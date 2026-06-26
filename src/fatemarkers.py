import igl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from src.meshharm import MeshHarm
from src import utils 
import vtk 
from sklearn.decomposition import PCA

class FateMarkers(): 

    def __init__(self):
        """Initialize the SpHarm with empty attributes."""
        self.v = None  # Vertices
        self.f = None  # Faces

    def align_with_pca(self):
        """
        Align the mesh using PCA.
        
        Returns:
            self: For method chaining
        """
        pca = PCA(n_components=3)
        pca.fit(self.v)
        self.v = pca.transform(self.v)
        self.transform_matrix = np.copy(pca.components_)
        self._orient_axes()  # resolve PCA sign ambiguity
        return self

    def _orient_axes(self):
        """Resolve the PCA sign ambiguity deterministically.

        PCA centres the vertices, so the vertex centroid is at the origin and
        carries no sign information. Instead, flip each principal axis so that
        the centroid of the *enclosed solid* (its centre of mass) lies on the
        positive side of that axis. Falls back to coordinate skewness if the
        mesh is not watertight (near-zero signed volume).

        Mirroring is allowed: the three axes are signed independently, so the
        result may be a reflection (this is fine for shape inspection).
        """
        v, f = self.v, self.f
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        vol6 = np.einsum('ij,ij->i', a, np.cross(b, c))   # 6 * signed tet volume
        V6 = vol6.sum()
        if abs(V6) > 1e-6:
            # centre of mass of the solid (tet centroids weighted by signed volume)
            score = (vol6[:, None] * (a + b + c)).sum(axis=0) / (4.0 * V6)
        else:
            score = np.mean(v ** 3, axis=0)              # skewness fallback

        signs = np.where(score >= 0, 1.0, -1.0)
        self.v = self.v * signs[None, :]
        self.transform_matrix = self.transform_matrix * signs[:, None]
        return self

    def load_mesh_from_file(self, path):
        """
        Load a mesh from a file (STL or OBJ).
        
        Args:
            path: Path to the mesh file
            
        Returns:
            self: For method chaining
        """
        if path.endswith('.obj'):
            self.v, self.f = igl.read_triangle_mesh(path)
        elif path.endswith('.vtp'):
            self._load_vtp(path) 
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        self._rescale_v()  # Center and normalize the vertices
            
        return self
    
    def precompute_eigens(self, lmax=15, sigma=0):
        self.lmax = lmax 
        k = int(lmax**2)

        # compute the harmonics modes 
        normalised_v = self.v/np.max(np.abs(self.v), axis=0)[np.newaxis, :] # important to normalise to ensure proper ordering of the harmonics 
        _, self.modes, self.mass_matrix = self._eig_decomp(normalised_v, self.f, k=k, sigma=sigma)

        # precompute the true eig decomp for the original mesh 
        self.eigvals, self.eigvecs, true_area_matrix = self._eig_decomp(self.v, self.f, k=k, sigma=sigma)
        self.area = true_area_matrix.diagonal().sum()

    def _rescale_v(self): 
        # Center and normalize
        self.v -= self.v.mean(0)
        # self.v /= np.mean(np.linalg.norm(self.v, axis=-1))
        self.v /= 10 # rescale to the units in length of a cell 
    
    def _load_vtp(self, path): 
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()

        # Extract vertices
        self.v = np.array(polydata.GetPoints().GetData())

        # Extract faces (cells)
        cells = polydata.GetPolys()
        cells.InitTraversal()

        # Create an array to store the faces
        n_faces = polydata.GetNumberOfPolys()
        faces = np.zeros((n_faces, 3), dtype=np.int64)

        # Extract all faces
        for i in range(n_faces):
            cell = vtk.vtkIdList()
            cells.GetNextCell(cell)
            for j in range(3):  # Assuming triangular faces
                faces[i, j] = cell.GetId(j)
        self.f = faces

        # Get the scalar fields 

        point_data = polydata.GetPointData()
        self.field_names = [] 
        self.fields = []
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            self.field_names.append(array.GetName())
            self.fields.append(np.array(array))
        self.fields = np.array(self.fields).T  # Transpose to match vertices

    def _refine_markers(self, annotation_names, exclusion_rules):
        """Combine multi-channel markers and apply biological mutual exclusion to
        the per-vertex fields, in place. Mirrors the CSV's *.cnt_exclusive
        definition so the displayed/encoded fates match the CSV.

        Args:
            annotation_names: friendly-name -> vtp field-name, or a list of
                field-names (duplicate channels are combined elementwise via max,
                e.g. 'ta' = cycd OR cyca). A combined marker is appended as a new
                per-vertex field named by the marker key (e.g. 'ta').
            exclusion_rules:  friendly-name -> list of markers whose presence at a
                vertex zeroes this marker there.

        Returns:
            self: For method chaining
        """
        # resolve each friendly marker to a single working column
        idx = {}
        for fr, flds in annotation_names.items():
            flds = [flds] if isinstance(flds, str) else list(flds)
            cols = [self.field_names.index(f) for f in flds if f in self.field_names]
            if not cols:
                continue
            if len(cols) == 1:
                idx[fr] = cols[0]
            else:  # combine duplicate channels into a new field named by the marker
                combined = self.fields[:, cols].max(axis=1)
                self.fields = np.column_stack([self.fields, combined])
                self.field_names = list(self.field_names) + [fr]
                idx[fr] = self.fields.shape[1] - 1

        # presence snapshot before any modification -> order-independent exclusion
        present = {fr: self.fields[:, i] > 0 for fr, i in idx.items()}
        orig = self.fields.copy()
        for marker, excluders in exclusion_rules.items():
            if marker not in idx:
                continue
            keep = present[marker].copy()
            for ex in excluders:
                if ex in present:
                    keep &= ~present[ex]
            self.fields[:, idx[marker]] = np.where(keep, orig[:, idx[marker]], 0)
        return self


    def _eig_decomp(self, v, f, k=100, sigma=0): 
        """
        Compute the first k eigenvalues and eigenvectors of the Laplace-Beltrami operator.
        
        Args:
            k: Number of eigenvalues/vectors to compute.
        Returns:
            eigvals: Eigenvalues
            eigvecs: Eigenvectors
        """
        L = -igl.cotmatrix(v, f)
        mass_matrix = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        eigvals, eigvecs = sp.linalg.eigsh(L, k=k, M=mass_matrix, sigma=sigma, which='LM')
        return eigvals, eigvecs, mass_matrix

    def compute_coefficients(self, fate=True):
        self.coeffs_v = self.modes.T @ (self.mass_matrix @ self.v)
        if not fate:
            return self.coeffs_v
        self.coeffs_fm = self.modes.T @ (self.mass_matrix @ self.fields)
        return self.coeffs_v, self.coeffs_fm

    def compute_hks_for_new_times(self, new_ts=[1, 5, 10], coeffs=True):
        hks = [] 
        for t in new_ts:
            hks.append(np.einsum('i, ji->j', np.exp(-self.eigvals*t), self.eigvecs**2))
        hks = np.array(hks).T 

        if coeffs: 
            coeffs_hks = self.modes.T @ (self.mass_matrix @ hks)
            return coeffs_hks
        else: 
            return hks 

    def reconstruct_from_coeffs(self, coeffs, lmax=8):
        '''
        Reconstruct the shape from harmonics coefficients.
        Args: 
            lmax: Maximum degree for the harmonics coefficients.
        Returns: 
            v_reconstructed (np.ndarray): Reconstructed vertex coordinates.
            Shape: (k, 3), where k = lmax**2 and 3 corresponds to the (x, y, z) coordinates of each vertex.
        '''
        if self.lmax >= lmax: 
            recon = self.modes[:, :int(lmax**2)] @ coeffs[:int(lmax**2), :]
        else: 
            raise ValueError(f"lmax {self.lmax} is less than the requested lmax {lmax}. Please compute coefficients with lmax >= {lmax} first.")
        return recon
    
    def compute_spectrum(self, coeffs, lmax=8): 
        spectrum = [] 
        if lmax <= self.lmax: 
            for i in range(lmax):
                spectrum.append(np.sum(coeffs[i**2:(i+1)**2]**2, axis=0))
        else: 
            raise ValueError(f"lmax {self.lmax} is less than the requested lmax {lmax}. Please compute coefficients with lmax >= {lmax} first.")
        return np.array(spectrum)
    
    def compute_recon_quality(self, lmax=None): 
        if lmax is None:
            lmax = self.lmax
        v_recon = self.reconstruct_from_coeffs(self.coeffs_v, lmax=lmax)
        diff = self.v - v_recon 
        return np.sqrt(np.sum(diff**2)/self.v.shape[0])/np.sqrt(self.area)

    def save_results(self, path, fate=True):
        filename = f"{path}_coeffs.npz"

        save_dict = {
            'coeffs_v': self.coeffs_v,
            'eigvals': self.eigvals,
            'eigvecs': self.eigvecs,
            'modes': self.modes,
            'mass_matrix': self.mass_matrix,
            'transform_matrix': self.transform_matrix,
            'lmax': self.lmax,
            'area': self.area
        }
        if fate:
            save_dict['coeffs_fm'] = self.coeffs_fm
            save_dict['field_names'] = self.field_names
        np.savez(filename, **save_dict)

        igl.write_triangle_mesh(path + '_transformed_mesh.obj', self.v, self.f)

    def load_results(self, path):
        data = np.load(path + '_coeffs.npz', allow_pickle=True)
        self.coeffs_v = data['coeffs_v']
        # fate coefficients are optional (shape-only runs skip them)
        if 'coeffs_fm' in data.files:
            self.coeffs_fm = data['coeffs_fm']
            self.field_names = data['field_names'].tolist()
        self.eigvals = data['eigvals']
        self.eigvecs = data['eigvecs']
        self.modes = data['modes']
        self.mass_matrix = data['mass_matrix'].item()
        self.lmax = int(data['lmax'])
        self.transform_matrix = data['transform_matrix']

        self.area = data['area']
        self.v, self.f = igl.read_triangle_mesh(path + '_transformed_mesh.obj')
