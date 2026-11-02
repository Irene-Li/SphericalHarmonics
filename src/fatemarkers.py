import igl
import numpy as np
from scipy.sparse.linalg import eigsh
import vtk
from sklearn.decomposition import PCA


def _polydata_mesh(polydata):
    """(vertices, faces) of a triangulated vtkPolyData."""
    v = np.array(polydata.GetPoints().GetData())
    cells = polydata.GetPolys()
    cells.InitTraversal()
    f = np.zeros((polydata.GetNumberOfPolys(), 3), dtype=np.int64)
    cell = vtk.vtkIdList()
    for i in range(len(f)):
        cells.GetNextCell(cell)
        for j in range(3):
            f[i, j] = cell.GetId(j)
    return v, f


class FateMarkers:
    """A true-geometry organoid mesh with its per-vertex fate fields, Laplace-Beltrami
    eigendecomposition and HKS."""

    def __init__(self):
        self.v = None  # Vertices
        self.f = None  # Faces

    def align_with_pca(self):
        """Rotate the mesh onto its principal axes, with their signs fixed."""
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
        """Load a .obj, .vtp or .stl mesh, centred and in units of 10 mesh units (about a cell)."""
        if path.endswith('.obj'):
            self.v, self.f = igl.read_triangle_mesh(path)
        elif path.endswith('.vtp'):
            self._load_vtp(path)
        elif path.endswith('.stl'):
            self._load_stl(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        self._rescale_v()
        return self

    def precompute_eigens(self, lmax=15, hks=True):
        """True-geometry Laplace-Beltrami eigendecomposition (lmax**2 pairs), which the
        HKS is built from, and the true surface area. hks=False (complexity-only) skips
        the eigensolve. The harmonic decomposition of fields and shape is SPHARM on the
        spherical map (src.spharm, compute_master_npz), not these eigenvectors."""
        self.lmax = lmax
        mass = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        self.eigvals = self.eigvecs = None
        if hks:
            L = -igl.cotmatrix(self.v, self.f)
            self.eigvals, self.eigvecs = eigsh(L, k=int(lmax**2), M=mass, sigma=0, which='LM')
        self.area = mass.diagonal().sum()

    def _rescale_v(self):
        self.v -= self.v.mean(0)
        self.v /= 10 # rescale to the units in length of a cell

    def _load_stl(self, path):
        """Geometry-only mesh (no per-vertex fields). Rejects incomplete files: a
        truncated/zero-filled ASCII STL can still parse into a partial mesh."""
        with open(path, 'rb') as fh:
            head = fh.read(5)
            fh.seek(max(0, fh.seek(0, 2) - 256))
            tail = fh.read()
        if head == b'solid' and b'endsolid' not in tail:
            raise ValueError(f"incomplete ASCII STL (no 'endsolid'): {path}")
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        v, f = _polydata_mesh(reader.GetOutput())
        if len(f) == 0:
            raise ValueError(f"STL has no faces: {path}")
        # vtkSTLReader gives float32 points; the .vtp meshes load as float64
        self.v = v.astype(np.float64)
        self.f = f
        self.field_names = []
        self.fields = None

    def _load_vtp(self, path):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()
        self.v, self.f = _polydata_mesh(polydata)
        point_data = polydata.GetPointData()
        self.field_names = [point_data.GetArray(i).GetName() for i in range(point_data.GetNumberOfArrays())]
        self.fields = np.array([np.array(point_data.GetArray(i)) for i in range(point_data.GetNumberOfArrays())]).T

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

    def compute_hks_for_new_times(self, new_ts=[1, 5, 10]):
        """Per-vertex HKS (n_verts, len(new_ts)) from the true-geometry eigendecomposition."""
        hks = []
        for t in new_ts:
            hks.append(np.einsum('i, ji->j', np.exp(-self.eigvals*t), self.eigvecs**2))
        return np.array(hks).T

    def save_results(self, path):
        """<path>_coeffs.npz (true-geometry eigendecomposition, PCA transform, area) and
        <path>_transformed_mesh.obj (the PCA-aligned mesh)."""
        save_dict = {
            'eigvals': self.eigvals,
            'eigvecs': self.eigvecs,
            'transform_matrix': self.transform_matrix,
            'lmax': self.lmax,
            'area': self.area
        }
        if self.eigvecs is None:            # complexity-only run (hks=False)
            del save_dict['eigvals'], save_dict['eigvecs']
        np.savez(f"{path}_coeffs.npz", **save_dict)
        igl.write_triangle_mesh(path + '_transformed_mesh.obj', self.v, self.f)

    def load_results(self, path):
        """Inverse of save_results."""
        data = np.load(path + '_coeffs.npz', allow_pickle=True)
        # absent for complexity-only (hks=False) runs
        self.eigvals = data['eigvals'] if 'eigvals' in data.files else None
        self.eigvecs = data['eigvecs'] if 'eigvecs' in data.files else None
        self.lmax = int(data['lmax'])
        self.transform_matrix = data['transform_matrix']
        self.area = float(data['area'])
        self.v, self.f = igl.read_triangle_mesh(path + '_transformed_mesh.obj')
