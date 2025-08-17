import igl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from src.meshharm import MeshHarm
from src import utils 
import vtk 

class FateMarkers(MeshHarm): 

    def load_mesh_from_file(self, path):
        """
        Load a mesh from a file (STL or OBJ).
        
        Args:
            path: Path to the mesh file
            
        Returns:
            self: For method chaining
        """
        if path.endswith('.stl'):
            self._load_stl(path)
        elif path.endswith('.obj'):
            self.v, self.f = igl.read_triangle_mesh(path)
        elif path.endswith('.vtp'):
            self._load_vtp(path) 
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        self._rescale_v()  # Center and normalize the vertices
            
        return self
    
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

    def compute_coefficients(self, lmax=15):
        self.lmax = lmax 
        k = int(lmax**2)
        self._eig_decomp(k=k)
        self.coeffs_v = self.eigvecs.T @ (self.mass_matrix @ self.v)
        self.coeffs_fm = self.eigvecs.T @ (self.mass_matrix @ self.fields)
        return self.coeffs_v, self.coeffs_fm

    def save_results(self, path):
        filename = f"{path}_coeffs.npz"

        save_dict = {
            'coeffs_v': self.coeffs_v, 
            'eigvals': self.eigvals,
            'eigvecs': self.eigvecs,
            'mass_matrix': self.mass_matrix,
            'transform_matrix': self.transform_matrix,
            'coeffs_fm': self.coeffs_fm,
        }


        np.savez(filename, coeffs_v=self.coeffs_v, coeffs_fm=self.coeffs_fm, 
                 eigvals=self.eigvals, eigvecs=self.eigvecs, mass_matrix=self.mass_matrix, 
                 field_names=self.field_names, transform_matrix=self.transform_matrix, lmax=self.lmax)
        
        igl.write_triangle_mesh(path + '_transformed_mesh.obj', self.v, self.f)

    def load_results(self, path): 
        data = np.load(path + '_coeffs.npz', allow_pickle=True)
        self.coeffs_v = data['coeffs_v']
        self.coeffs_fm = data['coeffs_fm']
        self.eigvals = data['eigvals']
        self.eigvecs = data['eigvecs']
        self.mass_matrix = data['mass_matrix'].item() 
        self.field_names = data['field_names'].tolist()
        self.lmax = int(data['lmax'])
        self.transform_matrix = data['transform_matrix']

        self.v, self.f = igl.read_triangle_mesh(path + '_transformed_mesh.obj')
