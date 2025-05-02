import numpy as np
import igl
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
import pyshtools as pysh
from sklearn.decomposition import PCA
import vtk
import utils


class SpHarm:
    """
    A class for mesh optimization and parameterization tasks.
    
    This class combines functionality from optimization.py and mesh_mapping.py,
    providing methods for mesh parameterization and optimization.
    """
    
    def __init__(self):
        """Initialize the SpHarm with empty attributes."""
        self.v = None  # Vertices
        self.f = None  # Faces
        self.coors = None  # Spherical coordinates
        self.areas = None  # Face areas
        self.edges = None  # Mesh edges
        self.transform_matrix = None  # Transformation matrix for PCA
        self.clm = None  # Spherical harmonics coefficients
        self.reconstructed_xyzs = None  # Reconstructed coordinates from SH coefficients
        
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
        else:
            raise ValueError(f"Unsupported file format: {path}")
            
        return self
    
    def _load_stl(self, path):
        """
        Load an STL file using VTK.
        
        Args:
            path: Path to the STL file
        """
        reader = vtk.vtkSTLReader()
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
        
        # Center and normalize
        self.v -= self.v.mean(0)
        self.v /= self.v.max()
    
    def load_from_arrays(self, vertices, faces):
        """
        Load mesh from vertex and face arrays.
        
        Args:
            vertices: Numpy array of vertices (n_vertices, 3)
            faces: Numpy array of faces (n_faces, 3)
            
        Returns:
            self: For method chaining
        """
        self.v = vertices
        self.f = faces
        return self
    
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
        return self
    
    def compute_initial_parameterization(self):
        """
        Compute spherical parameterization of the mesh.
        
        Returns:
            self: For method chaining
        """
        # Find extreme points as poles
        poles = np.array([np.argmax(self.v[:, 0]), np.argmin(self.v[:, 0])])
        pole_vals = np.array([0, np.pi])

        # List of all vertex indices and interior indices
        v_all = np.arange(self.v.shape[0])
        v_in = np.setdiff1d(v_all, poles)

        # Construct and slice up Laplacian
        l = igl.cotmatrix(self.v, self.f)
        M = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        Minv = sp.diags(1 / M.diagonal()) 
        l = Minv.dot(l)

        l_ii = l[v_in, :]
        l_ii = l_ii[:, v_in]

        l_ib = l[v_in, :]
        l_ib = l_ib[:, poles]

        # Solve PDE for theta
        theta_in = sp.linalg.spsolve(-l_ii, l_ib.dot(pole_vals))

        theta = np.empty_like(self.v[:, 1])
        theta[poles] = pole_vals 
        theta[v_in] = theta_in
        
        # Compute branch cuts and phi
        phi = self._compute_phi(theta, poles, v_in)
        
        # Store the coordinates
        self.coors = np.vstack([theta, phi]).T
        return self
    
    def _find_rhs(self, displacements, forward_vec, backward_vec, n):
        """Helper function for computing branch cuts."""
        x_director = forward_vec/np.linalg.norm(forward_vec)
        y_director = np.cross(forward_vec, n)
        y_director /= np.linalg.norm(y_director)

        x_decom = np.dot(displacements, x_director)
        y_decom = np.dot(displacements, y_director)
        angle = np.arctan2(y_decom, x_decom) % (2*np.pi) 

        ref_x = np.dot(backward_vec, x_director)
        ref_y = np.dot(backward_vec, y_director)
        ref_angle = np.arctan2(ref_y, ref_x) % (2*np.pi) 

        return angle > (ref_angle + 1e-4)
    
    def _compute_phi(self, theta, poles, v_in):
        """
        Compute phi coordinates by solving a PDE with branch cuts.
        
        Args:
            theta: Theta coordinates
            poles: Pole indices
            v_in: Interior vertex indices
            
        Returns:
            phi: Phi coordinates
        """
        neighbours = igl.adjacency_list(self.f)
        normals = igl.per_vertex_normals(self.v, self.f)
        
        # Find branch cuts
        i = np.argmin(theta)
        line1 = []
        line2 = []
        
        while theta[i] < np.pi:
            neighbours_i = np.array(neighbours[i])
            n = normals[i]

            max_index = np.argmax(theta[neighbours_i])
            new_i = neighbours_i[max_index]
            neighbours_i = np.delete(neighbours_i, max_index)

            if len(line1) > 1:
                u = (self.v[new_i] - self.v[i])
                m = self._find_rhs(self.v[neighbours_i] - self.v[i], u, self.v[old_i]-self.v[i], n)
                line2.extend(neighbours_i[m])

            line1.append(new_i)
            old_i = i
            i = new_i
            
        line1 = np.array(line1, dtype=int)
        line2 = np.array(np.unique(line2), dtype=int)

        sort_indices = np.argsort(theta[line2])
        line2 = line2[sort_indices]
        
        # Solve PDE for phi
        l = igl.cotmatrix(self.v, self.f)

        for p in poles:
            for nn in neighbours[p]:
                l[nn, nn] += l[nn, p]  # delete all the connections to the poles

        M = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        Minv = sp.diags(1 / M.diagonal())
        l = Minv.dot(l)

        lines = np.concatenate([line1, line2])
        line_vals = np.concatenate([np.zeros(len(line1)), 2*np.pi*np.ones(len(line2))])
        v_in2 = np.setdiff1d(v_in, lines)  # cut off poles and branch cuts
        
        l_ii = l[v_in2, :]
        l_ii = l_ii[:, v_in2]

        l_ib = l[v_in2, :]
        l_ib = l_ib[:, lines]

        # Solve PDE
        phi_in = sp.linalg.spsolve(-l_ii, l_ib.dot(line_vals))

        phi = np.empty_like(self.v[:, 1])
        phi[poles] = 0
        phi[v_in2] = phi_in
        phi[lines] = line_vals
        
        return phi
    
    def _prepare_optimization(self):
        """
        Prepare data for optimization.
        
        Returns:
            self: For method chaining
        """
        # Compute areas and normalize
        self.areas = igl.doublearea(self.v, self.f)/2
        self.areas /= np.sum(self.areas)
        self.areas *= 4*np.pi  # normalized area

        # Get edges
        self.edges = igl.edges(self.f)
        
        # Convert spherical coordinates to Cartesian 
        # This gives the initial mapping from real space to a unit sphere 
        theta, phi = self.coors[:, 0], self.coors[:, 1]
        x, y, z = utils.sph2cart(1, theta, phi)
        v_init = np.stack([x, y, z], axis=1) 
        
        # Convert to JAX arrays
        self.areas = jnp.array(self.areas)
        self.edges = jnp.array(self.edges)
        self.f = jnp.array(self.f)
        v_init = jnp.array(v_init)
        return v_init 
    
    # JAX functions for optimization
    @staticmethod
    @jit
    def jax_doublearea(v, f):
        """Compute double area of triangles with JAX."""
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        
        e1 = v1 - v0
        e2 = v2 - v0
        
        cross = jnp.cross(e1, e2)
        doublearea = jnp.linalg.norm(cross, axis=1)
        
        return doublearea
    
    @staticmethod
    @jit
    def area_constraint_vec(v, f, areas):
        """Compute area constraint vector."""
        area_test = SpHarm.jax_doublearea(v, f)/2
        return area_test/areas - 1  # Vector of constraint values
    
    @staticmethod
    @jit
    def norm_constraint_vec(v):
        """Compute norm constraint vector."""
        norms = jnp.linalg.norm(v, axis=1)
        return norms - 1  # Vector of constraint values
    
    @staticmethod
    @jit
    def lengths_ratio(v, edges):
        """Compute the objective function (length ratio)."""
        return jnp.sum(v[edges[:, 0]] * v[edges[:, 1]])
    
    @staticmethod
    @jit
    def augmented_lagrangian(v_flat, f, edges, areas, lambda_area, lambda_norm, mu):
        """
        Compute the augmented Lagrangian function.
        
        Args:
            v_flat: Flattened vertex array
            f: Faces
            edges: Edges
            areas: Areas
            lambda_area: Lagrange multipliers for area constraint
            lambda_norm: Lagrange multipliers for norm constraint
            mu: Penalty parameter
            
        Returns:
            Augmented Lagrangian value
        """
        v = v_flat.reshape(-1, 3)
        
        # Objective function
        obj_value = SpHarm.lengths_ratio(v, edges)
        
        # Compute constraint values
        c_area = SpHarm.area_constraint_vec(v, f, areas)
        c_norm = SpHarm.norm_constraint_vec(v)
        
        # Lagrangian terms
        lagrange_area = jnp.sum(lambda_area * c_area)
        lagrange_norm = jnp.sum(lambda_norm * c_norm)
        
        # Penalty terms
        penalty_area = (mu/2) * jnp.sum(jnp.square(c_area))
        penalty_norm = (mu/2) * jnp.sum(jnp.square(c_norm))
        
        return obj_value + lagrange_area + lagrange_norm + penalty_area + penalty_norm
    
    def optimize_primal(self, v_flat, lambda_area, lambda_norm, mu, learning_rate, num_steps):
        """
        Optimize the primal variables (vertices).
        
        Args:
            v_flat: Flattened vertex array
            lambda_area: Lagrange multipliers for area constraint
            lambda_norm: Lagrange multipliers for norm constraint
            mu: Penalty parameter
            learning_rate: Learning rate for optimizer
            num_steps: Number of optimization steps
            
        Returns:
            Optimized flattened vertex array
        """
        # Create the augmented Lagrangian for fixed multipliers
        def loss_fn(v_flat):
            return self.augmented_lagrangian(v_flat, self.f, self.edges, self.areas, 
                                            lambda_area, lambda_norm, mu)
        
        value_and_grad_fn = jit(value_and_grad(loss_fn))
        
        # Optimizer for primal variables
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(v_flat)
        
        # Optimization loop
        for i in range(num_steps):
            loss, grad_val = value_and_grad_fn(v_flat)
            updates, opt_state = optimizer.update(grad_val, opt_state)
            v_flat = optax.apply_updates(v_flat, updates)
        
        return v_flat
    
    def optimize(self, max_outer_iterations=20, primal_steps=100, verbose=True):
        """
        Run the augmented Lagrangian optimization method.
        This finds the best mapping of the mesh to a unit sphere that preserves the area of each trianglular face of the original mesh and the angle between the edges of the mesh.
        
        Args:
            max_outer_iterations: Maximum number of outer iterations
            primal_steps: Number of primal optimization steps per outer iteration
            
        Returns:
            self: For method chaining
        """
        # Initialize primal variables
        v_init = self._prepare_optimization()
        v_flat = v_init.reshape(-1)
        
        # Initialize Lagrange multipliers
        v = v_flat.reshape(-1, 3)
        lambda_area = jnp.zeros_like(self.area_constraint_vec(v, self.f, self.areas))
        lambda_norm = jnp.zeros_like(self.norm_constraint_vec(v))
        
        # Initialize penalty parameter
        mu = 10.0
        mu_max = 1e8
        mu_factor = 10.0
        
        # Learning rate for primal optimization
        learning_rate = 0.01
        
        # Outer loop - update Lagrange multipliers
        for outer_iter in range(max_outer_iterations):
            if verbose:
                print(f"Outer iteration {outer_iter}, mu = {mu}")
            
            # Optimize primal variables
            v_flat = self.optimize_primal(v_flat, lambda_area, lambda_norm, mu, learning_rate, primal_steps)
            v = v_flat.reshape(-1, 3)
            
            # Compute constraint violations
            c_area = self.area_constraint_vec(v, self.f, self.areas)
            c_norm = self.norm_constraint_vec(v)
            
            # Check constraint satisfaction
            area_violation = jnp.max(jnp.abs(c_area))
            norm_violation = jnp.max(jnp.abs(c_norm))
            
            if outer_iter % 5 == 0 and verbose:
                print(f"  Max area constraint violation: {area_violation}")
                print(f"  Max norm constraint violation: {norm_violation}")
            
            # Check for convergence
            if area_violation < 1e-6 and norm_violation < 1e-6:
                if verbose:
                    print("Constraints satisfied, optimization complete.")
                break
            
            # Update Lagrange multipliers
            lambda_area = lambda_area + mu * c_area
            lambda_norm = lambda_norm + mu * c_norm
            
            # Update penalty parameter
            if outer_iter > 0 and outer_iter % 3 == 0:
                mu = min(mu_max, mu * mu_factor)
        
        optimized_v = v_flat.reshape(-1, 3)

        # Convert the optimized coordinates on a unit sphere to spherical coordinates
        x, y, z = optimized_v.T
        _, theta, phi = utils.cart2sph(x, y, z)
        self.coors = np.vstack([theta, phi]).T

        # Revert jnp arrays to numpy arrays 
        self.f = np.array(self.f)
        self.v = np.array(self.v) 

        return self
    
    def compute_sh_coefficients(self, lmax=15):
        """
        Analyze optimized vertices using spherical harmonics.
        
        Args:
            lmax: Maximum degree for SH coefficients
            grid_lmax: Maximum degree for expanded grid
            
        Returns:
            clms: Spherical harmonics coefficients
        """
        theta, phi = self.coors[:, 0], self.coors[:, 1]
        
        # Convert to lat/lon
        lat, lon = utils.sph2latlon(theta, phi)
        
        
        self.clms = [] 
        for i in range(3):
            x = self.v[:, i]
            
            # Compute SH coefficients
            sh = pysh.SHCoeffs.from_least_squares(x, lat, lon, lmax)
            self.clms.append(sh.to_array())

        self.clms = np.array(self.clms)
    
        return self.clms
    
    def reconstruct_from_sh(self, grid_lmax=50):
        '''
        Reconstruct the shape from spherical harmonics coefficients.
        Args: 
            grid_lmax: lmax used when reconstructing the grid. To not lose information, it should be larger than the lmax used to compute the SH coefficients.

        Returns: 
            points: Reconstructed points in Cartesian coordinates, after undoing the PCA transformation.
            latlon_grid: (lat_grid, lon_grid) for the reconstructed points.
        '''
        points = [] 
        for i in range(3): 
            # Expand the SH coefficients for each coordinate 
            clm = self.clms[i]
            sh = pysh.SHCoeffs.from_array(clm)
            grid = sh.expand(lmax=grid_lmax)
            points.append(grid.to_array())

        points = np.array(points)
        points = np.einsum('ij,jkl->ikl', np.linalg.inv(self.transform_matrix), points)
        latlon_grid = np.meshgrid(grid.lats(), grid.lons(), indexing='ij')

        return points, latlon_grid 

    def save_results(self, path):
        """
        Save results to files.
        Assumes that self.v and self.f are post pca transformation. 
        
        Args:
            path: Path to save results
        """
        if self.transform_matrix is not None:
            np.save(path + '_transform_matrix.npy', self.transform_matrix)

        if self.coors is not None:
            np.save(path + '_coors.npy', self.coors)

        if self.clms is not None:
            np.save(path + '_clms.npy', self.clms)

        if self.v is not None and self.f is not None:
            igl.write_triangle_mesh(path + '_transformed_mesh.obj', self.v, self.f)

    def load_results(self, path):
        """
        Load results from files.
        
        Args:   
        """
        self.transform_matrix = np.load(path + '_transform_matrix.npy')
        self.coors = np.load(path + '_coors.npy')
        self.clms = np.load(path + '_clms.npy')
        self.v, self.f = igl.read_triangle_mesh(path + '_transformed_mesh.obj')


# Example usage:
if __name__ == "__main__":

    from tqdm import tqdm


    # Organoid 5 & 40 (fairly spherical), 27 & 33 & 42 (one crypt, elongated), 20 & 32 (two crypts, elongated), 23 & 35 (two crypts, at an angle like mickey mouse), 29 (three crypts), 28 & 38 (blobby)
    for n in tqdm([5, 20, 23, 27, 28, 29, 32, 33, 35, 38, 40, 42]):
        path = f'Data/mesh/{n}.stl'
        m = SpHarm()
        m.load_mesh_from_file(path)  
        m.align_with_pca() 
        m.compute_initial_parameterization()
        m.optimize(max_outer_iterations=100, primal_steps=100, verbose=False)
        clms = m.compute_sh_coefficients(lmax=15)
        m.save_results(f'sim/{n}')