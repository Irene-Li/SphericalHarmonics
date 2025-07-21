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
from spharm import SpHarm


class MeshHarm(SpHarm):
    """
    A class using direct harmonic coefficients to 
    
    This class combines functionality from optimization.py and mesh_mapping.py,
    providing methods for mesh parameterization and optimization.
    """    

    def _eig_decomp(self, k=100): 
        """
        Compute the first k eigenvalues and eigenvectors of the Laplace-Beltrami operator.
        
        Args:
            k: Number of eigenvalues/vectors to compute.
        Returns:
            eigvals: Eigenvalues
            eigvecs: Eigenvectors
        """
        L = -igl.cotmatrix(self.v, self.f)
        M = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        eigvals, eigvecs = sp.linalg.eigsh(L, k=k, M=M, sigma=0, which='LM')
        return eigvals, eigvecs, M

    
    def compute_hks_and_coor_coefficients(self, lmax=15, ts=[0.1, 1, 10]):
        '''
        Analyze optimized vertices using spherical harmonics.

        Args:
            lmax (int): Maximum degree for the harmonics coefficients. Determines the number of eigenfunctions used (k = lmax**2).
            ts (list of float): List of time scales at which to compute the heat kernel signature (HKS).

        Returns:
            coeffs_hks (np.ndarray): Harmonics coefficients for the heat kernel signature.
                Shape: (k, len(ts)), where k = lmax**2.
            coeffs_v (np.ndarray): Harmonics coefficients for the vertex coordinates.
                Shape: (k, 3), where k = lmax**2 and 3 corresponds to the (x, y, z) coordinates of each vertex.

        Notes:
            - Requires the mesh vertices (self.v) and faces (self.f) to be defined.
            - Uses the first k eigenvalues and eigenvectors of the Laplace-Beltrami operator.
        '''
        self.lmax = lmax 
        k = int(lmax**2)
        eigvals, self.eigvecs, mass_matrix = self._eig_decomp(k=k)

        hks = [] 
        for t in ts:
            hks.append(np.einsum('i, ji->j', np.exp(-eigvals*t), self.eigvecs**2))
        self.hks = np.array(hks).T 

        self.coeffs_hks = self.eigvecs.T @ (mass_matrix @ self.hks)
        self.coeffs_v = self.eigvecs.T @ (mass_matrix @ self.v)
        return self.coeffs_hks, self.coeffs_v
    
    def reconstruct_from_coeffs(self, lmax=15):
        '''
        Reconstruct the shape from harmonics coefficients.
        Args: 
            lmax: Maximum degree for the harmonics coefficients.
        Returns: 
            v_reconstructed (np.ndarray): Reconstructed vertex coordinates.
            Shape: (k, 3), where k = lmax**2 and 3 corresponds to the (x, y, z) coordinates of each vertex.
        '''
        if self.lmax >= lmax: 
            v_reconstructed = self.eigvecs[:, :int(lmax**2)] @ self.coeffs_v[:int(lmax**2), :]
        else: 
            raise ValueError(f"lmax {self.lmax} is less than the requested lmax {lmax}. Please compute coefficients with lmax >= {lmax} first.")
        return v_reconstructed
    
    def compute_spectrum(self, coeffs, lmax=15): 
        spectrum = [] 
        if lmax <= self.lmax: 
            for i in range(lmax):
                spectrum.append(np.sum(coeffs[i**2:(i+1)**2]**2))
        else: 
            raise ValueError(f"lmax {self.lmax} is less than the requested lmax {lmax}. Please compute coefficients with lmax >= {lmax} first.")
        return np.array(spectrum)
    
    def compute_recon_quality(self, lmax=None): 
        if lmax is None:
            lmax = self.lmax
        v_recon = self.reconstruct_from_coeffs(lmax=lmax)
        diff = self.v - v_recon 
        return np.linalg.norm(diff) / np.linalg.norm(self.v)

    def save_results(self, path):
        """
        Save results to files.
        Assumes that self.v and self.f are post pca transformation. 
        
        Args:
            path: Path to save results
        """
        if self.transform_matrix is not None:
            np.save(path + '_transform_matrix.npy', self.transform_matrix)

        if self.coeffs_v is not None:
            np.save(path + '_coeffs_v.npy', self.coeffs_v)

        if self.coeffs_hks is not None:
            np.save(path + '_coeffs_hks.npy', self.coeffs_hks)

        if self.eigvecs is not None:
            np.save(path + '_eigvecs.npy', self.eigvecs)

        if self.v is not None and self.f is not None:
            igl.write_triangle_mesh(path + '_transformed_mesh.obj', self.v, self.f)

    def load_results(self, path):
        """
        Load results from files.
        
        Args:   
        """
        self.transform_matrix = np.load(path + '_transform_matrix.npy') 
        self.coeffs_v = np.load(path + '_coeffs_v.npy') 
        self.coeffs_hks = np.load(path + '_coeffs_hks.npy') 
        self.eigvecs = np.load(path + '_eigvecs.npy') 
        self.lmax = int(np.sqrt(self.eigvecs.shape[1])) 

        mesh_path = path + '_transformed_mesh.obj'
        self.v, self.f = igl.read_triangle_mesh(mesh_path)
        


# Example usage:
if __name__ == "__main__":

    from tqdm import tqdm


    # Organoid 5 & 40 (fairly spherical), 27 & 33 & 42 (one crypt, elongated), 20 & 32 (two crypts, elongated), 23 & 35 (two crypts, at an angle like mickey mouse), 29 (three crypts), 28 & 38 (blobby)
    for n in tqdm([5, 20, 23, 27, 28, 29, 32, 33, 35, 38, 40, 42]):
        path = f'Data/mesh/{n}.stl'
        m = MeshHarm()
        m.load_mesh_from_file(path)  
        m.align_with_pca() 
        m.compute_hks_and_coor_coefficients(lmax=15)
        m.save_results(f'sim/meshharm/{n}')