import numpy as np
import igl
import scipy.sparse as sp
import pyshtools as pysh
from sklearn.decomposition import PCA
import vtk
import src.utils as utils
from src.spharm import SpHarm


class MeshHarm(SpHarm):
    """
    A class using direct harmonic coefficients to 
    
    This class combines functionality from optimization.py and mesh_mapping.py,
    providing methods for mesh parameterization and optimization.
    """    

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

    
    def compute_coefficients(self, lmax=10, hks=True, ts=[0.1, 1, 10]):
        '''
        Compute harmonics coefficients for

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
        self._eig_decomp(k=k)
        self.coeffs_v = self.eigvecs.T @ (self.mass_matrix @ self.v)

        if hks: 
            hks = [] 
            for t in ts:
                hks.append(np.einsum('i, ji->j', np.exp(-self.eigvals*t), self.eigvecs**2))
            self.hks = np.array(hks).T 

            self.coeffs_hks = self.eigvecs.T @ (self.mass_matrix @ self.hks)

    def smooth_mesh(self, new_lmax=None):
        '''
        Smooth the mesh by only retaining low-frequency spherical harmonics.

        Args:
            new_lmax (int): New maximum degree for the harmonics coefficients.
                Determines the number of eigenfunctions used (k = new_lmax**2).
        '''
        if new_lmax is not None: 
            self.lmax = new_lmax
        self.v = self.reconstruct_from_coeffs(self.coeffs_v, lmax=self.lmax)

    
    def compute_hks_for_new_times(self, new_ts=[1, 5, 10], coeffs=True):
        hks = [] 
        for t in new_ts:
            hks.append(np.einsum('i, ji->j', np.exp(-self.eigvals*t), self.eigvecs**2))
        hks = np.array(hks).T 

        if coeffs: 
            coeffs_hks = self.eigvecs.T @ (self.mass_matrix @ hks)
            return coeffs_hks
        else: 
            return hks 

    def reconstruct_from_coeffs(self, coeffs, lmax=15):
        '''
        Reconstruct the shape from harmonics coefficients.
        Args: 
            lmax: Maximum degree for the harmonics coefficients.
        Returns: 
            v_reconstructed (np.ndarray): Reconstructed vertex coordinates.
            Shape: (k, 3), where k = lmax**2 and 3 corresponds to the (x, y, z) coordinates of each vertex.
        '''
        if self.lmax >= lmax: 
            recon = self.eigvecs[:, :int(lmax**2)] @ coeffs[:int(lmax**2), :]
        else: 
            raise ValueError(f"lmax {self.lmax} is less than the requested lmax {lmax}. Please compute coefficients with lmax >= {lmax} first.")
        return recon
    
    def compute_spectrum(self, coeffs, lmax=15): 
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
        return np.sqrt(np.sum(diff**2)/self.v.shape[0])/np.sqrt(self.mass_matrix.diagonal().sum())

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
            np.save(path + '_eigvals.npy', self.eigvals)

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
        # self.eigvals = np.load(path + '_eigvals.npy')
        self.lmax = int(np.sqrt(self.eigvecs.shape[1])) 

        mesh_path = path + '_transformed_mesh.obj'
        self.v, self.f = igl.read_triangle_mesh(mesh_path)
        


# Example usage:
if __name__ == "__main__":

    from tqdm import tqdm
    import time


    # Organoid 5 & 40 (fairly spherical), 27 & 33 & 42 (one crypt, elongated), 20 & 32 (two crypts, elongated), 23 & 35 (two crypts, at an angle like mickey mouse), 29 (three crypts), 28 & 38 (blobby)
    times = {'load_mesh': [], 'align_pca': [], 'compute_coeffs': [], 'save_results': []}

    for n in tqdm([5, 20, 23, 27, 28, 29, 32, 33, 35, 38, 40, 42]):
        path = f'Data/mesh/{n}.stl'
        m = MeshHarm()
        
        start = time.time()
        m.load_mesh_from_file(path)
        times['load_mesh'].append(time.time() - start)
        
        start = time.time()
        m.align_with_pca()
        times['align_pca'].append(time.time() - start)
        
        start = time.time()
        m.compute_hks_and_coor_coefficients(lmax=15)
        times['compute_coeffs'].append(time.time() - start)
        
        start = time.time()
        m.save_results(f'sim/meshharm/{n}')
        times['save_results'].append(time.time() - start)

    # Print average times
    print(f"Average times:")
    print(f"Load mesh: {np.mean(times['load_mesh']):.4f}s")
    print(f"Align PCA: {np.mean(times['align_pca']):.4f}s")
    print(f"Compute coefficients: {np.mean(times['compute_coeffs']):.4f}s")
    print(f"Save results: {np.mean(times['save_results']):.4f}s")