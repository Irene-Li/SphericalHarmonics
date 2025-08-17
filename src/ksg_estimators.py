import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
import warnings

class KSGEstimator:
    """
    Implementation of the Kraskov-Stögbauer-Grassberger (KSG) estimators
    for mutual information anxwwd Shannon entropy using k-nearest neighbors.
    
    Reference:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). 
    Estimating mutual information. Physical review E, 69(6), 066138.
    """
    
    def __init__(self, k=3, base=np.e, alpha=0):
        """
        Initialize the KSG estimator.
        
        Parameters:
        -----------
        k : int, default=3
            Number of nearest neighbors to use
        base : float, default=np.e
            Base for logarithm (np.e for nats, 2 for bits)
        alpha : float, default=0
            Bias correction parameter (experimental)
        """
        self.k = k
        self.base = base
        self.alpha = alpha
    
    def _digamma(self, x):
        """Compute digamma function with proper base conversion."""
        return digamma(x) / np.log(self.base)
    
    def _add_noise(self, X, noise_level=1e-10):
        """Add small amount of noise to break ties in data."""
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, X.shape)
            return X + noise
        return X
    
    def _kraskov_mi1(self, x, y, k=None):
        """
        First KSG mutual information estimator (Algorithm 1).
        
        This uses the maximum norm in joint space and counts neighbors
        strictly within the k-th neighbor distance.
        """
        if k is None:
            k = self.k
            
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T
            
        N = len(x)
        
        # Add small noise to break ties
        x = self._add_noise(x)
        y = self._add_noise(y)
        
        # Joint space with maximum norm
        xy = np.hstack([x, y])
        
        # Find k-nearest neighbors in joint space using maximum norm (Chebyshev)
        nbrs_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
        distances_joint, _ = nbrs_joint.kneighbors(xy)
        
        # Distance to k-th neighbor (excluding self)
        epsilon = distances_joint[:, k]
        
        # Count neighbors in marginal spaces within epsilon/2
        nx = np.zeros(N)
        ny = np.zeros(N)
        
        for i in range(N):
            # Count points in x-space with distance < epsilon[i]/2
            dx = np.max(np.abs(x - x[i]), axis=1)
            nx[i] = np.sum(dx < epsilon[i])
            
            # Count points in y-space with distance < epsilon[i]/2  
            dy = np.max(np.abs(y - y[i]), axis=1)
            ny[i] = np.sum(dy < epsilon[i])
        
        # KSG estimator formula (Equation 8)
        mi = (self._digamma(k) - np.mean(self._digamma(nx + 1) + self._digamma(ny + 1)) 
              + self._digamma(N))
        
        return max(0, mi)  # MI should be non-negative
    
    def _kraskov_mi2(self, x, y, k=None):
        """
        Second KSG mutual information estimator (Algorithm 2).
        
        Uses rectangles rather than squares. The key difference is that
        ε_x(i) and ε_y(i) are the marginal distances that result from 
        finding the k-th neighbor in joint space.
        """
        if k is None:
            k = self.k
            
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T
            
        N = len(x)
        
        # Add small noise to break ties
        x = self._add_noise(x)
        y = self._add_noise(y)
        
        # Joint space with maximum norm
        xy = np.hstack([x, y])
        
        # Find k-nearest neighbors in joint space
        nbrs_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
        distances_joint, indices_joint = nbrs_joint.kneighbors(xy)
        
        nx = np.zeros(N, dtype=int)
        ny = np.zeros(N, dtype=int)
        
        for i in range(N):
            # For each point, look at all k nearest neighbors
            # and find the marginal distances ε_x and ε_y
            
            # Get the k nearest neighbor indices (excluding self at index 0)
            neighbor_indices = indices_joint[i, 1:k+1]
            
            # Calculate marginal distances to each of the k neighbors
            dx_to_neighbors = np.max(np.abs(x[neighbor_indices] - x[i]), axis=1)
            dy_to_neighbors = np.max(np.abs(y[neighbor_indices] - y[i]), axis=1)
            
            # ε_x(i) and ε_y(i) are the maximum marginal distances among the k neighbors
            eps_x = np.max(dx_to_neighbors)
            eps_y = np.max(dy_to_neighbors)
            
            # Count all points (including the k neighbors themselves) within these distances
            dx_all = np.max(np.abs(x - x[i]), axis=1)
            dy_all = np.max(np.abs(y - y[i]), axis=1)
            
            nx[i] = np.sum(dx_all <= eps_x) - 1  # subtract 1 to exclude point i itself
            ny[i] = np.sum(dy_all <= eps_y) - 1  # subtract 1 to exclude point i itself
        
        # KSG estimator formula (Equation 9)
        mi = (self._digamma(k) - 1.0/k - np.mean(self._digamma(nx + 1) + self._digamma(ny + 1))
              + self._digamma(N))
        
        return max(0, mi)
    
    def mutual_information(self, x, y, method=1):
        """
        Estimate mutual information between x and y.
        
        Parameters:
        -----------
        x : array-like, shape (n_samples,) or (n_samples, n_features)
            First variable
        y : array-like, shape (n_samples,) or (n_features,)
            Second variable  
        method : int, default=1
            Which KSG estimator to use (1 or 2)
            
        Returns:
        --------
        mi : float
            Estimated mutual information
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        
        if len(x) < self.k + 1:
            raise ValueError(f"Sample size {len(x)} is too small for k={self.k}")
        
        if method == 1:
            return self._kraskov_mi1(x, y)
        elif method == 2:
            return self._kraskov_mi2(x, y)
        else:
            raise ValueError("method must be 1 or 2")
    
    def entropy(self, x, k=None):
        """
        Estimate Shannon entropy using Kozachenko-Leonenko estimator.
        
        Parameters:
        -----------
        x : array-like, shape (n_samples,) or (n_samples, n_features)
            Input data
        k : int, optional
            Number of nearest neighbors (uses self.k if None)
            
        Returns:
        --------
        h : float
            Estimated entropy
        """
        if k is None:
            k = self.k
            
        x = np.atleast_2d(x)
        if x.shape[0] == 1:
            x = x.T
            
        N, d = x.shape
        
        if N < k + 1:
            raise ValueError(f"Sample size {N} is too small for k={k}")
        
        # Add small noise to break ties
        x = self._add_noise(x)
        
        # Find k-nearest neighbors using Euclidean distance
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(x)
        distances, _ = nbrs.kneighbors(x)
        
        # Distance to k-th neighbor (excluding self), this is ε(i)
        epsilon = distances[:, k]  
        
        # Volume constant for d-dimensional unit ball with Euclidean metric
        if d == 1:
            cd = 2.0
        else:
            # For Euclidean metric: cd = π^(d/2) / Γ(d/2 + 1)
            cd = (np.pi**(d/2.0)) / np.math.gamma(d/2.0 + 1)
        
        # Kozachenko-Leonenko estimator (Equation 20 from paper)
        # H(X) = -ψ(k) + ψ(N) + log(cd) + (d/N) * Σlog(ε(i))
        # Note: ε(i) in paper is 2 * distance to k-th neighbor
        h = (-self._digamma(k) + self._digamma(N) + np.log(cd) / np.log(self.base) + 
             d * np.mean(np.log(epsilon) / np.log(self.base)))
        
        return h
    
    def conditional_mutual_information(self, x, y, z):
        """
        Estimate conditional mutual information I(X;Y|Z).
        
        Uses the identity: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        
        Parameters:
        -----------
        x, y, z : array-like
            Input variables
            
        Returns:
        --------
        cmi : float
            Estimated conditional mutual information
        """
        # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        yz = np.column_stack([y, z]) if y.ndim == 1 else np.hstack([y, z])
        mi_xyz = self.mutual_information(x, yz)
        mi_xz = self.mutual_information(x, z)
        
        return mi_xyz - mi_xz
    
    def multivariate_mutual_information(self, variables):
        """
        Estimate multivariate mutual information for multiple variables.
        
        Uses the generalization from the paper (Equations 23 and 30).
        
        Parameters:
        -----------
        variables : list of arrays
            List of variables to compute MI for
            
        Returns:
        --------
        mi : float
            Estimated multivariate mutual information
        """
        if len(variables) < 2:
            raise ValueError("Need at least 2 variables")
        
        # Combine all variables
        X = np.column_stack(variables)
        N = len(X)
        
        # Add noise
        X = self._add_noise(X)
        
        # Find k-nearest neighbors in joint space
        nbrs = NearestNeighbors(n_neighbors=self.k+1, metric='chebyshev').fit(X)
        distances, _ = nbrs.kneighbors(X)
        epsilon = distances[:, self.k]
        
        # Count neighbors in each marginal space
        n_marginal = []
        for i, var in enumerate(variables):
            var = np.atleast_2d(var)
            if var.shape[0] == 1:
                var = var.T
            
            n_i = np.zeros(N)
            for j in range(N):
                dist = np.max(np.abs(var - var[j]), axis=1)
                n_i[j] = np.sum(dist < epsilon[j])
            n_marginal.append(n_i)
        
        # Multivariate MI formula (Algorithm 1 generalization)
        mi = (self._digamma(self.k) + (len(variables) - 1) * self._digamma(N) -
              sum(np.mean(self._digamma(n + 1)) for n in n_marginal))
        
        return max(0, mi)


def test_ksg_estimator():
    """Test the KSG estimator with known distributions."""
    np.random.seed(42)
    
    # Test 1: Independent Gaussian variables (MI should be close to 0)
    n_samples = 2000  # Increase sample size for better estimates
    x = np.random.normal(0, 1, n_samples)
    y = np.random.normal(0, 1, n_samples)
    
    ksg = KSGEstimator(k=5)  # Use k=5 for better bias-variance tradeoff
    mi_independent = ksg.mutual_information(x, y, method=1)
    print(f"MI for independent Gaussians: {mi_independent:.4f} (should be ≈ 0)")
    
    # Test 2: Correlated Gaussian variables
    rho = 0.8
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal(mean, cov, n_samples)
    x_corr, y_corr = data[:, 0], data[:, 1]
    
    # True MI for bivariate Gaussian: -0.5 * log(1 - rho^2)
    true_mi = -0.5 * np.log(1 - rho**2)
    
    mi_method1 = ksg.mutual_information(x_corr, y_corr, method=1)
    mi_method2 = ksg.mutual_information(x_corr, y_corr, method=2)
    
    print(f"True MI for correlated Gaussians (ρ={rho}): {true_mi:.4f}")
    print(f"Estimated MI (method 1): {mi_method1:.4f}")
    print(f"Estimated MI (method 2): {mi_method2:.4f}")
    print(f"Error method 1: {abs(mi_method1 - true_mi):.4f}")
    print(f"Error method 2: {abs(mi_method2 - true_mi):.4f}")
    
    # Test 3: Shannon entropy of Gaussian
    # True entropy of 1D Gaussian: 0.5 * log(2πe * σ²)
    sigma = 1.0
    x_gauss = np.random.normal(0, sigma, n_samples)
    true_entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    estimated_entropy = ksg.entropy(x_gauss)
    
    print(f"True entropy of Gaussian (σ={sigma}): {true_entropy:.4f}")
    print(f"Estimated entropy: {estimated_entropy:.4f}")
    print(f"Entropy error: {abs(estimated_entropy - true_entropy):.4f}")
    
    # Test 4: Test different k values for MI
    print("\nTesting different k values for MI estimation:")
    k_values = [1, 3, 5, 10, 20]
    for k in k_values:
        ksg_k = KSGEstimator(k=k)
        mi_k = ksg_k.mutual_information(x_corr, y_corr, method=1)
        print(f"k={k:2d}: MI = {mi_k:.4f}, Error = {abs(mi_k - true_mi):.4f}")
    
    # Test 5: Verify the conjecture from paper (MI=0 for independent variables)
    print(f"\nTesting conjecture for different distributions:")
    # Independent uniform
    x_unif = np.random.uniform(-1, 1, n_samples)
    y_unif = np.random.uniform(-1, 1, n_samples)
    mi_unif = ksg.mutual_information(x_unif, y_unif, method=1)
    print(f"Independent uniform: MI = {mi_unif:.6f}")
    
    # Independent exponential
    x_exp = np.random.exponential(1, n_samples)
    y_exp = np.random.exponential(2, n_samples)
    mi_exp = ksg.mutual_information(x_exp, y_exp, method=1)
    print(f"Independent exponential: MI = {mi_exp:.6f}")

if __name__ == "__main__":
    test_ksg_estimator()