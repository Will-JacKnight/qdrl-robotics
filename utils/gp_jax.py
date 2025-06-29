"""
Pure JAX implementation of Gaussian Process functionality with GPJax-compatible API
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Any, Dict
import functools


class Dataset:
    """Dataset container matching GPJax API"""
    
    def __init__(self, X: jnp.ndarray = None, y: jnp.ndarray = None):
        if X is None:
            self.X = jnp.empty((0, 1))
        else:
            self.X = X
            
        if y is None:
            self.y = jnp.empty((0, 1))
        else:
            self.y = y
    
    @property
    def n(self) -> int:
        """Number of data points"""
        return self.X.shape[0]
    
    def __add__(self, other):
        """Add datasets together"""
        if self.n == 0:
            return other
        if other.n == 0:
            return self
        
        return Dataset(
            X=jnp.concatenate([self.X, other.X], axis=0),
            y=jnp.concatenate([self.y, other.y], axis=0)
        )


class PredictiveDistribution:
    """Distribution returned by posterior predictions"""
    
    def __init__(self, mean: jnp.ndarray, variance: jnp.ndarray):
        self._mean = mean
        self._variance = variance
    
    def mean(self) -> jnp.ndarray:
        return self._mean
    
    def variance(self) -> jnp.ndarray:
        return self._variance
    
    def stddev(self) -> jnp.ndarray:
        return jnp.sqrt(self._variance)


class LatentDistribution:
    """Latent distribution from GP prediction"""
    
    def __init__(self, mean: jnp.ndarray, variance: jnp.ndarray):
        self._mean = mean
        self._variance = variance
    
    def mean(self) -> jnp.ndarray:
        return self._mean
    
    def variance(self) -> jnp.ndarray:
        return self._variance


class Gaussian:
    """Gaussian likelihood"""
    
    def __init__(self, num_datapoints: int, obs_stddev: float):
        self.num_datapoints = num_datapoints
        self.obs_stddev = obs_stddev
        self.obs_variance = obs_stddev ** 2
    
    def __call__(self, latent_dist: LatentDistribution) -> PredictiveDistribution:
        """Convert latent distribution to predictive distribution"""
        # Add observation noise to the variance
        predictive_variance = latent_dist.variance() + self.obs_variance
        return PredictiveDistribution(latent_dist.mean(), predictive_variance)


class Prior:
    """Gaussian Process Prior"""
    
    def __init__(self, kernel: Callable, mean_function: Callable):
        self.kernel = kernel
        self.mean_function = mean_function
        self.likelihood = None
    
    def __mul__(self, likelihood: Gaussian):
        """Create posterior by multiplying prior with likelihood"""
        return Posterior(self.kernel, self.mean_function, likelihood)


class Posterior:
    """Gaussian Process Posterior"""
    
    def __init__(self, kernel: Callable, mean_function: Callable, likelihood: Gaussian):
        self.kernel = kernel
        self.mean_function = mean_function
        self.likelihood_obj = likelihood
    
    def predict(self, test_inputs: jnp.ndarray, train_data: Dataset) -> LatentDistribution:
        """GP prediction returning latent distribution"""
        if train_data.n == 0:
            # No training data, return prior
            mean = self.mean_function(test_inputs)
            var = jnp.ones_like(mean)
            return LatentDistribution(mean, var)
        
        # Compute kernel matrices
        K_train = self.kernel(train_data.X, train_data.X)
        K_test_train = self.kernel(test_inputs, train_data.X)
        K_test = self.kernel(test_inputs, test_inputs)
        
        # Add noise to training kernel
        noise_var = self.likelihood_obj.obs_variance
        K_train_noisy = K_train + noise_var * jnp.eye(train_data.n)
        
        # Numerical stability
        jitter = 1e-6
        K_train_noisy = K_train_noisy + jitter * jnp.eye(train_data.n)
        
        # Cholesky decomposition for numerical stability
        try:
            L = jnp.linalg.cholesky(K_train_noisy)
        except:
            # Fallback to adding more jitter
            K_train_noisy = K_train_noisy + 1e-4 * jnp.eye(train_data.n)
            L = jnp.linalg.cholesky(K_train_noisy)
        
        # Mean function evaluation
        mean_train = self.mean_function(train_data.X)
        mean_test = self.mean_function(test_inputs)
        
        # Residuals
        y_residual = train_data.y.squeeze() - mean_train
        
        # Solve linear systems
        alpha = jax.scipy.linalg.solve_triangular(L, y_residual, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)
        
        v = jax.scipy.linalg.solve_triangular(L, K_test_train.T, lower=True)
        
        # Predictions
        mean_pred = mean_test + K_test_train @ alpha
        var_pred = jnp.diag(K_test) - jnp.sum(v**2, axis=0)
        
        # Ensure variance is positive
        var_pred = jnp.maximum(var_pred, 1e-8)
        
        return LatentDistribution(mean_pred, var_pred)
    
    def likelihood(self, latent_dist: LatentDistribution) -> PredictiveDistribution:
        """Convert latent to predictive distribution using likelihood"""
        return self.likelihood_obj(latent_dist)


class kernels:
    @staticmethod
    def Matern52(lengthscale: float = 1.0):
        """
        The Matérn kernel with smoothness parameter fixed at 2.5.


        Computes the covariance for pairs of inputs $(x, y)$ with
        lengthscale parameter $\ell$ and variance $\sigma^2$.

        $$
        k(x, y) = \sigma^2 \exp \Bigg(1+ \frac{\sqrt{5}\lvert x-y \rvert}{\ell^2} + \frac{5\lvert x - y \rvert^2}{3\ell^2} \Bigg)\exp\Bigg(-\frac{\sqrt{5}\lvert x-y\rvert}{\ell^2} \Bigg)
        $$
        """
        def kernel_fn(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
            # Handle single points
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1) 
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)
                
            # Compute pairwise distances
            dists = jnp.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=-1)
            scaled_dists = jnp.sqrt(5.0) * dists / lengthscale
            
            # Matern 5/2 formula
            kernel_vals = (1.0 + scaled_dists + (scaled_dists**2) / 3.0) * jnp.exp(-scaled_dists)
            return kernel_vals
        
        return kernel_fn


class mean_functions:
    @staticmethod
    def Zero():
        """Zero mean function"""
        def mean_fn(x: jnp.ndarray) -> jnp.ndarray:
            if x.ndim == 1:
                return jnp.zeros(1)
            return jnp.zeros(x.shape[0])
        return mean_fn


class likelihoods:
    @staticmethod
    def Gaussian(num_datapoints: int, obs_stddev: float):
        """Gaussian likelihood"""
        return Gaussian(num_datapoints, obs_stddev)


class gps:
    @staticmethod
    def Prior(kernel: Callable, mean_function: Callable):
        """Create a GP prior"""
        return Prior(kernel, mean_function) 