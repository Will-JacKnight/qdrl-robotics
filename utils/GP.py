import jax.numpy as jnp
import jax
from typing import NamedTuple, Optional

# Kernel functions
def matern52_kernel(x1: jnp.ndarray, x2: jnp.ndarray, lengthscale: float = 1.0):
    """Matern 5/2 kernel"""
    dists = jnp.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=-1)
    scaled_dists = jnp.sqrt(5.0) * dists / lengthscale
    
    kernel_vals = (1.0 + scaled_dists + (5.0/3.0) * (scaled_dists**2)) * jnp.exp(-scaled_dists)
    return kernel_vals

def zero_mean_function(x: jnp.ndarray) -> jnp.ndarray:
    """Zero mean function"""
    return jnp.zeros(x.shape[0])

class Dataset:
    """Base class for datasets.
    
    Args:
        x: input data
        y: output data
    """
    X: jnp.ndarray
    y: jnp.ndarray

    @property
    def n(self) -> int:
        return self.X.shape[0]
    
    def __add__(self, other):
        if self.X.size == 0:
            return other
        if other.X.size == 0:
            return self
        return Dataset(
            X=jnp.concatenate([self.X, other.X], axis=0),
            y=jnp.concatenate([self.y, other.y], axis=0)
        )
    
class GaussianProcess:
    def __init__(self, kernel_fn, mean_fn, noise: float = 1e-3):
        self.kernel_fn = kernel_fn
        self.mean_fn = mean_fn
        self.noise = noise
    
    def predict(self, test_inputs: jnp.ndarray, train_data: Dataset):
        """GP prediction with noise"""
        if train_data.n == 0:
            # No training data, return prior
            mean = self.mean_fn(test_inputs)
            var = jnp.ones_like(mean)
            return {"mean": mean, "variance": var}
        
        # Kernel matrices
        K_train = self.kernel_fn(train_data.X, train_data.X) + self.noise * jnp.eye(train_data.n)
        K_test_train = self.kernel_fn(test_inputs, train_data.X)
        K_test = self.kernel_fn(test_inputs, test_inputs)
        
        # Cholesky decomposition for numerical stability
        L = jnp.linalg.cholesky(K_train + 1e-6 * jnp.eye(train_data.n))
        
        # Solve linear systems
        alpha = jax.scipy.linalg.solve_triangular(L, train_data.y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)
        
        v = jax.scipy.linalg.solve_triangular(L, K_test_train.T, lower=True)
        
        # Predictions
        mean_pred = self.mean_fn(test_inputs) + K_test_train @ alpha
        var_pred = jnp.diag(K_test) - jnp.sum(v**2, axis=0)
        
        return {"mean": mean_pred.squeeze(), "variance": var_pred}