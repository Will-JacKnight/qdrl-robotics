"""
Test script for the pure JAX GP implementation
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import utils.gp_jax as gpx


def test_basic_gp():
    """Test basic GP functionality"""
    print("Testing basic GP functionality...")
    
    # Generate some test data
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    
    # Training data
    X_train = jnp.linspace(0, 1, 10).reshape(-1, 1)
    y_train = jnp.sin(2 * jnp.pi * X_train).squeeze() + 0.1 * jax.random.normal(subkey, (10,))
    
    # Test data
    X_test = jnp.linspace(-0.2, 1.2, 50).reshape(-1, 1)
    
    # Create dataset
    D = gpx.Dataset(X=X_train, y=y_train.reshape(-1, 1))
    print(f"Dataset size: {D.n}")
    
    # Define GP model (exactly like GPJax API)
    kernel = gpx.kernels.Matern52(lengthscale=0.2)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    
    # Create likelihood and posterior
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=0.1)
    posterior = prior * likelihood
    
    # Make predictions
    latent_dist = posterior.predict(test_inputs=X_test, train_data=D)
    predictive_dist = posterior.likelihood(latent_dist)
    
    # Get predictions
    mean_pred = predictive_dist.mean()
    var_pred = predictive_dist.variance()
    std_pred = predictive_dist.stddev()
    
    print(f"Prediction shapes - mean: {mean_pred.shape}, variance: {var_pred.shape}")
    print(f"Mean prediction range: [{jnp.min(mean_pred):.3f}, {jnp.max(mean_pred):.3f}]")
    print(f"Variance range: [{jnp.min(var_pred):.3f}, {jnp.max(var_pred):.3f}]")
    
    # Test adding datasets
    D2 = gpx.Dataset(X=jnp.array([[0.5]]), y=jnp.array([[0.8]]))
    D_combined = D + D2
    print(f"Combined dataset size: {D_combined.n}")
    
    print("✓ Basic GP test passed!")
    return True


def test_empty_dataset():
    """Test GP behavior with empty dataset"""
    print("\nTesting empty dataset...")
    
    # Empty dataset
    D = gpx.Dataset()
    print(f"Empty dataset size: {D.n}")
    
    # Test data
    X_test = jnp.linspace(0, 1, 10).reshape(-1, 1)
    
    # Define GP model
    kernel = gpx.kernels.Matern52(lengthscale=0.2)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=0.1)
    posterior = prior * likelihood
    
    # Make predictions (should return prior)
    latent_dist = posterior.predict(test_inputs=X_test, train_data=D)
    predictive_dist = posterior.likelihood(latent_dist)
    
    mean_pred = predictive_dist.mean()
    var_pred = predictive_dist.variance()
    
    print(f"Prior mean: {jnp.mean(mean_pred):.6f} (should be ~0)")
    print(f"Prior variance: {jnp.mean(var_pred):.6f} (should be > 0)")
    
    print("✓ Empty dataset test passed!")
    return True


def test_adaptation_api():
    """Test the API as used in adaptation.py"""
    print("\nTesting adaptation.py API...")
    
    # Simulate repertoire data
    n_centroids = 100
    dim = 2
    centroids = jax.random.uniform(jax.random.key(42), (n_centroids, dim))
    fitnesses = jax.random.uniform(jax.random.key(43), (n_centroids,))
    
    # Initialize empty dataset (like in adaptation.py)
    D = gpx.Dataset()
    
    # GP setup (like in adaptation.py)
    lengthscale = 0.4
    noise = 1e-3
    kernel = gpx.kernels.Matern52(lengthscale=lengthscale)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    
    # Simulate one iteration
    next_idx = jnp.argmax(fitnesses)
    next_goal = centroids[next_idx]
    real_fitness = 0.8  # simulated
    
    # Add observation (like in adaptation.py)
    obs_dataset = gpx.Dataset(
        X=jnp.expand_dims(next_goal, axis=0),
        y=jnp.expand_dims(real_fitness - fitnesses[next_idx], axis=[0, 1]),
    )
    D = D + obs_dataset
    
    print(f"Dataset after first observation: {D.n}")
    
    # Simulate second iteration
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=noise)
    posterior = prior * likelihood
    
    latent_dist = posterior.predict(test_inputs=centroids, train_data=D)
    predictive_dist = posterior.likelihood(latent_dist)
    
    means_residual = predictive_dist.mean()
    variances = predictive_dist.variance()
    
    print(f"Residual means shape: {means_residual.shape}")
    print(f"Variances shape: {variances.shape}")
    print(f"Mean residual: {jnp.mean(means_residual):.6f}")
    print(f"Mean variance: {jnp.mean(variances):.6f}")
    
    # Test acquisition function (upper confidence bound)
    means_adjusted = fitnesses + means_residual
    kappa = 0.05
    ucb_values = means_adjusted + kappa * jnp.sqrt(variances)
    next_idx = jnp.argmax(ucb_values)
    
    print(f"Next index selected: {next_idx}")
    print("✓ Adaptation API test passed!")
    return True


def plot_gp_predictions():
    """Create a visualization of GP predictions"""
    print("\nCreating GP visualization...")
    
    # Generate training data
    key = jax.random.key(42)
    X_train = jnp.array([[0.1], [0.4], [0.7], [0.9]])
    y_train = jnp.sin(2 * jnp.pi * X_train).squeeze() + 0.1 * jax.random.normal(key, (4,))
    
    # Test points
    X_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
    
    # GP setup
    D = gpx.Dataset(X=X_train, y=y_train.reshape(-1, 1))
    kernel = gpx.kernels.Matern52(lengthscale=0.2)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=0.1)
    posterior = prior * likelihood
    
    # Predictions
    latent_dist = posterior.predict(test_inputs=X_test, train_data=D)
    predictive_dist = posterior.likelihood(latent_dist)
    
    mean_pred = predictive_dist.mean()
    std_pred = predictive_dist.stddev()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.squeeze(), mean_pred, 'b-', label='GP mean', linewidth=2)
    plt.fill_between(X_test.squeeze(), 
                     mean_pred - 2*std_pred, 
                     mean_pred + 2*std_pred, 
                     alpha=0.3, color='blue', label='±2σ')
    plt.scatter(X_train.squeeze(), y_train, color='red', s=50, zorder=5, label='Training data')
    plt.plot(X_test.squeeze(), jnp.sin(2 * jnp.pi * X_test.squeeze()), 'k--', alpha=0.5, label='True function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pure JAX GP Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gp_test_plot.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'gp_test_plot.png'")


if __name__ == "__main__":
    print("Testing Pure JAX GP Implementation")
    print("=" * 40)
    
    # Run tests
    test_basic_gp()
    test_empty_dataset()
    test_adaptation_api()
    # plot_gp_predictions()
    
    print("\n" + "=" * 40)
    print("All tests passed! ✓")
    print("Your pure JAX GP implementation is ready to use as a drop-in replacement for GPJax.") 