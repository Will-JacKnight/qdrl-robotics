import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

def update_confidence(grid, x0, y0, amplitude=1., sigma=0.5):
    """Add a 2D Gaussian bump to a grid at (x0, y0) with given amplitude and sigma."""
    x = jnp.arange(grid.shape[0])
    y = jnp.arange(grid.shape[1])
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    gauss = amplitude * jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return grid + gauss

if __name__ == "__main__":
    # Example usage:
    grid = jnp.zeros((100, 100))
    x0, y0 = 50, 50  # Center
    amplitude = 10
    sigma = 5

    grid = update_confidence(grid, x0, y0, amplitude, sigma)

    # Convert to numpy for plotting
    grid_np = jax.device_get(grid)

    # Plot heatmap
    plt.imshow(grid_np, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()