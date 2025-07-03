from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax.numpy as jnp

from util import load_pkls

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from new_plot import plot_multidimensional_map_elites_grid

def plot_live_grid_update(
    repertoire: MapElitesRepertoire,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
    grid_shape: Tuple,
    output_dir: str,
) -> None:
    """
    plot grid changes during ite adaptation, and save the animation to output_path
    """


    plot_multidimensional_map_elites_grid(
        repertoire=repertoire,
        minval=min_bd,
        maxval=max_bd,
        grid_shape=grid_shape,
    )



if __name__ == "__main__":
    # demo
    fig, ax = plt.subplots()
    x = jnp.linspace(0, 2*jnp.pi, 100)
    line, = ax.plot(x, jnp.sin(x))

    def update(frame):
        line.set_ydata(jnp.sin(x + frame * 0.1))
        return line,

    ani = FuncAnimation(fig=fig, func=update, frames=100)
    ani.save("./animation.gif")  # or .mp4

    output_path = "./outputs/dcrl_20250628_173357"
    repertoire, metrics = load_pkls(output_path)
    breakpoint()
