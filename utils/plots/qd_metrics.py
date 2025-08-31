from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp

from utils.util import load_json
from utils.plots.config import apply_plot_style

# Apply shared plotting style
apply_plot_style()


def plot_corrected_coverage(
    model_paths: List[str],
    model_desc: List[str], 
    model_colors: List[str],
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plot corrected coverage evolution during training for multiple models.
    
    Args:
        model_paths: List of model base directories
        model_desc: List of model descriptions for legend
        model_colors: List of colors for each model
        ax: Optional existing axis to plot on
        
    Returns:
        Tuple of (figure, axis) where figure is None if ax was provided
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
        
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Corrected Coverage in %")
    
    for i, model_path in enumerate(model_paths):
        metrics = load_json(model_path, "metrics.json")
        args = load_json(model_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["real_evals_per_iter"]
        
        ax.plot(env_steps, metrics["coverage"], label=model_desc[i], color=model_colors[i])
    
    if fig is not None:
        ax.legend()
    plt.savefig("evaluations/corrected_coverage.png")

    return fig, ax


def plot_corrected_max_fitness(
    model_paths: List[str],
    model_desc: List[str], 
    model_colors: List[str],
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plot corrected maximum fitness evolution during training for multiple models.
    
    Args:
        model_paths: List of model base directories
        model_desc: List of model descriptions for legend
        model_colors: List of colors for each model
        ax: Optional existing axis to plot on
        
    Returns:
        Tuple of (figure, axis) where figure is None if ax was provided
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
        
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Corrected Maximum fitness")
    
    for i, model_path in enumerate(model_paths):
        metrics = load_json(model_path, "metrics.json")
        args = load_json(model_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["real_evals_per_iter"]
        
        ax.plot(env_steps, metrics["max_fitness"], label=model_desc[i], color=model_colors[i])
    
    if fig is not None:
        ax.legend()
    plt.savefig("evaluations/corrected_max_fitness.png")
    return fig, ax


def plot_corrected_qd_score(
    model_paths: List[str],
    model_desc: List[str], 
    model_colors: List[str],
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plot corrected QD score evolution during training for multiple models.
    
    Args:
        model_paths: List of model base directories
        model_desc: List of model descriptions for legend
        model_colors: List of colors for each model
        ax: Optional existing axis to plot on
        
    Returns:
        Tuple of (figure, axis) where figure is None if ax was provided
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
        
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Corrected QD Score")
    
    for i, model_path in enumerate(model_paths):
        metrics = load_json(model_path, "metrics.json")
        args = load_json(model_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["real_evals_per_iter"]
        
        ax.plot(env_steps, metrics["qd_score"], label=model_desc[i], color=model_colors[i])
    
    if fig is not None:
        ax.legend()
    plt.savefig("evaluations/corrected_qd_score.png")

    return fig, ax

