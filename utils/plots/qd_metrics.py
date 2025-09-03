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
from utils.plots.metrics_collector import collect_metrics
from plot_analysis import ModelInfo

# Apply shared plotting style
apply_plot_style()


def plot_illusory_coverage(
    model_paths: List[str],
    model_desc: List[str], 
    model_colors: List[str],
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plot illusory coverage evolution during training for multiple models.
    
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
    ax.set_ylabel("Illusory Coverage in %")
    
    for i, model_path in enumerate(model_paths):
        metrics = load_json(model_path, "metrics.json")
        args = load_json(model_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["real_evals_per_iter"]
        
        ax.plot(env_steps, metrics["coverage"], label=model_desc[i], color=model_colors[i])
    
    if fig is not None:
        ax.legend()
    plt.savefig("evaluations/illusory_coverage.png")

    return fig, ax


def plot_illusory_max_fitness(
    model_paths: List[str],
    model_desc: List[str], 
    model_colors: List[str],
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plot illusory maximum fitness evolution during training for multiple models.
    
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
    ax.set_ylabel("Illusory Maximum fitness")
    
    for i, model_path in enumerate(model_paths):
        metrics = load_json(model_path, "metrics.json")
        args = load_json(model_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["real_evals_per_iter"]
        
        ax.plot(env_steps, metrics["max_fitness"], label=model_desc[i], color=model_colors[i])
    
    if fig is not None:
        ax.legend()
    plt.savefig("evaluations/illusory_max_fitness.png")
    return fig, ax


def plot_illusory_qd_score(
    model_paths: List[str],
    model_desc: List[str], 
    model_colors: List[str],
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plot illusory QD score evolution during training for multiple models.
    
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
    ax.set_ylabel("Illusory QD Score")
    
    for i, model_path in enumerate(model_paths):
        metrics = load_json(model_path, "metrics.json")
        args = load_json(model_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["real_evals_per_iter"]
        
        ax.plot(env_steps, metrics["qd_score"], label=model_desc[i], color=model_colors[i])
    
    if fig is not None:
        ax.legend()
    plt.savefig("evaluations/illusory_qd_score.png")

    return fig, ax


def plot_final_corrected_qd_metrics(
    models: List[ModelInfo],
    metrics: str,
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
        
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Corrected Max Fitness")

    data = []
    color_indices = []

    for model in models:
        rep_metrics = collect_metrics(model)
    
        data.append(rep_metrics[metrics])
        color_indices.append(i)

    bp = ax.boxplot(data, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(model_colors[color_indices[i]])

    # center xticks for each damage case
    num_models = len(model_paths)
    ax.set_xticklabels(model_desc_abbr)

    handles = [plt.Line2D([0], [0], color=model_colors[i], lw=5) for i in range(num_models)]
    if fig is not None:
        ax.legend(handles, model_desc, loc="upper left")
    
    plt.savefig(f"evaluations/final_corrected_{metrics}.png")
    plt.close()
    return fig, ax