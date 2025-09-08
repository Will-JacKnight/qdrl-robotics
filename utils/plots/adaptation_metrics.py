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
from utils.plots.config import apply_plot_style, adjust_color, ModelInfo

# Apply shared plotting style
apply_plot_style()


def plot_adaptation_metrics(
    model_paths: List[str],
    model_desc: List[str],
    model_desc_abbr: List[str], 
    model_colors: List[str],
    damage_path: str,
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes, Axes]:
    """
    Plot adaptation steps and adaptation time metrics for multiple models.
    
    Args:
        model_paths: List of model base directories
        model_desc_abbr: List of abbreviated model descriptions for x-axis labels
        model_colors: List of colors for each model
        damage_path: Damage path to append to model paths
        ax: Optional existing axis to plot on
        
    Returns:
        Tuple of (figure, primary_axis, secondary_axis) where figure is None if ax was provided
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
        
    ax.set_xlabel("ITE Variants")
    ax.set_xticks(np.arange(len(model_desc_abbr)))
    ax.set_xticklabels(model_desc_abbr)
    ax_secondary = ax.twinx()
    ax.set_ylabel("Number of trials")
    ax_secondary.set_ylabel("Adaptation time (s)")
    ax.set_title(damage_path)
    
    width = 0.25
    
    for i, model_path in enumerate(model_paths):
        # eval final adaptation results
        eval_metrics = load_json(model_path + damage_path, "eval_metrics.json")
        
        ax.bar(i - width / 2, eval_metrics["global"]["adaptation_steps"], width=width, color=model_colors[i])
        ax_secondary.bar(i + width / 2, eval_metrics["global"]["adaptation_time"], width=width, color=adjust_color(model_colors[i]))
    
    plt.savefig("evaluations/adaptation_metrics.png")
    return fig, ax, ax_secondary


def plot_adaptation_step_speed_distribution(
    model: ModelInfo,
    damage_paths: str,
    ax: Optional[Axes] = None,
    ) -> Tuple[Optional[Figure], Axes]:
    """
    Plot performance distribution during ITE adaptation steps.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    
    ax.set_xlabel("Number of Adaptation Steps")
    ax.set_ylabel("Forward Velocity (m/s)")
    # ax.set_title(f"Performance distribution during ITE adaptation ({damage_path})")

    # speeds boxplot for every adaptation step
    for damage_path in damage_paths:
        eval_metrics = load_json(damage_path, "eval_metrics.json")

        bp = ax.boxplot(eval_metrics["iterative"]["step_speeds"], patch_artist=True)
        num_steps = len(eval_metrics["iterative"]["step_speeds"])
        ax.set_xticks(np.arange(1, num_steps + 1))
        ax.set_xticklabels(np.arange(0, num_steps))
        for box in bp['boxes']:
            box.set_facecolor(model.color)

    plt.savefig(f"evaluations/appendix/{model.model_desc_abbr}.png")
    return fig, ax
