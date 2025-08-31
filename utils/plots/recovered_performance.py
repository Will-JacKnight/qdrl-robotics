from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp


def plot_recovered_performance(
    model_paths: List[str],
    damage_paths: List[str],
    model_desc: List[str],
    model_desc_abbr: List[str],
    model_colors: List[str],
    ax: Optional[Axes] = None,
    ) -> Tuple[Optional[Figure], Axes]:

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig = None

    ax.set_xlabel("Damage cases")
    ax.set_ylabel("Performance (m/s)")
    ax.set_title("Best behavioural performance after ITE adaptation")    
    
    positions = []
    data = []
    color_indices = []
    width = 0.25
    group_spacing = 1.5
    
    for i, model_path in enumerate(model_paths):
        for j, damage_path in enumerate(damage_paths):
            eval_metrics = load_json(model_path + damage_path, "eval_metrics.json")
            positions.append(j * group_spacing + i * width)
            data.append(eval_metrics["iterative"]["step_speeds"][-1])
            color_indices.append(i)

    bp = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True)

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(model_colors[color_indices[i]])

    # center xticks for each damage case
    num_models = len(model_paths)
    centers = [j * group_spacing + (num_models - 1) * width / 2 for j in range(len(damage_paths))]
    ax.set_xticks(centers)
    ax.set_xticklabels(damage_paths)

    # ax.set_xticks()
    handles = [plt.Line2D([0], [0], color=model_colors[i], lw=5) for i in range(len(model_desc))]
    ax.legend(handles, model_desc, loc="upper left")
    plt.savefig("evaluations/final_performances.png")
    plt.close()

    return fig, ax