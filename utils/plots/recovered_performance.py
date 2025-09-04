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
from utils.plots.config import apply_plot_style, ModelInfo
from utils.plots.metrics_collector import collect_metrics

# Apply shared plotting style
apply_plot_style()


def plot_recovered_performance(
    models: List[ModelInfo],
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
    ax.set_ylabel("Fitness")
    # ax.set_title("Best behavioural performance after ITE adaptation")    
    
    positions = []
    data = []
    color_indices = []
    width = 0.25
    group_spacing = 1.5
    
    # for i, model_path in enumerate(model_paths):
    #     for j, damage_path in enumerate(damage_paths):
    #         # 1. step rewards distribution
    #         # eval_metrics = load_json(model_path + damage_path, "eval_metrics.json")
    #         # data.append(eval_metrics["iterative"]["step_speeds"][-1])

    #         # 2. fitness distribution via repetition runs 
    #         # rep_metrics = load_json(model_path + damage_path, "rep_metrics.json")
    #         # data.append(rep_metrics["best_real_fitness"])
    #         # positions.append(j * group_spacing + i * width)
    #         # color_indices.append(i) 

    for i, damage_path in enumerate(damage_paths):
        for j, model in enumerate(models):
            rep_metrics = collect_metrics(
                model=model, 
                filename="eval_metrics.json", 
                dict_key="global",
                damage_path=damage_path,
            )
            data.append(rep_metrics["best_real_fitness"])      

            positions.append(i * group_spacing + j * width)
            color_indices.append(j)
            
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
    fig.legend(handles, model_desc, ncol=len(model_desc), loc="lower center") # , loc="upper left"
    plt.savefig("evaluations/final_performances.png")
    plt.close()

    return fig, ax