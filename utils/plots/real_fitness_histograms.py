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

# change to multigrid
def plot_real_fitness_histograms(
    model_paths: List[str],
    damage_paths: List[str],
    model_desc: List[str],
    model_colors: List[str],
    num_bins: int = 2000,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
    ) -> None:
    # num_models = len(model_paths)
    num_damages = len(damage_paths)
    fig, axes = plt.subplots(nrows=2, ncols=num_damages // 2, figsize=(10 * (num_damages // 2), 10))
    axes = np.atleast_2d(axes)

    fig.supxlabel("Corrected Fitness")
    fig.supylabel("Frequency")
    # fig.suptitle("Real Fitness Distribution")

    def filter_top_k_list(data: List, top_k: float = 0.5) -> List:
        sorted_data = sorted(data, reverse=True)
        k_count = max(1, int(len(sorted_data) * top_k))
        return sorted_data[:k_count]

    for i, model_path in enumerate(model_paths):
        for j, damage_path in enumerate(damage_paths):

            eval_metrics = load_json(model_path + damage_path, "eval_metrics.json")
            real_fitnesses = np.array(eval_metrics["global"]["real_fitness"])
            # real_fitnesses = filter_top_k_list(eval_metrics["global"]["real_fitness"], top_k=0.2)
            
            # min_fitness = np.min(real_fitnesses)
            # max_fitness = np.max(real_fitnesses)
            # real_fitnesses = (real_fitnesses - min_fitness) / (max_fitness - min_fitness) * 100 + 0.0
            row = 0 if j < len(damage_paths) // 2 else 1
            col = j if row == 0 else j - len(damage_paths) // 2
            label = model_desc[i] if j == 0 else None
            axes[row][col].hist(
                real_fitnesses, 
                bins=num_bins, 
                range=(lower_bound, upper_bound),
                color=model_colors[i], 
                alpha=0.8,
                label=label,
                density=True
            )
            axes[row][col].set_title(damage_paths[j])

            # kde = gaussian_kde(real_fitness)
            # x_vals = np.linspace(lower_bound, upper_bound, 500)
            # axes[0][j].plot(x_vals, kde(x_vals), color=model_colors[i], linewidth=2)

    fig.legend(bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=len(model_desc))
    plt.savefig("evaluations/real_fitness_histogram.png")
    plt.close()

    return fig, axes