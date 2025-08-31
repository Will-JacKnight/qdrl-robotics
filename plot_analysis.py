from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import os
from scipy.stats import gaussian_kde
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp

from utils.util import load_json
from utils.plots.recovered_performance import plot_recovered_performance
from utils.plots.real_fitness_histograms import plot_real_fitness_histograms
from utils.plots.qd_metrics import plot_corrected_coverage, plot_corrected_max_fitness, plot_corrected_qd_score
from utils.plots.adaptation_metrics import plot_adaptation_metrics, plot_adaptation_step_speed_distribution
from utils.plots.config import baseline_colors, main_colors, apply_plot_style, adjust_color

# Apply shared plotting style
apply_plot_style()



def eval_multi_model_metrics(
    model_paths: List[str],
    model_desc: List[str],
    model_desc_abbr: List[str],
    model_colors: List[str],
    damage_path: str,
    ) -> None:
    """
    args: 
        - model_paths: model base directories in list format
        - model_desc: model descriptions in list format
        - model_colors: model colors in list format
    """

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    axes = axes.flatten()

    _, axes[0] = plot_corrected_coverage(model_paths, model_desc, model_colors, ax=axes[0])
    _, axes[1] = plot_corrected_max_fitness(model_paths, model_desc, model_colors, ax=axes[1])
    _, axes[2] = plot_corrected_qd_score(model_paths, model_desc, model_colors, ax=axes[2])
    _, axes[3], ax4_secondary = plot_adaptation_metrics(model_paths, model_desc_abbr, model_colors, damage_path, ax=axes[3])

    fig.legend() # loc='lower center', ncol=len(model_desc)
    plt.savefig("evaluations/eval_multi_model_metrics.png")
    plt.close()



if __name__ == "__main__":

    @dataclass
    class ModelInfo:
        model_path: str
        model_desc: str
        model_desc_abbr: str
        color: str

    def extract_model_attributes(models: List[ModelInfo]):
        model_paths  = [m.model_path for m in models]
        model_desc   = [m.model_desc for m in models]
        model_abbr   = [m.model_desc_abbr for m in models]
        model_colors = [m.color for m in models]
        return model_paths, model_desc, model_abbr, model_colors

    models = [
        ModelInfo(
            model_path="outputs/hpc/dcrl_20250723_160932/", 
            model_desc="original ITE: no dropouts", 
            model_desc_abbr="original ITE", 
            color="#9b59b6"
        ),
        ModelInfo(
            model_path="outputs/hpc/dcrl_20250813_213310/", 
            model_desc="variant 1: dropouts", 
            model_desc_abbr="variant 1", 
            color="#2171b5"
        ),
        # ModelInfo(
        #     model_path="outputs/hpc/dcrl_20250816_104912/", 
        #     model_desc="variant 2: dropouts + mapelites-sampling", # (same evaluation steps)
        #     model_desc_abbr="variant 2", 
        #     color="#f05d4d"
        # ),
        # ModelInfo(
        #     model_path="outputs/hpc/dcrl_20250816_154412/", 
        #     model_desc="variant 3: dropouts + mapelites-sampling", #  (same addition steps)
        #     model_desc_abbr="variant 3",
        #     color="#f05d4d"
        # ),
        # ModelInfo(
        #     model_path="outputs/hpc/dcrl_20250825_173441/", 
        #     model_desc="variant 4: dropouts + extract-map-elites",
        #     model_desc_abbr="variant 4", 
        #     color="#fec126"
        # ),
        ModelInfo(
            model_path="outputs/hpc/dcrl_20250827_170158/", 
            model_desc="variant 4: dropouts + extract-map-elites",
            model_desc_abbr="variant 4", 
            color="#f05d4d"
        ),
        
    ]

    damage_paths = [
        "physical_damage/FL_loose",
        # "physical_damage/BL_loose",
        "physical_damage/BL_BR_loose",
        # "physical_damage/FL_BR_loose",
        "sensory_damage/BL",
        # "sensory_damage/FL",
        # "sensory_damage/Rand1",
        "sensory_damage/Rand2",
    ]
    
    model_paths, model_desc, model_desc_abbr, model_colors = extract_model_attributes(models)

    plot_corrected_coverage(model_paths, model_desc, model_colors)
    plot_corrected_max_fitness(model_paths, model_desc, model_colors)
    plot_corrected_qd_score(model_paths, model_desc, model_colors)
    plot_adaptation_metrics(model_paths, model_desc_abbr, model_colors, damage_paths[0])

    plot_real_fitness_histograms(model_paths, damage_paths, model_desc, model_colors, num_bins=110, lower_bound=100, upper_bound=2300)
    # eval_multi_model_metrics(model_paths, model_desc, model_desc_abbr, model_colors, damage_paths[0])  
    
    # plot_recovered_performance(model_paths, damage_paths, model_desc, model_desc_abbr, model_colors)
    # plot_adaptation_step_speed_distribution(model_paths[2] + damage_paths[0], model_desc[2], model_colors[2])