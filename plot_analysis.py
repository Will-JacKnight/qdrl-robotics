from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import os
from scipy.stats import gaussian_kde

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
from utils.plots.qd_metrics import (
    plot_final_corrected_qd_metrics,
    plot_illusory_coverage, 
    plot_illusory_max_fitness, 
    plot_illusory_qd_score,
)
from utils.plots.adaptation_metrics import plot_adaptation_metrics, plot_adaptation_step_speed_distribution
from utils.plots.config import baseline_colors, main_colors, apply_plot_style, adjust_color, ModelInfo, extract_model_attributes

# Apply shared plotting style
apply_plot_style()
    

def eval_multi_model_metrics(
    models: List[ModelInfo],
    damage_paths: List[str],
    ) -> None:
    """
    args: 
        - model_paths: model base directories in list format
        - model_desc: model descriptions in list format
        - model_colors: model colors in list format
    """
    model_paths, model_desc, model_desc_abbr, model_colors = extract_model_attributes(models)

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # axes = axes.flatten()

    # _, axes[0] = plot_illusory_coverage(model_paths, model_desc, model_colors, ax=axes[0])


    # _, axes[0] = plot_illusory_max_fitness(model_paths, model_desc, model_colors, ax=axes[0])
    # _, axes[1] = plot_illusory_qd_score(model_paths, model_desc, model_colors, ax=axes[1])


    # _, axes[0] = plot_final_corrected_qd_metrics(models, "max_fitness", ax=axes[0])
    # _, axes[1] = plot_final_corrected_qd_metrics(models, "qd_score", ax=axes[1])


    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
    axes = axes.flatten()
    secondary_axes = []

    for i in range(len(damage_paths)):
        _, axes[i], ax_secondary = plot_adaptation_metrics(model_paths, model_desc_abbr, model_colors, damage_paths[i], ax=axes[i])
        secondary_axes.append(ax_secondary)

    # Create a single legend for the entire figure using handles from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(model_desc))
    
    plt.savefig("evaluations/eval_multi_model_metrics.png")
    plt.close()



if __name__ == "__main__":

    

    models = [
        ModelInfo(
            # model_path="outputs/hpc/dcrl_20250723_160932/", 
            model_path="outputs/final/dcrl_20250902_213836/",
            model_desc="original DCRL archive without dropouts", 
            model_desc_abbr="original ITE", 
            color="#9b59b6",
            rep_paths=[
                "outputs/final/dcrl_20250902_213836/",
                "outputs/final/dcrl_20250903_153618/",
                "outputs/final/dcrl_20250903_235427/",
            ],
        ),
        ModelInfo(
            # model_path="outputs/hpc/dcrl_20250813_213310/", 
            model_path="outputs/final/dcrl_20250902_213845/",
            model_desc="variant 1: original + dropouts", 
            model_desc_abbr="variant 1", 
            color="#2171b5",
            rep_paths=[
                "outputs/final/dcrl_20250902_213845/",
                "outputs/final/dcrl_20250903_195459/",
                "outputs/final/dcrl_20250903_235430/",

            ],
        ),
        ModelInfo(
            # model_path="outputs/hpc/dcrl_20250816_104912/", 
            model_path="outputs/final/dcrl_20250902_214441/",
            model_desc="variant 2: variant 1 + mapelites-sampling", # (same evaluation steps)
            model_desc_abbr="variant 2", 
            color="#f05d4d",
            rep_paths=[
                "outputs/final/dcrl_20250902_214441/",
                "outputs/final/dcrl_20250903_232733/",
                "outputs/final/dcrl_20250903_235433/",

            ],
        ),
        # ModelInfo(
        #     model_path="outputs/hpc/dcrl_20250816_154412/", 
        #     model_desc="variant 3: dropouts + mapelites-sampling", #  (same addition steps)
        #     model_desc_abbr="variant 3",
        #     color="#f05d4d"
        # ),
        ModelInfo(
            # model_path="outputs/hpc/dcrl_20250825_173441/", 
            # model_path="outputs/hpc/dcrl_20250827_170158/", 
            model_path="outputs/final/dcrl_20250902_214613/",
            model_desc="variant 3: variant 1 + extract-map-elites",
            model_desc_abbr="variant 3", 
            color="#ffcc00",
            rep_paths=[
                "outputs/final/dcrl_20250902_214613/",
                "outputs/final/dcrl_20250903_235425/",
                "outputs/final/dcrl_20250903_235437/",
            ],
        ),
        
    ]

    damage_paths = [
        "physical_damage/FL_loose",
        "physical_damage/BL_loose",
        "physical_damage/BL_BR_loose",
        "physical_damage/FL_BR_loose",
        "sensory_damage/BL",
        "sensory_damage/FL",
        "sensory_damage/Rand1",
        "sensory_damage/Rand2",
    ]
    
    model_paths, model_desc, model_desc_abbr, model_colors = extract_model_attributes(models)

    ###################
    # eval constructed qd archive
    ###################
    # plot_illusory_coverage(model_paths, model_desc, model_colors)
    # plot_illusory_max_fitness(model_paths, model_desc, model_colors)
    # plot_illusory_qd_score(model_paths, model_desc, model_colors)

    plot_final_corrected_qd_metrics(models, "max_fitness")
    plot_final_corrected_qd_metrics(models, "qd_score")
    plot_final_corrected_qd_metrics(models, "coverage")

    eval_multi_model_metrics(models, damage_paths)
    
    plot_real_fitness_histograms(model_paths, damage_paths, model_desc, model_colors, num_bins=110, lower_bound=100, upper_bound=2300)

    ###################
    # evaluate adaptation
    ###################
    plot_recovered_performance(models, model_paths, damage_paths, model_desc, model_desc_abbr, model_colors)

    
    # plot_adaptation_metrics(model_paths, model_desc_abbr, model_colors, damage_paths[0])
    # plot_adaptation_step_speed_distribution(model_paths[2] + damage_paths[0], model_desc[2], model_colors[2])