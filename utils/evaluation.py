from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np

from util import load_json

baseline_colors = [
    "#c6dbef",  # light blue
    "#9ecae1",  # medium light
    "#6baed6",  # mid blue
    "#4292c6",  # strong blue
    "#2171b5",  # dark blue
    "#084594",  # very dark blue
    "#9b59b6",  # medium-light purple
]

main_colors = [
    "#ffcc00",  # vivid yellow
    "#f1c40f",  # golden yellow 
    "#d62728", # highlight red
    "#2ecc71",  # mint green
    "#e67e22",  # "Carrot Orange"
]

# set the parameters
font_size = 14
params = {
    "axes.labelsize": font_size,
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "text.usetex": False,
    "figure.figsize": [10, 10],
}

mpl.rcParams.update(params)

def adjust_color(color, amount=1.1):
    """
    Darken or lighten a color by a given factor.
    Args:
        - amount: > 1 lighter, < 1 darker
    """
    c = mcolors.to_rgb(color)
    return tuple(min(1, max(0, channel * amount)) for channel in c)

def eval_single_model_metrics(
    damage_path: str,
    model_desc: str,
    model_color: str,
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    args: 
        - model_path: model base directory
        - model_desc: model description
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # ax.set_xlabel("Adaptation steps")
    # ax.set_ylabel("Performance (m/s)")
    # ax.set_title("Performance distribution during ITE adaptation")

    # speeds boxplot for every adaptation step
    eval_metrics = load_json(damage_path, "eval_metrics.json")

    bp = ax.boxplot(eval_metrics["iterative"]["step_speeds"], patch_artist=True)
    num_steps = len(eval_metrics["iterative"]["step_speeds"])
    ax.set_xticks(np.arange(1, num_steps + 1))
    ax.set_xticklabels(np.arange(0, num_steps))
    for box in bp['boxes']:
        box.set_facecolor(model_color)

    return fig, ax


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

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))
    axes = axes.flatten()

    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Coverage in %")
    axes[0].set_title("Coverage evolution during training")

    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Maximum fitness")
    axes[1].set_title("Maximum fitness evolution during training")

    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("QD Score")
    axes[2].set_title("QD Score evolution during training")

    axes[3].set_xlabel("ITE Variants")
    axes[3].set_ylabel("Performance (m/s)")
    axes[3].set_title("Best behavioural performance after ITE adaptation")    

    axes[4].set_xlabel("ITE Variants")
    # axes[4].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[4].set_xticks(np.arange(len(model_desc_abbr)))
    axes[4].set_xticklabels(model_desc_abbr)
    ax5_secondary = axes[4].twinx()
    axes[4].set_ylabel("Number of trials")
    ax5_secondary.set_ylabel("Adaptation time (s)")
    axes[4].set_title("Number of ITE trials / Adaptation time (s)")
    # axes[4].tick_params(axis='y', colors=baseline_colors[3])
    # ax5_secondary.tick_params(axis='y', colors=main_colors[1])
    width = 0.25

    # axes[5].set_xlabel("ITE Variants")
    # axes[5].set_ylabel("Best k Performances (m/s)")
    # axes[5].set_title("Top k behavioural performance after ITE adaptation")  
    axes[5].set_xlabel("Adaptation steps")
    axes[5].set_ylabel("Performance (m/s)")
    axes[5].set_title("Performance distribution during ITE adaptation (FL loose)")

    list_step_speeds = []
    # algo comparison
    for i, model_path in enumerate(model_paths):
        # eval training results
        metrics = load_json(model_path, "metrics.json")
        args = load_json(model_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["batch_size"]
        
        axes[0].plot(env_steps, metrics["coverage"], label=model_desc[i], color=model_colors[i])
        axes[1].plot(env_steps, metrics["max_fitness"], color=model_colors[i])
        axes[2].plot(env_steps, metrics["qd_score"], color=model_colors[i])

        # eval final adaptation results
        eval_metrics = load_json(model_path + damage_path, "eval_metrics.json")
        list_step_speeds.append(eval_metrics["iterative"]["step_speeds"][-1])

        axes[4].bar(i - width / 2, eval_metrics["global"]["adaptation_steps"], width=width, color=model_colors[i])
        ax5_secondary.bar(i + width / 2, eval_metrics["global"]["adaptation_time"], width=width, color=adjust_color(model_colors[i]))

    bp = axes[3].boxplot(list_step_speeds, labels=model_desc_abbr, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(model_colors[i])

    _, axes[5] = eval_single_model_metrics(model_paths[2] + damage_path, model_desc[2], model_colors[2], ax=axes[5])

    fig.legend(loc='lower center') # , ncol=len(model_desc)
    plt.savefig("evaluations/eval_multi_model_metrics.png")
    plt.close()


if __name__ == "__main__":

    model_paths = [
        "outputs/hpc/dcrl_20250723_160932",
        "outputs/hpc/dcrl_20250727_210952",
        "outputs/hpc/dcrl_20250811_152438",
        "outputs/hpc/dcrl_20250811_174154",
        # "outputs/hpc/dcrl_20250728_180401",
        # "outputs/hpc/dcrl_20250731_153529",
        # "outputs/hpc/dcrl_20250801_171556",
    ]

    model_desc = [
        "original ITE: no dropouts",
        "variant 1: dropout_rate=0.2",
        "variant 2: variant 1 + ResNet",
        "variant 3: variant 2 + LayerNorm"
        # "variant 2: variant 1 + random physical damage injection, trainiong_damage_rate=0.1",
        # "variant 3: variant 1 + random damage injection, training_damage_rate=0.05 (high intensity)",
        # "variant 4: variant 1 + random damage injection, training_damage_rate=0.05 (medium intensity)"
    ]

    model_desc_abbr = [
        "original ITE",
        "variant 1",
        "variant 2",
        "variant 3",
        # "variant 4",
    ]

    model_colors = [
        baseline_colors[2],
        baseline_colors[4],
        main_colors[3],
        main_colors[1],
        # main_colors[4],
    ]

    damage_paths = [
        "/physical_damage/FL_loose",
        "/physical_damage/BL_loose",
        "/physical_damage/BL_BR_loose",
        "/physical_damage/FL_BR_loose",
        "/sensory_damage/BL",
        "/sensory_damage/FL",
        "/sensory_damage/Rand1",
        "/sensory_damage/Rand2",
    ]

    # model_paths = [path + "/physical_damage/FL_loose" for path in model_paths]
    # eval_single_model_metrics(model_paths[2], model_desc[2])
    eval_multi_model_metrics(model_paths, model_desc, model_desc_abbr, model_colors, damage_paths[0])