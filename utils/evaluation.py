from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from util import load_json

baseline_colors = [
    "#c6dbef",  # light blue
    "#9ecae1",  # medium light
    "#6baed6",  # mid blue
    "#4292c6",  # strong blue
    "#2171b5",  # dark blue
    "#084594",  # very dark blue
]

main_colors = [
    "#ffcc00",  # vivid yellow
    "#f1c40f",  # golden yellow 
    "#d62728", # highlight red
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

def box_plot_performace(
    exp_paths: List[str],
    exp_desc: List[str],
) -> None:
    """
    args: 
        - exp_paths: experiment directories in list format
        - exp_desc: experiment descriptions in list format
    """
    plt.boxplot()
    plt.savefig("evaluations/performance_box_plot.png")
    plt.close()

def eval_training_metrics(
    exp_paths: List[str],
    exp_desc: List[str],
    exp_colors: List[str],
) -> None:
    """
    args: 
        - exp_paths: experiment directories in list format
        - exp_desc: experiment descriptions in list format
    """

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
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

    for i, exp_path in enumerate(exp_paths):
        metrics = load_json(exp_path, "metrics.json")
        args = load_json(exp_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["batch_size"]
        
        axes[0].plot(env_steps, metrics["coverage"], label=exp_desc[i], color=exp_colors[i])
        axes[1].plot(env_steps, metrics["max_fitness"], color=exp_colors[i])
        axes[2].plot(env_steps, metrics["qd_score"], color=exp_colors[i])

    fig.legend(loc='lower center', ncol=len(exp_desc))
    plt.savefig("evaluations/training_metrics_comparison.png")
    plt.close()


if __name__ == "__main__":
    exp_paths = [
        "outputs/hpc/dcrl_20250723_160932",
        "outputs/hpc/dcrl_20250727_210952",
        "outputs/hpc/dcrl_20250728_180401",
        "outputs/hpc/dcrl_20250731_153529"
    ]

    exp_desc = [
        "no dropouts",
        "dropout_rate=0.2",
        "random physical damage injection, trainiong_damage_rate=0.1",
        "random damage injection, training_damage_rate=0.05"
    ]

    exp_colors = [
        baseline_colors[2],
        baseline_colors[4],
        main_colors[0],
        main_colors[2],
    ]

    eval_training_metrics(exp_paths, exp_desc, exp_colors)