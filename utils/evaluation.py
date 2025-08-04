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

def eval_single_model_metrics(
    model_path: str,
    model_desc: str,
) -> None:
    """
    args: 
        - model_path: model base directory
        - model_desc: model description
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    axes = axes.flatten()

    axes[0].set_xlabel("Adaptation steps")
    axes[0].set_ylabel("Performance (m/s)")
    axes[0].set_title("Performance distribution during ITE adaptation")

    # speeds boxplot for every adaptation step
    eval_metrics = load_json(model_path, "eval_metrics.json")
    num_steps = len(eval_metrics["iterative"]["step_speeds"])
    blues = mpl.colormaps.get_cmap("blues", num_steps)
    colors = [blues(i) for i in range(num_steps)]
    for i, step in enumerate(eval_metrics["iterative"]["step_speeds"]):
        axes[0].boxplot(
            step, 
            positions=[i], 
            boxprops=dict(facecolor=colors[i])
        )

    fig.legend(loc='lower center', ncol=len(model_desc))
    plt.savefig("evaluations/eval_single_model_metrics.png")
    plt.close()
    


def eval_multi_model_metrics(
    model_paths: List[str],
    model_desc: List[str],
    model_colors: List[str],
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
    ax5_secondary = axes[4].twinx()
    axes[4].set_ylabel("Number of trials")
    ax5_secondary.set_ylabel("Adaptation time (s)")
    axes[4].set_title("Number of ITE trials")
    ax5_secondary.tick_params(axis='y', colors=main_colors[1])

    # axes[5].set_xlabel("ITE Variants")
    # axes[5].set_ylabel("Best k Performances (m/s)")
    # axes[5].set_title("Top k behavioural performance after ITE adaptation")  

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
        eval_metrics = load_json(model_path, "eval_metrics.json")
        axes[3].boxplot(eval_metrics["iterative"]["step_speeds"][-1], label=model_desc[i], color=model_colors[i])

        axes[4].scatter(model_desc, eval_metrics["global"]["adaptation_steps"], color=baseline_colors[2])
        ax5_secondary.scatter(model_desc, eval_metrics["global"]["adaptation_time"], color=main_colors[1])

        # axes[5].boxplot()

    fig.legend(loc='lower center', ncol=len(model_desc))
    plt.savefig("evaluations/eval_multi_model_metrics.png")
    plt.close()


if __name__ == "__main__":

    model_paths = [
        "outputs/hpc/dcrl_20250723_160932",
        "outputs/hpc/dcrl_20250727_210952",
        "outputs/hpc/dcrl_20250728_180401",
        "outputs/hpc/dcrl_20250731_153529",
    ]

    model_desc = [
        "no dropouts",
        "dropout_rate=0.2",
        "random physical damage injection, trainiong_damage_rate=0.1",
        "random damage injection, training_damage_rate=0.05"
    ]

    model_colors = [
        baseline_colors[2],
        baseline_colors[4],
        main_colors[0],
        main_colors[2],
    ]

    model_paths = [path + "/physical_damage/FL_loose" for path in model_paths]
    eval_single_model_metrics(model_paths[2], model_desc[2])
    eval_multi_model_metrics(model_paths, model_desc, model_colors)