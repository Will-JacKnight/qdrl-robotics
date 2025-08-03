from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import os

import matplotlib.pyplot as plt
import numpy as np

from util import load_json

def box_plot_performace(
    exp_paths: List[str],
) -> None:
    """
    args: 
        - exp_paths: experiment directories in list format
    """
    plt.boxplot()
    plt.savefig("evaluations/performance_box_plot.png")
    plt.close()

def eval_training_metrics(
    exp_paths: List[str],
    exp_desc: List[str],
) -> None:
    """
    args: 
        - exp_paths: experiment directories in list format
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
        metrics = load_json(exp_path, "metrics.json")[0]
        args = load_json(exp_path, "running_args.json")
        env_steps = np.arange(args["num_iterations"] + 1) * args["episode_length"] * args["batch_size"]
        
        axes[0].plot(env_steps, metrics["coverage"], label=exp_desc[i])
        axes[1].plot(env_steps, metrics["max_fitness"])
        axes[2].plot(env_steps, metrics["qd_score"])

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

    eval_training_metrics(exp_paths, exp_desc)