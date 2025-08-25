from typing import Dict, Any, Optional
import os
import pickle
import json
import jax.numpy as jnp
import numpy as np


def load_repertoire_and_metrics(
    output_path: str,
):
    with open(output_path + "/repertoire.pkl", "rb") as f:
        repertoire = pickle.load(f)

    metrics = load_json(output_path, "metrics.json")
    return repertoire, metrics


def save_repertoire_and_metrics(
    output_path: str, 
    repertoire,
    metrics: Dict,
) -> None:
    os.makedirs(output_path, exist_ok=True) 

    with open(output_path + "/repertoire.pkl", "wb") as f:
        pickle.dump(repertoire, f)
    
    log_metrics(output_path, "metrics.json", metrics)

def save_args(
    args: Any,
    filename: Optional[str] = "running_args.json",
):
    os.makedirs(args.output_path, exist_ok=True) 

    # Convert argparse.Namespace to a plain dictionary
    args_dict = vars(args)

    with open(os.path.join(args.output_path, filename), "w") as f:
        json.dump(args_dict, f, indent=4)

def log_metrics(
    exp_path: str, 
    filename: str,
    metrics: Dict,
):
    path = os.path.join(exp_path, filename)

    # if os.path.exists(path):
    #     with open(path, "r") as f:
    #         all_metrics = json.load(f)
    # else:
    #     all_metrics = []

    def convert_to_json_serializable(obj):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (jnp.generic, np.generic)):  # scalars like jnp.float32
            return obj.item()
        else:
            return obj
    
    serializable_metrics = convert_to_json_serializable(metrics)
    # all_metrics.append(serializable_metrics)

    with open(path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)

def load_json(
    path: str,
    filename: str,
):
    path = os.path.join(path, filename)
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    base_path = "outputs/hpc"
    for subdir in os.listdir(base_path):
        output_path = os.path.join(base_path, subdir)
        if not os.path.isdir(output_path):
            continue  # skip files

        metrics_path = os.path.join(output_path, "metrics.pkl")
        if not os.path.exists(metrics_path):
            print(f"Skipping {output_path}: no metrics.pkl found.")
            continue
        
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
        log_metrics(output_path, "metrics.json", metrics)