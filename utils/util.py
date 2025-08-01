from typing import Dict
import os
import pickle
import json
import jax.numpy as jnp
import numpy as np


def load_pkls(output_path: str):
    with open(output_path + "/repertoire.pkl", "rb") as f:
        repertoire = pickle.load(f)

    with open(output_path + "/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return repertoire, metrics


def save_pkls(output_path: str, repertoire, metrics) -> None:
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True) 

    with open(output_path + "/repertoire.pkl", "wb") as f:
        pickle.dump(repertoire, f)

    with open(output_path + "/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

def save_args(args):
    # Create directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True) 

    # Convert argparse.Namespace to a plain dictionary
    args_dict = vars(args)

    # Save as JSON
    with open(os.path.join(args.output_path, "running_args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

def log_metrics(exp_path: str, eval_metrics: Dict):
    path = os.path.join(exp_path, "eval_metrics.json")

    if os.path.exists(path):
        with open(path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []

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
    
    serializable_metrics = convert_to_json_serializable(eval_metrics)
    all_metrics.append(serializable_metrics)

    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)

def load_metrics(exp_path: str):
    with open(f"{exp_path}/eval_metrics.json", "r") as f:
        return json.load(f)