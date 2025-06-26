import os
import pickle


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