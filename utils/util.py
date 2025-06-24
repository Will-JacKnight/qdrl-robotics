import pickle


def load_pkls(output_path: str):
    with open(output_path + "/repertoire.pkl", "rb") as f:
        repertoire = pickle.load(f)

    with open(output_path + "/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return repertoire, metrics


def save_pkls(output_path: str, repertoire, metrics) -> None:
    # create of new folder in outputs with parameter as name
    # include brax rendering, repetoire, metrics, plot

    with open(output_path + "/repertoire.pkl", "wb") as f:
        pickle.dump(repertoire, f)

    with open(output_path + "/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)