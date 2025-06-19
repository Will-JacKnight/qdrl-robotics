import pickle


def load_pkls():
    with open("repertoire.pkl", "rb") as f:
        repertoire = pickle.load(f)

    with open("metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return repertoire, metrics


def save_pkls(repertoire, metrics) -> None:
    # create of new folder in outputs with parameter as name
    # include brax rendering, repetoire, metrics, plot

    with open("repertoire.pkl", "wb") as f:
        pickle.dump(repertoire, f)

    with open("metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)