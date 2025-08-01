from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import matplotlib.pyplot as plt

from utils.util import load_metrics

def performace_box_plot(
    exp_paths: List[str],
):
    plt.boxplot()
    plt.savefig("evaluations/performance_box_plot.png")
    plt.close()