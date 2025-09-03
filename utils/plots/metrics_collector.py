from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from plot_analysis import ModelInfo
from utils.util import load_json

def collect_metrics(
    model: ModelInfo,
) -> Dict:
    rep_metrics = {}
    for rep_path in model.rep_paths:
        _metrics = load_json(rep_path, "qd_metrics.json")
        (rep_metrics[k].append(v) for k, v in _metrics.items())
    return rep_metrics