from typing import Any, Dict, List, Optional, Tuple

from utils.plots.config import ModelInfo
from utils.util import load_json

def collect_metrics(
    model: ModelInfo,
    filename: str,
    dict_key: Optional[str] = None,
    damage_path: Optional[str] = "",
) -> Dict:
    rep_metrics = {}
    for rep_path in model.rep_paths:
        _metrics = load_json(rep_path + damage_path, filename)
        if dict_key:
            _metrics = _metrics[dict_key]
        for k, v in _metrics.items():
            if k not in rep_metrics:
                rep_metrics[k] = []
            rep_metrics[k].append(v)
    return rep_metrics