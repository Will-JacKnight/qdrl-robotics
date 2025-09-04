from typing import Any, Dict, List
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.colors as mcolors


# Color palettes
baseline_colors = [
    "#c6dbef",  # light blue
    "#9ecae1",  # medium light
    "#6baed6",  # mid blue
    "#4292c6",  # strong blue
    "#2171b5",  # dark blue
    "#084594",  # very dark blue
    "#9b59b6",  # medium-light purple
]

main_colors = [
    "#ffcc00",  # vivid yellow
    "#f1c40f",  # golden yellow 
    "#d62728",  # highlight red
    "#2ecc71",  # mint green
    "#e67e22",  # "Carrot Orange"
]

# Font and figure parameters
font_size = 14
plotting_params = {
    "axes.labelsize": font_size,
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "text.usetex": False,
    "figure.figsize": [10, 10],
}


def apply_plot_style():
    """Apply the shared plotting style to matplotlib."""
    mpl.rcParams.update(plotting_params)


def adjust_color(color, amount=1.1):
    """
    Darken or lighten a color by a given factor.
    Args:
        - amount: > 1 lighter, < 1 darker
    """
    c = mcolors.to_rgb(color)
    return tuple(min(1, max(0, channel * amount)) for channel in c)

@dataclass
class ModelInfo:
    model_path: str
    model_desc: str
    model_desc_abbr: str
    color: str
    rep_paths: List[str]

def extract_model_attributes(models: List[ModelInfo]):
    model_paths  = [m.model_path for m in models]
    model_desc   = [m.model_desc for m in models]
    model_abbr   = [m.model_desc_abbr for m in models]
    model_colors = [m.color for m in models]
    return model_paths, model_desc, model_abbr, model_colors
