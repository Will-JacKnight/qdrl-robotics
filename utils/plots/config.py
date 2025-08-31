"""
Shared plotting configuration and utilities for consistent styling across all plot modules.
"""

import matplotlib as mpl
import matplotlib.colors as mcolors
from typing import Any, Dict

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
