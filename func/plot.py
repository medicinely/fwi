"""Plotting algorithms."""
from pathlib import Path
from typing import Optional, Tuple, TypedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.style as mstyle
import numpy as np


class PlotCfg(TypedDict, total=False):
    """Type alias for a plot's configuration."""

    style: str
    label: str
    title: str
    xlabel: str
    ylabel: str
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    legend: bool
    path: str
    show: bool
    clear: bool
    cmap: str
    norm: str
    extent: Tuple[float, float, float, float]


def plot(
    data_x: np.ndarray, data_y: np.ndarray, config: Optional[PlotCfg] = None,
) -> Tuple[plt.Line2D]:
    """Plot 2D data (wrapper function for matplotlib.pyplot methods).
    Args:
        data_x (np.ndarray): Horizontal coordinate of the data points.
        data_y (np.ndarray): Vertical coordinate of the data points.
        config (Optional[PlotCfg], optional): Plot configuration.
    Returns:
        Tuple[plt.Line2D]: List of lines representing the plotted data.
    """
    if config is not None:
        with mstyle.context(config.get("style", "seaborn-whitegrid")):
            lines = plt.plot(data_x, data_y, label=config.get("label"))
            apply_config(config)
    else:
        lines = plt.plot(data_x, data_y)
        plt.tight_layout()
        plt.show()
        plt.clf()
    return lines


def imshow(data: np.ndarray, config: Optional[PlotCfg] = None,) -> None:
    """Plot 2D data (wrapper function for matplotlib.pyplot methods).
    Args:
        data_arr (np.ndarray): Data array.
        config (Optional[PlotCfg], optional): Plot configuration.
    Returns:
        Tuple[plt.Line2D]: List of lines representing the plotted data.
    """
    if config is not None:
        if config.get("norm") is not None:
            norm = config["norm"]
            if norm == "centered":
                offset = mcolors.TwoSlopeNorm(vcenter=0)
                data = offset(data)
            if norm == "logarithmic":
                linthresh = data.ptp() / 50
                offset = mcolors.SymLogNorm(linthresh, base=10)
                data = offset(data)

        with mstyle.context(config.get("style", "seaborn-whitegrid")):
            plt.imshow(
                data,
                config.get("cmap"),
                aspect="auto",
                extent=config.get("extent"),
                label=config.get("label"),
            )
            apply_config(config)

    else:
        plt.imshow(data)
        plt.tight_layout()
        plt.show()
        plt.clf()


def apply_config(config: PlotCfg) -> None:
    """Apply plot configuration.
    Args:
        config (PlotCfg): Plot configuration.
    """
    if "title" in config:
        plt.title(config["title"])
    if "xlabel" in config:
        plt.xlabel(config["xlabel"])
    if "ylabel" in config:
        plt.ylabel(config["ylabel"])
    if "xlim" in config:
        plt.xlim(config["xlim"])
    if "ylim" in config:
        plt.ylim(config["ylim"])
    if "legend" in config:
        if config["legend"] is True:
            plt.legend()

    plt.tight_layout()

    if config.get("path"):
        path = Path(config["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)

    if config.get("show") is True:
        plt.show()

    if config.get("clear") is not False:
        plt.clf()