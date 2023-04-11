from typing import Union, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from cleopatra.styles import DEFAULT_OPTIONS as style_defaults

DEFAULT_OPTIONS = dict(figsize=(5, 5), bins=15, color="#0504aa", alpha=0.7, rwidth=0.85)
DEFAULT_OPTIONS = style_defaults | DEFAULT_OPTIONS


class Statistic:
    """
    Statistical plots
    """

    def __init__(
        self,
        values: Union[List, np.ndarray],
    ):
        """

        Parameters
        ----------
        values: [list/array]
            values to be plotted as histogram.
        """
        self._values = values
        self._default_options = DEFAULT_OPTIONS

    @property
    def values(self):
        """numerical values"""
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    @property
    def default_options(self) -> Dict:
        """Default plot options"""
        return self._default_options

    def histogram(self, **kwargs) -> [Figure, Any, Dict]:
        """

        Parameters
        ----------
        **kwargs : [dict]
            keys:
                bins: [int]
                    number of bins.
                color: [str]
                    color of the bins
                alpha: [float]
                    degree of transparency.
                rwidth: [float]
                    width of thebins.
        """
        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {self.default_options}"
                )
            else:
                self.default_options[key] = val

        fig, ax = plt.subplots(figsize=self.default_options["figsize"])

        n, bins, patches = ax.hist(
            x=self.values,
            bins=self.default_options["bins"],
            color=self.default_options["color"],
            alpha=self.default_options["alpha"],
            rwidth=self.default_options["rwidth"],
        )
        plt.grid(axis="y", alpha=self.default_options["grid_alpha"])
        plt.xlabel(
            self.default_options["xlabel"],
            fontsize=self.default_options["xlabel_font_size"],
        )
        plt.ylabel(
            self.default_options["ylabel"],
            fontsize=self.default_options["ylabel_font_size"],
        )
        plt.xticks(fontsize=self.default_options["xtick_font_size"])
        plt.yticks(fontsize=self.default_options["ytick_font_size"])
        hist = {"n": n, "bins": bins, "patches": patches}
        # ax.yaxis.label.set_color("#27408B")
        # ax1.tick_params(axis="y", color="#27408B")
        return fig, ax, hist
