from typing import Union, List, Dict
import matplotlib.pyplot as plt
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

    def histogram(self, **kwargs):
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
        # ax1.hist(extracted_values, bins=15, alpha = 0.4) #width = 0.2,
        n, bins, patches = ax.hist(
            x=self.values,
            bins=self.default_options["bins"],
            color=self.default_options["color"],
            alpha=self.default_options["alpha"],
            rwidth=self.default_options["rwidth"],
        )
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Value", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.ylabel("Frequency", fontsize=15)
        return n, bins, patches
