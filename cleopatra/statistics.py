from typing import Union, List, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from cleopatra.styles import DEFAULT_OPTIONS as style_defaults

DEFAULT_OPTIONS = dict(
    figsize=(5, 5), bins=15, color=["#0504aa"], alpha=0.7, rwidth=0.85
)
DEFAULT_OPTIONS = style_defaults | DEFAULT_OPTIONS


class Statistic:
    """
    Statistical plots
    """

    def __init__(
        self,
        values: Union[List, np.ndarray],
        **kwargs,
    ):
        """

        Parameters
        ----------
        values: [list/array]
            values to be plotted as histogram.
        """
        self._values = values
        options_dict = DEFAULT_OPTIONS.copy()
        options_dict.update(kwargs)
        self._default_options = options_dict

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

    def histogram(self, **kwargs) -> [Figure, Axes, Dict]:
        """

        Parameters
        ----------
        **kwargs: [dict]
            keys:
                bins: [int]
                    number of bins.
                color: [str]
                    color of the bins
                alpha: [float]
                    degree of transparency.
                rwidth: [float]
                    width of the bins.

        Raises
        ------
        ValueError
            If the number of colors given by the `color` kwars is not equal to the number of samples.

        Examples
        --------
        - 1D data.

            - First genearte some random data and plot the histogram.

                >>> np.random.seed(1)
                >>> x = 4 + np.random.normal(0, 1.5, 200)
                >>> stat_plot = Statistic(x)
                >>> fig, ax, hist = stat_plot.histogram()

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

        n = []
        bins = []
        patches = []
        bins_val = self.default_options["bins"]
        color = self.default_options["color"]
        alpha = self.default_options["alpha"]
        rwidth = self.default_options["rwidth"]
        if self.values.ndim == 2:
            num_samples = self.values.shape[1]
            if len(color) != num_samples:
                raise ValueError(
                    f"The number of colors:{len(color)} should be equal to the number of samples:{num_samples}"
                )
        else:
            num_samples = 1

        for i in range(num_samples):
            if self.values.ndim == 1:
                vals = self.values
            else:
                vals = self.values[:, i]

            n_i, bins_i, patches_i = ax.hist(
                x=vals,
                bins=bins_val,
                color=color[i],
                alpha=alpha,
                rwidth=rwidth,
            )
            n.append(n_i)
            bins.append(bins_i)
            patches.append(patches_i)

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
        plt.show()
        return fig, ax, hist
