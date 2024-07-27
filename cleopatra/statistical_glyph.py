"""
The `Statistic` module provides a class for creating statistical plots, specifically histograms. The class, `Statistic`,
is designed to handle both 1D (single-dimensional) and 2D (multi-dimensional) data.

The class has the following key features:

1. Initialization: The class is initialized with a set of values (1D or 2D array) and optional keyword arguments.
    The keyword arguments can be used to customize the appearance of the histogram.

2. Properties:
    The class includes properties for accessing the values and default options.

3. Histogram method:
    The class has a method named `histogram` that generates a histogram plot based on the provided values and options.
    The method handles both 1D and 2D data, and it allows customization of various aspects of the plot, such as the
    number of bins, color, transparency, and axis labels.

4. Error handling:
    The class includes error handling mechanisms to ensure that the number of colors provided matches the number of
    samples in the data. It also checks for invalid keyword arguments.

5. Examples:
    The class includes examples demonstrating how to use the histogram method with 1D and 2D data. The examples also
    include doctests to verify the correctness of the code.

Here's an example of how to use the `Statistic` class:

```python
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from cleopatra.statistics import Statistic

# Create some random 1D data
np.random.seed(1)
data_1d = 4 + np.random.normal(0, 1.5, 200)

# Create a Statistic object with the 1D data
stat_plot_1d = Statistic(data_1d)

# Generate a histogram plot for the 1D data
fig_1d, ax_1d, hist_1d = stat_plot_1d.histogram()

# Create some random 2D data
data_2d = 4 + np.random.normal(0, 1.5, (200, 3))

# Create a Statistic object with the 2D data
stat_plot_2d = Statistic(data_2d, color=["red", "green", "blue"], alpha=0.4, rwidth=0.8)

# Generate a histogram plot for the 2D data
fig_2d, ax_2d, hist_2d = stat_plot_2d.histogram()
"""

from typing import Union, List, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

DEFAULT_OPTIONS = dict(
    figsize=(5, 5), bins=15, color=["#0504aa"], alpha=0.7, rwidth=0.85
)
DEFAULT_OPTIONS = STYLE_DEFAULTS | DEFAULT_OPTIONS


class StatisticalGlyph:
    """
    A class for creating statistical plots, specifically histograms.

    This class provides methods for initializing the class with numerical values and optional keyword arguments,
    and for creating histograms from the given values.

    Attributes:
    _values: numpy.ndarray
        The numerical values to be plotted as histograms.
    _default_options: dict
        The default options for creating histograms, including the number of bins, color, alpha, rwidth, grid_alpha,
        xlabel, ylabel, xlabel_font_size, ylabel_font_size, xtick_font_size, and ytick_font_size.

    Methods:
    __init__(self, values: Union[List, np.ndarray], **kwargs):
        Initializes the class with numerical values and optional keyword arguments.
    histogram(self, **kwargs) -> [Figure, Axes, Dict]:
        Creates a histogram from the given values and optional keyword arguments.

    Example:
    >>> np.random.seed(1)
    >>> x = 4 + np.random.normal(0, 1.5, 200)
    >>> stat_plot = StatisticalGlyph(x)
    >>> fig, ax, hist = stat_plot.histogram()
    >>> print(hist) # doctest: +SKIP
    {'n': [array([ 2.,  4.,  3., 10., 11., 20., 30., 27., 31., 25., 17.,  8.,  5.,
                        6.,  1.])], 'bins': [array([0.34774335, 0.8440597 , 1.34037605, 1.8366924 , 2.33300874,
                       2.82932509, 3.32564144, 3.82195778, 4.31827413, 4.81459048,
                       5.31090682, 5.80722317, 6.30353952, 6.79985587, 7.29617221,
                       7.79248856])], 'patches': [<BarContainer object of 15 artists>]}

    .. image:: /_images/one-histogram.png
        :alt: Example Image
        :align: center
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
                bins: int, Default is 15.
                    number of bins.
                color: List[str], default is ["#0504aa"]
                    color of the bins, the number of colors should be equal to the number of samples (columns of the
                    given array).
                alpha: float, default is 0.7
                     degree of transparency.
                rwidth: float, default is 0.85
                    width of the bins.
                grid_alpha:
                    alpha of the grid.
                xlabel: str
                    x-axis label.
                ylabel: str
                    y-axis label.
                xlabel_font_size: int
                    x-axis label font size.
                ylabel_font_size: int
                    y-axis label font size.
                xtick_font_size: int
                    x-axis tick font size.
                 ytick_font_size: int
                    y-axis tick font size.

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
                >>> stat_plot = StatisticalGlyph(x)
                >>> fig, ax, hist = stat_plot.histogram()
                >>> print(hist) # doctest: +SKIP
                {'n': [array([ 2.,  4.,  3., 10., 11., 20., 30., 27., 31., 25., 17.,  8.,  5.,
                        6.,  1.])], 'bins': [array([0.34774335, 0.8440597 , 1.34037605, 1.8366924 , 2.33300874,
                       2.82932509, 3.32564144, 3.82195778, 4.31827413, 4.81459048,
                       5.31090682, 5.80722317, 6.30353952, 6.79985587, 7.29617221,
                       7.79248856])], 'patches': [<BarContainer object of 15 artists>]}

        .. image:: /_images/one-histogram.png
            :alt: Example Image
            :align: center

        - 2D data.

            - First genearte some random data and plot the histogram.

                >>> np.random.seed(1)
                >>> x = 4 + np.random.normal(0, 1.5, (200, 3))
                >>> stat_plot = StatisticalGlyph(x, color=["red", "green", "blue"], alpha=0.4, rwidth=0.8)
                >>> fig, ax, hist = stat_plot.histogram()
                >>> print(hist) # doctest: +SKIP
                {'n': [array([ 1.,  2.,  4., 10., 13., 19., 20., 32., 27., 23., 24., 11.,  5.,
                        5.,  4.]), array([ 3.,  4.,  9., 12., 20., 41., 29., 32., 25., 14.,  9.,  1.,  0.,
                        0.,  1.]), array([ 3.,  4.,  6.,  7., 25., 26., 31., 24., 30., 19., 11.,  9.,  4.,
                        0.,  1.])], 'bins': [array([-0.1896275 ,  0.33461786,  0.85886323,  1.38310859,  1.90735396,
                        2.43159932,  2.95584469,  3.48009005,  4.00433542,  4.52858078,
                        5.05282615,  5.57707151,  6.10131688,  6.62556224,  7.14980761,
                        7.67405297]), array([-0.1738017 ,  0.50031202,  1.17442573,  1.84853945,  2.52265317,
                        3.19676688,  3.8708806 ,  4.54499432,  5.21910804,  5.89322175,
                        6.56733547,  7.24144919,  7.9155629 ,  8.58967662,  9.26379034,
                        9.93790406]), array([0.24033902, 0.7940688 , 1.34779857, 1.90152835, 2.45525813,
                       3.0089879 , 3.56271768, 4.11644746, 4.67017723, 5.22390701,
                       5.77763679, 6.33136656, 6.88509634, 7.43882612, 7.99255589,
                       8.54628567])], 'patches': [<BarContainer object of 15 artists>,
                       <BarContainer object of 15 artists>, <BarContainer object of 15 artists>]}

        .. image:: /_images/three-histogram.png
            :alt: Example Image
            :align: center
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
