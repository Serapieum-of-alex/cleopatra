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
```
"""

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS

DEFAULT_OPTIONS = {
    "figsize": (5, 5),
    "bins": 15,
    "color": ["#0504aa"],
    "alpha": 0.7,
    "rwidth": 0.85,
}
DEFAULT_OPTIONS = STYLE_DEFAULTS | DEFAULT_OPTIONS


class Statistic:
    """A class for creating statistical plots, specifically histograms.

    This class provides functionality for creating histogram plots from 1D or 2D numerical data.
    It supports customization of various aspects of the plots, including bin count, colors,
    transparency, and axis labels.

    Attributes
    ----------
    values : numpy.ndarray
        The numerical values to be plotted as histograms.
    default_options : dict
        The default options for creating histograms, including:
        - bins: Number of histogram bins
        - color: Colors for the histogram bars
        - alpha: Transparency of the bars
        - rwidth: Width of the bars
        - grid_alpha: Transparency of the grid
        - xlabel, ylabel: Axis labels
        - xlabel_font_size, ylabel_font_size: Font sizes for axis labels
        - xtick_font_size, ytick_font_size: Font sizes for axis ticks

    Methods
    -------
    histogram(**kwargs)
        Creates a histogram from the given values with customizable options.

    Notes
    -----
    The class can handle both 1D data (single histogram) and 2D data (multiple histograms
    overlaid on the same plot). For 2D data, the number of colors provided should match
    the number of data series (columns in the array).

    Examples
    --------
    Create a histogram from 1D data:
    ```python
    >>> import numpy as np
    >>> from cleopatra.statistics import Statistic
    >>> np.random.seed(1)
    >>> x = 4 + np.random.normal(0, 1.5, 200)
    >>> stat_plot = Statistic(x)
    >>> fig, ax, hist = stat_plot.histogram()

    ```
    Create a histogram from 2D data with custom colors:
    ```python
    >>> np.random.seed(1)
    >>> x = 4 + np.random.normal(0, 1.5, (200, 3))
    >>> stat_plot = Statistic(x, color=["red", "green", "blue"], alpha=0.4, rwidth=0.8)
    >>> fig, ax, hist = stat_plot.histogram()

    ```
    """

    def __init__(
        self,
        values: Union[List, np.ndarray],
        **kwargs,
    ):
        """Initialize the Statistic object with values and optional customization parameters.

        Parameters
        ----------
        values : Union[List, np.ndarray]
            The numerical values to be plotted as histograms. Can be:
            - 1D array/list for a single histogram
            - 2D array/list for multiple histograms (one per column)
        **kwargs : dict, optional
            Additional keyword arguments to customize the histogram appearance.
            Supported arguments include:
            - figsize : tuple, optional
                Figure size as (width, height) in inches, by default (5, 5).
            - bins : int, optional
                Number of histogram bins, by default 15.
            - color : List[str], optional
                Colors for the histogram bars, by default ["#0504aa"].
                For 2D data, the number of colors should match the number of columns.
            - alpha : float, optional
                Transparency of the histogram bars, by default 0.7.
                Values range from 0 (transparent) to 1 (opaque).
            - rwidth : float, optional
                Relative width of the bars, by default 0.85.
                Values range from 0 to 1.
            - grid_alpha : float, optional
                Transparency of the grid lines, by default 0.75.
            - xlabel, ylabel : str, optional
                Labels for the x and y axes.
            - xlabel_font_size, ylabel_font_size : int, optional
                Font sizes for the axis labels.
            - xtick_font_size, ytick_font_size : int, optional
                Font sizes for the axis tick labels.

        Examples
        --------
        Initialize with default options:
        ```python
        >>> import numpy as np
        >>> from cleopatra.statistics import Statistic
        >>> np.random.seed(1)
        >>> x = np.random.normal(0, 1, 100)
        >>> stat = Statistic(x)

        ```
        Initialize with custom options:
        ```python
        >>> stat_custom = Statistic(
        ...     x,
        ...     figsize=(8, 6),
        ...     bins=20,
        ...     color=["#FF5733"],
        ...     alpha=0.5,
        ...     rwidth=0.9,
        ...     xlabel="Values",
        ...     ylabel="Frequency",
        ...     xlabel_font_size=14,
        ...     ylabel_font_size=14
        ... )

        ```
        Initialize with 2D data:
        ```python
        >>> data_2d = np.random.normal(0, 1, (100, 3))
        >>> stat_2d = Statistic(
        ...     data_2d,
        ...     color=["red", "green", "blue"],
        ...     alpha=0.4
        ... )

        ```
        """
        self._values = values
        options_dict = DEFAULT_OPTIONS.copy()
        options_dict.update(kwargs)
        self._default_options = options_dict

    @property
    def values(self):
        """Get the numerical values to be plotted.

        Returns
        -------
        numpy.ndarray or list
            The numerical values stored in the object, which can be:
            - 1D array/list for a single histogram
            - 2D array/list for multiple histograms (one per column)

        Examples
        --------
        >>> import numpy as np
        >>> from cleopatra.statistics import Statistic
        >>> np.random.seed(1)
        >>> x = np.random.normal(0, 1, 100)
        >>> stat = Statistic(x)
        >>> values = stat.values
        >>> values.shape
        (100,)
        """
        return self._values

    @values.setter
    def values(self, values):
        """Set the numerical values to be plotted.

        Parameters
        ----------
        values : numpy.ndarray or list
            The new numerical values to be plotted as histograms. Can be:
            - 1D array/list for a single histogram
            - 2D array/list for multiple histograms (one per column)

        Examples
        --------
        ```python
        >>> import numpy as np
        >>> from cleopatra.statistics import Statistic
        >>> np.random.seed(1)
        >>> x1 = np.random.normal(0, 1, 100)
        >>> stat = Statistic(x1)
        >>> # Update with new values
        >>> x2 = np.random.normal(5, 2, 100)
        >>> stat.values = x2

        ```
        """
        self._values = values

    @property
    def default_options(self) -> Dict:
        """Get the default options for histogram plotting.

        This property returns the dictionary of default options used for creating
        histogram plots. These options can be modified by passing keyword arguments
        to the class constructor or to the histogram method.

        Returns
        -------
        Dict
            A dictionary containing the default options for histogram plotting, including:
            - figsize : tuple
                Figure size as (width, height) in inches.
            - bins : int
                Number of histogram bins.
            - color : List[str]
                Colors for the histogram bars.
            - alpha : float
                Transparency of the histogram bars.
            - rwidth : float
                Relative width of the bars.
            - grid_alpha : float
                Transparency of the grid lines.
            - xlabel, ylabel : str
                Labels for the x and y axes.
            - xlabel_font_size, ylabel_font_size : int
                Font sizes for the axis labels.
            - xtick_font_size, ytick_font_size : int
                Font sizes for the axis tick labels.

        Examples
        --------
        ```python
        >>> import numpy as np
        >>> from cleopatra.statistics import Statistic
        >>> np.random.seed(1)
        >>> x = np.random.normal(0, 1, 100)
        >>> stat = Statistic(x)
        >>> options = stat.default_options
        >>> print(options['bins'])
        15
        >>> print(options['alpha'])
        0.7

        ```
        """
        return self._default_options

    def histogram(self, **kwargs) -> [Figure, Axes, Dict]:
        """Create a histogram from the stored numerical values.

        This method generates a histogram visualization of the numerical values stored
        in the object. It can handle both 1D data (single histogram) and 2D data
        (multiple histograms overlaid on the same plot).

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to customize the histogram appearance.
            These will override any options set during initialization.
            Supported arguments include:
            - figsize : tuple, optional
                Figure size as (width, height) in inches, by default (5, 5).
            - bins : int, optional
                Number of histogram bins, by default 15.
            - color : List[str], optional
                Colors for the histogram bars, by default ["#0504aa"].
                For 2D data, the number of colors should match the number of columns.
            - alpha : float, optional
                Transparency of the histogram bars, by default 0.7.
                Values range from 0 (transparent) to 1 (opaque).
            - rwidth : float, optional
                Relative width of the bars, by default 0.85.
                Values range from 0 to 1.
            - grid_alpha : float, optional
                Transparency of the grid lines, by default 0.75.
            - xlabel, ylabel : str, optional
                Labels for the x and y axes.
            - xlabel_font_size, ylabel_font_size : int, optional
                Font sizes for the axis labels.
            - xtick_font_size, ytick_font_size : int, optional
                Font sizes for the axis tick labels.

        Returns
        -------
        Figure
            The matplotlib Figure object containing the histogram.
        Axes
            The matplotlib Axes object on which the histogram is drawn.
        Dict
            A dictionary containing the histogram data with keys:
            - 'n': List of arrays containing the histogram bin counts
            - 'bins': List of arrays containing the bin edges
            - 'patches': List of BarContainer objects representing the histogram bars

        Raises
        ------
        ValueError
            If an invalid keyword argument is provided.
        ValueError
            If the number of colors provided doesn't match the number of data series
            (columns) in 2D data.

        Notes
        -----
        For 2D data, multiple histograms will be overlaid on the same plot with
        different colors. The transparency (alpha) can be adjusted to make overlapping
        regions visible.

        Examples
        --------
        Create a histogram from 1D data:
        ```python
        >>> import numpy as np
        >>> from cleopatra.statistics import Statistic
        >>> np.random.seed(1)
        >>> x = 4 + np.random.normal(0, 1.5, 200)
        >>> stat_plot = Statistic(x)
        >>> fig, ax, hist = stat_plot.histogram()

        ```
        Create a histogram with custom bin count and labels:
        ```python
        >>> fig, ax, hist = stat_plot.histogram(
        ...     bins=20,
        ...     xlabel="Values",
        ...     ylabel="Frequency",
        ...     xlabel_font_size=14,
        ...     ylabel_font_size=14
        ... )

        ```
        Create a histogram from 2D data with custom colors:
        ```python
        >>> np.random.seed(1)
        >>> x = 4 + np.random.normal(0, 1.5, (200, 3))
        >>> stat_plot = Statistic(x)
        >>> fig, ax, hist = stat_plot.histogram(
        ...     color=["red", "green", "blue"],
        ...     alpha=0.4,
        ...     rwidth=0.8
        ... )

        ```
        Access the histogram data:
        ```python
        >>> # Get the bin counts for the first data series
        >>> bin_counts = hist['n'][0]
        >>> # Get the bin edges for the first data series
        >>> bin_edges = hist['bins'][0]

        ```
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
        plt.show()
        return fig, ax, hist
