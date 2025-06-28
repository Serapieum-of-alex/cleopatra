"""
Module: Array.

This module provides a class, `Array`, to handle 3D arrays and perform various operations on them,
such as plotting, animating, and displaying the array.

The `Array` class has the following functionalities:
- Initialize an array object with the provided parameters.
- Plot the array with optional parameters to customize the appearance and display cell values.
- Animate the array over time with optional parameters to customize the animation speed and display points.
- Display the array with optional parameters to customize the appearance and display point IDs.

The `Array` class has the following attributes:
- `arr`: The 3D array to be handled.
- `time`: The time values for animation.
- `points`: The points to be displayed on the array.
- `default_options`: A dictionary to store default options for plotting, animating, and displaying.

The `Array` class has the following methods:
- `plot`: Plot the array with optional parameters.
- `animate`: Animate the array over time with optional parameters.
- `display`: Display the array with optional parameters.
"""

import math
from typing import Any, Dict, List, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from hpc.indexing import get_indices2
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import LogFormatter

from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import MidpointNormalize

DEFAULT_OPTIONS = {
    "vmin": None,
    "vmax": None,
    "num_size": 8,
    "display_cell_value": False,
    "background_color_threshold": None,
    "id_color": "green",
    "id_size": 20,
    "precision": 2,
}
DEFAULT_OPTIONS = STYLE_DEFAULTS | DEFAULT_OPTIONS
SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4"]


class ArrayGlyph:
    """A class to handle arrays and perform various visualization operations on them.

    The ArrayGlyph class provides functionality for visualizing 2D and 3D arrays with
    various customization options. It supports plotting single arrays, RGB arrays,
    and creating animations from 3D arrays.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    extent : List
        The extent of the array [xmin, xmax, ymin, ymax].
    rgb : bool
        Whether the array is an RGB array.
    no_elem : int
        The number of elements in the array.
    anim : matplotlib.animation.FuncAnimation
        The animation object if created.

    Notes
    -----
    This class provides methods for:
    - Plotting arrays with customizable color scales, color bars, and annotations
    - Creating animations from 3D arrays
    - Displaying point values on arrays
    - Customizing plot appearance

    Examples
    --------
    Create a simple array plot:
    ```python
    >>> import numpy as np
    >>> from cleopatra.array_glyph import ArrayGlyph
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> array_glyph = ArrayGlyph(arr)
    >>> fig, ax = array_glyph.plot()

    Create an RGB plot from a 3D array:
    ```python
    >>> rgb_array = np.random.randint(0, 255, size=(3, 10, 10))
    >>> rgb_glyph = ArrayGlyph(rgb_array, rgb=[0, 1, 2])
    >>> fig, ax = rgb_glyph.plot()

    Create an animated plot from a 3D array:
    ```python
    >>> time_series = np.random.randint(1, 10, size=(5, 10, 10))
    >>> time_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
    >>> animated_glyph = ArrayGlyph(time_series)
    >>> anim = animated_glyph.animate(time_labels)
    """

    def __init__(
        self,
        array: np.ndarray,
        exclude_value: List = np.nan,
        extent: List = None,
        rgb: List[int] = None,
        surface_reflectance: int = None,
        cutoff: List = None,
        ax: Axes = None,
        fig: Figure = None,
        percentile: int = None,
        **kwargs,
    ):
        """Initialize the ArrayGlyph object with an array and optional parameters.

        Parameters
        ----------
        array : np.ndarray
            The array to be visualized. Can be a 2D array for single plots or a 3D array for RGB plots or animations.
        exclude_value : List or numeric, optional
            Value(s) used to mask cells out of the domain, by default np.nan.
            Can be a single value or a list of values to exclude.
        extent : List, optional
            The extent of the array in the format [xmin, ymin, xmax, ymax], by default None.
            If provided, the array will be plotted with these spatial boundaries.
        rgb : List[int], optional
            The indices of the red, green, and blue bands in the given array, by default None.
            If provided, the array will be treated as an RGB image.
            Can be a list of three values [r, g, b], or four values if alpha band is included [r, g, b, a].
        surface_reflectance : int, optional
            Surface reflectance value for normalizing satellite data, by default None.
            Typically 10000 for Sentinel-2 data.
        cutoff : List, optional
            Clip the range of pixel values for each band, by default None.
            Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
            Should be a list with one value per band.
        ax : matplotlib.axes.Axes, optional
            A pre-existing axes to plot on, by default None.
            If None, a new axes will be created.
        fig : matplotlib.figure.Figure, optional
            A pre-existing figure to plot on, by default None.
            If None, a new figure will be created.
        percentile : int, optional
            The percentile value to be used for scaling the array values, by default None.
            Used to enhance contrast by stretching the histogram.
        **kwargs : dict
            Additional keyword arguments for customizing the plot.
            Supported arguments include:
                figsize : tuple, optional
                    Figure size, by default (8, 8).
                vmin : float, optional
                    Minimum value for color scaling, by default min(array).
                vmax : float, optional
                    Maximum value for color scaling, by default max(array).
                title : str, optional
                    Title of the plot, by default 'Array Plot'.
                title_size : int, optional
                    Title font size, by default 15.
                cmap : str, optional
                    Colormap name, by default 'coolwarm_r'.

        Raises
        ------
        ValueError
            If an invalid keyword argument is provided.
        ValueError
            If rgb is provided but the array doesn't have enough dimensions.

        Examples
        --------
        Basic initialization with a 2D array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array_glyph = ArrayGlyph(arr)
        >>> fig, ax = array_glyph.plot()

        ```
        Initialization with custom figure size and title:
        ```python
        >>> array_glyph = ArrayGlyph(arr, figsize=(10, 8), title="Custom Array Plot")
        >>> fig, ax = array_glyph.plot()

        ```
        Initialization with RGB bands from a 3D array:
        ```python
        >>> rgb_array = np.random.randint(0, 255, size=(3, 10, 10))
        >>> rgb_glyph = ArrayGlyph(rgb_array, rgb=[0, 1, 2], surface_reflectance=255)
        >>> fig, ax = rgb_glyph.plot()

        ```
        Initialization with custom extent:
        ```python
        >>> array_glyph = ArrayGlyph(arr, extent=[0, 0, 10, 10])
        >>> fig, ax = array_glyph.plot()

        ```
        """
        self._default_options = DEFAULT_OPTIONS.copy()

        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {DEFAULT_OPTIONS}"
                )
            else:
                self.default_options[key] = val
        # first replace the no_data_value by nan
        # convert the array to float32 to be able to replace the no data value with nan
        if exclude_value is not np.nan:
            if len(exclude_value) > 1:
                mask = np.logical_or(
                    np.isclose(array, exclude_value[0], rtol=0.001),
                    np.isclose(array, exclude_value[1], rtol=0.001),
                )
            else:
                mask = np.isclose(array, exclude_value[0], rtol=0.0000001)
            array = ma.array(array, mask=mask, dtype=array.dtype)
        else:
            array = ma.array(array)

        # convert the extent from [xmin, ymin, xmax, ymax] to [xmin, xmax, ymin, ymax] as required by matplotlib.
        if extent is not None:
            extent = [extent[0], extent[2], extent[1], extent[3]]
        self.extent = extent

        if rgb is not None:
            self.rgb = True
            # prepare to plot rgb plot only if there are three arrays
            if array.shape[0] < 3:
                raise ValueError(
                    f"To plot RGB plot the given array should have only 3 arrays, given array have "
                    f"{array.shape[0]}"
                )
            else:
                array = self.prepare_array(
                    array,
                    rgb=rgb,
                    surface_reflectance=surface_reflectance,
                    cutoff=cutoff,
                    percentile=percentile,
                )
        else:
            self.rgb = False

        self._exclude_value = exclude_value

        self._vmax = (
            np.nanmax(array) if kwargs.get("vmax") is None else kwargs.get("vmax")
        )
        self._vmin = (
            np.nanmin(array) if kwargs.get("vmin") is None else kwargs.get("vmin")
        )

        self.arr = array
        # get the tick spacing that has 10 ticks only
        self.ticks_spacing = (self._vmax - self._vmin) / 10
        shape = array.shape
        if len(shape) == 3:
            no_elem = array[0, :, :].count()
        else:
            no_elem = array.count()

        self.no_elem = no_elem
        if fig is None:
            self.fig, self.ax = self.create_figure_axes()
        else:
            self.fig, self.ax = fig, ax

    def prepare_array(
        self,
        array: np.ndarray,
        rgb: List[int] = None,
        surface_reflectance: int = None,
        cutoff: List = None,
        percentile: int = None,
    ) -> np.ndarray:
        """Prepare an array for RGB visualization.

        This method processes a multi-band array to create an RGB image suitable for visualization.
        It can normalize the data using either percentile-based scaling or surface reflectance values.

        Parameters
        ----------
        array : np.ndarray
            The input array containing multiple bands. For RGB visualization,
            this should be a 3D array where the first dimension represents the bands.
        rgb : List[int], optional
            The indices of the red, green, and blue bands in the given array, by default None.
            If None, assumes the order is [3, 2, 1] (common for Sentinel-2 data).
        surface_reflectance : int, optional
            Surface reflectance value for normalizing satellite data, by default None.
            Typically 10000 for Sentinel-2 data or 255 for 8-bit imagery.
            Used to scale values to the range [0, 1].
        cutoff : List, optional
            Clip the range of pixel values for each band, by default None.
            Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
            Should be a list with one value per band.
        percentile : int, optional
            The percentile value to be used for scaling the array values, by default None.
            Used to enhance contrast by stretching the histogram.
            If provided, this takes precedence over surface_reflectance.

        Returns
        -------
        np.ndarray
            The prepared array with shape (height, width, 3) suitable for RGB visualization.
            Values are normalized to the range [0, 1].

        Raises
        ------
        ValueError
            If the array shape is incompatible with the provided RGB indices.

        Examples
        --------
        Prepare an array using percentile-based scaling:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a 3-band array (e.g., satellite image)
        >>> bands = np.random.randint(0, 10000, size=(3, 100, 100))
        >>> glyph = ArrayGlyph(np.zeros((1, 1)))  # Dummy initialization
        >>> rgb_array = glyph.prepare_array(bands, rgb=[0, 1, 2], percentile=2)
        >>> rgb_array.shape
        (100, 100, 3)
        >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
        True

        ```
        Prepare an array using surface reflectance normalization:
        ```python
        >>> rgb_array = glyph.prepare_array(bands, rgb=[0, 1, 2], surface_reflectance=10000)
        >>> rgb_array.shape
        (100, 100, 3)
        >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
        True

        ```
        Prepare an array with cutoff values:
        ```python
        >>> rgb_array = glyph.prepare_array(
        ...     bands, rgb=[0, 1, 2], surface_reflectance=10000, cutoff=[5000, 5000, 5000]
        ... )
        >>> rgb_array.shape
        (100, 100, 3)
        >>> np.all((0 <= rgb_array) & (rgb_array <= 1))
        True

        ```
        """
        # take the rgb arrays and reorder them to have the red-green-blue, if the order is not given, assume the
        # order as sentinel data. [3, 2, 1]
        array = array[rgb].transpose(1, 2, 0)

        if percentile is not None:
            array = self.scale_percentile(array, percentile=percentile)
        elif surface_reflectance is not None:
            array = self._prepare_sentinel_rgb(
                array,
                rgb=rgb,
                surface_reflectance=surface_reflectance,
                cutoff=cutoff,
            )
        return array

    def _prepare_sentinel_rgb(
        self,
        array: np.ndarray,
        rgb: List[int] = None,
        surface_reflectance: int = 10000,
        cutoff: List = None,
    ) -> np.ndarray:
        """Prepare Sentinel satellite data for RGB visualization.

        This method specifically handles Sentinel satellite imagery by normalizing the data
        using the provided surface reflectance value and optional cutoff values.

        Parameters
        ----------
        array : np.ndarray
            The input array with shape (height, width, 3) containing RGB bands.
            This array should already be transposed from the original band-first format.
        rgb : List[int], optional
            The indices of the red, green, and blue bands in the original array, by default None.
            Used only for cutoff application.
        surface_reflectance : int, optional
            Surface reflectance value for normalizing satellite data, by default 10000.
            Sentinel-2 data typically uses 10000 as the maximum reflectance value.
            Used to scale values to the range [0, 1].
        cutoff : List, optional
            Clip the range of pixel values for each band, by default None.
            Takes only pixel values from 0 to the value of the cutoff and scales them back to between 0 and 1.
            Should be a list with one value per band.

        Returns
        -------
        np.ndarray
            The prepared array with shape (height, width, 3) suitable for RGB visualization.
            Values are normalized to the range [0, 1].

        Examples
        --------
        Prepare Sentinel-2 data with default surface reflectance:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a simulated Sentinel-2 RGB array
        >>> rgb_data = np.random.randint(0, 10000, size=(100, 100, 3))
        >>> glyph = ArrayGlyph(np.zeros((1, 1)))  # Dummy initialization
        >>> normalized = glyph._prepare_sentinel_rgb(rgb_data)
        >>> np.all((0 <= normalized) & (normalized <= 1))
        True

        ```
        Prepare Sentinel-2 data with custom cutoff values:
        ```python
        >>> cutoffs = [8000, 7000, 9000]
        >>> normalized = glyph._prepare_sentinel_rgb(rgb_data, rgb=[0, 1, 2], cutoff=cutoffs)
        >>> np.all((0 <= normalized) & (normalized <= 1))
        True

        ```
        """
        array = np.clip(array / surface_reflectance, 0, 1)
        if cutoff is not None:
            array[0] = np.clip(rgb[0], 0, cutoff[0]) / cutoff[0]
            array[1] = np.clip(rgb[1], 0, cutoff[1]) / cutoff[1]
            array[2] = np.clip(rgb[2], 0, cutoff[2]) / cutoff[2]

        return array

    @staticmethod
    def scale_percentile(arr: np.ndarray, percentile: int = 1) -> np.ndarray:
        """Scale an array using percentile-based contrast stretching.

        This method enhances the contrast of an image by stretching the histogram
        based on percentile values. It calculates the lower and upper percentile values
        for each band and normalizes the data to the range [0, 1].

        Parameters
        ----------
        arr : np.ndarray
            The array to be scaled, with shape (height, width, bands).
            Typically an RGB image with 3 bands.
        percentile : int, optional
            The percentile value to be used for scaling, by default 1.
            This value determines how much of the histogram tails to exclude.
            Higher values result in more contrast stretching.
            Typical values range from 1 to 5.

        Returns
        -------
        np.ndarray
            The scaled array, normalized between 0 and 1, with the same shape as input.
            Data type is float32.

        Notes
        -----
        The method works by:
        1. Computing the lower percentile value for each band
        2. Computing the upper percentile value (100 - percentile) for each band
        3. Normalizing each band using these percentile values
        4. Clipping values to the range [0, 1]

        This is particularly useful for visualizing satellite imagery with high dynamic range.

        Examples
        --------
        Scale a single-band array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a test array with values between 0 and 10000
        >>> test_array = np.random.randint(0, 10000, size=(100, 100, 1))
        >>> scaled = ArrayGlyph.scale_percentile(test_array, percentile=2)
        >>> scaled.shape
        (100, 100, 1)
        >>> np.all((0 <= scaled) & (scaled <= 1))
        np.True_

        ```
        Scale an RGB array:
        ```python
        >>> rgb_array = np.random.randint(0, 10000, size=(100, 100, 3))
        >>> scaled = ArrayGlyph.scale_percentile(rgb_array, percentile=2)
        >>> scaled.shape
        (100, 100, 3)
        >>> np.all((0 <= scaled) & (scaled <= 1))
        np.True_

        ```
        Using different percentile values affects contrast:
        ```python
        >>> low_contrast = ArrayGlyph.scale_percentile(rgb_array, percentile=1)
        >>> high_contrast = ArrayGlyph.scale_percentile(rgb_array, percentile=5)
        >>> # Higher percentile typically results in higher contrast

        ```
        """
        rows, columns, bands = arr.shape
        # flatten image.
        arr = np.reshape(arr, [rows * columns, bands]).astype(np.float32)
        # lower percentile values (one value for each band).
        lower_percent = np.percentile(arr, percentile, axis=0)
        # 98 percentile values.
        upper_percent = np.percentile(arr, 100 - percentile, axis=0) - lower_percent
        # normalize the 3 bands using the percentile values for each band.
        arr = (arr - lower_percent[None, :]) / upper_percent[None, :]
        arr = np.reshape(arr, [rows, columns, bands])
        # discard outliers.
        arr = arr.clip(0, 1)

        return arr

    def __str__(self):
        """String representation of the Array object."""
        message = f"""
                    Min: {self.vmin}
                    Max: {self.vmax}
                    Exclude values: {self.exclude_value}
                    RGB: {self.rgb}
                """
        return message

    @property
    def vmin(self):
        """min value in the array"""
        return self._vmin

    @property
    def vmax(self):
        """max value in the array"""
        return self._vmax

    @property
    def exclude_value(self):
        """exclude_value"""
        return self._exclude_value

    @exclude_value.setter
    def exclude_value(self, value):
        self._exclude_value = value

    @property
    def default_options(self):
        """Default plot options"""
        return self._default_options

    @property
    def anim(self):
        """Animation function"""
        if hasattr(self, "_anim"):
            val = self._anim
        else:
            raise ValueError(
                "Please first use the function animate to create the animation object"
            )
        return val

    def create_figure_axes(self) -> Tuple[Figure, Axes]:
        """Create the figure and the axes.

        Returns
        -------
        fig: matplotlib.figure.Figure
            the created figure.
        ax: matplotlib.axes.Axes
            the created axes.
        """
        plt.ioff()  # to prevent the empty figure from being displayed
        fig, ax = plt.subplots(figsize=self.default_options["figsize"])

        return fig, ax

    def get_ticks(self) -> np.ndarray:
        """get a list of ticks for the color bar"""
        ticks_spacing = self.default_options["ticks_spacing"]
        vmax = self.default_options["vmax"]
        vmin = self.default_options["vmin"]
        remainder = np.round(math.remainder(vmax, ticks_spacing), 3)
        # np.mod(vmax, ticks_spacing) gives float point error, so we use the round function.
        if remainder == 0:
            ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
        else:
            try:
                ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
            except ValueError:
                raise ValueError(
                    "The number of ticks exceeded the max allowed size, possible errors"
                    f" is the value of the NodataValue you entered-{self.exclude_value}"
                )
            ticks = np.append(
                ticks,
                [int(vmax / ticks_spacing) * ticks_spacing + ticks_spacing],
            )
        return ticks

    def _plot_im_get_cbar_kw(
        self, ax: Axes, arr: np.ndarray, ticks: np.ndarray
    ) -> Tuple[AxesImage, Dict[str, str]]:
        """Plot a single image and get color bar keyword arguments.

        Parameters
        ----------
        ax: [axes]
            matplotlib figure axes.
        arr: [array]
            numpy array.
        ticks: [list]
            color bar ticks.

        Returns
        -------
        im: AxesImage
            image axes.
        cbar: Dict[str,str]
            color bar keyword arguments.
        """
        color_scale = self.default_options["color_scale"]
        cmap = self.default_options["cmap"]
        # get the vmin and vmax from the tick instead of the default values.
        vmin: float = ticks[0]  # self.default_options["vmin"]
        vmax: float = ticks[-1]  # self.default_options["vmax"]

        if color_scale.lower() == "linear":
            im = ax.matshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=self.extent)
            cbar_kw = {"ticks": ticks}
        elif color_scale.lower() == "power":
            im = ax.matshow(
                arr,
                cmap=cmap,
                norm=colors.PowerNorm(
                    gamma=self.default_options["gamma"], vmin=vmin, vmax=vmax
                ),
                extent=self.extent,
            )
            cbar_kw = {"ticks": ticks}
        elif color_scale.lower() == "sym-lognorm":
            im = ax.matshow(
                arr,
                cmap=cmap,
                norm=colors.SymLogNorm(
                    linthresh=self.default_options["line_threshold"],
                    linscale=self.default_options["line_scale"],
                    base=np.e,
                    vmin=vmin,
                    vmax=vmax,
                ),
                extent=self.extent,
            )
            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = {"ticks": ticks, "format": formatter}
        elif color_scale.lower() == "boundary-norm":
            if not self.default_options["bounds"]:
                bounds = ticks
                cbar_kw = {"ticks": ticks}
            else:
                bounds = self.default_options["bounds"]
                cbar_kw = {"ticks": self.default_options["bounds"]}
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(arr, cmap=cmap, norm=norm, extent=self.extent)
        elif color_scale.lower() == "midpoint":
            arr = arr.filled(np.nan)
            im = ax.matshow(
                arr,
                cmap=cmap,
                norm=MidpointNormalize(
                    midpoint=self.default_options["midpoint"],
                    vmin=vmin,
                    vmax=vmax,
                ),
                extent=self.extent,
            )
            cbar_kw = {"ticks": ticks}
        else:
            raise ValueError(
                f"Invalid color scale option: {color_scale}. Use 'linear', 'power', 'power-norm',"
                "'sym-lognorm', 'boundary-norm'"
            )

        return im, cbar_kw

    @staticmethod
    def _plot_text(
        ax: Axes, arr: np.ndarray, indices, default_options_dict: dict
    ) -> list:
        """plot values as a text in each cell.

        Parameters
        ----------
        ax:[matplotlib ax]
            matplotlib axes
        indices: [array]
            array with columns, (row, col)
        default_options_dict: Dict
            default options dictionary after updating the options.

        Returns
        -------
        list:
            list of the text object
        """
        # https://github.com/Serapieum-of-alex/cleopatra/issues/75
        # add text for the cell values
        add_text = lambda elem: ax.text(
            elem[1],
            elem[0],
            np.round(arr[elem[0], elem[1]], 2),
            ha="center",
            va="center",
            color="w",
            fontsize=default_options_dict["num_size"],
        )
        return list(map(add_text, indices))

    @staticmethod
    def _plot_point_values(ax, point_table: np.ndarray, pid_color, pid_size):
        write_points = lambda x: ax.text(
            x[2],
            x[1],
            x[0],
            ha="center",
            va="center",
            color=pid_color,
            fontsize=pid_size,
        )
        return list(map(write_points, point_table))

    def create_color_bar(self, ax: Axes, im: AxesImage, cbar_kw: dict) -> Colorbar:
        """Create Color bar.

        Parameters
        ----------
        ax: Axes
            matplotlib axes.
        im: AxesImage
            Image axes.
        cbar_kw: dict
            color bar keyword arguments.

        Returns
        -------
        Colorbar:
            colorbar object.
        """
        # im or cax is the last image added to the axes
        # im = ax.images[-1]
        cbar = ax.figure.colorbar(
            im,
            ax=ax,
            shrink=self.default_options["cbar_length"],
            orientation=self.default_options["cbar_orientation"],
            **cbar_kw,
        )
        # cbar.ax.set_ylabel(
        #     self.default_options["cbar_label"],
        #     rotation=self.default_options["cbar_label_rotation"],
        #     va=self.default_options["cbar_label_location"],
        #     fontsize=self.default_options["cbar_label_size"],
        # )
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(
            self.default_options["cbar_label"],
            fontsize=self.default_options["cbar_label_size"],
            loc=self.default_options["cbar_label_location"],
        )

        return cbar

    def plot(
        self,
        points: np.ndarray = None,
        point_color: str = "red",
        point_size: Union[int, float] = 100,
        pid_color="blue",
        pid_size: Union[int, float] = 10,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot the array with customizable visualization options.

        This method creates a visualization of the array with various customization options
        including color scales, color bars, cell value display, and point annotations.
        It supports both regular arrays and RGB arrays.

        Parameters
        ----------
        points : np.ndarray, optional
            Points to display on the array, by default None.
            Should be a 3-column array where:
            - First column: values to display for each point
            - Second column: row indices of the points in the array
            - Third column: column indices of the points in the array
        point_color : str, optional
            Color of the points, by default "red".
            Any valid matplotlib color string.
        point_size : Union[int, float], optional
            Size of the points, by default 100.
            Controls the marker size.
        pid_color : str, optional
            Color of the point value annotations, by default "blue".
            Any valid matplotlib color string.
        pid_size : Union[int, float], optional
            Size of the point value annotations, by default 10.
            Controls the font size of the annotations.
        **kwargs : dict
            Additional keyword arguments for customizing the plot.

            Plot appearance:
            ---------------
            title : str, optional
                Title of the plot, by default 'Array Plot'.
            title_size : int, optional
                Title font size, by default 15.
            cmap : str, optional
                Colormap name, by default 'coolwarm_r'.
            vmin : float, optional
                Minimum value for color scaling, by default min(array).
            vmax : float, optional
                Maximum value for color scaling, by default max(array).

            Color bar options:
            ----------------
            cbar_orientation : str, optional
                Orientation of the color bar, by default 'vertical'.
                Can be 'horizontal' or 'vertical'.
            cbar_label_rotation : float, optional
                Rotation angle of the color bar label, by default -90.
            cbar_label_location : str, optional
                Location of the color bar label, by default 'bottom'.
                Options: 'top', 'bottom', 'center', 'baseline', 'center_baseline'.
            cbar_length : float, optional
                Ratio to control the height/width of the color bar, by default 0.75.
            ticks_spacing : int, optional
                Spacing between ticks on the color bar, by default 2.
            cbar_label_size : int, optional
                Font size of the color bar label, by default 12.
            cbar_label : str, optional
                Label text for the color bar, by default 'Value'.

            Color scale options:
            ------------------
            color_scale : str, optional
                Type of color scaling to use, by default 'linear'.
                Options:
                - 'linear': Linear scale
                - 'power': Power-law normalization
                - 'sym-lognorm': Symmetrical logarithmic scale
                - 'boundary-norm': Discrete intervals based on boundaries
                - 'midpoint': Scale split at a specified midpoint
            gamma : float, optional
                Exponent for 'power' color scale, by default 0.5.
                Values < 1 emphasize lower values, values > 1 emphasize higher values.
            line_threshold : float, optional
                Threshold for 'sym-lognorm' color scale, by default 0.0001.
            line_scale : float, optional
                Scale factor for 'sym-lognorm' color scale, by default 0.001.
            bounds : List, optional
                Boundaries for 'boundary-norm' color scale, by default None.
                Defines the discrete intervals for color mapping.
            midpoint : float, optional
                Midpoint value for 'midpoint' color scale, by default 0.

            Cell value display options:
            -------------------------
            display_cell_value : bool, optional
                Whether to display the values of cells as text, by default False.
            num_size : int, optional
                Font size of the cell value text, by default 8.
            background_color_threshold : float, optional
                Threshold for cell value text color, by default None.
                If cell value > threshold, text is black; otherwise, text is white.
                If None, uses max(array)/2 as the threshold.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            A tuple containing:
            - fig: The matplotlib Figure object
            - ax: The matplotlib Axes object

        Raises
        ------
        ValueError
            If an invalid keyword argument is provided.

        Examples
        --------
        Basic array plot:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Customized Plot", title_size=18)
        >>> fig, ax = array.plot()

        ```
        Color bar customization:
        ```python
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Customized color bar", title_size=18)
        >>> fig, ax = array.plot(
        ...     cbar_orientation="horizontal",
        ...     cbar_label_rotation=-90,
        ...     cbar_label_location="center",
        ...     cbar_length=0.7,
        ...     cbar_label_size=12,
        ...     cbar_label="Discharge m3/s",
        ...     ticks_spacing=5,
        ...     color_scale="linear",
        ...     cmap="coolwarm_r",
        ... )

        ```
        Display cell values:
        ```python
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display array values", title_size=18)
        >>> fig, ax = array.plot(
        ...     display_cell_value=True,
        ...     num_size=12
        ... )

        ```
        Plot points at specific locations:
        ```python
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display Points", title_size=14)
        >>> points = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2]])
        >>> fig, ax = array.plot(
        ...     points=points,
        ...     point_color="black",
        ...     point_size=100,
        ...     pid_color="orange",
        ...     pid_size=30,
        ... )

        ```
        Power scale with different gamma values:
        ```python
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale", title_size=18)
        >>> fig, ax = array.plot(
        ...     cbar_label="Discharge m3/s",
        ...     color_scale="power",
        ...     gamma=0.5,  # Default value
        ...     cmap="coolwarm_r",
        ... )

        >>> # Higher gamma (0.8) emphasizes higher values less
        >>> fig, ax = array.plot(
        ...     color_scale="power",
        ...     gamma=0.8,
        ...     cmap="coolwarm_r",
        ... )

        >>> # Lower gamma (0.1) emphasizes higher values more
        >>> fig, ax = array.plot(
        ...     color_scale="power",
        ...     gamma=0.1,
        ...     cmap="coolwarm_r",
        ... )

        ```
        Logarithmic scale:
        ```python
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Logarithmic scale", title_size=18)
        >>> fig, ax = array.plot(
        ...     cbar_label="Discharge m3/s",
        ...     color_scale="sym-lognorm",
        ...     cmap="coolwarm_r",
        ... )

        >>> # Custom logarithmic scale parameters
        >>> fig, ax = array.plot(
        ...     color_scale="sym-lognorm",
        ...     line_threshold=0.015,
        ...     line_scale=0.1,
        ...     cmap="coolwarm_r",
        ... )

        ```
        Boundary scale with custom boundaries:
        ```python
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Boundary scale", title_size=18)
        >>> fig, ax = array.plot(
        ...     color_scale="boundary-norm",
        ...     cmap="coolwarm_r",
        ... )

        >>> # Custom boundaries
        >>> bounds = [0, 5, 10]
        >>> fig, ax = array.plot(
        ...     color_scale="boundary-norm",
        ...     bounds=bounds,
        ...     cmap="coolwarm_r",
        ... )

        ```
        Midpoint scale:
        ```python
        >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Midpoint scale", title_size=18)
        >>> fig, ax = array.plot(
        ...     color_scale="midpoint",
        ...     midpoint=2,
        ...     cmap="coolwarm_r",
        ... )

        ```
        """
        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {DEFAULT_OPTIONS}"
                )
            else:
                self.default_options[key] = val

        arr = self.arr
        fig, ax = self.fig, self.ax

        if self.rgb:
            ax.imshow(arr, extent=self.extent)
        else:
            # if user did not input ticks spacing use the calculated one.
            if "ticks_spacing" in kwargs.keys():
                self.default_options["ticks_spacing"] = kwargs["ticks_spacing"]
            else:
                self.default_options["ticks_spacing"] = self.ticks_spacing

            if "vmin" in kwargs.keys():
                self.default_options["vmin"] = kwargs["vmin"]
            else:
                self.default_options["vmin"] = self.vmin

            if "vmax" in kwargs.keys():
                self.default_options["vmax"] = kwargs["vmax"]
            else:
                self.default_options["vmax"] = self.vmax

            # creating the ticks/bounds
            ticks = self.get_ticks()
            im, cbar_kw = self._plot_im_get_cbar_kw(ax, arr, ticks)

            # Create colorbar
            self.create_color_bar(ax, im, cbar_kw)

        ax.set_title(
            self.default_options["title"], fontsize=self.default_options["title_size"]
        )

        if self.extent is None:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        optional_display = {}
        if self.default_options["display_cell_value"]:
            indices = get_indices2(arr, [np.nan])
            optional_display["cell_text_value"] = self._plot_text(
                ax, arr, indices, self.default_options
            )

        if points is not None:
            row = points[:, 1]
            col = points[:, 2]
            optional_display["points_scatter"] = ax.scatter(
                col, row, color=point_color, s=point_size
            )
            optional_display["points_id"] = self._plot_point_values(
                ax, points, pid_color, pid_size
            )

        # # Normalize the threshold to the image color range.
        # if self.default_options["background_color_threshold"] is not None:
        #     im.norm(self.default_options["background_color_threshold"])
        # else:
        #     im.norm(self.vmax) / 2.0
        plt.show()
        return fig, ax

    def animate(
        self,
        time: List[Any],
        points: np.ndarray = None,
        text_colors=("white", "black"),
        interval=200,
        text_loc: list[Any, Any] = None,
        point_color="red",
        point_size=100,
        pid_color="blue",
        pid_size=10,
        **kwargs,
    ):
        """Create an animation from a 3D array.

        This method creates an animation by iterating through the first dimension of a 3D array.
        Each slice of the array becomes a frame in the animation, with optional time labels,
        point annotations, and cell value displays.

        Parameters
        ----------
        time : List[Any]
            A list containing labels for each frame in the animation.
            These could be timestamps, frame numbers, or any other identifiers.
            The length of this list should match the first dimension of the array.
        points : np.ndarray, optional
            Points to display on the array, by default None.
            Should be a 3-column array where:
            - First column: values to display for each point
            - Second column: row indices of the points in the array
            - Third column: column indices of the points in the array
        text_colors : Tuple[str, str], optional
            Two colors to be used for cell value text, by default ("white", "black").
            The first color is used when the cell value is below the background_color_threshold,
            and the second color is used when the cell value is above the threshold.
        interval : int, optional
            Delay between frames in milliseconds, by default 200.
            Controls the speed of the animation (smaller values = faster animation).
        text_loc : list[Any, Any], optional
            Location of the time label text as [x, y] coordinates, by default None.
            If None, defaults to [0.1, 0.2].
        point_color : str, optional
            Color of the points, by default "red".
            Any valid matplotlib color string.
        point_size : int, optional
            Size of the points, by default 100.
            Controls the marker size.
        pid_color : str, optional
            Color of the point value annotations, by default "blue".
            Any valid matplotlib color string.
        pid_size : int, optional
            Size of the point value annotations, by default 10.
            Controls the font size of the annotations.
        **kwargs : dict
            Additional keyword arguments for customizing the animation.

            Plot appearance:
            ---------------
            title : str, optional
                Title of the plot, by default 'Array Plot'.
            title_size : int, optional
                Title font size, by default 15.
            cmap : str, optional
                Colormap name, by default 'coolwarm_r'.
            vmin : float, optional
                Minimum value for color scaling, by default min(array).
            vmax : float, optional
                Maximum value for color scaling, by default max(array).

            Color bar options:
            ----------------
            cbar_orientation : str, optional
                Orientation of the color bar, by default 'vertical'.
                Can be 'horizontal' or 'vertical'.
            cbar_label_rotation : float, optional
                Rotation angle of the color bar label, by default -90.
            cbar_label_location : str, optional
                Location of the color bar label, by default 'bottom'.
                Options: 'top', 'bottom', 'center', 'baseline', 'center_baseline'.
            cbar_length : float, optional
                Ratio to control the height/width of the color bar, by default 0.75.
            ticks_spacing : int, optional
                Spacing between ticks on the color bar, by default 2.
            cbar_label_size : int, optional
                Font size of the color bar label, by default 12.
            cbar_label : str, optional
                Label text for the color bar, by default 'Value'.

            Color scale options:
            ------------------
            color_scale : str, optional
                Type of color scaling to use, by default 'linear'.
                Options:
                - 'linear': Linear scale
                - 'power': Power-law normalization
                - 'sym-lognorm': Symmetrical logarithmic scale
                - 'boundary-norm': Discrete intervals based on boundaries
                - 'midpoint': Scale split at a specified midpoint
            gamma : float, optional
                Exponent for 'power' color scale, by default 0.5.
                Values < 1 emphasize lower values, values > 1 emphasize higher values.
            line_threshold : float, optional
                Threshold for 'sym-lognorm' color scale, by default 0.0001.
            line_scale : float, optional
                Scale factor for 'sym-lognorm' color scale, by default 0.001.
            bounds : List, optional
                Boundaries for 'boundary-norm' color scale, by default None.
                Defines the discrete intervals for color mapping.
            midpoint : float, optional
                Midpoint value for 'midpoint' color scale, by default 0.

            Cell value display options:
            -------------------------
            display_cell_value : bool, optional
                Whether to display the values of cells as text, by default False.
            num_size : int, optional
                Font size of the cell value text, by default 8.
            background_color_threshold : float, optional
                Threshold for cell value text color, by default None.
                If cell value > threshold, text is black; otherwise, text is white.
                If None, uses max(array)/2 as the threshold.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object that can be displayed in a notebook or saved to a file.

        Raises
        ------
        ValueError
            If an invalid keyword argument is provided.
        ValueError
            If the length of the time list doesn't match the first dimension of the array.

        Notes
        -----
        The animation is created by iterating through the first dimension of the array.
        For example, if the array has shape (10, 20, 30), the animation will have 10 frames,
        each showing a 20x30 slice of the array.

        To display the animation in a Jupyter notebook, you may need to use:
        ```python
        from IPython.display import HTML
        HTML(anim_obj.to_jshtml())
        ```

        To save the animation to a file, use the `save_animation` method after creating
        the animation.

        Examples
        --------
        Basic animation from a 3D array:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a 3D array with 5 frames, each 10x10
        >>> arr = np.random.randint(1, 10, size=(5, 10, 10))
        >>> # Create labels for each frame
        >>> frame_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
        >>> # Create the ArrayGlyph object
        >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated Array")
        >>> # Create the animation
        >>> anim_obj = animated_array.animate(frame_labels)

        ```
        Animation with custom interval (speed):
        ```python
        >>> # Slower animation (500ms between frames)
        >>> anim_obj = animated_array.animate(frame_labels, interval=500)
        >>> # Faster animation (100ms between frames)
        >>> anim_obj = animated_array.animate(frame_labels, interval=100)

        ```
        Animation with points:
        ```python
        >>> # Create points to display on the animation
        >>> points = np.array([[1, 2, 3], [2, 5, 5], [3, 8, 8]])
        >>> anim_obj = animated_array.animate(
        ...     frame_labels,
        ...     points=points,
        ...     point_color="black",
        ...     point_size=150,
        ...     pid_color="white",
        ...     pid_size=12
        ... )

        ```
        Animation with cell values displayed:
        ```python
        >>> anim_obj = animated_array.animate(
        ...     frame_labels,
        ...     display_cell_value=True,
        ...     num_size=10,
        ...     text_colors=("yellow", "blue")
        ... )

        ```
        Saving the animation to a file:
        ```python
        >>> # Create the animation first
        >>> anim_obj = animated_array.animate(frame_labels)
        >>> # Then save it to a file
        >>> animated_array.save_animation("animation.gif", fps=2)

        ```
        """
        if text_loc is None:
            text_loc = [0.1, 0.2]

        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {DEFAULT_OPTIONS}"
                )
            else:
                self.default_options[key] = val

        # if user did not input ticks spacing use the calculated one.
        if "ticks_spacing" in kwargs.keys():
            self.default_options["ticks_spacing"] = kwargs["ticks_spacing"]
        else:
            self.default_options["ticks_spacing"] = self.ticks_spacing

        if "vmin" in kwargs.keys():
            self.default_options["vmin"] = kwargs["vmin"]
        else:
            self.default_options["vmin"] = self.vmin

        if "vmax" in kwargs.keys():
            self.default_options["vmax"] = kwargs["vmax"]
        else:
            self.default_options["vmax"] = self.vmax

        # if optional_display
        precision = self.default_options["precision"]
        array = self.arr
        fig, ax = self.fig, self.ax

        ticks = self.get_ticks()
        im, cbar_kw = self._plot_im_get_cbar_kw(ax, array[0, :, :], ticks)

        # Create colorbar
        cbar = ax.figure.colorbar(
            im,
            ax=ax,
            shrink=self.default_options["cbar_length"],
            orientation=self.default_options["cbar_orientation"],
            **cbar_kw,
        )
        cbar.ax.set_ylabel(
            self.default_options["cbar_label"],
            rotation=self.default_options["cbar_label_rotation"],
            va=self.default_options["cbar_label_location"],
            fontsize=self.default_options["cbar_label_size"],
        )
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(
            self.default_options["title"], fontsize=self.default_options["title_size"]
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])

        if self.default_options["display_cell_value"]:
            indices = get_indices2(array[0, :, :], [np.nan])
            cell_text_value = self._plot_text(
                ax, array[0, :, :], indices, self.default_options
            )
            indices = np.array(indices)

        if points is not None:
            row = points[:, 1]
            col = points[:, 2]
            points_scatter = ax.scatter(col, row, color=point_color, s=point_size)
            points_id = self._plot_point_values(ax, points, pid_color, pid_size)

        # Normalize the threshold to the image color range.
        if self.default_options["background_color_threshold"] is not None:
            background_color_threshold = im.norm(
                self.default_options["background_color_threshold"]
            )
        else:
            background_color_threshold = im.norm(np.nanmax(array)) / 2.0

        day_text = ax.text(
            text_loc[0],
            text_loc[1],
            " ",
            fontsize=self.default_options["cbar_label_size"],
        )

        def init():
            """initialize the plot with the first array"""
            im.set_data(array[0, :, :])
            day_text.set_text("")
            output = [im, day_text]

            if points is not None:
                points_scatter.set_offsets(np.c_[col, row])
                output.append(points_scatter)
                update_points = lambda x: points_id[x].set_text(points[x, 0])
                list(map(update_points, range(len(col))))

                output += points_id

            if self.default_options["display_cell_value"]:
                vals = array[0, indices[:, 0], indices[:, 1]]
                update_cell_value = lambda x: cell_text_value[x].set_text(vals[x])
                list(map(update_cell_value, range(self.no_elem)))
                output += cell_text_value

            return output

        def animate_a(i):
            """plot for each element in the iterable."""
            im.set_data(array[i, :, :])
            day_text.set_text("Date = " + str(time[i])[0:10])
            output = [im, day_text]

            if points is not None:
                points_scatter.set_offsets(np.c_[col, row])
                output.append(points_scatter)

                for x in range(len(col)):
                    points_id[x].set_text(points[x, 0])

                output += points_id

            if self.default_options["display_cell_value"]:
                vals = array[i, indices[:, 0], indices[:, 1]]

                def update_cell_value(x):
                    """Update cell value"""
                    val = round(vals[x], precision)
                    kw = {
                        "color": text_colors[
                            int(im.norm(vals[x]) > background_color_threshold)
                        ]
                    }
                    cell_text_value[x].update(kw)
                    cell_text_value[x].set_text(val)

                list(map(update_cell_value, range(self.no_elem)))

                output += cell_text_value

            return output

        plt.tight_layout()

        anim = FuncAnimation(
            fig,
            animate_a,
            init_func=init,
            frames=np.shape(array)[0],
            interval=interval,
            blit=True,
        )
        self._anim = anim
        plt.show()
        return anim

    def save_animation(self, path: str, fps: int = 2):
        """Save the animation to a file.

        This method saves the animation created by the `animate` method to a file.
        The format of the output file is determined by the file extension in the path.

        Parameters
        ----------
        path : str
            The file path where the animation will be saved.
            The file extension determines the output format.
            Supported formats: gif, mov, avi, mp4.
        fps : int, optional
            Frames per second for the saved animation, by default 2.
            Higher values create faster animations, lower values create slower animations.

        Raises
        ------
        ValueError
            If the file extension is not one of the supported formats.
        FileNotFoundError
            If FFmpeg is not installed (required for mov, avi, and mp4 formats).

        Notes
        -----
        - For GIF format, the PillowWriter is used.
        - For MOV, AVI, and MP4 formats, FFMpegWriter is used, which requires FFmpeg to be installed.
        - You can download FFmpeg from https://ffmpeg.org/

        Examples
        --------
        Save an animation as a GIF file:
        ```python
        >>> import numpy as np
        >>> from cleopatra.array_glyph import ArrayGlyph
        >>> # Create a 3D array with 5 frames, each 10x10
        >>> arr = np.random.randint(1, 10, size=(5, 10, 10))
        >>> frame_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
        >>> animated_array = ArrayGlyph(arr)
        >>> anim_obj = animated_array.animate(frame_labels)
        >>> animated_array.save_animation("animation.gif")

        ```
        Save with a higher frame rate for a faster animation:
        ```python
        >>> animated_array.save_animation("animation.gif", fps=5)

        ```
        Save in MP4 format (requires FFmpeg):
        ```python
        >>> animated_array.save_animation("animation.mp4", fps=10)

        ```
        """
        video_format = path.split(".")[-1]
        if video_format not in SUPPORTED_VIDEO_FORMAT:
            raise ValueError(
                f"The given extension {video_format} implies a format that is not supported, "
                f"only {SUPPORTED_VIDEO_FORMAT} are supported"
            )

        if video_format == "gif":
            writer_gif = animation.PillowWriter(fps=fps)
            self.anim.save(path, writer=writer_gif)
        else:
            try:
                if video_format == "avi" or video_format == "mov":
                    writer_video = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    self.anim.save(path, writer=writer_video)
                elif video_format == "mp4":
                    writer_mp4 = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    self.anim.save(path, writer=writer_mp4)
            except FileNotFoundError:
                print(
                    "Please visit https://ffmpeg.org/ and download a version of ffmpeg compatible with your operating"
                    "system, for more details please check the method definition"
                )

    # @staticmethod
    # def plot_type_1(
    #     Y1,
    #     Y2,
    #     Points,
    #     PointsY,
    #     PointMaxSize=200,
    #     PointMinSize=1,
    #     X_axis_label="X Axis",
    #     LegendNum=5,
    #     LegendLoc=(1.3, 1),
    #     PointLegendTitle="Output 2",
    #     Ylim=[0, 180],
    #     Y2lim=[-2, 14],
    #     color1="#27408B",
    #     color2="#DC143C",
    #     color3="grey",
    #     linewidth=4,
    #     **kwargs,
    # ):
    #     """Plot_Type1.
    #
    #     !TODO Needs docs
    #
    #     Parameters
    #     ----------
    #     Y1 : TYPE
    #         DESCRIPTION.
    #     Y2 : TYPE
    #         DESCRIPTION.
    #     Points : TYPE
    #         DESCRIPTION.
    #     PointsY : TYPE
    #         DESCRIPTION.
    #     PointMaxSize : TYPE, optional
    #         DESCRIPTION. The default is 200.
    #     PointMinSize : TYPE, optional
    #         DESCRIPTION. The default is 1.
    #     X_axis_label : TYPE, optional
    #         DESCRIPTION. The default is 'X Axis'.
    #     LegendNum : TYPE, optional
    #         DESCRIPTION. The default is 5.
    #     LegendLoc : TYPE, optional
    #         DESCRIPTION. The default is (1.3, 1).
    #     PointLegendTitle : TYPE, optional
    #         DESCRIPTION. The default is "Output 2".
    #     Ylim : TYPE, optional
    #         DESCRIPTION. The default is [0,180].
    #     Y2lim : TYPE, optional
    #         DESCRIPTION. The default is [-2,14].
    #     color1 : TYPE, optional
    #         DESCRIPTION. The default is '#27408B'.
    #     color2 : TYPE, optional
    #         DESCRIPTION. The default is '#DC143C'.
    #     color3 : TYPE, optional
    #         DESCRIPTION. The default is "grey".
    #     linewidth : TYPE, optional
    #         DESCRIPTION. The default is 4.
    #     **kwargs : TYPE
    #         DESCRIPTION.
    #
    #     Returns
    #     -------
    #     ax1 : TYPE
    #         DESCRIPTION.
    #     TYPE
    #         DESCRIPTION.
    #     fig : TYPE
    #         DESCRIPTION.
    #     """
    #     fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    #
    #     ax2 = ax1.twinx()
    #
    #     ax1.plot(
    #         Y1[:, 0],
    #         Y1[:, 1],
    #         zorder=1,
    #         color=color1,
    #         linestyle=Styles.get_line_style(0),
    #         linewidth=linewidth,
    #         label="Model 1 Output1",
    #     )
    #
    #     if "Y1_2" in kwargs.keys():
    #         Y1_2 = kwargs["Y1_2"]
    #
    #         rows_axis1, cols_axis1 = np.shape(Y1_2)
    #
    #         if "Y1_2_label" in kwargs.keys():
    #             label = kwargs["Y2_2_label"]
    #         else:
    #             label = ["label"] * (cols_axis1 - 1)
    #         # first column is the x axis
    #         for i in range(1, cols_axis1):
    #             ax1.plot(
    #                 Y1_2[:, 0],
    #                 Y1_2[:, i],
    #                 zorder=1,
    #                 color=color2,
    #                 linestyle=Styles.get_line_style(i),
    #                 linewidth=linewidth,
    #                 label=label[i - 1],
    #             )
    #
    #     ax2.plot(
    #         Y2[:, 0],
    #         Y2[:, 1],
    #         zorder=1,
    #         color=color3,
    #         linestyle=Styles.get_line_style(6),
    #         linewidth=2,
    #         label="Output1-Diff",
    #     )
    #
    #     if "Y2_2" in kwargs.keys():
    #         Y2_2 = kwargs["Y2_2"]
    #         rows_axis2, cols_axis2 = np.shape(Y2_2)
    #
    #         if "Y2_2_label" in kwargs.keys():
    #             label = kwargs["Y2_2_label"]
    #         else:
    #             label = ["label"] * (cols_axis2 - 1)
    #
    #         for i in range(1, cols_axis2):
    #             ax1.plot(
    #                 Y2_2[:, 0],
    #                 Y2_2[:, i],
    #                 zorder=1,
    #                 color=color2,
    #                 linestyle=Styles.get_line_style(i),
    #                 linewidth=linewidth,
    #                 label=label[i - 1],
    #             )
    #
    #     if "Points1" in kwargs.keys():
    #         # first axis in the x axis
    #         Points1 = kwargs["Points1"]
    #
    #         vmax = np.max(Points1[:, 1:])
    #         vmin = np.min(Points1[:, 1:])
    #
    #         vmax = max(Points[:, 1].max(), vmax)
    #         vmin = min(Points[:, 1].min(), vmin)
    #
    #     else:
    #         vmax = max(Points)
    #         vmin = min(Points)
    #
    #     vmaxnew = PointMaxSize
    #     vminnew = PointMinSize
    #
    #     Points_scaled = [
    #         Scale.rescale(x, vmin, vmax, vminnew, vmaxnew) for x in Points[:, 1]
    #     ]
    #     f1 = np.ones(shape=(len(Points))) * PointsY
    #     scatter = ax2.scatter(
    #         Points[:, 0],
    #         f1,
    #         zorder=1,
    #         c=color1,
    #         s=Points_scaled,
    #         label="Model 1 Output 2",
    #     )
    #
    #     if "Points1" in kwargs.keys():
    #         row_points, col_points = np.shape(Points1)
    #         PointsY1 = kwargs["PointsY1"]
    #         f2 = np.ones_like(Points1[:, 1:])
    #
    #         for i in range(col_points - 1):
    #             Points1_scaled = [
    #                 Scale.rescale(x, vmin, vmax, vminnew, vmaxnew)
    #                 for x in Points1[:, i]
    #             ]
    #             f2[:, i] = PointsY1[i]
    #
    #             ax2.scatter(
    #                 Points1[:, 0],
    #                 f2[:, i],
    #                 zorder=1,
    #                 c=color2,
    #                 s=Points1_scaled,
    #                 label="Model 2 Output 2",
    #             )
    #
    #     # produce a legend with the unique colors from the scatter
    #     legend1 = ax2.legend(
    #         *scatter.legend_elements(), bbox_to_anchor=(1.1, 0.2)
    #     )  # loc="lower right", title="RIM"
    #
    #     ax2.add_artist(legend1)
    #
    #     # produce a legend with a cross section of sizes from the scatter
    #     handles, labels = scatter.legend_elements(
    #         prop="sizes", alpha=0.6, num=LegendNum
    #     )
    #     # L = [vminnew] + [float(i[14:-2]) for i in labels] + [vmaxnew]
    #     L = [float(i[14:-2]) for i in labels]
    #     labels1 = [
    #         round(Scale.rescale(x, vminnew, vmaxnew, vmin, vmax) / 1000) for x in L
    #     ]
    #
    #     legend2 = ax2.legend(
    #         handles, labels1, bbox_to_anchor=LegendLoc, title=PointLegendTitle
    #     )
    #     ax2.add_artist(legend2)
    #
    #     ax1.set_ylim(Ylim)
    #     ax2.set_ylim(Y2lim)
    #     #
    #     ax1.set_ylabel("Output 1 (m)", fontsize=12)
    #     ax2.set_ylabel("Output 1 - Diff (m)", fontsize=12)
    #     ax1.set_xlabel(X_axis_label, fontsize=12)
    #     ax1.xaxis.set_minor_locator(plt.MaxNLocator(10))
    #     ax1.tick_params(which="minor", length=5)
    #     fig.legend(
    #         loc="lower center",
    #         bbox_to_anchor=(1.3, 0.3),
    #         bbox_transform=ax1.transAxes,
    #         fontsize=10,
    #     )
    #     plt.rcParams.update({"ytick.major.size": 3.5})
    #     plt.rcParams.update({"font.size": 12})
    #     plt.title("Model Output Comparison", fontsize=15)
    #
    #     plt.subplots_adjust(right=0.7)
    #     # plt.tight_layout()
    #
    #     return (ax1, ax2), fig
