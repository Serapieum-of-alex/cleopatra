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

from typing import Any, Union, List, Tuple, Dict
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import numpy.ma as ma
from hpc.indexing import get_indices2

# from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap
import matplotlib.ticker as ticker
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import MidpointNormalize
from PIL import Image

DEFAULT_OPTIONS = dict(
    vmin=None,
    vmax=None,
    num_size=8,
    display_cell_value=False,
    background_color_threshold=None,
    id_color="green",
    id_size=20,
    precision=2,
)
DEFAULT_OPTIONS = STYLE_DEFAULTS | DEFAULT_OPTIONS
SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4"]
COLOR_SCALE = ["linear", "power", "lognorm", "boundary", "midpoint"]


class ArrayGlyph:
    """A class to handle 3D arrays and perform various operations on them."""

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
        """Array.

        Parameters
        ----------
        array: np.ndarray
            array.
        exclude_value: numeric, Optional, Default is np.nan.
            value used to fill cells out of the domain.
        extent: List, Default is None.
            [xmin, ymin, xmax, ymax].
        rgb: List
            the indices of the red, green, and blue bands in the given array. the `rgb` parameter can be a list of
            three values, or a list of four values if the alpha band is also included.
        surface_reflectance: int, Default is 10000.
            surface reflectance value of the sentinel data.
        cutoff: List, Default is None.
            clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
            and scale them back to between 0 and 1.
        percentile: int
            The percentile value to be used for scaling.
        **kwargs:
            figsize : [tuple], optional
                    figure size. The default is (8,8).

        the object does not need any parameters to be initialized.

        Examples
        --------
        - Create an array and instantiate the `Array` object.

            >>> import numpy as np
            >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> array = ArrayGlyph(arr)
            >>> fig, ax = array.plot()

        .. image:: /_images/image-plot.png
            :alt: Example Image
            :align: center
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

        self._arr = array
        # get the tick spacing that has 10 ticks only
        self.ticks_spacing = (self._vmax - self._vmin) / 10
        shape = array.shape
        if len(shape) == 3:
            no_elem = array[0, :, :].count()
        else:
            no_elem = array.count()

        self.no_elem = no_elem

        if fig is not None:
            self.fig, self.ax = fig, ax
        else:
            self.fig = None

    @property
    def arr(self):
        """array"""
        return self._arr

    @arr.setter
    def arr(self, value):
        self._arr = value

    def prepare_array(
        self,
        array: np.ndarray,
        rgb: List[int] = None,
        surface_reflectance: int = None,
        cutoff: List = None,
        percentile: int = None,
    ) -> np.ndarray:
        """Prepare Array.

        Parameters
        ----------
        array: np.ndarray
            array.
        rgb: List, Default is [3,2,1]
            the indices of the red, green, and blue bands in the given array.
        surface_reflectance: int, Default is 10000.
            surface reflectance value of the sentinel data.
        cutoff: List, Default is None.
            clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
            and scale them back to between 0 and 1).
        percentile: int
            The percentile value to be used for scaling.

        Returns
        -------
        np.ndarray: np.float32/input dtype
            the rgb 3d array is converted into 2d array to be plotted using the plt.imshow function.
            a float32 array normalized between 0 and 1 using the `percentile` values or the `surface_reflectance`.
            if the `percentile` or `surface_reflectance` values are not given, the function just reorders the values
            to have the red-green-blue order.

        .. note::

            - The `prepare_array` function is called in the constructor of the `ArrayGlyph` class to prepare the array,
              so you can provide the same parameters of the `prepare_array` function to the `ArrayGlyph constructor`.
            - The prepare function moves the first axes (the channel axis) to the last axes, and then scales the array
              using the percentile values. If the percentile is not given, the function scales the array using the
              surface reflectance values. If the surface reflectance is not given, the function scales the array using
              the cutoff values. If the cutoff is not given, the function scales the array using the sentinel data

        Examples
        --------
        - Create an array and instantiate the `ArrayGlyph` class.

            >>> import numpy as np
            >>> arr = np.random.randint(0, 255, size=(3, 5, 5)).astype(np.float32)
            >>> array_glyph = ArrayGlyph(arr)
            >>> print(array_glyph.arr.shape)
            (3, 5, 5)

        `rgb` channels:
            - Now let's use the `prepare_array` function with `rgb` channels as [0, 1, 2]. so the finction does not to
                reorder the chennels. but it just needs to move the first axis to the last axis.

            >>> rgb_array = array_glyph.prepare_array(arr, rgb=[0, 1, 2])
            >>> print(rgb_array.shape)
            (5, 5, 3)

            - If we compare the values of the first channel in the original array with the first array in the rgb array it
                should be the same.

            >>> np.testing.assert_equal(arr[0, :, :],rgb_array[:, :, 0])

        surface_reflectance:
            - if you provide the surface reflectance value, the function will scale the array using the surface reflectance
                value to a normalized rgb values.

            >>> array_glyph = ArrayGlyph(arr)
            >>> rgb_array = array_glyph.prepare_array(arr, surface_reflectance=10000, rgb=[0, 1, 2])
            >>> print(rgb_array.shape)
            (5, 5, 3)

            - if you print the values of the first channel, you will find all the values are between 0 and 1.
            >>> print(rgb_array[:, :, 0])
            [[0.0195 0.02   0.0109 0.0211 0.0087]
             [0.0112 0.0221 0.0035 0.0234 0.0141]
             [0.0116 0.0188 0.0001 0.0176 0.    ]
             [0.0014 0.0147 0.0043 0.0167 0.0117]
             [0.0083 0.0139 0.0186 0.02   0.0058]]

            - With the `surface_reflectance` parameter, you can also use the `cutoff` parameter to affect values that
                are above it, by rescaling them.

            >>> rgb_array = array_glyph.prepare_array(
            ...     arr, surface_reflectance=10000, rgb=[0, 1, 2], cutoff=[0.8, 0.8, 0.8]
            ... )
            >>> print(rgb_array[:, :, 0])
            [[0.     0.     0.     0.     0.    ]
             [1.     1.     1.     1.     1.    ]
             [1.     1.     1.     1.     1.    ]
             [0.0014 0.0147 0.0043 0.0167 0.0117]
             [0.0083 0.0139 0.0186 0.02   0.0058]]
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
        """Prepare for RGB plot.

        Parameters
        ----------
        array: np.ndarray
            array.
        rgb: List, Default is [3,2,1]
            the indices of the red, green, and blue bands in the given array.
        surface_reflectance: int, Default is 10000.
            surface reflectance value of the sentinel data.
        cutoff: List, Default is None.
            clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
            and scale them back to between 0 and 1).

        Returns
        -------
        np.ndarray:
            the rgb 3d array is converted into 2d array to be plotted using the plt.imshow function.
        """
        array = np.clip(array / surface_reflectance, 0, 1)
        if cutoff is not None:
            array[0] = np.clip(rgb[0], 0, cutoff[0]) / cutoff[0]
            array[1] = np.clip(rgb[1], 0, cutoff[1]) / cutoff[1]
            array[2] = np.clip(rgb[2], 0, cutoff[2]) / cutoff[2]

        return array

    @staticmethod
    def scale_percentile(arr: np.ndarray, percentile: int = 1) -> np.ndarray:
        """Scale the array.

        Parameters
        ----------
        arr: np.ndarray
            The array to be scaled.
        percentile: int
            The percentile value to be used for scaling.

        Returns
        -------
        np.ndarray: float32
            The scaled array, normalized between 0 and 1. using the percentile values.
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
        fig = plt.figure(figsize=self.default_options["figsize"])
        # gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
        ax = fig.add_subplot()  # gs[:,:]
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

        if color_scale.lower() == COLOR_SCALE[0]:
            # linear
            im = ax.matshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=self.extent)
            cbar_kw = dict(ticks=ticks)
        elif color_scale.lower() == COLOR_SCALE[1]:
            # power
            im = ax.matshow(
                arr,
                cmap=cmap,
                norm=colors.PowerNorm(
                    gamma=self.default_options["gamma"], vmin=vmin, vmax=vmax
                ),
                extent=self.extent,
            )
            cbar_kw = dict(ticks=ticks)
        elif color_scale.lower() == COLOR_SCALE[2]:
            # lognorm
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
            formatter = ticker.LogFormatter(10, labelOnlyBase=False)
            cbar_kw = dict(ticks=ticks, format=formatter)
        elif color_scale.lower() == COLOR_SCALE[3]:
            # boundary
            if not self.default_options["bounds"]:
                bounds = ticks
                cbar_kw = dict(ticks=ticks)
            else:
                bounds = self.default_options["bounds"]
                cbar_kw = dict(ticks=self.default_options["bounds"])
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(arr, cmap=cmap, norm=norm, extent=self.extent)
        elif color_scale.lower() == COLOR_SCALE[4]:
            # midpoint
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
            cbar_kw = dict(ticks=ticks)
        else:
            raise ValueError(
                f"Invalid color scale option: {color_scale}. Use {COLOR_SCALE}"
            )

        return im, cbar_kw

    def apply_colormap(self, cmap: Union[Colormap, str]) -> np.ndarray:
        """Apply a matplotlib colormap to an array.

            Create an RGB channel from the given array using the given colormap.

        Parameters
        ----------
        cmap: Colormap/str
            colormap.

        Returns
        -------
        np.ndarray: 8-bit array
            the array with the colormap applied.

        Examples
        --------
        - Create an array and instantiate the `Array` object.

            >>> import numpy as np
            >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> array = ArrayGlyph(arr)
            >>> rgb_array = array.apply_colormap("coolwarm_r")
            >>> print(rgb_array)
            [[[179   3  38]
              [221  96  76]
              [244 154 123]]
             [[244 196 173]
              [220 220 221]
              [183 207 249]]
             [[139 17 4 253]
              [ 96 128 232]
              [ 58  76 192]]]
              >>> print(rgb_array.dtype)
                uint8
        """
        colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        normed_data = (self.arr - self.arr.min()) / (self.arr.max() - self.arr.min())
        colored = colormap(normed_data)
        return (colored[:, :, :3] * 255).astype("uint8")

    def to_image(self, arr: np.ndarray = None) -> Image.Image:
        """Create an RGB image from an array.

            convert the array to an image.

        Parameters
        ----------
        arr: np.ndarray, default is None.
            array. if None, the array in the object will be used.

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr)
        >>> image = array.to_image()
        >>> print(image) # doctest: +SKIP
        <PIL.Image.Image image mode=RGB size=3x3 at 0x7F5E0D2F4C40>
        """
        if arr is None:
            arr = self.arr
        # This is done to scale the values between 0 and 255
        arr = arr if arr.dtype == "uint8" else self.scale_to_rgb()
        return Image.fromarray(arr).convert("RGB")

    def scale_to_rgb(self, arr: np.ndarray = None) -> np.ndarray:
        """Create an RGB image.

        Parameters
        ----------
        arr: np.ndarray, default is None.
            array. if None, the array in the object will be used.

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array = ArrayGlyph(arr)
        >>> rgb_array = array.scale_to_rgb()
        >>> print(rgb_array)
        [[28 56 85]
         [113 141 170]
         [198 226 255]]
         >>> print(rgb_array.dtype)
        """
        if arr is None:
            arr = self.arr
        # This is done to scale the values between 0 and 255
        return (arr * 255 / arr.max()).astype("uint8")

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
        """plot an array.

        Parameters
        ----------
        points : [array]
            3 column array with the first column as the value you want to display for the point, the second is the rows'
            index of the point in the array, and the third column as the column index in the array.
            - the second and third column tells the location of the point in the array.
        point_color: [str]
            color.
        point_size: [Any]
            size of the point.
        pid_color: [str]
            the annotation color of the point. Default is blue.
        pid_size: [Any]
            size of the point annotation.
        **kwargs: [dict]
            title: [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size: [integer], optional, default is 15.
                title size.
            cbar_orientation: [string], optional, default is 'vertical'
                orientation of the color bar horizontal/vertical.
            cbar_label_rotation: [number], optional, default is -90.
                rotation of the color bar label.
            cbar_label_location: str, optional, default is 'bottom'.
                location of the color bar title 'top', 'bottom', 'center', 'baseline', 'center_baseline'.
            cbar_length: float, optional
                ratio to control the height of the color bar. The default is 0.75.
            ticks_spacing: int, optional
                Spacing in the color bar ticks. The default is 2.
            cbar_label_size: integer, optional
                size of the color bar label. The default is 12.
            cbar_label: str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            color_scale : integer, optional, default is 1.
                there are 5 options to change the scale of the colors.

                1- `linear`:
                    linear scale.
                2- `power`:
                    for the power scale. Linearly map a given value to the 0-1 range and then apply a power-law
                    normalization over that range.
                3- `lognorm`:
                    the symmetrical logarithmic scale `SymLogNorm` is logarithmic in both the positive and
                    negative directions from the origin.
                4- `boundary`:
                    the Boundary scale generates a colormap index based on discrete intervals.
                5- `midpoint`:
                    the midpoint scale splits the scale into 2 halfs, be the given value.
            gamma: [float], optional, default is 0.5.
                value needed for the color_scale `power`.
            line_threshold: float, optional, default is 0.0001.
                value needed for the color_scale `lognorm`.
            line_scale: float, optional, default is 0.001.
                value needed for the color_scale `lognorm`.
            bounds: List, default is None,
                a list of number to be used as a discrete bounds for the color scale `boundary`.
            midpoint: float, optional, default is 0.
                value needed for the color_scale `midpoint`.
            cmap: str, optional, default is 'coolwarm_r'.
                color style.
            display_cell_value: bool
                True if you want to display the values of the cells as a text
            num_size: integer, optional, default is 8.
                size of the numbers plotted on top of each cell.
            background_color_threshold: [float/integer], optional, default is None.
                threshold value if the value of the cell is greater, the plotted
                numbers will be black, and if smaller the plotted number will be white
                if None given the max value/2 is considered.

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object

        Examples
        --------
        - Create an array and instantiate the `Array` object.

            >>> import numpy as np
            >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Customized Plot", title_size=18)
            >>> fig, ax = array.plot()

        .. image:: /_images/array-plot.png
            :alt: Example Image
            :align: center

        - Color bar customization:

            - Create an array and instantiate the `Array` object with custom options.

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

                .. image:: /_images/color-bar-customization.png
                    :alt: Example Image
                    :align: center
                - Color bar customization:

        - Display values for each cell:

            - you can display the values for each cell by using thr parameter `display_cell_value`, and customize how
                the values are displayed using the parameter `background_color_threshold` and `num_size`.

                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display array values", title_size=18)
                >>> fig, ax = array.plot(
                ...     display_cell_value=True,
                ...     num_size=12
                ... )

                .. image:: /_images/display-cell-values.png
                    :alt: Example Image
                    :align: center

        - Plot points at specific locations in the array:

            - you can display points in specific cells in the array and also display a value for each of these points.
                The point parameter takes an array with the first column as the values to be displayed on top of the
                points, the second and third columns are the row and column index of the point in the array.
            - The `point_color` and `point_size` parameters are used to customize the appearance of the points,
                while the `pid_color` and `pid_size` parameters are used to customize the appearance of the point
                IDs/text.

                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Display Points in top of the array", title_size=14)
                >>> points = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2]])
                >>> fig, ax = array.plot(
                ...     points=points,
                ...     point_color="black",
                ...     point_size=100,
                ...     pid_color="orange",
                ...     pid_size=30,
                ... )

                .. image:: /_images/display-points.png
                    :alt: Example Image
                    :align: center

        - Color scale customization:

            - Power scale.

                - The default power scale uses a gamma value of 0.5.

                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="power",
                    ...     cmap="coolwarm_r",
                    ... )

                    .. image:: /_images/power-scale.png
                        :alt: Example Image
                        :align: center

                - change the gamma of 0.8.

                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale: gamma=0.8", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="power",
                    ...     gamma=0.8,
                    ...     cmap="coolwarm_r",
                    ... )

                    .. image:: /_images/power-scale-gamma-0.8.png
                        :alt: Example Image
                        :align: center

                - change the gamma of 0.1.

                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Power scale: gamma=0.1", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="power",
                    ...     gamma=0.1,
                    ...     cmap="coolwarm_r",
                    ... )

                    .. image:: /_images/power-scale-gamma-0.1.png
                        :alt: Example Image
                        :align: center

            - Logarithmic scale.

                - the logarithmic scale uses to parameters `line_threshold` and `line_scale` with a default
                value if 0.0001, and 0.001 respectively.

                    >>> array = ArrayGlyph(arr, figsize=(6, 6), title="logarithmic scale", title_size=18)
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="lognorm",
                    ...     cmap="coolwarm_r",
                    ... )

                    .. image:: /_images/log-scale.png
                        :alt: Example Image
                        :align: center

                - you can change the `line_threshold` and `line_scale` values.

                    >>> array = ArrayGlyph(
                    ...     arr, figsize=(6, 6), title="Logarithmic scale: Customized Parameter", title_size=12
                    ... )
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="lognorm",
                    ...     cmap="coolwarm_r",
                    ...     line_threshold=0.015,
                    ...     line_scale=0.1,
                    ... )

                    .. image:: /_images/log-scale-custom-parameters.png
                        :alt: Example Image
                        :align: center

            - Defined boundary scale.

                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Defined boundary scale", title_size=18)
                >>> fig, ax = array.plot(
                ...     cbar_label_rotation=-90,
                ...     cbar_label="Discharge m3/s",
                ...     color_scale="boundary",
                ...     cmap="coolwarm_r",
                ... )

                .. image:: /_images/boundary-scale.png
                    :alt: Example Image
                    :align: center

                - You can also define the boundaries.

                    >>> array = ArrayGlyph(
                    ...     arr, figsize=(6, 6), title="Defined boundary scale: defined bounds", title_size=18
                    ... )
                    >>> bounds = [0, 5, 10]
                    >>> fig, ax = array.plot(
                    ...     cbar_label_rotation=-90,
                    ...     cbar_label="Discharge m3/s",
                    ...     color_scale="boundary",
                    ...     bounds=bounds,
                    ...     cmap="coolwarm_r",
                    ... )

                    .. image:: /_images/boundary-scale-defined-bounds.png
                        :alt: Example Image
                        :align: center

            - Midpoint scale.

                in the midpoint scale you can define a value that splits the scale into half.
                >>> array = ArrayGlyph(arr, figsize=(6, 6), title="Midpoint scale", title_size=18)
                >>> fig, ax = array.plot(
                ...     cbar_label_rotation=-90,
                ...     cbar_label="Discharge m3/s",
                ...     color_scale="midpoint",
                ...     cmap="coolwarm_r",
                ...     midpoint=2,
                ... )

                .. image:: /_images/midpoint-scale-costom-parameters.png
                    :alt: Example Image
                    :align: center
        """
        for key, val in kwargs.items():
            if key not in self.default_options.keys():
                raise ValueError(
                    f"The given keyword argument:{key} is not correct, possible parameters are,"
                    f" {DEFAULT_OPTIONS}"
                )
            else:
                self.default_options[key] = val

        if self.fig is None:
            self.fig, self.ax = self.create_figure_axes()

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

    def adjust_ticks(
        self,
        axis: str,
        multiply_value: Union[float, int] = 1,
        add_value: Union[float, int] = 0,
        fmt: str = "{0:g}",
        visible: bool = True,
    ):
        """Adjust the ticks of the axes.

        Parameters
        ----------
        axis: str
            x or y.
        multiply_value: Union[float, int]
            value to be multiplied.
        add_value: Union[float, int]
            value to be added.
        fmt: str, default is "{0:g}".
            format of the ticks.
            - 123.456 with the format "{0:f}" will give '123.456000'.
            - 123.456 with the format "{0:.2f}" will give '123.46'.
            - 123456.789 with the format "{0:e}" will give '1.234568e+05'.
            - 123456.789 with the format "{0:.2e}" will give '1.23e+05'.
            - 123456.789 with the format "{0:g}" will give '123457'.
            - 123456.789 with the format "{0:.2g}" will give '1.2e+05'.
            - 123 with the format "{0:d}" will give '123'.
         visible: bool, optional, default is True.
            Whether the ticks are visible or not.

        Returns
        -------
        None

        Examples
        --------
        - Create an array and instantiate the `ArrayGlyph` object.

            >>> import numpy as np
            >>> arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
            >>> extent = [34.62, 34.65, 31.82, 31.85]
            >>> my_glyph = ArrayGlyph(arr, extent=extent)
            >>> fig, ax = my_glyph.plot()

            .. image:: /_images/adjust_tick.png
                :alt: Example Image
                :align: center

        - Adjust the ticks of the x-axis.

            >>> my_glyph.adjust_ticks(axis='x', multiply_value=0.01, add_value=34.62, fmt="{0:.2f}")

            .. image:: /_images/adjust_tick-x.png
                :alt: Example Image
                :align: center

        - Adjust the ticks of the y-axis.

            >>> my_glyph.adjust_ticks(axis='y', multiply_value=0.01, add_value=31.82, fmt="{0:.2e}")

            .. image:: /_images/adjust_tick-y.png
                :alt: Example Image
                :align: center
        """
        if axis == "x":
            ticks_x = ticker.FuncFormatter(
                lambda x, pos: fmt.format(x * multiply_value + add_value)
            )
            self.ax.xaxis.set_major_formatter(ticks_x)
        else:
            ticks_y = ticker.FuncFormatter(
                lambda y, pos: fmt.format(y * multiply_value + add_value)
            )
            self.ax.yaxis.set_major_formatter(ticks_y)

        if not visible:
            if axis == "x":
                self.ax.get_xaxis().set_visible(visible)
            else:
                self.ax.get_yaxis().set_visible(visible)

        plt.show()

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
        """AnimateArray.

        plot an animation for 3d arrays

        Parameters
        ----------
        time : List[Any]
            A list containing the date of values for each frame in the animation.
        points : np.ndarray, optional
            A 3 column array with the first column as the value you want to display for the point, the second is the
            rows index of the point in the array, and the third column as the column index in the array.
            - the second and third column tells the location of the point in the array.
        point_color : str, optional, default is 'red'.
            The color of the points.
        point_size: int, optional, default is 100.
            The size of the point.
        pid_color : str, optional, default is blue.
            The annotation color of the point.
        pid_size : int, optional, default is 10.
            The size of the point annotation.
        text_colors : Tuple[str, str], optional, The default is ("white","black").
            Two colors to be used to plot the values on top of each cell.
        interval: int, optional, default is 200.
            number to control the speed of the animation.
        text_loc: list, optional, default is [0.1,0.2].
            location of the date text.
        **kwargs: [dict]
            figsize: [tuple], optional
                figure size. The default is (8,8).
            title: [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size: [integer], optional
                title size. The default is 15.
            cbar_orientation: [string], optional
                orientation of the color bar horizontal/vertical. The default is 'vertical'.
            cbar_label_rotation: [number], optional
                rotation of the color bar label. The default is -90.
            cbar_label_location: str, optional, default is 'bottom'.
                location of the color bar title 'top', 'bottom', 'center', 'baseline', 'center_baseline'.
            cbar_length: float, optional
                ratio to control the height of the color bar. The default is 0.75.
            ticks_spacing: int, optional
                Spacing in the color bar ticks. The default is 2.
            cbar_label_size: integer, optional
                size of the color bar label. The default is 12.
            cbar_label: str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            color_scale : integer, optional, default is 1.
                there are 5 options to change the scale of the colors.

                1- `linear`:
                    linear scale.
                2- `power`:
                    for the power scale. Linearly map a given value to the 0-1 range and then apply a power-law
                    normalization over that range.
                3- `lognorm`:
                    the symmetrical logarithmic scale `SymLogNorm` is logarithmic in both the positive and
                    negative directions from the origin.
                4- `boundary`:
                    the Boundary scale generates a colormap index based on discrete intervals.
                5- `midpoint`:
                    the midpoint scale splits the scale into 2 halfs, be the given value.
            gamma: [float], optional, default is 0.5.
                value needed for the color_scale `power`.
            line_threshold: float, optional, default is 0.0001.
                value needed for the color_scale `lognorm`.
            line_scale: float, optional, default is 0.001.
                value needed for the color_scale `lognorm`.
            bounds: List, default is None,
                a list of number to be used as a discrete bounds for the color scale `boundary`.
            midpoint: float, optional, default is 0.
                value needed for the color_scale `midpoint`.
            cmap: str, optional, default is 'coolwarm_r'.
                color style.
            display_cell_value: bool
                True if you want to display the values of the cells as a text
            num_size: integer, optional, default is 8.
                size of the numbers plotted on top of each cell.
            background_color_threshold: [float/integer], optional, default is None.
                threshold value if the value of the cell is greater, the plotted
                numbers will be black, and if smaller the plotted number will be white
                if None given the max value/2 is considered.

        Returns
        -------
        animation.FuncAnimation.

        Examples
        --------
        - First create a 3D array with the first dimension `frame_0 = arr[0, :, :]` as the dimension that the function
            will loop over as the frame, then create a list of what you want to be displayed with each frame (i.e.,
            time stamp, counter, ...)

            >>> import numpy as np
            >>> arr = np.random.randint(1, 10, size=(5, 10, 10))
            >>> animate_time_list = [1, 2, 3, 4, 5]
            >>> animated_array = ArrayGlyph(arr, figsize=(8, 8), title="Animated 3D array", title_size=18)
            >>> anim_obj = animated_array.animate(animate_time_list)

            .. image:: /_images/animated_array.gif
                :alt: Example Image
                :align: center

        - To save the animation to a file, use the `save_animation` method, and provide the frame per second `fps`
            parameter.

            >>> animated_array.save_animation("animated_array.gif", fps=2)
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

        if self.fig is None:
            self.fig, self.ax = self.create_figure_axes()

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
                    kw = dict(
                        color=text_colors[
                            int(im.norm(vals[x]) > background_color_threshold)
                        ]
                    )
                    cell_text_value[x].update(kw)
                    cell_text_value[x].set_text(val)

                list(map(update_cell_value, range(self.no_elem)))

                output += cell_text_value

            return output

        plt.tight_layout()
        # global anim
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
        """Save the animation.

            - video format is taken from the given path. available ["gif", "mov", "avi", "mp4"].

        Parameters
        ----------
        path: [str]
            path
        fps: [int]
            frames per second. Default is 2.
        """
        # ffmpegPath = os.getenv("HOME") + "/.matplotlib/ffmpeg-static/bin/ffmpeg.exe"
        # mpl.rcParams["animation.ffmpeg_path"] = ffmpegPath
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
