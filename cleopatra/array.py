"""plotting Array."""
from typing import Any, Union, List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from hpc.indexing import get_indices2

# from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib.ticker import LogFormatter
from cleopatra.styles import DEFAULT_OPTIONS as style_defaults
from cleopatra.styles import MidpointNormalize

DEFAULT_OPTIONS = dict(
    vmin=None,
    vmax=None,
    num_size=8,
    display_cell_value=False,
    background_color_threshold=None,
    id_color="green",
    id_size=20,
    precission=2,
)
DEFAULT_OPTIONS = style_defaults | DEFAULT_OPTIONS
SUPPORTED_VIDEO_FORMAT = ["gif", "mov", "avi", "mp4"]


class Array:
    """Map."""

    def __init__(
        self,
        array: np.ndarray,
        exclude_value: List = np.nan,
        extent: List = None,
        rgb: List[int] = None,
        surface_reflectance: int = 10000,
        cutoff: List = None,
        **kwargs,
    ):
        """Plot array.

        Parameters
        ----------
        array: [numpy array]
            array.
        exclude_value : [numeric]
            value used to fill cells out of the domain. Optional, Default is np.nan.
        extent: [List]
            [xmin, ymin, xmax, ymax]. Default is None.
        rgb: [List]
            Default is [3,2,1]
        surface_reflectance: [int]
            Default is 10000.
        cutoff: [List]
            clip the range of pixel values for each band. (take only the pixel values from 0 to value of the cutoff and
            scale them back to between 0 and 1. Default is None.

        the object does not need any parameters to be initialized.
        """
        # first replace the no_data_value by nan
        # convert the array to float32 to be able to replace the no data value with nan
        if exclude_value is not np.nan:
            array = array.astype(np.float32)
            if len(exclude_value) > 1:
                mask = np.logical_or(
                    np.isclose(array, exclude_value[0], rtol=0.001),
                    np.isclose(array, exclude_value[1], rtol=0.001),
                )
            else:
                mask = np.isclose(array, exclude_value[0], rtol=0.0000001)

            array[mask] = np.nan

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
                    f"{len(array.shape[0])}"
                )
            else:
                array = self._prepare_rgb(
                    array,
                    rgb=rgb,
                    surface_reflectance=surface_reflectance,
                    cutoff=cutoff,
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
            no_elem = np.size(array[0, :, :]) - np.count_nonzero(
                (array[0, np.isnan(array[0, :, :])])
            )
        else:
            no_elem = np.size(array[:, :]) - np.count_nonzero((array[np.isnan(array)]))

        self.no_elem = no_elem
        self._default_options = DEFAULT_OPTIONS.copy()

    def _prepare_rgb(
        self,
        array: np.ndarray,
        rgb: List[int] = None,
        surface_reflectance: int = 10000,
        cutoff: List = None,
    ) -> np.ndarray:
        """Prepare for RGB plot.

        Parameters
        ----------
        array: [numpy array]
            array.
        rgb: [List]
            Default is [3,2,1]
        surface_reflectance: [int]
            Default is  10000
        cutoff: [List]
            clip the range of pixel values for each band. (take only the pixel values from 0 to value of the cutoff and
            scale them back to between 0 and 1. Default is None.

        Returns
        -------

        """
        # take the rgb arrays and reorder them to have the red-green-blue, if the order is not given assume the
        # order as sentinel data. [3, 2, 1]
        array = array[rgb].transpose(1, 2, 0)
        array = np.clip(array / surface_reflectance, 0, 1)
        if cutoff is not None:
            array[0] = np.clip(rgb[0], 0, cutoff[0]) / cutoff[0]
            array[1] = np.clip(rgb[1], 0, cutoff[1]) / cutoff[1]
            array[2] = np.clip(rgb[2], 0, cutoff[2]) / cutoff[2]

        return array

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
                "please first use the function animate to create the animation object"
            )
        return val

    def get_ticks(self) -> np.ndarray:
        """get list of ticks for the color bar"""
        ticks_spacing = self.default_options["ticks_spacing"]
        vmax = self.default_options["vmax"]
        vmin = self.default_options["vmin"]
        if np.mod(vmax, ticks_spacing) == 0:
            ticks = np.arange(vmin, vmax + ticks_spacing, ticks_spacing)
        else:
            try:
                ticks = np.arange(vmin, vmax, ticks_spacing)
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

    def get_im_cbar(self, ax, arr: np.ndarray, ticks: np.ndarray):
        """

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
        im, cbar
        """
        color_scale = self.default_options["color_scale"]
        cmap = self.default_options["cmap"]
        vmin = self.default_options["vmin"]
        vmax = self.default_options["vmax"]

        if color_scale == 1:
            im = ax.matshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=self.extent)
            cbar_kw = dict(ticks=ticks)
        elif color_scale == 2:
            im = ax.matshow(
                arr,
                cmap=cmap,
                norm=colors.PowerNorm(
                    gamma=self.default_options["gamma"], vmin=vmin, vmax=vmax
                ),
                extent=self.extent,
            )
            cbar_kw = dict(ticks=ticks)
        elif color_scale == 3:
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
            cbar_kw = dict(ticks=ticks, format=formatter)
        elif color_scale == 4:
            if not self.default_options["bounds"]:
                bounds = ticks
                cbar_kw = dict(ticks=ticks)
            else:
                bounds = self.default_options["bounds"]
                cbar_kw = dict(ticks=self.default_options["bounds"])
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(arr, cmap=cmap, norm=norm, extent=self.extent)
        else:
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

        return im, cbar_kw

    @staticmethod
    def _plot_text(ax, arr: np.ndarray, indices, DEFAULT_OPTIONS: dict):
        """
            plot values as a text in each cell

        Parameters
        ----------
        ax:[matplotlib ax]
        indices: [array]
            array with columns, (row, col)

        DEFAULT_OPTIONS

        Returns
        -------
        list of the text object
        """
        # add text for the cell values
        add_text = lambda elem: ax.text(
            elem[1],
            elem[0],
            round(arr[elem[0], elem[1]], 2),
            ha="center",
            va="center",
            color="w",
            fontsize=DEFAULT_OPTIONS["num_size"],
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

    def plot(
        self,
        points: np.ndarray = None,
        point_color: str = "red",
        point_size: Union[int, float] = 100,
        pid_color="blue",
        pid_size: Union[int, float] = 10,
        **kwargs,
    ):
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
            the color of the annotation of the point. Default is blue.
        pid_size: [Any]
            size of the point annotation.
        **kwargs: [dict]
            keys:
                figsize : [tuple], optional
                    figure size. The default is (8,8).
                title: [str], optional
                    title of the plot. The default is 'Total Discharge'.
                title_size: [integer], optional
                    title size. The default is 15.
                orientation: [string], optional
                    orientation of the color bar horizontal/vertical. The default is 'vertical'.
                rotation: [number], optional
                    rotation of the color bar label. The default is -90.
                orientation: [string], optional
                    orientation of the color bar horizontal/vertical. The default is 'vertical'.
                cbar_length: [float], optional
                    ratio to control the height of the color bar. The default is 0.75.
                ticks_spacing: [integer], optional
                    Spacing in the color bar ticks. The default is 2.
                cbar_label_size: integer, optional
                    size of the color bar label. The default is 12.
                cbar_label: str, optional
                    label of the color bar. The default is 'Discharge m3/s'.
                color_scale : integer, optional
                    there are 5 options to change the scale of the colors. The default is 1.
                    1- color_scale 1 is the normal scale
                    2- color_scale 2 is the power scale
                    3- color_scale 3 is the SymLogNorm scale
                    4- color_scale 4 is the PowerNorm scale
                    5- color_scale 5 is the BoundaryNorm scale
                    ------------------------------------------------------------------
                    gamma: [float], optional
                        value needed for option 2. The default is 1./2.
                    line_threshold: [float], optional
                        value needed for option 3. The default is 0.0001.
                    line_scale: [float], optional
                        value needed for option 3. The default is 0.001.
                    bounds: [List]
                        a list of number to be used as a discrete bounds for the color scale 4.Default is None,
                    midpoint: [float], optional
                        value needed for option 5. The default is 0.
                    ------------------------------------------------------------------
                cmap: [str], optional
                    color style. The default is 'coolwarm_r'.
                display_cell_value: [bool]
                    True if you want to display the values of the cells as a text
                num_size: integer, optional
                    size of the numbers plotted on top of each cell. The default is 8.
                background_color_threshold: [float/integer], optional
                    threshold value if the value of the cell is greater, the plotted
                    numbers will be black, and if smaller the plotted number will be white
                    if None given the maxvalue/2 will be considered. The default is None.

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
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
        fig = plt.figure(figsize=self.default_options["figsize"])
        # gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
        ax = fig.add_subplot()  # gs[:,:]

        if self.rgb:
            ax.imshow(arr)
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
            im, cbar_kw = self.get_im_cbar(ax, arr, ticks)

            # Create colorbar
            cbar = ax.figure.colorbar(
                im,
                ax=ax,
                shrink=self.default_options["cbar_length"],
                orientation=self.default_options["orientation"],
                **cbar_kw,
            )
            cbar.ax.set_ylabel(
                self.default_options["cbar_label"],
                rotation=self.default_options["rotation"],
                va="bottom",
                fontsize=self.default_options["cbar_label_size"],
            )
            cbar.ax.tick_params(labelsize=10)

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
            optional_display["poits_id"] = self._plot_point_values(
                ax, points, pid_color, pid_size
            )

        # # Normalize the threshold to the images color range.
        # if self.default_options["background_color_threshold"] is not None:
        #     im.norm(self.default_options["background_color_threshold"])
        # else:
        #     im.norm(self.vmax) / 2.0

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
        """AnimateArray.

        plot an animation for 3d arrays

        Parameters
        ----------
        time : [dataframe]
            dataframe contains the date of values.
        points : [array]
            3 column array with the first column as the value you want to display for the point, the second is the rows
            index of the point in the array, and the third column as the column index in the array.
            - the second and third column tells the location of the point in the array.
        point_color: [str]
            color of the points. The default is 'red'.
        point_size: [Any]
            size of the point. The default is 100.
        pid_color: [str]
            the color of the anotation of the point. Default is blue.
        pid_size: [Any]
            size of the point annotation. The default is 10.
        text_colors : TYPE, optional
            Two colors to be used to plot the values i top of each cell. The default is ("white","black").
        interval : [integer], optional
            number to controlthe speed of the animation. The default is 200.
        text_loc : [list], optional
            location of the date text. The default is [0.1,0.2].
        **kwargs : [dict]
            figsize : [tuple], optional
                figure size. The default is (8,8).
            title : [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size : [integer], optional
                title size. The default is 15.
            orientation : [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            rotation : [number], optional
                rotation of the colorbar label. The default is -90.
            orientation : [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            cbar_length : [float], optional
                ratio to control the height of the colorbar. The default is 0.75.
            ticks_spacing : [integer], optional
                Spacing in the colorbar ticks. The default is 2.
            cbar_label_size : integer, optional
                size of the color bar label. The default is 12.
            cbar_label : str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            color_scale : integer, optional
                there are 5 options to change the scale of the colors. The default is 1.
                1- color_scale 1 is the normal scale
                2- color_scale 2 is the power scale
                3- color_scale 3 is the SymLogNorm scale
                4- color_scale 4 is the PowerNorm scale
                5- color_scale 5 is the BoundaryNorm scale
                ------------------------------------------------------------------
                gamma : [float], optional
                    value needed for option 2 . The default is 1./2..
                line_threshold : [float], optional
                    value needed for option 3. The default is 0.0001.
                line_scale : [float], optional
                    value needed for option 3. The default is 0.001.
                bounds: [List]
                    a list of number to be used as a discrete bounds for the color scale 4.Default is None,
                midpoint : [float], optional
                    value needed for option 5. The default is 0.
                ------------------------------------------------------------------
            cmap : [str], optional
                color style. The default is 'coolwarm_r'.
            display_cell_value : [bool]
                True if you want to display the values of the cells as a text
            num_size : integer, optional
                size of the numbers plotted intop of each cells. The default is 8.
            background_color_threshold : [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.

        Returns
        -------
        animation.FuncAnimation.
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
        precission = self.default_options["precission"]
        array = self.arr
        fig = plt.figure(60, figsize=self.default_options["figsize"])

        ax = fig.add_subplot()
        ticks = self.get_ticks()
        im, cbar_kw = self.get_im_cbar(ax, array[0, :, :], ticks)

        # Create colorbar
        cbar = ax.figure.colorbar(
            im,
            ax=ax,
            shrink=self.default_options["cbar_length"],
            orientation=self.default_options["orientation"],
            **cbar_kw,
        )
        cbar.ax.set_ylabel(
            self.default_options["cbar_label"],
            rotation=self.default_options["rotation"],
            va="bottom",
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
            poits_id = self._plot_point_values(ax, points, pid_color, pid_size)

        # Normalize the threshold to the images color range.
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
                update_points = lambda x: poits_id[x].set_text(points[x, 0])
                list(map(update_points, range(len(col))))

                output = output + poits_id

            if self.default_options["display_cell_value"]:
                vals = array[0, indices[:, 0], indices[:, 1]]
                update_cell_value = lambda x: cell_text_value[x].set_text(vals[x])
                list(map(update_cell_value, range(self.no_elem)))
                output = output + cell_text_value

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
                    poits_id[x].set_text(points[x, 0])

                output = output + poits_id

            if self.default_options["display_cell_value"]:
                vals = array[i, indices[:, 0], indices[:, 1]]

                def update_cell_value(x):
                    val = round(vals[x], precission)
                    kw = dict(
                        color=text_colors[
                            int(im.norm(vals[x]) > background_color_threshold)
                        ]
                    )
                    cell_text_value[x].update(kw)
                    cell_text_value[x].set_text(val)

                list(map(update_cell_value, range(self.no_elem)))

                output = output + cell_text_value

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
        return anim

    def save_animation(self, path: str, fps: int = 2):
        """Save gif file.

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
            writergif = animation.PillowWriter(fps=fps)
            self.anim.save(path, writer=writergif)
        else:
            try:
                if video_format == "avi" or video_format == "mov":
                    writervideo = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    self.anim.save(path, writer=writervideo)
                elif video_format == "mp4":
                    writermp4 = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    self.anim.save(path, writer=writermp4)
            except FileNotFoundError:
                print(
                    "please visit https://ffmpeg.org/ and download a version of ffmpeg compitable with your operating system, for more details please check the method definition"
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
