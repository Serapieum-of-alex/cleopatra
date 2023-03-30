"""plotting Array."""
from collections import OrderedDict
from typing import Any, List, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from numpy_utils.filter import get_indices2
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import LogFormatter


class Array:
    """Map."""

    figure_default_options = dict(
        ylabel="",
        xlabel="",
        legend="",
        legend_size=10,
        figsize=(10, 8),
        labelsize=10,
        fontsize=10,
        name="hist.tif",
        color1="#3D59AB",
        color2="#DC143C",
        linewidth=3,
        Axisfontsize=15,
    )

    line_styles = OrderedDict(
        [
            ("solid", (0, ())),  # 0
            ("loosely dotted", (0, (1, 10))),  # 1
            ("dotted", (0, (1, 5))),  # 2
            ("densely dotted", (0, (1, 1))),  # 3
            ("loosely dashed", (0, (5, 10))),  # 4
            ("dashed", (0, (5, 5))),  # 5
            ("densely dashed", (0, (5, 1))),  # 6
            ("loosely dashdotted", (0, (3, 10, 1, 10))),  # 7
            ("dashdotted", (0, (3, 5, 1, 5))),  # 8
            ("densely dashdotted", (0, (3, 1, 1, 1))),  # 9
            ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),  # 10
            ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),  # 11
            ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),  # 12
            ("densely dashdotdottededited", (0, (6, 1, 1, 1, 1, 1))),  # 13
        ]
    )

    marker_style_list = [
        "--o",
        ":D",
        "-.H",
        "--x",
        ":v",
        "--|",
        "-+",
        "-^",
        "--s",
        "-.*",
        "-.h",
    ]

    def __init__(self, array: np.ndarray, exculde_value: Union[int, float] = np.nan):
        """Plot array.

        Parameters
        ----------
        array: [numpy array]
            array.
        exculde_value : [numeric]
            value used to fill cells out of the domain. Optional, Default is np.nan
            needed only in case of plotting array

        the object does not need any parameters to be initialized.
        """
        # first replace the no_data_value by nan
        if exculde_value is not None:
            array[np.isclose(array, exculde_value, rtol=0.0000001)] = np.nan
        self.vmin = np.nanmin(array)
        self.vmax = np.nanmax(array)
        self.arr = array
        self.no_elem = np.size(array[:, :]) - np.count_nonzero((array[np.isnan(array)]))

    @staticmethod
    def get_line_style(style: Union[str, int] = "loosely dotted"):
        """LineStyle.

        Line styles for plotting

        Parameters
        ----------
        style : TYPE, optional
            DESCRIPTION. The default is 'loosely dotted'.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if isinstance(style, str):
            try:
                return Array.line_styles[style]
            except KeyError:
                msg = (
                    f" The style name you entered-{style}-does not exist please"
                    "choose from the available styles"
                )
                print(msg)
                print(list(Array.line_styles))
        else:
            return list(Array.line_styles.items())[style][1]

    @staticmethod
    def get_marker_style(style: int):
        """Marker styles for plotting.

        Parameters
        ----------
        style: [int]
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if style > len(Array.marker_style_list) - 1:
            style = style % len(Array.marker_style_list)
        return Array.marker_style_list[style]

    def plot(
        self,
        figsize: Tuple[int, int] = (8, 8),
        title: Any = "Total Discharge",
        title_size: Union[int, float] = 15,
        cbar_length: Union[int, float] = 0.75,
        orientation: str = "vertical",
        cbar_label_size: Union[int, float] = 12,
        cbar_label: str = "Color bar label",
        rotation: Union[int, float] = -90,
        ticks_spacing: Union[int, float] = 5,
        num_size: Union[int, float] = 8,
        color_scale: int = 1,
        cmap: str = "coolwarm_r",
        gamma: Union[int, float] = 0.5,
        linscale: Union[int, float] = 0.001,
        linthresh: Union[int, float] = 0.0001,
        bounds: List = None,
        midpoint: int = 0,
        display_cellvalue: bool = False,
        background_color_threshold=None,
        point_color: str = "red",
        point_size: Union[int, float] = 100,
        pid_color="blue",
        pid_size: Union[int, float] = 10,
        **kwargs,
    ):
        """plot an array.

        Parameters
        ----------
        figsize : [tuple], optional
            figure size. The default is (8,8).
        title : [str], optional
            title of the plot. The default is 'Total Discharge'.
        title_size : [integer], optional
            title size. The default is 15.
        cbar_length : [float], optional
            ratio to control the height of the colorbar. The default is 0.75.
        orientation : [string], optional
            orintation of the colorbar horizontal/vertical. The default is 'vertical'.
        cbar_label_size : integer, optional
            size of the color bar label. The default is 12.
        cbar_label : str, optional
            label of the color bar. The default is 'Discharge m3/s'.
        rotation : [number], optional
            rotation of the colorbar label. The default is -90.
        ticks_spacing : [integer], optional
            Spacing in the colorbar ticks. The default is 2.
        color_scale : integer, optional
            there are 5 options to change the scale of the colors. The default is 1.
            1- color_scale 1 is the normal scale
            2- color_scale 2 is the power scale
            3- color_scale 3 is the SymLogNorm scale
            4- color_scale 4 is the PowerNorm scale
            5- color_scale 5 is the BoundaryNorm scale
        gamma : [float], optional
            value needed for option 2 . The default is 1./2..
        linthresh : [float], optional
            value needed for option 3. The default is 0.0001.
        linscale : [float], optional
            value needed for option 3. The default is 0.001.
        bounds: [List]
            a list of number to be used as a discrete bounds for the color scale 4.Default is None,
        midpoint : [float], optional
            value needed for option 5. The default is 0.
        cmap : [str], optional
            color style. The default is 'coolwarm_r'.
        display_cellvalue : [bool]
            True if you want to display the values of the cells as a text
        num_size : integer, optional
            size of the numbers plotted intop of each cells. The default is 8.
        background_color_threshold : [float/integer], optional
            threshold value if the value of the cell is greater, the plotted
            numbers will be black and if smaller the plotted number will be white
            if None given the maxvalue/2 will be considered. The default is None.

        rotation : []

        midpoint : []

        **kwargs : [dict]
            keys:
                Points : [dataframe].
                    dataframe contains two columns 'row', and col to
                    plot the point at this location

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
        """
        arr = self.arr
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        # creating the ticks/bounds
        if np.mod(self.vmax, ticks_spacing) == 0:
            ticks = np.arange(self.vmin, self.vmax + ticks_spacing, ticks_spacing)
        else:
            try:
                ticks = np.arange(self.vmin, self.vmax, ticks_spacing)
            except ValueError:
                raise ValueError(
                    "The number of ticks exceeded the max allowed size, possible errors"
                    f"is the value of the NodataValue you entered-{self.exculde_value}"
                )
            ticks = np.append(
                ticks,
                [int(self.vmax / ticks_spacing) * ticks_spacing + ticks_spacing],
            )

        if color_scale == 1:
            im = ax.matshow(arr[:, :], cmap=cmap, vmin=self.vmin, vmax=self.vmax)
            cbar_kw = dict(ticks=ticks)
        elif color_scale == 2:
            im = ax.matshow(
                arr[:, :],
                cmap=cmap,
                norm=colors.PowerNorm(gamma=gamma, vmin=self.vmin, vmax=self.vmax),
            )
            cbar_kw = dict(ticks=ticks)
        elif color_scale == 3:
            im = ax.matshow(
                arr[:, :],
                cmap=cmap,
                norm=colors.SymLogNorm(
                    linthresh=linthresh,
                    linscale=linscale,
                    base=np.e,
                    vmin=self.vmin,
                    vmax=self.vmax,
                ),
            )

            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = dict(ticks=ticks, format=formatter)
        elif color_scale == 4:
            if not bounds:
                bounds = ticks
                cbar_kw = dict(ticks=ticks)
            else:
                cbar_kw = dict(ticks=bounds)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(arr[:, :], cmap=cmap, norm=norm)

        else:
            im = ax.matshow(
                arr[:, :],
                cmap=cmap,
                norm=MidpointNormalize(
                    midpoint=midpoint, vmin=self.vmin, vmax=self.vmax
                ),
            )
            cbar_kw = dict(ticks=ticks)

        # Create colorbar
        cbar = ax.figure.colorbar(
            im, ax=ax, shrink=cbar_length, orientation=orientation, **cbar_kw
        )
        cbar.ax.set_ylabel(
            cbar_label, rotation=rotation, va="bottom", fontsize=cbar_label_size
        )
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(title, fontsize=title_size)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])

        if display_cellvalue:
            indices = get_indices2(arr, [np.nan])
            # add text for the cell values
            add_text = lambda elem: ax.text(
                elem[1],
                elem[0],
                round(arr[elem[0], elem[1]], 2),
                ha="center",
                va="center",
                color="w",
                fontsize=num_size,
            )
            list(map(add_text, indices))

        # Normalize the threshold to the images color range.
        if background_color_threshold is not None:
            im.norm(background_color_threshold)
        else:
            im.norm(self.vmax) / 2.0

        if "points" in kwargs.keys():
            points = kwargs["points"]
            row = points[:, 1]
            col = points[:, 2]
            # IDs = points[:, 0]
            ax.scatter(col, row, color=point_color, s=point_size)
            # TODO: Points = ax.scatter(col, rows, color=point_color, s=point_size)
            #  return the scatter plot object (Points)
            write_points = lambda x: ax.text(
                x[2],
                x[1],
                x[0],
                ha="center",
                va="center",
                color=pid_color,
                fontsize=pid_size,
            )
            list(map(write_points, points))

        return fig, ax

    @staticmethod
    def animate(
        array,
        time,
        n_elem,
        ticks_spacing=2,
        figsize=(8, 8),
        plot_numbers=True,
        num_size=8,
        title="Total Discharge",
        title_size=15,
        background_color_threshold=None,
        cbar_label="Discharge m3/s",
        cbar_label_size=12,
        text_colors=("white", "black"),
        cbar_length=0.75,
        interval=200,
        cmap="coolwarm_r",
        text_loc=[0.1, 0.2],
        points_color="red",
        points_size=100,
        color_scale=1,
        gamma=0.5,
        line_threshold=0.0001,
        line_scale=0.001,
        midpoint=0,
        orientation="vertical",
        rotation=-90,
        pid_color="blue",
        pid_size=10,
        **kwargs,
    ):
        """AnimateArray.

        plot an animation for 3d arrays

        Parameters
        ----------
        array : [array]
            the array you want to animate.
        time : [dataframe]
            dataframe contains the date of values.
        n_elem : [integer]
            Number of the cells that has values.
        ticks_spacing : [integer], optional
            Spacing in the colorbar ticks. The default is 2.
        figsize : [tuple], optional
            figure size. The default is (8,8).
        plot_numbers : [bool], optional
            True to plot the values intop of each cell. The default is True.
        num_size : integer, optional
            size of the numbers plotted intop of each cells. The default is 8.
        title : [str], optional
            title of the plot. The default is 'Total Discharge'.
        title_size : [integer], optional
            title size. The default is 15.
        background_color_threshold : [float/integer], optional
            threshold value if the value of the cell is greater, the plotted
            numbers will be black and if smaller the plotted number will be white
            if None given the maxvalue/2 will be considered. The default is None.
        text_colors : TYPE, optional
            Two colors to be used to plot the values i top of each cell. The default is ("white","black").
        cbar_label : str, optional
            label of the color bar. The default is 'Discharge m3/s'.
        cbar_label_size : integer, optional
            size of the color bar label. The default is 12.
        cbar_length : [float], optional
            ratio to control the height of the colorbar. The default is 0.75.
        interval : [integer], optional
            number to controlthe speed of the animation. The default is 200.
        cmap : [str], optional
            color style. The default is 'coolwarm_r'.
        text_loc : [list], optional
            location of the date text. The default is [0.1,0.2].
        points_color : [str], optional
            color of the points. The default is 'red'.
        points_size : [integer], optional
            size of the points. The default is 100.
        pid_color : [str]
            the ID of the Point.The default is "blue".
        pid_size : [integer]
            size of the ID text. The default is 10.
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
            midpoint : [float], optional
                value needed for option 5. The default is 0.
            ------------------------------------------------------------------
        orientation : [string], optional
            orintation of the colorbar horizontal/vertical. The default is 'vertical'.
        rotation : [number], optional
            rotation of the colorbar label. The default is -90.
        **kwargs : [dict]
            keys:
                Points : [dataframe].
                    dataframe contains two columns 'cell_row', and cell_col to
                    plot the point at this location

        Returns
        -------
        animation.FuncAnimation.
        """
        fig = plt.figure(60, figsize=figsize)
        gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
        ax = fig.add_subplot(gs[:, :])
        ticks = np.arange(np.nanmin(array), np.nanmax(array), ticks_spacing)

        if color_scale == 1:
            im = ax.matshow(
                array[:, :, 0], cmap=cmap, vmin=np.nanmin(array), vmax=np.nanmax(array)
            )
            cbar_kw = dict(ticks=ticks)
        elif color_scale == 2:
            im = ax.matshow(
                array[:, :, 0],
                cmap=cmap,
                norm=colors.PowerNorm(
                    gamma=gamma, vmin=np.nanmin(array), vmax=np.nanmax(array)
                ),
            )
            cbar_kw = dict(ticks=ticks)
        elif color_scale == 3:
            im = ax.matshow(
                array[:, :, 0],
                cmap=cmap,
                norm=colors.SymLogNorm(
                    linthresh=line_threshold,
                    linscale=line_scale,
                    base=np.e,
                    vmin=np.nanmin(array),
                    vmax=np.nanmax(array),
                ),
            )
            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = dict(ticks=ticks, format=formatter)
        elif color_scale == 4:
            bounds = np.arange(np.nanmin(array), np.nanmax(array), ticks_spacing)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(array[:, :, 0], cmap=cmap, norm=norm)
            cbar_kw = dict(ticks=ticks)
        else:
            im = ax.matshow(
                array[:, :, 0], cmap=cmap, norm=MidpointNormalize(midpoint=midpoint)
            )
            cbar_kw = dict(ticks=ticks)

        # Create colorbar
        cbar = ax.figure.colorbar(
            im, ax=ax, shrink=cbar_length, orientation=orientation, **cbar_kw
        )
        cbar.ax.set_ylabel(cbar_label, rotation=rotation, va="bottom")
        cbar.ax.tick_params(labelsize=10)

        day_text = ax.text(text_loc[0], text_loc[1], " ", fontsize=cbar_label_size)
        ax.set_title(title, fontsize=title_size)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])
        Indexlist = list()

        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                if not np.isnan(array[x, y, 0]):
                    Indexlist.append([x, y])

        Textlist = list()
        for x in range(n_elem):
            Textlist.append(
                ax.text(
                    Indexlist[x][1],
                    Indexlist[x][0],
                    round(array[Indexlist[x][0], Indexlist[x][1], 0], 2),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=num_size,
                )
            )
        # Points = list()
        PoitsID = list()
        if "Points" in kwargs.keys():
            row = kwargs["Points"].loc[:, "cell_row"].tolist()
            col = kwargs["Points"].loc[:, "cell_col"].tolist()
            IDs = kwargs["Points"].loc[:, "id"].tolist()
            Points = ax.scatter(col, row, color=points_color, s=points_size)

            for i in range(len(row)):
                PoitsID.append(
                    ax.text(
                        col[i],
                        row[i],
                        IDs[i],
                        ha="center",
                        va="center",
                        color=pid_color,
                        fontsize=pid_size,
                    )
                )

        # Normalize the threshold to the images color range.
        if background_color_threshold is not None:
            background_color_threshold = im.norm(background_color_threshold)
        else:
            background_color_threshold = im.norm(np.nanmax(array)) / 2.0

        def init():
            im.set_data(array[:, :, 0])
            day_text.set_text("")

            output = [im, day_text]

            if "Points" in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs["Points"].loc[:, "cell_row"].tolist()
                col = kwargs["Points"].loc[:, "cell_col"].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)

                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])

                output = output + PoitsID

            if plot_numbers:
                for x in range(n_elem):
                    val = round(array[Indexlist[x][0], Indexlist[x][1], 0], 2)
                    Textlist[x].set_text(val)

                output = output + Textlist

            return output

        def animate_a(i):
            im.set_data(array[:, :, i])
            day_text.set_text("Date = " + str(time[i])[0:10])

            output = [im, day_text]

            if "Points" in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs["Points"].loc[:, "cell_row"].tolist()
                col = kwargs["Points"].loc[:, "cell_col"].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)

                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])

                output = output + PoitsID

            if plot_numbers:
                for x in range(n_elem):
                    val = round(array[Indexlist[x][0], Indexlist[x][1], i], 2)
                    kw = dict(
                        color=text_colors[
                            int(
                                im.norm(array[Indexlist[x][0], Indexlist[x][1], i])
                                > background_color_threshold
                            )
                        ]
                    )
                    Textlist[x].update(kw)
                    Textlist[x].set_text(val)

                output = output + Textlist

            return output

        plt.tight_layout()
        # global anim
        anim = FuncAnimation(
            fig,
            animate_a,
            init_func=init,
            frames=np.shape(array)[2],
            interval=interval,
            blit=True,
        )

        return anim

    @staticmethod
    def plot_type_1(
        Y1,
        Y2,
        Points,
        PointsY,
        PointMaxSize=200,
        PointMinSize=1,
        X_axis_label="X Axis",
        LegendNum=5,
        LegendLoc=(1.3, 1),
        PointLegendTitle="Output 2",
        Ylim=[0, 180],
        Y2lim=[-2, 14],
        color1="#27408B",
        color2="#DC143C",
        color3="grey",
        linewidth=4,
        **kwargs,
    ):
        """Plot_Type1.

        !TODO Needs docs

        Parameters
        ----------
        Y1 : TYPE
            DESCRIPTION.
        Y2 : TYPE
            DESCRIPTION.
        Points : TYPE
            DESCRIPTION.
        PointsY : TYPE
            DESCRIPTION.
        PointMaxSize : TYPE, optional
            DESCRIPTION. The default is 200.
        PointMinSize : TYPE, optional
            DESCRIPTION. The default is 1.
        X_axis_label : TYPE, optional
            DESCRIPTION. The default is 'X Axis'.
        LegendNum : TYPE, optional
            DESCRIPTION. The default is 5.
        LegendLoc : TYPE, optional
            DESCRIPTION. The default is (1.3, 1).
        PointLegendTitle : TYPE, optional
            DESCRIPTION. The default is "Output 2".
        Ylim : TYPE, optional
            DESCRIPTION. The default is [0,180].
        Y2lim : TYPE, optional
            DESCRIPTION. The default is [-2,14].
        color1 : TYPE, optional
            DESCRIPTION. The default is '#27408B'.
        color2 : TYPE, optional
            DESCRIPTION. The default is '#DC143C'.
        color3 : TYPE, optional
            DESCRIPTION. The default is "grey".
        linewidth : TYPE, optional
            DESCRIPTION. The default is 4.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        ax1 : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        fig : TYPE
            DESCRIPTION.
        """
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        ax2 = ax1.twinx()

        ax1.plot(
            Y1[:, 0],
            Y1[:, 1],
            zorder=1,
            color=color1,
            linestyle=Array.get_line_style(0),
            linewidth=linewidth,
            label="Model 1 Output1",
        )

        if "Y1_2" in kwargs.keys():
            Y1_2 = kwargs["Y1_2"]

            rows_axis1, cols_axis1 = np.shape(Y1_2)

            if "Y1_2_label" in kwargs.keys():
                label = kwargs["Y2_2_label"]
            else:
                label = ["label"] * (cols_axis1 - 1)
            # first column is the x axis
            for i in range(1, cols_axis1):
                ax1.plot(
                    Y1_2[:, 0],
                    Y1_2[:, i],
                    zorder=1,
                    color=color2,
                    linestyle=Array.get_line_style(i),
                    linewidth=linewidth,
                    label=label[i - 1],
                )

        ax2.plot(
            Y2[:, 0],
            Y2[:, 1],
            zorder=1,
            color=color3,
            linestyle=Array.get_line_style(6),
            linewidth=2,
            label="Output1-Diff",
        )

        if "Y2_2" in kwargs.keys():
            Y2_2 = kwargs["Y2_2"]
            rows_axis2, cols_axis2 = np.shape(Y2_2)

            if "Y2_2_label" in kwargs.keys():
                label = kwargs["Y2_2_label"]
            else:
                label = ["label"] * (cols_axis2 - 1)

            for i in range(1, cols_axis2):
                ax1.plot(
                    Y2_2[:, 0],
                    Y2_2[:, i],
                    zorder=1,
                    color=color2,
                    linestyle=Array.get_line_style(i),
                    linewidth=linewidth,
                    label=label[i - 1],
                )

        if "Points1" in kwargs.keys():
            # first axis in the x axis
            Points1 = kwargs["Points1"]

            vmax = np.max(Points1[:, 1:])
            vmin = np.min(Points1[:, 1:])

            vmax = max(Points[:, 1].max(), vmax)
            vmin = min(Points[:, 1].min(), vmin)

        else:
            vmax = max(Points)
            vmin = min(Points)

        vmaxnew = PointMaxSize
        vminnew = PointMinSize

        Points_scaled = [
            Scale.rescale(x, vmin, vmax, vminnew, vmaxnew) for x in Points[:, 1]
        ]
        f1 = np.ones(shape=(len(Points))) * PointsY
        scatter = ax2.scatter(
            Points[:, 0],
            f1,
            zorder=1,
            c=color1,
            s=Points_scaled,
            label="Model 1 Output 2",
        )

        if "Points1" in kwargs.keys():
            row_points, col_points = np.shape(Points1)
            PointsY1 = kwargs["PointsY1"]
            f2 = np.ones_like(Points1[:, 1:])

            for i in range(col_points - 1):
                Points1_scaled = [
                    Scale.rescale(x, vmin, vmax, vminnew, vmaxnew)
                    for x in Points1[:, i]
                ]
                f2[:, i] = PointsY1[i]

                ax2.scatter(
                    Points1[:, 0],
                    f2[:, i],
                    zorder=1,
                    c=color2,
                    s=Points1_scaled,
                    label="Model 2 Output 2",
                )

        # produce a legend with the unique colors from the scatter
        legend1 = ax2.legend(
            *scatter.legend_elements(), bbox_to_anchor=(1.1, 0.2)
        )  # loc="lower right", title="RIM"

        ax2.add_artist(legend1)

        # produce a legend with a cross section of sizes from the scatter
        handles, labels = scatter.legend_elements(
            prop="sizes", alpha=0.6, num=LegendNum
        )
        # L = [vminnew] + [float(i[14:-2]) for i in labels] + [vmaxnew]
        L = [float(i[14:-2]) for i in labels]
        labels1 = [
            round(Scale.rescale(x, vminnew, vmaxnew, vmin, vmax) / 1000) for x in L
        ]

        legend2 = ax2.legend(
            handles, labels1, bbox_to_anchor=LegendLoc, title=PointLegendTitle
        )
        ax2.add_artist(legend2)

        ax1.set_ylim(Ylim)
        ax2.set_ylim(Y2lim)
        #
        ax1.set_ylabel("Output 1 (m)", fontsize=12)
        ax2.set_ylabel("Output 1 - Diff (m)", fontsize=12)
        ax1.set_xlabel(X_axis_label, fontsize=12)
        ax1.xaxis.set_minor_locator(plt.MaxNLocator(10))
        ax1.tick_params(which="minor", length=5)
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(1.3, 0.3),
            bbox_transform=ax1.transAxes,
            fontsize=10,
        )
        plt.rcParams.update({"ytick.major.size": 3.5})
        plt.rcParams.update({"font.size": 12})
        plt.title("Model Output Comparison", fontsize=15)

        plt.subplots_adjust(right=0.7)
        # plt.tight_layout()

        return (ax1, ax2), fig


class Scale:
    """different scale object."""

    def __init__(self):
        """Different scale object."""
        pass

    @staticmethod
    def log_scale(minval, maxval):
        """log_scale.

            logarithmic scale

        Parameters
        ----------
        minval
        maxval

        Returns
        -------
        """

        def scalar(val):
            """scalar.

                scalar

            Parameters
            ----------
            val

            Returns
            -------
            """
            val = val + abs(minval) + 1
            return np.log10(val)

        return scalar

    @staticmethod
    def power_scale(minval, maxval):
        """power_scale.

            power scale

        Parameters
        ----------
        minval
        maxval

        Returns
        -------
        """

        def scalar(val):
            val = val + abs(minval) + 1
            return (val / 1000) ** 2

        return scalar

    @staticmethod
    def identity_scale(minval, maxval):
        """identity_scale.

            identity_scale

        Parameters
        ----------
        minval
        maxval

        Returns
        -------
        """

        def scalar(val):
            return 2

        return scalar

    @staticmethod
    def rescale(OldValue, OldMin, OldMax, NewMin, NewMax):
        """Rescale.

        Rescale nethod rescales a value between two boundaries to a new value
        bewteen two other boundaries
        inputs:
            1-OldValue:
                [float] value need to transformed
            2-OldMin:
                [float] min old value
            3-OldMax:
                [float] max old value
            4-NewMin:
                [float] min new value
            5-NewMax:
                [float] max new value
        output:
            1-NewValue:
                [float] transformed new value
        """
        OldRange = OldMax - OldMin
        NewRange = NewMax - NewMin
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        return NewValue


class MidpointNormalize(colors.Normalize):
    """MidpointNormalize.

    !TODO needs docs
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """MidpointNormalize.

        Parameters
        ----------
        vmin
        vmax
        midpoint
        clip
        """
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """MidpointNormalize.

        ! TODO needs docs

        Parameters
        ----------
        value : TYPE
            DESCRIPTION.
        clip : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))
