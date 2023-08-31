"""style related functionality"""
from collections import OrderedDict
from typing import Union
import matplotlib.colors as colors
import numpy as np

DEFAULT_OPTIONS = dict(
    figsize=(8, 8),
    title=None,
    title_size=15,
    ylabel="",
    ylabel_font_size=11,
    xlabel="",
    xlabel_font_size=11,
    xtick_font_size=11,
    ytick_font_size=11,
    legend="",
    legend_size=10,
    color_1="#3D59AB",
    color_2="#DC143C",
    line_width=3,
    cbar_length=0.75,
    orientation="vertical",
    cmap="coolwarm_r",
    cbar_label_size=12,
    cbar_label=None,
    rotation=-90,
    ticks_spacing=5,
    color_scale=1,
    gamma=0.5,
    line_scale=0.001,
    line_threshold=0.0001,
    bounds=None,
    midpoint=0,
    grid_alpha=0.75,
)


class Styles:
    """Styles"""

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
                return Styles.line_styles[style]
            except KeyError:
                msg = (
                    f" The style name you entered-{style}-does not exist please"
                    "choose from the available styles"
                )
                print(msg)
                print(list(Styles.line_styles))
        else:
            return list(Styles.line_styles.items())[style][1]

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
        if style > len(Styles.marker_style_list) - 1:
            style = style % len(Styles.marker_style_list)
        return Styles.marker_style_list[style]


class Scale:
    """different scale object."""

    def __init__(self):
        """Different scale object."""
        pass

    @staticmethod
    def log_scale(val):
        """log_scale.

            logarithmic scale

        Parameters
        ----------
        val

        Returns
        -------
        """

        # def scalar(val):
        #     """scalar.
        #
        #         scalar
        #
        #     Parameters
        #     ----------
        #     val
        #
        #     Returns
        #     -------
        #     """
        #   val = val + abs(minval) + 1
        # return scalar
        return np.log10(val)

    @staticmethod
    def power_scale(minval):
        """power_scale.

            power scale

        Parameters
        ----------
        minval

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
