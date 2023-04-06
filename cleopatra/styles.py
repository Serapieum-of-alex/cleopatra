"""style related functionality"""
import matplotlib.colors as colors
import numpy as np


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
