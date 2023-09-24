from typing import List, Union, Tuple
from matplotlib import colors as mcolors


class Colors:
    """Colors class for Cleopatra."""

    def __init__(self, hex_color: Union[List[str], str]):
        # convert the hex color to a list if it is a string
        if isinstance(hex_color, str):
            hex_color = [hex_color]

        self.hex_color = hex_color

    def is_valid_hex(self) -> List[bool]:
        """is_valid_hex.

            is_valid_hex

        Parameters
        ----------

        Returns
        -------

        """
        return [True if mcolors.is_color_like(col) else False for col in self.hex_color]

    def get_rgb(self, normalized: bool = True) -> List[Tuple[int, int]]:
        """get_rgb.

        Parameters
        ----------
        normalized: [int]
            True if you want the RGB values to be scaled between 0 and 1,  .Default is True.

        Returns
        -------
        List[Tuples
        """
        if normalized == 1:
            rgb = [mcolors.to_rgb(col) for col in self.hex_color]
        else:
            rgb = [
                tuple([int(c * 255) for c in mcolors.to_rgb(col)])
                for col in self.hex_color
            ]
        return rgb
