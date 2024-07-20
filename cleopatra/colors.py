from typing import List, Union, Tuple, Any
from matplotlib import colors as mcolors


class Colors:
    """Colors class for Cleopatra."""

    def __init__(
        self,
        color_value: Union[
            List[str], str, Tuple[float, float, float], List[Tuple[float, float, float]]
        ],
    ):
        """

        Parameters
        ----------
        color_value: List[str]/Tuple[float, float, float]/str.
            the color value could be a list of hex colors, a tuple of RGB values, or a single hex/RGB color.

        Examples
        --------
        - Create a color object from a hex color:

            >>> hex_number = "ff0000"
            >>> color = Colors(hex_number)
            >>> print(color.color_value)
            ['ff0000']

        - Create a color object from an RGB color (values are between 0 and 1):

            >>> rgb_color = (0.5, 0.2, 0.8)
            >>> color = Colors(rgb_color)
            >>> print(color.color_value)
            [(0.5, 0.2, 0.8)]

        - Create a color object from an RGB color (values are between 0 and 255):

            >>> rgb_color = (128, 51, 204)
            >>> color = Colors(rgb_color)
            >>> print(color.color_value)
            [(128, 51, 204)]
        """
        # convert the hex color to a list if it is a string
        if isinstance(color_value, str) or isinstance(color_value, tuple):
            color_value = [color_value]
        elif not isinstance(color_value, list):
            raise ValueError(
                "The color_value must be a list of hex colors, list of tuples (RGB color), a single hex "
                "or single RGB tuple color."
            )

        self._color_value = color_value

    def get_type(self) -> List[str]:
        """get_type.

        Returns
        -------
        List[str]

        Examples
        --------
        - Create a color object from a hex color:

            >>> hex_number = "#23a9dd"
            >>> color = Colors(hex_number)
            >>> print(color.get_type())
            ['hex']

        - Create a color object from an RGB color (values are between 0 and 1):

            >>> rgb_color = (0.5, 0.2, 0.8)
            >>> color = Colors(rgb_color)
            >>> print(color.get_type())
            ['rgb']

        - Create a color object from an RGB color (values are between 0 and 255):

            >>> rgb_color = (128, 51, 204)
            >>> color = Colors(rgb_color)
            >>> print(color.get_type())
            ['rgb']
        """
        color_type = []
        for color_i in self.color_value:
            if self.is_valid_rgb_i(color_i):
                color_type.append("rgb")
            elif self.is_valid_hex_i(color_i):
                color_type.append("hex")

        return color_type

    @property
    def color_value(self) -> Union[List[str], Tuple[float, float, float]]:
        """Color values given by the user.

        Returns
        -------
        List[str]
        """
        return self._color_value

    @property
    def hex_color(self) -> List[str]:
        """hex_color.

        Parameters
        ----------

        Returns
        -------
        List[str]
        """
        return self._hex_color

    def is_valid_hex(self) -> List[bool]:
        """is_valid_hex.

            is_valid_hex

        Parameters
        ----------

        Returns
        -------

        """
        return [self.is_valid_hex_i(col) for col in self.color_value]

    @staticmethod
    def is_valid_hex_i(hex_color: str) -> bool:
        """is_valid_hex for single color.


        Parameters
        ----------
        hex_color: str.
            single hex color.
        Returns
        -------
        bool
        """
        return True if mcolors.is_color_like(hex_color) else False

    def is_valid_rgb(self) -> List[bool]:
        """is_valid_rgb.

            is_valid_hex

        Parameters
        ----------

        Returns
        -------
        List[bool]
            List of boolean values for each color
        """
        return [self.is_valid_rgb_i(col) for col in self.color_value]

    @staticmethod
    def is_valid_rgb_i(rgb_tuple: Any) -> bool:
        """validate a single color whither it is rgb or not."""
        if isinstance(rgb_tuple, tuple) and len(rgb_tuple) == 3:
            if all(isinstance(value, int) for value in rgb_tuple):
                return all(0 <= value <= 255 for value in rgb_tuple)
            elif all(isinstance(value, float) for value in rgb_tuple):
                return all(0.0 <= value <= 1.0 for value in rgb_tuple)
        return False

    def to_rgb(
        self, normalized: bool = True
    ) -> List[Tuple[Union[int, float], Union[int, float]]]:
        """get_rgb.

        Parameters
        ----------
        normalized: int, Default is True.
            True if you want the RGB values to be scaled between 0 and 1. False if you want the RGB values to be scaled
            between 0 and 255.

        Returns
        -------
        List[Tuples]
        """
        if normalized == 1:
            rgb = [mcolors.to_rgb(col) for col in self.color_value]
        else:
            rgb = [
                tuple([int(c * 255) for c in mcolors.to_rgb(col)])
                for col in self.color_value
            ]
        return rgb
