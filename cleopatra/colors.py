from typing import List, Union, Tuple, Any
from matplotlib import colors as mcolors
from pathlib import Path
from PIL import Image


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

    @classmethod
    def create_from_image(cls, path: str) -> "Colors":
        """Create a color object from an image.

        if you have an image of a color ramp, and you want to extract the colors from it, you can use this method.

        .. image:: /_images/colors/color-ramp.png
            :alt: Example Image
            :align: center

        Parameters
        ----------
        path : str
            The path to the image file.

        Returns
        -------
        Colors
            A color object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> path = "examples/data/colors/color-ramp.png"
        >>> colors = Colors.create_from_image(path)
        >>> print(colors.color_value) # doctest: +SKIP
        [(9, 63, 8), (8, 68, 9), (5, 78, 7), (1, 82, 3), (0, 84, 0), (0, 85, 0), (1, 83, 0), (1, 81, 0), (1, 80, 1)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"The file {path} does not exist.")

        image = Image.open(path).convert("RGB")
        width, height = image.size
        color_values = [image.getpixel((x, int(height / 2))) for x in range(width)]

        return cls(color_values)

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
            ['rgb-normalized']

        - Create a color object from an RGB color (values are between 0 and 255):

            >>> rgb_color = (128, 51, 204)
            >>> color = Colors(rgb_color)
            >>> print(color.get_type())
            ['rgb']
        """
        color_type = []
        for color_i in self.color_value:
            if self.is_valid_rgb_norm(color_i):
                color_type.append("rgb-normalized")
            elif self.is_valid_rgb_255(color_i):
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

    def to_hex(self) -> List[str]:
        """Convert colors to hexdecimal format.

        Returns
        -------
        List[str]
            list of hec colors.

        Examples
        --------
        - Create a color object from a mixed list of hex and RGB colors:

            >>> mixed_color = [(128, 51, 204), "#23a9dd", (0.5, 0.2, 0.8)]
            >>> color = Colors(mixed_color)
            >>> print(color.to_hex())
            ['#8033cc', '#23a9dd', '#8033cc']
        """
        converted_color = []
        color_type = self.get_type()
        for ind, color_i in enumerate(self.color_value):
            if color_type[ind] == "hex":
                converted_color.append(color_i)
            elif color_type[ind] == "rgb":
                # Normalize the RGB values to be between 0 and 1
                rgb_color_normalized = tuple(value / 255 for value in color_i)
                converted_color.append(mcolors.to_hex(rgb_color_normalized))
            else:
                converted_color.append(mcolors.to_hex(color_i))
        return converted_color

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

        Returns
        -------
        List[bool]
            List of boolean values for each color
        """
        return [
            self.is_valid_rgb_norm(col) or self.is_valid_rgb_255(col)
            for col in self.color_value
        ]

    @staticmethod
    def is_valid_rgb_255(rgb_tuple: Any) -> bool:
        """validate a single color whither it is rgb or not."""
        if isinstance(rgb_tuple, tuple) and len(rgb_tuple) == 3:
            if all(isinstance(value, int) for value in rgb_tuple):
                return all(0 <= value <= 255 for value in rgb_tuple)
        return False

    @staticmethod
    def is_valid_rgb_norm(rgb_tuple: Any) -> bool:
        """validate a single color whither it is rgb or not."""
        if isinstance(rgb_tuple, tuple) and len(rgb_tuple) == 3:
            if all(isinstance(value, float) for value in rgb_tuple):
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
        color_type = self.get_type()
        rgb = []
        if normalized:
            for ind, color_i in enumerate(self.color_value):
                # if the color is in RGB format (0-255), normalize the values to be between 0 and 1
                if color_type[ind] == "rgb":
                    rgb_color_normalized = tuple(value / 255 for value in color_i)
                    rgb.append(rgb_color_normalized)
                else:
                    # any other format, just convert it to RGB
                    rgb.append(mcolors.to_rgb(color_i))
        else:
            for ind, color_i in enumerate(self.color_value):
                # if the color is in RGB format (0-255), normalize the values to be between 0 and 1
                if color_type[ind] == "rgb":
                    rgb.append(color_i)
                else:
                    # any other format, just convert it to RGB
                    rgb.append(tuple([int(c * 255) for c in mcolors.to_rgb(color_i)]))

        return rgb
