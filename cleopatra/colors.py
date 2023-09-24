from typing import List, Union
import re

# from PIL import ImageColor
HEX_COLOR_REGEX = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")


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
        return [
            True if HEX_COLOR_REGEX.search(col) else False for col in self.hex_color
        ]
