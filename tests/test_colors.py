from cleopatra.colors import Colors


class TestCreateColors:
    def test_create_from_hex(self):
        """test_create_colors_object."""
        hex_number = "ff0000"
        color = Colors(hex_number)
        assert color._color_value == [hex_number]

    def test_create_from_rgb(self):
        """test_create_colors_object."""
        hex_number = (128, 51, 204)
        color = Colors(hex_number)
        assert color._color_value == [hex_number]


def test_is_valid():
    """test_create_colors_object."""
    hex_number = ["ff0000", "#23a9dd"]
    color = Colors(hex_number)
    valid = color.is_valid_hex()
    assert valid == [False, True]


def test_to_rgb():
    """test_create_colors_object."""
    hex_number = ["#ff0000", "#23a9dd"]
    color = Colors(hex_number)
    rgb_scale_1 = color.to_rgb(normalized=True)
    assert rgb_scale_1 == [
        (1.0, 0.0, 0.0),
        (0.13725490196078433, 0.6627450980392157, 0.8666666666666667),
    ]
    rgb_scale_255 = color.to_rgb(normalized=False)
    assert rgb_scale_255 == [(255, 0, 0), (35, 169, 221)]
