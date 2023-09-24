from cleopatra.colors import Colors


def test_create_colors_object():
    """test_create_colors_object."""
    hex_number = "ff0000"
    color = Colors(hex_number)
    assert color.hex_color == [hex_number]


def test_is_valid():
    """test_create_colors_object."""
    hex_number = ["ff0000", "#23a9dd"]
    color = Colors(hex_number)
    valid = color.is_valid_hex()
    assert valid == [False, True]


def test_get_rgb():
    """test_create_colors_object."""
    hex_number = ["#ff0000", "#23a9dd"]
    color = Colors(hex_number)
    rgb_scale_1 = color.get_rgb(normalized=True)
    assert rgb_scale_1 == [
        (1.0, 0.0, 0.0),
        (0.13725490196078433, 0.6627450980392157, 0.8666666666666667),
    ]
    rgb_scale_255 = color.get_rgb(normalized=False)
    assert rgb_scale_255 == [(255, 0, 0), (35, 169, 221)]
