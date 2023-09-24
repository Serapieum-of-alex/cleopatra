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
