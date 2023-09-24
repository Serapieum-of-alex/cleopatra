from cleopatra.colors import Colors


def test_create_colors_object():
    """test_create_colors_object.

        test_create_colors_object

    Parameters
    ----------

    Returns
    -------

    """
    hex_number = "ff0000"
    color = Colors(hex_number)
    assert color.hex_color == hex_number
