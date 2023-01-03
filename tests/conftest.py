from typing import List
import numpy as np
import pytest


@pytest.fixture(scope="module")
def arr() -> np.ndarray:
    return np.load("examples/data/arr.npy")


@pytest.fixture(scope="module")
def no_data_value(arr: np.ndarray) -> float:
    return arr[0, 0]


@pytest.fixture(scope="module")
def rhine_dem_arr() -> np.ndarray:
    return np.load("examples/data/DEM5km_Rhine_burned_fill.npy")


@pytest.fixture(scope="module")
def rhine_no_data_val(rhine_dem_arr: np.ndarray) -> float:
    return rhine_dem_arr[0, 0]


@pytest.fixture(scope="module")
def cmap() -> str:
    return "terrain"


@pytest.fixture(scope="module")
def color_scale() -> List[int]:
    return [1, 2, 3, 4, 5]


@pytest.fixture(scope="module")
def ticks_spacing() -> int:
    return 500


@pytest.fixture(scope="module")
def color_scale_2_gamma() -> float:
    return 0.5


@pytest.fixture(scope="module")
def color_scale_3_linscale() -> float:
    return 0.001


@pytest.fixture(scope="module")
def color_scale_3_linthresh() -> float:
    return 0.0001


@pytest.fixture(scope="module")
def bounds() -> list:
    return [-559, 0, 440, 940, 1440, 1940, 2440, 2940, 3500]


@pytest.fixture(scope="module")
def midpoint() -> int:
    return 20


@pytest.fixture(scope="module")
def display_cellvalue() -> bool:
    return True


@pytest.fixture(scope="module")
def num_size() -> int:
    return 8


@pytest.fixture(scope="module")
def background_color_threshold():
    return None


@pytest.fixture(scope="module")
def IDsize() -> int:
    return 20


@pytest.fixture(scope="module")
def IDcolor() -> str:
    return "green"


@pytest.fixture(scope="module")
def Gaugesize() -> int:
    return 100


@pytest.fixture(scope="module")
def Gaugecolor() -> str:
    return "blue"
