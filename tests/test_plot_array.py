from typing import List
import numpy as np
from matplotlib.figure import Figure
from cleopatra.array import Array


class TestCreateArray:
    def test_create_instance(self, arr: np.ndarray, no_data_value: float):
        array = Array(arr, exculde_value=no_data_value)
        assert isinstance(array.arr, np.ndarray)
        assert np.isnan(array.arr[0, 0])
        assert array.no_elem == 89
        assert array.vmin == 0
        assert array.vmax == 88


class TestPlotArray:
    def test_plot_numpy_array(
        self,
        arr: np.ndarray,
        no_data_value: float,
    ):
        array = Array(arr, exculde_value=no_data_value)
        fig, ax = array.plot(title="Flow Accumulation")
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_1(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: List[int],
        ticks_spacing: int,
    ):
        array = Array(arr, exculde_value=no_data_value)
        fig, ax = array.plot(
            color_scale=color_scale[0], cmap=cmap, ticks_spacing=ticks_spacing
        )
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_2(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale_2_gamma: float,
        color_scale: List[int],
        ticks_spacing: int,
    ):
        array = Array(arr, exculde_value=no_data_value)
        fig, ax = array.plot(
            color_scale=color_scale[1],
            cmap=cmap,
            gamma=color_scale_2_gamma,
            ticks_spacing=ticks_spacing,
        )
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_3(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: List[int],
        ticks_spacing: int,
        color_scale_3_linscale: float,
        color_scale_3_linthresh: float,
    ):
        array = Array(arr, exculde_value=no_data_value)
        fig, ax = array.plot(
            color_scale=color_scale[2],
            linscale=color_scale_3_linscale,
            linthresh=color_scale_3_linthresh,
            cmap=cmap,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_4(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: List[int],
        ticks_spacing: int,
    ):
        array = Array(arr, exculde_value=no_data_value)
        fig, ax = array.plot(color_scale=color_scale[3], cmap=cmap, ticks_spacing=5)

        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_4_with_bounds(
        self,
        rhine_dem_arr: np.ndarray,
        cmap: str,
        color_scale: List[int],
        ticks_spacing: int,
        bounds: list,
        rhine_no_data_val: float,
    ):
        array = Array(rhine_dem_arr, exculde_value=rhine_no_data_val)
        fig, ax = array.plot(
            color_scale=color_scale[3],
            cmap=cmap,
            ticks_spacing=ticks_spacing,
            bounds=bounds,
            exculde_value=rhine_no_data_val,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_5(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: List[int],
        ticks_spacing: int,
        midpoint: int,
    ):
        array = Array(arr, exculde_value=no_data_value)
        fig, ax = array.plot(
            color_scale=color_scale[4],
            midpoint=midpoint,
            cmap=cmap,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_display_cell_values(
        self,
        arr: np.ndarray,
        no_data_value: float,
        ticks_spacing: int,
        display_cellvalue: bool,
        num_size,
        background_color_threshold,
    ):
        array = Array(arr, exculde_value=no_data_value)
        fig, ax = array.plot(
            display_cellvalue=display_cellvalue,
            num_size=num_size,
            background_color_threshold=background_color_threshold,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)

    # def test_plot_array_with_points(
    #         self,
    #         arr: np.ndarray,
    #         display_cellvalue: bool,
    #         points,
    #         num_size,
    #         background_color_threshold,
    #         ticks_spacing: int,
    #         id_size: int,
    #         id_color: str,
    #         point_size: int,
    #         Gaugecolor: str,
    # ):
    #     array = Array(arr, exculde_value=no_data_value)
    #     fig, ax = array.plot(
    #         Gaugecolor=Gaugecolor,
    #         point_size=point_size,
    #         id_color=id_color,
    #         id_size=id_size,
    #         points=points,
    #         display_cellvalue=display_cellvalue,
    #         NumSize=num_size,
    #         Backgroundcolorthreshold=background_color_threshold,
    #         ticks_spacing=ticks_spacing,
    #     )
    #
    #     assert isinstance(fig, Figure)
