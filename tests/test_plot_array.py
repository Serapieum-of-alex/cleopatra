import os
from typing import List
import numpy as np

from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from cleopatra.array import Array


class TestCreateArray:
    def test_create_instance(self, arr: np.ndarray, no_data_value: float):
        array = Array(arr, exclude_value=[no_data_value])
        assert isinstance(array.arr, np.ndarray)
        assert np.isnan(array.arr[0, 0])
        assert array.no_elem == 89
        assert array.vmin == 0
        assert array.vmax == 88


class TestRGB:
    def test_plot_rgb(self, sentinel_2: np.ndarray):
        array = Array(sentinel_2, rgb=[3, 2, 1], cutoff=[0.3, 0.3, 0.3])
        fig, ax = array.plot(title="Flow Accumulation")
        assert isinstance(fig, Figure)


class TestPlotArray:
    def test_plot_numpy_array(
        self,
        arr: np.ndarray,
        no_data_value: float,
    ):
        array = Array(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(title="Flow Accumulation")
        assert isinstance(fig, Figure)

    def test_plot_numpy_array_with_extent(
        self,
        arr: np.ndarray,
        no_data_value: float,
    ):
        extent = [
            -75.60441003848668,
            4.235054115032001,
            -75.09878783366909,
            4.704560448076901,
        ]
        array = Array(arr, exclude_value=[no_data_value], extent=extent)
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
        array = Array(arr, exclude_value=[no_data_value])
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
        array = Array(arr, exclude_value=[no_data_value])
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
        array = Array(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            color_scale=color_scale[2],
            line_scale=color_scale_3_linscale,
            line_threshold=color_scale_3_linthresh,
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
        array = Array(arr, exclude_value=[no_data_value])
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
        array = Array(rhine_dem_arr, exclude_value=[rhine_no_data_val])
        fig, ax = array.plot(
            color_scale=color_scale[3],
            cmap=cmap,
            ticks_spacing=ticks_spacing,
            bounds=bounds,
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
        array = Array(arr, exclude_value=[no_data_value])
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
        display_cell_value: bool,
        num_size,
        background_color_threshold,
    ):
        array = Array(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            display_cell_value=display_cell_value,
            num_size=num_size,
            background_color_threshold=background_color_threshold,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_with_points(
        self,
        arr: np.ndarray,
        no_data_value: float,
        display_cell_value: bool,
        points,
        num_size,
        background_color_threshold,
        ticks_spacing: int,
        id_size: int,
        id_color: str,
        point_size: int,
        gauge_color: str,
    ):
        array = Array(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            points=points,
            point_color=gauge_color,
            point_size=point_size,
            id_color=id_color,
            id_size=id_size,
            display_cell_value=display_cell_value,
            num_size=num_size,
            background_color_threshold=background_color_threshold,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)


class TestAnimate:
    def test_numpy_array(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        array = Array(coello_data, exclude_value=[no_data_value])
        anim_obj = array.animate(animate_time_list, title="Flow Accumulation")
        assert isinstance(anim_obj, FuncAnimation)

    def test_save_animation_gif(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "gif"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = Array(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)

    def test_save_animation_avi(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "avi"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = Array(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)

    def test_save_animation_mp4(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "mp4"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = Array(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)

    def test_save_animation_mov(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "mov"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = Array(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)
