import matplotlib

matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
import numpy as np

from cleopatra.array import Array

#%%
arr = np.load("examples/data/arr.npy")
exculde_value = arr[0, 0]
cmap = "terrain"
#%%
fig, ax = Array.plot(arr, exculde_value=exculde_value, Title="Flow Accumulation")
#%% test_plot_array_color_scale_1
ticks_spacing = 500
color_scale = [1, 2, 3, 4, 5]
fig, ax = Array.plot(
    arr,
    exculde_value=exculde_value,
    color_scale=color_scale[0],
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
#%% test_plot_array_color_scale_2
color_scale_2_gamma = 0.5
fig, ax = Array.plot(
    arr,
    exculde_value=exculde_value,
    color_scale=color_scale[1],
    cmap=cmap,
    gamma=color_scale_2_gamma,
    ticks_spacing=ticks_spacing,
)
#%% test_plot_array_color_scale_3
color_scale_3_linscale = 0.001
color_scale_3_linthresh = 0.0001
fig, ax = Array.plot(
    arr,
    exculde_value=exculde_value,
    color_scale=color_scale[2],
    linscale=color_scale_3_linscale,
    linthresh=color_scale_3_linthresh,
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
#%% test_plot_array_color_scale_4

fig, ax = Array.plot(
    arr,
    exculde_value=exculde_value,
    color_scale=color_scale[3],
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)

#%% test_plot_array_color_scale_5
midpoint = 20

fig, ax = Array.plot(
    arr,
    color_scale=color_scale[4],
    midpoint=midpoint,
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
# %% test_plot_array_display_cell_values
display_cellvalue = True
num_size = 8
background_color_threshold = None

fig, ax = Array.plot(
    arr,
    display_cellvalue=display_cellvalue,
    NumSize=num_size,
    Backgroundcolorthreshold=background_color_threshold,
    ticks_spacing=ticks_spacing,
)
