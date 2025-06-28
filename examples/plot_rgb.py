from cleopatra.config import Config

Config.set_matplotlib_backend()

import numpy as np

from cleopatra.array_glyph import ArrayGlyph

sentinel_2 = np.load("tests/data/s2a.npy")
extent = [34.626902783650785, 34.654007151597256, 31.82337186561403, 31.8504762335605]
# %% process the channels by excluding the top and lowest 1% of the values.
array = ArrayGlyph(sentinel_2, rgb=[3, 2, 1], percentile=1, extent=extent)
fig, ax = array.plot()
# %% process the channels by the surface reflectance of the sentinel data,
array = ArrayGlyph(sentinel_2, rgb=[3, 2, 1], surface_reflectance=10000)
array.plot()
# %%
array = ArrayGlyph(
    sentinel_2, rgb=[3, 2, 1], surface_reflectance=10000, cutoff=[0.3, 0.3, 0.3]
)
array.plot()
# %%
sentinel_2 = np.load("tests/data/gaza-20231002.npy")

array = ArrayGlyph(sentinel_2, rgb=[0, 1, 2])
array.plot()
