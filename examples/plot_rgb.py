import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from cleopatra.arrayglyph import ArrayGlyph

sentinel_2 = np.load("tests/data/s2a.npy")

array = ArrayGlyph(sentinel_2, rgb=[3, 2, 1], cutoff=[0.3, 0.3, 0.3])
array.plot()
