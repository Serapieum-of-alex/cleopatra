import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from cleopatra.array import Array

sentinel_2 = np.load("tests/data/s2a.npy")

array = Array(sentinel_2, rgb=[3, 2, 1], cutoff=[0.3, 0.3, 0.3])
array.plot()
