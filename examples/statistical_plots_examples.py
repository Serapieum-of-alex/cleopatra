import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from cleopatra.statistics import Statistic


np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)
stat_plot = Statistic(x)
res = stat_plot.histogram()
