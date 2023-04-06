import numpy as np
from cleopatra.statistics import Statistic


def test_histogram():
    # make data
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, 200)
    stat_plot = Statistic(x)
    res = stat_plot.histogram()
