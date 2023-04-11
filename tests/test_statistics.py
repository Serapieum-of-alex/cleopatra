import numpy as np
from matplotlib.figure import Figure
from cleopatra.statistics import Statistic


def test_histogram():
    # make data
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, 200)
    stat_plot = Statistic(x)
    fig, ax, hist = stat_plot.histogram()
    assert isinstance(fig, Figure)
    assert isinstance(hist, dict)
    assert ["n", "bins", "patches"] == list(hist.keys())
