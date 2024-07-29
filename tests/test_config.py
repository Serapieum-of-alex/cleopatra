import matplotlib.pyplot as plt

from cleopatra.config import Config, is_notebook


class TestSetMatplotlibBackend:

    def test_set_set_matplotlib_backend(self):
        Config.set_matplotlib_backend()
        backend = plt.get_backend()
        assert backend == 'TkAgg'


def test_is_notebook():
    assert not is_notebook()
