=======
History
=======

0.1.0 (2022-05-24)
------------------

* First release on PyPI.

0.2.4 (2022-12-26)
------------------

* bump up numpy versions to 1.23.5, add pandas

0.2.5 (2022-12-26)
------------------

* plot array with discrete bounds takes the bounds as a parameter

0.2.6 (2023-01-31)
------------------

* bump up versions
* add serapeum_utils as a dependency

0.2.7 (2023-01-31)
------------------
* bump up numpy to version 1.24.1

0.3.0 (2023-04-11)
------------------
* change API to work completly with numpy array inputs
* chenge to conda config
* add hpc-utils to filter and access arrays
* restructure the whole modules to array, statistics, and styles modules.
* all modules has classes.
* save animation function using ffmpeg.

0.3.1 (2023-04-17)
------------------
* plot RGB plots

0.3.2 (2023-04-23)
------------------
* bump up hpc version

0.3.3 (2023-04-25)
------------------
* change the default value for the color bar label.

0.3.4 (2023-04-26)
------------------
* pass the plot kwargs to the init of the array to scale the color bar using the vmin and vmax.

0.3.5 (2023-8-31)
------------------
* Update dependencies

0.4.0 (2023-9-24)
------------------
* Add a colors module to handle issues related to
  - Converting colors from one format to another
  - Creating colormaps

0.4.1 (2024-1-11)
------------------
* add extent to the array plot.

0.4.2 (2024-06-30)
------------------
* Update dependencies

0.4.3 (2024-07-13)
------------------
* Add extent to the array plot when plotting an rgb array.
* Add `ax`, and `fig` parameters to the `Array` constructor method to take an Axes and plot the array on it.
* Add `__str__` to the `Array` class.

0.5.0 (2024-07-22)
------------------

ArrayGlyph
^^^^^^^^^^
* rename the `Array` class to `ArrayGlyph`.
* add `scale_percentile` method to the `Array` class to scale the array using the percentile values.
* the `statistic.histogram` can plot multiple column array.
* change the `color_scale` values to be string (`linear`, "power", ...)
* the `kwargs` can be provided to the constructor or the `plot` method to plot the array.

Colors
^^^^^^
* rename the `get_rgb` to `to_rgb`
* add `get_type` to get the type of the color.
* add `to_hex` to convert the color to hex.
* add `to_rgb` to convert the color to rgb.

0.5.1 (2024-07-24)
------------------

ArrayGlyph
^^^^^^^^^^
* the ArrayGlyph constructor uses a masked array instead of a numpy array.


0.5.2 (2025-05-08)
------------------

Dev
^^^
* replace the setup.py with pyproject.toml
