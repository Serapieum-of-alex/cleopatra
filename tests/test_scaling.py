"""Tests for the `ColorScaling` grouped colour-scale object."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest

from cleopatra.styling.params import CellValues, Classify, Contour, DataStyle
from cleopatra.styling.scaling import (
    ColorScaling,
    _log_tick_positions,
    _symlog_tick_positions,
)


class TestColorScalingToOptions:
    """`ColorScaling.to_options` emits the flat colour-scale keys."""

    def test_non_midpoint_variant_does_not_leak_a_method_into_midpoint(self):
        """A non-midpoint scale emits the numeric `midpoint` default, not a method.

        Test scenario:
            Regression: the `midpoint` field once shadowed the `midpoint()`
            variant constructor, so `power()`/`linear()` emitted a bound
            method as the `midpoint` option instead of `0`.
        """
        options = ColorScaling.power(gamma=0.7).to_options()
        assert options["midpoint"] == 0, (
            f"midpoint should default to 0, got {options['midpoint']!r}"
        )
        assert isinstance(options["midpoint"], (int, float)), (
            f"midpoint must be numeric, got {type(options['midpoint'])}"
        )

    def test_midpoint_variant_carries_its_centre(self):
        """`ColorScaling.midpoint(at=X)` emits `X` as the `midpoint` option."""
        assert ColorScaling.midpoint(at=42).to_options()["midpoint"] == 42

    @pytest.mark.parametrize(
        "scale, key",
        [
            (ColorScaling.power(gamma=0.3), "color_scale"),
            (ColorScaling.boundary(bounds=[0, 1, 2]), "bounds"),
            (ColorScaling.sym_log(threshold=0.01, scale=0.1), "line_threshold"),
            (ColorScaling.log(), "color_scale"),
        ],
    )
    def test_variant_emits_all_six_keys(self, scale, key):
        """Every variant emits the full six-key option dict (full-scale reset).

        Args:
            scale: A `ColorScaling` variant.
            key: A key expected in the emitted options.
        """
        options = scale.to_options()
        assert set(options) == {
            "color_scale",
            "gamma",
            "line_threshold",
            "line_scale",
            "bounds",
            "midpoint",
        }, f"expected all six keys, got {set(options)}"
        assert key in options


class TestColorScalingBuildNorm:
    """`ColorScaling.build_norm` reproduces the scale's matplotlib norm."""

    def test_linear_without_levels_has_no_norm(self):
        """A plain linear scale returns no norm and passes ticks through."""
        norm, cbar_kw = ColorScaling.linear().build_norm(np.array([0.0, 5.0, 10.0]))
        assert norm is None, "linear scale should have no explicit norm"
        assert cbar_kw["extend"] == "neither"

    def test_midpoint_builds_a_midpoint_norm(self):
        """The midpoint scale builds a `MidpointNormalize` centred at `at`."""
        norm, _ = ColorScaling.midpoint(at=2.0).build_norm(np.array([0.0, 4.0]))
        assert type(norm).__name__ == "MidpointNormalize"
        assert norm.midpoint == 2.0, f"midpoint should be 2.0, got {norm.midpoint}"

    def test_power_builds_a_power_norm(self):
        """The power scale builds a `PowerNorm` with the given gamma."""
        norm, _ = ColorScaling.power(gamma=0.5).build_norm(np.array([0.0, 10.0]))
        assert isinstance(norm, mcolors.PowerNorm)
        assert norm.gamma == 0.5

    def test_log_builds_a_log_norm(self):
        """The log scale builds a `LogNorm` over the positive tick range."""
        norm, cbar_kw = ColorScaling.log().build_norm(np.array([1.0, 10.0, 100.0]))
        assert isinstance(norm, mcolors.LogNorm)
        assert (norm.vmin, norm.vmax) == (1.0, 100.0), (
            f"LogNorm should span the ticks, got ({norm.vmin}, {norm.vmax})"
        )
        assert cbar_kw["extend"] == "neither"

    def test_log_on_non_positive_range_raises(self):
        """A log scale whose range starts at zero raises, steering at sym_log."""
        scale = ColorScaling.log()
        ticks = np.array([0.0, 10.0, 100.0])
        with pytest.raises(ValueError, match="strictly-positive"):
            scale.build_norm(ticks)

    def test_log_on_constant_positive_data_widens_the_range(self):
        """A constant positive field (single tick) builds a LogNorm, not a crash.

        Test scenario:
            Uniform data yields one tick, so vmin == vmax. A log scale cannot
            span a zero-width range; the branch widens it (like the data-style
            path) rather than raising, matching the other scale kinds.
        """
        norm, _ = ColorScaling.log().build_norm(np.array([5.0]))
        assert isinstance(norm, mcolors.LogNorm)
        assert norm.vmin == 5.0, f"vmin should stay 5.0, got {norm.vmin}"
        assert norm.vmax == 6.0, f"vmax should widen to 6.0, got {norm.vmax}"

    def test_log_on_constant_negative_data_reports_real_bounds(self):
        """A constant non-positive field raises with its real bound, not a widened one.

        Test scenario:
            The degenerate-range widening applies only to strictly-positive
            constants, so an all-negative field is not widened before the error
            is built -- the message reports the real value and steers at sym_log.
        """
        scale = ColorScaling.log()
        ticks = np.array([-5.0])
        with pytest.raises(ValueError, match=r"vmin=-5\.0, vmax=-5\.0"):
            scale.build_norm(ticks)

    def test_log_options_round_trip(self):
        """`log()` emits `color_scale='lognorm'` and reconstructs to LOGNORM."""
        opts = ColorScaling.log().to_options()
        assert opts["color_scale"] == "lognorm"
        assert ColorScaling.from_options(opts).kind.name == "LOGNORM"

    def test_sym_log_bar_ticks_are_scale_aware_and_signed(self):
        """sym_log places decade-aligned bar ticks and keeps negative signs (#335).

        Test scenario:
            The linear ladder ([-24, 744] here) supplies vmin/vmax, but the bar
            ticks are the symlog decades within that range, and the formatter
            labels a negative decade with its sign -- unlike LogFormatter, which
            blanked non-decades and dropped the sign of -10.
        """
        _, cbar_kw = ColorScaling.sym_log(threshold=10.0, scale=1.0).build_norm(
            np.array([-24.0, 0.0, 744.0])
        )
        ticks = np.asarray(cbar_kw["ticks"])
        assert ticks.size >= 2, f"expected several bar ticks, got {ticks.tolist()}"
        assert ticks.min() >= -24.0, f"tick below vmin: {ticks.tolist()}"
        assert ticks.max() <= 744.0, f"tick above vmax: {ticks.tolist()}"
        assert ticks.min() < 0.0, f"a below-zero range should span a negative tick: {ticks.tolist()}"
        nonzero = ticks[ticks != 0.0]
        decades = np.log10(np.abs(nonzero))
        assert np.allclose(decades, np.round(decades)), f"non-decade ticks: {ticks.tolist()}"
        fmt = cbar_kw["format"]
        assert fmt(-10.0) == "-10", f"negative decade must keep its sign, got {fmt(-10.0)!r}"
        assert fmt(100.0) == "100", f"expected '100', got {fmt(100.0)!r}"

    def test_log_bar_ticks_are_decade_aligned(self):
        """log places decade bar ticks and a formatter that labels them (#335)."""
        _, cbar_kw = ColorScaling.log().build_norm(np.array([0.5, 744.0]))
        ticks = np.asarray(cbar_kw["ticks"])
        assert ticks.size >= 2, f"expected several bar ticks, got {ticks.tolist()}"
        assert ticks.min() >= 0.5, f"tick below vmin: {ticks.tolist()}"
        assert ticks.max() <= 744.0, f"tick above vmax: {ticks.tolist()}"
        decades = np.log10(ticks)
        assert np.allclose(decades, np.round(decades)), f"non-decade ticks: {ticks.tolist()}"
        assert cbar_kw["format"](10.0) == "10", "log formatter should label a decade"

    def test_non_linear_formatter_labels_arbitrary_positions(self):
        """The sym_log formatter labels any value, so set_ticks needs no set_ticklabels.

        Test scenario:
            The formatter is position-agnostic: a caller's non-decade tick (e.g.
            -5 or 42.5) is labelled with its plain value, which is what makes a
            later cbar.set_ticks([...]) readable without a paired set_ticklabels.
        """
        _, cbar_kw = ColorScaling.sym_log(threshold=10.0).build_norm(
            np.array([-24.0, 744.0])
        )
        fmt = cbar_kw["format"]
        assert fmt(-5.0) == "-5", f"expected '-5', got {fmt(-5.0)!r}"
        assert fmt(42.5) == "42.5", f"expected '42.5', got {fmt(42.5)!r}"

    def test_tick_positions_fall_back_to_the_ladder_when_sparse(self):
        """When no decade lands in range, the helpers return the caller's ladder."""
        ladder = np.array([1.0, 2.0, 3.0])
        assert _log_tick_positions(2.0, 5.0, ladder).tolist() == ladder.tolist(), (
            "log within one decade should fall back to the ladder"
        )
        assert _symlog_tick_positions(200.0, 500.0, 10.0, ladder).tolist() == (
            ladder.tolist()
        ), "symlog within one decade should fall back to the ladder"

    def test_formatter_normalizes_signed_zero(self):
        """The tick formatter renders a signed zero as '0', not '-0'."""
        fmt = ColorScaling.sym_log(threshold=10.0).build_norm(
            np.array([-24.0, 744.0])
        )[1]["format"]
        assert fmt(-0.0) == "0", f"signed zero should render '0', got {fmt(-0.0)!r}"
        assert fmt(0.0) == "0", f"zero should render '0', got {fmt(0.0)!r}"

    def test_log_ticks_stay_bounded_over_many_decades(self):
        """A log bar spanning many decades stays a handful of decade ticks, not hundreds."""
        _, cbar_kw = ColorScaling.log().build_norm(np.array([1e-6, 1e6]))
        ticks = np.asarray(cbar_kw["ticks"])
        assert 2 <= ticks.size <= 30, f"decade set should stay bounded, got {ticks.size}"
        decades = np.log10(ticks)
        assert np.allclose(decades, np.round(decades)), f"non-decade ticks: {ticks.tolist()}"


class TestParamGroupsEmitOnlySetFields:
    """`Contour`/`CellValues`/`DataStyle`/`Classify` emit only the fields set."""

    def test_empty_groups_emit_nothing(self):
        """A group with no fields set emits an empty option dict."""
        assert Contour().to_options() == {}
        assert CellValues().to_options() == {}
        assert Classify().to_options() == {}

    def test_classify_emits_only_set_fields(self):
        """`Classify` emits scheme/k/category_legend_kwargs only when given."""
        assert Classify(scheme="quantiles").to_options() == {"scheme": "quantiles"}
        assert Classify(k=4).to_options() == {"k": 4}
        assert Classify(category_legend_kwargs={"loc": "upper left"}).to_options() == {
            "category_legend_kwargs": {"loc": "upper left"}
        }
        assert Classify(scheme="quantiles", k=4).to_options() == {
            "scheme": "quantiles",
            "k": 4,
        }

    def test_contour_and_cells_emit_only_set_fields(self):
        """`Contour`/`CellValues` emit only the fields explicitly provided."""
        assert Contour(levels=5).to_options() == {"levels": 5}
        assert Contour(labels=True, label_kw={"fmt": "%.2f"}).to_options() == {
            "labels": True,
            "label_kw": {"fmt": "%.2f"},
        }
        assert CellValues(show=True, size=8, background_threshold=0.5).to_options() == {
            "display_cell_value": True,
            "num_size": 8,
            "background_color_threshold": 0.5,
        }

    def test_datastyle_unset_omits_but_explicit_none_clears(self):
        """`DataStyle` omits unset fields but emits an explicit `None` (clear)."""
        assert DataStyle().to_options() == {}
        assert DataStyle(style=None).to_options() == {"style": None}
        assert DataStyle(style="dem", hillshade=True).to_options() == {
            "style": "dem",
            "hillshade": True,
        }
