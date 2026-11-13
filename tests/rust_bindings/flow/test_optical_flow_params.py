# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The flow functions' keyword overrides of a preset's DIS parameters."""

import numpy as np
import pytest

from sfmtool.flow import (
    compute_optical_flow,
    compute_optical_flow_timed,
    compute_optical_flow_with_init,
    gpu_available,
)

_SHIFT = 0.6


def _texture(width: int, height: int, dx: float) -> np.ndarray:
    """A smooth uint8 texture sampled at (x - dx, y): the flow from the
    dx = 0 image to this one is (dx, 0) everywhere."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    x = x - dx
    v = 0.5 + 0.2 * np.sin(0.4 * x) + 0.2 * np.cos(0.33 * y + 0.1 * x)
    return np.round(v * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def pair():
    """A 64 x 64 pair moved by a sub-pixel horizontal shift."""
    return _texture(64, 64, 0.0), _texture(64, 64, _SHIFT)


def _median_error(flow_u: np.ndarray, flow_v: np.ndarray) -> float:
    inner = (slice(8, -8), slice(8, -8))
    return float(np.median(np.hypot(flow_u[inner] - _SHIFT, flow_v[inner])))


def test_finest_scale_zero_beats_high_quality_half_resolution(pair):
    """high_quality solves at half resolution at most; finest_scale=0 with
    dense patches solves at full resolution and is measurably closer."""
    a, b = pair
    hq = _median_error(
        *compute_optical_flow(a, b, preset="high_quality", use_gpu=False)
    )
    full = _median_error(
        *compute_optical_flow(
            a, b, preset="high_quality", use_gpu=False, finest_scale=0, patch_stride=1
        )
    )
    assert full < 0.05, full
    assert full < 0.7 * hq, (full, hq)


def test_overrides_equal_to_the_preset_change_nothing(pair):
    """Overriding every field with the preset's own value gives the preset's
    flow, so each keyword sets the field it names (a stride of 3 is
    high_quality's 0.75 overlap of a 12-pixel patch)."""
    a, b = pair
    expected = compute_optical_flow(a, b, preset="high_quality", use_gpu=False)
    got = compute_optical_flow(
        a,
        b,
        preset="high_quality",
        use_gpu=False,
        patch_size=12,
        patch_stride=3,
        grad_descent_iterations=16,
        finest_scale=1,
        variational_refinement=True,
        variational_jacobi_iterations=7,
        variational_outer_iterations_base=1,
        variational_alpha=10.0,
        variational_gamma=10.0,
        variational_delta=5.0,
        normalize_patches=True,
        gpu_min_pixels=50_000,
    )
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


@pytest.mark.parametrize(
    "override",
    [
        {"patch_size": 8},
        {"patch_stride": 1},
        {"grad_descent_iterations": 2},
        {"variational_refinement": False},
        {"variational_jacobi_iterations": 1},
        {"variational_outer_iterations_base": 3},
        {"variational_alpha": 1.0},
        {"variational_gamma": 0.0},
        {"variational_delta": 50.0},
        {"normalize_patches": False},
    ],
)
def test_each_override_changes_the_flow(pair, override):
    """Changing one field from high_quality's value changes the flow, so
    the keyword reaches the solver."""
    a, b = pair
    base = compute_optical_flow(a, b, preset="high_quality", use_gpu=False)
    changed = compute_optical_flow(
        a, b, preset="high_quality", use_gpu=False, **override
    )
    assert not np.array_equal(base[0], changed[0])


def test_timed_reports_the_overridden_levels(pair):
    a, b = pair
    timing = compute_optical_flow_timed(
        a, b, preset="fast", use_gpu=False, finest_scale=0, coarsest_scale=2
    )
    assert timing["levels_processed"] == 3
    assert [level[:3] for level in timing["per_level"]] == [
        (2, 16, 16),
        (1, 32, 32),
        (0, 64, 64),
    ]
    off = compute_optical_flow_timed(
        a, b, preset="high_quality", use_gpu=False, variational_refinement=False
    )
    assert off["variational_total"] == 0.0
    on = compute_optical_flow_timed(
        a, b, preset="fast", use_gpu=False, variational_refinement=True
    )
    assert on["variational_total"] > 0.0


def test_with_init_takes_overrides(pair):
    a, b = pair
    zero = np.zeros(a.shape, dtype=np.float32)
    hq = _median_error(
        *compute_optical_flow_with_init(
            a, b, zero, zero, preset="high_quality", use_gpu=False
        )
    )
    full = _median_error(
        *compute_optical_flow_with_init(
            a,
            b,
            zero,
            zero,
            preset="high_quality",
            use_gpu=False,
            finest_scale=0,
            patch_stride=1,
        )
    )
    assert full < 0.7 * hq, (full, hq)
    with pytest.raises(ValueError, match="patch_stride must be at least 1"):
        compute_optical_flow_with_init(a, b, zero, zero, patch_stride=0)


@pytest.mark.skipif(not gpu_available(), reason="no GPU adapter")
def test_gpu_min_pixels_routes_levels_to_the_gpu(pair):
    """With gpu_min_pixels 0 every level of a small image runs on the GPU,
    which the timing reports as levels with no separate DIS or variational
    time; with the default threshold they all run on the CPU."""
    a, b = pair
    gpu = compute_optical_flow_timed(
        a, b, preset="high_quality", use_gpu=True, gpu_min_pixels=0
    )
    assert all(level[3:] == (0.0, 0.0) for level in gpu["per_level"])
    cpu = compute_optical_flow_timed(a, b, preset="high_quality", use_gpu=True)
    assert all(level[4] > 0.0 for level in cpu["per_level"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"patch_size": 0}, "patch_size must be at least 1"),
        ({"patch_size": 65}, "patch_size 65 is larger than the 64x48 image"),
        ({"patch_stride": 0}, "patch_stride must be at least 1"),
        ({"grad_descent_iterations": 0}, "grad_descent_iterations must be at least 1"),
        (
            {"variational_jacobi_iterations": 0},
            "variational_jacobi_iterations must be at least 1",
        ),
        (
            {"variational_outer_iterations_base": 0},
            "variational_outer_iterations_base must be at least 1",
        ),
        ({"variational_alpha": -1.0}, "variational_alpha must be finite"),
        ({"variational_gamma": float("nan")}, "variational_gamma must be finite"),
        ({"variational_delta": float("inf")}, "variational_delta must be finite"),
        ({"finest_scale": 6}, "finest_scale 6 is a pyramid level less than one pixel"),
        ({"coarsest_scale": 6}, "coarsest_scale 6 is a pyramid level less than one"),
        (
            {"finest_scale": 2, "coarsest_scale": 1},
            "finest_scale 2 is larger than coarsest_scale 1",
        ),
    ],
)
def test_out_of_range_overrides_raise_value_error(override, message):
    a = _texture(64, 48, 0.0)
    with pytest.raises(ValueError, match=message):
        compute_optical_flow(a, a, use_gpu=False, **override)
    with pytest.raises(ValueError, match=message):
        compute_optical_flow_timed(a, a, use_gpu=False, **override)


def test_bad_override_types_raise():
    a = _texture(16, 16, 0.0)
    with pytest.raises(OverflowError):
        compute_optical_flow(a, a, patch_size=-1)
    with pytest.raises(TypeError):
        compute_optical_flow(a, a, patch_spacing=1)
    with pytest.raises(TypeError):
        # The overrides are keyword-only.
        compute_optical_flow(a, a, "default", False, 8)
