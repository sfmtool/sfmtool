# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``zncc_self_similarity_parts`` PyO3 binding."""

import numpy as np
import pytest

from sfmtool._sfmtool.patches import (
    zncc_self_similarity_parts,
    zncc_self_similarity_parts_stack,
)

R = 24


def _edge(angle_deg: float, size: int = R) -> np.ndarray:
    """A straight edge through the bitmap centre along ``angle_deg``, 150 grey
    levels high, with a one-pixel linear ramp across it."""
    s, c = np.sin(np.radians(angle_deg)), np.cos(np.radians(angle_deg))
    mid = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    across = -(x - mid) * s + (y - mid) * c
    return (50.0 + 150.0 * np.clip(0.5 + across, 0.0, 1.0)).astype(np.float32)


def test_the_result_carries_every_part_in_its_shape():
    rng = np.random.default_rng(1)
    bitmap = rng.integers(0, 256, size=(R, R, 3), dtype=np.uint8)
    out = zncc_self_similarity_parts(bitmap)
    assert isinstance(out["radius"], float)
    assert isinstance(out["radius_middle"], float)
    assert isinstance(out["tolerance"], float)
    assert out["radius_grid"].shape == (3, 3)
    assert out["radius_grid"].dtype == np.float64
    assert isinstance(out["radius_is_at_least"], bool)
    for suffix in ("", "_middle"):
        assert out[f"ellipse_axes{suffix}"].shape == (2,)
        assert out[f"ellipse_axes_is_at_least{suffix}"].dtype == np.bool_
        assert isinstance(out[f"ellipse_major_angle{suffix}"], float)
    assert out["ellipse_matrix"].shape == (2, 2)
    assert out["ellipse_axes_grid"].shape == (3, 3, 2)
    assert out["ellipse_axes_grid"].dtype == np.float64
    assert out["ellipse_axes_is_at_least_grid"].shape == (3, 3, 2)
    assert out["ellipse_major_angle_grid"].shape == (3, 3)
    # The radius is the ellipse's semi-major axis, whole, middle and per cell.
    assert out["radius"] == out["ellipse_axes"][0]
    assert out["radius_middle"] == out["ellipse_axes_middle"][0]
    np.testing.assert_array_equal(out["radius_grid"], out["ellipse_axes_grid"][..., 0])
    # The matrix's eigenvalues are the squared semi-axes.
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(out["ellipse_matrix"]))[::-1],
        out["ellipse_axes"] ** 2,
        rtol=1e-9,
    )
    assert out["surface"].shape == (7, 7)
    assert out["surface"][3, 3] == 1.0
    # The corners outside the disk of radius 3 are read too.
    assert np.isfinite(out["surface"]).all()
    # Random texture pins its position: its surface falls through the level
    # within a pixel.
    assert 0.0 < out["radius"] < 1.0


def test_an_edge_reads_a_long_ellipse_along_itself_and_a_corner_a_small_one():
    out = zncc_self_similarity_parts(_edge(0.0))
    assert out["radius"] == 3.0
    assert out["radius_is_at_least"]
    major, minor = out["ellipse_axes"]
    assert minor < 0.5 * major
    # The edge runs along x, so its major axis does too.
    angle = out["ellipse_major_angle"]
    assert min(angle, np.pi - angle) < 1e-6

    y, x = np.mgrid[0:R, 0:R]
    corner = np.where((x >= R // 2) & (y >= R // 2), 200, 50).astype(np.uint8)
    out = zncc_self_similarity_parts(corner)
    assert out["radius"] < 1.0
    assert not out["radius_is_at_least"]


def test_a_flat_bitmap_scores_the_maximum_and_has_no_surface():
    out = zncc_self_similarity_parts(np.full((R, R), 128, np.uint8))
    assert out["radius"] == 3.0
    assert out["radius_middle"] == 3.0
    assert (out["radius_grid"] == 3.0).all()
    assert out["tolerance"] == np.inf
    assert np.isnan(out["surface"]).all()
    np.testing.assert_array_equal(out["ellipse_axes"], [3.0, 3.0])
    assert out["ellipse_axes_is_at_least"].all()
    assert np.isnan(out["ellipse_major_angle"])


def test_uint8_float32_and_opaque_alpha_read_alike():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, size=(R, R, 3), dtype=np.uint8)
    rgba = np.concatenate([rgb, np.full((R, R, 1), 255, np.uint8)], axis=2)
    a = zncc_self_similarity_parts(rgb)
    b = zncc_self_similarity_parts(rgb.astype(np.float32))
    c = zncc_self_similarity_parts(rgba)
    for other in (b, c):
        assert other["radius"] == a["radius"]
        np.testing.assert_array_equal(other["radius_grid"], a["radius_grid"])
        np.testing.assert_allclose(other["surface"], a["surface"], atol=1e-12)


def test_samples_with_alpha_zero_carry_no_data():
    rng = np.random.default_rng(3)
    rgba = rng.integers(0, 256, size=(R, R, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[:, :8, 3] = 0
    junk = rgba.copy()
    junk[:, :8, :3] = rng.integers(0, 256, size=(R, 8, 3), dtype=np.uint8)
    a = zncc_self_similarity_parts(rgba)
    b = zncc_self_similarity_parts(junk)
    assert a["radius"] == b["radius"]
    assert np.isnan(a["radius_grid"][:, 0]).all()
    none = zncc_self_similarity_parts(np.zeros((R, R, 4), np.uint8))
    assert np.isnan(none["radius"])
    assert np.isnan(none["ellipse_axes"]).all()
    assert not none["ellipse_axes_is_at_least"].any()


def test_one_bitmap_reads_as_it_does_in_a_stack():
    rng = np.random.default_rng(4)
    stack = rng.integers(0, 256, size=(3, R, R, 4), dtype=np.uint8)
    stack[..., 3] = 255
    together = zncc_self_similarity_parts_stack(stack)
    for i in range(3):
        alone = zncc_self_similarity_parts(stack[i])
        assert alone["radius"] == together["radius"][i]
        assert alone["radius_middle"] == together["radius_middle"][i]
        np.testing.assert_array_equal(alone["radius_grid"], together["radius_grid"][i])
        np.testing.assert_array_equal(
            alone["ellipse_axes"], together["ellipse_axes"][i]
        )
        np.testing.assert_array_equal(
            alone["ellipse_axes_is_at_least"], together["ellipse_axes_is_at_least"][i]
        )
        assert alone["radius_is_at_least"] == together["radius_is_at_least"][i]
        np.testing.assert_equal(
            alone["ellipse_major_angle"], together["ellipse_major_angle"][i]
        )


def test_the_parameters_are_keywords():
    bitmap = _edge(90.0)
    out = zncc_self_similarity_parts(
        bitmap, max_radius=2, relative_tolerance=0.1, noise=1.0
    )
    assert out["radius"] == 2.0
    assert out["surface"].shape == (5, 5)
    # A larger noise raises the tolerance.
    loud = zncc_self_similarity_parts(bitmap, max_radius=2, noise=20.0)
    assert loud["tolerance"] > out["tolerance"]


def test_a_bitmap_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="square and at least 3 x 3"):
        zncc_self_similarity_parts(np.zeros((28, 24), np.uint8))
    with pytest.raises(ValueError, match="uint8 or float32"):
        zncc_self_similarity_parts(np.zeros((R, R), np.float64))
    with pytest.raises(ValueError, match="1 to 4 channels"):
        zncc_self_similarity_parts(np.zeros((R, R, 5), np.uint8))
