# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``zncc_self_similarity_parts`` PyO3 binding."""

import numpy as np
import pytest

from sfmtool._sfmtool.patches import zncc_self_similarity_parts

R = 24
SIDE = R + 2 * 3


def _edge(angle_deg: float, size: int = SIDE) -> np.ndarray:
    """A straight edge through the tile centre along ``angle_deg``, 150 grey
    levels high, with a one-pixel linear ramp across it."""
    s, c = np.sin(np.radians(angle_deg)), np.cos(np.radians(angle_deg))
    mid = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    across = -(x - mid) * s + (y - mid) * c
    return (50.0 + 150.0 * np.clip(0.5 + across, 0.0, 1.0)).astype(np.float32)


def test_the_result_carries_every_part_in_its_shape():
    rng = np.random.default_rng(1)
    tile = rng.integers(0, 256, size=(SIDE, SIDE, 3), dtype=np.uint8)
    out = zncc_self_similarity_parts(tile, R)
    assert isinstance(out["radius"], float)
    assert isinstance(out["radius_middle"], float)
    assert isinstance(out["tolerance"], float)
    assert out["radius_grid"].shape == (3, 3)
    assert out["radius_grid"].dtype == np.float64
    assert out["slide"].shape == (2,)
    assert out["slide_grid"].shape == (3, 3, 2)
    assert out["slide_grid"].dtype == np.float64
    assert out["surface"].shape == (7, 7)
    assert out["surface"][3, 3] == 1.0
    # Outside the disk of radius 3, there is no shift.
    assert np.isnan(out["surface"][0, 0]) and np.isnan(out["surface"][6, 6])
    # Random texture pins its position: its surface falls through the level
    # within a pixel.
    assert 0.0 < out["radius"] < 1.0


def test_an_edge_slides_along_itself_and_a_corner_does_not():
    out = zncc_self_similarity_parts(_edge(0.0), R)
    assert out["radius"] == 3.0
    assert abs(out["slide"][0]) > 0.99 and abs(out["slide"][1]) < 1e-9

    y, x = np.mgrid[0:SIDE, 0:SIDE]
    corner = np.where((x >= SIDE // 2) & (y >= SIDE // 2), 200, 50).astype(np.uint8)
    out = zncc_self_similarity_parts(corner, R)
    assert out["radius"] < 1.0
    np.testing.assert_array_equal(out["slide"], [0.0, 0.0])


def test_a_flat_tile_scores_the_maximum_and_has_no_surface():
    out = zncc_self_similarity_parts(np.full((SIDE, SIDE), 128, np.uint8), R)
    assert out["radius"] == 3.0
    assert out["radius_middle"] == 3.0
    assert (out["radius_grid"] == 3.0).all()
    assert out["tolerance"] == np.inf
    assert np.isnan(out["surface"]).all()


def test_uint8_float32_and_alpha_read_alike():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, size=(SIDE, SIDE, 3), dtype=np.uint8)
    rgba = np.concatenate(
        [rgb, rng.integers(0, 256, size=(SIDE, SIDE, 1), dtype=np.uint8)], axis=2
    )
    a = zncc_self_similarity_parts(rgb, R)
    b = zncc_self_similarity_parts(rgb.astype(np.float32), R)
    c = zncc_self_similarity_parts(rgba, R)
    for other in (b, c):
        assert other["radius"] == a["radius"]
        np.testing.assert_array_equal(other["radius_grid"], a["radius_grid"])
        np.testing.assert_allclose(other["surface"], a["surface"], atol=1e-12)


def test_the_parameters_are_keywords():
    tile = _edge(90.0, size=R + 4)
    out = zncc_self_similarity_parts(
        tile, R, max_radius=2, relative_tolerance=0.1, noise=1.0
    )
    assert out["radius"] == 2.0
    assert out["surface"].shape == (5, 5)
    # A larger noise raises the tolerance.
    loud = zncc_self_similarity_parts(tile, R, max_radius=2, noise=20.0)
    assert loud["tolerance"] > out["tolerance"]


def test_a_tile_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="needs a 30 x 30 tile"):
        zncc_self_similarity_parts(np.zeros((28, 28), np.uint8), R)
    with pytest.raises(ValueError, match="uint8 or float32"):
        zncc_self_similarity_parts(np.zeros((SIDE, SIDE), np.float64), R)
    with pytest.raises(ValueError, match="1 to 4 channels"):
        zncc_self_similarity_parts(np.zeros((SIDE, SIDE, 5), np.uint8), R)
