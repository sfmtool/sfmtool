# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``blur_matched_zncc_matrix``: the ZNCC between every pair of a
track's views' tiles, each pair blur-matched. See
``specs/core/patch/blur-matched-zncc.md``."""

import numpy as np
import pytest

from sfmtool._sfmtool.patches import blur_matched_zncc_matrix


def _texture(side=24, blur=0.0, seed=3):
    """A smooth three-channel texture, blurred by ``blur`` grid px."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:side, 0:side].astype(np.float64)
    out = np.zeros((side, side, 4))
    for c in range(3):
        v = np.full((side, side), 128.0)
        for _ in range(4):
            k = rng.uniform(-1.2, 1.2, size=2)
            amp = rng.uniform(15, 40)
            phase = rng.uniform(0, 2 * np.pi)
            gain = np.exp(-0.5 * blur * blur * (k @ k))
            v += amp * gain * np.sin(k[0] * x + k[1] * y + phase)
        out[..., c] = v
    out[..., 3] = 255
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def test_a_blurred_copy_reads_higher_blur_matched_than_plain():
    tiles = np.stack([_texture(), _texture(), _texture(blur=1.5)])
    plain = blur_matched_zncc_matrix(tiles, matching="plain")
    matched = blur_matched_zncc_matrix(tiles)
    assert plain["zncc"].shape == (3, 3)
    assert matched["zncc_grid"].shape == (3, 3, 3, 3)
    assert matched["ellipse_matrix"].shape == (3, 2, 2)
    np.testing.assert_allclose(np.diag(matched["zncc"]), 1.0)
    np.testing.assert_allclose(matched["zncc"], matched["zncc"].T)
    assert plain["pairs_blurred"] == 0
    assert matched["pairs"] == 3
    assert matched["blurred"][0, 2] and matched["blurred"][1, 2]
    for i in (0, 1):
        assert matched["zncc"][i, 2] > plain["zncc"][i, 2] + 0.01
    # The two sharp copies are the same tile: nothing to match.
    assert not matched["blurred"][0, 1]
    assert matched["zncc"][0, 1] == pytest.approx(1.0)


def test_a_ratio_and_given_ellipses_are_honoured():
    tiles = np.stack([_texture(), _texture(blur=1.0)])
    ellipses = blur_matched_zncc_matrix(tiles)["ellipse_matrix"]
    above = blur_matched_zncc_matrix(
        tiles,
        ellipses=ellipses,
        matching="blur_matched_above_ratio",
        min_ellipse_ratio=1e6,
    )
    assert above["pairs_blurred"] == 0
    # An ellipse that cannot be read leaves its pairs plain.
    unread = ellipses.copy()
    unread[1] = np.nan
    assert blur_matched_zncc_matrix(tiles, ellipses=unread)["pairs_blurred"] == 0
    ladder = blur_matched_zncc_matrix(tiles, kernel="isotropic_ladder")
    assert (
        ladder["zncc"][0, 1]
        > blur_matched_zncc_matrix(tiles, matching="plain")["zncc"][0, 1]
    )


def test_samples_without_data_are_left_out():
    tile = _texture()
    holed = tile.copy()
    holed[:, :6, 3] = 0
    holed[:, :6, :3] = 0
    out = blur_matched_zncc_matrix(np.stack([tile, holed]), matching="plain")
    # Over the samples both carry, the two are the same.
    assert out["zncc"][0, 1] == pytest.approx(1.0)


def test_bad_arguments_are_refused():
    tiles = np.stack([_texture(), _texture()])
    with pytest.raises(ValueError, match="kernel"):
        blur_matched_zncc_matrix(tiles, kernel="box")
    with pytest.raises(ValueError, match="matching"):
        blur_matched_zncc_matrix(tiles, matching="sharp")
    with pytest.raises(ValueError, match="ellipses"):
        blur_matched_zncc_matrix(tiles, ellipses=np.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="square"):
        blur_matched_zncc_matrix(np.zeros((2, 24, 20, 4), np.uint8))
