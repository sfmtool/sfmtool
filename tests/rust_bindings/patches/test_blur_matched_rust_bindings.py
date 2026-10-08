# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the blur-matching bindings: one tile's ``assess_blur``,
``blur_sigma_to_reach`` and ``blur_to_length``, and
``blur_matched_zncc_matrix``, the ZNCC between every pair of a track's views'
tiles, each pair blur-matched. See ``specs/core/patch/blur-matched-zncc.md``."""

import numpy as np
import pytest

from sfmtool._sfmtool.patches import (
    assess_blur,
    blur_matched_zncc_matrix,
    blur_sigma_to_reach,
    blur_to_length,
)


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


def test_assess_blur_keeps_both_axes_of_the_tile_and_each_probe():
    tile = _texture()
    a = assess_blur(tile)
    assert a["semi_axes"].shape == (2,)
    assert a["growth"].shape == (2, 2)
    np.testing.assert_allclose(a["probe_sigmas"], [0.4, 1.0])
    # The semi-axes are the square roots of the ellipse's eigenvalues.
    eig = np.sqrt(np.linalg.eigvalsh(a["ellipse_matrix"]))[::-1]
    np.testing.assert_allclose(a["semi_axes"], eig, rtol=1e-9)
    # Blurring lengthens both axes, more under the wider probe.
    assert np.all(a["growth"][0] > a["semi_axes"])
    assert np.all(a["growth"][1] > a["growth"][0])
    # The ellipse passed in is the one the assessment is set against.
    given = assess_blur(tile, ellipse=a["ellipse_matrix"])
    for key in ("semi_axes", "growth"):
        np.testing.assert_array_equal(given[key], a[key])
    # An ellipse that cannot be read gives no assessment.
    assert assess_blur(tile, ellipse=np.full((2, 2), np.nan)) is None


def test_blur_to_length_brings_the_tile_to_the_length_asked():
    sharp = _texture()
    blurry = _texture(blur=2.0)
    a = assess_blur(sharp)
    length = assess_blur(blurry)["semi_axes"][1]
    sigma = blur_sigma_to_reach(a, length)
    out = blur_to_length(sharp, a, length)
    assert out["sigma"] == sigma > 0
    assert out["samples"].shape == (24, 24, 3)
    assert out["samples"].dtype == np.float32
    # Read again, the blurred tile's semi-major axis is near the length.
    blurred = np.concatenate(
        [np.clip(np.round(out["samples"]), 0, 255), np.full((24, 24, 1), 255)],
        axis=-1,
    ).astype(np.uint8)
    major = assess_blur(blurred)["semi_axes"][0]
    assert major == pytest.approx(length, rel=0.1)


def test_blur_to_length_leaves_a_long_enough_tile_unblurred():
    tile = _texture(blur=1.0)
    a = assess_blur(tile)
    for length in (a["semi_axes"][0], 0.5 * a["semi_axes"][0]):
        assert blur_sigma_to_reach(a, length) == 0
        out = blur_to_length(tile, a, length)
        assert out["sigma"] == 0
        np.testing.assert_array_equal(out["samples"], tile[..., :3].astype(np.float32))
    # An assessment that does not grow gives no width.
    flat = {"semi_axes": a["semi_axes"], "growth": np.stack([a["semi_axes"]] * 2)}
    assert blur_sigma_to_reach(flat, 2 * a["semi_axes"][0]) is None
    assert blur_to_length(tile, flat, 2 * a["semi_axes"][0]) is None


def test_the_matrix_blurs_by_the_width_of_the_sharper_tiles_assessment():
    tiles = np.stack([_texture(), _texture(blur=2.0)])
    out = blur_matched_zncc_matrix(tiles)
    assert out["blurred"][0, 1]
    a = assess_blur(tiles[0])
    target = min(assess_blur(tiles[1])["semi_axes"][1], 2.0)
    assert out["blur_sigma"][0, 1] == blur_sigma_to_reach(a, target)


def test_bad_tile_arguments_are_refused():
    tile = _texture()
    with pytest.raises(ValueError, match="square"):
        assess_blur(np.zeros((24, 20, 4), np.uint8))
    with pytest.raises(ValueError, match="valid"):
        assess_blur(tile, valid=np.ones((24, 23), bool))
    with pytest.raises(ValueError, match="ellipse"):
        assess_blur(tile, ellipse=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="growth"):
        blur_sigma_to_reach({"semi_axes": [0.5, 0.4]}, 1.0)
    with pytest.raises(ValueError, match="assessment"):
        blur_to_length(tile, {"semi_axes": [0.5], "growth": [[1, 1], [2, 2]]}, 1.0)


@pytest.mark.parametrize("length", [-0.5, float("nan"), float("inf")])
def test_a_length_that_is_negative_or_not_finite_is_refused(length):
    tile = _texture()
    a = assess_blur(tile)
    with pytest.raises(ValueError, match="length"):
        blur_sigma_to_reach(a, length)
    with pytest.raises(ValueError, match="length"):
        blur_to_length(tile, a, length)


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
    # Only the sharper tile of a pair is blurred, and its width is reported
    # in its own row.
    sigma = matched["blur_sigma"]
    assert sigma.shape == (3, 3)
    assert sigma[0, 2] > 0 and sigma[1, 2] > 0
    assert sigma[2, 0] == 0 and sigma[2, 1] == 0
    assert np.all(plain["blur_sigma"] == 0)


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


def test_samples_without_data_are_left_out():
    tile = _texture()
    holed = tile.copy()
    holed[:, :6, 3] = 0
    holed[:, :6, :3] = 0
    out = blur_matched_zncc_matrix(np.stack([tile, holed]), matching="plain")
    # Over the samples both carry, the two are the same.
    assert out["zncc"][0, 1] == pytest.approx(1.0)


def test_valid_flags_samples_without_data():
    tile = _texture()
    holed = tile.copy()
    holed[:, :6, :3] = 0
    valid = np.ones((2, 24, 24), bool)
    valid[1, :, :6] = False
    out = blur_matched_zncc_matrix(
        np.stack([tile, holed]), valid=valid, matching="plain"
    )
    assert out["zncc"][0, 1] == pytest.approx(1.0)
    # Without the flags, the black columns are read as texture.
    unflagged = blur_matched_zncc_matrix(np.stack([tile, holed]), matching="plain")
    assert unflagged["zncc"][0, 1] < 0.99


def test_grey_and_alpha_read_the_grey_alone():
    grey = _texture()[..., :1]
    blurred = _texture(blur=1.5)[..., :1]
    tiles = np.stack([grey, blurred])
    with_alpha = np.concatenate([tiles, np.full_like(tiles, 255)], axis=-1)
    # Alpha is not a colour channel: two channels read as one.
    for matching in ("plain", "blur_matched"):
        np.testing.assert_allclose(
            blur_matched_zncc_matrix(with_alpha, matching=matching)["zncc"],
            blur_matched_zncc_matrix(tiles, matching=matching)["zncc"],
        )
    # Alpha 0 marks a sample without data.
    holed = with_alpha.copy()
    holed[1, :, :6, 0] = 0
    holed[1, :, :6, 1] = 0
    out = blur_matched_zncc_matrix(
        np.stack([with_alpha[0], holed[1]]), matching="plain"
    )
    expected = blur_matched_zncc_matrix(
        np.stack([with_alpha[0], with_alpha[1]]),
        valid=np.stack([np.ones((24, 24), bool), holed[1, ..., 1] > 0]),
        matching="plain",
    )
    assert out["zncc"][0, 1] == pytest.approx(expected["zncc"][0, 1])


def test_bad_arguments_are_refused():
    tiles = np.stack([_texture(), _texture()])
    with pytest.raises(ValueError, match="matching"):
        blur_matched_zncc_matrix(tiles, matching="sharp")
    with pytest.raises(ValueError, match="ellipses"):
        blur_matched_zncc_matrix(tiles, ellipses=np.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="square"):
        blur_matched_zncc_matrix(np.zeros((2, 24, 20, 4), np.uint8))
    with pytest.raises(ValueError, match="valid"):
        blur_matched_zncc_matrix(tiles, valid=np.ones((2, 24, 23), bool))
    with pytest.raises(ValueError, match="min_ellipse_ratio"):
        blur_matched_zncc_matrix(tiles, min_ellipse_ratio=0.5)
