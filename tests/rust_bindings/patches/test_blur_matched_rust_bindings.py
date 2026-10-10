# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the blur-matching bindings: one tile's ``assess_blur``,
``blur_sigma_to_reach`` and ``blur_to_length``, and ``score_against_bitmap``,
each observation's ZNCC with a point's stored bitmap, plain and with the
bitmap alone blurred. See ``specs/core/patch/blur-matched-zncc.md``."""

import numpy as np
import pytest

from sfmtool.patches import (
    assess_blur,
    blur_sigma_to_reach,
    blur_to_length,
    score_against_bitmap,
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


_SCALAR_KEYS = (
    "plain_zncc",
    "plain_zncc_middle",
    "blur_matched_zncc",
    "blur_matched_zncc_middle",
)
_GRID_KEYS = ("plain_zncc_grid", "blur_matched_zncc_grid")


def _assert_shapes(out, k):
    """One score per tile for each whole and middle reading, and one 3x3 grid
    per tile for each grid reading; no bare ``zncc`` key."""
    assert "zncc" not in out
    for key in _SCALAR_KEYS:
        assert out[key].shape == (k,), key
    for key in _GRID_KEYS:
        assert out[key].shape == (k, 3, 3), key


def _assert_read_plain(out, i):
    """Row ``i``'s bitmap was not blurred, so its blur-matched readings are
    its plain ones."""
    assert out["blur_sigma"][i] == 0
    assert out["blur_matched_zncc"][i] == out["plain_zncc"][i]
    assert out["blur_matched_zncc_middle"][i] == out["plain_zncc_middle"][i]
    np.testing.assert_array_equal(
        out["blur_matched_zncc_grid"][i], out["plain_zncc_grid"][i]
    )


def test_the_bitmap_alone_is_blurred_to_a_blurrier_observation():
    bitmap = _texture()
    tiles = np.stack([_texture(), _texture(blur=2.0)])
    out = score_against_bitmap(bitmap, tiles)
    _assert_shapes(out, 2)
    # The same tile as the bitmap reads 1, plain.
    assert out["plain_zncc"][0] == pytest.approx(1.0)
    assert out["plain_zncc_middle"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(out["plain_zncc_grid"][0], np.ones((3, 3)))
    _assert_read_plain(out, 0)
    # The blurrier observation has the bitmap blurred by the width the
    # bitmap's own assessment gives to reach its semi-minor axis.
    a = assess_blur(bitmap)
    target = min(assess_blur(tiles[1])["semi_axes"][1], 2.0)
    assert out["blur_sigma"][1] == blur_sigma_to_reach(a, target)
    assert out["blur_matched_zncc"][1] > out["plain_zncc"][1] + 0.01
    # The middle is read against the same blurred bitmap, and gains with it.
    assert out["blur_matched_zncc_middle"][1] > out["plain_zncc_middle"][1]
    assert not out["sharper_than_bitmap"].any()
    np.testing.assert_allclose(out["bitmap_semi_axes"], a["semi_axes"])


def test_an_observation_sharper_than_the_bitmap_is_read_plain():
    bitmap = _texture(blur=2.0)
    out = score_against_bitmap(bitmap, np.stack([_texture()]))
    _assert_shapes(out, 1)
    assert out["sharper_than_bitmap"][0]
    _assert_read_plain(out, 0)


def test_the_reference_is_not_computed():
    bitmap = _texture()
    tiles = np.stack([_texture(blur=1.5), bitmap])
    out = score_against_bitmap(bitmap, tiles, reference=1)
    _assert_shapes(out, 2)
    for key in _SCALAR_KEYS:
        assert out[key][1] == 1.0, key
    for key in _GRID_KEYS:
        np.testing.assert_array_equal(out[key][1], np.ones((3, 3)))
    assert out["blur_sigma"][1] == 0 and not out["sharper_than_bitmap"][1]
    alone = score_against_bitmap(bitmap, tiles[:1])
    for key in _SCALAR_KEYS + _GRID_KEYS:
        np.testing.assert_array_equal(out[key][0], alone[key][0])


def test_samples_without_data_are_left_out():
    bitmap = _texture()
    holed = bitmap.copy()
    holed[:, :6, :3] = 0
    holed[:, :6, 3] = 0
    out = score_against_bitmap(bitmap, np.stack([holed]))
    assert out["plain_zncc"][0] == pytest.approx(1.0)
    # The same through the valid flags.
    flagged = bitmap.copy()
    flagged[:, :6, :3] = 0
    valid = np.ones((1, 24, 24), bool)
    valid[0, :, :6] = False
    out = score_against_bitmap(bitmap, np.stack([flagged]), valid=valid)
    assert out["plain_zncc"][0] == pytest.approx(1.0)
    unflagged = score_against_bitmap(bitmap, np.stack([flagged]))
    assert unflagged["plain_zncc"][0] < 0.99


def test_bad_bitmap_arguments_are_refused():
    bitmap = _texture()
    tiles = np.stack([_texture()])
    with pytest.raises(ValueError, match="RGBA"):
        score_against_bitmap(bitmap[..., :3], tiles)
    with pytest.raises(ValueError, match="like the bitmap"):
        score_against_bitmap(bitmap, np.zeros((1, 20, 20, 4), np.uint8))
    with pytest.raises(ValueError, match="reference"):
        score_against_bitmap(bitmap, tiles, reference=1)
    with pytest.raises(ValueError, match="valid"):
        score_against_bitmap(bitmap, tiles, valid=np.ones((1, 24, 23), bool))
