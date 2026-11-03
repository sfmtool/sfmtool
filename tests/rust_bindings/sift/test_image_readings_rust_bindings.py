# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.sift.ImageToGray` and `image_readings_from_rgb` against an
independent numpy reference.

The reference below recomputes the reading image and both readings from the
definitions in `specs/formats/sift-file-format.md`: the grey formula evaluated
in numpy, the rounding to `quantize_bits` written out, the size rule, an exact
area average as a pair of dense weight matrices, and the Laplacian and Sobel
kernels as explicit sums of shifted slices. It calls nothing from sfmtool.
"""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.fileio import read_image_rgb
from sfmtool.sift import ImageReadings, ImageToGray, image_readings_from_rgb

TEST_DATA = Path(__file__).resolve().parents[3] / "test-data" / "images"
BT709 = "0.2126*R + 0.7152*G + 0.0722*B"
BT601 = "0.299*R + 0.587*G + 0.114*B"


# ── The reference ─────────────────────────────────────────────────────────


def _formula_value(formula, rgb):
    """The formula, for the few formulas these tests use, in double precision
    on [0, 1] channels, in the same order of operations as the formula text."""
    r, g, b = (rgb[..., c].astype(np.float64) / 255.0 for c in range(3))
    if formula == BT709:
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    if formula == BT601:
        return 0.299 * r + 0.587 * g + 0.114 * b
    if formula == "G":
        return g
    if formula == "0.2126*R**2.2 + 0.7152*G**2.2 + 0.0722*B**2.2":
        return 0.2126 * r**2.2 + 0.7152 * g**2.2 + 0.0722 * b**2.2
    raise AssertionError(f"no reference for {formula!r}")


def _quantize(v, bits):
    """round(clamp(v, 0, 1) * (2^b - 1)) / (2^b - 1), half rounded up."""
    levels = float(2**bits - 1)
    x = np.clip(v, 0.0, 1.0) * levels
    whole = np.floor(x)
    whole += (x - whole) >= 0.5
    return whole / levels


def _reading_size(w, h):
    long, short = max(w, h), min(w, h)
    if long <= 480:
        return w, h
    s = max(1, (2 * short * 480 + long) // (2 * long))
    return (480, s) if w >= h else (s, 480)


def _area_matrix(n, m):
    """The (m, n) matrix whose row d holds the area weights of output d: the
    overlap of [d n, (d+1) n) with [s m, (s+1) m), divided by n."""
    d = np.arange(m, dtype=np.int64)[:, None]
    s = np.arange(n, dtype=np.int64)[None, :]
    overlap = np.minimum((d + 1) * n, (s + 1) * m) - np.maximum(d * n, s * m)
    return np.clip(overlap, 0, None).astype(np.float64) / float(n)


def reference_readings(rgb, formula, quantize_bits=None):
    h, w, _ = rgb.shape
    grey = _formula_value(formula, rgb)
    if quantize_bits is not None:
        grey = _quantize(grey, quantize_bits)
    ow, oh = _reading_size(w, h)
    if ow < 3 or oh < 3:
        return 0.0, (0.0, 0.0, 0.0)
    image = _area_matrix(h, oh) @ grey @ _area_matrix(w, ow).T

    c = image[1:-1, 1:-1]
    left, right = image[1:-1, :-2], image[1:-1, 2:]
    up, down = image[:-2, 1:-1], image[2:, 1:-1]
    up_left, up_right = image[:-2, :-2], image[:-2, 2:]
    down_left, down_right = image[2:, :-2], image[2:, 2:]

    laplacian = left + right + up + down - 4.0 * c
    gx = (up_right + 2.0 * right + down_right - up_left - 2.0 * left - down_left) / 8.0
    gy = (down_left + 2.0 * down + down_right - up_left - 2.0 * up - up_right) / 8.0

    mean = laplacian.mean()
    sharpness = ((laplacian - mean) ** 2).mean()
    return sharpness, ((gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean())


def _assert_agrees(rgb, to_gray, rel=1e-12):
    got = image_readings_from_rgb(rgb, to_gray)
    sharpness, tensor = reference_readings(rgb, to_gray.formula, to_gray.quantize_bits)
    assert got.sharpness == pytest.approx(sharpness, rel=rel, abs=1e-300)
    # Exy can be near zero beside Exx and Eyy, so it is held to the tensor's
    # own scale.
    scale = max(tensor[0], tensor[1])
    for g, r in zip(got.gradient_structure_tensor_xx_yy_xy, tensor):
        assert abs(g - r) <= rel * scale, (got, tensor)
    return got


# ── Agreement with the reference ──────────────────────────────────────────


@pytest.mark.parametrize("number", [1, 7, 13])
def test_seoul_bull_images_match_the_reference(number):
    path = TEST_DATA / "seoul_bull_sculpture" / f"seoul_bull_sculpture_{number:02d}.jpg"
    rgb = read_image_rgb(path)
    assert rgb.shape == (480, 270, 3)  # read at its own size
    got = _assert_agrees(rgb, ImageToGray(BT709))
    assert got.sharpness > 0
    _assert_agrees(rgb, ImageToGray(BT709, quantize_bits=8))
    _assert_agrees(rgb, ImageToGray(BT601, quantize_bits=8))


def test_a_reduced_photograph_matches_the_reference():
    # 2040 x 1536 reduces by 4.25 to 480 x 361: fractional overlaps.
    rgb = read_image_rgb(TEST_DATA / "dino_dog_toy" / "dino_dog_toy_01.jpg")
    assert rgb.shape == (1536, 2040, 3)
    _assert_agrees(rgb, ImageToGray(BT709))
    _assert_agrees(rgb, ImageToGray("0.2126*R**2.2 + 0.7152*G**2.2 + 0.0722*B**2.2"))


@pytest.mark.parametrize(
    ("w", "h"),
    [(1283, 721), (129, 641), (641, 129), (961, 961), (300, 200), (960, 5), (4, 3)],
)
def test_synthetic_images_match_the_reference(w, h):
    rng = np.random.default_rng(w * 1000 + h)
    # Smooth structure plus noise, so the readings are not dominated by either.
    yy, xx = np.mgrid[0:h, 0:w]
    base = 127 + 80 * np.sin(xx / 17.0) * np.cos(yy / 11.0)
    rgb = np.clip(base[..., None] + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(
        np.uint8
    )
    _assert_agrees(rgb, ImageToGray(BT709))
    _assert_agrees(rgb, ImageToGray("G", quantize_bits=5))


def test_the_reference_size_rule_is_the_bindings():
    # A 960 x 5 image reduces to 480 x 3 (2.5 rounded up), so it has an
    # interior; at 960 x 3 it is 480 x 2 and reads 0.
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (5, 960, 3), dtype=np.uint8)
    assert image_readings_from_rgb(rgb, ImageToGray(BT709)).sharpness > 0
    rgb = rng.integers(0, 256, (3, 960, 3), dtype=np.uint8)
    flat = image_readings_from_rgb(rgb, ImageToGray(BT709))
    assert flat.sharpness == 0.0
    assert flat.gradient_structure_tensor_xx_yy_xy == (0.0, 0.0, 0.0)


def test_a_non_contiguous_array_reads_as_its_copy():
    rng = np.random.default_rng(5)
    big = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    view = big[::2, 10:250]
    assert not view.flags["C_CONTIGUOUS"]
    a = image_readings_from_rgb(view, ImageToGray(BT709))
    b = image_readings_from_rgb(np.ascontiguousarray(view), ImageToGray(BT709))
    assert a.sharpness == b.sharpness
    assert a.gradient_structure_tensor_xx_yy_xy == b.gradient_structure_tensor_xx_yy_xy


# ── ImageToGray ───────────────────────────────────────────────────────────


def test_image_to_gray_from_dict_and_json():
    d = {"formula": BT709, "quantize_bits": 8}
    a = ImageToGray.from_json(d)
    b = ImageToGray.from_json('{"formula": "' + BT709 + '", "quantize_bits": 8}')
    assert a == b == ImageToGray(BT709, quantize_bits=8)
    assert a.formula == BT709 and a.quantize_bits == 8
    assert a.to_dict() == d
    plain = ImageToGray.from_gray_formula("G")
    assert plain.quantize_bits is None
    assert plain.to_dict() == {"formula": "G"}
    assert plain != ImageToGray("G", quantize_bits=8)
    assert plain.gray(0.1, 0.7, 0.3) == 0.7
    # Half up: 0.5 with 1 bit is 1, where half to even would give 0.
    assert ImageToGray("G", quantize_bits=1).gray(0.0, 0.5, 0.0) == 1.0
    assert "quantize_bits=8" in repr(a)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"formula": "R - G"}, "does not parse"),
        ({"formula": "G", "quantize_bits": 0}, "1 to 16"),
        ({"formula": "G", "quantize_bits": 17}, "1 to 16"),
        ({"formula": "G", "quantize_bits": 8.5}, "1 to 16"),
        ({"formula": "G", "quantize_bits": None}, "1 to 16"),
        ({"quantize_bits": 8}, "no 'formula'"),
        ({"formula": 3}, "must be a string"),
        ({"formula": "G", "gamma": 2.2}, "unknown key"),
        ([1, 2], "JSON object"),
        ("{not json", "not valid JSON"),
    ],
)
def test_image_to_gray_refusals(value, message):
    with pytest.raises(ValueError, match=message):
        ImageToGray.from_json(value)


def test_image_to_gray_constructor_refusals():
    with pytest.raises(ValueError, match="does not parse"):
        ImageToGray("R +")
    for bits in (0, 17, -1, 300):
        with pytest.raises(ValueError, match="1 to 16"):
            ImageToGray("G", quantize_bits=bits)


# ── Errors ────────────────────────────────────────────────────────────────


def test_a_non_finite_reading_raises_naming_the_conversion():
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    rgb[:, 4, 0] = 0
    with pytest.raises(ValueError, match=r"R \*\* -1"):
        image_readings_from_rgb(rgb, ImageToGray("R ** -1"))
    # Rounding clamps the infinity to 1.
    got = image_readings_from_rgb(rgb, ImageToGray("R ** -1", quantize_bits=8))
    assert isinstance(got, ImageReadings)


def test_a_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        image_readings_from_rgb(np.zeros((4, 4, 4), np.uint8), ImageToGray("G"))
    with pytest.raises(TypeError):
        image_readings_from_rgb(np.zeros((4, 4), np.uint8), ImageToGray("G"))
