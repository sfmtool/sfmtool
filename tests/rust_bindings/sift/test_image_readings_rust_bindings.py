# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.sift.ImageToGray`, `gray_from_rgb` and `image_readings_from_rgb`
against independent numpy references, and `ImageReadings` from stored values
with its `axis_ratio` and `coherence`.

The readings reference below recomputes the reading image and both readings
from the definitions in `specs/formats/sift-file-format.md`, in double
precision: the grey formula evaluated in numpy, the rounding to
`quantize_bits` written out, the size rule, an exact area average as a pair of
dense weight matrices, and the Laplacian and Sobel kernels as explicit sums of
shifted slices. The format computes them in single precision, and states that
on photographs an evaluation in double precision agrees to within 1e-5 of each
reading (of Exx + Eyy for Exy), or 1e-12 absolute near 0, which is the
tolerance here; the test images are not made of colours at a tie of the
rounding to quantize_bits, where the two can differ by a level. The
grey reference evaluates the formula in single precision as the format
defines it, which `gray_from_rgb` matches bit for bit. Neither calls anything
from sfmtool.
"""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.fileio import read_image_rgb
from sfmtool.sift import (
    ImageReadings,
    ImageToGray,
    gray_from_rgb,
    image_readings_from_rgb,
)

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


def _gray_f32(formula, rgb, quantize_bits=None):
    """The grey samples as the format defines them: each channel the f32
    nearest v / 255, each literal the f32 nearest its text, each operation
    rounded to f32, in the formula's order; then the rounding to
    quantize_bits, with its product and quotient in f32."""
    f = np.float32
    r, g, b = (rgb[..., c].astype(f) / f(255) for c in range(3))
    if formula == BT709:
        v = (f("0.2126") * r + f("0.7152") * g) + f("0.0722") * b
    elif formula == BT601:
        v = (f("0.299") * r + f("0.587") * g) + f("0.114") * b
    elif formula == "G":
        v = g
    else:
        raise AssertionError(f"no reference for {formula!r}")
    assert v.dtype == f
    if quantize_bits is None:
        return v
    levels = f(2**quantize_bits - 1)
    x = np.clip(v, f(0), f(1)) * levels
    whole = np.floor(x)
    whole += (x - whole) >= f(0.5)
    return whole / levels


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


def _assert_agrees(rgb, to_gray, rel=1e-5, abs_=1e-12):
    got = image_readings_from_rgb(rgb, to_gray)
    sharpness, tensor = reference_readings(rgb, to_gray.formula, to_gray.quantize_bits)
    assert got.sharpness == pytest.approx(sharpness, rel=rel, abs=abs_)
    # Exy can be near zero beside Exx and Eyy, so it is held to the tensor's
    # own scale, Exx + Eyy.
    scale = tensor[0] + tensor[1]
    for g, r in zip(got.gradient_structure_tensor_xx_yy_xy, tensor):
        assert abs(g - r) <= rel * scale + abs_, (got, tensor)
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


# ── gray_from_rgb ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("formula", "bits"), [(BT709, None), (BT709, 8), (BT601, 8), ("G", 5)]
)
def test_gray_from_rgb_is_the_single_precision_definition(formula, bits):
    # Every 8-bit colour, as a 4096 x 4096 image: the binding's grey samples
    # are the format's single-precision ones, bit for bit.
    v = np.arange(1 << 24, dtype=np.uint32)
    rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1)
    rgb = rgb.astype(np.uint8).reshape(4096, 4096, 3)
    got = gray_from_rgb(rgb, ImageToGray(formula, quantize_bits=bits))
    assert got.dtype == np.float32 and got.shape == (4096, 4096)
    expected = _gray_f32(formula, rgb, bits)
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))


def test_gray_from_rgb_rounds_an_exact_tie_up():
    # 0.7152 * 14 + 0.0722 * 76 = 15.5 levels exactly: in single precision it
    # rounds up to 16, where the same formula in double precision reads 15.
    rgb = np.array([[[0, 14, 76]]], dtype=np.uint8)
    got = gray_from_rgb(rgb, ImageToGray(BT709, quantize_bits=8))
    assert got[0, 0] == np.float32(16) / np.float32(255)
    assert round(float(_formula_value(BT709, rgb)[0, 0]) * 255) == 15


def test_gray_from_rgb_refuses_a_wrong_shape():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        gray_from_rgb(np.zeros((4, 4, 4), np.uint8), ImageToGray("G"))


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
    # In single precision: the channel comes back as the f32 nearest 0.7.
    assert plain.gray(0.1, 0.7, 0.3) == float(np.float32(0.7))
    assert ImageToGray(BT709).gray(0.5, 0.25, 0.75) == float(
        np.float32("0.2126") * np.float32(0.5)
        + np.float32("0.7152") * np.float32(0.25)
        + np.float32("0.0722") * np.float32(0.75)
    )
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


# ── Stored readings and the values derived from them ──────────────────────


def test_image_readings_from_stored_values():
    readings = ImageReadings(0.0125, (0.004, 0.001, -0.001))
    assert readings.sharpness == np.float32(0.0125)
    assert readings.gradient_structure_tensor_xx_yy_xy == tuple(
        float(np.float32(v)) for v in (0.004, 0.001, -0.001)
    )
    # A list is taken as well as a tuple, as a .sift file's metadata holds it.
    assert ImageReadings(0.0125, [0.004, 0.001, -0.001]).coherence == (
        readings.coherence
    )


@pytest.mark.parametrize(
    "sharpness, tensor",
    [
        (-1.0, (0.004, 0.001, 0.001)),
        (float("nan"), (0.004, 0.001, 0.001)),
        (float("inf"), (0.004, 0.001, 0.001)),
        (1.0, (-0.004, 0.001, 0.001)),
        (1.0, (0.004, -0.001, 0.001)),
        (1.0, (float("nan"), 0.001, 0.001)),
        (1.0, (0.004, 0.001, float("-inf"))),
        # Too large for an f32, so infinite once stored as one.
        (1e39, (0.004, 0.001, 0.001)),
    ],
)
def test_image_readings_refuses_values_no_image_reads(sharpness, tensor):
    with pytest.raises(ValueError, match="must be finite"):
        ImageReadings(sharpness, tensor)


def test_image_readings_needs_three_tensor_values():
    with pytest.raises(ValueError, match="length 3"):
        ImageReadings(1.0, (0.004, 0.001))


@pytest.mark.parametrize(
    "tensor, axis_ratio, coherence",
    [
        ((0.004, 0.001, 0.001), np.log(4.0), np.sqrt(13.0) / 5.0),
        ((0.001, 0.004, -0.001), np.log(4.0), np.sqrt(13.0) / 5.0),
        ((0.002, 0.002, 0.0), 0.0, 0.0),
        ((0.004, 0.0, 0.0), np.inf, 1.0),
        ((0.0, 0.004, 0.0), np.inf, 1.0),
    ],
)
def test_axis_ratio_and_coherence(tensor, axis_ratio, coherence):
    readings = ImageReadings(1.0, tensor)
    assert readings.axis_ratio == pytest.approx(axis_ratio, rel=1e-6, abs=1e-7)
    assert readings.coherence == pytest.approx(coherence, rel=1e-6, abs=1e-7)


def test_a_flat_image_has_no_axis_ratio_and_zero_coherence():
    readings = ImageReadings(0.0, (0.0, 0.0, 0.0))
    assert readings.axis_ratio is None
    assert readings.coherence == 0.0
    measured = image_readings_from_rgb(
        np.full((32, 32, 3), 255, dtype=np.uint8), ImageToGray("G")
    )
    assert measured.axis_ratio is None
    assert measured.coherence == 0.0


def test_derived_values_of_measured_readings_match_their_definitions():
    """On a photograph, the derived values agree with the formulas evaluated
    in double precision on the stored tensor."""
    rgb = read_image_rgb(
        TEST_DATA / "seoul_bull_sculpture" / "seoul_bull_sculpture_01.jpg"
    )
    readings = image_readings_from_rgb(rgb, ImageToGray(BT709))
    exx, eyy, exy = readings.gradient_structure_tensor_xx_yy_xy
    assert exx > 0 and eyy > 0
    assert readings.axis_ratio == pytest.approx(abs(np.log(exx / eyy)), abs=1e-5)
    expected = np.sqrt((exx - eyy) ** 2 + 4 * exy**2) / (exx + eyy)
    assert readings.coherence == pytest.approx(expected, rel=1e-6)
