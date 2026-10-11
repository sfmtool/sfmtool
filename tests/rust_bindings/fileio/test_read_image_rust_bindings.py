# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the image readers ``sfmtool.fileio.read_image_rgb`` and
``read_image_rgba``, the Rust decoder that Python reads photographs with."""

import base64
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from sfmtool.fileio import (
    image_dimensions,
    image_has_alpha,
    read_image_rgb,
    read_image_rgba,
)

_DATA = Path(__file__).resolve().parents[3] / "test-data" / "images"


def _seoul_bull_images() -> list[Path]:
    return sorted((_DATA / "seoul_bull_sculpture").glob("*.jpg"))


def test_rgb_layout_is_y_x_rgb_uint8_c_contiguous():
    path = _seoul_bull_images()[0]
    rgb = read_image_rgb(path)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (480, 270, 3)
    assert rgb.flags.c_contiguous
    # Same orientation and channel order as OpenCV's decode, converted to RGB:
    # the two decoders differ by a few grey levels on a few values only.
    cv_rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    diff = np.abs(rgb.astype(np.int16) - cv_rgb)
    assert diff.max() <= 8
    assert (diff > 0).mean() < 0.1


def test_workspace_image_reader_is_the_rust_decoder():
    from sfmtool._workspace_image import read_workspace_image

    workspace = _DATA / "seoul_bull_sculpture"
    for path in _seoul_bull_images():
        expected = read_image_rgb(path)
        actual = read_workspace_image(workspace, path.name)
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)


def test_rgba_of_a_jpeg_is_opaque_with_the_rgb_colour():
    path = _seoul_bull_images()[0]
    rgba = read_image_rgba(path)
    assert rgba.dtype == np.uint8
    assert rgba.shape == (480, 270, 4)
    assert rgba.flags.c_contiguous
    assert (rgba[..., 3] == 255).all()
    np.testing.assert_array_equal(rgba[..., :3], read_image_rgb(path))


def test_rgba_keeps_a_png_alpha(tmp_path):
    rng = np.random.default_rng(3)
    rgba = rng.integers(0, 256, size=(5, 7, 4), dtype=np.uint8)
    rgba[0, 0, 3] = 0
    rgba[1, 2, 3] = 255
    path = tmp_path / "alpha.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    np.testing.assert_array_equal(read_image_rgba(path), rgba)
    np.testing.assert_array_equal(read_image_rgb(path), rgba[..., :3])
    assert image_has_alpha(path)
    assert not image_has_alpha(_seoul_bull_images()[0])


def test_the_contents_not_the_extension_choose_the_decoder(tmp_path):
    rgba = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    assert ok
    path = tmp_path / "actually_png.jpg"
    path.write_bytes(encoded.tobytes())

    assert image_has_alpha(path)
    assert image_dimensions(path) == (3, 2)
    np.testing.assert_array_equal(read_image_rgba(path), rgba)
    np.testing.assert_array_equal(read_image_rgb(path), rgba[..., :3])


def _with_exif_orientation(jpeg: bytes, orientation: int) -> bytes:
    """``jpeg`` with an APP1 Exif segment holding only an orientation tag."""
    tiff = (
        b"MM\x00\x2a\x00\x00\x00\x08"  # big-endian TIFF header, IFD at 8
        + struct.pack(">H", 1)  # one entry
        + struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)  # Orientation
        + struct.pack(">I", 0)  # no next IFD
    )
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    assert jpeg[:2] == b"\xff\xd8"
    return jpeg[:2] + app1 + jpeg[2:]


def test_exif_orientation_is_ignored(tmp_path):
    # A wide image whose top-left corner is bright: a rotation would change
    # both the shape and where the bright corner is.
    bgr = np.zeros((40, 64, 3), dtype=np.uint8)
    bgr[:10, :10] = 255
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    plain = tmp_path / "plain.jpg"
    tagged = tmp_path / "rotated_tag.jpg"
    plain.write_bytes(encoded.tobytes())
    tagged.write_bytes(_with_exif_orientation(encoded.tobytes(), 6))

    # The tag is one OpenCV honours by default, so the file really is tagged.
    assert cv2.imread(str(tagged), cv2.IMREAD_COLOR).shape[:2] == (64, 40)

    rgb = read_image_rgb(tagged)
    assert rgb.shape == (40, 64, 3)
    np.testing.assert_array_equal(rgb, read_image_rgb(plain))
    np.testing.assert_array_equal(read_image_rgba(tagged)[..., :3], rgb)

    from sfmtool._workspace_image import read_workspace_image

    np.testing.assert_array_equal(read_workspace_image(tmp_path, tagged.name), rgb)


# A 37 x 23 baseline JPEG at 4:2:0 with each component in a scan of its own,
# written by the `jpeg-encoder` crate with its optimized Huffman tables on.
# zune-jpeg 0.5.15, the `image` crate's JPEG decoder, decodes it to wrong
# pixels without an error.
_ONE_SCAN_PER_COMPONENT_JPEG = """
/9j/4AAQSkZJRgABAgAAAQABAAD/wAARCAAXACUDACIAAREBAhEB/9sAQwAFAwQEBAMFBAQEBQUF
BgcMCAcHBwcPCwsJDBEPEhIRDxERExYcFxMUGhURERghGBodHR8fHxMXIiQiHiQcHh8e/9sAQwEF
BQUHBgcOCAgOHhQRFB4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e
Hh4eHh4e/8QAFgABAQEAAAAAAAAAAAAAAAAABgcI/8QALBABAAECBAIJBQEAAAAAAAAAAQIDEQAF
EiExQQQTIlFhcYGR8DJSobHB4f/EABcBAQEBAQAAAAAAAAAAAAAAAAYHBAj/xAAlEQAABAUEAgMA
AAAAAAAAAAAAAQISAwQFEVEGIjGBIZGhweH/2gAIAQAAAD8AlOThpY2tvux8PLD/ACqjLTFPueKW
sH+YV5XSdV2Ldvq3bg7n9/GHmT0urDrLdl47d42/t8J8tjGdG850g4GsX94yZlFBagqRsg7WOW78
9sK8ogQkJF0x7UuW/wAvh9lFBlAUbxC6HjuPfz/OEmUUo2NEpfUS2AA7/HiYoWX0UohOkoBptFT0
tjG+TUY6yKp2RdLy4fPPDzKqOtYz1RIu6Wtwvw9MJ8ipydLANOw3C1+Xjz93FByekdZHS3LltO2z
zv5W78JuidFqVOjxYMZot5MT9Pr74//aAAgBAREAPwDFUtMnIck53XHvICVLTJyHJOd1a3vIKr0K
px7vj9HaRSkFD7FgBilIKX2LH2AK5CA4/A//2gAIAQIRAD8AudVjOuJrVYzrgREmtxi31hRia1hR
gZEUbjH/2Q==
"""


def test_a_jpeg_with_one_scan_per_component_reads_as_opencv_reads_it(tmp_path):
    jpeg = base64.b64decode(_ONE_SCAN_PER_COMPONENT_JPEG)
    assert jpeg.count(b"\xff\xda") == 3  # three start-of-scan markers
    path = tmp_path / "scans.jpg"
    path.write_bytes(jpeg)

    rgb = read_image_rgb(path)
    assert rgb.shape == (23, 37, 3)
    cv_rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    diff = np.abs(rgb.astype(np.int16) - cv_rgb)
    assert diff.mean() < 1.0
    np.testing.assert_array_equal(read_image_rgba(path)[..., :3], rgb)
    assert not image_has_alpha(path)


def test_a_missing_file_raises_file_not_found_naming_the_path(tmp_path):
    missing = tmp_path / "missing.jpg"
    for reader in (read_image_rgb, read_image_rgba, image_has_alpha):
        with pytest.raises(FileNotFoundError, match="missing.jpg"):
            reader(missing)


def test_an_undecodable_file_raises_oserror_naming_the_path(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    for reader in (read_image_rgb, read_image_rgba, image_has_alpha):
        with pytest.raises(OSError, match="bad.jpg"):
            reader(bad)
