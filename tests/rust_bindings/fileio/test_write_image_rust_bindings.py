# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the image writers ``sfmtool.fileio.write_image_rgb`` and
``write_image_rgba``, the Rust encoder that Python writes images with."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sfmtool.fileio import (
    read_image_rgb,
    read_image_rgba,
    write_image_rgb,
    write_image_rgba,
)

_DATA = Path(__file__).resolve().parents[3] / "test-data" / "images"
_PHOTO = _DATA / "seoul_bull_sculpture" / "seoul_bull_sculpture_01.jpg"


def _noise(shape, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def test_png_round_trips_rgb_and_rgba_bit_exactly(tmp_path):
    rgb = _noise((13, 21, 3))
    rgba = _noise((13, 21, 4), seed=1)
    rgba[0, 0, 3] = 0
    write_image_rgb(tmp_path / "rgb.png", rgb)
    write_image_rgba(tmp_path / "rgba.png", rgba)

    np.testing.assert_array_equal(read_image_rgb(tmp_path / "rgb.png"), rgb)
    np.testing.assert_array_equal(read_image_rgba(tmp_path / "rgba.png"), rgba)


def test_channels_are_written_in_rgb_order(tmp_path):
    # Pure red, written by the Rust writer and read back by OpenCV, which
    # returns BGR: red lands in OpenCV's last channel.
    red = np.zeros((8, 8, 3), np.uint8)
    red[..., 0] = 255
    for name in ("red.png", "red.jpg"):
        write_image_rgb(tmp_path / name, red)
        bgr = cv2.imread(str(tmp_path / name))
        assert bgr[4, 4, 2] > 240 and bgr[4, 4, 0] < 15 and bgr[4, 4, 1] < 15


def test_a_non_contiguous_array_is_copied(tmp_path):
    rgb = _noise((10, 30, 3))
    view = rgb[:, ::2]  # every other column: not C-contiguous
    assert not view.flags.c_contiguous
    write_image_rgb(tmp_path / "view.png", view)
    np.testing.assert_array_equal(read_image_rgb(tmp_path / "view.png"), view)


def test_jpeg_is_close_to_the_source_and_to_opencv(tmp_path):
    rgb = read_image_rgb(_PHOTO)
    write_image_rgb(tmp_path / "rust.jpg", rgb)
    cv2.imwrite(str(tmp_path / "cv.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    back = read_image_rgb(tmp_path / "rust.jpg").astype(np.int16)
    error = np.abs(back - rgb)
    assert error.mean() < 1.0
    # Both writers default to quality 95; the Rust encoder keeps full-resolution
    # chroma where OpenCV subsamples it, so its file is somewhat larger.
    rust_size = (tmp_path / "rust.jpg").stat().st_size
    cv_size = (tmp_path / "cv.jpg").stat().st_size
    assert 0.8 * cv_size < rust_size < 1.5 * cv_size


def test_jpeg_quality_defaults_to_95(tmp_path):
    rgb = read_image_rgb(_PHOTO)
    write_image_rgb(tmp_path / "default.jpg", rgb)
    write_image_rgb(tmp_path / "q95.jpg", rgb, jpeg_quality=95)
    write_image_rgb(tmp_path / "q50.jpg", rgb, jpeg_quality=50)

    default = (tmp_path / "default.jpg").read_bytes()
    assert default == (tmp_path / "q95.jpg").read_bytes()
    assert (tmp_path / "q50.jpg").stat().st_size < len(default)


def test_rgba_to_jpeg_raises_rather_than_dropping_alpha(tmp_path):
    path = tmp_path / "alpha.jpg"
    with pytest.raises(ValueError, match="alpha.jpg"):
        write_image_rgba(path, _noise((4, 4, 4)))
    assert not path.exists()


def test_an_unknown_extension_raises_value_error(tmp_path):
    path = tmp_path / "image.notaformat"
    with pytest.raises(ValueError, match="image.notaformat"):
        write_image_rgb(path, _noise((4, 4, 3)))
    assert not path.exists()


def test_an_unwritable_path_raises_naming_it(tmp_path):
    missing = tmp_path / "no_such_dir" / "image.png"
    with pytest.raises(FileNotFoundError, match="image.png"):
        write_image_rgb(missing, _noise((4, 4, 3)))
    # A directory where the file should go cannot be written either.
    (tmp_path / "taken.png").mkdir()
    with pytest.raises(OSError, match="taken.png"):
        write_image_rgb(tmp_path / "taken.png", _noise((4, 4, 3)))


@pytest.mark.parametrize("quality", [0, 101, -5])
def test_a_jpeg_quality_outside_1_to_100_raises(tmp_path, quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        write_image_rgb(tmp_path / "q.jpg", _noise((4, 4, 3)), jpeg_quality=quality)


def test_the_wrong_shape_or_dtype_raises(tmp_path):
    path = tmp_path / "x.png"
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        write_image_rgb(path, _noise((4, 4, 4)))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        write_image_rgb(path, _noise((4, 4)))
    with pytest.raises(ValueError, match=r"\(H, W, 4\)"):
        write_image_rgba(path, _noise((4, 4, 3)))
    with pytest.raises(TypeError, match="uint8"):
        write_image_rgb(path, np.zeros((4, 4, 3), np.float32))
    assert not path.exists()
