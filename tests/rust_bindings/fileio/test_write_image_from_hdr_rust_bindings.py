# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfmtool.fileio.write_image_from_hdr``, which converts a frame of
HDR video to sRGB by the fixed recipe of specs/core/camera/hdr-to-srgb.md and
writes it. The recipe's values are tested in Rust; these check the binding's
arguments, its layout and its errors."""

import numpy as np
import pytest

from sfmtool.fileio import (
    HDR_EXPOSURE_STOPS_LIMIT,
    HDR_TO_SRGB_RECIPE,
    HDR_TO_SRGB_RECIPE_VERSION,
    read_image_rgb,
    write_image_from_hdr,
)


def _planes(left: int, right: int, height: int = 8, width: int = 16) -> np.ndarray:
    """(3, H, W) 16-bit Y'CbCr planes, grey: the left half at 10-bit Y' code
    `left`, the right half at `right`."""
    planes = np.full((3, height, width), 32768, np.uint16)
    planes[0, :, : width // 2] = left * 64
    planes[0, :, width // 2 :] = right * 64
    return planes


def test_planes_are_converted_and_written(tmp_path):
    # HLG 75% and 50%: sRGB 229 and 137; PQ 203 cd/m² (code 573): 229.
    path = tmp_path / "frame.png"
    times = write_image_from_hdr(path, _planes(721, 502), transfer="HLG")
    # The call's two wall times.
    assert set(times) == {"conversion_seconds", "encoding_seconds"}
    assert all(isinstance(t, float) and t >= 0 for t in times.values())
    pixels = read_image_rgb(path)
    assert pixels.shape == (8, 16, 3)
    assert (pixels[:, :8] == 229).all() and (pixels[:, 8:] == 137).all()
    write_image_from_hdr(path, _planes(721, 502), transfer="HLG", exposure_stops=-1)
    assert read_image_rgb(path)[0, 0].tolist() == [187] * 3
    write_image_from_hdr(path, _planes(573, 573), transfer="PQ")
    assert read_image_rgb(path)[0, 0].tolist() == [229] * 3


def test_a_strided_array_is_read_in_row_order(tmp_path):
    planes = _planes(721, 502, width=32)[:, :, ::2]
    write_image_from_hdr(tmp_path / "frame.png", planes, transfer="HLG")
    pixels = read_image_rgb(tmp_path / "frame.png")
    assert (pixels[:, :8] == 229).all() and (pixels[:, 8:] == 137).all()


def test_constants():
    assert HDR_TO_SRGB_RECIPE == "sfmtool-hdr-to-srgb"
    assert HDR_TO_SRGB_RECIPE_VERSION == 1
    assert HDR_EXPOSURE_STOPS_LIMIT == 16.0


@pytest.mark.parametrize(
    "planes, kwargs, error, message",
    [
        (_planes(721, 502).astype(np.int32), {}, TypeError, "uint16"),
        (_planes(721, 502)[:2], {}, ValueError, "(3, H, W)"),
        (np.zeros((3, 0, 4), np.uint16), {}, ValueError, "(3, H, W)"),
        (_planes(721, 502), {"transfer": "hlg"}, ValueError, '"HLG" or "PQ"'),
        (_planes(721, 502), {"exposure_stops": 17.0}, ValueError, "exposure"),
        (_planes(721, 502), {"exposure_stops": float("nan")}, ValueError, "exposure"),
        (_planes(721, 502), {"jpeg_quality": 0}, ValueError, "jpeg_quality"),
    ],
)
def test_bad_arguments_raise(tmp_path, planes, kwargs, error, message):
    path = tmp_path / "frame.jpg"
    arguments = {"transfer": "HLG", **kwargs}
    with pytest.raises(error, match=message.replace("(", r"\(").replace(")", r"\)")):
        write_image_from_hdr(path, planes, **arguments)
    assert not path.exists()


def test_a_missing_directory_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="could not write image"):
        write_image_from_hdr(
            tmp_path / "missing" / "frame.jpg", _planes(721, 502), transfer="PQ"
        )
