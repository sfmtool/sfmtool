# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`Version2SiftFolder` converts a version 1 `.sift` file to version 2 through
the binding, and raises each refusal as a Python exception."""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from sfmtool.fileio import read_image_rgb, read_sift, read_sift_metadata, write_sift
from sfmtool.sift import ImageToGray, Version2SiftFolder, image_readings_from_rgb
from sfmtool.sift.file import xxh128_of_file
from sfmtool.workspace import upgrade_feature_options

from tests.conftest import SEOUL_BULL_DIR

_IMAGES = ("seoul_bull_sculpture_01.jpg", "seoul_bull_sculpture_02.jpg")


@pytest.fixture
def images(tmp_path: Path) -> list[Path]:
    """Two seoul_bull photographs in `images/` of a temporary directory."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in _IMAGES:
        shutil.copy(SEOUL_BULL_DIR / name, image_dir / name)
    return [image_dir / name for name in _IMAGES]


def _write_version_1(image: Path, feature_tool: str, feature_options: dict) -> Path:
    """A version 1 `.sift` file for `image` in `features/v1/` beside it, with
    features and a thumbnail that vary from entry to entry."""
    rgb = read_image_rgb(image)
    folder = image.parent / "features" / "v1"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{image.name}.sift"
    count = 5
    write_sift(
        path,
        {
            "feature_tool_metadata": {
                "feature_tool": feature_tool,
                "feature_type": "sift",
                "feature_options": feature_options,
            },
            "metadata": {
                "version": 1,
                "image_name": image.name,
                "image_file_xxh128": xxh128_of_file(image),
                "image_file_size": image.stat().st_size,
                "image_width": rgb.shape[1],
                "image_height": rgb.shape[0],
                "feature_count": count,
            },
            "positions_xy": np.arange(count * 2, dtype=np.float32).reshape(count, 2)
            * 1.5,
            "affine_shapes": np.arange(count * 4, dtype=np.float32).reshape(count, 2, 2)
            + 0.25,
            "descriptors": (np.arange(count * 128) % 251)
            .astype(np.uint8)
            .reshape(count, 128),
            "thumbnail_y_x_rgb": (np.arange(128 * 128 * 3) % 253)
            .astype(np.uint8)
            .reshape(128, 128, 3),
        },
    )
    return path


def test_an_opencv_file_converts_losslessly(images):
    image = images[0]
    v1_options = {"nfeatures": 0, "contrastThreshold": 0.04}
    source = _write_version_1(image, "opencv", v1_options)
    old = read_sift(source)

    folder = Version2SiftFolder(image.parent / "features" / "v2")
    assert repr(folder) == f"Version2SiftFolder({json.dumps(str(folder.path))})"
    converted = folder.convert_version_1_sift_file(source, image)
    assert set(converted) == {"path", "source_content_hash", "content_hash"}
    assert converted["path"] == folder.path / source.name
    new = read_sift(converted["path"])

    for key in ("positions_xy", "affine_shapes", "descriptors", "thumbnail_y_x_rgb"):
        assert new[key].tobytes() == old[key].tobytes(), key
    options = new["feature_tool_metadata"]["feature_options"]
    assert options == upgrade_feature_options("opencv", v1_options)
    assert options["image_to_gray"] == {
        "formula": "0.299*R + 0.587*G + 0.114*B",
        "quantize_bits": 8,
    }
    metadata = new["metadata"]
    assert metadata["version"] == 2
    expected = image_readings_from_rgb(
        read_image_rgb(image), ImageToGray.from_json(options["image_to_gray"])
    )
    assert metadata["sharpness"] == expected.sharpness
    assert metadata["gradient_structure_tensor_xx_yy_xy"] == list(
        expected.gradient_structure_tensor_xx_yy_xy
    )
    assert (
        converted["source_content_hash"] == read_sift_metadata(source)["content_hash"]
    )
    assert (
        converted["content_hash"]
        == read_sift_metadata(converted["path"])["content_hash"]
    )


def test_each_refusal_raises(images):
    image, other = images
    source = _write_version_1(image, "sfmtool", {"gray_formula": "G"})
    folder = Version2SiftFolder(image.parent / "features" / "v2")

    with pytest.raises(OSError, match="holds version 1 .sift files"):
        Version2SiftFolder(source.parent)
    with pytest.raises(FileNotFoundError, match="is not there"):
        folder.convert_version_1_sift_file(source, image.parent / "gone.jpg")
    with pytest.raises(OSError, match="not the image the features were extracted"):
        folder.convert_version_1_sift_file(source, other)
    with pytest.raises(FileNotFoundError):
        folder.convert_version_1_sift_file(source.parent / "gone.sift", image)

    converted = folder.convert_version_1_sift_file(source, image)
    again = Version2SiftFolder(image.parent / "features" / "again")
    with pytest.raises(ValueError, match="version 2 .sift file"):
        again.convert_version_1_sift_file(converted["path"], image)

    undistorted = _write_version_1(
        other,
        "sfmtool-undistort",
        {"source_feature_tool": "colmap", "source_feature_options": {}},
    )
    with pytest.raises(ValueError, match="run 'sfm undistort' again"):
        folder.convert_version_1_sift_file(undistorted, other)


def test_a_folder_that_cannot_be_made_raises(images, tmp_path):
    # A path that is not there and cannot be made: on Windows one under a
    # regular file, and on Linux and macOS, where that is a failure to list
    # instead, a symbolic link to nothing.
    if sys.platform == "win32":
        blocked = images[0] / "v2"
    else:
        blocked = tmp_path / "v2"
        os.symlink(tmp_path / "nowhere", blocked)
    with pytest.raises(OSError, match="^failed to create the folder ") as raised:
        Version2SiftFolder(blocked)
    assert type(raised.value) is OSError


def test_a_failed_write_raises_through_the_format_error_table(images):
    image = images[0]
    source = _write_version_1(image, "sfmtool", {})
    folder = Version2SiftFolder(image.parent / "features" / "v2")
    blocked = folder.path / source.name
    blocked.mkdir()
    # A file cannot be renamed over a directory. The format-error table raises
    # the OSError subclass for the I/O error's kind, not a plain OSError.
    with pytest.raises(OSError, match="^failed to write ") as raised:
        folder.convert_version_1_sift_file(source, image)
    assert isinstance(raised.value, (IsADirectoryError, PermissionError))
    assert str(blocked) in str(raised.value)
    assert list(folder.path.iterdir()) == [blocked]


def test_readings_that_are_not_finite_raise(images):
    image = images[0]
    # The square root of -1 is NaN at every pixel.
    source = _write_version_1(image, "sfmtool", {"gray_formula": "-1 ** 0.5"})
    folder = Version2SiftFolder(image.parent / "features" / "v2")
    with pytest.raises(
        OSError, match="^failed to take the readings of the image"
    ) as raised:
        folder.convert_version_1_sift_file(source, image)
    assert type(raised.value) is OSError
    assert list(folder.path.iterdir()) == []
