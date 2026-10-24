# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The Rust `.sift` builder (`build_sift_files`) writes the files the Python
`sfmtool` backend of `sfm sift --extract` writes."""

import shutil

import numpy as np
import pytest

from sfmtool.fileio import read_image_rgb, read_sift, write_image_rgb
from sfmtool.sift import build_sift_files
from sfmtool.sift.extract import image_files_to_sift_files
from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

from tests.conftest import SEOUL_BULL_DIR


@pytest.fixture
def images(tmp_path):
    """Two seoul_bull photographs, and the first one enlarged to 512 x 768
    (an integer reduction to the 128 x 128 thumbnail) and to 256 x 256 (the
    2 x 2 reduction OpenCV runs on its vector path)."""
    source = SEOUL_BULL_DIR
    paths = []
    for name in ("seoul_bull_sculpture_01.jpg", "seoul_bull_sculpture_02.jpg"):
        shutil.copy(source / name, tmp_path / name)
        paths.append(tmp_path / name)
    rgb = read_image_rgb(paths[0])
    for w, h in ((512, 768), (256, 256)):
        ys = np.arange(h) * rgb.shape[0] // h
        xs = np.arange(w) * rgb.shape[1] // w
        path = tmp_path / f"scaled_{w}x{h}.png"
        write_image_rgb(path, np.ascontiguousarray(rgb[ys][:, xs]))
        paths.append(path)
    return paths


def test_rust_and_python_write_the_same_files(images):
    options = get_default_sfmtool_feature_options()
    python_paths = image_files_to_sift_files(
        images,
        feature_tool="sfmtool",
        feature_options=options,
        feature_prefix_dir="features/python",
    )
    rust_paths, extracted, skipped = build_sift_files(images, options, "features/rust")
    assert (extracted, skipped) == (len(images), 0)
    assert [p.name for p in rust_paths] == [p.name for p in python_paths]
    for python_path, rust_path in zip(python_paths, rust_paths):
        assert rust_path == python_path.parent.parent / "rust" / python_path.name
        python_sift = read_sift(python_path)
        rust_sift = read_sift(rust_path)
        assert (
            rust_sift["feature_tool_metadata"] == python_sift["feature_tool_metadata"]
        )
        assert rust_sift["metadata"] == python_sift["metadata"]
        for key in (
            "positions_xy",
            "affine_shapes",
            "descriptors",
            "thumbnail_y_x_rgb",
        ):
            np.testing.assert_array_equal(rust_sift[key], python_sift[key], err_msg=key)
        # Every stored byte agrees, so the hash over the content does too.
        assert rust_sift["content_hash"] == python_sift["content_hash"]

    # A second build finds every file up to date.
    _, extracted, skipped = build_sift_files(images, options, "features/rust")
    assert (extracted, skipped) == (0, len(images))


def test_invalid_options_and_unreadable_images_raise(images, tmp_path):
    with pytest.raises(ValueError, match="params\\['sigma'\\]"):
        build_sift_files(images, {"sigma": "wide"}, "features/rust")
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(OSError, match="broken.jpg"):
        build_sift_files([broken], {}, "features/rust")
