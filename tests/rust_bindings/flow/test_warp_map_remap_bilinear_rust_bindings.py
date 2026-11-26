# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`WarpMap.remap_bilinear` reading a photograph of an `ImagePyramidSet`, as the
strip montages read their photographs. See ``specs/python-bindings.md`` and
``specs/core/camera/image-warping.md`` § "Python API"."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from sfmtool.fileio import read_image_rgb
from sfmtool.flow import WarpMap
from sfmtool.patches import ImagePyramidSet
from sfmtool.photographs import PhotographCache


def _photograph(path: Path, width: int, height: int) -> Path:
    rng = np.random.default_rng(width * height)
    cv2.imwrite(str(path), rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    return path


def _map(width: int, height: int) -> WarpMap:
    """A 40 x 30 map that turns and scales across the photograph, with
    samples off its edges and two marked invalid."""
    rows, cols = np.mgrid[0:30, 0:40].astype(np.float32)
    map_x = 0.9 * cols - 0.35 * rows + 0.3 * width - 6.0
    map_y = 0.35 * cols + 0.9 * rows + 0.3 * height - 7.0
    map_x[0, 0] = map_y[5, 7] = np.nan
    return WarpMap.from_numpy(map_x, map_y)


def test_a_photograph_in_core_is_sampled_as_its_array_is(tmp_path):
    first = _photograph(tmp_path / "first.png", 96, 80)
    second = _photograph(tmp_path / "second.png", 120, 90)
    pyramids = ImagePyramidSet.read([first, second])

    for index, path in enumerate((first, second)):
        array = read_image_rgb(path)
        warp = _map(array.shape[1], array.shape[0])
        expected = warp.remap_bilinear(array)
        assert expected[0, 0].tolist() == [0, 0, 0]
        np.testing.assert_array_equal(
            warp.remap_bilinear(pyramids, image_index=index), expected
        )


def test_a_photograph_out_of_core_is_sampled_from_its_full_resolution_level(
    tmp_path,
):
    (tmp_path / ".sfm-workspace.json").write_text(
        json.dumps(
            {
                "version": 2,
                "mipmap_options": {
                    "out_of_core_long_side": 64,
                    "jpeg_quality": 95,
                    "full_resolution_jpeg_quality": 95,
                },
            }
        )
    )
    path = _photograph(tmp_path / "big.png", 300, 200)
    cache = PhotographCache()
    pyramids = ImagePyramidSet.read([path], cache=cache)
    warp = _map(300, 200)

    sampled = warp.remap_bilinear(pyramids, image_index=0)

    # A PNG is not read by region, so level 0 is its full-resolution mip
    # level file, a JPEG, whose pixels are not the original's.
    level0 = cache.read_level(path, 0)
    np.testing.assert_array_equal(sampled, warp.remap_bilinear(level0))
    assert not np.array_equal(sampled, warp.remap_bilinear(read_image_rgb(path)))
    assert cache.stats()["decodes"] == 0


def test_an_index_is_given_with_a_set_and_only_with_a_set(tmp_path):
    path = _photograph(tmp_path / "first.png", 64, 48)
    pyramids = ImagePyramidSet.read([path])
    warp = _map(64, 48)

    with pytest.raises(ValueError, match="needs image_index"):
        warp.remap_bilinear(pyramids)
    with pytest.raises(IndexError, match="image_index 1 is past the set's 1 images"):
        warp.remap_bilinear(pyramids, image_index=1)
    with pytest.raises(
        ValueError,
        match="image_index 0 names a photograph of an ImagePyramidSet, but image is an array",
    ):
        warp.remap_bilinear(read_image_rgb(path), image_index=0)


def test_a_photograph_changed_after_the_read_fails_naming_it(tmp_path):
    path = _photograph(tmp_path / "first.png", 64, 48)
    second = _photograph(tmp_path / "second.png", 64, 48)
    # A budget of one byte keeps no tile of the first photograph once the
    # second is read, so the sampling decodes the file again.
    pyramids = ImagePyramidSet.read(
        [path, second], cache=PhotographCache(budget_bytes=1)
    )
    _photograph(path, 70, 50)
    later = os.stat(path).st_mtime + 10
    os.utime(path, (later, later))

    with pytest.raises(
        RuntimeError, match=r"first\.png changed while it was being read"
    ):
        _map(64, 48).remap_bilinear(pyramids, image_index=0)
