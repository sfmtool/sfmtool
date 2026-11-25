# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.photographs.PhotographCache`: its budgets, `build_mipmaps`, `stats`,
`layout` and its reads, in a temporary workspace whose `mipmap_options` sets a
threshold of 64 px."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sfmtool import ProgressCounter
from sfmtool.fileio import read_image_rgb
from sfmtool.photographs import PhotographCache

_MIPMAP_OPTIONS = {
    "out_of_core_long_side": 64,
    "jpeg_quality": 95,
    "full_resolution_jpeg_quality": 95,
}


def _photograph(path: Path, width: int, height: int) -> Path:
    rng = np.random.default_rng(width * height)
    cv2.imwrite(str(path), rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    return path


def test_build_mipmaps_makes_the_out_of_core_mipmaps_and_reports_each_path(tmp_path):
    (tmp_path / ".sfm-workspace.json").write_text(
        json.dumps({"version": 2, "mipmap_options": _MIPMAP_OPTIONS})
    )
    big = _photograph(tmp_path / "big.png", 300, 200)
    small = _photograph(tmp_path / "small.png", 60, 40)
    gone = tmp_path / "gone.png"
    cache = PhotographCache(budget_bytes=0)
    counter = ProgressCounter()

    report = cache.build_mipmaps([big, small, gone], progress=counter)

    assert report["made"] == [big]
    assert report["current"] == []
    assert report["in_core"] == [small]
    [(path, message)] = report["failed"]
    assert path == gone and str(gone) in message
    assert counter.value == 1
    mip1 = tmp_path / ".sfm-mipmaps" / "big.png-mip1-q95.jpg"
    assert mip1.is_file()
    stats = cache.stats()
    assert sorted(stats) == [
        "budget_bytes",
        "build_budget_bytes",
        "build_reserved_bytes",
        "bytes",
        "decodes",
        "entries",
        "hits",
        "mipmaps_made",
        "misses",
    ]
    assert (stats["budget_bytes"], stats["mipmaps_made"], stats["decodes"]) == (0, 1, 0)
    assert stats["build_reserved_bytes"] == 0

    again = cache.build_mipmaps([str(big)])
    assert (again["made"], again["current"]) == ([], [big])
    assert cache.stats()["mipmaps_made"] == 1


def test_build_mipmaps_outside_a_mipmap_workspace_makes_nothing(tmp_path):
    big = _photograph(tmp_path / "big.png", 300, 200)
    report = PhotographCache().build_mipmaps([big])
    assert report == {"made": [], "current": [], "in_core": [big], "failed": []}
    assert not (tmp_path / ".sfm-mipmaps").exists()


def test_budgets_are_the_given_ones_and_a_negative_one_is_refused():
    stats = PhotographCache(budget_bytes=5, build_budget_bytes=7).stats()
    assert (stats["budget_bytes"], stats["build_budget_bytes"]) == (5, 7)
    with pytest.raises(ValueError, match="budget_bytes"):
        PhotographCache(budget_bytes=-1)
    with pytest.raises(ValueError, match="build_budget_bytes"):
        PhotographCache(build_budget_bytes=-1)
    with pytest.raises(TypeError):
        PhotographCache(budget_bytes="1")


def _mipmap_workspace(path: Path) -> Path:
    (path / ".sfm-workspace.json").write_text(
        json.dumps({"version": 2, "mipmap_options": _MIPMAP_OPTIONS})
    )
    return path


def test_layout_describes_an_in_core_and_an_out_of_core_photograph(tmp_path):
    ws = _mipmap_workspace(tmp_path)
    big = _photograph(ws / "big.png", 300, 200)
    small = _photograph(ws / "small.png", 60, 40)
    cache = PhotographCache(budget_bytes=0)

    out = cache.layout(big)
    assert (out.width, out.height, out.levels) == (300, 200, 8)
    assert out.out_of_core and out.region_readable is False
    # Level 1, 150 px wide, is the first to fit in one 512 px tile.
    assert out.stored_levels == 1
    assert out.level_sizes[:3] == [(300, 200), (150, 100), (75, 50)]
    assert out.tile_size == 512
    assert not (ws / ".sfm-mipmaps").exists(), "a layout makes no mipmap"

    inside = cache.layout(str(small))
    assert (inside.width, inside.height) == (60, 40)
    assert not inside.out_of_core and inside.region_readable is None
    assert inside.stored_levels == 0
    assert len(inside.level_sizes) == inside.levels
    assert "PhotographLayout(width=60, height=40" in repr(inside)


def test_reads_are_read_only_arrays_of_the_pixels_kernels_sample(tmp_path):
    ws = _mipmap_workspace(tmp_path)
    big = _photograph(ws / "big.png", 300, 200)
    small = _photograph(ws / "small.png", 60, 40)
    cache = PhotographCache()
    counter = ProgressCounter()

    level1 = cache.read_level(big, 1, progress=counter)
    assert counter.value == 1, "the read made the mipmap"
    assert (ws / ".sfm-mipmaps" / "big.png-mip1-q95.jpg").is_file()
    assert level1.shape == (100, 150, 3) and level1.dtype == np.uint8
    assert level1.flags.c_contiguous and not level1.flags.writeable
    with pytest.raises(ValueError):
        level1[0, 0, 0] = 1
    counter = ProgressCounter()
    region = cache.read_region(big, 1, 20, 7, 145, 39, progress=counter)
    assert counter.value == 0, "nothing made again"
    np.testing.assert_array_equal(region, level1[7:39, 20:145])
    assert not region.flags.writeable

    # In core every level is the decode's own, as read_image_rgb reads it.
    whole = read_image_rgb(small)
    np.testing.assert_array_equal(cache.read_level(small, 0), whole)
    np.testing.assert_array_equal(
        cache.read_region(small, 0, 3, 4, 50, 40), whole[4:40, 3:50]
    )
    level2 = cache.read_level(small, 2)
    assert level2.shape == (10, 15, 3)
    tile = cache.read_tile(small, 2, 0, 0)
    np.testing.assert_array_equal(tile, level2)
    assert not tile.flags.writeable


def test_reads_refuse_what_the_photograph_does_not_have(tmp_path):
    ws = _mipmap_workspace(tmp_path)
    small = _photograph(ws / "small.png", 60, 40)
    cache = PhotographCache(budget_bytes=0)
    levels = cache.layout(small).levels
    for read in (
        lambda: cache.read_level(small, levels),
        lambda: cache.read_region(small, 0, 0, 0, 61, 10),
        lambda: cache.read_region(small, 0, 5, 5, 5, 10),
        lambda: cache.read_tile(small, 0, 1, 0),
        lambda: cache.read_tile(small, 300, 0, 0),
    ):
        with pytest.raises(ValueError, match="small.png|past"):
            read()
    for read in (
        lambda: cache.read_level(small, -1),
        lambda: cache.read_tile(small, 0, 0, -1),
        lambda: cache.read_region(small, 0, -1, 0, 5, 5),
    ):
        with pytest.raises(ValueError, match="must be 0 or more"):
            read()
    with pytest.raises(TypeError):
        cache.read_level(small, "0")

    gone = ws / "gone.png"
    with pytest.raises(FileNotFoundError, match="gone.png"):
        cache.layout(gone)
    with pytest.raises(FileNotFoundError, match="gone.png"):
        cache.read_level(gone, 0)

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".sfm-workspace.json").write_text(
        json.dumps({"version": 2, "mipmap_options": {"out_of_core_long_side": "big"}})
    )
    photo = _photograph(broken / "a.png", 60, 40)
    with pytest.raises(RuntimeError, match="out_of_core_long_side"):
        cache.layout(photo)
    with pytest.raises(RuntimeError, match="out_of_core_long_side"):
        cache.read_tile(photo, 0, 0, 0)
