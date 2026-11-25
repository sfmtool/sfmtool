# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.photographs.PhotographCache`: its budgets, `build_mipmaps` and
`stats`, in a temporary workspace whose `mipmap_options` sets a threshold of
64 px."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sfmtool import ProgressCounter
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
