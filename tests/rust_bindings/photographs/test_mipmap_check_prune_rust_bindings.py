# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.photographs.check_mipmaps` and `prune_mipmaps`: their reports and
counters, in a temporary workspace whose `mipmap_options` sets a threshold of
64 px. What counts as an orphan, and the guards on what a prune deletes, are
tested in Rust (`crates/sfmtool-core/src/camera/mipmap/prune/tests.rs`)."""

import json
from pathlib import Path

import cv2
import numpy as np

from sfmtool import ProgressCounter
from sfmtool.photographs import PhotographCache, check_mipmaps, prune_mipmaps


def _workspace(root: Path) -> tuple[Path, Path]:
    """A workspace with a 300 x 200 photograph, out of core, and a 60 x 40
    one, in core."""
    (root / ".sfm-workspace.json").write_text(
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
    rng = np.random.default_rng(3)
    paths = []
    for name, (width, height) in (("big.png", (300, 200)), ("small.png", (60, 40))):
        path = root / name
        cv2.imwrite(str(path), rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        paths.append(path)
    return paths[0], paths[1]


def test_check_mipmaps_reports_each_photograph(tmp_path):
    big, small = _workspace(tmp_path)
    gone = tmp_path / "gone.png"
    counter = ProgressCounter()

    before = check_mipmaps([big, small, gone], progress=counter)

    assert before["missing"] == [big]
    assert before["in_core"] == [small]
    assert (before["current"], before["stale"]) == ([], [])
    [(path, message)] = before["failed"]
    assert path == gone and str(gone) in message
    assert counter.value == 3
    assert not (tmp_path / ".sfm-mipmaps").exists()

    PhotographCache(budget_bytes=0).build_mipmaps([big])
    after = check_mipmaps([str(big)])
    assert after["current"] == [big]


def test_prune_mipmaps_deletes_an_orphan_and_a_dry_run_does_not(tmp_path):
    big, _ = _workspace(tmp_path)
    PhotographCache(budget_bytes=0).build_mipmaps([big])
    mipmaps = tmp_path / ".sfm-mipmaps"
    orphan = mipmaps / "gone.png-mip1-q95.jpg"
    orphan.write_bytes((mipmaps / "big.png-mip1-q95.jpg").read_bytes())

    dry = prune_mipmaps([tmp_path], dry_run=True)
    assert dry["orphans"] == [(orphan, "its photograph gone.png no longer exists")]
    assert orphan.is_file()

    counter = ProgressCounter()
    report = prune_mipmaps([tmp_path], progress=counter)
    assert report["orphans"] == dry["orphans"]
    assert not orphan.exists()
    assert report["kept"] == [
        mipmaps / f"big.png-mip{level}-q95.jpg" for level in (0, 1)
    ]
    assert report["other_files"] == [mipmaps / "CACHEDIR.TAG"]
    assert (report["links"], report["failed"], report["outside_workspaces"]) == (
        [],
        [],
        [],
    )
    assert counter.value == 4
