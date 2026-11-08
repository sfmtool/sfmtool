# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that `read_matches` returns the workspace directory a `.matches` file
records, under the `workspace_dir` key, and that a `MatchesFile` opened from
the same path reports the same directory as its `workspace_dir`.

The rule (relative path, then absolute path, then the nearest marked directory
above the file) is `resolve_workspace_dir` in `sfmtool-matches-format`, whose
Rust tests cover each step; these check that the bindings report what that
rule finds, and `None` when it finds nothing.
"""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.fileio import MatchesFile, read_matches, write_matches


def _matches_data(relative_path: str, absolute_path: Path) -> dict:
    """A one-pair `.matches` dict recording the two given workspace paths."""
    return {
        "metadata": {
            "version": 4,
            "matching_method": "sequential",
            "matching_tool": "sfmtool",
            "matching_tool_version": "0.2",
            "matching_options": {},
            "workspace": {
                "absolute_path": str(absolute_path),
                "relative_path": relative_path,
                "contents": {
                    "feature_tool": "sfmtool",
                    "feature_type": "sift",
                    "feature_options": {},
                    "feature_prefix_dir": "features/sift-sfmtool-abc123",
                },
            },
            "timestamp": "2026-11-07T10:00:00Z",
            "image_count": 2,
            "image_pair_count": 1,
            "match_count": 1,
            "has_two_view_geometries": False,
            "has_clusters": False,
            "has_cluster_patches": False,
        },
        "image_names": ["a.jpg", "b.jpg"],
        "feature_tool_hashes": [b"\x00" * 16] * 2,
        "sift_content_hashes": [b"\x01" * 16] * 2,
        "feature_counts": np.full(2, 10, dtype=np.uint32),
        "image_dims": np.full((2, 2), [640, 480], dtype=np.uint32),
        "has_clusters": False,
        "image_index_pairs": np.array([[0, 1]], dtype=np.uint32),
        "match_counts": np.array([1], dtype=np.uint32),
        "match_feature_indexes": np.zeros((1, 2), dtype=np.uint32),
        "match_descriptor_distances": np.array([100.0], dtype=np.float32),
        "has_cluster_patches": False,
        "has_two_view_geometries": False,
    }


@pytest.mark.parametrize(
    ("file_dir", "relative_path", "absolute_dir", "expected"),
    [
        # The relative path names a marked directory, ahead of another one.
        ("out", "../ws", "other", "ws"),
        # The relative directory has no marker, so the absolute path is next.
        ("out", "../plain", "other", "other"),
        # Neither recorded directory has a marker; the file is inside `ws`.
        ("ws/matches", "../../plain", "plain", "ws"),
        # Nothing names or encloses a marked directory.
        ("out", "../plain", "plain", None),
    ],
)
def test_read_matches_resolves_the_recorded_workspace(
    tmp_path, file_dir, relative_path, absolute_dir, expected
):
    for sub in ["ws/matches", "other", "plain", "out"]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    for workspace in ["ws", "other"]:
        (tmp_path / workspace / ".sfm-workspace.json").write_text("{}")
    path = tmp_path / file_dir / "x.matches"
    write_matches(path, _matches_data(relative_path, tmp_path / absolute_dir))

    workspace_dir = read_matches(path)["workspace_dir"]

    if expected is None:
        assert workspace_dir is None
    else:
        assert Path(workspace_dir).resolve() == (tmp_path / expected).resolve()
    assert MatchesFile(path).workspace_dir == workspace_dir
