# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the `sfm cluster-patches` CLI command.

The cluster-bearing input file is produced the way users produce it: by
`sfm match --cluster`, which persists the matcher's clusters as its primary
artifact (programmatic construction of cluster-bearing dicts is covered by
`test_matches_clusters.py` and `test_pairs_from_matches.py`).
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main

# matches_format::ClusterMemberStatus discriminants.
STATUS_REFERENCE = 0
STATUS_KEPT = 1
VALID_STATUSES = {0, 1, 2, 3, 4, 5}


@pytest.fixture
def cluster_matches_file(isolated_seoul_bull_17_images) -> Path:
    """A cluster-bearing .matches file from `sfm match --cluster`."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent
    out = workspace_dir / "matches" / "clusters.matches"

    runner = CliRunner()
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        main,
        [
            "match",
            "--cluster",
            "--clusters-output",
            str(out),
            "--output",
            str(workspace_dir / "tvg-matches" / "verified.matches"),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    return out


def test_cluster_patches_end_to_end(cluster_matches_file: Path):
    from sfmtool._sfmtool.io import read_matches, verify_matches

    result = CliRunner().invoke(
        main, ["cluster-patches", "-i", str(cluster_matches_file)]
    )
    assert result.exit_code == 0, result.output

    out_path = cluster_matches_file.with_name("clusters-patches.matches")
    assert out_path.exists()

    valid, errors = verify_matches(out_path)
    assert valid, errors

    src = read_matches(cluster_matches_file)
    data = read_matches(out_path)
    assert data["has_clusters"]
    assert data["has_cluster_patches"]
    # Images + clusters sections copied verbatim.
    assert data["image_names"] == src["image_names"]
    np.testing.assert_array_equal(data["cluster_starts"], src["cluster_starts"])
    np.testing.assert_array_equal(data["member_images"], src["member_images"])
    np.testing.assert_array_equal(data["member_features"], src["member_features"])
    assert data["refine_options"]["radius"] == 4.0
    assert data["refine_options"]["min_zncc"] == 0.85

    statuses = data["member_status"]
    starts = data["cluster_starts"]
    assert set(np.unique(statuses).tolist()) <= VALID_STATUSES
    assert STATUS_REFERENCE in statuses
    assert STATUS_KEPT in statuses

    # > 50% of multi-member clusters keep at least one member.
    cluster_count = len(starts) - 1
    multi = kept_any = 0
    for c in range(cluster_count):
        s = statuses[starts[c] : starts[c + 1]]
        if len(s) < 2:
            continue
        multi += 1
        if (s == STATUS_KEPT).any():
            kept_any += 1
    assert multi > 0
    assert kept_any / multi > 0.5, f"{kept_any}/{multi} clusters kept a member"

    # References: in range and flagged Reference; unrefinable clusters use
    # the sentinel.
    refs = data["reference_members"]
    for c in range(cluster_count):
        r = int(refs[c])
        if r == 0xFFFFFFFF:
            continue
        assert starts[c] <= r < starts[c + 1]
        assert statuses[r] == STATUS_REFERENCE

    # ZNCC of kept members clears the gate; signals are finite for evaluated
    # members.
    zncc = data["member_zncc"]
    kept_mask = statuses == STATUS_KEPT
    assert (zncc[kept_mask] >= 0.85 - 1e-6).all()
    shift = data["member_shift_px"]
    assert (shift[kept_mask] <= 3.0 + 1e-6).all()


def test_cluster_patches_rejects_existing_output_and_enriched_input(
    cluster_matches_file: Path,
):
    runner = CliRunner()
    first = runner.invoke(main, ["cluster-patches", "-i", str(cluster_matches_file)])
    assert first.exit_code == 0, first.output
    out_path = cluster_matches_file.with_name("clusters-patches.matches")

    # Write-once: the default output now exists.
    again = runner.invoke(main, ["cluster-patches", "-i", str(cluster_matches_file)])
    assert again.exit_code != 0
    assert "already exists" in again.output

    # An already-enriched file is rejected up front.
    enriched = runner.invoke(
        main,
        [
            "cluster-patches",
            "-i",
            str(out_path),
            "-o",
            str(out_path.with_name("twice.matches")),
        ],
    )
    assert enriched.exit_code != 0
    assert "already carries" in enriched.output


def test_cluster_patches_rejects_pairwise_input(tmp_path: Path):
    """A pairwise .matches file (no clusters) is rejected with guidance."""
    from sfmtool._sfmtool.io import write_matches

    path = tmp_path / "pairwise.matches"
    data = {
        "metadata": {
            "version": 3,
            "matching_method": "exhaustive",
            "matching_tool": "test",
            "matching_tool_version": "0",
            "matching_options": {},
            "workspace": {
                "absolute_path": str(tmp_path),
                "relative_path": "",
                "contents": {
                    "feature_tool": "opencv",
                    "feature_type": "sift",
                    "feature_options": {},
                    "feature_prefix_dir": "",
                },
            },
            "timestamp": "2026-07-09T10:00:00Z",
            "image_count": 2,
            "image_pair_count": 1,
            "match_count": 1,
            "has_two_view_geometries": False,
            "has_clusters": False,
            "has_cluster_patches": False,
        },
        "image_names": ["a.jpg", "b.jpg"],
        "feature_tool_hashes": np.zeros((2, 16), dtype=np.uint8),
        "sift_content_hashes": np.zeros((2, 16), dtype=np.uint8),
        "feature_counts": np.array([1, 1], dtype=np.uint32),
        "image_index_pairs": np.array([[0, 1]], dtype=np.uint32),
        "match_counts": np.array([1], dtype=np.uint32),
        "match_feature_indexes": np.array([[0, 0]], dtype=np.uint32),
        "match_descriptor_distances": np.array([1.0], dtype=np.float32),
        "has_clusters": False,
        "has_cluster_patches": False,
        "has_two_view_geometries": False,
    }
    write_matches(path, data)

    result = CliRunner().invoke(main, ["cluster-patches", "-i", str(path)])
    assert result.exit_code != 0
    assert "match --cluster" in result.output
