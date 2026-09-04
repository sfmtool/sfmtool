# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the `sfm cluster-patches` CLI command.

The cluster-bearing input file is produced the way users produce it: by
`sfm match --cluster`, which persists the matcher's clusters as its primary
artifact (programmatic construction of cluster-bearing dicts is covered by
`../matching/test_matches_clusters.py` and
`../matching/test_pairs_from_matches.py`).
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main

# matches_format::ClusterMemberStatus discriminants.
STATUS_REFERENCE = 0
STATUS_KEPT = 1
VALID_STATUSES = {0, 1, 2, 3, 4, 5, 6}


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

    # --resolution 16 (vs the 25-per-axis default) samples a coarser template to
    # keep this end-to-end test cheap; the assertions below are structural
    # (status classes, >50% clusters keep a member, gated ZNCC/shift, finite
    # residuals) and hold at the lower grid.
    result = CliRunner().invoke(
        main, ["cluster-patches", "-i", str(cluster_matches_file), "--resolution", "16"]
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
    np.testing.assert_array_equal(data["image_dims"], src["image_dims"])
    np.testing.assert_array_equal(data["cluster_starts"], src["cluster_starts"])
    np.testing.assert_array_equal(data["member_images"], src["member_images"])
    np.testing.assert_array_equal(data["member_features"], src["member_features"])
    assert data["refine_options"]["patch_size"] == 12.0
    assert data["refine_options"]["min_zncc"] == 0.85
    assert data["refine_options"]["max_keypoint_uncertainty"] == 0.35

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

    # Warp-consistency residuals: finite exactly for the fitted population
    # (kept members and references of clusters with >= 2 fitted members),
    # non-negative where finite, NaN for rejected/not-evaluated members.
    consistency = data["member_consistency_residual"]
    assert consistency.dtype == np.float32
    finite = np.isfinite(consistency)
    assert finite.any()
    assert (consistency[finite] >= 0.0).all()
    assert not finite[(statuses != STATUS_REFERENCE) & (statuses != STATUS_KEPT)].any()


def test_matcher_output_states_the_detections(cluster_matches_file: Path):
    """The matcher's own file carries its members' detected geometry, copied
    bit-for-bit from the `.sift` rows their feature indexes name."""
    from sfmtool._sfmtool.io import read_matches, read_sift_partial

    data = read_matches(cluster_matches_file)
    assert data["metadata"]["version"] == 6
    positions = data["member_positions"]
    shapes = data["member_affine_shapes"]
    assert positions.dtype == np.float32 and shapes.dtype == np.float32
    assert positions.shape == (data["metadata"]["cluster_member_count"], 2)
    assert shapes.shape == (data["metadata"]["cluster_member_count"], 2, 2)
    assert np.isfinite(positions).all() and np.isfinite(shapes).all()

    workspace_dir = cluster_matches_file.parent.parent
    prefix = data["metadata"]["workspace"]["contents"]["feature_prefix_dir"]
    member_images = np.asarray(data["member_images"])
    member_features = np.asarray(data["member_features"])
    for i, name in enumerate(data["image_names"]):
        on_image = member_images == i
        feats = member_features[on_image]
        if not len(feats):
            continue
        rel = Path(name)
        sift = read_sift_partial(
            str(workspace_dir / rel.parent / prefix / f"{rel.name}.sift"),
            int(feats.max()) + 1,
        )
        # tobytes(), not allclose: these are verbatim copies, not conversions.
        assert sift["positions_xy"][feats].tobytes() == positions[on_image].tobytes()
        assert sift["affine_shapes"][feats].tobytes() == shapes[on_image].tobytes()


#: The statuses whose rows the refinement cascade measured. Everything else
#: (duplicate_image, not_evaluated, rejected_unlocalizable) it never fitted.
MEASURED_STATUSES = (0, 1, 2, 3)


def test_enriched_output_states_the_refinement(cluster_matches_file: Path):
    """The refined file's geometry is the refinement's, exactly: the kernel's
    float64 answer downcast for every member it measured, and the input's own
    detection -- byte for byte -- for every member it never fitted."""
    import cv2

    from sfmtool._sfmtool.io import read_matches
    from sfmtool._sfmtool.matching import refine_cluster_patches

    result = CliRunner().invoke(
        main, ["cluster-patches", "-i", str(cluster_matches_file), "--resolution", "16"]
    )
    assert result.exit_code == 0, result.output
    src = read_matches(cluster_matches_file)
    data = read_matches(cluster_matches_file.with_name("clusters-patches.matches"))

    statuses = data["member_status"]
    positions = data["member_positions"]
    shapes = data["member_affine_shapes"]
    assert positions.dtype == np.float32 and shapes.dtype == np.float32
    # No NaN anywhere: every member has a real position and shape, and the
    # status alone says which reading it is.
    assert np.isfinite(positions).all() and np.isfinite(shapes).all()
    assert "member_affines" not in data

    measured = np.isin(statuses, MEASURED_STATUSES)
    assert measured.any() and (~measured).any()
    # An unfitted member keeps the detection it came in with, bit for bit.
    assert (
        positions[~measured].tobytes() == src["member_positions"][~measured].tobytes()
    )
    assert (
        shapes[~measured].tobytes() == src["member_affine_shapes"][~measured].tobytes()
    )

    # A measured member's value is the kernel's own float64 answer under the
    # defined downcast -- re-run the kernel on the same seeds and compare.
    workspace_dir = cluster_matches_file.parent.parent
    images, seed_pos, seed_shp = [], [], []
    member_images = np.asarray(src["member_images"])
    member_features = np.asarray(src["member_features"])
    for i, name in enumerate(src["image_names"]):
        images.append(
            np.ascontiguousarray(
                cv2.imread(str(workspace_dir / name), cv2.IMREAD_COLOR)
            )
        )
        count = int(src["feature_counts"][i])
        p = np.zeros((count, 2), dtype=np.float32)
        a = np.zeros((count, 2, 2), dtype=np.float32)
        on_image = member_images == i
        feats = member_features[on_image]
        p[feats] = src["member_positions"][on_image]
        a[feats] = src["member_affine_shapes"][on_image]
        seed_pos.append(p)
        seed_shp.append(a)
    kernel = refine_cluster_patches(
        images,
        seed_pos,
        seed_shp,
        src["cluster_starts"],
        src["member_images"],
        src["member_features"],
        radius=6.0,
        resolution=16,
    )
    np.testing.assert_array_equal(kernel["member_status"], statuses)
    np.testing.assert_array_equal(
        positions[measured], kernel["member_positions"][measured].astype(np.float32)
    )
    np.testing.assert_array_equal(
        shapes[measured], kernel["member_affine_shapes"][measured].astype(np.float32)
    )

    # A reference member is refined against itself, so its refined shape and
    # position are exactly the detections the matcher's file states.
    refs = data["reference_members"]
    refs = refs[refs != 0xFFFFFFFF]
    assert len(refs) > 0
    np.testing.assert_array_equal(shapes[refs], src["member_affine_shapes"][refs])
    np.testing.assert_array_equal(positions[refs], src["member_positions"][refs])


def test_cluster_patches_rejects_existing_output_and_enriched_input(
    cluster_matches_file: Path,
):
    runner = CliRunner()
    # --resolution 16 (vs the 25 default): this test only exercises the
    # write-once / already-enriched guards, so the coarser grid is purely a
    # speedup and changes none of the assertions.
    first = runner.invoke(
        main, ["cluster-patches", "-i", str(cluster_matches_file), "--resolution", "16"]
    )
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
            "version": 4,
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
        "image_dims": np.array([[64, 48], [64, 48]], dtype=np.uint32),
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
