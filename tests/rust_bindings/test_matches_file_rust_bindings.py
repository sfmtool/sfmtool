# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `MatchesFile` handle binding: single-parse array access,
file-level cluster selection (`select_clusters`), and `save` round-trips."""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.io import MatchesFile, verify_matches, write_matches

UNREFINABLE = np.iinfo(np.uint32).max


def _cluster_patch_dict() -> dict:
    """4 images, 3 clusters, 8 members.

    Cluster 0 = members 0..3 on images (0, 1, 2): reference 0, kept, kept.
    Cluster 1 = members 3..5 on images (1, 2): unrefinable.
    Cluster 2 = members 5..8 on images (2, 3, 0): reference 5, kept, kept.
    """
    # The backbone's geometry -- one position and one shape per member. This
    # fixture is a cluster-patches output, so the members the cascade measured
    # carry its answer; members 3 and 4 were never fitted and keep the
    # detections they came in with (distinct values, so a mix-up shows).
    statuses = np.array([0, 1, 1, 5, 5, 0, 1, 1], dtype=np.uint8)
    member_positions = np.array(
        [[10.0 + k, 20.0 + k] for k in range(8)], dtype=np.float32
    )
    member_affine_shapes = np.array(
        [[[1.1, 0.05], [0.02, 0.9]] for _ in range(8)], dtype=np.float32
    )
    # A cluster's reference member carries `S_ref`, its own detector affine
    # shape -- non-singular and deliberately not the identity.
    for ref in (0, 5):
        member_affine_shapes[ref] = [[2.0, 0.5], [0.25, 1.5]]
    return {
        "metadata": {
            "version": 4,
            "matching_method": "cluster",
            "matching_tool": "sfmtool",
            "matching_tool_version": "0.2",
            "matching_options": {"d": 8},
            "workspace": {
                "absolute_path": "/tmp/workspace",
                "relative_path": "..",
                "contents": {
                    "feature_tool": "sfmtool",
                    "feature_type": "sift",
                    "feature_options": {},
                    "feature_prefix_dir": "features/sift-sfmtool-abc123",
                },
            },
            "timestamp": "2026-07-09T10:00:00Z",
            "image_count": 4,
            "cluster_count": 3,
            "cluster_member_count": 8,
            "has_two_view_geometries": False,
            "has_clusters": True,
            "has_cluster_patches": True,
        },
        "image_names": [f"frames/frame_{j:03d}.jpg" for j in range(4)],
        "feature_tool_hashes": [b"\x00" * 16] * 4,
        "sift_content_hashes": [b"\x01" * 16] * 4,
        "feature_counts": np.array([100, 150, 200, 120], dtype=np.uint32),
        "image_dims": np.array(
            [[640, 480], [640, 480], [1024, 768], [800, 600]], dtype=np.uint32
        ),
        "has_clusters": True,
        "cluster_starts": np.array([0, 3, 5, 8], dtype=np.uint32),
        "member_images": np.array([0, 1, 2, 1, 2, 2, 3, 0], dtype=np.uint32),
        "member_features": np.array([0, 1, 2, 5, 10, 11, 12, 13], dtype=np.uint32),
        "member_positions": member_positions,
        "member_affine_shapes": member_affine_shapes,
        "matcher_options": {"d": 8, "min_size": 2},
        "has_cluster_patches": True,
        "reference_members": np.array([0, UNREFINABLE, 5], dtype=np.uint32),
        "member_status": statuses,
        "member_zncc": np.array(
            [1.0, 0.9, 0.8, np.nan, np.nan, 1.0, 0.95, 0.88], dtype=np.float32
        ),
        "member_shift_px": np.array(
            [0.0, 1.0, 2.0, np.nan, np.nan, 0.0, 0.5, 0.6], dtype=np.float32
        ),
        "member_consistency_residual": np.array(
            [0.10, 0.20, 0.30, np.nan, np.nan, 0.15, np.nan, 0.35], dtype=np.float32
        ),
        "refine_options": {"patch_size": 12.0, "resolution": 15},
        "has_two_view_geometries": False,
    }


@pytest.fixture
def matches_path(tmp_path):
    path = tmp_path / "clusters-patches.matches"
    write_matches(path, _cluster_patch_dict())
    return path


def test_accessors_match_source_arrays(matches_path):
    src = _cluster_patch_dict()
    mf = MatchesFile(matches_path)

    assert mf.has_clusters and mf.has_cluster_patches
    assert mf.image_names == src["image_names"]
    npt.assert_array_equal(mf.image_dims, src["image_dims"])
    npt.assert_array_equal(mf.feature_counts, src["feature_counts"])
    npt.assert_array_equal(mf.cluster_starts, src["cluster_starts"])
    npt.assert_array_equal(mf.member_images, src["member_images"])
    npt.assert_array_equal(mf.member_features, src["member_features"])
    npt.assert_array_equal(mf.reference_members, src["reference_members"])
    npt.assert_array_equal(mf.member_status, src["member_status"])
    npt.assert_array_equal(mf.member_zncc, src["member_zncc"])
    npt.assert_array_equal(
        mf.member_consistency_residual, src["member_consistency_residual"]
    )
    assert mf.matcher_options == src["matcher_options"]
    assert mf.refine_options == src["refine_options"]
    assert mf.metadata["cluster_count"] == 3
    assert len(mf.content_xxh128) == 32

    # The geometry accessors read the backbone, at its stored float32; every
    # row carries a real value, and member_status says what it means.
    npt.assert_array_equal(mf.member_positions(), src["member_positions"])
    npt.assert_array_equal(mf.member_affine_shapes(), src["member_affine_shapes"])
    assert mf.member_positions().dtype == np.float32
    assert mf.member_affine_shapes().dtype == np.float32
    assert np.isfinite(mf.member_positions()).all()

    # The shapes are ABSOLUTE; the reference->member warp comes back by
    # inverting the cluster's own reference member's.
    shapes = np.asarray(mf.member_affine_shapes(), dtype=np.float64)
    s_ref = shapes[int(mf.reference_members[0])]
    w = shapes[1] @ np.linalg.inv(s_ref)
    npt.assert_allclose(w @ s_ref, shapes[1], atol=1e-12)
    # refine_radius halves the full patch edge; worst consistency is the
    # per-cluster max finite residual (inf when none).
    assert mf.refine_radius == 6.0
    npt.assert_array_equal(
        mf.cluster_worst_consistency(),
        np.array([np.float32(0.30), np.inf, np.float32(0.35)], dtype=np.float64),
    )


def test_select_clusters_restriction_and_save(matches_path, tmp_path):
    mf = MatchesFile(matches_path)
    names = mf.image_names

    # Restrict away image 0: cluster 0 survives on images (1, 2) with its
    # reference (on image 0) absent -> sentinel; cluster 1 is dropped at the
    # source (unrefinable); cluster 2 survives on images (2, 3) with its
    # reference kept.
    sel = mf.select_clusters(min_span=2, restrict_images=names[1:])
    assert sel.image_names == names[1:]
    npt.assert_array_equal(sel.cluster_starts, [0, 2, 4])
    npt.assert_array_equal(sel.member_images, [0, 1, 1, 2])
    npt.assert_array_equal(sel.member_features, [1, 2, 11, 12])
    npt.assert_array_equal(sel.reference_members, [UNREFINABLE, 2])
    npt.assert_array_equal(sel.member_status, [1, 1, 0, 1])
    provenance = sel.metadata["matching_options"]["cluster_selection"]
    assert provenance["source_content_xxh128"] == mf.content_xxh128
    assert provenance["accepted_statuses"] == ["reference", "kept"]

    # The selection saves as a complete, verifiable .matches file and reads
    # back identically.
    out = tmp_path / "selected.matches"
    sel.save(out)
    valid, errors = verify_matches(out)
    assert valid, errors
    reread = MatchesFile(out)
    npt.assert_array_equal(reread.cluster_starts, sel.cluster_starts)
    npt.assert_array_equal(reread.member_positions(), sel.member_positions())
    npt.assert_array_equal(reread.member_affine_shapes(), sel.member_affine_shapes())
    npt.assert_array_equal(reread.reference_members, sel.reference_members)


def test_select_clusters_restrict_cluster_ids(matches_path, tmp_path):
    mf = MatchesFile(matches_path)

    # Cluster-id restriction: only source cluster 2 is requested (cluster 1 is
    # dropped at the source anyway, being unrefinable).  The image table is
    # untouched — the id axis does not shape it.
    sel = mf.select_clusters(min_span=2, restrict_cluster_ids=[2])
    assert sel.image_names == mf.image_names
    npt.assert_array_equal(sel.cluster_starts, [0, 3])
    npt.assert_array_equal(sel.member_images, [2, 3, 0])
    npt.assert_array_equal(sel.member_features, [11, 12, 13])
    npt.assert_array_equal(sel.reference_members, [0])
    provenance = sel.metadata["matching_options"]["cluster_selection"]
    assert provenance["restrict_cluster_ids"] == [2]

    # Composition with the image axis: each applies its own.
    both = mf.select_clusters(
        min_span=2,
        restrict_images=mf.image_names[1:],
        restrict_cluster_ids=[0, 2],
    )
    npt.assert_array_equal(both.cluster_starts, [0, 2, 4])
    npt.assert_array_equal(both.member_images, [0, 1, 1, 2])
    npt.assert_array_equal(both.reference_members, [UNREFINABLE, 2])
    assert both.image_names == mf.image_names[1:]

    # Requesting every id changes nothing but the provenance key, and a
    # selection that does not request the axis writes the same bytes as before
    # the option existed (the key is simply absent).
    all_ids = mf.select_clusters(min_span=2, restrict_cluster_ids=[0, 1, 2])
    plain = mf.select_clusters(min_span=2)
    npt.assert_array_equal(all_ids.cluster_starts, plain.cluster_starts)
    npt.assert_array_equal(all_ids.member_features, plain.member_features)
    assert (
        "restrict_cluster_ids"
        not in (plain.metadata["matching_options"]["cluster_selection"])
    )
    a, b = tmp_path / "plain.matches", tmp_path / "none.matches"
    plain.save(a)
    mf.select_clusters(min_span=2, restrict_cluster_ids=None).save(b)
    assert a.read_bytes() == b.read_bytes()

    # A saved id-restricted selection verifies and re-reads.
    out = tmp_path / "restricted.matches"
    sel.save(out)
    valid, errors = verify_matches(out)
    assert valid, errors
    npt.assert_array_equal(MatchesFile(out).member_features, sel.member_features)


def test_select_clusters_errors(matches_path):
    mf = MatchesFile(matches_path)
    with pytest.raises(ValueError, match="min_span"):
        mf.select_clusters(min_span=1)
    with pytest.raises(ValueError, match="not in the source image table"):
        mf.select_clusters(restrict_images=["frames/nope.jpg"])
    with pytest.raises(ValueError, match="ClusterMemberStatus"):
        mf.select_clusters(accepted_statuses=["bogus"])
    # The file has 3 clusters, so id 3 names none of them.
    with pytest.raises(ValueError, match="restrict_cluster_ids id 3"):
        mf.select_clusters(restrict_cluster_ids=[3])
    # Statuses accept ints and names interchangeably.
    by_int = mf.select_clusters(accepted_statuses=[0, 1])
    by_name = mf.select_clusters(accepted_statuses=["reference", "kept"])
    npt.assert_array_equal(by_int.cluster_starts, by_name.cluster_starts)
