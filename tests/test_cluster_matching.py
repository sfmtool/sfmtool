# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for background-floor track-cluster matching (`sfm match --cluster`)."""

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.feature_match._cluster_matching import cluster_match
from sfmtool.sift.file import write_sift


N_IMAGES = 4
N_POINTS = 5
N_BACKGROUND = 30
DIM = 128


def _write_synthetic_sift(path: Path, descriptors: np.ndarray, image_name: str):
    """Write a .sift file holding the given descriptors (positions arbitrary)."""
    feature_count = len(descriptors)
    feature_tool_metadata = {
        "feature_tool": "pytest",
        "feature_type": "sift",
        "feature_options": {},
    }
    metadata = {
        "version": 1,
        "image_name": image_name,
        "image_file_xxh128": "a" * 32,
        "image_file_size": 12345,
        "image_width": 640,
        "image_height": 480,
        "feature_count": feature_count,
    }
    rng = np.random.default_rng(0)
    position = rng.random((feature_count, 2), dtype=np.float32) * np.array(
        [640, 480], dtype=np.float32
    )
    affine_shape = rng.random((feature_count, 2, 2), dtype=np.float32) - 0.5
    thumbnail = np.zeros((128, 128, 3), dtype=np.uint8)
    write_sift(
        path,
        feature_tool_metadata,
        metadata,
        position,
        affine_shape,
        descriptors,
        thumbnail,
    )


def _synthetic_sift_set(tmp_path: Path, seed=42) -> tuple[list[Path], list[Path]]:
    """N_IMAGES synthetic .sift files with N_POINTS planted cross-image points.

    Each image's features start with the planted observations (base descriptor
    + small jitter), so a planted feature's index equals its point id.
    """
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 256, size=(N_POINTS, DIM), dtype=np.int16)

    image_paths, sift_paths = [], []
    for i in range(N_IMAGES):
        jitter = rng.integers(-2, 3, size=(N_POINTS, DIM), dtype=np.int16)
        planted = np.clip(bases + jitter, 0, 255).astype(np.uint8)
        background = rng.integers(0, 256, size=(N_BACKGROUND, DIM), dtype=np.uint8)
        descriptors = np.vstack([planted, background])

        image_path = tmp_path / f"image_{i:02d}.jpg"
        sift_path = tmp_path / f"image_{i:02d}.jpg.sift"
        _write_synthetic_sift(sift_path, descriptors, image_path.name)
        image_paths.append(image_path)
        sift_paths.append(sift_path)
    return image_paths, sift_paths


class TestClusterMatch:
    def test_cluster_invariants(self, tmp_path):
        image_paths, sift_paths = _synthetic_sift_set(tmp_path)
        clusters, pairs = cluster_match(image_paths, sift_paths, d=16)

        starts = clusters.cluster_starts
        assert starts[0] == 0
        assert np.all(np.diff(starts.astype(np.int64)) >= 0)
        assert starts[-1] == len(clusters.member_images)

        seen = set()
        for c in range(len(starts) - 1):
            lo, hi = int(starts[c]), int(starts[c + 1])
            members = clusters.member_images[lo:hi]
            # One feature per image, sorted by image; spans >= min_size images.
            assert np.all(np.diff(members.astype(np.int64)) > 0)
            assert hi - lo >= 2
            # Disjoint: no (image, feature) in two clusters.
            for m in range(lo, hi):
                key = (int(members[m - lo]), int(clusters.member_features[m]))
                assert key not in seen
                seen.add(key)

        # The planted points come back as full-span clusters.
        assert len(starts) - 1 == N_POINTS
        for c in range(len(starts) - 1):
            lo, hi = int(starts[c]), int(starts[c + 1])
            feats = clusters.member_features[lo:hi]
            assert hi - lo == N_IMAGES
            assert np.all(feats == feats[0]) and feats[0] < N_POINTS

    def test_pair_matches_one_to_one(self, tmp_path):
        image_paths, sift_paths = _synthetic_sift_set(tmp_path)
        _, pairs = cluster_match(image_paths, sift_paths, d=16)

        assert np.all(pairs.image_index_pairs[:, 0] < pairs.image_index_pairs[:, 1])
        assert pairs.match_counts.sum() == len(pairs.match_feature_indexes)

        offset = 0
        for k in range(len(pairs.image_index_pairs)):
            count = int(pairs.match_counts[k])
            block = pairs.match_feature_indexes[offset : offset + count]
            # One-to-one per image pair: no feature repeats on either side.
            assert len(np.unique(block[:, 0])) == count
            assert len(np.unique(block[:, 1])) == count
            offset += count

    def test_max_feature_count_caps_indices(self, tmp_path):
        image_paths, sift_paths = _synthetic_sift_set(tmp_path)
        cap = N_POINTS + 10
        _, pairs = cluster_match(image_paths, sift_paths, d=16, max_feature_count=cap)
        assert len(pairs.match_feature_indexes) > 0
        assert np.all(pairs.match_feature_indexes < cap)


class TestClusterCli:
    def test_cluster_honors_camera_model(
        self, isolated_seoul_bull_17_images: list[Path]
    ):
        # --camera-model is accepted with --cluster: it feeds the geometric
        # verification step. (Remove the committed camera_config.json first,
        # since an explicit model conflicts with a resolved config.)
        workspace_dir = isolated_seoul_bull_17_images[0].parent
        (workspace_dir / "camera_config.json").unlink()

        result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
        assert result.exit_code == 0, result.output
        result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
        assert result.exit_code == 0, result.output

        output_path = workspace_dir / "tvg-matches" / "cluster.matches"
        result = CliRunner().invoke(
            main,
            [
                "match",
                "--cluster",
                "--camera-model",
                "SIMPLE_RADIAL",
                "--output",
                str(output_path),
                str(workspace_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_path.exists()

        from sfmtool._sfmtool import read_matches

        matches_data = read_matches(str(output_path))
        # The model fed geometric verification: TVGs are present.
        assert matches_data["has_two_view_geometries"]
        assert matches_data["tvg_metadata"]["inlier_count"] > 0

    def test_cluster_and_exhaustive_rejected(self, isolated_seoul_bull_image: Path):
        result = CliRunner().invoke(
            main,
            ["match", "--cluster", "--exhaustive", str(isolated_seoul_bull_image)],
        )
        assert result.exit_code != 0
        assert "Cannot specify more than one matching method" in result.output

    def test_match_cluster_end_to_end(
        self, isolated_seoul_bull_17_images: list[Path], tmp_path
    ):
        workspace_dir = isolated_seoul_bull_17_images[0].parent

        result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
        assert result.exit_code == 0, result.output

        result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
        assert result.exit_code == 0, result.output

        output_path = workspace_dir / "tvg-matches" / "cluster.matches"
        result = CliRunner().invoke(
            main,
            [
                "match",
                "--cluster",
                "--output",
                str(output_path),
                str(workspace_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Running cluster matching" in result.output
        assert "track clusters" in result.output
        assert output_path.exists()

        from sfmtool._sfmtool import read_matches

        matches_data = read_matches(str(output_path))
        meta = matches_data["metadata"]
        assert meta["matching_method"] == "cluster"
        assert meta["matching_tool"] == "sfmtool"
        assert meta["matching_options"]["mode"] == "background-floor"
        assert meta["matching_options"]["d"] == 10
        assert meta["matching_options"]["alpha"] == 0.8
        assert meta["image_count"] == 17
        assert meta["image_pair_count"] > 0
        assert meta["match_count"] > 0
        # Geometric verification ran: TVGs are embedded.
        assert matches_data["has_two_view_geometries"]
        assert matches_data["tvg_metadata"]["inlier_count"] > 0

        # The .matches feeds the existing COLMAP DB consumer unchanged.
        db_path = tmp_path / "colmap.db"
        result = CliRunner().invoke(
            main,
            ["to-colmap-db", str(output_path), "--out-db", str(db_path)],
        )
        assert result.exit_code == 0, result.output
        assert db_path.exists()
