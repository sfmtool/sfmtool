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
    def test_cluster_rejects_camera_model(self, isolated_seoul_bull_image: Path):
        # The clustering uses no intrinsics and verifies nothing, so a camera
        # model would be inert; it belongs to --derive-pairs.
        result = CliRunner().invoke(
            main,
            [
                "match",
                "--cluster",
                "--camera-model",
                "SIMPLE_RADIAL",
                str(isolated_seoul_bull_image),
            ],
        )
        assert result.exit_code != 0
        assert "--camera-model does not apply to --cluster" in result.output
        assert "--derive-pairs" in result.output

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

        clusters_path = workspace_dir / "matches" / "cluster-clusters.matches"
        result = CliRunner().invoke(
            main,
            [
                "match",
                "--cluster",
                "--output",
                str(clusters_path),
                str(workspace_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Running cluster matching" in result.output
        assert "track clusters" in result.output
        assert clusters_path.exists()

        from sfmtool._sfmtool.io import read_matches, verify_matches

        # The clusters file is the whole output: no COLMAP database is opened
        # and no verified pairwise file is written.
        assert not (workspace_dir / "tvg-matches").exists()

        valid, errors = verify_matches(str(clusters_path))
        assert valid, errors

        cluster_data = read_matches(str(clusters_path))
        cmeta = cluster_data["metadata"]
        assert cluster_data["has_clusters"]
        assert not cluster_data["has_two_view_geometries"]
        assert cmeta["matching_method"] == "cluster"
        assert cmeta["matching_tool"] == "sfmtool"
        assert cmeta["image_count"] == 17
        assert cmeta["cluster_count"] > 0
        assert cmeta["cluster_member_count"] > 0
        assert cmeta["matching_options"]["mode"] == "background-floor"
        assert cmeta["matching_options"]["d"] == 10
        assert cmeta["matching_options"]["alpha"] == 0.8
        assert cluster_data["matcher_options"]["preset"] == "accurate"

        # Nothing in the cluster path is nondeterministic, so a second run on
        # the same corpus produces the same backbone bit for bit.
        rerun_path = workspace_dir / "matches" / "rerun-clusters.matches"
        result = CliRunner().invoke(
            main,
            ["match", "--cluster", "--output", str(rerun_path), str(workspace_dir)],
        )
        assert result.exit_code == 0, result.output
        rerun_data = read_matches(str(rerun_path))
        for key in ("cluster_starts", "member_images", "member_features"):
            np.testing.assert_array_equal(rerun_data[key], cluster_data[key])

        # The cluster file's pairwise view needs no sift lookup, so its
        # distances are NaN placeholders.
        from sfmtool.feature_match import pairs_from_matches

        derived = pairs_from_matches(cluster_data)
        assert len(derived["image_index_pairs"]) > 0
        assert np.isnan(derived["match_descriptor_distances"]).all()

        # Consumer smoke test: `sfm inspect` reports cluster stats.
        result = CliRunner().invoke(main, ["inspect", str(clusters_path), "-v"])
        assert result.exit_code == 0, result.output
        assert "Backbone" in result.output
        assert "Cluster size" in result.output
        assert "Matches per pair (derived)" in result.output

        # Consumer smoke test: the COLMAP DB consumer accepts the cluster
        # file directly (pairs derived at read time; no TVGs to write).
        cluster_db_path = tmp_path / "clusters-colmap.db"
        result = CliRunner().invoke(
            main,
            ["to-colmap-db", str(clusters_path), str(cluster_db_path)],
        )
        assert result.exit_code == 0, result.output
        assert cluster_db_path.exists()

    def test_cluster_default_output_lands_in_matches(
        self, isolated_seoul_bull_17_images: list[Path]
    ):
        workspace_dir = isolated_seoul_bull_17_images[0].parent

        result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
        assert result.exit_code == 0, result.output
        result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
        assert result.exit_code == 0, result.output

        result = CliRunner().invoke(main, ["match", "--cluster", str(workspace_dir)])
        assert result.exit_code == 0, result.output

        written = list((workspace_dir / "matches").glob("*.matches"))
        assert len(written) == 1, written
        assert written[0].stem.endswith("-clusters")
        assert not (workspace_dir / "tvg-matches").exists()


class TestDerivePairsCli:
    """`sfm match --derive-pairs`: the verified pairwise+TVG boundary file."""

    def _cluster_workspace(self, images: list[Path]) -> tuple[Path, Path]:
        workspace_dir = images[0].parent

        result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
        assert result.exit_code == 0, result.output
        result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
        assert result.exit_code == 0, result.output

        clusters_path = workspace_dir / "matches" / "seoul-clusters.matches"
        result = CliRunner().invoke(
            main,
            ["match", "--cluster", "--output", str(clusters_path), str(workspace_dir)],
        )
        assert result.exit_code == 0, result.output
        return workspace_dir, clusters_path

    def test_derive_pairs_end_to_end(
        self, isolated_seoul_bull_17_images: list[Path], tmp_path
    ):
        workspace_dir, clusters_path = self._cluster_workspace(
            isolated_seoul_bull_17_images
        )

        result = CliRunner().invoke(
            main, ["match", "--derive-pairs", str(clusters_path)]
        )
        assert result.exit_code == 0, result.output

        # Default output: tvg-matches/, with the "-clusters" suffix dropped.
        out_path = workspace_dir / "tvg-matches" / "seoul.matches"
        assert out_path.exists()

        from sfmtool._sfmtool.io import read_matches, verify_matches

        valid, errors = verify_matches(str(out_path))
        assert valid, errors

        derived_file = read_matches(str(out_path))
        meta = derived_file["metadata"]
        assert derived_file["has_two_view_geometries"]
        assert derived_file["tvg_metadata"]["inlier_count"] > 0
        assert not derived_file.get("has_clusters", False)
        assert meta["matching_method"] == "cluster"
        assert meta["matching_tool"] == "sfmtool"
        assert meta["image_count"] == 17
        assert meta["image_pair_count"] > 0
        assert meta["match_count"] > 0
        # The matcher's own options ride along, and a provenance record names
        # the clusters file this was derived from.
        assert meta["matching_options"]["mode"] == "background-floor"
        provenance = meta["matching_options"]["derived_pairs"]
        assert provenance["source_path"].endswith("seoul-clusters.matches")

        from sfmtool._sfmtool.io import MatchesFile

        assert provenance["source_content_xxh128"] == (
            MatchesFile(clusters_path).content_xxh128
        )

        cluster_data = read_matches(str(clusters_path))
        # Both files list the same images in the same order, so indices are
        # directly comparable.
        assert list(cluster_data["image_names"]) == list(derived_file["image_names"])

        # Verification culls pairs below COLMAP's min_num_matches (15), so the
        # verified pairs are a subset of the cluster expansion with identical
        # match sets on every surviving pair.
        from sfmtool.feature_match import pairs_from_matches

        def _per_pair_matches(pairs_dict):
            out = {}
            offset = 0
            for (i, j), count in zip(
                pairs_dict["image_index_pairs"], pairs_dict["match_counts"]
            ):
                count = int(count)
                block = pairs_dict["match_feature_indexes"][offset : offset + count]
                out[(int(i), int(j))] = block[np.lexsort((block[:, 1], block[:, 0]))]
                offset += count
            return out

        expansion = _per_pair_matches(pairs_from_matches(cluster_data))
        verified = _per_pair_matches(derived_file)
        assert len(verified) > 0
        assert set(verified) <= set(expansion)
        for key, verified_block in verified.items():
            np.testing.assert_array_equal(expansion[key], verified_block)
        for key in set(expansion) - set(verified):
            assert len(expansion[key]) < 15, key

        # The derived file feeds the COLMAP DB consumer unchanged.
        db_path = tmp_path / "colmap.db"
        result = CliRunner().invoke(main, ["to-colmap-db", str(out_path), str(db_path)])
        assert result.exit_code == 0, result.output
        assert db_path.exists()

    def test_derive_pairs_explicit_output(
        self, isolated_seoul_bull_17_images: list[Path], tmp_path
    ):
        _, clusters_path = self._cluster_workspace(isolated_seoul_bull_17_images)

        out_path = tmp_path / "explicit.matches"
        result = CliRunner().invoke(
            main,
            ["match", "--derive-pairs", str(clusters_path), "-o", str(out_path)],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()

        from sfmtool._sfmtool.io import read_matches

        assert read_matches(str(out_path))["has_two_view_geometries"]

    def test_derive_pairs_rejects_pairs_only_input(
        self, isolated_seoul_bull_17_images: list[Path]
    ):
        workspace_dir = isolated_seoul_bull_17_images[0].parent

        result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
        assert result.exit_code == 0, result.output
        result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
        assert result.exit_code == 0, result.output

        pairs_path = workspace_dir / "exhaustive.matches"
        result = CliRunner().invoke(
            main,
            ["match", "--exhaustive", "-o", str(pairs_path), str(workspace_dir)],
        )
        assert result.exit_code == 0, result.output

        result = CliRunner().invoke(main, ["match", "--derive-pairs", str(pairs_path)])
        assert result.exit_code != 0
        assert "stores no clusters" in result.output

    def test_derive_pairs_rejects_image_paths(self, isolated_seoul_bull_image: Path):
        result = CliRunner().invoke(
            main, ["match", "--derive-pairs", str(isolated_seoul_bull_image.parent)]
        )
        assert result.exit_code != 0
        assert "exactly one clusters-bearing .matches file" in result.output

    def test_derive_pairs_rejects_other_method(self, isolated_seoul_bull_image: Path):
        result = CliRunner().invoke(
            main,
            [
                "match",
                "--derive-pairs",
                "--cluster",
                str(isolated_seoul_bull_image),
            ],
        )
        assert result.exit_code != 0
        assert "Cannot specify more than one matching method" in result.output

    def test_derive_pairs_rejects_matcher_options(
        self, isolated_seoul_bull_image: Path
    ):
        result = CliRunner().invoke(
            main,
            [
                "match",
                "--derive-pairs",
                "--cluster-d",
                "20",
                str(isolated_seoul_bull_image),
            ],
        )
        assert result.exit_code != 0
        assert "--cluster-d only applies to --cluster matching" in result.output

    def test_derive_pairs_rejects_image_set_options(
        self, isolated_seoul_bull_image: Path
    ):
        result = CliRunner().invoke(
            main,
            [
                "match",
                "--derive-pairs",
                "--max-features",
                "500",
                str(isolated_seoul_bull_image),
            ],
        )
        assert result.exit_code != 0
        assert "--max-features" in result.output
        assert "reads its image set from the clusters file" in result.output
