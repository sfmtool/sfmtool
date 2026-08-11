# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `sfmtool.feature_match.pairs_from_matches` — the single
derived-pairs view over `read_matches` dicts (stored pairs verbatim, or the
canonical cluster expansion)."""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.feature_match import pairs_from_matches
from sfmtool.feature_match._cluster_matching import cluster_match
from sfmtool.sift.file import SiftReader, write_sift

N_IMAGES = 4
N_POINTS = 5
N_BACKGROUND = 30
DIM = 128


def _write_synthetic_sift(path: Path, descriptors: np.ndarray, image_name: str):
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
    """N_IMAGES synthetic .sift files with N_POINTS planted cross-image points."""
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


def _cluster_data_dict(
    clusters, feature_counts: np.ndarray, image_names: list[str]
) -> dict:
    """Minimal cluster-backbone dict, as `read_matches` shapes it."""
    return {
        "has_clusters": True,
        "image_names": image_names,
        "feature_counts": feature_counts,
        "cluster_starts": clusters.cluster_starts,
        "member_images": clusters.member_images,
        "member_features": clusters.member_features,
    }


def _canonical_expansion(clusters, sift_paths, feature_counts):
    """The expansion straight through `clusters_to_pair_matches`."""
    from sfmtool._sfmtool.matching import clusters_to_pair_matches

    descriptors = []
    for sp, count in zip(sift_paths, feature_counts):
        with SiftReader(sp) as reader:
            descriptors.append(reader.read_descriptors(count=int(count)))
    corpus = np.ascontiguousarray(np.concatenate(descriptors, axis=0))
    image_starts = np.zeros(len(feature_counts) + 1, dtype=np.uint32)
    np.cumsum(feature_counts, out=image_starts[1:])
    return clusters_to_pair_matches(
        clusters.cluster_starts,
        clusters.member_images,
        clusters.member_features,
        corpus,
        image_starts,
    )


def test_stored_pairs_passthrough():
    """Pairwise-backbone dicts return the stored arrays verbatim."""
    data = {
        "has_clusters": False,
        "image_index_pairs": np.array([[0, 1], [0, 2]], dtype=np.uint32),
        "match_counts": np.array([2, 1], dtype=np.uint32),
        "match_feature_indexes": np.array([[0, 3], [1, 4], [2, 5]], dtype=np.uint32),
        "match_descriptor_distances": np.array([1.5, 2.5, 3.5], dtype=np.float32),
    }
    result = pairs_from_matches(data)
    for key in (
        "image_index_pairs",
        "match_counts",
        "match_feature_indexes",
        "match_descriptor_distances",
    ):
        assert result[key] is data[key]
        np.testing.assert_array_equal(result[key], data[key])

    # A dict with no has_clusters key at all (version <= 2 file) also passes
    # straight through.
    del data["has_clusters"]
    result = pairs_from_matches(data)
    assert result["match_counts"] is data["match_counts"]


def test_cluster_expansion_equals_canonical(tmp_path):
    """The helper's expansion equals clusters_to_pair_matches on the same
    clusters, distances included, when sift_paths is given."""
    image_paths, sift_paths = _synthetic_sift_set(tmp_path)
    clusters, _ = cluster_match(image_paths, sift_paths, d=16)
    feature_counts = np.full(N_IMAGES, N_POINTS + N_BACKGROUND, dtype=np.uint32)
    data = _cluster_data_dict(clusters, feature_counts, [p.name for p in image_paths])

    result = pairs_from_matches(data, sift_paths=sift_paths)
    pairs, counts, feats, dists = _canonical_expansion(
        clusters, sift_paths, feature_counts
    )

    np.testing.assert_array_equal(result["image_index_pairs"], pairs)
    np.testing.assert_array_equal(result["match_counts"], counts)
    np.testing.assert_array_equal(result["match_feature_indexes"], feats)
    np.testing.assert_array_equal(result["match_descriptor_distances"], dists)
    assert np.isfinite(result["match_descriptor_distances"]).all()
    assert result["match_descriptor_distances"].dtype == np.float32


def test_cluster_expansion_nan_distances_without_sift(tmp_path):
    """Without sift_paths the index arrays are identical and every distance
    is NaN."""
    image_paths, sift_paths = _synthetic_sift_set(tmp_path)
    clusters, _ = cluster_match(image_paths, sift_paths, d=16)
    feature_counts = np.full(N_IMAGES, N_POINTS + N_BACKGROUND, dtype=np.uint32)
    data = _cluster_data_dict(clusters, feature_counts, [p.name for p in image_paths])

    result = pairs_from_matches(data)
    with_sift = pairs_from_matches(data, sift_paths=sift_paths)

    np.testing.assert_array_equal(
        result["image_index_pairs"], with_sift["image_index_pairs"]
    )
    np.testing.assert_array_equal(result["match_counts"], with_sift["match_counts"])
    np.testing.assert_array_equal(
        result["match_feature_indexes"], with_sift["match_feature_indexes"]
    )
    assert result["match_descriptor_distances"].dtype == np.float32
    assert len(result["match_descriptor_distances"]) == len(
        result["match_feature_indexes"]
    )
    assert np.isnan(result["match_descriptor_distances"]).all()


def test_feature_counts_cap_respected(tmp_path):
    """When the file's feature_counts are below the .sift files' feature
    counts (matching ran with --max-features), the helper reads only the
    capped prefix — indices line up and distances match the capped corpus."""
    cap = N_POINTS + 10
    image_paths, sift_paths = _synthetic_sift_set(tmp_path)
    clusters, _ = cluster_match(image_paths, sift_paths, d=16, max_feature_count=cap)
    feature_counts = np.full(N_IMAGES, cap, dtype=np.uint32)
    data = _cluster_data_dict(clusters, feature_counts, [p.name for p in image_paths])

    result = pairs_from_matches(data, sift_paths=sift_paths)
    pairs, counts, feats, dists = _canonical_expansion(
        clusters, sift_paths, feature_counts
    )

    assert np.all(result["match_feature_indexes"] < cap)
    np.testing.assert_array_equal(result["image_index_pairs"], pairs)
    np.testing.assert_array_equal(result["match_counts"], counts)
    np.testing.assert_array_equal(result["match_feature_indexes"], feats)
    np.testing.assert_array_equal(result["match_descriptor_distances"], dists)


def test_sift_paths_length_mismatch_rejected(tmp_path):
    image_paths, sift_paths = _synthetic_sift_set(tmp_path)
    clusters, _ = cluster_match(image_paths, sift_paths, d=16)
    feature_counts = np.full(N_IMAGES, N_POINTS + N_BACKGROUND, dtype=np.uint32)
    data = _cluster_data_dict(clusters, feature_counts, [p.name for p in image_paths])

    with pytest.raises(ValueError, match="entries for"):
        pairs_from_matches(data, sift_paths=sift_paths[:-1])
