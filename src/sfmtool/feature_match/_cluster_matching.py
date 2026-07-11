# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Track-cluster matching via the per-point background floor.

Instead of matching image pairs, concatenates every image's SIFT descriptors
into one corpus and clusters it directly: each descriptor keeps the cross-image
neighbours within ``alpha x`` its ``d``-th-nearest distance (its background
floor), and density-ordered seeding materializes those candidates into track
clusters — the matcher's primary output. A derived view expands the clusters
into the per-image-pair matches the existing ``.matches`` pipeline consumes.

The heavy lifting (kd-tree forest build, k-NN query, clustering, pair
expansion) happens in Rust; see ``specs/core/track-cluster-matching.md``.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from .._sfmtool.matching import (
    background_floor_clusters as _rust_background_floor_clusters,
    clusters_to_pair_matches as _rust_clusters_to_pair_matches,
)
from ..sift.file import SiftReader


class ClusterSet(NamedTuple):
    """Materialized track clusters in CSR form (the primary artefact)."""

    cluster_starts: np.ndarray  # (C+1,) uint32
    member_images: np.ndarray  # (M,) uint32
    member_features: np.ndarray  # (M,) uint32


class PairArrays(NamedTuple):
    """Per-image-pair match arrays in the `.matches` parallel-array form."""

    image_index_pairs: np.ndarray  # (P, 2) uint32, i < j, sorted
    match_counts: np.ndarray  # (P,) uint32
    match_feature_indexes: np.ndarray  # (M, 2) uint32, grouped by pair
    match_descriptor_distances: np.ndarray  # (M,) float32, Euclidean L2


def cluster_match(
    image_paths: list[Path],
    sift_paths: list[Path],
    *,
    d: int = 10,
    alpha: float = 0.8,
    min_size: int = 2,
    preset: str = "accurate",
    max_feature_count: Optional[int] = None,
) -> tuple[ClusterSet, PairArrays]:
    """Run the background-floor matcher over every image's SIFT descriptors.

    Loads each image's descriptors (capped at max_feature_count to match the
    feature indices used downstream), concatenates them into one (N, 128) uint8
    corpus with a CSR image_starts array, and calls
    sfmtool.background_floor_clusters followed by
    sfmtool.clusters_to_pair_matches. Returns both: the clusters
    (cluster_starts, member_images, member_features — the primary artefact) and
    the four parallel pair arrays (image_index_pairs, match_counts,
    match_feature_indexes, match_descriptor_distances) for the .matches writer.
    """
    assert len(image_paths) == len(sift_paths)

    def read_one(sift_path):
        with SiftReader(sift_path) as reader:
            return reader.read_descriptors(count=max_feature_count)

    # Decode the .sift descriptor blocks through a thread pool (the reader
    # releases the GIL for the ZIP/zstd work); map preserves input order so
    # the corpus concatenation order is unchanged.
    with ThreadPoolExecutor() as pool:
        descriptors = list(pool.map(read_one, sift_paths))

    image_starts = np.zeros(len(descriptors) + 1, dtype=np.uint32)
    image_starts[1:] = np.cumsum([len(desc) for desc in descriptors])
    if descriptors:
        corpus = np.ascontiguousarray(np.concatenate(descriptors, axis=0))
    else:
        corpus = np.zeros((0, 128), dtype=np.uint8)

    clusters = ClusterSet(
        *_rust_background_floor_clusters(
            corpus,
            image_starts,
            d=d,
            alpha=alpha,
            min_size=min_size,
            preset=preset,
        )
    )
    pairs = PairArrays(
        *_rust_clusters_to_pair_matches(
            clusters.cluster_starts,
            clusters.member_images,
            clusters.member_features,
            corpus,
            image_starts,
        )
    )
    return clusters, pairs
