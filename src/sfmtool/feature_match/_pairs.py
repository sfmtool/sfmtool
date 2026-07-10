# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Derived pairwise view of a `.matches` dict.

`pairs_from_matches` is the single place consumers obtain the four pairwise
match arrays from `read_matches` output. Pairwise-backbone files return their
stored arrays verbatim; cluster-backbone files are expanded through the
canonical `sfmtool._sfmtool.matching.clusters_to_pair_matches` expansion, with
descriptor distances recomputed from the referenced `.sift` files when the
caller supplies them (and NaN otherwise — the expansion itself needs no
descriptors).
"""

from pathlib import Path

import numpy as np

#: SIFT descriptor width; the corpus handed to the expansion is (N, 128) uint8.
_DESCRIPTOR_DIM = 128


def pairs_from_matches(data: dict, sift_paths: list[Path] | None = None) -> dict:
    """Return the four pairwise match arrays for a ``read_matches`` dict.

    Args:
        data: A dict as returned by ``sfmtool._sfmtool.io.read_matches`` (or a
            compatible dict carrying either the pairwise or the cluster
            backbone).
        sift_paths: Optional per-image ``.sift`` paths, parallel to
            ``data["image_names"]``. Only consulted for cluster-bearing files,
            where the stored form carries no descriptor distances: when given,
            distances are recomputed from the first ``feature_counts[i]``
            descriptors of each file (the cap used during matching, so member
            indices line up); when omitted, distances are NaN.

    Returns:
        Dict with keys ``image_index_pairs`` (P, 2) uint32,
        ``match_counts`` (P,) uint32, ``match_feature_indexes`` (M, 2) uint32,
        and ``match_descriptor_distances`` (M,) float32. For pairwise-backbone
        files these are the stored arrays, verbatim; for cluster-backbone
        files they are the canonical cluster expansion.
    """
    if not data.get("has_clusters", False):
        return {
            "image_index_pairs": data["image_index_pairs"],
            "match_counts": data["match_counts"],
            "match_feature_indexes": data["match_feature_indexes"],
            "match_descriptor_distances": data["match_descriptor_distances"],
        }

    from .._sfmtool.matching import clusters_to_pair_matches

    feature_counts = np.asarray(data["feature_counts"], dtype=np.uint32)
    image_starts = np.zeros(len(feature_counts) + 1, dtype=np.uint32)
    np.cumsum(feature_counts, out=image_starts[1:])
    total_features = int(image_starts[-1])

    if sift_paths is not None:
        if len(sift_paths) != len(feature_counts):
            raise ValueError(
                f"sift_paths has {len(sift_paths)} entries for "
                f"{len(feature_counts)} images"
            )
        from ..sift.file import SiftReader

        descriptors = []
        for i, sift_path in enumerate(sift_paths):
            count = int(feature_counts[i])
            with SiftReader(sift_path) as reader:
                desc = reader.read_descriptors(count=count)
            if len(desc) < count:
                raise ValueError(
                    f"{sift_path}: has {len(desc)} descriptors but the "
                    f".matches images section records {count} features"
                )
            descriptors.append(desc[:count])
        if descriptors:
            corpus = np.ascontiguousarray(np.concatenate(descriptors, axis=0))
        else:
            corpus = np.zeros((0, _DESCRIPTOR_DIM), dtype=np.uint8)
    else:
        # The expansion only needs the corpus for distances; a zero corpus
        # yields the same pairs/indexes, and the distances are replaced below.
        corpus = np.zeros((total_features, _DESCRIPTOR_DIM), dtype=np.uint8)

    (
        image_index_pairs,
        match_counts,
        match_feature_indexes,
        match_descriptor_distances,
    ) = clusters_to_pair_matches(
        data["cluster_starts"],
        data["member_images"],
        data["member_features"],
        corpus,
        image_starts,
    )
    if sift_paths is None:
        match_descriptor_distances = np.full(
            len(match_feature_indexes), np.nan, dtype=np.float32
        )
    return {
        "image_index_pairs": image_index_pairs,
        "match_counts": match_counts,
        "match_feature_indexes": match_feature_indexes,
        "match_descriptor_distances": match_descriptor_distances,
    }
