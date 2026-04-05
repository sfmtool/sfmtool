# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for finding corresponding 3D points between SfM reconstructions."""

import numpy as np

from ._sfmtool import SfmrReconstruction, find_point_correspondences_py


def find_point_correspondences(
    source_recon: SfmrReconstruction,
    target_recon: SfmrReconstruction,
    shared_images: list[tuple[int, int]],
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    """Find corresponding 3D points between two reconstructions.

    Uses shared images to find features that appear in both reconstructions
    (at the same feature index), then maps those features to their respective
    3D points to establish point-to-point correspondences.

    Args:
        source_recon: Source SfmrReconstruction object
        target_recon: Target SfmrReconstruction object
        shared_images: List of (source_img_idx, target_img_idx) pairs

    Returns:
        Tuple of:
        - correspondences: Dict mapping source_point_id -> target_point_id
        - source_positions: (N, 3) array of source point positions
        - target_positions: (N, 3) array of target point positions

    Raises:
        ValueError: If no point correspondences are found
    """
    shared_src = np.array([s for s, _ in shared_images], dtype=np.uint32)
    shared_tgt = np.array([t for _, t in shared_images], dtype=np.uint32)

    source_ids, target_ids = find_point_correspondences_py(
        source_recon.track_image_indexes.astype(np.uint32),
        source_recon.track_feature_indexes.astype(np.uint32),
        source_recon.track_point_ids.astype(np.uint32),
        target_recon.track_image_indexes.astype(np.uint32),
        target_recon.track_feature_indexes.astype(np.uint32),
        target_recon.track_point_ids.astype(np.uint32),
        shared_src,
        shared_tgt,
    )

    if len(source_ids) == 0:
        raise ValueError(
            "No point correspondences found. "
            "This may indicate no shared features or incompatible reconstructions."
        )

    correspondences = dict(zip(source_ids.tolist(), target_ids.tolist()))
    source_positions = source_recon.positions[source_ids]
    target_positions = target_recon.positions[target_ids]
    return correspondences, source_positions, target_positions
