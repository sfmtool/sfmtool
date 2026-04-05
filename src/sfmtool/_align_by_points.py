# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Point-based alignment for SfM reconstructions."""

import numpy as np

from ._align import AlignmentResult, kabsch_algorithm
from ._point_correspondence import find_point_correspondences
from ._sfmtool import SfmrReconstruction


def estimate_alignment_from_points(
    source_recon: SfmrReconstruction,
    target_recon: SfmrReconstruction,
    shared_images: list[tuple[int, int]],
    min_points: int = 10,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_percentile: float = 95.0,
) -> AlignmentResult:
    """Estimate SE3 transform by aligning corresponding 3D points.

    Finds 3D points that appear in both reconstructions (via shared feature
    observations), then computes the optimal similarity transform using
    the Kabsch algorithm with optional RANSAC outlier rejection.

    Args:
        source_recon: Source SfmrReconstruction object
        target_recon: Target SfmrReconstruction object
        shared_images: List of (source_img_idx, target_img_idx) pairs
        min_points: Minimum number of point correspondences required
        use_ransac: Apply RANSAC for outlier rejection
        ransac_iterations: Number of RANSAC iterations
        ransac_percentile: Percentile of correspondence distances to use as RANSAC
            threshold (default: 95.0).

    Returns:
        AlignmentResult with transform from source to target
    """
    correspondences, source_positions, target_positions = find_point_correspondences(
        source_recon, target_recon, shared_images
    )

    n_points = len(source_positions)
    if n_points < min_points:
        raise ValueError(
            f"Insufficient point correspondences: found {n_points}, "
            f"need at least {min_points}. "
            f"This may indicate incompatible reconstructions or "
            f"insufficient shared features."
        )

    computed_threshold = None
    distances_for_stats = None

    if use_ransac and n_points > min_points:
        prelim_transform = kabsch_algorithm(source_positions, target_positions)
        transformed = prelim_transform.apply_to_points(source_positions)
        distances_for_stats = np.linalg.norm(transformed - target_positions, axis=1)

        computed_threshold = float(
            np.percentile(distances_for_stats, ransac_percentile)
        )

        from ._sfmtool import ransac_alignment_rs

        inlier_mask = ransac_alignment_rs(
            source_positions,
            target_positions,
            max_iterations=ransac_iterations,
            threshold=computed_threshold,
            seed=42,
        )
        source_positions = source_positions[inlier_mask]
        target_positions = target_positions[inlier_mask]

        n_inliers = np.sum(inlier_mask)

        if n_inliers < min_points:
            raise ValueError(
                f"RANSAC rejected too many points: {n_inliers} inliers "
                f"remaining (rejected {n_points - n_inliers} outliers), "
                f"need at least {min_points}."
            )
    else:
        n_inliers = n_points

    transform = kabsch_algorithm(source_positions, target_positions)

    transformed_source = transform.apply_to_points(source_positions)
    distances = np.linalg.norm(transformed_source - target_positions, axis=1)
    rms_error = np.sqrt(np.mean(distances**2))

    result = AlignmentResult(
        source_id="source",
        target_id="target",
        transform=transform,
        matches=[],
    )

    result.total_rms_error = float(rms_error)
    result.confidence = max(0.0, 1.0 - rms_error / 0.1)
    result.n_point_correspondences = n_points
    result.n_inliers = n_inliers
    result.point_rms_error = float(rms_error)
    result.computed_threshold = computed_threshold
    result.ransac_percentile = ransac_percentile
    result.distances_for_stats = distances_for_stats

    return result
