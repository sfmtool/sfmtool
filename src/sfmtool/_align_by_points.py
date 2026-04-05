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


def estimate_alignment_from_points_with_logging(
    source_recon: SfmrReconstruction,
    target_recon: SfmrReconstruction,
    shared_images: list[tuple[int, int]],
    min_points: int = 10,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_percentile: float = 95.0,
    verbose: bool = True,
) -> AlignmentResult:
    """Estimate alignment with progress logging.

    Wrapper around estimate_alignment_from_points that provides
    detailed logging for CLI usage.
    """
    import click

    if verbose:
        click.echo("  Finding 3D point correspondences...")

    result = estimate_alignment_from_points(
        source_recon=source_recon,
        target_recon=target_recon,
        shared_images=shared_images,
        min_points=min_points,
        use_ransac=use_ransac,
        ransac_iterations=ransac_iterations,
        ransac_percentile=ransac_percentile,
    )

    if verbose:
        click.echo(f"    Found {result.n_point_correspondences} point correspondences")

        if use_ransac and result.distances_for_stats is not None:
            from ._histogram_utils import print_histogram

            distances = result.distances_for_stats
            click.echo("\n    Correspondence distance statistics:")
            click.echo(f"      Min:    {np.min(distances):.6f}")
            click.echo(f"      Max:    {np.max(distances):.6f}")
            click.echo(f"      Mean:   {np.mean(distances):.6f}")
            click.echo(f"      Median: {np.median(distances):.6f}")

            percentiles = [50, 75, 90, 95, 99]
            click.echo("\n    Percentiles:")
            for p in percentiles:
                val = np.percentile(distances, p)
                marker = (
                    " <-- threshold" if abs(p - result.ransac_percentile) < 0.1 else ""
                )
                click.echo(f"      {p:3d}th: {val:.6f}{marker}")

            click.echo()
            print_histogram(
                distances,
                title="Distance distribution",
                num_buckets=60,
                show_stats=False,
            )

            if result.computed_threshold is not None:
                click.echo(
                    f"\n    Using {result.ransac_percentile:.1f}th percentile as threshold: {result.computed_threshold:.6f}"
                )
            click.echo()

        if use_ransac:
            n_outliers = result.n_point_correspondences - result.n_inliers
            inlier_ratio = (
                result.n_inliers / result.n_point_correspondences
                if result.n_point_correspondences > 0
                else 0
            )
            click.echo(f"    RANSAC: {result.n_inliers} inliers, {n_outliers} outliers")
            click.echo(f"    Inlier ratio: {inlier_ratio * 100:.1f}%")

            if inlier_ratio < 0.5:
                click.echo(
                    click.style(
                        f"    WARNING: Low inlier ratio ({inlier_ratio * 100:.1f}%)! "
                        "Alignment quality may be poor.",
                        fg="yellow",
                        bold=True,
                    )
                )
        click.echo(f"    Point RMS error: {result.point_rms_error:.4f}")

    return result
