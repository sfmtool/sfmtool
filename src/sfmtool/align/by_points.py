# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Point-based alignment for SfM reconstructions."""

import numpy as np

from .core import AlignmentResult, estimate_alignment
from .._point_correspondence import (
    find_point_correspondences,
    find_point_correspondences_by_coordinate,
)
from .._sfmtool.reconstruction import SfmrReconstruction


def _matches_by_feature_index(
    source_recon: SfmrReconstruction, target_recon: SfmrReconstruction
) -> bool:
    """Whether point alignment matches the two reconstructions' observations by
    feature index.

    Only when both have feature indexes (``sift_files``). Otherwise, when either
    is ``embedded_patches``, observations are matched by keypoint position: the
    inline keypoints of an ``embedded_patches`` side, the ``.sift`` positions of
    a ``sift_files`` one. This is the rule ``align_reconstructions`` in
    ``sfmtool-core`` applies for the SfM Explorer.
    """
    return (
        source_recon.track_feature_indexes is not None
        and target_recon.track_feature_indexes is not None
    )


def estimate_alignment_from_points(
    source_recon: SfmrReconstruction,
    targets: list[tuple[SfmrReconstruction, list[tuple[int, int]]]],
    min_points: int = 10,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_percentile: float = 95.0,
) -> AlignmentResult:
    """Estimate the similarity transform onto ``targets`` from corresponding 3D points.

    ``targets`` holds ``(target_recon, shared_images)`` rows, every target in
    the same coordinate frame, with ``shared_images`` the
    ``(source image index, target image index)`` pairs of the source's images
    with that target's. For each target, the 3D points that appear in both
    reconstructions are found through their observations in the shared images,
    matched by feature index or by keypoint position (see
    :func:`_matches_by_feature_index`). The correspondences of all targets are
    fitted together, by a least-squares similarity fit with optional RANSAC
    outlier rejection over all of them.

    A target that yields no correspondence (its finder raises ``ValueError``)
    adds none; when no target yields one, the targets' errors are raised
    together as one ``ValueError``.

    Args:
        source_recon: Source SfmrReconstruction object
        targets: ``(target_recon, shared_images)`` rows, in one frame
        min_points: Minimum number of point correspondences required
        use_ransac: Apply RANSAC for outlier rejection
        ransac_iterations: Number of RANSAC iterations
        ransac_percentile: Percentile of correspondence distances to use as RANSAC
            threshold (default: 95.0).

    Returns:
        AlignmentResult with transform from source to the targets' frame. Its
        ``n_point_correspondences_per_target`` holds each target's count.
    """
    source_parts = []
    target_parts = []
    per_target = []
    errors = []
    for target_recon, shared_images in targets:
        if _matches_by_feature_index(source_recon, target_recon):
            find = find_point_correspondences
        else:
            find = find_point_correspondences_by_coordinate
        try:
            _, source_found, target_found = find(
                source_recon, target_recon, shared_images
            )
        except ValueError as e:
            errors.append(str(e))
            per_target.append(0)
            continue
        source_parts.append(source_found)
        target_parts.append(target_found)
        per_target.append(len(source_found))
    if not source_parts:
        raise ValueError("; ".join(errors))
    source_positions = np.concatenate(source_parts)
    target_positions = np.concatenate(target_parts)

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
        prelim_transform = estimate_alignment(source_positions, target_positions)
        transformed = prelim_transform.apply_to_points(source_positions)
        distances_for_stats = np.linalg.norm(transformed - target_positions, axis=1)

        computed_threshold = float(
            np.percentile(distances_for_stats, ransac_percentile)
        )

        from .._sfmtool.analysis import ransac_alignment

        inlier_mask = ransac_alignment(
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

    transform = estimate_alignment(source_positions, target_positions)

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
    result.n_point_correspondences_per_target = per_target
    result.n_inliers = n_inliers
    result.point_rms_error = float(rms_error)
    result.computed_threshold = computed_threshold
    result.ransac_percentile = ransac_percentile
    result.distances_for_stats = distances_for_stats

    return result


def estimate_alignment_from_points_with_logging(
    source_recon: SfmrReconstruction,
    targets: list[tuple[SfmrReconstruction, list[tuple[int, int]]]],
    target_labels: list[str],
    min_points: int = 10,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_percentile: float = 95.0,
    verbose: bool = True,
) -> AlignmentResult:
    """Estimate alignment with progress logging.

    Wrapper around estimate_alignment_from_points that provides
    detailed logging for CLI usage. ``target_labels`` names each of
    ``targets`` in that log.
    """
    import click

    if verbose:
        click.echo("  Finding 3D point correspondences...")

    result = estimate_alignment_from_points(
        source_recon=source_recon,
        targets=targets,
        min_points=min_points,
        use_ransac=use_ransac,
        ransac_iterations=ransac_iterations,
        ransac_percentile=ransac_percentile,
    )

    if verbose:
        for (target_recon, _), label, count in zip(
            targets, target_labels, result.n_point_correspondences_per_target
        ):
            how = (
                "feature index"
                if _matches_by_feature_index(source_recon, target_recon)
                else "keypoint position"
            )
            click.echo(f"    With {label} (by {how}): {count}")
        click.echo(f"    Found {result.n_point_correspondences} point correspondences")

        if use_ransac and result.distances_for_stats is not None:
            from .._histogram_utils import print_histogram

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
