# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Find and group corresponding 3D points across multiple reconstructions."""

import click
import numpy as np

from .._histogram_utils import print_histogram
from .._point_correspondence import _finite_point_pairs
from .._sfmtool.reconstruction import SfmrReconstruction


def find_point_correspondences(
    reconstructions: list[SfmrReconstruction],
    image_mapping: dict[str, list[tuple[int, int]]],
    merge_percentile: float,
) -> list[list[tuple[int, int]]]:
    """Find corresponding 3D points across reconstructions.

    Uses the Rust-backed pairwise point correspondence finder for each pair of
    reconstructions, then groups results transitively using union-find. A pair
    in which either point is at infinity is dropped, because its position is a
    direction and would distort the distance statistics.

    Returns:
        List of correspondence groups, where each group is a list of (recon_idx, point_id)
        tuples that represent the same physical 3D point.
    """
    # Build shared image lists for each pair of reconstructions
    pair_shared_images: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for img_name, occurrences in image_mapping.items():
        if len(occurrences) < 2:
            continue
        for i, (recon_idx1, img_idx1) in enumerate(occurrences):
            for recon_idx2, img_idx2 in occurrences[i + 1 :]:
                pair_key = (recon_idx1, recon_idx2)
                if pair_key not in pair_shared_images:
                    pair_shared_images[pair_key] = []
                pair_shared_images[pair_key].append((img_idx1, img_idx2))

    # Find pairwise correspondences, then group with union-find
    potential_correspondences = {}

    for (recon_idx1, recon_idx2), shared_images in pair_shared_images.items():
        ids_1, ids_2, _ = _finite_point_pairs(
            reconstructions[recon_idx1],
            reconstructions[recon_idx2],
            shared_images,
        )

        for pt_id1, pt_id2 in zip(ids_1.tolist(), ids_2.tolist()):
            key1 = (recon_idx1, pt_id1)
            key2 = (recon_idx2, pt_id2)

            if key1 not in potential_correspondences:
                potential_correspondences[key1] = {key1}
            if key2 not in potential_correspondences:
                potential_correspondences[key2] = {key2}

            set1 = potential_correspondences[key1]
            set2 = potential_correspondences[key2]
            merged_set = set1 | set2

            for key in merged_set:
                potential_correspondences[key] = merged_set

    # Deduplicate correspondence groups
    seen = set()
    unique_groups = []

    for key, group in potential_correspondences.items():
        group_tuple = tuple(sorted(group))
        if group_tuple in seen:
            continue
        seen.add(group_tuple)
        unique_groups.append(list(group))

    click.echo(
        f"  Found {len(unique_groups)} potential correspondence groups based on features"
    )

    # Compute distances for all multi-point groups
    group_max_distances = np.array(
        [
            _compute_group_max_distance(reconstructions, group)
            for group in unique_groups
            if len(group) >= 2
        ]
    )

    if len(group_max_distances) > 0:
        click.echo("\n  Correspondence distance statistics:")
        click.echo(f"    Min:  {np.min(group_max_distances):.6f}")
        click.echo(f"    Max:  {np.max(group_max_distances):.6f}")
        click.echo(f"    Mean: {np.mean(group_max_distances):.6f}")
        click.echo(f"    Median: {np.median(group_max_distances):.6f}")

        percentiles = [50, 75, 90, 95, 99]
        click.echo("\n  Percentiles:")
        for p in percentiles:
            val = np.percentile(group_max_distances, p)
            click.echo(f"    {p:3d}th: {val:.6f}")

        click.echo("\n  Distance distribution (max distance from centroid per group):")
        print_histogram(
            group_max_distances,
            title="Max distances",
            num_buckets=60,
            show_stats=False,
        )
        click.echo()

        computed_threshold = np.percentile(group_max_distances, merge_percentile)
        click.echo(
            f"  Using {merge_percentile:.1f}th percentile as threshold: {computed_threshold:.6f}"
        )

        accepted_mask = group_max_distances <= computed_threshold
        correspondence_groups = [
            group
            for group, accept in zip(unique_groups, accepted_mask)
            if accept and len(group) >= 2
        ]
        rejected_count = np.sum(~accepted_mask)
    else:
        correspondence_groups = [group for group in unique_groups if len(group) < 2]
        rejected_count = 0
        click.echo("  No correspondences to compute percentile from")

    click.echo(
        f"  Accepted {len(correspondence_groups)} groups, rejected {rejected_count} outliers"
    )

    return correspondence_groups


def _compute_group_max_distance(
    reconstructions: list[SfmrReconstruction],
    group: list[tuple[int, int]],
) -> float:
    """Compute the maximum distance from centroid for a correspondence group."""
    if len(group) < 2:
        return 0.0

    positions = []
    for recon_idx, point_id in group:
        positions.append(reconstructions[recon_idx].positions[point_id])

    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)

    return np.max(distances)
