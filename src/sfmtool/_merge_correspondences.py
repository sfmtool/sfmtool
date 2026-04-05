# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Find and merge corresponding 3D points across multiple reconstructions."""

from collections import defaultdict

import click
import numpy as np

from ._histogram_utils import print_histogram
from ._sfmtool import SfmrReconstruction, find_point_correspondences_py


def _find_pairwise_correspondences(
    recon_a: SfmrReconstruction,
    recon_b: SfmrReconstruction,
    shared_images: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Find corresponding point IDs between two reconstructions using Rust backend."""
    shared_a = np.array([s for s, _ in shared_images], dtype=np.uint32)
    shared_b = np.array([t for _, t in shared_images], dtype=np.uint32)

    return find_point_correspondences_py(
        recon_a.track_image_indexes.astype(np.uint32),
        recon_a.track_feature_indexes.astype(np.uint32),
        recon_a.track_point_ids.astype(np.uint32),
        recon_b.track_image_indexes.astype(np.uint32),
        recon_b.track_feature_indexes.astype(np.uint32),
        recon_b.track_point_ids.astype(np.uint32),
        shared_a,
        shared_b,
    )


def find_point_correspondences(
    reconstructions: list[SfmrReconstruction],
    image_mapping: dict[str, list[tuple[int, int]]],
    merge_percentile: float,
) -> list[list[tuple[int, int]]]:
    """Find corresponding 3D points across reconstructions.

    Uses the Rust-backed pairwise point correspondence finder for each pair of
    reconstructions, then groups results transitively using union-find.

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
        ids_1, ids_2 = _find_pairwise_correspondences(
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


def merge_points_and_tracks(
    reconstructions: list[SfmrReconstruction],
    point_correspondences: list[list[tuple[int, int]]],
    image_mapping: dict[str, list[tuple[int, int]]],
) -> tuple[dict, dict]:
    """Merge 3D points and tracks from multiple reconstructions.

    Points that correspond across reconstructions are merged by averaging their
    positions. A union-find structure ensures transitive merging when multiple
    points share observations.
    """
    # Build reverse image mapping: (recon_idx, old_img_idx) -> merged_img_idx
    reverse_image_mapping = {}
    for merged_idx, (img_name, occurrences) in enumerate(image_mapping.items()):
        for recon_idx, old_idx in occurrences:
            if (recon_idx, old_idx) not in reverse_image_mapping:
                reverse_image_mapping[(recon_idx, old_idx)] = merged_idx

    merged_positions = []
    merged_colors = []
    merged_errors = []
    merged_track_image_indexes = []
    merged_track_feature_indexes = []
    merged_track_point_ids = []

    merged_points_set = set()
    obs_to_points = defaultdict(list)
    temp_points = []

    # Step 1: Create temporary points from correspondence groups
    for group in point_correspondences:
        positions = []
        colors = []
        errors = []
        observations = []

        for recon_idx, point_id in group:
            recon = reconstructions[recon_idx]
            positions.append(recon.positions[point_id])
            colors.append(recon.colors[point_id])
            errors.append(recon.errors[point_id])
            merged_points_set.add((recon_idx, point_id))

            mask = recon.track_point_ids == point_id
            for img_idx, feat_idx in zip(
                recon.track_image_indexes[mask],
                recon.track_feature_indexes[mask],
            ):
                if (recon_idx, img_idx) in reverse_image_mapping:
                    merged_img_idx = reverse_image_mapping[(recon_idx, img_idx)]
                    observations.append((merged_img_idx, feat_idx))

        temp_point_id = len(temp_points)
        temp_points.append(
            {
                "position": np.mean(positions, axis=0),
                "color": np.mean(colors, axis=0).astype(np.uint8),
                "error": np.mean(errors),
                "observations": set(observations),
            }
        )

        for obs in temp_points[temp_point_id]["observations"]:
            obs_to_points[obs].append(temp_point_id)

    # Step 2: Add unique points (not in any correspondence group)
    for recon_idx, recon in enumerate(reconstructions):
        for point_id in range(recon.point_count):
            if (recon_idx, point_id) in merged_points_set:
                continue

            observations = []
            mask = recon.track_point_ids == point_id
            for img_idx, feat_idx in zip(
                recon.track_image_indexes[mask],
                recon.track_feature_indexes[mask],
            ):
                if (recon_idx, img_idx) in reverse_image_mapping:
                    merged_img_idx = reverse_image_mapping[(recon_idx, img_idx)]
                    observations.append((merged_img_idx, feat_idx))

            temp_point_id = len(temp_points)
            temp_points.append(
                {
                    "position": recon.positions[point_id],
                    "color": recon.colors[point_id],
                    "error": recon.errors[point_id],
                    "observations": set(observations),
                }
            )

            for obs in temp_points[temp_point_id]["observations"]:
                obs_to_points[obs].append(temp_point_id)

    # Step 3: Union-find to merge points that share observations
    parent = list(range(len(temp_points)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for obs, point_ids in obs_to_points.items():
        if len(point_ids) > 1:
            for i in range(1, len(point_ids)):
                union(point_ids[0], point_ids[i])

    # Step 4: Group points by their root
    point_groups = defaultdict(list)
    for i in range(len(temp_points)):
        root = find(i)
        point_groups[root].append(i)

    # Step 5: Create final merged points
    for root, point_ids in point_groups.items():
        positions = [temp_points[i]["position"] for i in point_ids]
        colors = [temp_points[i]["color"] for i in point_ids]
        errors = [temp_points[i]["error"] for i in point_ids]
        all_observations = set()
        for i in point_ids:
            all_observations.update(temp_points[i]["observations"])

        merged_point_id = len(merged_positions)
        merged_positions.append(np.mean(positions, axis=0))
        merged_colors.append(np.mean(colors, axis=0).astype(np.uint8))
        merged_errors.append(np.mean(errors))

        for merged_img_idx, feat_idx in all_observations:
            merged_track_image_indexes.append(merged_img_idx)
            merged_track_feature_indexes.append(feat_idx)
            merged_track_point_ids.append(merged_point_id)

    merged_points = {
        "positions": np.array(merged_positions),
        "colors": np.array(merged_colors, dtype=np.uint8),
        "errors": np.array(merged_errors),
    }

    merged_tracks = {
        "image_indexes": np.array(merged_track_image_indexes, dtype=np.int32),
        "feature_indexes": np.array(merged_track_feature_indexes, dtype=np.int32),
        "point_ids": np.array(merged_track_point_ids, dtype=np.int32),
    }

    return merged_points, merged_tracks
