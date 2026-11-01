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
    reconstructions, then groups the pairs transitively: a group is a
    connected component of the graph whose edges are the pairs. A pair in
    which either point is at infinity is dropped, because its position is a
    direction and would distort the distance statistics.

    Returns:
        List of correspondence groups, where each group is a list of (recon_idx, point_id)
        tuples that represent the same physical 3D point. The groups come in
        the order in which their first point appears in the pairs, and the
        points of a group in the order in which they first appear.
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

    # Number the points of all the reconstructions together: point p of
    # reconstruction r is node offsets[r] + p.
    point_counts = [r.point_count for r in reconstructions]
    offsets = np.concatenate([[0], np.cumsum(point_counts)]).astype(np.int64)

    # Find pairwise correspondences, each an edge between two nodes
    first_nodes = []
    second_nodes = []
    for (recon_idx1, recon_idx2), shared_images in pair_shared_images.items():
        ids_1, ids_2, _ = _finite_point_pairs(
            reconstructions[recon_idx1],
            reconstructions[recon_idx2],
            shared_images,
        )
        first_nodes.append(offsets[recon_idx1] + np.asarray(ids_1, dtype=np.int64))
        second_nodes.append(offsets[recon_idx2] + np.asarray(ids_2, dtype=np.int64))

    empty = np.zeros(0, dtype=np.int64)
    members, group_starts = _connected_groups(
        np.concatenate([empty, *first_nodes]), np.concatenate([empty, *second_nodes])
    )
    member_recon = np.searchsorted(offsets, members, side="right") - 1
    member_point = members - offsets[member_recon]
    group_count = len(group_starts)

    click.echo(
        f"  Found {group_count} potential correspondence groups based on features"
    )

    if group_count > 0:
        positions_xyzw = [np.asarray(r.positions_xyzw) for r in reconstructions]
        group_max_distances = _group_max_distances(
            positions_xyzw, member_recon, member_point, group_starts
        )

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
        group_ends = np.append(group_starts[1:], len(members))
        recon_list = member_recon.tolist()
        point_list = member_point.tolist()
        correspondence_groups = [
            list(zip(recon_list[start:end], point_list[start:end]))
            for start, end in zip(
                group_starts[accepted_mask].tolist(),
                group_ends[accepted_mask].tolist(),
            )
        ]
        rejected_count = np.sum(~accepted_mask)
    else:
        correspondence_groups = []
        rejected_count = 0
        click.echo("  No correspondences to compute percentile from")

    click.echo(
        f"  Accepted {len(correspondence_groups)} groups, rejected {rejected_count} outliers"
    )

    return correspondence_groups


def _connected_groups(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Group the nodes joined by the edges ``first[i]``--``second[i]``.

    Each group is a connected component of the graph of the edges; a node on
    no edge is in no group. The nodes are read in the order ``first[0],
    second[0], first[1], second[1], ...``. The groups come in the order in
    which their first node is read, and the nodes of a group in the order in
    which they are first read.

    Returns:
        ``(members, group_starts)``: the nodes of every group, one group after
        another, and the index in ``members`` at which each group starts.
    """
    if len(first) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    stream = np.empty(2 * len(first), dtype=np.int64)
    stream[0::2] = first
    stream[1::2] = second
    nodes, first_read = np.unique(stream, return_index=True)
    # Renumber the nodes by when they are first read: nodes[by_reading[k]] is
    # the k-th node read.
    by_reading = np.argsort(first_read, kind="stable")
    rank = np.empty(len(nodes), dtype=np.int64)
    rank[by_reading] = np.arange(len(nodes), dtype=np.int64)
    edge_a = rank[np.searchsorted(nodes, first)]
    edge_b = rank[np.searchsorted(nodes, second)]

    # Label each renumbered node with the smallest number in its component:
    # give both ends of every edge the smaller of their labels, then replace
    # each label by the label of the node it names, until no edge joins two
    # different labels.
    labels = np.arange(len(nodes), dtype=np.int64)
    while True:
        smaller = np.minimum(labels[edge_a], labels[edge_b])
        updated = labels.copy()
        np.minimum.at(updated, edge_a, smaller)
        np.minimum.at(updated, edge_b, smaller)
        while True:
            followed = updated[updated]
            if np.array_equal(followed, updated):
                break
            updated = followed
        if np.array_equal(updated, labels):
            break
        labels = updated

    # A component's label is the number of its first node read, so a stable
    # sort by label orders the groups, and keeps each group's nodes in the
    # order they were read.
    order = np.argsort(labels, kind="stable")
    members = nodes[by_reading[order]]
    sorted_labels = labels[order]
    group_starts = np.flatnonzero(
        np.concatenate([[True], sorted_labels[1:] != sorted_labels[:-1]])
    )
    return members, group_starts


def _group_max_distances(
    positions_xyzw: list[np.ndarray],
    member_recon: np.ndarray,
    member_point: np.ndarray,
    group_starts: np.ndarray,
) -> np.ndarray:
    """The largest distance of a group's points from their centroid, per group.

    Member ``i`` is point ``member_point[i]`` of reconstruction
    ``member_recon[i]``, and ``positions_xyzw[r]`` is reconstruction ``r``'s
    ``positions_xyzw``. The members of a group are contiguous; group ``g``
    starts at member ``group_starts[g]``, and every group has at least one
    member. Every member must be a finite point: a point at infinity stores a
    direction, not a position, so one raises ``ValueError``.
    """
    member_count = len(member_recon)
    positions = np.empty((member_count, 3), dtype=np.float64)
    for recon_idx, xyzw in enumerate(positions_xyzw):
        in_recon = member_recon == recon_idx
        if not np.any(in_recon):
            continue
        rows = xyzw[member_point[in_recon]]
        if np.any(rows[:, 3] == 0.0):
            raise ValueError(
                "A correspondence group holds a point at infinity of "
                f"reconstruction {recon_idx}, which has no position"
            )
        positions[in_recon] = rows[:, :3]

    group_count = len(group_starts)
    group_sizes = np.diff(np.append(group_starts, member_count))
    member_group = np.repeat(np.arange(group_count), group_sizes)
    sums = np.column_stack(
        [
            np.bincount(member_group, weights=positions[:, axis], minlength=group_count)
            for axis in range(3)
        ]
    )
    centroids = sums / group_sizes[:, None]
    distances = np.linalg.norm(positions - centroids[member_group], axis=1)
    return np.maximum.reduceat(distances, group_starts)
