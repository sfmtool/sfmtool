# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Print per-image connectivity information table with motion path analysis."""

import time
from collections import defaultdict, deque
from pathlib import Path

import click
import numpy as np

from ._sfmtool import RotQuaternion, SfmrReconstruction
from ._image_pair_graph import compute_camera_directions


def _compute_camera_centers(quaternions, translations):
    """Compute world-space camera centers from quaternions and translations."""
    num_images = len(quaternions)
    camera_centers = np.zeros((num_images, 3), dtype=np.float64)
    for i in range(num_images):
        quat = RotQuaternion.from_wxyz_array(quaternions[i])
        R_cam_from_world = quat.to_rotation_matrix()
        R_world_from_cam = R_cam_from_world.T
        camera_centers[i] = -R_world_from_cam @ translations[i]
    return camera_centers


def _compute_rotation_angle(quat_a, quat_b):
    """Compute rotation angle in degrees between two RotQuaternion objects."""
    return np.degrees((quat_b * quat_a.conjugate()).angle())


def _slerp_halfway(quat_prev, quat_next):
    """Compute SLERP interpolation at t=0.5 between two RotQuaternion objects."""
    return quat_prev.slerp(quat_next, 0.5)


def _analyze_motion_path(recon, camera_centers, quaternions):
    """Analyze successive frame motion and detect discontinuities."""
    num_images = len(camera_centers)

    successive_translations = np.zeros(num_images - 1)
    successive_rotations = np.zeros(num_images - 1)

    for i in range(num_images - 1):
        successive_translations[i] = np.linalg.norm(
            camera_centers[i + 1] - camera_centers[i]
        )
        quat_i = RotQuaternion.from_wxyz_array(quaternions[i])
        quat_j = RotQuaternion.from_wxyz_array(quaternions[i + 1])
        successive_rotations[i] = _compute_rotation_angle(quat_i, quat_j)

    interpolation_trans_diffs = np.zeros(max(num_images - 2, 0))
    interpolation_rot_diffs = np.zeros(max(num_images - 2, 0))

    for idx, i in enumerate(range(1, num_images - 1)):
        center_prev = camera_centers[i - 1]
        center_curr = camera_centers[i]
        center_next = camera_centers[i + 1]

        quat_prev = RotQuaternion.from_wxyz_array(quaternions[i - 1])
        quat_curr = RotQuaternion.from_wxyz_array(quaternions[i])
        quat_next = RotQuaternion.from_wxyz_array(quaternions[i + 1])

        interpolated_center = (center_prev + center_next) / 2.0
        interpolated_quat = _slerp_halfway(quat_prev, quat_next)

        interpolation_trans_diffs[idx] = np.linalg.norm(
            center_curr - interpolated_center
        )
        interpolation_rot_diffs[idx] = _compute_rotation_angle(
            interpolated_quat, quat_curr
        )

    trans_mean = successive_translations.mean()
    trans_std = successive_translations.std()
    rot_mean = successive_rotations.mean()
    rot_std = successive_rotations.std()

    discontinuity_threshold_sigma = 2.5
    trans_threshold = trans_mean + discontinuity_threshold_sigma * trans_std
    rot_threshold = rot_mean + discontinuity_threshold_sigma * rot_std

    discontinuities = []
    for i in range(len(successive_translations)):
        is_trans = successive_translations[i] > trans_threshold
        is_rot = successive_rotations[i] > rot_threshold
        if is_trans or is_rot:
            discontinuities.append(
                {
                    "frame_pair": (i, i + 1),
                    "translation": successive_translations[i],
                    "rotation": successive_rotations[i],
                    "trans_flag": is_trans,
                    "rot_flag": is_rot,
                }
            )

    return (
        successive_translations,
        successive_rotations,
        interpolation_trans_diffs,
        interpolation_rot_diffs,
        discontinuities,
        trans_threshold,
        rot_threshold,
        discontinuity_threshold_sigma,
    )


def _print_motion_analysis(
    recon,
    successive_translations,
    successive_rotations,
    interpolation_trans_diffs,
    interpolation_rot_diffs,
    discontinuities,
    trans_threshold,
    rot_threshold,
    discontinuity_threshold_sigma,
):
    """Print motion discontinuity analysis results."""
    trans_mean = successive_translations.mean()
    trans_std = successive_translations.std()
    rot_mean = successive_rotations.mean()
    rot_std = successive_rotations.std()

    click.echo("")
    click.echo("=" * 120)
    click.echo("MOTION PATH DISCONTINUITY ANALYSIS")
    click.echo("=" * 120)
    click.echo("")
    click.echo("Statistics for successive frame motion:")
    click.echo(
        f"  Translation distance: mean={trans_mean:.3f}, std={trans_std:.3f}, threshold={trans_threshold:.3f}"
    )
    click.echo(
        f"  Rotation angle:       mean={rot_mean:.1f}\u00b0, std={rot_std:.1f}\u00b0, threshold={rot_threshold:.1f}\u00b0"
    )
    click.echo(
        f"  Threshold: {discontinuity_threshold_sigma}\u03c3 (standard deviations)"
    )
    click.echo("")

    if len(interpolation_trans_diffs) > 0:
        click.echo("Difference from linear interpolation for interior images:")
        click.echo(
            f"  Translation difference: mean={interpolation_trans_diffs.mean():.3f}, "
            f"std={interpolation_trans_diffs.std():.3f}, "
            f"max={interpolation_trans_diffs.max():.3f}"
        )
        click.echo(
            f"  Rotation difference:    mean={interpolation_rot_diffs.mean():.1f}\u00b0, "
            f"std={interpolation_rot_diffs.std():.1f}\u00b0, "
            f"max={interpolation_rot_diffs.max():.1f}\u00b0"
        )
        click.echo("")

    if discontinuities:
        click.echo(f"Found {len(discontinuities)} discontinuities:")
        click.echo("")
        for disc in discontinuities:
            i, j = disc["frame_pair"]
            trans = disc["translation"]
            rot = disc["rotation"]

            flags = []
            if disc["trans_flag"]:
                flags.append(f"TRANSLATION: {trans:.3f} (>{trans_threshold:.3f})")
            if disc["rot_flag"]:
                flags.append(f"ROTATION: {rot:.1f}\u00b0 (>{rot_threshold:.1f}\u00b0)")

            flag_str = ", ".join(flags)
            click.echo(f"  {recon.image_names[i]} \u2192 {recon.image_names[j]}")
            click.echo(f"    {flag_str}")
            click.echo(f"    Translation: {trans:.3f}, Rotation: {rot:.1f}\u00b0")
            click.echo("")
    else:
        click.echo("No significant discontinuities detected.")
        click.echo("")

    click.echo("All successive frame motions:")
    click.echo("")
    click.echo(
        f"{'From':<30} {'To':<30} {'Trans Dist':>12} {'Rot Angle':>12} {'Flags':<20}"
    )
    click.echo("-" * 120)

    for i in range(len(successive_translations)):
        trans = successive_translations[i]
        rot = successive_rotations[i]

        flags = []
        if trans > trans_threshold:
            flags.append("TRANS!")
        if rot > rot_threshold:
            flags.append("ROT!")
        flag_str = " ".join(flags) if flags else ""

        from_name = (
            recon.image_names[i]
            if len(recon.image_names[i]) <= 28
            else "..." + recon.image_names[i][-25:]
        )
        to_name = (
            recon.image_names[i + 1]
            if len(recon.image_names[i + 1]) <= 28
            else "..." + recon.image_names[i + 1][-25:]
        )

        click.echo(
            f"{from_name:<30} {to_name:<30} {trans:>12.3f} {rot:>12.1f}\u00b0 {flag_str:<20}"
        )

    click.echo("")

    if len(interpolation_trans_diffs) > 0:
        num_images = len(successive_translations) + 1
        click.echo("Interior image differences from linear interpolation:")
        click.echo(
            "(Difference from linear interpolation between previous and next poses)"
        )
        click.echo("")
        click.echo(f"{'Image':<50} {'Trans Diff':>12} {'Rot Diff':>12}")
        click.echo("-" * 120)

        for idx, i in enumerate(range(1, num_images - 1)):
            trans_diff = interpolation_trans_diffs[idx]
            rot_diff = interpolation_rot_diffs[idx]

            img_name = (
                recon.image_names[i]
                if len(recon.image_names[i]) <= 48
                else "..." + recon.image_names[i][-45:]
            )

            click.echo(f"{img_name:<50} {trans_diff:>12.3f} {rot_diff:>12.1f}\u00b0")

        click.echo("")

    click.echo("=" * 120)
    click.echo("")


def print_images_table(recon: SfmrReconstruction, recon_name: str | None = None):
    """Print a detailed per-image connectivity table."""
    overall_start = time.perf_counter()

    if recon_name is None:
        recon_name = recon.source_metadata.get("source_path", "reconstruction")
        if "/" in recon_name or "\\" in recon_name:
            recon_name = Path(recon_name).name

    step_start = time.perf_counter()
    num_images = recon.image_count
    quaternions = recon.quaternions_wxyz
    translations = recon.translations
    image_indexes = recon.track_image_indexes
    points3d_indexes = recon.track_point_ids
    click.echo(f"Loading data: {time.perf_counter() - step_start:.2f}s")

    click.echo(f"\nPer-image connectivity table for: {recon_name}")
    click.echo("=" * 120)
    click.echo(f"Total images: {num_images}")
    click.echo("")

    # Count observations per image
    step_start = time.perf_counter()
    obs_per_image = np.zeros(num_images, dtype=np.int32)
    for img_idx in image_indexes:
        obs_per_image[img_idx] += 1
    click.echo(f"Step 1 (Count observations): {time.perf_counter() - step_start:.2f}s")

    # Compute camera centers
    step_start = time.perf_counter()
    camera_centers = _compute_camera_centers(quaternions, translations)
    click.echo(
        f"Step 2 (Compute camera centers): {time.perf_counter() - step_start:.2f}s"
    )

    # Pairwise camera center distances
    step_start = time.perf_counter()
    dist_matrix = np.zeros((num_images, num_images), dtype=np.float64)
    for i in range(num_images):
        for j in range(i + 1, num_images):
            dist = np.linalg.norm(camera_centers[i] - camera_centers[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    closest_dist = np.zeros(num_images, dtype=np.float64)
    tenth_closest_dist = np.zeros(num_images, dtype=np.float64)

    for i in range(num_images):
        distances = [dist_matrix[i, j] for j in range(num_images) if i != j]
        if distances:
            distances_sorted = sorted(distances)
            closest_dist[i] = distances_sorted[0]
            tenth_closest_dist[i] = (
                distances_sorted[9] if len(distances_sorted) >= 10 else np.nan
            )
        else:
            closest_dist[i] = np.nan
            tenth_closest_dist[i] = np.nan
    click.echo(
        f"Step 3 (Compute distance matrix): {time.perf_counter() - step_start:.2f}s"
    )

    # Build covisibility matrix
    step_start = time.perf_counter()
    point_to_images = defaultdict(list)
    for obs_idx in range(len(points3d_indexes)):
        point_id = points3d_indexes[obs_idx]
        image_id = image_indexes[obs_idx]
        point_to_images[point_id].append(image_id)

    covis_matrix = np.zeros((num_images, num_images), dtype=np.int32)
    for point_id, img_list in point_to_images.items():
        unique_images = sorted(set(img_list))
        for i in range(len(unique_images)):
            for j in range(i + 1, len(unique_images)):
                img_i = unique_images[i]
                img_j = unique_images[j]
                covis_matrix[img_i, img_j] += 1
                covis_matrix[img_j, img_i] += 1

    closest_by_shared = {}
    for i in range(num_images):
        max_shared = 0
        closest_img = None
        for j in range(num_images):
            if i != j and covis_matrix[i, j] > max_shared:
                max_shared = covis_matrix[i, j]
                closest_img = j
        if closest_img is not None:
            closest_by_shared[i] = (closest_img, max_shared)
    click.echo(
        f"Step 4 (Build covisibility matrix): {time.perf_counter() - step_start:.2f}s"
    )

    # Compute camera view directions
    step_start = time.perf_counter()
    camera_directions = compute_camera_directions(quaternions)
    click.echo(
        f"Step 5 (Compute view directions): {time.perf_counter() - step_start:.2f}s"
    )

    def compute_view_angle(i, j):
        dot_product = np.dot(camera_directions[i], camera_directions[j])
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)

    # Build graph and compute graph distances
    step_start = time.perf_counter()
    graph = defaultdict(list)
    for i in range(num_images):
        for j in range(num_images):
            if i != j and covis_matrix[i, j] > 0:
                graph[i].append(j)
    click.echo(
        f"Step 6a (Build graph adjacency): {time.perf_counter() - step_start:.2f}s"
    )

    step_start = time.perf_counter()
    closest_far_graph = {}

    for i in range(num_images):
        images_within_10 = {}
        visited = {i: 0}
        queue = deque([(i, 0)])

        while queue:
            node, dist = queue.popleft()
            if dist >= 10:
                continue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    images_within_10[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        candidates = []
        for j in range(num_images):
            if i != j and j not in images_within_10:
                candidates.append((j, dist_matrix[i, j]))

        if len(candidates) > 0:
            closest_idx, _ = min(candidates, key=lambda x: x[1])

            visited_from_j = {closest_idx: 0}
            queue_j = deque([(closest_idx, 0)])
            nodes_processed = 0
            max_nodes = 100
            found_distance = None

            while queue_j and nodes_processed < max_nodes:
                node, dist_from_j = queue_j.popleft()
                nodes_processed += 1

                if node in images_within_10:
                    dist_from_i = images_within_10[node]
                    found_distance = dist_from_i + dist_from_j
                    break

                for neighbor in graph[node]:
                    if neighbor not in visited_from_j:
                        visited_from_j[neighbor] = dist_from_j + 1
                        queue_j.append((neighbor, dist_from_j + 1))

            if found_distance is not None:
                graph_dist = found_distance
            else:
                if visited_from_j:
                    max_depth_from_j = max(visited_from_j.values())
                    graph_dist = f">{10 + max_depth_from_j}"
                else:
                    graph_dist = ">10"

            closest_far_graph[i] = (closest_idx, graph_dist)
    click.echo(
        f"Step 6b (Compute graph distances): {time.perf_counter() - step_start:.2f}s"
    )

    # Motion analysis
    step_start = time.perf_counter()
    motion_results = _analyze_motion_path(recon, camera_centers, quaternions)
    click.echo(
        f"Step 7 (Compute motion discontinuities): {time.perf_counter() - step_start:.2f}s"
    )

    _print_motion_analysis(recon, *motion_results)

    # Print table
    step_start = time.perf_counter()
    click.echo("")
    click.echo("Column descriptions:")
    click.echo("  Image: Image name (relative POSIX path within workspace)")
    click.echo("  Obs: Number of observations (features) for this image")
    click.echo("  Dist1: 3D distance to closest other camera center")
    click.echo("  Dist10: 3D distance to 10th closest other camera center")
    click.echo("  ClosestShared: Image with highest number of shared 3D points")
    click.echo("  SharedMetrics: (# shared points, 3D distance, view angle in degrees)")
    click.echo(
        "  GraphDist>10: Closest image (by 3D distance) with graph distance > 10"
    )
    click.echo(
        "  GraphDist>10Metrics: (graph distance, 3D distance, view angle in degrees)"
    )
    click.echo("")

    for i in range(num_images):
        img_name = recon.image_names[i]
        obs_count = obs_per_image[i]
        dist1_str = f"{closest_dist[i]:.3f}" if not np.isnan(closest_dist[i]) else "N/A"
        dist10_str = (
            f"{tenth_closest_dist[i]:.3f}"
            if not np.isnan(tenth_closest_dist[i])
            else "N/A"
        )

        if i in closest_by_shared:
            closest_idx, shared_count = closest_by_shared[i]
            closest_name = recon.image_names[closest_idx]
            closest_3d_dist = dist_matrix[i, closest_idx]
            closest_angle = compute_view_angle(i, closest_idx)
            closest_shared_str = closest_name
            shared_metrics_str = (
                f"({shared_count}, {closest_3d_dist:.3f}, {closest_angle:.1f}\u00b0)"
            )
        else:
            closest_shared_str = "N/A"
            shared_metrics_str = "N/A"

        if i in closest_far_graph:
            far_graph_idx, graph_dist = closest_far_graph[i]
            far_graph_name = recon.image_names[far_graph_idx]
            far_graph_3d_dist = dist_matrix[i, far_graph_idx]
            far_graph_angle = compute_view_angle(i, far_graph_idx)
            closest_far_graph_str = far_graph_name
            far_graph_metrics_str = (
                f"({graph_dist}, {far_graph_3d_dist:.3f}, {far_graph_angle:.1f}\u00b0)"
            )
        else:
            closest_far_graph_str = "N/A"
            far_graph_metrics_str = "N/A"

        click.echo(f"Image: {img_name}")
        click.echo(f"  Observations: {obs_count}")
        click.echo(f"  Closest camera distance: {dist1_str}")
        click.echo(f"  10th closest camera distance: {dist10_str}")
        click.echo(f"  Closest by shared observations: {closest_shared_str}")
        click.echo(f"    Metrics: {shared_metrics_str}")
        click.echo(f"  Closest with graph distance > 10: {closest_far_graph_str}")
        click.echo(f"    Metrics: {far_graph_metrics_str}")
        click.echo("")

    click.echo(f"Step 8 (Print table): {time.perf_counter() - step_start:.2f}s")
    click.echo("")
    click.echo(f"Total time: {time.perf_counter() - overall_start:.2f}s")
    click.echo("")
