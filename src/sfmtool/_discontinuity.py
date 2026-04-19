# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Discontinuity analysis for image sequences and reconstructions."""

from pathlib import Path

import click
import numpy as np

from ._flow_analysis import (
    _compare_flow_representations,
    _compute_in_bounds_mask,
    _flow_magnitude,
    _load_gray,
)
from ._sfmtool import (
    RotQuaternion,
    compute_optical_flow,
    compute_optical_flow_with_init,
)
from .visualization._discontinuity_display import (
    _print_sample_point,
    _print_summary,
    _save_flow_images,
)


def analyze_image_sequence(
    image_paths: list[Path],
    *,
    frame_numbers: list[int] | None = None,
    initial_stride: int = 1,
    min_stride: int = 1,
    max_stride: int = 32,
    grid_size: int = 3,
    adaptive: bool = True,
    save_flow_dir: Path | None = None,
    sequence_name: str = "flow",
) -> list[dict]:
    """Analyze an image sequence for discontinuities using optical flow.

    Walks through the sequence with an adaptive stride, computing local (i, i+1)
    and stride (i, i+N) flow at each sample point. Shows per-tile magnitude and
    mean vector grids for both flows.

    When initial_stride is 1, every frame is sampled with local (i→i+1) compared
    against stride (i→i+2).

    Args:
        image_paths: Ordered list of image file paths.
        frame_numbers: Actual frame numbers for each image (for display).
            If None, uses 0-based array indices.
        initial_stride: Starting stride N. Use 1 to sample every frame with
            a stride of 2.
        min_stride: Minimum stride (floor for adaptive shrinking).
        max_stride: Maximum stride (ceiling for adaptive growing).
        grid_size: Tile grid size (grid_size x grid_size).
        adaptive: Whether to adaptively adjust the stride. If False, the
            stride stays fixed at initial_stride.
        save_flow_dir: If provided, save flow color images to this directory.

    Returns:
        List of per-sample-point result dicts.
    """
    if frame_numbers is None:
        frame_numbers = list(range(len(image_paths)))
    n_images = len(image_paths)
    if n_images < 2:
        return []

    results = []
    stride = initial_stride
    i = 0

    # Cache loaded grayscale images to avoid reloading
    gray_cache: dict[int, np.ndarray] = {}

    def get_gray(idx: int) -> np.ndarray:
        if idx not in gray_cache:
            gray_cache[idx] = _load_gray(image_paths[idx])
        return gray_cache[idx]

    while i < n_images - 1:
        gray_i = get_gray(i)

        # Local flow: i -> i+1 (computed once per sample point)
        gray_next = get_gray(i + 1)
        local_u, local_v = compute_optical_flow(
            gray_i, gray_next, preset="high_quality"
        )

        # Inner loop: may retry with a smaller stride if the ratio is off
        while True:
            # When stride is 1, compare i→i+1 vs i→i+2 (bump to 2 for comparison)
            compare_stride = max(stride, 2)
            effective_stride = min(compare_stride, n_images - 1 - i)

            result = {
                "frame_number": frame_numbers[i],
                "frame_name": image_paths[i].name,
                "next_frame_name": image_paths[i + 1].name,
                "stride": effective_stride,
            }

            if effective_stride >= 2:
                # Stride flow: i -> i+N, using scaled local flow as init
                gray_stride = get_gray(i + effective_stride)
                init_u = local_u * effective_stride
                init_v = local_v * effective_stride
                stride_u, stride_v = compute_optical_flow_with_init(
                    gray_i, gray_stride, init_u, init_v, preset="high_quality"
                )

                # In-bounds mask from scaled local flow
                in_bounds = _compute_in_bounds_mask(init_u, init_v)

                # Median magnitudes using only in-bounds pixels
                local_mag = _flow_magnitude(local_u, local_v)
                stride_mag = _flow_magnitude(stride_u, stride_v)
                if in_bounds.any():
                    local_median_mag = float(np.median(local_mag[in_bounds]))
                    stride_median_mag = float(np.median(stride_mag[in_bounds]))
                else:
                    local_median_mag = float(np.median(local_mag))
                    stride_median_mag = float(np.median(stride_mag))

                comparison = _compare_flow_representations(
                    local_u,
                    local_v,
                    stride_u,
                    stride_v,
                    effective_stride,
                    grid_size=grid_size,
                )

                result["local_median_magnitude"] = local_median_mag
                result["stride_frame_name"] = image_paths[i + effective_stride].name
                result["stride_median_magnitude"] = stride_median_mag
                result["expected_magnitude_ratio"] = effective_stride
                result["actual_magnitude_ratio"] = (
                    stride_median_mag / local_median_mag
                    if local_median_mag > 0.01
                    else None
                )
                result["local_tile_mags"] = comparison["local_tile_mags"]
                result["stride_tile_mags"] = comparison["stride_tile_mags"]
                result["diff_tile_mags"] = comparison["diff_tile_mags"]
                result["local_tile_means"] = comparison["local_tile_means"]
                result["stride_tile_means"] = comparison["stride_tile_means"]
                result["diff_tile_means"] = comparison["diff_tile_means"]
                result["local_hist"] = comparison["local_hist"]
                result["stride_hist"] = comparison["stride_hist"]
                result["in_bounds_pct"] = comparison["in_bounds_pct"]

                if save_flow_dir is not None:
                    _save_flow_images(
                        save_flow_dir,
                        sequence_name,
                        frame_numbers[i],
                        frame_numbers[i + 1],
                        local_u,
                        local_v,
                        frame_numbers[i + effective_stride],
                        stride_u,
                        stride_v,
                    )
            else:
                # No stride comparison, report unmasked local magnitude
                local_mag = _flow_magnitude(local_u, local_v)
                result["local_median_magnitude"] = float(np.median(local_mag))
                if save_flow_dir is not None:
                    _save_flow_images(
                        save_flow_dir,
                        sequence_name,
                        frame_numbers[i],
                        frame_numbers[i + 1],
                        local_u,
                        local_v,
                        None,
                        None,
                        None,
                    )

            results.append(result)
            _print_sample_point(result)

            if not adaptive:
                break  # no stride adjustment

            # Adaptive stride based on ratio/stride and in-bounds coverage.
            #
            # Ratio bands (log-symmetric):
            #   Grow:   0.85 < ratio/stride < 1/0.85  — consistent
            #   Keep:   0.75 < ratio/stride < 1/0.75  — mild deviation
            #   Shrink: outside the keep band          — something changed
            #
            # In-bounds coverage modifies the decision:
            #   < 25%:  force shrink (data too sparse to trust)
            #   25-50%: suppress grow (keep or shrink only)
            #   > 50%:  use ratio bands as normal
            ratio = result.get("actual_magnitude_ratio")
            in_bounds_pct = result.get("in_bounds_pct", 100.0)
            if ratio is not None and effective_stride >= 2:
                normalized = ratio / effective_stride

                # Determine action from ratio
                if normalized < 0.75 or normalized > 1.0 / 0.75:
                    action = "shrink"
                    reason = f"ratio/stride={normalized:.2f}, outside [0.75, 1.33]"
                elif 0.85 < normalized < 1.0 / 0.85:
                    action = "grow"
                    reason = f"ratio/stride={normalized:.2f}, inside [0.85, 1.18]"
                else:
                    action = "keep"
                    reason = ""

                # In-bounds coverage overrides
                if in_bounds_pct < 25:
                    if action != "shrink":
                        action = "shrink"
                        reason = f"in-bounds={in_bounds_pct:.0f}%<25%"
                elif in_bounds_pct < 50:
                    if action == "grow":
                        action = "keep"

                if action == "shrink":
                    new_stride = max(stride // 2, min_stride)
                    if new_stride < stride:
                        new_effective = min(new_stride, n_images - 1 - i)
                        if new_effective < effective_stride:
                            click.echo(f"  ↓ stride {stride}→{new_stride} ({reason})")
                            stride = new_stride
                            result["superseded"] = True
                            continue  # retry this frame with smaller stride
                        stride = new_stride
                elif action == "grow":
                    new_stride = min(stride * 2, max_stride)
                    if new_stride > stride:
                        click.echo(f"  ↑ stride {stride}→{new_stride} ({reason})")
                        stride = new_stride

            break  # done with this sample point

        i += stride

        # Evict old cache entries to limit memory
        evict_below = i - 1
        for key in list(gray_cache.keys()):
            if key < evict_below:
                del gray_cache[key]

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Reconstruction analysis
# ---------------------------------------------------------------------------


def _rotation_angle_deg(qa: RotQuaternion, qb: RotQuaternion) -> float:
    """Rotation angle in degrees between two quaternions."""
    return float(np.degrees((qb * qa.conjugate()).angle()))


def _extrapolate_pose(
    centers: list[np.ndarray],
    quats: list[RotQuaternion],
    ts: list[float],
    t_target: float,
    degree: int,
) -> tuple[np.ndarray, RotQuaternion]:
    """Extrapolate a camera pose using polynomial fit on centers and quaternions.

    Both translation and rotation use polynomial fitting.  For the quaternion,
    each (w, x, y, z) component is fit independently and the result is
    re-normalized to unit length.

    Args:
        centers: Camera center positions (at least degree+1).
        quats: Quaternion orientations (same length as centers).
        ts: Parameter values (e.g. frame indices) for the known poses.
        t_target: Parameter value to extrapolate to.
        degree: Polynomial degree (1 for linear, 2 for quadratic).

    Returns:
        (predicted_center, predicted_quaternion)
    """
    ts_arr = np.array(ts, dtype=np.float64)

    # Polynomial fit for translation
    centers_arr = np.array([c for c in centers], dtype=np.float64)
    predicted_center = np.zeros(3)
    for axis in range(3):
        coeffs = np.polyfit(ts_arr, centers_arr[:, axis], degree)
        predicted_center[axis] = np.polyval(coeffs, t_target)

    # Polynomial fit for rotation: fit each quaternion component, normalize.
    # Flip signs to ensure all quaternions are in the same hemisphere as the
    # first, so the polynomial doesn't interpolate through the wrong side.
    q_arr = np.array([q.to_wxyz_array() for q in quats], dtype=np.float64)
    for j in range(1, len(quats)):
        if np.dot(q_arr[0], q_arr[j]) < 0:
            q_arr[j] = -q_arr[j]

    predicted_wxyz = np.zeros(4)
    for comp in range(4):
        coeffs = np.polyfit(ts_arr, q_arr[:, comp], degree)
        predicted_wxyz[comp] = np.polyval(coeffs, t_target)

    # RotQuaternion constructor normalizes to unit length
    predicted_quat = RotQuaternion(*predicted_wxyz)

    return predicted_center, predicted_quat


def _extrapolation_error(
    centers: list[np.ndarray],
    quats: list[RotQuaternion],
    ts: list[float],
    t_target: float,
    actual_center: np.ndarray,
    actual_quat: RotQuaternion,
    degree: int,
) -> tuple[float, float]:
    """Compute translation and rotation extrapolation errors for one prediction."""
    pred_center, pred_quat = _extrapolate_pose(centers, quats, ts, t_target, degree)
    trans_err = float(np.linalg.norm(pred_center - actual_center))
    rot_err = _rotation_angle_deg(pred_quat, actual_quat)
    return trans_err, rot_err


def _compute_extrapolation_errors(
    seq_centers: list[np.ndarray],
    seq_quats: list[RotQuaternion],
    seq_frame_numbers: list[int],
) -> list[dict]:
    """Compute left and right extrapolation errors for each frame in a sequence.

    For each frame i, extrapolate from the left (i-2, i-1 linear and
    i-3, i-2, i-1 quadratic) and from the right (i+1, i+2 linear and
    i+1, i+2, i+3 quadratic).  Reports the minimum of the two fits for
    each direction, so a discontinuity at the far end of the 3-point
    window doesn't inflate the error for the near side.

    Returns a list of per-frame dicts with extrapolation errors.
    """
    n = len(seq_centers)
    results = []

    for i in range(n):
        entry = {
            "seq_idx": i,
            "frame_number": seq_frame_numbers[i],
        }
        t_target = float(seq_frame_numbers[i])

        # Left extrapolation
        if i >= 2:
            # Linear from nearest 2: i-2, i-1
            lin_centers = [seq_centers[i - 2], seq_centers[i - 1]]
            lin_quats = [seq_quats[i - 2], seq_quats[i - 1]]
            lin_ts = [
                float(seq_frame_numbers[i - 2]),
                float(seq_frame_numbers[i - 1]),
            ]
            lin_t, lin_r = _extrapolation_error(
                lin_centers,
                lin_quats,
                lin_ts,
                t_target,
                seq_centers[i],
                seq_quats[i],
                degree=1,
            )

            if i >= 3:
                # Quadratic from nearest 3: i-3, i-2, i-1
                quad_centers = [
                    seq_centers[i - 3],
                    seq_centers[i - 2],
                    seq_centers[i - 1],
                ]
                quad_quats = [
                    seq_quats[i - 3],
                    seq_quats[i - 2],
                    seq_quats[i - 1],
                ]
                quad_ts = [
                    float(seq_frame_numbers[i - 3]),
                    float(seq_frame_numbers[i - 2]),
                    float(seq_frame_numbers[i - 1]),
                ]
                quad_t, quad_r = _extrapolation_error(
                    quad_centers,
                    quad_quats,
                    quad_ts,
                    t_target,
                    seq_centers[i],
                    seq_quats[i],
                    degree=2,
                )
                entry["left_trans_err"] = min(lin_t, quad_t)
                entry["left_rot_err"] = min(lin_r, quad_r)
            else:
                entry["left_trans_err"] = lin_t
                entry["left_rot_err"] = lin_r
        else:
            entry["left_trans_err"] = None
            entry["left_rot_err"] = None

        # Right extrapolation
        if i + 2 < n:
            # Linear from nearest 2: i+1, i+2
            lin_centers = [seq_centers[i + 2], seq_centers[i + 1]]
            lin_quats = [seq_quats[i + 2], seq_quats[i + 1]]
            lin_ts = [
                float(seq_frame_numbers[i + 2]),
                float(seq_frame_numbers[i + 1]),
            ]
            lin_t, lin_r = _extrapolation_error(
                lin_centers,
                lin_quats,
                lin_ts,
                t_target,
                seq_centers[i],
                seq_quats[i],
                degree=1,
            )

            if i + 3 < n:
                # Quadratic from nearest 3: i+1, i+2, i+3
                quad_centers = [
                    seq_centers[i + 3],
                    seq_centers[i + 2],
                    seq_centers[i + 1],
                ]
                quad_quats = [
                    seq_quats[i + 3],
                    seq_quats[i + 2],
                    seq_quats[i + 1],
                ]
                quad_ts = [
                    float(seq_frame_numbers[i + 3]),
                    float(seq_frame_numbers[i + 2]),
                    float(seq_frame_numbers[i + 1]),
                ]
                quad_t, quad_r = _extrapolation_error(
                    quad_centers,
                    quad_quats,
                    quad_ts,
                    t_target,
                    seq_centers[i],
                    seq_quats[i],
                    degree=2,
                )
                entry["right_trans_err"] = min(lin_t, quad_t)
                entry["right_rot_err"] = min(lin_r, quad_r)
            else:
                entry["right_trans_err"] = lin_t
                entry["right_rot_err"] = lin_r
        else:
            entry["right_trans_err"] = None
            entry["right_rot_err"] = None

        results.append(entry)

    return results


def _compute_shared_point_counts(
    recon,
    image_indexes: list[int],
) -> dict[tuple[int, int], int]:
    """Build a lookup of shared 3D point counts for adjacent image pairs.

    Returns a dict mapping (img_idx_a, img_idx_b) → shared_count.
    """
    from ._image_pair_graph import build_covisibility_pairs

    all_pairs = build_covisibility_pairs(recon)
    pair_counts = {}
    idx_set = set(image_indexes)
    for i, j, count in all_pairs:
        if i in idx_set and j in idx_set:
            pair_counts[(i, j)] = count
            pair_counts[(j, i)] = count
    return pair_counts


def _compute_per_image_mean_errors(
    recon,
    image_indexes: list[int],
) -> dict[int, float]:
    """Compute mean reprojection error per image for a set of image indices."""
    errors = {}
    for img_idx in image_indexes:
        obs_data = recon.compute_observation_reprojection_errors(img_idx)
        obs_errors = obs_data[:, 1]
        valid = obs_errors[~np.isnan(obs_errors)]
        errors[img_idx] = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    return errors


def analyze_reconstruction(
    recon,
    *,
    range_numbers: set[int] | None = None,
) -> list[dict]:
    """Analyze a reconstruction for pose discontinuities in sequential data.

    Detects numbered sequences among the reconstruction's image names, then
    for each sequence computes extrapolation errors to find pose discontinuities.
    Reports covisibility and reprojection error context for flagged frames.

    Args:
        recon: A loaded SfmrReconstruction.
        range_numbers: If provided, only include images whose file number
            is in this set.

    Returns:
        List of per-sequence result dicts.
    """
    from deadline.job_attachments.api import summarize_paths_by_sequence

    from ._filenames import number_from_filename

    image_names = recon.image_names
    num_images = recon.image_count
    quaternions_wxyz = recon.quaternions_wxyz
    translations = recon.translations

    # Build name → image index mapping
    name_to_idx = {}
    for i in range(num_images):
        name = image_names[i]
        if range_numbers is not None:
            file_number = number_from_filename(name)
            if file_number is None or file_number not in range_numbers:
                continue
        name_to_idx[name] = i

    # Detect sequences
    summaries = summarize_paths_by_sequence(list(name_to_idx.keys()))
    numbered_sequences = [s for s in summaries if s.index_set]

    if not numbered_sequences:
        click.echo("No numbered sequences detected among the reconstruction's images.")
        return []

    click.echo(
        f"Reconstruction: {num_images} images, {recon.point_count:,} points, "
        f"{recon.observation_count:,} observations"
    )
    click.echo(f"Found {len(numbered_sequences)} sequence(s)")

    all_sequence_results = []

    for seq in numbered_sequences:
        sorted_indexes = sorted(seq.index_set)

        # Build ordered lists of poses for this sequence
        seq_image_names = []
        seq_image_indexes = []  # index into the reconstruction
        seq_frame_numbers = []
        seq_centers = []
        seq_quats = []

        for idx in sorted_indexes:
            # Normalize separators — summarize_paths_by_sequence may produce
            # backslashes on Windows while image names use forward slashes.
            fname = (seq.path % idx).replace("\\", "/")
            if fname not in name_to_idx:
                continue
            img_idx = name_to_idx[fname]
            quat = RotQuaternion.from_wxyz_array(quaternions_wxyz[img_idx])
            center = quat.camera_center(translations[img_idx])

            seq_image_names.append(fname)
            seq_image_indexes.append(img_idx)
            seq_frame_numbers.append(idx)
            seq_centers.append(center)
            seq_quats.append(quat)

        if len(seq_image_names) < 4:
            click.echo(
                f"\nSkipping sequence {seq.path}: "
                f"only {len(seq_image_names)} image(s) (need at least 4)"
            )
            continue

        seq_base = seq.path.rsplit(".", 1)[0].split("%")[0].rstrip("_")
        click.echo(f"\nAnalyzing sequence: {seq.path} ({len(seq_image_names)} frames)")

        # Step 1: Pose extrapolation errors
        extrap_results = _compute_extrapolation_errors(
            seq_centers, seq_quats, seq_frame_numbers
        )

        # Compute successive distances and rotations for context
        n = len(seq_centers)
        successive_trans = np.array(
            [np.linalg.norm(seq_centers[i + 1] - seq_centers[i]) for i in range(n - 1)]
        )
        successive_rots = np.array(
            [_rotation_angle_deg(seq_quats[i], seq_quats[i + 1]) for i in range(n - 1)]
        )

        # Compute median successive motion as scale reference
        median_trans = float(np.median(successive_trans))
        median_rot = float(np.median(successive_rots))

        # Print per-frame extrapolation errors
        click.echo(
            f"\n  Successive motion --"
            f" median translation: {median_trans:.4f},"
            f" median rotation: {median_rot:.2f}°"
        )
        # Translation threshold: 3x median successive motion.  Using 1x
        # is too tight — normal trajectory curvature produces extrapolation
        # errors on the order of the step size.  3x leaves room for that
        # while still catching real jumps.  Invariant to pruning since it's
        # based on trajectory properties, not error statistics.
        trans_threshold = 3.0 * median_trans

        # Rotation threshold: fixed at 15°.  Unlike translation, rotation
        # extrapolation quality depends on the smoothness of the trajectory,
        # not on the rotation rate.  A quadratic extrapolation from 3
        # smooth neighbors should predict within a few degrees regardless
        # of how fast the camera is rotating.
        rot_threshold = 15.0

        click.echo(
            f"  Extrapolation thresholds:"
            f"  trans>{trans_threshold:.4f} (3x median step)"
            f"  rot>{rot_threshold:.1f}° (fixed)"
        )
        click.echo("")
        click.echo(
            f"  {'Frame':>7}  {'Image':<40}  "
            f"{'L.trans':>7}  {'L.rot':>6}  "
            f"{'R.trans':>7}  {'R.rot':>6}  {'Flag'}"
        )
        click.echo("  " + "-" * 95)

        flagged_frames = []

        for er in extrap_results:
            i = er["seq_idx"]
            frame_num = er["frame_number"]
            name = seq_image_names[i]
            if len(name) > 40:
                name = "..." + name[-37:]

            lt = er["left_trans_err"]
            lr = er["left_rot_err"]
            rt = er["right_trans_err"]
            rr = er["right_rot_err"]

            lt_s = f"{lt:.4f}" if lt is not None else "-"
            lr_s = f"{lr:.2f}°" if lr is not None else "-"
            rt_s = f"{rt:.4f}" if rt is not None else "-"
            rr_s = f"{rr:.2f}°" if rr is not None else "-"

            # Determine flags
            flags = []
            if lt is not None and lt > trans_threshold:
                flags.append("L.t")
            if lr is not None and lr > rot_threshold:
                flags.append("L.r")
            if rt is not None and rt > trans_threshold:
                flags.append("R.t")
            if rr is not None and rr > rot_threshold:
                flags.append("R.r")

            flag_str = ",".join(flags) if flags else ""

            click.echo(
                f"  {frame_num:>7}  {name:<40}  "
                f"{lt_s:>7}  {lr_s:>6}  "
                f"{rt_s:>7}  {rr_s:>6}  {flag_str}"
            )

            if flags:
                flagged_frames.append(
                    {
                        "seq_idx": i,
                        "frame_number": frame_num,
                        "image_name": seq_image_names[i],
                        "image_index": seq_image_indexes[i],
                        "left_trans_err": lt,
                        "left_rot_err": lr,
                        "right_trans_err": rt,
                        "right_rot_err": rr,
                        "flags": flags,
                    }
                )

        # Step 2: Aggregate per-frame flags into per-edge discontinuities.
        # L flags on frame i implicate edge (i-1) → (i).
        # R flags on frame i implicate edge (i) → (i+1).
        flagged_edges: dict[tuple[int, int], list[str]] = {}
        for f in flagged_frames:
            idx = f["seq_idx"]
            for flag in f["flags"]:
                if flag.startswith("L") and idx > 0:
                    edge = (idx - 1, idx)
                elif flag.startswith("R") and idx < n - 1:
                    edge = (idx, idx + 1)
                else:
                    continue
                flagged_edges.setdefault(edge, []).append(
                    f"frame {f['frame_number']} {flag}"
                )

        # Step 3: Cluster adjacent edges and keep only the core edge per
        # cluster.  A single discontinuity at edge (A, A+1) causes the
        # neighboring edges to also get flagged because even the nearest 2
        # extrapolation points straddle the break.  We cluster consecutive
        # flagged edges and pick the one with the most evidence (flag count),
        # breaking ties by lowest shared 3D point count.
        core_edges: dict[tuple[int, int], list[str]] = {}
        pair_counts: dict[tuple[int, int], int] = {}
        if flagged_edges:
            # Compute shared point counts for all flagged edges up front
            context_indexes = set()
            for a, b in flagged_edges:
                context_indexes.add(seq_image_indexes[a])
                context_indexes.add(seq_image_indexes[b])
            pair_counts = _compute_shared_point_counts(recon, list(context_indexes))

            # Build clusters of adjacent edges
            sorted_edges = sorted(flagged_edges.keys())
            clusters: list[list[tuple[int, int]]] = []
            current_cluster: list[tuple[int, int]] = [sorted_edges[0]]
            for edge in sorted_edges[1:]:
                prev = current_cluster[-1]
                if edge[0] <= prev[1]:  # adjacent or overlapping
                    current_cluster.append(edge)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [edge]
            clusters.append(current_cluster)

            # Pick the core edge per cluster: most evidence, then fewest
            # shared points
            for cluster in clusters:
                best_edge = max(
                    cluster,
                    key=lambda e: (
                        len(flagged_edges[e]),
                        -pair_counts.get(
                            (seq_image_indexes[e[0]], seq_image_indexes[e[1]]),
                            0,
                        ),
                    ),
                )
                core_edges[best_edge] = flagged_edges[best_edge]

        all_sequence_results.append(
            {
                "sequence": seq.path,
                "sequence_name": seq_base,
                "frame_count": len(seq_image_names),
                "extrap_results": extrap_results,
                "flagged_frames": flagged_frames,
                "flagged_edges": flagged_edges,
                "core_edges": core_edges,
                "pair_counts": pair_counts,
                "seq_image_indexes": seq_image_indexes,
                "seq_frame_numbers": seq_frame_numbers,
                "seq_centers": seq_centers,
                "seq_quats": seq_quats,
                "median_trans": median_trans,
                "median_rot": median_rot,
            }
        )

    # --- Summary section: all sequences grouped at the end ---
    click.echo("")
    click.echo("=" * 80)
    click.echo("Summary")
    click.echo("=" * 80)

    total_discontinuities = sum(len(s["core_edges"]) for s in all_sequence_results)

    for s in all_sequence_results:
        core_edges = s["core_edges"]
        seq_name = s["sequence_name"]
        click.echo(
            f"\n  {seq_name}: {s['frame_count']} frames, "
            f"{len(core_edges)} discontinuity(s)"
        )

        if not core_edges:
            click.echo("    No pose discontinuities detected.")
            continue

        # Compute reprojection errors for core edges
        seq_img_indexes = s["seq_image_indexes"]
        seq_frm_numbers = s["seq_frame_numbers"]
        seq_centers = s["seq_centers"]
        seq_quats = s["seq_quats"]
        pair_counts = s["pair_counts"]
        core_img_indexes = set()
        for a, b in core_edges:
            core_img_indexes.add(seq_img_indexes[a])
            core_img_indexes.add(seq_img_indexes[b])
        reproj_errors = _compute_per_image_mean_errors(recon, list(core_img_indexes))

        n_seq = len(seq_centers)
        click.echo(
            f"    {'Edge':>14}  {'Dist(prev)':>10}  {'Dist':>8}  "
            f"{'Dist(next)':>10}  {'Rot':>7}  "
            f"{'SharedPts':>10}  {'Err(A)':>8}  {'Err(B)':>8}"
        )
        click.echo("    " + "-" * 97)

        for (a, b), evidence in sorted(core_edges.items()):
            img_a = seq_img_indexes[a]
            img_b = seq_img_indexes[b]
            frame_a = seq_frm_numbers[a]
            frame_b = seq_frm_numbers[b]

            dist_prev = (
                float(np.linalg.norm(seq_centers[a] - seq_centers[a - 1]))
                if a > 0
                else None
            )
            dist = float(np.linalg.norm(seq_centers[b] - seq_centers[a]))
            dist_next = (
                float(np.linalg.norm(seq_centers[b + 1] - seq_centers[b]))
                if b + 1 < n_seq
                else None
            )
            rot = _rotation_angle_deg(seq_quats[a], seq_quats[b])

            shared = pair_counts.get((img_a, img_b), 0)
            err_a = reproj_errors.get(img_a, float("nan"))
            err_b = reproj_errors.get(img_b, float("nan"))
            err_a_s = f"{err_a:.3f}px" if not np.isnan(err_a) else "N/A"
            err_b_s = f"{err_b:.3f}px" if not np.isnan(err_b) else "N/A"

            # Determine if translation and/or rotation triggered this edge
            has_t = any(" L.t" in e or " R.t" in e for e in evidence)
            has_r = any(" L.r" in e or " R.r" in e for e in evidence)

            # Highlight flagged values with < >
            if has_t:
                dp_s = f"{dist_prev:.4f}" if dist_prev is not None else "-"
                dist_s = f"<{dist:.4f}>"
                dn_s = f"{dist_next:.4f}" if dist_next is not None else "-"
            else:
                dp_s = f"{dist_prev:.4f}" if dist_prev is not None else "-"
                dist_s = f" {dist:.4f} "
                dn_s = f"{dist_next:.4f}" if dist_next is not None else "-"

            if has_r:
                rot_s = f"<{rot:.2f}°>"
            else:
                rot_s = f" {rot:.2f}° "

            edge_str = f"{frame_a}->{frame_b}"
            click.echo(
                f"    {edge_str:>14}  {dp_s:>10}  {dist_s:>10}  "
                f"{dn_s:>10}  {rot_s:>9}  "
                f"{shared:>10}  {err_a_s:>8}  {err_b_s:>8}"
            )

    click.echo("")
    if total_discontinuities == 0:
        click.echo("No pose discontinuities detected in any sequence.")
    else:
        click.echo(
            f"Total: {total_discontinuities} discontinuity(s) "
            f"across {len(all_sequence_results)} sequence(s)."
        )

    return all_sequence_results
