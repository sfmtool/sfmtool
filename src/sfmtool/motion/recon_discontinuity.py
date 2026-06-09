# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Pose discontinuity analysis for SfM reconstructions."""

import click
import numpy as np

from .constants import (
    OBS_WINDOW,
    OBS_Z_THRESHOLD,
    OVERLAP_BASELINE_WINDOW,
    OVERLAP_DROP_THRESHOLD,
    OVERLAP_WINDOW,
    POSE_ROT_DEG,
    POSE_TRANS_FACTOR,
    STEP_RATIO_THRESHOLD,
    STEP_RATIO_WINDOW,
)
from .._sfmtool import RotQuaternion


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
    from .._image_pair_graph import build_covisibility_pairs

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


# ---------------------------------------------------------------------------
# Secondary discontinuity signals (complement pose extrapolation).
# See `constants.py` for the rationale behind each signal and
# its threshold/window constants.
# ---------------------------------------------------------------------------


def _build_per_image_point_sets(recon) -> list[set[int]]:
    """Return per-image sets of 3D point ids observed in that image."""
    obs_counts = np.asarray(recon.observation_counts, dtype=np.int64)
    tii = np.asarray(recon.track_image_indexes, dtype=np.int64)
    offsets = np.concatenate([[np.int64(0)], np.cumsum(obs_counts)])
    n_images = recon.image_count
    per_image: list[set[int]] = [set() for _ in range(n_images)]
    for p in range(len(obs_counts)):
        for img_idx in tii[offsets[p] : offsets[p + 1]]:
            per_image[int(img_idx)].add(p)
    return per_image


def _compute_step_ratios(
    seq_centers: list[np.ndarray],
    window: int = STEP_RATIO_WINDOW,
) -> list[float | None]:
    """Per-edge ratio of local median step size before vs. after.

    For each edge i→(i+1), compute the ratio max(m_pre/m_post, m_post/m_pre)
    where m_pre is the median step length in the `window-1` edges immediately
    before i and m_post is the median in the `window-1` edges after.  The
    edge itself is excluded — we are testing whether the surrounding motion
    magnitude changes across it, not whether this one step is an outlier.

    Returns None for edges without at least 2 edges of context on each side.
    """
    n = len(seq_centers)
    if n < 2:
        return []
    step_sizes = [
        float(np.linalg.norm(seq_centers[i + 1] - seq_centers[i])) for i in range(n - 1)
    ]
    n_edges = len(step_sizes)
    ratios: list[float | None] = [None] * n_edges
    for i in range(n_edges):
        pre = step_sizes[max(0, i - window + 1) : i]
        post = step_sizes[i + 1 : min(n_edges, i + window)]
        if len(pre) < 2 or len(post) < 2:
            continue
        m_pre = float(np.median(pre))
        m_post = float(np.median(post))
        if m_pre <= 0 or m_post <= 0:
            continue
        ratios[i] = max(m_pre / m_post, m_post / m_pre)
    return ratios


def _compute_overlap_drops(
    per_image_points: list[set[int]],
    seq_image_indexes: list[int],
    window: int = OVERLAP_WINDOW,
    baseline_window: int = OVERLAP_BASELINE_WINDOW,
) -> list[float | None]:
    """Per-edge covisibility drop relative to a local baseline of edges.

    For each edge i→(i+1):
      cross = |P_pre ∩ P_post| / min(|P_pre|, |P_post|)
        where P_pre  = union of tracks observed in images [i-w+1 .. i]
              P_post = union of tracks observed in images [i+1 .. i+w]
      baseline = median of the surrounding edges' `cross` values
      drop = baseline / cross      (larger = stronger break)

    Returns all None when the sequence has fewer than 3*window frames — on
    short sequences typical track lifetimes exceed the window and the test
    cannot distinguish a real break from natural track aging.  Returns +inf
    for a per-edge cross of 0 (no tracks survive the edge).
    """
    n = len(seq_image_indexes)
    n_edges = max(0, n - 1)
    if n < 3 * window:
        return [None] * n_edges
    overlaps: list[float | None] = [None] * n_edges

    for i in range(n_edges):
        lo_l = max(0, i - window + 1)
        hi_l = i + 1
        lo_r = i + 1
        hi_r = min(n, i + 1 + window)
        pre: set[int] = set()
        for k in range(lo_l, hi_l):
            pre |= per_image_points[seq_image_indexes[k]]
        post: set[int] = set()
        for k in range(lo_r, hi_r):
            post |= per_image_points[seq_image_indexes[k]]
        if not pre or not post:
            continue
        denom = min(len(pre), len(post))
        overlaps[i] = len(pre & post) / denom if denom > 0 else None

    drops: list[float | None] = [None] * n_edges
    for i in range(n_edges):
        own = overlaps[i]
        if own is None:
            continue
        lo = max(0, i - baseline_window)
        hi = min(n_edges, i + baseline_window + 1)
        nearby = [
            overlaps[j] for j in range(lo, hi) if j != i and overlaps[j] is not None
        ]
        if len(nearby) < 3:
            continue
        baseline = float(np.median(nearby))
        if own == 0:
            drops[i] = float("inf")
        elif baseline <= 0:
            continue
        else:
            drops[i] = baseline / own
    return drops


def _compute_obs_z_scores(
    per_image_points: list[set[int]],
    seq_image_indexes: list[int],
    window: int = OBS_WINDOW,
) -> list[float | None]:
    """Per-frame robust z-score of observation count.

    Uses a symmetric rolling window of `window` frames centered on i
    (excluding i itself) and MAD-based scale (σ ≈ 1.4826·MAD).  Returns
    None for frames without enough neighbors or with zero MAD.
    """
    n = len(seq_image_indexes)
    if n < 5:
        return [None] * n
    nobs = np.array(
        [len(per_image_points[seq_image_indexes[i]]) for i in range(n)],
        dtype=np.float64,
    )
    half = window // 2
    z_scores: list[float | None] = [None] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = np.concatenate([nobs[lo:i], nobs[i + 1 : hi]])
        if len(window_vals) < 5:
            continue
        med = float(np.median(window_vals))
        mad = float(np.median(np.abs(window_vals - med)))
        if mad <= 0:
            continue
        z_scores[i] = (nobs[i] - med) / (1.4826 * mad)
    return z_scores


def _flag_frame(
    *,
    left_trans_err: float | None,
    left_rot_err: float | None,
    right_trans_err: float | None,
    right_rot_err: float | None,
    step_ratio: float | None,
    overlap_drop: float | None,
    obs_z: float | None,
    trans_threshold: float,
    rot_threshold: float,
) -> list[str]:
    """Determine discontinuity flags for a single frame.

    The single source of truth for per-frame thresholding, shared by the
    console table in `analyze_reconstruction` and the JSON serializer in
    `report.py` so the two cannot silently diverge.

    `step_ratio` and `overlap_drop` are the values of the *landing* edge
    `(i-1, i)`; both callers pass `None` for the first frame (no landing
    edge), so the "Step"/"Cov" flags never fire on frame 0.
    """
    flags: list[str] = []
    if left_trans_err is not None and left_trans_err > trans_threshold:
        flags.append("L.t")
    if left_rot_err is not None and left_rot_err > rot_threshold:
        flags.append("L.r")
    if right_trans_err is not None and right_trans_err > trans_threshold:
        flags.append("R.t")
    if right_rot_err is not None and right_rot_err > rot_threshold:
        flags.append("R.r")
    if step_ratio is not None and step_ratio > STEP_RATIO_THRESHOLD:
        flags.append("Step")
    if overlap_drop is not None and overlap_drop > OVERLAP_DROP_THRESHOLD:
        flags.append("Cov")
    if obs_z is not None and obs_z < -OBS_Z_THRESHOLD:
        flags.append("Obs")
    return flags


def _aggregate_flagged_edges(
    flagged_frames: list[dict], n: int
) -> dict[tuple[int, int], list[str]]:
    """Aggregate per-frame flags into per-edge discontinuity evidence.

    - L flags on frame i implicate edge (i-1, i).
    - R flags on frame i implicate edge (i, i+1).
    - Step and Cov flags are edge-level — they sit on the landing frame i and
      implicate edge (i-1, i).
    - Obs flags are per-frame context only: attached to any adjacent edge that
      is already flagged by another signal, but they never flag an edge alone
      (a single dim frame is not a discontinuity).

    Returns a dict mapping edge `(a, b)` → list of evidence strings.
    """
    flagged_edges: dict[tuple[int, int], list[str]] = {}
    for f in flagged_frames:
        idx = f["seq_idx"]
        for flag in f["flags"]:
            if flag == "Obs":
                continue  # attached below as endpoint context
            if flag.startswith("L") and idx > 0:
                edge = (idx - 1, idx)
            elif flag.startswith("R") and idx < n - 1:
                edge = (idx, idx + 1)
            elif flag in ("Step", "Cov") and idx > 0:
                edge = (idx - 1, idx)  # landing edge
            else:
                continue
            flagged_edges.setdefault(edge, []).append(
                f"frame {f['frame_number']} {flag}"
            )

    # Attach Obs context to adjacent edges that are already flagged.
    for f in flagged_frames:
        if "Obs" not in f["flags"]:
            continue
        idx = f["seq_idx"]
        for edge in ((idx - 1, idx), (idx, idx + 1)):
            if edge in flagged_edges:
                flagged_edges[edge].append(f"frame {f['frame_number']} Obs")

    return flagged_edges


def _select_core_edges(
    flagged_edges: dict[tuple[int, int], list[str]],
    seq_image_indexes: list[int],
    recon,
) -> tuple[dict[tuple[int, int], list[str]], dict[tuple[int, int], int]]:
    """Cluster adjacent flagged edges and keep only the core edge per cluster.

    A single discontinuity at edge (A, A+1) causes the neighboring edges to
    also get flagged because even the nearest 2 extrapolation points straddle
    the break.  Consecutive flagged edges are clustered and the one with the
    most evidence (flag count) is kept, breaking ties by lowest shared 3D
    point count.

    Returns `(core_edges, pair_counts)` where `core_edges` maps the kept edge
    → its evidence, and `pair_counts` maps `(img_a, img_b)` → shared 3D point
    count for every flagged edge (used both here for tie-breaking and by the
    summary report).
    """
    core_edges: dict[tuple[int, int], list[str]] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    if not flagged_edges:
        return core_edges, pair_counts

    # Compute shared point counts for all flagged edges up front.
    context_indexes = set()
    for a, b in flagged_edges:
        context_indexes.add(seq_image_indexes[a])
        context_indexes.add(seq_image_indexes[b])
    pair_counts = _compute_shared_point_counts(recon, list(context_indexes))

    # Build clusters of adjacent edges.
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

    # Pick the core edge per cluster: most evidence, then fewest shared points.
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

    return core_edges, pair_counts


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

    from .._filenames import number_from_filename

    # Imported lazily to avoid a module-level import cycle: _recon_console
    # imports the pure helpers (`_flag_frame`, `_rotation_angle_deg`) from here.
    from ._recon_console import print_frame_table, print_summary

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

    # Per-image point sets drive the overlap-drop and obs-outlier signals.
    # Built once up front; reused across sequences.
    per_image_points = _build_per_image_point_sets(recon)

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

        # Step 2: Secondary signals.  These are complementary to pose
        # extrapolation — polynomial fits can absorb smooth scale/slope
        # changes that these three catch directly.
        step_ratios = _compute_step_ratios(seq_centers)  # per-edge, len n-1
        overlap_drops = _compute_overlap_drops(
            per_image_points, seq_image_indexes
        )  # per-edge, len n-1
        obs_z_scores = _compute_obs_z_scores(
            per_image_points, seq_image_indexes
        )  # per-frame, len n

        # Thresholds for this sequence (consumed by both the console table and
        # the JSON report).
        # Translation threshold: POSE_TRANS_FACTOR × median successive
        # motion.  Using 1× is too tight — normal trajectory curvature
        # produces extrapolation errors on the order of the step size.
        # The factor leaves room for that while still catching real jumps.
        # Invariant to pruning since it's based on trajectory properties,
        # not error statistics.
        trans_threshold = POSE_TRANS_FACTOR * median_trans
        # Rotation threshold: fixed.  Unlike translation, rotation
        # extrapolation quality depends on the smoothness of the trajectory,
        # not on the rotation rate.  A quadratic extrapolation from 3
        # smooth neighbors should predict within a few degrees regardless
        # of how fast the camera is rotating.
        rot_threshold = POSE_ROT_DEG

        # Per-frame flags.  StepR/CovR come from the landing edge (i-1, i), so
        # they annotate the frame the edge lands on; both are None when
        # landing_edge < 0 and thus never fire those flags on frame 0.
        flagged_frames = []
        for er in extrap_results:
            i = er["seq_idx"]
            landing_edge = i - 1  # edge index for edge (i-1, i)
            step_r = step_ratios[landing_edge] if landing_edge >= 0 else None
            cov_r = overlap_drops[landing_edge] if landing_edge >= 0 else None
            obs_z = obs_z_scores[i] if i < len(obs_z_scores) else None

            flags = _flag_frame(
                left_trans_err=er["left_trans_err"],
                left_rot_err=er["left_rot_err"],
                right_trans_err=er["right_trans_err"],
                right_rot_err=er["right_rot_err"],
                step_ratio=step_r,
                overlap_drop=cov_r,
                obs_z=obs_z,
                trans_threshold=trans_threshold,
                rot_threshold=rot_threshold,
            )

            if flags:
                flagged_frames.append(
                    {
                        "seq_idx": i,
                        "frame_number": er["frame_number"],
                        "image_name": seq_image_names[i],
                        "image_index": seq_image_indexes[i],
                        "left_trans_err": er["left_trans_err"],
                        "left_rot_err": er["left_rot_err"],
                        "right_trans_err": er["right_trans_err"],
                        "right_rot_err": er["right_rot_err"],
                        "step_ratio": step_r,
                        "overlap_drop": cov_r,
                        "obs_z": obs_z,
                        "flags": flags,
                    }
                )

        # Aggregate per-frame flags into per-edge discontinuities, then
        # cluster adjacent edges down to one core edge per break.
        flagged_edges = _aggregate_flagged_edges(flagged_frames, n)
        core_edges, pair_counts = _select_core_edges(
            flagged_edges, seq_image_indexes, recon
        )

        # Mean reprojection error per endpoint of each core edge (context for
        # the summary report).
        core_img_indexes: set[int] = set()
        for a, b in core_edges:
            core_img_indexes.add(seq_image_indexes[a])
            core_img_indexes.add(seq_image_indexes[b])
        reproj_errors = (
            _compute_per_image_mean_errors(recon, list(core_img_indexes))
            if core_edges
            else {}
        )

        seq_result = {
            "sequence": seq.path,
            "sequence_name": seq_base,
            "frame_count": len(seq_image_names),
            "extrap_results": extrap_results,
            "flagged_frames": flagged_frames,
            "flagged_edges": flagged_edges,
            "core_edges": core_edges,
            "pair_counts": pair_counts,
            "seq_image_names": seq_image_names,
            "seq_image_indexes": seq_image_indexes,
            "seq_frame_numbers": seq_frame_numbers,
            "seq_centers": seq_centers,
            "seq_quats": seq_quats,
            "median_trans": median_trans,
            "median_rot": median_rot,
            "step_ratios": step_ratios,
            "overlap_drops": overlap_drops,
            "obs_z_scores": obs_z_scores,
            "trans_threshold": trans_threshold,
            "rot_threshold": rot_threshold,
            "core_edge_reproj_errors": reproj_errors,
        }
        all_sequence_results.append(seq_result)

        # Render this sequence's per-frame table.
        print_frame_table(seq_result)

    # Summary section: all sequences grouped at the end.
    print_summary(all_sequence_results)

    return all_sequence_results
