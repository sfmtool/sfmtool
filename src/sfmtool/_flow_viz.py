# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""
Optical flow visualization for image pairs.

This module provides functionality to visualize dense optical flow between
two images, optionally comparing flow-based keypoint advection against
correspondences from an SfM reconstruction.
"""

import colorsys
import random
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ._sfmtool import KdTree2d, advect_points as _rust_advect_points
from ._sfmtool import compute_optical_flow as _rust_compute_optical_flow
from ._histogram_utils import print_histogram
from ._sift_file import SiftReader, get_sift_path_for_image

if TYPE_CHECKING:
    from ._sfmtool import SfmrReconstruction


def _find_nearest_within_tolerance(
    query_points: np.ndarray,
    target_points: np.ndarray,
    tolerance: float,
) -> dict[int, tuple[int, float]]:
    """Find nearest target point for each query point within tolerance.

    Returns dict mapping query_idx -> (target_idx, distance) for points
    within tolerance. Uses a KD-tree for efficient nearest-neighbor search.
    """
    if len(query_points) == 0 or len(target_points) == 0:
        return {}

    tree = KdTree2d(np.asarray(target_points, dtype=np.float32))
    query_f32 = np.asarray(query_points, dtype=np.float32)
    nearest_idx = tree.nearest(query_f32)

    nearest_pos = target_points[nearest_idx]
    diffs = query_f32 - nearest_pos
    distances = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)

    result: dict[int, tuple[int, float]] = {}
    for qi in np.where(distances <= tolerance)[0]:
        result[int(qi)] = (int(nearest_idx[qi]), float(distances[qi]))

    return result


def _get_color_palette(n_colors: int) -> list:
    """Generate a cycling color palette with distinct colors randomized.

    Returns list of (B, G, R) tuples for use with cv2.
    """
    colors = []
    for i in range(n_colors):
        hue = i / max(n_colors, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    random.Random(42).shuffle(colors)
    return colors


def _flow_to_color(flow_u: np.ndarray, flow_v: np.ndarray) -> np.ndarray:
    """Convert optical flow to color using the Middlebury color wheel convention.

    Direction maps to hue, magnitude maps to saturation. Zero flow is white,
    strong flow is vivid color. Value is always full brightness, so small
    motions show as pastel tints rather than vanishing into black.

    Returns BGR image (uint8).
    """
    mag = np.sqrt(flow_u**2 + flow_v**2)
    angle = np.arctan2(flow_v, flow_u)

    # Normalize magnitude for visualization (99th percentile avoids outlier blowout)
    max_mag = np.percentile(mag, 99) if mag.max() > 0 else 1.0
    mag_norm = np.clip(mag / max_mag, 0, 1)

    hsv = np.zeros((*flow_u.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[:, :, 1] = (mag_norm * 255).astype(np.uint8)  # Saturation = magnitude
    hsv[:, :, 2] = 255  # Value = full brightness

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _get_shared_feature_pairs(
    recon: "SfmrReconstruction",
    image1_name: str,
    image2_name: str,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Extract shared feature pairs between two images from reconstruction tracks.

    Returns (image1_idx, image2_idx, list of (feat1_idx, feat2_idx) pairs).
    """
    image_names = recon.image_names
    image_indexes = recon.track_image_indexes
    feature_indexes = recon.track_feature_indexes
    point_ids = recon.track_point_ids

    observations = np.column_stack([image_indexes, feature_indexes])

    image1_idx = None
    image2_idx = None
    for idx, name in enumerate(image_names):
        if Path(name).name == Path(image1_name).name or name == image1_name:
            image1_idx = idx
        if Path(name).name == Path(image2_name).name or name == image2_name:
            image2_idx = idx

    if image1_idx is None:
        raise ValueError(f"Image '{image1_name}' not found in reconstruction")
    if image2_idx is None:
        raise ValueError(f"Image '{image2_name}' not found in reconstruction")

    obs1_mask = observations[:, 0] == image1_idx
    obs2_mask = observations[:, 0] == image2_idx

    obs1 = observations[obs1_mask]
    obs2 = observations[obs2_mask]
    point_ids1 = point_ids[obs1_mask]
    point_ids2 = point_ids[obs2_mask]

    shared_point_ids = np.intersect1d(point_ids1, point_ids2)

    feature_pairs = []
    for point_id in shared_point_ids:
        feat1_idx = obs1[point_ids1 == point_id, 1][0]
        feat2_idx = obs2[point_ids2 == point_id, 1][0]
        feature_pairs.append((int(feat1_idx), int(feat2_idx)))

    return image1_idx, image2_idx, feature_pairs


def draw_flow_visualization(
    image1_path: str | Path,
    image2_path: str | Path,
    output_path: str | Path | None = None,
    preset: str = "default",
    feature_size: int = 4,
    line_thickness: int = 1,
    max_features: int | None = None,
    side_by_side: bool = False,
    recon: "SfmrReconstruction | None" = None,
    advection_tolerance: float = 3.0,
    descriptor_threshold: float = 100.0,
) -> None:
    """Visualize optical flow between two images.

    Without a reconstruction (recon=None):
        Draws flow-colored arrows from SIFT keypoint positions in image1
        to their advected positions, overlaid on the target image (image2).
        A side-by-side or separate output is produced.

    With a reconstruction (recon provided):
        Compares flow advection against reconstruction correspondences:
        - GREEN arrows: flow advection agrees with sfmr correspondence
          (advected position lands within tolerance of the matched keypoint)
        - RED arrows: sfmr correspondences that flow does NOT explain
          (advected position is far from the matched keypoint)
        - YELLOW dots: flow advection hits (keypoints that land near a
          keypoint in image2) that are NOT sfmr correspondences
    """
    image1_path = Path(image1_path)
    image2_path = Path(image2_path)
    if output_path is not None:
        output_path = Path(output_path)

    # Load images
    img1_bgr = cv2.imread(str(image1_path))
    img2_bgr = cv2.imread(str(image2_path))
    if img1_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image1_path}")
    if img2_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image2_path}")

    # Convert to grayscale for flow computation
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    # Compute optical flow using Rust DIS implementation
    print(f"Computing optical flow ({preset} preset)...")
    flow_u, flow_v = _rust_compute_optical_flow(gray1, gray2, preset=preset)
    mag = np.sqrt(flow_u**2 + flow_v**2)
    print(
        f"  Flow magnitude: mean={mag.mean():.1f}, max={mag.max():.1f}, "
        f"median={np.median(mag):.1f} px"
    )

    # Print flow component histograms
    # Use symmetric range so zero is visually centered
    u_flat = flow_u.ravel()
    v_flat = flow_v.ravel()
    max_abs = max(np.abs(u_flat).max(), np.abs(v_flat).max(), 1.0)
    print_histogram(u_flat, "Flow X (horizontal)", min_val=-max_abs, max_val=max_abs)
    print_histogram(v_flat, "Flow Y (vertical)", min_val=-max_abs, max_val=max_abs)

    # Load SIFT features (workspace lookup handled by get_sift_path_for_image)
    sift1_path = get_sift_path_for_image(image1_path)
    sift2_path = get_sift_path_for_image(image2_path)

    if not sift1_path.exists():
        raise FileNotFoundError(f"SIFT file not found: {sift1_path}")
    if not sift2_path.exists():
        raise FileNotFoundError(f"SIFT file not found: {sift2_path}")

    with SiftReader(sift1_path) as reader:
        positions1 = reader.read_positions()
    with SiftReader(sift2_path) as reader:
        positions2 = reader.read_positions()

    print(f"  Image 1: {len(positions1)} keypoints")
    print(f"  Image 2: {len(positions2)} keypoints")

    # Advect image1 keypoints to image2 using flow
    positions1_f32 = positions1.astype(np.float32)
    advected = _rust_advect_points(positions1_f32, flow_u, flow_v)

    # Build spatial index for image2 keypoints (for proximity matching)
    h2, w2 = img2_bgr.shape[:2]

    # Find which advected points land near an image2 keypoint
    in_bounds = (
        (advected[:, 0] >= 0)
        & (advected[:, 0] < w2)
        & (advected[:, 1] >= 0)
        & (advected[:, 1] < h2)
    )
    in_bounds_indices = np.where(in_bounds)[0]
    advected_in_bounds = advected[in_bounds_indices]

    hits = _find_nearest_within_tolerance(
        advected_in_bounds, positions2, advection_tolerance
    )

    # Map back to original indices
    hit_pairs = []  # (img1_kp_idx, img2_kp_idx)
    hit_distances = []  # spatial distance for each hit
    for local_qi, (ti, dist) in hits.items():
        original_qi = in_bounds_indices[local_qi]
        hit_pairs.append((int(original_qi), int(ti)))
        hit_distances.append(dist)

    print(
        f"  Flow advection hits: {len(hit_pairs)} keypoints land within "
        f"{advection_tolerance}px of an image2 keypoint"
    )

    if len(hit_distances) > 0:
        hit_dist_arr = np.array(hit_distances)
        print_histogram(hit_dist_arr, "Advection hit distance (px)")

        # Compute descriptor distances for hit pairs
        with SiftReader(sift1_path) as reader:
            descriptors1 = reader.read_descriptors()
        with SiftReader(sift2_path) as reader:
            descriptors2 = reader.read_descriptors()

        idx1 = np.array([p[0] for p in hit_pairs])
        idx2 = np.array([p[1] for p in hit_pairs])
        desc_dists = np.linalg.norm(
            descriptors1[idx1].astype(np.float32)
            - descriptors2[idx2].astype(np.float32),
            axis=1,
        )
        print_histogram(desc_dists, "Descriptor distance (L2) for advection hits")

        # Filtered stats: only hits with descriptor distance below threshold
        good_mask = desc_dists <= descriptor_threshold
        n_good = good_mask.sum()
        print(
            f"  Descriptor-filtered hits (L2 <= {descriptor_threshold}): "
            f"{n_good} / {len(hit_pairs)} "
            f"({100 * n_good / len(hit_pairs):.1f}%)"
        )
        if n_good > 0:
            print_histogram(
                hit_dist_arr[good_mask],
                f"Advection hit distance (px), L2 <= {descriptor_threshold}",
            )
            print_histogram(
                desc_dists[good_mask],
                f"Descriptor distance (L2), filtered <= {descriptor_threshold}",
            )

    if recon is not None:
        # Get shared feature pairs from reconstruction
        _img1_idx, _img2_idx, sfmr_pairs = _get_shared_feature_pairs(
            recon, image1_path.name, image2_path.name
        )
        print(f"  SfMR correspondences: {len(sfmr_pairs)}")

        # Compute distance from advected position to SfMR target
        sfmr_distances = []
        for feat1_idx, feat2_idx in sfmr_pairs:
            if feat1_idx >= len(advected):
                sfmr_distances.append((feat1_idx, feat2_idx, float("inf")))
                continue
            adv_pos = advected[feat1_idx]
            target_pos = positions2[feat2_idx]
            dist = np.sqrt(
                (adv_pos[0] - target_pos[0]) ** 2 + (adv_pos[1] - target_pos[1]) ** 2
            )
            sfmr_distances.append((feat1_idx, feat2_idx, float(dist)))

        green_pairs = [
            (f1, f2) for f1, f2, d in sfmr_distances if d <= advection_tolerance
        ]
        red_pairs = [
            (f1, f2) for f1, f2, d in sfmr_distances if d > advection_tolerance
        ]

        finite_dists = np.array([d for _, _, d in sfmr_distances if np.isfinite(d)])
        if len(finite_dists) > 0:
            print_histogram(
                finite_dists,
                "SfMR advection error (flow predicted vs SfMR matched)",
                show_stats=False,
            )
            print(
                f"      Mean: {finite_dists.mean():.1f}px, "
                f"Median: {np.median(finite_dists):.1f}px, "
                f"P90: {np.percentile(finite_dists, 90):.1f}px, "
                f"Max: {finite_dists.max():.1f}px"
            )

        sfmr_set = set(sfmr_pairs)
        yellow_pairs = []
        for local_qi, (ti, _dist) in hits.items():
            original_qi = int(in_bounds_indices[local_qi])
            pair = (original_qi, int(ti))
            if pair not in sfmr_set:
                yellow_pairs.append(pair)

        print(
            f"  Flow agrees with SfMR (green, <={advection_tolerance}px): {len(green_pairs)}"
        )
        print(f"  SfMR not explained by flow (red): {len(red_pairs)}")
        print(f"  Flow hits not in SfMR (yellow): {len(yellow_pairs)}")

    if output_path is None:
        return

    if recon is not None:
        _draw_comparison_mode(
            img1_bgr=img1_bgr,
            img2_bgr=img2_bgr,
            positions1=positions1,
            positions2=positions2,
            advected=advected,
            flow_u=flow_u,
            flow_v=flow_v,
            recon=recon,
            image1_name=image1_path.name,
            image2_name=image2_path.name,
            output_path=output_path,
            feature_size=feature_size,
            line_thickness=line_thickness,
            max_features=max_features,
            side_by_side=side_by_side,
            advection_tolerance=advection_tolerance,
        )
    else:
        _draw_flow_only_mode(
            img1_bgr=img1_bgr,
            img2_bgr=img2_bgr,
            positions1=positions1,
            positions2=positions2,
            advected=advected,
            flow_u=flow_u,
            flow_v=flow_v,
            output_path=output_path,
            feature_size=feature_size,
            line_thickness=line_thickness,
            max_features=max_features,
            side_by_side=side_by_side,
            advection_tolerance=advection_tolerance,
        )


def _draw_flow_only_mode(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    positions1: np.ndarray,
    positions2: np.ndarray,
    advected: np.ndarray,
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    output_path: Path,
    feature_size: int,
    line_thickness: int,
    max_features: int | None,
    side_by_side: bool,
    advection_tolerance: float,
) -> None:
    """Draw flow visualization without reconstruction comparison.

    Shows flow color field on image1, and on image2 shows advected keypoints
    with arrows from image1 keypoints to where they land. Keypoints that
    land near an image2 keypoint are highlighted.
    """
    h2, w2 = img2_bgr.shape[:2]

    # Find which advected points land near an image2 keypoint
    in_bounds = (
        (advected[:, 0] >= 0)
        & (advected[:, 0] < w2)
        & (advected[:, 1] >= 0)
        & (advected[:, 1] < h2)
    )
    in_bounds_indices = np.where(in_bounds)[0]
    advected_in_bounds = advected[in_bounds_indices]

    hits = _find_nearest_within_tolerance(
        advected_in_bounds, positions2, advection_tolerance
    )

    # Map back to original indices
    hit_pairs = []  # (img1_kp_idx, img2_kp_idx)
    for local_qi, (ti, _dist) in hits.items():
        original_qi = in_bounds_indices[local_qi]
        hit_pairs.append((int(original_qi), int(ti)))

    # --- Save standalone flow color image ---
    _save_flow_color_image(flow_u, flow_v, output_path)

    # --- Draw image 1: source with keypoints ---
    vis1 = img1_bgr.copy()

    # Draw source keypoints for hits
    if max_features is not None:
        hit_pairs = hit_pairs[:max_features]

    colors = _get_color_palette(max(len(hit_pairs), 1))
    for i, (qi, _ti) in enumerate(hit_pairs):
        pt = (int(positions1[qi, 0]), int(positions1[qi, 1]))
        cv2.circle(vis1, pt, feature_size, colors[i % len(colors)], -1)

    # --- Draw image 2: advected positions with arrows ---
    vis2 = img2_bgr.copy()

    for i, (qi, ti) in enumerate(hit_pairs):
        src = (int(advected[qi, 0]), int(advected[qi, 1]))
        dst = (int(positions2[ti, 0]), int(positions2[ti, 1]))
        color = colors[i % len(colors)]
        # Draw advected position
        cv2.circle(vis2, src, feature_size, color, -1)
        # Draw actual keypoint position
        cv2.circle(vis2, dst, feature_size + 1, color, 1)
        # Draw line from advected to actual
        cv2.line(vis2, src, dst, color, line_thickness)

    _save_output(vis1, vis2, output_path, side_by_side)


def _draw_comparison_mode(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    positions1: np.ndarray,
    positions2: np.ndarray,
    advected: np.ndarray,
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    recon: "SfmrReconstruction",
    image1_name: str,
    image2_name: str,
    output_path: Path,
    feature_size: int,
    line_thickness: int,
    max_features: int | None,
    side_by_side: bool,
    advection_tolerance: float,
) -> None:
    """Draw flow vs reconstruction comparison visualization.

    Three categories of features are drawn:
    - GREEN: sfmr correspondences where flow agrees (advected pos near matched keypoint)
    - RED: sfmr correspondences where flow disagrees (advected pos far from matched keypoint)
    - YELLOW: flow hits that are NOT sfmr correspondences
    """
    h2, w2 = img2_bgr.shape[:2]

    # Get shared feature pairs from reconstruction
    _img1_idx, _img2_idx, sfmr_pairs = _get_shared_feature_pairs(
        recon, image1_name, image2_name
    )

    # Compute distance from advected position to SfMR target for each correspondence
    sfmr_distances = []  # (feat1_idx, feat2_idx, distance)
    for feat1_idx, feat2_idx in sfmr_pairs:
        if feat1_idx >= len(advected):
            sfmr_distances.append((feat1_idx, feat2_idx, float("inf")))
            continue
        adv_pos = advected[feat1_idx]
        target_pos = positions2[feat2_idx]
        dist = np.sqrt(
            (adv_pos[0] - target_pos[0]) ** 2 + (adv_pos[1] - target_pos[1]) ** 2
        )
        sfmr_distances.append((feat1_idx, feat2_idx, float(dist)))

    # Split into green/red by tolerance
    green_pairs = [(f1, f2) for f1, f2, d in sfmr_distances if d <= advection_tolerance]
    red_pairs = [(f1, f2) for f1, f2, d in sfmr_distances if d > advection_tolerance]

    # Find flow advection hits that are NOT sfmr correspondences
    in_bounds = (
        (advected[:, 0] >= 0)
        & (advected[:, 0] < w2)
        & (advected[:, 1] >= 0)
        & (advected[:, 1] < h2)
    )
    in_bounds_indices = np.where(in_bounds)[0]
    advected_in_bounds = advected[in_bounds_indices]

    sfmr_set = set(sfmr_pairs)
    flow_hits = _find_nearest_within_tolerance(
        advected_in_bounds, positions2, advection_tolerance
    )

    yellow_pairs = []  # (feat1_idx, feat2_idx) - flow hit but not in sfmr
    for local_qi, (ti, _dist) in flow_hits.items():
        original_qi = int(in_bounds_indices[local_qi])
        pair = (original_qi, int(ti))
        if pair not in sfmr_set:
            yellow_pairs.append(pair)

    # Apply max_features limit proportionally
    if max_features is not None:
        total = len(green_pairs) + len(red_pairs) + len(yellow_pairs)
        if total > max_features:
            ratio = max_features / total
            green_pairs = green_pairs[: max(1, int(len(green_pairs) * ratio))]
            red_pairs = red_pairs[: max(1, int(len(red_pairs) * ratio))]
            yellow_pairs = yellow_pairs[: max(1, int(len(yellow_pairs) * ratio))]

    # --- Save standalone flow color image ---
    _save_flow_color_image(flow_u, flow_v, output_path)

    # Colors (BGR)
    GREEN = (0, 200, 0)
    RED = (0, 0, 200)
    YELLOW = (0, 200, 200)

    # --- Draw image 1: source keypoints colored by category ---
    vis1 = img1_bgr.copy()

    for feat1_idx, _feat2_idx in yellow_pairs:
        if feat1_idx < len(positions1):
            pt = (int(positions1[feat1_idx, 0]), int(positions1[feat1_idx, 1]))
            cv2.circle(vis1, pt, feature_size, YELLOW, -1)

    for feat1_idx, _feat2_idx in red_pairs:
        if feat1_idx < len(positions1):
            pt = (int(positions1[feat1_idx, 0]), int(positions1[feat1_idx, 1]))
            cv2.circle(vis1, pt, feature_size, RED, -1)

    for feat1_idx, _feat2_idx in green_pairs:
        if feat1_idx < len(positions1):
            pt = (int(positions1[feat1_idx, 0]), int(positions1[feat1_idx, 1]))
            cv2.circle(vis1, pt, feature_size, GREEN, -1)

    # --- Draw image 2: target keypoints and advected positions ---
    vis2 = img2_bgr.copy()

    # Yellow: flow-only hits (dot at advected position)
    for feat1_idx, feat2_idx in yellow_pairs:
        if feat1_idx < len(advected):
            adv = (int(advected[feat1_idx, 0]), int(advected[feat1_idx, 1]))
            tgt = (int(positions2[feat2_idx, 0]), int(positions2[feat2_idx, 1]))
            cv2.circle(vis2, adv, feature_size, YELLOW, -1)
            cv2.circle(vis2, tgt, feature_size + 1, YELLOW, 1)
            cv2.line(vis2, adv, tgt, YELLOW, line_thickness)

    # Red: sfmr correspondence that flow misses
    for feat1_idx, feat2_idx in red_pairs:
        tgt = (int(positions2[feat2_idx, 0]), int(positions2[feat2_idx, 1]))
        cv2.circle(vis2, tgt, feature_size, RED, -1)
        if feat1_idx < len(advected):
            adv = (int(advected[feat1_idx, 0]), int(advected[feat1_idx, 1]))
            if 0 <= adv[0] < w2 and 0 <= adv[1] < img2_bgr.shape[0]:
                cv2.line(vis2, tgt, adv, RED, line_thickness)

    # Green: flow agrees with sfmr (draw both, they should be close)
    for feat1_idx, feat2_idx in green_pairs:
        tgt = (int(positions2[feat2_idx, 0]), int(positions2[feat2_idx, 1]))
        cv2.circle(vis2, tgt, feature_size, GREEN, -1)
        if feat1_idx < len(advected):
            adv = (int(advected[feat1_idx, 0]), int(advected[feat1_idx, 1]))
            cv2.circle(vis2, adv, feature_size - 1, GREEN, 1)

    _save_output(vis1, vis2, output_path, side_by_side)


def _direction_color_bgr(u: float, v: float) -> tuple[int, int, int]:
    """Get the Middlebury color for a unit flow direction, fully saturated."""
    angle = np.arctan2(v, u)
    hue = int((angle + np.pi) / (2 * np.pi) * 179)
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def _draw_flow_legend(image: np.ndarray) -> None:
    """Draw a direction color legend in the top-left corner of the flow image."""
    directions = [
        ("Right", 1, 0),
        ("Left", -1, 0),
        ("Down", 0, 1),
        ("Up", 0, -1),
    ]

    # Scale legend relative to image height so it's readable at any resolution
    h = image.shape[0]
    scale = max(h / 500, 1.0)
    swatch_size = int(14 * scale)
    padding = int(8 * scale)
    gap = int(6 * scale)
    line_height = swatch_size + gap
    text_scale = 0.45 * scale
    text_thickness = max(1, int(scale))

    # Measure text to size the background
    max_text_w = 0
    for label, _, _ in directions:
        (tw, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        max_text_w = max(max_text_w, tw)

    bg_w = padding + swatch_size + gap + max_text_w + padding
    bg_h = padding + line_height * len(directions) + padding

    # Semi-transparent dark background
    roi = image[0:bg_h, 0:bg_w]
    overlay = roi.copy()
    overlay[:] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, roi)

    y = padding
    for label, u, v in directions:
        color = _direction_color_bgr(u, v)
        # Draw swatch
        x0 = padding
        cv2.rectangle(image, (x0, y), (x0 + swatch_size, y + swatch_size), color, -1)
        # Draw label
        text_x = x0 + swatch_size + gap
        text_y = y + swatch_size - 2
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
        y += line_height


def _save_flow_color_image(
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    output_path: Path,
) -> None:
    """Save standalone flow color visualization as a separate image file.

    Uses the Middlebury color wheel convention: direction maps to hue,
    magnitude maps to saturation, white means zero flow.
    """
    flow_color = _flow_to_color(flow_u, flow_v)
    _draw_flow_legend(flow_color)
    stem = output_path.stem
    suffix = output_path.suffix
    flow_path = output_path.parent / f"{stem}_flow{suffix}"
    cv2.imwrite(str(flow_path), flow_color)
    print(f"Saved flow color image to {flow_path}")


def _save_output(
    vis1: np.ndarray,
    vis2: np.ndarray,
    output_path: Path,
    side_by_side: bool,
) -> None:
    """Save visualization as side-by-side or separate images."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if side_by_side:
        # Resize to same height if needed
        h1, h2 = vis1.shape[0], vis2.shape[0]
        if h1 != h2:
            target_h = max(h1, h2)
            if h1 < target_h:
                scale = target_h / h1
                vis1 = cv2.resize(vis1, (int(vis1.shape[1] * scale), target_h))
            if h2 < target_h:
                scale = target_h / h2
                vis2 = cv2.resize(vis2, (int(vis2.shape[1] * scale), target_h))
        combined = np.hstack([vis1, vis2])
        cv2.imwrite(str(output_path), combined)
        print(f"Saved side-by-side visualization to {output_path}")
    else:
        stem = output_path.stem
        suffix = output_path.suffix
        path_a = output_path.parent / f"{stem}_A{suffix}"
        path_b = output_path.parent / f"{stem}_B{suffix}"
        cv2.imwrite(str(path_a), vis1)
        cv2.imwrite(str(path_b), vis2)
        print(f"Saved visualizations to {path_a} and {path_b}")
