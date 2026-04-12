# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Flow-based feature matching for video sequences.

Uses dense optical flow to find feature correspondences between sequential
image pairs. Instead of using ANN descriptor matching to find candidate
correspondences, this advects keypoint positions through the flow field and
matches by spatial proximity, using descriptor distance only as a filter on
spatially-matched candidates.

Uses a sliding window approach: at each step, computes the adjacent flow
and advects all tracked keypoint positions one hop forward. This produces
matches at multiple baselines in a single O(N) sweep with O(window_size *
N_features) memory — much less than the previous approach of composing
full-resolution flow fields.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .._sfmtool import (
    KdTree2d,
    advect_points as _rust_advect_points,
    compute_optical_flow as _rust_compute_optical_flow,
    match_candidates_by_descriptor as _rust_match_candidates,
)
from .._sift_file import SiftReader


_SPATIAL_CANDIDATES_K = 5
_SPATIAL_CANDIDATES_RADIUS = 10.0


def _flow_match_pair(
    positions1: np.ndarray,
    positions2: np.ndarray,
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    spatial_tolerance: float,
    descriptor_threshold: float,
) -> np.ndarray:
    """Match features between a single image pair using precomputed flow.

    Advects keypoints from image 1 through the flow field, finds the closest
    K candidate keypoints in image 2 within spatial_tolerance, then selects
    the candidate with the best descriptor match for each source feature.

    Args:
        positions1: (N, 2) float32 keypoint positions in image 1 (x, y).
        positions2: (M, 2) float32 keypoint positions in image 2 (x, y).
        descriptors1: (N, 128) uint8 descriptors for image 1.
        descriptors2: (M, 128) uint8 descriptors for image 2.
        flow_u: (H, W) float32 horizontal displacement field.
        flow_v: (H, W) float32 vertical displacement field.
        spatial_tolerance: Max pixel distance for spatial matching.
        descriptor_threshold: Max L2 descriptor distance for filtering.

    Returns:
        (K, 2) uint32 array of (feat1_idx, feat2_idx) matched pairs.
    """
    if len(positions1) == 0 or len(positions2) == 0:
        return np.zeros((0, 2), dtype=np.uint32)

    h, w = flow_u.shape

    # Advect image1 keypoints through flow
    pos1_f32 = np.asarray(positions1, dtype=np.float32)
    advected = _rust_advect_points(pos1_f32, flow_u, flow_v)

    # Filter to advected points that land in-bounds
    in_bounds = (
        (advected[:, 0] >= 0)
        & (advected[:, 0] < w)
        & (advected[:, 1] >= 0)
        & (advected[:, 1] < h)
    )
    in_bounds_idx = np.where(in_bounds)[0]
    if len(in_bounds_idx) == 0:
        return np.zeros((0, 2), dtype=np.uint32)

    advected_in_bounds = advected[in_bounds_idx]

    # Find the closest K candidate keypoints within spatial_tolerance for each
    # advected point, then select the one with the best descriptor match.
    target_f32 = np.asarray(positions2, dtype=np.float32)
    tree = KdTree2d(target_f32)
    k = _SPATIAL_CANDIDATES_K
    # (Q, K) uint32 — indices into positions2, u32::MAX for empty slots
    candidates = tree.nearest_k_within_radius(
        advected_in_bounds, k, _SPATIAL_CANDIDATES_RADIUS
    )

    in_bounds_idx_u32 = np.asarray(in_bounds_idx, dtype=np.uint32)
    desc1_u8 = np.ascontiguousarray(descriptors1, dtype=np.uint8)
    desc2_u8 = np.ascontiguousarray(descriptors2, dtype=np.uint8)

    return _rust_match_candidates(
        candidates, in_bounds_idx_u32, desc1_u8, desc2_u8, descriptor_threshold
    )


def _flow_match_from_advected(
    advected_positions: np.ndarray,
    valid_mask: np.ndarray,
    descriptors1: np.ndarray,
    positions2: np.ndarray,
    descriptors2: np.ndarray,
    spatial_tolerance: float,
    descriptor_threshold: float,
) -> np.ndarray:
    """Match features using pre-advected positions (no flow field needed).

    Args:
        advected_positions: (N, 2) float32 predicted positions in target image.
        valid_mask: (N,) bool mask of positions that are still in-bounds.
        descriptors1: (N, 128) uint8 descriptors for source image.
        positions2: (M, 2) float32 keypoint positions in target image.
        descriptors2: (M, 128) uint8 descriptors for target image.
        spatial_tolerance: Max pixel distance for spatial matching.
        descriptor_threshold: Max L2 descriptor distance for filtering.

    Returns:
        (K, 2) uint32 array of (feat1_idx, feat2_idx) matched pairs.
    """
    in_bounds_idx = np.where(valid_mask)[0]
    if len(in_bounds_idx) == 0 or len(positions2) == 0:
        return np.zeros((0, 2), dtype=np.uint32)

    advected_in_bounds = advected_positions[in_bounds_idx]

    target_f32 = np.asarray(positions2, dtype=np.float32)
    tree = KdTree2d(target_f32)
    candidates = tree.nearest_k_within_radius(
        advected_in_bounds, _SPATIAL_CANDIDATES_K, _SPATIAL_CANDIDATES_RADIUS
    )

    in_bounds_idx_u32 = np.asarray(in_bounds_idx, dtype=np.uint32)
    desc1_u8 = np.ascontiguousarray(descriptors1, dtype=np.uint8)
    desc2_u8 = np.ascontiguousarray(descriptors2, dtype=np.uint8)

    return _rust_match_candidates(
        candidates, in_bounds_idx_u32, desc1_u8, desc2_u8, descriptor_threshold
    )


def _load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _load_features(
    sift_path: Path, max_feature_count: Optional[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Load SIFT positions and descriptors from a .sift file."""
    with SiftReader(sift_path) as reader:
        positions = reader.read_positions(count=max_feature_count)
        descriptors = reader.read_descriptors(count=max_feature_count)
    return positions, descriptors


def flow_match_sequential(
    image_paths: list[Path],
    sift_paths: list[Path],
    preset: str = "default",
    spatial_tolerance: float = 3.0,
    descriptor_threshold: float = 250.0,
    window_size: int = 5,
    max_feature_count: Optional[int] = None,
    trace_path: Optional[Path] = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Match features across a sequence of images using optical flow.

    Uses a sliding window: at each step, computes flow(j-1, j), then
    advects all tracked keypoint positions one hop forward through that
    flow. This produces matches at multiple baselines in a single O(N)
    sweep with O(window_size * N_features) memory.

    Args:
        image_paths: Ordered sequence of image file paths.
        sift_paths: Corresponding .sift file paths.
        preset: Optical flow quality preset ("fast", "default", "high_quality").
        spatial_tolerance: Max pixel distance for advected→keypoint matching.
        descriptor_threshold: Max L2 descriptor distance for filtering.
        window_size: Number of source images to track. Produces matches
            from adjacent (skip=1) up to skip=window_size.
        max_feature_count: Max features per image (None = all).
        trace_path: If set, write a Chrome Trace Event Format JSON file
            for visualization in Perfetto / chrome://tracing.

    Returns:
        Dict mapping (image_idx_i, image_idx_j) to (M, 2) uint32 array
        of matched feature index pairs.
    """
    n_images = len(image_paths)
    if n_images < 2:
        return {}

    # Tracing support (Chrome Trace Event Format for Perfetto)
    trace_events = [] if trace_path is not None else None
    trace_lock = threading.Lock() if trace_events is not None else None
    trace_t0 = time.perf_counter() if trace_events is not None else 0.0
    _tid_map: dict[int, int] = {}
    _tid_counter = [0]

    def _trace(name, cat, t_start, t_end, **args):
        if trace_events is None:
            return
        ident = threading.get_ident()
        with trace_lock:
            if ident not in _tid_map:
                _tid_map[ident] = _tid_counter[0]
                _tid_counter[0] += 1
            tid = _tid_map[ident]
        ts_us = (t_start - trace_t0) * 1_000_000
        dur_us = (t_end - t_start) * 1_000_000
        event = {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": ts_us,
            "dur": dur_us,
            "pid": 1,
            "tid": tid,
        }
        if args:
            event["args"] = args
        with trace_lock:
            trace_events.append(event)

    all_matches: dict[tuple[int, int], np.ndarray] = {}

    # Window of tracked keypoint positions. Each entry tracks where a source
    # image's keypoints have landed after being advected through successive
    # adjacent flow fields.
    # (source_image_idx, advected_positions (N,2) f32, valid_mask (N,) bool,
    #  original_positions (N,2) f32, descriptors (N,128) u8)
    window: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    print(f"Flow matching {n_images} images (window={window_size})...", flush=True)

    # Stream images and features through the sliding window.
    prev_gray = _load_gray(image_paths[0])
    prev_pos, prev_desc = _load_features(sift_paths[0], max_feature_count)

    def _compute_flow_traced(gray_a, gray_b, a_idx, b_idx):
        """Compute optical flow with tracing — runs on background thread."""
        t0 = time.perf_counter()
        result = _rust_compute_optical_flow(gray_a, gray_b, preset=preset)
        _trace(f"optical_flow({a_idx},{b_idx})", "gpu", t0, time.perf_counter())
        return result

    # Submit the first flow computation to the background thread.
    executor = ThreadPoolExecutor(max_workers=1)
    next_gray = _load_gray(image_paths[1])
    next_flow_future = executor.submit(_compute_flow_traced, prev_gray, next_gray, 0, 1)

    for j in range(1, n_images):
        curr_gray = next_gray
        curr_pos, curr_desc = _load_features(sift_paths[j], max_feature_count)

        # Block until this flow is ready.
        new_flow_u, new_flow_v = next_flow_future.result()
        h, w = new_flow_u.shape

        # Submit the next flow computation while we do advect+match.
        if j + 1 < n_images:
            next_gray = _load_gray(image_paths[j + 1])
            next_flow_future = executor.submit(
                _compute_flow_traced, curr_gray, next_gray, j, j + 1
            )

        # Advect all tracked positions one hop through the new adjacent flow
        for k in range(len(window)):
            src_idx, positions, valid, src_pos, src_desc = window[k]
            valid_idx = np.where(valid)[0]
            if len(valid_idx) == 0:
                continue

            t0 = time.perf_counter()
            new_positions = _rust_advect_points(
                positions[valid_idx], new_flow_u, new_flow_v
            )
            _trace(f"advect({src_idx},{j})", "cpu", t0, time.perf_counter())

            # Update positions and kill points that went out of bounds
            still_valid = (
                (new_positions[:, 0] >= 0)
                & (new_positions[:, 0] < w)
                & (new_positions[:, 1] >= 0)
                & (new_positions[:, 1] < h)
            )
            positions[valid_idx] = new_positions
            valid[valid_idx] &= still_valid
            window[k] = (src_idx, positions, valid, src_pos, src_desc)

        # Add new slot: source=image j-1, positions start at their keypoints
        new_slot_pos = np.asarray(prev_pos, dtype=np.float32).copy()
        # Advect the just-added slot through the current flow (j-1 → j)
        t0 = time.perf_counter()
        advected_new = _rust_advect_points(new_slot_pos, new_flow_u, new_flow_v)
        _trace(f"advect({j - 1},{j})", "cpu", t0, time.perf_counter())

        new_valid = (
            (advected_new[:, 0] >= 0)
            & (advected_new[:, 0] < w)
            & (advected_new[:, 1] >= 0)
            & (advected_new[:, 1] < h)
        )
        window.append((j - 1, advected_new, new_valid, prev_pos, prev_desc))

        # Trim window to max size
        if len(window) > window_size:
            window.pop(0)

        # Match all window source images against the current image
        for k in range(len(window)):
            src_idx, positions, valid, src_pos, src_desc = window[k]

            t0 = time.perf_counter()
            matches = _flow_match_from_advected(
                positions,
                valid,
                src_desc,
                curr_pos,
                curr_desc,
                spatial_tolerance,
                descriptor_threshold,
            )
            _trace(
                f"match({src_idx},{j})",
                "cpu",
                t0,
                time.perf_counter(),
                n_matches=len(matches),
            )
            if len(matches) > 0:
                all_matches[(src_idx, j)] = matches
                skip = j - src_idx
                if skip == 1:
                    print(
                        f"  [{j}/{n_images - 1}] Pair ({src_idx}, {j}): {len(matches)} matches",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{j}/{n_images - 1}] Pair ({src_idx}, {j}): {len(matches)} matches (skip={skip})",
                        flush=True,
                    )

        # Retain current features for next iteration (they become prev)
        prev_pos, prev_desc = curr_pos, curr_desc

    executor.shutdown(wait=False)

    total = sum(len(m) for m in all_matches.values())
    print(f"Flow matching complete: {total} matches across {len(all_matches)} pairs")

    if trace_events is not None and trace_path is not None:
        for ident, tid in _tid_map.items():
            label = "main" if tid == 0 else f"flow-worker-{tid}"
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 1,
                    "tid": tid,
                    "args": {"name": label},
                }
            )
        trace_data = {"traceEvents": trace_events}
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as f:
            json.dump(trace_data, f)
        print(f"Trace written to {trace_path}")

    return all_matches
