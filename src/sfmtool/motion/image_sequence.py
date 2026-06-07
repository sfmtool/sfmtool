# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Discontinuity analysis for image sequences via optical flow."""

from pathlib import Path

import click
import numpy as np

from .flow_stats import (
    _compare_flow_representations,
    _compute_in_bounds_mask,
    _flow_magnitude,
    _load_gray,
)
from .._sfmtool import (
    compute_optical_flow,
    compute_optical_flow_with_init,
)
from ..visualization._discontinuity_display import (
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
