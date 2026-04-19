# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Optical flow primitives used by discontinuity analysis."""

from pathlib import Path

import cv2
import numpy as np


def _load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _flow_magnitude(flow_u: np.ndarray, flow_v: np.ndarray) -> np.ndarray:
    """Compute per-pixel flow magnitude."""
    return np.sqrt(flow_u**2 + flow_v**2)


def _compute_in_bounds_mask(flow_u: np.ndarray, flow_v: np.ndarray) -> np.ndarray:
    """Compute a boolean mask of pixels whose flow stays within the image.

    A pixel at (x, y) is in-bounds if (x + flow_u, y + flow_v) lands inside
    the image dimensions.
    """
    h, w = flow_u.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dest_x = xx + flow_u
    dest_y = yy + flow_v
    return (dest_x >= 0) & (dest_x < w) & (dest_y >= 0) & (dest_y < h)


def _flow_histogram_6x6(
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a 6x6 directional histogram of (u, v) flow vectors.

    Bin range is 1/3 of the larger image dimension on each side of zero,
    giving 6 bins that span the full possible motion range. Returns the
    histogram normalized to sum to 1 (percentages).

    If mask is provided, only pixels where mask is True are included.
    """
    h, w = flow_u.shape
    extent = max(h, w) / 3.0

    if mask is not None:
        u_vals = flow_u[mask]
        v_vals = flow_v[mask]
    else:
        u_vals = flow_u.ravel()
        v_vals = flow_v.ravel()

    hist, _, _ = np.histogram2d(
        v_vals,  # v (vertical) on rows
        u_vals,  # u (horizontal) on columns
        bins=6,
        range=[[-extent, extent], [-extent, extent]],
    )
    total = hist.sum()
    if total > 0:
        hist = hist / total * 100.0
    return hist


def _flow_tile_means(
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    grid_size: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute mean flow vector per spatial tile.

    Returns an (grid_size, grid_size, 2) array of mean (u, v) per tile.
    If mask is provided, only pixels where mask is True are included.
    """
    h, w = flow_u.shape
    tile_h = h // grid_size
    tile_w = w // grid_size
    means = np.zeros((grid_size, grid_size, 2), dtype=np.float64)
    for ty in range(grid_size):
        for tx in range(grid_size):
            y0 = ty * tile_h
            y1 = y0 + tile_h if ty < grid_size - 1 else h
            x0 = tx * tile_w
            x1 = x0 + tile_w if tx < grid_size - 1 else w
            if mask is not None:
                tile_mask = mask[y0:y1, x0:x1]
                if tile_mask.any():
                    means[ty, tx, 0] = flow_u[y0:y1, x0:x1][tile_mask].mean()
                    means[ty, tx, 1] = flow_v[y0:y1, x0:x1][tile_mask].mean()
            else:
                means[ty, tx, 0] = flow_u[y0:y1, x0:x1].mean()
                means[ty, tx, 1] = flow_v[y0:y1, x0:x1].mean()
    return means


def _flow_tile_magnitudes(
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    grid_size: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute median flow magnitude per spatial tile.

    Returns a (grid_size, grid_size) array.
    If mask is provided, only pixels where mask is True are included.
    """
    h, w = flow_u.shape
    tile_h = h // grid_size
    tile_w = w // grid_size
    mags = np.zeros((grid_size, grid_size), dtype=np.float64)
    for ty in range(grid_size):
        for tx in range(grid_size):
            y0 = ty * tile_h
            y1 = y0 + tile_h if ty < grid_size - 1 else h
            x0 = tx * tile_w
            x1 = x0 + tile_w if tx < grid_size - 1 else w
            tile_u = flow_u[y0:y1, x0:x1]
            tile_v = flow_v[y0:y1, x0:x1]
            tile_mag = _flow_magnitude(tile_u, tile_v)
            if mask is not None:
                tile_mask = mask[y0:y1, x0:x1]
                if tile_mask.any():
                    mags[ty, tx] = np.median(tile_mag[tile_mask])
            else:
                mags[ty, tx] = np.median(tile_mag)
    return mags


def _compare_flow_representations(
    local_u: np.ndarray,
    local_v: np.ndarray,
    stride_u: np.ndarray,
    stride_v: np.ndarray,
    stride: int,
    grid_size: int = 3,
) -> dict:
    """Compare local flow (scaled by stride) against stride flow.

    Returns per-tile magnitude grids, mean vectors, and 6x6 directional
    histograms for both flows.
    """
    scaled_local_u = local_u * stride
    scaled_local_v = local_v * stride

    # Mask out pixels whose scaled local flow would leave the image.
    # These pixels can't contribute meaningful stride flow either, so
    # exclude them from all comparisons for cleaner data.
    in_bounds = _compute_in_bounds_mask(scaled_local_u, scaled_local_v)

    # Per-tile magnitudes (masked)
    local_tile_mags = _flow_tile_magnitudes(
        scaled_local_u, scaled_local_v, grid_size, mask=in_bounds
    )
    stride_tile_mags = _flow_tile_magnitudes(
        stride_u, stride_v, grid_size, mask=in_bounds
    )

    # Per-tile mean vectors (masked)
    local_tile_means = _flow_tile_means(
        scaled_local_u, scaled_local_v, grid_size, mask=in_bounds
    )
    stride_tile_means = _flow_tile_means(stride_u, stride_v, grid_size, mask=in_bounds)

    # 6x6 directional histograms (masked)
    local_hist = _flow_histogram_6x6(scaled_local_u, scaled_local_v, mask=in_bounds)
    stride_hist = _flow_histogram_6x6(stride_u, stride_v, mask=in_bounds)

    total_pixels = scaled_local_u.shape[0] * scaled_local_u.shape[1]
    in_bounds_count = int(in_bounds.sum())

    # Difference: stride - scaled local
    diff_tile_mags = stride_tile_mags - local_tile_mags
    diff_tile_means = stride_tile_means - local_tile_means

    return {
        "local_tile_mags": local_tile_mags,
        "stride_tile_mags": stride_tile_mags,
        "diff_tile_mags": diff_tile_mags,
        "local_tile_means": local_tile_means,
        "stride_tile_means": stride_tile_means,
        "diff_tile_means": diff_tile_means,
        "local_hist": local_hist,
        "stride_hist": stride_hist,
        "in_bounds_pct": in_bounds_count / total_pixels * 100.0,
    }
