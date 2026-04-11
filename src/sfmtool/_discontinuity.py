# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Discontinuity analysis for image sequences and reconstructions."""

from pathlib import Path

import click
import cv2
import numpy as np

from ._flow_viz import _draw_flow_legend, _flow_to_color
from ._sfmtool import (
    RotQuaternion,
    compute_optical_flow,
    compute_optical_flow_with_init,
)


def _load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8."""
    img = cv2.imread(str(path))
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


_DIRECTION_ARROWS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]


def _arrow_for_vector(u: float, v: float) -> str:
    """Return a Unicode arrow character for the dominant direction of (u, v)."""
    if u * u + v * v < 0.01:
        return "·"
    # atan2 with -v because screen y is downward
    angle = np.arctan2(-v, u)
    # Quantize to 8 directions
    idx = int(round(angle / (np.pi / 4))) % 8
    return _DIRECTION_ARROWS[idx]


_BAR_BLOCKS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587"


def _render_bar(value: float, max_value: float) -> str:
    """Render a single value as a block character. Uses 7 levels (up to ▇)."""
    if max_value <= 0 or value <= 0:
        return " "
    level = int((value / max_value) * 7)
    level = max(1, min(7, level))
    return _BAR_BLOCKS[level]


def _print_bar_grids_side_by_side(
    label_a: str,
    grid_a: np.ndarray,
    label_b: str,
    grid_b: np.ndarray,
):
    """Print two 2D grids as block bars side by side with a shared range."""
    max_val = max(grid_a.max(), grid_b.max())

    rows = grid_a.shape[0]
    cols_a = grid_a.shape[1]
    cols_b = grid_b.shape[1]

    # Center labels over their boxes (border + content + border)
    box_width_a = cols_a + 2  # │ + content + │
    box_width_b = cols_b + 2
    click.echo(f"  {label_a:^{box_width_a}s}   {label_b:^{box_width_b}s}")
    click.echo(f"  Flow direction (0,0 at center), count 0 to {max_val:.0f}:")

    top_a = "\u250c" + "\u2500" * cols_a + "\u2510"
    top_b = "\u250c" + "\u2500" * cols_b + "\u2510"
    click.echo(f"  {top_a}   {top_b}")

    for r in range(rows):
        bar_a = "".join(_render_bar(grid_a[r, c], max_val) for c in range(cols_a))
        bar_b = "".join(_render_bar(grid_b[r, c], max_val) for c in range(cols_b))
        click.echo(f"  \u2502{bar_a}\u2502   \u2502{bar_b}\u2502")

    bot_a = "\u2514" + "\u2500" * cols_a + "\u2518"
    bot_b = "\u2514" + "\u2500" * cols_b + "\u2518"
    click.echo(f"  {bot_a}   {bot_b}")


def _format_mag_row(mags: np.ndarray, means: np.ndarray, row: int) -> str:
    """Format one row of a magnitude grid with direction arrows."""
    cols = mags.shape[1]
    col_width = 10
    cells = []
    for c in range(cols):
        arrow = _arrow_for_vector(means[row, c, 0], means[row, c, 1])
        cells.append(f"{mags[row, c]:.1f}{arrow}")
    return " ".join(f"{c:>{col_width}s}" for c in cells)


def _format_diff_row(diffs: np.ndarray, means: np.ndarray, row: int) -> str:
    """Format one row of a signed difference grid with direction arrows."""
    cols = diffs.shape[1]
    col_width = 10
    cells = []
    for c in range(cols):
        val = diffs[row, c]
        arrow = _arrow_for_vector(means[row, c, 0], means[row, c, 1])
        cells.append(f"{val:+.1f}{arrow}")
    return " ".join(f"{c:>{col_width}s}" for c in cells)


def _print_mag_grids_side_by_side(
    label_a: str,
    mags_a: np.ndarray,
    means_a: np.ndarray,
    label_b: str,
    mags_b: np.ndarray,
    means_b: np.ndarray,
    label_diff: str,
    mags_diff: np.ndarray,
    means_diff: np.ndarray,
):
    """Print three magnitude grids with arrows side by side."""
    rows = mags_a.shape[0]
    cols = mags_a.shape[1]
    row_width = cols * 10 + (cols - 1)  # col_width * cols + spaces
    click.echo(f"  {label_a:<{row_width}s}    {label_b:<{row_width}s}    {label_diff}")
    for r in range(rows):
        row_a = _format_mag_row(mags_a, means_a, r)
        row_b = _format_mag_row(mags_b, means_b, r)
        row_d = _format_diff_row(mags_diff, means_diff, r)
        click.echo(f"  {row_a}    {row_b}    {row_d}")


def _print_vector_grid(label: str, grid: np.ndarray):
    """Print a grid of (u, v) vectors."""
    rows = grid.shape[0]
    cols = grid.shape[1]
    click.echo(f"  {label}:")
    for r in range(rows):
        cells = []
        for c in range(cols):
            u, v = grid[r, c, 0], grid[r, c, 1]
            cells.append(f"({u:+.1f},{v:+.1f})")
        click.echo("    " + " ".join(f"{c:>14s}" for c in cells))


def _save_flow_images(
    output_dir: Path,
    base_name: str,
    from_number: int,
    to_local_number: int,
    local_u: np.ndarray,
    local_v: np.ndarray,
    to_stride_number: int | None,
    stride_u: np.ndarray | None,
    stride_v: np.ndarray | None,
) -> None:
    """Save flow color images (Middlebury convention) for a sample point."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Local flow: from → from+1
    local_color = _flow_to_color(local_u, local_v)
    _draw_flow_legend(local_color)
    local_path = output_dir / f"{base_name}_from_{from_number}_to_{to_local_number}.jpg"
    cv2.imwrite(str(local_path), local_color)

    # Stride flow: from → from+N
    if stride_u is not None and stride_v is not None:
        stride_color = _flow_to_color(stride_u, stride_v)
        _draw_flow_legend(stride_color)
        stride_path = (
            output_dir / f"{base_name}_from_{from_number}_to_{to_stride_number}.jpg"
        )
        cv2.imwrite(str(stride_path), stride_color)


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


def _print_sample_point(result: dict):
    """Print detailed output for a single sample point."""
    frame_num = result["frame_number"]
    stride = result["stride"]
    click.echo("")
    click.echo(f"Frame {frame_num}: {result['frame_name']}")
    click.echo(
        f"  Local: → {result['next_frame_name']}"
        f"  median magnitude: {result['local_median_magnitude']:.1f}"
    )

    if "stride_tile_mags" not in result:
        return

    ratio = result.get("actual_magnitude_ratio")
    ratio_str = f"{ratio:.2f}" if ratio is not None else "N/A"
    click.echo(
        f"  Stride: → {result['stride_frame_name']}"
        f"  median magnitude: {result['stride_median_magnitude']:.1f}"
        f"  (ratio: {ratio_str}, expected: {stride})"
    )

    click.echo("")
    _print_mag_grids_side_by_side(
        f"Local×{stride} tile magnitudes",
        result["local_tile_mags"],
        result["local_tile_means"],
        "Stride tile magnitudes",
        result["stride_tile_mags"],
        result["stride_tile_means"],
        "Difference (stride-local)",
        result["diff_tile_mags"],
        result["diff_tile_means"],
    )

    in_bounds = result.get("in_bounds_pct", 100.0)
    click.echo(f"  ({in_bounds:.0f}% of pixels in bounds)")
    click.echo("")
    _print_bar_grids_side_by_side(
        f"Local×{stride} (v\u2193 u\u2192)",
        result["local_hist"],
        "Stride (v\u2193 u\u2192)",
        result["stride_hist"],
    )


def _classify_ratio(normalized: float) -> str:
    """Classify a ratio/stride value into a human-readable description."""
    if normalized < 0.5:
        return "strong deceleration"
    elif normalized < 0.75:
        return "deceleration"
    elif normalized > 2.0:
        return "strong acceleration"
    elif normalized > 1.33:
        return "acceleration"
    return ""


def _print_summary(results: list[dict]):
    """Print a summary of likely discontinuities from analysis results."""
    # Collect frames with notable ratio deviations, skipping results
    # that were superseded by a retry with a shorter stride.
    flagged = []
    for r in results:
        if r.get("superseded"):
            continue
        ratio = r.get("actual_magnitude_ratio")
        stride = r.get("stride", 0)
        if ratio is None or stride < 2:
            continue
        normalized = ratio / stride
        if normalized < 0.75 or normalized > 1.0 / 0.75:
            flagged.append(
                {
                    "frame_number": r["frame_number"],
                    "frame_name": r["frame_name"],
                    "stride": stride,
                    "ratio": ratio,
                    "normalized": normalized,
                    "in_bounds_pct": r.get("in_bounds_pct", 100.0),
                    "local_mag": r.get("local_median_magnitude", 0),
                }
            )

    click.echo("")
    if not flagged:
        click.echo(
            "Summary: no discontinuities detected — motion is consistent "
            "across all sampled intervals."
        )
        return

    click.echo(f"Summary: {len(flagged)} discontinuity/discontinuities detected:")
    click.echo("")
    for f in flagged:
        desc = _classify_ratio(f["normalized"])
        click.echo(
            f"  Frame {f['frame_number']} ({f['frame_name']}): "
            f"ratio/stride={f['normalized']:.2f} "
            f"(stride {f['stride']}, local mag {f['local_mag']:.0f}) "
            f"— {desc}"
        )


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
