# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Text/image display helpers for discontinuity analysis output."""

from pathlib import Path

import click
import cv2
import numpy as np

from ._flow_display import _draw_flow_legend, _flow_to_color


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
