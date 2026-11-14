# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Heatmap rendering utilities for visualization.

Renders quality metrics as colored overlays on source images.
"""

from pathlib import Path

import cv2
import numpy as np

from .._sfmtool.fileio import read_image_rgb, write_image_rgb
from ._colormap import value_to_color


def render_heatmap_overlay(
    image_path: Path,
    feature_positions: np.ndarray,
    feature_values: np.ndarray,
    output_path: Path,
    *,
    metric_name: str = "metric",
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    radius: int = 5,
    alpha: float = 0.7,
    show_colorbar: bool = True,
) -> None:
    """Render a heatmap overlay on an image.

    Args:
        image_path: Path to source image
        feature_positions: (N, 2) array of feature (x, y) positions
        feature_values: (N,) array of values to visualize
        output_path: Path to save output image
        metric_name: Name of metric for colorbar label
        colormap: Colormap name to use
        vmin: Minimum value for color scale (default: data min)
        vmax: Maximum value for color scale (default: data max)
        radius: Radius of feature circles in pixels
        alpha: Opacity of overlay (0.0-1.0)
        show_colorbar: Whether to add a colorbar legend
    """
    # Load image
    image = read_image_rgb(image_path)
    height, width = image.shape[:2]

    # Create overlay
    overlay = image.copy()

    # Compute value range
    valid_values = feature_values[~np.isnan(feature_values)]
    if len(valid_values) == 0:
        # No valid values, save original image
        write_image_rgb(output_path, image)
        return

    if vmin is None:
        vmin = float(np.min(valid_values))
    if vmax is None:
        vmax = float(np.max(valid_values))

    # Draw circles for each feature
    for i in range(len(feature_positions)):
        x, y = feature_positions[i]
        value = feature_values[i]

        # Skip if position is outside image
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        # Get color
        if np.isnan(value):
            color = (128, 128, 128)  # Gray for NaN
        else:
            color = value_to_color(value, vmin, vmax, colormap)

        # Draw filled circle
        cv2.circle(overlay, (int(x), int(y)), radius, color, -1)

    # Blend overlay with original
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Add colorbar if requested
    if show_colorbar:
        result = _add_colorbar(result, metric_name, colormap, vmin, vmax)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image_rgb(output_path, result)


def _add_colorbar(
    image: np.ndarray,
    label: str,
    colormap: str,
    vmin: float,
    vmax: float,
    bar_width: int = 20,
    margin: int = 10,
) -> np.ndarray:
    """Add a colorbar legend to the right side of an image.

    Args:
        image: RGB image array
        label: Label for the colorbar
        colormap: Colormap name
        vmin: Minimum value
        vmax: Maximum value
        bar_width: Width of colorbar in pixels
        margin: Margin around colorbar

    Returns:
        Image with colorbar added
    """
    height, width = image.shape[:2]

    # Create colorbar strip
    bar_height = height - 2 * margin
    colorbar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

    for y in range(bar_height):
        # Map y position to value (top = max, bottom = min)
        t = 1.0 - (y / (bar_height - 1))
        value = vmin + t * (vmax - vmin)
        color = value_to_color(value, vmin, vmax, colormap)
        colorbar[y, :] = color

    # Create extended canvas
    total_width = width + bar_width + margin * 3 + 60  # Extra space for labels
    canvas = np.zeros((height, total_width, 3), dtype=np.uint8)
    canvas[:, :, :] = 255  # White background for label area

    # Copy original image
    canvas[:, :width, :] = image

    # Copy colorbar
    canvas[
        margin : margin + bar_height, width + margin : width + margin + bar_width, :
    ] = colorbar

    # Draw border around colorbar
    cv2.rectangle(
        canvas,
        (width + margin - 1, margin - 1),
        (width + margin + bar_width, margin + bar_height),
        (0, 0, 0),
        1,
    )

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (0, 0, 0)
    thickness = 1

    # Max value at top
    cv2.putText(
        canvas,
        f"{vmax:.2g}",
        (width + margin + bar_width + 5, margin + 12),
        font,
        font_scale,
        font_color,
        thickness,
    )

    # Min value at bottom
    cv2.putText(
        canvas,
        f"{vmin:.2g}",
        (width + margin + bar_width + 5, margin + bar_height - 2),
        font,
        font_scale,
        font_color,
        thickness,
    )

    # Label in middle
    mid_y = margin + bar_height // 2
    cv2.putText(
        canvas,
        label,
        (width + margin + bar_width + 5, mid_y),
        font,
        font_scale,
        font_color,
        thickness,
    )

    return canvas
