# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Heatmap rendering utilities for visualization.

Renders quality metrics as colored overlays on source images.
"""

from pathlib import Path

import cv2
import numpy as np

from .._sfmtool import RotQuaternion
from ._colormap import value_to_color


def compute_triangulation_angles(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    translations: np.ndarray,
    track_image_indexes: np.ndarray,
    track_point_ids: np.ndarray,
) -> np.ndarray:
    """Compute maximum triangulation angle for each 3D point.

    The triangulation angle is the maximum angle between any two rays
    observing a point. Larger angles indicate more reliable triangulation.

    Args:
        positions: (M, 3) array of 3D point positions
        quaternions_wxyz: (N, 4) array of camera quaternions
        translations: (N, 3) array of camera translations
        track_image_indexes: Array of image indices for track observations
        track_point_ids: Array of 3D point indices for track observations

    Returns:
        (M,) array of maximum triangulation angles in degrees for each point
    """
    num_points = len(positions)
    num_images = len(translations)

    # Compute camera centers (positions in world coordinates)
    # Camera center = -R^T @ t
    camera_centers = np.zeros((num_images, 3), dtype=np.float64)
    for img_idx in range(num_images):
        q = RotQuaternion.from_wxyz_array(quaternions_wxyz[img_idx])
        camera_centers[img_idx] = q.camera_center(translations[img_idx])

    # Build list of observing cameras for each point
    point_observers = [[] for _ in range(num_points)]
    for obs_idx in range(len(track_image_indexes)):
        point_idx = track_point_ids[obs_idx]
        img_idx = track_image_indexes[obs_idx]
        point_observers[point_idx].append(img_idx)

    # Compute max angle for each point
    max_angles = np.zeros(num_points, dtype=np.float64)

    for point_idx in range(num_points):
        observers = point_observers[point_idx]
        if len(observers) < 2:
            max_angles[point_idx] = 0.0
            continue

        point_pos = positions[point_idx]
        max_angle = 0.0

        # Check all pairs of observing cameras
        for i in range(len(observers)):
            for j in range(i + 1, len(observers)):
                cam_i = camera_centers[observers[i]]
                cam_j = camera_centers[observers[j]]

                # Compute rays from cameras to point
                ray_i = point_pos - cam_i
                ray_j = point_pos - cam_j

                # Normalize rays
                norm_i = np.linalg.norm(ray_i)
                norm_j = np.linalg.norm(ray_j)

                if norm_i < 1e-10 or norm_j < 1e-10:
                    continue

                ray_i = ray_i / norm_i
                ray_j = ray_j / norm_j

                # Compute angle between rays
                cos_angle = np.clip(np.dot(ray_i, ray_j), -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180.0 / np.pi

                max_angle = max(max_angle, angle)

        max_angles[point_idx] = max_angle

    return max_angles


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
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert BGR to RGB for processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Create overlay
    overlay = image.copy()

    # Compute value range
    valid_values = feature_values[~np.isnan(feature_values)]
    if len(valid_values) == 0:
        # No valid values, save original image
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
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
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


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
