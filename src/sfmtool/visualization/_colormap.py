# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Colormap utilities for visualization.

Provides functions to map scalar values to colors for heatmap visualization.
"""

import numpy as np

# Predefined colormaps as (position, r, g, b) tuples
# Position is 0.0-1.0, colors are 0-255
COLORMAPS = {
    # Viridis-like: purple -> blue -> teal -> green -> yellow
    "viridis": [
        (0.0, 68, 1, 84),
        (0.25, 59, 82, 139),
        (0.5, 33, 145, 140),
        (0.75, 94, 201, 98),
        (1.0, 253, 231, 37),
    ],
    # Plasma-like: purple -> pink -> orange -> yellow
    "plasma": [
        (0.0, 13, 8, 135),
        (0.25, 126, 3, 168),
        (0.5, 204, 71, 120),
        (0.75, 248, 149, 64),
        (1.0, 240, 249, 33),
    ],
    # Jet: blue -> cyan -> green -> yellow -> red
    "jet": [
        (0.0, 0, 0, 128),
        (0.11, 0, 0, 255),
        (0.35, 0, 255, 255),
        (0.5, 0, 255, 0),
        (0.65, 255, 255, 0),
        (0.89, 255, 0, 0),
        (1.0, 128, 0, 0),
    ],
    # Cool to warm: blue -> white -> red
    "coolwarm": [
        (0.0, 59, 76, 192),
        (0.5, 221, 221, 221),
        (1.0, 180, 4, 38),
    ],
    # Error map: green (good) -> yellow -> red (bad)
    "error": [
        (0.0, 0, 200, 0),
        (0.5, 255, 255, 0),
        (1.0, 255, 0, 0),
    ],
    # Track length: blue (few) -> green -> yellow -> red (many)
    "tracks": [
        (0.0, 0, 0, 255),
        (0.33, 0, 200, 200),
        (0.66, 200, 200, 0),
        (1.0, 255, 50, 50),
    ],
}


def value_to_color(
    value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colormap: str = "viridis",
) -> tuple[int, int, int]:
    """Map a scalar value to an RGB color.

    Args:
        value: The scalar value to map
        vmin: Minimum value (maps to first color)
        vmax: Maximum value (maps to last color)
        colormap: Name of colormap to use

    Returns:
        Tuple of (r, g, b) integers in range [0, 255]
    """
    if colormap not in COLORMAPS:
        raise ValueError(
            f"Unknown colormap: {colormap}. Available: {list(COLORMAPS.keys())}"
        )

    cmap = COLORMAPS[colormap]

    # Normalize to [0, 1]
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))

    # Find surrounding control points
    for i in range(len(cmap) - 1):
        t0, r0, g0, b0 = cmap[i]
        t1, r1, g1, b1 = cmap[i + 1]

        if t0 <= t <= t1:
            # Interpolate within this segment
            if t1 == t0:
                frac = 0.0
            else:
                frac = (t - t0) / (t1 - t0)

            r = int(r0 + frac * (r1 - r0))
            g = int(g0 + frac * (g1 - g0))
            b = int(b0 + frac * (b1 - b0))
            return (r, g, b)

    # Should not reach here, but return last color
    return (cmap[-1][1], cmap[-1][2], cmap[-1][3])


def apply_colormap(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: str = "viridis",
) -> np.ndarray:
    """Apply colormap to an array of values.

    Args:
        values: Array of scalar values (any shape)
        vmin: Minimum value (default: values.min())
        vmax: Maximum value (default: values.max())
        colormap: Name of colormap to use

    Returns:
        Array of shape (*values.shape, 3) with RGB values in [0, 255]
    """
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    # Flatten for processing
    flat = values.flatten()
    colors = np.zeros((len(flat), 3), dtype=np.uint8)

    for i, v in enumerate(flat):
        if np.isnan(v):
            colors[i] = (128, 128, 128)  # Gray for NaN
        else:
            colors[i] = value_to_color(v, vmin, vmax, colormap)

    return colors.reshape((*values.shape, 3))
