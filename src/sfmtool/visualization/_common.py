# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the visualization modules."""

import colorsys
import random


def get_color_palette(n_colors: int) -> list[tuple[int, int, int]]:
    """Generate a cycling color palette with distinct colors randomized.

    Returns list of (B, G, R) tuples for use with cv2. ``n_colors <= 0``
    returns an empty list.
    """
    if n_colors <= 0:
        return []
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    random.Random(42).shuffle(colors)
    return colors
