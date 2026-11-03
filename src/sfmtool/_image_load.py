# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared image decoding helpers.

Lives at the package root so that `feature_match/`, `motion/` and
`visualization/` — which each need the same grayscale decode for optical flow —
can reach it without importing across a sibling subpackage's private surface.
"""

from pathlib import Path

import numpy as np

from .fileio import read_image_rgb
from .sift import ImageToGray, gray_from_rgb

# The grey the optical flow kernels read: BT.601 luma rounded to 8 bits, the
# 8-bit grey image they take. It is made by sfmtool's one RGB-to-grey
# conversion, and it is within one level of OpenCV's ``COLOR_RGB2GRAY``, the
# conversion the flow presets were tuned on, for all but 0.12% of the 8-bit
# colours.
_FLOW_IMAGE_TO_GRAY = ImageToGray("0.299*R + 0.587*G + 0.114*B", quantize_bits=8)


def flow_gray(rgb: np.ndarray) -> np.ndarray:
    """An ``(H, W, 3)`` uint8 RGB image as the ``(H, W)`` uint8 grey image
    the optical flow kernels read: BT.601 luma rounded to 8 bits, through
    ``sfmtool.sift.gray_from_rgb``."""
    levels = gray_from_rgb(rgb, _FLOW_IMAGE_TO_GRAY) * np.float32(255)
    return np.rint(levels).astype(np.uint8)


def load_gray(path: Path) -> np.ndarray:
    """Load an image as the grayscale uint8 image of ``flow_gray``.

    Decoded by ``read_image_rgb``, the Rust decoder, with the EXIF orientation
    ignored.
    """
    return flow_gray(read_image_rgb(path))
