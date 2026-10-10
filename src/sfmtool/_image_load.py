# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared image decoding helpers.

Lives at the package root so that both `feature_match/` and `motion/` — which
each need the same grayscale decode for optical flow — can reach it without
importing across a sibling subpackage's private surface.
"""

from pathlib import Path

import cv2
import numpy as np

from .fileio import read_image_rgb


def load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8.

    Decoded by ``read_image_rgb``, the Rust decoder, with the EXIF orientation
    ignored; the grey conversion is OpenCV's ``COLOR_RGB2GRAY``.
    """
    return cv2.cvtColor(read_image_rgb(path), cv2.COLOR_RGB2GRAY)
