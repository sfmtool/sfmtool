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


def load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
