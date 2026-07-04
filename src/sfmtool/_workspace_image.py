# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a reconstruction's source images from its workspace.

Photometric code paths (patch-normal refinement, the ``compare --strips``
montage) reach back for the workspace's source images the same way the
SIFT-reading filters reach for ``.sift`` files: ``workspace_dir / image_name``.
A missing or unreadable image is a hard error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_workspace_image(workspace_dir: str | Path, image_name: str) -> np.ndarray:
    """Read one workspace image as a contiguous **RGB** array.

    The whole sfmtool-side image pipeline works in RGB (matching the SIFT
    extractor, which converts right after ``imread``), so this load boundary
    converts BGR→RGB once. The Rust patch/bitmap renderers sample source pixels
    in their native channel order, so feeding them RGB makes the stored
    ``patch_bitmaps_y_x_rgba`` genuinely RGB (as its name, the ``.sfmr`` spec,
    and the GUI atlas all require). Consumers that hand an array to cv2 for
    drawing or ``imwrite`` must convert RGB→BGR at that sink.

    ``image_name`` is the workspace-relative path as stored in
    ``recon.image_names``. Raises ``FileNotFoundError`` if the file is missing
    or cannot be decoded.
    """
    import cv2  # heavy module, only needed here

    path = Path(workspace_dir) / image_name
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Source image not found or unreadable: {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
