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

    The image is decoded by ``sfmtool.fileio.read_image_rgb``, the decoder the
    Rust code and the viewer read photographs with, so the pixels here equal
    the ones the viewer's bench reads, and the EXIF orientation is ignored as
    it is there and in every SIFT extractor. The Rust patch/bitmap renderers
    sample source pixels in their native channel order, so feeding them RGB
    makes the stored ``patch_bitmaps_y_x_rgba`` genuinely RGB (as its name,
    the ``.sfmr`` spec, and the GUI atlas all require). Consumers that hand an
    array to cv2 for drawing or ``imwrite`` must convert RGB→BGR at that sink.

    ``image_name`` is the workspace-relative path as stored in
    ``recon.image_names``. Raises ``FileNotFoundError`` if the file is missing
    and ``OSError`` if it cannot be decoded.
    """
    from .fileio import read_image_rgb

    return read_image_rgb(Path(workspace_dir) / image_name)
