# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared source-image loading for the photometric ``sfm xform`` operations.

The patch operations (``--refine-normals``, ``--refine-keypoints``,
``--localize-keypoints``) all read the workspace's source images to score
cross-view photometric consensus. They each need one full-resolution image per
reconstruction image, loaded up front.
"""

import numpy as np

from .._sfmtool.reconstruction import SfmrReconstruction


def load_workspace_images(recon: SfmrReconstruction) -> list[np.ndarray]:
    """Load every source image, parallel to the reconstruction's images.

    Resolves ``workspace_dir / image_name`` exactly as the SIFT-reading filters
    resolve their ``.sift`` paths. A missing image (or unresolvable workspace)
    is a hard error. The patch bindings require one full-resolution image per
    reconstruction image (matching its camera resolution), so all are loaded up
    front.
    """
    from sfmtool._workspace_image import read_workspace_image

    return [
        read_workspace_image(recon.workspace_dir, name) for name in recon.image_names
    ]
