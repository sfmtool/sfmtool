# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Similarity transformation (SE3 with scale)."""

from .._sfmtool import Se3Transform, SfmrReconstruction


class SimilarityTransform:
    """Apply a full SE3 similarity transform (rotation, translation, scale)."""

    def __init__(self, transform: Se3Transform):
        self.transform = transform

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return self.transform @ recon

    def description(self) -> str:
        t = self.transform.translation
        scale = self.transform.scale
        rot = self.transform.rotation
        return (
            f"Similarity transform: scale={scale:.3f}, "
            f"translation=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}), "
            f"rotation=({rot.w:.3f}, {rot.x:.3f}, {rot.y:.3f}, {rot.z:.3f})"
        )
