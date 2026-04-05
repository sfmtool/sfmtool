# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale transformation."""

from .._sfmtool import Se3Transform, SfmrReconstruction


class ScaleTransform:
    """Scale reconstruction by a uniform factor."""

    def __init__(self, scale: float):
        if scale <= 0:
            raise ValueError(f"Scale factor must be positive, got {scale}")
        self.scale = scale
        self.transform = Se3Transform(scale=scale)

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return self.transform @ recon

    def description(self) -> str:
        return f"Scale by {self.scale:.3f}"
