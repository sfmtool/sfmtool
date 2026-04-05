# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Rotation transformation."""

import numpy as np

from .._sfmtool import Se3Transform, SfmrReconstruction


class RotateTransform:
    """Rotate reconstruction around an axis."""

    def __init__(self, axis: np.ndarray, angle_rad: float):
        self.transform = Se3Transform.from_axis_angle(axis, angle_rad)
        axis = np.asarray(axis, dtype=float)
        self.axis = axis / np.linalg.norm(axis)
        self.angle_rad = angle_rad

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return self.transform @ recon

    def description(self) -> str:
        angle_deg = np.degrees(self.angle_rad)
        return f"Rotate {angle_deg:.2f}\u00b0 around axis ({self.axis[0]:.3f}, {self.axis[1]:.3f}, {self.axis[2]:.3f})"
