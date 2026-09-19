# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Align to another reconstruction transformation."""

from pathlib import Path

from .._sfmtool.reconstruction import SfmrReconstruction
from ._align_points_base import AlignToPointsTransform


class AlignToTransform(AlignToPointsTransform):
    """Align reconstruction to match another reconstruction using 3D point correspondences."""

    def __init__(self, reference_path: Path):
        self.reference_path = Path(reference_path).absolute()
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {self.reference_path}")
        if self.reference_path.suffix.lower() != ".sfmr":
            raise ValueError(
                f"Reference must be a .sfmr file, got: {self.reference_path}"
            )
        self._reference_recon = SfmrReconstruction.load(self.reference_path)

    def _target_recon(self) -> SfmrReconstruction:
        return self._reference_recon

    def _target_label(self) -> str:
        return self.reference_path.name

    def description(self) -> str:
        return f"Align to {self.reference_path.name}"
