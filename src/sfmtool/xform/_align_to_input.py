# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Align to input reconstruction transformation."""

from .._sfmtool.reconstruction import SfmrReconstruction
from ._align_points_base import AlignToPointsTransform


class AlignToInputTransform(AlignToPointsTransform):
    """Align reconstruction back to original input reconstruction."""

    _original_input: SfmrReconstruction | None = None

    def __init__(self):
        pass

    @classmethod
    def set_original_input(cls, recon: SfmrReconstruction):
        cls._original_input = recon.clone_with_changes()

    def _target_recon(self) -> SfmrReconstruction:
        if self._original_input is None:
            raise RuntimeError(
                "Original input reconstruction not set. "
                "This transform must be used within apply_transforms()."
            )
        return self._original_input

    def _target_label(self) -> str:
        return "original input reconstruction"

    def description(self) -> str:
        return "Align to original input"
