# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Interface definitions for xform transformations."""

from typing import Protocol

from .._sfmtool.reconstruction import SfmrReconstruction


class Transform(Protocol):
    """Protocol for transformations that can be applied to reconstructions.

    A step that reads photographs also sets ``reads_photographs = True`` and
    takes ``apply(recon, cache=None)``: ``apply_transforms`` passes it the
    run's ``PhotographCache``, which it reads through with
    ``ImagePyramidSet.read(recon, cache=cache)``.
    """

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        """Apply this transformation to a reconstruction."""
        ...

    def description(self) -> str:
        """Return a human-readable description of this transformation."""
        ...
