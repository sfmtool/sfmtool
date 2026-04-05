# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Interface definitions for xform transformations."""

from typing import Protocol

from .._sfmtool import SfmrReconstruction


class Transform(Protocol):
    """Protocol for transformations that can be applied to reconstructions."""

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        """Apply this transformation to a reconstruction."""
        ...

    def description(self) -> str:
        """Return a human-readable description of this transformation."""
        ...
