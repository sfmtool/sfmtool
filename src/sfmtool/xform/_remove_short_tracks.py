# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove short tracks filter."""

import numpy as np

from .._sfmtool import SfmrReconstruction


class RemoveShortTracksFilter:
    """Remove 3D points with track length <= size."""

    def __init__(self, max_size: int):
        if max_size < 2:
            raise ValueError(f"Track size must be >= 2, got {max_size}")
        self.max_size = max_size

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        points_to_keep_mask = recon.observation_counts > self.max_size

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after removing tracks with length <= {self.max_size}"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with track length <= {self.max_size} ({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Remove tracks with length <= {self.max_size}"
