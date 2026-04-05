# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter points by reprojection error."""

import numpy as np

from .._sfmtool import SfmrReconstruction


class FilterByReprojectionErrorTransform:
    """Remove 3D points with reprojection error > threshold."""

    def __init__(self, threshold: float):
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        self.threshold = threshold

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        points_to_keep_mask = recon.errors <= self.threshold

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after filtering by reprojection error <= {self.threshold} pixels"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with reprojection error > {self.threshold:.2f} pixels ({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Filter by reprojection error <= {self.threshold:.2f} pixels"
