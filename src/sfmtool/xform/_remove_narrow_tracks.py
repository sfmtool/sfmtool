# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove narrow tracks filter."""

import numpy as np

from .._sfmtool import SfmrReconstruction


class RemoveNarrowTracksFilter:
    """Remove 3D points with viewing angle span less than threshold."""

    def __init__(self, min_angle_rad: float):
        if min_angle_rad <= 0:
            raise ValueError(f"Minimum angle must be positive, got {min_angle_rad}")
        self.min_angle_rad = min_angle_rad

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        print(
            f"  Computing viewing angles for {recon.point_count} points across {recon.image_count} images..."
        )

        from .._sfmtool import compute_narrow_track_mask

        points_to_keep_mask = compute_narrow_track_mask(
            recon.quaternions_wxyz,
            recon.translations,
            recon.positions,
            recon.track_point_ids,
            recon.track_image_indexes,
            self.min_angle_rad,
        )

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after removing tracks with viewing angle < {np.degrees(self.min_angle_rad):.2f}\u00b0"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with viewing angle < {np.degrees(self.min_angle_rad):.2f}\u00b0 ({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return (
            f"Remove tracks with viewing angle < {np.degrees(self.min_angle_rad):.2f}\u00b0"
        )
