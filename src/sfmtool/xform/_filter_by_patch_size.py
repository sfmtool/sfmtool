# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter points by world-space patch size (coarse-patch cull)."""

import numpy as np

from .._sfmtool.reconstruction import SfmrReconstruction


class FilterByPatchSizeTransform:
    """Remove 3D points whose world-space patch size exceeds a multiple of the
    per-reconstruction median.

    Each patch's characteristic world size is the geometric mean of its two
    world half-extents, ``sqrt(|half_extent[0]| * |half_extent[1]|)``. Under the
    ``feature_size`` extent policy this tracks the keypoint's SIFT scale, so the
    largest patches are the coarsest features. The keep threshold is
    data-derived: ``size <= multiplier * median(size)`` over all patches, so it
    adapts to each reconstruction's own scale rather than fixing an absolute
    world size. Requires an ``embedded_patches`` reconstruction — the per-point
    patch frames carry the world half-extents this scores.
    """

    def __init__(self, multiplier: float):
        if multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {multiplier}")
        self.multiplier = multiplier

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        cloud = recon.patches
        if cloud is None:
            raise ValueError(
                "Filtering by patch size needs the per-point patch frames, which "
                "this reconstruction has none of (not an embedded_patches "
                "reconstruction). Convert first with `sfm embed-patches` or "
                "`sfm xform --to-embedded-patches`."
            )

        n = len(cloud)
        if n == 0:
            raise ValueError("No patches to filter by patch size")

        # Characteristic world size per patch: the geometric mean of its two
        # world half-extents.
        sizes = np.array(
            [
                np.sqrt(abs(cloud[i].half_extent[0]) * abs(cloud[i].half_extent[1]))
                for i in range(n)
            ]
        )

        threshold = self.multiplier * float(np.median(sizes))
        points_to_keep_mask = np.ascontiguousarray(sizes <= threshold, dtype=bool)

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after filtering by patch size "
                f"<= {self.multiplier}x median"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with world-space patch size > "
            f"{self.multiplier:.2f}x median ({threshold:.6f}) "
            f"({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Filter by patch size <= {self.multiplier:.2f}x median"
