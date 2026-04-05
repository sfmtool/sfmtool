# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove isolated points filter."""

import numpy as np

from .._sfmtool import SfmrReconstruction


class RemoveIsolatedPointsFilter:
    """Remove 3D points whose nearest neighbor distance exceeds a threshold."""

    def __init__(self, factor: float, value_spec: str):
        if factor <= 0:
            raise ValueError(f"Factor must be positive, got {factor}")
        self.factor = factor
        self.value_spec = value_spec

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if recon.point_count < 2:
            raise ValueError("Need at least 2 points to compute nearest neighbors")

        print(
            f"  Computing nearest neighbor distances for {recon.point_count} points..."
        )

        from .._sfmtool import KdTree3d

        nn_distances = KdTree3d(recon.positions).nearest_neighbor_distances()

        if self.value_spec == "median":
            reference_value = np.median(nn_distances)
            ref_desc = "median"
        elif self.value_spec.endswith("percent") or self.value_spec.endswith(
            "percentile"
        ):
            if self.value_spec.endswith("percentile"):
                percentile_str = self.value_spec[: -len("percentile")]
            else:
                percentile_str = self.value_spec[: -len("percent")]

            try:
                percentile = float(percentile_str)
            except ValueError:
                raise ValueError(
                    f"Invalid percentile specification: '{self.value_spec}'"
                )

            if not 0 <= percentile <= 100:
                raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

            reference_value = np.percentile(nn_distances, percentile)
            ref_desc = f"{percentile}th percentile"
        else:
            raise ValueError(
                f"Invalid value_spec: '{self.value_spec}'. "
                f"Expected 'median', '<N>percent', or '<N>percentile'"
            )

        threshold = self.factor * reference_value

        print(f"    Reference value ({ref_desc}): {reference_value:.6f}")
        print(f"    Threshold ({self.factor}\u00d7 reference): {threshold:.6f}")

        points_to_keep_mask = nn_distances <= threshold

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after removing isolated points (threshold: {threshold:.6f})"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} isolated points ({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Remove isolated points (NN > {self.factor}\u00d7 {self.value_spec})"
