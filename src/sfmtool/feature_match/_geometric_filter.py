# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration for geometric filtering in feature matching.

The geometric filtering algorithms are implemented in Rust
(sfmtool-core/src/feature_match/geometric_filter.rs) and exposed
via Python bindings. This module provides the Python-side configuration
dataclass used by the matching pipeline.
"""

from dataclasses import dataclass


@dataclass
class GeometricFilterConfig:
    """Configuration for two-stage geometric filtering.

    Attributes:
        enable_geometric_filtering: Whether to apply geometric filtering

        # Stage 1: Orientation check (always performed)
        max_angle_difference: Maximum allowed angle difference in degrees (default: 15.0)

        # Stage 2: Size check (conditional on triangulation reliability)
        min_triangulation_angle: Minimum ray angle in degrees for reliable triangulation (default: 5.0)
        geometric_size_ratio_min: Minimum allowed size ratio (default: 0.8)
        geometric_size_ratio_max: Maximum allowed size ratio (default: 1.25)
    """

    enable_geometric_filtering: bool = True

    # Stage 1: Orientation check (always)
    max_angle_difference: float = 15.0

    # Stage 2: Size check (conditional)
    min_triangulation_angle: float = 5.0
    geometric_size_ratio_min: float = 0.8
    geometric_size_ratio_max: float = 1.25

    def is_size_ratio_valid(self, ratio: float) -> bool:
        """Check if size ratio is within thresholds."""
        return self.geometric_size_ratio_min <= ratio <= self.geometric_size_ratio_max

    def is_angle_diff_valid(self, angle_diff_degrees: float) -> bool:
        """Check if angle difference is within threshold."""
        return angle_diff_degrees <= self.max_angle_difference
