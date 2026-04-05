# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Feature matching module for registered images.

Provides automatic feature matching using sort-and-sweep algorithms,
with automatic selection between rectified and polar matching based
on camera geometry.

Supports geometric filtering using affine shape consistency for improved
accuracy and performance.
"""

from ._core import match_image_pair, match_registered_images
from ._geometric_filter import GeometricFilterConfig

__all__ = ["match_image_pair", "match_registered_images", "GeometricFilterConfig"]
