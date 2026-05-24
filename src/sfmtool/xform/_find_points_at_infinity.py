# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Transforms that discover or reclassify points at infinity.

Unlike the other xform operations, ``FindPointsAtInfinityTransform`` is
*additive*: it appends new points and tracks rather than transforming or
removing existing geometry, so the point count grows. See
specs/drafts/xform-find-points-at-infinity.md.
"""

from .._sfmtool import SfmrReconstruction


class FindPointsAtInfinityTransform:
    """Discover points at infinity (and near-infinite distant points).

    Clusters keypoint directions across all images, confirms clusters with
    SIFT descriptors, classifies each surviving track as a ``w = 0`` point or
    a finite distant point, and appends the new points and tracks. Reads the
    workspace ``.sift`` files, so they must still be present where the
    reconstruction was created.
    """

    def __init__(
        self,
        eps_deg: float,
        desc_thresh: float = 200.0,
        min_views: int = 2,
        max_features: int | None = None,
        ratio: float = 0.8,
        noise_floor_px: float = 1.0,
    ):
        if eps_deg <= 0:
            raise ValueError(f"eps_deg must be positive, got {eps_deg}")
        if min_views < 2:
            raise ValueError(f"min_views must be >= 2, got {min_views}")
        self.eps_deg = eps_deg
        self.desc_thresh = desc_thresh
        self.min_views = min_views
        self.max_features = max_features
        self.ratio = ratio
        self.noise_floor_px = noise_floor_px

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return recon.find_points_at_infinity(
            self.eps_deg,
            self.desc_thresh,
            self.ratio,
            self.min_views,
            self.max_features,
            self.noise_floor_px,
        )

    def description(self) -> str:
        return (
            f"Find points at infinity (eps={self.eps_deg}°, "
            f"desc_thresh={self.desc_thresh}, min_views={self.min_views}, "
            f"max_features={self.max_features})"
        )


class ClassifyPointsAtInfinityTransform:
    """Reclassify existing finite points whose depth is unconstrained.

    Only relabels already-triangulated points as ``w = 0``; it finds no new
    points and leaves the point count unchanged.
    """

    def __init__(self, noise_floor_px: float = 1.0):
        self.noise_floor_px = noise_floor_px

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return recon.classify_points_at_infinity(self.noise_floor_px)

    def description(self) -> str:
        return f"Classify points at infinity (noise_floor={self.noise_floor_px}px)"
