# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Transforms that discover or reclassify points at infinity.

Unlike the other xform operations, ``FindPointsAtInfinityTransform`` is
*additive*: it appends new points and tracks rather than transforming or
removing existing geometry, so the point count grows. See
specs/cli/reconstruction/xform/find-points-at-infinity.md.
"""

import numpy as np

from .._sfmtool.reconstruction import SfmrReconstruction


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
    """Decide every existing point with the point-or-bearing test.

    A finite point whose rays ask for no depth is demoted to ``w = 0``, and a
    point at infinity whose rays ask for one is promoted to the fitted point.
    It finds no new points and leaves the point count unchanged. ``sigma_px``
    overrides the measured reprojection noise the rays are weighted by. See
    specs/core/reconstruction/batch-triangulation-api.md, "Consumers".
    """

    def __init__(self, sigma_px: float | None = None):
        if sigma_px is not None and not (np.isfinite(sigma_px) and sigma_px > 0):
            raise ValueError(f"sigma_px must be finite and positive, got {sigma_px}")
        self.sigma_px = sigma_px

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        result, summary = recon.classify_points_at_infinity(self.sigma_px)
        for line in classify_summary_lines(summary, recon):
            print(f"    {line}")
        return result

    def description(self) -> str:
        if self.sigma_px is None:
            return "Classify points at infinity (measured noise)"
        return f"Classify points at infinity (sigma_px={self.sigma_px}px)"


def classify_summary_lines(summary: dict, recon: SfmrReconstruction) -> list[str]:
    """What ``classify_points_at_infinity`` did on ``recon``, one statement per line."""
    from ..analyze.point_or_bearing import no_noise_reason

    sigma = summary["sigma_px"]
    if sigma is None:
        reason = no_noise_reason(recon)
        return [f"{reason[0].upper()}{reason[1:]}; the points are left as they are"]
    noise = summary["noise"]
    if noise is None:
        lines = [f"Noise level: {sigma:.4g} px (given)"]
    else:
        lines = [
            f"Noise level: {sigma:.4g} px, measured over "
            f"{noise['observation_count']:,} observations, "
            f"{noise['outlier_count']:,} excluded as outliers"
        ]
    lines.append(
        f"Promoted to finite: {summary['promoted']:,}; "
        f"demoted to infinity: {summary['demoted']:,}; "
        f"kept: {summary['kept']:,}"
    )
    if summary["refitted"]:
        lines.append(
            "Moved off a position behind or on top of a camera: "
            f"{summary['refitted']:,}"
        )
    if summary["bearing_behind_camera"]:
        lines.append(
            f"Left finite, bearing behind a camera: {summary['bearing_behind_camera']:,}"
        )
    if summary["no_usable_point"]:
        lines.append(
            f"Left at infinity, no usable fitted point: {summary['no_usable_point']:,}"
        )
    if summary["left_unusable"]:
        lines.append(
            "Left finite behind or on top of a camera, no usable point or bearing: "
            f"{summary['left_unusable']:,}"
        )
    if summary["unscored"]:
        lines.append(
            f"Not scored (fewer than two usable rays): {summary['unscored']:,}"
        )
    return lines
