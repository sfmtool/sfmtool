# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter points by keypoint positional uncertainty (patch localizability)."""

import numpy as np

from .._sfmtool import SfmrReconstruction


class FilterByLocalizabilityTransform:
    """Remove 3D points whose predicted keypoint position uncertainty exceeds a
    threshold.

    The per-point ``σ_pos`` is measured in **patch-grid px** — the intrinsic,
    resolution-independent unit — as the noise-normalized weak-axis structure-tensor
    uncertainty of the point's cross-view consensus ``patch_bitmaps`` (see
    ``specs/core/patch-localizability.md``). It grades the *conditioning* of a
    keypoint's localization — corner vs. edge vs. flat — independently of whether
    the views agree, so it catches the aperture blind spot the reprojection /
    agreement gates miss. Grid px is chosen over source-image px because it
    transfers across datasets of different resolution (the source-px form folds in
    a per-point focal/depth scale that makes a fixed threshold cull wildly
    different fractions per dataset). Computed on demand from the recon's stored
    consensus + geometry; no source images are read.
    """

    def __init__(self, threshold: float, sigma_noise: float = 3.0):
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        if sigma_noise <= 0:
            raise ValueError(f"sigma_noise must be positive, got {sigma_noise}")
        self.threshold = threshold
        self.sigma_noise = sigma_noise

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        bitmaps = recon.patch_bitmaps
        if bitmaps is None:
            raise ValueError(
                "Filtering by keypoint uncertainty needs per-point patch bitmaps "
                "(a cross-view consensus to score), which this reconstruction has "
                "none of. Produce them with `sfm embed-patches` or "
                "`sfm xform --refine-keypoints bitmaps=true` first."
            )
        cloud = recon.patches
        if cloud is None:
            raise ValueError(
                "Filtering by keypoint uncertainty needs the per-point patch "
                "frames, which this reconstruction has none of (not an "
                "embedded_patches reconstruction)."
            )

        result = cloud.score_localizability(
            recon, bitmaps, sigma_noise=self.sigma_noise
        )
        sigma_pos = np.asarray(result["sigma_pos_grid"], dtype=float)

        # Drop points whose σ_pos (patch-grid px) exceeds τ. A point that could
        # not be scored (empty consensus) has σ_pos = NaN; ``NaN > τ`` is False,
        # so it is kept — the filter only removes points it has positive evidence
        # are poorly localized.
        points_to_keep_mask = ~(sigma_pos > self.threshold)

        if not np.any(points_to_keep_mask):
            raise ValueError(
                "No points remain after filtering by keypoint uncertainty "
                f"<= {self.threshold} grid px"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with keypoint uncertainty > "
            f"{self.threshold:.2f} grid px "
            f"({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Filter by keypoint uncertainty <= {self.threshold:.2f} grid px"
