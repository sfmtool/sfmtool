# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter points by the ZNCC self-similarity radius of their stored patch bitmap."""

import numpy as np

from .._sfmtool.patches import (
    DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS,
    zncc_self_similarity_parts_stack,
)
from .._sfmtool.reconstruction import SfmrReconstruction

#: The default bar on a point's stored bitmap: the same 2.5 patch-grid px as
#: the keypoint localizer's and cluster refinement's member gates and the bench.
DEFAULT_MAX_ZNCC_SELF_SIMILARITY_RADIUS = DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS


def points_passing_zncc_self_similarity_radius(
    bitmaps: np.ndarray, bar: float
) -> tuple[np.ndarray, np.ndarray]:
    """Which points' stored bitmaps pass a bar on the ZNCC self-similarity radius.

    Each ``(R, R, C)`` bitmap of the ``(N, R, R, C)`` stack is read the overlap
    way (``zncc_self_similarity_parts_stack``; see
    ``specs/core/patch/zncc-self-similarity-radius.md``), as the bench and the
    member gates read their tiles: at each shift only the samples inside the
    bitmap on both sides, and whose alpha is above 0 on both sides, are
    correlated. A point passes when its whole bitmap's radius is at or below
    ``bar``; a ``NaN`` radius fails. A bitmap with no sample carrying data (a
    point with no bitmap, whose row is zero) has no reading and passes, as
    the bench's painting passes a row with no reading. A ``bar`` of ``0`` or
    less turns the cull off, and since the radius reads at most 3, a bar of 3
    or more turns nothing out.

    Returns ``(passes, radius)``, both of length ``N``: the boolean mask and the
    whole bitmap's radius in patch-grid px (``NaN`` where there is no reading).
    """
    result = zncc_self_similarity_parts_stack(bitmaps)
    radius = np.asarray(result["radius"], dtype=float)
    if not bar > 0:
        return np.ones(len(radius), dtype=bool), radius
    covered = np.asarray(result["covered"], dtype=bool)
    return ~covered | (radius <= bar), radius


class FilterByZnccSelfSimilarityRadiusTransform:
    """Remove 3D points whose stored bitmap can slide over itself too far.

    The ZNCC self-similarity radius of a point's stored ``patch_bitmaps`` row is
    how far, in patch-grid px up to 3, the bitmap can shift over itself and
    still match itself as well as a true match between two views would. A
    corner or a busy texture reads under 1; a straight edge, which slides along
    itself, and a flat patch read 3. The filter removes points whose radius is
    over ``threshold``, with the pass rule of
    :func:`points_passing_zncc_self_similarity_radius`: ``0`` turns it off, and
    a point with no bitmap is kept. It reads the bitmaps the reconstruction
    stores; no source images are read.
    """

    def __init__(self, threshold: float = DEFAULT_MAX_ZNCC_SELF_SIMILARITY_RADIUS):
        if not threshold >= 0:
            raise ValueError(f"Threshold must be 0 or more, got {threshold}")
        self.threshold = threshold

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        bitmaps = recon.patch_bitmaps
        if bitmaps is None:
            raise ValueError(
                "Filtering by the ZNCC self-similarity radius needs per-point patch "
                "bitmaps to score, which this "
                "reconstruction has none of. Produce them with `sfm embed-patches` "
                "or `sfm xform --add-patch-bitmaps` first."
            )

        points_to_keep_mask, _ = points_passing_zncc_self_similarity_radius(
            bitmaps, self.threshold
        )

        if not np.any(points_to_keep_mask):
            raise ValueError(
                "No points remain after filtering by ZNCC self-similarity radius "
                f"<= {self.threshold} grid px"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with ZNCC self-similarity radius > "
            f"{self.threshold:.2f} grid px "
            f"({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Filter by ZNCC self-similarity radius <= {self.threshold:.2f} grid px"
