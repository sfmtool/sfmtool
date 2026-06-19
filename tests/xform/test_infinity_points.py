# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that xform point filters and bundle adjustment handle points at infinity.

Filters whose score is undefined for a direction (reprojection error,
triangulation angle, neighbour distance) pass points at infinity through
untouched; filters scoring track length or feature size score them normally.
Bundle adjustment materialises them, refines, then reclassifies. See
specs/formats/sfmr-file-format.md.
"""

import numpy as np

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.xform import (
    BundleAdjustTransform,
    FilterByReprojectionErrorTransform,
    RemoveIsolatedPointsFilter,
    RemoveNarrowTracksFilter,
    RemoveShortTracksFilter,
)


def _inject_infinity(recon: SfmrReconstruction, indices) -> SfmrReconstruction:
    """Return a copy with the points at ``indices`` turned into points at infinity."""
    positions = recon.positions_xyzw.copy()
    for i in indices:
        positions[i] = [0.0, 0.0, 1.0, 0.0]
    return recon.clone_with_changes(positions=positions)


def test_reprojection_filter_keeps_high_error_infinity_points(
    seoul_bull_sfmr_only,
):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    indices = list(range(5))

    # The injected points are at infinity AND given a huge reprojection error:
    # a finite point with that error would be filtered out.
    positions = recon.positions_xyzw.copy()
    errors = recon.errors.copy()
    for i in indices:
        positions[i] = [0.0, 0.0, 1.0, 0.0]
        errors[i] = 1000.0
    recon = recon.clone_with_changes(positions=positions, errors=errors)
    assert recon.point_is_at_infinity[indices].all()

    result = FilterByReprojectionErrorTransform(threshold=2.0).apply(recon)

    # Every infinity point survived despite failing the error criterion.
    assert int(result.point_is_at_infinity.sum()) == len(indices)
    # Finite points were still filtered.
    assert result.point_count < recon.point_count


def test_short_tracks_filter_scores_infinity_points(
    seoul_bull_sfmr_only,
):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    # Turn the shortest-track points into points at infinity.
    short_idx = np.flatnonzero(recon.observation_counts <= 3)[:5]
    assert len(short_idx) >= 1
    recon = _inject_infinity(recon, short_idx)
    assert int(recon.point_is_at_infinity.sum()) == len(short_idx)

    result = RemoveShortTracksFilter(max_size=3).apply(recon)

    # Track length is well-defined for a point at infinity, so short-track
    # infinity points are filtered out like any other short track.
    assert int(result.point_is_at_infinity.sum()) == 0


def test_narrow_tracks_filter_keeps_infinity_points(
    seoul_bull_sfmr_only,
):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    recon = _inject_infinity(recon, range(5))

    result = RemoveNarrowTracksFilter(min_angle_rad=np.radians(2.0)).apply(recon)

    assert int(result.point_is_at_infinity.sum()) == 5


def test_isolated_filter_keeps_infinity_points(
    seoul_bull_sfmr_only,
):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    recon = _inject_infinity(recon, range(5))

    result = RemoveIsolatedPointsFilter(factor=2.0, value_spec="median").apply(recon)

    assert int(result.point_is_at_infinity.sum()) == 5


def test_bundle_adjust_handles_infinity_points(
    seoul_bull_workspace,
):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    recon = _inject_infinity(recon, range(5))
    assert int(recon.point_is_at_infinity.sum()) == 5

    # Bundle adjustment materialises the infinity points, refines, then
    # reclassifies — it must not crash and must preserve the point count.
    result = BundleAdjustTransform().apply(recon)

    assert result.point_count == recon.point_count
    assert np.all(np.isfinite(result.positions))
