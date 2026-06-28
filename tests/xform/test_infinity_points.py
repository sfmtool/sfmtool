# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that xform point filters and bundle adjustment handle points at infinity.

Filters whose score is undefined for a direction (triangulation angle, neighbour
distance) pass points at infinity through untouched; filters scoring a quantity
that is well-defined regardless of ``w`` — track length, feature size, or
reprojection error (a ``w = 0`` point still projects) — score them normally.
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


def test_reprojection_filter_scores_infinity_points(
    seoul_bull_sfmr_only,
):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    indices = list(range(5))

    # A point at infinity has a well-defined reprojection error, so it is scored
    # by the filter like any finite point. Give the injected infinity points a
    # huge error (filtered) and a tiny error (kept) to prove both outcomes.
    positions = recon.positions_xyzw.copy()
    errors = recon.errors.copy()
    for i in indices:
        positions[i] = [0.0, 0.0, 1.0, 0.0]
    errors[indices[0]] = 1000.0  # fails the criterion -> removed
    errors[indices[1:]] = 0.1  # passes the criterion -> kept
    recon = recon.clone_with_changes(positions=positions, errors=errors)
    assert recon.point_is_at_infinity[indices].all()
    n_infinity_before = int(recon.point_is_at_infinity.sum())

    result = FilterByReprojectionErrorTransform(threshold=2.0).apply(recon)

    # The high-error infinity point was removed; the low-error ones survived.
    assert int(result.point_is_at_infinity.sum()) == n_infinity_before - 1


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
