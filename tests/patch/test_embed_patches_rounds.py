# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``embed_patches`` round structure: the default round/sweep counts,
the sub-pixel LK sweep, and multi-round refinement. See
``specs/core/patch/sift-to-patch-reconstruction.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.io import verify_sfmr

from .conftest import load_images


def test_embed_patches_default_is_two_rounds_one_sweep(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """The default ``embed_patches`` call (no ``subpixel=`` / ``rounds=`` kwargs) is
    bit-for-bit equivalent to passing ``subpixel=1, rounds=2``. Pins the default so
    flipping it in code can't slip in silently.

    Scope: this only pins the **default kwarg values** (one LK sweep, two rounds).
    It does NOT pin the broader behavioral contract — defending that would need a
    baseline artifact compared against this build's output, which this test does
    not carry.
    """
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"
    images = load_images(recon)

    # resolution=12 (vs the resolution=24 default) keeps this a comparison of
    # default-vs-explicit kwargs at a cheaper sampling grid — both sides use the
    # same grid, so the equivalence being pinned is unaffected. The sibling
    # test_embed_patches_command.py tests run at this same low resolution.
    default = embed_patches(recon, images, patch_size=10.0, resolution=12)
    explicit = embed_patches(
        recon, images, patch_size=10.0, subpixel=1, rounds=2, resolution=12
    )

    assert default.point_count == explicit.point_count
    np.testing.assert_array_equal(
        np.asarray(default.keypoints_xy), np.asarray(explicit.keypoints_xy)
    )


def test_embed_patches_subpixel_lk_round_trips(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """``embed_patches(subpixel=1)`` produces a valid ``embedded_patches``
    reconstruction that round-trips through ``.sfmr``, and its per-view
    keypoints differ from the no-refinement baseline (``subpixel=0``) — the
    refiner actually moved something (it ran end-to-end, not a no-op splice).
    """
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)

    # Pin rounds=1 so the subpixel pass is the terminal step: it feeds nothing
    # downstream, so the only membership change it can cause is its own.
    # (At rounds>=2 the round-1 keypoints also feed the next round's normal
    # refinement + grazing drop — covered by
    # test_embed_patches_multiple_rounds_round_trips.)
    # resolution=12 (vs default 24) is a cheaper sampling grid; the assertion is
    # a relative baseline-vs-refined comparison at a fixed grid, so it holds.
    baseline = embed_patches(
        recon, images, patch_size=10.0, subpixel=0, rounds=1, resolution=12
    )
    refined = embed_patches(
        recon, images, patch_size=10.0, subpixel=1, rounds=1, resolution=12
    )

    assert baseline.feature_source == refined.feature_source == "embedded_patches"

    # The subpixel pass can move the point count in BOTH directions, so bound
    # the difference symmetrically rather than asserting a direction:
    #
    # - Shrink: it drops views that won't co-register (`max_shift_px`, low LOO
    #   ZNCC, grazing) and then culls points left below `min_views`.
    # - Grow: `_refine_subpixel` runs in both configurations (render-only at
    #   `subpixel=0`), and the compaction's culled-point signal is the
    #   consensus validity at whatever keypoints the pass ends with — the
    #   seeds for the baseline, the LK-refined keypoints here. Refinement can
    #   flip a marginal consensus from invalid to valid, RESCUING a point the
    #   baseline culls. (`embed_patches` itself is deterministic — 4/4
    #   bit-identical repeat runs on a fixed solve — so a grown count is this
    #   rescue, not run-to-run jitter.)
    #
    # Do not require the counts to be *equal* either. The fixture re-solves
    # the reconstruction every session and the solve is not reproducible
    # (COLMAP's geometric verification during matching is nondeterministic),
    # so whether a marginal point clears those gates varies run to run.
    # Measured over 45 solves: 39 identical and 6 that culled exactly one
    # two-view point; a later CI solve rescued one (937 vs 936).
    assert abs(refined.point_count - baseline.point_count) <= 2, (
        f"subpixel moved the point count by "
        f"{refined.point_count - baseline.point_count} "
        f"({baseline.point_count} -> {refined.point_count}); "
        "expected a local refinement, not a rebuild"
    )

    # The refiner must actually have moved keypoints — otherwise the splice is
    # a no-op and the wiring is broken. Compare multisets rounded to 1e-3
    # rather than row-by-row: a cull shifts every later row, so a positional
    # comparison is only valid when nothing was culled. A keypoint that moved
    # by less than ~1e-3 rounds into its original bucket and counts as
    # unmoved, preserving the original threshold's intent.
    def _buckets(recon):
        return [(round(float(x), 3), round(float(y), 3)) for x, y in recon.keypoints_xy]

    base_seen = set(_buckets(baseline))
    ref_buckets = _buckets(refined)
    moved = sum(1 for b in ref_buckets if b not in base_seen)
    # Measured 94-95% of observations move; 50% leaves ample headroom while
    # still failing a no-op (0%) or a near-no-op.
    assert moved > 0.5 * len(ref_buckets), (
        f"subpixel=1 moved only {moved}/{len(ref_buckets)} keypoints "
        "(wiring is a no-op?)"
    )

    # Round-trip the refined recon through .sfmr to confirm it's structurally
    # valid (the path the CLI takes).
    out = tmp_path / "refined.sfmr"
    refined.save(str(out), operation="embed-patches subpixel=1")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.feature_source == "embedded_patches"
    assert reloaded.point_count == refined.point_count


def test_embed_patches_multiple_rounds_round_trips(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """``rounds > 1`` alternates normal- and keypoint-refinement, feeding each
    round into the next and re-pruning grazing observations. The output is a valid
    ``embedded_patches`` recon; the per-round grazing drop can only shrink the
    observation/point set, never grow it. A per-round ``progress`` callback fires
    once per round."""
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)

    # resolution=12 (vs default 24) is a cheaper sampling grid; the assertions
    # are relative (three-rounds vs one-round monotonicity) at a fixed grid.
    one = embed_patches(
        recon, images, patch_size=10.0, subpixel=1, rounds=1, resolution=12
    )

    lines: list[str] = []
    three = embed_patches(
        recon,
        images,
        patch_size=10.0,
        subpixel=1,
        rounds=3,
        resolution=12,
        progress=lines.append,
    )

    assert three.feature_source == "embedded_patches"
    # Per-round grazing pruning can only remove points/observations, so three
    # rounds keeps no more points than one.
    assert three.point_count <= one.point_count
    assert three.point_count > 0
    assert (
        np.asarray(three.keypoints_xy).shape[0] <= np.asarray(one.keypoints_xy).shape[0]
    )

    # One per-round summary line (the "normal Δ ..." metric line) per round.
    # Phase lines share the "round N/M:" prefix, so match the summary distinctly.
    assert sum("normal Δ" in line for line in lines) == 3

    out = tmp_path / "rounds.sfmr"
    three.save(str(out), operation="embed-patches rounds=3")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
