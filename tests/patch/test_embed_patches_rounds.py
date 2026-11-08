# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``embed_patches`` round structure: the default round count,
the sub-pixel LK pass, and multi-round refinement. See
``specs/core/patch/sift-to-patch-reconstruction.md``.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.fileio import verify_sfmr

from .conftest import load_images


@pytest.fixture(scope="module")
def scene(seoul_bull_workspace_once: Path):
    """The seoul_bull reconstruction and its images. Read them; do not modify
    them or their workspace."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once)
    return recon, load_images(recon)


@pytest.fixture(scope="module")
def one_round(scene):
    """One round with the sub-pixel pass, at ``patch_size=10`` and
    ``resolution=12``, and its ``progress`` lines. Read them; do not modify
    them."""
    from sfmtool._embed_patches import embed_patches

    recon, images = scene
    lines: list[str] = []
    out = embed_patches(
        recon,
        images,
        patch_size=10.0,
        subpixel=True,
        rounds=1,
        resolution=12,
        progress=lines.append,
    )
    return out, lines


def test_embed_patches_default_is_two_rounds_with_subpixel(default_embed_patches_run):
    """The default ``embed_patches`` call (no ``subpixel=`` / ``rounds=`` kwargs) is
    bit-for-bit equivalent to passing ``subpixel=True, rounds=2``. Pins the default so
    flipping it in code can't slip in silently.

    Scope: this only pins the **default kwarg values** (the LK pass on, two rounds).
    It does NOT pin the broader behavioral contract — defending that would need a
    baseline artifact compared against this build's output, which this test does
    not carry.
    """
    from sfmtool._embed_patches import embed_patches

    run = default_embed_patches_run
    recon, images = run.recon, run.images
    assert recon.feature_source == "sift_files"

    # resolution=12 (vs the resolution=24 default) keeps this a comparison of
    # default-vs-explicit kwargs at a cheaper sampling grid — both sides use the
    # same grid, so the equivalence being pinned is unaffected. The default
    # side is default_embed_patches_run, which other patch tests also read.
    default = run.out
    explicit = embed_patches(recon, images, subpixel=True, rounds=2, resolution=12)

    assert default.point_count == explicit.point_count
    np.testing.assert_array_equal(
        np.asarray(default.keypoints_xy), np.asarray(explicit.keypoints_xy)
    )


def test_embed_patches_subpixel_lk_round_trips(scene, one_round, tmp_path: Path):
    """``embed_patches(subpixel=True)`` produces a valid ``embedded_patches``
    reconstruction that round-trips through ``.sfmr``, and its per-view
    keypoints differ from the no-refinement baseline (``subpixel=False``) — the
    refiner actually moved something (it ran end-to-end, not a no-op splice).
    """
    from sfmtool._embed_patches import embed_patches

    recon, images = scene

    # Pin rounds=1 so the subpixel pass is the terminal step: it feeds nothing
    # downstream, so the only membership change it can cause is its own.
    # (At rounds>=2 the round-1 keypoints also feed the next round's normal
    # refinement + grazing drop — covered by
    # test_embed_patches_multiple_rounds_round_trips.)
    # resolution=12 (vs default 24) is a cheaper sampling grid; the assertion is
    # a relative baseline-vs-refined comparison at a fixed grid, so it holds.
    baseline = embed_patches(
        recon, images, patch_size=10.0, subpixel=False, rounds=1, resolution=12
    )
    refined, _ = one_round

    assert baseline.feature_source == refined.feature_source == "embedded_patches"

    # The subpixel pass can move the point count in BOTH directions, so bound
    # the difference symmetrically rather than asserting a direction:
    #
    # - Shrink: it drops views that won't co-register (`max_shift_px`, low
    #   ZNCC against the reference, grazing) and then culls points left below
    #   `min_views`.
    # - Grow: `_refine_subpixel` runs in both configurations (render-only at
    #   `subpixel=False`), and the compaction's culled-point signal is the
    #   consensus validity at whatever keypoints the pass ends with — the
    #   seeds for the baseline, the LK-refined keypoints here. Refinement can
    #   flip a marginal consensus from invalid to valid, RESCUING a point the
    #   baseline culls. (`embed_patches` itself is deterministic — 4/4
    #   bit-identical repeat runs on a fixed reconstruction, so a grown count
    #   is this rescue, not run-to-run jitter.)
    #
    # Do not require the counts to be *equal* either: which marginal points
    # clear those gates is a property of the input geometry, not of the wiring
    # this test pins. The bound says the pass stays a local refinement.
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
        f"subpixel=True moved only {moved}/{len(ref_buckets)} keypoints "
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


def _positions(recon: SfmrReconstruction) -> list[tuple[float, ...]]:
    """Each point's position, in point order. ``embed_patches`` copies positions
    through unchanged, so a position names the same point in two runs whose
    compactions number the survivors differently. A few input points share a
    position, so the comparisons count positions with a ``Counter``."""
    return [tuple(row) for row in np.asarray(recon.positions).tolist()]


def _observed_images(
    recon: SfmrReconstruction, keys: set[tuple[float, ...]]
) -> dict[tuple[float, ...], set[int]]:
    """The images observing each point whose position is in ``keys``."""
    positions = _positions(recon)
    observed: dict[tuple[float, ...], set[int]] = {
        key: set() for key in positions if key in keys
    }
    for pid, img in zip(
        np.asarray(recon.track_point_indexes).tolist(),
        np.asarray(recon.track_image_indexes).tolist(),
    ):
        if positions[pid] in keys:
            observed[positions[pid]].add(int(img))
    return observed


def _self_similarity_culled(lines: list[str]) -> int:
    """The number of points the round-1 ZNCC self-similarity cull dropped, read
    from its ``progress`` line (absent when it dropped none)."""
    counts = [
        int(m.group(1))
        for line in lines
        if (m := re.search(r"culled (\d+) points whose bitmap", line))
    ]
    assert len(counts) <= 1
    return counts[0] if counts else 0


def test_embed_patches_multiple_rounds_round_trips(scene, one_round, tmp_path: Path):
    """``rounds > 1`` alternates normal- and keypoint-refinement, feeding each
    round into the next and re-pruning grazing observations. The output is a valid
    ``embedded_patches`` recon. Every point a one-round or a three-round run keeps
    is an input point that survived round 1's self-similarity cull, and a point
    both runs keep is observed by the three-round run in a subset of the images
    the one-round run observes it in. A per-round ``progress`` callback fires
    once per round.

    The point count is not monotone in the number of rounds. The final cull
    drops each point the last sub-pixel pass rendered no valid bitmap for, but a
    point with no valid bitmap after round 1 has no self-similarity reading, so
    the early cull keeps it and the later rounds carry it forward; at the normals
    and keypoints of round 3 its bitmap can be valid, and the three-round run
    then keeps a point the one-round run drops."""
    from sfmtool._embed_patches import embed_patches

    recon, images = scene

    # resolution=12 (vs default 24) is a cheaper sampling grid; both runs use
    # it, so the comparison between them is unaffected.
    one, one_lines = one_round

    lines: list[str] = []
    three = embed_patches(
        recon,
        images,
        patch_size=10.0,
        subpixel=True,
        rounds=3,
        resolution=12,
        progress=lines.append,
    )

    assert three.feature_source == "embedded_patches"
    assert three.point_count > 0

    # Round 1 is the same in both runs, so both drop the same points by ZNCC
    # self-similarity radius, and no later step brings a point back: together
    # the two outputs hold input points only, and no more of them than that
    # cull left.
    n_culled = _self_similarity_culled(lines)
    assert _self_similarity_culled(one_lines) == n_culled
    input_counts = Counter(_positions(recon))
    one_counts = Counter(_positions(one))
    three_counts = Counter(_positions(three))
    assert one_counts <= input_counts
    assert three_counts <= input_counts
    assert (one_counts | three_counts).total() <= recon.point_count - n_culled

    # A point's views are fixed in round 1 and each later round's grazing cut
    # only removes views, so where both runs keep a point the three-round run
    # observes it in no image the one-round run does not. Compared only for
    # positions no other input point shares, so each names one point.
    unique = {key for key, n in input_counts.items() if n == 1}
    common = unique & one_counts.keys() & three_counts.keys()
    assert common
    one_images = _observed_images(one, common)
    three_images = _observed_images(three, common)
    for key in common:
        assert three_images[key] <= one_images[key], key

    # One per-round summary line (the "normal Δ ..." metric line) per round.
    # Phase lines share the "round N/M:" prefix, so match the summary distinctly.
    assert sum("normal Δ" in line for line in lines) == 3

    out = tmp_path / "rounds.sfmr"
    three.save(str(out), operation="embed-patches rounds=3")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
