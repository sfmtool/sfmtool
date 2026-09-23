# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Binding tests for ``PatchCloud.validate_member_coherence``.

Drives the pairwise member-agreement kernel over a real reconstruction's patches
and source images — the multi-view rendering path the Rust unit tests cannot
exercise without on-disk imagery — and pins the *binding* surface: the dict keys
and dtypes, the ``k×k`` matrix, the two member-list sources (tracks vs an explicit
``member_views`` map), and the unscored-member contract. The decision rule itself
is covered by the Rust tests. See ``specs/core/patch/member-coherence-validation.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool.patches import CameraViews, PatchCloud
from sfmtool._sfmtool.reconstruction import SfmrReconstruction

from .conftest import load_images, sample_point_ids

VERDICTS = {"keep_all", "split", "retire"}
KEYS = {
    "point_index",
    "members",
    "verdict",
    "kept",
    "block",
    "scored",
    "support",
    "n_support",
    "margin",
    "min_intra",
    "max_cross",
    "effective_bar",
    "effective_margin_gate",
    "core_center",
    "core_scatter",
    "relative_flagged",
    "exonerated",
    "retained_deficit",
    "sharpness_deficit",
}


@pytest.fixture(scope="module")
def scene(seoul_bull_workspace_once: Path):
    """A patch cloud, its reconstruction and its images (built once)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0
    return recon, cloud, images


def _validate(scene, **kwargs):
    recon, cloud, images = scene
    kwargs.setdefault("resolution", 12)
    return cloud.validate_member_coherence(recon, images, **kwargs)


def test_result_dicts_have_the_documented_keys_and_dtypes(scene):
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=60)
    results = _validate(scene, point_indexes=sample)
    assert len(results) == len(sample)

    for r in results:
        assert set(r.keys()) == KEYS
        members = r["members"]
        # uint32, not int32: these are image indices straight off the track.
        assert members.dtype == np.uint32
        k = len(members)
        assert k >= 1
        # Deduplicated, and every other per-member array is parallel to it.
        assert len(set(members.tolist())) == k
        for name in ("kept", "block", "scored"):
            assert r[name].dtype == np.bool_
            assert r[name].shape == (k,)
        assert r["verdict"] in VERDICTS
        assert isinstance(r["point_index"], int)
        assert isinstance(r["support"], int)
        assert isinstance(r["n_support"], int)
        assert 0 <= r["support"] <= k
        assert r["n_support"] >= 0
        for name in (
            "margin",
            "min_intra",
            "max_cross",
            "effective_bar",
            "effective_margin_gate",
            "core_center",
            "core_scatter",
        ):
            assert isinstance(r[name], float)
        # The thresholds the rule really ran at bracket the absolute pair: the
        # bar is a floor that the relative term raises, the margin gate a
        # ceiling it lowers. NaN means no sweep ran at all.
        if not np.isnan(r["effective_bar"]):
            assert r["effective_bar"] >= 0.65
            assert r["effective_bar"] <= 0.99
            assert 0.0 < r["effective_margin_gate"] <= 0.05
        # The statistics are reported exactly when the relative term was active.
        assert np.isnan(r["core_center"]) == np.isnan(r["core_scatter"])
        if not np.isnan(r["core_scatter"]):
            assert r["core_scatter"] >= 0.005

        # The verdict's own invariants, at the binding boundary.
        kept, block, scored = r["kept"], r["block"], r["scored"]
        assert int(block.sum()) == r["support"]
        # The block is drawn from the scored members only.
        assert not (block & ~scored).any()
        if r["verdict"] == "keep_all":
            assert kept.all()
        elif r["verdict"] == "retire":
            assert not kept.any()
        else:
            # A split keeps its block plus everything with no evidence against it.
            assert np.array_equal(kept, block | ~scored)
        # A correlated track has a real common support behind it.
        if scored.any():
            assert r["n_support"] >= 8


def test_return_matrix_gives_a_symmetric_unit_diagonal_kxk(scene):
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=40)
    results = _validate(scene, point_indexes=sample, return_matrix=True)

    saw_finite_pair = False
    for r in results:
        k = len(r["members"])
        zncc = r["zncc"]
        assert zncc.dtype == np.float64
        assert zncc.shape == (k, k)
        assert np.array_equal(np.diagonal(zncc), np.ones(k))
        # Symmetric, NaNs included (nan != nan, so compare the mask separately).
        assert np.array_equal(np.isnan(zncc), np.isnan(zncc.T))
        finite = np.isfinite(zncc)
        assert np.allclose(zncc[finite], zncc.T[finite], atol=0.0, rtol=0.0)
        # `scored` is exactly "has a finite off-diagonal entry".
        off_diagonal = finite & ~np.eye(k, dtype=bool)
        assert np.array_equal(off_diagonal.any(axis=1), r["scored"])
        saw_finite_pair |= bool(off_diagonal.any())
    assert saw_finite_pair, "no pair was correlated anywhere in the sample"


def test_point_indexes_selects_exactly_that_subset(scene):
    _, cloud, _ = scene
    every = {int(r["point_index"]) for r in _validate(scene)}
    sample = sample_point_ids(cloud, n=25)
    subset = _validate(scene, point_indexes=sample)
    assert [int(r["point_index"]) for r in subset] == sorted(sample)
    assert set(sample) <= every


def test_member_views_overrides_the_track_and_dedups_first_seen_wins(scene):
    recon, cloud, _ = scene
    pid = sample_point_ids(cloud, n=1)[0]
    n_images = recon.image_count
    assert n_images >= 3

    from_track = _validate(scene, point_indexes=[pid])[0]
    # An explicit list, in an order the track does not use, with duplicates: the
    # binding keeps first-seen order and drops the repeats.
    wanted = [2, 0, 2, 1, 0]
    overridden = _validate(scene, point_indexes=[pid], member_views={pid: wanted})[0]
    assert overridden["members"].tolist() == [2, 0, 1]
    assert overridden["members"].tolist() != from_track["members"].tolist()
    assert len(overridden["kept"]) == 3


def test_member_views_rejects_an_out_of_range_image(scene):
    recon, cloud, _ = scene
    pid = sample_point_ids(cloud, n=1)[0]
    with pytest.raises(ValueError, match="out of range"):
        _validate(scene, point_indexes=[pid], member_views={pid: [recon.image_count]})


def test_camera_views_requires_member_views(scene):
    recon, cloud, images = scene
    views = CameraViews(
        recon.cameras,
        np.asarray(recon.quaternions_wxyz, np.float64),
        np.asarray(recon.translations, np.float64),
        np.asarray(recon.camera_indexes, np.uint32),
    )
    with pytest.raises(ValueError, match="member_views is required"):
        cloud.validate_member_coherence(views, images, resolution=12)


def test_unscored_members_are_kept(scene):
    """The contract, through the binding: a member nothing could correlate is
    missing evidence, not contrary evidence, so no verdict evicts it.

    Forced by handing every point the *whole* image list as its members — most
    images do not cover a given patch, so they fail the coverage gate and come
    back unscored."""
    recon, cloud, _ = scene
    sample = sample_point_ids(cloud, n=40)
    everything = list(range(recon.image_count))
    results = _validate(
        scene,
        point_indexes=sample,
        member_views={int(p): everything for p in sample},
    )

    n_unscored = 0
    for r in results:
        scored, kept = r["scored"], r["kept"]
        n_unscored += int((~scored).sum())
        if r["verdict"] == "retire":
            continue  # the point itself is refused; nothing ships
        assert kept[~scored].all(), (
            f"point {r['point_index']}: an unscored member was evicted ({r['verdict']})"
        )
    assert n_unscored > 0, "no member went unscored — the test proves nothing"


def test_min_support_pixels_fails_open_on_the_whole_track(scene):
    """Above any achievable common support, every track goes unscored and every
    member is kept — and the support count is still reported."""
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=30)
    baseline = _validate(scene, point_indexes=sample)
    gated = _validate(scene, point_indexes=sample, min_support_pixels=10_000)

    assert any(r["scored"].any() for r in baseline), "baseline scored nothing"
    for base, r in zip(baseline, gated):
        assert r["point_index"] == base["point_index"]
        assert not r["scored"].any()
        assert r["verdict"] == "keep_all"
        assert r["kept"].all()
        assert r["support"] == 0
        assert r["n_support"] == base["n_support"]


@pytest.fixture(scope="module")
def embedded_scene(scene):
    """The same cloud over an ``embedded_patches`` reconstruction — the mode that
    carries inline per-observation keypoints, and so the only one in which
    keypoint anchoring is anything but a no-op."""
    from sfmtool import _embed_patches as ep

    recon, _cloud, images = scene
    emb = ep.embed_patches(recon, images, resolution=12)
    assert emb.feature_source == "embedded_patches"
    # The embedded recon carries its own cloud; rebuilding one would need the
    # .sift scales the embed just replaced.
    return emb, emb.patches, images


def test_keypoint_anchoring_is_on_by_default_and_moves_the_matrix(embedded_scene):
    """The default anchors members at their stored keypoints; ``False`` anchors
    them at the reprojection.

    On a converged reconstruction the two are close — the residuals are sub-pixel
    — so this pins the *plumbing* (the flag reaches the render and moves the
    numbers), not a magnitude. The score direction is covered by the Rust test
    that injects a known residual."""
    recon, cloud, images = embedded_scene
    sample = sample_point_ids(cloud, n=80)
    kw = dict(point_indexes=sample, resolution=12, return_matrix=True)
    anchored = cloud.validate_member_coherence(recon, images, **kw)
    projected = cloud.validate_member_coherence(
        recon, images, keypoint_anchor=False, **kw
    )
    default = cloud.validate_member_coherence(recon, images, **kw)

    # The default is the anchored arm.
    for a, d in zip(anchored, default):
        np.testing.assert_array_equal(np.asarray(a["zncc"]), np.asarray(d["zncc"]))

    moved = 0
    for a, p in zip(anchored, projected):
        assert a["point_index"] == p["point_index"]
        za, zp = np.asarray(a["zncc"]), np.asarray(p["zncc"])
        assert za.shape == zp.shape
        both = np.isfinite(za) & np.isfinite(zp)
        if both.any() and not np.allclose(za[both], zp[both], atol=1e-9):
            moved += 1
    assert moved > 0, "anchoring changed no point's matrix - the flag is inert"


def test_keypoint_anchoring_is_a_no_op_without_inline_keypoints(scene):
    """A ``sift_files`` reconstruction carries feature indexes, not keypoints, so
    every member falls back to projection anchoring and the flag cannot bite."""
    recon, cloud, images = scene
    assert recon.feature_source == "sift_files"
    sample = sample_point_ids(cloud, n=40)
    kw = dict(point_indexes=sample, resolution=12, return_matrix=True)
    anchored = cloud.validate_member_coherence(recon, images, **kw)
    projected = cloud.validate_member_coherence(
        recon, images, keypoint_anchor=False, **kw
    )
    for a, p in zip(anchored, projected):
        np.testing.assert_array_equal(np.asarray(a["zncc"]), np.asarray(p["zncc"]))


def test_keypoint_anchoring_survives_a_member_views_override(embedded_scene):
    """An overridden member list is re-keyed against the point's own track, so a
    member the point really observes still carries its keypoint and one it does
    not falls back to its projection - either way the call goes through."""
    recon, cloud, images = embedded_scene
    sample = sample_point_ids(cloud, n=20)
    everything = list(range(recon.image_count))
    out = cloud.validate_member_coherence(
        recon,
        images,
        resolution=12,
        point_indexes=sample,
        member_views={int(p): everything for p in sample},
    )
    assert len(out) == len(sample)
    for r in out:
        assert len(r["members"]) == recon.image_count
        assert r["verdict"] in VERDICTS


def test_self_bar_k_zero_disables_the_relative_term(scene):
    """``self_bar_k=0`` reports the absolute thresholds, no statistics, and the
    verdicts the rule reached before the self-normalized bar existed; raising it
    can only ever tighten, never loosen."""
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=60)
    off = _validate(scene, point_indexes=sample, self_bar_k=0.0)
    on = _validate(scene, point_indexes=sample, self_bar_k=1.5)

    for r in off:
        assert r["effective_bar"] == 0.65
        assert r["effective_margin_gate"] == 0.05
        assert np.isnan(r["core_center"]) and np.isnan(r["core_scatter"])

    engaged = 0
    for a, b in zip(off, on):
        assert a["point_index"] == b["point_index"]
        # The relative term is a tightening: a higher bar and a lower gate, and
        # never the other way round.
        assert b["effective_bar"] >= a["effective_bar"]
        assert b["effective_margin_gate"] <= a["effective_margin_gate"]
        engaged += b["effective_bar"] > a["effective_bar"]
        if b["effective_bar"] == a["effective_bar"]:
            # An inactive or collapsed relative term decides exactly as the
            # absolute rule does.
            assert b["verdict"] == a["verdict"]
            assert np.array_equal(b["kept"], a["kept"])
    assert engaged > 0, "the relative term never engaged â€” the test proves nothing"


def test_the_effective_bar_is_the_core_centre_less_k_scatters(scene):
    """Wherever the relative term is active, the reported bar is exactly
    ``centre - self_bar_k * scatter`` clamped into ``[bar, 0.99]`` â€” the bar the
    block sweep ran at, not a summary of it."""
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=80)
    k_self = 2.0
    active = 0
    for r in _validate(scene, point_indexes=sample, self_bar_k=k_self):
        if np.isnan(r["core_center"]):
            # Inactive: too few intra-block pairs, or no sweep at all.
            assert np.isnan(r["effective_bar"]) or r["effective_bar"] == 0.65
            continue
        active += 1
        want = min(r["core_center"] - k_self * r["core_scatter"], 0.99)
        assert r["effective_bar"] == pytest.approx(max(0.65, want), abs=1e-12)
        assert r["effective_margin_gate"] == pytest.approx(
            min(0.05, r["core_scatter"]), abs=1e-12
        )
    assert active > 0, "the relative term was never active"


def test_exoneration_keys_and_per_member_shapes(scene):
    """The three exoneration columns and the sharpness column are present, are
    parallel to ``members``, and carry the documented dtypes."""
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=40)
    for r in _validate(scene, point_indexes=sample):
        k = len(np.asarray(r["members"]))
        for key, dtype in (
            ("relative_flagged", np.bool_),
            ("exonerated", np.bool_),
            ("retained_deficit", np.float64),
            ("sharpness_deficit", np.float64),
        ):
            arr = np.asarray(r[key])
            assert arr.shape == (k,), f"{key} must be parallel to members"
            assert arr.dtype == dtype, f"{key} dtype"
        # Sparing is a subset of flagging, always.
        flagged = np.asarray(r["relative_flagged"], bool)
        spared = np.asarray(r["exonerated"], bool)
        assert not (spared & ~flagged).any()
        # A ratio is only ever reported for a flagged member.
        rd = np.asarray(r["retained_deficit"], float)
        assert not np.isfinite(rd[~flagged]).any()


def test_return_matrix_carries_the_coarse_scales(scene):
    """``return_matrix`` yields the coarse tables alongside the full-scale one,
    same shape, coarsest last, with their factors."""
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=10)
    # 12 admits one halving (12/2 = 6); 12/4 = 3 is below the floor.
    for r in _validate(scene, point_indexes=sample, return_matrix=True):
        z = np.asarray(r["zncc"])
        coarse = r["zncc_coarse"]
        factors = list(r["coarse_factors"])
        assert factors == [2]
        assert len(coarse) == len(factors)
        for table in coarse:
            t = np.asarray(table)
            assert t.shape == z.shape
            np.testing.assert_array_equal(np.diag(t), np.ones(len(t)))
            # A member unscored at full scale is unscored at every scale.
            np.testing.assert_array_equal(np.isfinite(t), np.isfinite(z))


def test_exoneration_only_ever_keeps_members(scene):
    """Turning exoneration on can only add members back: every point's kept set
    at the shipping threshold is a superset of the raw rule's, the verdicts only
    move towards keeping, and the reported block never moves at all."""
    # The whole cloud, not a sample: the relative term reaches a handful of this
    # fixture's tracks and a subsample reliably misses all of them.
    off = _validate(scene, exoneration_ratio=0.0)
    # A ratio of 1 spares every flagged member whose deficit decays at all, which
    # is what makes the DIRECTION of the effect testable on any fixture; the
    # shipping threshold is a calibration, and where it lands is not this test's
    # business.
    on = _validate(scene, exoneration_ratio=1.0)

    rank = {"retire": 0, "split": 1, "keep_all": 2}
    spared_any = 0
    for a, b in zip(off, on):
        assert a["point_index"] == b["point_index"]
        # The block the sweep produced is a property of the matrix and the
        # thresholds, so exoneration must not touch it.
        np.testing.assert_array_equal(a["block"], b["block"])
        assert a["support"] == b["support"]
        assert a["effective_bar"] == b["effective_bar"] or (
            np.isnan(a["effective_bar"]) and np.isnan(b["effective_bar"])
        )
        # Flagging is the same either way; only acting on it differs.
        np.testing.assert_array_equal(a["relative_flagged"], b["relative_flagged"])
        assert not np.asarray(a["exonerated"], bool).any()
        keep_a = np.asarray(a["kept"], bool)
        keep_b = np.asarray(b["kept"], bool)
        assert not (keep_a & ~keep_b).any(), "exoneration must never evict"
        assert rank[b["verdict"]] >= rank[a["verdict"]]
        spared_any += int(np.asarray(b["exonerated"], bool).any())
    assert spared_any > 0, "nothing was ever spared â€” the test proves nothing"


def test_exoneration_is_inert_without_the_relative_term(scene):
    """With ``self_bar_k=0`` nothing is ever flagged, so no value of
    ``exoneration_ratio`` can change a single verdict."""
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=80)
    base = _validate(scene, point_indexes=sample, self_bar_k=0.0, exoneration_ratio=0.0)
    for tau in (0.5, 0.90, 1.0):
        other = _validate(
            scene, point_indexes=sample, self_bar_k=0.0, exoneration_ratio=tau
        )
        for a, b in zip(base, other):
            assert a["verdict"] == b["verdict"]
            np.testing.assert_array_equal(a["kept"], b["kept"])
            assert not np.asarray(b["relative_flagged"], bool).any()
            assert not np.asarray(b["exonerated"], bool).any()


def test_sharpness_is_measured_off_the_verdict(scene):
    """Sharpness describes the observations a point ships, so it is reported for
    scored members regardless of verdict and does not move with the exoneration
    knob."""
    _, cloud, _ = scene
    sample = sample_point_ids(cloud, n=80)
    off = _validate(scene, point_indexes=sample, exoneration_ratio=0.0)
    on = _validate(scene, point_indexes=sample, exoneration_ratio=0.90)
    measured = 0
    for a, b in zip(off, on):
        sa = np.asarray(a["sharpness_deficit"], float)
        sb = np.asarray(b["sharpness_deficit"], float)
        np.testing.assert_array_equal(np.isnan(sa), np.isnan(sb))
        np.testing.assert_allclose(sa[~np.isnan(sa)], sb[~np.isnan(sb)])
        # Where it is defined at all, it is defined for scored members.
        scored = np.asarray(a["scored"], bool)
        assert not np.isfinite(sa[~scored]).any()
        measured += int(np.isfinite(sa).sum())
    assert measured > 0, "nothing was measured â€” the test proves nothing"
