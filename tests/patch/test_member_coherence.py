# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Binding tests for ``PatchCloud.validate_member_coherence``.

Drives the pairwise member-agreement kernel over a real reconstruction's patches
and source images — the multi-view rendering path the Rust unit tests cannot
exercise without on-disk imagery — and pins the *binding* surface: the dict keys
and dtypes, the ``k×k`` matrix, the two member-list sources (tracks vs an explicit
``member_views`` map), and the unscored-member contract. The decision rule itself
is covered by the Rust tests. See ``specs/core/member-coherence-validation.md``.
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
        for name in ("margin", "min_intra", "max_cross"):
            assert isinstance(r[name], float)

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
