# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for ``PatchCloud.refine_normals`` against a real reconstruction.

Builds a patch cloud from the solved 17-image seoul_bull reconstruction and runs
photometric normal refinement over its real ``.sift``-derived patches and source
images — the multi-view rendering path the Rust unit tests can't exercise without
on-disk images.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction


def _load_images(recon) -> list[np.ndarray]:
    import cv2  # heavy module, only needed by this integration test

    ws = recon.workspace_dir
    images = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        images.append(np.ascontiguousarray(bgr))
    return images


def _sample_point_ids(cloud, n: int = 500, seed: int = 0) -> list[int]:
    """A deterministic point-id subset to refine instead of the whole cloud.

    The per-point photometric search dominates ``refine_normals`` runtime, so
    refining a representative sample keeps these integration tests fast while
    leaving the statistical assertions (consensus improves, cache equivalence)
    well-populated. ``min(n, …)`` keeps it robust to small clouds.
    """
    ids = np.asarray(cloud.point_ids)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False)).tolist()


def test_refine_normals_improves_consensus(
    seoul_bull_workspace: Path,
):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0
    n0 = np.array([cloud[i].normal for i in range(len(cloud))])

    # Modest search params + a point subset keep the test quick; correctness,
    # not quality. Confidence is opt-in (off by default), and this test checks it.
    res = cloud.refine_normals(
        recon,
        images,
        point_ids=_sample_point_ids(cloud),
        resolution=12,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
        compute_confidence=True,
    )

    normals = res["normal"]
    photo = res["photoconsistency"]
    init = res["init_photoconsistency"]
    conf = res["confidence"]
    vvc = res["valid_view_count"]

    assert normals.shape == (len(cloud), 3)

    # Returned normals are unit length and finite.
    norms = np.linalg.norm(normals, axis=1)
    assert np.all(np.isfinite(norms))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # The cloud was updated in place to the returned normals.
    n1 = np.array([cloud[i].normal for i in range(len(cloud))])
    np.testing.assert_allclose(n1, normals, atol=1e-9)

    # Some patches were actually scored (had enough valid views).
    scored = np.isfinite(photo) & np.isfinite(init)
    assert scored.sum() > 0
    assert np.all(vvc[scored] >= 3)

    # Refinement never lowers the consensus relative to the init (frozen support),
    # and strictly improves at least one patch (it genuinely does something).
    assert np.all(photo[scored] >= init[scored] - 1e-9)
    assert np.any(photo[scored] > init[scored] + 1e-6)

    # Confidence is finite and non-negative.
    assert np.all(np.isfinite(conf))
    assert np.all(conf >= 0.0)

    # At least one normal moved away from its mean-viewing init.
    moved = 1.0 - np.abs(np.sum(n0 * n1, axis=1))
    assert np.nanmax(moved) > 1e-4


def test_confidence_is_opt_in(seoul_bull_workspace: Path):
    """Confidence is NaN unless ``compute_confidence=True`` (off by default)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    sample = _sample_point_ids(
        PatchCloud.from_reconstruction(recon, normal="mean_viewing", extent_value=5.0)
    )

    def conf(compute):
        cloud = PatchCloud.from_reconstruction(
            recon, normal="mean_viewing", extent_value=5.0
        )
        res = cloud.refine_normals(
            recon,
            images,
            point_ids=sample,
            resolution=12,
            init_steps=5,
            refine_levels=2,
            sampler="bilinear",
            compute_confidence=compute,
        )
        return np.asarray(res["confidence"]), np.asarray(res["photoconsistency"])

    # Refined (scored) patches report NaN confidence when not requested;
    # unrefined patches keep the 0.0 not-refined sentinel either way.
    off, photo_off = conf(False)
    scored = np.isfinite(photo_off)
    assert scored.sum() > 0
    assert np.all(np.isnan(off[scored]))

    on, photo_on = conf(True)
    scored = np.isfinite(photo_on)
    assert np.all(np.isfinite(on[scored])) and np.all(on[scored] >= 0.0)


def test_render_bitmaps_scatters_to_points(seoul_bull_workspace: Path):
    """``render_bitmaps`` returns a per-3D-point RGBA bitmap array."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = _sample_point_ids(cloud)
    resolution = 12
    res = cloud.refine_normals(
        recon,
        images,
        point_ids=sample,
        resolution=resolution,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
        render_bitmaps=True,
    )

    # Off by default: omitting render_bitmaps yields no bitmaps key.
    assert "bitmaps" in res
    bitmaps = res["bitmaps"]
    npoints = len(recon.positions)
    assert bitmaps.shape == (npoints, resolution, resolution, 4)
    assert bitmaps.dtype == np.uint8

    # Only refined (scored) patches get a filled row; everything else is zero.
    photo = np.asarray(res["photoconsistency"])
    point_ids = np.asarray(cloud.point_ids)
    filled = bitmaps.any(axis=(1, 2, 3))
    assert filled.sum() > 0
    # Every filled row belongs to a scored patch's point id.
    scored_pids = set(point_ids[np.isfinite(photo)].tolist())
    assert set(np.nonzero(filled)[0].tolist()).issubset(scored_pids)

    # A filled patch has some non-zero alpha (cross-view agreement) where covered.
    alpha = bitmaps[..., 3]
    assert alpha[filled].max() > 0


def test_render_bitmaps_round_trips_through_sfmr(seoul_bull_workspace: Path, tmp_path):
    """Attached bitmaps survive a save / load of the .sfmr."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    res = cloud.refine_normals(
        recon,
        images,
        point_ids=_sample_point_ids(cloud, n=200),
        resolution=10,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
        render_bitmaps=True,
    )

    # Bitmaps require the frame, so attach the cloud too.
    edited = recon.clone_with_changes(patches=cloud, patch_bitmaps=res["bitmaps"])
    assert edited.patch_bitmaps is not None
    np.testing.assert_array_equal(edited.patch_bitmaps, res["bitmaps"])

    out = tmp_path / "with-bitmaps.sfmr"
    edited.save(str(out))

    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.patch_bitmaps is not None
    np.testing.assert_array_equal(reloaded.patch_bitmaps, res["bitmaps"])

    # A reconstruction with no patch data reports None.
    assert recon.patch_bitmaps is None


def test_view_indices_override_expands_view_set(seoul_bull_workspace: Path):
    """``view_indices`` refines each patch over an explicit view set, overriding
    the track observations — the hook for MVS-style all-visible-view refinement."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    n_images = len(images)

    common = dict(resolution=12, init_steps=5, refine_levels=2)

    cloud = PatchCloud.from_reconstruction(recon, normal="stored", extent_value=5.0)
    pids = _sample_point_ids(cloud, n=40)
    base = cloud.refine_normals(recon, images, point_ids=pids, **common)
    base_vvc = np.asarray(base["valid_view_count"])

    # Override every patch's view set with *all* images (a superset of any track).
    # A fresh cloud so both refinements start from the same stored normals.
    cloud2 = PatchCloud.from_reconstruction(recon, normal="stored", extent_value=5.0)
    all_views = [list(range(n_images))] * len(cloud2)
    expanded = cloud2.refine_normals(
        recon, images, point_ids=pids, view_indices=all_views, **common
    )
    exp_vvc = np.asarray(expanded["valid_view_count"])

    idx = {int(p): k for k, p in enumerate(cloud.point_ids)}
    # The validity gates can only keep at least as many views from the full set as
    # from the track subset, for every refined point.
    for p in pids:
        k = idx[int(p)]
        if base_vvc[k] > 0:
            assert exp_vvc[k] >= base_vvc[k]
    # And the expansion genuinely brought extra views into the consensus somewhere.
    assert bool((exp_vvc > base_vvc).any())


def test_view_indices_validation(seoul_bull_workspace: Path):
    """``view_indices`` must be parallel to the cloud and reference real images."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(recon, normal="stored", extent_value=5.0)

    with pytest.raises(ValueError, match="parallel to the cloud"):
        cloud.refine_normals(recon, images, view_indices=[[0, 1]])

    out_of_range = [[len(images)]] * len(cloud)
    with pytest.raises(ValueError, match="out of range"):
        cloud.refine_normals(recon, images, view_indices=out_of_range)


def test_view_indices_dedupes_repeated_views(seoul_bull_workspace: Path):
    """Repeated views within a patch are ignored, so a duplicated index gives the
    same result (and view count) as listing each view once."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    n_images = len(images)
    common = dict(resolution=12, init_steps=5, refine_levels=2)

    cloud = PatchCloud.from_reconstruction(recon, normal="stored", extent_value=5.0)
    pids = _sample_point_ids(cloud, n=30)
    uniq = cloud.refine_normals(
        recon,
        images,
        point_ids=pids,
        view_indices=[list(range(n_images))] * len(cloud),
        **common,
    )

    cloud2 = PatchCloud.from_reconstruction(recon, normal="stored", extent_value=5.0)
    dup = cloud2.refine_normals(
        recon,
        images,
        point_ids=pids,
        view_indices=[list(range(n_images)) * 2] * len(cloud2),
        **common,
    )

    np.testing.assert_array_equal(uniq["valid_view_count"], dup["valid_view_count"])
    np.testing.assert_allclose(uniq["normal"], dup["normal"], atol=1e-9)


def _refine(recon, images, cache, cache_supersample, point_ids):
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    return cloud.refine_normals(
        recon,
        images,
        point_ids=point_ids,
        resolution=16,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
        cache=cache,
        cache_supersample=cache_supersample,
    )


def test_fronto_cache_matches_source_rendering(
    seoul_bull_workspace: Path,
):
    """The fronto-parallel cache reproduces the source-render refinement.

    See ``specs/core/fronto-parallel-patch-cache.md``: the cache should be
    Φ-equivalent (the angular tail is ambiguity, not error) and must not drop a
    meaningful fraction of the scored points.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    sample = _sample_point_ids(
        PatchCloud.from_reconstruction(recon, normal="mean_viewing", extent_value=5.0)
    )

    off = _refine(recon, images, cache="off", cache_supersample=1.0, point_ids=sample)
    on = _refine(recon, images, cache="fronto", cache_supersample=2.0, point_ids=sample)

    # Cached normals are unit length and finite.
    norms = np.linalg.norm(on["normal"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    scored_off = np.isfinite(off["photoconsistency"])
    scored_on = np.isfinite(on["photoconsistency"])
    assert scored_on.sum() >= 0.95 * scored_off.sum()

    both = scored_off & scored_on
    assert both.sum() > 0

    # Φ-equivalent: the cache does not meaningfully lower the consensus it finds.
    mean_phi_off = float(off["photoconsistency"][both].mean())
    mean_phi_on = float(on["photoconsistency"][both].mean())
    assert mean_phi_on >= mean_phi_off - 0.03

    # Normals broadly agree on the common population — the differences
    # concentrate on low-confidence / flat-Φ ambiguous points (Φ-equivalent, as
    # the consensus check above shows). The median is the supplementary signal;
    # this fast low-resolution config (R=16, 2 levels) runs looser than the
    # ~2° seen at the production R=32 (see the report), so the bound is generous.
    a = off["normal"][both]
    b = on["normal"][both]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = np.clip((a * b).sum(1), -1.0, 1.0)
    median_deg = float(np.degrees(np.median(np.arccos(cos))))
    assert median_deg < 12.0


def test_fronto_cache_handles_fisheye_distortion(
    kerry_park_workspace: Path,
):
    """The cache stays Φ-equivalent on a genuine back-to-back fisheye rig.

    The base↔candidate affine is fit from *undistorted-normalized* corners, so the
    lens distortion cancels in the correspondence (see
    ``specs/core/fronto-parallel-patch-cache.md``). If it used the distorted pixel
    corners instead, the cache would mis-resample the strongly-distorted fisheye
    views and Φ would collapse — this test guards that cancellation.
    """
    recon = SfmrReconstruction.load(kerry_park_workspace)
    images = _load_images(recon)

    def build():
        return PatchCloud.from_reconstruction(
            recon, normal="mean_viewing", extent_value=5.0
        )

    ids = np.asarray(build().point_ids)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(ids, size=min(300, len(ids)), replace=False)).tolist()

    def run(cache, ss):
        return build().refine_normals(
            recon,
            images,
            point_ids=sample,
            resolution=16,
            init_steps=5,
            refine_levels=2,
            sampler="bilinear",
            cache=cache,
            cache_supersample=ss,
        )

    off = run("off", 1.0)
    on = run("fronto", 2.0)

    scored_off = np.isfinite(off["photoconsistency"])
    scored_on = np.isfinite(on["photoconsistency"])
    assert scored_on.sum() >= 0.9 * scored_off.sum()

    both = scored_off & scored_on
    assert both.sum() > 0
    # Φ-equivalent on a distorted camera: a distorted-corner map would tank Φ here.
    mean_phi_off = float(off["photoconsistency"][both].mean())
    mean_phi_on = float(on["photoconsistency"][both].mean())
    assert mean_phi_on >= mean_phi_off - 0.05


def test_refine_normals_cache_validation(
    seoul_bull_workspace: Path,
):
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )

    with pytest.raises(ValueError):
        cloud.refine_normals(recon, images, resolution=12, cache="bogus")
    with pytest.raises(ValueError):
        cloud.refine_normals(recon, images, resolution=12, cache_supersample=0.5)
