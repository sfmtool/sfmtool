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


def test_refine_normals_improves_consensus(
    sfmrfile_reconstruction_with_17_images: Path,
):
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    images = _load_images(recon)

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0
    n0 = np.array([cloud[i].normal for i in range(len(cloud))])

    # Modest search params keep the test quick; correctness, not quality.
    res = cloud.refine_normals(
        recon,
        images,
        resolution=12,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
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


def _refine(recon, images, cache, cache_supersample):
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    return cloud.refine_normals(
        recon,
        images,
        resolution=16,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
        cache=cache,
        cache_supersample=cache_supersample,
    )


def test_fronto_cache_matches_source_rendering(
    sfmrfile_reconstruction_with_17_images: Path,
):
    """The fronto-parallel cache reproduces the source-render refinement.

    See ``specs/core/fronto-parallel-patch-cache.md``: the cache should be
    Φ-equivalent (the angular tail is ambiguity, not error) and must not drop a
    meaningful fraction of the scored points.
    """
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    images = _load_images(recon)

    off = _refine(recon, images, cache="off", cache_supersample=1.0)
    on = _refine(recon, images, cache="fronto", cache_supersample=2.0)

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
    sfmrfile_reconstruction_kerry_park: Path,
):
    """The cache stays Φ-equivalent on a genuine back-to-back fisheye rig.

    The base↔candidate affine is fit from *undistorted-normalized* corners, so the
    lens distortion cancels in the correspondence (see
    ``specs/core/fronto-parallel-patch-cache.md``). If it used the distorted pixel
    corners instead, the cache would mis-resample the strongly-distorted fisheye
    views and Φ would collapse — this test guards that cancellation.
    """
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_kerry_park)
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
    sfmrfile_reconstruction_with_17_images: Path,
):
    import pytest

    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )

    with pytest.raises(ValueError):
        cloud.refine_normals(recon, images, resolution=12, cache="bogus")
    with pytest.raises(ValueError):
        cloud.refine_normals(recon, images, resolution=12, cache_supersample=0.5)
