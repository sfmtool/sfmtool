# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""``PatchCloud.refine_normals(use_stored_keypoints=...)`` against a real recon.

Exercises the stored-keypoint anchoring path: on an ``embedded_patches``
reconstruction each view's patch is positioned at its stored per-observation
keypoint (instead of the reprojected point center). Verifies the call runs and
returns the documented dict shape, and that requesting it on a ``sift_files``
reconstruction is rejected.
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


def _sample_point_ids(cloud, n: int = 200, seed: int = 0) -> list[int]:
    ids = np.asarray(cloud.point_ids)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False)).tolist()


def test_use_stored_keypoints_runs_on_embedded_patches(seoul_bull_workspace: Path):
    sift_recon = SfmrReconstruction.load(seoul_bull_workspace)
    recon = sift_recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
    assert recon.feature_source == "embedded_patches"
    images = _load_images(recon)

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    assert len(cloud) > 0

    point_ids = _sample_point_ids(cloud)
    res = cloud.refine_normals(
        recon,
        images,
        point_ids=point_ids,
        use_stored_keypoints=True,
        resolution=12,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
    )

    # Documented dict shape, parallel to the cloud.
    n = len(cloud)
    assert res["normal"].shape == (n, 3)
    for key in ("photoconsistency", "init_photoconsistency", "confidence"):
        assert res[key].shape == (n,)
    assert res["valid_view_count"].shape == (n,)
    # Refined normals are finite unit-ish vectors where a patch was refined.
    norms = np.linalg.norm(res["normal"], axis=1)
    assert np.all(np.isfinite(res["normal"]))
    assert np.all(norms > 0.0)


def test_use_stored_keypoints_differs_from_centered(seoul_bull_workspace: Path):
    # Anchoring at the stored detection keypoints should move at least some
    # refined results relative to the point-centered refine (the solve carries
    # nonzero reprojection error, so the keypoints sit off the projected center).
    sift_recon = SfmrReconstruction.load(seoul_bull_workspace)
    recon = sift_recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
    images = _load_images(recon)
    point_ids = None

    cloud_a = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    cloud_b = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    common = dict(
        point_ids=point_ids,
        resolution=12,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
        cache="off",
    )
    centered = cloud_a.refine_normals(
        recon, images, use_stored_keypoints=False, **common
    )
    anchored = cloud_b.refine_normals(
        recon, images, use_stored_keypoints=True, **common
    )

    # Most patches refine to similar normals, but the keypoint anchoring must
    # perturb at least one refined normal (otherwise the offset is a no-op).
    dots = np.einsum("ij,ij->i", centered["normal"], anchored["normal"])
    assert np.any(np.abs(dots - 1.0) > 1e-6), (
        "stored keypoints should change some normals"
    )


def test_use_stored_keypoints_rejects_sift_files(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    # Pass NO images: the rejection must fire before any image decode / pyramid
    # build (fail-fast), so an empty image list still raises the right error
    # rather than an image-count mismatch.
    with pytest.raises(
        ValueError, match="use_stored_keypoints requires an embedded_patches"
    ):
        cloud.refine_normals(
            recon,
            [],
            use_stored_keypoints=True,
            resolution=12,
            init_steps=5,
            refine_levels=2,
        )


def test_stored_keypoints_at_reprojection_match_centered(seoul_bull_workspace: Path):
    # Provable mapping guard: overwrite every stored keypoint with its own
    # observation's reprojection of the point center. Anchoring on those (zero
    # offset) must then reproduce the centered refine. A transposed (x/y) or
    # mis-keyed (point/image) binding map would anchor on the WRONG 3D point and
    # diverge sharply — something `differs_from_centered` (any nonzero offset
    # perturbs normals) cannot detect.
    from sfmtool._sfmtool.geometry import RigidTransform

    sift_recon = SfmrReconstruction.load(seoul_bull_workspace)
    recon = sift_recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
    images = _load_images(recon)

    positions = np.asarray(recon.positions, np.float64)
    quats = np.asarray(recon.quaternions_wxyz, np.float64)
    trans = np.asarray(recon.translations, np.float64)
    cam_idx = np.asarray(recon.camera_indexes)
    cams = list(recon.cameras)
    rot = [
        np.asarray(
            RigidTransform.from_wxyz_translation(
                quats[i].tolist(), trans[i].tolist()
            ).to_rotation_matrix(),
            np.float64,
        )
        for i in range(len(quats))
    ]
    tpid = np.asarray(recon.track_point_ids)
    timg = np.asarray(recon.track_image_indexes)
    reproj = np.zeros((len(tpid), 2), np.float32)
    for j in range(len(tpid)):
        im = int(timg[j])
        pc = rot[im] @ positions[int(tpid[j])] + trans[im]
        u, v = cams[int(cam_idx[im])].project(pc[0] / pc[2], pc[1] / pc[2])
        reproj[j] = (u, v)

    recon_reproj = recon.clone_with_changes(keypoints_xy=reproj)
    common = dict(
        resolution=12, init_steps=5, refine_levels=2, sampler="bilinear", cache="off"
    )
    cloud_c = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    cloud_a = PatchCloud.from_reconstruction(
        recon_reproj, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    pids = _sample_point_ids(cloud_c, n=120)
    centered = cloud_c.refine_normals(
        recon, images, point_ids=pids, use_stored_keypoints=False, **common
    )
    anchored = cloud_a.refine_normals(
        recon_reproj, images, point_ids=pids, use_stored_keypoints=True, **common
    )

    # Reprojection-anchored ≈ centered (zero offset). A mis-mapping would diverge
    # far past this tight bound.
    dots = np.einsum("ij,ij->i", centered["normal"], anchored["normal"])
    assert np.all(np.abs(dots - 1.0) < 1e-3), (
        "anchoring on the reprojection must match the centered refine; "
        f"max deviation {np.max(np.abs(dots - 1.0)):.2e}"
    )


def test_use_stored_keypoints_with_view_indices(seoul_bull_workspace: Path):
    # view_indices override + use_stored_keypoints: track images resolve to a
    # stored keypoint; a non-track image added per patch resolves to None (left
    # centered). Exercises the mixed Some/None path through the binding's
    # (point, image) -> keypoint map.
    sift_recon = SfmrReconstruction.load(seoul_bull_workspace)
    recon = sift_recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )

    num_images = len(recon.image_names)
    by_point: dict[int, set[int]] = {}
    for p, im in zip(
        np.asarray(recon.track_point_ids).tolist(),
        np.asarray(recon.track_image_indexes).tolist(),
    ):
        by_point.setdefault(int(p), set()).add(int(im))

    # view_indices must be parallel to every cloud patch.
    view_indices: list[list[int]] = []
    for pid in np.asarray(cloud.point_ids).tolist():
        obs = set(by_point.get(int(pid), set()))
        extra = next((i for i in range(num_images) if i not in obs), None)
        if extra is not None:
            obs = obs | {extra}  # a non-track image -> None keypoint
        view_indices.append(sorted(obs) if obs else [0])

    point_ids = _sample_point_ids(cloud, n=100)
    res = cloud.refine_normals(
        recon,
        images,
        point_ids=point_ids,
        view_indices=view_indices,
        use_stored_keypoints=True,
        resolution=12,
        init_steps=5,
        refine_levels=2,
    )
    n = len(cloud)
    assert res["normal"].shape == (n, 3)
    assert np.all(np.isfinite(res["normal"]))
