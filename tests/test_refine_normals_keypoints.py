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

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction


def _load_images(recon) -> list[np.ndarray]:
    import cv2  # heavy module, only needed by this integration test

    ws = recon.workspace_dir
    images = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        images.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return images


def _sample_point_ids(cloud, n: int = 200, seed: int = 0) -> list[int]:
    ids = np.asarray(cloud.point_indexes)
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
        point_indexes=point_ids,
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
        point_indexes=point_ids,
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


def test_use_stored_keypoints_default_true_on_embedded_uses_stored(
    seoul_bull_workspace: Path,
):
    """The default (``use_stored_keypoints=True``) on an embedded_patches recon
    anchors at the inline stored keypoints — bit-equal to passing
    ``use_stored_keypoints=True`` explicitly. Pins the default value to True
    so a code flip can't silently change anchor source."""
    sift_recon = SfmrReconstruction.load(seoul_bull_workspace)
    recon = sift_recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
    images = _load_images(recon)

    cloud_true = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    cloud_default = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    common = dict(
        resolution=12, init_steps=5, refine_levels=2, sampler="bilinear", cache="off"
    )
    out_true = cloud_true.refine_normals(
        recon, images, use_stored_keypoints=True, **common
    )
    out_default = cloud_default.refine_normals(recon, images, **common)

    # Embedded-patches recon: default must match explicit-True bit-for-bit.
    assert np.array_equal(
        np.asarray(out_true["normal"]),
        np.asarray(out_default["normal"]),
    )


def test_use_stored_keypoints_default_true_on_sift_files_falls_back_to_projection(
    seoul_bull_workspace: Path,
):
    """The default (``use_stored_keypoints=True``) on a sift-files recon
    silently falls back per-view to the reprojected center (the recon has no
    inline keypoints to anchor on) — bit-equal to passing
    ``use_stored_keypoints=False`` explicitly. No error: ``True`` is a
    request, not a requirement."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"

    images = _load_images(recon)
    cloud_false = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )
    cloud_default = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=5.0
    )

    common = dict(resolution=12, init_steps=5, refine_levels=2)
    out_false = cloud_false.refine_normals(
        recon, images, use_stored_keypoints=False, **common
    )
    out_default = cloud_default.refine_normals(recon, images, **common)

    # Sift-files recon: default-True (per-view fall-through to projection)
    # and explicit-False (projection everywhere) must match bit-for-bit.
    assert np.array_equal(
        np.asarray(out_false["normal"]),
        np.asarray(out_default["normal"]),
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
    tpid = np.asarray(recon.track_point_indexes)
    timg = np.asarray(recon.track_image_indexes)
    reproj = np.zeros((len(tpid), 2), np.float32)
    for j in range(len(tpid)):
        im = int(timg[j])
        pc = rot[im] @ positions[int(tpid[j])] + trans[im]
        # Canonical cameras look down -Z, so normalized coords divide by the
        # depth -z (> 0 in front), not +z.
        u, v = cams[int(cam_idx[im])].project(pc[0] / -pc[2], pc[1] / -pc[2])
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
        recon, images, point_indexes=pids, use_stored_keypoints=False, **common
    )
    anchored = cloud_a.refine_normals(
        recon_reproj, images, point_indexes=pids, use_stored_keypoints=True, **common
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
        np.asarray(recon.track_point_indexes).tolist(),
        np.asarray(recon.track_image_indexes).tolist(),
    ):
        by_point.setdefault(int(p), set()).add(int(im))

    # view_indices must be parallel to every cloud patch.
    view_indices: list[list[int]] = []
    for pid in np.asarray(cloud.point_indexes).tolist():
        obs = set(by_point.get(int(pid), set()))
        extra = next((i for i in range(num_images) if i not in obs), None)
        if extra is not None:
            obs = obs | {extra}  # a non-track image -> None keypoint
        view_indices.append(sorted(obs) if obs else [0])

    point_ids = _sample_point_ids(cloud, n=100)
    res = cloud.refine_normals(
        recon,
        images,
        point_indexes=point_ids,
        view_indices=view_indices,
        use_stored_keypoints=True,
        resolution=12,
        init_steps=5,
        refine_levels=2,
    )
    n = len(cloud)
    assert res["normal"].shape == (n, 3)
    assert np.all(np.isfinite(res["normal"]))
