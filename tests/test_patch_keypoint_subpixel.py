# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for ``PatchCloud.refine_keypoints`` (the continuous ECC /
Gauss-Newton subpixel refiner) against a real reconstruction.

Builds a patch cloud from a solved reconstruction and refines the per-view
keypoints over the real ``.sift``-derived patches and source images — the
multi-view rendering + continuous solve the Rust unit tests exercise only
synthetically. See ``specs/core/keypoint-subpixel-refinement.md``.

The refiner is a *local* refiner that changes no view membership and is never
worse than the seed; the checks here are the binding's array contract, the
unchanged view set, and the never-worse-than-seed guarantee (compared against a
zero-step "seed only" run on the same data).
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


def _sample_point_ids(cloud, n: int = 200, seed: int = 0) -> list[int]:
    ids = np.asarray(cloud.point_ids)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False)).tolist()


def _rotation_matrices(recon) -> np.ndarray:
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w * w + x * x + y * y + z * z
    s = np.where(n > 0, 2.0 / n, 0.0)
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - s * (y * y + z * z)
    R[:, 0, 1] = s * (x * y - z * w)
    R[:, 0, 2] = s * (x * z + y * w)
    R[:, 1, 0] = s * (x * y + z * w)
    R[:, 1, 1] = 1 - s * (x * x + z * z)
    R[:, 1, 2] = s * (y * z - x * w)
    R[:, 2, 0] = s * (x * z - y * w)
    R[:, 2, 1] = s * (y * z + x * w)
    R[:, 2, 2] = 1 - s * (x * x + y * y)
    return R


def _project(recon, point_xyz: np.ndarray, image_idx: int):
    R = _rotation_matrices(recon)[image_idx]
    t = np.asarray(recon.translations, dtype=np.float64)[image_idx]
    x_cam = R @ point_xyz + t
    if x_cam[2] <= 0:
        return None
    cam = recon.cameras[int(np.asarray(recon.camera_indexes)[image_idx])]
    return np.asarray(cam.project(x_cam[0] / x_cam[2], x_cam[1] / x_cam[2]))


def test_refine_keypoints_array_contract_and_never_worse(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0

    sample = _sample_point_ids(cloud)
    view_sets = {
        int(r["point_id"]): np.asarray(r["admitted"]).tolist()
        for r in cloud.select_views(recon, images, point_ids=sample, resolution=12)
    }

    common = dict(
        recon=recon,
        images=images,
        view_sets=view_sets,
        point_ids=sample,
        resolution=12,
    )
    # Seed-only baseline (no GN steps) and the full continuous refine.
    seed = cloud.refine_keypoints(**common, max_gn_steps=0)
    refined = cloud.refine_keypoints(**common, max_gn_steps=10)

    assert {int(r["point_id"]) for r in refined} == set(sample)
    seed_by_pid = {int(r["point_id"]): r for r in seed}
    positions = np.asarray(recon.positions, dtype=np.float64)

    moved_any = False
    improved_any = False
    for r in refined:
        pid = int(r["point_id"])
        views = np.asarray(r["views"], dtype=np.int64)
        kpts = np.asarray(r["keypoints"], dtype=np.float64)
        offs = np.asarray(r["offsets_px"], dtype=np.float64)
        scores = np.asarray(r["scores"], dtype=np.float64)

        # Parallel arrays, no duplicates, indices in range.
        assert kpts.shape == (len(views), 2)
        assert offs.shape == (len(views),)
        assert scores.shape == (len(views),)
        assert len(set(views.tolist())) == len(views)
        assert np.all(views >= 0) and np.all(views < len(images))

        # View set is UNCHANGED versus the seed run (no membership change, same order).
        sviews = np.asarray(seed_by_pid[pid]["views"], dtype=np.int64)
        assert np.array_equal(views, sviews), (
            f"point {pid}: refine changed the view set/order"
        )

        # The reported offset is the keypoint's distance from the projection.
        for k, image_idx in enumerate(views.tolist()):
            proj = _project(recon, positions[pid], image_idx)
            assert proj is not None
            assert np.allclose(np.linalg.norm(kpts[k] - proj), offs[k], atol=1e-6)

        # Never worse than the seed: every finite refined score is >= its seed score.
        sscores = np.asarray(seed_by_pid[pid]["scores"], dtype=np.float64)
        both_finite = np.isfinite(scores) & np.isfinite(sscores)
        assert np.all(scores[both_finite] >= sscores[both_finite] - 1e-6), (
            f"point {pid}: a refined score dropped below the seed"
        )
        if np.any(scores[both_finite] > sscores[both_finite] + 1e-4):
            improved_any = True

        skpts = np.asarray(seed_by_pid[pid]["keypoints"], dtype=np.float64)
        if skpts.shape == kpts.shape and np.any(
            np.linalg.norm(kpts - skpts, axis=1) > 1e-3
        ):
            moved_any = True

    # The refiner must actually do something on real data.
    assert moved_any, "refinement never moved any keypoint"
    assert improved_any, "refinement never improved any view's ECC score"


def test_refine_keypoints_defaults_to_track(seoul_bull_workspace: Path):
    """With no view_sets, each point refines over its track; the view set is the
    track's views, in order, unchanged by the local refiner."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    point_ids = np.asarray(recon.track_point_ids)
    image_idxs = np.asarray(recon.track_image_indexes)
    tracks: dict[int, set[int]] = {}
    for pid, im in zip(point_ids.tolist(), image_idxs.tolist()):
        tracks.setdefault(int(pid), set()).add(int(im))

    sample = _sample_point_ids(cloud, n=120)
    results = cloud.refine_keypoints(recon, images, point_ids=sample, resolution=12)
    for r in results:
        pid = int(r["point_id"])
        views = set(np.asarray(r["views"], dtype=np.int64).tolist())
        assert views.issubset(tracks[pid])


def test_refine_keypoints_empty_view_set_yields_empty_arrays(
    seoul_bull_workspace: Path,
):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    pid = int(np.asarray(cloud.point_ids)[0])
    res = cloud.refine_keypoints(
        recon, images, view_sets={pid: []}, point_ids=[pid], resolution=12
    )
    assert len(res) == 1
    assert np.asarray(res[0]["views"]).shape == (0,)
    assert np.asarray(res[0]["keypoints"]).shape == (0, 2)
    assert np.asarray(res[0]["offsets_px"]).shape == (0,)
    assert np.asarray(res[0]["scores"]).shape == (0,)


def test_refine_keypoints_rejects_out_of_range_view_index(seoul_bull_workspace: Path):
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    pid = int(np.asarray(cloud.point_ids)[0])
    bad = {pid: [0, len(images)]}
    with pytest.raises(ValueError):
        cloud.refine_keypoints(
            recon, images, view_sets=bad, point_ids=[pid], resolution=12
        )
