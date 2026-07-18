# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for ``PatchCloud.select_views`` against real reconstructions.

Builds a patch cloud from a solved reconstruction and runs photometric
patch-view selection over its real ``.sift``-derived patches and source images —
the multi-view rendering path the Rust unit tests can't exercise without on-disk
images. See ``specs/core/patch-view-selection.md``.

``seoul_bull`` is the convex case: the selected view set must be a superset of
each point's track. ``kerry_park`` (a back-to-back fisheye rig) is the
non-convex / cluttered stress case, exercised for shape and sanity.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.patches import PatchCloud


def _load_images(recon) -> list[np.ndarray]:
    import cv2  # heavy module, only needed by this integration test

    ws = recon.workspace_dir
    images = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        images.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return images


def _sample_point_ids(cloud, n: int = 300, seed: int = 0) -> list[int]:
    """A deterministic point-id subset, to keep the per-point search fast."""
    ids = np.asarray(cloud.point_indexes)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False)).tolist()


def _track_views(recon) -> dict[int, set[int]]:
    """Map each 3D-point id to the set of image indices in its track."""
    point_ids = np.asarray(recon.track_point_indexes)
    image_idxs = np.asarray(recon.track_image_indexes)
    tracks: dict[int, set[int]] = {}
    for pid, image_idx in zip(point_ids.tolist(), image_idxs.tolist()):
        tracks.setdefault(int(pid), set()).add(int(image_idx))
    return tracks


def _rotation_matrices(recon) -> np.ndarray:
    """Per-image world-to-camera rotation matrices (Mx3x3) from wxyz quaternions."""
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


def _camera_frame_point(recon, point_xyz: np.ndarray, image_idx: int) -> np.ndarray:
    """The 3D point in image `image_idx`'s camera frame: X_cam = R·X + t."""
    R = _rotation_matrices(recon)[image_idx]
    t = np.asarray(recon.translations, dtype=np.float64)[image_idx]
    return R @ point_xyz + t


def _geometric_candidate_set(recon, pid: int, patch, point_xyz: np.ndarray) -> set[int]:
    """Every image that *geometrically* sees the point: in front of the camera
    (positive camera-frame depth), front-facing patch, and projecting in-frame.
    This is (approximately) the set view selection vets photometrically; the test
    checks that at least one such candidate beyond the always-admitted track is
    rejected. It is only an approximation of the Rust gate — this Python recompute
    can disagree at sub-pixel frame/cheirality boundaries — so it must not be used
    for a strict per-point subset assertion."""
    from sfmtool._sfmtool.geometry import RigidTransform

    cams = recon.cameras
    cam_idx = np.asarray(recon.camera_indexes)
    R = _rotation_matrices(recon)
    t = np.asarray(recon.translations, dtype=np.float64)
    quats = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    out: set[int] = set()
    for i in range(recon.image_count):
        x_cam = R[i] @ point_xyz + t[i]
        # Canonical cameras look down -Z: in front means camera-space z < 0, and
        # the normalized image coords divide by the depth -z (> 0 in front).
        if x_cam[2] >= 0:
            continue  # behind camera
        pose = RigidTransform.from_wxyz_translation(quats[i].tolist(), t[i].tolist())
        if not patch.is_front_facing(pose):
            continue
        cam = cams[int(cam_idx[i])]
        u, v = cam.project(x_cam[0] / -x_cam[2], x_cam[1] / -x_cam[2])
        if 0 <= u < cam.width and 0 <= v < cam.height:
            out.add(i)
    return out


def test_select_views_superset_of_track_on_convex_dataset(seoul_bull_workspace: Path):
    """On the convex seoul_bull case the selected set contains the whole track."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0

    sample = _sample_point_ids(cloud)
    results = cloud.select_views(
        recon,
        images,
        point_indexes=sample,
        resolution=12,
    )

    # Only the sampled patches are returned.
    assert len(results) == len(sample)
    returned_pids = {int(r["point_index"]) for r in results}
    assert returned_pids == set(sample)

    tracks = _track_views(recon)
    expanded_any = False
    for r in results:
        pid = int(r["point_index"])
        admitted = np.asarray(r["admitted"], dtype=np.int64)
        scores = np.asarray(r["scores"], dtype=np.float64)

        # admitted / scores are parallel.
        assert admitted.shape == scores.shape
        # No duplicates and all in range.
        assert len(set(admitted.tolist())) == len(admitted)
        assert np.all(admitted >= 0) and np.all(admitted < len(images))

        track = tracks[pid]
        # The selected set is a superset of the point's track (track always kept).
        assert track.issubset(set(admitted.tolist())), (
            f"point {pid}: admitted {admitted.tolist()} missing track {sorted(track)}"
        )
        if len(admitted) > len(track):
            expanded_any = True

    # The point of the algorithm: at least one point gained an extra view.
    assert expanded_any, "selection never expanded any track on the convex dataset"


def test_select_views_self_agreement_and_threshold(seoul_bull_workspace: Path):
    """Where a reference was built, admitted candidates clear the relative bar."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    tracks = _track_views(recon)

    min_rel = 0.7
    results = cloud.select_views(
        recon,
        images,
        point_indexes=_sample_point_ids(cloud),
        resolution=12,
        min_relative_zncc=min_rel,
        min_self_agreement=0.3,
    )

    checked = 0
    for r in results:
        sa = float(r["self_agreement"])
        if not np.isfinite(sa):
            continue  # no reference: track admitted verbatim
        pid = int(r["point_index"])
        track = tracks[pid]
        admitted = np.asarray(r["admitted"], dtype=np.int64)
        scores = np.asarray(r["scores"], dtype=np.float64)
        bar = min_rel * max(sa, 0.3)
        for view_idx, score in zip(admitted.tolist(), scores.tolist()):
            if view_idx in track:
                continue  # track views are unconditional
            # A vetted (non-track) candidate must have cleared the bar.
            assert np.isfinite(score)
            assert score >= bar - 1e-9, f"point {pid}: {score} < bar {bar}"
            checked += 1
    # The dataset should produce at least some vetted candidates to check.
    assert checked > 0


def test_select_views_runs_on_nonconvex_fisheye_rig(kerry_park_workspace: Path):
    """Shape / sanity on the non-convex kerry_park fisheye-rig stress case."""
    recon = SfmrReconstruction.load(kerry_park_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0

    sample = _sample_point_ids(cloud, n=150)
    results = cloud.select_views(recon, images, point_indexes=sample, resolution=12)

    assert len(results) == len(sample)
    tracks = _track_views(recon)
    for r in results:
        pid = int(r["point_index"])
        admitted = np.asarray(r["admitted"], dtype=np.int64)
        scores = np.asarray(r["scores"], dtype=np.float64)
        assert admitted.shape == scores.shape
        assert len(set(admitted.tolist())) == len(admitted)
        assert np.all(admitted >= 0) and np.all(admitted < len(images))
        # Track is always kept, even on the non-convex case.
        assert tracks[pid].issubset(set(admitted.tolist()))


def test_select_views_rejects_some_geometrically_visible_candidate(
    seoul_bull_workspace: Path,
):
    """Selection is not a rubber stamp: across the sample at least one
    geometrically-visible candidate (beyond the always-admitted track) is vetted
    away.

    We deliberately do NOT assert ``admitted ⊆ visible`` per point. Track views
    are force-admitted regardless of the geometric gate, and this Python recompute
    of "geometrically visible" disagrees with the Rust gate at sub-pixel
    frame/cheirality boundaries (platform float differences) — so a per-point
    subset check is flaky (it failed on macOS for a track view sitting right on a
    frame edge). The load-bearing invariant is that some visible non-track
    candidate is rejected; admitted views are otherwise covered by the cheirality
    and threshold tests."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    positions = np.asarray(recon.positions, dtype=np.float64)
    pid_to_patch_idx = {
        int(p): i for i, p in enumerate(np.asarray(cloud.point_indexes))
    }
    tracks = _track_views(recon)

    sample = _sample_point_ids(cloud)
    results = cloud.select_views(recon, images, point_indexes=sample, resolution=12)

    rejected_any = False
    for r in results:
        pid = int(r["point_index"])
        admitted = set(np.asarray(r["admitted"], dtype=np.int64).tolist())
        patch = cloud[pid_to_patch_idx[pid]]
        cand = _geometric_candidate_set(recon, pid, patch, positions[pid])
        # A geometrically-visible candidate beyond the always-admitted track that
        # was not admitted == the photometric vetting did real work.
        if (cand - tracks[pid]) - admitted:
            rejected_any = True
            break
    assert rejected_any, (
        "selection never rejected a geometrically-visible candidate — "
        "the photometric vetting is not doing anything"
    )


def test_select_views_admitted_points_are_in_front_of_camera(
    kerry_park_workspace: Path,
):
    """B1 cheirality: no admitted view may have the point behind its camera, even
    on the wide-fisheye kerry_park rig where a behind-camera point can project
    in-frame.

    This is the *finite*-point cheirality invariant (`R·X + t`.z < 0 — canonical
    cameras look down -Z), so build a finite-only cloud; a point at infinity has a
    different in-front test (`R·d`.z, no translation) and is covered by the core
    tests.
    """
    recon = SfmrReconstruction.load(kerry_park_workspace)
    images = _load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0, exclude_points_at_infinity=True
    )
    positions = np.asarray(recon.positions, dtype=np.float64)

    sample = _sample_point_ids(cloud, n=150)
    results = cloud.select_views(recon, images, point_indexes=sample, resolution=12)

    for r in results:
        pid = int(r["point_index"])
        admitted = np.asarray(r["admitted"], dtype=np.int64)
        for image_idx in admitted.tolist():
            x_cam = _camera_frame_point(recon, positions[pid], image_idx)
            assert x_cam[2] < 0, (
                f"point {pid} admitted into image {image_idx} but is behind that "
                f"camera (canonical depth -z = {-x_cam[2]})"
            )


def test_select_views_infinity_admitted_are_in_front(kerry_park_workspace: Path):
    """w == 0 cheirality: an admitted view of a point at infinity must look toward
    its direction — `(R·d).z < 0` (canonical cameras look down -Z), with no
    translation (every ray to the point is parallel to `d`). Complements the
    finite-point B1 test above, on the same wide-fisheye rig (kerry_park carries
    points at infinity)."""
    recon = SfmrReconstruction.load(kerry_park_workspace)
    images = _load_images(recon)
    # Default includes points at infinity.
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    is_inf = np.asarray(recon.point_is_at_infinity)
    positions = np.asarray(recon.positions, dtype=np.float64)
    rot = _rotation_matrices(recon)

    inf_ids = [int(p) for p in np.asarray(cloud.point_indexes) if is_inf[int(p)]]
    assert inf_ids, "kerry_park should carry points at infinity"
    results = cloud.select_views(recon, images, point_indexes=inf_ids, resolution=12)

    checked = 0
    for r in results:
        pid = int(r["point_index"])
        d = positions[pid]  # unit direction for a w == 0 point
        for image_idx in np.asarray(r["admitted"], dtype=np.int64).tolist():
            z = float((rot[image_idx] @ d)[2])
            assert z < 0, (
                f"infinity point {pid} admitted into image {image_idx} but its "
                f"direction points away from the camera (R·d).z = {z}"
            )
            checked += 1
    assert checked > 0, "no admitted infinity views were checked"
