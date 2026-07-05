# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""``classify_points_at_infinity`` — the single-viewpoint branch.

When a track's observing cameras all sit at essentially one optical centre — a
camera panning in place, or a solver that collapsed a run of frames — there is
no camera motion to give a depth cue, so a finite point looks just like an
infinite one and its triangulated position is meaningless (often landing right
on the cameras). The classifier stores such a point as ``w = 0`` with a
direction recovered from its keypoints, which also unblocks FeatureSize patch
sizing (``to_embedded_patches``), whose ``σ·d/f`` world size vanishes at zero
viewing distance ``d``.

This needs on-disk ``.sift`` files (the bearing is unprojected from the stored
keypoints), so it lives on the Python side over the ``seoul_bull_workspace``
fixture rather than as a Rust unit test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool._sfmtool.geometry import RigidTransform
from sfmtool.sift.file import get_sift_path_from_recon, read_sift_partial


def _world_to_camera_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """The world-to-camera rotation ``R`` for a stored WXYZ quaternion."""
    return np.asarray(
        RigidTransform.from_wxyz_translation(
            quaternion_wxyz.tolist(), [0.0, 0.0, 0.0]
        ).to_rotation_matrix(),
        dtype=np.float64,
    )


def _expected_keypoint_bearing(recon: SfmrReconstruction, point_idx: int) -> np.ndarray:
    """The mean world bearing of a point's observation keypoints, recomputed
    independently of the Rust classifier (unproject each ``.sift`` detection,
    rotate camera→world, average, normalise)."""
    tp = np.asarray(recon.track_point_indexes)
    ti = np.asarray(recon.track_image_indexes)
    tf = np.asarray(recon.track_feature_indexes)
    names = recon.image_names
    cams = recon.cameras
    cam_idx = np.asarray(recon.camera_indexes)
    quats = np.asarray(recon.quaternions_wxyz, dtype=np.float64)

    acc = np.zeros(3)
    for j in np.nonzero(tp == point_idx)[0]:
        imi, fi = int(ti[j]), int(tf[j])
        data = read_sift_partial(
            str(get_sift_path_from_recon(recon, names[imi])), fi + 1
        )
        u, v = np.asarray(data["positions_xy"])[fi].astype(np.float64)
        ray_cam = np.asarray(cams[int(cam_idx[imi])].pixel_to_ray(u, v), np.float64)
        ray_world = _world_to_camera_rotation(quats[imi]).T @ ray_cam
        acc += ray_world / np.linalg.norm(ray_world)
    return acc / np.linalg.norm(acc)


def _collapse_observers_onto_center(
    recon: SfmrReconstruction, point_idx: int
) -> tuple[SfmrReconstruction, np.ndarray]:
    """Return a copy of ``recon`` whose ``point_idx``'s observing cameras all
    share one world centre ``C`` (rotations kept) and whose point sits at ``C`` —
    a zero-baseline collapse. Non-observing cameras keep their spread so the
    camera-cloud extent (the classifier's scene scale) stays large.
    """
    ti = np.asarray(recon.track_image_indexes)
    tp = np.asarray(recon.track_point_indexes)
    observers = np.unique(ti[tp == point_idx])

    quats = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    translations = np.asarray(recon.translations, dtype=np.float64).copy()
    # Shared centre: the first observer's current camera centre.
    r0 = _world_to_camera_rotation(quats[observers[0]])
    center = -r0.T @ translations[observers[0]]
    # For each observer, choose t = -R·C so its centre is exactly C.
    for imi in observers:
        r = _world_to_camera_rotation(quats[imi])
        translations[imi] = -r @ center

    positions = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    positions[point_idx] = np.append(center, 1.0)  # finite, at the shared centre

    collapsed = recon.clone_with_changes(translations=translations, positions=positions)
    return collapsed, center


def _pick_point_with_observer_count(recon: SfmrReconstruction, count: int) -> int:
    """A finite point observed by exactly ``count`` images (so collapsing its
    observers leaves the rest of the camera cloud spread)."""
    counts = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=recon.point_count
    )
    finite = ~np.asarray(recon.point_is_at_infinity)
    candidates = np.nonzero((counts == count) & finite)[0]
    assert candidates.size, f"no finite point observed by exactly {count} images"
    return int(candidates[0])


def test_coincident_cameras_classified_as_infinity(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    pidx = _pick_point_with_observer_count(recon, 2)
    collapsed, _center = _collapse_observers_onto_center(recon, pidx)

    # The point is finite before classification (the solve placed it at a
    # position), but its cameras all sit at one optical centre.
    assert not bool(np.asarray(collapsed.point_is_at_infinity)[pidx])

    classified = collapsed.classify_points_at_infinity(1.0)
    at_inf = np.asarray(classified.point_is_at_infinity)

    # It is demoted to a point at infinity, with a unit bearing that matches the
    # independent keypoint-unprojection mean.
    assert bool(at_inf[pidx])
    xyzw = np.asarray(classified.positions_xyzw)[pidx]
    assert xyzw[3] == 0.0
    np.testing.assert_allclose(np.linalg.norm(xyzw[:3]), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        xyzw[:3], _expected_keypoint_bearing(recon, pidx), atol=1e-6
    )


def test_coincident_cameras_unblock_feature_size_embedding(
    seoul_bull_workspace: Path,
):
    """A point coincident with its cameras breaks FeatureSize sizing (``d≈0``);
    classifying it to infinity fixes ``to_embedded_patches``."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    pidx = _pick_point_with_observer_count(recon, 2)
    collapsed, _center = _collapse_observers_onto_center(recon, pidx)

    # Before: the coincident point has no readable world size → the sizing error.
    with pytest.raises(Exception, match="coincident with the camera centre"):
        collapsed.to_embedded_patches(
            normal="mean_viewing", extent="feature_size", extent_value=2.5
        )

    # After classifying it to infinity, the angular (distance-free) size applies
    # and the conversion succeeds.
    classified = collapsed.classify_points_at_infinity(1.0)
    embedded = classified.to_embedded_patches(
        normal="mean_viewing", extent="feature_size", extent_value=2.5
    )
    assert embedded.point_count == classified.point_count
    assert bool(np.asarray(embedded.point_is_at_infinity)[pidx])


def test_spread_cameras_are_not_demoted(seoul_bull_workspace: Path):
    """The baseline gate is tight: an ordinary well-triangulated point (real
    baseline) is left finite — the branch only fires on a near-perfect collapse."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    before_inf = int(np.count_nonzero(recon.point_is_at_infinity))
    classified = recon.classify_points_at_infinity(1.0)
    after_inf = int(np.count_nonzero(classified.point_is_at_infinity))
    # A forward-facing capture has few/no unconstrained points and certainly no
    # zero-baseline collapse, so classification does not sweep the scene to
    # infinity.
    assert after_inf < recon.point_count * 0.5
    assert after_inf >= before_inf
