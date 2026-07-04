# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the coordinate-convention conversion bindings and the
`sfmtool.colmap.convention` helper wrappers (COLMAP <-> canonical)."""

import numpy as np
import pytest

import sfmtool._sfmtool.geometry as geometry
from sfmtool.colmap import convention

# Fixed non-trivial batch of world-to-camera poses (WXYZ quats, translations).
_QUATS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [
            0.9128709291752769,
            0.18257418583505536,
            0.2738612787525831,
            0.2738612787525831,
        ],
        [0.2, -0.5, 0.7, 0.3],  # normalized by the bindings
    ]
)
_TRANS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.3, -1.2, 2.5],
        [10.0, -20.0, 30.0],
    ]
)

_SQRT_HALF = np.sqrt(0.5)


def test_convention_functions_registered():
    for name in (
        "poses_colmap_to_canonical",
        "poses_canonical_to_colmap",
        "relative_poses_conjugate_s",
        "world_rotate_w",
        "world_rotate_w_inverse",
    ):
        assert callable(getattr(geometry, name)), f"missing binding: {name}"


def test_pose_round_trip_batch():
    q_can, t_can = geometry.poses_colmap_to_canonical(_QUATS, _TRANS)
    q_back, t_back = geometry.poses_canonical_to_colmap(q_can, t_can)

    # Quaternions have a sign ambiguity: compare either q or -q.
    normalized = _QUATS / np.linalg.norm(_QUATS, axis=1, keepdims=True)
    for got, expected in zip(q_back, normalized):
        assert np.allclose(got, expected, atol=1e-12) or np.allclose(
            got, -expected, atol=1e-12
        )
    np.testing.assert_allclose(t_back, _TRANS, atol=1e-12)


def test_identity_colmap_pose_maps_to_w_rotation():
    # R' = S.I.W^T = W, a -90 degree rotation about X:
    # quaternion wxyz = (cos 45, -sin 45, 0, 0).
    q_can, t_can = geometry.poses_colmap_to_canonical(
        np.array([[1.0, 0.0, 0.0, 0.0]]), np.zeros((1, 3))
    )
    expected = np.array([_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0])
    assert np.allclose(q_can[0], expected, atol=1e-12) or np.allclose(
        q_can[0], -expected, atol=1e-12
    )
    np.testing.assert_allclose(t_can[0], np.zeros(3), atol=1e-12)


def test_translation_flips_by_s():
    q_can, t_can = geometry.poses_colmap_to_canonical(
        np.array([[1.0, 0.0, 0.0, 0.0]]), np.array([[1.0, 2.0, 3.0]])
    )
    np.testing.assert_allclose(t_can[0], [1.0, -2.0, -3.0], atol=1e-12)


def test_relative_pose_conjugation_is_involutive():
    q1, t1 = geometry.relative_poses_conjugate_s(_QUATS, _TRANS)
    q2, t2 = geometry.relative_poses_conjugate_s(q1, t1)
    normalized = _QUATS / np.linalg.norm(_QUATS, axis=1, keepdims=True)
    for got, expected in zip(q2, normalized):
        assert np.allclose(got, expected, atol=1e-12) or np.allclose(
            got, -expected, atol=1e-12
        )
    np.testing.assert_allclose(t2, _TRANS, atol=1e-12)


def test_world_rotate_w_axis_mapping():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],  # COLMAP -Y "up" becomes +Z up
        ]
    )
    np.testing.assert_allclose(geometry.world_rotate_w(vectors), expected, atol=1e-12)

    round_trip = geometry.world_rotate_w_inverse(geometry.world_rotate_w(vectors))
    np.testing.assert_allclose(round_trip, vectors, atol=1e-12)


def test_shape_validation():
    with pytest.raises(ValueError):
        geometry.poses_colmap_to_canonical(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        geometry.poses_colmap_to_canonical(np.zeros((2, 4)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        geometry.world_rotate_w(np.zeros((2, 4)))


# ── Python helper wrappers (sfmtool.colmap.convention) ────────────────────


def test_helper_single_pose_shapes():
    q, t = convention.pose_colmap_to_canonical([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert q.shape == (4,) and t.shape == (3,)
    np.testing.assert_allclose(t, [1.0, -2.0, -3.0], atol=1e-12)

    q_back, t_back = convention.pose_canonical_to_colmap(q, t)
    assert np.allclose(np.abs(q_back), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(t_back, [1.0, 2.0, 3.0], atol=1e-12)


def test_helper_batch_matches_bindings():
    q, t = convention.pose_colmap_to_canonical(_QUATS, _TRANS)
    q_expected, t_expected = geometry.poses_colmap_to_canonical(_QUATS, _TRANS)
    np.testing.assert_allclose(q, q_expected, atol=1e-15)
    np.testing.assert_allclose(t, t_expected, atol=1e-15)

    q_rel, t_rel = convention.relative_pose_conjugate_s(_QUATS[1], _TRANS[1])
    assert q_rel.shape == (4,) and t_rel.shape == (3,)


def test_helper_mixed_single_batch_rejected():
    with pytest.raises(ValueError):
        convention.pose_colmap_to_canonical(_QUATS, _TRANS[0])


def test_helper_points_xyzw_carry_w():
    points = np.array(
        [
            [1.0, 2.0, 3.0, 1.0],  # finite point
            [0.0, 1.0, 0.0, 0.0],  # infinity direction
        ]
    )
    rotated = convention.points_xyzw_rotate_w(points)
    np.testing.assert_allclose(
        rotated,
        [
            [1.0, 3.0, -2.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        atol=1e-12,
    )
    round_trip = convention.points_xyzw_rotate_w_inverse(rotated)
    np.testing.assert_allclose(round_trip, points, atol=1e-12)

    single = convention.points_xyzw_rotate_w(points[0])
    assert single.shape == (4,)
    np.testing.assert_allclose(single, [1.0, 3.0, -2.0, 1.0], atol=1e-12)


def test_helper_world_rotate_single_vector():
    v = convention.world_rotate_w([0.0, 1.0, 0.0])
    assert v.shape == (3,)
    np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(
        convention.world_rotate_w_inverse(v), [0.0, 1.0, 0.0], atol=1e-12
    )


def test_helper_rigid3d_round_trip():
    pycolmap = pytest.importorskip("pycolmap")

    # A non-trivial COLMAP world-to-camera pose (exactly unit-norm quaternion:
    # pycolmap stores the quaternion verbatim, without normalizing).
    quat_xyzw = np.array([1.0, 1.5, 1.5, 5.0])
    quat_xyzw /= np.linalg.norm(quat_xyzw)
    translation = np.array([0.3, -1.2, 2.5])
    rigid = pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(quat_xyzw),
        translation=translation,
    )

    q_can, t_can = convention.rigid3d_colmap_to_canonical(rigid)
    assert q_can.shape == (4,) and t_can.shape == (3,)
    np.testing.assert_allclose(t_can, [0.3, 1.2, -2.5], atol=1e-12)

    rigid_back = convention.canonical_pose_to_rigid3d(q_can, t_can)
    back_xyzw = np.asarray(rigid_back.rotation.quat)
    assert np.allclose(back_xyzw, quat_xyzw, atol=1e-7) or np.allclose(
        back_xyzw, -quat_xyzw, atol=1e-7
    )
    np.testing.assert_allclose(rigid_back.translation, translation, atol=1e-12)
