# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the epipolar geometry helpers and intrinsic matrix construction."""

import numpy as np
import pycolmap

from sfmtool.camera.cameras import get_intrinsic_matrix
from sfmtool.feature_match._geometry import (
    check_rectification_safe,
    compute_epipole,
    get_essential_matrix,
    get_fundamental_matrix,
)


class TestEssentialMatrix:
    def test_identity_cameras(self):
        R = np.eye(3)
        t = np.zeros(3)
        E = get_essential_matrix(R, t, R, t)
        np.testing.assert_allclose(E, np.zeros((3, 3)), atol=1e-10)

    def test_lateral_motion(self):
        R = np.eye(3)
        t1 = np.zeros(3)
        t2 = np.array([1.0, 0.0, 0.0])
        E = get_essential_matrix(R, t1, R, t2)
        # E should have rank 2
        _, S, _ = np.linalg.svd(E)
        assert S[2] < 1e-10
        assert S[0] > 1e-10

    def test_epipolar_constraint(self):
        """x2^T E x1 = 0 for corresponding points."""
        R1 = np.eye(3)
        t1 = np.zeros(3)
        R2 = np.eye(3)
        t2 = np.array([1.0, 0.0, 0.0])
        E = get_essential_matrix(R1, t1, R2, t2)

        # A 3D point and its projections in normalized coords
        P = np.array([5.0, 3.0, 10.0])
        x1 = R1 @ P + t1
        x2 = R2 @ P + t2
        x1 /= x1[2]
        x2 /= x2[2]

        assert abs(x2 @ E @ x1) < 1e-10


class TestFundamentalMatrix:
    def test_rank_2(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t1 = np.zeros(3)
        t2 = np.array([1.0, 0.0, 0.0])
        F = get_fundamental_matrix(K, R, t1, K, R, t2)
        _, S, _ = np.linalg.svd(F)
        assert S[2] < 1e-10

    def test_epipolar_constraint_pixels(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        R1, R2 = np.eye(3), np.eye(3)
        t1 = np.zeros(3)
        t2 = np.array([1.0, 0.0, 0.0])
        F = get_fundamental_matrix(K, R1, t1, K, R2, t2)

        P = np.array([5.0, 3.0, 10.0])
        x1 = K @ (R1 @ P + t1)
        x2 = K @ (R2 @ P + t2)
        x1 /= x1[2]
        x2 /= x2[2]

        assert abs(x2 @ F @ x1) < 1e-8


class TestEpipole:
    def test_at_infinity(self):
        F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        epipole, is_at_inf = compute_epipole(F)
        assert is_at_inf

    def test_finite_epipole(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        R1, R2 = np.eye(3), np.eye(3)
        t1 = np.zeros(3)
        t2 = np.array([0.0, 0.0, 1.0])  # Forward motion
        F = get_fundamental_matrix(K, R1, t1, K, R2, t2)
        epipole, is_at_inf = compute_epipole(F)
        assert not is_at_inf


class TestRectificationSafe:
    def test_lateral_motion_safe(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t1 = np.zeros(3)
        t2 = np.array([1.0, 0.0, 0.0])
        assert check_rectification_safe(K, R, t1, K, R, t2, 640, 480)

    def test_forward_motion_unsafe(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t1 = np.zeros(3)
        t2 = np.array([0.0, 0.0, 1.0])
        assert not check_rectification_safe(K, R, t1, K, R, t2, 640, 480)


# ===== Intrinsic Matrix Tests =====


class TestGetIntrinsicMatrix:
    def test_pycolmap_pinhole(self):
        cam = pycolmap.Camera(
            model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240]
        )
        K = get_intrinsic_matrix(cam)
        assert K.shape == (3, 3)
        assert K[0, 0] == 500
        assert K[1, 1] == 500
        assert K[0, 2] == 320
        assert K[1, 2] == 240

    def test_pycolmap_simple_radial(self):
        cam = pycolmap.Camera(
            model="SIMPLE_RADIAL", width=640, height=480, params=[500, 320, 240, 0.1]
        )
        K = get_intrinsic_matrix(cam)
        assert K[0, 0] == 500
        assert K[1, 1] == 500  # Same as fx for SIMPLE_RADIAL
