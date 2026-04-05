# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the inspect command modules."""

import numpy as np
import pytest

from sfmtool._histogram_utils import estimate_z_from_histogram
from sfmtool._inspect_images import (
    _compute_camera_centers,
    _compute_rotation_angle,
    _slerp_halfway,
    _analyze_motion_path,
)
from sfmtool._sfmtool import RotQuaternion


class TestEstimateZFromHistogram:
    """Tests for the consolidated estimate_z_from_histogram function."""

    def test_uniform_histogram_50th_percentile(self):
        counts = np.ones(10, dtype=np.float64)
        result = estimate_z_from_histogram(counts, 0.0, 10.0, 50.0)
        assert abs(result - 5.0) < 0.5

    def test_all_in_first_bin(self):
        counts = np.zeros(10, dtype=np.float64)
        counts[0] = 100
        result = estimate_z_from_histogram(counts, 0.0, 10.0, 50.0)
        assert result < 1.0

    def test_all_in_last_bin(self):
        counts = np.zeros(10, dtype=np.float64)
        counts[-1] = 100
        result = estimate_z_from_histogram(counts, 0.0, 10.0, 50.0)
        assert result > 9.0

    def test_empty_histogram(self):
        counts = np.zeros(10, dtype=np.float64)
        result = estimate_z_from_histogram(counts, 0.0, 10.0, 50.0)
        assert result == 5.0

    def test_invalid_percentile_low(self):
        counts = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="Percentile must be between"):
            estimate_z_from_histogram(counts, 0.0, 10.0, -1.0)

    def test_invalid_percentile_high(self):
        counts = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="Percentile must be between"):
            estimate_z_from_histogram(counts, 0.0, 10.0, 101.0)

    def test_0th_percentile(self):
        counts = np.ones(10, dtype=np.float64)
        result = estimate_z_from_histogram(counts, 0.0, 10.0, 0.0)
        assert result == 0.0

    def test_100th_percentile(self):
        counts = np.ones(10, dtype=np.float64)
        result = estimate_z_from_histogram(counts, 0.0, 10.0, 100.0)
        assert result == 10.0

    def test_bimodal_histogram(self):
        counts = np.zeros(20, dtype=np.float64)
        counts[2:5] = 50
        counts[15:18] = 50
        result = estimate_z_from_histogram(counts, 0.0, 20.0, 50.0)
        assert 4.0 < result < 16.0


class TestComputeCameraCenters:
    def test_identity_rotation(self):
        quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])
        translations = np.array([[1.0, 2.0, 3.0]])
        centers = _compute_camera_centers(quaternions, translations)
        np.testing.assert_allclose(centers[0], [-1.0, -2.0, -3.0])

    def test_zero_translation(self):
        quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])
        translations = np.array([[0.0, 0.0, 0.0]])
        centers = _compute_camera_centers(quaternions, translations)
        np.testing.assert_allclose(centers[0], [0.0, 0.0, 0.0])


class TestComputeRotationAngle:
    def test_identity_rotation(self):
        q = RotQuaternion(1.0, 0.0, 0.0, 0.0)
        assert _compute_rotation_angle(q, q) == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_rotation(self):
        q1 = RotQuaternion(1.0, 0.0, 0.0, 0.0)
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        q2 = RotQuaternion(c, 0.0, 0.0, s)
        angle = _compute_rotation_angle(q1, q2)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_180_degree_rotation(self):
        q1 = RotQuaternion(1.0, 0.0, 0.0, 0.0)
        q2 = RotQuaternion(0.0, 0.0, 0.0, 1.0)
        angle = _compute_rotation_angle(q1, q2)
        assert angle == pytest.approx(180.0, abs=0.1)


class TestSlerpHalfway:
    def test_identity_inputs(self):
        q = RotQuaternion(1.0, 0.0, 0.0, 0.0)
        result = _slerp_halfway(q, q)
        assert result.w == pytest.approx(1.0, abs=1e-6)

    def test_halfway_is_equidistant(self):
        q1 = RotQuaternion(1.0, 0.0, 0.0, 0.0)
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        q2 = RotQuaternion(c, 0.0, 0.0, s)

        mid = _slerp_halfway(q1, q2)
        angle1 = _compute_rotation_angle(q1, mid)
        angle2 = _compute_rotation_angle(mid, q2)
        assert angle1 == pytest.approx(angle2, abs=0.5)


class TestAnalyzeMotionPath:
    def test_basic_motion_analysis(self):
        class MockRecon:
            image_count = 3
            image_names = ["img0.jpg", "img1.jpg", "img2.jpg"]

        recon = MockRecon()
        quaternions = np.array([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float64)
        camera_centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        result = _analyze_motion_path(recon, camera_centers, quaternions)
        (
            succ_trans,
            succ_rot,
            interp_trans,
            interp_rot,
            discs,
            trans_thresh,
            rot_thresh,
            sigma,
        ) = result

        np.testing.assert_allclose(succ_trans, [1.0, 1.0])
        np.testing.assert_allclose(succ_rot, [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(interp_trans, [0.0], atol=1e-6)
        assert len(discs) == 0
