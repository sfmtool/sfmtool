# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for sweep matching: the geometric filter config, the rectified and polar
sweeps, and ``match_image_pair``."""

import numpy as np
import pytest
import pycolmap

from sfmtool.feature_match import GeometricFilterConfig, match_image_pair
from sfmtool.colmap.convention import flip_camera_pose_matrix_s
from sfmtool._sfmtool.matching import (
    mutual_best_match_sweep_py,
    mutual_best_match_sweep_geometric_py,
    polar_mutual_best_match_py,
    polar_mutual_best_match_geometric_py,
)


class TestGeometricFilterConfig:
    def test_defaults(self):
        config = GeometricFilterConfig()
        assert config.enable_geometric_filtering is True
        assert config.max_angle_difference == 15.0

    def test_size_ratio_valid(self):
        config = GeometricFilterConfig(
            geometric_size_ratio_min=0.8, geometric_size_ratio_max=1.25
        )
        assert config.is_size_ratio_valid(1.0)
        assert config.is_size_ratio_valid(0.8)
        assert not config.is_size_ratio_valid(0.5)
        assert not config.is_size_ratio_valid(2.0)

    def test_angle_diff_valid(self):
        config = GeometricFilterConfig(max_angle_difference=15.0)
        assert config.is_angle_diff_valid(10.0)
        assert config.is_angle_diff_valid(15.0)
        assert not config.is_angle_diff_valid(20.0)


# ===== Rectified Sweep Matching Tests =====


def _make_geometric_params(n_features1, n_features2, forward=False):
    """Create synthetic geometric filtering parameters for testing."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = 500.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0

    pose1 = pycolmap.Rigid3d(
        pycolmap.Rotation3d([0, 0, 0, 1]),
        np.array([0.0, 0.0, 0.0]),
    )
    if forward:
        pose2 = pycolmap.Rigid3d(
            pycolmap.Rotation3d([0, 0, 0, 1]),
            np.array([0.0, 0.0, -1.0]),
        )
    else:
        pose2 = pycolmap.Rigid3d(
            pycolmap.Rotation3d([0, 0, 0, 1]),
            np.array([1.0, 0.0, 0.0]),
        )

    affine_shapes1 = np.tile(np.eye(2) * 5.0, (n_features1, 1, 1))
    affine_shapes2 = np.tile(np.eye(2) * 5.0, (n_features2, 1, 1))

    R1 = pose1.rotation.matrix()
    R2 = pose2.rotation.matrix()
    R_2d = (R2 @ R1.T)[:2, :2]

    config = GeometricFilterConfig()

    return affine_shapes1, affine_shapes2, K, K, pose1, pose2, R_2d, config


def _geometric_args(geometric):
    """Flatten `_make_geometric_params` output into the Rust matchers' form.

    The Rust geometric matchers take COLMAP/OpenCV-frame poses as matrices,
    while the poses handed in here are canonical, so the camera frames are
    S-flipped (S-only, D3) — the same conversion ``match_registered_images``
    performs at the Rust boundary. Note it is *that* path this mirrors, not
    ``match_image_pair``, which passes its canonical poses to
    ``match_image_pair_py`` unflipped and lets Rust convert.
    """
    aff1, aff2, K1, K2, pose1, pose2, _R_2d, config = geometric
    R1, t1 = flip_camera_pose_matrix_s(pose1.rotation.matrix(), pose1.translation)
    R2, t2 = flip_camera_pose_matrix_s(pose2.rotation.matrix(), pose2.translation)
    poses = (
        np.asarray(aff1.reshape(-1, 4), dtype=np.float64),
        np.asarray(aff2.reshape(-1, 4), dtype=np.float64),
        np.asarray(K1, dtype=np.float64),
        np.asarray(K2, dtype=np.float64),
        R1,
        R2,
        t1,
        t2,
    )
    thresholds = (
        config.max_angle_difference,
        config.min_triangulation_angle,
        config.geometric_size_ratio_min,
        config.geometric_size_ratio_max,
    )
    return poses, thresholds


def _as_positions(a):
    return np.asarray(a, dtype=np.float64)


def _as_descriptors(a):
    return np.asarray(a, dtype=np.uint8)


def mutual_best_match_sweep(
    keypoints1,
    descriptors1,
    keypoints2,
    descriptors2,
    window_size,
    threshold=None,
    geometric=None,
):
    """Bidirectional rectified sweep match through the Rust matcher."""
    k1, d1 = _as_positions(keypoints1), _as_descriptors(descriptors1)
    k2, d2 = _as_positions(keypoints2), _as_descriptors(descriptors2)
    if geometric is None:
        return mutual_best_match_sweep_py(k1, d1, k2, d2, window_size, threshold)
    (aff1, aff2, K1, K2, R1, R2, t1, t2), thresholds = _geometric_args(geometric)
    return mutual_best_match_sweep_geometric_py(
        k1,
        d1,
        k2,
        d2,
        aff1,
        aff2,
        K1,
        K2,
        R1,
        R2,
        t1,
        t2,
        window_size,
        threshold,
        *thresholds,
    )


def polar_mutual_best_match(
    positions1,
    descriptors1,
    positions2,
    descriptors2,
    F,
    window_size=15,
    threshold=None,
    min_radius=10.0,
    geometric=None,
):
    """Bidirectional polar sweep match through the Rust matcher.

    Mirrors the production contract: the Rust matchers return ``None`` when the
    epipole is at infinity, which callers surface as a ``ValueError``.
    """
    p1, d1 = _as_positions(positions1), _as_descriptors(descriptors1)
    p2, d2 = _as_positions(positions2), _as_descriptors(descriptors2)
    f_arr = np.asarray(F, dtype=np.float64)
    if geometric is None:
        result = polar_mutual_best_match_py(
            p1, d1, p2, d2, f_arr, window_size, threshold, min_radius
        )
    else:
        (aff1, aff2, K1, K2, R1, R2, t1, t2), thresholds = _geometric_args(geometric)
        result = polar_mutual_best_match_geometric_py(
            p1,
            d1,
            p2,
            d2,
            aff1,
            aff2,
            f_arr,
            K1,
            K2,
            R1,
            R2,
            t1,
            t2,
            window_size,
            threshold,
            min_radius,
            *thresholds,
        )
    if result is None:
        raise ValueError("Epipole is at infinity - use standard rectification instead")
    return result


class TestRectifiedSweepMatching:
    @pytest.mark.parametrize("use_geometric_filter", [False, True])
    def test_sliding_window(self, use_geometric_filter):
        kpts1 = np.array([[0, 10], [0, 20], [0, 30]])
        descs1 = np.array([[1.0] * 128, [2.0] * 128, [3.0] * 128], dtype=np.float32)

        kpts2 = np.array([[0, 11], [0, 21], [0, 31], [0, 41], [0, 51]])
        descs2 = np.array(
            [[1.0] * 128, [2.0] * 128, [3.0] * 128, [4.0] * 128, [5.0] * 128],
            dtype=np.float32,
        )

        geometric = None
        if use_geometric_filter:
            geometric = _make_geometric_params(3, 5)

        matches = mutual_best_match_sweep(
            kpts1, descs1, kpts2, descs2, window_size=2, geometric=geometric
        )
        assert len(matches) == 3
        actual_pairs = {(m[0], m[1]) for m in matches}
        assert actual_pairs == {(0, 0), (1, 1), (2, 2)}

    @pytest.mark.parametrize("use_geometric_filter", [False, True])
    def test_unsorted_input(self, use_geometric_filter):
        kpts1 = np.array([[0, 30], [0, 10]])
        descs1 = np.array([[3.0] * 128, [1.0] * 128], dtype=np.float32)
        kpts2 = np.array([[0, 11], [0, 31]])
        descs2 = np.array([[1.0] * 128, [3.0] * 128], dtype=np.float32)

        geometric = None
        if use_geometric_filter:
            geometric = _make_geometric_params(2, 2)

        matches = mutual_best_match_sweep(
            kpts1, descs1, kpts2, descs2, window_size=2, geometric=geometric
        )
        actual_pairs = {(m[0], m[1]) for m in matches}
        assert actual_pairs == {(1, 0), (0, 1)}


# ===== Polar Sweep Matching Tests =====


class TestPolarSweepMatching:
    @pytest.mark.parametrize("use_geometric_filter", [False, True])
    def test_basic(self, use_geometric_filter):
        rng = np.random.default_rng(42)
        e1 = np.array([320.0, 240.0])
        e2 = np.array([330.0, 250.0])

        e1_h = np.array([e1[0], e1[1], 1.0])
        e2_h = np.array([e2[0], e2[1], 1.0])

        def skew(v):
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        proj = np.eye(3) - np.outer(e1_h, e1_h) / np.dot(e1_h, e1_h)
        F = skew(e2_h) @ proj
        F = F / np.linalg.norm(F)

        n_features = 20
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        radii = 50 + 100 * rng.random(n_features)

        positions1 = np.column_stack(
            [
                e1[0] + radii * np.cos(angles),
                e1[1] + radii * np.sin(angles),
            ]
        )
        positions2 = np.column_stack(
            [
                e2[0] + radii * np.cos(angles + 0.05),
                e2[1] + radii * np.sin(angles + 0.05),
            ]
        )

        descriptors1 = rng.integers(0, 255, (n_features, 128), dtype=np.uint8)
        descriptors2 = descriptors1.copy()
        positions2 += rng.standard_normal((n_features, 2)) * 2

        geometric = None
        if use_geometric_filter:
            geometric = _make_geometric_params(n_features, n_features, forward=True)

        matches = polar_mutual_best_match(
            positions1,
            descriptors1,
            positions2,
            descriptors2,
            F,
            window_size=15,
            min_radius=10.0,
            geometric=geometric,
        )
        assert len(matches) >= n_features // 2

    @pytest.mark.parametrize("use_geometric_filter", [False, True])
    def test_no_false_matches(self, use_geometric_filter):
        rng = np.random.default_rng(123)
        e1 = np.array([320.0, 240.0])
        e2 = np.array([330.0, 250.0])
        e2_h = np.array([e2[0], e2[1], 1.0])
        e2_skew = np.array(
            [[0, -e2_h[2], e2_h[1]], [e2_h[2], 0, -e2_h[0]], [-e2_h[1], e2_h[0], 0]]
        )
        F = e2_skew @ np.eye(3)
        F = F / np.linalg.norm(F)

        n_features = 10
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        radii = 80 + 50 * rng.random(n_features)

        positions1 = np.column_stack(
            [e1[0] + radii * np.cos(angles), e1[1] + radii * np.sin(angles)]
        )
        positions2 = np.column_stack(
            [e2[0] + radii * np.cos(angles), e2[1] + radii * np.sin(angles)]
        )

        descriptors1 = np.zeros((n_features, 128), dtype=np.uint8)
        descriptors2 = np.full((n_features, 128), 255, dtype=np.uint8)

        geometric = None
        if use_geometric_filter:
            geometric = _make_geometric_params(n_features, n_features, forward=True)

        matches = polar_mutual_best_match(
            positions1,
            descriptors1,
            positions2,
            descriptors2,
            F,
            window_size=15,
            threshold=100.0,
            min_radius=10.0,
            geometric=geometric,
        )
        assert len(matches) == 0

    @pytest.mark.parametrize("use_geometric_filter", [False, True])
    def test_epipole_at_infinity_raises(self, use_geometric_filter):
        F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        positions1 = np.array([[100, 100], [200, 200]], dtype=np.float32)
        positions2 = np.array([[110, 110], [210, 210]], dtype=np.float32)
        descriptors1 = np.zeros((2, 128), dtype=np.uint8)
        descriptors2 = np.zeros((2, 128), dtype=np.uint8)

        geometric = None
        if use_geometric_filter:
            geometric = _make_geometric_params(2, 2, forward=True)

        with pytest.raises(ValueError, match="at infinity"):
            polar_mutual_best_match(
                positions1,
                descriptors1,
                positions2,
                descriptors2,
                F,
                geometric=geometric,
            )

    @pytest.mark.parametrize("use_geometric_filter", [False, True])
    def test_empty_after_radius_filtering(self, use_geometric_filter):
        e2 = np.array([100.0, 100.0])
        e2_h = np.array([e2[0], e2[1], 1.0])
        e2_skew = np.array(
            [[0, -e2_h[2], e2_h[1]], [e2_h[2], 0, -e2_h[0]], [-e2_h[1], e2_h[0], 0]]
        )
        F = e2_skew @ np.eye(3)
        F = F / np.linalg.norm(F)

        positions1 = np.array([[101, 101], [99, 99]], dtype=np.float32)
        positions2 = np.array([[102, 102], [98, 98]], dtype=np.float32)
        descriptors1 = np.zeros((2, 128), dtype=np.uint8)
        descriptors2 = np.zeros((2, 128), dtype=np.uint8)

        geometric = None
        if use_geometric_filter:
            geometric = _make_geometric_params(2, 2, forward=True)

        matches = polar_mutual_best_match(
            positions1,
            descriptors1,
            positions2,
            descriptors2,
            F,
            min_radius=50.0,
            geometric=geometric,
        )
        assert len(matches) == 0


# ===== Match Image Pair Tests =====


class TestMatchImagePair:
    def test_lateral_motion_finds_matches(self):
        """Test that match_image_pair finds matches with identical features."""
        rng = np.random.default_rng(42)
        cam = pycolmap.Camera(
            model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240]
        )

        pose_i = pycolmap.Rigid3d(
            pycolmap.Rotation3d([0, 0, 0, 1]), np.array([0.0, 0.0, 0.0])
        )
        pose_j = pycolmap.Rigid3d(
            pycolmap.Rotation3d([0, 0, 0, 1]), np.array([0.5, 0.0, 0.0])
        )

        n = 50
        positions = rng.uniform([50, 50], [590, 430], (n, 2)).astype(np.float64)
        descriptors = rng.integers(0, 255, (n, 128), dtype=np.uint8)

        matches = match_image_pair(
            pose_i,
            pose_j,
            cam,
            cam,
            positions,
            descriptors,
            positions,
            descriptors,
            window_size=30,
        )

        assert len(matches) >= n // 2
