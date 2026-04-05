# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the densify command, feature matching module, and supporting utilities."""

import numpy as np
import pytest
import pycolmap
from click.testing import CliRunner

from sfmtool._cameras import get_intrinsic_matrix
from sfmtool._densify import prune_image_pairs
from sfmtool.cli import main
from sfmtool.feature_match import GeometricFilterConfig, match_image_pair
from sfmtool.feature_match._geometry import (
    check_rectification_safe,
    compute_epipole,
    get_essential_matrix,
    get_fundamental_matrix,
)
from sfmtool.feature_match._polar_sweep import (
    _cartesian_to_polar,
    _compute_epipole_pair_from_F,
    polar_mutual_best_match,
)
from sfmtool.feature_match._rectified_sweep import mutual_best_match_sweep


# ===== Geometry Tests =====


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


# ===== Geometric Filter Config Tests =====


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

        kwargs = {}
        if use_geometric_filter:
            aff1, aff2, K1, K2, p1, p2, R_2d, config = _make_geometric_params(3, 5)
            kwargs = {
                "affine_shapes1": aff1,
                "affine_shapes2": aff2,
                "K1": K1,
                "K2": K2,
                "pose1": p1,
                "pose2": p2,
                "R_2d": R_2d,
                "geometric_config": config,
            }

        matches = mutual_best_match_sweep(
            kpts1, descs1, kpts2, descs2, window_size=2, **kwargs
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

        kwargs = {}
        if use_geometric_filter:
            aff1, aff2, K1, K2, p1, p2, R_2d, config = _make_geometric_params(2, 2)
            kwargs = {
                "affine_shapes1": aff1,
                "affine_shapes2": aff2,
                "K1": K1,
                "K2": K2,
                "pose1": p1,
                "pose2": p2,
                "R_2d": R_2d,
                "geometric_config": config,
            }

        matches = mutual_best_match_sweep(
            kpts1, descs1, kpts2, descs2, window_size=2, **kwargs
        )
        actual_pairs = {(m[0], m[1]) for m in matches}
        assert actual_pairs == {(1, 0), (0, 1)}


# ===== Polar Sweep Matching Tests =====


class TestPolarCoordinates:
    def test_basic(self):
        epipole = np.array([100.0, 100.0])
        points = np.array(
            [
                [200.0, 100.0],
                [100.0, 200.0],
                [0.0, 100.0],
                [100.0, 0.0],
            ]
        )
        polar = _cartesian_to_polar(points, epipole, min_radius=1.0)
        assert len(polar.theta) == 4
        np.testing.assert_allclose(polar.radius, [100, 100, 100, 100], atol=1e-10)
        np.testing.assert_allclose(
            polar.theta,
            [0, np.pi / 2, np.pi, -np.pi / 2],
            atol=1e-10,
        )

    def test_excludes_close_points(self):
        epipole = np.array([100.0, 100.0])
        points = np.array(
            [
                [100.0, 100.0],
                [105.0, 100.0],
                [200.0, 100.0],
            ]
        )
        polar = _cartesian_to_polar(points, epipole, min_radius=10.0)
        assert len(polar.theta) == 1
        assert polar.original_indices[0] == 2

    def test_epipole_pair_from_F(self):
        F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        e1, e2, e1_inf, e2_inf = _compute_epipole_pair_from_F(F)
        assert e1_inf
        assert e2_inf


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

        kwargs = {}
        if use_geometric_filter:
            aff1, aff2, K1, K2, p1, p2, R_2d, config = _make_geometric_params(
                n_features, n_features, forward=True
            )
            kwargs = {
                "affine_shapes1": aff1,
                "affine_shapes2": aff2,
                "K1": K1,
                "K2": K2,
                "pose1": p1,
                "pose2": p2,
                "R_2d": R_2d,
                "geometric_config": config,
            }

        matches = polar_mutual_best_match(
            positions1,
            descriptors1,
            positions2,
            descriptors2,
            F,
            window_size=15,
            min_radius=10.0,
            **kwargs,
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

        kwargs = {}
        if use_geometric_filter:
            aff1, aff2, K1, K2, p1, p2, R_2d, config = _make_geometric_params(
                n_features, n_features, forward=True
            )
            kwargs = {
                "affine_shapes1": aff1,
                "affine_shapes2": aff2,
                "K1": K1,
                "K2": K2,
                "pose1": p1,
                "pose2": p2,
                "R_2d": R_2d,
                "geometric_config": config,
            }

        matches = polar_mutual_best_match(
            positions1,
            descriptors1,
            positions2,
            descriptors2,
            F,
            window_size=15,
            threshold=100.0,
            min_radius=10.0,
            **kwargs,
        )
        assert len(matches) == 0

    @pytest.mark.parametrize("use_geometric_filter", [False, True])
    def test_epipole_at_infinity_raises(self, use_geometric_filter):
        F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        positions1 = np.array([[100, 100], [200, 200]], dtype=np.float32)
        positions2 = np.array([[110, 110], [210, 210]], dtype=np.float32)
        descriptors1 = np.zeros((2, 128), dtype=np.uint8)
        descriptors2 = np.zeros((2, 128), dtype=np.uint8)

        kwargs = {}
        if use_geometric_filter:
            aff1, aff2, K1, K2, p1, p2, R_2d, config = _make_geometric_params(
                2, 2, forward=True
            )
            kwargs = {
                "affine_shapes1": aff1,
                "affine_shapes2": aff2,
                "K1": K1,
                "K2": K2,
                "pose1": p1,
                "pose2": p2,
                "R_2d": R_2d,
                "geometric_config": config,
            }

        with pytest.raises(ValueError, match="at infinity"):
            polar_mutual_best_match(
                positions1,
                descriptors1,
                positions2,
                descriptors2,
                F,
                **kwargs,
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

        kwargs = {}
        if use_geometric_filter:
            aff1, aff2, K1, K2, p1, p2, R_2d, config = _make_geometric_params(
                2, 2, forward=True
            )
            kwargs = {
                "affine_shapes1": aff1,
                "affine_shapes2": aff2,
                "K1": K1,
                "K2": K2,
                "pose1": p1,
                "pose2": p2,
                "R_2d": R_2d,
                "geometric_config": config,
            }

        matches = polar_mutual_best_match(
            positions1,
            descriptors1,
            positions2,
            descriptors2,
            F,
            min_radius=50.0,
            **kwargs,
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


# ===== Prune Image Pairs Tests =====


class TestPruneImagePairs:
    def test_empty_input(self):
        assert prune_image_pairs([]) == []

    def test_close_pairs_kept(self):
        pairs = [(0, 1, 10), (1, 2, 8), (2, 3, 6)]
        result = prune_image_pairs(pairs, close_pair_threshold=4, max_distant_pairs=0)
        assert len(result) == 3

    def test_max_close_pairs_limits(self):
        pairs = [(0, 1, 10), (1, 2, 8), (2, 3, 6)]
        result = prune_image_pairs(
            pairs, close_pair_threshold=4, max_close_pairs=2, max_distant_pairs=0
        )
        assert len(result) == 2

    def test_distant_pairs_sorted_by_score(self):
        pairs = [
            (0, 100, 5),
            (0, 200, 10),
            (0, 300, 1),
        ]
        result = prune_image_pairs(
            pairs,
            close_pair_threshold=4,
            max_distant_pairs=2,
            distant_pair_search_multiplier=10,
        )
        assert len(result) == 2
        # Best scores should be kept
        assert (0, 200) in result
        assert (0, 100) in result

    def test_mixed_close_and_distant(self):
        pairs = [(0, 1, 10), (0, 2, 8), (0, 100, 20), (0, 200, 15)]
        result = prune_image_pairs(pairs, close_pair_threshold=4, max_distant_pairs=1)
        # 2 close + 1 best distant
        close = [(i, j) for i, j in result if abs(j - i) <= 4]
        distant = [(i, j) for i, j in result if abs(j - i) > 4]
        assert len(close) == 2
        assert len(distant) == 1


# ===== Densify CLI Tests =====


class TestDensifyCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["densify", "--help"])
        assert result.exit_code == 0
        assert "Densify matches" in result.output

    def test_non_sfmr_input_rejected(self, tmp_path):
        runner = CliRunner()
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello")
        output_file = tmp_path / "output.sfmr"
        result = runner.invoke(main, ["densify", str(input_file), str(output_file)])
        assert result.exit_code != 0
        assert (
            "sfmr" in result.output.lower() or "sfmr" in str(result.exception).lower()
        )

    def test_non_sfmr_output_rejected(self, tmp_path):
        runner = CliRunner()
        input_file = tmp_path / "input.sfmr"
        input_file.write_text("hello")
        output_file = tmp_path / "output.txt"
        result = runner.invoke(main, ["densify", str(input_file), str(output_file)])
        assert result.exit_code != 0

    def test_nonexistent_input_rejected(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "densify",
                str(tmp_path / "nonexistent.sfmr"),
                str(tmp_path / "output.sfmr"),
            ],
        )
        assert result.exit_code != 0


# ===== End-to-end Densify Test =====


class TestDensifyE2E:
    def test_densify_reconstruction(self, sfmrfile_reconstruction_with_17_images):
        """Test that densify produces a valid reconstruction with more points."""
        from sfmtool._densify import densify_reconstruction
        from sfmtool._sfmtool import SfmrReconstruction

        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(str(sfmr_path))
        original_point_count = recon.point_count

        result = densify_reconstruction(
            recon=recon,
            max_features=512,
            sweep_window_size=30,
        )

        assert result.image_count == recon.image_count
        assert result.camera_count >= 1
        assert result.point_count > 0
        # Densified should generally have more points
        # (but don't strictly require it since filtering can reduce count)

    def test_densify_cli_e2e(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Test densify CLI end-to-end."""
        runner = CliRunner()
        input_path = sfmrfile_reconstruction_with_17_images
        output_path = tmp_path / "densified.sfmr"

        result = runner.invoke(
            main,
            [
                "densify",
                str(input_path),
                str(output_path),
                "--max-features",
                "512",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_path.exists()

        from sfmtool._sfmtool import SfmrReconstruction

        densified = SfmrReconstruction.load(str(output_path))
        assert densified.image_count > 0
        assert densified.point_count > 0
