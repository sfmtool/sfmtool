# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the evenly_distributed_sphere_points Rust binding."""

import numpy as np
import pytest

from sfmtool._sfmtool import KdTree3d, evenly_distributed_sphere_points


def _nn_distances(points: np.ndarray) -> np.ndarray:
    return KdTree3d(points.astype(np.float32)).nearest_neighbor_distances()


class TestEvenlyDistributedSpherePoints:
    def test_returns_correct_shape_and_dtype(self):
        points = evenly_distributed_sphere_points(100)
        assert points.shape == (100, 3)
        assert points.dtype == np.float32

    def test_points_are_unit_norm(self):
        points = evenly_distributed_sphere_points(200)
        norms = np.linalg.norm(points, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_zero_points(self):
        points = evenly_distributed_sphere_points(0)
        assert points.shape == (0, 3)
        assert points.dtype == np.float32

    def test_single_point(self):
        points = evenly_distributed_sphere_points(1)
        assert points.shape == (1, 3)
        norm = np.linalg.norm(points[0])
        assert abs(norm - 1.0) < 1e-5

    def test_zero_iterations_still_returns_unit_points(self):
        # With iterations=0 we just get the random initialization, but it
        # still has to be on the unit sphere.
        points = evenly_distributed_sphere_points(50, iterations=0)
        assert points.shape == (50, 3)
        norms = np.linalg.norm(points, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_relaxation_improves_uniformity(self):
        n = 500
        rough = evenly_distributed_sphere_points(n, iterations=0)
        smooth = evenly_distributed_sphere_points(n, iterations=50)
        rough_std = float(np.std(_nn_distances(rough)))
        smooth_std = float(np.std(_nn_distances(smooth)))
        # Random uniform sampling has high NN-distance variance; relaxation
        # should at least halve it.
        assert smooth_std < rough_std * 0.5, (
            f"relaxation did not cut NN variance enough: "
            f"rough_std={rough_std}, smooth_std={smooth_std}"
        )

    def test_density_matches_expected_spacing(self):
        # Mean NN distance for an evenly tiled unit sphere is approximately
        # √(4π/n) — within ~30% of that ballpark.
        n = 1000
        points = evenly_distributed_sphere_points(n)
        expected = np.sqrt(4.0 * np.pi / n)
        mean_nn = float(np.mean(_nn_distances(points)))
        assert 0.7 * expected < mean_nn < 1.3 * expected, (
            f"mean NN spacing {mean_nn} not close to expected {expected}"
        )

    def test_two_points_become_approximately_antipodal(self):
        # With fixed step length the algorithm has a limit cycle around
        # antipodal — confirm only that the points end up in opposite
        # hemispheres of each other.
        points = evenly_distributed_sphere_points(2, iterations=500)
        dot = float(np.dot(points[0], points[1]))
        assert dot < -0.95, f"two points did not converge near antipodal: dot={dot}"

    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_various_sizes(self, n):
        points = evenly_distributed_sphere_points(n)
        assert points.shape == (n, 3)
        norms = np.linalg.norm(points, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_seed_makes_result_deterministic(self):
        a = evenly_distributed_sphere_points(80, seed=42)
        b = evenly_distributed_sphere_points(80, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_points(self):
        a = evenly_distributed_sphere_points(80, seed=1)
        b = evenly_distributed_sphere_points(80, seed=2)
        assert not np.array_equal(a, b)
