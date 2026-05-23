# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the homogeneous-point accessors on the SfmrReconstruction binding."""

import numpy as np
import pytest

from sfmtool._sfmtool import SfmrReconstruction


class TestHomogeneousPointAccessors:
    def test_finite_reconstruction_accessors(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        m = len(recon.positions)
        assert m > 0

        positions = recon.positions
        positions_xyzw = recon.positions_xyzw
        at_infinity = recon.point_is_at_infinity

        assert positions.shape == (m, 3)
        assert positions_xyzw.shape == (m, 4)
        assert at_infinity.shape == (m,)
        assert at_infinity.dtype == bool

        # A normally solved reconstruction contains only finite points.
        assert not at_infinity.any()
        assert np.array_equal(positions_xyzw[:, 3], np.ones(m))
        assert np.array_equal(positions_xyzw[:, :3], positions)

    def test_clone_with_changes_homogeneous_positions(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        positions_xyzw = recon.positions_xyzw.copy()
        assert len(positions_xyzw) >= 2

        # Turn the first point into a point at infinity: direction (0, 0, 2),
        # w = 0. clone_with_changes must normalise the direction to unit length.
        positions_xyzw[0] = [0.0, 0.0, 2.0, 0.0]
        clone = recon.clone_with_changes(positions=positions_xyzw)

        at_infinity = clone.point_is_at_infinity
        assert at_infinity[0]
        assert not at_infinity[1:].any()

        assert clone.positions_xyzw[0, 3] == 0.0
        np.testing.assert_allclose(clone.positions[0], [0.0, 0.0, 1.0])

    def test_clone_with_changes_rejects_all_zero_homogeneous_coordinate(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        positions_xyzw = recon.positions_xyzw.copy()
        assert len(positions_xyzw) > 0
        # (0, 0, 0, 0) is no point at all: w = 0 with a zero direction.
        positions_xyzw[0] = [0.0, 0.0, 0.0, 0.0]

        with pytest.raises(ValueError, match="all-zero homogeneous coordinate"):
            recon.clone_with_changes(positions=positions_xyzw)

    def test_clone_with_changes_euclidean_positions_stay_finite(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        # An (N, 3) Euclidean array keeps every point finite (w = 1).
        clone = recon.clone_with_changes(positions=recon.positions)
        assert not clone.point_is_at_infinity.any()
        assert np.array_equal(clone.positions_xyzw[:, 3], np.ones(len(clone.positions)))


class TestInfinityConversions:
    def test_classify_preserves_point_count(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        classified = recon.classify_points_at_infinity()
        # Reclassification never adds or drops points or observations.
        assert classified.point_count == recon.point_count
        assert classified.observation_count == recon.observation_count

    def test_classify_detects_a_far_point(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        # Pick a well-observed point and push it far away so its observation
        # rays become parallel — its parallax collapses below the noise floor.
        idx = int(np.argmax(recon.observation_counts))
        positions = recon.positions_xyzw.copy()
        positions[idx] = [0.0, 0.0, 1.0e7, 1.0]
        recon = recon.clone_with_changes(positions=positions)

        classified = recon.classify_points_at_infinity()
        assert classified.point_is_at_infinity[idx]
        # A point at infinity stores a unit-length direction.
        np.testing.assert_allclose(
            np.linalg.norm(classified.positions[idx]), 1.0, atol=1e-9
        )

    def test_classify_noise_floor_is_monotone(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        # A larger noise floor can only classify a superset of points.
        strict = recon.classify_points_at_infinity(noise_floor_px=0.01)
        loose = recon.classify_points_at_infinity(noise_floor_px=1.0e4)
        assert loose.point_is_at_infinity.sum() >= strict.point_is_at_infinity.sum()

    def test_materialize_makes_every_point_finite(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        positions = recon.positions_xyzw.copy()
        n_infinity = min(5, len(positions))
        for i in range(n_infinity):
            positions[i] = [0.0, 0.0, 1.0, 0.0]
        recon = recon.clone_with_changes(positions=positions)
        assert recon.point_is_at_infinity[:n_infinity].all()

        materialized = recon.materialize_points_at_infinity()
        assert not materialized.point_is_at_infinity.any()
        assert materialized.point_count == recon.point_count
        assert np.all(np.isfinite(materialized.positions))

    def test_materialize_leaves_finite_points_unchanged(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        # No points at infinity — materialise is a no-op on the positions.
        materialized = recon.materialize_points_at_infinity()
        np.testing.assert_array_equal(materialized.positions, recon.positions)


class TestWorldSpaceUnit:
    def test_default_is_none(self, sfmrfile_reconstruction_with_17_images):
        # A freshly solved reconstruction is in arbitrary units.
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        assert recon.world_space_unit is None

    def test_clone_sets_and_clears_unit(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)

        scaled = recon.clone_with_changes(world_space_unit="m")
        assert scaled.world_space_unit == "m"
        # The original is untouched.
        assert recon.world_space_unit is None

        cleared = scaled.clone_with_changes(world_space_unit=None)
        assert cleared.world_space_unit is None

    def test_unit_survives_save_load_roundtrip(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        scaled = recon.clone_with_changes(world_space_unit="mm")

        out_path = tmp_path / "scaled.sfmr"
        scaled.save(out_path)

        reloaded = SfmrReconstruction.load(out_path)
        assert reloaded.world_space_unit == "mm"
