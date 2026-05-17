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
