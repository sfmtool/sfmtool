# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``OrientedPatch.render_view_tile``: one view's ``R x R`` tile of a
patch, rendered as the bench renders an observation's tile, with the per-view
readings the reference-view rule takes on it. See
``specs/core/patch/reference-view.md``."""

import numpy as np
import pytest

from sfmtool._sfmtool.geometry import CameraIntrinsics, RigidTransform
from sfmtool._sfmtool.patches import OrientedPatch


def _pinhole(w=64, h=48, f=60.0):
    return CameraIntrinsics(
        "PINHOLE",
        w,
        h,
        {
            "focal_length_x": f,
            "focal_length_y": f,
            "principal_point_x": w / 2.0,
            "principal_point_y": h / 2.0,
        },
    )


def _identity():
    return RigidTransform.from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def _photograph(w=64, h=48):
    y, x = np.mgrid[0:h, 0:w]
    grey = (40 + (x * 7 + y * 13) % 150).astype(np.uint8)
    return np.dstack([grey, grey, grey])


def test_a_patch_facing_the_camera_renders_a_whole_tile():
    # Canonical cameras look down -Z; this patch faces back along +Z.
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = patch.render_view_tile(_pinhole(), _identity(), _photograph(), resolution=16)
    assert tile["samples"].shape == (16, 16, 3)
    assert tile["samples"].dtype == np.uint8
    assert tile["valid"].shape == (16, 16) and tile["valid"].all()
    assert tile["coverage"] == 1.0
    assert tile["clipped_share"] == 0.0
    assert tile["viewing_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert tile["tilt_direction_deg"] is None
    assert tile["sampler"] in {"bilinear_mip", "anisotropic"}
    assert tile["jacobian"].shape == (2, 2)
    np.testing.assert_allclose(tile["placement"].center, [0, 0, -4])


def test_anchoring_on_a_keypoint_moves_the_tile_and_reads_the_angle_there():
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = patch.render_view_tile(
        _pinhole(), _identity(), _photograph(), keypoint=(42.0, 24.0), resolution=16
    )
    # The anchored patch stays in its plane and its centre projects onto the
    # keypoint, 10 px right of the principal point: 10 / 60 of the depth.
    center = tile["placement"].center
    assert center[2] == pytest.approx(-4.0)
    assert abs(center[0]) == pytest.approx(4.0 * 10.0 / 60.0)
    # The ray to it leans off the normal by atan(10 / 60).
    assert tile["viewing_angle_deg"] == pytest.approx(np.degrees(np.arctan(10 / 60)))
    assert tile["tilt_direction_deg"] is not None


def test_a_blown_out_photograph_is_clipped_and_a_tile_off_its_edge_is_partial():
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    white = np.full((48, 64, 3), 255, np.uint8)
    tile = patch.render_view_tile(_pinhole(), _identity(), white, resolution=16)
    assert tile["clipped_share"] == 1.0
    edge = OrientedPatch([2.1, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = edge.render_view_tile(_pinhole(), _identity(), _photograph(), resolution=16)
    assert 0.0 < tile["coverage"] < 1.0
    assert tile["coverage"] == pytest.approx(tile["valid"].mean())


def test_a_fixed_sampler_and_bad_arguments():
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = patch.render_view_tile(
        _pinhole(), _identity(), _photograph(), resolution=8, sampler="bilinear"
    )
    assert tile["sampler"] == "bilinear"
    with pytest.raises(ValueError, match="resolution"):
        patch.render_view_tile(_pinhole(), _identity(), _photograph(), resolution=1)
    with pytest.raises(ValueError, match="sampler"):
        patch.render_view_tile(
            _pinhole(), _identity(), _photograph(), sampler="nearest"
        )
    with pytest.raises(ValueError, match="camera"):
        patch.render_view_tile(_pinhole(), _identity(), _photograph(32, 32))
