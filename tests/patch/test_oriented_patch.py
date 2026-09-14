# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for OrientedPatch and WarpMap.from_patch."""

from __future__ import annotations

import numpy as np

from sfmtool._sfmtool.patches import OrientedPatch
from sfmtool._sfmtool.geometry import CameraIntrinsics, RigidTransform
from sfmtool._sfmtool.flow import WarpMap


def _pinhole(f, cx, cy, w, h):
    return CameraIntrinsics(
        "PINHOLE",
        w,
        h,
        {
            "focal_length_x": f,
            "focal_length_y": f,
            "principal_point_x": cx,
            "principal_point_y": cy,
        },
    )


def _identity():
    return RigidTransform.from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_from_patch_projects_fronto_parallel_plane():
    f, cx, cy = 500.0, 320.0, 240.0
    cam = _pinhole(f, cx, cy, 640, 480)
    pose = _identity()  # world == camera frame
    d, h = 4.0, 0.5
    # Canonical cameras look down -Z, so a visible patch sits at z = -d (in
    # front). The projection formulas below are unchanged: the normalized coords
    # divide by the depth -z = d.
    patch = OrientedPatch([0, 0, -d], [1, 0, 0], [0, 1, 0], [h, h])

    r = 8
    wm = WarpMap.from_patch(patch, cam, pose, r)
    assert (wm.width, wm.height) == (r, r)

    map_x, map_y = wm.to_numpy()
    step = 2.0 / r
    for col in range(r):
        for row in range(r):
            s = (col + 0.5) * step - 1.0
            t = (row + 0.5) * step - 1.0
            # Columns run with +u_axis; rows run with -v_axis (the raster
            # reverses v to render un-mirrored). The canonical camera projects
            # with +Y up (image-y = cy - fy·y_n), which cancels that raster
            # reversal, so image-y tracks +t (was -t under the old +Y-down
            # COLMAP camera). The patch is at z = -d, so x_n = s·h/d, y_n = t·h/d.
            assert abs(map_x[row, col] - (f * s * h / d + cx)) < 1e-2
            assert abs(map_y[row, col] - (f * t * h / d + cy)) < 1e-2


def test_from_center_normal_and_front_facing():
    patch = OrientedPatch.from_center_normal([0, 0, 5], [0, 0, -1], [0, 1, 0], [1, 1])
    np.testing.assert_allclose(np.asarray(patch.normal), [0, 0, -1], atol=1e-9)
    assert patch.is_front_facing(_identity())

    back = OrientedPatch.from_center_normal([0, 0, 5], [0, 0, 1], [0, 1, 0], [1, 1])
    assert not back.is_front_facing(_identity())


def test_from_patch_remaps_an_image():
    cam = _pinhole(500.0, 320.0, 240.0, 640, 480)
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.5, 0.5])
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :320] = (255, 0, 0)
    out = np.asarray(
        WarpMap.from_patch(patch, cam, _identity(), 16).remap_bilinear(img)
    )
    assert out.shape == (16, 16, 3)


def _simple_radial(f, cx, cy, k1, w, h):
    return CameraIntrinsics(
        "SIMPLE_RADIAL",
        w,
        h,
        {
            "focal_length": f,
            "principal_point_x": cx,
            "principal_point_y": cy,
            "radial_distortion_k1": k1,
        },
    )


def _shape_from_frame(patch, cam, keypoint):
    """The format's frame-to-shape rule, in the camera's own frame.

    Project the patch anchor and the tips of its two half-vectors; the pixel
    differences are the shape's columns. Written against an identity pose, so
    world and camera coordinates coincide.
    """
    center = np.asarray(patch.center)
    u = np.asarray(patch.u_axis) * patch.half_extent[0]
    v = np.asarray(patch.v_axis) * patch.half_extent[1]
    k = np.asarray(cam.ray_to_pixel(list(center)))
    np.testing.assert_allclose(k, keypoint, atol=1e-9)
    pu = np.asarray(cam.ray_to_pixel(list(center + u)))
    pv = np.asarray(cam.ray_to_pixel(list(center + v)))
    return np.array([[pu[0] - k[0], pv[0] - k[0]], [pu[1] - k[1], pv[1] - k[1]]])


def test_from_affine_shape_at_depth_round_trips_the_shape():
    """A frame built from a shape at a depth projects back to that shape.

    A patch-convention shape (negative determinant, since `v` points image-up
    while pixel rows count down) comes back unchanged, for a pinhole and for a
    radially distorted camera: both directions go through the camera model.
    """
    keypoint = [412.0, 173.0]
    shape = [[7.5, 1.2], [1.9, -6.1]]
    depth = 5.25
    for cam in (
        _pinhole(520.0, 320.0, 240.0, 640, 480),
        _simple_radial(520.0, 320.0, 240.0, -0.11, 640, 480),
    ):
        patch = OrientedPatch.from_affine_shape_at_depth(
            cam, _identity(), keypoint, shape, depth
        )
        assert patch is not None
        assert patch.w == 1.0
        assert patch.is_front_facing(_identity())
        assert abs(np.linalg.norm(np.asarray(patch.center)) - depth) < 1e-9
        np.testing.assert_allclose(
            _shape_from_frame(patch, cam, keypoint), shape, atol=1e-9
        )


def test_from_affine_shape_at_depth_refuses_a_bad_depth():
    cam = _pinhole(520.0, 320.0, 240.0, 640, 480)
    shape = [[6.0, 0.0], [0.0, -6.0]]
    for depth in (0.0, -2.0, float("nan")):
        assert (
            OrientedPatch.from_affine_shape_at_depth(
                cam, _identity(), [320.0, 240.0], shape, depth
            )
            is None
        )
    assert (
        OrientedPatch.from_affine_shape_at_depth(
            cam, _identity(), [320.0, 240.0], [[0.0, 0.0], [0.0, 0.0]], 3.0
        )
        is None
    )
