# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for OrientedPatch and WarpMap.from_patch."""

from __future__ import annotations

import numpy as np

from sfmtool._sfmtool import OrientedPatch
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
    patch = OrientedPatch([0, 0, d], [1, 0, 0], [0, 1, 0], [h, h])

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
            # reverses v to render un-mirrored), so image-y uses -t.
            assert abs(map_x[row, col] - (f * s * h / d + cx)) < 1e-2
            assert abs(map_y[row, col] - (f * -t * h / d + cy)) < 1e-2


def test_from_center_normal_and_front_facing():
    patch = OrientedPatch.from_center_normal([0, 0, 5], [0, 0, -1], [0, 1, 0], [1, 1])
    np.testing.assert_allclose(np.asarray(patch.normal), [0, 0, -1], atol=1e-9)
    assert patch.is_front_facing(_identity())

    back = OrientedPatch.from_center_normal([0, 0, 5], [0, 0, 1], [0, 1, 0], [1, 1])
    assert not back.is_front_facing(_identity())


def test_from_patch_remaps_an_image():
    cam = _pinhole(500.0, 320.0, 240.0, 640, 480)
    patch = OrientedPatch([0, 0, 4], [1, 0, 0], [0, 1, 0], [0.5, 0.5])
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :320] = (255, 0, 0)
    out = np.asarray(
        WarpMap.from_patch(patch, cam, _identity(), 16).remap_bilinear(img)
    )
    assert out.shape == (16, 16, 3)
