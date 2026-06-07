# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests validating Rust distortion/undistortion against pycolmap."""

import numpy as np
import pycolmap
import pytest

from sfmtool._sfmtool import CameraIntrinsics

CAMERA_MODELS = [
    (
        "SIMPLE_PINHOLE",
        640,
        480,
        [500.0, 320.0, 240.0],
        {"focal_length": 500.0, "principal_point_x": 320.0, "principal_point_y": 240.0},
    ),
    (
        "PINHOLE",
        640,
        480,
        [500.0, 502.0, 320.0, 240.0],
        {
            "focal_length_x": 500.0,
            "focal_length_y": 502.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
        },
    ),
    (
        "SIMPLE_RADIAL",
        640,
        480,
        [500.0, 320.0, 240.0, 0.05],
        {
            "focal_length": 500.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
        },
    ),
    (
        "RADIAL",
        640,
        480,
        [500.0, 320.0, 240.0, 0.05, -0.02],
        {
            "focal_length": 500.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
            "radial_distortion_k2": -0.02,
        },
    ),
    (
        "OPENCV",
        640,
        480,
        [500.0, 502.0, 320.0, 240.0, 0.05, -0.02, 0.001, -0.002],
        {
            "focal_length_x": 500.0,
            "focal_length_y": 502.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
            "radial_distortion_k2": -0.02,
            "tangential_distortion_p1": 0.001,
            "tangential_distortion_p2": -0.002,
        },
    ),
    (
        "OPENCV_FISHEYE",
        640,
        480,
        [500.0, 502.0, 320.0, 240.0, 0.05, -0.02, 0.005, -0.003],
        {
            "focal_length_x": 500.0,
            "focal_length_y": 502.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
            "radial_distortion_k2": -0.02,
            "radial_distortion_k3": 0.005,
            "radial_distortion_k4": -0.003,
        },
    ),
    (
        "FULL_OPENCV",
        640,
        480,
        [
            500.0,
            502.0,
            320.0,
            240.0,
            0.05,
            -0.02,
            0.001,
            -0.002,
            0.005,
            -0.003,
            0.001,
            -0.001,
        ],
        {
            "focal_length_x": 500.0,
            "focal_length_y": 502.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
            "radial_distortion_k2": -0.02,
            "tangential_distortion_p1": 0.001,
            "tangential_distortion_p2": -0.002,
            "radial_distortion_k3": 0.005,
            "radial_distortion_k4": -0.003,
            "radial_distortion_k5": 0.001,
            "radial_distortion_k6": -0.001,
        },
    ),
    (
        "SIMPLE_RADIAL_FISHEYE",
        640,
        480,
        [500.0, 320.0, 240.0, 0.05],
        {
            "focal_length": 500.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
        },
    ),
    (
        "RADIAL_FISHEYE",
        640,
        480,
        [500.0, 320.0, 240.0, 0.05, -0.02],
        {
            "focal_length": 500.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
            "radial_distortion_k2": -0.02,
        },
    ),
    (
        "THIN_PRISM_FISHEYE",
        640,
        480,
        [
            500.0,
            502.0,
            320.0,
            240.0,
            0.05,
            -0.02,
            0.001,
            -0.002,
            0.005,
            -0.003,
            0.001,
            -0.001,
        ],
        {
            "focal_length_x": 500.0,
            "focal_length_y": 502.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.05,
            "radial_distortion_k2": -0.02,
            "tangential_distortion_p1": 0.001,
            "tangential_distortion_p2": -0.002,
            "radial_distortion_k3": 0.005,
            "radial_distortion_k4": -0.003,
            "thin_prism_sx1": 0.001,
            "thin_prism_sy1": -0.001,
        },
    ),
    (
        "RAD_TAN_THIN_PRISM_FISHEYE",
        640,
        480,
        [
            500.0,
            502.0,
            320.0,
            240.0,
            0.05,
            -0.02,
            0.005,
            -0.003,
            0.001,
            -0.001,
            0.001,
            -0.002,
            0.0005,
            -0.0005,
            0.0002,
            -0.0002,
        ],
        {
            "focal_length_x": 500.0,
            "focal_length_y": 502.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
            "radial_distortion_k0": 0.05,
            "radial_distortion_k1": -0.02,
            "radial_distortion_k2": 0.005,
            "radial_distortion_k3": -0.003,
            "radial_distortion_k4": 0.001,
            "radial_distortion_k5": -0.001,
            "tangential_distortion_p0": 0.001,
            "tangential_distortion_p1": -0.002,
            "thin_prism_s0": 0.0005,
            "thin_prism_s1": -0.0005,
            "thin_prism_s2": 0.0002,
            "thin_prism_s3": -0.0002,
        },
    ),
]

IMAGE_PLANE_POINTS = [
    (0.0, 0.0),
    (0.1, 0.0),
    (0.0, 0.1),
    (0.1, 0.1),
    (-0.2, 0.15),
    (0.3, -0.2),
    (-0.1, -0.3),
    (0.05, -0.05),
    (-0.3, 0.25),
    (0.4, 0.3),
]


def _make_cameras(model_name, width, height, pycolmap_params, rust_params):
    py_cam = pycolmap.Camera(
        model=model_name, width=width, height=height, params=pycolmap_params
    )
    rust_cam = CameraIntrinsics(model_name, width, height, rust_params)
    return py_cam, rust_cam


@pytest.mark.parametrize(
    "model_name, width, height, pycolmap_params, rust_params",
    CAMERA_MODELS,
    ids=[m[0] for m in CAMERA_MODELS],
)
@pytest.mark.parametrize("x, y", IMAGE_PLANE_POINTS)
def test_project_matches_pycolmap(
    model_name, width, height, pycolmap_params, rust_params, x, y
):
    py_cam, rust_cam = _make_cameras(
        model_name, width, height, pycolmap_params, rust_params
    )
    py_pixel = py_cam.img_from_cam(np.array([x, y, 1.0]))
    rust_pixel = rust_cam.project(x, y)
    np.testing.assert_allclose(
        rust_pixel, py_pixel, atol=1e-10, err_msg=f"{model_name} project({x}, {y})"
    )


@pytest.mark.parametrize(
    "model_name, width, height, pycolmap_params, rust_params",
    CAMERA_MODELS,
    ids=[m[0] for m in CAMERA_MODELS],
)
@pytest.mark.parametrize("x, y", IMAGE_PLANE_POINTS)
def test_unproject_matches_pycolmap(
    model_name, width, height, pycolmap_params, rust_params, x, y
):
    py_cam, rust_cam = _make_cameras(
        model_name, width, height, pycolmap_params, rust_params
    )
    pixel = py_cam.img_from_cam(np.array([x, y, 1.0]))
    py_result = py_cam.cam_from_img(pixel)
    rust_result = rust_cam.unproject(pixel[0], pixel[1])
    np.testing.assert_allclose(
        rust_result,
        py_result,
        atol=1e-8,
        err_msg=f"{model_name} unproject({pixel[0]:.4f}, {pixel[1]:.4f})",
    )


@pytest.mark.parametrize(
    "model_name, width, height, pycolmap_params, rust_params",
    CAMERA_MODELS,
    ids=[m[0] for m in CAMERA_MODELS],
)
@pytest.mark.parametrize("x, y", IMAGE_PLANE_POINTS)
def test_project_unproject_round_trip(
    model_name, width, height, pycolmap_params, rust_params, x, y
):
    rust_cam = CameraIntrinsics(model_name, width, height, rust_params)
    u, v = rust_cam.project(x, y)
    x_rt, y_rt = rust_cam.unproject(u, v)
    np.testing.assert_allclose(
        [x_rt, y_rt], [x, y], atol=1e-8, err_msg=f"{model_name} round-trip ({x}, {y})"
    )


@pytest.mark.parametrize(
    "model_name, width, height, pycolmap_params, rust_params",
    CAMERA_MODELS,
    ids=[m[0] for m in CAMERA_MODELS],
)
def test_project_batch_matches_single(
    model_name, width, height, pycolmap_params, rust_params
):
    rust_cam = CameraIntrinsics(model_name, width, height, rust_params)
    points = np.array(IMAGE_PLANE_POINTS)
    batch_result = rust_cam.project_batch(points)
    for i, (x, y) in enumerate(IMAGE_PLANE_POINTS):
        single_result = rust_cam.project(x, y)
        np.testing.assert_allclose(
            batch_result[i],
            single_result,
            atol=1e-15,
            err_msg=f"{model_name} project_batch[{i}]",
        )


@pytest.mark.parametrize(
    "model_name, width, height, pycolmap_params, rust_params",
    CAMERA_MODELS,
    ids=[m[0] for m in CAMERA_MODELS],
)
def test_unproject_batch_matches_single(
    model_name, width, height, pycolmap_params, rust_params
):
    rust_cam = CameraIntrinsics(model_name, width, height, rust_params)
    pixels = np.array([rust_cam.project(x, y) for x, y in IMAGE_PLANE_POINTS])
    batch_result = rust_cam.unproject_batch(pixels)
    for i in range(len(IMAGE_PLANE_POINTS)):
        single_result = rust_cam.unproject(pixels[i, 0], pixels[i, 1])
        np.testing.assert_allclose(
            batch_result[i],
            single_result,
            atol=1e-15,
            err_msg=f"{model_name} unproject_batch[{i}]",
        )


@pytest.mark.parametrize(
    "model_name, width, height, pycolmap_params, rust_params",
    CAMERA_MODELS,
    ids=[m[0] for m in CAMERA_MODELS],
)
def test_unproject_batch_matches_pycolmap(
    model_name, width, height, pycolmap_params, rust_params
):
    py_cam, rust_cam = _make_cameras(
        model_name, width, height, pycolmap_params, rust_params
    )
    pixels = np.array(
        [py_cam.img_from_cam(np.array([x, y, 1.0])) for x, y in IMAGE_PLANE_POINTS]
    )
    rust_results = rust_cam.unproject_batch(pixels)
    for i in range(len(IMAGE_PLANE_POINTS)):
        py_result = py_cam.cam_from_img(pixels[i])
        np.testing.assert_allclose(
            rust_results[i],
            py_result,
            atol=1e-8,
            err_msg=f"{model_name} unproject_batch[{i}] vs pycolmap",
        )
