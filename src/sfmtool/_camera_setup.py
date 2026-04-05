# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera model inference and descriptor wrapping for COLMAP database setup."""

from pathlib import Path

import numpy as np
import pycolmap


def _wrap_descriptors(descriptors):
    """Wrap numpy descriptors as FeatureDescriptors for pycolmap 4.x."""
    if isinstance(descriptors, np.ndarray):
        return pycolmap.FeatureDescriptors(
            data=descriptors, type=pycolmap.FeatureExtractorType.SIFT
        )
    return descriptors


def _infer_camera(
    image_path: str | Path, camera_model: str | None = None
) -> pycolmap.Camera:
    """Infer a camera from an image, optionally overriding the camera model."""
    cam = pycolmap.infer_camera_from_image(str(image_path))
    if camera_model is not None:
        target_model = getattr(pycolmap.CameraModelId, camera_model.upper())
        # Build new params for the target model using the inferred focal length and principal point
        old_params = cam.params
        if cam.model.name in (
            "SIMPLE_PINHOLE",
            "SIMPLE_RADIAL",
            "SIMPLE_RADIAL_FISHEYE",
        ):
            f, cx, cy = old_params[0], old_params[1], old_params[2]
            fx, fy = f, f
        else:
            fx, fy, cx, cy = old_params[0], old_params[1], old_params[2], old_params[3]

        # For fisheye models, use a more appropriate initial focal length.
        # The default EXIF-based estimate (1.2x image size) assumes rectilinear projection,
        # which is far too large for equidistant fisheye lenses.
        # A ~190 deg FOV equidistant fisheye has f ~ image_size / pi.
        if target_model.name in (
            "OPENCV_FISHEYE",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
            "THIN_PRISM_FISHEYE",
            "RAD_TAN_THIN_PRISM_FISHEYE",
        ):
            fisheye_f = min(cam.width, cam.height) / 3.14159
            fx, fy = fisheye_f, fisheye_f

        new_cam = pycolmap.Camera()
        new_cam.model = target_model
        new_cam.width = cam.width
        new_cam.height = cam.height

        model_name = target_model.name
        if model_name in ("SIMPLE_RADIAL_FISHEYE", "SIMPLE_RADIAL"):
            new_cam.params = [fx, cx, cy, 0]
        elif model_name in ("RADIAL_FISHEYE", "RADIAL"):
            new_cam.params = [fx, cx, cy, 0, 0]
        elif model_name == "OPENCV_FISHEYE":
            new_cam.params = [fx, fy, cx, cy, 0, 0, 0, 0]
        elif model_name == "OPENCV":
            new_cam.params = [fx, fy, cx, cy, 0, 0, 0, 0]
        elif model_name == "SIMPLE_PINHOLE":
            new_cam.params = [fx, cx, cy]
        elif model_name == "PINHOLE":
            new_cam.params = [fx, fy, cx, cy]
        elif model_name == "THIN_PRISM_FISHEYE":
            new_cam.params = [fx, fy, cx, cy, 0, 0, 0, 0, 0, 0, 0, 0]
        elif model_name == "RAD_TAN_THIN_PRISM_FISHEYE":
            new_cam.params = [fx, fy, cx, cy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif model_name == "FULL_OPENCV":
            new_cam.params = [fx, fy, cx, cy, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raise ValueError(f"Unsupported camera model: {model_name}")

        cam = new_cam
    return cam
