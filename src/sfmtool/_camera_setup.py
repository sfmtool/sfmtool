# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera model inference and descriptor wrapping for COLMAP database setup."""

from pathlib import Path

import numpy as np
import pycolmap

from ._camera_config import CameraConfigError, CameraConfigResolver
from ._cameras import (
    _CAMERA_PARAM_NAMES,
    FOCAL_PRINCIPAL_PARAM_NAMES,
    pycolmap_camera_to_intrinsics,
)

_ASPECT_RATIO_TOLERANCE = 1e-3


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


def _distortion_param_names(model: str) -> list[str]:
    """Return the distortion-parameter names for a camera model."""
    return [
        name
        for name in _CAMERA_PARAM_NAMES[model]
        if name not in FOCAL_PRINCIPAL_PARAM_NAMES
    ]


def build_intrinsics_from_camera_config(
    camera_config: dict | None,
    image_path: Path,
    camera_model_override: str | None,
):
    """Build a `CameraIntrinsics` for an image, honoring `camera_config`.

    Returns `(intrinsics, treat_as_prior)`. `treat_as_prior` is `True` only
    when the camera_config supplies a full calibration block (model + size +
    parameters); the caller should set `has_prior_focal_length=True` on the
    resulting `pycolmap.Camera`.

    Branches:

    - `camera_config is None` — fall back to EXIF inference (with optional
      `camera_model_override`).
    - `camera_config` has only `model` — same as None but use the file's
      model as the override.
    - `camera_config` has distortion-only `parameters` (no focal/principal)
      — infer focal+principal from EXIF, overlay distortion coefficients.
    - `camera_config` has full `parameters + width + height` — build directly,
      scaling focal+principal if the actual image resolution differs from the
      calibration resolution (uniform scale only). Aspect-ratio mismatch
      raises `CameraConfigError`.
    """
    if camera_config is None:
        cam = _infer_camera(image_path, camera_model_override)
        return pycolmap_camera_to_intrinsics(cam), False

    model = camera_config["model"]
    parameters = camera_config.get("parameters")
    has_full_size = "width" in camera_config and "height" in camera_config

    has_focal_or_principal = False
    if parameters is not None:
        has_focal_or_principal = any(
            name in parameters for name in FOCAL_PRINCIPAL_PARAM_NAMES
        )

    # Model-only branch
    if parameters is None:
        cam = _infer_camera(image_path, model)
        return pycolmap_camera_to_intrinsics(cam), False

    # Distortion-only branch
    if not has_focal_or_principal:
        cam = _infer_camera(image_path, model)
        intrinsics = pycolmap_camera_to_intrinsics(cam)
        intrinsics_dict = intrinsics.to_dict()
        for name in _distortion_param_names(model):
            intrinsics_dict["parameters"][name] = float(parameters.get(name, 0.0))
        from ._sfmtool import CameraIntrinsics

        return CameraIntrinsics.from_dict(intrinsics_dict), False

    # Full-block branch — must include width and height (validated in load)
    if not has_full_size:
        raise CameraConfigError(
            "camera_config: 'width' and 'height' are required when "
            "'parameters' includes a focal length or principal point"
        )

    calib_w = int(camera_config["width"])
    calib_h = int(camera_config["height"])
    actual_w, actual_h = _read_image_size(image_path)

    if (actual_w, actual_h) == (calib_w, calib_h):
        scale = 1.0
    else:
        actual_aspect = actual_w / actual_h
        calib_aspect = calib_w / calib_h
        if abs(actual_aspect - calib_aspect) >= _ASPECT_RATIO_TOLERANCE:
            raise CameraConfigError(
                f"camera_config aspect mismatch: actual {actual_w}x{actual_h} "
                f"(aspect {actual_aspect:.4f}) vs calibration "
                f"{calib_w}x{calib_h} (aspect {calib_aspect:.4f})"
            )
        scale = actual_w / calib_w

    scaled_params: dict[str, float] = {}
    for name in _CAMERA_PARAM_NAMES[model]:
        value = float(parameters[name])
        if name in FOCAL_PRINCIPAL_PARAM_NAMES:
            scaled_params[name] = value * scale
        else:
            scaled_params[name] = value

    from ._sfmtool import CameraIntrinsics

    intrinsics = CameraIntrinsics.from_dict(
        {
            "model": model,
            "width": actual_w,
            "height": actual_h,
            "parameters": scaled_params,
        }
    )
    return intrinsics, True


def _read_image_size(image_path: Path) -> tuple[int, int]:
    """Return `(width, height)` of an image file in pixels."""
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise CameraConfigError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def intrinsics_for_image(
    image_path: Path,
    camera_config_resolver: CameraConfigResolver | None,
    camera_model_override: str | None,
):
    """Resolve intrinsics for an image, consulting `camera_config.json`.

    `camera_config_resolver` may be `None` (no workspace), in which case the function
    behaves as if no camera_config exists. Returns `(intrinsics, prior_flag)`.
    """
    camera_config: dict | None = None
    if camera_config_resolver is not None:
        resolved = camera_config_resolver.resolve_for_image(Path(image_path))
        if resolved is not None:
            camera_config = resolved[1]
    return build_intrinsics_from_camera_config(
        camera_config, Path(image_path), camera_model_override
    )


def _check_camera_model_conflict(
    image_paths,
    camera_config_resolver: CameraConfigResolver | None,
    camera_model_override: str | None,
) -> None:
    """Raise click.UsageError if any image resolves a camera_config.json AND
    `--camera-model` was supplied. The two mechanisms are not allowed to mix.
    """
    if camera_model_override is None or camera_config_resolver is None:
        return
    import click

    for image_path in image_paths:
        resolved = camera_config_resolver.resolve_for_image(Path(image_path))
        if resolved is not None:
            config_path, _ = resolved
            raise click.UsageError(
                f"--camera-model cannot be used together with "
                f"camera_config.json. Image {image_path} resolves to "
                f"{config_path}. Either remove --camera-model or delete the "
                f"camera_config.json."
            )
