# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera conversion utilities for pycolmap interop."""

import numpy as np

# Parameter names that represent focal length or principal point (as opposed
# to distortion coefficients). Used by callers that need to treat the two
# groups differently — e.g. resolution scaling, schema validation.
FOCAL_PRINCIPAL_PARAM_NAMES = frozenset(
    {
        "focal_length",
        "focal_length_x",
        "focal_length_y",
        "principal_point_x",
        "principal_point_y",
    }
)

_CAMERA_PARAM_NAMES = {
    "PINHOLE": [
        "focal_length_x",
        "focal_length_y",
        "principal_point_x",
        "principal_point_y",
    ],
    "SIMPLE_PINHOLE": [
        "focal_length",
        "principal_point_x",
        "principal_point_y",
    ],
    "SIMPLE_RADIAL": [
        "focal_length",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
    ],
    "RADIAL": [
        "focal_length",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
        "radial_distortion_k2",
    ],
    "OPENCV": [
        "focal_length_x",
        "focal_length_y",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
        "radial_distortion_k2",
        "tangential_distortion_p1",
        "tangential_distortion_p2",
    ],
    "OPENCV_FISHEYE": [
        "focal_length_x",
        "focal_length_y",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
        "radial_distortion_k2",
        "radial_distortion_k3",
        "radial_distortion_k4",
    ],
    "FULL_OPENCV": [
        "focal_length_x",
        "focal_length_y",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
        "radial_distortion_k2",
        "tangential_distortion_p1",
        "tangential_distortion_p2",
        "radial_distortion_k3",
        "radial_distortion_k4",
        "radial_distortion_k5",
        "radial_distortion_k6",
    ],
    "SIMPLE_RADIAL_FISHEYE": [
        "focal_length",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
    ],
    "RADIAL_FISHEYE": [
        "focal_length",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
        "radial_distortion_k2",
    ],
    "THIN_PRISM_FISHEYE": [
        "focal_length_x",
        "focal_length_y",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k1",
        "radial_distortion_k2",
        "tangential_distortion_p1",
        "tangential_distortion_p2",
        "radial_distortion_k3",
        "radial_distortion_k4",
        "thin_prism_sx1",
        "thin_prism_sy1",
    ],
    "RAD_TAN_THIN_PRISM_FISHEYE": [
        "focal_length_x",
        "focal_length_y",
        "principal_point_x",
        "principal_point_y",
        "radial_distortion_k0",
        "radial_distortion_k1",
        "radial_distortion_k2",
        "radial_distortion_k3",
        "radial_distortion_k4",
        "radial_distortion_k5",
        "tangential_distortion_p0",
        "tangential_distortion_p1",
        "thin_prism_s0",
        "thin_prism_s1",
        "thin_prism_s2",
        "thin_prism_s3",
    ],
}

# Canonical COLMAP camera-model vocabulary accepted by every `--camera-model`
# flag (`solve`, `match`, `camrig create`). Sourced from the parameter-name
# table so the commands cannot drift apart.
#
# sfmtool's own non-COLMAP models (`EQUIRECTANGULAR`, `EQUIDISTANT_FISHEYE`)
# are deliberately absent: the flag feeds `pycolmap.CameraModelId`, which only
# knows COLMAP's models.
CAMERA_MODEL_NAMES = tuple(_CAMERA_PARAM_NAMES)

# sfmtool-native (non-COLMAP) camera models, with their parameter names. These
# are storable in `.sfmr` and understood by the Rust core, but have to cross
# the COLMAP boundary as a carrier model.
EQUIDISTANT_FISHEYE = "EQUIDISTANT_FISHEYE"

# COLMAP has no distortion-free equidistant model, so `SIMPLE_RADIAL_FISHEYE`
# at `k = 0` — the identical `theta = r/f` map — carries it in both
# directions. Mirrors `sfmr-colmap`'s `EQUIDISTANT_FISHEYE_CARRIER`.
EQUIDISTANT_FISHEYE_CARRIER = "SIMPLE_RADIAL_FISHEYE"

_NATIVE_CAMERA_PARAM_NAMES = {
    EQUIDISTANT_FISHEYE: [
        "focal_length",
        "principal_point_x",
        "principal_point_y",
    ],
}


def get_intrinsic_matrix(camera) -> np.ndarray:
    """Extract intrinsic matrix K from a camera object.

    Accepts either a pycolmap.Camera or a CameraIntrinsics object.

    Args:
        camera: pycolmap.Camera or CameraIntrinsics object

    Returns:
        3x3 intrinsic matrix K
    """
    from .._sfmtool.geometry import CameraIntrinsics

    if isinstance(camera, CameraIntrinsics):
        return np.asarray(camera.intrinsic_matrix())

    model_name = camera.model.name
    params = camera.params

    if model_name in (
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "THIN_PRISM_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE",
    ):
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    elif model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model_name == "RADIAL":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model_name == "RADIAL_FISHEYE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        raise ValueError(f"Unsupported camera model: {model_name}")

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def colmap_camera_from_intrinsics(camera_meta, *, width=None, height=None):
    """Convert CameraIntrinsics to pycolmap.Camera object.

    An sfmtool-native model is exported through its COLMAP carrier:
    `EQUIDISTANT_FISHEYE` becomes `SIMPLE_RADIAL_FISHEYE` with `k = 0`, which
    parameterizes the identical map (see `EQUIDISTANT_FISHEYE_CARRIER`).

    Args:
        camera_meta: CameraIntrinsics object
        width: Image width in pixels. If None, uses width from camera_meta.
        height: Image height in pixels. If None, uses height from camera_meta.

    Returns:
        pycolmap.Camera object
    """
    import pycolmap

    d = camera_meta.to_dict()
    model = d["model"]
    params = d["parameters"]

    if width is None:
        width = camera_meta.width
    if height is None:
        height = camera_meta.height

    if model == EQUIDISTANT_FISHEYE:
        model = EQUIDISTANT_FISHEYE_CARRIER
        # The carrier's extra parameter is the zero the native model implies.
        params = dict(params)
        params.setdefault("radial_distortion_k1", 0.0)

    param_names = _CAMERA_PARAM_NAMES.get(model)
    if param_names is None:
        raise ValueError(f"Unsupported camera model: {model}")

    param_list = [params[name] for name in param_names]

    return pycolmap.Camera(
        model=model,
        width=width,
        height=height,
        params=param_list,
    )


def pycolmap_camera_to_intrinsics(camera, *, claim_native=False):
    """Convert a pycolmap.Camera to a CameraIntrinsics object.

    With `claim_native=True`, a COLMAP camera whose parameters make it exactly
    one of sfmtool's native models is relabelled as that model: a
    `SIMPLE_RADIAL_FISHEYE` whose `k` is **exactly** `0.0` becomes
    `EQUIDISTANT_FISHEYE` with `k` dropped. Any other `k`, however small, is
    left alone. This is the import half of `colmap_camera_from_intrinsics`'s
    export, and mirrors `sfmr-colmap`'s `claim_native_camera_model`.

    It is off by default because the same conversion also builds the *initial*
    camera for a COLMAP solve, where the caller's requested model (from
    `--camera-model` or a `camera_config.json`) must survive verbatim — a
    freshly-initialized `SIMPLE_RADIAL_FISHEYE` has `k = 0` and is still meant
    to be refined as one. Callers importing a *solved* reconstruction pass
    `claim_native=True`.
    """
    from .._sfmtool.geometry import CameraIntrinsics

    model_name = camera.model.name
    params = camera.params
    names = _CAMERA_PARAM_NAMES.get(
        model_name, [f"param_{i}" for i in range(len(params))]
    )
    parameters = {name: float(value) for name, value in zip(names, params)}

    if (
        claim_native
        and model_name == EQUIDISTANT_FISHEYE_CARRIER
        and parameters.get("radial_distortion_k1") == 0.0
    ):
        model_name = EQUIDISTANT_FISHEYE
        parameters = {
            name: parameters[name] for name in _NATIVE_CAMERA_PARAM_NAMES[model_name]
        }

    return CameraIntrinsics.from_dict(
        {
            "model": model_name,
            "width": camera.width,
            "height": camera.height,
            "parameters": parameters,
        }
    )
