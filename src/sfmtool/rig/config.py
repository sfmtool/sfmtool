# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Rig configuration loading and image-to-sensor matching.

`rig_config.json` is COLMAP's `rig_configurator` rig config verbatim — there
are no sfmtool-specific extensions. Loading is therefore **strict**: any key
this schema does not define is refused rather than ignored, because a key
sfmtool skips is a key COLMAP would skip too, and silently dropped intrinsics
turn into a degenerate solve that nothing else flags.

See `specs/workspace/rig-config.md` for the file format and semantics.
"""

import difflib
import json
from pathlib import Path

import numpy as np

from ..camera.cameras import _CAMERA_PARAM_NAMES, CAMERA_MODEL_NAMES


class RigConfigError(Exception):
    """Raised when a `rig_config.json` file is malformed or unusable."""


_RIG_KEYS = ("cameras",)

_SENSOR_KEYS = (
    "cam_from_rig_rotation",
    "cam_from_rig_translation",
    "camera_model_name",
    "camera_params",
    "image_prefix",
    "ref_sensor",
)

# Guidance for keys that are not in the schema but that authors reach for
# anyway. Keyed by the offending key name, at either nesting level.
_KEY_HINTS = {
    "camera_intrinsics": (
        "intrinsics in this file are per sensor: give each entry of 'cameras' a "
        "'camera_model_name' plus a positional 'camera_params' array"
    ),
    "intrinsics": (
        "intrinsics in this file are per sensor: give each entry of 'cameras' a "
        "'camera_model_name' plus a positional 'camera_params' array"
    ),
    "model": "use 'camera_model_name'",
    "camera_model": "use 'camera_model_name'",
    "parameters": (
        "use 'camera_params', a flat positional array in COLMAP's parameter order"
    ),
    "params": (
        "use 'camera_params', a flat positional array in COLMAP's parameter order"
    ),
    "width": "this file carries no image dimensions; they are read from the images",
    "height": "this file carries no image dimensions; they are read from the images",
    "cam_from_rig": (
        "split the pose into 'cam_from_rig_rotation' (WXYZ) and "
        "'cam_from_rig_translation'"
    ),
    "sensors": "use 'cameras'",
    "prefix": "use 'image_prefix'",
}

_SCHEMA_NOTE = (
    "rig_config.json is exactly COLMAP's rig config schema; keys outside it "
    "are refused rather than ignored"
)


def _is_number(value) -> bool:
    """True for a JSON number (bools are not numbers here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _reject_unknown_keys(entry: dict, allowed: tuple[str, ...], where: str) -> None:
    """Raise `RigConfigError` naming the first unknown key in `entry`."""
    unknown = sorted(k for k in entry if k not in allowed)
    if not unknown:
        return
    key = unknown[0]
    hint = _KEY_HINTS.get(key)
    if hint is None:
        close = difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)
        if close:
            hint = f"did you mean {close[0]!r}?"
        else:
            hint = f"valid keys here are {list(allowed)}"
    raise RigConfigError(f"{where}: unknown key {key!r}. {hint} ({_SCHEMA_NOTE})")


def _validate_sensor(cam, where: str, seen_prefixes: dict[str, str]) -> bool:
    """Validate one sensor entry. Returns whether it is the reference sensor.

    `seen_prefixes` maps every `image_prefix` already claimed, anywhere in the
    file, to the location that claimed it; it is updated in place.
    """
    if not isinstance(cam, dict):
        raise RigConfigError(f"{where}: must be an object")

    _reject_unknown_keys(cam, _SENSOR_KEYS, where)

    prefix = cam.get("image_prefix")
    if prefix is None:
        raise RigConfigError(f"{where}: missing required key 'image_prefix'")
    if not isinstance(prefix, str) or not prefix:
        raise RigConfigError(f"{where}: 'image_prefix' must be a non-empty string")
    if "\\" in prefix:
        raise RigConfigError(
            f"{where}: 'image_prefix' {prefix!r} must use forward slashes"
        )
    if prefix in seen_prefixes:
        raise RigConfigError(
            f"{where}: 'image_prefix' {prefix!r} is already used by "
            f"{seen_prefixes[prefix]}; each sensor needs a distinct prefix"
        )
    seen_prefixes[prefix] = where

    ref_sensor = cam.get("ref_sensor", False)
    if not isinstance(ref_sensor, bool):
        raise RigConfigError(f"{where}: 'ref_sensor' must be true or false")

    rotation = cam.get("cam_from_rig_rotation")
    translation = cam.get("cam_from_rig_translation")
    if ref_sensor:
        for key in ("cam_from_rig_rotation", "cam_from_rig_translation"):
            if key in cam:
                raise RigConfigError(
                    f"{where}: the reference sensor defines the rig frame, so its "
                    f"cam_from_rig is the identity; remove {key!r}"
                )
    else:
        if rotation is None:
            raise RigConfigError(
                f"{where}: non-reference sensors need 'cam_from_rig_rotation' "
                f"(a WXYZ quaternion)"
            )
        if not isinstance(rotation, list) or len(rotation) != 4:
            raise RigConfigError(
                f"{where}: 'cam_from_rig_rotation' must be 4 numbers (WXYZ), got "
                f"{rotation!r}"
            )
        if not all(_is_number(v) for v in rotation):
            raise RigConfigError(f"{where}: 'cam_from_rig_rotation' must be numbers")
        if all(float(v) == 0.0 for v in rotation):
            raise RigConfigError(
                f"{where}: 'cam_from_rig_rotation' is all zeros, which is not a "
                f"rotation; the identity is [1, 0, 0, 0]"
            )
        if translation is not None:
            if not isinstance(translation, list) or len(translation) != 3:
                raise RigConfigError(
                    f"{where}: 'cam_from_rig_translation' must be 3 numbers, got "
                    f"{translation!r}"
                )
            if not all(_is_number(v) for v in translation):
                raise RigConfigError(
                    f"{where}: 'cam_from_rig_translation' must be numbers"
                )

    model_name = cam.get("camera_model_name")
    if model_name is not None:
        if not isinstance(model_name, str):
            raise RigConfigError(f"{where}: 'camera_model_name' must be a string")
        if model_name not in _CAMERA_PARAM_NAMES:
            hint = difflib.get_close_matches(
                model_name.upper(), CAMERA_MODEL_NAMES, n=1, cutoff=0.6
            )
            suffix = (
                f"did you mean {hint[0]!r}?"
                if hint
                else f"must be one of {list(CAMERA_MODEL_NAMES)}"
            )
            raise RigConfigError(
                f"{where}: unknown camera model {model_name!r}. {suffix}"
            )

    camera_params = cam.get("camera_params")
    if camera_params is not None:
        if model_name is None:
            raise RigConfigError(
                f"{where}: 'camera_params' requires 'camera_model_name' — "
                f"COLMAP's parameter array is positional, so the model names it"
            )
        if not isinstance(camera_params, list):
            raise RigConfigError(f"{where}: 'camera_params' must be an array")
        if not all(_is_number(v) for v in camera_params):
            raise RigConfigError(f"{where}: 'camera_params' must be numbers")
        expected = _CAMERA_PARAM_NAMES[model_name]
        if len(camera_params) != len(expected):
            raise RigConfigError(
                f"{where}: 'camera_params' for {model_name} needs "
                f"{len(expected)} values ({', '.join(expected)}), got "
                f"{len(camera_params)}"
            )

    return ref_sensor


def validate_rig_config(data, source: str) -> list[dict]:
    """Validate a parsed `rig_config.json` document, returning it unchanged.

    `source` is used verbatim as the prefix of every error message — pass the
    file path. Raises `RigConfigError` on any schema violation, including keys
    the schema does not define.
    """
    if isinstance(data, dict):
        raise RigConfigError(
            f"{source}: top level must be a JSON array of rig objects; wrap the "
            f"rig object in [ ... ]"
        )
    if not isinstance(data, list):
        raise RigConfigError(f"{source}: top level must be a JSON array of rig objects")
    if not data:
        raise RigConfigError(
            f"{source}: no rigs declared; the array needs at least one rig object, "
            f"or delete the file"
        )

    seen_prefixes: dict[str, str] = {}
    for rig_idx, rig in enumerate(data):
        where = f"{source}: rig {rig_idx}"
        if not isinstance(rig, dict):
            raise RigConfigError(f"{where}: must be an object")
        _reject_unknown_keys(rig, _RIG_KEYS, where)

        cameras = rig.get("cameras")
        if cameras is None:
            raise RigConfigError(f"{where}: missing required key 'cameras'")
        if not isinstance(cameras, list) or not cameras:
            raise RigConfigError(
                f"{where}: 'cameras' must be a non-empty array of sensor entries"
            )

        ref_indexes: list[int] = []
        for sensor_idx, cam in enumerate(cameras):
            is_ref = _validate_sensor(
                cam, f"{where}, camera {sensor_idx}", seen_prefixes
            )
            if is_ref:
                ref_indexes.append(sensor_idx)
        if not ref_indexes:
            raise RigConfigError(
                f"{where}: no reference sensor; exactly one entry of 'cameras' must "
                f'set "ref_sensor": true'
            )
        if len(ref_indexes) > 1:
            raise RigConfigError(
                f'{where}: {len(ref_indexes)} sensors set "ref_sensor": true '
                f"(cameras {ref_indexes}); exactly one may"
            )

    return data


def _load_rig_config(workspace_dir: Path) -> list[dict] | None:
    """Load and validate `rig_config.json` from a workspace directory.

    Returns `None` when the file does not exist. Raises `RigConfigError` when
    it exists but violates the schema.
    """
    rig_config_path = workspace_dir / "rig_config.json"
    if not rig_config_path.exists():
        return None
    try:
        with open(rig_config_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RigConfigError(f"Invalid JSON in {rig_config_path}: {e}") from None
    return validate_rig_config(data, str(rig_config_path))


def _match_image_to_sensor(
    image_rel_path: str,
    rig_configs: list[dict],
) -> tuple[int, int] | None:
    """Match an image path to a rig and sensor using image_prefix.

    Returns:
        Tuple of (rig_index, sensor_index) if matched, None otherwise.
    """
    norm_path = image_rel_path.replace("\\", "/")
    for rig_idx, rig_config in enumerate(rig_configs):
        for sensor_idx, cam in enumerate(rig_config["cameras"]):
            prefix = cam["image_prefix"]
            if norm_path.startswith(prefix):
                return (rig_idx, sensor_idx)
    return None


def _infer_frame_key(image_rel_path: str, prefix: str) -> str:
    """Extract the frame key from an image path by removing the sensor prefix."""
    return image_rel_path.replace("\\", "/").removeprefix(prefix)


def _sensor_from_rig_pose(cam_config: dict):
    """Build a pycolmap.Rigid3d sensor_from_rig pose from a rig config camera entry.

    Per D4 of the coordinate-convention migration, ``rig_config.json`` stays in
    COLMAP convention: it mirrors COLMAP's own rig-config schema and this loader
    feeds COLMAP DB setup directly, so the pose is used verbatim with **no**
    S-conjugation (unlike the canonical ``.camrig`` path). If a consumer ever
    uses these poses on the ``.sfmr`` (canonical) side, it must convert.

    Returns pycolmap.Rigid3d or None if no pose is specified.
    """
    if "cam_from_rig_rotation" not in cam_config:
        return None

    import pycolmap

    # rig_config.json stores quaternion as WXYZ, pycolmap uses XYZW
    wxyz = cam_config["cam_from_rig_rotation"]
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)
    rotation = pycolmap.Rotation3d(xyzw)

    translation = np.array(
        cam_config.get("cam_from_rig_translation", [0.0, 0.0, 0.0]), dtype=np.float64
    )

    return pycolmap.Rigid3d(rotation, translation)
