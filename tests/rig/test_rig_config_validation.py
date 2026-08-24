# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict-validation tests for ``rig_config.json``.

``rig_config.json`` is COLMAP's rig config schema verbatim, so the loader
refuses anything outside that schema instead of ignoring it. See
``specs/workspace/rig-config.md``.
"""

import json
from pathlib import Path

import pytest

from sfmtool.rig.config import (
    RigConfigError,
    _load_rig_config,
    validate_rig_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KERRY_PARK_RIG_CONFIG = REPO_ROOT / "test-data" / "images" / "kerry_park"


def _valid_rig() -> list[dict]:
    """A minimal two-sensor rig that passes validation."""
    return [
        {
            "cameras": [
                {
                    "image_prefix": "left/",
                    "ref_sensor": True,
                    "camera_model_name": "PINHOLE",
                    "camera_params": [100.0, 100.0, 240.0, 240.0],
                },
                {
                    "image_prefix": "right/",
                    "ref_sensor": False,
                    "cam_from_rig_rotation": [0, 0, 1, 0],
                    "cam_from_rig_translation": [0, 0, -0.03],
                    "camera_model_name": "PINHOLE",
                    "camera_params": [100.0, 100.0, 240.0, 240.0],
                },
            ]
        }
    ]


def _write(workspace_dir: Path, data) -> Path:
    path = workspace_dir / "rig_config.json"
    path.write_text(json.dumps(data))
    return path


# --------------------------------------------------------------------------- #
# Valid configs keep loading.
# --------------------------------------------------------------------------- #


def test_checked_in_kerry_park_config_validates() -> None:
    """The shipped kerry_park rig config must keep loading unchanged."""
    rig_configs = _load_rig_config(KERRY_PARK_RIG_CONFIG)
    assert rig_configs is not None
    assert len(rig_configs) == 1
    assert len(rig_configs[0]["cameras"]) == 2


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert _load_rig_config(tmp_path) is None


def test_minimal_valid_config_round_trips(tmp_path: Path) -> None:
    _write(tmp_path, _valid_rig())
    assert _load_rig_config(tmp_path) == _valid_rig()


def test_model_only_sensor_is_valid(tmp_path: Path) -> None:
    """``camera_model_name`` without ``camera_params`` is the model-hint tier."""
    data = _valid_rig()
    for cam in data[0]["cameras"]:
        del cam["camera_params"]
    _write(tmp_path, data)
    assert _load_rig_config(tmp_path) is not None


def test_sensor_without_intrinsics_is_valid(tmp_path: Path) -> None:
    """Uncalibrated sensors are allowed; they fall back to inference."""
    data = _valid_rig()
    for cam in data[0]["cameras"]:
        del cam["camera_params"]
        del cam["camera_model_name"]
    _write(tmp_path, data)
    assert _load_rig_config(tmp_path) is not None


def test_omitted_ref_sensor_flag_defaults_to_false(tmp_path: Path) -> None:
    data = _valid_rig()
    del data[0]["cameras"][1]["ref_sensor"]
    _write(tmp_path, data)
    assert _load_rig_config(tmp_path) is not None


def test_omitted_translation_is_valid(tmp_path: Path) -> None:
    data = _valid_rig()
    del data[0]["cameras"][1]["cam_from_rig_translation"]
    _write(tmp_path, data)
    assert _load_rig_config(tmp_path) is not None


# --------------------------------------------------------------------------- #
# Unknown keys.
# --------------------------------------------------------------------------- #


def test_rig_level_camera_intrinsics_block_is_rejected(tmp_path: Path) -> None:
    """The KerryPark360 failure shape: a hand-invented rig-level intrinsics
    block that the loader used to ignore, silently falling back to EXIF."""
    data = _valid_rig()
    for cam in data[0]["cameras"]:
        del cam["camera_model_name"]
        del cam["camera_params"]
    data[0]["camera_intrinsics"] = {
        "model": "OPENCV_FISHEYE",
        "width": 960,
        "height": 960,
        "parameters": {"focal_length_x": 257.9, "focal_length_y": 257.4},
    }
    _write(tmp_path, data)

    with pytest.raises(RigConfigError) as excinfo:
        _load_rig_config(tmp_path)

    message = str(excinfo.value)
    assert "camera_intrinsics" in message
    assert "camera_model_name" in message
    assert "camera_params" in message
    assert "rig 0" in message
    assert "rig_config.json" in message


def test_unknown_sensor_level_key_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig"] = {"rotation": [1, 0, 0, 0]}
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="unknown key 'cam_from_rig'"):
        _load_rig_config(tmp_path)


def test_unknown_key_suggests_the_close_alternative(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["image_prefixes"] = "left/"
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="did you mean 'image_prefix'"):
        _load_rig_config(tmp_path)


def test_sensor_level_intrinsics_dict_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["parameters"] = {"focal_length_x": 100.0}
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="use 'camera_params'"):
        _load_rig_config(tmp_path)


def test_width_height_keys_are_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["width"] = 480
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="carries no image dimensions"):
        _load_rig_config(tmp_path)


# --------------------------------------------------------------------------- #
# Intrinsics.
# --------------------------------------------------------------------------- #


def test_wrong_camera_params_length_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["camera_params"] = [100.0, 100.0, 240.0]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="needs 4 values"):
        _load_rig_config(tmp_path)


def test_unknown_camera_model_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["camera_model_name"] = "FISHEYE"
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="unknown camera model 'FISHEYE'"):
        _load_rig_config(tmp_path)


def test_camera_params_without_model_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    del data[0]["cameras"][0]["camera_model_name"]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="requires 'camera_model_name'"):
        _load_rig_config(tmp_path)


def test_non_numeric_camera_params_are_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["camera_params"] = ["100", 100.0, 240.0, 240.0]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="'camera_params' must be numbers"):
        _load_rig_config(tmp_path)


# --------------------------------------------------------------------------- #
# Reference sensor and poses.
# --------------------------------------------------------------------------- #


def test_missing_ref_sensor_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["ref_sensor"] = False
    data[0]["cameras"][0]["cam_from_rig_rotation"] = [1, 0, 0, 0]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="no reference sensor"):
        _load_rig_config(tmp_path)


def test_two_ref_sensors_are_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["ref_sensor"] = True
    del data[0]["cameras"][1]["cam_from_rig_rotation"]
    del data[0]["cameras"][1]["cam_from_rig_translation"]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="exactly one may"):
        _load_rig_config(tmp_path)


def test_ref_sensor_carrying_a_pose_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["cam_from_rig_rotation"] = [1, 0, 0, 0]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="cam_from_rig is the identity"):
        _load_rig_config(tmp_path)


def test_non_ref_sensor_without_rotation_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    del data[0]["cameras"][1]["cam_from_rig_rotation"]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="need 'cam_from_rig_rotation'"):
        _load_rig_config(tmp_path)


def test_three_element_rotation_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig_rotation"] = [0, 1, 0]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="must be 4 numbers"):
        _load_rig_config(tmp_path)


def test_zero_quaternion_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig_rotation"] = [0, 0, 0, 0]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="all zeros"):
        _load_rig_config(tmp_path)


def test_wrong_translation_length_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig_translation"] = [0, 0]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="must be 3 numbers"):
        _load_rig_config(tmp_path)


# --------------------------------------------------------------------------- #
# Structure and prefixes.
# --------------------------------------------------------------------------- #


def test_top_level_object_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, _valid_rig()[0])

    with pytest.raises(RigConfigError, match=r"wrap the rig object in \[ \.\.\. \]"):
        _load_rig_config(tmp_path)


def test_empty_array_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, [])

    with pytest.raises(RigConfigError, match="no rigs declared"):
        _load_rig_config(tmp_path)


def test_missing_cameras_key_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, [{}])

    with pytest.raises(RigConfigError, match="missing required key 'cameras'"):
        _load_rig_config(tmp_path)


def test_empty_cameras_array_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, [{"cameras": []}])

    with pytest.raises(RigConfigError, match="non-empty array"):
        _load_rig_config(tmp_path)


def test_missing_image_prefix_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    del data[0]["cameras"][0]["image_prefix"]
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="missing required key 'image_prefix'"):
        _load_rig_config(tmp_path)


def test_backslash_image_prefix_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["image_prefix"] = "left\\"
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="must use forward slashes"):
        _load_rig_config(tmp_path)


def test_duplicate_image_prefix_across_rigs_is_rejected(tmp_path: Path) -> None:
    data = _valid_rig() + _valid_rig()
    _write(tmp_path, data)

    with pytest.raises(RigConfigError, match="distinct prefix"):
        _load_rig_config(tmp_path)


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "rig_config.json").write_text("[{,}]")

    with pytest.raises(RigConfigError, match="Invalid JSON"):
        _load_rig_config(tmp_path)


def test_error_messages_name_the_offending_location(tmp_path: Path) -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["bogus"] = 1
    path = _write(tmp_path, data)

    with pytest.raises(RigConfigError) as excinfo:
        _load_rig_config(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "rig 0, camera 1" in message


def test_validate_rig_config_returns_the_document() -> None:
    data = _valid_rig()
    assert validate_rig_config(data, "<memory>") is data
