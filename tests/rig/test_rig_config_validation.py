# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict-validation tests for ``rig_config.json``, the input of
``sfm camrig cp``.

``rig_config.json`` is COLMAP's rig config schema verbatim, so the validator
refuses anything outside that schema instead of ignoring it. See
``specs/workspace/rig-config.md``.
"""

import copy

import pytest

from sfmtool.rig.config import RigConfigError, validate_rig_config


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


# What the messages name the file as.
SOURCE = "ws/rig_config.json"


# --------------------------------------------------------------------------- #
# Valid configs pass.
# --------------------------------------------------------------------------- #


def test_minimal_valid_config_round_trips() -> None:
    assert validate_rig_config(_valid_rig(), SOURCE) == _valid_rig()


def test_model_only_sensor_is_valid() -> None:
    """``camera_model_name`` without ``camera_params`` is the model-hint tier."""
    data = _valid_rig()
    for cam in data[0]["cameras"]:
        del cam["camera_params"]
    expected = copy.deepcopy(data)
    assert validate_rig_config(data, SOURCE) == expected


def test_sensor_without_intrinsics_is_valid() -> None:
    """Uncalibrated sensors are allowed; they fall back to inference."""
    data = _valid_rig()
    for cam in data[0]["cameras"]:
        del cam["camera_params"]
        del cam["camera_model_name"]
    expected = copy.deepcopy(data)
    assert validate_rig_config(data, SOURCE) == expected


def test_omitted_ref_sensor_flag_defaults_to_false() -> None:
    """A sensor with no ``ref_sensor`` flag is a non-reference sensor. The
    validator applies that default when it counts the reference sensors and
    checks their poses, and returns the document without writing the flag in.
    ``sfm camrig cp`` applies the same default when it orders and poses the
    sensors, which ``test_each_rig_of_the_file_becomes_a_camrig`` in
    ``tests/camrig/test_camrig_cp_rig_config.py`` checks."""
    data = _valid_rig()
    del data[0]["cameras"][1]["ref_sensor"]
    expected = copy.deepcopy(data)
    assert validate_rig_config(data, SOURCE) == expected

    # Without its pose, the unflagged sensor is refused as a non-reference
    # sensor; with a default of true it would be a second reference sensor.
    del data[0]["cameras"][1]["cam_from_rig_rotation"]
    del data[0]["cameras"][1]["cam_from_rig_translation"]
    with pytest.raises(RigConfigError, match="need 'cam_from_rig_rotation'"):
        validate_rig_config(data, SOURCE)


def test_omitted_translation_is_valid() -> None:
    data = _valid_rig()
    del data[0]["cameras"][1]["cam_from_rig_translation"]
    expected = copy.deepcopy(data)
    assert validate_rig_config(data, SOURCE) == expected


# --------------------------------------------------------------------------- #
# Unknown keys.
# --------------------------------------------------------------------------- #


def test_rig_level_camera_intrinsics_block_is_rejected() -> None:
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

    with pytest.raises(RigConfigError) as excinfo:
        validate_rig_config(data, SOURCE)

    message = str(excinfo.value)
    assert "camera_intrinsics" in message
    assert "camera_model_name" in message
    assert "camera_params" in message
    assert "rig 0" in message
    assert "rig_config.json" in message


def test_unknown_sensor_level_key_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig"] = {"rotation": [1, 0, 0, 0]}

    with pytest.raises(RigConfigError, match="unknown key 'cam_from_rig'"):
        validate_rig_config(data, SOURCE)


def test_unknown_key_suggests_the_close_alternative() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["image_prefixes"] = "left/"

    with pytest.raises(RigConfigError, match="did you mean 'image_prefix'"):
        validate_rig_config(data, SOURCE)


def test_sensor_level_intrinsics_dict_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["parameters"] = {"focal_length_x": 100.0}

    with pytest.raises(RigConfigError, match="use 'camera_params'"):
        validate_rig_config(data, SOURCE)


def test_width_height_keys_are_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["width"] = 480

    with pytest.raises(RigConfigError, match="carries no image dimensions"):
        validate_rig_config(data, SOURCE)


# --------------------------------------------------------------------------- #
# Intrinsics.
# --------------------------------------------------------------------------- #


def test_wrong_camera_params_length_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["camera_params"] = [100.0, 100.0, 240.0]

    with pytest.raises(RigConfigError, match="needs 4 values"):
        validate_rig_config(data, SOURCE)


def test_unknown_camera_model_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["camera_model_name"] = "FISHEYE"

    with pytest.raises(RigConfigError, match="unknown camera model 'FISHEYE'"):
        validate_rig_config(data, SOURCE)


def test_camera_params_without_model_is_rejected() -> None:
    data = _valid_rig()
    del data[0]["cameras"][0]["camera_model_name"]

    with pytest.raises(RigConfigError, match="requires 'camera_model_name'"):
        validate_rig_config(data, SOURCE)


def test_non_numeric_camera_params_are_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["camera_params"] = ["100", 100.0, 240.0, 240.0]

    with pytest.raises(RigConfigError, match="'camera_params' must be numbers"):
        validate_rig_config(data, SOURCE)


# --------------------------------------------------------------------------- #
# Reference sensor and poses.
# --------------------------------------------------------------------------- #


def test_missing_ref_sensor_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["ref_sensor"] = False
    data[0]["cameras"][0]["cam_from_rig_rotation"] = [1, 0, 0, 0]

    with pytest.raises(RigConfigError, match="no reference sensor"):
        validate_rig_config(data, SOURCE)


def test_two_ref_sensors_are_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["ref_sensor"] = True
    del data[0]["cameras"][1]["cam_from_rig_rotation"]
    del data[0]["cameras"][1]["cam_from_rig_translation"]

    with pytest.raises(RigConfigError, match="exactly one may"):
        validate_rig_config(data, SOURCE)


def test_ref_sensor_carrying_a_pose_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["cam_from_rig_rotation"] = [1, 0, 0, 0]

    with pytest.raises(RigConfigError, match="cam_from_rig is the identity"):
        validate_rig_config(data, SOURCE)


def test_non_ref_sensor_without_rotation_is_rejected() -> None:
    data = _valid_rig()
    del data[0]["cameras"][1]["cam_from_rig_rotation"]

    with pytest.raises(RigConfigError, match="need 'cam_from_rig_rotation'"):
        validate_rig_config(data, SOURCE)


def test_three_element_rotation_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig_rotation"] = [0, 1, 0]

    with pytest.raises(RigConfigError, match="must be 4 numbers"):
        validate_rig_config(data, SOURCE)


def test_zero_quaternion_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig_rotation"] = [0, 0, 0, 0]

    with pytest.raises(RigConfigError, match="all zeros"):
        validate_rig_config(data, SOURCE)


def test_wrong_translation_length_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["cam_from_rig_translation"] = [0, 0]

    with pytest.raises(RigConfigError, match="must be 3 numbers"):
        validate_rig_config(data, SOURCE)


# --------------------------------------------------------------------------- #
# Structure and prefixes.
# --------------------------------------------------------------------------- #


def test_top_level_object_is_rejected() -> None:
    data = _valid_rig()[0]

    with pytest.raises(RigConfigError, match=r"wrap the rig object in \[ \.\.\. \]"):
        validate_rig_config(data, SOURCE)


def test_empty_array_is_rejected() -> None:
    data = []

    with pytest.raises(RigConfigError, match="no rigs declared"):
        validate_rig_config(data, SOURCE)


def test_missing_cameras_key_is_rejected() -> None:
    data = [{}]

    with pytest.raises(RigConfigError, match="missing required key 'cameras'"):
        validate_rig_config(data, SOURCE)


def test_empty_cameras_array_is_rejected() -> None:
    data = [{"cameras": []}]

    with pytest.raises(RigConfigError, match="non-empty array"):
        validate_rig_config(data, SOURCE)


def test_missing_image_prefix_is_rejected() -> None:
    data = _valid_rig()
    del data[0]["cameras"][0]["image_prefix"]

    with pytest.raises(RigConfigError, match="missing required key 'image_prefix'"):
        validate_rig_config(data, SOURCE)


def test_backslash_image_prefix_is_rejected() -> None:
    data = _valid_rig()
    data[0]["cameras"][0]["image_prefix"] = "left\\"

    with pytest.raises(RigConfigError, match="must use forward slashes"):
        validate_rig_config(data, SOURCE)


def test_duplicate_image_prefix_across_rigs_is_rejected() -> None:
    data = _valid_rig() + _valid_rig()

    with pytest.raises(RigConfigError, match="distinct prefix"):
        validate_rig_config(data, SOURCE)


def test_error_messages_name_the_offending_location() -> None:
    data = _valid_rig()
    data[0]["cameras"][1]["bogus"] = 1

    with pytest.raises(RigConfigError) as excinfo:
        validate_rig_config(data, SOURCE)

    message = str(excinfo.value)
    assert SOURCE in message
    assert "rig 0, camera 1" in message


def test_validate_rig_config_returns_the_document() -> None:
    data = _valid_rig()
    assert validate_rig_config(data, "<memory>") is data
