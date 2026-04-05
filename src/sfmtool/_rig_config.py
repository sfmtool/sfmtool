# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Rig configuration loading and image-to-sensor matching."""

import json
from pathlib import Path

import numpy as np


def _load_rig_config(workspace_dir: Path) -> list[dict] | None:
    """Load rig_config.json from a workspace directory, if it exists."""
    rig_config_path = workspace_dir / "rig_config.json"
    if not rig_config_path.exists():
        return None
    with open(rig_config_path) as f:
        return json.load(f)


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
