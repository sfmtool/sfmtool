# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Equirectangular panorama to perspective rig conversion.

Converts equirectangular (360) panoramas into perspective face images
suitable for rig-aware SfM using a standard 6-face cubemap layout.
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from ._sfmtool import RotQuaternion

# Standard cubemap: 6 faces looking along +Z, +X, -Z, -X, +Y, -Y
#
# Convention: sensor_from_rig rotation. The rig frame is the front camera frame.
# So front is identity, and other faces rotate the rig frame to point in their direction.
#
# These are defined as rig_from_sensor (where each sensor looks in rig coords),
# then inverted to get sensor_from_rig.
#
# Front: looks along +Z (identity)
# Right: looks along +X (rotate -90° around Y)
# Back: looks along -Z (rotate 180° around Y)
# Left: looks along -X (rotate +90° around Y)
# Top: looks along +Y (rotate -90° around X)
# Bottom: looks along -Y (rotate +90° around X)

_CUBEMAP_FACE_NAMES = ["front", "right", "back", "left", "top", "bottom"]

_Y_AXIS = [0.0, 1.0, 0.0]
_X_AXIS = [1.0, 0.0, 0.0]


def _cubemap_rotations() -> list[RotQuaternion]:
    """Get sensor_from_rig rotations for 6 cubemap faces.

    The rig frame is defined as the front camera frame (looking along +Z).
    Each rotation transforms from rig coordinates to sensor coordinates.
    """
    # rig_from_sensor rotations, then invert to get sensor_from_rig
    rig_from_sensor = [
        RotQuaternion.identity(),  # front: +Z
        RotQuaternion.from_axis_angle(_Y_AXIS, math.radians(-90)),  # right: +X
        RotQuaternion.from_axis_angle(_Y_AXIS, math.radians(180)),  # back: -Z
        RotQuaternion.from_axis_angle(_Y_AXIS, math.radians(90)),  # left: -X
        RotQuaternion.from_axis_angle(_X_AXIS, math.radians(-90)),  # top: +Y
        RotQuaternion.from_axis_angle(_X_AXIS, math.radians(90)),  # bottom: -Y
    ]
    return [q.inverse() for q in rig_from_sensor]


def extract_perspective_face(
    equirect: np.ndarray,
    rotation: RotQuaternion,
    face_size: int,
) -> np.ndarray:
    """Extract a perspective face image from an equirectangular panorama.

    Args:
        equirect: Equirectangular image, shape (H, W) or (H, W, C).
        rotation: sensor_from_rig rotation for this face.
        face_size: Output face image size (square).

    Returns:
        Face image of shape (face_size, face_size) or (face_size, face_size, C).
    """
    h, w = equirect.shape[:2]
    f = face_size / 2.0  # focal length for 90° FOV

    # Create pixel grid for the face image
    u = np.arange(face_size, dtype=np.float64) - (face_size - 1) / 2.0
    v = np.arange(face_size, dtype=np.float64) - (face_size - 1) / 2.0
    uu, vv = np.meshgrid(u, v)

    # 3D ray directions in face-local coordinates (camera looks along +Z)
    rays = np.stack([uu / f, -vv / f, np.ones_like(uu)], axis=-1)  # (S, S, 3)

    # Transform rays to rig/world coordinates
    # rotation is sensor_from_rig, so rig_from_sensor = inverse = transpose
    R_sensor_from_rig = rotation.to_rotation_matrix()
    R_rig_from_sensor = R_sensor_from_rig.T
    rays_world = rays @ R_rig_from_sensor.T  # broadcast matmul: (S,S,3) @ (3,3)

    # Convert to spherical coordinates
    x, y, z = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    lon = np.arctan2(x, z)  # longitude: [-pi, pi], 0 = front (+Z)
    lat = np.arctan2(y, np.sqrt(x**2 + z**2))  # latitude: [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    # longitude -> horizontal: -pi maps to 0, +pi maps to W
    # latitude -> vertical: +pi/2 maps to 0 (top), -pi/2 maps to H (bottom)
    src_x = (lon / np.pi + 1.0) * 0.5 * w - 0.5
    src_y = (0.5 - lat / np.pi) * h - 0.5

    # Use cv2.remap for bilinear interpolation
    src_x = src_x.astype(np.float32)
    src_y = src_y.astype(np.float32)

    face = cv2.remap(
        equirect,
        src_x,
        src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return face


def default_face_size(pano_width: int) -> int:
    """Compute the default face size from panorama width.

    Default: pano_width / 4 (since 4 faces span the 360° equator).
    """
    return pano_width // 4


def generate_rig_config_json(
    face_names: list[str],
    prefix_base: str = "",
    rotations: list[RotQuaternion] | None = None,
    translations: list[list[float]] | None = None,
    camera_intrinsics: dict | None = None,
) -> dict:
    """Generate a COLMAP-compatible rig config entry.

    Args:
        face_names: List of face/sensor names.
        prefix_base: Workspace-relative path prefix for image directories.
            If non-empty, should end with '/'.
        rotations: Optional list of sensor_from_rig rotations (one per face).
            The ref sensor (index 0) rotation is ignored. Non-ref sensors get
            cam_from_rig_rotation written as WXYZ quaternion. These rotations
            use the Y-up ray convention (as used in extract_perspective_face);
            they are converted to COLMAP's Y-down convention via [w, -x, y, -z]
            when written.
        translations: Optional list of sensor_from_rig translations [x, y, z]
            (one per face). The ref sensor (index 0) translation is ignored.
            Non-ref sensors get cam_from_rig_translation written as [X, Y, Z].
        camera_intrinsics: Optional dict with camera intrinsics info shared by
            all sensors in this rig. Must contain "model" (e.g. "PINHOLE",
            "OPENCV_FISHEYE"). May also contain model-specific parameters.

    Returns:
        Dict with "cameras" list and optional "camera_intrinsics",
        suitable as one entry in rig_config.json.
    """
    result = {}
    if camera_intrinsics is not None:
        result["camera_intrinsics"] = camera_intrinsics
    cameras = []
    for i, name in enumerate(face_names):
        cam = {"image_prefix": f"{prefix_base}{name}/"}
        if i == 0:
            cam["ref_sensor"] = True
        else:
            cam["ref_sensor"] = False
            if rotations is not None:
                q = rotations[i]
                # Convert from Y-up extraction convention to COLMAP's Y-down
                # convention: negate x and z quaternion components.
                cam["cam_from_rig_rotation"] = [q.w, -q.x, q.y, -q.z]
            if translations is not None:
                cam["cam_from_rig_translation"] = translations[i]
        cameras.append(cam)
    result["cameras"] = cameras
    return result


def build_rig_frame_data(
    face_names: list[str],
    rotations: list[RotQuaternion],
    num_panoramas: int,
    camera_index: int = 0,
) -> dict:
    """Build rig_frame_data dict for writing into .sfmr files.

    Args:
        face_names: List of sensor names.
        rotations: List of sensor_from_rig RotQuaternion rotations.
        num_panoramas: Number of panoramas (= number of frames).
        camera_index: Camera intrinsics index for all sensors (same PINHOLE model).

    Returns:
        Dictionary matching the rig_frame_data format expected by sfmr_file.write_sfm().
    """
    num_sensors = len(face_names)
    num_images = num_panoramas * num_sensors

    # Rig metadata
    rigs_metadata = {
        "rig_count": 1,
        "sensor_count": num_sensors,
        "rigs": [
            {
                "name": "pano2rig",
                "sensor_count": num_sensors,
                "sensor_offset": 0,
                "ref_sensor_name": face_names[0],
                "sensor_names": face_names,
            }
        ],
    }

    # Sensor arrays
    sensor_camera_indexes = np.full(num_sensors, camera_index, dtype=np.uint32)
    sensor_quaternions_wxyz = np.zeros((num_sensors, 4), dtype=np.float64)
    sensor_translations_xyz = np.zeros((num_sensors, 3), dtype=np.float64)

    for i, q in enumerate(rotations):
        sensor_quaternions_wxyz[i] = q.to_wxyz_array()

    # Frames metadata
    frames_metadata = {"frame_count": num_panoramas}

    # Frame arrays
    rig_indexes = np.zeros(num_panoramas, dtype=np.uint32)  # all frames use rig 0

    # Image->sensor and image->frame mappings
    # Images are ordered: face0_pano0, face0_pano1, ..., face1_pano0, face1_pano1, ...
    # i.e., grouped by face subdirectory
    image_sensor_indexes = np.zeros(num_images, dtype=np.uint32)
    image_frame_indexes = np.zeros(num_images, dtype=np.uint32)

    for sensor_idx in range(num_sensors):
        for frame_idx in range(num_panoramas):
            img_idx = sensor_idx * num_panoramas + frame_idx
            image_sensor_indexes[img_idx] = sensor_idx
            image_frame_indexes[img_idx] = frame_idx

    return {
        "rigs_metadata": rigs_metadata,
        "sensor_camera_indexes": sensor_camera_indexes,
        "sensor_quaternions_wxyz": sensor_quaternions_wxyz,
        "sensor_translations_xyz": sensor_translations_xyz,
        "frames_metadata": frames_metadata,
        "rig_indexes": rig_indexes,
        "image_sensor_indexes": image_sensor_indexes,
        "image_frame_indexes": image_frame_indexes,
    }


def find_panorama_images(input_dir: Path) -> list[Path]:
    """Find panorama images in a directory.

    Looks for common image extensions and returns them sorted by name.
    """
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in extensions:
            images.append(p)
    return images


def convert_panoramas(
    input_dir: Path,
    output_dir: Path,
    *,
    face_size: int | None = None,
    jpeg_quality: int = 95,
) -> tuple[int, int, list[str]]:
    """Convert equirectangular panoramas to perspective face images.

    Args:
        input_dir: Directory containing equirectangular panorama images.
        output_dir: Output workspace directory.
        face_size: Face image size in pixels. If None, derived from panorama width.
        jpeg_quality: JPEG quality for output images (1-100).

    Returns:
        Tuple of (num_panoramas, face_size, face_names).
    """
    pano_paths = find_panorama_images(input_dir)
    if not pano_paths:
        raise ValueError(f"No panorama images found in {input_dir}")

    face_names = _CUBEMAP_FACE_NAMES
    rotations = _cubemap_rotations()

    # Read first panorama to determine face size
    first_pano = cv2.imread(str(pano_paths[0]))
    if first_pano is None:
        raise ValueError(f"Failed to read panorama image: {pano_paths[0]}")

    pano_h, pano_w = first_pano.shape[:2]
    if face_size is None:
        face_size = default_face_size(pano_w)

    # Create output directories
    face_dirs = []
    for name in face_names:
        d = output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        face_dirs.append(d)

    # Process each panorama
    for pano_path in pano_paths:
        pano = cv2.imread(str(pano_path))
        if pano is None:
            raise ValueError(f"Failed to read panorama image: {pano_path}")

        stem = pano_path.stem

        for i, (name, rotation) in enumerate(zip(face_names, rotations)):
            face = extract_perspective_face(pano, rotation, face_size)

            out_path = face_dirs[i] / f"{stem}.jpg"
            cv2.imwrite(
                str(out_path),
                face,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )

    return len(pano_paths), face_size, face_names


def write_rig_config(
    output_dir: Path,
    face_names: list[str],
    rotations: list[RotQuaternion] | None = None,
    translations: list[list[float]] | None = None,
    camera_intrinsics: dict | None = None,
) -> None:
    """Write rig configuration into an existing workspace.

    Appends a new rig entry to rig_config.json (COLMAP-compatible).

    Args:
        output_dir: Workspace directory.
        face_names: List of face/sensor names.
        rotations: Optional list of sensor_from_rig rotations (one per face).
        translations: Optional list of sensor_from_rig translations [x, y, z]
            (one per face).
        camera_intrinsics: Optional dict with camera intrinsics info shared by
            all sensors (e.g. {"model": "PINHOLE"}).
    """
    from ._workspace import find_workspace_for_path

    workspace_dir = find_workspace_for_path(output_dir)
    if workspace_dir is None:
        raise RuntimeError(
            f"No workspace found at or above {output_dir}. "
            f"Initialize one with 'sfm init'."
        )

    # Compute workspace-relative prefix for the output directory
    output_rel = Path(os.path.relpath(output_dir, workspace_dir))
    prefix_base = output_rel.as_posix()
    if prefix_base == ".":
        prefix_base = ""
    else:
        prefix_base = prefix_base + "/"

    # Write COLMAP-compatible rig_config.json
    # If an existing rig entry has overlapping image prefixes, replace it;
    # otherwise append as a new rig entry.
    new_rig = generate_rig_config_json(
        face_names, prefix_base, rotations, translations, camera_intrinsics
    )
    new_prefixes = {c["image_prefix"] for c in new_rig["cameras"]}
    rig_config_path = workspace_dir / "rig_config.json"
    if rig_config_path.exists():
        with open(rig_config_path) as f:
            rig_config = json.load(f)
        replaced = False
        for i, existing_rig in enumerate(rig_config):
            existing_prefixes = {c["image_prefix"] for c in existing_rig["cameras"]}
            if existing_prefixes & new_prefixes:
                rig_config[i] = new_rig
                replaced = True
                break
        if not replaced:
            rig_config.append(new_rig)
    else:
        rig_config = [new_rig]
    with open(rig_config_path, "w") as f:
        json.dump(rig_config, f, indent=2)
