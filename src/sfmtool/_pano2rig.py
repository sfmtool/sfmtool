# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Equirectangular panorama to perspective rig conversion.

Converts equirectangular (360) panoramas into perspective face images
suitable for rig-aware SfM using a standard 6-face cubemap layout.
"""

import math
from pathlib import Path

import cv2
import numpy as np

from ._sfmtool import RotQuaternion

# Per-sensor frame filename template. The `%06d` field is both the output
# filename and a `.camrig` frame field, so the same string names the extracted
# face images and the rig's image pattern.
_PANO_FRAME_PATTERN = "frame_%06d.jpg"

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
    first_pano = cv2.imread(
        str(pano_paths[0]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
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

    # Process each panorama. Faces are named by frame index — the panorama's
    # position in sorted order — so each sensor's images carry a `.camrig`
    # frame field and frames pair up across faces.
    for frame_idx, pano_path in enumerate(pano_paths):
        pano = cv2.imread(
            str(pano_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if pano is None:
            raise ValueError(f"Failed to read panorama image: {pano_path}")

        frame_name = _PANO_FRAME_PATTERN % frame_idx
        for i, (name, rotation) in enumerate(zip(face_names, rotations)):
            face = extract_perspective_face(pano, rotation, face_size)

            out_path = face_dirs[i] / frame_name
            cv2.imwrite(
                str(out_path),
                face,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )

    return len(pano_paths), face_size, face_names


def write_pano_camrig(
    camrig_path: Path,
    *,
    rig_name: str,
    face_names: list[str],
    rotations: list[RotQuaternion],
    face_size: int,
) -> None:
    """Write a six-face cubemap rig to a ``.camrig`` file.

    All six faces share one square 90°-FOV ``PINHOLE`` camera and one optical
    centre, so the camera pool holds a single entry and every translation is
    zero. ``rotations`` are the ``sensor_from_rig`` rotations in the Y-up ray
    convention ``extract_perspective_face`` uses; conjugating by the Y-flip
    (negating the x and z quaternion components) converts each to COLMAP's
    Y-down camera convention, which the ``.camrig`` format stores. Each
    sensor's image pattern is ``<face>/frame_%06d.jpg``, relative to the
    directory holding the ``.camrig`` file (the rig root).
    """
    from ._sfmtool import CameraIntrinsics, write_camrig

    # 90° FOV over a square face: half-FOV is 45°, so fx = fy = face_size / 2.
    # The principal point sits at the image centre, face_size / 2 in COLMAP's
    # pixel-coordinate convention.
    half = face_size / 2.0
    camera = CameraIntrinsics.from_dict(
        {
            "model": "PINHOLE",
            "width": face_size,
            "height": face_size,
            "parameters": {
                "focal_length_x": half,
                "focal_length_y": half,
                "principal_point_x": half,
                "principal_point_y": half,
            },
        }
    ).to_dict()

    quaternions_wxyz = np.array(
        [[q.w, -q.x, q.y, -q.z] for q in rotations], dtype=np.float64
    )
    translations_xyz = np.zeros((len(face_names), 3), dtype=np.float64)

    patterns = [f"{name}/{_PANO_FRAME_PATTERN}" for name in face_names]
    write_camrig(
        path=str(camrig_path),
        name=rig_name,
        rig_type="cubemap",
        cameras=[camera],
        sensor_image_patterns=patterns,
        camera_indexes=[0] * len(face_names),
        quaternions_wxyz=quaternions_wxyz,
        translations_xyz=translations_xyz,
    )
