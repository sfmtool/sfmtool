# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a pinhole .sfmr reconstruction to a Nerfstudio dataset directory.

Produces the layout `ns-process-data` writes from an undistorted COLMAP project:
`transforms.json`, `sparse_pc.ply`, `images/`, and downsampled `images_N/`
pyramids. The output is consumable directly by `ns-train`.
"""

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ._sfmtool import RotQuaternion, SfmrReconstruction


# Nerfstudio's conventional applied_transform: swaps Y<->Z, negates Y.
# Stored as 3x4 in transforms.json; embedded in 4x4 form here for chained matmuls.
_APPLIED_TRANSFORM_3x4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

_APPLIED_TRANSFORM_4x4 = np.vstack([_APPLIED_TRANSFORM_3x4, [0.0, 0.0, 0.0, 1.0]])


def frame_transform_matrix(
    quaternion_wxyz: np.ndarray, translation_xyz: np.ndarray
) -> np.ndarray:
    """Return the 4x4 transform_matrix Nerfstudio expects for one frame.

    Input is COLMAP-style camera-from-world (OpenCV axes: +x right, +y down,
    +z forward). Output is world-from-camera in OpenGL axes (+x right, +y up,
    +z back), composed with the standard nerfstudio applied_transform so the
    matrix is in the same post-applied space ns-process-data writes.
    """
    q = RotQuaternion(
        float(quaternion_wxyz[0]),
        float(quaternion_wxyz[1]),
        float(quaternion_wxyz[2]),
        float(quaternion_wxyz[3]),
    )
    R_cam_from_world = q.to_rotation_matrix()

    cam_from_world = np.eye(4, dtype=np.float64)
    cam_from_world[:3, :3] = np.asarray(R_cam_from_world, dtype=np.float64)
    cam_from_world[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)

    world_from_cam = np.linalg.inv(cam_from_world)
    world_from_cam[:, 1] *= -1.0  # OpenCV +y down -> OpenGL +y up
    world_from_cam[:, 2] *= -1.0  # OpenCV +z forward -> OpenGL +z back

    return _APPLIED_TRANSFORM_4x4 @ world_from_cam


def apply_transform_to_points(positions: np.ndarray) -> np.ndarray:
    """Apply the applied_transform to (N, 3) world-space points."""
    pos = np.asarray(positions, dtype=np.float64)
    return pos @ _APPLIED_TRANSFORM_3x4[:, :3].T + _APPLIED_TRANSFORM_3x4[:, 3]


def write_sparse_ply(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY of (M, 3) points and (M, 3) uint8 colors.

    Positions are written verbatim — the caller is responsible for applying
    `apply_transform_to_points` first if alignment with the camera poses is
    required.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    col = np.ascontiguousarray(colors, dtype=np.uint8)
    if pos.shape[1] != 3 or col.shape[1] != 3 or len(pos) != len(col):
        raise ValueError(
            f"positions {pos.shape} and colors {col.shape} must be (M, 3) with matching M"
        )

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pos)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pos, col):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _camera_intrinsics_dict(cam) -> dict:
    fx, fy = cam.focal_lengths
    cx, cy = cam.principal_point
    return {
        "w": int(cam.width),
        "h": int(cam.height),
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
    }


def build_transforms_json(
    recon: SfmrReconstruction, frame_basenames: list[str]
) -> dict:
    """Assemble the transforms.json dict for the given reconstruction.

    Single-camera reconstructions hoist intrinsics to the top level; multi-
    camera reconstructions emit per-frame intrinsics. `applied_transform` and
    `ply_file_path` are always present at the top level.
    """
    if len(frame_basenames) != recon.image_count:
        raise ValueError(
            f"frame_basenames count {len(frame_basenames)} != image count {recon.image_count}"
        )

    cameras = recon.cameras
    camera_indexes = np.asarray(recon.camera_indexes)
    quaternions = np.asarray(recon.quaternions_wxyz)
    translations = np.asarray(recon.translations)

    single_camera = len(cameras) == 1 or len(np.unique(camera_indexes)) == 1

    transforms: dict = {}
    if single_camera:
        transforms.update(_camera_intrinsics_dict(cameras[int(camera_indexes[0])]))

    frames = []
    for i in range(recon.image_count):
        cam_idx = int(camera_indexes[i])
        matrix = frame_transform_matrix(quaternions[i], translations[i])
        entry: dict = {
            "file_path": f"images/{frame_basenames[i]}",
            "transform_matrix": matrix.tolist(),
            "colmap_im_id": i + 1,
        }
        if not single_camera:
            entry.update(_camera_intrinsics_dict(cameras[cam_idx]))
        frames.append(entry)
    transforms["frames"] = frames

    transforms["applied_transform"] = _APPLIED_TRANSFORM_3x4.tolist()
    transforms["ply_file_path"] = "sparse_pc.ply"
    return transforms


def _place_one_image(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _build_pyramid_for_image(
    src_image: np.ndarray,
    output_dir: Path,
    basename: str,
    num_downscales: int,
    jpeg_quality: int,
) -> None:
    if num_downscales <= 0:
        return
    h, w = src_image.shape[:2]
    for level in range(1, num_downscales + 1):
        factor = 2**level
        new_w = max(1, w // factor)
        new_h = max(1, h // factor)
        downscaled = cv2.resize(src_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        level_dir = output_dir / f"images_{factor}"
        level_dir.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(level_dir / basename),
            downscaled,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(f"Failed to write pyramid image: {level_dir / basename}")


def export_to_nerfstudio(
    recon: SfmrReconstruction,
    output_dir: Path,
    *,
    num_downscales: int = 3,
    jpeg_quality: int = 95,
    include_colmap: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Write a complete nerfstudio dataset directory derived from `recon`.

    Returns a small summary dict (counts, output_dir) for the CLI to print.
    """
    for i, cam in enumerate(recon.cameras):
        if cam.has_distortion:
            raise ValueError(
                f"Camera {i} ({cam.model}) has nonzero distortion. "
                "Run `sfm undistort` first to produce a pinhole reconstruction."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = Path(recon.workspace_dir)
    image_names = recon.image_names
    basenames = [Path(name).name for name in image_names]

    seen: dict[str, int] = {}
    for i, basename in enumerate(basenames):
        if basename in seen:
            raise ValueError(
                f"Duplicate destination filename {basename!r} from images "
                f"#{seen[basename]} and #{i}."
            )
        seen[basename] = i

    for i, (image_name, basename) in enumerate(zip(image_names, basenames)):
        if progress_callback is not None:
            progress_callback(i, recon.image_count, image_name)
        src = workspace_dir / image_name
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        dst = images_dir / basename
        _place_one_image(src, dst)

        if num_downscales > 0:
            src_image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if src_image is None:
                raise RuntimeError(f"Failed to load image for pyramid: {src}")
            _build_pyramid_for_image(
                src_image, output_dir, basename, num_downscales, jpeg_quality
            )

    if progress_callback is not None:
        progress_callback(recon.image_count, recon.image_count, "")

    transforms = build_transforms_json(recon, basenames)
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=4)

    transformed_positions = apply_transform_to_points(np.asarray(recon.positions))
    write_sparse_ply(
        output_dir / "sparse_pc.ply", transformed_positions, np.asarray(recon.colors)
    )

    if include_colmap:
        from ._colmap_io import save_colmap_binary

        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        save_colmap_binary(recon, sparse_dir)

    return {
        "output_dir": str(output_dir),
        "image_count": recon.image_count,
        "point_count": recon.point_count,
        "camera_count": recon.camera_count,
        "single_camera": "fl_x" in transforms,
        "num_downscales": num_downscales,
        "include_colmap": include_colmap,
    }
