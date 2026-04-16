# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""
Undistort images from an SfM reconstruction using camera parameters.

For each image in a reconstruction, this module loads the corresponding camera
model and undistorts the image, saving the result to a specified output directory.
"""

import json
import os
from pathlib import Path

import cv2

from ._sfmtool import SfmrReconstruction, WarpMap


def undistort_reconstruction_images(
    recon: SfmrReconstruction,
    output_dir: Path,
    fit: str = "inside",
    resampling_filter: str = "aniso",
    progress_callback=None,
) -> tuple[int, str]:
    """
    Undistort all images in a reconstruction using their camera parameters.

    Images are saved with the same filenames in the specified output directory.
    Camera parameters for undistorted images are saved to undistorted_cameras.json.

    Args:
        recon: The SfmrReconstruction containing camera parameters and image metadata
        output_dir: Directory where undistorted images will be saved
        fit: "inside" (no black borders) or "outside" (no cropping)
        resampling_filter: "aniso" (anisotropic, higher quality) or "bilinear"
        progress_callback: Optional callback(current, total, image_name) for progress

    Returns:
        Tuple of (number_of_images_processed, output_directory_path)
    """
    workspace_dir = Path(recon.workspace_dir)
    image_names = recon.image_names
    cameras_meta = recon.cameras
    camera_indexes = recon.camera_indexes

    image_count = len(image_names)
    print(
        f"Undistorting {image_count} images (fit={fit}, filter={resampling_filter})..."
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build pinhole cameras and warp maps per unique camera (not per image).
    pinhole_cameras = {}
    warp_maps = {}

    # Track per-image metadata
    image_entries = []

    # Process each image
    for i, image_name in enumerate(image_names):
        if progress_callback:
            progress_callback(i, image_count, image_name)

        # Construct path to original image file
        image_path = workspace_dir / image_name

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get camera for this image
        cam_idx = camera_indexes[i]
        cam_meta = cameras_meta[cam_idx]

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            continue

        height, width = image.shape[:2]

        # Build (or reuse) the pinhole camera and warp map for this camera index.
        if cam_idx not in pinhole_cameras:
            if fit == "outside":
                pinhole = cam_meta.best_fit_outside_pinhole(width, height)
            else:
                pinhole = cam_meta.best_fit_inside_pinhole(width, height)
            pinhole_cameras[cam_idx] = pinhole
            warp_maps[cam_idx] = WarpMap.from_cameras(src=cam_meta, dst=pinhole)

        warp = warp_maps[cam_idx]

        # Warp the image
        if resampling_filter == "aniso":
            undistorted = warp.remap_aniso(image)
        else:
            undistorted = warp.remap_bilinear(image)

        # Save undistorted image using canonical POSIX path (preserves directory structure)
        output_path = output_dir / image_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), undistorted)
        if not success:
            print(f"Warning: Failed to save undistorted image: {output_path}")
            continue

        image_entries.append({"name": image_name, "camera_index": cam_idx})

        if i < 3 or (i + 1) % 10 == 0:
            pinhole = pinhole_cameras[cam_idx]
            print(
                f"  [{i + 1}/{image_count}] {image_name}: "
                f"{width}x{height} -> {pinhole.width}x{pinhole.height}"
            )

    if progress_callback:
        progress_callback(image_count, image_count, "")

    # Build deduplicated camera list keyed by original camera index.
    # Remap to contiguous indices for the JSON output.
    sorted_cam_idxs = sorted(pinhole_cameras)
    cam_idx_to_json_idx = {idx: j for j, idx in enumerate(sorted_cam_idxs)}
    camera_metadata = {
        "cameras": [pinhole_cameras[idx].to_dict() for idx in sorted_cam_idxs],
        "images": [
            {"name": e["name"], "camera_index": cam_idx_to_json_idx[e["camera_index"]]}
            for e in image_entries
        ],
    }

    camera_json_path = output_dir / "undistorted_cameras.json"
    with open(camera_json_path, "w") as f:
        json.dump(camera_metadata, f, indent=2)

    print(f"\nUndistorted {image_count} images")
    print(f"Output directory: {output_dir}")
    print(f"Camera metadata saved to: {camera_json_path}")

    return image_count, str(output_dir)
