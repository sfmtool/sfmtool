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

import pycolmap

from ._cameras import colmap_camera_from_intrinsics, pycolmap_camera_to_intrinsics
from ._sfmtool import SfmrReconstruction


def undistort_reconstruction_images(
    recon: SfmrReconstruction,
    output_dir: Path,
    progress_callback=None,
) -> tuple[int, str]:
    """
    Undistort all images in a reconstruction using their camera parameters.

    Images are saved with the same filenames in the specified output directory.
    Camera parameters for undistorted images are saved to undistorted_cameras.json.

    Args:
        recon: The SfmrReconstruction containing camera parameters and image metadata
        output_dir: Directory where undistorted images will be saved
        progress_callback: Optional callback(current, total, image_name) for progress

    Returns:
        Tuple of (number_of_images_processed, output_directory_path)
    """
    workspace_dir = Path(recon.workspace_dir)
    image_names = recon.image_names
    cameras_meta = recon.cameras
    camera_indexes = recon.camera_indexes

    image_count = len(image_names)
    print(f"Undistorting {image_count} images...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track undistorted cameras (may differ from original if dimensions change)
    undistorted_cameras = []
    undistorted_image_sizes = []

    # Process each image
    for i, image_name in enumerate(image_names):
        if progress_callback:
            progress_callback(i, image_count, image_name)

        # Construct path to original image file
        image_path = workspace_dir / image_name

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Get camera for this image
        cam_idx = camera_indexes[i]
        cam_meta = cameras_meta[cam_idx]

        # Load image
        bitmap = pycolmap.Bitmap.read(str(image_path), as_rgb=True)
        if bitmap is None:
            print(f"Warning: Failed to load image: {image_path}")
            continue

        # Get image dimensions
        array = bitmap.to_array()
        height, width = array.shape[:2]

        # Create pycolmap Camera (use actual image dimensions, not metadata dimensions)
        camera = colmap_camera_from_intrinsics(cam_meta, width=width, height=height)

        # Configure undistortion options - force output size to match input
        options = pycolmap.UndistortCameraOptions()
        options.min_scale = 1.0
        options.max_scale = 1.0

        # Undistort
        try:
            undistorted_bitmap, undistorted_camera = pycolmap.undistort_image(
                options, bitmap, camera
            )
        except Exception as e:
            print(f"Error undistorting {image_name}: {e}")
            continue

        # Save undistorted image using canonical POSIX path (preserves directory structure)
        output_path = output_dir / image_name

        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = undistorted_bitmap.write(str(output_path))
        if not success:
            print(f"Warning: Failed to save undistorted image: {output_path}")
            continue

        # Store undistorted camera info
        undistorted_cameras.append(pycolmap_camera_to_intrinsics(undistorted_camera))
        undistorted_image_sizes.append(
            {
                "original_path": image_name,
                "undistorted_path": image_name,
                "width": undistorted_camera.width,
                "height": undistorted_camera.height,
            }
        )

        if i < 3 or (i + 1) % 10 == 0:
            print(
                f"  [{i + 1}/{image_count}] {image_name}: "
                f"{width}x{height} -> {undistorted_camera.width}x{undistorted_camera.height}"
            )

    if progress_callback:
        progress_callback(image_count, image_count, "")

    # Save undistorted camera metadata
    camera_metadata = {
        "cameras": [cam.to_dict() for cam in undistorted_cameras],
        "image_sizes": undistorted_image_sizes,
        "note": "Cameras for undistorted images. All distortion parameters removed (PINHOLE model).",
    }

    camera_json_path = output_dir / "undistorted_cameras.json"
    with open(camera_json_path, "w") as f:
        json.dump(camera_metadata, f, indent=2)

    print(f"\nUndistorted {image_count} images")
    print(f"Output directory: {output_dir}")
    print(f"Camera metadata saved to: {camera_json_path}")

    return image_count, str(output_dir)
