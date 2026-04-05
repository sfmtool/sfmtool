# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sfmtool._sift_file import SiftExtractionError, feature_size, xxh128_of_file

__all__ = [
    "get_default_opencv_feature_options",
    "opencv_keypoint_to_affine_shape",
    "extract_sift_with_opencv",
]


def get_default_opencv_feature_options():
    """Get default OpenCV SIFT extraction options as a dict.

    Returns:
        Dict of OpenCV SIFT options
    """
    return {
        "nfeatures": 0,
        "nOctaveLayers": 3,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6,
    }


def opencv_keypoint_to_affine_shape(keypoint):
    """Convert OpenCV KeyPoint to affine shape matrix.

    OpenCV keypoint provides size (diameter) and angle (orientation in degrees).
    The affine shape matrix transforms the unit circle to an ellipse oriented
    according to the keypoint's orientation and scaled by its radius.

    Note: OpenCV's KeyPoint.size is the DIAMETER of the meaningful neighborhood,
    but SIFT scale is the radius, so we divide by 2.

    Args:
        keypoint: OpenCV KeyPoint with .size and .angle attributes

    Returns:
        Numpy array of shape (2, 2) with affine shape matrix
    """
    angle_rad = np.radians(keypoint.angle)
    scale = keypoint.size / 2.0

    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    affine_shape = np.array(
        [
            [scale * cos_angle, -scale * sin_angle],
            [scale * sin_angle, scale * cos_angle],
        ],
        dtype=np.float32,
    )

    return affine_shape


def extract_sift_with_opencv(
    image_filename_list: list[str | Path],
    feature_options: dict,
    num_threads: int = -1,
):
    """Extract SIFT features from image files using OpenCV.

    Args:
        image_filename_list: List of absolute paths to image files
        feature_options: Dict containing OpenCV SIFT options
        num_threads: Number of threads for feature extraction (-1 uses all cores)

    Returns:
        List of tuples with (feature_tool_metadata, metadata, positions,
        affine_shapes, descriptors, thumbnail) for each image in order

    Raises:
        SiftExtractionError: If image loading fails or feature extraction fails
    """
    image_filename_list = [
        Path(os.path.normpath(os.path.abspath(p))) for p in image_filename_list
    ]

    options = feature_options

    sift = cv2.SIFT_create(
        nfeatures=options.get("nfeatures", 0),
        nOctaveLayers=options.get("nOctaveLayers", 3),
        contrastThreshold=options.get("contrastThreshold", 0.04),
        edgeThreshold=options.get("edgeThreshold", 10),
        sigma=options.get("sigma", 1.6),
    )

    def process_single_image(image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise SiftExtractionError(f"Failed to load image: {image_path}")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) == 0:
            keypoints = []
            descriptors = np.zeros((0, 128), dtype=np.uint8)
        else:
            descriptors = descriptors.astype(np.uint8)

        positions = np.array(
            [[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32
        )
        affine_shapes = np.array(
            [opencv_keypoint_to_affine_shape(kp) for kp in keypoints], dtype=np.float32
        )

        # Sort by feature size (largest first) to match COLMAP behavior
        if len(keypoints) > 0:
            sizes = feature_size(affine_shapes)
            sorted_indices = np.argsort(sizes)[::-1]
            positions = positions[sorted_indices]
            affine_shapes = affine_shapes[sorted_indices]
            descriptors = descriptors[sorted_indices]

        file_size = image_path.stat().st_size
        file_xxh128 = xxh128_of_file(image_path)

        feature_tool_metadata = {
            "feature_tool": "opencv",
            "feature_type": "sift",
            "feature_options": feature_options,
        }

        metadata = {
            "version": 1,
            "image_name": image_path.name,
            "image_file_xxh128": file_xxh128,
            "image_file_size": file_size,
            "image_width": width,
            "image_height": height,
            "feature_count": len(keypoints),
        }

        thumbnail = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

        return (
            feature_tool_metadata,
            metadata,
            positions,
            affine_shapes,
            descriptors,
            thumbnail,
        )

    if num_threads == 1:
        print(
            f"Extracting features from {len(image_filename_list)} image(s) using 1 thread"
        )
        results = [process_single_image(img_path) for img_path in image_filename_list]
    else:
        if num_threads == -1:
            max_workers = os.cpu_count()
        else:
            max_workers = num_threads

        print(
            f"Extracting features from {len(image_filename_list)} image(s) "
            f"using {max_workers} thread(s)"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_image, image_filename_list))

    return results
