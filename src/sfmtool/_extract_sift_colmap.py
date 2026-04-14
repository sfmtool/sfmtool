# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from sfmtool._sift_file import SiftExtractionError, feature_size, xxh128_of_file

__all__ = [
    "get_colmap_feature_options",
    "extract_sift_with_colmap",
    "read_colmap_db_sift",
]


def get_colmap_feature_options(
    *,
    domain_size_pooling=False,
    max_image_size=4096,
    max_num_features=None,
    estimate_affine_shape=False,
):
    """Get COLMAP feature extraction options as a flat dict.

    Args:
        domain_size_pooling: Enable domain size pooling for SIFT (default: False)
        max_image_size: Maximum image size for processing (default: 4096)
        max_num_features: Maximum number of features to extract per image.
                         If None, uses COLMAP default (8192).
        estimate_affine_shape: Enable affine shape estimation (default: False).
                              Incompatible with GPU SIFT.

    Returns:
        Dict of feature options that affect feature output
    """
    sift_options = pycolmap.SiftExtractionOptions(
        estimate_affine_shape=estimate_affine_shape,
        domain_size_pooling=domain_size_pooling,
    )
    if max_num_features is not None:
        sift_options.max_num_features = max_num_features

    sift_dict = sift_options.todict()
    # Convert enums to their names
    sift_dict = {k: (v.name if hasattr(v, "name") else v) for k, v in sift_dict.items()}

    # Extract the relevant sift options into a flat dict
    _sift_keys = {
        "max_num_features",
        "domain_size_pooling",
        "estimate_affine_shape",
        "peak_threshold",
        "edge_threshold",
        "upright",
        "normalization",
    }
    # Include DSP options when domain_size_pooling is enabled
    if domain_size_pooling:
        _sift_keys |= {"dsp_min_scale", "dsp_max_scale", "dsp_num_scales"}

    result = {k: v for k, v in sift_dict.items() if k in _sift_keys}
    result["max_image_size"] = max_image_size

    # Use None for max_num_features when it's the COLMAP default
    if max_num_features is None:
        result["max_num_features"] = None

    return result


def extract_sift_with_colmap(
    image_filename_list: list[str | Path],
    feature_options: dict,
    num_threads: int = -1,
):
    """Extract SIFT features from image files using COLMAP.

    Creates a temporary COLMAP database, extracts SIFT features, and converts
    them to the standard .sift format.

    Args:
        image_filename_list: List of absolute paths to image files
        feature_options: Dict containing feature extraction options
        num_threads: Number of threads for feature extraction (-1 uses all cores)

    Returns:
        List of tuples with (feature_tool_metadata, metadata, positions,
        affine_shapes, descriptors, thumbnail) for each image in order

    Raises:
        SiftExtractionError: If image is not in database or features have wrong shape
    """
    image_filename_list = [
        Path(os.path.normpath(os.path.abspath(p))) for p in image_filename_list
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        db_file = tmpdir_path / "colmap_db.db"

        # Find common path (this will fail if paths are on different drives on Windows)
        image_path = Path(os.path.commonpath(image_filename_list))

        if image_path.is_file():
            image_path = image_path.parent

        image_names = [
            p.relative_to(image_path).as_posix() for p in image_filename_list
        ]

        # Map flat feature_options back to pycolmap's nested structure
        _sift_option_keys = {
            "max_num_features",
            "domain_size_pooling",
            "estimate_affine_shape",
            "peak_threshold",
            "edge_threshold",
            "upright",
            "normalization",
            "dsp_min_scale",
            "dsp_max_scale",
            "dsp_num_scales",
        }
        sift_opts = {
            k: v
            for k, v in feature_options.items()
            if k in _sift_option_keys and v is not None
        }
        extraction_kwargs = {}
        if "max_image_size" in feature_options:
            extraction_kwargs["max_image_size"] = feature_options["max_image_size"]

        if num_threads != -1:
            extraction_kwargs["num_threads"] = num_threads

        extraction_options = pycolmap.FeatureExtractionOptions(
            **extraction_kwargs, sift=pycolmap.SiftExtractionOptions(**sift_opts)
        )

        pycolmap.extract_features(
            db_file, image_path, image_names, extraction_options=extraction_options
        )

        with pycolmap.Database.open(db_file) as db:
            return [
                read_colmap_db_sift(db, image_path, image_name, feature_options)
                for image_name in image_names
            ]


def read_colmap_db_sift(
    db, image_path: str | Path, image_name: str, feature_options: dict
):
    """Read SIFT features from a COLMAP database for a single image.

    Extracts SIFT keypoints and descriptors from the COLMAP database, sorts
    them by feature size (largest first), and packages them with metadata.

    Args:
        db: COLMAP Database object
        image_path: Directory containing the image
        image_name: Name of the image (relative path from image_path)
        feature_options: Dict with feature extraction options

    Returns:
        Tuple with (feature_tool_metadata, metadata, positions, affine_shapes,
        descriptors, thumbnail)

    Raises:
        SiftExtractionError: If image not in database or features have wrong shape
    """
    image_path = Path(image_path)
    image_file = image_path / image_name

    image = db.read_image_with_name(image_name)
    if image is None:
        raise SiftExtractionError(f"Image {image_name} is not in the DB.")
    camera = db.read_camera(image.camera_id)

    width = camera.width
    height = camera.height

    keypoints = db.read_keypoints(image.image_id)
    if keypoints.shape[1] != 6:
        raise SiftExtractionError(
            f"Features for image {image_name} have {keypoints.shape[1]} entries "
            f"instead of the expected 6."
        )
    keypoints = keypoints.view(
        np.dtype([("position", np.float32, (2,)), ("affine_shape", np.float32, (2, 2))])
    )[:, 0]

    descriptors = db.read_descriptors(image.image_id)
    # pycolmap 4.x returns FeatureDescriptors object; extract numpy data
    if hasattr(descriptors, "data"):
        descriptors = descriptors.data
    if descriptors.shape[1] != 128:
        raise SiftExtractionError(
            f"Descriptors for image {image_name} have {descriptors.shape[1]} entries "
            f"instead of the expected 128."
        )
    descriptors = descriptors.view(np.dtype([("descriptor", np.uint8, 128)]))[:, 0]

    file_size = image_file.stat().st_size
    file_xxh128 = xxh128_of_file(image_file)

    # Sort the features descending by size
    sizes = feature_size(keypoints["affine_shape"])
    sorted_indexes = np.argsort(sizes)[::-1]
    keypoints = keypoints[sorted_indexes]
    descriptors = descriptors[sorted_indexes]

    feature_tool_metadata = {
        "feature_tool": "colmap",
        "feature_type": "sift",
        "feature_options": feature_options,
    }

    metadata = {
        "version": 1,
        "image_name": image_name,
        "image_file_xxh128": file_xxh128,
        "image_file_size": file_size,
        "image_width": width,
        "image_height": height,
        "feature_count": len(keypoints),
    }

    # Generate 128x128 RGB thumbnail
    img = cv2.imread(str(image_file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    thumbnail = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

    return (
        feature_tool_metadata,
        metadata,
        keypoints["position"],
        keypoints["affine_shape"],
        descriptors["descriptor"],
        thumbnail,
    )
