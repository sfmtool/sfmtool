# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""sfmtool SIFT extraction backend (the toolkit's own Rust implementation).

Wraps the ``sfmtool._sfmtool.extract_sift`` PyO3 binding so it plugs into the
same extraction pipeline as the COLMAP and OpenCV backends. The Rust core
parallelizes internally (rayon), so images are processed sequentially here.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from sfmtool.sift.file import SiftExtractionError, xxh128_of_file

__all__ = [
    "get_default_sfmtool_feature_options",
    "extract_sift_with_sfmtool",
]


def get_default_sfmtool_feature_options() -> dict:
    """Get default options for the sfmtool SIFT backend.

    The keys mirror the output-defining fields of ``SiftParams::default()`` in
    ``sfmtool-core`` and are passed straight through to the
    ``sfmtool._sfmtool.extract_sift`` binding. They also feed the feature-cache
    hash, so changing any of them yields a distinct cache directory.

    Hardware/performance-only knobs (e.g. thread count) are intentionally
    excluded — they do not change the feature output.

    Returns:
        Dict of sfmtool SIFT options.
    """
    return {
        "octave_layers": 3,
        "sigma": 1.6,
        "blur_radius_factor": 2.25,
        "input_sigma": 0.5,
        "double_image": True,
        "contrast_threshold": 0.0067,
        "edge_threshold": 10.0,
        "max_num_features": 8192,
        "orientation_bins": 36,
        "peak_ratio": 0.8,
        "descriptor_width": 4,
        "descriptor_bins": 8,
        "descriptor_magnification": 3.0,
        "descriptor_clamp": 0.2,
        # BT.709 luma (matches COLMAP). The grayscale conversion is output-defining,
        # so it is recorded here to pin it in the feature-tool hash even though the
        # version 1 .sift layout does not require an image_to_gray field.
        "gray_formula": "0.2126*R + 0.7152*G + 0.0722*B",
    }


def extract_sift_with_sfmtool(
    image_filename_list: list[str | Path],
    feature_options: dict,
    num_threads: int = -1,
):
    """Extract SIFT features from image files using the sfmtool Rust backend.

    Args:
        image_filename_list: List of absolute paths to image files
        feature_options: Dict of sfmtool SIFT options (see
            ``get_default_sfmtool_feature_options``). Empty/None uses the
            Rust defaults.
        num_threads: Accepted for interface compatibility. The Rust core
            parallelizes within each image via rayon (using all cores for
            ``-1``); images are processed sequentially here.

    Returns:
        List of tuples with (feature_tool_metadata, metadata, positions,
        affine_shapes, descriptors, thumbnail) for each image in order

    Raises:
        SiftExtractionError: If image loading fails or feature extraction fails
    """
    from sfmtool._sfmtool import extract_sift as _rust_extract_sift

    image_filename_list = [
        Path(os.path.normpath(os.path.abspath(p))) for p in image_filename_list
    ]

    # The binding rejects unknown keys, and only the output-defining keys belong
    # here; pass None when there are no overrides so the Rust defaults apply.
    params = feature_options or None

    print(
        f"Extracting features from {len(image_filename_list)} image(s) "
        f"using the sfmtool backend"
    )

    def process_single_image(image_path):
        image = cv2.imread(
            str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if image is None:
            raise SiftExtractionError(f"Failed to load image: {image_path}")

        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # The binding already returns C-contiguous float32/float32/uint8 arrays,
        # and keypoints already sorted by descending feature size (the .sift
        # ordering every consumer expects) — so no re-contiguize and no re-sort
        # here (matching the COLMAP/OpenCV backends).
        positions, affine_shapes, descriptors = _rust_extract_sift(rgb, params)

        file_size = image_path.stat().st_size
        file_xxh128 = xxh128_of_file(image_path)

        feature_tool_metadata = {
            "feature_tool": "sfmtool",
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
            "feature_count": len(positions),
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

    return [process_single_image(img_path) for img_path in image_filename_list]
