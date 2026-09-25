# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract SIFT features and write them to .sift files."""

import textwrap
from pathlib import Path

from sfmtool._path_summary import summarize_path_list


class SiftExtractionError(Exception):
    """Exception raised when SIFT feature extraction fails."""


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------


def image_files_to_sift_files(
    image_filename_list: list[str | Path],
    feature_path: str | Path | None = None,
    num_threads: int = -1,
    feature_tool: str | None = "colmap",
    feature_options: dict | None = None,
    feature_prefix_dir: str | None = None,
) -> list[Path]:
    """Extract and write SIFT features for a list of images.

    Processes images into .sift files using the specified tool (COLMAP or OpenCV),
    skipping images that already have up-to-date .sift files based on modification
    timestamps. Processing is done in chunks of 500 images for progress reporting.

    Args:
        image_filename_list: List of absolute paths to image files
        feature_path: Optional directory to write .sift files to.
        num_threads: Number of threads for feature extraction (-1 uses all cores)
        feature_tool: Feature extraction tool: "colmap" (default) or "opencv"
        feature_options: Optional dict with tool-specific options.
                         If None, uses defaults for the specified tool.
        feature_prefix_dir: Optional relative path from each image's parent to the
                           features directory. When provided, takes precedence over
                           computing from tool+hash.

    Returns:
        List of paths to the created/verified .sift files (in same order as input)

    Raises:
        SiftExtractionError: If feature extraction fails for any image
    """
    from sfmtool.sift.file import (
        _SIFT_ZSTD_LEVEL,
        _SiftWriteStream,
        _validate_sift_write,
        get_feature_tool_xxh128,
        get_feature_type_for_tool,
    )
    from sfmtool.sift.extract_colmap import (
        extract_sift_with_colmap,
        get_colmap_feature_options,
    )
    from sfmtool.sift.extract_opencv import (
        extract_sift_with_opencv,
        get_default_opencv_feature_options,
    )
    from sfmtool.sift.extract_sfmtool import (
        extract_sift_with_sfmtool,
        get_default_sfmtool_feature_options,
    )

    image_filename_list = [Path(p) for p in image_filename_list]
    if feature_path:
        feature_path = Path(feature_path)

    if feature_tool is None:
        feature_tool = "colmap"
    feature_tool = feature_tool.lower()
    if feature_tool == "opencv":
        if feature_options is None:
            feature_options = get_default_opencv_feature_options()
        extraction_fn = extract_sift_with_opencv
    elif feature_tool == "sfmtool":
        if feature_options is None:
            feature_options = get_default_sfmtool_feature_options()
        extraction_fn = extract_sift_with_sfmtool
    else:  # colmap
        if feature_options is None:
            feature_options = get_colmap_feature_options()
        extraction_fn = extract_sift_with_colmap

    feature_type = get_feature_type_for_tool(feature_tool, feature_options)

    if feature_path:
        sift_filename_list = [
            feature_path / (p.name + ".sift") for p in image_filename_list
        ]
        feature_path.mkdir(parents=True, exist_ok=True)
    elif feature_prefix_dir:
        sift_filename_list = [
            p.parent / feature_prefix_dir / (p.name + ".sift")
            for p in image_filename_list
        ]
        sift_dirs = {p.parent for p in sift_filename_list}
        for d in sift_dirs:
            d.mkdir(parents=True, exist_ok=True)
    else:
        feature_tool_xxh128 = get_feature_tool_xxh128(
            feature_tool, feature_type, feature_options
        )

        sift_filename_list = [
            p.parent
            / "features"
            / f"{feature_type}-{feature_tool_xxh128}"
            / (p.name + ".sift")
            for p in image_filename_list
        ]
        sift_dirs = {p.parent for p in sift_filename_list}
        for d in sift_dirs:
            d.mkdir(parents=True, exist_ok=True)

    # Check modification times to skip up-to-date files
    mtime_pairs = [
        (
            p.stat().st_mtime,
            s.stat().st_mtime if s.exists() else None,
        )
        for p, s in zip(image_filename_list, sift_filename_list)
    ]
    files_skip_mask = [
        sift_mtime is not None and sift_mtime >= image_mtime
        for image_mtime, sift_mtime in mtime_pairs
    ]

    image_filename_filtered_list = [
        filename
        for skip, filename in zip(files_skip_mask, image_filename_list)
        if not skip
    ]
    sift_filename_filtered_list = [
        filename
        for skip, filename in zip(files_skip_mask, sift_filename_list)
        if not skip
    ]

    if image_filename_filtered_list:
        chunk_size = 500
        # Compress + write each .sift on the shared rayon pool via SiftWriteQueue
        # (write_sift releases the GIL for the zstd/ZIP work). Because the save
        # is a pool *task* rather than a separate OS thread, it never
        # oversubscribes the cores: one worker runs the save while the extract's
        # par_iter proceeds on the rest, so the save of image i overlaps the
        # extract of image i+1 without the barrier busy-spin a contending
        # external thread would cause. See specs/core/features/sift.md.
        with _SiftWriteStream() as writer:
            for index_start in range(0, len(image_filename_filtered_list), chunk_size):
                # extraction_fn may return a list (COLMAP/OpenCV, which shell out
                # to batch binaries) or an in-order generator (the sfmtool
                # backend). In the generator case the zip pulls one result at a
                # time, so extraction and writing stream per image.
                sift_list = extraction_fn(
                    image_filename_filtered_list[
                        index_start : index_start + chunk_size
                    ],
                    feature_options,
                    num_threads=num_threads,
                )
                for sift, sift_filename in zip(
                    sift_list,
                    sift_filename_filtered_list[index_start : index_start + chunk_size],
                ):
                    writer.submit(
                        str(sift_filename),
                        _validate_sift_write(*sift),
                        _SIFT_ZSTD_LEVEL,
                    )

    print()
    if len(sift_filename_filtered_list) != len(sift_filename_list):
        print(
            f"Existing SIFT features already processed for "
            f"{len(sift_filename_list) - len(sift_filename_filtered_list)} / "
            f"{len(sift_filename_list)} image(s)"
        )
    tool_label = f" ({feature_tool.upper()})" if feature_tool != "colmap" else ""
    print(
        f"New SIFT feature extraction{tool_label}: "
        f"{len(sift_filename_filtered_list)} / {len(sift_filename_list)} image(s)"
    )
    print()
    print("Image files:")
    print(textwrap.indent(summarize_path_list(image_filename_list), "  "))
    print("SIFT features files:")
    print(textwrap.indent(summarize_path_list(sift_filename_list), "  "))
    return sift_filename_list


def image_files_to_sift_files_opencv(
    image_filename_list: list[str | Path],
    feature_path: str | Path | None = None,
    num_threads: int = -1,
) -> list[Path]:
    """Extract and write SIFT features for a list of images using OpenCV.

    Convenience wrapper around image_files_to_sift_files() with tool="opencv".

    Args:
        image_filename_list: List of absolute paths to image files
        feature_path: Optional directory to write .sift files to
        num_threads: Number of threads for feature extraction (-1 uses all cores)

    Returns:
        List of paths to the created/verified .sift files (in same order as input)
    """
    return image_files_to_sift_files(
        image_filename_list=image_filename_list,
        feature_path=feature_path,
        num_threads=num_threads,
        feature_tool="opencv",
    )
