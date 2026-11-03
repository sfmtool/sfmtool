# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract SIFT features and write them to .sift files."""

import textwrap
from pathlib import Path

from sfmtool._path_summary import summarize_path_list


class SiftExtractionError(Exception):
    """Exception raised when SIFT feature extraction fails."""


# The feature tools whose grey conversion is the tool's own, not a setting.
_FIXED_CONVERSION_TOOLS = ("colmap", "opencv")


def _image_readings_fields(image_path, rgb, feature_options: dict) -> dict:
    """The two version 2 ``metadata`` readings of a decoded image.

    ``rgb`` is the ``(H, W, 3)`` uint8 decode the thumbnail is made from, and
    the conversion is ``feature_options["image_to_gray"]``. Returns the
    ``sharpness`` and ``gradient_structure_tensor_xx_yy_xy`` keys of a
    ``.sift`` file's metadata.

    Raises:
        SiftExtractionError: when the options carry no ``image_to_gray``, or
            the readings are not finite under it; the message names the image
            and the conversion.
    """
    from sfmtool.sift import ImageToGray, image_readings_from_rgb

    try:
        image_to_gray = ImageToGray.from_json(feature_options["image_to_gray"])
    except KeyError:
        raise SiftExtractionError(
            "The feature options carry no 'image_to_gray', so they are version 1 "
            "settings and no .sift file is written from them; run 'sfm ws upgrade'."
        ) from None
    except ValueError as e:
        raise SiftExtractionError(str(e)) from e
    try:
        readings = image_readings_from_rgb(rgb, image_to_gray)
    except ValueError as e:
        raise SiftExtractionError(f"Image {image_path}: {e}") from e
    return {
        "sharpness": readings.sharpness,
        "gradient_structure_tensor_xx_yy_xy": list(
            readings.gradient_structure_tensor_xx_yy_xy
        ),
    }


def _check_versions_never_mix(
    feature_tool: str,
    feature_options: dict,
    sift_dirs,
    settings_source: str | None,
    settings_from_sfmr: bool = False,
) -> None:
    """Refuse to write version 2 ``.sift`` files where versions would mix.

    Called only when at least one file is to be written. Refuses an
    ``sfmtool-undistort`` tool, whose files only ``sfm undistort`` makes;
    options with no top-level ``image_to_gray`` (version 1 settings); a
    ``colmap`` or ``opencv`` conversion other than the one the tool makes; and
    a folder in ``sift_dirs`` that already holds a version 1 ``.sift`` file.
    ``sfm ws upgrade`` is suggested only when the settings are a workspace
    file's, not a ``.sfmr`` record's (``settings_from_sfmr``).
    """
    from sfmtool.fileio import SiftFile
    from sfmtool.workspace import tool_image_to_gray

    if feature_tool == "sfmtool-undistort":
        raise SiftExtractionError(
            "The feature tool is 'sfmtool-undistort': the .sift files of an "
            "undistorted workspace are made by 'sfm undistort' from its source's, "
            "and cannot be extracted. Run 'sfm undistort' again to write them."
        )
    upgrade = (
        "The record is the .sfmr's own, so 'sfm ws upgrade' does not change it."
        if settings_from_sfmr
        else "Run 'sfm ws upgrade', which moves the workspace to version 2 and a "
        "new folder."
    )
    if "image_to_gray" not in feature_options:
        where = settings_source or "The feature options"
        raise SiftExtractionError(
            f"{where} holds version 1 settings (its feature options carry no "
            "'image_to_gray'), and no .sift file is written from version 1 "
            f"settings. {upgrade}"
        )
    if feature_tool in _FIXED_CONVERSION_TOOLS:
        row = tool_image_to_gray(feature_tool).to_dict()
        if feature_options["image_to_gray"] != row:
            raise SiftExtractionError(
                f"The '{feature_tool}' feature tool converts images to grey one way, "
                f"image_to_gray {row}, and the feature options record "
                f"{feature_options['image_to_gray']}."
            )
    for sift_dir in sorted(set(sift_dirs)):
        if not sift_dir.is_dir():
            continue
        for path in sorted(sift_dir.glob("*.sift")):
            try:
                with SiftFile(path) as reader:
                    version = reader.metadata["version"]
            except OSError:
                # A file that cannot be read says nothing about the folder.
                continue
            if version < 2:
                raise SiftExtractionError(
                    f"The folder {sift_dir} holds version 1 .sift files, and version 2 "
                    f"files are never written beside them. {upgrade}"
                )


# The `write_sift` dict keys of the six values each extraction backend returns
# for an image, in the order it returns them.
_SIFT_DATA_KEYS = (
    "feature_tool_metadata",
    "metadata",
    "positions_xy",
    "affine_shapes",
    "descriptors",
    "thumbnail_y_x_rgb",
)


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
    settings_source: str | None = None,
    settings_from_sfmr: bool = False,
) -> list[Path]:
    """Extract and write SIFT features for a list of images.

    Processes images into .sift files using the specified tool (COLMAP or OpenCV),
    skipping images that already have up-to-date .sift files based on modification
    timestamps. Processing is done in chunks of 500 images for progress reporting.
    The files are ``.sift`` version 2.

    When at least one file is to be written, it refuses the
    ``sfmtool-undistort`` tool, whose files only ``sfm undistort`` makes,
    version 1 settings
    (``feature_options`` with no ``image_to_gray``, as a version 1 workspace
    file's are), a ``colmap`` or ``opencv`` ``image_to_gray`` other than the
    tool's, and a target folder that already holds a version 1 ``.sift`` file.
    When every file is up to date it writes nothing and refuses nothing.

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
        settings_source: What the settings were read from, such as the
                         workspace file's path, for the refusal of version 1
                         settings to name.
        settings_from_sfmr: Whether the settings are a ``.sfmr``'s record
                         rather than a workspace file's, in which case a
                         refusal does not suggest ``sfm ws upgrade``.

    Returns:
        List of paths to the created/verified .sift files (in same order as input)

    Raises:
        SiftExtractionError: If feature extraction fails for any image
    """
    from sfmtool._sfmtool.fileio import SiftWriteQueue
    from sfmtool.sift.file import get_feature_tool_xxh128, get_feature_type_for_tool
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
    elif feature_prefix_dir:
        sift_filename_list = [
            p.parent / feature_prefix_dir / (p.name + ".sift")
            for p in image_filename_list
        ]
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
        sift_dirs = {p.parent for p in sift_filename_filtered_list}
        _check_versions_never_mix(
            feature_tool,
            feature_options,
            sift_dirs,
            settings_source,
            settings_from_sfmr,
        )
        for d in sift_dirs:
            d.mkdir(parents=True, exist_ok=True)
        chunk_size = 500
        # Compress + write each .sift on the shared rayon pool via
        # SiftWriteQueue, which bounds the saves in flight and joins them at the
        # end of the block (draining instead if an exception leaves it). See
        # specs/core/features/sift.md.
        with SiftWriteQueue() as writer:
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
                    writer.submit(sift_filename, dict(zip(_SIFT_DATA_KEYS, sift)))

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
