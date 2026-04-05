# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""SIFT feature file I/O and utilities.

Provides SiftReader for reading .sift files, write_sift for writing them,
and helper functions for path resolution and feature analysis.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xxhash

from sfmtool._sfmtool import (
    read_sift as _core_read_sift,
    read_sift_metadata,
    read_sift_partial,
    write_sift as _core_write_sift,
)

if TYPE_CHECKING:
    from sfmtool._sfmtool import SfmrReconstruction

__all__ = [
    "SiftExtractionError",
    "SiftReader",
    "write_sift",
    "get_feature_type_for_tool",
    "get_sift_path_for_image",
    "get_used_features_from_reconstruction",
    "print_sift_summary",
    "image_files_to_sift_files",
    "image_files_to_sift_files_opencv",
    "draw_sift_features",
    "xxh128_of_file",
    "get_feature_tool_xxh128",
    "compute_orientation",
    "feature_size",
    "feature_size_x",
    "feature_size_y",
]


class SiftExtractionError(Exception):
    """Exception raised when SIFT feature extraction fails."""

    pass


# ---------------------------------------------------------------------------
# Hashing utilities (previously in _sift_utils.py)
# ---------------------------------------------------------------------------


def xxh128_of_file(filename: str | Path) -> str:
    """Compute XXH128 hash of a file.

    Args:
        filename: Path to the file to hash

    Returns:
        Hexadecimal string representation of the XXH128 hash
    """
    xxh128 = xxhash.xxh3_128()
    with open(filename, "rb") as fh:
        for b in iter(lambda: fh.read(2**16), b""):
            xxh128.update(b)
    return xxh128.hexdigest()


def get_feature_tool_xxh128(
    feature_tool: str, feature_type: str, feature_options: dict
) -> str:
    """Calculate the XXH128 hash for a feature extraction configuration.

    Args:
        feature_tool: Name of the feature tool (e.g. "colmap", "opencv")
        feature_type: Feature type string (e.g. "sift-colmap", "sift-opencv")
        feature_options: Dict of feature extraction options

    Returns:
        Hexadecimal string representation of the XXH128 hash
    """
    canonical = json.dumps(
        {
            "feature_tool": feature_tool,
            "feature_type": feature_type,
            "feature_options": feature_options,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return xxhash.xxh3_128(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Affine shape / feature size utilities (previously in _sift_utils.py)
# ---------------------------------------------------------------------------


def compute_orientation(affine_shape):
    """Compute orientation angle from affine shape matrix.

    Uses the same method as COLMAP: atan2(a21, a11).

    Args:
        affine_shape: Array of shape (N, 2, 2) or (2, 2)

    Returns:
        Array of shape (N,) with orientations in radians, or a scalar.
    """
    return np.arctan2(affine_shape[..., 1, 0], affine_shape[..., 0, 0])


def feature_size_x(affine_shape):
    """Calculate x-axis feature size from affine shape matrix.

    Args:
        affine_shape: Array of shape (N, 2, 2)

    Returns:
        Array of shape (N,) with x-axis feature sizes
    """
    return np.sqrt(affine_shape[:, 0, 0] ** 2 + affine_shape[:, 1, 0] ** 2)


def feature_size_y(affine_shape):
    """Calculate y-axis feature size from affine shape matrix.

    Args:
        affine_shape: Array of shape (N, 2, 2)

    Returns:
        Array of shape (N,) with y-axis feature sizes
    """
    return np.sqrt(affine_shape[:, 0, 1] ** 2 + affine_shape[:, 1, 1] ** 2)


def feature_size(affine_shape):
    """Calculate average feature size from affine shape matrix.

    Args:
        affine_shape: Array of shape (N, 2, 2)

    Returns:
        Array of shape (N,) with average feature sizes
    """
    col0_norms = np.linalg.norm(affine_shape[:, :, 0], axis=1)
    col1_norms = np.linalg.norm(affine_shape[:, :, 1], axis=1)
    return 0.5 * (col0_norms + col1_norms)


# ---------------------------------------------------------------------------
# Feature type naming
# ---------------------------------------------------------------------------


def get_feature_type_for_tool(feature_tool: str, feature_options: dict) -> str:
    """Get the feature type string for directory naming based on tool and options.

    Args:
        feature_tool: Feature extraction tool name ("colmap" or "opencv")
        feature_options: Dict with tool-specific options

    Returns:
        Feature type string (e.g. "sift-colmap", "sift-colmap-dsp", "sift-opencv")
    """
    feature_tool = feature_tool.lower()

    if feature_tool == "opencv":
        return "sift-opencv"
    elif feature_tool == "colmap":
        parts = ["sift-colmap"]

        if feature_options.get("domain_size_pooling"):
            parts.append("dsp")

        max_features = feature_options.get("max_num_features")
        if max_features is not None and max_features != 8192:
            parts.append(f"max{max_features}")

        return "-".join(parts)
    else:
        return f"sift-{feature_tool}"


# ---------------------------------------------------------------------------
# Validation schemas
# ---------------------------------------------------------------------------

_FEATURE_TOOL_METADATA_SCHEMA = {
    "feature_tool": str,
    "feature_type": str,
    "feature_options": dict,
}
_FEATURE_TOOL_METADATA_KEYS = set(_FEATURE_TOOL_METADATA_SCHEMA.keys())

_SIFT_METADATA_SCHEMA = {
    "version": int,
    "image_name": str,
    "image_file_xxh128": str,
    "image_file_size": int,
    "image_width": int,
    "image_height": int,
    "feature_count": int,
}
_SIFT_METADATA_KEYS = set(_SIFT_METADATA_SCHEMA.keys())


def _validate_input(metadata, keys, schema):
    """Validate metadata against a schema."""
    metadata_keys = set(metadata.keys())
    if metadata_keys != keys:
        extra_keys = ", ".join(sorted(metadata_keys.difference(keys)))
        missing_keys = ", ".join(sorted(keys.difference(metadata_keys)))
        raise ValueError(
            f"The metadata has incorrect keys.\nExtra: {extra_keys}\nMissing: {missing_keys}"
        )

    for field_name, field_type in schema.items():
        if not isinstance(metadata[field_name], field_type):
            raise ValueError(
                f"The metadata field {field_name} has type {type(metadata[field_name])}, expected {field_type}"
            )


# ---------------------------------------------------------------------------
# SiftReader
# ---------------------------------------------------------------------------


class SiftReader:
    """Reader for .sift feature files, delegating I/O to Rust bindings."""

    def __init__(self, filename: str | Path):
        self.filename = Path(filename)
        try:
            meta = read_sift_metadata(str(self.filename))
        except OSError as e:
            if not self.filename.exists():
                raise FileNotFoundError(str(e)) from e
            raise
        self.feature_tool_metadata = meta["feature_tool_metadata"]
        self.metadata = meta["metadata"]
        self.content_hash = meta["content_hash"]
        self._data = None

    @classmethod
    def for_image(
        cls,
        image_path: str | Path,
        feature_tool: str | None = None,
        feature_options: dict | None = None,
    ):
        """Open a SiftReader for the .sift file corresponding to an image."""
        if feature_tool is None:
            feature_tool = "colmap"
        sift_path = get_sift_path_for_image(
            image_path,
            feature_tool=feature_tool,
            feature_options=feature_options,
        )
        return cls(sift_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._data = None

    def _ensure_data(self, count=None):
        if count is not None:
            return read_sift_partial(str(self.filename), count)
        if self._data is None:
            self._data = _core_read_sift(str(self.filename))
        return self._data

    def read_positions(self, count=None):
        return self._ensure_data(count)["positions_xy"]

    def read_affine_shapes(self, count=None):
        return self._ensure_data(count)["affine_shapes"]

    def read_descriptors(self, count=None):
        return self._ensure_data(count)["descriptors"]

    def read_positions_and_shapes(self, count=None):
        data = self._ensure_data(count)
        return data["positions_xy"], data["affine_shapes"]

    def read_thumbnail(self):
        """Read the 128x128 RGB thumbnail from the .sift file."""
        return self._ensure_data()["thumbnail_y_x_rgb"]


# ---------------------------------------------------------------------------
# write_sift
# ---------------------------------------------------------------------------


def write_sift(
    filename: str | Path,
    feature_tool_metadata,
    metadata,
    position,
    affine_shape,
    descriptor,
    thumbnail,
    *,
    zstd_level=5,
):
    """Write a .sift file with SIFT feature data.

    Validates metadata and array shapes/dtypes in Python, then delegates
    ZIP/zstd I/O to the Rust backend.
    """
    filename = Path(filename)

    _validate_input(
        feature_tool_metadata,
        _FEATURE_TOOL_METADATA_KEYS,
        _FEATURE_TOOL_METADATA_SCHEMA,
    )
    _validate_input(metadata, _SIFT_METADATA_KEYS, _SIFT_METADATA_SCHEMA)

    if not (
        metadata["feature_count"]
        == len(position)
        == len(affine_shape)
        == len(descriptor)
    ):
        raise ValueError(
            f"Lengths of input arrays don't match feature count {metadata['feature_count']}: "
            f"{len(position)}, {len(affine_shape)}, {len(descriptor)}"
        )

    feature_count = metadata["feature_count"]
    position = np.asarray(position)
    affine_shape = np.asarray(affine_shape)
    descriptor = np.asarray(descriptor)

    if position.dtype != np.float32:
        raise ValueError(
            f"Data dtype {position.dtype} does not match required element dtype float32"
        )
    if affine_shape.dtype != np.float32:
        raise ValueError(
            f"Data dtype {affine_shape.dtype} does not match required element dtype float32"
        )
    if descriptor.dtype != np.uint8:
        raise ValueError(
            f"Data dtype {descriptor.dtype} does not match required element dtype uint8"
        )

    if position.shape != (feature_count, 2):
        raise ValueError(
            f"Data shape {position.shape} does not match required feature count "
            f"and element shape ({feature_count}, 2)"
        )
    if affine_shape.shape != (feature_count, 2, 2):
        raise ValueError(
            f"Data shape {affine_shape.shape} does not match required feature count "
            f"and element shape ({feature_count}, 2, 2)"
        )
    if descriptor.shape != (feature_count, 128):
        raise ValueError(
            f"Data shape {descriptor.shape} does not match required feature count "
            f"and element shape ({feature_count}, 128)"
        )

    thumbnail = np.asarray(thumbnail)
    if thumbnail.shape != (128, 128, 3):
        raise ValueError(
            f"Thumbnail shape {thumbnail.shape} does not match required shape (128, 128, 3)"
        )
    if thumbnail.dtype != np.uint8:
        raise ValueError(
            f"Thumbnail dtype {thumbnail.dtype} does not match required dtype uint8"
        )

    data = {
        "feature_tool_metadata": feature_tool_metadata,
        "metadata": metadata,
        "positions_xy": position,
        "affine_shapes": affine_shape,
        "descriptors": descriptor,
        "thumbnail_y_x_rgb": thumbnail,
    }
    _core_write_sift(str(filename), data, zstd_level)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def get_sift_path_for_image(
    image_filename: str | Path,
    feature_tool: str | None = None,
    feature_options: dict | None = None,
) -> Path:
    """Determine the .sift file path for a given image filename.

    If feature_tool and feature_options are not provided, attempts to find
    the workspace configuration and use its ``feature_prefix_dir``.

    Args:
        image_filename: Absolute path to an image file
        feature_tool: Feature extraction tool name. If None, tries workspace config.
        feature_options: Dict with feature extraction options. If None, uses defaults.

    Returns:
        Path where the corresponding .sift file should be located
    """
    image_filename = Path(image_filename).resolve()

    if feature_tool is None or feature_options is None:
        from sfmtool._workspace import find_workspace_for_path, load_workspace_config

        workspace_dir = find_workspace_for_path(image_filename)
        if workspace_dir is not None:
            workspace_config = load_workspace_config(workspace_dir)
            if feature_tool is None and feature_options is None:
                if "feature_prefix_dir" not in workspace_config:
                    raise RuntimeError(
                        f"SfM workspace {workspace_dir} is missing required "
                        f"field feature_prefix_dir. Re-initialize the workspace "
                        f"with 'sfm init'."
                    )
                return (
                    image_filename.parent
                    / workspace_config["feature_prefix_dir"]
                    / (image_filename.name + ".sift")
                )
            if feature_tool is None:
                feature_tool = workspace_config["feature_tool"]
            if feature_options is None:
                feature_options = workspace_config["feature_options"]

    if feature_tool is None:
        feature_tool = "colmap"
    feature_tool = feature_tool.lower()

    if feature_options is None:
        from sfmtool._extract_sift_colmap import get_colmap_feature_options
        from sfmtool._extract_sift_opencv import get_default_opencv_feature_options

        if feature_tool == "opencv":
            feature_options = get_default_opencv_feature_options()
        else:
            feature_options = get_colmap_feature_options()

    feature_type = get_feature_type_for_tool(feature_tool, feature_options)
    feature_tool_hash = get_feature_tool_xxh128(
        feature_tool, feature_type, feature_options
    )

    return (
        image_filename.parent
        / "features"
        / f"{feature_type}-{feature_tool_hash}"
        / (image_filename.name + ".sift")
    )


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------


def get_used_features_from_reconstruction(
    recon: "SfmrReconstruction", image_path: str | Path
) -> np.ndarray:
    """Get feature indices used in a reconstruction for a specific image.

    Args:
        recon: SfmrReconstruction object
        image_path: Path to the image file

    Returns:
        Array of feature indices (0-based) used in the reconstruction.
        Empty array if the image is not in the reconstruction.

    Raises:
        ValueError: If image path cannot be resolved relative to workspace
    """
    image_path = Path(image_path).resolve()

    workspace_dir = Path(recon.workspace_dir)
    image_names = recon.image_names
    image_indexes = np.asarray(recon.track_image_indexes)
    feature_indexes = np.asarray(recon.track_feature_indexes)

    try:
        image_name = image_path.relative_to(workspace_dir).as_posix()
    except ValueError:
        raise ValueError(
            f"Image {image_path} is not relative to workspace {workspace_dir}"
        )

    image_index = None
    for idx, name in enumerate(image_names):
        if name == image_name:
            image_index = idx
            break

    if image_index is None:
        return np.array([], dtype=np.uint32)

    mask = image_indexes == image_index
    return feature_indexes[mask]


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_sift_summary(sift_filename: str | Path, verbose: bool = False):
    """Read a .sift file and print a human-readable summary.

    Args:
        sift_filename: Path to the .sift file
        verbose: If True, also print file hashes and tool options
    """
    print(f"--- Summary for {sift_filename} ---")
    with SiftReader(sift_filename) as reader:
        meta = reader.metadata
        tool_meta = reader.feature_tool_metadata
        print(f"  Image Name:         {meta['image_name']}")
        print(f"  Image Dimensions:   {meta['image_width']}x{meta['image_height']}")
        print(f"  Feature Count:      {meta['feature_count']}")
        print(f"  Feature Tool:       {tool_meta['feature_tool']}")

        if verbose:
            hashes = reader.content_hash
            print(f"  Image File Size:    {meta['image_file_size']} bytes")
            print(f"  Image File Hash:    {meta['image_file_xxh128']}")
            print(f"  Feature Tool Hash:  {hashes['feature_tool_xxh128']}")
            print(f"  Content Hash:       {hashes['content_xxh128']}")
            print(f"  Feature Tool Options: {tool_meta['feature_options']}")

        print("  Top 5 Features (by size):")
        positions = reader.read_positions(count=5)
        affine_shapes = reader.read_affine_shapes(count=5)
        sizes_x = feature_size_x(affine_shapes)
        sizes_y = feature_size_y(affine_shapes)
        for i, (pos, sx, sy) in enumerate(zip(positions, sizes_x, sizes_y), 1):
            print(f"    {i}. pos=({pos[0]:.2f}, {pos[1]:.2f}), size=({sx:.2f}, {sy:.2f})")


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------


def _summarize_paths(paths: list[Path]) -> str:
    """Summarize a list of paths for display."""
    if not paths:
        return "(none)"
    parents = {p.parent for p in paths}
    lines = []
    for parent in sorted(parents):
        children = sorted(p.name for p in paths if p.parent == parent)
        if len(children) <= 3:
            lines.append(f"{parent}/  ({', '.join(children)})")
        else:
            lines.append(
                f"{parent}/  ({children[0]}, ..., {children[-1]}) "
                f"[{len(children)} files]"
            )
    return "\n".join(lines)


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
    from sfmtool._extract_sift_colmap import (
        extract_sift_with_colmap,
        get_colmap_feature_options,
    )
    from sfmtool._extract_sift_opencv import (
        extract_sift_with_opencv,
        get_default_opencv_feature_options,
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
        for index_start in range(0, len(image_filename_filtered_list), chunk_size):
            sift_list = extraction_fn(
                image_filename_filtered_list[index_start : index_start + chunk_size],
                feature_options,
                num_threads=num_threads,
            )
            for sift, sift_filename in zip(
                sift_list,
                sift_filename_filtered_list[index_start : index_start + chunk_size],
            ):
                write_sift(sift_filename, *sift)

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
    print("  " + _summarize_paths(image_filename_list).replace("\n", "\n  "))
    print("SIFT features files:")
    print("  " + _summarize_paths(sift_filename_list).replace("\n", "\n  "))
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_sift_features(
    image_path: str | Path,
    output_path: str | Path,
    max_features: int | None = None,
    feature_indices: "np.ndarray | None" = None,
    feature_tool: str | None = None,
    feature_options: dict | None = None,
) -> None:
    """Draw SIFT features with affine shape ellipses on an image.

    Args:
        image_path: Path to the input image file
        output_path: Path where the output image should be saved
        max_features: Optional limit on number of features to draw (draws largest first)
        feature_indices: Optional array of feature indices to draw.
        feature_tool: Feature extraction tool name
        feature_options: Optional dict with feature tool options.

    Raises:
        FileNotFoundError: If image or SIFT file doesn't exist
    """
    import cv2

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    try:
        with SiftReader.for_image(
            image_path,
            feature_tool=feature_tool,
            feature_options=feature_options,
        ) as reader:
            positions, affine_shapes = reader.read_positions_and_shapes(
                count=max_features
            )
    except FileNotFoundError:
        sift_path = get_sift_path_for_image(
            image_path,
            feature_tool=feature_tool,
            feature_options=feature_options,
        )
        raise FileNotFoundError(
            f"SIFT file not found for image {image_path}. "
            f"Expected at: {sift_path}. "
            f"Run 'sfm sift --extract' first to generate SIFT features."
        )

    if feature_indices is not None:
        positions = positions[feature_indices]
        affine_shapes = affine_shapes[feature_indices]

    for pos, affine_matrix in zip(positions, affine_shapes):
        center_x, center_y = float(pos[0]), float(pos[1])
        center = (int(round(center_x)), int(round(center_y)))

        # SVD to get ellipse parameters
        _U, s, _Vt = np.linalg.svd(affine_matrix)
        axis_a = float(s[0])
        axis_b = float(s[1])

        angle_rad = compute_orientation(affine_matrix)
        angle_deg = np.degrees(angle_rad)

        cv2.ellipse(
            image,
            center,
            (int(round(axis_a)), int(round(axis_b))),
            angle_deg,
            0,
            360,
            (0, 255, 0),  # green in BGR
            1,
        )
        cv2.circle(image, center, 2, (0, 0, 255), -1)  # red center

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Drew {len(positions)} features on {image_path.name}")
    print(f"  Saved to: {output_path}")
