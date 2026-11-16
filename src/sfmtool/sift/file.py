# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for `.sift` files: where an image's file lives, and arithmetic on
the affine shapes it stores. The features folder an image's file goes in is
named by `sfmtool.workspace.feature_prefix_dir`.

Reading and writing a `.sift` file is `sfmtool.fileio`'s: `SiftFile` opens
one, and `write_sift` and `SiftWriteQueue` write them.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xxhash

# The feature type a workspace records for a tool, named by the one Rust
# implementation, which the workspace upgrade also names folders with.
from ..workspace import feature_type_for_tool as get_feature_type_for_tool

if TYPE_CHECKING:
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

__all__ = [
    "get_feature_type_for_tool",
    "get_sift_path_for_image",
    "get_sift_path_from_recon",
    "get_used_features_from_reconstruction",
    "xxh128_of_file",
    "compute_orientation",
    "feature_size",
    "feature_size_x",
    "feature_size_y",
]


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
# Path resolution
# ---------------------------------------------------------------------------


def get_sift_path_for_image(
    image_filename: str | Path,
    feature_tool: str | None = None,
    feature_options: dict | None = None,
    *,
    ignore_workspace: bool = False,
) -> Path:
    """Determine the .sift file path for a given image filename.

    When ``image_filename`` lies inside an SfM workspace whose config has
    ``feature_prefix_dir``, that prefix is used directly — explicit
    ``feature_tool`` / ``feature_options`` arguments are ignored. The
    workspace's stored prefix is the source of truth and avoids re-hashing
    options that may have drifted from what was used at extraction time.

    For images outside a workspace (or in a workspace missing
    ``feature_prefix_dir``), or when ``ignore_workspace`` is true, the path is
    recomputed from the supplied ``feature_tool`` / ``feature_options``,
    defaulting to colmap with default options. ``ignore_workspace`` is for a
    caller whose tool overrides the workspace's, as ``sfm sift --tool`` does,
    so it finds the files that extraction with the same tool wrote.
    """
    image_filename = Path(image_filename).resolve()

    from sfmtool._workspace import load_workspace_config
    from sfmtool.workspace import feature_prefix_dir, find_workspace_for_path

    workspace_dir = (
        None if ignore_workspace else find_workspace_for_path(image_filename)
    )
    if workspace_dir is not None:
        workspace_config = load_workspace_config(workspace_dir)
        prefix = workspace_config.get("feature_prefix_dir")
        if prefix:
            return image_filename.parent / prefix / (image_filename.name + ".sift")

    if feature_tool is None:
        feature_tool = "colmap"
    feature_tool = feature_tool.lower()

    if feature_options is None:
        from sfmtool.sift.extract_colmap import get_colmap_feature_options
        from sfmtool.sift.extract_opencv import get_default_opencv_feature_options
        from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

        if feature_tool == "opencv":
            feature_options = get_default_opencv_feature_options()
        elif feature_tool == "sfmtool":
            feature_options = get_default_sfmtool_feature_options()
        else:
            feature_options = get_colmap_feature_options()

    feature_type = get_feature_type_for_tool(feature_tool, feature_options)
    prefix = feature_prefix_dir(feature_tool, feature_type, feature_options)
    return image_filename.parent / prefix / (image_filename.name + ".sift")


def get_sift_path_from_recon(
    recon: "SfmrReconstruction", image_name: str | Path
) -> Path:
    """Resolve an image's ``.sift`` path from the reconstruction's own metadata.

    The feature directory is taken from the ``feature_prefix_dir`` recorded in
    the reconstruction (the features actually used to build it), joined under the
    recon's workspace directory. Use this — rather than
    :func:`get_sift_path_for_image` — whenever the path must reflect the features
    a given reconstruction was built from, since the image's *current* on-disk
    workspace config may have drifted to a different feature set.

    ``image_name`` is the workspace-relative path as stored in
    ``recon.image_names`` (e.g. ``images/foo.jpg``).
    """
    prefix = recon.metadata()["workspace"]["contents"]["feature_prefix_dir"]
    image_rel = Path(image_name)
    return (
        Path(recon.workspace_dir) / image_rel.parent / prefix / f"{image_rel.name}.sift"
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
