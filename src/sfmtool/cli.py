# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from io import UnsupportedOperation
from pathlib import Path

import click

from ._cli_group import CategoryGroup
from ._workspace import find_workspace_for_path


@click.help_option("--help", "-h")
@click.group(cls=CategoryGroup)
def main():
    """SfM Tool - Fun with Structure from Motion."""
    # Disable output buffering for real-time progress feedback
    # Use UTF-8 encoding for Unicode support (e.g., histogram block characters)
    # Skip in test environments where stdout/stderr don't have fileno()
    try:
        sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, encoding="utf-8")
        sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, encoding="utf-8")
    except AttributeError, UnsupportedOperation:
        pass


# Every top-level command except `version`: its name, the `--help` category it
# is listed under, the module in `sfmtool._commands` and the attribute that
# define it, and its one-line help. A command's module is imported only when the
# command is run, or its own `--help` is asked for, so `sfm --help` lists the
# commands from this table. The one-line help is the first sentence of the
# command's docstring; `tests/test_lazy_loading.py` checks that the two
# agree.
COMMANDS = (
    # Workspace
    ("ws", "Workspace", "ws", "ws", "Workspace-related operations."),
    (
        "pano2rig",
        "Workspace",
        "pano2rig",
        "pano2rig",
        "Convert equirectangular panoramas to perspective face images for rig-aware SfM.",
    ),
    (
        "insv2rig",
        "Workspace",
        "insv2rig",
        "insv2rig",
        "Extract dual-fisheye frames from an Insta360 .insv video file.",
    ),
    ("camrig", "Workspace", "camrig", "camrig", "Build .camrig camera rig files."),
    # Image Feature
    (
        "sift",
        "Image Feature",
        "sift",
        "sift",
        "Extract SIFT features and visualize .sift feature files.",
    ),
    (
        "match",
        "Image Feature",
        "match",
        "match",
        "Match features between image pairs and write a .matches file.",
    ),
    (
        "cluster-patches",
        "Image Feature",
        "cluster_patches",
        "cluster_patches",
        "Refine a cluster-bearing .matches file into patch clusters.",
    ),
    # Reconstruction
    (
        "solve",
        "Reconstruction",
        "solve",
        "solve",
        "Run structure from motion on images or a .matches file.",
    ),
    (
        "xform",
        "Reconstruction",
        "xform",
        "xform",
        "Apply transformations to a .sfmr file.",
    ),
    (
        "inspect",
        "Reconstruction",
        "inspect",
        "inspect",
        "Inspect an sfmtool file, image, or 3D point and print a summary.",
    ),
    (
        "analyze",
        "Reconstruction",
        "analyze",
        "analyze",
        "Run a deep-analysis report on a .sfmr reconstruction.",
    ),
    ("compare", "Reconstruction", "compare", "compare", "Compare two .sfmr files."),
    (
        "align",
        "Reconstruction",
        "align",
        "align",
        "Align multiple .sfmr reconstructions.",
    ),
    (
        "merge",
        "Reconstruction",
        "merge",
        "merge",
        "Merge multiple aligned .sfmr files.",
    ),
    (
        "densify",
        "Reconstruction",
        "densify",
        "densify",
        "Densify matches in a .sfmr file (experimental).",
    ),
    (
        "motion",
        "Reconstruction",
        "motion",
        "motion",
        "Analyze camera motion in image sequences or reconstructions for discontinuities.",
    ),
    (
        "embed-patches",
        "Reconstruction",
        "embed_patches",
        "embed_patches_command",
        "Convert a sift_files reconstruction to embedded_patches.",
    ),
    (
        "estimate-intrinsics",
        "Reconstruction",
        "estimate_intrinsics",
        "estimate_intrinsics",
        "Estimate a shared focal length and camera model from cluster matches.",
    ),
    # Visualization
    (
        "explorer",
        "Visualization",
        "explorer",
        "explorer",
        "Launch the SfM Explorer 3D viewer.",
    ),
    (
        "epipolar",
        "Visualization",
        "epipolar",
        "epipolar",
        "Visualize epipolar geometry between two images from a reconstruction.",
    ),
    (
        "heatmap",
        "Visualization",
        "heatmap",
        "heatmap",
        "Visualize reconstruction quality metrics as heatmaps on images.",
    ),
    (
        "render-patches",
        "Visualization",
        "render_patches",
        "render_patches_command",
        "Render a reconstruction's oriented patches on top of its source images.",
    ),
    (
        "panorama",
        "Visualization",
        "panorama",
        "panorama",
        "Render an equirectangular panorama from a posed reconstruction.",
    ),
    (
        "web-export",
        "Visualization",
        "web_export",
        "web_export",
        "Write a reconstruction as a web page that draws it in 3D.",
    ),
    # Image Processing
    (
        "flow",
        "Image Processing",
        "flow",
        "flow",
        "Visualize optical flow between two images.",
    ),
    (
        "undistort",
        "Image Processing",
        "undistort",
        "undistort",
        "Undistort all images in a reconstruction using camera parameters.",
    ),
    # COLMAP Interop
    (
        "to-colmap-bin",
        "COLMAP Interop",
        "to_colmap_bin",
        "to_colmap_bin",
        "Convert a .sfmr file to COLMAP .bin format.",
    ),
    (
        "to-colmap-db",
        "COLMAP Interop",
        "to_colmap_db",
        "to_colmap_db",
        "Create a COLMAP database from a .sfmr or .matches file.",
    ),
    (
        "from-colmap-bin",
        "COLMAP Interop",
        "from_colmap_bin",
        "from_colmap_bin",
        "Convert COLMAP .bin files to a .sfmr file.",
    ),
    (
        "to-nerfstudio",
        "COLMAP Interop",
        "to_nerfstudio",
        "to_nerfstudio",
        "Convert a pinhole .sfmr reconstruction to a Nerfstudio dataset.",
    ),
)

for _name, _category, _module, _attribute, _short_help in COMMANDS:
    main.add_lazy_command(
        _name,
        f"sfmtool._commands.{_module}",
        _attribute,
        short_help=_short_help,
        category=_category,
    )


@main.command()
def version():
    """Print the version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        pkg_version = _pkg_version("sfmtool")
    except PackageNotFoundError:
        pkg_version = "unknown"
    click.echo(f"sfmtool {pkg_version}")


def deduce_workspace(paths: set[Path]) -> Path:
    """Deduce the workspace directory from a set of directory paths.

    Args:
        paths: Set of directory paths within the workspace.

    Returns:
        Path to the workspace directory

    Raises:
        click.ClickException: If no workspace is found
    """
    if not paths:
        raise click.ClickException("No input paths provided to deduce workspace.")

    common_parent = Path(os.path.commonpath([str(p.absolute()) for p in paths]))

    workspace_dir = find_workspace_for_path(common_parent)
    if workspace_dir is None:
        raise click.ClickException(
            "No workspace found for paths. "
            "Please initialize a workspace using 'sfm ws init' in a parent directory."
        )

    return workspace_dir
