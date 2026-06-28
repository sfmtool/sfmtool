# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""SIFT feature extraction command."""

import os
import textwrap
from pathlib import Path

import click
from deadline.job_attachments.api import summarize_path_list

from .._cli_utils import timed_command
from .._filenames import expand_paths
from .._sfmtool import RangeExpr
from .._workspace import find_workspace_for_path, load_workspace_config
from ..sift.file import (
    SiftExtractionError,
    draw_sift_features,
    get_used_features_from_reconstruction,
    image_files_to_sift_files,
)


@click.command("sift")
@timed_command
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--extract",
    "-e",
    "extract_flag",
    is_flag=True,
    help="Extract features from the specified image files.",
)
@click.option(
    "--draw",
    "-d",
    "draw_output_dir",
    type=click.Path(),
    help="Draw SIFT features with ellipses on images and save to this directory.",
)
@click.option(
    "--filter-sfm",
    "filter_sfm_path",
    type=click.Path(exists=True),
    help="Only draw features used in this .sfm reconstruction file (only with --draw).",
)
@click.option(
    "--range",
    "-r",
    "range_expr",
    help="A range expression of file numbers to use from the input directories",
)
@click.option(
    "--num-threads",
    "-t",
    type=int,
    default=-1,
    help="Number of threads for feature extraction (default: -1 uses all available cores)",
)
@click.option(
    "--tool",
    type=click.Choice(["colmap", "opencv", "sfmtool"], case_sensitive=False),
    default=None,
    help="Feature extraction tool. If not specified, uses workspace configuration.",
)
@click.option(
    "--dsp/--no-dsp",
    "domain_size_pooling",
    default=None,
    help="Enable/disable domain size pooling (COLMAP only, default: disabled for better performance)",
)
def sift(
    paths,
    extract_flag,
    draw_output_dir,
    filter_sfm_path,
    range_expr,
    num_threads,
    tool,
    domain_size_pooling,
):
    """Extract SIFT features and visualize .sift feature files.

    Use `sfm inspect <FILE.sift>` to inspect a feature file.
    """
    # Count how many modes are active
    active_modes = sum([extract_flag, draw_output_dir is not None])
    if active_modes != 1:
        raise click.UsageError("Exactly one of --extract or --draw must be specified.")

    # Validate --filter-sfm is only used with --draw
    if filter_sfm_path and not draw_output_dir:
        raise click.UsageError("--filter-sfm can only be used with --draw mode.")

    # Default to the current directory when no paths are given
    if not paths:
        paths = (".",)

    paths = [Path(p) for p in paths]
    if draw_output_dir:
        draw_output_dir = Path(draw_output_dir)
    if filter_sfm_path:
        filter_sfm_path = Path(filter_sfm_path)

    numbers = None
    if range_expr:
        numbers = RangeExpr(range_expr)

    # Convert any directories into lists of image paths for processing
    filenames = expand_paths(
        paths, extensions=(".png", ".jpg", ".jpeg"), numbers=numbers
    )

    # Validate we have files to process
    if not filenames:
        raise click.UsageError(
            "There were no image files to process in the provided directories."
        )

    # Validate CLI arguments before mode branching
    if domain_size_pooling is not None and tool is None:
        raise click.UsageError(
            "--dsp/--no-dsp can only be used with --tool. "
            "To change DSP settings, reinitialize the workspace with 'sfm ws init --dsp'."
        )

    # Determine feature tool and options based on CLI vs workspace
    tool_from_cli = tool is not None
    feature_prefix_dir = None

    if tool_from_cli:
        if domain_size_pooling is not None and tool.lower() != "colmap":
            raise click.UsageError(
                f"The --dsp/--no-dsp option is only supported for COLMAP, not {tool}"
            )

        if tool.lower() == "colmap":
            from ..sift.extract_colmap import get_colmap_feature_options

            dsp = domain_size_pooling if domain_size_pooling is not None else False
            feature_options = get_colmap_feature_options(domain_size_pooling=dsp)
        elif tool.lower() == "sfmtool":
            from ..sift.extract_sfmtool import get_default_sfmtool_feature_options

            feature_options = get_default_sfmtool_feature_options()
        else:  # opencv
            from ..sift.extract_opencv import get_default_opencv_feature_options

            feature_options = get_default_opencv_feature_options()
    else:
        # Workspace mode: detect workspace and use its configuration
        absolute_paths = [os.path.abspath(f) for f in filenames]
        common_parent = Path(os.path.commonpath(absolute_paths))
        if common_parent.is_file():
            common_parent = common_parent.parent

        workspace_dir = find_workspace_for_path(common_parent)
        if workspace_dir is None:
            raise click.ClickException(
                f"No workspace found for images at {common_parent}. "
                "Either:\n"
                "  1. Initialize a workspace with 'sfm ws init', or\n"
                "  2. Specify --tool explicitly"
            )

        try:
            config = load_workspace_config(workspace_dir)
        except RuntimeError as e:
            raise click.ClickException(str(e))
        tool = config["feature_tool"]
        feature_options = config["feature_options"]
        feature_prefix_dir = config["feature_prefix_dir"]

        click.echo(f"Using workspace: {workspace_dir}")
        click.echo(f"  Feature tool: {tool}")

    if extract_flag:
        click.echo(
            f"Given {len(paths)} path(s) to process, expanded to {len(filenames)} filename(s):"
        )
        click.echo(textwrap.indent(summarize_path_list(filenames), "  "))

        try:
            image_files_to_sift_files(
                filenames,
                num_threads=num_threads,
                feature_tool=tool,
                feature_options=feature_options,
                feature_prefix_dir=feature_prefix_dir,
            )
        except SiftExtractionError as e:
            raise click.ClickException(str(e))

    elif draw_output_dir:
        draw_output_dir.mkdir(parents=True, exist_ok=True)

        click.echo(
            f"Given {len(paths)} path(s) to process, expanded to {len(filenames)} filename(s)"
        )

        if not filenames:
            raise click.UsageError(
                "There were no image files to process in the provided directories."
            )

        # Load reconstruction once if --filter-sfm was specified
        filter_recon = None
        if filter_sfm_path:
            from sfmtool._sfmtool import SfmrReconstruction

            filter_recon = SfmrReconstruction.load(filter_sfm_path)

        for image_path in filenames:
            image_basename = image_path.name
            output_path = draw_output_dir / image_basename

            try:
                feature_indices = None
                if filter_recon:
                    feature_indices = get_used_features_from_reconstruction(
                        filter_recon, image_path
                    )
                    if len(feature_indices) == 0:
                        click.echo(
                            f"Warning: {image_basename} not found in reconstruction "
                            f"or has no features used",
                            err=True,
                        )

                draw_sift_features(
                    image_path,
                    output_path,
                    feature_indices=feature_indices,
                    feature_tool=tool,
                    feature_options=feature_options,
                )
            except FileNotFoundError as e:
                click.echo(f"Error: {e}", err=True)
            except Exception as e:
                click.echo(f"Error processing {image_path}: {e}", err=True)

        click.echo(f"\nAll visualizations saved to: {draw_output_dir}")
