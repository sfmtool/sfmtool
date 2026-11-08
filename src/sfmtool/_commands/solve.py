# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Structure from motion solve command."""

import os
import tempfile
from pathlib import Path

import click

from .._cli_utils import timed_command
from .._filenames import expand_paths
from .._sfmtool.reconstruction import RangeExpr
from ..camera.cameras import CAMERA_MODEL_NAMES
from ._range_options import (
    matches_image_filter,
    parse_image_selection,
    range_option,
    resolve_list_images,
)

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@click.command("solve")
@timed_command
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--colmap-dir",
    "colmap_dir",
    type=click.Path(file_okay=False),
    help="Directory for the COLMAP database and intermediate files.",
)
@click.option(
    "--max-features",
    "max_feature_count",
    type=click.IntRange(min=1),
    help="Maximum number of features to use from each image.",
)
@range_option(
    "A range expression of file numbers to use from the input directories, "
    "or from the images of a .matches file.",
    on="paths",
    list_help_extra=" With a .matches file, use only the file's images that "
    "are among the list's images, with the list read from the file's "
    "workspace.",
)
@click.option(
    "--incremental",
    "-i",
    "incremental",
    is_flag=True,
    help="Run incremental structure from motion.",
)
@click.option(
    "--global",
    "-g",
    "global_mode",
    is_flag=True,
    help="Run global structure from motion using GLOMAP.",
)
@click.option(
    "--sfmr-dir",
    "sfmr_dir",
    type=click.Path(),
    help="Directory for .sfmr files. Default: workspace/sfmr.",
)
@click.option(
    "--seed",
    "-s",
    "random_seed",
    type=int,
    default=None,
    help="Random seed for reproducible reconstructions.",
)
@click.option(
    "--output",
    "-o",
    "output_sfm_file",
    type=click.Path(),
    help="Output .sfmr file path.",
)
@click.option(
    "--seq-overlap",
    "seq_overlap",
    type=str,
    help="Sequential overlap mode: 'WINDOW,OVERLAP' (e.g., '100,20').",
)
@click.option(
    "--refine-rig/--no-refine-rig",
    "refine_rig",
    default=True,
    help="Refine sensor-from-rig poses during bundle adjustment.",
)
@click.option(
    "--flow-match",
    "flow_match",
    is_flag=True,
    help="Use optical flow-based matching instead of exhaustive descriptor matching.",
)
@click.option(
    "--flow-preset",
    "flow_preset",
    type=click.Choice(["fast", "default", "high_quality"]),
    default="default",
    help="Optical flow quality preset for --flow-match. Default: default.",
)
@click.option(
    "--flow-skip",
    "flow_wide_baseline_skip",
    type=click.IntRange(min=1),
    default=5,
    help="Sliding window size for --flow-match. Default: 5.",
)
@click.option(
    "--camera-model",
    "camera_model",
    type=click.Choice(CAMERA_MODEL_NAMES, case_sensitive=False),
    default=None,
    help="Camera model to use: overrides auto-detection, and refits to this "
    "model the cameras a .camrig or camera_config.json supplies.",
)
@click.option(
    "--detect-infinity/--no-detect-infinity",
    "detect_infinity",
    default=True,
    help="Reclassify points whose depth the solve could not pin down as "
    "points at infinity. Enabled by default.",
)
def solve(
    paths,
    colmap_dir,
    max_feature_count,
    range_expr,
    named_image_list,
    incremental,
    global_mode,
    sfmr_dir,
    random_seed,
    output_sfm_file,
    seq_overlap,
    refine_rig,
    flow_match,
    flow_preset,
    flow_wide_baseline_skip,
    camera_model,
    detect_infinity,
):
    """Run structure from motion on images or a .matches file.

    Input can be image paths/directories (runs feature matching internally)
    or a single .matches file (uses pre-computed matches).

    Examples:
        # From images (incremental)
        sfm solve -i path/to/images/

        # From a pre-computed .matches file
        sfm solve -i tvg-matches/my.matches

        # Use global SfM (GLOMAP)
        sfm solve -g path/to/images/

        # Sequential overlap mode
        sfm solve -i path/to/images/ --seq-overlap 100,20

        # The images of a named image list of the workspace
        sfm solve -i -r walk1_sharp

        # The first 30 images of the list, from its pre-computed matches
        sfm solve -g tvg-matches/walk1_sharp.matches -r 'walk1_sharp[:30]'
    """
    from ..cli import deduce_workspace
    from .._solve_driver import _run_sequential_overlap_sfm, _run_sfm

    selection = parse_image_selection(range_expr, named_image_list)
    range_expr = selection.range_expr
    given_matches = len(paths) == 1 and paths[0].endswith(".matches")
    if selection.list_name is not None and paths and not given_matches:
        raise click.UsageError(
            "A named image list replaces the image paths; give image paths or "
            "a list, not both. A list given with a single .matches file keeps "
            "the file's images that are among the list's images."
        )
    if not paths and selection.list_name is None:
        raise click.UsageError("Must provide image paths or a .matches file.")

    if incremental and global_mode:
        raise click.UsageError(
            "Cannot specify both --incremental and --global. Choose one mode."
        )

    if not incremental and not global_mode:
        raise click.UsageError(
            "Must specify either --incremental (-i) or --global (-g) mode."
        )

    if seq_overlap and output_sfm_file:
        raise click.UsageError(
            "--seq-overlap cannot be used with --output. "
            "Sequential overlap mode only supports automatic filename generation."
        )

    if colmap_dir:
        colmap_dir = Path(colmap_dir)

    # Detect whether input is a single .matches file or image paths
    matches_file = None
    if given_matches:
        matches_file = Path(paths[0])
        if not matches_file.exists():
            raise click.UsageError(f"Matches file not found: {matches_file}")
    elif any(p.endswith(".matches") for p in paths):
        raise click.UsageError(
            "Cannot mix .matches files with image paths. "
            "Provide either a single .matches file or image paths/directories."
        )

    if matches_file is not None:
        if seq_overlap:
            raise click.UsageError("--seq-overlap cannot be used with a .matches file.")
        if flow_match:
            raise click.UsageError("--flow-match cannot be used with a .matches file.")

        # The COLMAP mapper reads its correspondence graph from the database's
        # two-view geometry table, so a clusters-bearing file — which by
        # format never carries two-view geometries — would hand it an empty
        # graph and register nothing. Say so instead of solving into silence.
        from .._sfmtool.fileio import read_matches_metadata

        try:
            matches_metadata = read_matches_metadata(str(matches_file))
        except Exception as e:
            raise click.ClickException(str(e))
        if matches_metadata.get("has_clusters", False):
            raise click.UsageError(
                f"{matches_file} is a clusters-bearing .matches file and "
                "carries no two-view geometries, which the mapper needs. "
                f"Run 'sfm match --derive-pairs {matches_file}' first and "
                "solve from the .matches file it writes."
            )

        image_filter = matches_image_filter(
            selection, matches_file, extensions=_IMAGE_EXTENSIONS
        )

        try:
            if colmap_dir:
                _run_sfm(
                    [],
                    None,
                    colmap_dir,
                    max_feature_count,
                    incremental,
                    sfmr_dir,
                    random_seed,
                    output_sfm_file=output_sfm_file,
                    refine_rig=refine_rig,
                    camera_model=camera_model,
                    matches_file=matches_file,
                    image_filter=image_filter,
                    detect_infinity=detect_infinity,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="colmap_") as temp_colmap_dir:
                    _run_sfm(
                        [],
                        None,
                        Path(temp_colmap_dir),
                        max_feature_count,
                        incremental,
                        sfmr_dir,
                        random_seed,
                        output_sfm_file=output_sfm_file,
                        refine_rig=refine_rig,
                        camera_model=camera_model,
                        matches_file=matches_file,
                        image_filter=image_filter,
                        detect_infinity=detect_infinity,
                    )
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e))
        return

    # Standard path: solve from image paths, or from a named image list
    extensions = _IMAGE_EXTENSIONS
    if selection.list_name is not None:
        list_images = resolve_list_images(selection, extensions=extensions)
        absolute_paths = [Path(os.path.normpath(p)) for p in list_images.paths]
        workspace_dir = list_images.workspace_dir
    else:
        paths = [Path(p) for p in paths]

        numbers = None
        if range_expr:
            numbers = RangeExpr(range_expr)

        filenames = expand_paths(paths, extensions=extensions, numbers=numbers)

        if not filenames:
            raise click.UsageError(
                "No image files to process in the provided directories."
            )

        absolute_paths = [Path(os.path.normpath(os.path.abspath(p))) for p in filenames]

        workspace_dir = deduce_workspace({p.parent for p in absolute_paths})

    if seq_overlap:
        try:
            _run_sequential_overlap_sfm(
                absolute_paths,
                workspace_dir,
                colmap_dir,
                max_feature_count,
                incremental,
                sfmr_dir,
                random_seed,
                seq_overlap,
                refine_rig,
                camera_model=camera_model,
                detect_infinity=detect_infinity,
            )
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e))
        return

    matching_mode = "flow" if flow_match else "exhaustive"

    try:
        if colmap_dir:
            _run_sfm(
                absolute_paths,
                workspace_dir,
                colmap_dir,
                max_feature_count,
                incremental,
                sfmr_dir,
                random_seed,
                output_sfm_file=output_sfm_file,
                refine_rig=refine_rig,
                camera_model=camera_model,
                matching_mode=matching_mode,
                flow_preset=flow_preset,
                flow_wide_baseline_skip=flow_wide_baseline_skip,
                detect_infinity=detect_infinity,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="colmap_") as temp_colmap_dir:
                _run_sfm(
                    absolute_paths,
                    workspace_dir,
                    Path(temp_colmap_dir),
                    max_feature_count,
                    incremental,
                    sfmr_dir,
                    random_seed,
                    output_sfm_file=output_sfm_file,
                    refine_rig=refine_rig,
                    camera_model=camera_model,
                    matching_mode=matching_mode,
                    flow_preset=flow_preset,
                    flow_wide_baseline_skip=flow_wide_baseline_skip,
                    detect_infinity=detect_infinity,
                )
    except Exception as e:
        raise click.ClickException(str(e))
