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
@click.option(
    "--range",
    "-r",
    "range_expr",
    help="A range expression of file numbers to use from the input directories.",
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
    help="Camera model to use (overrides auto-detection).",
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
    """
    from ..cli import deduce_workspace
    from .._solve_driver import _run_sequential_overlap_sfm, _run_sfm

    if not paths:
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
    if len(paths) == 1 and paths[0].endswith(".matches"):
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
        from .._sfmtool.io import read_matches_metadata

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
                    range_expr=range_expr,
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
                        range_expr=range_expr,
                        detect_infinity=detect_infinity,
                    )
        except click.UsageError:
            raise
        except Exception as e:
            raise click.ClickException(str(e))
        return

    # Standard path: solve from image paths
    paths = [Path(p) for p in paths]

    numbers = None
    if range_expr:
        numbers = RangeExpr(range_expr)

    filenames = expand_paths(
        paths, extensions=(".png", ".jpg", ".jpeg"), numbers=numbers
    )

    if not filenames:
        raise click.UsageError("No image files to process in the provided directories.")

    absolute_paths = [Path(os.path.normpath(os.path.abspath(p))) for p in filenames]

    workspace_dir = deduce_workspace({p.parent for p in absolute_paths})

    from ..camera.config import CameraConfigResolver
    from ..camera.setup import _check_camera_model_conflict

    camera_config_resolver = CameraConfigResolver(workspace_dir)
    _check_camera_model_conflict(absolute_paths, camera_config_resolver, camera_model)

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
