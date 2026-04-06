# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a COLMAP database from a .sfmr or .matches file."""

from pathlib import Path

import click

from .._cli_utils import timed_command


@click.command("to-colmap-db")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--out-db",
    "output_db_path",
    required=True,
    type=click.Path(),
    help="Output path for the COLMAP database file (e.g., database.db).",
)
@click.option(
    "--max-features",
    "max_features",
    type=click.IntRange(min=1),
    help="Maximum number of features per image (only for .sfmr input).",
)
@click.option(
    "--no-guided-matching",
    "no_guided_matching",
    is_flag=True,
    help="Disable pre-population of two-view geometries (only for .sfmr input).",
)
@click.option(
    "--camera-model",
    type=str,
    default=None,
    help="Camera model override (e.g., OPENCV, PINHOLE). Only for .matches input.",
)
def to_colmap_db(
    input_path,
    output_db_path,
    max_features,
    no_guided_matching,
    camera_model,
):
    """Create a COLMAP database from a .sfmr or .matches file.

    INPUT_PATH must be either a .sfmr reconstruction file or a .matches file.

    \b
    From a .sfmr file:
      Creates a database with camera intrinsics, pose priors, keypoints,
      descriptors, and optionally two-view geometries with fundamental
      matrices computed from relative poses (for guided matching).

    \b
    From a .matches file:
      Creates a database with cameras, keypoints, descriptors, and
      pre-computed matches/two-view geometries from the file.

    Example usage:

    \b
        # From a reconstruction (with guided matching data)
        sfm to-colmap-db reconstruction.sfmr --out-db database.db

        # From a reconstruction, skip two-view geometry pre-population
        sfm to-colmap-db reconstruction.sfmr --out-db database.db --no-guided-matching

        # From pre-computed matches
        sfm to-colmap-db matches.matches --out-db database.db
    """
    input_path = Path(input_path)
    output_db_path = Path(output_db_path)
    suffix = input_path.suffix.lower()

    if suffix == ".sfmr":
        _from_sfmr(input_path, output_db_path, max_features, no_guided_matching)
    elif suffix == ".matches":
        _from_matches(input_path, output_db_path, camera_model)
    else:
        raise click.UsageError(
            f"Input must be a .sfmr or .matches file, got: {input_path}"
        )


def _from_sfmr(input_path, output_db_path, max_features, no_guided_matching):
    """Create COLMAP DB from a .sfmr reconstruction."""
    from .._sfmtool import SfmrReconstruction
    from .._to_colmap_db import create_colmap_db_from_reconstruction
    from .._workspace import find_workspace_for_path, load_workspace_config

    workspace_dir = find_workspace_for_path(input_path.parent)
    if workspace_dir is not None:
        load_workspace_config(workspace_dir)

    try:
        recon = SfmrReconstruction.load(input_path)
        create_colmap_db_from_reconstruction(
            recon=recon,
            output_db_path=output_db_path,
            max_feature_count=max_features,
            populate_two_view_geometries=not no_guided_matching,
        )
    except Exception as e:
        raise click.ClickException(str(e))


def _from_matches(input_path, output_db_path, camera_model):
    """Create COLMAP DB from a .matches file."""
    from .._colmap_db import _setup_for_sfm_from_matches

    try:
        colmap_dir = output_db_path.parent
        db_path, _image_dir, _image_paths = _setup_for_sfm_from_matches(
            input_path,
            colmap_dir,
            camera_model=camera_model,
        )
        # Rename if the caller specified a different output path
        if db_path.resolve() != output_db_path.resolve():
            import shutil

            shutil.move(str(db_path), str(output_db_path))
            click.echo(f"Database: {output_db_path}")
    except Exception as e:
        raise click.ClickException(str(e))
