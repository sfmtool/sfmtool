# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a .sfmr file to COLMAP binary format."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from ._range_options import apply_range_filter, range_options, validate_range_options


@click.command("to-colmap-bin")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_sfmr", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@range_options
def to_colmap_bin(
    input_sfmr: str,
    output_dir: str,
    range_expr: str | None,
    filter_points: bool,
):
    """Convert a .sfmr file to COLMAP .bin format.

    Exports the reconstruction to COLMAP's binary format, creating files
    in the output directory:

    \b
    - cameras.bin: Camera intrinsic parameters
    - images.bin: Image poses and camera assignments
    - points3D.bin: 3D point cloud with tracks
    - rigs.bin: Rig definitions
    - frames.bin: Frame groupings

    This is useful for importing sfmtool reconstructions into COLMAP for
    further processing, visualization, or dense reconstruction.

    Example usage:

    \b
        sfm to-colmap-bin reconstruction.sfmr colmap_output/
        sfm to-colmap-bin reconstruction.sfmr colmap_output/ -r 10-50
        sfm to-colmap-bin reconstruction.sfmr colmap_output/ -r 10-50 --filter-points
        colmap gui --import_path colmap_output/
    """
    from ..colmap.io import save_colmap_binary
    from .._sfmtool.reconstruction import SfmrReconstruction

    input_path = Path(input_sfmr)
    output_path = Path(output_dir)

    if input_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Input must be a .sfmr file: {input_path}")

    validate_range_options(range_expr, filter_points)

    try:
        recon = SfmrReconstruction.load(input_path)

        if range_expr is not None:
            recon = apply_range_filter(recon, range_expr, filter_points, echo=print)

        save_colmap_binary(recon, output_path)
    except Exception as e:
        raise click.ClickException(str(e))
