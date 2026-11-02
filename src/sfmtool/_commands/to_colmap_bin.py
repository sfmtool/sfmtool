# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a .sfmr file to COLMAP binary format."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from ._sfmr_path import check_sfmr_path
from ._range_options import (
    apply_image_selection,
    parse_image_selection,
    range_options,
    validate_range_options,
)


@click.command("to-colmap-bin")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_sfmr", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@range_options
def to_colmap_bin(
    input_sfmr: str,
    output_dir: str,
    range_expr: tuple[str, ...],
    named_image_list: str | None,
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

    input_path = check_sfmr_path(input_sfmr, "Input path")
    output_path = Path(output_dir)

    selection = parse_image_selection(range_expr, named_image_list)
    validate_range_options(selection, filter_points)

    try:
        recon = SfmrReconstruction.load(input_path)

        if selection.given:
            recon = apply_image_selection(
                recon, selection, filter_points, echo=print, sfmr_path=input_path
            )

        save_colmap_binary(recon, output_path)
    except Exception as e:
        raise click.ClickException(str(e))
