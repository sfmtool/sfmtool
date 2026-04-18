# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a .sfmr file to COLMAP binary format."""

from pathlib import Path

import click
import numpy as np
from openjd.model import IntRangeExpr

from .._cli_utils import timed_command
from .._filenames import number_from_filename


@click.command("to-colmap-bin")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_sfmr", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--range",
    "-r",
    "range_expr",
    default=None,
    help="Export only images whose file number matches this range expression "
    "(e.g. '10-50' or '0-9,20-29'). Observations on excluded images are dropped.",
)
@click.option(
    "--filter-points",
    "filter_points",
    is_flag=True,
    default=False,
    help="With --range, also drop 3D points that have no remaining observations. "
    "Default is to keep all 3D points.",
)
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
    from .._colmap_io import save_colmap_binary
    from .._sfmtool import SfmrReconstruction

    input_path = Path(input_sfmr)
    output_path = Path(output_dir)

    if input_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Input must be a .sfmr file: {input_path}")

    if filter_points and range_expr is None:
        raise click.UsageError("--filter-points requires --range")

    try:
        recon = SfmrReconstruction.load(input_path)

        if range_expr is not None:
            recon = _apply_range_filter(recon, range_expr, filter_points)

        save_colmap_binary(recon, output_path)
    except Exception as e:
        raise click.ClickException(str(e))


def _apply_range_filter(recon, range_expr_str: str, filter_points: bool):
    """Subset the reconstruction by image file number range."""
    range_numbers = set(IntRangeExpr.from_str(range_expr_str))

    image_names = recon.image_names
    keep_indices: list[int] = []
    for i, name in enumerate(image_names):
        file_number = number_from_filename(name)
        if file_number is not None and file_number in range_numbers:
            keep_indices.append(i)

    if not keep_indices:
        available_numbers = sorted(
            n
            for n in (number_from_filename(name) for name in image_names)
            if n is not None
        )
        raise ValueError(
            f"No images remain after applying range filter '{range_expr_str}'. "
            f"Available file numbers: {available_numbers}"
        )

    print(
        f"  Applied range filter '{range_expr_str}': "
        f"keeping {len(keep_indices)} of {len(image_names)} images"
    )

    indices_arr = np.array(keep_indices, dtype=np.uint32)
    return recon.subset_by_image_indices(
        indices_arr, drop_orphaned_points=filter_points
    )
