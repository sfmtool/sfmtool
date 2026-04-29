# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a pinhole .sfmr reconstruction to a Nerfstudio dataset directory."""

from pathlib import Path

import click
import numpy as np
from openjd.model import IntRangeExpr

from .._cli_utils import timed_command
from .._filenames import number_from_filename


@click.command("to-nerfstudio")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_sfmr", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output dataset directory (default: <input>_nerfstudio/ next to the .sfmr).",
)
@click.option(
    "--num-downscales",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Number of image pyramid levels (writes images_2/, images_4/, ...). 0 disables.",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(min=1, max=100),
    default=95,
    show_default=True,
    help="JPEG quality for downsampled pyramid images.",
)
@click.option(
    "--include-colmap",
    is_flag=True,
    default=False,
    help="Also write a sparse/ directory with COLMAP .bin files.",
)
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
def to_nerfstudio(
    input_sfmr: str,
    output_dir: str | None,
    num_downscales: int,
    jpeg_quality: int,
    include_colmap: bool,
    range_expr: str | None,
    filter_points: bool,
):
    """Convert a pinhole .sfmr reconstruction to a Nerfstudio dataset.

    Writes transforms.json, sparse_pc.ply, images/, and pyramid directories
    in the layout that nerfstudio's NerfstudioDataParser accepts. The output
    is directly trainable with `ns-train nerfacto --data <output>`.

    Input cameras must be pinhole (zero distortion). Run `sfm undistort`
    first if your reconstruction has lens distortion.

    Example usage:

    \b
        sfm to-nerfstudio undistorted.sfmr -o my_dataset/
        sfm to-nerfstudio undistorted.sfmr -o my_dataset/ -r 10-50
        ns-train nerfacto --data my_dataset/
    """
    from .._sfmtool import SfmrReconstruction
    from .._to_nerfstudio import export_to_nerfstudio

    input_path = Path(input_sfmr)
    if input_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Input must be a .sfmr file: {input_path}")

    if filter_points and range_expr is None:
        raise click.UsageError("--filter-points requires --range")

    if output_dir is None:
        out_path = input_path.parent / f"{input_path.stem}_nerfstudio"
    else:
        out_path = Path(output_dir)

    try:
        recon = SfmrReconstruction.load(input_path)
        if range_expr is not None:
            recon = _apply_range_filter(recon, range_expr, filter_points)
    except Exception as e:
        raise click.ClickException(str(e))

    def _progress(current: int, total: int, name: str) -> None:
        if name:
            click.echo(f"  [{current + 1}/{total}] {name}")

    try:
        summary = export_to_nerfstudio(
            recon,
            out_path,
            num_downscales=num_downscales,
            jpeg_quality=jpeg_quality,
            include_colmap=include_colmap,
            progress_callback=_progress,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise click.ClickException(str(e))

    click.echo(
        f"Wrote nerfstudio dataset to {summary['output_dir']}: "
        f"{summary['image_count']} images, {summary['point_count']} points, "
        f"{'single' if summary['single_camera'] else 'multi'}-camera, "
        f"{summary['num_downscales']} pyramid levels"
        + (", + sparse/" if summary["include_colmap"] else "")
    )


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

    click.echo(
        f"  Applied range filter '{range_expr_str}': "
        f"keeping {len(keep_indices)} of {len(image_names)} images"
    )

    indices_arr = np.array(keep_indices, dtype=np.uint32)
    return recon.subset_by_image_indices(
        indices_arr, drop_orphaned_points=filter_points
    )
