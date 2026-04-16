# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Undistort images command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._undistort_images import undistort_reconstruction_images


@click.command("undistort")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True))
@click.option(
    "--fit",
    type=click.Choice(["inside", "outside"]),
    default="inside",
    help=(
        "Pinhole fit mode. 'inside' (default) ensures no black borders — all "
        "output pixels are backed by source data. 'outside' ensures no "
        "cropping — all source pixels are included, but borders may be black."
    ),
)
@click.option(
    "--filter",
    "resampling_filter",
    type=click.Choice(["aniso", "bilinear"]),
    default="aniso",
    help=(
        "Resampling filter. 'aniso' (default) uses anisotropic filtering for "
        "higher quality, reducing aliasing in areas of high compression. "
        "'bilinear' is faster but may alias near image edges."
    ),
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Output directory. Defaults to <sfmr_stem>_undistorted/ next to the .sfmr file.",
)
def undistort(
    reconstruction_path,
    fit,
    resampling_filter,
    output_dir,
):
    """Undistort all images in a reconstruction using camera parameters.

    For each image in the reconstruction, this command loads the corresponding
    camera model and undistorts the image to a best-fit pinhole camera with
    square pixels.

    By default, undistorted images are saved next to the .sfmr file:
        reconstruction.sfmr -> reconstruction_undistorted/

    Camera parameters are saved to: <output_dir>/undistorted_cameras.json

    The undistorted cameras will be PINHOLE models with all distortion removed.

    RECONSTRUCTION_PATH must be a .sfmr file.

    Example:
        sfm undistort reconstruction.sfmr
        sfm undistort reconstruction.sfmr --fit outside
        sfm undistort reconstruction.sfmr -o /tmp/undistorted
    """
    reconstruction_path = Path(reconstruction_path)

    # Validate .sfmr extension
    if reconstruction_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
        )

    try:
        from .._sfmtool import SfmrReconstruction

        # Load reconstruction
        recon = SfmrReconstruction.load(reconstruction_path)

        # Compute output directory based on .sfmr filename if not specified
        if output_dir is None:
            output_dir = (
                reconstruction_path.parent / f"{reconstruction_path.stem}_undistorted"
            )
        else:
            output_dir = Path(output_dir)

        # Undistort all images
        image_count, output_dir_str = undistort_reconstruction_images(
            recon=recon,
            output_dir=output_dir,
            fit=fit,
            resampling_filter=resampling_filter,
        )

        click.echo(f"\nSuccessfully undistorted {image_count} images")
        click.echo(f"Output directory: {output_dir_str}")

    except Exception as e:
        raise click.ClickException(str(e))
