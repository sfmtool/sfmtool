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
def undistort(
    reconstruction_path,
):
    """Undistort all images in a reconstruction using camera parameters.

    For each image in the reconstruction, this command loads the corresponding
    camera model and undistorts the image. Output images maintain the same
    size as input images (scale is fixed to 1.0).

    Undistorted images are saved to a directory based on the .sfmr filename:
        reconstruction.sfmr -> reconstruction_undistorted/

    Camera parameters are saved to: <output_dir>/undistorted_cameras.json

    The undistorted cameras will be PINHOLE models with all distortion removed.

    RECONSTRUCTION_PATH must be a .sfmr file.

    Example:
        sfm undistort reconstruction.sfmr
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

        # Compute output directory based on .sfmr filename
        output_dir = (
            reconstruction_path.parent / f"{reconstruction_path.stem}_undistorted"
        )

        # Undistort all images
        image_count, output_dir_str = undistort_reconstruction_images(
            recon=recon,
            output_dir=output_dir,
        )

        click.echo(f"\nSuccessfully undistorted {image_count} images")
        click.echo(f"Output directory: {output_dir_str}")

    except Exception as e:
        raise click.ClickException(str(e))
