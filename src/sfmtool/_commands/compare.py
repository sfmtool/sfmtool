# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare reconstructions command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._compare import compare_reconstructions
from .._sfmtool import SfmrReconstruction


@click.command("compare")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction1", type=click.Path(exists=True), required=True)
@click.argument("reconstruction2", type=click.Path(exists=True), required=True)
def compare(reconstruction1, reconstruction2):
    """Compare two .sfmr files.

    This command compares two .sfmr files by:
    - Aligning them and reporting the transformation
    - Comparing camera intrinsics
    - Comparing images and their extrinsics
    - Comparing feature usage for matching images

    RECONSTRUCTION1 and RECONSTRUCTION2 must be .sfmr files.
    RECONSTRUCTION1 will be used as the reference for alignment.

    Example:

        sfm compare reconstruction1.sfmr reconstruction2.sfmr
    """
    recon1_path = Path(reconstruction1)
    recon2_path = Path(reconstruction2)

    if recon1_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"First reconstruction must be a .sfmr file: {reconstruction1}"
        )
    if recon2_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Second reconstruction must be a .sfmr file: {reconstruction2}"
        )

    try:
        recon1 = SfmrReconstruction.load(str(recon1_path))
        recon2 = SfmrReconstruction.load(str(recon2_path))

        compare_reconstructions(
            recon1, recon2, recon1_name=recon1_path.name, recon2_name=recon2_path.name
        )
    except Exception as e:
        raise click.ClickException(str(e))
