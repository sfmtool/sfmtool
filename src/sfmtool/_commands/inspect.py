# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruction inspection command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._inspect_summary import print_reconstruction_summary
from .._inspect_graphs import print_covisibility_graph, print_frustum_intersection_graph
from .._inspect_depth import print_z_range
from .._inspect_images import print_images_table
from .._inspect_metrics import print_metrics_analysis
from .._sfmtool import SfmrReconstruction


@click.command("inspect")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True))
@click.option(
    "--coviz",
    "coviz_flag",
    is_flag=True,
    help="Construct and print the covisibility graph.",
)
@click.option(
    "--z-range",
    "z_range_flag",
    is_flag=True,
    help="Print per-image Z depth ranges and histograms from stored depth statistics.",
)
@click.option(
    "--frustum",
    "frustum_flag",
    is_flag=True,
    help="Construct and print the frustum intersection graph.",
)
@click.option(
    "--images",
    "images_flag",
    is_flag=True,
    help="Print per-image connectivity information table.",
)
@click.option(
    "--metrics",
    "metrics_flag",
    is_flag=True,
    help="Print per-image metrics analysis (reprojection error breakdown).",
)
@click.option(
    "--range",
    "-r",
    "range_expr",
    help="Range expression of file numbers to include (e.g. '1-10'). Only with --metrics.",
)
@click.option(
    "--near-percentile",
    "near_percentile",
    type=click.FloatRange(0.0, 100.0),
    default=5.0,
    help="Percentile for near Z plane (default: 5.0).",
)
@click.option(
    "--far-percentile",
    "far_percentile",
    type=click.FloatRange(0.0, 100.0),
    default=95.0,
    help="Percentile for far Z plane (default: 95.0).",
)
@click.option(
    "--samples",
    "num_samples",
    type=click.IntRange(min=100),
    default=100,
    help="Number of Monte Carlo samples per frustum pair (default: 100).",
)
def inspect(
    reconstruction_path,
    coviz_flag,
    z_range_flag,
    frustum_flag,
    images_flag,
    metrics_flag,
    range_expr,
    near_percentile,
    far_percentile,
    num_samples,
):
    """Inspect the contents of a .sfmr file.

    Without any options, prints a summary of the reconstruction including
    metadata, camera info, and statistics.

    With --coviz, constructs and prints the covisibility graph showing
    which images share 3D points and how many points they share.

    With --z-range, prints per-image Z depth ranges and histograms from
    the depth statistics stored in the .sfmr file.

    With --frustum, constructs and prints the frustum intersection graph
    showing which camera frustums overlap and their estimated intersection
    volumes.

    With --images, prints a detailed per-image connectivity table showing
    for each image: number of observations, distances to other cameras,
    closest images by shared observations, and graph connectivity metrics.

    With --metrics, prints per-image metrics analysis showing reprojection
    error breakdown (mean, median, max), observation count, and mean track
    length for each image.

    RECONSTRUCTION_PATH must be a .sfmr file.

    Examples:

        sfm inspect reconstruction.sfmr

        sfm inspect reconstruction.sfmr --coviz

        sfm inspect reconstruction.sfmr --z-range

        sfm inspect reconstruction.sfmr --frustum

        sfm inspect reconstruction.sfmr --frustum --near-percentile 10 --far-percentile 90

        sfm inspect reconstruction.sfmr --images

        sfm inspect reconstruction.sfmr --metrics

        sfm inspect reconstruction.sfmr --metrics --range 1-10
    """
    reconstruction_path = Path(reconstruction_path)

    active_modes = sum(
        [coviz_flag, z_range_flag, frustum_flag, images_flag, metrics_flag]
    )
    if active_modes > 1:
        raise click.UsageError(
            "--coviz, --z-range, --frustum, --images, and --metrics are mutually exclusive."
        )

    if range_expr is not None and not metrics_flag:
        raise click.UsageError("--range can only be used with --metrics.")

    if not frustum_flag:
        if near_percentile != 5.0:
            raise click.UsageError("--near-percentile can only be used with --frustum.")
        if far_percentile != 95.0:
            raise click.UsageError("--far-percentile can only be used with --frustum.")
        if num_samples != 100:
            raise click.UsageError("--samples can only be used with --frustum.")

    if frustum_flag and near_percentile >= far_percentile:
        raise click.UsageError(
            f"--near-percentile ({near_percentile}) must be less than "
            f"--far-percentile ({far_percentile})."
        )

    if reconstruction_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
        )

    try:
        recon_name = reconstruction_path.name

        if metrics_flag:
            print_metrics_analysis(
                reconstruction_path, recon_name=recon_name, range_expr=range_expr
            )
        else:
            recon = SfmrReconstruction.load(reconstruction_path)

            if coviz_flag:
                print_covisibility_graph(recon, recon_name=recon_name)
            elif z_range_flag:
                print_z_range(recon, recon_name=recon_name)
            elif frustum_flag:
                print_frustum_intersection_graph(
                    recon,
                    near_percentile=near_percentile,
                    far_percentile=far_percentile,
                    num_samples=num_samples,
                    recon_name=recon_name,
                )
            elif images_flag:
                print_images_table(recon, recon_name=recon_name)
            else:
                print_reconstruction_summary(recon, recon_name=recon_name)
    except Exception as e:
        raise click.ClickException(str(e))
