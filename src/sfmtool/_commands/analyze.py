# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruction analysis command."""

import math
from pathlib import Path

import click

from .._cli_utils import timed_command
from ._sfmr_path import check_sfmr_path
from ..analyze.graphs import print_covisibility_graph, print_frustum_intersection_graph
from ..analyze.depth import print_depth_reliability, print_z_range
from ..analyze.images import print_images_table
from ..analyze.metrics import print_metrics_analysis
from .._sfmtool.reconstruction import SfmrReconstruction


def _require_finite(ctx, param, value):
    """Reject NaN and infinity, which FloatRange lets through."""
    if value is not None and not math.isfinite(value):
        raise click.BadParameter(f"{value} is not a finite number.")
    return value


@click.command("analyze")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True, dir_okay=False))
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
    "--depth-reliability",
    "depth_reliability_flag",
    is_flag=True,
    help="Print per-point triangulation conditioning (inverse-depth z-score, "
    "condition number) and the point-or-bearing likelihood-ratio test, with the "
    "points whose verdict disagrees with how they are stored.",
)
@click.option(
    "--sigma-px",
    "sigma_px",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    callback=_require_finite,
    help="Per-axis pixel noise the point-or-bearing test weights its rays by "
    "(default: the reconstruction's measured reprojection noise). Only with "
    "--depth-reliability.",
)
@click.option(
    "--depth-likelihood-ratio-threshold",
    "depth_likelihood_ratio_threshold",
    type=click.FloatRange(min=0.0),
    default=None,
    callback=_require_finite,
    help="The threshold the point-or-bearing test's verdict applies (default: 25): "
    "a point is finite when its bearing cost and either its depth score or its "
    "midpoint bound reach it. Only with --depth-reliability.",
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
def analyze(
    reconstruction_path,
    coviz_flag,
    z_range_flag,
    frustum_flag,
    images_flag,
    metrics_flag,
    depth_reliability_flag,
    sigma_px,
    depth_likelihood_ratio_threshold,
    range_expr,
    near_percentile,
    far_percentile,
    num_samples,
):
    """Run a deep-analysis report on a .sfmr reconstruction.

    Exactly one analysis mode must be selected. For a quick summary of any
    sfmtool file, use `sfm inspect` instead.

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

    With --depth-reliability, prints per-point triangulation conditioning: the
    inverse-depth z-score (depth / sigma, low => near-infinity) and the
    normal-matrix condition number, summarised across the finite points. It
    then reports the point-or-bearing likelihood-ratio test: the measured
    reprojection noise it weights rays by, its depth score over finite points
    and points at infinity, and the points whose verdict disagrees with how
    they are stored. --sigma-px and --depth-likelihood-ratio-threshold
    override the test's noise level and threshold.

    RECONSTRUCTION_PATH must be a .sfmr file.

    Examples:

        sfm analyze reconstruction.sfmr --coviz

        sfm analyze reconstruction.sfmr --z-range

        sfm analyze reconstruction.sfmr --frustum

        sfm analyze reconstruction.sfmr --frustum --near-percentile 10 --far-percentile 90

        sfm analyze reconstruction.sfmr --images

        sfm analyze reconstruction.sfmr --metrics

        sfm analyze reconstruction.sfmr --metrics --range 1-10

        sfm analyze reconstruction.sfmr --depth-reliability

        sfm analyze reconstruction.sfmr --depth-reliability --sigma-px 0.5
    """
    reconstruction_path = Path(reconstruction_path)

    active_modes = sum(
        [
            coviz_flag,
            z_range_flag,
            frustum_flag,
            images_flag,
            metrics_flag,
            depth_reliability_flag,
        ]
    )
    if active_modes == 0:
        raise click.UsageError(
            "Select an analysis mode: "
            "--coviz, --z-range, --frustum, --images, --metrics, or "
            "--depth-reliability."
        )
    if active_modes > 1:
        raise click.UsageError(
            "--coviz, --z-range, --frustum, --images, --metrics, and "
            "--depth-reliability are mutually exclusive."
        )

    if range_expr is not None and not metrics_flag:
        raise click.UsageError("--range can only be used with --metrics.")

    if not depth_reliability_flag:
        if sigma_px is not None:
            raise click.UsageError(
                "--sigma-px can only be used with --depth-reliability."
            )
        if depth_likelihood_ratio_threshold is not None:
            raise click.UsageError(
                "--depth-likelihood-ratio-threshold can only be used with "
                "--depth-reliability."
            )

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

    reconstruction_path = check_sfmr_path(reconstruction_path, "Reconstruction path")

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
            elif depth_reliability_flag:
                print_depth_reliability(
                    recon,
                    recon_name=recon_name,
                    sigma_px=sigma_px,
                    threshold=depth_likelihood_ratio_threshold,
                )
    except Exception as e:
        raise click.ClickException(str(e))
