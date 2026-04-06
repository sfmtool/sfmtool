# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Densify reconstruction command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._densify import densify_reconstruction
from .._sfmtool import SfmrReconstruction


@click.command("densify")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_sfmr", type=click.Path(exists=True))
@click.argument("output_sfmr", type=click.Path())
@click.option(
    "--max-features",
    "max_feature_count",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of features to use from each image (uses largest features)",
)
@click.option(
    "--sweep-window-size",
    type=int,
    default=30,
    help="Window size for sweep matching (default: 30)",
)
@click.option(
    "--distance-threshold",
    type=float,
    default=None,
    help="Maximum descriptor distance for matches (default: None)",
)
@click.option(
    "--ba-refine-focal-length",
    is_flag=True,
    help="Refine focal length in bundle adjustment",
)
@click.option(
    "--ba-refine-principal-point",
    is_flag=True,
    help="Refine principal point in bundle adjustment",
)
@click.option(
    "--ba-refine-extra-params",
    is_flag=True,
    help="Refine extra camera parameters in bundle adjustment",
)
@click.option(
    "--filter-max-reproj-error",
    type=float,
    default=4.0,
    help="Maximum reprojection error for filtering points (default: 4.0 pixels)",
)
@click.option(
    "--filter-min-track-length",
    type=int,
    default=3,
    help="Minimum track length (observations) for filtering points (default: 3)",
)
@click.option(
    "--filter-min-tri-angle",
    type=float,
    default=1.5,
    help="Minimum triangulation angle for filtering points (default: 1.5 degrees)",
)
@click.option(
    "--filter-isolated-median-ratio",
    type=float,
    default=2.0,
    help="Filter points more isolated than the median NN distance times this number (default: 2.0, 0 to disable)",
)
@click.option(
    "--close-pair-threshold",
    type=int,
    default=4,
    help="Maximum distance between image numbers to be considered 'close' (default: 4)",
)
@click.option(
    "--max-close-pairs",
    type=int,
    default=None,
    help="Maximum number of close pairs to keep (default: None = all close pairs)",
)
@click.option(
    "--max-distant-pairs",
    type=int,
    default=5000,
    help="Maximum number of distant pairs to keep (default: 5000)",
)
@click.option(
    "--distant-pair-search-multiplier",
    type=int,
    default=3,
    help="Search through this many times max-distant-pairs to find the best ones (default: 3)",
)
@click.option(
    "--frustum",
    "include_frustum_pairs",
    is_flag=True,
    help="Also find and match frustum intersection pairs (slower, finds pairs with no shared points yet).",
)
@click.option(
    "--enable-geometric-filtering",
    is_flag=True,
    help="Enable motion-invariant geometric filtering using affine shapes (experimental)",
)
@click.option(
    "--geometric-size-ratio-max",
    type=float,
    default=1.25,
    help="Maximum allowed size ratio for geometric filtering (default: 1.25)",
)
@click.option(
    "--geometric-angle-diff-max",
    type=float,
    default=15.0,
    help="Maximum allowed angle difference in degrees for geometric filtering (default: 15.0)",
)
def densify(
    input_sfmr: str,
    output_sfmr: str,
    max_feature_count: int | None,
    sweep_window_size: int,
    distance_threshold: float | None,
    ba_refine_focal_length: bool,
    ba_refine_principal_point: bool,
    ba_refine_extra_params: bool,
    filter_max_reproj_error: float,
    filter_min_track_length: int,
    filter_min_tri_angle: float,
    filter_isolated_median_ratio: float,
    close_pair_threshold: int,
    max_close_pairs: int | None,
    max_distant_pairs: int,
    distant_pair_search_multiplier: int,
    include_frustum_pairs: bool,
    enable_geometric_filtering: bool,
    geometric_size_ratio_max: float,
    geometric_angle_diff_max: float,
):
    """Densify matches in a .sfmr file.

    By default, finds covisibility pairs (pairs already sharing 3D points) and
    sweep-matches them at higher features to find new correspondences.

    With --frustum, additionally finds frustum intersection pairs (geometric
    overlap) that don't already share 3D points.

    Example usage:

        sfm densify input.sfmr output.sfmr --max-features 2048

        sfm densify input.sfmr output.sfmr --max-features 2048 --frustum

        sfm densify input.sfmr output.sfmr --ba-refine-focal-length
    """
    input_path = Path(input_sfmr)
    output_path = Path(output_sfmr)

    if input_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Input must be a .sfmr file: {input_path}")

    if output_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Output must be a .sfmr file: {output_path}")

    # Build bundle adjustment options
    ba_options = None
    if ba_refine_focal_length or ba_refine_principal_point or ba_refine_extra_params:
        ba_options = {
            "refine_focal_length": ba_refine_focal_length,
            "refine_principal_point": ba_refine_principal_point,
            "refine_extra_params": ba_refine_extra_params,
        }

    # Build geometric filtering config if enabled
    geometric_config = None
    if enable_geometric_filtering:
        from ..feature_match import GeometricFilterConfig

        geometric_size_ratio_min = 1.0 / geometric_size_ratio_max

        geometric_config = GeometricFilterConfig(
            enable_geometric_filtering=True,
            geometric_size_ratio_min=geometric_size_ratio_min,
            geometric_size_ratio_max=geometric_size_ratio_max,
            max_angle_difference=geometric_angle_diff_max,
        )

    try:
        click.echo(f"Loading reconstruction from {input_path.name}...")
        recon = SfmrReconstruction.load(input_path)
        click.echo(f"  Images: {recon.image_count}")
        click.echo(f"  Points: {recon.point_count}")
        click.echo(f"  Cameras: {recon.camera_count}")

        result = densify_reconstruction(
            recon=recon,
            max_features=max_feature_count,
            sweep_window_size=sweep_window_size,
            distance_threshold=distance_threshold,
            ba_options=ba_options,
            filter_max_reproj_error=filter_max_reproj_error,
            filter_min_track_length=filter_min_track_length,
            filter_min_tri_angle=filter_min_tri_angle,
            filter_isolated_median_ratio=filter_isolated_median_ratio,
            close_pair_threshold=close_pair_threshold,
            max_close_pairs=max_close_pairs,
            max_distant_pairs=max_distant_pairs,
            distant_pair_search_multiplier=distant_pair_search_multiplier,
            geometric_config=geometric_config,
            include_frustum_pairs=include_frustum_pairs,
        )

        click.echo(f"\nSaving to {output_path.name}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(
            str(output_path),
            operation="densify",
            tool_name="densify",
            tool_options={
                "max_features": max_feature_count,
                "sweep_window_size": sweep_window_size,
                "distance_threshold": distance_threshold,
                "ba_options": ba_options,
            },
        )
        click.echo("Done!")

    except Exception as e:
        raise click.ClickException(str(e))
