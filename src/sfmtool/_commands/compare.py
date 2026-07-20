# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare reconstructions command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._compare import compare_reconstructions
from .._sfmtool.reconstruction import SfmrReconstruction


def _parse_labels(text: str) -> tuple[str, str]:
    """Parse a ``LEFT,RIGHT`` column-label string for --strips-labels."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2 or not all(parts):
        raise click.UsageError(f"--strips-labels must be 'LEFT,RIGHT' (got {text!r})")
    return parts[0], parts[1]


@click.command("compare")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction1", type=click.Path(exists=True), required=True)
@click.argument("reconstruction2", type=click.Path(exists=True), required=True)
@click.option(
    "--by-coordinate/--by-feature-index",
    "by_coordinate",
    default=None,
    help="Match corresponding 3D points by 2D keypoint coordinate (works across "
    "different SIFT backends) or by feature index (requires identical .sift "
    "files). Default: auto-select coordinate matching when every shared image "
    "uses a different SIFT file.",
)
@click.option(
    "--pixel-threshold",
    type=click.FloatRange(min=0, min_open=True),
    default=2.0,
    show_default=True,
    help="Max 2D keypoint distance (pixels) for --by-coordinate matching.",
)
@click.option(
    "--fragments",
    is_flag=True,
    default=False,
    help="Always print the fragment decomposition (RANSAC similarity components "
    "over the shared cameras). Without the flag the section appears only when "
    "the decomposition finds more than one component or outlier frames.",
)
@click.option(
    "--fragment-pos-threshold",
    type=click.FloatRange(min=0, min_open=True),
    default=3.5,
    show_default=True,
    help="Fragment consensus: max position error as % of the reference scene "
    "scale for a camera to join a component.",
)
@click.option(
    "--fragment-rot-threshold",
    type=click.FloatRange(min=0, min_open=True),
    default=5.0,
    show_default=True,
    help="Fragment consensus: max rotation error in degrees for a camera to "
    "join a component.",
)
@click.option(
    "--fragment-min-size",
    type=click.IntRange(min=2),
    default=5,
    show_default=True,
    help="Smallest camera count that still counts as a fragment component; "
    "smaller consensus sets are reported as individual outlier frames.",
)
@click.option(
    "--strips",
    "strips_out",
    type=click.Path(),
    default=None,
    help="Render the corresponding points the two solves place most differently "
    "as a side-by-side patch-strip montage written to this PNG (reference left, "
    "target right). Needs each reconstruction's workspace images and .sift files.",
)
@click.option(
    "--strips-num",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="Number of differing points to render with --strips.",
)
@click.option(
    "--strips-views",
    type=click.IntRange(min=0),
    default=8,
    show_default=True,
    help="Cap tiles (observing views) per strip with --strips (0 = all).",
)
@click.option(
    "--strips-context",
    type=click.IntRange(min=0),
    default=96,
    show_default=True,
    help="Render wider NxN-pixel context tiles around each point (must exceed "
    "the 32px patch), with a green box marking the validated extent (0 = off).",
)
@click.option(
    "--strips-rank",
    type=click.Choice(
        [
            "overview",
            "distance",
            "view-angle",
            "ncc",
            "ncc-gap",
            "image-radius",
            "feature-size",
            "world-size",
        ]
    ),
    default="overview",
    show_default=True,
    help="What --strips surfaces: overview (a few points from each category), or a "
    "single quantity - distance (alignment disagreement), view-angle "
    "(triangulation angle), ncc (per-solve photoconsistency), ncc-gap (gap "
    "between the solves; slower), image-radius (keypoint distance from the "
    "principal point), feature-size (keypoint feature size in pixels), or "
    "world-size (feature size as a metric surface footprint). Use --strips-end "
    "to pick the end of the axis.",
)
@click.option(
    "--strips-end",
    type=click.Choice(["high", "low"]),
    default=None,
    help="Which end of the --strips-rank axis to show (default: the axis's natural "
    "end, e.g. high for distance/ncc-gap/image-radius, low for view-angle/ncc).",
)
@click.option(
    "--strips-refine/--strips-no-refine",
    default=True,
    show_default=True,
    help="Refine patch normals for photoconsistency before rendering --strips.",
)
@click.option(
    "--strips-labels",
    default="reference,target",
    show_default=True,
    metavar="LEFT,RIGHT",
    help="Names for the two columns in --strips (RECONSTRUCTION1,RECONSTRUCTION2), "
    "used in the headers and the per-row R/T and unique-only labels.",
)
def compare(
    reconstruction1,
    reconstruction2,
    by_coordinate,
    pixel_threshold,
    fragments,
    fragment_pos_threshold,
    fragment_rot_threshold,
    fragment_min_size,
    strips_out,
    strips_num,
    strips_views,
    strips_context,
    strips_rank,
    strips_end,
    strips_refine,
    strips_labels,
):
    """Compare two .sfmr files.

    This command compares two .sfmr files by:
    - Aligning them and reporting the transformation
    - Comparing camera intrinsics
    - Comparing images and their extrinsics
    - Decomposing the shared cameras into internally-rigid fragments
    - Comparing feature usage for matching images
    - Comparing corresponding 3D points

    The fragment decomposition detects solves whose cameras form several
    internally-consistent groups at different scales/orientations (which a
    single alignment papers over), plus individually misplaced frames. It is
    printed when it finds anything beyond one clean component; --fragments
    prints it always.

    RECONSTRUCTION1 and RECONSTRUCTION2 must be .sfmr files.
    RECONSTRUCTION1 will be used as the reference for alignment.

    Corresponding 3D points are matched either by feature index (the default
    when both reconstructions share their .sift files) or by 2D keypoint
    coordinate (``--by-coordinate``), which works even when the two
    reconstructions were built with different feature backends.

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
        recon1 = SfmrReconstruction.load(recon1_path)
        recon2 = SfmrReconstruction.load(recon2_path)

        compare_reconstructions(
            recon1,
            recon2,
            recon1_name=recon1_path.name,
            recon2_name=recon2_path.name,
            by_coordinate=by_coordinate,
            pixel_threshold=pixel_threshold,
            fragments=fragments,
            fragment_pos_threshold=fragment_pos_threshold,
            fragment_rot_threshold=fragment_rot_threshold,
            fragment_min_size=fragment_min_size,
            strips_out=strips_out,
            strips_num=strips_num,
            strips_views=strips_views,
            strips_context=strips_context,
            strips_rank=strips_rank,
            strips_end=strips_end,
            strips_refine=strips_refine,
            strips_labels=_parse_labels(strips_labels),
        )
    except Exception as e:
        raise click.ClickException(str(e))
