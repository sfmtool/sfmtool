# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Optical flow visualization command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._flow_viz import draw_flow_visualization


@click.command("flow")
@timed_command
@click.help_option("--help", "-h")
@click.argument("image1", type=click.Path(exists=True))
@click.argument("image2", type=click.Path(exists=True))
@click.option(
    "--draw",
    "-d",
    "output_path",
    type=click.Path(),
    help="Save flow visualization to this path. If omitted, only prints statistics.",
)
@click.option(
    "--preset",
    type=click.Choice(["fast", "default", "high_quality"]),
    default="default",
    help="Optical flow quality preset. Default: default.",
)
@click.option(
    "--reconstruction",
    "-r",
    "reconstruction_path",
    type=click.Path(exists=True),
    help="Compare flow against correspondences from this .sfmr reconstruction.",
)
@click.option(
    "--max-features",
    "max_features",
    type=click.IntRange(min=1),
    help="Maximum number of features to visualize. Default: all.",
)
@click.option(
    "--tolerance",
    "tolerance",
    type=click.FloatRange(min=0.1),
    default=3.0,
    help="Pixel tolerance for matching advected positions to keypoints. Default: 3.0.",
)
@click.option(
    "--descriptor-threshold",
    "descriptor_threshold",
    type=click.FloatRange(min=0.0),
    default=100.0,
    help="L2 descriptor distance threshold for filtering good matches. Default: 100.0.",
)
@click.option(
    "--feature-size",
    "feature_size",
    type=click.IntRange(min=1),
    default=4,
    help="Size of feature point markers in pixels. Default: 4.",
)
@click.option(
    "--line-thickness",
    "line_thickness",
    type=click.IntRange(min=1),
    default=1,
    help="Thickness of lines in pixels. Default: 1.",
)
@click.option(
    "--side-by-side/--separate",
    "side_by_side",
    default=False,
    help="Output a single side-by-side image or two separate images (_A and _B).",
)
@click.option(
    "--pairs-dir",
    "pairs_dir",
    type=click.Path(),
    help="Process all adjacent image pairs from the reconstruction and save to this directory.",
)
def flow(
    image1,
    image2,
    output_path,
    preset,
    reconstruction_path,
    max_features,
    tolerance,
    descriptor_threshold,
    feature_size,
    line_thickness,
    side_by_side,
    pairs_dir,
):
    """Visualize optical flow between two images.

    Computes dense optical flow from IMAGE1 to IMAGE2 using the Rust DIS
    algorithm, then visualizes the flow field overlaid with SIFT keypoint
    advection.

    \b
    Without --reconstruction:
        Shows flow-colored arrows from SIFT keypoints in IMAGE1 to their
        advected positions in IMAGE2. Keypoints that land near an IMAGE2
        keypoint are highlighted with connecting lines.

    \b
    With --reconstruction:
        Compares flow advection against reconstruction correspondences:
        - GREEN: flow agrees with sfmr (advected position near matched keypoint)
        - RED: sfmr correspondence that flow does not explain
        - YELLOW: flow hit that is not an sfmr correspondence

    \b
    Print statistics only (no image output):
        sfm flow img1.jpg img2.jpg

    \b
    Draw visualization:
        sfm flow img1.jpg img2.jpg --draw output.png

    \b
    Compare with reconstruction:
        sfm flow img1.jpg img2.jpg --draw output.png -r reconstruction.sfmr

    \b
    Side-by-side output:
        sfm flow img1.jpg img2.jpg --draw output.png --side-by-side

    \b
    High quality flow:
        sfm flow img1.jpg img2.jpg --draw output.png --preset high_quality
    """
    image1_path = Path(image1)
    image2_path = Path(image2)

    # Load reconstruction if provided
    recon = None
    if reconstruction_path is not None:
        reconstruction_path = Path(reconstruction_path)
        if reconstruction_path.suffix.lower() != ".sfmr":
            raise click.UsageError(
                f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
            )
        from .._sfmtool import SfmrReconstruction

        recon = SfmrReconstruction.load(reconstruction_path)

    click.echo(f"Image 1: {image1_path}")
    click.echo(f"Image 2: {image2_path}")
    if recon is not None:
        click.echo(
            f"Reconstruction: {reconstruction_path} ({recon.image_count} images, {recon.point_count} points)"
        )

    try:
        draw_flow_visualization(
            image1_path=image1_path,
            image2_path=image2_path,
            output_path=output_path,
            preset=preset,
            feature_size=feature_size,
            line_thickness=line_thickness,
            max_features=max_features,
            side_by_side=side_by_side,
            recon=recon,
            advection_tolerance=tolerance,
            descriptor_threshold=descriptor_threshold,
        )
    except Exception as e:
        raise click.ClickException(str(e))
