# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge reconstructions command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._merge import merge_reconstructions
from .._sfmtool import SfmrReconstruction


@click.command("merge")
@timed_command
@click.help_option("--help", "-h")
@click.argument(
    "reconstruction_paths", nargs=-1, type=click.Path(exists=True), required=True
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    help="Output path for merged .sfmr file.",
    required=True,
)
@click.option(
    "--merge-percentile",
    type=float,
    default=95.0,
    help="Percentile of correspondence distances to use as threshold (default: 95.0).",
)
def merge(reconstruction_paths, output_path, merge_percentile):
    """Merge multiple aligned .sfmr files.

    This command merges multiple .sfmr files that are already in the same
    coordinate frame (e.g., from 'sfm align') into a single unified reconstruction.

    The merge process:
    - Deduplicates cameras with identical parameters
    - Merges images (uses first occurrence's pose for overlapping images)
    - Finds corresponding 3D points across reconstructions using feature matching
    - Filters correspondences using percentile-based thresholding (scale-invariant)
    - Merges corresponding points (averages positions, combines observations)
    - Refines camera poses using PnP + RANSAC against the merged point cloud
    - Combines all tracks into a unified reconstruction

    RECONSTRUCTION_PATHS must be .sfmr files that are already aligned.

    Example:

        sfm merge aligned/ref.sfmr aligned/recon1.sfmr -o merged.sfmr
    """
    if len(reconstruction_paths) < 2:
        raise click.UsageError("Need at least 2 reconstructions to merge")

    input_paths = []
    for p in reconstruction_paths:
        path = Path(p)
        if path.suffix.lower() != ".sfmr":
            raise click.UsageError(f"Reconstruction path must be a .sfmr file: {p}")
        input_paths.append(path)

    output_path_obj = Path(output_path)
    if output_path_obj.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Output path must be a .sfmr file: {output_path}")

    try:
        click.echo("\nLoading reconstructions...")
        reconstructions = []
        for path in input_paths:
            click.echo(f"  Loading {path.name}...")
            recon = SfmrReconstruction.load(str(path))
            reconstructions.append(recon)
            click.echo(f"    Images: {recon.image_count}, Points: {recon.point_count}")

        result = merge_reconstructions(
            reconstructions=reconstructions,
            merge_percentile=merge_percentile,
        )

        click.echo(f"\nSaving merged reconstruction to {output_path_obj.name}...")
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        result.save(
            str(output_path_obj),
            operation="merge",
            tool_options={"merge_percentile": merge_percentile},
        )
        click.echo(f"Saved to: {output_path_obj}")

    except Exception as e:
        raise click.ClickException(str(e))
