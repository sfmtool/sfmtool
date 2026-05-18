# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and print covisibility and frustum intersection graphs."""

from pathlib import Path

import click
import numpy as np

from ._sfmtool import SfmrReconstruction
from ._histogram_utils import estimate_z_from_histogram
from ._image_pair_graph import (
    _has_valid_depth_statistics,
    build_covisibility_pairs,
    build_frustum_intersection_pairs,
)


def print_covisibility_graph(recon: SfmrReconstruction, recon_name: str | None = None):
    """Build and print the covisibility graph of the reconstruction."""
    if recon_name is None:
        recon_name = recon.source_metadata.get("source_path", "reconstruction")
        if "/" in recon_name or "\\" in recon_name:
            recon_name = Path(recon_name).name

    pairs = build_covisibility_pairs(recon, angle_threshold_deg=90.0)

    click.echo(f"\nCovisibility graph for: {recon_name}")
    click.echo("=" * 70)
    click.echo(f"Images: {recon.image_count}")
    click.echo(f"3D points: {recon.point_count}")
    click.echo("")

    click.echo("Image pairs with shared 3D points:")
    click.echo("(showing count of shared points for each pair)")
    click.echo("")

    if not pairs:
        click.echo("No image pairs share 3D points (after angle culling).")
    else:
        total_possible = recon.image_count * (recon.image_count - 1) // 2
        click.echo(
            f"Total image pairs: {len(pairs)} out of {total_possible} total possible"
        )
        click.echo("")

        counts = [count for _, _, count in pairs]
        click.echo("Statistics:")
        click.echo(f"  Max shared points: {max(counts)}")
        click.echo(f"  Min shared points: {min(counts)}")
        click.echo(f"  Mean shared points: {np.mean(counts):.1f}")
        click.echo(f"  Median shared points: {int(np.median(counts))}")
        click.echo("")

        click.echo("Image pairs (sorted by shared point count):")
        for img_i, img_j, count in pairs:
            name_i = recon.image_names[img_i]
            name_j = recon.image_names[img_j]
            click.echo(f"  {name_i} <-> {name_j}: {count} points")

    click.echo("")


def print_frustum_intersection_graph(
    recon: SfmrReconstruction,
    near_percentile: float = 5.0,
    far_percentile: float = 95.0,
    num_samples: int = 10000,
    recon_name: str | None = None,
):
    """Build and print the frustum intersection graph of the reconstruction."""
    if recon_name is None:
        recon_name = recon.source_metadata.get("source_path", "reconstruction")
        if "/" in recon_name or "\\" in recon_name:
            recon_name = Path(recon_name).name

    depth_stats = recon.depth_statistics
    if not _has_valid_depth_statistics(depth_stats):
        raise click.ClickException(
            f"No depth statistics found in {recon_name}.\n"
            "Depth statistics are required for frustum intersection analysis.\n"
            "Re-run the reconstruction to generate depth statistics."
        )

    hist_counts = recon.depth_histogram_counts

    click.echo(f"\nFrustum intersection graph for: {recon_name}")
    click.echo("=" * 70)
    click.echo(f"Images: {recon.image_count}")
    click.echo(f"Z range percentiles: {near_percentile}% to {far_percentile}%")
    click.echo(f"Monte Carlo samples: {num_samples}")
    click.echo("")

    images_data = depth_stats["images"]
    click.echo("Image frustum Z ranges:")
    for img_idx in range(recon.image_count):
        img_data = images_data[img_idx]
        name = recon.image_names[img_idx]

        if img_data["histogram_min_z"] is None:
            click.echo(f"  {name}: no depth data, skipping")
            continue

        min_z = img_data["histogram_min_z"]
        max_z = img_data["histogram_max_z"]

        near_z = estimate_z_from_histogram(
            hist_counts[img_idx], min_z, max_z, near_percentile
        )
        far_z = estimate_z_from_histogram(
            hist_counts[img_idx], min_z, max_z, far_percentile
        )

        if near_z >= far_z:
            near_z = min_z
            far_z = max_z

        click.echo(f"  {name}: Z=[{near_z:.3f}, {far_z:.3f}]")

    click.echo("")
    click.echo("Computing frustum intersections...")

    try:
        pairs = build_frustum_intersection_pairs(
            recon,
            near_percentile=near_percentile,
            far_percentile=far_percentile,
            num_samples=num_samples,
            angle_threshold_deg=90.0,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    if not pairs:
        click.echo("\nNo frustum intersections found.")
    else:
        volumes = [vol for _, _, vol in pairs]
        total_possible = recon.image_count * (recon.image_count - 1) // 2
        click.echo(
            f"\nTotal image pairs with intersection: {len(pairs)} out of {total_possible} total possible"
        )
        click.echo("")
        click.echo("Statistics:")
        click.echo(f"  Max intersection volume: {max(volumes):.3f} cubic units")
        click.echo(f"  Min intersection volume: {min(volumes):.3f} cubic units")
        click.echo(f"  Mean intersection volume: {np.mean(volumes):.3f} cubic units")
        click.echo(
            f"  Median intersection volume: {np.median(volumes):.3f} cubic units"
        )
        click.echo("")

        click.echo("Image pairs (sorted by intersection volume):")
        for img_i, img_j, volume in pairs:
            name_i = recon.image_names[img_i]
            name_j = recon.image_names[img_j]
            click.echo(f"  {name_i} <-> {name_j}: {volume:.3f} cu")

    click.echo("")
