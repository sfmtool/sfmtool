# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Print per-camera Z (depth) ranges from stored depth statistics."""

from pathlib import Path

import click

from ._sfmtool import SfmrReconstruction
from ._histogram_utils import render_histogram_string
from ._image_pair_graph import _has_valid_depth_statistics


def print_z_range(recon: SfmrReconstruction, recon_name: str | None = None):
    """Print per-image Z depth ranges and histograms from stored depth statistics."""
    if recon_name is None:
        recon_name = recon.source_metadata.get("source_path", "reconstruction")
        if "/" in recon_name or "\\" in recon_name:
            recon_name = Path(recon_name).name

    depth_stats = recon.depth_statistics
    if not _has_valid_depth_statistics(depth_stats):
        raise click.ClickException(
            f"No depth statistics found in {recon_name}.\n"
            "Depth statistics are computed during reconstruction and stored in "
            "the .sfmr file."
        )

    observed_counts = recon.depth_histogram_counts
    num_buckets = depth_stats["num_histogram_buckets"]

    click.echo(f"\nZ range statistics for: {recon_name}")
    click.echo("=" * 70)
    click.echo(f"Images: {recon.image_count}")
    click.echo(f"3D points: {recon.point_count}")
    click.echo(f"Histogram buckets: {num_buckets}")
    click.echo("")

    images_data = depth_stats["images"]
    valid_ranges = [img for img in images_data if img["histogram_min_z"] is not None]

    if not valid_ranges:
        click.echo("No images have depth statistics.")
        return

    all_min = [img["histogram_min_z"] for img in valid_ranges]
    all_max = [img["histogram_max_z"] for img in valid_ranges]
    all_observed = [img["observed"]["count"] for img in valid_ranges]

    click.echo("Summary statistics:")
    click.echo(f"  Images with data: {len(valid_ranges)}/{recon.image_count}")
    click.echo(f"  Min Z range: [{min(all_min):.3f}, {max(all_min):.3f}]")
    click.echo(f"  Max Z range: [{min(all_max):.3f}, {max(all_max):.3f}]")
    click.echo(
        f"  Observed points per image: [{min(all_observed)}, {max(all_observed)}]"
    )
    click.echo("")

    def hist_to_str(hist_counts):
        hist_counts = hist_counts[0::2] + hist_counts[1::2]
        return render_histogram_string(hist_counts)

    click.echo("Per-image Z ranges:")
    for img_idx, img_data in enumerate(images_data):
        name = recon.image_names[img_idx]
        if img_data["histogram_min_z"] is not None:
            min_z = img_data["histogram_min_z"]
            max_z = img_data["histogram_max_z"]
            obs = img_data["observed"]

            click.echo(f"  {name}:")
            click.echo(f"    Z range: [{min_z:.3f}, {max_z:.3f}]")

            if obs["count"] > 0:
                obs_hist = hist_to_str(observed_counts[img_idx])
                click.echo(
                    f"    Observed:  n={obs['count']:4d}, "
                    f"med={obs['median_z']:.3f}, mean={obs['mean_z']:.3f}"
                )
                click.echo(f"               [{obs_hist}]")
            else:
                click.echo("    Observed:  (none)")
        else:
            click.echo(f"  {name}: no data")

    click.echo("")
