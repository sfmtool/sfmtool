# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Print per-camera Z (depth) ranges from stored depth statistics."""

from pathlib import Path

import click
import numpy as np

from .._sfmtool.reconstruction import SfmrReconstruction
from .._histogram_utils import (
    create_histogram,
    print_histogram,
    render_histogram_string,
)
from .._image_pair_graph import _has_valid_depth_statistics

# Matches sfmtool_core::analysis::infinity::DEFAULT_INVERSE_DEPTH_Z_CUTOFF: below this a
# point's depth is statistically indistinguishable from infinity.
DEPTH_RELIABILITY_Z_CUTOFF = 4.0


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


def print_depth_reliability(
    recon: SfmrReconstruction,
    recon_name: str | None = None,
    noise_px: float = 1.0,
):
    """Print per-point triangulation observability diagnostics.

    For each finite, >=2-view point this reports the inverse-depth z-score
    (depth / sigma_depth) — the scale-free reliability signal, where low values
    mean the depth is statistically indistinguishable from infinity — and the
    normal-matrix condition number, the cheap geometric proxy that scales with
    track length. Points at infinity and sub-2-view points have no finite depth
    and are excluded.
    """
    if recon_name is None:
        recon_name = recon.source_metadata.get("source_path", "reconstruction")
        if "/" in recon_name or "\\" in recon_name:
            recon_name = Path(recon_name).name

    diag = recon.triangulation_diagnostics(noise_px=noise_px)
    z = np.asarray(diag["inverse_depth_z"])
    cond = np.asarray(diag["condition_number"])

    click.echo(f"\nDepth reliability for: {recon_name}")
    click.echo("=" * 70)
    click.echo(
        f"3D points: {recon.point_count}  ({recon.infinity_point_count} at infinity)"
    )
    click.echo(f"Noise floor: {noise_px:g} px")

    # The condition number is purely geometric (camera centers + ray directions)
    # and so will look healthy even when the camera intrinsics are degenerate;
    # the inverse-depth z column is the only one that reflects bad focals
    # (via sigma_rad = noise / focal_max, which goes to inf for focal=0). Flag
    # the case loudly so users do not read a normal-looking condition-number
    # histogram as a clean bill of health.
    bad_cameras = sum(
        1
        for cam in recon.cameras
        if not all(np.isfinite(f) and f > 0.0 for f in cam.focal_lengths)
    )
    if bad_cameras:
        click.echo(
            click.style(
                f"\nWARNING: {bad_cameras} of {recon.camera_count} camera(s) have "
                "non-positive or non-finite focal length. The inverse-depth z "
                "column collapses to ~0 for their points; the condition number "
                "is a purely geometric proxy and does not reflect the broken "
                "intrinsics.",
                fg="yellow",
                bold=True,
            )
        )

    finite = np.isfinite(z)
    n = int(finite.sum())
    if n == 0:
        click.echo("\nNo finite, >=2-view points to diagnose.")
        return
    zf = z[finite]

    below = int((zf < DEPTH_RELIABILITY_Z_CUTOFF).sum())
    click.echo(f"\nDiagnosed points: {n:,}")
    click.echo("\nInverse-depth z (depth/sigma; low => near-infinity):")
    click.echo(f"  Median: {np.median(zf):.2f}")
    click.echo(f"  Mean:   {zf.mean():.2f}")
    click.echo(f"  Min:    {zf.min():.2f}    Max: {zf.max():.2f}")
    click.echo(
        f"  Below z={DEPTH_RELIABILITY_Z_CUTOFF:g} (near-infinity): "
        f"{below:,} ({100.0 * below / n:.1f}%)"
    )
    hi = max(float(np.percentile(zf, 99)), DEPTH_RELIABILITY_Z_CUTOFF)
    print_histogram(zf, "Inverse-depth z", min_val=0.0, max_val=hi, show_stats=False)

    cond_f = cond[np.isfinite(cond)]
    if cond_f.size:
        click.echo("\nCondition number of the normal matrix:")
        click.echo(f"  Median: {np.median(cond_f):.1f}")
        click.echo(f"  Max:    {cond_f.max():.1f}")
        # Log-scale histogram: the condition number spans orders of magnitude.
        log_cond = np.log10(np.clip(cond_f, 1.0, None))
        counts, edges = create_histogram(log_cond, num_buckets=64)
        if counts.size:
            print_histogram(
                log_cond,
                "log10(condition number)",
                min_val=float(edges[0]),
                max_val=float(edges[-1]),
                show_stats=False,
            )

    click.echo("")
