# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Heatmap visualization command for reconstruction quality metrics."""

import re
from pathlib import Path

import click
import numpy as np

from .._cli_utils import timed_command
from .._sift_file import SiftReader, get_sift_path_for_image
from ..visualization._heatmap_renderer import (
    compute_triangulation_angles,
    render_heatmap_overlay,
)


def _insert_metric_before_number(stem: str, metric: str) -> str:
    """Insert metric name before trailing number in filename stem.

    Examples:
        seoul_bull_sculpture_07, reproj -> seoul_bull_sculpture_reproj_07
        image_001, angle -> image_angle_001
        frame12, tracks -> frame_tracks_12
        noNumber, reproj -> noNumber_reproj
    """
    # Match trailing digits (with optional underscore/dash separator)
    match = re.search(r"([_-]?)(\d+)$", stem)
    if match:
        separator, number = match.groups()
        prefix = stem[: match.start()]
        # Use underscore as separator if prefix doesn't end with one
        if prefix and not prefix.endswith(("_", "-")):
            prefix += "_"
        return f"{prefix}{metric}{separator}{number}"
    else:
        # No trailing number, just append
        return f"{stem}_{metric}"


@click.command("heatmap")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Output directory for heatmap images.",
)
@click.option(
    "--metric",
    "metric",
    type=click.Choice(["reproj", "tracks", "angle", "all"]),
    default="all",
    help="Metric to visualize: reproj (reprojection error), tracks (track length), "
    "angle (triangulation angle), or all (default).",
)
@click.option(
    "--colormap",
    "colormap",
    type=click.Choice(["viridis", "plasma", "jet", "coolwarm", "error", "tracks"]),
    default=None,
    help="Colormap to use. Default depends on metric.",
)
@click.option(
    "--radius",
    "radius",
    type=int,
    default=5,
    help="Radius of feature circles in pixels (default: 5).",
)
@click.option(
    "--alpha",
    "alpha",
    type=float,
    default=0.7,
    help="Opacity of overlay (0.0-1.0, default: 0.7).",
)
def heatmap(
    reconstruction_path,
    output_dir,
    metric,
    colormap,
    radius,
    alpha,
):
    """Visualize reconstruction quality metrics as heatmaps on images.

    Renders colored overlays showing per-feature quality metrics:

    \b
    - reproj: Reprojection error (pixels) - how well 3D points project back
    - tracks: Track length - number of images observing each point
    - angle: Triangulation angle (degrees) - baseline quality

    Each metric uses an appropriate colormap by default:
    - reproj: error (green=good, red=bad)
    - tracks: tracks (blue=few, red=many)
    - angle: viridis (purple=small, yellow=large)

    Output images are saved with metric suffix (e.g., image_001_reproj.png).

    Examples:

        # Generate all heatmaps for a reconstruction
        sfm heatmap result.sfmr -o heatmaps/

        # Generate only reprojection error heatmaps
        sfm heatmap result.sfmr -o heatmaps/ --metric reproj

        # Use custom colormap and larger markers
        sfm heatmap result.sfmr -o heatmaps/ --colormap jet --radius 8
    """
    reconstruction_path = Path(reconstruction_path)
    output_dir = Path(output_dir)

    # Validate .sfmr extension
    if reconstruction_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
        )

    # Determine which metrics to generate
    if metric == "all":
        metrics = ["reproj", "tracks", "angle"]
    else:
        metrics = [metric]

    # Default colormaps for each metric
    default_colormaps = {
        "reproj": "error",
        "tracks": "tracks",
        "angle": "viridis",
    }

    try:
        # Load reconstruction
        from .._sfmtool import SfmrReconstruction

        click.echo(f"Loading reconstruction: {reconstruction_path}")
        recon = SfmrReconstruction.load(reconstruction_path)

        click.echo(f"  Images: {recon.image_count}")
        click.echo(f"  3D points: {recon.point_count}")
        click.echo(f"  Observations: {recon.observation_count}")

        # Pre-compute triangulation angles if needed
        if "angle" in metrics:
            click.echo("Computing triangulation angles...")
            tri_angles = compute_triangulation_angles(
                recon.positions,
                recon.quaternions_wxyz,
                recon.translations,
                recon.track_image_indexes,
                recon.track_point_ids,
            )
        else:
            tri_angles = None

        # Build index from (image_idx, feature_idx) -> point3d_idx
        click.echo("Building feature-to-point mapping...")

        # Create mapping from image_idx to list of (feature_idx, point3d_idx)
        image_features: dict[int, list[tuple[int, int]]] = {}
        for obs_idx in range(len(recon.track_image_indexes)):
            img_idx = recon.track_image_indexes[obs_idx]
            feat_idx = recon.track_feature_indexes[obs_idx]
            point_idx = recon.track_point_ids[obs_idx]

            if img_idx not in image_features:
                image_features[img_idx] = []
            image_features[img_idx].append((feat_idx, point_idx))

        # Process each image
        output_dir.mkdir(parents=True, exist_ok=True)
        total_images = len(image_features)
        processed = 0

        for img_idx in sorted(image_features.keys()):
            image_name = recon.image_names[img_idx]
            image_path = Path(recon.workspace_dir) / image_name

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Get feature positions from SIFT file
            sift_path = get_sift_path_for_image(image_path)
            if not sift_path.exists():
                raise FileNotFoundError(f"SIFT file not found: {sift_path}")

            with SiftReader(sift_path) as reader:
                all_positions = reader.read_positions()

            # Get tracked features for this image
            features = image_features[img_idx]
            feat_indices = np.array([f[0] for f in features], dtype=np.int32)
            point_indices = np.array([f[1] for f in features], dtype=np.int32)

            # Get positions of tracked features only
            tracked_positions = all_positions[feat_indices]

            # Generate heatmap for each metric
            for m in metrics:
                # Get metric values for tracked points
                if m == "reproj":
                    values = recon.errors[point_indices]
                    metric_label = "Reproj Error (px)"
                elif m == "tracks":
                    values = recon.observation_counts[point_indices].astype(np.float64)
                    metric_label = "Track Length"
                elif m == "angle":
                    values = tri_angles[point_indices]
                    metric_label = "Tri Angle (deg)"
                else:
                    continue

                # Determine colormap
                cmap = colormap if colormap else default_colormaps[m]

                # Generate output filename with metric before number
                stem = Path(image_name).stem
                output_stem = _insert_metric_before_number(stem, m)
                output_path = output_dir / f"{output_stem}.png"

                # Render heatmap
                render_heatmap_overlay(
                    image_path,
                    tracked_positions,
                    values,
                    output_path,
                    metric_name=metric_label,
                    colormap=cmap,
                    radius=radius,
                    alpha=alpha,
                )

            processed += 1
            if processed % 10 == 0 or processed == total_images:
                click.echo(f"  Processed {processed}/{total_images} images")

        click.echo(f"\nHeatmaps saved to: {output_dir}")

    except Exception as e:
        raise click.ClickException(str(e))
