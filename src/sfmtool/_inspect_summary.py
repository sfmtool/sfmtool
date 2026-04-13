# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Print a summary of reconstruction contents (metadata, cameras, statistics)."""

import textwrap
from pathlib import Path

import click
import numpy as np

from ._sfmtool import KdTree3d, SfmrReconstruction
from ._histogram_utils import print_histogram


def print_reconstruction_summary(
    recon: SfmrReconstruction, recon_name: str | None = None
):
    """Print a summary of the reconstruction file."""
    metadata = recon.source_metadata

    if recon_name is None:
        recon_name = metadata.get("source_path", "reconstruction")
        if "/" in recon_name or "\\" in recon_name:
            recon_name = Path(recon_name).name

    click.echo(f"\nReconstruction file: {recon_name}")
    click.echo("=" * 70)

    # Metadata
    click.echo("\nMetadata:")
    click.echo(f"  Operation: {metadata.get('operation', 'unknown')}")
    click.echo(f"  Tool: {metadata.get('tool', 'unknown')}")
    click.echo(f"  Tool version: {metadata.get('tool_version', 'unknown')}")
    click.echo(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")

    # Workspace
    workspace_info = metadata.get("workspace", {})
    click.echo("\nWorkspace:")
    click.echo(f"  Absolute path: {workspace_info.get('absolute_path', 'unknown')}")
    relative_path = workspace_info.get("relative_path", "unknown")
    click.echo(f"  Relative path: {relative_path}")
    click.echo(f"  Resolved workspace: {recon.workspace_dir}")
    click.echo(f"  Feature tool: {workspace_info.get('feature_tool', 'unknown')}")

    # Counts
    click.echo("\nReconstruction summary:")
    click.echo(f"  Images: {recon.image_count}")

    # Image path summarization (optional dependency)
    try:
        from deadline.job_attachments.api import summarize_path_list

        click.echo("  Image paths:")
        click.echo(
            textwrap.indent(summarize_path_list(recon.image_names).rstrip(), "    ")
        )
    except ImportError:
        pass

    click.echo(f"  Cameras: {recon.camera_count}")
    click.echo(f"  3D points: {recon.point_count}")
    click.echo(f"  Observations: {recon.observation_count}")

    if recon.point_count > 0:
        avg_obs = recon.observation_count / recon.point_count
        click.echo(f"  Avg observations per point: {avg_obs:.2f}")

    # Camera information
    from ._cameras import _CAMERA_PARAM_NAMES

    click.echo("\nCameras:")
    for idx, cam in enumerate(recon.cameras):
        click.echo(f"  Camera {idx}: {cam.model} {cam.width}x{cam.height}")
        params = cam.parameters
        if params:
            canonical_order = _CAMERA_PARAM_NAMES.get(cam.model)
            if canonical_order is not None:
                keys = list(canonical_order)
                extra = sorted(set(params.keys()) - set(keys))
                keys.extend(extra)
            else:
                keys = list(params.keys())

            name_width = max(len(k) for k in keys)
            click.echo(f"    {'Parameter':<{name_width}}  {'Value':>14}")
            click.echo(f"    {'-' * name_width}  {'-' * 14}")
            for key in keys:
                val = params.get(key)
                if val is not None:
                    click.echo(f"    {key:<{name_width}}  {val:>14.6f}")

    # Rig information
    rfd = recon.rig_frame_data
    if rfd is not None:
        rigs_meta = rfd["rigs_metadata"]
        click.echo("\nRig configuration:")
        click.echo(f"  Rigs: {rigs_meta['rig_count']}")
        click.echo(f"  Total sensors: {rigs_meta['sensor_count']}")
        click.echo(f"  Frames: {rfd['frames_metadata']['frame_count']}")
        for rig_def in rigs_meta.get("rigs", []):
            click.echo(f"  Rig '{rig_def['name']}':")
            click.echo(f"    Sensors: {rig_def['sensor_count']}")
            click.echo(f"    Reference sensor: {rig_def['ref_sensor_name']}")
            sensor_names = rig_def.get("sensor_names", [])
            offset = rig_def.get("sensor_offset", 0)
            for i, sensor_name in enumerate(sensor_names):
                cam_idx = int(rfd["sensor_camera_indexes"][offset + i])
                is_ref = sensor_name == rig_def["ref_sensor_name"]
                ref_marker = " (ref)" if is_ref else ""
                if not is_ref:
                    t = rfd["sensor_translations_xyz"][offset + i]
                    click.echo(
                        f"    {sensor_name}{ref_marker}: camera {cam_idx}, "
                        f"translation=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})"
                    )
                else:
                    click.echo(f"    {sensor_name}{ref_marker}: camera {cam_idx}")
    else:
        click.echo("\nRig configuration: none")

    # 3D point statistics
    if recon.point_count > 0:
        positions = recon.positions
        errors = recon.errors

        click.echo("\n3D Point statistics:")
        click.echo("  Position range:")
        click.echo(f"    X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
        print_histogram(positions[:, 0], "X distribution", show_stats=False)
        click.echo(f"    Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        print_histogram(positions[:, 1], "Y distribution", show_stats=False)
        click.echo(f"    Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        print_histogram(positions[:, 2], "Z distribution", show_stats=False)

        click.echo("  Reprojection error:")
        click.echo(f"    Mean: {errors.mean():.3f} pixels")
        click.echo(f"    Median: {np.median(errors):.3f} pixels")
        click.echo(f"    Min: {errors.min():.3f} pixels")
        click.echo(f"    Max: {errors.max():.3f} pixels")
        print_histogram(errors, "Error distribution", show_stats=False)

    # Observation statistics
    if recon.point_count > 0:
        observation_counts = recon.observation_counts

        click.echo("\nObservation statistics:")
        click.echo(f"  Min observations per point: {observation_counts.min()}")
        click.echo(f"  Max observations per point: {observation_counts.max()}")
        click.echo(
            f"  Median observations per point: {int(np.median(observation_counts))}"
        )
        print_histogram(
            observation_counts, "Track length distribution", show_stats=False
        )

    # Nearest neighbor distances
    if recon.point_count > 1:
        positions = recon.positions
        nn_distances = KdTree3d(positions).nearest_neighbor_distances()

        click.echo("\nNearest neighbor distances:")
        click.echo(f"  Min: {nn_distances.min():.6f}")
        click.echo(f"  Max: {nn_distances.max():.6f}")
        click.echo(f"  Mean: {nn_distances.mean():.6f}")
        click.echo(f"  Median: {np.median(nn_distances):.6f}")
        print_histogram(nn_distances, "NN distance distribution", show_stats=False)

    click.echo("")
