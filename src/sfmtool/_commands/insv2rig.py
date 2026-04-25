# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract dual-fisheye frames from Insta360 .insv video files."""

import math
from pathlib import Path

import click

from .._cli_utils import timed_command
from .._insv2rig import extract_insv_frames
from .._pano2rig import write_rig_config
from .._sfmtool import RotQuaternion
from .._workspace import find_workspace_for_path

# Insta360 X5 rig geometry (calibrated from SfM, validated against physical measurement).
# The two fisheye lenses are back-to-back, with the right eye rotated 180 degrees
# around Y relative to the left eye. The optical center baseline is ~29mm along
# the optical axis (-Z in the left/rig camera frame).
#
# These are sensor_from_rig transforms in COLMAP's Y-down convention.
# Left eye (ref sensor): identity rotation, zero translation.
# Right eye: 180 deg around Y, translated [0, 0, -0.0307] m in rig frame.
_X5_BASELINE_M = 0.0307
_X5_RIGHT_ROTATION = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], math.radians(180))
# cam_from_rig_translation = -R * center_in_rig
# center_in_rig = [0, 0, -0.0307], R = Ry(180) = diag(-1,1,-1)
# t = -R * [0,0,-0.0307] = -[0, 0, 0.0307] = [0, 0, -0.0307]
_X5_RIGHT_TRANSLATION = [0.0, 0.0, -_X5_BASELINE_M]

# Calibrated OPENCV_FISHEYE intrinsics for the Insta360 X5 at 3840x3840.
# These were refined via SfM bundle adjustment on a single X5 unit and need
# further validation across different units to confirm consistency.
_X5_CAMERA_INTRINSICS = {
    "model": "OPENCV_FISHEYE",
    "width": 3840,
    "height": 3840,
    "parameters": {
        "focal_length_x": 1031.741638,
        "focal_length_y": 1029.728817,
        "principal_point_x": 1920.0,
        "principal_point_y": 1920.0,
        "radial_distortion_k1": 0.042219,
        "radial_distortion_k2": -0.011493,
        "radial_distortion_k3": 0.010094,
        "radial_distortion_k4": -0.003034,
    },
}


@click.command("insv2rig")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Output directory for fisheye images (inside a workspace).",
)
def insv2rig(input_file, output_dir):
    """Extract dual-fisheye frames from an Insta360 .insv video file.

    Reads an Insta360 .insv video file and extracts every frame as a pair
    of fisheye images (left and right), suitable for rig-aware SfM.

    The output directory must be inside an initialized workspace (via 'sfm ws init').

    Example usage:

        sfm ws init my_workspace/
        sfm insv2rig video.insv -o my_workspace/images/seq1

    """
    input_file = Path(input_file)
    output_dir = Path(output_dir).resolve()

    try:
        if find_workspace_for_path(output_dir) is None:
            raise RuntimeError(
                f"No workspace found at or above {output_dir}. "
                f"Initialize one with 'sfm ws init'."
            )

        click.echo(f"Input: {input_file}")
        click.echo(f"Output: {output_dir}")
        num_frames, sensor_names = extract_insv_frames(input_file, output_dir)

        click.echo(f"Extracted {num_frames} frames into {len(sensor_names)} sensors")

        # Build rig geometry: left eye is ref sensor (identity), right eye
        # is rotated 180 deg with calibrated baseline.
        rotations = [RotQuaternion.identity(), _X5_RIGHT_ROTATION]
        translations = [[0.0, 0.0, 0.0], _X5_RIGHT_TRANSLATION]

        write_rig_config(
            output_dir,
            sensor_names,
            rotations=rotations,
            translations=translations,
            camera_intrinsics=_X5_CAMERA_INTRINSICS,
        )

        click.echo(f"Output: {output_dir}")
        click.echo(f"  Sensor directories: {output_dir}/<name>/")
        click.echo()
        click.echo("Next steps:")
        click.echo(f"  sfm sift --extract {output_dir}")
        click.echo(f"  sfm solve -i {output_dir}   # or -g for global SfM")

    except Exception as e:
        raise click.ClickException(str(e))
