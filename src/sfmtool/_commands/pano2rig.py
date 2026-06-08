# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert equirectangular panoramas to perspective rig images."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from ..rig.pano2rig import (
    _cubemap_rotations,
    convert_panoramas,
    write_pano_camrig,
)
from .._workspace import find_workspace_for_path


@click.command("pano2rig")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Output directory for face images (inside a workspace).",
)
@click.option(
    "--face-size",
    type=int,
    default=None,
    help="Face image size in pixels (default: pano_width / 4).",
)
@click.option(
    "--jpeg-quality",
    type=int,
    default=95,
    help="JPEG quality for output face images (1-100, default: 95).",
)
def pano2rig(input_dir, output_dir, face_size, jpeg_quality):
    """Convert equirectangular panoramas to perspective face images for rig-aware SfM.

    Reads equirectangular (360) panorama images from INPUT_DIR and generates
    perspective face images under the OUTPUT directory, then writes a .camrig
    file describing the six-face cubemap rig.

    Generates a standard 6-face cubemap (front, right, back, left,
    top, bottom) with 90-degree FOV. Face images are written to
    <output>/front/, <output>/right/, etc.

    The output directory must be inside an initialized workspace (via 'sfm ws init').

    Example usage:

        sfm ws init my_workspace/
        sfm pano2rig panoramas/ -o my_workspace/images

        sfm ws init my_workspace/
        sfm pano2rig panoramas/ -o my_workspace/images --face-size 1024

    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir).resolve()

    try:
        if find_workspace_for_path(output_dir) is None:
            raise RuntimeError(
                f"No workspace found at or above {output_dir}. "
                f"Initialize one with 'sfm ws init'."
            )

        click.echo(f"Input: {input_dir}")
        click.echo(f"Output: {output_dir}")
        num_panos, actual_face_size, face_names = convert_panoramas(
            input_dir,
            output_dir,
            face_size=face_size,
            jpeg_quality=jpeg_quality,
        )

        click.echo(
            f"Extracted {len(face_names)} faces from {num_panos} panoramas "
            f"(face size: {actual_face_size}x{actual_face_size})"
        )

        camrig_path = output_dir / "cubemap.camrig"
        write_pano_camrig(
            camrig_path,
            rig_name="cubemap",
            face_names=face_names,
            rotations=_cubemap_rotations(),
            face_size=actual_face_size,
        )

        click.echo(f"Output: {output_dir}")
        click.echo(f"  Face directories: {output_dir}/<name>/")
        click.echo(f"  Camera rig: {camrig_path}")
        click.echo()
        click.echo("Next steps:")
        click.echo(f"  sfm sift --extract {output_dir}")
        click.echo(f"  sfm solve -i {output_dir}   # or -g for global SfM")

    except Exception as e:
        raise click.ClickException(str(e))
