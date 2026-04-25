# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera-related commands (`sfm cam ...`)."""

import json
from pathlib import Path

import click

from .._cli_utils import timed_command


@click.group("cam")
@click.help_option("--help", "-h")
def cam():
    """Camera-related operations on reconstructions and workspaces."""


@cam.command("cp")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_sfmr", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_json", type=click.Path(dir_okay=False))
@click.option(
    "--index",
    "camera_index",
    type=click.IntRange(min=0),
    default=None,
    help="Index of the camera to copy (required when the reconstruction has "
    "more than one camera). Indexes match the reconstruction's camera order.",
)
def cp(input_sfmr: str, output_json: str, camera_index: int | None):
    """Copy a camera's intrinsics from a `.sfmr` file to a `camera_config.json`.

    Reads the chosen camera's intrinsics from INPUT_SFMR and writes them to
    OUTPUT_JSON in the `camera_config.json` schema (`version: 1`). If the
    reconstruction has exactly one camera, `--index` may be omitted.

    Example usage:

    \b
        # Single-camera reconstruction
        sfm cam cp sfmr/solve_001.sfmr camera_config.json

        # Multi-camera reconstruction
        sfm cam cp sfmr/multi.sfmr --index 1 phone/camera_config.json
    """
    from .._cameras import _CAMERA_PARAM_NAMES
    from .._sfmtool import SfmrReconstruction

    input_path = Path(input_sfmr)
    output_path = Path(output_json)

    if input_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Input must be a .sfmr file: {input_path}")

    try:
        recon = SfmrReconstruction.load(input_path)
    except Exception as e:
        raise click.ClickException(f"Failed to load {input_path}: {e}")

    cameras = recon.cameras
    camera_count = len(cameras)
    if camera_count == 0:
        raise click.ClickException(
            f"{input_path} contains no cameras — nothing to copy."
        )

    if camera_index is None:
        if camera_count > 1:
            raise click.UsageError(
                f"{input_path} has {camera_count} cameras; pass --index N "
                f"to choose one (valid range: 0..{camera_count - 1})."
            )
        camera_index = 0
    elif camera_index >= camera_count:
        raise click.UsageError(
            f"--index {camera_index} is out of range; "
            f"{input_path} has {camera_count} camera(s) "
            f"(valid range: 0..{camera_count - 1})."
        )

    intrinsics_dict = cameras[camera_index].to_dict()
    model = intrinsics_dict["model"]
    if model in _CAMERA_PARAM_NAMES:
        ordered_params = {
            name: intrinsics_dict["parameters"][name]
            for name in _CAMERA_PARAM_NAMES[model]
            if name in intrinsics_dict["parameters"]
        }
        intrinsics_dict["parameters"] = ordered_params

    output = {"version": 1, "camera_intrinsics": intrinsics_dict}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    click.echo(
        f"Wrote camera {camera_index} ({model}, "
        f"{intrinsics_dict.get('width', '?')}x{intrinsics_dict.get('height', '?')}) "
        f"to {output_path}"
    )
