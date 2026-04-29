# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Transform reconstruction command."""

import re
import sys
from pathlib import Path

import click
import numpy as np
from openjd.model import IntRangeExpr

from .._cli_utils import timed_command
from ..xform import (
    AlignToInputTransform,
    AlignToTransform,
    BundleAdjustTransform,
    ExcludeGlobFilter,
    ExcludeRangeFilter,
    FilterByReprojectionErrorTransform,
    IncludeGlobFilter,
    IncludeRangeFilter,
    RemoveIsolatedPointsFilter,
    RemoveLargeFeaturesFilter,
    RemoveNarrowTracksFilter,
    RemoveShortTracksFilter,
    RotateTransform,
    ScaleByMeasurementsTransform,
    ScaleTransform,
    SwitchCameraModelTransform,
    TranslateTransform,
    apply_transforms,
)


def parse_angle(angle_str: str) -> float:
    """Parse angle string with unit suffix.

    Args:
        angle_str: Angle string like "90deg" or "1.57rad"

    Returns:
        Angle in radians
    """
    match = re.match(r"^([+-]?[\d.]+)(deg|degrees|rad|radians)$", angle_str.strip())
    if not match:
        raise ValueError(
            f"Invalid angle format: '{angle_str}'. Expected format: <number><unit> "
            f"where unit is 'deg', 'degrees', 'rad', or 'radians'"
        )

    value_str, unit = match.groups()
    value = float(value_str)

    if unit in ("deg", "degrees"):
        return np.radians(value)
    elif unit in ("rad", "radians"):
        return value
    else:
        raise ValueError(f"Unrecognized angle unit: {unit}")


def _auto_output_path(input_path: Path) -> Path:
    """Generate an output path of the form {stem}-transformed[-N].sfmr next to the input.

    Picks ``{stem}-transformed.sfmr`` if available, otherwise the smallest
    counter starting at 2: ``{stem}-transformed-2.sfmr``, ``-3.sfmr``, ...
    """
    base = input_path.with_name(f"{input_path.stem}-transformed.sfmr")
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = input_path.with_name(
            f"{input_path.stem}-transformed-{counter}.sfmr"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def parse_transform_args(args: list[str]) -> list:
    """Parse command-line arguments to extract transforms in order."""
    transforms = []
    i = 0

    while i < len(args):
        arg = args[i]

        if arg == "--rotate":
            if i + 1 >= len(args):
                raise click.UsageError("--rotate requires an argument")
            i += 1
            param = args[i]

            parts = param.split(",")
            if len(parts) != 4:
                raise click.UsageError(
                    f"--rotate expects 4 comma-separated values (axisX,axisY,axisZ,angle), got: {param}"
                )

            try:
                axis_x = float(parts[0])
                axis_y = float(parts[1])
                axis_z = float(parts[2])
                angle_rad = parse_angle(parts[3])
            except ValueError as e:
                raise click.UsageError(f"Invalid --rotate parameter '{param}': {e}")

            axis = np.array([axis_x, axis_y, axis_z])
            transforms.append(RotateTransform(axis, angle_rad))

        elif arg == "--translate":
            if i + 1 >= len(args):
                raise click.UsageError("--translate requires an argument")
            i += 1
            param = args[i]

            parts = param.split(",")
            if len(parts) != 3:
                raise click.UsageError(
                    f"--translate expects 3 comma-separated values (X,Y,Z), got: {param}"
                )

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except ValueError as e:
                raise click.UsageError(f"Invalid --translate parameter '{param}': {e}")

            translation = np.array([x, y, z])
            transforms.append(TranslateTransform(translation))

        elif arg == "--scale":
            if i + 1 >= len(args):
                raise click.UsageError("--scale requires an argument")
            i += 1
            param = args[i]

            try:
                scale_factor = float(param)
            except ValueError as e:
                raise click.UsageError(f"Invalid --scale parameter '{param}': {e}")

            transforms.append(ScaleTransform(scale_factor))

        elif arg == "--remove-short-tracks":
            if i + 1 >= len(args):
                raise click.UsageError("--remove-short-tracks requires an argument")
            i += 1
            param = args[i]

            try:
                max_size = int(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --remove-short-tracks parameter '{param}': {e}"
                )

            transforms.append(RemoveShortTracksFilter(max_size))

        elif arg == "--bundle-adjust":
            transforms.append(BundleAdjustTransform())

        elif arg == "--remove-narrow-tracks":
            if i + 1 >= len(args):
                raise click.UsageError("--remove-narrow-tracks requires an argument")
            i += 1
            param = args[i]

            try:
                min_angle_rad = parse_angle(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --remove-narrow-tracks parameter '{param}': {e}"
                )

            transforms.append(RemoveNarrowTracksFilter(min_angle_rad))

        elif arg == "--remove-isolated":
            if i + 1 >= len(args):
                raise click.UsageError("--remove-isolated requires an argument")
            i += 1
            param = args[i]

            parts = param.split(",")
            if len(parts) != 2:
                raise click.UsageError(
                    f"--remove-isolated expects 2 comma-separated values (factor,value_spec), got: {param}"
                )

            try:
                factor = float(parts[0])
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid factor in --remove-isolated '{param}': {e}"
                )

            value_spec = parts[1]
            transforms.append(RemoveIsolatedPointsFilter(factor, value_spec))

        elif arg == "--align-to":
            if i + 1 >= len(args):
                raise click.UsageError("--align-to requires an argument")
            i += 1
            param = args[i]

            reference_path = Path(param)
            transforms.append(AlignToTransform(reference_path))

        elif arg == "--align-to-input":
            transforms.append(AlignToInputTransform())

        elif arg == "--remove-large-features":
            if i + 1 >= len(args):
                raise click.UsageError("--remove-large-features requires an argument")
            i += 1
            param = args[i]

            try:
                max_size = float(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --remove-large-features parameter '{param}': {e}"
                )

            transforms.append(RemoveLargeFeaturesFilter(max_size))

        elif arg == "--filter-by-reprojection-error":
            if i + 1 >= len(args):
                raise click.UsageError(
                    "--filter-by-reprojection-error requires an argument"
                )
            i += 1
            param = args[i]

            try:
                threshold = float(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --filter-by-reprojection-error parameter '{param}': {e}"
                )

            transforms.append(FilterByReprojectionErrorTransform(threshold))

        elif arg == "--include-range":
            if i + 1 >= len(args):
                raise click.UsageError("--include-range requires an argument")
            i += 1
            param = args[i]

            try:
                range_expr = IntRangeExpr.from_str(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --include-range parameter '{param}': {e}"
                )

            transforms.append(IncludeRangeFilter(range_expr))

        elif arg == "--exclude-range":
            if i + 1 >= len(args):
                raise click.UsageError("--exclude-range requires an argument")
            i += 1
            param = args[i]

            try:
                range_expr = IntRangeExpr.from_str(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --exclude-range parameter '{param}': {e}"
                )

            transforms.append(ExcludeRangeFilter(range_expr))

        elif arg == "--scale-by-measurements":
            if i + 1 >= len(args):
                raise click.UsageError("--scale-by-measurements requires an argument")
            i += 1
            param = args[i]

            measurements_path = Path(param)
            if not measurements_path.exists():
                raise click.UsageError(
                    f"Measurements file not found: {measurements_path}"
                )

            transforms.append(ScaleByMeasurementsTransform(measurements_path))

        elif arg == "--include-glob":
            if i + 1 >= len(args):
                raise click.UsageError("--include-glob requires an argument")
            i += 1
            transforms.append(IncludeGlobFilter(args[i]))

        elif arg == "--exclude-glob":
            if i + 1 >= len(args):
                raise click.UsageError("--exclude-glob requires an argument")
            i += 1
            transforms.append(ExcludeGlobFilter(args[i]))

        elif arg == "--camera-model":
            if i + 1 >= len(args):
                raise click.UsageError("--camera-model requires an argument")
            i += 1
            param = args[i]

            try:
                transforms.append(SwitchCameraModelTransform(param))
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --camera-model parameter '{param}': {e}"
                )

        i += 1

    return transforms


@click.command("xform")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False)
@click.option(
    "--rotate",
    multiple=True,
    help="Rotate around axis: axisX,axisY,axisZ,angle (e.g., '0,1,0,90deg')",
)
@click.option(
    "--translate",
    multiple=True,
    help="Translate by vector: X,Y,Z (e.g., '3,5,-2')",
)
@click.option(
    "--scale",
    multiple=True,
    help="Scale by factor: S (e.g., '2.0')",
)
@click.option(
    "--remove-short-tracks",
    multiple=True,
    help="Remove points with track length <= size (e.g., '2')",
)
@click.option(
    "--bundle-adjust",
    is_flag=True,
    multiple=True,
    help="Apply bundle adjustment to refine camera poses and 3D points",
)
@click.option(
    "--remove-narrow-tracks",
    multiple=True,
    help="Remove points with viewing angle < threshold (e.g., '5deg')",
)
@click.option(
    "--remove-large-features",
    multiple=True,
    help="Remove points where max SIFT feature size > threshold in pixels (e.g., '50')",
)
@click.option(
    "--remove-isolated",
    multiple=True,
    help="Remove isolated points (factor,value_spec) (e.g., '3.0,median')",
)
@click.option(
    "--align-to",
    multiple=True,
    help="Align to another reconstruction (path to .sfmr file)",
)
@click.option(
    "--align-to-input",
    is_flag=True,
    multiple=True,
    help="Align back to original input reconstruction",
)
@click.option(
    "--filter-by-reprojection-error",
    multiple=True,
    help="Remove points with reprojection error > threshold (e.g., '2.0')",
)
@click.option(
    "--scale-by-measurements",
    multiple=True,
    help="Scale to physical units using a YAML measurements file with known point-pair distances",
)
@click.option(
    "--include-range",
    multiple=True,
    help="Keep only images with file numbers in range (e.g., '1-10', '1,3,5-7')",
)
@click.option(
    "--exclude-range",
    multiple=True,
    help="Exclude images with file numbers in range (e.g., '1-10', '1,3,5-7')",
)
@click.option(
    "--include-glob",
    multiple=True,
    help="Keep only images whose name matches a glob pattern (e.g., '*fisheye_left*')",
)
@click.option(
    "--exclude-glob",
    multiple=True,
    help="Exclude images whose name matches a glob pattern (e.g., '*fisheye_right*')",
)
@click.option(
    "--camera-model",
    multiple=True,
    help=(
        "Switch every camera to a different COLMAP model "
        "(e.g., 'RADIAL' to add a k2 term for bundle adjustment to refine). "
        "Shared parameters carry over; new ones initialize to zero."
    ),
)
@click.pass_context
def xform(ctx, input_path, output_path, **kwargs):
    """Apply transformations to a .sfmr file.

    This command applies a sequence of transformations and filters to a
    reconstruction in a single pass. Transformations are applied in the
    order they appear on the command line.

    INPUT_PATH must be a .sfmr file.
    OUTPUT_PATH is the path for the output .sfmr file. If omitted, the
    output is written next to the input as ``{stem}-transformed.sfmr``,
    falling back to ``{stem}-transformed-2.sfmr`` (then ``-3``, ...) when
    that name is taken.

    Available transformations:

    \b
    Geometric Transformations:
      --rotate axisX,axisY,axisZ,angle    Rotate around axis
      --translate X,Y,Z                   Translate by vector
      --scale S                           Scale by factor
      --scale-by-measurements FILE        Scale to physical units using measurements YAML

    \b
    Filters:
      --include-range RANGE               Keep only images with file numbers in range
      --exclude-range RANGE               Exclude images with file numbers in range
      --include-glob PATTERN              Keep only images matching glob pattern
      --exclude-glob PATTERN              Exclude images matching glob pattern
      --remove-short-tracks size          Remove points with track length <= size
      --remove-narrow-tracks angle        Remove points with viewing angle < threshold
      --remove-large-features size        Remove points with max feature size > threshold
      --remove-isolated factor,spec       Remove isolated points (NN distance filter)
      --filter-by-reprojection-error val  Remove points with reprojection error > threshold

    \b
    Camera model:
      --camera-model NAME                 Switch every camera's model (e.g. RADIAL)

    \b
    Optimization:
      --bundle-adjust                     Apply bundle adjustment

    \b
    Alignment:
      --align-to path.sfmr                Align to another reconstruction
      --align-to-input                    Align back to original input

    Examples:

    \b
        # Rotate 90 degrees around Y axis
        sfm xform in.sfmr out.sfmr --rotate 0,1,0,90deg

    \b
        # Translate then rotate (order matters!)
        sfm xform in.sfmr out.sfmr --translate 3,5,-2 --rotate 0,1,0,90deg

    \b
        # Filter short tracks, then scale
        sfm xform in.sfmr out.sfmr --remove-short-tracks 2 --scale 0.5

    \b
        # Multiple operations in sequence
        sfm xform in.sfmr out.sfmr \\
            --remove-short-tracks 2 \\
            --rotate 1,0,0,90deg \\
            --translate 0,0,-5 \\
            --scale 0.01

    \b
        # Filter and optimize with bundle adjustment
        sfm xform in.sfmr out.sfmr --remove-short-tracks 2 --bundle-adjust

    \b
        # Upgrade SIMPLE_RADIAL → RADIAL to refine k2 during bundle adjustment
        sfm xform in.sfmr out.sfmr --camera-model RADIAL --bundle-adjust
    """
    from .._sfmtool import SfmrReconstruction

    input_path = Path(input_path)
    output_path_provided = output_path is not None

    if input_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Input path must be a .sfmr file, got: {input_path}")

    if output_path_provided:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".sfmr":
            raise click.UsageError(
                f"Output path must be a .sfmr file, got: {output_path}"
            )
    else:
        output_path = _auto_output_path(input_path)

    # Parse transforms from sys.argv to preserve order
    try:
        xform_idx = sys.argv.index("xform")
        # Skip 'xform', input_path, and output_path (if it was supplied).
        transform_args_start = xform_idx + (3 if output_path_provided else 2)
        transform_args = sys.argv[transform_args_start:]
    except (ValueError, IndexError):
        transform_args = []

    try:
        transforms = parse_transform_args(transform_args)
    except ValueError as e:
        raise click.UsageError(str(e))

    if not transforms:
        raise click.UsageError(
            "At least one transformation must be specified. "
            "Options: --rotate, --translate, --scale, --scale-by-measurements, "
            "--include-range, --exclude-range, "
            "--include-glob, --exclude-glob, --remove-short-tracks, --remove-narrow-tracks, "
            "--remove-large-features, --remove-isolated, --filter-by-reprojection-error, "
            "--camera-model, --bundle-adjust, --align-to, --align-to-input"
        )

    try:
        click.echo(f"Loading reconstruction from: {input_path}")
        recon = SfmrReconstruction.load(input_path)
        click.echo(f"  Images: {recon.image_count}")
        click.echo(f"  Points: {recon.point_count}")
        click.echo(f"  Cameras: {recon.camera_count}")

        recon = apply_transforms(
            recon=recon,
            transforms=transforms,
        )

        transform_descriptions = [t.description() for t in transforms]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"\nWriting transformed reconstruction to: {output_path}")
        recon.save(
            str(output_path),
            operation="xform",
            tool_options={"transforms": transform_descriptions},
        )

        click.echo("\nTransformed reconstruction saved to:")
        click.echo(f"  {output_path}")

    except Exception as e:
        raise click.ClickException(str(e))
