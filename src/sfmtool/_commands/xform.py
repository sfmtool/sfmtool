# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Transform reconstruction command."""

import sys
from pathlib import Path

import click

from .._cli_utils import timed_command
from ..xform import apply_transforms
from ..xform._arg_parser import auto_output_path, parse_transform_args


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
    "--refine-normals",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Refine per-point surface normals by photometric cross-view consensus. "
        "Optional comma-separated key=value params (e.g. "
        "'angular_range_deg=25,init_steps=7'; 'bitmaps=true' also renders the "
        "per-point patch textures). Requires an embedded_patches reconstruction "
        "(convert first with --to-embedded-patches); reads the workspace source "
        "images, which must still be present where it was created."
    ),
)
@click.option(
    "--to-embedded-patches",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Convert sift_files → embedded_patches without photometric adaptation: "
        "mean-view uv frames, keypoints + image hashes copied from the .sift files. "
        "Optional comma-separated key=value params (e.g. "
        "'normal=mean_viewing,extent=feature_size,extent_value=10'). Reads the "
        ".sift files, which must still be present where the reconstruction was "
        "created. After this op the reconstruction is embedded_patches."
    ),
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
    "--include-by-distribution",
    multiple=True,
    help="Keep COUNT strategically distributed cameras/rig frames; append ',verbose' for a per-step trace (e.g., '16' or '16,verbose')",
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
@click.option(
    "--find-points-at-infinity",
    multiple=True,
    help="Discover points at infinity: eps_deg[,desc_thresh[,min_views[,noise_floor_px]]] (e.g. '0.1,200,2,1.0')",
)
@click.option(
    "--classify-points-at-infinity",
    multiple=True,
    help="Reclassify finite points whose depth is unconstrained as points at infinity: noise_floor_px (e.g. '1.0')",
)
@click.option(
    "--max-features",
    type=click.IntRange(min=1),
    default=None,
    help="Cap features per image for --find-points-at-infinity (largest first)",
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
      --include-by-distribution COUNT[,verbose]  Keep COUNT well-distributed cameras/rig frames

    \b
    Points at infinity:
      --find-points-at-infinity SPEC      Discover points at infinity: eps_deg[,desc_thresh[,min_views[,noise_floor_px]]]
      --classify-points-at-infinity NOISE Reclassify unconstrained finite points as points at infinity
      --max-features N                    Cap features per image for --find-points-at-infinity (largest first)

    \b
    Camera model:
      --camera-model NAME                 Switch every camera's model (e.g. RADIAL)

    \b
    Optimization:
      --bundle-adjust                     Apply bundle adjustment
      --refine-normals [PARAMS]           Refine per-point normals by photometric consensus (reads source images)

    \b
    Representation:
      --to-embedded-patches [PARAMS]      Convert sift_files → embedded_patches (no photometric adaptation; reads .sift)

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

    \b
        # Discover points at infinity, capping features per image
        sfm xform in.sfmr out.sfmr --find-points-at-infinity 0.1,200,2 --max-features 2000

    \b
        # Bundle-adjust, then refine surface normals against the final geometry
        sfm xform in.sfmr out.sfmr --bundle-adjust --refine-normals angular_range_deg=25,init_steps=7
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
        output_path = auto_output_path(input_path)

    # Parse transforms from sys.argv to preserve order
    try:
        xform_idx = sys.argv.index("xform")
        # Skip 'xform', input_path, and output_path (if it was supplied).
        transform_args_start = xform_idx + (3 if output_path_provided else 2)
        transform_args = sys.argv[transform_args_start:]
    except (ValueError, IndexError):
        transform_args = []

    try:
        transforms = parse_transform_args(
            transform_args, max_features=kwargs.get("max_features")
        )
    except ValueError as e:
        raise click.UsageError(str(e))

    # --max-features only feeds --find-points-at-infinity; reject it when that
    # operation isn't in the chain so it isn't silently ignored.
    from ..xform import FindPointsAtInfinityTransform

    if ctx.get_parameter_source(
        "max_features"
    ) == click.ParameterSource.COMMANDLINE and not any(
        isinstance(t, FindPointsAtInfinityTransform) for t in transforms
    ):
        raise click.UsageError(
            "--max-features only applies to --find-points-at-infinity, "
            "which was not requested."
        )

    if not transforms:
        raise click.UsageError(
            "At least one transformation must be specified. "
            "Options: --rotate, --translate, --scale, --scale-by-measurements, "
            "--include-range, --exclude-range, "
            "--include-glob, --exclude-glob, --remove-short-tracks, --remove-narrow-tracks, "
            "--remove-large-features, --remove-isolated, --filter-by-reprojection-error, "
            "--include-by-distribution, "
            "--find-points-at-infinity, --classify-points-at-infinity, "
            "--camera-model, --bundle-adjust, --refine-normals, --to-embedded-patches, "
            "--align-to, --align-to-input"
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
