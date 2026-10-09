# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Transform reconstruction command."""

import click

from .._cli_utils import timed_command
from ._sfmr_path import check_sfmr_path
from ..xform import apply_transforms
from ..xform._arg_parser import (
    OrderedArgsCommand,
    auto_output_path,
    check_against_click,
    command_args,
    parse_xform_args,
)


@click.command("xform", cls=OrderedArgsCommand)
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
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Apply bundle adjustment to refine camera poses and 3D points. A "
        "reconstruction with a SFMTOOL_FISHEYE or SFMTOOL_PINHOLE camera is "
        "adjusted by sfmtool with the focal and the lens distortion released, "
        "and each point stored as a position or a direction by its rays at the "
        "end; 'cameras=0+1' releases only those cameras' lenses, and "
        "'cross=off' keeps every point in the representation it has (e.g. "
        "'--bundle-adjust cameras=0,cross=off')."
    ),
)
@click.option(
    "--refine-normals",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Refine per-point surface normals by photometric cross-view consensus. "
        "Optional comma-separated key=value params (e.g. "
        "'angular_range_deg=25,init_steps=7'). Renders and persists the "
        "per-point RGBA patch textures by default so the output is "
        "self-contained; pass 'bitmaps=false' to skip the render (e.g. on an "
        "intermediate stage). A point that stores a reference observation "
        "keeps it and is rendered from it. Requires an embedded_patches reconstruction "
        "(convert first with --to-embedded-patches); reads the workspace source "
        "images, which must still be present where it was created."
    ),
)
@click.option(
    "--refine-keypoints",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Refine per-observation 2D keypoints to sub-pixel by photometric "
        "cross-view alignment (never worse than the seed; the track structure "
        "is unchanged). Optional comma-separated key=value params (e.g. "
        "'max_outer_sweeps=2,sampler=anisotropic'). Renders and persists the "
        "per-point RGBA patch textures at the refined keypoints by default so "
        "the output is self-contained; pass 'bitmaps=false' to skip the render "
        "(e.g. on an intermediate stage). A point that stores a reference "
        "observation keeps it and is rendered from it. Requires an embedded_patches "
        "reconstruction (convert first with --to-embedded-patches); reads the "
        "workspace source images, which must still be present where it was "
        "created."
    ),
)
@click.option(
    "--localize-keypoints",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Localize per-observation 2D keypoints by discrete cross-view search "
        "(congealing). Structural, not in-place: views that don't co-register "
        "are dropped, points falling below min_views are culled, and the track "
        "structure is rebuilt from the survivors; stored patch bitmaps are "
        "dropped (re-run --refine-keypoints to regenerate them, since it "
        "renders bitmaps by default), and each point keeps its reference "
        "observation where its track still holds that image. "
        "Optional comma-separated key=value params (e.g. "
        "'search=8,min_views=3'). Requires an embedded_patches reconstruction "
        "(convert first with --to-embedded-patches); reads the workspace "
        "source images, which must still be present where it was created."
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
        "'normal=mean_viewing,extent=feature_size,extent_value=5'). Reads the "
        ".sift files, which must still be present where the reconstruction was "
        "created. After this op the reconstruction is embedded_patches."
    ),
)
@click.option(
    "--drop-thumbnails",
    is_flag=True,
    multiple=True,
    help=(
        "Discard the per-image thumbnail column, keeping every row. Reads no "
        "files. --add-thumbnails builds it back from the .sift files or the "
        "photographs."
    ),
)
@click.option(
    "--drop-patch-bitmaps",
    is_flag=True,
    multiple=True,
    help=(
        "Discard the per-point patch bitmap column, keeping the patch frames, "
        "normals and reference observations, so a later step can render the "
        "same bitmaps onto them."
    ),
)
@click.option(
    "--add-thumbnails",
    is_flag=True,
    multiple=True,
    help=(
        "Build the thumbnail column from each image's verified .sift copy, which "
        "is already reduced, and otherwise from its source photograph, decoded and "
        "resized as the SIFT extractors do. An embedded_patches file checks each "
        "photograph it reads against its recorded image hash; an image neither "
        "source can supply fails the step. A no-op when thumbnails are present."
    ),
)
@click.option(
    "--add-patch-bitmaps",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Render the patch bitmap column at the stored frames and keypoints, "
        "moving nothing: each point from its stored reference observation, and "
        "a point with none from the observation the reference-view rule picks, "
        "which is recorded; where the rule picks no view, or reaches its pick "
        "only through its last fallback, the bitmap is the views' fused mean and the point "
        "records -1. Optional 'resolution=<R>,sampler=<S>' (defaults 24 and "
        "per_view). Requires an embedded_patches reconstruction; reads the "
        "workspace source images. A no-op when bitmaps are present."
    ),
)
@click.option(
    "--minimal",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Write the smallest file that still holds the whole reconstruction: "
        "--drop-patch-bitmaps --drop-thumbnails at this position, and a save "
        "with an empty absolute workspace path and tool_options "
        "holding only this invocation's transforms. Optional "
        "'wspath=<path>' records that as the relative workspace path instead of "
        "measuring one, e.g. wspath=. for an output written inside its workspace."
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
    "--filter-by-zncc-self-similarity-radius",
    multiple=True,
    help=(
        "Remove points whose stored patch bitmap can slide over itself further "
        "than this and still match itself, in patch-grid px from 0 to 3 (e.g., "
        "'2.5'): the ZNCC self-similarity radius, which reads under 1 for a "
        "corner or texture and 3 for an edge or a flat patch. 0 turns it off. "
        "Needs a reconstruction with patch bitmaps."
    ),
)
@click.option(
    "--filter-by-patch-size",
    multiple=True,
    help=(
        "Remove points whose world-space patch size exceeds a multiple of the "
        "median (e.g. '3.0'). Size is the geometric mean of a patch's two world "
        "half-extents; the threshold is data-derived per reconstruction. Needs "
        "an embedded_patches reconstruction."
    ),
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
        "Fit a camera of another model to each camera over the angles where it "
        "is trusted: MODEL[,coeffs=N,fit_to=DEG,spline_domain=DEG,cameras=0+1] "
        "(e.g., 'SFMTOOL_FISHEYE,coeffs=8' or 'RADIAL'). Prints a per-camera report."
    ),
)
@click.option(
    "--find-points-at-infinity",
    multiple=True,
    help=(
        "Discover points at infinity: eps_deg[,desc_thresh[,min_views[,sigma_px]]] "
        "(e.g. '0.1,200,2'). Each candidate track is decided with the "
        "point-or-bearing test; the optional sigma_px is the per-axis pixel "
        "noise to weight the rays by, measured from the reconstruction by default."
    ),
)
@click.option(
    "--classify-points-at-infinity",
    is_flag=False,
    flag_value="",
    multiple=True,
    help=(
        "Decide every point with the point-or-bearing test: demote a finite "
        "point whose rays ask for no depth to a point at infinity, and promote "
        "a point at infinity whose rays ask for one to the fitted point. The "
        "optional value is the per-axis pixel noise to weight the rays by "
        "(e.g. '0.5'); by default it is measured from the reconstruction."
    ),
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
      --filter-by-zncc-self-similarity-radius val  Remove points whose patch bitmap slides over itself > val (patch-grid px)
      --filter-by-patch-size MULT         Remove points with world-space patch size > MULT x median
      --include-by-distribution COUNT[,verbose]  Keep COUNT well-distributed cameras/rig frames

    \b
    Points at infinity:
      --find-points-at-infinity SPEC      Discover points at infinity: eps_deg[,desc_thresh[,min_views[,sigma_px]]]
      --classify-points-at-infinity [SIGMA_PX]  Store each point as the point-or-bearing test decides
      --max-features N                    Cap features per image for --find-points-at-infinity (largest first)

    \b
    Camera model:
      --camera-model MODEL[,KEY=VAL...]   Fit another model to each camera (e.g. SFMTOOL_FISHEYE,coeffs=8)

    \b
    Optimization:
      --bundle-adjust [cameras=0+1,cross=off]  Apply bundle adjustment
      --refine-normals [PARAMS]           Refine per-point normals by photometric consensus (reads source images)
      --refine-keypoints [PARAMS]         Refine per-observation keypoints to sub-pixel (reads source images)
      --localize-keypoints [PARAMS]       Cross-view keypoint search; drops non-registering views (reads source images)

    \b
    Representation:
      --to-embedded-patches [PARAMS]      Convert sift_files → embedded_patches (no photometric adaptation; reads .sift)

    \b
    Heavy columns:
      --drop-thumbnails                   Discard the per-image thumbnails
      --drop-patch-bitmaps                Discard the per-point patch bitmaps (frames kept)
      --add-thumbnails                    Build thumbnails from the .sift files, else the photographs
      --add-patch-bitmaps [PARAMS]        Render patch bitmaps at the stored frames (reads source images)
      --minimal [PARAMS]                  Drop both, and save minimal metadata (for a file that travels); wspath=<path> states the recorded workspace path

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
        # Move a fisheye to the spline model, then refine its focal and spline
        sfm xform in.sfmr out.sfmr --camera-model SFMTOOL_FISHEYE,coeffs=8 --bundle-adjust

    \b
        # Discover points at infinity, capping features per image
        sfm xform in.sfmr out.sfmr --find-points-at-infinity 0.1,200,2 --max-features 2000

    \b
        # Bundle-adjust, then refine surface normals against the final geometry
        sfm xform in.sfmr out.sfmr --bundle-adjust --refine-normals angular_range_deg=25,init_steps=7

    \b
        # Refine keypoints to sub-pixel and refine normals in one pass
        sfm xform in.sfmr out.sfmr --refine-keypoints --refine-normals

    \b
        # Search keypoints into the photometric basin (drops non-registering
        # views), then sharpen the survivors to sub-pixel
        sfm xform in.sfmr out.sfmr --localize-keypoints --refine-keypoints

    \b
        # The smallest file for a repository, and the same with thumbnails kept
        sfm xform in.sfmr out.sfmr --minimal
        sfm xform in.sfmr out.sfmr --minimal --add-thumbnails

    \b
        # A ground truth checked in inside its own workspace, so the file records
        # the workspace it sits in rather than the path it was written from
        sfm xform in.sfmr ws/ground_truth.sfmr --minimal wspath=.

    \b
        # Re-render the patch bitmaps at a different resolution
        sfm xform in.sfmr out.sfmr --drop-patch-bitmaps --add-patch-bitmaps resolution=32
    """
    from .._sfmtool.reconstruction import SfmrReconstruction

    raw_input_path, raw_output_path = input_path, output_path
    input_path = check_sfmr_path(input_path, "Input path")
    output_path_provided = output_path is not None

    if output_path_provided:
        output_path = check_sfmr_path(output_path, "Output path")
    else:
        output_path = auto_output_path(input_path)

    # Walk the command's own arguments again to keep the transforms in order,
    # then check that walk against what Click collected.
    try:
        parsed = parse_xform_args(
            command_args(ctx), max_features=kwargs.get("max_features")
        )
    except ValueError as e:
        raise click.UsageError(str(e))
    check_against_click(parsed, kwargs, [raw_input_path, raw_output_path])
    transforms = parsed.transforms

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
            "--filter-by-zncc-self-similarity-radius, --filter-by-patch-size, "
            "--include-by-distribution, "
            "--find-points-at-infinity, --classify-points-at-infinity, "
            "--camera-model, --bundle-adjust, --refine-normals, --refine-keypoints, "
            "--localize-keypoints, --to-embedded-patches, "
            "--drop-thumbnails, --drop-patch-bitmaps, --add-thumbnails, "
            "--add-patch-bitmaps, --minimal, "
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
        # --minimal anywhere in the chain marks the save: the save rewrites the
        # metadata after every step has run, so clearing it at the step's own
        # position would be undone.
        from ..xform import MinimalTransform

        minimal_steps = [t for t in transforms if isinstance(t, MinimalTransform)]
        minimal = bool(minimal_steps)
        # A stated workspace path belongs to the save too, so the last --minimal
        # that names one is the one the output records.
        workspace_path = next(
            (
                t.workspace_path
                for t in reversed(minimal_steps)
                if t.workspace_path is not None
            ),
            None,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"\nWriting transformed reconstruction to: {output_path}")
        recon.save(
            str(output_path),
            operation="xform",
            tool_options={"transforms": transform_descriptions},
            minimal=minimal,
            workspace_path=workspace_path,
        )

        click.echo("\nTransformed reconstruction saved to:")
        click.echo(f"  {output_path}")

    except click.UsageError:
        # A step that finds the chain misapplied to this reconstruction (e.g.
        # ``--bundle-adjust coeffs=`` with no spline camera) says so as a usage
        # error rather than as a failure of the step.
        raise
    except Exception as e:
        raise click.ClickException(str(e))
