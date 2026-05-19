# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera rig file commands (`sfm camrig ...`)."""

import math
from pathlib import Path

import click

from .._cameras import _CAMERA_PARAM_NAMES
from .._cli_utils import timed_command


@click.group("camrig")
@click.help_option("--help", "-h")
def camrig():
    """Build .camrig camera rig files.

    Use `sfm inspect <FILE.camrig>` to inspect a rig file.
    """


# COLMAP camera-model names accepted by `--camera-model`, taken from the
# canonical parameter-name table so the two cannot drift apart.
_CAMERA_MODELS = tuple(_CAMERA_PARAM_NAMES)


@camrig.command("create")
@timed_command
@click.help_option("--help", "-h")
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.argument("image_pattern")
@click.option(
    "--camera-model",
    type=click.Choice(_CAMERA_MODELS, case_sensitive=False),
    default=None,
    help="COLMAP camera model. Overrides the EXIF-inferred model; "
    "required when --params is used.",
)
@click.option(
    "--resolution",
    default=None,
    help="Image resolution as WIDTHxHEIGHT (e.g. 4000x3000). Every matched "
    "image must have this resolution.",
)
@click.option(
    "--focal-length",
    type=float,
    default=None,
    help="Focal length in pixels; sets both fx and fy.",
)
@click.option(
    "--focal-length-x",
    type=float,
    default=None,
    help="Focal length fx in pixels.",
)
@click.option(
    "--focal-length-y",
    type=float,
    default=None,
    help="Focal length fy in pixels.",
)
@click.option(
    "--principal-point-x",
    type=float,
    default=None,
    help="Principal point cx in pixels.",
)
@click.option(
    "--principal-point-y",
    type=float,
    default=None,
    help="Principal point cy in pixels.",
)
@click.option(
    "--params",
    default=None,
    help="Full camera parameter list in COLMAP order, comma-separated "
    "(e.g. 1400,1400,960,540). Requires --camera-model.",
)
@click.option(
    "--name",
    default=None,
    help="Rig name stored in the file (default: the output file stem).",
)
def create(
    output_file,
    image_pattern,
    camera_model,
    resolution,
    focal_length,
    focal_length_x,
    focal_length_y,
    principal_point_x,
    principal_point_y,
    params,
    name,
):
    """Create a .camrig file for a directory of images.

    Scans the images matched by IMAGE_PATTERN — a .camrig image pattern
    (globs and/or %d-style frame fields) resolved relative to the OUTPUT_FILE
    directory and stored verbatim in the rig — and writes a one-camera
    .camrig. Intrinsics come from pycolmap EXIF inference unless overridden.
    The command fails if the matched images are inconsistent (mixed
    resolutions, mixed inferred models, varying focal lengths) so the caller
    can narrow the pattern and build separate rigs.

    Example usage:

        sfm camrig create my_images.camrig 'images/*'

        sfm camrig create rig.camrig '*.jpg' --camera-model OPENCV \\
            --resolution 4000x3000 \\
            --params 2800,2800,2000,1500,-0.08,0.01,0,0

    """
    from .._camrig_create import build_camrig_from_images

    try:
        summary = build_camrig_from_images(
            Path(output_file),
            image_pattern,
            camera_model=camera_model,
            resolution=resolution,
            focal_length=focal_length,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            principal_point_x=principal_point_x,
            principal_point_y=principal_point_y,
            params=params,
            name=name,
        )
    except Exception as e:
        raise click.ClickException(str(e))

    cam = summary["camera"]
    cam_params = cam["parameters"]
    focal = cam_params.get("focal_length_x", cam_params.get("focal_length"))
    click.echo(f"Wrote {summary['output_file']}")
    click.echo(f"  images:   {summary['image_count']}")
    click.echo(f"  pattern:  {summary['pattern']}")
    click.echo(f"  camera:   {cam['model']} {cam['width']}x{cam['height']}")
    click.echo(f"  focal:    {focal:.1f} px")


@camrig.command("cp")
@timed_command
@click.help_option("--help", "-h")
@click.argument("source", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--rig",
    "rig_index",
    type=click.IntRange(min=0),
    default=None,
    help="`.sfmr` source only: index of the rig to copy.",
)
@click.option(
    "--camera",
    "camera_index",
    type=click.IntRange(min=0),
    default=None,
    help="Index into the source's camera pool; copies that one camera as a "
    "single-sensor rig.",
)
@click.option(
    "--sensors",
    "sensors_expr",
    default=None,
    help="`.camrig` source only: sensor indices to keep, as an integer range "
    "expression (e.g. 0-2, 0,2,4).",
)
@click.option(
    "--pattern",
    default=None,
    help="Image pattern for the output sensor. Only valid with --camera.",
)
@click.option(
    "--name",
    default=None,
    help="Rig name stored in the file (default: the source rig name, then the "
    "output file stem).",
)
def cp(source, output_file, rig_index, camera_index, sensors_expr, pattern, name):
    """Build a .camrig by copying from a .sfmr or .camrig file.

    SOURCE is a `.sfmr` reconstruction or a `.camrig` file; OUTPUT_FILE is the
    `.camrig` to write. A selector chooses what to copy: `--rig` (a whole rig,
    `.sfmr` only), `--camera` (one camera as a single-sensor rig), or
    `--sensors` (a subset of sensors, `.camrig` only). With no selector, a
    `.sfmr` falls back to its lone rig or lone camera and a `.camrig` is copied
    whole.

    Example usage:

    \b
        # Harvest refined intrinsics from a solved reconstruction
        sfm camrig cp sfmr/solve_001.sfmr photos.camrig --camera 0
        # Copy a solved rig into a reusable .camrig
        sfm camrig cp sfmr/rig_solve.sfmr studio.camrig --rig 0
        # Take three faces out of a six-face cubemap rig
        sfm camrig cp cubemap.camrig front.camrig --sensors 0-2
    """
    from .._camrig_cp import CamrigCpError, copy_from_camrig, copy_from_sfmr

    src = Path(source)
    out = Path(output_file)

    if camera_index is not None and rig_index is not None:
        raise click.UsageError("--rig and --camera are mutually exclusive.")
    if camera_index is not None and sensors_expr is not None:
        raise click.UsageError("--sensors and --camera are mutually exclusive.")
    if rig_index is not None and sensors_expr is not None:
        raise click.UsageError("--rig and --sensors are mutually exclusive.")
    if pattern is not None and camera_index is None:
        raise click.UsageError(
            "--pattern applies only with --camera (single-sensor output)."
        )

    suffix = src.suffix.lower()
    try:
        if suffix == ".sfmr":
            if sensors_expr is not None:
                raise click.UsageError(
                    "--sensors applies to a .camrig source; use --rig for a "
                    ".sfmr reconstruction."
                )
            summary = copy_from_sfmr(
                src,
                out,
                rig_index=rig_index,
                camera_index=camera_index,
                pattern=pattern,
                name=name,
            )
        elif suffix == ".camrig":
            if rig_index is not None:
                raise click.UsageError(
                    "--rig applies to a .sfmr source; a .camrig holds exactly "
                    "one rig — use --sensors to pick a subset of its sensors."
                )
            summary = copy_from_camrig(
                src,
                out,
                sensors_expr=sensors_expr,
                camera_index=camera_index,
                pattern=pattern,
                name=name,
            )
        else:
            raise click.UsageError(f"SOURCE must be a .sfmr or .camrig file: {src}")
    except CamrigCpError as e:
        raise click.ClickException(str(e))

    click.echo(f"Wrote {summary['output_file']}")
    click.echo(
        f"  sensors:  {summary['sensor_count']}  "
        f"cameras: {summary['camera_count']}  "
        f"rig type: {summary['rig_type']}"
    )
    if not summary["image_backed"]:
        click.echo(
            "  note:     written geometry-only (no image patterns could be "
            "inferred for every sensor)"
        )


@camrig.command("spherical-tiles")
@timed_command
@click.help_option("--help", "-h")
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--n",
    "n_tiles",
    type=int,
    required=True,
    help="Number of tiles in the rig (>= 2).",
)
@click.option(
    "--equirect-width",
    type=int,
    default=None,
    help="Target equirectangular width in pixels; sets the per-tile pixel "
    "size to 2*pi / width. Mutually exclusive with --arc-per-pixel.",
)
@click.option(
    "--arc-per-pixel",
    type=float,
    default=None,
    help="Angular size of one tile pixel, in radians. Mutually exclusive "
    "with --equirect-width.",
)
@click.option(
    "--overlap-factor",
    type=float,
    default=1.15,
    show_default=True,
    help="Tile FOV safety margin over the worst-case gap between tiles.",
)
@click.option(
    "--centre",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 0.0),
    show_default=True,
    help="Rig optical centre in world space (three floats: x y z).",
)
@click.option(
    "--atlas-cols",
    type=int,
    default=None,
    help="Atlas column count (default: ceil(sqrt(n))).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Relaxer seed for reproducible tile placement.",
)
@click.option(
    "--name",
    default=None,
    help="Rig name stored in the file (default: the output file stem).",
)
def spherical_tiles(
    output_file,
    n_tiles,
    equirect_width,
    arc_per_pixel,
    overlap_factor,
    centre,
    atlas_cols,
    seed,
    name,
):
    """Build a spherical tile rig and write it to a .camrig file.

    Discretises the sphere into N co-centric pinhole tiles and stores the
    rig as OUTPUT_FILE. The tile look directions come from a relaxed
    sphere-point set; pass --seed for a reproducible layout.

    Example usage:

        sfm camrig spherical-tiles tiles.camrig --n 1280 --equirect-width 1024

        sfm camrig spherical-tiles tiles.camrig --n 320 --arc-per-pixel 0.012 \\
            --overlap-factor 1.2 --seed 42

    """
    if (equirect_width is None) == (arc_per_pixel is None):
        raise click.UsageError(
            "Pass exactly one of --equirect-width or --arc-per-pixel."
        )
    if equirect_width is not None:
        if equirect_width < 1:
            raise click.UsageError("--equirect-width must be >= 1.")
        arc_per_pixel = 2.0 * math.pi / equirect_width

    from .._sfmtool import SphericalTileRig

    output_file = Path(output_file)
    try:
        rig = SphericalTileRig(
            n=n_tiles,
            arc_per_pixel=arc_per_pixel,
            centre=list(centre),
            overlap_factor=overlap_factor,
            atlas_cols=atlas_cols,
            seed=seed,
        )
        rig.write_camrig(str(output_file), name=name)
    except Exception as e:
        raise click.ClickException(str(e))

    atlas_w, atlas_h = rig.atlas_size
    click.echo(f"Wrote {output_file}")
    click.echo(f"  tiles:      {len(rig)}")
    click.echo(f"  patch size: {rig.patch_size} px")
    click.echo(f"  atlas:      {atlas_w}x{atlas_h} px ({rig.atlas_cols} cols)")
    click.echo(f"  tile FOV:   {math.degrees(2.0 * rig.half_fov_rad):.2f} deg")
