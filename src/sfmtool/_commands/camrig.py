# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera rig file commands (`sfm camrig ...`)."""

import math
from pathlib import Path

import click

from .._cli_utils import timed_command


@click.group("camrig")
@click.help_option("--help", "-h")
def camrig():
    """Build and inspect .camrig camera rig files."""


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


@camrig.command("inspect")
@click.help_option("--help", "-h")
@click.argument("camrig_file", type=click.Path(exists=True, dir_okay=False))
def inspect(camrig_file):
    """Verify a .camrig file and print its metadata.

    Recomputes the file's content hashes, checks its structural
    constraints, and prints the rig metadata.

    Example usage:

        sfm camrig inspect tiles.camrig

    """
    from .._sfmtool import read_camrig_metadata, verify_camrig

    camrig_file = Path(camrig_file)
    try:
        info = read_camrig_metadata(str(camrig_file))
        valid, errors = verify_camrig(str(camrig_file))
    except Exception as e:
        raise click.ClickException(str(e))

    meta = info["metadata"]
    content_hash = info["content_hash"]
    click.echo(f"File:         {camrig_file}")
    click.echo(f"Format ver:   {meta['version']}")
    click.echo(f"Name:         {meta['name']}")
    click.echo(f"Rig type:     {meta['rig_type']}")
    click.echo(f"Sensors:      {meta['sensor_count']}")
    click.echo(f"Cameras:      {meta['camera_count']}")

    attrs = meta.get("rig_attributes")
    if isinstance(attrs, dict) and attrs:
        click.echo("Attributes:")
        for key, value in attrs.items():
            click.echo(f"  {key}: {value}")

    click.echo(f"Content hash: {content_hash['content_xxh128']}")
    if valid:
        click.echo("Integrity:    OK")
    else:
        click.echo("Integrity:    FAILED")
        for err in errors:
            click.echo(f"  {err}")
        raise click.ClickException("camrig verification failed")
