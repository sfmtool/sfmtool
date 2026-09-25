# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Write a reconstruction as static web files a browser draws in 3D."""

from pathlib import Path

import click

from .._cli_utils import timed_command


def _size(n: int) -> str:
    if n >= 2**20:
        return f"{n / 2**20:.1f} MB"
    if n >= 2**10:
        return f"{n / 2**10:.0f} KB"
    return f"{n} B"


@click.command("web-export")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory to write. Created if missing; refused if it exists and is "
    "not empty, unless --overwrite.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace the files this command writes in an existing directory.",
)
@click.option(
    "--no-patches",
    is_flag=True,
    help="Leave out the patch atlas; points are drawn as round splats in their "
    "stored colour.",
)
@click.option(
    "--no-thumbnails",
    is_flag=True,
    help="Leave out the thumbnail atlas; frustums are drawn as wireframes only.",
)
@click.option(
    "--patch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Resample patch tiles to this many texels a side (for example 12) to "
    "shrink the atlas. Default: the file's bitmap size.",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(1, 100),
    default=85,
    show_default=True,
    help="JPEG quality for both atlases.",
)
@click.option(
    "--max-points",
    type=click.IntRange(min=1),
    default=None,
    help="Keep the N points with the most observations; the rest are left out.",
)
@click.option(
    "--start-image",
    default=None,
    help="Open the page looking through this image's camera (its name as the "
    "reconstruction stores it) instead of framing the whole scene.",
)
@click.option(
    "--single-file",
    is_flag=True,
    help="Write one self-contained index.html with the data and atlases inline; "
    "refused if it would pass 16 MB.",
)
def web_export(
    reconstruction_path,
    output_dir,
    overwrite,
    no_patches,
    no_thumbnails,
    patch_size,
    jpeg_quality,
    max_points,
    start_image,
    single_file,
):
    """Write a reconstruction as a web page that draws it in 3D.

    The output directory holds static files: a viewer page (index.html), its
    script (web-export.js), the scene description (scene.json) and JPEG atlases
    of the patch bitmaps and the image thumbnails. Open index.html from a
    static web server, point an iframe at it, or publish the directory as the
    files of a claude.ai artifact. The page draws the points, their textured
    patches and the cameras with each image's thumbnail, and can be turned
    with a mouse or with touch on a phone. three.js is loaded from
    cdn.jsdelivr.net.

    The thumbnails of the photographs are in the output, so sharing it shares
    them. Use --no-thumbnails to leave them out.

    Examples:

    \b
        sfm web-export scene.sfmr -o site/
        sfm web-export scene.sfmr -o site/ --patch-size 12 --max-points 50000
        sfm web-export scene.sfmr -o small/ --single-file
    """
    from .._sfmtool.reconstruction import SfmrReconstruction
    from ..web_export import WebExportError, export_reconstruction

    reconstruction_path = Path(reconstruction_path)
    output_dir = Path(output_dir)
    if reconstruction_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
        )

    click.echo(f"Loading reconstruction: {reconstruction_path}")
    try:
        recon = SfmrReconstruction.load(reconstruction_path)
    except Exception as e:
        raise click.ClickException(str(e))
    click.echo(f"  Images: {recon.image_count}")
    click.echo(f"  3D points: {recon.point_count}")

    try:
        report = export_reconstruction(
            recon,
            output_dir,
            source_name=reconstruction_path.name,
            overwrite=overwrite,
            single_file=single_file,
            patches=not no_patches,
            thumbnails=not no_thumbnails,
            patch_size=patch_size,
            jpeg_quality=jpeg_quality,
            max_points=max_points,
            start_image=start_image,
        )
    except WebExportError as e:
        raise click.UsageError(str(e))

    line = f"  Points written: {report['points']}"
    if report["points_at_infinity"]:
        line += f" ({report['points_at_infinity']} at infinity)"
    if report["points_left_out"]:
        line += f", {report['points_left_out']} left out"
    click.echo(line)
    if report["patches"]:
        rendered = (
            ", bitmaps rendered from the photographs"
            if report["patch_bitmaps_rendered"]
            else ""
        )
        click.echo(
            f"  Patches: {report['patches']} at {report['patch_size']} px{rendered}"
        )
    thumbs = report["thumbnails"]
    if any(thumbs.values()):
        sources = [
            f"{n} {what}"
            for what, n in (
                ("from the file", thumbs["file"]),
                ("from .sift files", thumbs["sift"]),
                ("from photographs", thumbs["photographs"]),
                ("placeholders", thumbs["placeholders"]),
            )
            if n
        ]
        click.echo(f"  Thumbnails: {', '.join(sources)}")
    click.echo(
        f"  Atlases decode to {_size(report['decoded_atlas_bytes'])} of GPU memory"
    )
    total = 0
    click.echo(f"\nWrote to {output_dir}:")
    for name, size in report["files"]:
        total += size
        click.echo(f"  {name:<18} {_size(size):>9}")
    click.echo(f"  {'total':<18} {_size(total):>9}")
    for warning in report["warnings"]:
        click.echo(f"Warning: {warning}", err=True)
