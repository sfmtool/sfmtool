# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render oriented patches from a reconstruction onto its source images."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._feature_source import require_embedded_patches
from ..visualization._patch_renderer import MODES, PatchRenderError, render_patches


def _parse_rgb(ctx, param, value: str) -> tuple[int, int, int]:
    """Parse an ``R,G,B`` triple for --border-color."""
    try:
        parts = tuple(int(c) for c in value.split(","))
    except ValueError:
        raise click.BadParameter(f"expected three integers 'R,G,B', got {value!r}")
    if len(parts) != 3 or not all(0 <= c <= 255 for c in parts):
        raise click.BadParameter(
            f"expected three 0-255 integers 'R,G,B', got {value!r}"
        )
    return parts


@click.command("render-patches")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Output directory for the rendered overlay images.",
)
@click.option(
    "--mode",
    type=click.Choice(MODES),
    default="texture",
    help="Patch fill: texture (warp the patch bitmap), normal (shade by surface "
    "normal), flat (point colour), or wire (outline only). Default: texture.",
)
@click.option("--border/--no-border", default=False, help="Outline each patch quad.")
@click.option(
    "--border-color",
    callback=_parse_rgb,
    default="0,255,0",
    help="Border colour as 'R,G,B' (default: 0,255,0).",
)
@click.option(
    "--border-thickness", type=int, default=1, help="Border thickness in pixels."
)
@click.option(
    "--alpha",
    type=float,
    default=1.0,
    help="Global patch opacity (0.0-1.0, default: 1.0).",
)
@click.option(
    "--opaque",
    "opaque_threshold",
    is_flag=False,
    flag_value="0.1",
    default=None,
    type=float,
    help="Texture mode: paint texels whose confidence alpha exceeds this "
    "threshold (0-1) at full opacity and drop the rest, for an honest alignment "
    "check with no bleed-through. Bare --opaque uses 0.1; omit for the "
    "confidence-weighted blend.",
)
@click.option(
    "--scale",
    type=float,
    default=1.0,
    help="Shrink/grow each patch quad about its centre (e.g. 0.5). Note: scaling "
    "away from 1.0 breaks texture-to-image alignment.",
)
@click.option(
    "--upscale",
    type=float,
    default=1.0,
    help="Upscale the output canvas (and projected coords) for legibility on "
    "small images.",
)
@click.option(
    "--backface-cull/--no-backface-cull",
    default=True,
    help="Skip patches whose normal faces away from the camera (default: on).",
)
@click.option(
    "--images",
    "image_filter",
    multiple=True,
    help="Restrict to images whose name contains this substring (repeatable). "
    "Default: all registered images.",
)
def render_patches_command(
    reconstruction_path,
    output_dir,
    mode,
    border,
    border_color,
    border_thickness,
    alpha,
    opaque_threshold,
    scale,
    upscale,
    backface_cull,
    image_filter,
):
    """Render a reconstruction's oriented patches on top of its source images.

    For each registered image, every per-point oriented patch (the quad spanned
    by its in-plane half-extent vectors, centred on the point) is projected into
    the frame and composited onto the source image, for visually inspecting the
    reconstruction's geometry and patches.

    The reconstruction must be ``embedded_patches`` (it carries the per-point
    patch frames), e.g. from:

    \b
        sfm xform in.sfmr out.sfmr --to-embedded-patches
        # for --mode texture, build bitmaps too:
        sfm xform in.sfmr out.sfmr --to-embedded-patches --refine-normals bitmaps=true

    Patches are painted back-to-front (painter's algorithm); there is no true
    occlusion, so a distant patch can show through a nearer one.

    Examples:

    \b
        # Textured patches with outlines (needs bitmaps)
        sfm render-patches out.sfmr -o renders/ --border
        # Honest alignment check, no confidence bleed-through
        sfm render-patches out.sfmr -o renders/ --opaque
        # Normal field on small images, scaled up for legibility
        sfm render-patches out.sfmr -o renders/ --mode normal --upscale 3
    """
    reconstruction_path = Path(reconstruction_path)
    output_dir = Path(output_dir)

    if reconstruction_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
        )

    if opaque_threshold is not None and not 0.0 <= opaque_threshold <= 1.0:
        raise click.UsageError("--opaque threshold must be between 0 and 1")

    try:
        from .._sfmtool.reconstruction import SfmrReconstruction

        click.echo(f"Loading reconstruction: {reconstruction_path}")
        recon = SfmrReconstruction.load(reconstruction_path)
        require_embedded_patches(recon, "sfm render-patches")
        click.echo(f"  Images: {recon.image_count}")
        click.echo(f"  3D points: {recon.point_count}")

        def _progress(name: str, n: int, path: Path) -> None:
            click.echo(f"  {name}: {n} patches -> {path.name}")

        results = render_patches(
            recon,
            output_dir,
            mode=mode,
            border=border,
            border_color=border_color,
            border_thickness=border_thickness,
            alpha=alpha,
            opaque_threshold=opaque_threshold,
            scale=scale,
            upscale=upscale,
            backface_cull=backface_cull,
            image_filter=image_filter or None,
            progress=_progress,
        )
    except PatchRenderError as e:
        raise click.UsageError(str(e))
    except click.UsageError:
        raise  # the embedded_patches precondition (and any other UsageError)
    except Exception as e:
        raise click.ClickException(str(e))

    if not results:
        click.echo("No images matched; nothing rendered.")
    else:
        click.echo(f"\nRendered {len(results)} image(s) to: {output_dir}")
