# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render an equirectangular panorama from a reconstruction."""

from pathlib import Path

import click
from click.core import ParameterSource

from .._cli_utils import timed_command


@click.command("panorama")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output panorama image path (e.g. pano.png).",
)
@click.option(
    "--image-dir",
    "image_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory the reconstruction's image names are relative to "
    "(default: the reconstruction's workspace directory).",
)
@click.option(
    "--range",
    "-r",
    "range_expr",
    default=None,
    help="Composite only images whose file number matches this range "
    "expression (e.g. '10-50' or '0-9,20-29').",
)
@click.option(
    "--near-image",
    "near_image",
    default=None,
    help="Composite only images spatially near this reference image (matched "
    "by name or path suffix). Requires --near-count and/or --near-radius.",
)
@click.option(
    "--near-count",
    "near_count",
    type=int,
    default=None,
    help="With --near-image, keep the N images whose cameras are closest to "
    "the reference (the reference is always included).",
)
@click.option(
    "--near-radius",
    "near_radius",
    type=float,
    default=None,
    help="With --near-image, keep images whose cameras are within this "
    "world-space distance of the reference.",
)
@click.option(
    "--equirect-width",
    "equirect_width",
    type=int,
    default=2160,
    help="Output width in pixels; height is width / 2 (default: 2160).",
)
@click.option(
    "--n-tiles",
    "n_tiles",
    type=int,
    default=320,
    help="Number of spherical tiles in the synthesized rig (default: 320; "
    "ignored when --camrig is given).",
)
@click.option(
    "--camrig",
    "camrig_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Load the spherical-tile rig from this .camrig file instead of "
    "synthesizing one. Takes precedence over --n-tiles (which is then "
    "ignored); --equirect-width controls only the output resolution.",
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=32,
    help="Tiles composited per batch; smaller bounds peak memory (default: 32).",
)
@click.option(
    "--dtype",
    "dtype",
    type=click.Choice(["float32", "float16"]),
    default="float32",
    help="Per-batch stack storage; float16 halves memory at some precision "
    "cost (default: float32).",
)
@click.option(
    "-k",
    "k",
    type=int,
    default=1,
    help="Nearest tiles blended when resampling; k=1 is closest-tile (default: 1).",
)
@click.option(
    "--seed",
    "seed",
    type=int,
    default=1234,
    help="Rig relaxer seed (default: 1234).",
)
@click.option(
    "--inlier-threshold",
    "inlier_threshold",
    type=float,
    default=8.0,
    help="Photometric RANSAC inlier threshold in luma units (default: 8.0).",
)
@click.option(
    "--gamma",
    "gamma",
    type=float,
    default=1.0,
    help="Photometric RANSAC tone exponent (default: 1.0).",
)
@click.option(
    "--ransac-seed",
    "ransac_seed",
    type=int,
    default=0,
    help="Photometric RANSAC seed (default: 0).",
)
def panorama(
    reconstruction_path,
    output_path,
    image_dir,
    range_expr,
    near_image,
    near_count,
    near_radius,
    equirect_width,
    n_tiles,
    camrig_path,
    batch_size,
    dtype,
    k,
    seed,
    inlier_threshold,
    gamma,
    ransac_seed,
):
    """Render an equirectangular panorama from a posed reconstruction.

    Composites the reconstruction's source images onto a sphere of spherical
    tiles, selects the photometrically-agreeing cluster per tile, and resamples
    the result through a full-sphere equirectangular camera. Uncovered regions
    are written as black.

    Examples:

        # Render a 2160x1080 panorama
        sfm panorama result.sfmr -o pano.png

        # Higher resolution with images in an explicit directory
        sfm panorama result.sfmr -o pano.png --equirect-width 4096 \\
            --image-dir images/

        # Render with a pre-built rig (tile density decoupled from output size)
        sfm panorama result.sfmr -o pano.png --camrig tiles.camrig \\
            --equirect-width 4096
    """
    import cv2
    import numpy as np

    from ..rig.panorama import (
        render_equirect_panorama,
        resolve_panorama_rig,
        select_source_indices,
    )
    from .._sfmtool import SfmrReconstruction

    reconstruction_path = Path(reconstruction_path)
    output_path = Path(output_path)
    if camrig_path is not None:
        camrig_path = Path(camrig_path)

    if reconstruction_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
        )
    if camrig_path is not None:
        if camrig_path.suffix.lower() != ".camrig":
            raise click.UsageError(
                f"--camrig must be a .camrig file, got: {camrig_path}"
            )
        ctx = click.get_current_context()
        if ctx.get_parameter_source("n_tiles") == ParameterSource.COMMANDLINE:
            click.echo("Note: --n-tiles is ignored because --camrig was supplied.")
    if equirect_width < 2 or equirect_width % 2 != 0:
        raise click.UsageError("--equirect-width must be a positive even integer.")
    if (near_count is not None or near_radius is not None) and near_image is None:
        raise click.UsageError("--near-count / --near-radius require --near-image.")
    if near_image is not None and near_count is None and near_radius is None:
        raise click.UsageError(
            "--near-image requires --near-count and/or --near-radius."
        )
    if near_count is not None and near_count < 1:
        raise click.UsageError("--near-count must be >= 1.")
    if near_radius is not None and near_radius <= 0:
        raise click.UsageError("--near-radius must be > 0.")

    try:
        click.echo(f"Loading reconstruction: {reconstruction_path}")
        recon = SfmrReconstruction.load(reconstruction_path)
        click.echo(f"  Images: {recon.image_count}, cameras: {recon.camera_count}")

        indices = select_source_indices(
            recon,
            range_expr=range_expr,
            near_image=near_image,
            near_count=near_count,
            near_radius=near_radius,
        )
        if indices is not None:
            click.echo(
                f"  Selected {len(indices)} of {recon.image_count} images "
                "after range/near-image filtering"
            )
            recon = recon.subset_by_image_indices(indices, drop_orphaned_points=False)

        base_dir = (
            Path(image_dir) if image_dir is not None else Path(recon.workspace_dir)
        )

        rig = resolve_panorama_rig(
            equirect_width=equirect_width,
            n_tiles=n_tiles,
            camrig_path=camrig_path,
            seed=seed,
        )
        tile_fx = rig.tile_camera().focal_lengths[0]
        if camrig_path is not None:
            click.echo(f"Loaded rig from {camrig_path}")
        click.echo(
            f"Rendering {equirect_width}x{equirect_width // 2} panorama "
            f"(tiles={rig.n}, tile_fx={tile_fx:.1f}px, k={k}) from images in {base_dir}"
        )

        pano = render_equirect_panorama(
            recon,
            base_dir,
            rig=rig,
            equirect_width=equirect_width,
            batch_size=batch_size,
            dtype=dtype,
            k=k,
            inlier_threshold=inlier_threshold,
            gamma=gamma,
            ransac_seed=ransac_seed,
        )

        # Uncovered samples are NaN; flatten them to black for image output.
        pano = np.where(np.isnan(pano), 0.0, pano)
        pano_u8 = np.clip(pano, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(pano_u8, cv2.COLOR_RGB2BGR)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), bgr):
            raise RuntimeError(f"Failed to write panorama to {output_path}")
        click.echo(f"Wrote panorama: {output_path}")

    except Exception as e:
        raise click.ClickException(str(e))
