# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a sift_files reconstruction to embedded_patches (photometric pipeline)."""

from pathlib import Path

import click

from .._cli_utils import timed_command


@click.command("embed-patches")
@timed_command
@click.help_option("--help", "-h")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False)
@click.option(
    "--min-relative-zncc",
    type=float,
    default=0.7,
    show_default=True,
    help=(
        "Minimum ZNCC a view must reach, as a fraction of the reference's own "
        "agreement — admits candidate views and drops poorly-registering ones "
        "during congealing."
    ),
)
@click.option(
    "--max-iters",
    type=int,
    default=5,
    show_default=True,
    help="Max congealing rounds per point (stops early at convergence).",
)
@click.option(
    "--search",
    type=float,
    default=6.0,
    show_default=True,
    help="Max total per-view in-plane drift, in patch-grid pixels.",
)
@click.option(
    "--max-shift-px",
    type=float,
    default=3.0,
    show_default=True,
    help=(
        "Discard an observation whose keypoint sits more than this from the "
        "point's projection, in source-image pixels."
    ),
)
@click.option(
    "--min-views",
    type=int,
    default=2,
    show_default=True,
    help="Drop a point left with fewer surviving observations after discards.",
)
@click.option(
    "--patch-size",
    type=float,
    default=5.0,
    show_default=True,
    help=(
        "Surfel size — the full patch edge length (in feature-size multiples), "
        "halved to the library half-extent and passed to to_embedded_patches."
    ),
)
@click.option(
    "--search-resolution-multiplier",
    "search_resolution_multiplier",
    type=float,
    default=1.0,
    show_default=True,
    help=(
        "Multiplier m for the discrete cross-view search: it runs at resolution "
        "round(m·R). 1.0 is the no-op; >1 (the supersampled grid) resolves "
        "sub-pixel offsets at a cost that grows ~m². See "
        "specs/core/keypoint-localization-search-cache.md."
    ),
)
@click.option(
    "--subpixel",
    type=click.Choice(["none", "lk", "lk_per_move"]),
    default="none",
    show_default=True,
    help=(
        "Optional photometric sub-pixel pass applied to the localizer's output. "
        "'none' (default) skips it; 'lk' runs LK / ECC Gauss-Newton refinement "
        "with the per-sweep consensus (max_outer_sweeps=1); 'lk_per_move' uses "
        "the per-move (Gauss-Seidel) incremental consensus with "
        "max_outer_sweeps=5. See specs/core/keypoint-subpixel-refinement.md."
    ),
)
def embed_patches_command(
    input_path,
    output_path,
    min_relative_zncc,
    max_iters,
    search,
    max_shift_px,
    min_views,
    patch_size,
    search_resolution_multiplier,
    subpixel,
):
    """Convert a sift_files reconstruction to embedded_patches.

    Builds a patch frame for each point (mean-viewing normal, refined
    photometrically), expands and vets each point's view set, congeals the
    per-view keypoints to sub-pixel, then writes a NEW embedded_patches .sfmr that
    loads and verifies with no .sift companion. The input is never modified.

    INPUT_PATH is a sift_files reconstruction (e.g. straight from `sfm solve`);
    its .sift files must still be present where it was created. OUTPUT_PATH, when
    omitted, is written next to the input as `<stem>-embedded.sfmr` (then
    `-embedded-2.sfmr`, ... if taken), mirroring `sfm xform`.

    The observation set is the input track reshaped — expanded by vetting, trimmed
    by per-view discards, then compacted — so point and observation counts
    generally differ from the input.

    \b
    Examples:
        # Straight from a solve; writes solve-embedded.sfmr next to the input.
        sfm embed-patches solve.sfmr

    \b
        # Explicit output, tighter budgets.
        sfm embed-patches solve.sfmr out.sfmr \\
            --max-iters 3 --search 4 --min-relative-zncc 0.75
    """
    from .._embed_patches import embed_patches
    from .._sfmtool import SfmrReconstruction
    from .._workspace_image import read_workspace_image
    from ..xform._arg_parser import auto_output_path

    input_path = Path(input_path)
    if input_path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"Input path must be a .sfmr file, got: {input_path}")

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".sfmr":
            raise click.UsageError(
                f"Output path must be a .sfmr file, got: {output_path}"
            )
    else:
        output_path = auto_output_path(input_path, suffix="embedded")

    try:
        click.echo(f"Loading reconstruction from: {input_path}")
        recon = SfmrReconstruction.load(input_path)
        if recon.feature_source != "sift_files":
            raise click.UsageError(
                f"Input is already {recon.feature_source}; nothing to convert "
                "(embed-patches converts sift_files → embedded_patches)."
            )
        click.echo(f"  Images: {recon.image_count}")
        click.echo(f"  Points: {recon.point_count}")

        click.echo("Loading source images...")
        images = [
            read_workspace_image(recon.workspace_dir, name)
            for name in recon.image_names
        ]

        click.echo(
            "Building + refining patches, selecting views, localizing keypoints..."
        )
        result = embed_patches(
            recon,
            images,
            min_relative_zncc=min_relative_zncc,
            patch_size=patch_size,
            max_shift_px=max_shift_px,
            min_views=min_views,
            max_iters=max_iters,
            search=search,
            search_resolution_multiplier=search_resolution_multiplier,
            subpixel=subpixel,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_path), operation="embed-patches")
        click.echo("\nWrote embedded_patches reconstruction:")
        click.echo(f"  {output_path}")
        click.echo(f"  Points: {result.point_count}  Images: {result.image_count}")
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e))
