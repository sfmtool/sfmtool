# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a sift_files reconstruction to embedded_patches (photometric pipeline)."""

import time
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
    type=click.IntRange(min=0),
    default=1,
    show_default=True,
    help=(
        "Number of LK / ECC Gauss-Newton outer sweeps for the photometric "
        "sub-pixel keypoint refinement (always the per-sweep consensus variant). "
        "`0` disables the sub-pixel pass (the localizer's keypoints are used as "
        "is); `N >= 1` runs the refiner with `max_outer_sweeps = N`. Applied once "
        "per round (see --rounds). See specs/core/keypoint-subpixel-refinement.md."
    ),
)
@click.option(
    "--rounds",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help=(
        "Number of (normal-refinement, keypoint-refinement) rounds, alternating "
        "the two. Each round photometrically re-refines every patch normal, then "
        "re-refines its keypoints (the LK sub-pixel pass, see --subpixel), feeding "
        "each result into the next. The discrete keypoint localizer runs once in "
        "the first round as the seed; later rounds refine from the previous "
        "round's normals and keypoints."
    ),
)
@click.option(
    "--max-obliquity-deg",
    type=click.FloatRange(min=0.0, max=90.0),
    default=80.0,
    show_default=True,
    help=(
        "After round 1, drop each observation that views its surfel more than this "
        "many degrees off the (refined) patch normal — a grazing view renders as a "
        "cross-view-consistent but degenerate smear that biases the consensus and "
        "pulls the normal toward grazing over subsequent rounds. `90` keeps all "
        "views (disables the filter)."
    ),
)
@click.option(
    "--obliquity-weight-power",
    type=click.FloatRange(min=0.0),
    default=2.0,
    show_default=True,
    help=(
        "Exponent p of the multiplicative obliquity view-weight |v̂·n|^p folded "
        "into the robust normal-refinement consensus (use A). 0 disables it; 2 "
        "(default) is the cos²θ foreshortening weight that softly down-weights "
        "oblique views — a continuous complement to the hard --max-obliquity-deg "
        "cut."
    ),
)
@click.option(
    "--fronto-prior-weight",
    type=click.FloatRange(min=0.0),
    default=0.05,
    show_default=True,
    help=(
        "Weight λ of the additive fronto-parallel prior λ·mean(v̂·n)² on each "
        "candidate normal during refinement (use B). 0 disables it; the small "
        "default pulls a low-parallax (flat-Φ) normal toward facing the cameras "
        "instead of drifting to a photometrically-equivalent tilt (a distorted "
        "surfel), without overriding a normal that real parallax constrains."
    ),
)
@click.option(
    "--refine-max-views",
    "refine_max_views",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help=(
        "Cap the round-2+ normal-refinement basis at the N most "
        "normal-informative views per point (a D-optimal geometric pick: "
        "least-oblique anchor plus azimuthally-complementary oblique views). "
        "0 uses all views. Only the refinement basis shrinks — all observations "
        "stay in the output, and the consensus bitmaps are still fused over the "
        "full view set. See specs/core/patch-normal-refine-view-subset.md."
    ),
)
@click.option(
    "--localize-search-strategy",
    "localize_search_strategy",
    type=click.Choice(["plus_descent", "exhaustive"]),
    default="plus_descent",
    show_default=True,
    help=(
        "Per-(view, round) shift-grid traversal inside the keypoint localizer's "
        "search_shift. 'plus_descent' (default) is steepest-descent on the 4 "
        "axis neighbors, scoring ~6 cells per call via an AVX2 single-position "
        "vgather kernel; ~1.9× faster end-to-end on dino at comparable accuracy "
        "(median per-observation keypoint shift vs exhaustive ~0.05 px, 91 % "
        "within 1 px). 'exhaustive' scores the full (2·margin+1)² grid via the "
        "SIMD SAXPY accumulator — the global-argmax fallback, no local-optima "
        "risk. See specs/core/keypoint-localization-search-cache.md."
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
    rounds,
    max_obliquity_deg,
    obliquity_weight_power,
    fronto_prior_weight,
    refine_max_views,
    localize_search_strategy,
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
        click.echo(f"  Observations: {recon.observation_count}")

        image_names = recon.image_names
        n_images = len(image_names)
        click.echo(f"Loading source images ({n_images})...")
        # Report every ~5% (min every image) so a large image set (1000+) shows
        # steady progress through the decode instead of one silent block.
        report_every = max(1, n_images // 20)
        load_start = time.perf_counter()
        images = []
        for i, name in enumerate(image_names):
            images.append(read_workspace_image(recon.workspace_dir, name))
            if (i + 1) % report_every == 0 or (i + 1) == n_images:
                click.echo(f"  loaded {i + 1}/{n_images} images")
        click.echo(
            f"  loaded {n_images} images in {time.perf_counter() - load_start:.1f}s"
        )

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
            rounds=rounds,
            max_obliquity_deg=max_obliquity_deg,
            obliquity_weight_power=obliquity_weight_power,
            fronto_prior_weight=fronto_prior_weight,
            max_refine_views=refine_max_views,
            localize_search_strategy=localize_search_strategy,
            progress=click.echo,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"Writing {result.point_count} points to {output_path}...")
        result.save(str(output_path), operation="embed-patches")
        click.echo("\nWrote embedded_patches reconstruction:")
        click.echo(f"  {output_path}")
        click.echo(f"  Points: {result.point_count}  Images: {result.image_count}")
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e))
