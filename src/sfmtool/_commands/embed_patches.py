# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a sift_files reconstruction to embedded_patches (photometric pipeline)."""

import time
from concurrent.futures import ThreadPoolExecutor

import click

from .._cli_utils import timed_command
from ._sfmr_path import check_sfmr_path


# Extra positional arguments are collected rather than refused by click, so a
# count left over from `--subpixel N` is reported as that in any position.
@click.command("embed-patches", context_settings={"allow_extra_args": True})
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
        "Minimum ZNCC a view must reach, as a fraction of its peers' — admits "
        "candidate views, and drops poorly-registering ones when the views are "
        "aligned to the reference render, by their blur-matched score against "
        "it."
    ),
)
@click.option(
    "--min-absolute-zncc",
    type=float,
    default=0.5,
    show_default=True,
    help=(
        "Refuse an observation whose blur-matched score against the point's "
        "reference render is below this absolute floor. 0 disables it."
    ),
)
@click.option(
    "--max-member-zncc-self-similarity-radius",
    type=float,
    default=2.5,
    show_default=True,
    help=(
        "Refuse an observation whose OWN patch tile pins no 2D position: its "
        "ZNCC self-similarity radius, how far the tile can slide over itself "
        "and still match itself, is above this, in patch-grid pixels. It "
        "throws out a flat sky or water crop, or a lone straight edge, before "
        "it is scored. The radius reads at most 3, so 3 or more turns nothing "
        "out; 0 disables it. The default, 2.5, is the bench's bar. See "
        "specs/core/patch/zncc-self-similarity-radius.md."
    ),
)
@click.option(
    "--search",
    type=float,
    default=6.0,
    show_default=True,
    help=(
        "The reach of each view's search around its starting keypoint, in "
        "patch-grid pixels."
    ),
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
    default=11.0,
    show_default=True,
    help=(
        "Full edge length of each point's patch, in multiples of the SIFT "
        "feature size. A smaller patch gives the normal and keypoint "
        "refinement less texture to compare."
    ),
)
@click.option(
    "--subpixel/--no-subpixel",
    default=True,
    show_default=True,
    help=(
        "Run the photometric sub-pixel keypoint refinement (LK / ECC "
        "Gauss-Newton against the point's reference render) once per round "
        "(see --rounds). --no-subpixel uses the localizer's keypoints as they "
        "are. See specs/core/patch/keypoint-subpixel-refinement.md."
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
        "In every round, before the sub-pixel keypoint refinement, drop each "
        "observation that views its surfel more than this many degrees off the "
        "(refined) patch normal, the reference observation included — a grazing "
        "view renders as a cross-view-consistent but degenerate smear that biases "
        "the consensus and pulls the normal toward grazing over subsequent rounds. "
        "`90` keeps all views (disables the filter)."
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
    default=8,
    show_default=True,
    help=(
        "In rounds 2 and later, refine each point's normal against at most N "
        "of its views, the ones that constrain the normal most. 0 uses all "
        "views. Every observation stays in the output; only the views the "
        "normal is refined against are limited. See "
        "specs/core/patch/patch-normal-refine-view-subset.md."
    ),
)
@click.option(
    "--max-zncc-self-similarity-radius",
    type=click.FloatRange(min=0.0),
    default=2.5,
    show_default=True,
    help=(
        "Cull points whose patch bitmap pins no 2D position, "
        "EARLY — right after round 1's localize + sub-pixel refine, before the "
        "multi-round refinement: its ZNCC self-similarity radius, how far the "
        "bitmap can slide over itself and still match itself, is above this, in "
        "patch-grid pixels. It removes points on a straight edge or a flat "
        "patch, which the agreement gate lets through. Each shift is correlated "
        "over the samples the bitmap holds on both sides, as the member gates "
        "read their tiles. The radius reads at most 3, so 3 or more turns "
        "nothing out; 0 disables the cull. The default, 2.5, is the member "
        "gate's. See specs/core/patch/zncc-self-similarity-radius.md."
    ),
)
@click.option(
    "--localize-search-strategy",
    "localize_search_strategy",
    type=click.Choice(["exhaustive", "plus_descent"]),
    default="exhaustive",
    show_default=True,
    help=(
        "How the keypoint localizer searches each view's shift window for the "
        "best match to the reference render. 'exhaustive' (default) scores "
        "every shift in the window and takes the highest peak. 'plus_descent' "
        "climbs from the starting keypoint to the nearest peak, scoring fewer "
        "shifts, and stops at a side peak more often when the start is 2 px or "
        "more off. See specs/core/patch/patch-keypoint-localization.md."
    ),
)
@click.option(
    "--sampler",
    type=click.Choice(["per_view", "bilinear", "bilinear_mip", "anisotropic"]),
    default="per_view",
    show_default=True,
    help=(
        "How every step that renders a patch from an image (normal "
        "refinement, view selection, keypoint localization, sub-pixel "
        "refinement and the stored bitmap) samples the image pyramid. "
        "'per_view' chooses for each view: 'anisotropic' where 'bilinear_mip' "
        "would read the view's less compressed axis too coarsely, "
        "'bilinear_mip' otherwise. The other three use one sampler for every "
        "view: 'bilinear' reads the full-resolution level only; "
        "'bilinear_mip' reads the pyramid level nearest the view's "
        "compression, which limits aliasing on views at a different scale; "
        "'anisotropic' also samples obliquely seen patches correctly; its "
        "cost against 'bilinear_mip' depends on the view and on whether the "
        "CPU has AVX2. See specs/core/camera/image-warping.md."
    ),
)
def embed_patches_command(
    input_path,
    output_path,
    min_relative_zncc,
    min_absolute_zncc,
    max_member_zncc_self_similarity_radius,
    search,
    max_shift_px,
    min_views,
    patch_size,
    subpixel,
    rounds,
    max_obliquity_deg,
    obliquity_weight_power,
    fronto_prior_weight,
    refine_max_views,
    max_zncc_self_similarity_radius,
    localize_search_strategy,
    sampler,
):
    """Convert a sift_files reconstruction to embedded_patches.

    Builds a patch frame for each point (mean-viewing normal, refined
    photometrically), expands and vets each point's view set, aligns the
    per-view keypoints to each point's reference render and refines them to
    sub-pixel, then writes a NEW embedded_patches .sfmr that
    loads and verifies with no .sift companion. The input is never modified.

    INPUT_PATH is a sift_files reconstruction (e.g. straight from `sfm solve`);
    its .sift files must still be present where it was created. OUTPUT_PATH, when
    omitted, is written next to the input as `<stem>-embedded.sfmr` (then
    `-embedded-2.sfmr`, ... if taken), mirroring `sfm xform`.

    The observation set is the input track reshaped — expanded by vetting, trimmed
    by per-view discards, then compacted — so point and observation counts
    generally differ from the input.

    An input that is already embedded_patches and stores reference
    observations keeps them: each such point's views are aligned to its own
    reference observation and its bitmap is rendered from it at the final
    keypoints. A point takes the reference-view rule's pick where it stores
    none, where the obliquity cut or the localizer drops its reference, or
    where its reference does not render at its keypoint.

    \b
    Examples:
        # Straight from a solve; writes solve-embedded.sfmr next to the input.
        sfm embed-patches solve.sfmr

    \b
        # Explicit output, tighter budgets.
        sfm embed-patches solve.sfmr out.sfmr \\
            --search 4 --min-relative-zncc 0.75
    """
    from .._embed_patches import embed_patches
    from .._sfmtool.patches import DEFAULT_ANISOTROPIC_THRESHOLD
    from .._sfmtool.reconstruction import SfmrReconstruction
    from .._workspace_image import read_workspace_image
    from ..xform._arg_parser import auto_output_path

    input_path = check_sfmr_path(input_path, "Input path")

    # `--subpixel N` from before the flag became a switch: the count is read
    # as the output path, or as an extra argument after it.
    extra = list(click.get_current_context().args)
    count = next(
        (a for a in [output_path, *extra] if a is not None and a.isdigit()), None
    )
    if count is not None:
        raise click.UsageError(
            f"--subpixel no longer takes a count (got {count}); use "
            "--subpixel or --no-subpixel, and --rounds for the number of rounds."
        )
    if extra:
        raise click.UsageError(f"Got unexpected extra arguments ({' '.join(extra)})")

    if output_path is not None:
        output_path = check_sfmr_path(output_path, "Output path")
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
        # Decode in a thread pool (cv2 releases the GIL), collecting results in
        # submission order so `images` stays parallel to `image_names`.
        with ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(read_workspace_image, recon.workspace_dir, name)
                for name in image_names
            ]
            images = []
            try:
                for i, future in enumerate(futures):
                    images.append(future.result())
                    if (i + 1) % report_every == 0 or (i + 1) == n_images:
                        click.echo(f"  loaded {i + 1}/{n_images} images")
            except BaseException:
                # Fail fast: without this, the pool's __exit__ would finish
                # decoding every queued image before the error surfaces.
                for f in futures:
                    f.cancel()
                raise
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
            min_absolute_zncc=min_absolute_zncc,
            max_member_zncc_self_similarity_radius=max_member_zncc_self_similarity_radius,
            patch_size=patch_size,
            max_shift_px=max_shift_px,
            min_views=min_views,
            search=search,
            subpixel=subpixel,
            rounds=rounds,
            max_obliquity_deg=max_obliquity_deg,
            obliquity_weight_power=obliquity_weight_power,
            fronto_prior_weight=fronto_prior_weight,
            max_refine_views=refine_max_views,
            max_zncc_self_similarity_radius=max_zncc_self_similarity_radius,
            localize_search_strategy=localize_search_strategy,
            sampler=sampler,
            progress=click.echo,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"Writing {result.point_count} points to {output_path}...")
        # Record embed's own knobs. `save` MERGES into `metadata.tool_options`,
        # so without this the written file advertises the upstream solve's
        # options as if they were this run's — and the gates below decide which
        # observations survive, which is exactly what a reader of the file needs
        # to know.
        result.save(
            str(output_path),
            operation="embed-patches",
            tool_options={
                "patch_size": patch_size,
                "rounds": rounds,
                "subpixel": subpixel,
                "min_views": min_views,
                "search": search,
                "max_shift_px": max_shift_px,
                "min_relative_zncc": min_relative_zncc,
                "min_absolute_zncc": min_absolute_zncc,
                "max_member_zncc_self_similarity_radius": max_member_zncc_self_similarity_radius,
                "max_zncc_self_similarity_radius": max_zncc_self_similarity_radius,
                "max_obliquity_deg": max_obliquity_deg,
                "obliquity_weight_power": obliquity_weight_power,
                "fronto_prior_weight": fronto_prior_weight,
                "refine_max_views": refine_max_views,
                "localize_search_strategy": localize_search_strategy,
                "sampler": sampler,
                # The sampler rule's threshold the renders were made under,
                # from which a reader works out each render's sampler from its
                # zoom; none for a fixed sampler.
                "anisotropic_threshold": (
                    DEFAULT_ANISOTROPIC_THRESHOLD if sampler == "per_view" else None
                ),
            },
        )
        click.echo("\nWrote embedded_patches reconstruction:")
        click.echo(f"  {output_path}")
        click.echo(f"  Points: {result.point_count}  Images: {result.image_count}")
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e))
