# `sfm embed-patches` Command

## Overview

`sfm embed-patches` makes photometric patch matching, in place of SIFT
descriptor matching, decide a reconstruction's tracks and keypoints. Its input
is a `sift_files` `.sfmr`, typically straight from `sfm solve`, whose
observations are SIFT detections joined into tracks by descriptor matching. The
command gives each point a small oriented patch, renders the patch in every
image that geometrically sees the point, and compares the renders by ZNCC to
decide both which images belong to the point's track and where in each image
the point lies. An image that SIFT matching missed joins the track when its
render agrees with the others, an observation whose render does not agree is
dropped, and each kept keypoint is placed by aligning its render to the
render of the point's reference observation rather than copied from the SIFT
detection. The point and
observation counts therefore usually differ from the input. The output is a new
`embedded_patches` `.sfmr`, in which each point carries its patch and an image
of its appearance and each observation carries its keypoint inline. The `.sift`
files are read in one step, which sizes each patch from the SIFT feature scale,
seeds the keypoints at the SIFT detections and reads the image hashes, so they
must be present for the run; the input file is never modified.

The conversion:

1. **Build a patch frame.** A keypoint anchors the point's surfel, so each point
   needs a `(u, v)` frame. The command initializes each frame (normal from the
   mean viewing direction, via `to_embedded_patches`) and refines that normal
   photometrically (the `refine_normals` kernel, which re-persists the frame).
2. **Derive the keypoints over an expanded, vetted view set.** For each point,
   expand the track with the other views that geometrically see the surfel,
   photometrically vet them against a track-seeded template, and align each
   view's keypoint to the point's reference render: the tile of its reference
   observation, which the reference-view rule picks from the views' renders at
   their starting keypoints, and whose keypoint is not moved. The discrete
   keypoint localizer places each view, and the sub-pixel refiner then refines
   it against the same reference.
3. **Write an `embedded_patches` file.** Drop the `.sift`-link columns, add the
   inline keypoints, pin image identity directly, and compact — so the result
   verifies and loads with no `.sift` present.

This is a Reconstruction-category command (`src/sfmtool/_commands/`).

## Command Syntax

```bash
sfm embed-patches INPUT.sfmr [OUTPUT.sfmr] [options]
```

`INPUT.sfmr` is a `sift_files` reconstruction (e.g. straight from `sfm solve`).
The command builds a patch frame for each point (initialize + refine its normal).
The result is written as an `embedded_patches` file.
When `OUTPUT.sfmr` is omitted it is written next to the input as
`<stem>-embedded.sfmr`; if that name is taken, a numeric suffix is appended
starting at 2 (`<stem>-embedded-2.sfmr`, `-3`, …), mirroring `sfm xform`. The
input is never overwritten — writing over it requires passing its path explicitly
as `OUTPUT`.

## Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--min-relative-zncc` | float | 0.7 | Minimum ZNCC a view must reach, as a fraction of its peers' — used both to admit candidate views (against the track-seeded template) and, when the views are aligned to the reference render, to drop a view whose ZNCC against it is below this fraction of the median over the views other than the reference. |
| `--min-absolute-zncc` | float | `0.5` | Discard an **observation** whose ZNCC against the point's reference render is below this absolute floor, however few views the point has left. On a two-view point the relative bar compares the one view's score with itself and always passes, so this floor is what refuses a pair of unrelated surfaces, and a point it leaves below `--min-views` is dropped whole. `0` disables it. |
| `--max-member-zncc-self-similarity-radius` | float | `2.5` | Discard an **observation** whose own patch tile pins no 2D position: its [ZNCC self-similarity radius](../../core/patch/zncc-self-similarity-radius.md), how far the tile can slide over itself and still match itself, is above this, in **patch-grid** pixels. Applied to one view's tile before it is scored against anything, so a flat sky or water crop, or a lone straight edge, is not aligned at all. The radius reads at most `3`, so `3` or more turns nothing out. `0` disables it. The default is `2.5`, the bench's bar; see [the member gate's default](../../core/patch/patch-keypoint-localization.md#the-member-gates-default). |
| `--search` | float | 6 | The reach of each view's search around its starting keypoint, in **patch-grid** pixels. |
| `--max-shift-px` | float | 3.0 | Discard an **observation** whose keypoint sits more than this from the point's projection, in **source-image** pixels (an absolute distance, not the move from the seed). |
| `--min-views` | int | 2 | Drop a point left with fewer surviving observations after discards. |
| `--patch-size` | float | `11.0` | Surfel size — the full patch edge length (in feature-size multiples) used to render the patch. Halved to the library half-extent and passed to `to_embedded_patches` (`extent="feature_size"`), the step that builds the frame. The default sits at SIFT's ~12× descriptor window — the patch carries roughly the texture context the detector itself deems characteristic of the feature. Smaller patches starve the normal/keypoint refiners (weaker normals, more self-similarity culls); larger ones trade observation yield away to the grazing cull for marginal accuracy gains. |
| `--subpixel` / `--no-subpixel` | flag | on | Run the photometric sub-pixel keypoint refinement (LK / ECC Gauss–Newton against the point's reference render, one pass) once per round. `--no-subpixel` moves no keypoint: the localizer's keypoints are used as they are, and the final round still runs the stage render-only to render each point's stored bitmap and validity at those keypoints. See [`specs/core/patch/keypoint-subpixel-refinement.md`](../../core/patch/keypoint-subpixel-refinement.md). |
| `--rounds` | int ≥ 1 | `2` | Number of (normal-refinement, keypoint-refinement) rounds, alternating the two. Round 1 runs the SIFT-anchored normal refine, the discrete localizer (the seed), then the sub-pixel keypoint refine; each subsequent round re-refines every normal against the previous round's keypoints, then re-refines the keypoints against the new normals — a fixed-point alternation. The default `2` runs one refinement pass on top of the seed round (most of the normal/keypoint convergence gain lands in the first extra round); raise it for the tail. The per-point view set (membership) is fixed after round 1 (and per-round grazing drops only shrink it). With a `progress` sink (the CLI wires `click.echo`) each round prints its mean normal change (deg) and mean keypoint shift (px). |
| `--max-obliquity-deg` | float 0–90 | `80` | After round 1, drop every observation viewing its surfel more than this many degrees off the refined normal (`\|v̂·n\| < cos θ`). A grazing view renders as a cross-view-consistent but **degenerate smear** that satisfies the consensus yet erases surface texture, so over multiple rounds it drags the normal toward grazing (a self-reinforcing failure). Dropping those views keeps the round alternation on the well-observed, near-frontal views. `90` disables the filter. The obliquity is exactly the `inspect --strips` magenta-dot radius (`sin θ`). (With the fronto prior on by default, surfels stay near-frontal, so this cut fires far less often — it is the backstop for the residual grazing observations.) |
| `--obliquity-weight-power` | float ≥ 0 | `2` | Exponent `p` of the multiplicative **obliquity view-weight** `\|v̂·n\|^p` folded into the robust normal-refinement consensus (use A). `0` disables it — the consensus runs as before. `2` (default) is the `cos²θ` foreshortening weight: it softly down-weights a view the more obliquely it sees the surfel — a continuous complement to the hard `--max-obliquity-deg` cut, on points whose views span a range of obliquities. On a low-parallax point (all views near-collinear, hence near-equal obliquity) it renormalizes away; that case is what `--fronto-prior-weight` addresses. See [`specs/core/patch/patch-normal-refinement.md`](../../core/patch/patch-normal-refinement.md). |
| `--fronto-prior-weight` | float ≥ 0 | `0.05` | Weight `λ` of the additive **fronto-parallel prior** `λ·mean_v (v̂·n)²` on each candidate normal during refinement (use B). `0` disables it. It rewards normals that face the observing cameras, supplying the constraint the data can't when `Φ` is flat — the narrow-baseline degeneracy where every candidate tilt shifts all views' patches identically, so a low-parallax surfel drifts to a photometrically-equivalent tilt and renders distorted (a stop sign's octagon shears into a smear). The prior lands it fronto-parallel instead; wherever real parallax curves `Φ` the small prior is overruled, so well-constrained normals are unaffected. The `0.05` default (with `--obliquity-weight-power 2`) straightens low-parallax surfels at negligible photoconsistency cost. See [`specs/core/patch/patch-normal-refinement.md`](../../core/patch/patch-normal-refinement.md). |
| `--refine-max-views` | int ≥ 0 | `8` | Cap the **round-2+ normal-refinement basis** at the `N` most normal-informative views per point — the D-optimal geometric pick of [`specs/core/patch/patch-normal-refine-view-subset.md`](../../core/patch/patch-normal-refine-view-subset.md) (a least-oblique appearance anchor plus a greedy information-determinant fill; always the best `N`, no fall-back-to-all). `0` disables the cap (use all views). Applies only to the fine-tuning rounds, whose view set is the `select_views`-expanded one; the round-1 (raw-track) refine is untouched. **Lossless for the output**: only the refinement basis shrinks — every observation stays, and the reference view each stored bitmap is rendered from is picked from the full view set. The default `8` cuts end-to-end time by about a third against all views on large view sets (29–37 % on the Spain Soapmaker sweep, with the round-2 refinement itself 3.4–16.8× faster for `K` from 10 down to 3; see that spec § "Choice of `K`"). |
| `--max-zncc-self-similarity-radius` | float ≥ 0 | `2.5` | Drop a **point** whose round-1 stored bitmap (the render the sub-pixel stage makes after round 1) pins no 2D position: its [ZNCC self-similarity radius](../../core/patch/zncc-self-similarity-radius.md), how far the bitmap can slide over itself and still match itself, is above this, in **patch-grid** pixels. Read right after round 1's sub-pixel refine, before the multi-round refinement, so the points it drops cost nothing further. It is read [the overlap way](../../core/patch/zncc-self-similarity-radius.md#the-overlap-reading), as the member gates read their tiles: each shift is correlated over the samples the bitmap holds, with alpha above 0, on both sides. It drops points on a straight edge or a flat patch, which the agreement gates let through. A point with no bitmap has no reading and is left to the track thresholds. The radius reads at most `3`, so `3` or more turns nothing out; `0` disables it and skips the round-1 bitmap render on a multi-round run. The default is `2.5`, the member gate's bar. |
| `--localize-search-strategy` | choice | `plus_descent` | Per-view shift-grid traversal inside the keypoint localizer's `search_shift`. `plus_descent` (default) is steepest-descent on the 4 axis neighbors, scoring ~6 cells per call via an AVX2 single-position vgather kernel — ~1.9× faster end-to-end on dino at comparable accuracy (median per-observation keypoint shift vs `exhaustive` ~0.05 px, 91 % within 1 px). `exhaustive` scores the full `(2·margin+1)²` grid via the SIMD SAXPY accumulator — the global-argmax fallback, no local-optima risk. See [`specs/core/patch/keypoint-localization-search-cache.md`](../../core/patch/keypoint-localization-search-cache.md). |
| `--sampler` | choice | `per_view` | Pyramid sampler for every photometric kernel in the pipeline (normal refinement, view selection, keypoint localization, sub-pixel refinement, the stored bitmap's render): `per_view` applies the sampler rule to each view, rendering it with `anisotropic` where `bilinear_mip` would read its less compressed axis at least 1.5× too coarsely and with `bilinear_mip` otherwise, so every kernel reads the same view through the same sampler (see [`specs/core/camera/image-warping.md`](../../core/camera/image-warping.md) § "Choosing the sampler per view"). The other three render every view with one sampler: `bilinear_mip` taps the mip level nearest the warp's compression, bounding aliasing on cross-scale views at ~bilinear cost; `anisotropic` also resolves oblique footprints, with the AVX2 kernel at 0.65–1.55× the cost of `bilinear_mip` per tile (the most on views compressed 10 times or more along one axis) and 1.8–4× it on a CPU without AVX2 (the value+gradient render the sub-pixel refinement reads has no AVX2 kernel and costs 2.8–7×); `bilinear` taps the full-resolution level only. |

The two `--search` / `--max-shift-px` budgets are in different units on purpose:
`--search` bounds the alignment in the patch's own grid (the localizer's
search), while the discard gate `--max-shift-px` is read back in
source-image pixels (the pipeline's quality control). See the core specs.

**Failures are discarded, not back-filled** — every keypoint written reflects a
real registration (see Behaviour Notes), so the output observation set is the
input track reshaped (expanded by vetting, trimmed by drops), not copied through.

## Behaviour Notes

- **View set.** Each point is localized over its track **plus** the other views
  that geometrically see the surfel and pass photometric vetting (based on
  `--min-relative-zncc`).
- **Alignment to the reference render.** Each point's views are aligned, in one
  pass, to its reference render: the `R×R` tile of its reference observation at
  that observation's starting keypoint. The reference is the reference-view
  rule's pick from the views' renders at their starting keypoints; where the
  rule picks none it would store, the template is the views' fused mean and no
  view is the reference. An observed view starts at its stored (SIFT) keypoint,
  and a view the vetting added starts at the point's projection. The
  reference's keypoint is not moved. Every other view is searched once within
  `±--search` patch-grid px of its start, and the sub-pixel refiner then
  refines it against the same reference render in each round; later rounds keep
  the reference.
- **Observation thresholds.** A view is dropped if it can't be localized
  cleanly (grazing view, out-of-frame keypoint), if its own tile pins no 2D
  position (its ZNCC self-similarity radius above
  `--max-member-zncc-self-similarity-radius`, when that gate is on), if its
  keypoint sits more than `--max-shift-px` from the point's projection, if its
  ZNCC against the reference render is below `--min-absolute-zncc`, or if that
  ZNCC falls below `--min-relative-zncc` of the median over the views other
  than the reference. The reference faces none of these gates after the
  grazing check, and a point can come out of localization below `--min-views`
  and be dropped whole.
- **Reference bitmaps.** Each surviving point's stored bitmap is the tile of
  its **reference observation**, the one its views were aligned to, at the
  final per-view keypoints of the sub-pixel keypoint-refinement stage
  (`refine_keypoints(render_bitmaps=True)`; with `--no-subpixel` the stage
  still runs render-only at the localizer's keypoints), and
  `tracks/reference_observations` records which observation it is
  ([reference-view.md](../../core/patch/reference-view.md) § "The stored
  bitmap"). Where the views were aligned to the fused mean, the bitmap is the
  views' fused mean and the point records `-1`. Points at infinity go through the same
  `w`-aware render path and get a real bitmap — no zero-row exemption. An
  input that is already `embedded_patches` and stores reference observations
  keeps them: the compaction carries each point's reference to the
  observation of the same image in its final track
  (`compact_to_embedded_patches`), so only a point at
  `-1`, or one whose reference image the refinement dropped, takes the rule's
  pick. Every point with a reference, kept or picked, then has its bitmap
  rendered again from that observation through the compacted value's stored
  `f32` keypoints and frame (`render_from_references`), so dropping and adding
  the bitmaps later gives the same bytes; a fused mean stays as the pass
  rendered it.
- **Self-similarity cull.** After round 1 the sub-pixel stage renders each
  point's bitmap (whatever the round count, while the cull is on),
  and a point whose bitmap's ZNCC self-similarity radius is over
  `--max-zncc-self-similarity-radius` is dropped before round 2. The reading
  depends on the point's own bitmap alone, so dropping a point early changes
  no other point's bitmap. On the seoul_bull and kerry_park solves, measured on fused-mean bitmaps
  before the stored bitmap was the reference view's render, it dropped 59
  of 837 and 292 of 1,886 points at the default. The run records the bar
  in the file's `tool_options` as `max_zncc_self_similarity_radius`; files
  written before the cull read the radius carry `max_keypoint_uncertainty` there, and
  nothing reads either key back.
- **Sampler.** The run records `--sampler` in `tool_options` as `sampler`, and
  beside it `anisotropic_threshold`, the sampler rule's threshold the renders
  were made under (`1.5`), or null for a fixed sampler. With the threshold and a
  view's zoom a reader can work out which sampler rendered it.
- **Track thresholds.** A point is dropped whole when its support count falls
  below `--min-views`, or when the sub-pixel stage produced **no valid
  bitmap** for it (fewer than two views, or no reference view and fewer than two
  of its views render at their final keypoints for the fused mean) —
  the same rule for finite and infinity points, so no kept point carries an
  all-black bitmap.
- **Image identity.** For each surviving image, `images/image_file_hashes[i]` is
  copied from the image's `.sift` `image_file_xxh128` metadata field (hex → 16
  bytes, the same decode already used for `sift_content_hashes`).

## Errors

- Input is already `embedded_patches` → error (nothing to convert).
- A referenced image has no resolvable `.sift` (needed for `image_file_hashes`)
  → error naming the image.

## Output

An `embedded_patches` `.sfmr` that loads and verifies with no
`.sift` companion. Its observation set is the input track reshaped (expanded by
vetting, filtered by discards, compacted), so point and observation counts
generally differ from the input.

## Usage Examples

```bash
# Straight from a solve: builds the patch frame, then writes the result next to
# the input as solve-embedded.sfmr (no output arg).
sfm embed-patches solve.sfmr

# Explicit output, tighter budgets.
sfm embed-patches solve.sfmr out.sfmr \
  --search 4 --min-relative-zncc 0.75
```

## Module Layout

The command lives in [embed_patches.py](../../../src/sfmtool/_commands/embed_patches.py)
and its orchestration in [_embed_patches.py](../../../src/sfmtool/_embed_patches.py),
which drives the Rust patch kernels through the `sfmtool._sfmtool` bindings.

- `src/sfmtool/_commands/embed_patches.py` — the Click command (argument
  parsing, validation, default-output derivation, image load, write-out). The
  image decode runs in a thread pool (cv2 releases the GIL), preserving
  `image_names` order.
- `src/sfmtool/_embed_patches.py::embed_patches` — the orchestration: a single
  `SfmrReconstruction.to_embedded_patches` bridge (the only `.sift` read) followed
  by the Rust patch kernels exposed on `PatchCloud`, run over the embedded recon
  (`refine_normals(use_stored_keypoints=True)` → `select_views` →
  `localize_keypoints`) and the `compact_to_embedded_patches` write tail. The
  cloud is read from the embedded recon's stored frames (`recon.patches`) and the
  image hashes from `recon.image_file_hashes`, both set by the bridge — no second
  `.sift` read. See the [pipeline
  spec](../../core/patch/sift-to-patch-reconstruction.md) for where the hot loops live in
  `sfmtool-core`.
- `src/sfmtool/_patch_compaction.py` — the write/compaction tail
  (`compact_to_embedded_patches`, shared with the `xform localize-keypoints` op)
  plus the `image_file_hashes_from_sift` / `image_file_hashes_from_images`
  identity-hash helpers (available standalone; the orchestration itself sources
  hashes from the embedded recon, not these).
- `src/sfmtool/_progress.py` — the `_timed_step` / `_poll_progress` progress
  helpers the orchestration wraps each Rust pass in.

## Non-goals

- **A normal from the patch's pieces.** Each round's normal step is the
  photometric search; the command does not fit the depths of the patch's
  pieces and take the plane through them. A normal from gated grid pieces is
  proposed in [piece-gated-grid-normal.md](../../drafts/piece-gated-grid-normal.md).
- **Two patch resolutions.** Every round renders every patch at one grid
  resolution `R`; no round runs on a coarse grid and then moves the patches
  that can support it to a finer one. A coarse and a fine tier over the same
  half-extent are proposed in [two-tier-patch-density.md](../../drafts/two-tier-patch-density.md).

## Open questions

- The discard gates (`--min-relative-zncc`, `--max-shift-px`) want tuning across
  the datasets before the defaults are fixed.
- v1 always (re)builds the patch frame for simplicity; should reusing an
  already-present frame (skipping the rebuild) be offered later as an
  optimization?
