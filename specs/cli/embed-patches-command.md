# `sfm embed-patches` Command

_Status: implemented (`src/sfmtool/_commands/embed_patches.py`, orchestration in
`src/sfmtool/_embed_patches.py::embed_patches`). Converts a reconstruction from
`sift_files` to `embedded_patches` (the modes are specified in
[sfmr-file-format.md](../formats/sfmr-file-format.md), "Observation source").
Pipeline and the per-point algorithms it calls:
[sift-to-patch-reconstruction.md](../core/sift-to-patch-reconstruction.md)._

> _**Re-layered (2026-06-25):** `embed-patches` remains the `sift_files` entry
> point, but its first pipeline step is now a single call to the Rust
> `SfmrReconstruction.to_embedded_patches` binding (the sole sift-consuming
> step — it reads the `.sift` files for keypoints, frames, and image hashes);
> normal refinement (anchored on the stored keypoints via
> `use_stored_keypoints`), view selection, and keypoint localization then run as
> `embedded_patches → embedded_patches` steps on its result. See the operating
> contract in
> [sift-to-patch-reconstruction.md](../core/sift-to-patch-reconstruction.md)._

## Overview

Convert a `sift_files` reconstruction into an `embedded_patches` `.sfmr` — a
wholesale switch of `feature_source`. Each observation's reference into an
external `.sift` file is replaced by an inline, patch-derived 2D keypoint, and a
**new** `.sfmr` is written that needs no `.sift` companion. The input file is
never modified.

The conversion:

1. **Build a patch frame.** A keypoint anchors the point's surfel, so each point
   needs a `(u, v)` frame. The command initializes each frame (normal from the
   mean viewing direction, via `to_embedded_patches`) and refines that normal
   photometrically (the `refine_normals` kernel, which re-persists the frame).
2. **Derive the keypoints over an expanded, vetted view set.** For each point,
   expand the track with the other views that geometrically see the surfel,
   photometrically vet them against a track-seeded template, and
   congeal — refining each view's projected keypoint with the group-wise
   sub-pixel keypoint-localization algorithm.
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
| `--min-relative-zncc` | float | 0.7 | Minimum ZNCC a view must reach, as a fraction of the reference's own agreement — used both to admit candidate views (vs the track) and to drop poorly-registering views during congealing (vs the views' median LOO ZNCC). |
| `--max-iters` | int | 5 | Max congealing rounds per point (stops early at convergence). |
| `--search` | float | 6 | Max total per-view in-plane drift, in **patch-grid** pixels. |
| `--max-shift-px` | float | 3.0 | Discard an **observation** whose keypoint sits more than this from the point's projection, in **source-image** pixels (an absolute distance, not the move from the seed). |
| `--min-views` | int | 2 | Drop a point left with fewer surviving observations after discards. |
| `--patch-size` | float | `5.0` | Surfel size — the full patch edge length (in feature-size multiples) used to render the patch. Halved to the library half-extent and passed to `to_embedded_patches` (`extent="feature_size"`), the step that builds the frame. |
| `--search-resolution-multiplier` | float | `1.0` | Multiplier `m` for the discrete cross-view search; the search runs at resolution `round(m·R)`. `1.0` is the no-op; `>1` (the supersampled grid) resolves sub-pixel offsets directly at a cost that grows ~`m²`. See [`specs/core/keypoint-localization-search-cache.md`](../core/keypoint-localization-search-cache.md). |
| `--subpixel` | int ≥ 0 | `1` | LK / ECC Gauss–Newton `max_outer_sweeps` for the photometric sub-pixel keypoint refinement (always the per-sweep consensus variant), applied once per round. `0` disables the keypoint movement (the localizer's keypoints are used as is; the final round still runs the stage render-only to fuse each point's consensus bitmap + validity at those keypoints); `N ≥ 1` runs the refiner with that many sweeps. See [`specs/core/keypoint-subpixel-refinement.md`](../core/keypoint-subpixel-refinement.md). |
| `--rounds` | int ≥ 1 | `2` | Number of (normal-refinement, keypoint-refinement) rounds, alternating the two. Round 1 runs the SIFT-anchored normal refine, the discrete localizer (the seed), then the sub-pixel keypoint refine; each subsequent round re-refines every normal against the previous round's keypoints, then re-refines the keypoints against the new normals — a fixed-point alternation. The default `2` runs one refinement pass on top of the seed round (most of the normal/keypoint convergence gain lands in the first extra round); raise it for the tail. The per-point view set (membership) is fixed after round 1 (and per-round grazing drops only shrink it). With a `progress` sink (the CLI wires `click.echo`) each round prints its mean normal change (deg) and mean keypoint shift (px). |
| `--max-obliquity-deg` | float 0–90 | `80` | After round 1, drop every observation viewing its surfel more than this many degrees off the refined normal (`\|v̂·n\| < cos θ`). A grazing view renders as a cross-view-consistent but **degenerate smear** that satisfies the consensus yet erases surface texture, so over multiple rounds it drags the normal toward grazing (a self-reinforcing failure). Dropping those views keeps the round alternation on the well-observed, near-frontal views. `90` disables the filter. The obliquity is exactly the `inspect --strips` magenta-dot radius (`sin θ`). (With the fronto prior on by default, surfels stay near-frontal, so this cut fires far less often — it is the backstop for the residual grazing observations.) |
| `--obliquity-weight-power` | float ≥ 0 | `2` | Exponent `p` of the multiplicative **obliquity view-weight** `\|v̂·n\|^p` folded into the robust normal-refinement consensus (use A). `0` disables it — the consensus runs as before. `2` (default) is the `cos²θ` foreshortening weight: it softly down-weights a view the more obliquely it sees the surfel — a continuous complement to the hard `--max-obliquity-deg` cut, on points whose views span a range of obliquities. On a low-parallax point (all views near-collinear, hence near-equal obliquity) it renormalizes away; that case is what `--fronto-prior-weight` addresses. See [`specs/core/patch-normal-refinement.md`](../core/patch-normal-refinement.md). |
| `--fronto-prior-weight` | float ≥ 0 | `0.05` | Weight `λ` of the additive **fronto-parallel prior** `λ·mean_v (v̂·n)²` on each candidate normal during refinement (use B). `0` disables it. It rewards normals that face the observing cameras, supplying the constraint the data can't when `Φ` is flat — the narrow-baseline degeneracy where every candidate tilt shifts all views' patches identically, so a low-parallax surfel drifts to a photometrically-equivalent tilt and renders distorted (a stop sign's octagon shears into a smear). The prior lands it fronto-parallel instead; wherever real parallax curves `Φ` the small prior is overruled, so well-constrained normals are unaffected. The `0.05` default (with `--obliquity-weight-power 2`) straightens low-parallax surfels at negligible photoconsistency cost. See [`specs/core/patch-normal-refinement.md`](../core/patch-normal-refinement.md). |
| `--refine-max-views` | int ≥ 0 | `8` | Cap the **round-2+ normal-refinement basis** at the `N` most normal-informative views per point — the D-optimal geometric pick of [`specs/core/patch-normal-refine-view-subset.md`](../core/patch-normal-refine-view-subset.md) (a least-oblique appearance anchor plus a greedy information-determinant fill; always the best `N`, no fall-back-to-all). `0` disables the cap (use all views). Applies only to the fine-tuning rounds, whose view set is the `select_views`-expanded one; the round-1 (raw-track) refine is untouched. **Lossless for the output**: only the refinement basis shrinks — every observation stays, and the consensus bitmaps are still fused over the full view set. The default `8` roughly halves end-to-end time vs all-views on large view sets. |
| `--localize-search-strategy` | choice | `plus_descent` | Per-(view, round) shift-grid traversal inside the keypoint localizer's `search_shift`. `plus_descent` (default) is steepest-descent on the 4 axis neighbors, scoring ~6 cells per call via an AVX2 single-position vgather kernel — ~1.9× faster end-to-end on dino at comparable accuracy (median per-observation keypoint shift vs `exhaustive` ~0.05 px, 91 % within 1 px). `exhaustive` scores the full `(2·margin+1)²` grid via the SIMD SAXPY accumulator — the global-argmax fallback, no local-optima risk. See [`specs/core/keypoint-localization-search-cache.md`](../core/keypoint-localization-search-cache.md). |

The two `--search` / `--max-shift-px` budgets are in different units on purpose:
`--search` bounds the registration in the patch's own grid (the congealing
algorithm), while the discard gate `--max-shift-px` is read back in
source-image pixels (the pipeline's quality control). See the core specs.

**Failures are discarded, not back-filled** — every keypoint written reflects a
real registration (see Behaviour Notes), so the output observation set is the
input track reshaped (expanded by vetting, trimmed by drops), not copied through.

## Behaviour Notes

- **View set.** Each point is localized over its track **plus** the other views
  that geometrically see the surfel and pass photometric vetting (based on
  `--min-relative-zncc`).
- **Per-point congealing.** Render the admitted views' patches, build the robust
  leave-one-out consensus, search each view's sub-pixel shift, iterate to
  convergence; emit `keypoint = project_i(X_p) + δ`. Views that won't co-register
  are dropped *as it goes*, so the survivors refine against a cleaner consensus.
- **Observation thresholds.** A view is dropped, in-loop, if it can't be localized
  cleanly (grazing view, out-of-frame keypoint), if its keypoint sits more than
  `--max-shift-px` from the point's projection, or if its leave-one-out ZNCC
  agreement falls below `--min-relative-zncc` of the views' median LOO ZNCC.
- **Reference bitmaps.** Each surviving point's stored bitmap is the cross-view
  **consensus texture fused in the sub-pixel keypoint-refinement stage** at the
  final per-view keypoints (`refine_keypoints(render_bitmaps=True)`; with
  `--subpixel 0` the stage still runs render-only at the localizer's keypoints).
  Points at infinity go through the same `w`-aware render path and get a real
  consensus bitmap — no zero-row exemption. (Bitmaps are no longer sourced from
  normal refinement, whose render lagged the final keypoints by one round.)
- **Track thresholds.** A point is dropped whole when its support count falls
  below `--min-views`, or when the sub-pixel stage produced **no valid consensus
  bitmap** for it (fewer than two of its views render at their final keypoints) —
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
  --max-iters 3 --search 4 --min-relative-zncc 0.75
```

## Module Layout

- `src/sfmtool/_commands/embed_patches.py` — the Click command (argument
  parsing, validation, default-output derivation, image load, write-out).
- `src/sfmtool/_embed_patches.py::embed_patches` — the orchestration: a single
  `SfmrReconstruction.to_embedded_patches` bridge (the only `.sift` read) followed
  by the Rust patch kernels exposed on `PatchCloud`, run over the embedded recon
  (`refine_normals(use_stored_keypoints=True)` → `select_views` →
  `localize_keypoints`) and the `compact_to_embedded_patches` write tail. The
  cloud is read from the embedded recon's stored frames (`recon.patches`) and the
  image hashes from `recon.image_file_hashes`, both set by the bridge — no second
  `.sift` read. (`image_file_hashes_from_sift` / `image_file_hashes_from_images`
  remain available helpers.) See the [pipeline
  spec](../core/sift-to-patch-reconstruction.md) for where the hot loops live in
  `sfmtool-core`.

## Open questions

- The discard gates (`--min-relative-zncc`, `--max-shift-px`) want tuning across
  the datasets before the defaults are fixed.
- v1 always (re)builds the patch frame for simplicity; should reusing an
  already-present frame (skipping the rebuild) be offered later as an
  optimization?
