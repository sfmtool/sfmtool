# `sfm embed-patches` Command

_Status: implemented (`src/sfmtool/_commands/embed_patches.py`, orchestration in
`src/sfmtool/_embed_patches.py::embed_patches`). Converts a reconstruction from
`sift_files` to `embedded_patches` (the modes are specified in
[sfmr-file-format.md](../formats/sfmr-file-format.md), "Observation source").
Pipeline and the per-point algorithms it calls:
[sift-to-patch-reconstruction.md](../core/sift-to-patch-reconstruction.md)._

> _**Planned (2026-06-25):** `embed-patches` remains the `sift_files` entry
> point. Its first pipeline step becomes a single call to the Rust
> `SfmrReconstruction.to_embedded_patches` binding (the sole sift-consuming
> step — it reads the `.sift` files for keypoints, frames, and image hashes);
> normal refinement, view selection, and keypoint localization then run as
> `embedded_patches → embedded_patches` steps on its result. See the operating
> contract in
> [sift-to-patch-reconstruction.md](../core/sift-to-patch-reconstruction.md) and
> the design lock in `reports/2026-06-25-embedded-patches-precondition-plan.md`._

## Overview

Convert a `sift_files` reconstruction into an `embedded_patches` `.sfmr` — a
wholesale switch of `feature_source`. Each observation's reference into an
external `.sift` file is replaced by an inline, patch-derived 2D keypoint, and a
**new** `.sfmr` is written that needs no `.sift` companion. The input file is
never modified.

The conversion:

1. **Build a patch frame.** A keypoint anchors the point's surfel, so each point
   needs a `(u, v)` frame. The command initializes each frame (normal from the
   mean viewing direction) and refines that normal photometrically (the
   `refine-normals` machinery, with `save_patches`).
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
| `--patch-size` | float | inherits `refine-normals`' `extent_value` | Surfel size — the full patch edge length (in feature-size multiples) used to render the patch. Halved to the library half-extent, matching `refine-normals`. |

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
- **Track thresholds.** The only point-level cull is support count: a point left
  with fewer than `--min-views` kept views is dropped whole.
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
- `src/sfmtool/_embed_patches.py::embed_patches` — the orchestration: it chains the
  Rust patch kernels exposed on `PatchCloud` (`from_reconstruction` →
  `refine_normals` → `select_views` → `localize_keypoints`) and the
  `compact_to_embedded_patches` write tail. `image_file_hashes_from_sift` reads the
  per-image identity hashes from the `.sift` metadata. See the [pipeline
  spec](../core/sift-to-patch-reconstruction.md) for where the hot loops live in
  `sfmtool-core`.

## Open questions

- The discard gates (`--min-relative-zncc`, `--max-shift-px`) want tuning across
  the datasets before the defaults are fixed.
- v1 always (re)builds the patch frame for simplicity; should reusing an
  already-present frame (skipping the rebuild) be offered later as an
  optimization?
