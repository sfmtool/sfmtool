# Two-Tier Patch Density

**Status:** Draft. Decided:
- a patch is sampled at one of two densities: a coarse tier of `R = 12` grid pixels per edge and a fine tier of `R = 24`, over the same half-extent in the world, so that upgrading a patch changes how finely its surface is read and not which surface it reads;
- the coarse tier is where variety is cheap: candidate evaluation, the anchor floor and the relaxation stages run there;
- the fine tier is where precision is bought: a survivor is upgraded, its normal re-estimated, its keypoints re-localised and only then bundle-adjusted;
- a track is upgraded only when its footprint in the reference view's photograph covers at least `R_fine` pixels and its reference view's sharpness supports the finer grid; otherwise it stays coarse and the file says so;
- the upgrade is one ordered step, resample then normal then keypoints then adjustment, and no consumer reads a fine bitmap whose keypoints were localised on the coarse one;
- the footprint is fixed by the anchor floor of [patch-footprint-selection.md](patch-footprint-selection.md) and the surface step of [surface-footprint-analysis.md](surface-footprint-analysis.md), and the tier only chooses the density at which that footprint is read.

Not decided: whether a reconstruction stores both tiers or replaces the coarse bitmap on upgrade; whether the cluster-patches file carries a second size; whether `R_fine` is always double or chosen from the footprint. See [Open questions](#open-questions).

Amends:
- [formats/sfmr-file-format.md](../formats/sfmr-file-format.md): the patch resolution, currently one per file
- [formats/matches-file-format.md](../formats/matches-file-format.md): `patch_size` of the cluster-patches file
- [core/patch/patch-cloud.md](../core/patch/patch-cloud.md): the resolution a cloud is rendered at
- [cli/reconstruction/embed-patches-command.md](../cli/reconstruction/embed-patches-command.md): the rounds

Amended by [surface-footprint-analysis.md](surface-footprint-analysis.md), which chooses the footprint the two tiers share.

Related drafts: [sharper-patch-bitmap.md](sharper-patch-bitmap.md) decides what the stored bitmap is and how views are scored against it; this draft decides at what density and does not restate that. [patch-footprint-selection.md](patch-footprint-selection.md) supplies each anchor's floor, and [surface-footprint-analysis.md](surface-footprint-analysis.md) the footprint above it, in reference-view photograph pixels. [piece-gated-grid-normal.md](piece-gated-grid-normal.md) is the normal step of the upgrade and needs the fine tier for its per-cell radii.

## Purpose

A patch's bitmap is a square of `R × R` samples of the surface the patch covers. The number `R` is one value for a whole reconstruction, and every operation that renders, compares or localises patches pays for it: the cost of resampling a view, of a ZNCC, of a self-similarity sweep, and of the search cache all grow with `R²`. The project's fleet of cluster-patches files is at `R = 12`, and the ground-truth reconstructions it checks against are at `R = 24`.

Those two numbers are not in tension. They serve different stages. Early in a reconstruction's life the question is which of many candidates is right, and a cheap reading over all of them beats a precise reading over one. Late in its life the question is how precisely the survivor can be placed, and a finer reading of its surfaces is where precision comes from. This draft makes that split explicit: a coarse tier for breadth and a fine tier for depth, with a defined step between them.

Fixing the half-extent across the two tiers is what makes the step a resampling rather than a re-embedding. A finer grid over the same square reads the same surface in more detail. A larger square at the same grid would read a different surface, with the planarity hazards that choosing the footprint for a whole surface, in [surface-footprint-analysis.md](surface-footprint-analysis.md), handles.

### Why this matters for the seed

The seed stage evaluates up to eight candidates per capture. Scored against the approved ground truths, it finds a correct candidate in 7 of 8 captures and picks it in 4 of 8. Photometric scoring of the candidates, which is what the choice lacks, is affordable across all eight only at the coarse tier. The survivor is then the one track set worth paying the fine tier for.

## The upgrade step

The step is defined as an order, because each stage invalidates the one after it:

1. **Resample.** Each view is rendered into the `R_fine` grid over the unchanged half-extent, with the sampler [sharper-patch-bitmap.md](sharper-patch-bitmap.md) chooses per view.
2. **Normal.** The piece-gated grid estimator runs on the fine tiles, with the coarse normal as its prior. Only at this tier do its per-cell radii carry data.
3. **Keypoints.** The keypoints are re-localised on the fine bitmap with the new normal. A keypoint localised on the coarse bitmap is stale after the normal moves, and the project has already paid once for reading stale keypoints into an adjustment.
4. **Adjustment.** Bundle adjustment reads the re-localised keypoints.

A caller that performs stages 1 and 2 and skips 3 produces a reconstruction whose observations do not match its bitmaps. The file does not forbid that, but the step as a unit does, and the embedding command runs the unit.

### Refusing the upgrade

A track is upgraded only when its footprint, as the surface step sets it, covers in its reference view at least `R_fine` photograph pixels across the patch, and the reference view's blur-matched sharpness is finer than the coarse grid's spacing. A distant or blurred patch fails both and stays at `R = 12`; rendering it at 24 would interpolate, and a comparison against an interpolated bitmap rewards views that are equally blurred. The track's tier is recorded so a consumer knows which bitmaps are data.

### The coarse normal as a prior with a weight

The coarse normal seeds the fine estimate but does not bind it. The fine estimator can overrule it, because a coarse mistake on a low-texture patch, locked in as a hard start, would never be corrected. The prior enters as the fallback for an axis the fine pieces do not fix, per the determinacy verdict, and not as a term in the fit.

## Theory

### Cost

A render, a ZNCC and a self-similarity sweep are `O(R²)` per view. The search cache is `O(R²)` in memory per track. Going from 12 to 24 is a factor of four in all of them. A seed evaluation over eight candidates at the fine tier would cost as much as thirty-two at the coarse tier, which is the budget the coarse tier spends on variety instead.

### Why the half-extent is fixed

The patch-size sweep in the project's records placed the elbow of quality against size near twelve times the detector scale, and the hand-set sizes of the ground truths sit near a 12 px half-extent in the reference photograph. [patch-footprint-selection.md](patch-footprint-selection.md) sets a floor on the size per anchor and [surface-footprint-analysis.md](surface-footprint-analysis.md) chooses the size above it per surface. That choice is about which surface the patch covers. Density is about how finely that surface is read. Coupling them, so that an upgrade also widens the square, would reintroduce the planarity failures the size choice avoided and make every fine-tier reading incomparable with the coarse one it replaced.

### Evaluating on fixed populations

The fine tier can change which members survive a gate, because sharper bitmaps refuse members the coarse ones admitted. A comparison of lens or pose quality between tiers must be made on the observation set both tiers accept, or the gain from sharpness is confounded with the loss of members. This is the same rule the project applies to lens-state comparisons.

## Format

Two tiers mean two patch resolutions can exist for one reconstruction. The options:

- **Replace on upgrade.** The file holds one `R`. Upgrading a reconstruction rewrites its bitmaps at 24 and its unupgraded tracks are rendered at 24 anyway, interpolated, with a per-track flag saying so. Simplest; loses the honest coarse bitmap for refused tracks.
- **Two resolutions per file.** Each track records its own `R` from the pair. Bitmap storage becomes ragged. Consumers that assume one grid, such as the display, need a per-track `R`.
- **Two files.** The coarse product and the fine product are separate reconstructions with the same tracks. Clean for the seed, which already writes candidate files, and awkward for a single edited reconstruction.

The cluster-patches file has the same question at `patch_size`. The fleet refresh standardised on 12, and the file is a precursor the seed reads once, so the proposal is that the cluster-patches file stays at one size and the fine tier is only ever a property of a reconstruction.

## Implementation notes

- `R` is read in many places as a file-level constant. Before any per-track `R`, a grep for every reader of the patch resolution is the first task, so the format decision is made knowing the cost.
- The resample of stage 1 must use the per-view sampler choice from the sharper-bitmap work, not the mip sampler alone; at 24 the anisotropy that choice handles is twice as visible.
- The seed's hypothesis releases carry no patch frames today. For the seed to hand a survivor to the fine tier, the release must carry frames, which a photometric score of the candidates would require anyway.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `R_coarse` | 12 | Grid edge of the coarse tier. |
| `R_fine` | 24 | Grid edge of the fine tier. |
| `min_footprint_ref_px` | `R_fine` | A track whose reference-view footprint is narrower is not upgraded. |

## Testing

A track upgraded and then downgraded by rendering at 12 again gives bitmaps equal within sampler tolerance to the original. A track whose footprint is under `R_fine` is refused and its tier reads coarse. On the `kerry_park` and `seoul_bull_sculpture` ground truths, keypoint residuals on the fixed observation set accepted by both tiers are no worse at the fine tier than at the coarse, and the normal error is lower.

## Non-goals

More than two tiers. Two cover the breadth-then-depth split; a third would need a use the first two do not serve.

Changing the half-extent on upgrade. A patch that needs a different size is re-embedded at the footprint the surface step chooses.

## Open questions

- **Format choice.** Replace, two resolutions, or two files. The grep over readers of `R` decides what the second option costs.
- **Always double.** `R_fine = 2 R_coarse` keeps the resample an exact subdivision; choosing `R_fine` from the footprint would read more where the photographs allow it. Start with double.
- **Partial upgrade.** Whether a reconstruction with some tracks refused is one tier or two for the purposes of hashing and display.
