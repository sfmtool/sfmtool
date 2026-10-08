# Patch Footprint Selection

**Status:** Draft. Decided:
- the size of a patch in the world, its half-extent, is chosen per track from a short ladder of sizes rather than fixed at one multiple of the detector's scale, and the chosen size is the smallest on the ladder that the track's own readings call good;
- three readings decide it, all of them kernels that exist: the ZNCC self-similarity radius of the patch and of its nine cells, the member coherence ZNCC across the track's views, and the blur-matched sharpness of each view's tile;
- the thresholds are derived from the distribution of those readings over the capture, not fixed constants;
- the primitive reports the chosen half-extent together with the footprint it covers in the reference view's photograph pixels, because that number decides whether a finer bitmap is honest for the track.

Not decided: the ladder's rungs; whether the sweep runs on the coarse tier only or re-checks at the fine tier; whether a track may change size after it is first embedded. See [Open questions](#open-questions).

Amends:
- [core/patch/patch-cloud.md](../core/patch/patch-cloud.md): how a patch's half-extent is set
- [core/patch/zncc-self-similarity-radius.md](../core/patch/zncc-self-similarity-radius.md): a consumer that reads the whole, middle and cell radii across sizes
- [core/patch/member-coherence-validation.md](../core/patch/member-coherence-validation.md) and [core/patch/blur-matched-zncc.md](../core/patch/blur-matched-zncc.md): consumers
- [core/patch/sift-to-patch-reconstruction.md](../core/patch/sift-to-patch-reconstruction.md): the size rule the embedding starts from

Related drafts: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) reads the same per-cell radii; [two-tier-patch-density.md](two-tier-patch-density.md) uses the reported footprint in photograph pixels to decide whether a track may be upgraded.

## Purpose

A patch is a small oriented square in the world, and its size decides what the patch's bitmap contains. Too small and the bitmap holds too little texture to be found again in another photograph: it slides over itself and matches in several places. Too large and the square reaches past the surface it sits on, across a depth edge or round a corner, and the views stop agreeing because they are no longer looking at one plane. Between those limits there is a range of sizes that work, and the smallest size in that range is the one to want, because a smaller patch is cheaper to resample, stays planar more often, and resolves finer structure.

The project sizes every patch at a fixed multiple of the SIFT detector's scale. A sweep over that multiple found an elbow near twelve times the scale, with improvements on eleven of the measured captures, and that is the current default. One multiple for every track is a compromise: a patch on a brick wall could be smaller, a patch on a smooth painted surface needs to be larger, and the detector's scale knows nothing about either.

This draft proposes a primitive that tries a short ladder of sizes for a track and picks the smallest good one, using readings the project already computes for other purposes.

### Why this matters for the seed

The seed stage evaluates up to eight candidate reconstructions per capture at small patch sizes, where everything is cheap. Scored against the approved ground truths, the seed finds a correct candidate in 7 of 8 captures and picks it in 4 of 8. Scoring those candidates photometrically would be only as good as the patches it reads, and a patch sized wrong scores a correct candidate badly. Per-track sizing is a precondition for trusting such a score.

## Rust API

The primitive will live in `sfmtool-core`'s `patch` module beside the self-similarity reading, bound as a `PatchCloud` method.

```rust
/// The sizes a track may take, as multiples of the detector's scale.
pub struct FootprintLadder {
    pub multiples: Vec<f32>,   // ascending, e.g. [6, 9, 12, 18, 24]
}

/// Why a size was called good or not.
pub struct FootprintReading {
    pub multiple: f32,
    pub radius_px: f32,            // whole-patch self-similarity radius
    pub cell_radii_px: [[f32; 3]; 3],
    pub coherence_zncc: f32,       // median member coherence
    pub sharpness_px: f32,         // reference view's blur-matched sharpness
    pub footprint_ref_px: f32,     // edge of the patch in the reference view's photograph
}

pub struct FootprintChoice {
    pub chosen: Option<usize>,     // index into the ladder; None when no rung is good
    pub readings: Vec<FootprintReading>,
}

impl PatchCloud {
    pub fn select_footprints(
        &self,
        views: &[ProjectedImage],
        ladder: &FootprintLadder,
        gates: &FootprintGates,      // derived from the capture; see Theory
    ) -> Vec<FootprintChoice>;
}
```

**Why this shape.** A ladder rather than a continuous search, because every stored bitmap, the consensus basis and the display assume all patches of a reconstruction share one grid, and a small set of allowed half-extents keeps that true per rung. Every reading is returned, not only the choice, because the choice rule will be tuned by looking at readings across a capture, and because [two-tier-patch-density.md](two-tier-patch-density.md) reads `footprint_ref_px` from the same sweep. The gates arrive as an argument so that one capture-level pass derives them and every track reads the same ones.

**Example.** The embedding step sweeps the ladder for every track at the coarse tier, derives the gates from the distribution of the readings, picks a size per track, and only then renders the bitmaps that are stored.

## Theory

### The three readings and what each bounds

**The self-similarity radius bounds the size from below.** A patch that slides over itself and matches cannot be localised. The radius falls as the patch grows, because more texture enters the square, until it reaches the floor set by the texture's own scale. The smallest size whose whole-patch radius and middle radius pass the gate, and whose cell radii pass on enough cells to fix a normal, is the first candidate. The cells matter because the piece-gated normal estimator needs them; a size whose whole patch passes while most cells fail gives a localisable patch with no usable normal.

**Member coherence bounds the size from above.** The views of one patch agree while the patch covers one plane. As the patch grows past a depth edge or round a corner, the resampled tiles of the views disagree where the surface departs from the plane, and the median pairwise ZNCC across the track falls. The coherence reading at each size, compared with its value at the smallest size that passes the radius gate, shows where the plateau ends.

**Blur-matched sharpness bounds the density, not the size.** Each view's tile has a sharpness; a view cannot show detail finer than its blur. The sharpness does not change the right half-extent, but it tells the density tier whether sampling the patch finer would read anything. The primitive reports the reference view's sharpness so the tier can read it without a second render.

### The choice rule

The chosen rung is the smallest on the ladder where the radius gate passes on the whole, the middle and at least `min_cells` cells, and the coherence is within `coherence_drop` of the maximum coherence over the ladder. The second clause stops the rule choosing a size that is localisable but already past the plane; the first stops it choosing a size that is planar but not findable.

### Gates from the capture, not constants

The gate on the radius and the allowed coherence drop are read from the distribution over the capture at the coarse tier: the radius gate at a quantile of the whole-patch radii at the default size, the coherence drop at a quantile of the per-track drop between adjacent rungs. A capture of a smooth building and a capture of a gravel path have different distributions and should not share constants. The project's experience with fixed thresholds elsewhere is the reason.

### Footprint in reference-view pixels

The patch's edge, projected into the reference view, covers some number of photograph pixels. A bitmap of `R` grid pixels over a footprint of fewer than `R` photograph pixels is interpolation, not data. The primitive reports the footprint so that the density tier can refuse to upgrade a track whose footprint is under its target `R`, and so that the sweep's own readings at a size are known to be honest.

## Implementation notes

- The sweep renders each view's tile at every rung. At the coarse tier, `R = 12`, a five-rung ladder is five renders per view per track, which is cheap. At the fine tier it is not, which is why the sweep belongs to the coarse tier.
- The self-similarity radii and the coherence matrix are the existing kernels called on the rendered tile; no new photometric kernel is needed.
- A per-track half-extent makes the stored patches of one reconstruction differ in world size while sharing one `R`. Nothing in the bitmap storage assumes equal half-extents; the display scales each patch by its own placement.

## Determinism and precision

Readings are `f32` on tile intensities, as the kernels they call are. The choice is a discrete rung and must be the same for any thread count. Gate derivation from quantiles is a sort over the capture's readings, in `f64`, with ties broken by track index.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `multiples` | to be measured; around `[6, 9, 12, 18, 24]` | The ladder of half-extents as multiples of the detector scale. |
| `min_cells` | 5 | Cells that must pass the radius gate for a rung to count as good. |
| `radius_gate_quantile` | to be measured | Quantile of the capture's whole-patch radii at the default size that sets the gate. |
| `coherence_drop_quantile` | to be measured | Quantile of per-track coherence drops that sets the allowed drop. |

## Testing

A synthetic planar texture with a known repeat length: the chosen rung is the first whose half-extent exceeds the repeat. A planar texture meeting a second plane at a known distance from the patch centre: the chosen rung stops before the half-extent reaches the edge. A texture-free patch: no rung is good, `chosen` is `None`. On the checked-in ground truths for `seoul_bull_sculpture` and `kerry_park`, the per-track choices must not raise the median normal error or the keypoint localisation residual against the fixed-multiple default, measured on a fixed observation set.

## Non-goals

Changing the detector's scale. The ladder is relative to it, and the detector is unchanged.

Anisotropic patches. The patch stays square; a surface whose good extent differs along two directions gets the smaller.

## Open questions

- **The rungs.** Geometric spacing around the current default of twelve is the natural start; whether anything above 24 is ever chosen decides the top of the ladder.
- **Re-check at the fine tier.** Upgrading to `R = 24` reveals texture the coarse tier could not see, which can only lower the radius. Whether that justifies a second sweep is a cost question.
- **Changing size after embedding.** A track that later gains or loses views could want a different size. Fixing the size at embedding is simpler and is the proposal; a resize is a re-embed.
