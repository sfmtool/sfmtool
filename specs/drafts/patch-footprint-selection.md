# Patch Footprint Selection

**Status:** Draft. Decided:
- per-point selection of a patch's size by a reading is not the rule the hand-set sizes of the two ground truths follow: no per-point rule tried beat a constant 12 px half-extent across captures ([the drafted gates](patch-footprint-selection-measurements.md#whether-the-readings-predict-the-humans-size), [a rule search](patch-footprint-selection-measurements.md#a-rule-for-the-hand-set-sizes-2026-10-09), [the nine cell radii](patch-footprint-selection-measurements.md#the-nine-cell-radii-against-the-hand-set-sizes-2026-10-09), [the cell-resolve variants](patch-footprint-selection-measurements.md#the-smallest-footprint-at-which-the-cells-resolve-2026-10-09), [a feature quota](patch-footprint-selection-measurements.md#a-quota-of-distinct-features-against-the-hand-set-sizes-2026-10-09));
- the per-point primitive is an anchor floor: the smallest footprint, on a ladder of half-extents in reference-view photograph pixels read at native density, at which all nine cells of the patch pass the self-similarity bar (the cell-resolve rule v1). It is used to place anchors that resolve a normal and an affine shape, not to set the size a patch finally takes;
- the footprint above the floor is chosen per surface, in a later step proposed in [surface-footprint-analysis.md](surface-footprint-analysis.md).

Not decided: the surface step's inputs, which that draft leaves open. See [Open questions](#open-questions).

Amends:
- [core/patch/patch-cloud.md](../core/patch/patch-cloud.md): a floor below which a patch's half-extent is not set
- [core/patch/zncc-self-similarity-radius.md](../core/patch/zncc-self-similarity-radius.md): a consumer that reads the nine cell radii across a ladder of footprints
- [core/patch/sift-to-patch-reconstruction.md](../core/patch/sift-to-patch-reconstruction.md): the size rule the embedding starts from

Amended by [surface-footprint-analysis.md](surface-footprint-analysis.md), which chooses the footprint above the floor per surface.

Related drafts: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) reads the same per-cell radii to estimate a normal; [two-tier-patch-density.md](two-tier-patch-density.md) chooses the density over a footprint that this draft and the surface step fix. The measurements behind this draft are in [patch-footprint-selection-measurements.md](patch-footprint-selection-measurements.md).

## Purpose

An anchor is a patch placed to localise a point and to resolve its normal and its affine shape across the views. Both depend on the patch's cells: the cell plane normal and the piece-gated grid normal fit a plane through the nine cells of a three-by-three split, and an affine shape is fixed by cells that localise at spread positions. A cell that slides over itself and matches in several places contributes nothing to either. This draft proposes a primitive that finds, for each point, the smallest footprint at which all nine cells localise: the **anchor floor**. Below it the patch cannot resolve a normal on both axes; at it, the cells can.

The floor is not the size the patch should finally have. Measured against the half-extents the maintainer set by hand on the two checked-in ground truths, no per-point reading reproduced the hand-set sizes better than a constant 12 px half-extent across captures: not the self-similarity, coherence and sharpness gates this draft first proposed, not five families of rules over footprint and scale band, not a learned model on 52 per-point readings, not the cell-resolve variants and not a quota of distinct features. What the hand-set sizes do follow is the surface: a point's nearest neighbour on the same surface has a size within a factor of 1.25 on 73% (`seoul_bull`) and 84% (`kerry_park`) of the points that have such a neighbour, while sizes differ by a factor of two between surfaces. The smallest footprint at which all nine cells pass is a floor the human exceeds, by a median factor of 1.37 and 1.51 where the human is above it. So the per-point question is only where an anchor can resolve a normal and an affine, and the footprint a patch takes above that is decided for a surface as a whole.

## Rust API

The primitive will live in `sfmtool-core`'s `patch` module beside the self-similarity reading, bound as a `PatchCloud` method.

```rust
/// Footprints to try, as half-extents in the reference view's photograph pixels.
pub struct FootprintLadder {
    pub half_extents_px: Vec<f32>,  // ascending, e.g. [3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 30, 40]
}

pub struct AnchorFloorOptions {
    pub cell_bar: f32,       // 2.5 grid samples, the bench's self-similarity bar
    pub max_radius: u32,     // 3, the self-similarity search's cap
    pub min_resolution: u32, // 6, the smallest native grid
    pub max_resolution: u32, // 80, the largest native grid
}

pub struct AnchorFloor {
    pub rung: Option<usize>,        // index into the ladder; None when no rung resolves
    pub half_extent_px: f32,        // the rung's half-extent; NaN when rung is None
    pub half_extent_world: f64,     // the same footprint in the world; NaN when rung is None
    pub cell_radii: [[f32; 3]; 3],  // the nine radii at the rung, in grid samples, row-major
}

impl PatchCloud {
    pub fn anchor_footprint_floor(
        &self,
        views: &[ProjectedImage],
        ladder: &FootprintLadder,
        options: &AnchorFloorOptions,
    ) -> Vec<AnchorFloor>;
}
```

**Why this shape.** The ladder is in reference-view photograph pixels because the readings that decide the floor are read on the reference view's tile, and because the hand-set sizes follow a footprint in the photograph rather than the detector's scale ([below](#why-the-ladder-is-in-reference-pixels)). The output is the floor and the nine radii at it, not a final size: the surface step reads the floors of a group's anchors, and the radii tell it which cells were near the bar. The radii at the other rungs are not returned, since no consumer reads them; a measurement that needs them calls the self-similarity kernel directly. `None` is a result, not an error: a patch whose only structure is one straight edge never resolves, and the surface step has to know that.

**Example.** The embedding step computes the floor for every track before rendering stored bitmaps, places anchors only where the floor exists, and hands the anchors with their normals to the surface step, which sets one half-extent per surface at or above its anchors' floors.

## Theory

### The floor

The floor is the smallest rung at which every one of the nine cells of the reference view's tile has a ZNCC self-similarity radius at or under the bar of 2.5 grid samples, with the search capped at 3. A cell at the cap does not localise at all. Requiring all nine, rather than a count, is what makes the floor a statement about resolving a normal and an affine: with all nine localising, the cells span the square in both directions, and a plane and an affine map through them are fixed on both axes. A count of five can be met by five cells along one edge.

On the two ground truths the floor exists for 98% (`seoul_bull`) and 93% (`kerry_park`) of points. The hand-set size is at or above it on 70% and 61% of those, by a median factor of 1.37 and 1.51 where it is above, and under it on the rest by a median factor of 0.71 and 0.65 ([measurement](patch-footprint-selection-measurements.md#the-smallest-footprint-at-which-the-cells-resolve-2026-10-09)). The points under it are mostly small features the human sized tightly: a stripe crossing on the sculpture's coils, distant buildings and treelines, the sculpture's edge against the lawn, where some cells stay capped until the footprint takes in a second surface. Of the readings of "resolve" measured, this one is the tightest floor that holds on most points. Requiring every cell to localise within a quarter of its own side (v2) holds on fewer, and "the count of passing cells stops rising" (v5) holds on 94% to 97% but sits a median 2.3 to 2.6 times below the human, at about 4 px, which says little.

The whole-tile radius is not part of the rule: at native density it passes on 98% to 100% of points at every multiple of the human's size from 0.5× to 2×, so it does not separate one rung from another ([measurement](patch-footprint-selection-measurements.md#the-nine-cell-radii-against-the-hand-set-sizes-2026-10-09)).

### Why native density

Each rung is rendered at `R = round(2·h)` grid samples for a half-extent of `h` reference px, clamped to `min_resolution..max_resolution`, so one sample covers about one photograph pixel. A fixed grid gives a different floor. At `R = 24` a small footprint is oversampled, the cells hold interpolated samples, and they pass at footprints the photograph cannot resolve: the `R = 24` floor falls at 5 to 8 px, the human is above it by a median factor of 1.79 and 2.15, and as a predictor of the hand-set size it scores worse than the native floor on both captures ([measurement](patch-footprint-selection-measurements.md#the-smallest-footprint-at-which-the-cells-resolve-2026-10-09)). In the other direction, a grid coarser than the photograph averages fine texture away; on these captures that case is rare, 12 of 380 `kerry_park` points and none on `seoul_bull` ([measurement](patch-footprint-selection-measurements.md#readings-at-two-densities)), but it would grow with the top of the ladder. Native density removes both.

### Why the ladder is in reference pixels

The hand-set half-extent in photograph pixels does not follow the detector's scale `σ`: a regression of log half-extent in px on log `σ` has a slope of 0.11 and 0.02, where a fixed multiple of `σ` would give 1, and the spread of the half-extent in px is smaller than the spread of its ratio to `σ` ([measurement](patch-footprint-selection-measurements.md#what-the-human-chose)). The fleet's cluster patches, sized at a multiple of `σ`, are more than about 1.5 times away from the hand-set size on half of the matched clusters ([measurement](patch-footprint-selection-measurements.md#clusters-against-the-humans-size)). A ladder in multiples of `σ` spends its rungs on the variation of `σ`; a ladder in reference px spends them on the footprint. The ground-truth keypoints are mostly not SIFT detections, so the figures on `σ` are taken on the minority of points with a SIFT keypoint within 2 px.

### Why member coherence and sharpness are not part of it

The first form of this draft bounded the size from above by member coherence and reported the reference view's blur-matched sharpness. Neither belongs in the floor:

- **Coherence.** Per point, the coherence plateau does not cap the size within a factor of two: on most points coherence stays within the gate up to twice the hand-set size ([measurement](patch-footprint-selection-measurements.md#whether-the-readings-predict-the-humans-size)). Over a grid of footprint and scale band, the best first-passing rule without a coherence bar scores the same as with one ([measurement](patch-footprint-selection-measurements.md#the-rule-families)). Coherence also needs every view rendered at every rung, where the floor needs only the reference view.
- **Sharpness.** `assess_blur`'s semi-axes on a tile are that tile's own self-similarity ellipse, so its major axis equals the whole-tile radius on every tile measured and adds nothing as a size reading ([measurement](patch-footprint-selection-measurements.md#whether-the-readings-predict-the-humans-size)). It bounds the density a footprint can be read at, which [two-tier-patch-density.md](two-tier-patch-density.md) uses, not the footprint.

## Implementation notes

- Only the reference view is rendered, one tile per rung. With the ladder above, the tiles hold about 17,600 samples in all per track, under three times one `R = 80` tile.
- At native density the samples of a smaller rung are the central samples of a larger rung's tile, since every rung's grid is the same affine map of the patch plane at one sample per reference pixel. Rendering the top rung once and cropping it is then equal to rendering each rung, provided the sampler chooses the same pyramid level for both; a test checks that the two agree before the crop is used.
- The cell radii come from `zncc_self_similarity_parts` on the rung's tile, with the cell split it already uses. No new photometric kernel is needed.
- The sweep stops at the first rung that resolves.

## Determinism and precision

Readings are `f32` on tile intensities, as the kernel they call is. The floor is a discrete rung and must be the same for any thread count.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `half_extents_px` | `[3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 30, 40]` | The ladder, in reference-view photograph px; the ladder the measurements used. |
| `cell_bar` | 2.5 grid samples | A cell passes when its self-similarity radius is at or under this. The bench's bar. |
| `max_radius` | 3 | The self-similarity search's cap; a cell at it does not localise. |
| `min_resolution`, `max_resolution` | 6, 80 | The clamp on the native grid `R = round(2·h)`. |

## Testing

A random texture at the scale of one photograph pixel: the floor is the smallest rung. The same texture blurred by a Gaussian of growing width: the floor rises with the width and never falls. A uniform tile: no rung resolves and `rung` is `None`. A tile whose only structure is one straight edge: no rung resolves. The crop of a top-rung render equals the render of the smaller rung. On the checked-in ground truths for `seoul_bull_sculpture` and `kerry_park`, the floor exists for the measured shares of points (98% and 93%) and lies at or under the hand-set half-extent on the measured shares (70% and 61%), within a few points.

## Non-goals

Choosing the footprint a patch finally takes. That is a property of the surface the patch lies on and is proposed in [surface-footprint-analysis.md](surface-footprint-analysis.md).

Changing the detector's scale. The ladder does not refer to it.

Anisotropic patches. The patch stays square.

## Open questions

- **The surface step's inputs.** Which readings set a surface's footprint above its anchors' floors is open in [surface-footprint-analysis.md](surface-footprint-analysis.md).
- **Where the human went under the floor.** On 30% and 39% of the points with a floor, the hand-set size is under it, mostly small features whose cells stay capped until the footprint crosses onto a second surface. Whether such a point gets no anchor, an anchor at the floor that the surface step later shrinks, or an anchor whose capped cells are left out of the normal, is open.
- **Reference view only.** The floor is read on the reference view. Whether a view at a grazing angle or at a different zoom should be able to raise it is open.
