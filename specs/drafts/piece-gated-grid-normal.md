# Piece-Gated Grid Normal

**Status:** Draft. Decided:
- the normal of a patch is estimated from the fitted centres of its pieces, where the pieces are the nine cells of the ZNCC grid the self-similarity reading already splits the bitmap into, so that each piece carries a self-similarity radius computed once and read by both;
- a piece whose own radius fails the gate the whole-patch rule applies is not fitted at all, and a piece that passes is weighted by how sharply its centre is pinned along its sighting ray;
- the estimator reports which axes of the normal the surviving pieces fix, in the form the adjacency-surfel estimator reports its determinacy verdict, and leaves an unfixed axis at its prior;
- once this estimator covers the callers of the photometric normal search, that search is retired together with the fronto-parallel cache and the D-optimal view subset that exist only to serve it.

Not decided: the weight's functional form; whether the gated pieces are fitted independently along their rays or jointly with a shared plane constraint; the overlap between pieces; what the estimator does on a patch where fewer than three pieces pass. See [Open questions](#open-questions).

Amends:
- [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal": the Grid Plane step
- [core/analysis/adjacency-surfel-normals.md](../core/analysis/adjacency-surfel-normals.md): the determinacy verdict is shared
- [core/patch/patch-normal-refinement.md](../core/patch/patch-normal-refinement.md), [core/patch/fronto-parallel-patch-cache.md](../core/patch/fronto-parallel-patch-cache.md), [core/patch/patch-normal-refine-view-subset.md](../core/patch/patch-normal-refine-view-subset.md): retired when their callers move
- [cli/reconstruction/embed-patches-command.md](../cli/reconstruction/embed-patches-command.md) and [cli/reconstruction/xform/refine-normals-command.md](../cli/reconstruction/xform/refine-normals-command.md): the normal step they run

Related drafts: [surface-co-solve.md](surface-co-solve.md) § "Normal estimators" lists the family this estimator belongs to and should defer to this draft for the grid estimator. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) supplies the per-view readings the weights use. [patch-footprint-selection.md](patch-footprint-selection.md) reads the same per-piece radii to choose the patch size. [cell-plane-normals.md](cell-plane-normals.md) fits the same plane and emits the same verdict from the cluster-patches file's stored cell displacements, with no rendering, and defines the `NormalDeterminacy` type this estimator would share.

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in the world, and its normal says which way the square faces. The normal decides how every photograph is resampled into the patch's bitmap, so a wrong normal blurs the bitmap, shifts the keypoints that are localised on it, and misleads every photometric reading taken from it afterwards.

The project has two production ways of estimating a normal. The photometric search turns the normal through two degrees of freedom and keeps the orientation at which the views agree best. The grid estimator, available in the Track View as *Grid Plane Normal*, cuts the patch into a grid of smaller pieces, fits each piece's depth along its sighting rays, and takes the plane through the fitted centres. The photometric search is weak where the texture is weak, has a median error of 16 to 23 degrees at the true position in the surface co-solve harness, and never recovers a normal far from the mean viewing direction. The grid estimator does not have that limit, because a piece's depth comes from parallax, not from the shape of a correlation peak over orientation.

The grid estimator has a different fault. It treats every piece as equally trustworthy. A piece over a flat or repeating part of the patch has a correlation that is nearly constant along its ray, so its fitted depth is arbitrary, and one arbitrary depth among nine tilts the fitted plane with no sign in the plane's residual that anything went wrong. The whole patch can score well on self-similarity while one corner or one half scores badly, which is the usual case on a depth edge, an object boundary or a patch that straddles texture and sky.

This draft proposes gating and weighting the pieces by their own self-similarity before any geometry is fitted. The self-similarity reading already splits the bitmap into the nine cells of the ZNCC grid and reports a radius for each, so the per-piece trust is a number the code already computes.

### Why this matters for the seed

The seed stage produces up to eight candidate reconstructions per capture and then chooses among them. Scored against the eight approved ground truths, a correct candidate exists in 7 of 8 captures and the seed's own rule picks it in 4 of 8. The rule reads only geometric residuals, and those cannot distinguish a right camera layout from a collapsed one at the same reprojection error. Photometric scoring of candidates is the independent signal that can, and it needs patch normals that are right often enough to be trusted.

## Rust API

The estimator will live beside the current grid estimator in [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), with the pose-free part of the piece fit shared with [cluster_refine](../../crates/sfmtool-core/src/patch/cluster_refine/mod.rs) as [cluster-patches-piecewise-refinement.md](cluster-patches-piecewise-refinement.md) proposes.

```rust
/// How the pieces of a patch are gated and weighted before the plane fit.
pub struct PieceGateParams {
    /// A piece whose self-similarity radius exceeds this, in grid px of the
    /// piece's own render, is not fitted. Defaults to the whole-patch gate.
    pub max_piece_radius_px: f32,
    /// A piece whose along-ray fit has less than this curvature at its
    /// optimum is not fitted, because its depth is not pinned.
    pub min_ray_curvature: f32,
    /// Fewer surviving pieces than this leaves the normal at its prior.
    pub min_pieces: usize,
}

/// The normal, which of its axes the pieces fixed, and why each piece was kept or dropped.
pub struct GatedNormalReport {
    pub normal: [f64; 3],
    pub determinacy: Determinacy,          // shared with adjacency-surfel-normals
    pub pieces: [[PieceVerdict; 3]; 3],
    pub plane_rms: f64,
}

pub enum PieceVerdict {
    Fitted { weight: f32, radius_px: f32, ray_curvature: f32 },
    RadiusGate { radius_px: f32 },
    CurvatureGate { ray_curvature: f32 },
    NoViews,
}

pub fn piece_gated_grid_normal(
    track: &EditableTrack,
    recon: &EditedReconstruction,
    views: &[ProjectedImage],
    gate: &PieceGateParams,
) -> Result<(EditableTrack, GatedNormalReport), NormalError>;
```

**Why this shape.** The pieces are fixed at the ZNCC grid's 3×3 split rather than a free `pieces` count, because the per-piece radius is only available at that split, and computing a second split would cost a second sweep over every view for no gain in the cases this draft is about. The report keeps every piece's verdict, because a patch whose normal is wrong is diagnosed by asking which pieces were trusted, and the Track View should show that the way it shows the reference view's per-view verdicts. The determinacy type is shared with the adjacency-surfel estimator so that a caller choosing between the two reads one vocabulary.

**Example.** The bench's *Grid Plane Normal* button calls this in place of `grid_normal`, shows the nine verdicts as a 3×3 overlay on the patch bitmap, and tilts the patch only along the fixed axes.

## Theory

### The piece gate

The self-similarity radius of a bitmap says how far the bitmap can slide over itself and still match. The reading reports it for the whole bitmap, for its middle square, and for each of the nine cells of the ZNCC grid. The whole-patch rule already refuses a member whose radius exceeds a gate, because such a member cannot be localised. A piece is a cell, and the same rule applies to it for the same reason: a cell that matches itself over a wide shift has no single depth along its ray that the views agree on.

Gating before the fit is the point. A robust plane fit over nine centres cannot tell one wrong piece from a real tilt, because a tilted plane through eight points and an outlier and a flat plane through nine noisy points are both consistent with nine numbers. The piece's own radius decides without any geometry, and a gated piece contributes nothing rather than a down-weighted error.

### The along-ray fit and its curvature

Each surviving piece is fitted along its sighting rays the way the bench's fit moves a whole patch: the piece's centre slides along the ray from the reference view and the summed ZNCC against the other views is maximised. The curvature of that score at its optimum is the second trust signal. A piece with good texture but seen only over a narrow baseline has a sharp self-similarity but a flat score along the ray, and its depth is still not pinned. The curvature catches what the radius cannot.

The weight is a function of both. The draft does not fix the form; the candidates are the inverse radius, the ray curvature, or their product normalised across the surviving pieces. The functional form is an open question to be settled by measurement on the ground-truth captures.

### The plane and its determinacy

The plane is the weighted least-variance direction through the fitted centres. With nine well-spread pieces it is fully determined. After gating, the survivors can be collinear (one row of a patch across a depth edge) or too few. The adjacency-surfel estimator already defines a determinacy verdict for exactly this situation: both axes fixed, one axis fixed with the free axis named, or none. The gated estimator emits the same verdict. A caller tilts the patch only about the fixed axes and leaves the other at the prior, which is the mean viewing direction unless the caller has a better one.

### Why the whole-patch-good, parts-bad case is the common one

A patch on a planar textured surface passes everywhere and the gate changes nothing. The patches the gate exists for sit where a surface ends: a depth edge, an occluding contour, a sign against the sky. There the pieces over the surface are good and the pieces over the background are bad or belong to a different plane. The right normal is the one the good pieces define, which is the physical surface the patch mostly covers. Averaging good and bad pieces gives a plane that belongs to neither surface. So the estimator prefers the plane of the survivors, even when that is only one half of the patch, and reports the one-axis verdict when the survivors are collinear rather than filling the missing axis from bad data.

### Retiring the photometric search

The photometric normal search has three production callers: the embed-patches rounds, the `xform --refine-normals` command, and the bench's *Fit Normal*. When each has moved to the gated grid estimator, the search and its two support modules, the fronto-parallel cache and the D-optimal view subset, have no caller and are deleted. The fronto-parallel prior goes with them; the gated estimator's prior is the mean viewing direction, and the sweep over the fronto prior in the project's records showed the prior degrading normals everywhere it was applied.

The order is: build the estimator and measure it against the two harness ground truths; move the bench button and measure by hand; move embed-patches; move `xform --refine-normals`; delete.

## Implementation notes

- The nine per-piece radii are already computed by the self-similarity parts reading, as `zncc_self_similarity_radius_grid`. The estimator reads them from the same measurement and never recomputes them.
- The along-ray fit of a piece is the bench fit restricted to a cell; it should share the fit's code rather than copy it, so that the piece fit and the whole-patch fit cannot drift apart in their sampler or their pyramid rule.
- The piece's render for its self-similarity reading is the piece's own `R/3 × R/3` tile. At `R = 12` that is 4×4 px, which is too small to carry a meaningful radius. The estimator therefore needs `R ≥ 24` to read piece radii, which ties it to the fine tier in [two-tier-patch-density.md](two-tier-patch-density.md). At the coarse tier the gate falls back to the whole-patch and middle radii alone, and the estimator reports that it did.

## Determinism and precision

Per-piece fits are `f32` on tile intensities, as the bench fit is. The plane fit is `f64` because the fitted centres are world coordinates and the least-variance direction is a near-cancelling eigenproblem on nine points. Output is equal within tolerance for any thread count, with the same discrete piece verdicts; no content hash depends on it.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_piece_radius_px` | the whole-patch gate | A piece with a larger self-similarity radius is not fitted. |
| `min_ray_curvature` | to be measured | A piece with a flatter along-ray score is not fitted. |
| `min_pieces` | 3 | Fewer survivors leaves the normal at its prior. |

## Testing

A synthetic planar patch with uniform texture: all nine pieces fitted, normal within 0.5° of truth, both axes fixed. The same patch with one third replaced by a flat region: three pieces gated by radius, both axes still fixed, normal within 1°. The same patch with one half replaced by a second plane at a different depth: the survivors are one row or one column, the verdict names the free axis, and the fixed axis agrees with the surviving half's plane. A patch with all pieces over flat texture: fewer than `min_pieces`, normal unchanged, verdict none. Against the surface co-solve harness's two ground truths, the median normal error must beat the photometric search's 16 to 23 degrees.

## Non-goals

Joint estimation of depth and normal. The estimator takes the depth as fitted and turns the patch about its centre; the joint solve is the subject of [surface-co-solve.md](surface-co-solve.md).

Normals before poses exist. A normal is a 3D quantity and needs sighting rays. The pose-free precursor, per-piece 2D displacements, is [cluster-patches-piecewise-refinement.md](cluster-patches-piecewise-refinement.md).

## Open questions

- **Weight form.** Inverse radius, ray curvature, or their product. Decide by measuring normal error on the two harness ground truths with each.
- **Independent or joint piece fits.** Fitting each piece's depth alone is simple and parallel. Fitting all survivors with a shared plane constraint uses the plane to regularise weak pieces but reintroduces the coupling the gate was meant to remove. Start independent.
- **Piece overlap.** The ZNCC grid's cells do not overlap. Overlapping pieces give more centres but correlated ones. Start without overlap, since the radii are only available for the non-overlapping cells.
- **Fewer than three survivors.** Leave at the prior, or fall back to the adjacency-surfel estimate where neighbours exist. The latter is the better answer in a dense reconstruction and the only answer on the bench is the former.
