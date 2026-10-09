# Surface Footprint Analysis

**Status:** Draft. Decided:
- a patch's final footprint is chosen for a surface, not for a point: a step after anchor placement groups the anchors that lie on one surface and gives the group one footprint at or above its anchors' floors;
- the grouping reads anchors with normals, each normal from the [cell plane normal kernel](../core/patch/cell-plane-normals.md) or from the gated grid normal of [piece-gated-grid-normal.md](piece-gated-grid-normal.md);
- the group's footprint is one value per group.

Not decided: the reading that sets a group's factor above the floor; whether the group's footprint is also the footprint the fine density tier keeps. See [Open questions](#open-questions).

Amends:
- [patch-footprint-selection.md](patch-footprint-selection.md): its per-point primitive is now the anchor floor, and the footprint above the floor is chosen here
- [two-tier-patch-density.md](two-tier-patch-density.md): the footprint its tiers share is the one this step chooses
- [surface-co-solve.md](surface-co-solve.md) § "[Patch size, and where a surface ends](surface-co-solve.md#patch-size-and-where-a-surface-ends)": the size of a patch is set for its surface; the grouping and the stopping rules of that draft's flood fill are the ideas this step draws on, and they are not restated here

Related: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) and [core/patch/cell-plane-normals.md](../core/patch/cell-plane-normals.md) supply the anchor normals; [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) decides when a cell plane normal may be read at all; [core/analysis/adjacency-surfel-normals.md](../core/analysis/adjacency-surfel-normals.md) fits a plane over a point's image-space neighbours, the adjacency this step groups by. The measurements are in [patch-footprint-selection-measurements.md](patch-footprint-selection-measurements.md).

## Purpose

The two checked-in ground truths carry patch half-extents that the maintainer set by hand, and those sizes are a property of the surface, not of the point. A point's nearest neighbour on the same surface has a size within a factor of 1.25 on 73% (`seoul_bull`) and 84% (`kerry_park`) of the points that have such a neighbour, with a median difference under 1%. Between surfaces the sizes differ, often by a factor of two. No per-point reading tried, from self-similarity and coherence gates to a learned model on 52 readings and a quota of distinct features, reproduced the sizes better than a constant 12 px half-extent across captures ([measurements](patch-footprint-selection-measurements.md#noise-floor)).

[patch-footprint-selection.md](patch-footprint-selection.md) therefore reduces the per-point question to the anchor floor: the smallest footprint at which all nine cells of a patch localise, which is where an anchor can resolve a normal and an affine shape. The hand-set size lies above that floor on most points, by a median factor of about 1.4. This draft proposes the step that chooses how far above: group the anchors that lie on one surface, and give the group one footprint.

The anchors' cells and their cell plane normals are readings of the anchor, taken at the floor. Whether they depend on the footprint has not been measured: the cell plane normals were measured at one patch size only ([cluster-patch-refinement-measurements.md](../core/patch/cluster-patch-refinement-measurements.md)). If they prove independent of it, the step can group anchors by their floor-level normals and change the footprint afterwards without reading the normals again.

## Proposal

### Grouping

Anchors are grouped when they lie on one surface: adjacent, coplanar and with consistent normals. Adjacency is the image-space adjacency of the [observation adjacency graph](../core/analysis/observation-adjacency-graph.md) that [adjacency-surfel-normals.md](../core/analysis/adjacency-surfel-normals.md) fits over. Coplanarity and consistency read each anchor's centre and normal. The measurements counted two ground-truth points as on one surface when their normals are within 20°, each centre lies within `0.25·d + 0.1·(h₁ + h₂)` of the other's plane, and they are within `d ≤ 3·(h₁ + h₂)` of each other, for `d` their distance and `h` their world half-extents ([noise floor](patch-footprint-selection-measurements.md#noise-floor)). That test is the starting point for the grouping, with the anchors' floors in place of the hand-set sizes.

An anchor whose normal is not determined on both axes, or fails the precision gate of [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md), joins a group by adjacency and position only, and does not decide the group's plane.

### One footprint per group

The group's footprint is one value, set from the group as a whole: a level at or above its anchors' floors, raised by a factor read from the group's region. Every patch of the group takes that footprint.

### The factor above the floor

What sets the factor is open. The hand-set sizes show a trend: smaller where strong features lie near the point, larger on surfaces with many weak features. Over a fixed 12 px footprint, the count of DoG responses above 0.03 correlates −0.56 (`kerry_park`) and −0.26 (`seoul_bull`) with the log of the hand-set size, and the density of Laplacian blobs +0.43 and +0.35 ([measurement](patch-footprint-selection-measurements.md#a-quota-of-distinct-features-against-the-hand-set-sizes-2026-10-09)). The grey variance of the photograph within 32 px of the point correlates −0.56 and −0.49 ([measurement](patch-footprint-selection-measurements.md#the-rule-families)). No per-point rule built on these reproduced the sizes across captures. The candidate is the same statistics taken over a group's region, where the noise of a single point's neighbourhood averages out.

## Open questions

- **The reading that sets the factor.** Region-level texture statistics over the group, such as the count of strong DoG responses, the density of weak blobs and the grey variance, are the candidates. Whether any of them, taken over a group rather than a point, reproduces the hand-set sizes better than the constant is the measurement to make.
- **What the fine tier keeps.** Whether the group's footprint is also the footprint [two-tier-patch-density.md](two-tier-patch-density.md) keeps when it resamples a patch at the fine density, or whether the fine tier reads a different footprint, is open.
- **Combining the floors.** How the group's level is formed from its anchors' floors, the largest of them or a quantile, is open. The largest lets one anchor whose cells stay capped raise the whole group.
- **Units.** Whether the group's one value is a world half-extent or a footprint in each anchor's reference-view px is open. The hand-set world half-extent grows with depth at a slope of 0.7 to 0.8 on a log-log scale, so on a surface that recedes from the cameras the two differ ([measurement](patch-footprint-selection-measurements.md#what-the-human-chose)).
