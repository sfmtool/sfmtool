# Cluster Patches Piecewise Refinement

**Status:** Draft. Decided:
- the cluster-patches refinement, which today fits one affine shape per member against the cluster's template, also fits a 2D displacement for each of the member's nine cells, the same cells the self-similarity reading splits the bitmap into;
- those per-cell displacements are stored in the cluster-patches file beside the member's shape, with a per-cell status, so that a consumer with poses can turn them into a normal without rendering anything;
- a cell whose own self-similarity radius fails the member gate is not fitted and is stored as refused;
- the seed's writer turns the per-cell displacements into a normal and a determinacy verdict once poses exist, replacing the tilt solve on whole-member warps its debug snapshots carry today.

Not decided: whether cells are fitted as pure shifts or as shift plus scale; whether the cell fit runs on every member or only on kept ones; the `.matches` version that carries the columns. See [Open questions](#open-questions).

Amends:
- [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md): a second stage of the cascade
- [core/patch/cluster-patches.md](../core/patch/cluster-patches.md): what the refined file holds
- [formats/matches-file-format.md](../formats/matches-file-format.md): new per-member columns

Related drafts: [cluster-patches-consumers-amendment.md](cluster-patches-consumers-amendment.md) § "Seeding `embed-patches` frames from patch clusters" asks whether the refined shapes can start the embedding's frames; this draft answers that the per-cell displacements can, and that section should point here. [piece-gated-grid-normal.md](piece-gated-grid-normal.md) is the estimator that consumes the displacements once poses exist.

## Purpose

Before any camera pose is known, the project groups matching features across images into clusters and refines, for each member of a cluster, an affine shape that maps the cluster's template onto that member's photograph. That shape says how the surface around the feature appears in each view: its scale, its orientation and its skew. It is a two-dimensional fact about one photograph and needs no geometry to compute.

A normal is a three-dimensional fact and needs sighting rays, so it cannot be computed at this stage. But the raw material for a normal can. If the member's patch is cut into pieces and each piece is registered to the template separately, the pieces' displacements describe how the surface's appearance varies across the patch in that view. Once poses exist, each piece's displacement in each view becomes a sighting ray, the pieces' depths follow from parallax, and a plane through them is the normal. The piecewise refinement is the pose-free half of the piece-gated grid normal.

Today the seed's writer derives a normal from the whole-member affine shapes, by a tilt solve with a fronto-parallel prior and an 80° cap, and only for its debug snapshots; a hypothesis release carries no patch frames at all. That reads one shape per member and cannot see that half of the patch lies on a different surface. The per-cell displacements can.

### Why this matters for the seed

The seed stage reads the cluster-patches file and nothing else about appearance. Scored against the approved ground truths, it finds a correct candidate in 7 of 8 captures and picks it in 4 of 8, with a choice rule that reads no photometric evidence. Giving the seed normals it can trust, from a file it already reads, is the cheapest route to scoring its candidates photometrically.

## Rust API

The refinement lives in [cluster_refine](../../crates/sfmtool-core/src/patch/cluster_refine/mod.rs), bound as `sfmtool._sfmtool.matching.refine_cluster_patches`. The piecewise stage is a second pass after the affine cascade.

```rust
/// Per-cell registration of one member against the template, after its
/// affine shape is fixed.
pub struct CellRefinement {
    /// Displacement of each cell's centre from where the affine shape
    /// places it, in template grid px.
    pub shift_px: [[[f32; 2]; 3]; 3],
    /// Windowed ZNCC of each cell at its optimum.
    pub zncc: [[f32; 3]; 3],
    /// Fitted, refused by self-similarity, refused by ZNCC, or not attempted.
    pub status: [[CellStatus; 3]; 3],
}

pub struct ClusterPatchRefinement {
    // existing per-member fields …
    pub cells: Vec<CellRefinement>,      // one per member, kept members only
}
```

**Why this shape.** Displacements are relative to the affine shape, not absolute, so a consumer that ignores them reads the file exactly as before. They are in template grid px, the unit every other refined quantity in the file uses. The status is per cell because the point of the stage is to know which cells to trust; a single per-member flag would discard that.

**Example.** The seed's writer, given poses, back-projects each kept member's nine cell centres plus their displacements to rays, intersects them per cell across members, and hands the nine fitted centres with their statuses to the gated plane fit.

## Theory

### What a cell's displacement means

The affine shape is the best single linear map from the template to the member. If the surface under the patch is one plane, the true map from template to member is a homography, and the affine shape is its first-order approximation at the patch centre. The homography's departure from affine across the patch is small for a small patch and grows with tilt. A cell's residual displacement after the affine map is that departure sampled at the cell's centre. Across nine cells it is a sampling of the second-order term, which is what carries the tilt.

If the surface is not one plane, a cell over the second surface displaces by its parallax relative to the first, which is not small and does not fit the homography model. That cell has a low ZNCC at its best shift, or a shift inconsistent with its neighbours, and is refused. This is how the stage finds the whole-good, parts-bad case before any geometry exists.

### Why gate by the cell's own self-similarity

A cell over flat or repeating texture registers anywhere. Its shift is noise and its ZNCC may still be high, because a flat region correlates well with a flat region at every shift. The self-similarity radius of the cell, read on the member's own tile, says the registration is not meaningful, and the cell is refused on that alone. The whole-member gate in the current refinement does the same for the member; this applies it per cell.

### From displacements to a normal

With poses, a cell's centre in a member is a pixel, and that pixel is a ray. The same cell in each kept member gives a bundle of rays whose nearest point is the cell's position in the world. Nine positions, minus refused cells, are the input to the piece-gated plane fit. The displacements were measured photometrically at refinement time; the normal costs no rendering at seed time. That is the property the seed needs, since its releases carry no frames and it renders nothing.

## Format

Per kept member, three new entries in the cluster-patches file: `member_cell_shift_px` of shape `(members, 3, 3, 2)`, `member_cell_zncc` of shape `(members, 3, 3)`, and `member_cell_status` of shape `(members, 3, 3)` with its legend in the section's metadata, following the convention `member_status` adopted in version 7. They are optional entries introduced in the next version; a reader of an older file has no cells and a consumer that needs them says so.

## Implementation notes

- The cell fit starts from the affine shape's placement of the cell and searches shifts only, within a bound of a few grid px. The affine cascade has already removed scale and rotation; a cell that needs more than a shift is over a different surface and should be refused, not fitted.
- At `patch_size = 12` a cell is 4×4 template px. That is enough for a shift registration against a textured template, and not enough for a self-similarity ellipse. The per-cell gate at that size reads the cell's ZNCC curvature over the shift search instead, and the self-similarity gate is used when the refinement runs at a larger size.
- The template is the cluster's reference member. The displacements are relative to that member's frame, so the reference member's own cells displace by zero by construction and carry no information; the normal fit uses the other members.

## Determinism and precision

`f32` throughout, matching the affine cascade. Per-member work is independent and parallel; output equal within tolerance for any thread count, with the same cell statuses.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `cell_shift_bound_px` | 2.0 | Search bound for a cell's shift from its affine placement. |
| `min_cell_zncc` | to be measured | A cell below this at its optimum is refused. |
| `min_cell_curvature` | to be measured | A cell whose ZNCC over the shift search is flatter is refused. |

## Testing

A synthetic planar cluster seen from a tilted view: the cell shifts follow the homography's second-order term, within sampler tolerance, and all nine cells are fitted. The same cluster with one third of the template over a second plane: the three cells over it are refused by ZNCC or shift inconsistency. Reading an older `.matches` file gives no cells and no error. On the `seoul_bull_sculpture` ground truth, the normals the seed's writer derives from the cells have lower median error than the tilt solve on whole-member warps.

## Non-goals

A normal in the cluster-patches file. The file is pose-free and stays so; a normal is derived by whoever holds poses.

Fitting cells with a full affine. A cell that needs one is over another surface.

## Open questions

- **Shift only, or shift and scale.** Scale per cell would read the perspective term's radial component; it also costs a two-dimensional search per cell. Start with shift.
- **All members or kept members.** Refused members have no trustworthy affine shape to start from. Kept members only.
- **Version.** Whether these columns ride with the next planned `.matches` version or get their own.
