# Cluster Patches Piecewise Refinement

**Status:** Draft. Decided:
- the cluster-patches refinement, which today fits one affine shape per member against the cluster's template, fits a 2D displacement for each of the member's nine cells, the same cells the self-similarity reading splits the bitmap into;
- the fit is a loop with two levels: the photograph is rendered once per iteration into a working patch on the template's grid, the nine cells are registered within that working patch, the cell shifts are fitted back to an affine update of the member's shape, and the loop renders again until the update is small;
- the stored per-cell displacement is the residual to the converged affine, so the affine absorbs the first-order correction and the residual carries the second-order term a normal is derived from;
- those per-cell displacements are stored in the cluster-patches file beside the member's shape, with a per-cell status, so that a consumer with poses can turn them into a normal without rendering anything;
- a cell whose own reading fails its gate is not fitted and is stored as refused, and the affine update is fitted only to the cells that survive, dropping to a similarity or a shift when too few do;
- the seed's writer turns the per-cell displacements into a normal and a determinacy verdict once poses exist, replacing the tilt solve on whole-member warps its debug snapshots carry today.

Not decided: whether cells are fitted as pure shifts or as shift plus scale; whether the cell fit runs on every member or only on kept ones; the `.matches` version that carries the columns; whether the loop replaces the affine cascade or runs after it. See [Open questions](#open-questions).

Amends:
- [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md): the shape fit and a second output per member
- [core/patch/cluster-patches.md](../core/patch/cluster-patches.md): what the refined file holds
- [formats/matches-file-format.md](../formats/matches-file-format.md): new per-member columns

Related drafts: [cluster-patches-consumers-amendment.md](cluster-patches-consumers-amendment.md) § "Seeding `embed-patches` frames from patch clusters" asks whether the refined shapes can start the embedding's frames; this draft answers that the per-cell displacements can, and that section points here. [piece-gated-grid-normal.md](piece-gated-grid-normal.md) is the estimator that consumes the displacements once poses exist. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) supplies the per-view sampler the working patch should be rendered with.

## Purpose

Before any camera pose is known, the project groups matching features across images into clusters and refines, for each member of a cluster, an affine shape that maps the cluster's template onto that member's photograph. That shape says how the surface around the feature appears in each view: its scale, its orientation and its skew. It is a two-dimensional fact about one photograph and needs no geometry to compute.

A normal is a three-dimensional fact and needs sighting rays, so it cannot be computed at this stage. But the raw material for a normal can. If the member's patch is cut into pieces and each piece is registered to the template separately, the pieces' displacements describe how the surface's appearance varies across the patch in that view. Once poses exist, each piece's displacement in each view becomes a sighting ray, the pieces' depths follow from parallax, and a plane through them is the normal. The piecewise refinement is the pose-free half of the piece-gated grid normal.

Today the seed's writer derives a normal from the whole-member affine shapes, by a tilt solve with a fronto-parallel prior and an 80° cap, and only for its debug snapshots; a hypothesis release carries no patch frames at all. That reads one shape per member and cannot see that half of the patch lies on a different surface. The per-cell displacements can.

### Why this matters for the seed

The seed stage reads the cluster-patches file and nothing else about appearance. Scored against the approved ground truths, it finds a correct candidate in 7 of 8 captures and picks it in 4 of 8, with a choice rule that reads no photometric evidence. Giving the seed normals it can trust, from a file it already reads, is the cheapest route to scoring its candidates photometrically.

## Rust API

The refinement lives in [cluster_refine](../../crates/sfmtool-core/src/patch/cluster_refine/mod.rs), bound as `sfmtool._sfmtool.matching.refine_cluster_patches`.

```rust
/// Per-cell registration of one member against the template, as the
/// residual to the member's converged affine shape.
pub struct CellRefinement {
    /// Displacement of each cell's centre from where the affine shape
    /// places it, in template grid px.
    pub shift_px: [[[f32; 2]; 3]; 3],
    /// Windowed ZNCC of each cell at its optimum.
    pub zncc: [[f32; 3]; 3],
    /// Fitted, refused by its reading, refused by ZNCC, refused because its
    /// best shift lies on the search bound, or not attempted.
    pub status: [[CellStatus; 3]; 3],
    /// Iterations the two-level loop ran before its affine update fell
    /// below the tolerance, or the cap.
    pub iterations: u8,
}

pub struct PiecewiseParams {
    /// Search bound for a cell's shift from its affine placement, grid px.
    /// Also the margin the working patch is rendered with on each side.
    pub cell_shift_bound_px: f32,
    /// A cell below this ZNCC at its optimum is refused.
    pub min_cell_zncc: f32,
    /// A cell whose ZNCC over the shift search is flatter is refused.
    pub min_cell_curvature: f32,
    /// The loop stops when the affine update moves every cell centre by
    /// less than this, grid px.
    pub update_tolerance_px: f32,
    /// Iteration cap.
    pub max_iterations: u8,
}

pub struct ClusterPatchRefinement {
    // existing per-member fields …
    pub cells: Vec<Option<CellRefinement>>, // one per member; None unless kept
}
```

**Why this shape.** Displacements are relative to the affine shape, not absolute, so a consumer that ignores them reads the file exactly as before. They are in template grid px, the unit every other refined quantity in the file uses. The status is per cell because the point of the stage is to know which cells to trust; a single per-member flag would discard that. The cells are one entry per member, `None` for a member that is not kept, so they index like every other per-member array and a consumer cannot pair a member with another member's cells. The iteration count is reported because a member that hit the cap is one whose affine never settled, and a consumer may want to treat its residuals as less trustworthy. The shift bound and the render margin are one parameter because they must be equal: a search that can reach a shift the render did not cover reads outside the tile.

**Example.** The seed's writer, given poses, back-projects each kept member's nine cell centres plus their displacements to rays, intersects them per cell across members, and hands the nine fitted centres with their statuses to the gated plane fit.

## Theory

### The two-level loop

A member's fit runs on two levels. The outer level touches the photograph; the inner level touches only a small tile.

1. **Render.** The member's current affine shape maps the template's grid into the photograph, and the photograph is resampled along that map into a working patch: the template's `R × R` grid plus a margin of `cell_shift_bound_px` on every side. This is the only step that reads the photograph. It is one gather of a few hundred samples from a region of the image that is contiguous in the sense the sampler cares about.
2. **Register the cells.** Each of the nine cells of the template is compared with the working patch over every shift within the bound, by windowed ZNCC, and its optimum is read at sub-pixel precision from the peak. The margin guarantees that every shift of every cell, edge cells included, reads pixels the render covered. All nine searches run on one tile that fits in the first-level cache; none of them touches the photograph.
3. **Fit the affine update.** The nine cell centres and their measured shifts are nine point correspondences between the template and the working patch. A weighted least-squares affine through the surviving cells is the update to the member's shape: the current shape composed with that affine. A cell that was refused in step 2 carries no weight.
4. **Loop.** If the update moves any cell centre by more than `update_tolerance_px`, the shape is updated and the loop returns to step 1. Otherwise it stops, and the cell shifts measured at the last pass are the stored residuals.

This is the inverse-compositional alignment pattern with a piecewise translation model. Given the starting shape the current affine cascade provides, it converges in two or three iterations; each iteration is one render and nine small searches, where the cascade's simplex search is one photograph sampling per objective evaluation.

### What converges, and what is left over

The affine update absorbs everything nine translations can express as one linear map: a shift, a scale, a rotation and a skew. What it cannot absorb is the part of the cell shifts that varies across the patch in a way no affine matches. For a planar surface that is the second-order term of the homography, which is what the normal needs. For a surface that is not one plane it is the parallax of the cells that lie off the plane, which is large and is what refuses those cells.

The stored displacement is defined as the residual to the converged affine. That definition is only exact at convergence, which is why the loop's exit reads the update's size rather than the ZNCC: a member on a non-planar surface converges to the best affine and leaves a large, honest residual, instead of iterating against a score floor it cannot reach.

### Which cells are refused, and what the update does then

A cell over flat or repeating texture registers anywhere. Its shift is noise, and its ZNCC may still be high because a flat region correlates with a flat region at every shift. The cell's reading says so: at the sizes the cluster-patches file uses, a cell is too small for a self-similarity ellipse, so the reading is the curvature of the cell's ZNCC over its shift search. A flat curvature refuses the cell. A low ZNCC at the optimum refuses it for the other reason, that it is over a different surface. The whole-member gate in the current refinement does the same for the member; this applies it per cell.

The affine update is fitted to the survivors. Six parameters need at least three non-collinear cells, and a trustworthy fit needs more. A rule on the count and the spread of the survivors chooses the model: a full affine with five or more well-spread cells, a similarity with three or more otherwise, a pure shift below that, and no update when none survive, in which case the member keeps its cascade shape and all nine cells are stored as not attempted. "Otherwise" covers three or four cells, five or more that lie along one row or column and so pin only one axis of an affine, and five or more well-spread cells whose affine normal equations are singular: each falls back one step, to the similarity, before the shift.

### Double resampling

The cell shifts are measured on a tile that is itself an interpolation of the photograph, and the stored displacement is read through that sampler. For a shift read as a sub-pixel ZNCC peak this is a small blur, not a bias, and the fine tier re-measures whatever it needs. It does argue for rendering the working patch with the per-view sampler choice of [sharper-patch-bitmap.md](sharper-patch-bitmap.md), since a view compressed along one axis loses most along that axis.

### From displacements to a normal

With poses, a cell's centre in a member is a pixel, and that pixel is a ray. The same cell in each kept member gives a bundle of rays whose nearest point is the cell's position in the world. Nine positions, minus refused cells, are the input to the piece-gated plane fit. The displacements were measured photometrically at refinement time; the normal costs no rendering at seed time. That is the property the seed needs, since its releases carry no frames and it renders nothing.

## Format

Per kept member, four new entries in the cluster-patches file: `member_cell_shift_px` of shape `(members, 3, 3, 2)`, `member_cell_zncc` of shape `(members, 3, 3)`, `member_cell_status` of shape `(members, 3, 3)` with its legend in the section's metadata, following the convention `member_status` adopted in version 7, and `member_cell_iterations` of shape `(members,)`. They are optional entries introduced in the next version; a reader of an older file has no cells and a consumer that needs them says so.

## Implementation notes

- The working patch is rendered at the template's grid plus the margin, once per iteration, and nothing in the inner loop reads the photograph. A profile of the loop should show the render as the only cache-missing step.
- The cell search is a shift search only. The affine cascade and the loop's own updates have removed scale and rotation; a cell that needs more than a shift is over a different surface and should be refused, not fitted.
- At `patch_size = 12` a cell is 4×4 template px. That is enough for a shift registration against a textured template, and not enough for a self-similarity ellipse, which is why the per-cell gate reads curvature here and the self-similarity gate applies when the refinement runs at a larger size.
- The template is the cluster's reference member. The displacements are relative to that member's frame, so the reference member's own cells displace by zero by construction and carry no information; the normal fit uses the other members.
- The affine update fit is a weighted least squares on at most nine points; it is solved in `f64` and composed into the `f32` shape, because the update is small and its composition with the shape is where a near-identity matrix is multiplied repeatedly.
- The stored residual is exact to the returned shape. The last render measured cell `c` at `c + d` of its grid, and the returned shape's grid maps to that one by the last update `c ↦ A·c + b`, so the residual is `A⁻¹·(d − (A·c + b − c))`. Subtracting the update's movement without the `A⁻¹` leaves an error of the size of `(A⁻¹ − I)` applied to the residual.
- The residuals follow the homography's second-order term less closely as the term grows. On the synthetic plane with `h = [0.003, −0.002]` (term up to 0.52 grid px) the largest residual error over the nine cells is 0.099 grid px and the RMS 0.054; at twice that, `h = [0.006, −0.004]` (term up to 1.08 grid px), the largest is 0.31 and the RMS 0.13.
- The loop's result is all or nothing. When any iteration fails, the first or a later one, by a failed render, no surviving cell, or an update that reflects (`det A ≤ 0`) or is not finite, the member keeps its cascade shape and all nine cells are stored as not attempted; what earlier iterations fitted is discarded, because the cells were read at a shape that would not be the one returned.
- After the loop moves a member, its whole-patch ZNCC, its parts and its shift from the seed are read again at the new shape, and the cascade's acceptance gates, `min_zncc` and `max_shift_px`, are applied to those readings. A member that fails either, or whose new support leaves the frame, keeps its cascade shape and readings, with all nine cells not attempted.
- A cell's search reads the window's sum and sum of squares from summed-area tables of the working patch, built once per render; the template side is mean-removed, so the cross term needs no window mean and is the only pass over the window per shift.
- The stage is off by default (`ClusterRefineParams::piecewise` is `None`) until the cluster-patches file carries the cells; the milestone that adds them turns it on from the CLI, and the fleet comparison decides the default.

## Determinism and precision

`f32` on tile intensities, matching the affine cascade; `f64` for the affine update solve. Per-member work is independent and parallel; output equal within tolerance for any thread count, with the same cell statuses and the same iteration counts.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `cell_shift_bound_px` | 2.0 | Search bound for a cell's shift from its affine placement, and the working patch's margin. |
| `min_cell_zncc` | to be measured | A cell below this at its optimum is refused. |
| `min_cell_curvature` | to be measured | A cell whose ZNCC over the shift search is flatter is refused. |
| `update_tolerance_px` | 0.05 | The loop stops when no cell centre moves by more. |
| `max_iterations` | 5 | Iteration cap. |

## Testing

A synthetic planar cluster seen from a tilted view, with the starting shape perturbed by a known affine: the loop recovers the affine within tolerance in at most three iterations, the cell shifts follow the homography's second-order term within sampler tolerance, and all nine cells are fitted. The same cluster with one third of the template over a second plane: the three cells over it are refused, the update is fitted to the other six, and the recovered affine matches the first plane. A member whose every cell is flat: no update, nine cells not attempted, cascade shape unchanged. Reading an older `.matches` file gives no cells and no error. On the fleet, the converged affine shapes agree with the cascade's within tolerance on members the cascade kept, and the refinement's wall time does not rise. On the `seoul_bull_sculpture` ground truth, the normals the seed's writer derives from the cells have lower median error than the tilt solve on whole-member warps.

## Non-goals

A normal in the cluster-patches file. The file is pose-free and stays so; a normal is derived by whoever holds poses.

Fitting cells with a full affine. A cell that needs one is over another surface.

## Open questions

- **Shift only, or shift and scale.** Scale per cell would read the perspective term's radial component; it also costs a two-dimensional search per cell. Start with shift.
- **All members or kept members.** Refused members have no trustworthy affine shape to start from. Kept members only.
- **Version.** Whether these columns ride with the next planned `.matches` version or get their own.
- **Replace or follow the cascade.** The loop can start from the cascade's shape, or from the detection's shape with the cascade removed. Start after the cascade, measure how many iterations the loop needs from the detection alone, and decide whether the cascade still earns its cost.
