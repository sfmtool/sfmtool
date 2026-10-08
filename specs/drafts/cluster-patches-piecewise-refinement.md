# Cluster Patches Piecewise Refinement

**Status:** Draft. Decided:
- the cluster-patches refinement, which today fits one affine shape per member against the cluster's template, also measures a 2D displacement for each of the member's nine cells, the same cells the self-similarity reading splits the bitmap into;
- the stage is a measurement taken at the cascade's shape: the photograph is rendered once into a working patch on the template's grid, the nine cells are registered within that working patch, and a robust affine map is fitted to the nine shifts; the member's shape, position, ZNCC and shift stay exactly as the cascade produced them;
- the affine map is fitted by iteratively reweighted least squares with a Tukey biweight, and a cell the robust fit gives no weight is stored as `refused_outlier`, a cell status `.matches` version 9 adds;
- the stored per-cell displacement is the displacement of the cell's centre from where the member's stored shape places it, with no fitted affine map removed; a consumer that wants only the part no affine map matches, the second-order term a normal is derived from, fits and removes an affine map from the nine displacements itself;
- a two-level loop that applies the fitted map as an update of the member's shape, accepted only while the whole-member windowed ZNCC the cascade maximised does not fall, is an option, `PiecewiseParams::move_shape`, off by default: on five fleet entries it moved 0.2% to 1.4% of kept members, each to a shape of higher ZNCC, and on `KerryPark480` those 55 members alone turned the seed's passing pick into a failing one ([ablation](cluster-patches-piecewise-refinement-measurements.md#subset-with-the-acceptance-rule-2026-10-08));
- those per-cell displacements are stored in the cluster-patches file beside the member's shape, with a per-cell status, so that a consumer with poses can turn them into a normal without rendering anything;
- a cell whose own reading fails its gate is not fitted and is stored as refused, and the affine map is fitted only to the cells that survive, dropping to a similarity or a shift when too few do;
- the cell fit runs on kept members only, since a member that is not kept has no trustworthy affine shape to start from, and `.matches` version 8 refuses a file that stores readings on any other member;
- the seed's writer turns the per-cell displacements into a normal and a determinacy verdict once poses exist, replacing the tilt solve on whole-member warps its debug snapshots carry today.

Not decided: whether cells are fitted as pure shifts or as shift plus scale; whether the stage runs by default; whether the shapes the loop moves are better or worse than the cascade's. See [Open questions](#open-questions). The measurements behind these questions are in [cluster-patches-piecewise-refinement-measurements.md](cluster-patches-piecewise-refinement-measurements.md); its [fleet conclusions](cluster-patches-piecewise-refinement-measurements.md#what-these-measurements-decide), those of its [subset run with the acceptance rule](cluster-patches-piecewise-refinement-measurements.md#subset-with-the-acceptance-rule-2026-10-08) and those of its [subset run of the measurement](cluster-patches-piecewise-refinement-measurements.md#subset-with-the-shape-left-to-the-cascade-2026-10-08) are summarised under [Fleet results](#fleet-results).

Amends:
- [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md): a second output per member, and an optional refit of the shape
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
/// Per-cell registration of one member against the template, relative to
/// the shape the stage returns for the member.
pub struct CellRefinement {
    /// Displacement of each cell's centre from where the returned shape
    /// places it, in template grid px, with no fitted affine map removed.
    pub shift_px: [[[f32; 2]; 3]; 3],
    /// Windowed ZNCC of each cell at its optimum.
    pub zncc: [[f32; 3]; 3],
    /// Fitted, refused by its reading, refused by ZNCC, refused because its
    /// best shift lies on the search bound, refused as an outlier by the
    /// robust fit, or not attempted.
    pub status: [[CellStatus; 3]; 3],
    /// Renders the stage made: 1 without `move_shape`.
    pub iterations: u8,
    /// Measured (without `move_shape`), not run, or why the loop stopped:
    /// converged, the cap, an update rejected for lowering the whole-member
    /// ZNCC, or an update that did not shrink.
    pub stop: LoopStop,
    /// Whether the last fitted update was applied to the returned shape;
    /// always false without `move_shape`.
    pub final_update_accepted: bool,
}

/// How far the whole-member ZNCC may fall at an updated shape before the
/// update is rejected (with `move_shape`).
pub const ACCEPT_ZNCC_TOLERANCE: f64 = 1e-4;

pub struct PiecewiseParams {
    /// Whether the fitted affine map may move the member's shape. Default
    /// false: measure once and leave the member exactly as the cascade left it.
    pub move_shape: bool,
    /// Search bound for a cell's shift from its affine placement, grid px.
    /// Also the margin the working patch is rendered with on each side.
    pub cell_shift_bound_px: f32,
    /// A cell below this ZNCC at its optimum is refused.
    pub min_cell_zncc: f32,
    /// A cell whose ZNCC over the shift search is flatter is refused.
    pub min_cell_curvature: f32,
    /// With `move_shape`: the loop stops when the affine update moves every
    /// cell centre by less than this, grid px.
    pub update_tolerance_px: f32,
    /// With `move_shape`: iteration cap.
    pub max_iterations: u8,
}

pub struct ClusterRefineResult {
    // existing per-member fields …
    pub cells: Vec<Option<CellRefinement>>, // one per member; None unless kept
}

/// In sfmtool-core: the result's cells as the format's columns, one row per
/// member, `NaN` / `NaN` / not attempted / `0` for a member without cells.
pub fn member_cell_data(cells: &[Option<CellRefinement>]) -> MemberCellData;

// In sfmtool-matches-format (`cells.rs`):

/// The four per-cell columns of `cluster_patches/`, member-parallel.
pub struct MemberCellData {
    pub shift_px: Array4<f32>,  // (M, 3, 3, 2)
    pub zncc: Array3<f32>,      // (M, 3, 3)
    pub status: Array3<u8>,     // (M, 3, 3) ClusterCellStatus codes
    pub iterations: Array1<u8>, // (M,)
}

/// The stored cell status, with the same discriminants as `CellStatus`.
#[repr(u8)]
pub enum ClusterCellStatus {
    Fitted = 0,
    RefusedCurvature = 1,
    RefusedZncc = 2,
    NotAttempted = 3,
    RefusedBound = 4,
    RefusedOutlier = 5, // version 9
}
```

The binding `refine_cluster_patches` takes `piecewise=` (default `False`) and the six `PiecewiseParams` settings as keyword arguments under their field names, `move_shape=`, `cell_shift_bound_px=`, `min_cell_zncc=`, `min_cell_curvature=`, `update_tolerance_px=` and `max_iterations=`, each `None` by default for the Rust default. With `piecewise=True` it returns the four columns as `member_cell_shift_px`, `member_cell_zncc`, `member_cell_status` and `member_cell_iterations`, the settings it ran with as `piecewise_options`, and two per-member readings the file does not store: `member_cell_loop_stop`, the `LoopStop` code (`5`, measured, for every kept member with readings when `move_shape` is off), and `member_cell_update_accepted`, whether the last fitted update was applied. Without it those keys are `None`. `sfm cluster-patches --piecewise` runs the stage at the defaults and exposes no setting, so it never moves a shape.

**Why this shape.** Displacements are relative to the affine shape, not absolute, so a consumer that ignores them reads the file exactly as before. They are the displacements as measured, with no fitted affine map removed, because the stored shape and the displacements together then say where each cell's content lies, in either mode, and a consumer that wants the second-order term alone can fit and remove the affine part itself, whereas one that wants the whole displacement could not recover the part a stored residual had dropped. They are in template grid px, the unit every other refined quantity in the file uses. The status is per cell because the point of the stage is to know which cells to trust; a single per-member flag would discard that. The cells are one entry per member, `None` for a member that is not kept, so they index like every other per-member array and a consumer cannot pair a member with another member's cells. The iteration count is reported because, with `move_shape`, a member that hit the cap is one whose affine never settled, and a consumer may want to treat its displacements as less trustworthy. The shift bound and the render margin are one parameter because they must be equal: a search that can reach a shift the render did not cover reads outside the tile. `stop` and `final_update_accepted` are kept in the result in both modes, reading measured and `false` without `move_shape`, rather than being made conditional on the mode, so the result has one shape.

**Example.** The seed's writer, given poses, back-projects each kept member's nine cell centres plus their displacements to rays, intersects them per cell across members, and hands the nine fitted centres with their statuses to the gated plane fit.

## Theory

### The measurement

A member's cells are measured at the shape the affine cascade found for it. The stage touches the photograph once, and everything after that touches only a small tile.

1. **Render.** The member's affine shape maps the template's grid into the photograph, and the photograph is resampled along that map into a working patch: the template's `R × R` grid plus a margin of `cell_shift_bound_px` on every side. This is the only step that reads the photograph. It is one gather of a few hundred samples from a region of the image that is contiguous in the sense the sampler cares about.
2. **Register the cells.** Each of the nine cells of the template is compared with the working patch over every shift within the bound, by windowed ZNCC, and its optimum is read at sub-pixel precision from the peak. The margin guarantees that every shift of every cell, edge cells included, reads pixels the render covered. All nine searches run on one tile that fits in the first-level cache; none of them touches the photograph.
3. **Fit the affine map.** The nine cell centres and their measured shifts are nine point correspondences between the template and the working patch. A robust affine map is fitted through the surviving cells. A cell that was refused in step 2 carries no weight, and a cell whose shift disagrees with the others is given none by the fit and is stored as `refused_outlier`. The map itself is not stored and does not move the shape; it is what decides which cells agree with each other.

The stored displacement of a cell is its shift as measured in step 2: the displacement of the cell's centre from where the member's stored shape, the cascade's, places it. The member's shape, position, whole-member ZNCC and shift from the seed are the cascade's, bit for bit.

### What the displacements carry

The displacements carry three things.

- **A first-order part.** The cascade found the affine shape that maximises one windowed ZNCC over the whole patch. Each cell's optimum is found separately, and at `patch_size = 12` a cell is 4×4 template px, so its shift is noisy and the nine cell optima do not agree exactly with the whole-patch optimum. The part of that disagreement one affine map can express is in the displacements. On the fleet subset the fitted cells' displacements have a median length of 0.17 to 0.36 grid px, most of it this part.
- **The second-order term.** For a planar surface, the part of the cell shifts no affine map matches is the second-order term of the homography, which is what the normal needs.
- **Parallax.** For a surface that is not one plane, the cells that lie off the plane the others agree on are displaced by their parallax, which is large and is what refuses those cells.

A consumer that back-projects a cell centre plus its displacement through the stored shape reads where the cell's content lies, which is what the definition is for. A consumer that wants only the second-order term fits an affine map to the nine displacements, over the cells stored as `fitted`, and removes it. The stage does not store its own fitted map, because a consumer that refits from the stored displacements gets it back, while a stored residual with the map removed would lose the first-order part for good.

### Why the shape is left to the cascade

The cell fit could move the shape: its affine map is an update the cells agree on. But the two objectives disagree. The cell fit asks for the affine map that best explains nine separate shifts; the cascade found the one that maximises the whole-patch ZNCC, the score the member was kept on. A loop that applied the cell fit's map unconditionally moved 80% to 99.6% of kept members on the fleet and lowered every moved member's ZNCC. With an acceptance rule that applies an update only when the whole-patch ZNCC does not fall, the loop moves 0.2% to 1.4% of kept members, each to a higher ZNCC. Even that changes the seed: on `KerryPark480` the 55 members it moves, 0.34% of the kept members, alone turn the seed's passing pick into a failing one. ZNCC cannot say whether those shapes are worse, and until it is known the stage leaves every shape where the cascade put it. The loop is kept behind `PiecewiseParams::move_shape` for that question. The measurements are under [Fleet results](#fleet-results).

### The shape-moving loop (`move_shape`)

With `move_shape`, the fitted map is an update of the member's shape and the stage runs on two levels. The outer level touches the photograph; the inner level touches only the working patch.

1. **Render**, at the current shape, as in the measurement.
2. **Register the cells**, as in the measurement.
3. **Fit the affine update.** The robust affine map through the surviving cells is the update: the current shape composed with that map.
4. **Accept or reject.** The whole-member windowed ZNCC, the objective the affine cascade maximised, is read at the updated shape. If it is lower than at the current shape by more than `ACCEPT_ZNCC_TOLERANCE`, or lower than at the starting shape at all, the update is rejected: the shape is not moved and the loop stops. The tolerance lets the loop cross a flat stretch above the start; the floor at the start keeps it from accumulating over several updates.
5. **Loop.** The loop stops with the update applied if it moves no cell centre by more than `update_tolerance_px`, or if it was the last of `max_iterations` renders. It stops if the update's largest cell-centre movement is not smaller than the previous update's, which means the loop is alternating between shapes rather than converging; it then keeps whichever of the shape before and the shape after the update has the higher whole-member ZNCC. Otherwise the shape is updated and the loop returns to step 1.

This is the inverse-compositional alignment pattern with a piecewise translation model. On a synthetic plane, from a starting shape a few percent of scale and a few degrees off, it converges in two or three iterations; each iteration is one render, nine small searches and one reading of the whole-member ZNCC, where the cascade's simplex search is one photograph sampling per objective evaluation. On captures, starting from the shape the cascade found, it almost always stops at its first iteration, because its first update lowers the score the cascade maximised.

The stored displacement keeps its definition, the displacement from where the returned shape places the cell. When the last update was applied, the last render's shifts are carried into the updated shape's grid, so the update's affine part is gone from them because the shape absorbed it, and what remains is the part no affine map matches. When the last update was rejected, or an oscillation kept the shape before it, the returned shape is the one the last render was made at, and the shifts are stored as measured, first-order part included, as in the measurement.

The affine update absorbs everything nine translations can express as one linear map: a shift, a scale, a rotation and a skew. The loop's exit reads the update's size and the score's direction rather than a score floor: a member on a non-planar surface stops at the best shape the score accepts and leaves a large, honest displacement, instead of iterating against a floor it cannot reach.

### Which cells are refused, and what the fit does then

A cell over flat or repeating texture registers anywhere. Its shift is noise, and its ZNCC may still be high because a flat region correlates with a flat region at every shift. The cell's reading says so: at the sizes the cluster-patches file uses, a cell is too small for a self-similarity ellipse, so the reading is the curvature of the cell's ZNCC over its shift search. A flat curvature refuses the cell. A low ZNCC at the optimum refuses it for the other reason, that it is over a different surface. The whole-member gate in the current refinement does the same for the member; this applies it per cell.

The affine map is fitted to the survivors. Six parameters need at least three non-collinear cells, and a trustworthy fit needs more. A rule on the count and the spread of the survivors chooses the model: a full affine with five or more well-spread cells, a similarity with three or more otherwise, a pure shift below that, and no fit when none survive, in which case all nine cells are stored as not attempted. "Otherwise" covers three or four cells, five or more that lie along one row or column and so pin only one axis of an affine, and five or more well-spread cells whose affine normal equations are singular: each falls back one step, to the similarity, before the shift.

A cell can pass both of its own gates and still be wrong: a repeated texture can give it a sharp, high-scoring peak at the wrong shift, and a depth edge inside the cell can give it a peak that belongs to the other surface. Such a cell disagrees with the affine map the others agree on. The map is therefore fitted by iteratively reweighted least squares: a first fit weighted by each cell's curvature, then three rounds in which each cell's weight is its curvature times a Tukey biweight of its residual to the previous round's fit. The residual scale is 1.4826 times the median residual, floored at 0.1 grid px so that cells agreeing to a few hundredths of a pixel do not turn a cell a tenth of a pixel off into an outlier, and the biweight's cut-off is 4.685 scales. A cell at or past the cut-off has weight zero and is stored as `refused_outlier`, with its shift. The model choice above reads the cells with a positive weight.

### Double resampling

The cell shifts are measured on a tile that is itself an interpolation of the photograph, and the stored displacement is read through that sampler. For a shift read as a sub-pixel ZNCC peak this is a small blur, not a bias, and the fine tier re-measures whatever it needs. It does argue for rendering the working patch with the per-view sampler choice of [sharper-patch-bitmap.md](sharper-patch-bitmap.md), since a view compressed along one axis loses most along that axis.

### From displacements to a normal

With poses, a cell's centre in a member is a pixel, and that pixel is a ray. The same cell in each kept member gives a bundle of rays whose nearest point is the cell's position in the world. Nine positions, minus refused cells, are the input to the piece-gated plane fit. The displacements were measured photometrically at refinement time; the normal costs no rendering at seed time. That is the property the seed needs, since its releases carry no frames and it renders nothing.

## Format

Per member, with readings only for kept members, four new entries in the cluster-patches file: `member_cell_shift_px` of shape `(members, 3, 3, 2)`, `member_cell_zncc` of shape `(members, 3, 3)`, `member_cell_status` of shape `(members, 3, 3)` with its legend `member_cell_status_names` in the section's metadata, following the convention `member_status` adopted in version 7, and `member_cell_iterations` of shape `(members,)`. They are optional entries of `.matches` version 8, present together with the legend or absent together with it, and a member that is not kept carries `NaN` / `NaN` / `not_attempted` / `0`. A reader of an older file has no cells and a consumer that needs them says so. Version 9 adds `refused_outlier` to the names the legend may state. The format's legend rule lets a file name any subset of the statuses the format defines, but a reader refuses a name it does not define, and a writer states the whole legend, so without the bump a version 8 reader would refuse a file that claims version 8. Why the loop stopped and whether its last update was applied are in the Rust result and the binding only; storing them in the file is a follow-up, if a consumer turns out to need them. The format side is specified in [formats/matches-file-format.md](../formats/matches-file-format.md) § "Per-cell entries"; the Rust types are `MemberCellData` and `ClusterCellStatus` in [cells.rs](../../crates/sfmtool-matches-format/src/cells.rs), filled from `ClusterRefineResult::cells` by `member_cell_data` in [piecewise.rs](../../crates/sfmtool-core/src/patch/cluster_refine/piecewise.rs). The binding takes `piecewise=True` and returns the four arrays; `sfm cluster-patches --piecewise` writes them and records the six settings in `refine_options` beside `piecewise`, `move_shape` among them. The format defines `member_cell_shift_px` as the displacement from where the member's stored shape places the cell, with no affine map removed, which is what both modes store.

## Implementation notes

- The working patch is rendered at the template's grid plus the margin, once per member without `move_shape` and once per iteration with it, and nothing in the cell search reads the photograph. A profile of the stage should show the render as the only cache-missing step.
- Without `move_shape` the stage never reads the whole-member ZNCC: `run_loop` returns after the one render and fit, before the score closure is called, and `refine_kept_member_cells` sees an unmoved member and stores only its cells. That is what keeps every other member output bit-identical to a run without the stage, which a kernel test checks, and it is why the stage costs one render and nine searches per kept member and nothing more.
- A failed render, no surviving cell, or a fitted map that reflects (`det A ≤ 0`) or is not finite leaves all nine cells not attempted after one render, in both modes; the member's shape is the cascade's either way.
- The cell search is a shift search only. The affine cascade has removed scale and rotation; a cell that needs more than a shift is over a different surface and should be refused, not fitted.
- At `patch_size = 12` a cell is 4×4 template px. That is enough for a shift registration against a textured template, and not enough for a self-similarity ellipse, which is why the per-cell gate reads curvature here and the self-similarity gate applies when the refinement runs at a larger size.
- The template is the cluster's reference member. The displacements are relative to that member's frame, so the reference member's own cells displace by zero by construction and carry no information; the normal fit uses the other members.
- The affine fit is a weighted least squares on at most nine points, run four times for the reweighting, in `f64`. With `move_shape` it is composed into the `f32` shape, because the update is small and its composition with the shape is where a near-identity matrix is multiplied repeatedly.
- A cell's search reads the window's sum and sum of squares from summed-area tables of the working patch, built once per render; the template side is mean-removed, so the cross term needs no window mean and is the only pass over the window per shift.
- The stage is off by default (`ClusterRefineParams::piecewise` is `None`, the binding's `piecewise=False`, `sfm cluster-patches --no-piecewise`). `--piecewise` turns it on at the default settings, which measure without moving the shape, and stores the cells.

### With `move_shape`

- The whole-member ZNCC the acceptance rule reads is the cascade's own evaluation, `eval_zncc` at the map the shape and position give, through the same pyramid-level choice and tile; the loop reads it once at the starting shape and once per iteration. A shape whose support leaves the frame cannot be read and counts as the lowest score, so its update is rejected. `ACCEPT_ZNCC_TOLERANCE` (`1e-4`) lets an update through that lowers the score by less than that from the current shape, so the loop can cross a flat stretch, but never below the score at the starting shape: a first version applied the tolerance per update, and on the subset the stored ZNCC of members the loop moved over several updates fell by up to 0.0004.
- The oscillation test compares the largest cell-centre movement of each update with the one before. A converging loop's movements shrink; a loop whose cells pull the shape back and forth between two readings repeats a movement of the same size, and continuing would end at whichever shape the cap happens to land on. Of the two shapes, the loop keeps the one with the higher whole-member ZNCC, the shape after the update on a tie.
- The stored displacement is exact to the returned shape. When the last update was applied, the last render measured cell `c` at `c + d` of its grid, and the returned shape's grid maps to that one by the last update `c ↦ A·c + b`, so the displacement is `A⁻¹·(d − (A·c + b − c))`. Subtracting the update's movement without the `A⁻¹` leaves an error of the size of `(A⁻¹ − I)` applied to the displacement. When it was not applied, the displacement is `d`.
- After a converged loop, the displacements follow the homography's second-order term less closely as the term grows. On the synthetic plane with `h = [0.003, −0.002]` the loop converges with its last update applied; the largest error over the nine cells is 0.095 grid px and the RMS 0.054, against true displacements of up to 0.63 grid px (the robust fit weights the corner cells, where the term is largest, below the others, so the affine it leaves follows the middle cells and the corners keep more of the term). At twice that, `h = [0.006, −0.004]`, the second update would lower the whole-member ZNCC and is rejected; the largest error is 0.29 and the RMS 0.12, against true displacements of up to 1.08 grid px.
- The loop's result is all or nothing when an iteration fails. When any iteration fails, the first or a later one, the member keeps its cascade shape and all nine cells are stored as not attempted; what earlier iterations fitted is discarded, because the cells were read at a shape that would not be the one returned. A rejected update is not a failure: the member keeps the shape the last render was made at, with that render's readings.
- After the loop moves a member, its whole-patch ZNCC, its parts and its shift from the seed are read again at the new shape, and the cascade's acceptance gates, `min_zncc` and `max_shift_px`, are applied to those readings. A member that fails either, or whose new support leaves the frame, keeps its cascade shape and readings, with all nine cells not attempted. The re-read ZNCC must also be at least the cascade's stored ZNCC: the loop's floor is its own reading of the starting shape, which can differ from the cascade's stored value in the last bits, and the stored value is what a consumer compares. The acceptance rule already keeps the score at or above the start and rejects a shape whose support leaves the frame, so these gates refuse only a member those last bits put under, or one the loop carried past `max_shift_px`.

## Determinism and precision

`f32` on tile intensities, matching the affine cascade; `f64` for the affine fit. Per-member work is independent and parallel; output equal within tolerance for any thread count, with the same cell statuses and the same iteration counts. Without `move_shape`, every member output other than the cells is bit-identical to a run without the stage.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `move_shape` | false | Whether the fitted affine map may move the member's shape, by the loop under [The shape-moving loop](#the-shape-moving-loop-move_shape). Off, the stage measures once at the cascade's shape. |
| `cell_shift_bound_px` | 2.0 | Search bound for a cell's shift from its affine placement, and the working patch's margin. |
| `min_cell_zncc` | 0.8 | A cell below this at its optimum is refused. The code's default (`DEFAULT_MIN_CELL_ZNCC`); provisional, since the fleet's cell ZNCC distribution has no valley to place it in. |
| `min_cell_curvature` | 0.02 | A cell whose ZNCC over the shift search is flatter is refused, in ZNCC per grid px². The code's default (`DEFAULT_MIN_CELL_CURVATURE`); provisional on the same terms. |
| `update_tolerance_px` | 0.05 | With `move_shape`: the loop stops when no cell centre moves by more. |
| `max_iterations` | 5 | With `move_shape`: iteration cap. Without it the stage renders once. |

Fixed in the code rather than exposed: `ACCEPT_ZNCC_TOLERANCE` = `1e-4`, three reweighting rounds, the Tukey cut-off of 4.685 residual scales, and the residual scale's floor of 0.1 grid px.

## Testing

**The measurement (default).** A synthetic planar cluster seen from a tilted view, with the starting shape perturbed by a known affine: the stage renders once, reads no whole-member ZNCC, leaves the shape and position at the start, fits all nine cells, and stores displacements that match the perturbation's affine, the displacement from the start's placement of each cell to where the plane puts it, within a fifth of a grid px; they are bit for bit the shifts the loop reads at its first render. Inside `refine_cluster_patches`, every member output but the cells, shape, position, ZNCC, its parts and shift, is bit-identical with the stage on and off, and the kept member reads one iteration and `Measured`. A cell whose shift is off the affine the other eight agree on is stored as `refused_outlier`, and every cell's shift is stored as read. A failed render leaves nine cells not attempted after one render.

**The loop (`move_shape`).** On the same cluster the loop recovers the affine within tolerance in at most three iterations with its last update accepted, the cell shifts follow the homography's second-order term within sampler tolerance, and all nine cells are fitted. The same cluster with one third of the template over a second plane: the three cells over it are refused, the update is fitted to the other six, and the recovered affine matches the first plane to 0.12 px, the whole-member ZNCC, which reads the second plane too, stopping the loop 0.10 px short of it. An update that would lower the whole-member ZNCC is rejected: the shape is unchanged and the stored shifts are the displacements from it, carrying the start's whole first-order error. An update that would carry the support out of the frame is rejected the same way. A loop whose updates alternate without shrinking stops at its second iteration with the shape of higher ZNCC. A cell whose shift is off the affine the other eight agree on gets weight zero and is stored as `refused_outlier`, and the update matches the inliers' affine. A member whose every cell is flat: no update, nine cells not attempted, cascade shape unchanged.

**Both.** Reading an older `.matches` file gives no cells and no error. On the fleet, the measurement leaves every member output bit-identical to the cascade's, and the seed's result on the ground-truth entries does not change. On the `seoul_bull_sculpture` ground truth, the normals the seed's writer derives from the cells have lower median error than the tilt solve on whole-member warps.

## Fleet results

The data is in [cluster-patches-piecewise-refinement-measurements.md](cluster-patches-piecewise-refinement-measurements.md).

- **The loop without the acceptance rule** (commit `f787be3b`, plain weighted least squares, [fleet conclusions](cluster-patches-piecewise-refinement-measurements.md#what-these-measurements-decide)): the loop moved 80% to 99.6% of kept members, by a median of 0.16 to 0.71 grid px and up to 10. It lowered every moved member's whole-member ZNCC, and 23.9% of members alternated at the cap. The refinement cost 1.6× as much, and the seed's `KerryPark480` pick turned from pass to fail.
- **The loop with the acceptance rule, the robust fit and the oscillation stop** ([subset run](cluster-patches-piecewise-refinement-measurements.md#subset-with-the-acceptance-rule-2026-10-08) on five entries):
  - The first update is rejected for 98.5% to 99.8% of kept members. Under 1.5% of members move, each to a shape of higher whole-member ZNCC, and no member's ZNCC falls.
  - Nothing reaches the cap or oscillates, and the cost is 1.12× to 1.29×.
  - `SeoulBull`'s seed result is unchanged. `KerryPark480`'s pick still fails, and its failure is reproduced by moving only the 55 members (0.34%) the loop moved there; the cells and the consistency residuals alone leave the seed exactly as the cascade does.
- **The measurement, the default** ([subset run](cluster-patches-piecewise-refinement-measurements.md#subset-with-the-shape-left-to-the-cascade-2026-10-08) on the same five entries): every member output other than the cells (shape, position, ZNCC, shift, consistency residual) is byte-for-byte the cascade-only run's on all five entries, the CPU cost is 1.09× to 1.17×, and the cell status shares match the loop's to 0.1 point. The seed on `SeoulBull` and `KerryPark480` releases the same candidates, byte for byte, as with the cascade file, so `KerryPark480`'s pick `h00` passes again.

The stage stays off by default (`--no-piecewise`) until a consumer reads the cells; when it is on, it measures and does not move the shape.

## Non-goals

A normal in the cluster-patches file. The file is pose-free and stays so; a normal is derived by whoever holds poses.

Fitting cells with a full affine. A cell that needs one is over another surface.

## Open questions

- **Shift only, or shift and scale.** Scale per cell would read the perspective term's radial component; it also costs a two-dimensional search per cell. Start with shift.
- **Are the moved shapes better or worse?** The loop moves under 1.5% of kept members, each to a higher whole-member ZNCC, and on `KerryPark480` those members change the seed's pick. Scoring the moved members' shapes against the `kerry_park` ground truth, or finding which of them carry the seed's choice, would say whether `move_shape` should ever be on, or whether the cascade's simplex should itself search further.
- **Replace or follow the cascade.** If the loop is ever turned on, it can start from the cascade's shape, or from the detection's shape with the cascade removed. Measuring how many iterations it needs from the detection alone would say whether the cascade still earns its cost.
