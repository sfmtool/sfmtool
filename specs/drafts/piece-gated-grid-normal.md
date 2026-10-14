# Piece-Gated Grid Normal

**Status:** Draft. Decided:
- the bench's *Grid Plane Normal* is the baseline this draft improves on, because the photometric normal search is known to be the weaker estimator, which is why Surface Tiling Mode runs the Grid Plane Normal and not *Fit Normal*;
- each part of the patch is measured by registering one cell of the three-by-three ZNCC grid split, by a shift search on one render per view, the measurement the piecewise refinement of cluster patches makes, instead of by a full bench fit per piece;
- each cell is gated in each view before any geometry, by the piecewise refinement's rules (peak curvature, cell ZNCC, search bound, and a robust affine fit per view), and those rules are shared code with the piecewise refinement, not a copy;
- the surviving cells are triangulated and a plane fitted through them with the weighting, residual floors, Tukey loop and three-way determinacy verdict of the [cell plane normal kernel](../core/patch/cell-plane-normals.md), with that math factored out so it takes per-view pixels from a track as well as a cluster file's columns;
- the patch is turned to the fitted normal and the measurement repeated, under the acceptance and stopping rules of the piecewise refinement's shape-moving loop;
- the split is the nine cells of the ZNCC grid, without overlap;
- once this estimator covers the callers of the photometric normal search, that search is retired together with the fronto-parallel cache and the D-optimal view subset that exist only to serve it.
- the working grid follows the reference view's footprint (decided by the maintainer): with `working_resolution` `None` its side is the patch's longer side in the reference view's photograph pixels at the starting frame, about one sample per photograph pixel, rounded up and clamped to `R ..= 2R` for `R` the resolution the track's bitmap is rendered at; the cap because one sample per pixel is unbounded against the patch and the cost grows with the square of the side, the floor so the grid is never coarser than the bitmap's. It is chosen once per step, so every render of the loop is on one grid;
- the step changes the normal only on a good estimate (decided by the maintainer): only a loop that ends `Converged` with at least one axis fixed turns the patch (`GridPlaneOutcome::Turned`). Every other ending leaves the normal exactly as the step was given it, even when earlier turns were accepted, and is reported, not refused: `Undetermined` when the cells cannot constrain the normal (verdict none, or no view scored), `Unstable` when they constrain it but the loop did not settle (a rejected turn, an oscillation, or the render cap, including a cap reached with every turn accepted, because the step wants stable convergence).

Not decided: which score the loop's acceptance rule reads; whether the step moves the patch's centre onto the fitted plane; whether a cell registration cannot settle falls back to a full piece fit; whether the first iteration can read the localizer's tiles. See [Open questions](#open-questions).

Built: the shared kernels (#954) and the bench step `grid_plane_normal` in [bench/normal/grid_plane.rs](../../crates/sfmtool-core/src/bench/normal/grid_plane.rs), with its tests, specified in [editable-track.md](../core/bench/editable-track.md) § "Estimating the normal". It stands beside `PieceLayout::Grid`, which is unchanged, so the two can be measured against each other; Track View's button, Surface Tiling Mode, the wire's `fit_bench_track_normal` and the Python bindings still run `PieceLayout::Grid`. The step answers two open questions provisionally, the acceptance score and moving the centre; the density, decided by the maintainer with the synthetic evidence recorded below, is built as decided; see [What the built step chose](#what-the-built-step-chose). Remaining: the measurement against today's Grid Plane Normal and the hand review, then the moves listed under [Retiring the photometric search](#retiring-the-photometric-search).

Amends:
- [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal": `PieceLayout::Grid` of `finite_difference_normal`, the *Grid Plane Normal* step
- [gui/track-view.md](../gui/track-view.md) § "Surface Tiling Mode": the Grid Plane Normal step of the tiling sequence runs this estimator
- [core/patch/cell-plane-normals.md](../core/patch/cell-plane-normals.md): its triangulation, plane fit and `NormalDeterminacy` are factored out and shared
- [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md#piecewise-refinement): the cell registration, the cell gates and the robust affine fit are factored out and shared
- [core/patch/patch-normal-refinement.md](../core/patch/patch-normal-refinement.md), [core/patch/fronto-parallel-patch-cache.md](../core/patch/fronto-parallel-patch-cache.md), [core/patch/patch-normal-refine-view-subset.md](../core/patch/patch-normal-refine-view-subset.md): retired when their callers move
- [cli/reconstruction/embed-patches-command.md](../cli/reconstruction/embed-patches-command.md) and [cli/reconstruction/xform/refine-normals-command.md](../cli/reconstruction/xform/refine-normals-command.md): the normal step they run

Related drafts: [surface-co-solve.md](surface-co-solve.md) § "Normal estimators" lists the family this estimator belongs to and should defer to this draft for the grid estimator. [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) decides when a cell plane normal may be read, which applies to this estimator's output too. [two-tier-patch-density.md](two-tier-patch-density.md) chooses the density the patch is stored at, which bears on the density this estimator renders at. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) decides how views are scored against a bitmap, which bears on the loop's acceptance score.

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in the world, and its normal says which way the square faces. The normal decides how every photograph is resampled into the patch's bitmap, so a wrong normal blurs the bitmap, shifts the keypoints that are localised on it, and misleads every photometric reading taken from it afterwards.

The project has two production ways of estimating a normal. The photometric search turns the normal through two degrees of freedom and keeps the orientation at which the views agree best; [surface-co-solve.md](surface-co-solve.md) records a median error of 16 to 23 degrees for it at the true position in the surface co-solve harness, and it is weak where the texture is weak. The other is the bench's *Grid Plane Normal*, which reads depth by parallax instead of reading the shape of a correlation peak over orientation. Because the photometric search is the weaker of the two, Track View's Surface Tiling Mode runs the Grid Plane Normal in its sequence: Fit, Find matches by geometry, Fit, Grid Plane Normal, Fit, Commit ([gui/track-view.md](../gui/track-view.md) § "Surface Tiling Mode"). This draft improves the Grid Plane Normal.

### The Grid Plane Normal today

`finite_difference_normal` with `PieceLayout::Grid` in [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), specified in [editable-track.md](../core/bench/editable-track.md) § "Estimating the normal":

1. It cuts the patch into `pieces × pieces` square pieces that together span its side. Track View's default is 2×2 with no overlap (`SplitSettings::default` in [bench.rs](../../crates/sfm-explorer/src/bench.rs) takes `MIN_PIECES`, which is 2, and an overlap of 0), and Surface Tiling Mode cuts the patch with the split settings Track View holds.
2. Each piece is a copy of the track resized about its centre and slid along the plane (`resize_patch`, `translate_patch`), and is given a full bench `fit`: localize and sub-pixel refine every sighting against the reference render, re-triangulate, render the bitmap, and evaluate. The fit runs with `FitOptions::default()`, whose localizer is `open_localizer()`, so the member self-similarity gate and the ZNCC bars are off.
3. A piece whose fit fails, or which the fit puts at infinity, is left out.
4. The normal is that of the unweighted least-squares plane through the fitted centres: the direction they spread in least. When the centres lie along a line, the line fixes only the turn about its perpendicular, and the normal is the patch's own with its component along the line removed.
5. The patch is turned to that normal by `tilt_patch`, and the track is read back as a fit ends. Nothing tests whether the turn made the track better.

It has five faults this draft addresses:

- **No per-piece trust gate.** A piece over a flat or repeating part of the patch fits to an arbitrary depth, and with the localizer's gates off nothing refuses it.
- **Equal weights.** A piece pinned by a wide baseline and one pinned by a narrow one pull on the plane equally.
- **No outlier rejection.** One wrong centre among four tilts the plane, and the plane's residual gives no sign of it, because a tilted plane through three good centres and one bad one and a flat plane through four noisy ones both fit four numbers.
- **No acceptance rule.** The patch is turned whether or not the turn improves it.
- **Cost.** It runs `pieces²` full fits, and every fit goes back to the photographs several times: to render each view's search tile, to refine, to render the bitmap, and to evaluate.

The whole patch can read as well textured while one corner or one half is flat or lies on another surface, which is the usual case on a depth edge, an occluding contour, or a patch that straddles texture and sky. These are the patches whose normals the faults above get wrong.

### The proposal

Measure each part of the patch by a shift registration of one cell on one render per view, the measurement the piecewise refinement of cluster patches makes ([cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md#piecewise-refinement)). After the one render per view, the registration reads only a small tile held in cache, so the step never goes back to the photograph to resample, which is what makes it faster than a fit per piece. Gate every cell in every view by the piecewise refinement's rules before any geometry, triangulate the surviving cells and fit a weighted, robust plane through them as the [cell plane normal kernel](../core/patch/cell-plane-normals.md) does, then turn the patch and measure again until the cells stop moving.

### Why this matters for the seed

The seed stage produces up to eight candidate reconstructions per capture and then chooses among them. Scored against the eight approved ground truths, a correct candidate exists in 7 of 8 captures and the seed's own rule picks it in 4 of 8. The rule reads only geometric residuals, and those cannot distinguish a right camera layout from a collapsed one at the same reprojection error. Photometric scoring of candidates is the independent signal that can, and it needs patch normals that are right often enough to be trusted.

## Rust API

The bench step replaces `PieceLayout::Grid` in [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), beside `fit_normal` and `finite_difference_normal`, which keeps `PieceLayout::Cross`. It builds on two crate-visible kernels in `patch/`, each factored out of an existing module rather than written again:

- **Cell registration**, [patch/cell_registration.rs](../../crates/sfmtool-core/src/patch/cell_registration.rs), factored out of [cluster_refine/piecewise.rs](../../crates/sfmtool-core/src/patch/cluster_refine/piecewise.rs): the `WorkingPatch` with its summed-area tables (`WorkingPatch::new`), the per-cell shift search with its sub-pixel peak and curvature (`read_cell`, `read_cells`), the three cell gates (the bound in `CellLayout`, the two bars in `CellGateOptions`), and the robust affine fit (`fit_update_robust`, `fit_pass`). The piecewise refinement calls the same functions, with its `PiecewiseParams` supplying the bound and the two bars.
- **Cell triangulation and plane fit**, [patch/cell_plane_fit.rs](../../crates/sfmtool-core/src/patch/cell_plane_fit.rs), factored out of [cell_plane_normals.rs](../../crates/sfmtool-core/src/patch/cell_plane_normals.rs): `PosedView::ray_through` turns a pixel of a posed view into a ray carrying the noise floor the caller gives it, and `fit_cell_plane` takes the nine cells' rays and returns a `CellPlaneNormal`: each cell's position, covariance and residual (its private `triangulate`), and the covariance- and Tukey-weighted plane with its `NormalDeterminacy` (`weighted_plane`, `combined_weights` and the `determinacy` rule). `cell_plane_normals` keeps its public signature and builds its rays and their floors from the cluster file's columns; the bench step builds them from a track.

```rust
/// How [`grid_plane_normal`] measures the cells and when its loop stops.
pub struct GridPlaneNormalOptions {
    /// Search bound for a cell's shift, grid px of the working grid; also the
    /// margin the context tile is rendered with. As the piecewise refinement's
    /// `cell_shift_bound_px`, 2.0.
    pub cell_shift_bound_px: f32,
    /// A cell whose ZNCC at its optimum is below this is refused in that view.
    /// `DEFAULT_MIN_CELL_ZNCC`, 0.8.
    pub min_cell_zncc: f32,
    /// A cell whose ZNCC peak is flatter than this is refused in that view.
    /// `DEFAULT_MIN_CELL_CURVATURE`, 0.02 ZNCC per grid px².
    pub min_cell_curvature: f32,
    /// Side of the working grid the views are rendered on. `None` renders at
    /// the reference view's footprint of the patch, clamped to `R ..= 2R`.
    pub working_resolution: Option<u32>,
    /// Triangulation, floors, Tukey loop and determinacy, as the cell plane
    /// normal kernel reads them.
    pub plane: CellPlaneParams,
    /// The loop stops when a turn moves no cell centre by more than this, grid px.
    pub update_tolerance_px: f32,
    /// The most renders the loop makes.
    pub max_iterations: u8,
    /// The reading the step ends with.
    pub fit: FitOptions,
}

/// What the cells said, carried by `NormalEstimate::GridPlane`.
pub struct GridPlaneCells {
    pub determinacy: NormalDeterminacy,      // shared with cell_plane_normals
    pub cell_status: [CellPlaneStatus; 9],   // in the plane, plane outlier, too few rays, …
    pub cell_rays: [u32; 9],                 // rays each cell was triangulated from
    pub cell_weight: [f64; 9],
    /// Per observation of the track, each cell's registration status in that
    /// view at the last render: fitted, refused for curvature, ZNCC, bound,
    /// as an outlier of the view's affine fit, or not attempted.
    pub view_cells: Vec<[CellStatus; 9]>,
    pub plane_rms: f64,
    /// Distance of the fitted plane from the patch centre along the normal.
    pub plane_offset: f64,
    pub iterations: u8,
    pub stop: LoopStop,                      // the piecewise loop's stop reasons
}

pub fn grid_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GridPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError>;
```

**Why this shape.** The signature is the other two normal steps', so Track View's button, the wire's `fit_bench_track_normal` with `method` `grid_plane`, and Surface Tiling Mode's step call it the way they call `finite_difference_normal` today, and it ends with the same `NormalReport`. The split is fixed at the ZNCC grid's nine cells, so there is no `pieces` or `overlap`: the cells are the ones `grid_cell_centres` defines for every other per-cell reading, and the cell gates and the plane fit are only defined on that split. Track View's *per axis* and *overlap* boxes stay for *Finite Diff Normal*. The report keeps every cell's verdict and every view's per-cell status, because a wrong normal is diagnosed by asking which cells were trusted and why each view refused the others, and the statuses reuse `CellStatus`, `CellPlaneStatus` and `LoopStop`, so the bench, the cluster file and the cell plane kernel use one set of words. `NormalDeterminacy` is the cell plane kernel's type, so a caller choosing between the two estimators reads one verdict. The settings are `*Options` because a caller chooses them; `CellPlaneParams` keeps its existing name.

**Example.** Surface Tiling Mode's Grid Plane Normal step calls `grid_plane_normal(&track, &edited, &images, &GridPlaneNormalOptions::default(), &progress)`, and Track View shows the nine cell verdicts as a three-by-three overlay on the patch bitmap and turns the patch only about the axes the cells fixed.

## Theory

### The measurement

**Render.** Each `in` view is rendered once at the track's current frame, centred on the view's keypoint, through the patch's plane: the patch re-anchored in its plane on the keypoint's ray, and every grid sample projected into the photograph by the camera model, the render `WarpMap::from_patch` makes for every bench tile. It is not an affine approximation. The render covers the working grid plus a margin of `cell_shift_bound_px` on every side, as the piecewise refinement's working patch does, and it is the only read of the photographs. The reference view is rendered the same way, without the margin, and its nine cells are the templates.

**Register.** Each of the nine cells of each view is compared with the same cell of the reference render by windowed ZNCC at every whole shift within the bound. At the best whole shift a quadratic fitted to the three-by-three neighbourhood gives the sub-pixel shift, and the smaller eigenvalue of its negated Hessian is the peak's curvature: how sharply the cell is pinned.

**Gate.** A cell is refused in a view before any geometry, by the piecewise refinement's rules: for curvature when the reference is flat over it or its peak is flatter than `min_cell_curvature`, which catches flat and repeating texture; for ZNCC when its score at the optimum is below `min_cell_zncc`, which catches a cell over a different surface; and for the bound when its best whole shift lies on the search bound. The surviving cells of the view are then fitted with the robust affine map of the piecewise refinement, curvature-weighted IRLS with a Tukey biweight, and a cell the map gives no weight is refused as an outlier in that view. A repeated texture can give a cell a sharp, high-scoring peak at the wrong shift, and only its disagreement with the other cells exposes it.

A cell is refused per view, not per patch: a cell occluded in one view keeps its rays from the others.

### Why a displacement here is an error

In the cluster file a cell's displacement is measured against an affine shape, so it carries the shape's first-order disagreement with the cells and the homography's second-order term as well as any parallax. Here the render is induced by the patch's plane, so if the normal and the depth are right, every cell of every view lands where the reference's does and its displacement is zero up to noise. A displacement measures what remains wrong with the current normal and depth, and a view's common shift says its keypoint is off. That is what lets the step loop: it is the inverse-compositional pattern of the piecewise refinement's `move_shape` loop, with the plane in place of the affine shape.

### Triangulating the cells

A surviving cell's content in view `i` lies at its centre plus its displacement on the working grid, and the render's map takes that grid position to a pixel of the photograph, and so to a ray. The reference contributes its cell centre undisplaced. The rays of one cell across the views are triangulated as [cell-plane-normals.md](../core/patch/cell-plane-normals.md) § "Triangulating a cell" does: the depth-weighted nearest point of the rays in pixels, refused for too few rays, too narrow a baseline, or a point behind a camera; its residual floored at the precision of a measured shift mapped into each view, with the render map's local Jacobian taking the place of the cluster file's `(patch_size / R) · ‖S‖`; and its covariance `σ² Λ⁻¹`.

### The plane and its determinacy

The plane through the triangulated cells is the cell plane kernel's: weighted by the inverse of each cell's variance along the normal times its Tukey weight, over `irls_iters` passes on standardised residuals. Its verdict is `NormalDeterminacy`: both axes when the live cells spread in two directions within the plane, one axis with the line named as the free axis when they lie along a line, and none with fewer than `min_cells` live cells or no spread. Under one axis the step turns the patch only about the fixed axis and keeps its rotation about the free axis where the patch had it, as today's line rule does; under none it does not turn the patch.

### Turning and looping

The step turns the patch to the fitted normal with `tilt_patch`, which keeps the centre, rebuilds each sighting at the in-plane offset it had, and stops `MAX_TILT_DEG` (80°) from any observation. It then renders and registers again. A turn is kept only when the score does not fall, the piecewise loop's acceptance rule, with the same tolerance and the same floor at the starting score; which score is an open question. The loop stops when a turn moves no cell centre by more than `update_tolerance_px`, at `max_iterations` renders, when a turn's largest cell movement does not shrink from the one before (oscillation, keeping whichever of the two frames scores higher), or when a turn is rejected. With a normal far from the truth, cells near the patch's edge register at or past the bound; they are refused for the bound on the first render and return as the turns bring them inside it.

The step ends as today's does, by reading the track back with the fit's evaluation over the turned square.

### The patch that is good in parts

A patch on one textured plane passes everywhere, and the gates change nothing. The patches the gates exist for sit where a surface ends. There the cells over the surface register and the cells over the background are refused for curvature, for ZNCC, or as outliers of the view's affine fit; a cell that passes in every view but lies off the plane the others agree on is refused by the robust plane fit. The normal is the plane of the survivors, even when they are one half of the patch, and when they lie along one row or column the verdict says one axis is free rather than filling it from refused cells.

### Retiring the photometric search

The photometric normal search has three production callers: the embed-patches rounds, `sfm xform --refine-normals`, and the bench's *Fit Normal*. When each has moved to this estimator, the search and its two support modules, the fronto-parallel cache and the D-optimal view subset, have no caller and are deleted. The fronto-parallel prior goes with them; a sweep over that prior found it degraded normals everywhere it was applied.

The order is: build the shared kernels and the bench step and measure the step against today's Grid Plane Normal; move Surface Tiling Mode and the Track View button; move embed-patches; move `xform --refine-normals`; delete.

## Implementation notes

- **The localizer's context tile is the same kind of render, and is not reusable as it stands.** The keypoint localizer renders, per view, a context tile through `render_context` ([keypoint_localize.rs:342](../../crates/sfmtool-core/src/patch/keypoint_localize.rs)): the patch re-anchored on the view's starting keypoint by `shifted_center`, scaled to `R + 2·margin` samples, and projected by `WarpMap::from_patch`, one sample per grid pixel. That is a plane-induced render on the patch grid with a margin, as step one needs, and its margin, `⌈search⌉` (6 at the default `search`), covers the cell bound of 2. But the tile cannot be read by this step for three reasons. It is private and lives only inside the per-view loop of the alignment ([keypoint_localize/align.rs:321](../../crates/sfmtool-core/src/patch/keypoint_localize/align.rs)), dropped once that view is searched. It is rendered at the frame and keypoints the track had before the fit, while the step runs after a fit has re-triangulated and moved the centre. And it is centred per-channel `f32` planes with a separate validity map, where the cell search reads an interleaved working patch with `NaN` for missing samples. What is shared is the render path, `render_context`'s use of `shifted_center` and `WarpMap::from_patch`, not the tiles. Whether a first iteration run inside a fit could read them is an open question.
- `grid_cell_centres` and `grid_bounds` live in `patch/normal_refine/support.rs` with `pub(in crate::patch)` visibility. They outlive the photometric search, so retiring the search keeps them, and the shared kernels belong in `patch/` where they are visible; the bench step in `bench/` calls those kernels, not the helpers.
- The piecewise refinement renders its working patch through `sample_grid_window` from an affine shape. The plane-induced render produces the same interleaved samples, so the shared code starts at `WorkingPatch::new` and the two stages differ only in how they render.
- Factoring out of `cell_plane_normals` must leave its output unchanged, and factoring out of the piecewise refinement must leave a `.matches` file bit for bit what it was; both kernels' existing determinism tests check it.
- The glossary's **working patch** entry names both renders: the affine render of the piecewise refinement and the plane-induced **context tile** this step renders, the localizer's word for the same kind of render. The glossary's **grid-plane normal** entry names `grid_plane_normal` beside `PieceLayout::Grid` for now; when the layout is retired it no longer has pieces, `pieces` or `overlap`.
- A track at infinity is refused as today, by `normal_preconditions`.

## Determinism and precision

Renders and cell scores are `f32` on tile intensities, as the piecewise refinement's are; the sub-pixel peak, the affine fits, the triangulation and the plane are `f64`, as in the two kernels they come from. Each view's render and registration is independent and can run in parallel, with no randomness and a fixed pass count; the output is equal for any thread count. No content hash depends on it.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `cell_shift_bound_px` | `2.0` | Shift search bound and render margin, working grid px; the piecewise refinement's default |
| `min_cell_zncc` | `0.8` | A cell below this ZNCC in a view is refused in that view (`DEFAULT_MIN_CELL_ZNCC`) |
| `min_cell_curvature` | `0.02` | A cell whose peak is flatter in a view is refused in that view (`DEFAULT_MIN_CELL_CURVATURE`) |
| `working_resolution` | `None` (the reference view's footprint, clamped to `R ..= 2R`) | Side of the working grid; decided by the maintainer, see Status |
| `plane` | `CellPlaneParams::default()` | `include_refused_outlier` false, `min_rays` 2, `min_triangulation_angle_deg` 2.0, `cell_shift_precision_grid_px` 0.1, `ray_noise_floor_px` 0.05, `irls_iters` 3, `tukey_c` 4.685, `min_cells` 3, `det_aniso` 0.10 |
| `update_tolerance_px` | `0.05` | The loop stops when no cell centre moves by more; the piecewise loop's default |
| `max_iterations` | `5` | Render cap; the piecewise loop's default |

The two cell bars are the piecewise refinement's provisional defaults, set on synthetic tests; they are not measured for this use.

## Testing

Kernel tests on synthetic planes rendered through known poses:

- A textured planar patch started 10° to 20° off its normal: the loop converges, all nine cells are in the plane, both axes are fixed, and the normal is within 0.5° of truth.
- The same patch with one third replaced by a flat region: those three cells are refused for curvature in every view, both axes are still fixed, and the normal is within 1° of truth.
- A patch whose halves lie on two planes at different depths: the cells of one half are refused, by ZNCC, as affine outliers or by the plane fit, the verdict is one axis, the free axis lies along the surviving half's row or column, and the fixed axis agrees with that half's plane.
- A patch over flat texture throughout: no cell survives, the verdict is none, and the patch is not turned.
- A turn that lowers the acceptance score is not kept.
- The piecewise refinement's and the cell plane kernel's tests pass unchanged after the factoring, and a `.matches` file written before and after is identical.

Measurement against today's Grid Plane Normal, at its Track View default of 2×2 pieces, on the `seoul_bull_sculpture` and `kerry_park` ground truths: the normal error of each against the ground-truth normals, the share of tracks given each verdict, and the time each takes per track. Then a hand review in SfM Explorer: Surface Tiling Mode run with each estimator over the same patches, the turned patches compared by a person without knowing which estimator turned which. The estimator replaces today's only if the review prefers it; a lower error on the ground truths is not enough on its own.

## Non-goals

**Joint estimation of depth and normal.** The step turns the patch about its centre and reports the plane's offset; placing the centre is the following Fit's work in the tiling sequence, unless the open question below decides otherwise. The joint solve is the subject of [surface-co-solve.md](surface-co-solve.md).

**Normals before poses exist.** A normal needs sighting rays. Without poses, the per-cell displacements of the piecewise refinement are what is measured, and [cell-plane-normals.md](../core/patch/cell-plane-normals.md) turns them into a normal once poses exist.

**Moving the sightings.** The loop's renders follow each view's keypoint as `tilt_patch` rebuilds it; the step does not write a view's common cell shift back to its keypoint. Placing keypoints is the fit's work.

## What the built step chose

The bench step follows the Rust API above, with these differences and with these choices where the draft left the detail open. Each is provisional until the measurement.

- **Names.** `NormalEstimate::GridPlane` still names `PieceLayout::Grid`'s report, so the new report is carried by `NormalEstimate::GridPlaneCells(GridPlaneCells)`. `NormalEstimate` is no longer `Copy`, since `GridPlaneCells` holds a `Vec`. When `PieceLayout::Grid` is retired the variant can take the plain name.
- **Two scores in the report.** `GridPlaneCells` also carries `score_before` and `score_after`, the acceptance score at the start and at the last render the loop kept. Its readings are those of the last render the loop kept, not of a rejected render.
- **The acceptance score** is the mean over the non-reference views of the windowed ZNCC of each view's tile at zero shift against the reference's render, over the samples both carry and averaged over the reference's textured channels. The window is the cluster refinement's default, `PatchWindow::GaussianDisk { sigma: 0.5 }`, the window the piecewise loop's whole-member ZNCC reads with, so the rule this step copies reads the same kind of score. This is the cheapest score the step already has the tiles for, and it answers the open question provisionally. On the textured plane and the plane with a flat column, every measured angle is the same as with an unwindowed whole-tile ZNCC to the stated precision. On the synthetic depth step it fails: at one sample per photograph pixel the first turn toward the near plane lowered the score from two of five starts, and the patch was not turned, because the window weights the tile's middle and the middle is the far plane; the unwindowed score failed from one of the five, a different start. One of the two is 12 degrees about the axis the column fixes with 6 about the column, the start the depth-step test used under the unwindowed score; on the one-sample-per-pixel grid the test started 15 degrees about the axis the column fixes with -8 about the column, one of the three starts the windowed score turned from there. On the default footprint grid that start ends on a rejected turn too, so the test now runs there from 20 degrees about `-x` with 4 about the column, one of the two of five starts that converge. A score over the cells the gates kept would not read the refused ones.
- **The reference** is the fit's choice (the reference the track holds where its row is `in` with a keypoint, else the reference-view rule's pick), made once when the step starts and kept through the loop. One row per image takes part.
- **The render** places the context tile with `OrientedPatch::anchored_at_keypoint`, the same re-anchoring `shifted_center` of the localizer's offset gives, and reads the cell positions through `shifted_center`. The sampler is the one `fit.refine.sampler` picks from the view's Jacobian at the working resolution, as for the bench's tiles. A row with no keypoint is rendered at the frame's own projection.
- **The working resolution** `None` follows the reference view's footprint, as decided above. The footprint is read from the reference render map's Jacobian at the patch centre (`patch_grid_jacobian` on the `R`-sample grid at the reference's keypoint): its two columns are the patch's `u` and `v` axes in photograph px per grid px, and the side is `R` times the longer column, rounded up, clamped to `R ..= 2R`, and `R` where there is no reference or no Jacobian. `R` is the reconstruction's patch resolution where it stores bitmaps and otherwise `fit.refine.resolution`, at least 3, and is also the resolution the reference rule renders at. `GridPlaneCells::working_resolution` reports the side. On the tests' scene it comes out 33, for a patch facing the cameras and for one tilted 10 to 20 degrees alike: the footprint reads a fraction over 32 px (32.0000005 facing, 32.00002 to 32.11 tilted), and rounding up takes it to 33.
- **A view whose affine fit fails** (reflects or is not finite) keeps none of its cells, which read `NotAttempted`, as the piecewise refinement stores a failed fit.
- **The reference's row** in `view_cells` reads `Fitted` for a textured template, `RefusedCurvature` for a flat one and `NotAttempted` for one partly off its photograph; it contributes a ray only for a textured template.
- **The ray floor** uses the render map's local Jacobian by central differences of half a grid px on each axis at the cell's registered position.
- **A turn's cell movement** is `2ρ sin(θ/2)` for a turn of `θ` (the angle `tilt_patch` reached) and `ρ` the farthest cell centre from the patch centre, in grid px: the chord the turn carries that centre through.
- **The loop's order.** A turn under `update_tolerance_px` is taken without a render to score it (`Converged`). At `max_iterations` renders the last render's turn is not taken, having no render to score it (`Cap`). A rejected turn is rendered, so it counts as a render. A start that no view can be scored at stops the loop at once (`NotRun`), and cells that fix no axis stop it where it is (`Measured`). The piecewise loop instead scores even a converged update, because its score does not need a render.
- **Under one axis** the target normal is the patch's own with its component along the free axis removed, not the plane kernel's one-axis normal, which takes its turn about the line from the mean viewing direction.
- **The centre does not move**, as the open question's first option; the plane's offset is reported.
- **Not parallel.** The views are rendered and registered one after another, so the per-render cancel check stays simple.

### Synthetic evidence on the density

On the tests' scene (three pinholes 4 units from a textured plane, baselines 0.6 to 1.0 units, a patch whose side spans 32 photograph px), the angle between the step's normal and the plane's, from starts 10 to 20 degrees off, by working grid:

| Working grid | Photograph px per grid px | Textured plane | One column flat |
|---|---|---|---|
| 12 | 2.7 | 3.2 to 4.0° | |
| 24 (the bitmap grid, `R`) | 1.33 | 0.67 to 0.90° | 1.4 to 3.3° |
| 28 | 1.14 | 0.32 to 0.43° | 0.40 to 0.64° |
| 32 | 1.0 | 0.10 to 0.27° | 0.09 to 0.24° |
| the footprint (33 here, the default) | 0.97 | 0.37 to 0.53° | 1.27 and 1.86° from the two of four starts that converge; the other two unstable and unchanged |
| 40 | 0.8 | 0.18 to 0.28° | 0.31 to 0.46° |
| 48 | 0.67 | 0.18 to 0.26° | |

Rendering with a fixed bilinear sampler instead of the sampler rule's choice changed the 24-sample results by under 0.03°, so the loss is the grid's coarseness against the photograph and not the sampler. The error falls until the grid reaches about one sample per photograph pixel and is flat beyond, which favours a working grid that follows the reference view's footprint over one fixed in samples. The footprint grid, 33 samples on this scene, does less well than exactly 32, so the error is not a smooth function of the density alone; how the `[0, R/3, R − R/3, R]` split falls may enter. The tests hold the default grid to 0.6 degrees on the textured plane (the draft's 0.5 holds from three of four starts) and run the flat-column case at exactly 32. On the depth step the default grid turns from two of five starts, the fixed axis ending 0.06 and 0.17° off, and leaves the other three unchanged as unstable. This is one scene; the ground-truth measurement decides.

## Open questions

- **The acceptance score.** The windowed ZNCC of each view's whole tile against the reference render at the zero shift, read from the renders the loop already made, is cheap and is the cluster loop's choice in kind. The bench's evaluation, the blur-matched score the bars judge, is what a person reads, but costs another pass over the photographs per iteration. Start with the reference ZNCC, and compare the turns each accepts. The built step starts with the windowed reference ZNCC, provisionally, and has already shown it refusing a correct turn when most of the tile is refused cells (see [What the built step chose](#what-the-built-step-chose)); a score over the kept cells is the next candidate.
- **Moving the centre.** The fitted plane may lie off the patch's centre. Keeping the centre, as `tilt_patch` does, leaves depth to the following Fit; moving it to where the reference ray meets the plane would let the loop correct depth and normal together, and the displacements would then tend to zero. The built step keeps the centre and reports the offset.
- **A fallback to a full piece fit.** A cell registration cannot settle, for example one at the search bound when the normal is far off, could be given today's full piece fit as a fallback. A second loop iteration, after the first turn has brought the cell inside the bound, may make it unnecessary. Decide by counting, on the ground truths, the cells that end the loop refused for the bound.
- **Reading the localizer's tiles.** The first iteration of the loop could read the context tiles a fit's localizer renders, if the localizer kept them and the step ran inside the fit, at the frame the fit started from. Every later iteration renders again anyway. Whether saving one render per view is worth the coupling is to be measured.
- **Fewer than three live cells.** The step leaves the patch unturned. Where the reconstruction is dense, the [adjacency surfel](../core/analysis/adjacency-surfel-normals.md) estimate from the point's neighbours could stand in; on the bench, leaving the patch is the only answer.
