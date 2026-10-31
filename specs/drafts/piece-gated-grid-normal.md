# Piece-Gated Grid Normal

**Status:** Draft. Decided:
- the bench's *Grid Plane Normal* is the baseline this draft improves on, because the photometric normal search is known to be the weaker estimator;
- each part of the patch is measured by registering one cell of the three-by-three ZNCC grid split, by a shift search on one render per view, the measurement the piecewise refinement of cluster patches makes, instead of by a full bench fit per piece;
- each cell is gated in each view before any geometry, by the piecewise refinement's rules (peak curvature, cell ZNCC, search bound, and a robust affine fit per view), and those rules are shared code with the piecewise refinement, not a copy;
- the surviving cells are triangulated and a plane fitted through them with the weighting, residual floors, Tukey loop and three-way determinacy verdict of the [cell plane normal kernel](../core/patch/cell-plane-normals.md), with that math factored out so it takes per-view pixels from a track as well as a cluster file's columns;
- the patch is turned to the fitted normal and the measurement repeated, under the acceptance and stopping rules of the piecewise refinement's shape-moving loop;
- the split is the nine cells of the ZNCC grid, without overlap;
- once this estimator covers the callers of the photometric normal search, that search is retired together with the fronto-parallel cache and the D-optimal view subset that exist only to serve it.
- the working grid follows the reference view's footprint (decided by the maintainer): with `working_resolution` `None` its side is the patch's longer side in the reference view's photograph pixels at the starting frame, about one sample per photograph pixel, rounded up and clamped to `R ..= 2R` for `R` the resolution the track's bitmap is rendered at; the cap because one sample per pixel is unbounded against the patch and the cost grows with the square of the side, the floor so the grid is never coarser than the bitmap's. It is chosen once per step, so every render of the loop is on one grid;
- the step changes the normal only on a good estimate (decided by the maintainer): only a loop that ends `Converged` with at least one axis fixed turns the patch (`GridPlaneOutcome::Turned`). Every other ending leaves the normal exactly as the step was given it, even when earlier turns were accepted, and is reported, not refused: `Undetermined` when the cells cannot constrain the normal (verdict none, or no view scored), `Unstable` when they constrain it but the loop did not settle (a rejected turn, an oscillation, or the render cap, including a cap reached with every turn accepted, because the step wants stable convergence).
- the cell ZNCC bar defaults to `0.6` (decided by the maintainer), the step's own `BENCH_MIN_CELL_ZNCC`, below the piecewise refinement's `0.8`, which keeps its own; see [The cell ZNCC bar](#the-cell-zncc-bar-2026-10-15).

Not decided: which score the loop's acceptance rule reads; whether the step moves the patch's centre onto the fitted plane; whether a cell registration cannot settle falls back to a full piece fit; whether the first iteration can read the localizer's tiles. See [Open questions](#open-questions).

Built: the shared kernels (#954) and the bench step `grid_plane_normal` in [bench/normal/grid_plane.rs](../../crates/sfmtool-core/src/bench/normal/grid_plane.rs), with its tests, specified in [editable-track.md](../core/bench/editable-track.md) § "Estimating the normal". It stands beside `PieceLayout::Grid`, which is unchanged, so the two can be measured against each other; Track View's button and the wire's `fit_bench_track_normal` still run `PieceLayout::Grid`, and Surface Tiling Mode now runs *Grown Plane Normal* ([core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md)) instead; the Python bindings in `sfmtool.bench` bind both, as `finite_difference_normal` and `grid_plane_normal`. The step answers two open questions provisionally, the acceptance score and moving the centre; the density, decided by the maintainer with the synthetic evidence recorded below, is built as decided; see [What the built step chose](#what-the-built-step-chose). The measurement against today's Grid Plane Normal on the two ground truths is recorded under [Measurement against today's Grid Plane Normal](#measurement-against-todays-grid-plane-normal-2026-10-14). A sweep of the search bound and the acceptance rule, with precision and baseline gates applied after the fact, is recorded under [Sweep of the gates, the bound and the acceptance rule](#sweep-of-the-gates-the-bound-and-the-acceptance-rule-2026-10-15); the options it added (`acceptance`, `max_normal_precision_deg`) are off by default. [An addendum](#addendum-todays-step-at-33-and-a-third-capture-2026-10-15) adds today's step at 3×3 as a baseline and a third, candidate ground truth. Remaining: the hand review, then the moves listed under [Retiring the photometric search](#retiring-the-photometric-search).

Amends:
- [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal": `PieceLayout::Grid` of `finite_difference_normal`, the *Grid Plane Normal* step
- [core/patch/cell-plane-normals.md](../core/patch/cell-plane-normals.md): its triangulation, plane fit and `NormalDeterminacy` are factored out and shared
- [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md#piecewise-refinement): the cell registration, the cell gates and the robust affine fit are factored out and shared
- `core/patch/patch-normal-refinement.md` (deleted), `core/patch/fronto-parallel-patch-cache.md` (deleted), `core/patch/patch-normal-refine-view-subset.md` (deleted): retired when their callers move
- [cli/reconstruction/embed-patches-command.md](../cli/reconstruction/embed-patches-command.md) and [cli/reconstruction/xform/refine-normals-command.md](../cli/reconstruction/xform/refine-normals-command.md): the normal step they run

Related drafts: [surface-co-solve.md](surface-co-solve.md) § "Normal estimators" lists the family this estimator belongs to and should defer to this draft for the grid estimator. [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) decides when a cell plane normal may be read, which applies to this estimator's output too. [two-tier-patch-density.md](two-tier-patch-density.md) chooses the density the patch is stored at, which bears on the density this estimator renders at. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) decides how views are scored against a bitmap, which bears on the loop's acceptance score.

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in the world, and its normal says which way the square faces. The normal decides how every photograph is resampled into the patch's bitmap, so a wrong normal blurs the bitmap, shifts the keypoints that are localised on it, and misleads every photometric reading taken from it afterwards.

The project has two production ways of estimating a normal. The photometric search turns the normal through two degrees of freedom and keeps the orientation at which the views agree best; [surface-co-solve.md](surface-co-solve.md) records a median error of 16 to 23 degrees for it at the true position in the surface co-solve harness, and it is weak where the texture is weak. The other is the bench's *Grid Plane Normal*, which reads depth by parallax instead of reading the shape of a correlation peak over orientation. Track View's Surface Tiling Mode ran the Grid Plane Normal in its sequence, as the stronger of the two, until it moved to *Grown Plane Normal* ([core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md), [gui/track-view.md](../gui/track-view.md) § "Surface Tiling Mode"). This draft improves the Grid Plane Normal.

### The Grid Plane Normal today

`finite_difference_normal` with `PieceLayout::Grid` in [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), specified in [editable-track.md](../core/bench/editable-track.md) § "Estimating the normal":

1. It cuts the patch into `pieces × pieces` square pieces that together span its side. Track View's default is 2×2 with no overlap (`SplitSettings::default` in [bench.rs](../../crates/sfm-explorer/src/bench.rs) takes `MIN_PIECES`, which is 2, and an overlap of 0).
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

The bench step replaces `PieceLayout::Grid` in [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), beside `finite_difference_normal`, which keeps `PieceLayout::Cross`. It builds on two crate-visible kernels in `patch/`, each factored out of an existing module rather than written again:

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
    /// `BENCH_MIN_CELL_ZNCC`, 0.6.
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

**Why this shape.** The signature is the other two normal steps', so Track View's button and the wire's `fit_bench_track_normal` with `method` `grid_plane` call it the way they call `finite_difference_normal` today, and it ends with the same `NormalReport`. The split is fixed at the ZNCC grid's nine cells, so there is no `pieces` or `overlap`: the cells are the ones `grid_cell_centres` defines for every other per-cell reading, and the cell gates and the plane fit are only defined on that split. Track View's *per axis* and *overlap* boxes stay for *Finite Diff Normal*. The report keeps every cell's verdict and every view's per-cell status, because a wrong normal is diagnosed by asking which cells were trusted and why each view refused the others, and the statuses reuse `CellStatus`, `CellPlaneStatus` and `LoopStop`, so the bench, the cluster file and the cell plane kernel use one set of words. `NormalDeterminacy` is the cell plane kernel's type, so a caller choosing between the two estimators reads one verdict. The settings are `*Options` because a caller chooses them; `CellPlaneParams` keeps its existing name.

**Example.** Track View's *Grid Plane Normal* button calls `grid_plane_normal(&track, &edited, &images, &GridPlaneNormalOptions::default(), &progress)`, and Track View shows the nine cell verdicts as a three-by-three overlay on the patch bitmap and turns the patch only about the axes the cells fixed.

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

The step turns the patch to the fitted normal with `tilt_patch`, which keeps the centre and so moves no keypoint, and stops `MAX_TILT_DEG` (80°) from any observation. It then renders and registers again. A turn is kept only when the score does not fall, the piecewise loop's acceptance rule, with the same tolerance and the same floor at the starting score; which score is an open question. The loop stops when a turn moves no cell centre by more than `update_tolerance_px`, at `max_iterations` renders, when a turn's largest cell movement does not shrink from the one before (oscillation, keeping whichever of the two frames scores higher), or when a turn is rejected. With a normal far from the truth, cells near the patch's edge register at or past the bound; they are refused for the bound on the first render and return as the turns bring them inside it.

The step ends as today's does, by reading the track back with the fit's evaluation over the turned square.

### The patch that is good in parts

A patch on one textured plane passes everywhere, and the gates change nothing. The patches the gates exist for sit where a surface ends. There the cells over the surface register and the cells over the background are refused for curvature, for ZNCC, or as outliers of the view's affine fit; a cell that passes in every view but lies off the plane the others agree on is refused by the robust plane fit. The normal is the plane of the survivors, even when they are one half of the patch, and when they lie along one row or column the verdict says one axis is free rather than filling it from refused cells.

### Retiring the photometric search

The photometric normal search had two production callers: the embed-patches rounds and `sfm xform --refine-normals`. The bench's *Fit Normal* was the third; Track View and the wire offer *Grown Plane Normal* ([core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md)) in its place, and `sfmtool_core::bench::fit_normal` and its Python binding have been deleted (PR #1080). Every command that turned patch normals now runs the grown-plane normal rather than this estimator (PRs #1076 to #1079), and the search, with its two support modules, the fronto-parallel cache and the D-optimal view subset, and the fronto-parallel prior, has been deleted (PR #1082); a sweep over that prior had found it degraded normals everywhere it was applied.

The order is: build the shared kernels and the bench step and measure the step against today's Grid Plane Normal; move the Track View button; move embed-patches; move `xform --refine-normals`; delete.

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
| `min_cell_zncc` | `0.6` | A cell below this ZNCC in a view is refused in that view (`BENCH_MIN_CELL_ZNCC`); decided by the maintainer, see [The cell ZNCC bar](#the-cell-zncc-bar-2026-10-15) |
| `min_cell_curvature` | `0.02` | A cell whose peak is flatter in a view is refused in that view (`DEFAULT_MIN_CELL_CURVATURE`) |
| `working_resolution` | `None` (the reference view's footprint, clamped to `R ..= 2R`) | Side of the working grid; decided by the maintainer, see Status |
| `plane` | `CellPlaneParams::default()` | `include_refused_outlier` false, `min_rays` 2, `min_triangulation_angle_deg` 2.0, `cell_shift_precision_grid_px` 0.1, `ray_noise_floor_px` 0.05, `irls_iters` 3, `tukey_c` 4.685, `min_cells` 3, `det_aniso` 0.10 |
| `update_tolerance_px` | `0.05` | The loop stops when no cell centre moves by more; the piecewise loop's default |
| `max_iterations` | `5` | Render cap; the piecewise loop's default |

`min_cell_curvature` is the piecewise refinement's provisional default, set on synthetic tests and not measured for this use. `min_cell_zncc` is the step's own, `0.6`, below the piecewise refinement's `DEFAULT_MIN_CELL_ZNCC` (`0.8`), which stays the cluster pass's bar.

## Testing

Kernel tests on synthetic planes rendered through known poses:

- A textured planar patch started 10° to 20° off its normal: the loop converges, all nine cells are in the plane, both axes are fixed, and the normal is within 0.5° of truth.
- The same patch with one third replaced by a flat region: those three cells are refused for curvature in every view, both axes are still fixed, and the normal is within 1° of truth.
- A patch whose halves lie on two planes at different depths: the cells of one half are refused, by ZNCC, as affine outliers or by the plane fit, the verdict is one axis, the free axis lies along the surviving half's row or column, and the fixed axis agrees with that half's plane.
- A patch over flat texture throughout: no cell survives, the verdict is none, and the patch is not turned.
- A turn that lowers the acceptance score is not kept.
- The piecewise refinement's and the cell plane kernel's tests pass unchanged after the factoring, and a `.matches` file written before and after is identical.

Measurement against today's Grid Plane Normal, at its Track View default of 2×2 pieces, on the `seoul_bull_sculpture` and `kerry_park` ground truths: the normal error of each against the ground-truth normals, the share of tracks given each verdict, and the time each takes per track. Then a hand review in SfM Explorer: Track View's normal step run with each estimator over the same patches, the turned patches compared by a person without knowing which estimator turned which. The estimator replaces today's only if the review prefers it; a lower error on the ground truths is not enough on its own.

## Non-goals

**Joint estimation of depth and normal.** The step turns the patch about its centre and reports the plane's offset; placing the centre is the following Fit's work in the tiling sequence, unless the open question below decides otherwise. The joint solve is the subject of [surface-co-solve.md](surface-co-solve.md).

**Normals before poses exist.** A normal needs sighting rays. Without poses, the per-cell displacements of the piecewise refinement are what is measured, and [cell-plane-normals.md](../core/patch/cell-plane-normals.md) turns them into a normal once poses exist.

**Moving the sightings.** The loop's renders follow each view's keypoint as `tilt_patch` rebuilds it; the step does not write a view's common cell shift back to its keypoint. Placing keypoints is the fit's work.

## What the built step chose

The bench step follows the Rust API above, with these differences and with these choices where the draft left the detail open. Each is provisional until the measurement.

- **Names.** `NormalEstimate::GridPlane` still names `PieceLayout::Grid`'s report, so the new report is carried by `NormalEstimate::GridPlaneCells(Box<GridPlaneCells>)`, boxed because the report, with its per-render trace, is much larger than the other variants. `NormalEstimate` is no longer `Copy`, since `GridPlaneCells` holds a `Vec`. When `PieceLayout::Grid` is retired the variant can take the plain name.
- **Two scores in the report.** `GridPlaneCells` also carries `score_before` and `score_after`, the acceptance score at the start and at the last render the loop kept. Its readings are those of the last render the loop kept, not of a rejected render.
- **The acceptance score** is the mean over the non-reference views of the windowed ZNCC of each view's tile at zero shift against the reference's render, over the samples both carry and averaged over the reference's textured channels. The window is the cluster refinement's default, `PatchWindow::GaussianDisk { sigma: 0.5 }`, the window the piecewise loop's whole-member ZNCC reads with, so the rule this step copies reads the same kind of score. This is the cheapest score the step already has the tiles for, and it answers the open question provisionally. On the textured plane and the plane with a flat column, every measured angle is the same as with an unwindowed whole-tile ZNCC to the stated precision. On the synthetic depth step it fails: at one sample per photograph pixel the first turn toward the near plane lowered the score from two of five starts, and the patch was not turned, because the window weights the tile's middle and the middle is the far plane; the unwindowed score failed from one of the five, a different start. One of the two is 12 degrees about the axis the column fixes with 6 about the column, the start the depth-step test used under the unwindowed score; on the one-sample-per-pixel grid the test started 15 degrees about the axis the column fixes with -8 about the column, one of the three starts the windowed score turned from there. On the default footprint grid that start ends on a rejected turn too, so the test now runs there from 20 degrees about `-x` with 4 about the column, the one of five starts that converges there. A score over the cells the gates kept would not read the refused ones.
- **The reference** is the fit's choice (the reference the track holds where its row is `in` with a keypoint, else the reference-view rule's pick), made once when the step starts and kept through the loop. One row per image takes part.
- **The render** places the context tile with `OrientedPatch::anchored_at_keypoint`, the same re-anchoring `shifted_center` of the localizer's offset gives, and reads the cell positions through `shifted_center`. The sampler is the one `fit.refine.sampler` picks from the view's Jacobian at the working resolution, as for the bench's tiles. A row with no keypoint is rendered at the frame's own projection.
- **The working resolution** `None` follows the reference view's footprint, as decided above. The footprint is read from the reference render map's Jacobian at the patch centre (`patch_grid_jacobian` on the `R`-sample grid at the reference's keypoint): its two columns are the patch's `u` and `v` axes in photograph px per grid px, and the side is `R` times the longer column, rounded up, clamped to `R ..= 2R`, and `R` where there is no reference or no Jacobian. `R` is the reconstruction's patch resolution where it stores bitmaps and otherwise `fit.refine.resolution`, at least 3, and is also the resolution the reference rule renders at. `GridPlaneCells::working_resolution` reports the side. The footprint is rounded up after a thousandth of a pixel is taken off, so a footprint that is a whole number of pixels up to the Jacobian's own error keeps that number: the Jacobian is a central difference half a grid px either side of the centre, and reads the 32 px patch facing the cameras as 32.0000005. On the tests' scene the side comes out 32 facing the cameras and from 10 degrees about `x` and 15 about `y` (footprints 32.00002 and 32.00004 px, under the thousandth), and 33 from 20 degrees about `x + y` and 15 about `x - y` (32.04 and 32.11 px).
- **A view whose affine fit fails** (reflects or is not finite) keeps none of its cells, which read `NotAttempted`, as the piecewise refinement stores a failed fit.
- **The reference's row** in `view_cells` reads `Fitted` for a textured template, `RefusedCurvature` for a flat one and `NotAttempted` for one partly off its photograph; it contributes a ray only for a textured template.
- **The ray floor** uses the render map's local Jacobian by central differences of half a grid px on each axis at the cell's registered position.
- **A turn's cell movement** is `2ρ sin(θ/2)` for a turn of `θ` (the angle `tilt_patch` reached) and `ρ` the farthest cell centre from the patch centre, in grid px: the chord the turn carries that centre through.
- **The loop's order.** A turn under `update_tolerance_px` is taken without a render to score it (`Converged`). At `max_iterations` renders the last render's turn is not taken, having no render to score it (`Cap`). A rejected turn is rendered, so it counts as a render. A start that no view can be scored at stops the loop at once (`NotRun`), and cells that fix no axis stop it where it is (`Measured`). The piecewise loop instead scores even a converged update, because its score does not need a render.
- **Under one axis** the target normal is the patch's own with its component along the free axis removed, not the plane kernel's one-axis normal, which takes its turn about the line from the mean viewing direction.
- **The centre does not move**, as the open question's first option; the plane's offset is reported.
- **Not parallel.** The views are rendered and registered one after another, so the per-render cancel check stays simple.
- **A detail record.** `GridPlaneNormalOptions::record_detail` (off by default) keeps, for every render, the tiles, every cell's reading in every view, each view's affine fit, each cell's rays and triangulation, and the plane, in `GridPlaneCells::detail` (see [editable-track.md](../core/bench/editable-track.md#estimating-the-normal)). [scripts/track_at_pixel/normal_sources/grid_plane_detail.py](../../scripts/track_at_pixel/normal_sources/grid_plane_detail.py) records it for named points from the evaluation's mean-view start and from the ground-truth normal, with today's Grid Plane Normal at 2×2 and 3×3 beside it.

### Synthetic evidence on the density

On the tests' scene (three pinholes 4 units from a textured plane, baselines 0.6 to 1.0 units, a patch whose side spans 32 photograph px), the angle between the step's normal and the plane's, from starts 10 to 20 degrees off, by working grid:

| Working grid | Photograph px per grid px | Textured plane | One column flat |
|---|---|---|---|
| 12 | 2.7 | 3.2 to 4.0° | |
| 24 (the bitmap grid, `R`) | 1.33 | 0.67 to 0.90° | 1.4 to 3.3° |
| 28 | 1.14 | 0.32 to 0.43° | 0.40 to 0.64° |
| 32 | 1.0 | 0.10 to 0.27° | 0.09 to 0.24° |
| the footprint (32 or 33 here, the default) | 1.0 or 0.97 | 0.14 to 0.53° | 0.24 and 0.13° from the two of four starts that converge, both on 32; the other two, on 33, unstable and unchanged |
| 40 | 0.8 | 0.18 to 0.28° | 0.31 to 0.46° |
| 48 | 0.67 | 0.18 to 0.26° | |

Rendering with a fixed bilinear sampler instead of the sampler rule's choice changed the 24-sample results by under 0.03°, so the loss is the grid's coarseness against the photograph and not the sampler. The error falls until the grid reaches about one sample per photograph pixel and is flat beyond, which favours a working grid that follows the reference view's footprint over one fixed in samples. Where the footprint grid comes out 33 it does less well than exactly 32 (0.53 and 0.38° on the textured plane, against 0.19 and 0.10° at 32 from the same starts), so the error is not a smooth function of the density alone; how the `[0, R/3, R − R/3, R]` split falls may enter. The tests hold the default grid to 0.6 degrees on the textured plane (the draft's 0.5 holds from three of four starts) and run the flat-column case at exactly 32. On the depth step the default grid turns from one of five starts, the fixed axis ending 0.06° off, and leaves the other four unchanged as unstable. This is one scene; the ground-truth measurement decides.

## Measurement against today's Grid Plane Normal (2026-10-14)

**Question.** On the two checked-in ground truths, how close to the ground-truth normal does `grid_plane_normal` at its defaults turn a patch, compared with today's Grid Plane Normal (`finite_difference_normal`, `PieceLayout::Grid`, 2×2 pieces, no overlap, Track View's default); how often does it turn at all; and what does each cost per call?

**Setup.**

- **Script.** [scripts/track_at_pixel/normal_sources/grid_plane_eval.py](../../scripts/track_at_pixel/normal_sources/grid_plane_eval.py), through the `sfmtool.bench` bindings `create_track`, `tilt_patch`, `fit`, `finite_difference_normal` and `grid_plane_normal`. `prepare` copies each ground truth into the track-at-pixel cache workspace in the system temporary directory; nothing is written beside `test-data`.
- **Points.** Every finite point seen in two or more images: all 266 of `seoul_bull_sculpture_ground_truth_v01.sfmr` and all 382 of `kerry_park_ground_truth_v01.sfmr`. No sample.
- **Start.** Each point is put on the bench from the ground truth (its position, its sightings at their stored keypoints, its stored half-extent), its patch turned with `tilt_patch` to the mean viewing direction of its sightings (the `mean_view` of `normal_diag.py`), and then given one bench `fit`. Both steps run from that one fitted track. The fit is kept because it was then what preceded the Grid Plane Normal in Surface Tiling Mode's sequence and it gives both steps keypoints localized at the start's frame. A fit does not change the normal, so the start's error is the mean view's. It moved the centre a median 0.11 (`seoul_bull`) and 0.09 (`kerry_park`) half-extents off the ground-truth position, and no fit made a point a direction. A control run without the fit (`--no-fit`) is below.
- **Errors.** The angle between a step's normal and the ground-truth normal, both turned to face the first sighting's camera, folded to 0–90°, as in `normal_diag.py`. The ground-truth normal is `u × v` of the ground truth's stored patch frame. A step that leaves the normal keeps the start's error.
- **Time.** Wall time of each binding call, single-threaded, the photographs held in one prebuilt `ImagePyramidSet`. Medians vary by about 20% between repeated runs.
- **Commit and machine.** Commit `62d22282` (on `db9f99ff`); Intel Core i9-14900HX, 64 GB, Windows 11, release build of the extension.
- **Commands.** From the repository root, each dataset to its own directory, which also receives its `cameras.json`:

  ```bash
  pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_eval.py seoul_bull --out OUT/seoul_bull
  pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_eval.py \
      test-data/images/kerry_park/kerry_park_ground_truth_v01.sfmr --out OUT/kerry_park
  # the two controls, each on both datasets: the same commands with
  #   --start ground_truth
  #   --no-fit
  ```

  Each run took 7 to 10 s on `seoul_bull` and 24 to 30 s on `kerry_park`, loading included.

**Error to the ground-truth normal, all points (degrees).**

| dataset | points | normal | median | p75 | under 10° | under 20° |
|---|---|---|---|---|---|---|
| `seoul_bull` | 266 | start (mean view) | 33.7 | 47.6 | 11% | 24% |
| | | today's 2×2 Grid Plane Normal | 31.5 | 48.6 | 9% | 26% |
| | | `grid_plane_normal` | 33.4 | 47.6 | 11% | 25% |
| `kerry_park` | 382 | start (mean view) | 34.6 | 51.7 | 15% | 31% |
| | | today's 2×2 Grid Plane Normal | 29.3 | 47.6 | 16% | 36% |
| | | `grid_plane_normal` | 32.7 | 51.2 | 20% | 34% |

Today's step turned every point and refused none; all four pieces fitted on every point but two of `kerry_park`, where three did. It left the normal further from the truth than the start on 38% (`seoul_bull`) and 29% (`kerry_park`) of points.

**Outcomes of `grid_plane_normal`.**

| dataset | Turned | Undetermined | Unstable, rejected | Unstable, oscillation | Unstable, cap |
|---|---|---|---|---|---|
| `seoul_bull` | 7 (2.6%) | 93 (35.0%) | 149 (56.0%) | 14 (5.3%) | 3 (1.1%) |
| `kerry_park` | 27 (7.1%) | 163 (42.7%) | 171 (44.8%) | 17 (4.5%) | 4 (1.0%) |

Every Undetermined point stopped `Measured` (the last render's cells fixed no axis); none stopped `NotRun`. The working grid was the floor `R` = 24 on 147 and 170 points and the cap 48 on 21 and 21.

**The points `grid_plane_normal` turned, paired with today's step on the same points (degrees).**

| dataset | points | normal | median | p75 | under 10° | under 20° |
|---|---|---|---|---|---|---|
| `seoul_bull` | 7 | start | 39.6 | 40.4 | 29% | 29% |
| | | today's 2×2 | 20.9 | 28.7 | 14% | 43% |
| | | `grid_plane_normal` | 15.0 | 27.1 | 29% | 71% |
| `kerry_park` | 27 | start | 22.4 | 42.0 | 19% | 48% |
| | | today's 2×2 | 15.9 | 39.1 | 33% | 56% |
| | | `grid_plane_normal` | 4.5 | 8.4 | 78% | 85% |

On these points the new step's normal is closer to the truth than today's on 4 of 7 and 18 of 27.

**Time per call (median ms).**

| dataset | fit | today's 2×2 | `grid_plane_normal` |
|---|---|---|---|
| `seoul_bull` | 4 | 13 | 3 |
| `kerry_park` | 10 | 34 | 6 |

**Controls.**

- **From the ground-truth normal** (`--start ground_truth`). Today's step moves every point off the truth, by a median 10.8° (p75 31.3°) on `seoul_bull` and 3.6° (p75 8.4°) on `kerry_park`. `grid_plane_normal` turns 11 (4.1%) and 48 (12.6%) points, by a median 2.9° and 2.3°, and leaves the rest: 71 and 72 Undetermined, 184 and 262 Unstable, of which 171 and 245 stopped on a rejected turn.
- **Without the fit** (`--no-fit`, both steps from the ground-truth position and keypoints at the mean-view normal). The Turned and Undetermined counts are within four points of the fitted run's (Turned 7 and 29, Undetermined 97 and 163); among the Unstable on `kerry_park`, 160 stopped on a rejected turn against 171, and 10 at the cap against 4. Today's step ends at a median 32.1° and 27.2°; on the 29 points `grid_plane_normal` turned on `kerry_park`, it ends at a median 5.4° against today's 18.1°. The fit neither helps nor harms either step much here.

**Cell readings.** Over every observation row of every point at the last render of the mean-view run, the registration statuses are 44% fitted, 33% refused for the bound, 15% for ZNCC and 5% for curvature on `seoul_bull`, and 32%, 43%, 16% and 8% on `kerry_park`. The Undetermined points started furthest off (median 47.1° and 43.2°), against 39.6° and 22.4° for the Turned and 31.4° and 27.1° for the Unstable. From the ground-truth normal the bound still refuses 24% and 31% of cell readings.

**Points to look at in Track View** (ground-truth point indexes; the first ten of each kind in index order, and the Turned points with the largest error):

| dataset | kind | points |
|---|---|---|
| `seoul_bull` | Undetermined | 3, 18, 20, 26, 29, 30, 33, 40, 41, 49 |
| | Unstable, rejected | 1, 2, 4, 5, 8, 10, 11, 12, 13, 14 |
| | Unstable, oscillation | 0, 6, 17, 32, 37, 63, 74, 100, 101, 132 |
| | Unstable, cap | 22, 133, 153 |
| | Turned, largest error first (°) | 7 (71.4), 165 (38.7), 9 (15.6), 226 (15.0), 19 (12.1), 160 (7.9), 250 (7.5) |
| `kerry_park` | Undetermined | 10, 12, 28, 29, 32, 33, 40, 41, 47, 48 |
| | Unstable, rejected | 0, 1, 2, 5, 6, 7, 9, 11, 13, 14 |
| | Unstable, oscillation | 3, 4, 8, 21, 30, 44, 51, 93, 117, 144 |
| | Unstable, cap | 258, 260, 267, 325 |
| | Turned, largest error first (°) | 114 (48.1), 169 (34.6), 238 (25.3), 132 (22.4), 159 (12.4), 96 (12.2), 353 (9.4), 178 (7.3), 319 (6.8), 196 (6.5) |

Each row of the script's JSONL carries the point's ground-truth position and normal, the start normal, the fitted frame, each step's resulting frame, and the reference and `in` sightings with their keypoints; `cameras.json` carries the images' names, centres, rotations and lenses. These points can be drawn from them without running the steps again.

**What the numbers show.**

- Over all points neither step moves the median far from the mean-view start: today's lowers it by 2 and 5°, the new one by under 2°. The new step changes few normals, so its whole-set figures are close to the start's.
- `grid_plane_normal` rarely turns a patch: 2.6% and 7.1% of points from the mean view, 4.1% and 12.6% from the truth itself. Most endings are a turn the acceptance rule rejected (56% and 45% of points from the mean view, 64% of both from the truth) or cells that fixed no axis (35% and 43% from the mean view).
- Where it turns, it is much closer to the truth than today's step on `kerry_park` (median 4.5° against 15.9°, 78% under 10°). On `seoul_bull` its median on the 7 turned points is lower (15.0° against 20.9°), with one turn 71° off (point 7).
- From the truth, today's step moves every point off it, by a median 3.6° to 10.8°; `grid_plane_normal` leaves 87% to 96% of them where they were and turns the rest by a median of 2 to 3°.
- The search bound refuses a third or more of the cell readings, and the points whose cells fixed no axis started furthest off. That a start 30° or more off puts the outer cells past the 2 px bound at the first render is an inference from this, not measured here; it is the case the [open question](#open-questions) on a fallback to a full piece fit asks to count.
- The new step costs a fifth to a quarter of today's per call.

**What the numbers cannot show.**

- **The ground-truth normals are estimates, and most were set with today's step.** They are the patch frames stored in the two `.sfmr` files, not measured surfaces, and per the maintainer most of them were set with today's Grid Plane Normal at 3×3 pieces, so agreement with them partly measures agreement with that method (see [today's step at 3×3](#addendum-todays-step-at-33-and-a-third-capture-2026-10-15)). Where a step disagrees with one, the numbers cannot say which of the two is wrong; the [cell plane normal measurements](../core/patch/cluster-patch-refinement-measurements.md#cell-plane-normals-against-the-ground-truths-2026-10-08) carry the same caveat. Today's step moving off the truth on every point, and the new step's rejected turns from the truth, can each be a fault of the step or of the truth.
- Whether a person prefers the turned patches. The hand review decides that, and a lower error here is not enough on its own.
- Why turns are rejected. Whether the windowed whole-tile score refuses correct turns (the open question on a score over the kept cells) or the proposed turns are wrong is not separated here. From the truth, a rejected first turn ends Unstable with the normal unchanged, so Unstable there includes points already at the normal the score prefers.
- Two small captures, 648 points in all; the Turned sets are 7 and 27 points.

## Sweep of the gates, the bound and the acceptance rule (2026-10-15)

**Question.** The [measurement above](#measurement-against-todays-grid-plane-normal-2026-10-14) found the step turning 3% to 7% of points. A diagnostic showed that taking the first plane the cells give, with no acceptance test, turns most points and, where the views are wide apart, lands closer to the truth than the start and than today's 2×2 step, while where they are close together it can be far off. This sweep measures, on both ground truths, how the search bound and the acceptance rule change that, how the loop converges, and whether a gate on the plane's predicted precision or on the views' baseline separates the good planes from the bad. No default changes.

**Setup.**

- **Options added for the experiment** (defaults unchanged): `GridPlaneNormalOptions::acceptance`, `WindowedZncc` (today's rule) or `None` (every rendered turn is kept; the loop still stops on the tolerance, the render cap, an oscillation, or a render that proposes no turn); `max_normal_precision_deg`, a precision gate, not used in the runs (the bars below are applied after the fact). `GridPlaneCells` reports the plane's predicted precision (`σ = 1 / √(Σⱼ wⱼ (dⱼ · e)²)` as [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) defines it, along the weaker in-plane axis under both axes and along the fixed one under one axis), the views' baseline (the largest angle between two `in` sightings' directions from the patch centre), and a trace with one entry per render: the score, the verdict, the cells in the plane, the precision, the normal the proposed turn reaches, the turn's largest cell movement, and whether the loop took it.
- **Points, start, errors.** As in the measurement above: every finite point seen twice or more (266 and 382), the same fitted mean-view start, the angle to the ground truth's stored patch normal folded to 0–90°. Today's 2×2 Grid Plane Normal and the start are the baselines and are the same in every variant.
- **Variants.** `cell_shift_bound_px` 2, 4 and 6 × acceptance `windowed_zncc` and `none`, all with `max_iterations` 8; everything else at its default. `bound2_windowed_zncc` is the default step with the higher render cap.
- **Estimators.** *new*: the normal the step left (it differs from the start only on `Turned` points). *first plane*: the normal the first render's turn would reach, read from the trace; it does not depend on the acceptance rule, so it is reported once per bound. With `update_tolerance_px` very large it is what the step itself returns, as the diagnostic ran it.
- **Baseline bins.** Under 5°, 5–10°, 10–20°, 20–40°, 40° and over. `seoul_bull` has 67, 58, 14, 30 and 97 points in them; `kerry_park` 12, 21, 48, 100 and 201.
- **Commit and machine.** Commit `a26e3483` for the step, the binding and the run script, `3d52abc2` for the summary script; Intel Core i9-14900HX, 64 GB, Windows 11, release build.
- **Commands.** From the repository root, for each dataset (`seoul_bull`, and `test-data/images/kerry_park/kerry_park_ground_truth_v01.sfmr` as `kerry_park`), bound `B` in 2, 4, 6 and acceptance `A` in `windowed_zncc`, `none`:

  ```bash
  pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_eval.py DATASET \
      --out OUT/DATASET/boundB_A \
      --new-opt max_iterations=8 --new-opt cell_shift_bound_px=B --new-opt acceptance=A
  pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_sweep.py OUT/DATASET
  ```

  Each run took 9 to 11 s on `seoul_bull` and 33 to 54 s on `kerry_park`, loading included; the median `grid_plane_normal` call took 3 to 6 ms and 7 to 22 ms (the slowest `bound6_none` on `kerry_park`), against 16 to 17 ms and 41 to 58 ms for today's 2×2.

**Outcomes.** `converged` is `Turned`; `measured` (and `not_run`, none here) is `Undetermined`; the rest are `Unstable`.

`seoul_bull`:

| variant | converged | measured | rejected | oscillation | cap | not_run |
|---|---|---|---|---|---|---|
| bound2_none | 19 | 132 | 0 | 111 | 4 | 0 |
| bound2_windowed_zncc | 8 | 93 | 151 | 14 | 0 | 0 |
| bound4_none | 35 | 111 | 0 | 118 | 2 | 0 |
| bound4_windowed_zncc | 10 | 69 | 175 | 12 | 0 | 0 |
| bound6_none | 53 | 104 | 0 | 108 | 1 | 0 |
| bound6_windowed_zncc | 13 | 62 | 183 | 8 | 0 | 0 |

`kerry_park`:

| variant | converged | measured | rejected | oscillation | cap | not_run |
|---|---|---|---|---|---|---|
| bound2_none | 59 | 182 | 0 | 139 | 2 | 0 |
| bound2_windowed_zncc | 28 | 163 | 171 | 20 | 0 | 0 |
| bound4_none | 92 | 139 | 0 | 147 | 4 | 0 |
| bound4_windowed_zncc | 26 | 122 | 219 | 15 | 0 | 0 |
| bound6_none | 109 | 128 | 0 | 137 | 8 | 0 |
| bound6_windowed_zncc | 31 | 105 | 229 | 17 | 0 | 0 |

**Start and today's 2×2 by baseline** (all points, degrees):

`seoul_bull`:

| baseline | points | start med | start <20 | 2x2 med | 2x2 <10 | 2x2 <20 | 2x2 closer than start | 3x3 med | 3x3 <10 | 3x3 <20 | 3x3 closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <5 | 67 | 26.1 | 39% | 45.0 | 0% | 12% | 37% | 51.5 | 1% | 7% | 31% |
| 5-10 | 58 | 28.9 | 34% | 29.9 | 16% | 33% | 45% | 32.6 | 7% | 28% | 41% |
| 10-20 | 14 | 27.0 | 43% | 18.6 | 29% | 64% | 57% | 21.4 | 21% | 50% | 50% |
| 20-40 | 30 | 47.4 | 13% | 26.5 | 13% | 33% | 80% | 41.8 | 3% | 17% | 77% |
| >=40 | 97 | 38.2 | 8% | 28.2 | 6% | 25% | 85% | 34.8 | 2% | 12% | 82% |

`kerry_park`:

| baseline | points | start med | start <20 | 2x2 med | 2x2 <10 | 2x2 <20 | 2x2 closer than start | 3x3 med | 3x3 <10 | 3x3 <20 | 3x3 closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <5 | 12 | 0.7 | 83% | 71.1 | 0% | 0% | 8% | 71.8 | 0% | 8% | 8% |
| 5-10 | 21 | 41.3 | 38% | 44.3 | 24% | 33% | 48% | 24.2 | 29% | 38% | 52% |
| 10-20 | 48 | 28.6 | 46% | 19.7 | 23% | 50% | 54% | 17.9 | 27% | 54% | 52% |
| 20-40 | 100 | 39.9 | 13% | 30.2 | 11% | 32% | 79% | 35.4 | 4% | 21% | 67% |
| >=40 | 201 | 29.5 | 33% | 26.5 | 18% | 37% | 78% | 29.1 | 14% | 36% | 71% |

**The first plane by baseline.** `proposed` is how many points' first render proposed a turn; the start and today's step are on those same points.

`seoul_bull`:

| variant | baseline | points | proposed | first med | first p75 | first <10 | first <20 | start med | 2x2 med | 3x3 med | closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bound2 | <5 | 67 | 46 | 39.8 | 48.7 | 7% | 11% | 28.2 | 34.2 | 45.3 | 39% |
| bound2 | 5-10 | 58 | 50 | 44.7 | 64.3 | 4% | 16% | 29.1 | 30.5 | 32.6 | 36% |
| bound2 | 10-20 | 14 | 8 | 38.7 | 60.7 | 12% | 25% | 27.0 | 18.6 | 17.5 | 38% |
| bound2 | 20-40 | 30 | 11 | 23.7 | 36.0 | 27% | 45% | 40.5 | 27.1 | 33.6 | 82% |
| bound2 | >=40 | 97 | 71 | 18.1 | 26.2 | 25% | 56% | 34.0 | 23.7 | 30.9 | 80% |
| bound4 | <5 | 67 | 49 | 40.5 | 56.3 | 6% | 12% | 29.8 | 34.6 | 44.6 | 35% |
| bound4 | 5-10 | 58 | 51 | 46.7 | 61.6 | 4% | 12% | 29.1 | 30.2 | 32.7 | 33% |
| bound4 | 10-20 | 14 | 11 | 50.9 | 68.2 | 9% | 18% | 21.5 | 18.1 | 19.7 | 36% |
| bound4 | 20-40 | 30 | 18 | 19.6 | 26.8 | 28% | 50% | 40.7 | 25.1 | 36.8 | 89% |
| bound4 | >=40 | 97 | 74 | 13.1 | 22.0 | 32% | 68% | 34.0 | 24.2 | 30.9 | 82% |
| bound6 | <5 | 67 | 50 | 41.0 | 58.2 | 6% | 12% | 28.2 | 34.8 | 45.3 | 36% |
| bound6 | 5-10 | 58 | 52 | 47.7 | 62.0 | 4% | 10% | 29.1 | 29.9 | 33.6 | 35% |
| bound6 | 10-20 | 14 | 11 | 50.9 | 65.3 | 9% | 27% | 21.5 | 18.1 | 19.7 | 36% |
| bound6 | 20-40 | 30 | 19 | 16.5 | 24.4 | 32% | 58% | 40.9 | 23.2 | 37.4 | 89% |
| bound6 | >=40 | 97 | 79 | 12.9 | 23.5 | 32% | 68% | 34.0 | 24.8 | 31.0 | 77% |

`kerry_park`:

| variant | baseline | points | proposed | first med | first p75 | first <10 | first <20 | start med | 2x2 med | 3x3 med | closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bound2 | <5 | 12 | 5 | 77.2 | 78.3 | 0% | 0% | 0.8 | 60.0 | 70.2 | 0% |
| bound2 | 5-10 | 21 | 8 | 59.7 | 70.9 | 0% | 25% | 31.7 | 45.1 | 34.8 | 38% |
| bound2 | 10-20 | 48 | 36 | 34.6 | 51.7 | 17% | 39% | 18.3 | 22.3 | 17.9 | 47% |
| bound2 | 20-40 | 100 | 62 | 20.6 | 32.5 | 19% | 50% | 36.8 | 24.1 | 33.3 | 71% |
| bound2 | >=40 | 201 | 118 | 11.3 | 22.3 | 43% | 72% | 22.5 | 20.5 | 20.9 | 75% |
| bound4 | <5 | 12 | 9 | 76.6 | 78.7 | 0% | 0% | 0.8 | 65.8 | 70.2 | 0% |
| bound4 | 5-10 | 21 | 12 | 65.3 | 72.1 | 0% | 17% | 20.6 | 46.4 | 36.1 | 25% |
| bound4 | 10-20 | 48 | 42 | 35.2 | 58.1 | 14% | 36% | 18.3 | 19.7 | 17.9 | 48% |
| bound4 | 20-40 | 100 | 68 | 19.1 | 31.4 | 29% | 50% | 37.1 | 25.7 | 33.6 | 74% |
| bound4 | >=40 | 201 | 143 | 11.4 | 19.4 | 47% | 76% | 25.3 | 22.6 | 23.0 | 76% |
| bound6 | <5 | 12 | 9 | 73.4 | 78.2 | 0% | 0% | 0.8 | 65.8 | 70.2 | 0% |
| bound6 | 5-10 | 21 | 12 | 72.1 | 73.4 | 0% | 8% | 20.6 | 46.4 | 36.1 | 17% |
| bound6 | 10-20 | 48 | 42 | 39.1 | 63.0 | 14% | 33% | 18.3 | 19.7 | 17.9 | 48% |
| bound6 | 20-40 | 100 | 70 | 19.3 | 32.6 | 30% | 54% | 37.1 | 26.1 | 33.6 | 76% |
| bound6 | >=40 | 201 | 152 | 12.3 | 23.8 | 47% | 68% | 25.5 | 24.0 | 25.0 | 72% |

**The step as built, by baseline** (`Turned` points; start and today's 2×2 on the same points):

`seoul_bull`:

| variant | baseline | points | turned | new med | new p75 | new <10 | new <20 | start med | 2x2 med | 2x2 <10 | 3x3 med | 3x3 <10 | new closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bound2_none | <5 | 67 | 4 | 27.2 | 50.9 | 25% | 50% | 24.9 | 25.4 | 0% | 28.9 | 0% | 25% |
| bound2_none | 5-10 | 58 | 3 | 71.4 | 73.9 | 0% | 0% | 6.3 | 33.6 | 0% | 34.9 | 0% | 33% |
| bound2_none | 10-20 | 14 | 1 | 10.5 | 10.5 | 0% | 100% | 32.4 | 7.9 | 100% | 10.0 | 0% | 100% |
| bound2_none | 20-40 | 30 | 2 | 27.1 | 32.9 | 0% | 50% | 40.3 | 18.1 | 50% | 33.2 | 0% | 100% |
| bound2_none | >=40 | 97 | 9 | 12.1 | 22.7 | 33% | 56% | 32.1 | 25.4 | 0% | 27.0 | 0% | 78% |
| bound2_windowed_zncc | <5 | 67 | 2 | 11.2 | 13.1 | 50% | 100% | 37.4 | 17.0 | 0% | 26.7 | 0% | 50% |
| bound2_windowed_zncc | 5-10 | 58 | 1 | 71.4 | 71.4 | 0% | 0% | 6.3 | 33.6 | 0% | 46.1 | 0% | 0% |
| bound2_windowed_zncc | 10-20 | 14 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound2_windowed_zncc | 20-40 | 30 | 2 | 27.1 | 32.9 | 0% | 50% | 40.3 | 18.1 | 50% | 33.2 | 0% | 100% |
| bound2_windowed_zncc | >=40 | 97 | 3 | 12.1 | 17.4 | 33% | 67% | 39.0 | 27.0 | 0% | 34.8 | 0% | 100% |
| bound4_none | <5 | 67 | 7 | 44.3 | 64.0 | 14% | 14% | 35.2 | 29.4 | 0% | 25.3 | 14% | 14% |
| bound4_none | 5-10 | 58 | 6 | 74.8 | 77.2 | 0% | 0% | 30.7 | 44.9 | 0% | 44.0 | 0% | 0% |
| bound4_none | 10-20 | 14 | 3 | 24.1 | 41.4 | 0% | 33% | 32.4 | 7.9 | 67% | 10.0 | 33% | 67% |
| bound4_none | 20-40 | 30 | 6 | 18.1 | 22.1 | 33% | 50% | 44.3 | 26.8 | 0% | 44.1 | 0% | 100% |
| bound4_none | >=40 | 97 | 13 | 14.6 | 18.8 | 38% | 77% | 35.3 | 22.2 | 0% | 31.7 | 0% | 77% |
| bound4_windowed_zncc | <5 | 67 | 4 | 41.2 | 55.8 | 25% | 25% | 25.8 | 24.5 | 0% | 24.3 | 25% | 25% |
| bound4_windowed_zncc | 5-10 | 58 | 1 | 43.9 | 43.9 | 0% | 0% | 33.9 | 42.7 | 0% | 46.3 | 0% | 0% |
| bound4_windowed_zncc | 10-20 | 14 | 1 | 24.1 | 24.1 | 0% | 0% | 35.4 | 40.2 | 0% | 43.5 | 0% | 100% |
| bound4_windowed_zncc | 20-40 | 30 | 3 | 22.6 | 26.0 | 0% | 33% | 40.9 | 30.4 | 0% | 44.5 | 0% | 100% |
| bound4_windowed_zncc | >=40 | 97 | 1 | 22.2 | 22.2 | 0% | 0% | 22.0 | 17.4 | 0% | 21.5 | 0% | 0% |
| bound6_none | <5 | 67 | 11 | 45.4 | 69.5 | 9% | 9% | 36.1 | 34.3 | 0% | 31.9 | 0% | 27% |
| bound6_none | 5-10 | 58 | 11 | 71.3 | 76.5 | 0% | 0% | 29.6 | 47.1 | 9% | 42.7 | 0% | 9% |
| bound6_none | 10-20 | 14 | 4 | 38.2 | 61.8 | 0% | 50% | 23.9 | 8.0 | 75% | 8.4 | 50% | 50% |
| bound6_none | 20-40 | 30 | 4 | 19.3 | 25.1 | 25% | 50% | 44.0 | 22.9 | 25% | 42.2 | 0% | 100% |
| bound6_none | >=40 | 97 | 23 | 11.3 | 24.9 | 35% | 70% | 36.4 | 21.5 | 0% | 31.5 | 0% | 83% |
| bound6_windowed_zncc | <5 | 67 | 3 | 44.6 | 66.4 | 33% | 33% | 10.5 | 19.6 | 0% | 27.7 | 0% | 0% |
| bound6_windowed_zncc | 5-10 | 58 | 3 | 43.8 | 53.8 | 0% | 0% | 33.9 | 42.7 | 0% | 46.3 | 0% | 33% |
| bound6_windowed_zncc | 10-20 | 14 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound6_windowed_zncc | 20-40 | 30 | 2 | 24.7 | 29.0 | 0% | 50% | 40.7 | 22.9 | 0% | 32.0 | 0% | 100% |
| bound6_windowed_zncc | >=40 | 97 | 5 | 11.3 | 15.9 | 40% | 80% | 40.2 | 22.2 | 0% | 34.8 | 0% | 100% |

`kerry_park`:

| variant | baseline | points | turned | new med | new p75 | new <10 | new <20 | start med | 2x2 med | 2x2 <10 | 3x3 med | 3x3 <10 | new closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bound2_none | <5 | 12 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound2_none | 5-10 | 21 | 1 | 74.7 | 74.7 | 0% | 0% | 52.5 | 42.8 | 0% | 36.8 | 0% | 0% |
| bound2_none | 10-20 | 48 | 9 | 19.1 | 35.0 | 44% | 56% | 37.5 | 15.0 | 11% | 8.2 | 56% | 78% |
| bound2_none | 20-40 | 100 | 10 | 11.4 | 22.1 | 50% | 60% | 34.5 | 14.3 | 30% | 19.9 | 10% | 70% |
| bound2_none | >=40 | 201 | 39 | 5.3 | 8.2 | 77% | 82% | 20.4 | 17.0 | 33% | 20.5 | 33% | 87% |
| bound2_windowed_zncc | <5 | 12 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound2_windowed_zncc | 5-10 | 21 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound2_windowed_zncc | 10-20 | 48 | 1 | 9.4 | 9.4 | 100% | 100% | 13.4 | 10.3 | 0% | 8.2 | 100% | 100% |
| bound2_windowed_zncc | 20-40 | 100 | 2 | 3.5 | 3.7 | 100% | 100% | 35.9 | 9.0 | 50% | 20.3 | 0% | 100% |
| bound2_windowed_zncc | >=40 | 201 | 25 | 5.3 | 12.2 | 72% | 80% | 22.4 | 25.3 | 32% | 22.3 | 32% | 88% |
| bound4_none | <5 | 12 | 1 | 62.5 | 62.5 | 0% | 0% | 26.7 | 22.1 | 0% | 22.9 | 0% | 0% |
| bound4_none | 5-10 | 21 | 5 | 76.9 | 77.9 | 0% | 0% | 5.0 | 52.1 | 0% | 32.8 | 0% | 0% |
| bound4_none | 10-20 | 48 | 19 | 58.1 | 74.4 | 37% | 37% | 35.1 | 17.7 | 16% | 21.7 | 32% | 47% |
| bound4_none | 20-40 | 100 | 16 | 19.5 | 38.0 | 44% | 50% | 35.8 | 21.3 | 25% | 32.6 | 6% | 75% |
| bound4_none | >=40 | 201 | 51 | 4.5 | 7.4 | 82% | 86% | 25.4 | 23.4 | 25% | 25.3 | 20% | 88% |
| bound4_windowed_zncc | <5 | 12 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound4_windowed_zncc | 5-10 | 21 | 1 | 72.0 | 72.0 | 0% | 0% | 52.5 | 42.8 | 0% | 36.8 | 0% | 0% |
| bound4_windowed_zncc | 10-20 | 48 | 3 | 9.4 | 44.0 | 67% | 67% | 13.4 | 12.2 | 0% | 8.2 | 67% | 67% |
| bound4_windowed_zncc | 20-40 | 100 | 3 | 8.7 | 19.0 | 67% | 67% | 57.3 | 10.5 | 33% | 41.0 | 0% | 100% |
| bound4_windowed_zncc | >=40 | 201 | 19 | 4.1 | 7.1 | 84% | 89% | 20.9 | 20.9 | 21% | 20.8 | 21% | 89% |
| bound6_none | <5 | 12 | 1 | 57.7 | 57.7 | 0% | 0% | 26.7 | 22.1 | 0% | 22.9 | 0% | 0% |
| bound6_none | 5-10 | 21 | 5 | 76.5 | 77.7 | 0% | 0% | 5.0 | 52.1 | 0% | 32.8 | 0% | 0% |
| bound6_none | 10-20 | 48 | 17 | 58.1 | 74.6 | 35% | 35% | 21.5 | 15.0 | 18% | 16.0 | 29% | 47% |
| bound6_none | 20-40 | 100 | 24 | 21.4 | 37.2 | 38% | 46% | 37.3 | 28.0 | 17% | 35.7 | 4% | 71% |
| bound6_none | >=40 | 201 | 62 | 4.6 | 11.2 | 74% | 84% | 27.1 | 25.0 | 23% | 27.1 | 16% | 84% |
| bound6_windowed_zncc | <5 | 12 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound6_windowed_zncc | 5-10 | 21 | 1 | 72.0 | 72.0 | 0% | 0% | 52.5 | 42.8 | 0% | 36.8 | 0% | 0% |
| bound6_windowed_zncc | 10-20 | 48 | 5 | 21.0 | 73.0 | 40% | 40% | 13.4 | 12.2 | 0% | 13.8 | 40% | 60% |
| bound6_windowed_zncc | 20-40 | 100 | 2 | 18.5 | 24.1 | 50% | 50% | 57.3 | 33.4 | 50% | 55.9 | 0% | 100% |
| bound6_windowed_zncc | >=40 | 201 | 23 | 3.2 | 6.0 | 83% | 87% | 32.4 | 31.7 | 17% | 32.3 | 17% | 87% |

**Convergence under `acceptance` `none`.** Per render, over the points whose loop reached that render with a proposal: the proposed normal's error, its median change from the same point's first plane, and the turn's largest cell movement (grid px), with the share at or under today's tolerance, 0.05 px.

`seoul_bull`:

| variant | render | points | proposed med | proposed p75 | change from first, med | move med px | move p10 | move p90 | moves <= 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| bound2_none | 1 | 186 | 25.9 | 47.7 | +0.0 | 8.45 | 3.02 | 13.81 | 0% |
| bound2_none | 2 | 160 | 23.8 | 44.0 | +0.9 | 2.21 | 0.61 | 7.69 | 1% |
| bound2_none | 3 | 119 | 22.3 | 36.2 | +0.7 | 0.85 | 0.11 | 4.92 | 7% |
| bound2_none | 4 | 76 | 21.0 | 38.2 | +2.4 | 0.68 | 0.08 | 3.03 | 7% |
| bound2_none | 5 | 41 | 19.5 | 34.4 | +2.4 | 0.38 | 0.05 | 1.85 | 10% |
| bound2_none | 6 | 18 | 18.7 | 30.1 | +2.8 | 0.24 | 0.10 | 1.00 | 6% |
| bound2_none | 7 | 9 | 19.6 | 23.4 | +9.6 | 0.36 | 0.12 | 0.67 | 0% |
| bound2_none | 8 | 4 | 16.8 | 29.4 | +8.9 | 0.14 | 0.08 | 0.30 | 0% |
| bound4_none | 1 | 203 | 24.6 | 49.5 | +0.0 | 9.47 | 3.67 | 14.24 | 0% |
| bound4_none | 2 | 176 | 23.8 | 44.5 | +1.7 | 2.17 | 0.58 | 7.95 | 1% |
| bound4_none | 3 | 139 | 21.7 | 40.3 | +1.3 | 0.77 | 0.08 | 3.95 | 8% |
| bound4_none | 4 | 92 | 18.4 | 37.9 | +2.2 | 0.40 | 0.04 | 1.89 | 11% |
| bound4_none | 5 | 50 | 15.2 | 31.2 | +2.0 | 0.21 | 0.04 | 1.49 | 14% |
| bound4_none | 6 | 22 | 10.2 | 20.3 | -1.2 | 0.15 | 0.03 | 0.88 | 23% |
| bound4_none | 7 | 6 | 10.6 | 17.5 | -2.2 | 0.14 | 0.05 | 0.22 | 17% |
| bound4_none | 8 | 2 | 8.1 | 10.2 | +2.1 | 0.47 | 0.17 | 0.77 | 0% |
| bound6_none | 1 | 211 | 24.4 | 49.5 | +0.0 | 10.25 | 4.32 | 14.87 | 0% |
| bound6_none | 2 | 182 | 23.8 | 45.3 | +2.1 | 2.11 | 0.45 | 5.70 | 3% |
| bound6_none | 3 | 146 | 22.4 | 42.3 | +2.4 | 0.75 | 0.04 | 4.51 | 12% |
| bound6_none | 4 | 89 | 17.7 | 35.7 | +2.6 | 0.39 | 0.02 | 1.96 | 17% |
| bound6_none | 5 | 49 | 14.6 | 27.1 | +0.8 | 0.25 | 0.02 | 1.36 | 18% |
| bound6_none | 6 | 18 | 10.8 | 16.8 | +0.8 | 0.24 | 0.04 | 1.03 | 22% |
| bound6_none | 7 | 6 | 8.4 | 9.7 | +4.8 | 0.22 | 0.08 | 0.78 | 17% |
| bound6_none | 8 | 1 | 10.5 | 10.5 | +5.0 | 0.16 | 0.16 | 0.16 | 0% |

`kerry_park`:

| variant | render | points | proposed med | proposed p75 | change from first, med | move med px | move p10 | move p90 | moves <= 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| bound2_none | 1 | 229 | 16.9 | 33.3 | +0.0 | 6.75 | 1.50 | 12.39 | 1% |
| bound2_none | 2 | 207 | 12.9 | 31.8 | -0.3 | 1.51 | 0.30 | 4.81 | 5% |
| bound2_none | 3 | 161 | 11.0 | 30.3 | -0.8 | 0.67 | 0.06 | 3.16 | 8% |
| bound2_none | 4 | 106 | 9.9 | 22.5 | -1.7 | 0.27 | 0.03 | 1.71 | 14% |
| bound2_none | 5 | 52 | 7.2 | 28.3 | -2.4 | 0.16 | 0.02 | 1.73 | 27% |
| bound2_none | 6 | 16 | 9.1 | 18.3 | -3.1 | 0.24 | 0.05 | 1.07 | 12% |
| bound2_none | 7 | 5 | 15.9 | 29.5 | -1.7 | 0.35 | 0.10 | 1.52 | 20% |
| bound2_none | 8 | 2 | 11.6 | 13.9 | -7.1 | 0.78 | 0.35 | 1.21 | 0% |
| bound4_none | 1 | 274 | 15.4 | 35.1 | +0.0 | 8.05 | 2.10 | 13.60 | 0% |
| bound4_none | 2 | 253 | 13.1 | 29.9 | -0.1 | 1.70 | 0.25 | 4.96 | 5% |
| bound4_none | 3 | 215 | 10.3 | 30.2 | +0.0 | 0.60 | 0.02 | 2.97 | 13% |
| bound4_none | 4 | 147 | 10.5 | 27.8 | -0.2 | 0.43 | 0.03 | 2.08 | 15% |
| bound4_none | 5 | 80 | 6.2 | 22.3 | -1.1 | 0.18 | 0.02 | 1.55 | 24% |
| bound4_none | 6 | 36 | 8.6 | 24.4 | +0.4 | 0.24 | 0.03 | 1.35 | 17% |
| bound4_none | 7 | 11 | 12.1 | 18.8 | +2.1 | 0.09 | 0.01 | 1.73 | 27% |
| bound4_none | 8 | 4 | 5.7 | 9.1 | +1.2 | 0.52 | 0.19 | 1.13 | 0% |
| bound6_none | 1 | 285 | 17.4 | 40.4 | +0.0 | 8.71 | 2.22 | 13.84 | 0% |
| bound6_none | 2 | 262 | 13.4 | 35.1 | -0.1 | 1.46 | 0.21 | 4.65 | 7% |
| bound6_none | 3 | 218 | 10.8 | 30.6 | -0.5 | 0.57 | 0.01 | 2.75 | 15% |
| bound6_none | 4 | 146 | 7.3 | 25.1 | -1.1 | 0.26 | 0.02 | 2.42 | 18% |
| bound6_none | 5 | 78 | 6.1 | 23.3 | -1.4 | 0.17 | 0.02 | 0.98 | 27% |
| bound6_none | 6 | 32 | 6.4 | 27.5 | -2.3 | 0.13 | 0.02 | 0.79 | 19% |
| bound6_none | 7 | 18 | 4.7 | 12.8 | -1.8 | 0.10 | 0.06 | 0.64 | 6% |
| bound6_none | 8 | 11 | 6.3 | 12.7 | -0.7 | 0.10 | 0.02 | 0.28 | 27% |

`seoul_bull`:

| variant | points with 2+ proposals | first is best | last better than first | last worse than first by 5+ deg |
|---|---|---|---|---|
| bound2_none | 160 | 49% | 39% | 39% |
| bound4_none | 176 | 51% | 38% | 40% |
| bound6_none | 182 | 56% | 34% | 43% |

`kerry_park`:

| variant | points with 2+ proposals | first is best | last better than first | last worse than first by 5+ deg |
|---|---|---|---|---|
| bound2_none | 207 | 39% | 54% | 26% |
| bound4_none | 253 | 42% | 49% | 25% |
| bound6_none | 262 | 41% | 48% | 24% |

**Turns the ZNCC rule rejected** (the `windowed_zncc` variants): whether the rejected turn's normal was closer to the truth than the frame it was proposed from.

`seoul_bull`:

| variant | baseline | rejected | would lower the error | by 5+ deg | would raise it by 5+ deg |
|---|---|---|---|---|---|
| bound2_windowed_zncc | <5 | 38 | 21% | 18% | 66% |
| bound2_windowed_zncc | 5-10 | 43 | 23% | 16% | 60% |
| bound2_windowed_zncc | 10-20 | 4 | 50% | 25% | 50% |
| bound2_windowed_zncc | 20-40 | 9 | 33% | 11% | 33% |
| bound2_windowed_zncc | >=40 | 57 | 37% | 14% | 23% |
| bound4_windowed_zncc | <5 | 41 | 22% | 20% | 61% |
| bound4_windowed_zncc | 5-10 | 45 | 22% | 18% | 60% |
| bound4_windowed_zncc | 10-20 | 8 | 38% | 12% | 62% |
| bound4_windowed_zncc | 20-40 | 13 | 62% | 46% | 15% |
| bound4_windowed_zncc | >=40 | 68 | 43% | 16% | 18% |
| bound6_windowed_zncc | <5 | 43 | 30% | 28% | 56% |
| bound6_windowed_zncc | 5-10 | 46 | 28% | 22% | 54% |
| bound6_windowed_zncc | 10-20 | 10 | 40% | 20% | 60% |
| bound6_windowed_zncc | 20-40 | 17 | 59% | 35% | 12% |
| bound6_windowed_zncc | >=40 | 67 | 37% | 15% | 27% |

`kerry_park`:

| variant | baseline | rejected | would lower the error | by 5+ deg | would raise it by 5+ deg |
|---|---|---|---|---|---|
| bound2_windowed_zncc | <5 | 3 | 0% | 0% | 67% |
| bound2_windowed_zncc | 5-10 | 6 | 0% | 0% | 50% |
| bound2_windowed_zncc | 10-20 | 30 | 40% | 27% | 47% |
| bound2_windowed_zncc | 20-40 | 53 | 47% | 34% | 38% |
| bound2_windowed_zncc | >=40 | 79 | 56% | 28% | 19% |
| bound4_windowed_zncc | <5 | 4 | 0% | 0% | 75% |
| bound4_windowed_zncc | 5-10 | 7 | 14% | 14% | 71% |
| bound4_windowed_zncc | 10-20 | 35 | 29% | 20% | 51% |
| bound4_windowed_zncc | 20-40 | 61 | 46% | 36% | 34% |
| bound4_windowed_zncc | >=40 | 112 | 54% | 28% | 15% |
| bound6_windowed_zncc | <5 | 5 | 0% | 0% | 80% |
| bound6_windowed_zncc | 5-10 | 8 | 12% | 12% | 75% |
| bound6_windowed_zncc | 10-20 | 33 | 27% | 15% | 52% |
| bound6_windowed_zncc | 20-40 | 66 | 50% | 39% | 26% |
| bound6_windowed_zncc | >=40 | 117 | 50% | 30% | 21% |

**The first plane's error against its predicted precision** (degrees):

`seoul_bull`:

| variant | precision | points | first med | first p75 | first <10 | first <20 | start med |
|---|---|---|---|---|---|---|---|
| bound2 | <2 | 24 | 23.9 | 39.3 | 25% | 46% | 35.7 |
| bound2 | 2-5 | 55 | 18.9 | 27.5 | 22% | 55% | 34.1 |
| bound2 | 5-10 | 42 | 25.6 | 46.3 | 10% | 21% | 30.4 |
| bound2 | 10-20 | 43 | 44.3 | 58.0 | 12% | 21% | 31.0 |
| bound2 | 20-45 | 15 | 36.6 | 40.0 | 0% | 7% | 21.3 |
| bound2 | >=45 | 7 | 47.5 | 51.0 | 0% | 0% | 66.9 |
| bound4 | <2 | 46 | 11.8 | 22.9 | 41% | 67% | 36.5 |
| bound4 | 2-5 | 57 | 21.4 | 32.3 | 12% | 47% | 34.0 |
| bound4 | 5-10 | 42 | 31.9 | 57.8 | 12% | 21% | 27.8 |
| bound4 | 10-20 | 32 | 42.4 | 58.7 | 9% | 12% | 26.6 |
| bound4 | 20-45 | 20 | 40.1 | 52.0 | 5% | 10% | 27.6 |
| bound4 | >=45 | 6 | 50.5 | 55.3 | 0% | 0% | 38.7 |
| bound6 | <2 | 54 | 12.0 | 23.3 | 35% | 70% | 38.2 |
| bound6 | 2-5 | 55 | 20.6 | 34.9 | 16% | 45% | 33.5 |
| bound6 | 5-10 | 42 | 31.9 | 57.1 | 14% | 26% | 26.0 |
| bound6 | 10-20 | 35 | 40.5 | 60.9 | 9% | 14% | 29.4 |
| bound6 | 20-45 | 18 | 38.8 | 49.1 | 0% | 0% | 23.6 |
| bound6 | >=45 | 7 | 53.5 | 58.0 | 0% | 0% | 66.9 |

`kerry_park`:

| variant | precision | points | first med | first p75 | first <10 | first <20 | start med |
|---|---|---|---|---|---|---|---|
| bound2 | <2 | 52 | 12.0 | 34.9 | 40% | 60% | 23.3 |
| bound2 | 2-5 | 85 | 13.2 | 26.8 | 33% | 65% | 26.2 |
| bound2 | 5-10 | 56 | 18.1 | 31.4 | 32% | 54% | 31.8 |
| bound2 | 10-20 | 30 | 21.7 | 45.1 | 7% | 50% | 32.8 |
| bound2 | 20-45 | 6 | 34.9 | 48.8 | 0% | 17% | 33.8 |
| bound2 | >=45 | 0 | - | - | - | - | - |
| bound4 | <2 | 82 | 8.8 | 27.9 | 51% | 72% | 25.4 |
| bound4 | 2-5 | 104 | 15.7 | 30.4 | 33% | 59% | 26.4 |
| bound4 | 5-10 | 62 | 20.1 | 40.0 | 26% | 48% | 33.6 |
| bound4 | 10-20 | 22 | 26.6 | 56.5 | 5% | 36% | 37.2 |
| bound4 | 20-45 | 4 | 57.6 | 63.7 | 0% | 25% | 27.0 |
| bound4 | >=45 | 0 | - | - | - | - | - |
| bound6 | <2 | 96 | 12.3 | 34.7 | 48% | 64% | 25.6 |
| bound6 | 2-5 | 106 | 17.2 | 38.4 | 33% | 57% | 30.8 |
| bound6 | 5-10 | 56 | 21.3 | 41.3 | 27% | 48% | 29.9 |
| bound6 | 10-20 | 20 | 25.1 | 48.9 | 10% | 40% | 35.6 |
| bound6 | 20-45 | 6 | 50.7 | 65.3 | 0% | 0% | 42.5 |
| bound6 | >=45 | 1 | 14.5 | 14.5 | 0% | 100% | 50.0 |

**Gates on the first plane.** Every gate over all points, and the precision bar of 5° by baseline:

`seoul_bull`:

| variant | gate | baseline | points | kept | first med | first <10 | first <20 | start med | 2x2 med | 3x3 med |
|---|---|---|---|---|---|---|---|---|---|---|
| bound2 | precision <= 5 | <5 | 67 | 3 | 63.1 | 0% | 0% | 14.7 | 69.5 | 53.6 |
| bound2 | precision <= 5 | 5-10 | 58 | 14 | 41.9 | 7% | 14% | 40.9 | 24.3 | 24.9 |
| bound2 | precision <= 5 | 10-20 | 14 | 4 | 39.7 | 25% | 50% | 32.8 | 13.1 | 11.0 |
| bound2 | precision <= 5 | 20-40 | 30 | 7 | 23.7 | 29% | 43% | 39.9 | 16.0 | 28.5 |
| bound2 | precision <= 5 | >=40 | 97 | 51 | 15.1 | 27% | 67% | 35.0 | 23.4 | 31.0 |
| bound2 | precision <= 5 | all | 266 | 79 | 19.7 | 23% | 52% | 35.0 | 23.5 | 30.6 |
| bound2 | precision <= 10 | all | 266 | 121 | 23.7 | 18% | 41% | 33.3 | 25.2 | 31.0 |
| bound2 | precision <= 15 | all | 266 | 150 | 24.2 | 17% | 37% | 32.7 | 25.9 | 31.3 |
| bound2 | precision <= 20 | all | 266 | 164 | 25.4 | 16% | 36% | 32.7 | 26.7 | 31.6 |
| bound2 | baseline >= 5 | all | 266 | 140 | 24.2 | 17% | 39% | 32.6 | 25.3 | 30.9 |
| bound2 | baseline >= 10 | all | 266 | 90 | 19.3 | 24% | 52% | 34.0 | 23.6 | 30.7 |
| bound2 | baseline >= 20 | all | 266 | 82 | 18.2 | 26% | 55% | 34.5 | 24.2 | 30.9 |
| bound4 | precision <= 5 | <5 | 67 | 6 | 62.5 | 0% | 17% | 12.6 | 68.8 | 53.1 |
| bound4 | precision <= 5 | 5-10 | 58 | 15 | 33.1 | 7% | 7% | 40.6 | 25.1 | 31.1 |
| bound4 | precision <= 5 | 10-20 | 14 | 6 | 45.1 | 17% | 33% | 32.8 | 16.8 | 12.6 |
| bound4 | precision <= 5 | 20-40 | 30 | 13 | 16.9 | 23% | 54% | 40.9 | 28.2 | 39.0 |
| bound4 | precision <= 5 | >=40 | 97 | 63 | 12.7 | 33% | 75% | 34.9 | 24.8 | 31.5 |
| bound4 | precision <= 5 | all | 266 | 103 | 16.2 | 25% | 56% | 35.0 | 25.4 | 32.7 |
| bound4 | precision <= 10 | all | 266 | 145 | 21.6 | 21% | 46% | 33.5 | 27.0 | 31.7 |
| bound4 | precision <= 15 | all | 266 | 167 | 23.1 | 20% | 41% | 32.8 | 27.1 | 32.0 |
| bound4 | precision <= 20 | all | 266 | 177 | 23.2 | 19% | 40% | 32.8 | 27.4 | 32.5 |
| bound4 | baseline >= 5 | all | 266 | 154 | 22.7 | 21% | 44% | 32.7 | 25.3 | 31.1 |
| bound4 | baseline >= 10 | all | 266 | 103 | 14.4 | 29% | 59% | 34.1 | 23.4 | 30.9 |
| bound4 | baseline >= 20 | all | 266 | 92 | 13.5 | 32% | 64% | 35.1 | 24.2 | 31.4 |
| bound6 | precision <= 5 | <5 | 67 | 6 | 62.5 | 0% | 17% | 12.6 | 68.8 | 53.1 |
| bound6 | precision <= 5 | 5-10 | 58 | 17 | 34.3 | 6% | 6% | 31.1 | 25.1 | 31.1 |
| bound6 | precision <= 5 | 10-20 | 14 | 5 | 12.5 | 20% | 60% | 33.2 | 18.1 | 13.1 |
| bound6 | precision <= 5 | 20-40 | 30 | 14 | 16.4 | 29% | 64% | 43.3 | 27.6 | 39.5 |
| bound6 | precision <= 5 | >=40 | 97 | 67 | 12.4 | 33% | 73% | 35.0 | 24.9 | 31.5 |
| bound6 | precision <= 5 | all | 266 | 109 | 16.4 | 26% | 58% | 35.3 | 26.3 | 32.9 |
| bound6 | precision <= 10 | all | 266 | 151 | 20.1 | 23% | 49% | 33.8 | 27.1 | 32.2 |
| bound6 | precision <= 15 | all | 266 | 170 | 23.0 | 21% | 44% | 33.3 | 27.5 | 32.8 |
| bound6 | precision <= 20 | all | 266 | 186 | 23.4 | 20% | 42% | 33.2 | 27.7 | 32.8 |
| bound6 | baseline >= 5 | all | 266 | 161 | 22.8 | 21% | 45% | 33.2 | 25.4 | 31.3 |
| bound6 | baseline >= 10 | all | 266 | 109 | 14.8 | 29% | 62% | 34.9 | 23.7 | 31.0 |
| bound6 | baseline >= 20 | all | 266 | 98 | 13.5 | 32% | 66% | 35.3 | 24.7 | 31.6 |

`kerry_park`:

| variant | gate | baseline | points | kept | first med | first <10 | first <20 | start med | 2x2 med | 3x3 med |
|---|---|---|---|---|---|---|---|---|---|---|
| bound2 | precision <= 5 | <5 | 12 | 2 | 78.2 | 0% | 0% | 0.7 | 68.2 | 45.1 |
| bound2 | precision <= 5 | 5-10 | 21 | 1 | 70.6 | 0% | 0% | 5.0 | 43.4 | 32.8 |
| bound2 | precision <= 5 | 10-20 | 48 | 20 | 35.6 | 20% | 35% | 28.3 | 26.7 | 20.5 |
| bound2 | precision <= 5 | 20-40 | 100 | 29 | 14.6 | 28% | 62% | 36.1 | 16.9 | 30.4 |
| bound2 | precision <= 5 | >=40 | 201 | 85 | 11.0 | 44% | 72% | 21.8 | 20.0 | 20.5 |
| bound2 | precision <= 5 | all | 382 | 137 | 13.2 | 36% | 63% | 25.4 | 20.6 | 21.9 |
| bound2 | precision <= 10 | all | 382 | 193 | 14.6 | 35% | 60% | 25.6 | 20.6 | 23.4 |
| bound2 | precision <= 15 | all | 382 | 214 | 16.1 | 32% | 60% | 26.1 | 21.6 | 23.5 |
| bound2 | precision <= 20 | all | 382 | 223 | 16.6 | 31% | 59% | 26.7 | 21.8 | 23.5 |
| bound2 | baseline >= 5 | all | 382 | 224 | 16.5 | 31% | 59% | 27.6 | 21.7 | 23.7 |
| bound2 | baseline >= 10 | all | 382 | 216 | 16.0 | 32% | 60% | 27.6 | 21.4 | 23.5 |
| bound2 | baseline >= 20 | all | 382 | 180 | 13.5 | 35% | 64% | 29.0 | 21.4 | 25.0 |
| bound4 | precision <= 5 | <5 | 12 | 4 | 78.5 | 0% | 0% | 0.8 | 77.0 | 70.6 |
| bound4 | precision <= 5 | 5-10 | 21 | 4 | 65.6 | 0% | 0% | 11.2 | 48.2 | 35.4 |
| bound4 | precision <= 5 | 10-20 | 48 | 29 | 44.9 | 21% | 34% | 13.4 | 29.0 | 16.7 |
| bound4 | precision <= 5 | 20-40 | 100 | 41 | 14.6 | 37% | 59% | 37.0 | 22.6 | 33.3 |
| bound4 | precision <= 5 | >=40 | 201 | 108 | 9.4 | 51% | 80% | 23.4 | 20.7 | 21.4 |
| bound4 | precision <= 5 | all | 382 | 186 | 13.1 | 41% | 65% | 25.5 | 22.7 | 23.4 |
| bound4 | precision <= 10 | all | 382 | 248 | 13.8 | 37% | 60% | 26.9 | 23.5 | 25.0 |
| bound4 | precision <= 15 | all | 382 | 260 | 13.9 | 36% | 60% | 27.1 | 24.3 | 25.0 |
| bound4 | precision <= 20 | all | 382 | 270 | 14.9 | 34% | 59% | 28.0 | 24.6 | 25.5 |
| bound4 | baseline >= 5 | all | 382 | 265 | 14.1 | 35% | 60% | 30.2 | 24.3 | 25.3 |
| bound4 | baseline >= 10 | all | 382 | 253 | 13.7 | 37% | 62% | 30.3 | 22.6 | 24.8 |
| bound4 | baseline >= 20 | all | 382 | 211 | 13.1 | 41% | 67% | 31.9 | 22.8 | 27.0 |
| bound6 | precision <= 5 | <5 | 12 | 4 | 78.6 | 0% | 0% | 1.6 | 77.8 | 71.8 |
| bound6 | precision <= 5 | 5-10 | 21 | 6 | 72.2 | 0% | 0% | 4.3 | 49.4 | 36.7 |
| bound6 | precision <= 5 | 10-20 | 48 | 31 | 47.6 | 19% | 35% | 13.4 | 17.7 | 16.7 |
| bound6 | precision <= 5 | 20-40 | 100 | 43 | 15.9 | 37% | 60% | 36.1 | 25.7 | 33.3 |
| bound6 | precision <= 5 | >=40 | 201 | 118 | 10.3 | 50% | 71% | 25.3 | 22.4 | 22.9 |
| bound6 | precision <= 5 | all | 382 | 202 | 13.9 | 40% | 60% | 26.9 | 24.0 | 25.3 |
| bound6 | precision <= 10 | all | 382 | 258 | 15.6 | 37% | 57% | 27.6 | 25.4 | 26.3 |
| bound6 | precision <= 15 | all | 382 | 271 | 16.0 | 36% | 57% | 29.5 | 25.6 | 26.6 |
| bound6 | precision <= 20 | all | 382 | 278 | 17.0 | 35% | 56% | 29.4 | 25.4 | 26.3 |
| bound6 | baseline >= 5 | all | 382 | 276 | 16.2 | 36% | 57% | 31.5 | 25.4 | 26.7 |
| bound6 | baseline >= 10 | all | 382 | 264 | 15.0 | 37% | 59% | 32.1 | 24.3 | 25.8 |
| bound6 | baseline >= 20 | all | 382 | 222 | 13.4 | 41% | 64% | 32.6 | 24.6 | 29.0 |

**Precision gates on the step as built** (`Turned` points; bars of 5° and 10°):

`seoul_bull`:

| variant | gate | turned | kept | new med | new <10 | new <20 | start med | 2x2 med | 3x3 med |
|---|---|---|---|---|---|---|---|---|---|
| bound2_none | precision <= 5 | 19 | 14 | 19.1 | 21% | 50% | 35.7 | 24.5 | 26.5 |
| bound2_none | precision <= 10 | 19 | 18 | 22.7 | 22% | 44% | 32.2 | 26.2 | 26.5 |
| bound2_windowed_zncc | precision <= 5 | 8 | 5 | 15.6 | 20% | 60% | 39.6 | 27.0 | 34.8 |
| bound2_windowed_zncc | precision <= 10 | 8 | 7 | 15.6 | 29% | 57% | 39.0 | 27.0 | 34.8 |
| bound4_none | precision <= 5 | 35 | 26 | 18.5 | 27% | 54% | 35.3 | 29.9 | 33.9 |
| bound4_none | precision <= 10 | 35 | 31 | 20.7 | 26% | 48% | 35.2 | 24.6 | 31.7 |
| bound4_windowed_zncc | precision <= 5 | 10 | 6 | 26.8 | 0% | 17% | 40.7 | 35.3 | 42.6 |
| bound4_windowed_zncc | precision <= 10 | 10 | 9 | 24.1 | 11% | 22% | 40.5 | 29.4 | 27.7 |
| bound6_none | precision <= 5 | 53 | 39 | 18.1 | 23% | 51% | 35.4 | 27.0 | 31.5 |
| bound6_none | precision <= 10 | 53 | 45 | 23.2 | 22% | 47% | 34.1 | 27.0 | 31.5 |
| bound6_windowed_zncc | precision <= 5 | 13 | 9 | 16.1 | 22% | 56% | 40.5 | 21.5 | 31.5 |
| bound6_windowed_zncc | precision <= 10 | 13 | 12 | 21.9 | 25% | 50% | 40.3 | 21.8 | 33.2 |

`kerry_park`:

| variant | gate | turned | kept | new med | new <10 | new <20 | start med | 2x2 med | 3x3 med |
|---|---|---|---|---|---|---|---|---|---|
| bound2_none | precision <= 5 | 59 | 52 | 6.4 | 71% | 77% | 25.4 | 16.5 | 19.9 |
| bound2_none | precision <= 10 | 59 | 55 | 6.4 | 71% | 76% | 25.4 | 16.0 | 19.8 |
| bound2_windowed_zncc | precision <= 5 | 28 | 26 | 4.9 | 73% | 81% | 23.9 | 15.9 | 20.0 |
| bound2_windowed_zncc | precision <= 10 | 28 | 28 | 4.9 | 75% | 82% | 23.9 | 15.9 | 20.0 |
| bound4_none | precision <= 5 | 92 | 81 | 6.0 | 67% | 70% | 26.7 | 22.6 | 23.5 |
| bound4_none | precision <= 10 | 92 | 88 | 6.6 | 64% | 67% | 27.1 | 24.0 | 25.1 |
| bound4_windowed_zncc | precision <= 5 | 26 | 23 | 5.4 | 78% | 83% | 31.9 | 16.3 | 20.8 |
| bound4_windowed_zncc | precision <= 10 | 26 | 26 | 5.9 | 77% | 81% | 34.7 | 18.4 | 25.7 |
| bound6_none | precision <= 5 | 109 | 96 | 6.3 | 64% | 71% | 31.5 | 26.5 | 26.8 |
| bound6_none | precision <= 10 | 109 | 104 | 7.0 | 59% | 66% | 31.5 | 25.6 | 25.8 |
| bound6_windowed_zncc | precision <= 5 | 31 | 28 | 4.1 | 79% | 82% | 35.0 | 28.7 | 31.2 |
| bound6_windowed_zncc | precision <= 10 | 31 | 30 | 4.8 | 73% | 77% | 35.0 | 25.6 | 31.2 |

**What the numbers show.**

- **The views' baseline divides the points.** Where it is 20° or more (127 points on `seoul_bull`, 301 on `kerry_park`), the first plane at bound 4 lands at a median 19.6° and 13.1° (20–40°, 40° and over) on `seoul_bull` and 19.1° and 11.4° on `kerry_park`, against a start of 25° to 41° and today's 2×2 at 22.6° to 25.7°, and is closer to the truth than the start on 74% to 89% of the points that propose one. Under 10°, it is worse than the start in median on both captures (40° to 47° against 29° to 30° on `seoul_bull`, 65° to 77° against 1° and 21° on `kerry_park`) and closer than the start on only 0% to 39%. Today's 2×2 is also worse than the start under 5° (45.0° against 26.1°, 71.1° against 0.7°). The 10–20° bin is mixed: the first plane is worse than the start in median there on both.
- **The step as built turns its best points where the baseline is wide.** At 40° and over on `kerry_park` its turned points end at a median 3.2° to 5.3° in every variant, against today's 17° to 32° on the same points. Its turned points under 10° are few (0 to 11 per bin and variant) and mostly far off (medians from 11° to 77°, most above 40°).
- **The ZNCC rule vetoes useful turns at wide baselines and bad ones at narrow baselines.** Of the turns it rejected, those that would have lowered the error are 37% to 43% (`seoul_bull`) and 50% to 56% (`kerry_park`) at 40° and over, and 0% to 30% under 10°, where 50% to 80% would have raised it by 5° or more. It acts in part as a narrow-baseline guard.
- **Dropping the acceptance test turns more points, but the loop rarely converges.** At bound 2 it turns 19 and 59 points instead of 8 and 28, at bound 6 53 and 109. Oscillation then ends 108 to 147 loops per variant, more than converge, and more loops end `measured`, with cells that fix no axis (`seoul_bull` bound 2: 93 to 132; `kerry_park` bound 2: 163 to 182), likely because a turn the score would have refused leaves the next render's cells refused (inferred, not traced).
- **Later renders help a little on `kerry_park` and drift on `seoul_bull`.** Over renders 2 to 6, the median change of a point's proposal from its own first plane is mostly between 0 and −3° on `kerry_park` and between +1° and +3° on `seoul_bull`. The first plane is the best of a point's proposals on 49% to 56% (`seoul_bull`) and 39% to 42% (`kerry_park`) of points with two or more, and the last is 5° or more worse than the first on 39% to 43% and 24% to 26%.
- **Today's tolerance is far below the movements the loop settles to.** The largest cell movement falls from a median 7 to 10 px at the first render to 1.5 to 2.2 px at the second and 0.1 to 0.9 px from the third on, and at no render are more than 27% of the turns at or under 0.05 px. A cell centre lies about `√2 R / 3`, 11 to 23 px, from the middle on the 24- to 48-sample grids, so 0.05 px is a turn of about 0.13° to 0.25°, while 0.5 px would be about 1.3° to 2.5°.
- **The predicted precision ranks the wide-baseline planes but does not catch the narrow-baseline failures.** Over all first planes the median error rises with the predicted precision (bound 4: 11.8° under 2° to 40.1° at 20–45° on `seoul_bull`; 8.8° to 57.6° on `kerry_park`). But under 10° baseline the first planes predicted within 5° have median errors of 33° to 79°, and those planes are turned a median 68° (`seoul_bull`) and 78° (`kerry_park`) from the mean viewing direction, against 28° to 34° for the planes at 20° and over. The precision reads each cell's variance along the fitted normal; a plane turned nearly edge-on to the views reads it across the rays, where even close views place a cell well, so a wrong edge-on plane reads as precise. This explanation is inferred from those two medians, not tested.
- **A baseline gate does as well as the precision gate over all points.** On the first plane at bound 4, a precision bar of 5° keeps 103 of 266 points at a median 16.2° and 186 of 382 at 13.1°; a baseline bar of 20° keeps 92 at 13.5° and 211 at 13.1°, and 10° keeps 103 at 14.4° and 253 at 13.7°. On the step as built, a precision bar of 5° keeps 88% to 93% of the turned points on `kerry_park`, at medians of 4.1° to 6.4°.
- **A wider bound helps a little.** Bound 4 makes more first renders propose (203 against 186, 274 against 229) and is as accurate or better at 20° and over (`seoul_bull` 40° and over 13.1° against 18.1°); bound 6 adds few proposals over 4 and costs more on `kerry_park`.

**What the numbers cannot show.**

- **The ground-truth normals are estimates**, the patch frames stored in the two `.sfmr` files, and per the maintainer most of them were set with today's Grid Plane Normal at 3×3 pieces, so agreement with them partly measures agreement with that method (see [today's step at 3×3](#addendum-todays-step-at-33-and-a-third-capture-2026-10-15)). Where a step disagrees with one, the numbers cannot say which is wrong. On `kerry_park` the 12 points under 5° baseline have a ground-truth normal a median 0.7° from their mean viewing direction, which suggests that those normals were set from the viewing direction rather than measured. If so, the errors under 10° there measure distance from the viewing direction, not from the surface. On `seoul_bull` that bin's start error is 26°, so this does not hold there.
- **Narrow baselines are measured, and these estimators fail on them against these truths**; whether a better measurement could succeed there is not shown. The narrow bins on `kerry_park` are small (12 and 21 points).
- The first plane and the gates are read from the traces after the fact. A gate inside the loop, or a larger tolerance, changes the later renders, which these runs do not show.
- Whether a person prefers the turned patches; the hand review decides that.

**Candidates the data supports** (for the maintainer; none is adopted here):

- A **baseline gate** on the step (propose no turn below about 20° between the views), as the guard the narrow baselines need. The precision gate as defined is not a substitute, unless its variance is read so that an edge-on plane is not reported as precise.
- With that gate in place, **taking the first plane, or turns without the ZNCC veto**, at wide baselines: there the first plane is closer to the truth than the start on three quarters or more of the points, and the veto rejects about half the turns that would help.
- **`cell_shift_bound_px` 4** rather than 2.
- An **`update_tolerance_px` near the movements seen from the third render on** (0.1 to 0.9 px), or a fixed small number of renders, instead of 0.05 px, which few loops reach. Not measured here.

## Addendum: today's step at 3×3 and a third capture (2026-10-15)

**Why.** Per the maintainer, most of the two ground truths' normals were set with today's Grid Plane Normal at 3×3 pieces, not at Track View's default 2×2, so agreement with them partly measures agreement with that method. This addendum runs today's step at 3×3 beside 2×2 on every point, and repeats the sweep above on a third capture.

**Setup.**

- **Today's step at 3×3.** `grid_plane_eval.py --also-old-pieces 3` runs `finite_difference_normal` with `layout="grid"`, `pieces=3`, `overlap=0.0` from the same fitted track as every other estimator and records it in `old3_*` fields beside the 2×2 `old_*` ones. `grid_plane_sweep.py` adds a 3×3 column wherever 2×2 appears; the tables of the sweep above now carry it. All variants were run again with it; every value the sweep above reported came out the same to the last printed digit.
- **The third capture.** `dino_dog_toy_ground_truth_candidate_v006.sfmr` from another checkout's `dino_dog_toy_ws` workspace: 85 images at 2040×1536, 961 finite points seen in two or more images, a median track of 13 images. It is a candidate ground truth, not an approved one. `prepare` was extended to find the workspace above the `.sfmr` (the file sits in `sfmr/` under the workspace), to copy it to the same relative path in the cache workspace so that it resolves its workspace to the cache, and to stop if it does not; a listing of the source workspace before and after showed no file of it changed by the runs. All 961 points were run, no sample. Its baseline bins hold 9, 25, 45, 179 and 703 points.
- **Variants, estimators, errors.** As in the sweep above: bound 2, 4, 6 × acceptance `windowed_zncc`, `none`, `max_iterations` 8, from the fitted mean-view start, angles to the stored patch normal folded to 0–90°.
- **Commit and machine.** Commit `5f660960` for the scripts (the step and binding as at `a26e3483`); Intel Core i9-14900HX, 64 GB, Windows 11, release build.
- **Commands.** As in the sweep above, with `--also-old-pieces 3` added to each run, and for the third capture
  `C:/Dev/prod3/sfmtool/dino_dog_toy_ws/sfmr/dino_dog_toy_ground_truth_candidate_v006.sfmr` as `DATASET`, written to `OUT/dino_dog_toy/boundB_A`. A run took 20 to 32 s on `seoul_bull`, 77 to 107 s on `kerry_park` and 649 to 1235 s on `dino_dog_toy`, loading included. The median call of today's step took 21, 38 and 103 ms at 2×2 and 47, 87 and 227 ms at 3×3; `grid_plane_normal` 4 to 6, 7 to 13 and 31 to 108 ms.

**Today's step at 2×2 and 3×3, all points** (degrees; median / p75 / under 10° / under 20°):

| dataset | points | start | 2×2 | 3×3 | 3×3 closer than 2×2 | angle between the two, median |
|---|---|---|---|---|---|---|
| `seoul_bull` | 266 | 33.7 / 47.6 / 11% / 24% | 31.5 / 48.6 / 9% / 26% | 37.5 / 52.2 / 4% / 17% | 31% | 12.6 |
| `kerry_park` | 382 | 34.6 / 51.7 / 15% / 31% | 29.3 / 47.6 / 16% / 36% | 31.3 / 50.9 / 13% / 34% | 37% | 7.5 |
| `dino_dog_toy` | 961 | 30.9 / 43.5 / 6% / 23% | 21.2 / 33.8 / 27% / 48% | 26.8 / 38.9 / 17% / 34% | 17% | 5.7 |

By baseline, `seoul_bull`:

| baseline | points | start med | start <20 | 2x2 med | 2x2 <10 | 2x2 <20 | 2x2 closer than start | 3x3 med | 3x3 <10 | 3x3 <20 | 3x3 closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <5 | 67 | 26.1 | 39% | 45.0 | 0% | 12% | 37% | 51.5 | 1% | 7% | 31% |
| 5-10 | 58 | 28.9 | 34% | 29.9 | 16% | 33% | 45% | 32.6 | 7% | 28% | 41% |
| 10-20 | 14 | 27.0 | 43% | 18.6 | 29% | 64% | 57% | 21.4 | 21% | 50% | 50% |
| 20-40 | 30 | 47.4 | 13% | 26.5 | 13% | 33% | 80% | 41.8 | 3% | 17% | 77% |
| >=40 | 97 | 38.2 | 8% | 28.2 | 6% | 25% | 85% | 34.8 | 2% | 12% | 82% |

`kerry_park`:

| baseline | points | start med | start <20 | 2x2 med | 2x2 <10 | 2x2 <20 | 2x2 closer than start | 3x3 med | 3x3 <10 | 3x3 <20 | 3x3 closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <5 | 12 | 0.7 | 83% | 71.1 | 0% | 0% | 8% | 71.8 | 0% | 8% | 8% |
| 5-10 | 21 | 41.3 | 38% | 44.3 | 24% | 33% | 48% | 24.2 | 29% | 38% | 52% |
| 10-20 | 48 | 28.6 | 46% | 19.7 | 23% | 50% | 54% | 17.9 | 27% | 54% | 52% |
| 20-40 | 100 | 39.9 | 13% | 30.2 | 11% | 32% | 79% | 35.4 | 4% | 21% | 67% |
| >=40 | 201 | 29.5 | 33% | 26.5 | 18% | 37% | 78% | 29.1 | 14% | 36% | 71% |

`dino_dog_toy`:

| baseline | points | start med | start <20 | 2x2 med | 2x2 <10 | 2x2 <20 | 2x2 closer than start | 3x3 med | 3x3 <10 | 3x3 <20 | 3x3 closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <5 | 9 | 48.5 | 22% | 11.2 | 33% | 67% | 67% | 20.4 | 22% | 44% | 67% |
| 5-10 | 25 | 16.9 | 64% | 3.0 | 76% | 84% | 84% | 4.5 | 72% | 84% | 84% |
| 10-20 | 45 | 34.4 | 33% | 7.5 | 53% | 67% | 93% | 11.9 | 49% | 53% | 84% |
| 20-40 | 179 | 50.1 | 8% | 37.8 | 13% | 25% | 75% | 46.0 | 8% | 13% | 75% |
| >=40 | 703 | 28.1 | 25% | 19.8 | 27% | 50% | 92% | 24.9 | 15% | 36% | 87% |

**`dino_dog_toy`: outcomes.**

| variant | converged | measured | rejected | oscillation | cap | not_run |
|---|---|---|---|---|---|---|
| bound2_none | 304 | 247 | 0 | 395 | 15 | 0 |
| bound2_windowed_zncc | 105 | 208 | 589 | 59 | 0 | 0 |
| bound4_none | 330 | 175 | 0 | 446 | 10 | 0 |
| bound4_windowed_zncc | 112 | 155 | 650 | 44 | 0 | 0 |
| bound6_none | 350 | 158 | 0 | 444 | 9 | 0 |
| bound6_windowed_zncc | 118 | 140 | 675 | 28 | 0 | 0 |

**`dino_dog_toy`: the step as built, by baseline** (`Turned` points; start and today's step on the same points):

| variant | baseline | points | turned | new med | new p75 | new <10 | new <20 | start med | 2x2 med | 2x2 <10 | 3x3 med | 3x3 <10 | new closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bound2_none | <5 | 9 | 1 | 4.3 | 4.3 | 100% | 100% | 49.5 | 10.0 | 100% | 7.5 | 100% | 100% |
| bound2_none | 5-10 | 25 | 7 | 23.1 | 43.6 | 43% | 43% | 16.5 | 10.0 | 43% | 7.6 | 57% | 71% |
| bound2_none | 10-20 | 45 | 2 | 30.2 | 43.2 | 50% | 50% | 20.1 | 1.4 | 100% | 2.4 | 100% | 50% |
| bound2_none | 20-40 | 179 | 27 | 3.8 | 7.1 | 81% | 81% | 31.7 | 4.8 | 56% | 16.1 | 33% | 85% |
| bound2_none | >=40 | 703 | 267 | 1.3 | 2.8 | 90% | 94% | 25.0 | 14.4 | 40% | 21.1 | 23% | 96% |
| bound2_windowed_zncc | <5 | 9 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound2_windowed_zncc | 5-10 | 25 | 1 | 4.4 | 4.4 | 100% | 100% | 23.0 | 2.7 | 100% | 3.5 | 100% | 100% |
| bound2_windowed_zncc | 10-20 | 45 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound2_windowed_zncc | 20-40 | 179 | 5 | 1.5 | 5.0 | 100% | 100% | 18.3 | 1.0 | 100% | 5.6 | 80% | 100% |
| bound2_windowed_zncc | >=40 | 703 | 99 | 1.3 | 2.3 | 91% | 98% | 22.6 | 11.4 | 45% | 17.5 | 26% | 96% |
| bound4_none | <5 | 9 | 1 | 4.3 | 4.3 | 100% | 100% | 49.5 | 10.0 | 100% | 7.5 | 100% | 100% |
| bound4_none | 5-10 | 25 | 6 | 13.8 | 51.6 | 50% | 50% | 12.2 | 11.2 | 50% | 9.3 | 50% | 67% |
| bound4_none | 10-20 | 45 | 6 | 6.3 | 44.3 | 67% | 67% | 21.5 | 3.0 | 67% | 3.4 | 67% | 67% |
| bound4_none | 20-40 | 179 | 34 | 2.8 | 6.7 | 85% | 91% | 35.2 | 13.9 | 44% | 25.5 | 26% | 91% |
| bound4_none | >=40 | 703 | 283 | 1.2 | 2.1 | 96% | 98% | 26.0 | 15.4 | 38% | 21.6 | 21% | 98% |
| bound4_windowed_zncc | <5 | 9 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound4_windowed_zncc | 5-10 | 25 | 1 | 4.4 | 4.4 | 100% | 100% | 23.0 | 2.7 | 100% | 3.5 | 100% | 100% |
| bound4_windowed_zncc | 10-20 | 45 | 1 | 1.9 | 1.9 | 100% | 100% | 65.0 | 40.2 | 0% | 48.9 | 0% | 100% |
| bound4_windowed_zncc | 20-40 | 179 | 7 | 2.1 | 4.4 | 100% | 100% | 19.9 | 2.4 | 86% | 5.6 | 57% | 100% |
| bound4_windowed_zncc | >=40 | 703 | 103 | 1.0 | 1.8 | 98% | 98% | 24.4 | 12.0 | 42% | 18.2 | 23% | 98% |
| bound6_none | <5 | 9 | 1 | 4.3 | 4.3 | 100% | 100% | 49.5 | 10.0 | 100% | 7.5 | 100% | 100% |
| bound6_none | 5-10 | 25 | 7 | 20.5 | 46.1 | 43% | 43% | 16.5 | 10.0 | 43% | 7.6 | 57% | 71% |
| bound6_none | 10-20 | 45 | 9 | 2.4 | 8.4 | 78% | 78% | 62.5 | 14.5 | 33% | 36.5 | 33% | 78% |
| bound6_none | 20-40 | 179 | 36 | 2.8 | 7.0 | 78% | 83% | 35.1 | 12.3 | 47% | 24.6 | 28% | 92% |
| bound6_none | >=40 | 703 | 297 | 1.2 | 2.1 | 95% | 98% | 26.8 | 15.4 | 38% | 22.4 | 21% | 97% |
| bound6_windowed_zncc | <5 | 9 | 0 | - | - | - | - | - | - | - | - | - | - |
| bound6_windowed_zncc | 5-10 | 25 | 1 | 4.4 | 4.4 | 100% | 100% | 23.0 | 2.7 | 100% | 3.5 | 100% | 100% |
| bound6_windowed_zncc | 10-20 | 45 | 1 | 2.1 | 2.1 | 100% | 100% | 65.0 | 40.2 | 0% | 48.9 | 0% | 100% |
| bound6_windowed_zncc | 20-40 | 179 | 8 | 1.7 | 2.8 | 100% | 100% | 24.0 | 1.8 | 75% | 6.7 | 62% | 100% |
| bound6_windowed_zncc | >=40 | 703 | 108 | 1.1 | 2.0 | 95% | 97% | 24.4 | 12.4 | 44% | 19.2 | 26% | 98% |

**`dino_dog_toy`: the first plane, by baseline.**

| variant | baseline | points | proposed | first med | first p75 | first <10 | first <20 | start med | 2x2 med | 3x3 med | closer than start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bound2 | <5 | 9 | 6 | 12.3 | 23.3 | 50% | 67% | 49.0 | 10.8 | 11.8 | 83% |
| bound2 | 5-10 | 25 | 24 | 16.4 | 23.3 | 29% | 62% | 17.4 | 2.9 | 4.0 | 62% |
| bound2 | 10-20 | 45 | 35 | 30.3 | 58.0 | 14% | 37% | 22.6 | 5.6 | 4.4 | 49% |
| bound2 | 20-40 | 179 | 77 | 8.5 | 17.8 | 60% | 75% | 38.6 | 20.6 | 31.4 | 86% |
| bound2 | >=40 | 703 | 617 | 5.0 | 10.3 | 74% | 91% | 27.0 | 18.3 | 23.2 | 97% |
| bound4 | <5 | 9 | 6 | 12.3 | 32.7 | 50% | 67% | 49.0 | 10.8 | 11.8 | 83% |
| bound4 | 5-10 | 25 | 25 | 17.6 | 30.4 | 32% | 60% | 16.9 | 3.0 | 4.5 | 60% |
| bound4 | 10-20 | 45 | 36 | 31.7 | 56.5 | 31% | 39% | 22.8 | 5.8 | 5.1 | 50% |
| bound4 | 20-40 | 179 | 89 | 7.8 | 13.5 | 65% | 82% | 40.9 | 23.3 | 34.3 | 91% |
| bound4 | >=40 | 703 | 657 | 3.4 | 6.8 | 86% | 96% | 27.7 | 19.0 | 24.0 | 98% |
| bound6 | <5 | 9 | 6 | 12.3 | 35.2 | 50% | 67% | 49.0 | 10.8 | 11.8 | 83% |
| bound6 | 5-10 | 25 | 25 | 17.6 | 30.4 | 32% | 60% | 16.9 | 3.0 | 4.5 | 60% |
| bound6 | 10-20 | 45 | 36 | 30.0 | 56.5 | 31% | 39% | 22.8 | 5.8 | 5.1 | 50% |
| bound6 | 20-40 | 179 | 92 | 7.8 | 13.6 | 64% | 80% | 41.2 | 24.4 | 34.5 | 92% |
| bound6 | >=40 | 703 | 665 | 3.3 | 6.3 | 86% | 96% | 27.7 | 19.1 | 24.3 | 98% |

**`dino_dog_toy`: convergence under `acceptance` `none`.**

| variant | render | points | proposed med | proposed p75 | change from first, med | move med px | move p10 | move p90 | moves <= 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| bound2_none | 1 | 759 | 5.7 | 12.5 | +0.0 | 8.15 | 2.54 | 15.69 | 0% |
| bound2_none | 2 | 732 | 2.6 | 7.0 | -1.2 | 1.20 | 0.23 | 4.97 | 2% |
| bound2_none | 3 | 654 | 2.2 | 5.3 | -1.4 | 0.31 | 0.06 | 2.53 | 7% |
| bound2_none | 4 | 506 | 1.8 | 4.0 | -2.0 | 0.14 | 0.02 | 1.18 | 24% |
| bound2_none | 5 | 277 | 1.7 | 3.6 | -2.9 | 0.12 | 0.02 | 0.92 | 28% |
| bound2_none | 6 | 119 | 2.0 | 4.2 | -4.0 | 0.12 | 0.02 | 1.12 | 27% |
| bound2_none | 7 | 41 | 1.8 | 4.3 | -3.9 | 0.10 | 0.03 | 0.72 | 17% |
| bound2_none | 8 | 18 | 2.3 | 3.6 | -5.5 | 0.10 | 0.04 | 0.59 | 17% |
| bound4_none | 1 | 813 | 4.0 | 8.7 | +0.0 | 9.38 | 3.55 | 16.89 | 0% |
| bound4_none | 2 | 797 | 2.2 | 5.4 | -1.1 | 1.00 | 0.24 | 4.87 | 1% |
| bound4_none | 3 | 743 | 1.9 | 4.4 | -1.1 | 0.28 | 0.05 | 1.91 | 10% |
| bound4_none | 4 | 578 | 1.8 | 3.7 | -1.5 | 0.16 | 0.02 | 1.05 | 24% |
| bound4_none | 5 | 302 | 1.8 | 4.3 | -1.9 | 0.15 | 0.02 | 1.01 | 24% |
| bound4_none | 6 | 122 | 2.0 | 3.9 | -2.4 | 0.16 | 0.03 | 0.79 | 21% |
| bound4_none | 7 | 35 | 1.7 | 4.5 | -1.9 | 0.13 | 0.03 | 0.51 | 20% |
| bound4_none | 8 | 11 | 4.6 | 5.8 | -4.8 | 0.25 | 0.05 | 1.08 | 9% |
| bound6_none | 1 | 824 | 3.9 | 8.4 | +0.0 | 9.48 | 3.66 | 16.77 | 0% |
| bound6_none | 2 | 814 | 2.2 | 5.4 | -0.9 | 0.98 | 0.23 | 4.91 | 2% |
| bound6_none | 3 | 761 | 1.9 | 4.8 | -1.0 | 0.27 | 0.05 | 1.96 | 11% |
| bound6_none | 4 | 582 | 1.7 | 3.7 | -1.4 | 0.15 | 0.02 | 1.06 | 24% |
| bound6_none | 5 | 299 | 1.8 | 3.7 | -1.9 | 0.13 | 0.02 | 0.75 | 23% |
| bound6_none | 6 | 130 | 1.9 | 4.1 | -2.5 | 0.13 | 0.03 | 0.67 | 19% |
| bound6_none | 7 | 51 | 1.7 | 4.7 | -2.3 | 0.12 | 0.03 | 0.56 | 25% |
| bound6_none | 8 | 11 | 1.6 | 4.1 | -2.3 | 0.21 | 0.04 | 0.31 | 18% |

| variant | points with 2+ proposals | first is best | last better than first | last worse than first by 5+ deg |
|---|---|---|---|---|
| bound2_none | 732 | 24% | 70% | 6% |
| bound4_none | 797 | 25% | 68% | 6% |
| bound6_none | 814 | 25% | 67% | 6% |

**`dino_dog_toy`: turns the ZNCC rule rejected.**

| variant | baseline | rejected | would lower the error | by 5+ deg | would raise it by 5+ deg |
|---|---|---|---|---|---|
| bound2_windowed_zncc | <5 | 6 | 67% | 17% | 33% |
| bound2_windowed_zncc | 5-10 | 23 | 35% | 13% | 26% |
| bound2_windowed_zncc | 10-20 | 32 | 34% | 28% | 50% |
| bound2_windowed_zncc | 20-40 | 68 | 46% | 24% | 18% |
| bound2_windowed_zncc | >=40 | 460 | 56% | 15% | 4% |
| bound4_windowed_zncc | <5 | 6 | 67% | 33% | 33% |
| bound4_windowed_zncc | 5-10 | 24 | 29% | 12% | 25% |
| bound4_windowed_zncc | 10-20 | 33 | 39% | 27% | 52% |
| bound4_windowed_zncc | 20-40 | 75 | 52% | 25% | 13% |
| bound4_windowed_zncc | >=40 | 512 | 52% | 13% | 4% |
| bound6_windowed_zncc | <5 | 6 | 67% | 33% | 33% |
| bound6_windowed_zncc | 5-10 | 24 | 29% | 12% | 25% |
| bound6_windowed_zncc | 10-20 | 35 | 40% | 26% | 49% |
| bound6_windowed_zncc | 20-40 | 80 | 52% | 25% | 12% |
| bound6_windowed_zncc | >=40 | 530 | 52% | 13% | 4% |

**`dino_dog_toy`: the first plane's error against its predicted precision.**

| variant | precision | points | first med | first p75 | first <10 | first <20 | start med |
|---|---|---|---|---|---|---|---|
| bound2 | <2 | 209 | 2.9 | 5.9 | 85% | 91% | 18.3 |
| bound2 | 2-5 | 281 | 4.9 | 9.3 | 77% | 93% | 28.4 |
| bound2 | 5-10 | 137 | 9.8 | 15.6 | 52% | 82% | 31.7 |
| bound2 | 10-20 | 71 | 14.8 | 24.4 | 31% | 68% | 38.6 |
| bound2 | 20-45 | 42 | 9.0 | 24.4 | 55% | 71% | 37.9 |
| bound2 | >=45 | 19 | 20.3 | 37.7 | 37% | 47% | 36.5 |
| bound4 | <2 | 304 | 2.6 | 4.4 | 92% | 95% | 21.2 |
| bound4 | 2-5 | 303 | 4.2 | 7.8 | 82% | 94% | 31.4 |
| bound4 | 5-10 | 101 | 7.6 | 14.2 | 63% | 87% | 36.0 |
| bound4 | 10-20 | 62 | 12.2 | 25.0 | 40% | 68% | 36.0 |
| bound4 | 20-45 | 30 | 9.0 | 24.9 | 57% | 70% | 38.1 |
| bound4 | >=45 | 13 | 8.6 | 35.0 | 62% | 69% | 41.4 |
| bound6 | <2 | 328 | 2.6 | 4.4 | 91% | 95% | 22.6 |
| bound6 | 2-5 | 304 | 4.3 | 7.5 | 82% | 94% | 31.2 |
| bound6 | 5-10 | 96 | 6.5 | 15.2 | 61% | 84% | 36.9 |
| bound6 | 10-20 | 55 | 13.2 | 24.6 | 38% | 69% | 33.5 |
| bound6 | 20-45 | 28 | 9.0 | 27.1 | 57% | 71% | 38.0 |
| bound6 | >=45 | 13 | 8.6 | 35.0 | 62% | 69% | 41.4 |

**`dino_dog_toy`: gates on the first plane** (every gate over all points, and the precision bar of 5° by baseline), and on the step as built:

| variant | gate | baseline | points | kept | first med | first <10 | first <20 | start med | 2x2 med | 3x3 med |
|---|---|---|---|---|---|---|---|---|---|---|
| bound2 | precision <= 5 | <5 | 9 | 0 | - | - | - | - | - | - |
| bound2 | precision <= 5 | 5-10 | 25 | 1 | 70.9 | 0% | 0% | 7.9 | 58.2 | 28.9 |
| bound2 | precision <= 5 | 10-20 | 45 | 10 | 23.7 | 20% | 40% | 45.5 | 14.7 | 22.5 |
| bound2 | precision <= 5 | 20-40 | 179 | 32 | 5.1 | 69% | 81% | 29.9 | 6.5 | 16.4 |
| bound2 | precision <= 5 | >=40 | 703 | 447 | 3.9 | 83% | 95% | 24.0 | 13.8 | 19.9 |
| bound2 | precision <= 5 | all | 961 | 490 | 4.0 | 80% | 92% | 24.1 | 13.7 | 19.9 |
| bound2 | precision <= 10 | all | 961 | 627 | 4.8 | 74% | 90% | 25.9 | 16.0 | 22.0 |
| bound2 | precision <= 15 | all | 961 | 672 | 5.1 | 71% | 89% | 26.4 | 16.6 | 22.5 |
| bound2 | precision <= 20 | all | 961 | 698 | 5.3 | 70% | 88% | 26.7 | 16.7 | 22.7 |
| bound2 | baseline >= 5 | all | 961 | 753 | 5.7 | 68% | 86% | 27.7 | 17.7 | 23.2 |
| bound2 | baseline >= 10 | all | 961 | 729 | 5.5 | 70% | 87% | 28.0 | 18.2 | 23.8 |
| bound2 | baseline >= 20 | all | 961 | 694 | 5.1 | 72% | 89% | 28.1 | 18.5 | 24.2 |
| bound4 | precision <= 5 | <5 | 9 | 0 | - | - | - | - | - | - |
| bound4 | precision <= 5 | 5-10 | 25 | 3 | 36.3 | 0% | 0% | 7.9 | 35.0 | 28.9 |
| bound4 | precision <= 5 | 10-20 | 45 | 14 | 18.8 | 43% | 50% | 64.6 | 30.9 | 36.9 |
| bound4 | precision <= 5 | 20-40 | 179 | 49 | 5.6 | 67% | 84% | 39.6 | 16.3 | 27.6 |
| bound4 | precision <= 5 | >=40 | 703 | 541 | 3.1 | 91% | 97% | 25.7 | 16.1 | 22.0 |
| bound4 | precision <= 5 | all | 961 | 607 | 3.2 | 87% | 95% | 26.3 | 16.3 | 22.5 |
| bound4 | precision <= 10 | all | 961 | 708 | 3.5 | 84% | 94% | 27.4 | 17.7 | 23.3 |
| bound4 | precision <= 15 | all | 961 | 756 | 3.6 | 81% | 92% | 28.0 | 18.2 | 23.7 |
| bound4 | precision <= 20 | all | 961 | 770 | 3.7 | 80% | 91% | 28.0 | 18.1 | 23.7 |
| bound4 | baseline >= 5 | all | 961 | 807 | 4.0 | 79% | 90% | 28.2 | 18.5 | 24.4 |
| bound4 | baseline >= 10 | all | 961 | 782 | 3.8 | 81% | 91% | 28.4 | 19.0 | 24.8 |
| bound4 | baseline >= 20 | all | 961 | 746 | 3.6 | 83% | 94% | 28.5 | 19.4 | 25.1 |
| bound6 | precision <= 5 | <5 | 9 | 0 | - | - | - | - | - | - |
| bound6 | precision <= 5 | 5-10 | 25 | 3 | 36.3 | 0% | 0% | 7.9 | 35.0 | 28.9 |
| bound6 | precision <= 5 | 10-20 | 45 | 16 | 12.6 | 44% | 56% | 65.7 | 28.5 | 36.9 |
| bound6 | precision <= 5 | 20-40 | 179 | 50 | 5.6 | 66% | 82% | 41.0 | 16.6 | 28.7 |
| bound6 | precision <= 5 | >=40 | 703 | 563 | 3.0 | 90% | 97% | 26.2 | 16.7 | 22.4 |
| bound6 | precision <= 5 | all | 961 | 632 | 3.1 | 87% | 94% | 26.8 | 16.9 | 22.9 |
| bound6 | precision <= 10 | all | 961 | 728 | 3.5 | 83% | 93% | 27.7 | 18.2 | 23.8 |
| bound6 | precision <= 15 | all | 961 | 766 | 3.6 | 82% | 92% | 28.0 | 18.3 | 24.0 |
| bound6 | precision <= 20 | all | 961 | 783 | 3.7 | 80% | 91% | 28.0 | 18.3 | 24.0 |
| bound6 | baseline >= 5 | all | 961 | 818 | 3.9 | 79% | 91% | 28.2 | 18.8 | 24.6 |
| bound6 | baseline >= 10 | all | 961 | 793 | 3.7 | 81% | 92% | 28.4 | 19.1 | 25.1 |
| bound6 | baseline >= 20 | all | 961 | 757 | 3.6 | 83% | 94% | 28.9 | 19.6 | 25.3 |

| variant | gate | turned | kept | new med | new <10 | new <20 | start med | 2x2 med | 3x3 med |
|---|---|---|---|---|---|---|---|---|---|
| bound2_none | precision <= 5 | 304 | 287 | 1.4 | 90% | 93% | 25.0 | 13.7 | 19.9 |
| bound2_none | precision <= 10 | 304 | 295 | 1.4 | 89% | 92% | 25.0 | 13.7 | 19.9 |
| bound2_windowed_zncc | precision <= 5 | 105 | 101 | 1.3 | 94% | 100% | 21.8 | 10.6 | 16.6 |
| bound2_windowed_zncc | precision <= 10 | 105 | 105 | 1.4 | 91% | 98% | 22.6 | 10.7 | 17.1 |
| bound4_none | precision <= 5 | 330 | 316 | 1.2 | 95% | 97% | 26.7 | 15.0 | 21.7 |
| bound4_none | precision <= 10 | 330 | 323 | 1.2 | 94% | 97% | 26.2 | 14.7 | 21.5 |
| bound4_windowed_zncc | precision <= 5 | 112 | 111 | 1.0 | 98% | 98% | 24.4 | 11.5 | 18.1 |
| bound4_windowed_zncc | precision <= 10 | 112 | 112 | 1.0 | 98% | 98% | 23.8 | 11.4 | 17.9 |
| bound6_none | precision <= 5 | 350 | 333 | 1.3 | 93% | 96% | 27.7 | 14.9 | 22.7 |
| bound6_none | precision <= 10 | 350 | 345 | 1.3 | 92% | 95% | 27.7 | 14.9 | 22.7 |
| bound6_windowed_zncc | precision <= 5 | 118 | 117 | 1.2 | 96% | 97% | 24.5 | 12.0 | 18.8 |
| bound6_windowed_zncc | precision <= 10 | 118 | 118 | 1.2 | 96% | 97% | 24.4 | 12.0 | 18.5 |

**What the numbers show.**

- **One 3×3 press from the mean-view start does not reproduce the stored normals.** On all three captures today's step at 3×3 ends further from the ground truth than at 2×2 (median 37.5° against 31.5°, 31.3° against 29.3°, 26.8° against 21.2°), and closer than 2×2 on only 17% to 37% of points; the two land a median 5.7° to 12.6° apart. That most stored normals were set with the 3×3 step does not make one press from this start agree with them. They were presumably set from other starts or positions, or over several presses; this is inferred, not checked.
- **On `dino_dog_toy` the new step agrees with the candidate's normals far more closely than on the other two captures.** Its turned points at 40° and over end at a median 1.0° to 1.3° in every variant (90% to 98% under 10°), against 11.4° to 15.4° for today's 2×2 and 17.5° to 22.4° for 3×3 on the same points; on `kerry_park` the same figure was 3.2° to 5.3°. The first plane at bound 4 lands at a median 3.4° at 40° and over and 7.8° at 20–40°, against a start of 27.7° and 40.9°, and is closer than the start on 98% and 91% of those points.
- **The step turns more points here.** `windowed_zncc` turns 105 to 118 points (11% to 12%) and `none` 304 to 350 (32% to 36%), against 3% to 8% and 7% to 29% on the other two.
- **Narrow baselines go the other way from the new step.** Under 20° on `dino_dog_toy` today's 2×2 is close to the candidate's normals (median 3.0° at 5–10°, 7.5° at 10–20°) while the first plane is not (17.6° and 31.7° at bound 4, against a start of 16.9° and 22.8°).
- **Later renders help on `dino_dog_toy`.** The median change of a point's proposal from its own first plane is −0.9° to −4.0° over renders 2 to 6; the first plane is the best of a point's proposals on 24% to 25% of points with two or more, the last is better than the first on 67% to 70%, and 5° or more worse on 6%. The cell movements settle as on the other two: a median 8 to 9.5 px at the first render, about 1 px at the second, 0.3 px at the third and 0.1 to 0.25 px after, with at most 28% at or under 0.05 px at any render.
- **The ZNCC rule rejects useful turns here.** At 40° and over 52% to 56% of the turns it rejected would have lowered the error and 4% would have raised it by 5° or more; at 10–20° about half would have raised it by 5° or more.
- **The predicted precision ranks the first planes cleanly on this capture.** At bound 4 the median error is 2.6° under 2° predicted, 4.2° at 2–5°, 7.6° at 5–10° and 12.2° at 10–20°. A precision bar of 5° keeps 607 of 961 first planes at a median 3.2° (87% under 10°); a baseline bar of 20° keeps 746 at 3.6° (83%), and 10° keeps 782 at 3.8°.

**What the numbers cannot show.**

- **How the candidate's normals were set.** It is a candidate ground truth, not approved, and its provenance is not recorded here. Agreement within about a degree with the new step's turned points, and within a few degrees with today's 2×2 at narrow baselines, may mean that parts of it were set by those methods or by a related one; if so, those parts measure agreement with a method, not with the surface.
- The other caveats of the sweep above hold: the ground-truth normals are estimates, most of the two approved ones were set with today's 3×3 step per the maintainer, and the gates and the first plane are read from the traces after the fact.

## The cell ZNCC bar (2026-10-15)

**Question.** In the walkthrough of the sweep's runs, many cells refused for a ZNCC under `0.8` looked like usable data. How do the first plane and the step change when the bar is lowered?

**Setup.** The `bound4_none` variant of the sweep above (`cell_shift_bound_px` 4, acceptance `none`, `max_iterations` 8) was run again at `min_cell_zncc` 0.6, 0.4 and 0.0 and compared with its 0.8 run, on every point of `seoul_bull` (266), `kerry_park` (382) and the `dino_dog_toy` candidate ground truth v006 (961). Starts, estimators and errors are as in the sweep above.

**Views' baseline 20° and over, 0.8 against 0.6** (errors in degrees, medians):

| dataset | points | cells refused for ZNCC | points with a first plane | first plane error | turned | turned error |
|---|---|---|---|---|---|---|
| `seoul_bull` | 127 | 19% → 5% | 92 → 125 | 13.5 → 13.0 | 19 → 39 | 14.7 → 8.1 |
| `kerry_park` | 301 | 26% → 10% | 211 → 279 | 13.1 → 10.1 | 67 → 106 | 5.3 → 3.3 |
| `dino_dog_toy` | 882 | 27% → 6% | 746 → 833 | 3.6 → 3.5 | 317 → 391 | 1.2 → 1.1 |

**What the numbers show.**

- At narrow baselines on `dino_dog_toy` the first plane's median error falls from 16.8° to 8.3° under 10° and from 31.7° to 6.8° at 10–20°. `seoul_bull` stays poor at narrow baselines.
- Below 0.6 the first plane gets worse at baselines of 20° and over: median 10.1°, 14.4° and 18.3° at 0.6, 0.4 and 0.0 on `kerry_park`, and 3.5°, 4.6° and 7.3° on `dino_dog_toy`.
- Even at 0.6 the step turns only 31% to 44% of the points at 20° and over. The stopping rules lose more points than the bar does.

**What the numbers cannot show.** The errors are against the stored normals, which are estimates, mostly set with today's 3×3 Grid Plane Normal; the `dino_dog_toy` file is a candidate ground truth, not an approved one.

**Decision.** The maintainer chose 0.6 as the step's default, `BENCH_MIN_CELL_ZNCC`, after reviewing the walkthrough. The piecewise refinement of cluster patches keeps its own default of 0.8.

## Open questions

- **A small oscillation.** When the loop oscillates within a cycle of normals a few times `update_tolerance_px` across, the cycle's mean could be taken as the estimate and confirmed by one more render, instead of leaving the normal unchanged as unstable. The maintainer's decision: wait for real examples from the evaluation against today's Grid Plane Normal on real captures, and keep the step simple until then.
- **A score over the kept cells.** Whether the acceptance score should read only the cells the gates kept, so that refused cells, such as the far plane of a depth step, cannot reject a correct turn. The same decision applies: it waits for real examples from that evaluation.
- **The acceptance score.** The windowed ZNCC of each view's whole tile against the reference render at the zero shift, read from the renders the loop already made, is cheap and is the cluster loop's choice in kind. The bench's evaluation, the blur-matched score the bars judge, is what a person reads, but costs another pass over the photographs per iteration. Start with the reference ZNCC, and compare the turns each accepts. The built step starts with the windowed reference ZNCC, provisionally, and has already shown it refusing a correct turn when most of the tile is refused cells (see [What the built step chose](#what-the-built-step-chose)); a score over the kept cells is the next candidate.
- **Moving the centre.** The fitted plane may lie off the patch's centre. Keeping the centre, as `tilt_patch` does, leaves depth to the following Fit; moving it to where the reference ray meets the plane would let the loop correct depth and normal together, and the displacements would then tend to zero. The built step keeps the centre and reports the offset.
- **A fallback to a full piece fit.** A cell registration cannot settle, for example one at the search bound when the normal is far off, could be given today's full piece fit as a fallback. A second loop iteration, after the first turn has brought the cell inside the bound, may make it unnecessary. Decide by counting, on the ground truths, the cells that end the loop refused for the bound. For the wide views whose cells the starting plane puts past any bound, [core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md) specifies a separate step that fits one shared plane directly to the photographs, growing each view's window from the centre outward and taking the views in order of their angle from the reference.
- **Reading the localizer's tiles.** The first iteration of the loop could read the context tiles a fit's localizer renders, if the localizer kept them and the step ran inside the fit, at the frame the fit started from. Every later iteration renders again anyway. Whether saving one render per view is worth the coupling is to be measured.
- **Fewer than three live cells.** The step leaves the patch unturned. Where the reconstruction is dense, the [adjacency surfel](../core/analysis/adjacency-surfel-normals.md) estimate from the point's neighbours could stand in; on the bench, leaving the patch is the only answer.
