# Grown Plane Normal

**Status:** Draft. Proposed: a fourth bench normal step, `grown_plane_normal`, beside `fit_normal`, `finite_difference_normal` and `grid_plane_normal`, that fits one plane shared by every view directly to the photographs. It is a separate step and not an option of `grid_plane_normal`, because it measures nothing that step measures: it has no cells, no shift search on a fixed render and no triangulation.

Decided: the step's existence as a separate step, and that it writes each joined view's keypoint back from the shift it solved for (§ [The keypoints the step writes](#the-keypoints-the-step-writes)). Not decided: the optimiser, the window schedule and the spot rule, the order views are taken in, the convergence bars, the weighting of the shared refinement, what the step reports, and whether it also moves the patch's centre. See [Open questions](#open-questions).

Built: nothing. A Python prototype, not in the repository, was run on one point of the `dino_dog_toy` candidate ground truth; its results are under [Prototype on one point](#prototype-on-one-point-2026-10-15).

Amends: [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal", which would describe a fourth step.

Related drafts: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) designs `grid_plane_normal` and records the measurements this step is compared against. [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) defines the predicted precision of a fitted normal, which bears on the proposed ordering rule. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) decides how a view is scored against the reference at matched sharpness, which bears on blur matching here.

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in the world, and its normal says which way the square faces. Each photograph that sees the patch is resampled onto the square through that plane, so a wrong normal stretches and shears every view's picture of the patch against the others, blurs the patch's stored bitmap and misleads every reading taken from it afterwards.

This draft proposes a way to estimate the normal by fitting the plane to the photographs themselves. One view is the reference. For every other view, the plane predicts where each part of the reference's picture of the patch lands in that view's photograph. The step finds the plane under which the views' pictures agree with the reference's, by ZNCC. It reaches a plane far from the starting one in two ways. Within a view it registers a small window near the patch centre first and grows the window outward, because a small window tolerates a large error in the plane. Across views it takes the views closest in angle to the reference first and the wide views last, each starting from the plane the earlier views agreed on, because a wide view is where a wrong plane distorts the picture most. The step turns the patch to the plane it found, and writes each joined view's keypoint at the pixel where that plane and the view's own shift put the patch.

### Why a new step

`grid_plane_normal` ([piece-gated-grid-normal.md](piece-gated-grid-normal.md)) measures the normal by parallax: it renders each view once through the current plane, registers each of nine cells by a shift search within a small bound, gates each cell in each view by a robust affine fit, triangulates the cells and fits a plane through them, then turns the patch and repeats. Each part of that chain needs the starting plane to be near enough that every cell of a view lands within the shift bound and that each cell's picture in the view differs from the reference's by little more than a shift, since a cell is registered by shift alone. In a wide view a wrong plane breaks both: the cells far from the centre are displaced past the bound, and the cells whose picture is stretched or turned against the reference's fall under the cell ZNCC bar.

This step replaces the nine-cell shift search, the per-view affine gate and the cell triangulation with a direct fit of the plane to the photographs. It uses the plane's own prediction to reach the views the shift search cannot: a view taken late in the sequence is rendered through a plane fitted on the views before it, so its picture already arrives close to the reference's.

### Why the name

The glossary's **fit normal** / **finite-difference normal** / **grid-plane normal** entry names each bench normal step for the evidence it reads. *Grown plane normal* says the estimate is a plane and that its support grows: the window over which a view is registered grows from a few samples to the whole grid, and the set of views the plane is fitted to grows one view at a time. It pairs with *Grid Plane Normal* in Track View's list. `homography_normal` names the warp but not the schedule that is the method's point, and *homography* is not bench vocabulary. `incremental_plane_normal` says the views are added one at a time but not that each view's window grows. *Register* is not used in the name, because the glossary gives it the meaning of giving an image a pose. When the name is settled, a glossary entry is added with the step.

## Rust API

Proposed: the step lives beside `grid_plane_normal` under [bench/normal/](../../crates/sfmtool-core/src/bench/normal/) and has the signature every bench normal step has ([bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs)), so the Python bindings in `sfmtool.bench` can call it the way they call the other three, and Track View and the wire's `fit_bench_track_normal` the way they call `fit_normal` and `finite_difference_normal` (neither calls `grid_plane_normal` yet).

```rust
/// How [`grown_plane_normal`] grows its windows and when a view joins.
pub struct GrownPlaneNormalOptions {
    /// Side of the working grid the reference tile is sampled on. 48.
    pub working_resolution: u32,
    /// Side of the first window, grid px, placed at each candidate spot. 7.
    pub spot_window: u32,
    /// How many candidate spots are tried: the most contrasted
    /// non-overlapping first windows in the reference tile's middle cell. 5.
    pub spot_candidates: u32,
    /// Half-widths of the later windows about a spot, grid px, the last
    /// covering the whole grid. [6, 10, 16, 24].
    pub window_half_widths: Vec<u32>,
    /// The first window's search for the view's shift: bound and step, image px.
    /// 12.0 and 0.5.
    pub spot_search_px: f32,
    pub spot_search_step_px: f32,
    /// A view joins only when its whole-grid ZNCC is at least this. 0.7.
    pub min_grid_zncc: f32,
    /// A view joins only when no later window's ZNCC falls more than this below
    /// the best earlier window's. 0.1.
    pub max_window_zncc_drop: f32,
    /// A ZNCC is read only when at least this share of the window's samples
    /// lands on the photograph. 0.6.
    pub min_coverage: f32,
    /// The reading the step ends with.
    pub fit: FitOptions,
}

/// What happened to one observation, carried by `NormalEstimate::GrownPlane`.
pub enum GrownPlaneView {
    /// The reference, with the keypoint the step wrote back for it: the one
    /// it had before the step.
    Reference { keypoint: [f64; 2] },
    /// Joined the shared plane, from the shared plane or, on a retry, from the
    /// starting plane; with its shift, image px, and its whole-grid ZNCC.
    /// `was` is its keypoint before the step and `keypoint` the one the step
    /// wrote, both image px; `keypoint` is `None` where that pixel is off the
    /// photograph and nothing was written.
    Joined {
        from_start: bool,
        shift: [f64; 2],
        grid_zncc: f32,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    /// Left out after both starts and one retry at the back of the queue. The
    /// step writes no keypoint for it, nor for `NotAttempted`.
    LeftOut { reason: GrownPlaneRefusal, grid_zncc: f32 },
    NotAttempted,
}

pub fn grown_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GrownPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError>;
```

**Why this shape.** The signature is the other steps', and the step ends almost as they do: it turns the patch with `tilt_patch`, then writes the keypoints of the reference and of every joined view, then reads the track back, so a caller can swap one step for another and compare `NormalReport`s. The write comes after the turn because `tilt_patch` itself moves every keypoint, and would carry the written ones off the pixels the step found. The settings are `*Options` because a caller chooses them. The report is per observation because a wrong normal is diagnosed by asking which views joined, in what order and from which start, and why the others were left out, and because a person checking the write needs each view's keypoint before and after it. Its exact fields are an open question. The step changes the normal and writes keypoints only when at least one view joins, and leaves the track as given otherwise, as `grid_plane_normal` does on an ending that is not `Turned`. It refuses a track at infinity and one with fewer than two `in` observations through `normal_preconditions`, and it reports progress and is cancelled per view through `progress`, since it can run longer than a second.

**Example.** `grown_plane_normal(&track, &edited, &images, &GrownPlaneNormalOptions::default(), &Progress::none())` returns the turned track and a report saying, for example, "15 of 17 views joined, 2 left out; 15 keypoints written".

## Theory

### One plane shared by all views

The model is one plane, the same for every view. The warp from the reference view's working grid into any other view is the plane-induced map: each sample of the reference tile is followed along its reference ray to the plane, and that point is projected into the view's photograph through the camera's lens model, where the photograph is sampled. There is no free per-view affine map; the plane predicts each view's local affine map from its tilt alone.

The plane has two unknowns, a tilt `(a, b)` of the starting frame's normal `n` along the frame's own axes `u` and `v`:

```text
normal = unit(n + a·u + b·v)
```

through the point where the reference keypoint's ray meets the starting frame. Each view adds its own image shift, two unknowns in image px, applied to the projected positions. The shift absorbs error in the point's position: a wrong depth along the reference ray moves every sample of a wide view by about the same amount, up to about 8 image px on the example below (image 36's shift was (−6.75, −4.25) px), which the tilt cannot express and should not try to.

Every evaluation samples the photograph directly through the current plane, reading the photograph around the projected footprint, rather than resampling a tile rendered once at the starting plane. A tile rendered at a wrong plane has already lost detail along the axis that plane squeezed, and resampling it cannot recover that detail.

### Near to far within a view

Under a plane that is wrong, a sample at distance `r` from the window's centre is displaced in the view by about `‖A − I‖·r`, where `A` is the error in the view's local affine map. A small window therefore tolerates a large error in the map, and a map fitted over radius `r` is accurate enough to bring radius about `2r` within capture of the next fit.

Stage 1 places a window of `spot_window` grid px (7×7) at a spot in the middle cell of the reference tile. It finds the view's shift alone, at the current plane, by an exhaustive search over shifts of up to `spot_search_px` (12) image px along each axis, on a lattice of `spot_search_step_px` (0.5) image px. Each later stage widens the window about the same spot to the next of `window_half_widths` (6, 10, 16 and 24 grid px on the 48-sample working grid, the last the whole grid) and refines the tilt and the view's shift together for the best ZNCC over the window. The prototype used a pattern search; Gauss–Newton on the ZNCC is the obvious alternative, and the choice is open.

**Which spot.** The prototype's sequence used one spot, the most contrasted window of the middle cell, contrast being the sum over the window of the squared forward differences in x and y. The proposal is to take several candidate spots instead, the `spot_candidates` most contrasted non-overlapping windows of the middle cell, grow the view from each, and keep the result that converges with the best whole-grid ZNCC. A cheaper alternative is to grow only from the candidate whose stage-1 match has the best ZNCC; on the example that candidate converged in both views where it was tried. The several-spot rule was tested on single views only, not in the sequence (see [Prototype on one point](#prototype-on-one-point-2026-10-15)).

### Views in sequence, sharing the plane

The reference is the step's reference view, by the rule `grid_plane_normal` and the fit use: the reference the track holds where its row is `in` with a keypoint, else the reference-view rule's pick. The other `in` views are taken in order of their angle from the reference as seen from the point. Each view starts from the plane found so far. When a view converges it joins, and the shared tilt is refined over every joined view, for the best mean whole-grid ZNCC, each view's shift being fitted again after. A view that fails is tried again from the starting plane. A view that fails from both goes to the back of the queue once, to be tried again from the shared plane after the views behind it have joined, and is otherwise left out.

### The keypoints the step writes

Proposed. The step solves for each joined view's shift, and the shift says where that view's photograph places the patch, so the step writes it back to the track rather than discarding it. The keypoint is the track-stage measurement `TrackMeasurement::keypoint` ([bench/track.rs](../../crates/sfmtool-core/src/bench/track.rs)): the pixel a commit writes and the place every reading of the row is anchored at, which the fit sets from its kernels and `sight_observation` sets by hand.

- **A joined view.** Its keypoint is set to the pixel where the final shared plane and the view's final shift put the patch's anchor point. The anchor is the point where the reference keypoint's ray meets the final plane; every plane the step tries passes through the point where that ray meets the starting frame, so the anchor is that same point. It is projected into the view's photograph through the camera's lens model, as every sample of the view's window is, and the view's shift is added. That is the pixel the view's whole-grid ZNCC was read around. A pixel off the photograph is not written, and the report says so.
- **The reference.** It keeps the keypoint it had before the step. The anchor lies on the reference keypoint's ray, so it projects back to that keypoint, and the reference's shift is zero by construction: the reference tile is the template every other view is compared against, sampled at the plane's own positions, so it has no shift to fit. The step writes that keypoint back after the turn, because the turn moves it.
- **Every other row**: a view left out, one never attempted, and every row not `in`. The step writes nothing for them. They keep the keypoint the turn gives them, as under every other normal step: `tilt_patch` keeps each sighting's offset from the centre in the patch's plane, rebuilds that point on the turned plane and projects it (`place_keypoints` in [bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs)).

**Readings at the old keypoint.** Every other field of a row's `TrackMeasurement` was read at its keypoint and says nothing about a new one. The precedent is the two steps that write keypoints by geometry or by hand: `place_keypoints`, which `tilt_patch`, the translations and a resize from an edge run, and `sight_observation` each replace the row's measurement with one that carries only the new keypoint, and `sight_observation`'s doc gives the reason, that an evaluation recomputes everything else from the track as it stands. The step does the same for each row it writes, so `walked_px` and `walked_to`, which only a fit sets, go with the rest. The turn has already done this for every row and dropped the bitmap, and the readback that ends the step renders a new bitmap from the reference and reads every row again at the keypoint it now has, so no reading is left that a written keypoint makes stale. The write changes no verdict and pins nothing, as a fit pins nothing: a pixel the step found is not one a person ruled on.

**The centre does not move.** The step turns the patch about its centre, as the other normal steps do. A depth error along the reference ray shows up as the per-view shifts, up to about 8 image px on the example below, so the written keypoints can imply a different position than the patch's centre; the readback's `projection_offset_px` then shows, for each written row, how far its keypoint sits from where the centre projects. Whether the step should also re-triangulate the centre from the written keypoints, or leave that to the next fit, is an open question.

How far the keypoints move was not measured. The prototype's shifts are offsets from where the plane projects the anchor, not from the keypoints the track held, so the 8 px is not the size of the move.

### When a view converges

A view joins when all three hold:

- its whole-grid ZNCC is at least `min_grid_zncc` (0.7);
- as the window grows, the window ZNCC falls no more than `max_window_zncc_drop` (0.1) below the best of the earlier stages. A view that is not converging loses agreement as the window grows (image 65 on the example: 0.80, 0.65, 0.31 and 0.27 over stages 2 to 5), where a converging view keeps it (image 62: 0.85, 0.82, 0.86 and 0.86);
- stage 1's best shift is not on the edge of its search.

Every ZNCC is the mean over the colour channels, over the samples that land on the photograph, and is read only when at least `min_coverage` (60%) of the window's samples do.

### A proposed ordering rule

Not prototyped. The angle from the reference is a proxy for how far the current plane's prediction is from the truth in a view. The plane's uncertainty, carried into each remaining view, predicts that error directly: the map error over a cell, given the plane's covariance. The proposal is to take next the view the current plane is predicted to get closest, and to defer a view whose predicted squeeze is past about 1.7×. That bar sits below what was captured on the example: from a single view, image 62, whose start needed a stretch of about 2×, was captured, and images 65 and 66, whose starts were squeezed by about 3× and 4×, were not.

### Relation to `grid_plane_normal`

`grid_plane_normal` reads the plane from where nine cells register, by shift alone, on a render made at the current plane, and gates each cell in each view by its ZNCC, its curvature, the bound and the view's robust affine fit; it can only measure what lands inside the shift bound. This step instead treats the plane as the thing being fitted, and every view's picture as evidence for it. The two share the reference rule, the starting frame and the final turn and readback. Where `grid_plane_normal` ends `Turned`, the two should agree; the cases this step is for are the wide views `grid_plane_normal`'s gates refuse at the starting plane.

## Implementation notes

- **Cost is unknown.** The prototype is Python and its time does not transfer. What dominates is one direct render of the window per optimiser evaluation per view, and one render per joined view per evaluation of the shared refinement. Gauss–Newton would cut the evaluations against a pattern search at the cost of a Jacobian of the render with respect to the tilt and the shift.
- The render is the plane-induced map `WarpMap::from_patch` makes for every bench tile, applied over a window rather than the whole tile; the shift is applied in the photograph after projection, not by re-anchoring the patch on the view's keypoint. The search therefore reads no view's keypoint but the reference's.
- The write sits between the turn and the readback. The other steps end through `turn_and_read` ([bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs)), which runs `tilt_patch` and then `evaluate_rendering_bitmap` with nothing between them; this step needs its keypoint write between the two. The anchor and each joined view's final shift are kept from the search for it.
- A track at infinity is refused by `normal_preconditions`, as for the other steps: its patch faces along its own bearing and has no plane to fit.

## Determinism and precision

Photograph samples and ZNCC sums are `f32`, as for every bench tile; the tilt, the shifts and the plane are `f64`. The views are taken in a fixed order, ties in angle broken by observation index, with no randomness, so the output is the same for any thread count. No content hash depends on it.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `working_resolution` | `48` | Side of the working grid, samples; the prototype's value |
| `spot_window` | `7` | Side of the first window, grid px |
| `spot_candidates` | `5` | Candidate spots tried per view, the centre not among them; a proposal, not measured: the prototype's sequence used 1, and the single-view test tried 5 and the centre |
| `window_half_widths` | `[6, 10, 16, 24]` | Half-widths of the later windows, grid px |
| `spot_search_px` | `12.0` | Bound of the first window's shift search, image px |
| `spot_search_step_px` | `0.5` | Step of that search, image px |
| `min_grid_zncc` | `0.7` | Whole-grid ZNCC a view needs to join |
| `max_window_zncc_drop` | `0.1` | Largest fall of the window ZNCC below its best earlier stage |
| `min_coverage` | `0.6` | Share of a window's samples on the photograph for its ZNCC to be read |

Every default but `spot_candidates` is the prototype's, set on one point.

## Testing

Synthetic planes rendered through known poses, as `grid_plane_normal`'s tests do:

- A textured plane started 20° off its normal, seen from views out to 80°: every view joins and the normal is within 1° of truth.
- The same scene with the point's depth along the reference ray off by several image px in the wide views: the per-view shifts absorb it and the normal is unchanged within the same tolerance.
- A wide view whose start is squeezed past capture: it fails from the starting plane and joins from the shared plane after the nearer views.
- A flat patch: no view converges, and the normal and every keypoint are left as given.
- The written keypoints: in the depth-error scene, each joined view's keypoint is, to within a fraction of a px, the projection of the true surface point on the reference keypoint's ray, the reference's keypoint is the one it had before the step, a view left out and a row that is `out` carry the keypoint `tilt_patch` gives them, and every row's readings are those of the readback at its new keypoint.

Then the measurement on every point of `seoul_bull_sculpture`, `kerry_park` and the `dino_dog_toy` candidate, against `grid_plane_normal` and today's Grid Plane Normal at 2×2 and 3×3, from the same fitted mean-view start as [piece-gated-grid-normal.md](piece-gated-grid-normal.md)'s sweep, and a hand review in SfM Explorer.

## Prototype on one point (2026-10-15)

**Question.** Can a plane fitted directly to the photographs, near to far within each view and view by view across the track, reach the stored normal from a start that `grid_plane_normal` and the cluster kernel do not recover from, on a point with wide views?

**Setup.** One point, `pt3d_0fdc9e3b_810` of `dino_dog_toy_ground_truth_candidate_v006.sfmr`, with 17 `in` views beside the reference, image 23. The start is the mean viewing direction, 22.8° from the stored normal. The stored normal is itself an estimate: most stored normals were set with today's Grid Plane Normal at 3×3, and the file is a candidate ground truth, not an approved one. One point says what the method can do, not how often it does it.

**Today's steps from the same start.**

- `grid_plane_normal` (bar 0.6, bound 6, one render): the first plane was 7.0° from the stored normal, 6.5° at bound 4, and the step ended `Unstable`.
- The cluster kernel (the bench's `set_stage` to cluster, then `evaluate`, with image 30 picked as the reference) kept the 13 views out to 43°, whose affine maps were within 0.01 to 0.2 of the stored normal's map, and refused images 09, 62, 65 and 66 (ZNCC 0.833, 0.217, 0.198 and 0.458). Its refinement stays near its seed: image 62's seed was squeezed to 0.54 of the stored normal's map and ended at 0.52.

**The model.** With the stored normal and one free shift per view, every view matches the reference well: mean cell ZNCC 0.79 to 0.95, and image 62 0.841 against 0.111 through the starting plane. The plane-induced warp therefore accounts for the wide views' rotation and stretch; for image 62 that is a 2.07× stretch along an oblique axis and a 4.7° rotation.

**Near to far, one view at a time from the starting plane.** Image 62, about 80° (79.8°) from the reference, reached 0.34° from the stored normal at the second stage and 0.83° at the end, whole-grid ZNCC 0.86. Images 25 and 36 reached 1.1° and 0.9°. Images 65 and 66, whose starts were squeezed to 0.33 and 0.24 of the stored normal's map, failed, ending about 11° off.

**Blur matching.** Blurring the reference to the view's sampling did not change the single-view results. At the stored normal image 65 has 1.45 to 1.97 photograph px per grid px, finer than the reference, so the reference needs no blur; the softness seen under the starting plane is that plane's upsampling, 0.52 photograph px per grid px along one axis.

**In sequence.** The shared plane was 2.0° from the stored normal after the first view (image 24, 6.8° from the reference), 0.2° to 0.6° after the ninth, and 0.59° at the end. 15 of the 17 views joined, among them images 62, 65 and 66, each started from the shared plane and ending 0.79°, 0.57° and 0.42° from the stored normal. Two were left out: image 10, which ended 1.5° off with whole-grid ZNCC 0.806 while its window ZNCC fell from 0.92 to 0.81, and image 09, 0.9° off with whole-grid ZNCC 0.694, under the 0.7 bar. Both ended near the stored normal, so on this point the two convergence bars are close to their edge.

**The starting spot.** The maintainer suspected the single-view failures came from the spot the contrast rule picked. Single views were run again from six candidate spots: the five most contrasted non-overlapping 7×7 windows of the middle cell, ranked by contrast, among them grid (20, 21), (29, 19), (18, 31) and (27, 31), and the geometric centre (24, 24).

- Images 65 and 66 from the starting plane, 22.8° off: the most contrasted spot, (20, 21), the one the method picked, failed in both (65 ending 12.6° off, 66 11.6°), and so did the centre (10.3° and 10.5°). Other spots converged from the starting plane in a single view: image 65 at (29, 19) to 0.40°, though its stage-1 match was on the edge of the search, so the convergence test flagged it although it ended right, and at (27, 31) to 0.43°; image 66 at (29, 19) to 0.34°, at (18, 31) to 0.51° and at (27, 31) to 0.43°. For these two views the spot decided the outcome, and started at a good spot they need no plane from other views.
- The spot with the best stage-1 ZNCC converged in both: (27, 31) in each, at 0.825 for image 65 and 0.803 for image 66, where the picked spot scored 0.694 and 0.713.
- Images 09 and 10 from the stored normal. For image 09 the spot did not change the verdict: it ended about 1° off from four of five spots (from (18, 31) it wandered to 15°), with whole-grid ZNCC about 0.69, just under the 0.7 bar, from all of them. For image 10 the spot changed the verdict but not the result: every spot ended 1.4° to 1.6° from the stored normal with whole-grid ZNCC 0.805 to 0.807, and its window ZNCC fell from its best by 0.07 from (29, 19), which passed the bars (ending 1.41° off), and by 0.11 to 0.15 from the other four, which failed the window-drop bar (from (20, 21) it ran 0.922, 0.923, 0.883, 0.832 and 0.810 over stages 1 to 5). Both views land near the stored normal (image 09 from all but one spot), so it is the bars that left them out of the sequence, not a wrong spot.

**What the numbers cannot show.** How often the method converges, on which kinds of point it fails, and what it costs. The full sweep has not been run.

## Non-goals

**Moving the centre.** The step turns the patch about its centre, as the other normal steps do, and does not move it. It writes the keypoints of the views that joined (§ [The keypoints the step writes](#the-keypoints-the-step-writes)); whether it should also re-triangulate the centre from them is an open question.

**Per-cell one-dimensional constraints.** A cell over stripes pins its shift across the stripes only. Using such constraints was discussed but is not part of this method.

## Open questions

- **The optimiser.** Pattern search, as prototyped, or Gauss–Newton on the ZNCC.
- **The window schedule and the spot rule.** On the example the most contrasted window of the middle cell was the wrong first window for images 65 and 66, and other candidates converged; whether to grow every candidate and keep the best converged result, or only the candidate with the best stage-1 ZNCC, how many candidates to take, and what to do when the centre is blank. Neither rule has been run in the sequence.
- **The order of the views** beyond angle from the reference; the predicted-error rule above is the candidate.
- **The convergence bars.** Both views the bars left out on the example ended within about 1° to 1.5° of the stored normal, image 09 from four of five starting spots and image 10 from all five, though one spot let image 10 pass the bars. Judging a view by its agreement with the shared plane, rather than by its own ZNCC readings, is an alternative.
- **Weighting the shared refinement.** Whether views should be weighted, since a wide view carries more information about the tilt per degree than a narrow one.
- **Blur matching the other way.** A view finer than the reference aliases when sampled bilinearly from the full-resolution photograph; sampling it from a coarser pyramid level, or blurring it, may be needed.
- **What the step reports**: each view's outcome and start, which joined and which were left out, each written view's keypoint before and after the write, the shared plane, and the plane's predicted precision.
- **The centre after the write.** A depth error shows up as the per-view shifts, so the written keypoints can imply a different centre than the patch's. Whether the step should re-triangulate the centre from the written keypoints before the readback, or leave it to the next fit, is not decided. The step's plane passes through the anchor and the turned patch's through its centre, so the two planes also differ in depth wherever the anchor is not the centre.
- **A bound on the write.** A fit keeps a sighting at its seed when its kernels would move it further than `max_shift_px` on the patch's plane, and records the refusal in `walked_px` and `walked_to`. Whether this step should bound its write the same way, and what it measures the move from, is open; the size of the moves on real tracks has not been measured.
- **The full sweep** over every point of `seoul_bull_sculpture`, `kerry_park` and `dino_dog_toy` is not done.
