# Grown Plane Normal

**Status:** Draft. Proposed: a fourth bench normal step, `grown_plane_normal`, beside `fit_normal`, `finite_difference_normal` and `grid_plane_normal`, that fits one plane shared by every view directly to the photographs. It is a separate step and not an option of `grid_plane_normal`, because it measures nothing that step measures: it has no cells, no shift search on a fixed render and no triangulation.

Decided: the step's existence as a separate step; that it writes each joined view's keypoint back from the shift it solved for (§ [The keypoints the step writes](#the-keypoints-the-step-writes)); that a view joins by its agreement with the shared plane, scaled by the plane's precision, rather than by convergence bars on its own ZNCC readings (§ [When a view agrees with the shared plane](#when-a-view-agrees-with-the-shared-plane)); that the step turns the patch to the shared plane whenever at least one view joined and reports the plane's precision as its confidence, rather than rejecting a plane it found (§ [Turn and report](#turn-and-report)); and that stage 1 starts each view from the shift its own keypoint implies (§ [Near to far within a view](#near-to-far-within-a-view)). Not decided: the optimiser, the window schedule and the spot rule, the order views are taken in, the ZNCC tolerance and the other numbers of the agreement rule, the weighting of the shared refinement, the report's exact fields, whether it also moves the patch's centre, and every rule of the optional wide-pair stage (§ [An optional wide-pair stage](#an-optional-wide-pair-stage)). See [Open questions](#open-questions).

Built: nothing. A Python prototype, not in the repository, was run on one point of the `dino_dog_toy` candidate ground truth ([Prototype on one point](#prototype-on-one-point-2026-10-15)), and a second round of it, with the four decided changes above, on four points ([Prototype on four points](#prototype-on-four-points-2026-10-16)). The multi-spot rule and the optional wide-pair stage were not part of either.

Amends: [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal", which would describe a fourth step.

Related drafts: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) designs `grid_plane_normal` and records the measurements this step is compared against. [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) defines the predicted precision of a fitted normal, which bears on the proposed ordering rule. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) decides how a view is scored against the reference at matched sharpness, which bears on blur matching here.

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in the world, and its normal says which way the square faces. Each photograph that sees the patch is resampled onto the square through that plane, so a wrong normal stretches and shears every view's picture of the patch against the others, blurs the patch's stored bitmap and misleads every reading taken from it afterwards.

This draft proposes a way to estimate the normal by fitting the plane to the photographs themselves. One view is the reference. For every other view, the plane predicts where each part of the reference's picture of the patch lands in that view's photograph. The step finds the plane under which the views' pictures agree with the reference's, by ZNCC. It reaches a plane far from the starting one in two ways. Within a view it registers a small window near the patch centre first and grows the window outward, because a small window tolerates a large error in the plane. Across views it takes the views closest in angle to the reference first and the wide views last, each starting from the plane the earlier views agreed on, because a wide view is where a wrong plane distorts the picture most. An optional last stage, off by default, compares two joined views on opposite sides of the reference with each other, since they are further apart than either is from the reference. The step turns the patch to the plane it found whenever at least one view joined, reports how precisely the views determine that plane, and writes each joined view's keypoint at the pixel where that plane and the view's own shift put the patch.

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
    /// How many contrasted candidate spots are tried: the most contrasted
    /// non-overlapping first windows in the reference tile's middle cell.
    /// The geometric centre is tried as well. 4.
    pub spot_candidates: u32,
    /// Half-widths of the later windows about a spot, as fractions of the
    /// working grid's side, each rounded to grid px; one no wider than the
    /// first window's half-width is dropped, and the last covers the whole
    /// grid. [1/8, 1/5, 1/3, 1/2], which is 6, 10, 16, 24 on a 48 grid.
    pub window_fractions: Vec<f32>,
    /// The first window's search for the view's shift about the shift the
    /// view's keypoint implies: bound, image px. 3.0.
    pub keypoint_search_px: f32,
    /// The search widens to `spot_search_px` when no shift within
    /// `keypoint_search_px` reaches this ZNCC, or the best is on its edge. 0.7.
    pub min_keypoint_zncc: f32,
    /// The widened search: bound about the keypoint-implied shift, and the
    /// step of both searches, image px. 12.0 and 0.5.
    pub spot_search_px: f32,
    pub spot_search_step_px: f32,
    /// The ZNCC difference noise can produce; turns a view's information
    /// about the tilt into a covariance. 0.01.
    pub zncc_tolerance: f32,
    /// Finite-difference step of the information estimate, tilt units. 0.03.
    pub tilt_step: f64,
    /// A view agrees when the Mahalanobis distance between its own plane and
    /// the shared plane, the two covariances added, is at most this. 3.0.
    pub max_agreement_distance: f64,
    /// A view agrees only when, held to the shared plane with its shift
    /// fitted again, its whole-grid ZNCC is at least this; the first view
    /// joins on this alone. 0.5.
    pub min_held_zncc: f32,
    /// σ above this, degrees, is reported as low confidence. 2.0.
    pub low_confidence_sigma_deg: f64,
    /// A ZNCC is read only when at least this share of the window's samples
    /// lands on the photograph. 0.6.
    pub min_coverage: f32,
    /// Run the wide-pair stage after the last view has joined or been left
    /// out. Off by default; not prototyped.
    pub wide_pair_stage: bool,
    /// The reading the step ends with.
    pub fit: FitOptions,
}

/// What the wide-pair stage did, carried by `NormalEstimate::GrownPlane`;
/// `None` there when the stage is off.
pub enum GrownPlaneWidePair {
    /// The stage was on but did not run: fewer than two views joined, or no
    /// joined pair is wider than the widest joined view's angle from the
    /// reference.
    Skipped,
    /// The stage ran on these pairs of observation indices, each with its
    /// angle at the point, degrees.
    Ran {
        pairs: Vec<(usize, usize, f64)>,
        /// The shared plane's normal before the stage and the one the stage
        /// found, world frame.
        before: [f64; 3],
        found: [f64; 3],
        /// Whether `found` was kept; otherwise the step goes on with `before`.
        accepted: bool,
    },
}

/// What happened to one observation, carried by `NormalEstimate::GrownPlane`.
pub enum GrownPlaneView {
    /// The reference, with the keypoint the step wrote back for it: the one
    /// it had before the step, or the centre's projection where it had none.
    Reference { keypoint: [f64; 2] },
    /// Joined the shared plane, from the shared plane or, on a retry, from the
    /// starting plane; with its shift, image px, and its whole-grid ZNCC.
    /// `widened` says stage 1 needed the full search. `distance` is the
    /// Mahalanobis distance of its own plane from the shared plane, `None`
    /// for the first view, and `held_zncc` its whole-grid ZNCC held to the
    /// shared plane. `was` is its keypoint before the step and `keypoint` the
    /// one the step wrote, both image px; `keypoint` is `None` where that
    /// pixel is off the photograph and nothing was written.
    Joined {
        from_start: bool,
        widened: bool,
        shift: [f64; 2],
        grid_zncc: f32,
        distance: Option<f64>,
        held_zncc: f32,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    /// Left out after both starts and one retry at the back of the queue, with
    /// the last try's distance and held ZNCC. The step writes no keypoint for
    /// it, nor for `NotAttempted`.
    LeftOut { reason: GrownPlaneRefusal, distance: Option<f64>, held_zncc: f32 },
    NotAttempted,
}

/// How precisely the joined views determine the shared plane, carried by
/// `NormalEstimate::GrownPlane`; `None` there when no view joined.
pub struct GrownPlanePrecision {
    /// σ along the plane's weakest and strongest directions, degrees.
    pub sigma_weak_deg: f64,
    pub sigma_strong_deg: f64,
    /// `sigma_weak_deg` is over `low_confidence_sigma_deg`. The patch is
    /// turned either way.
    pub low_confidence: bool,
}

pub fn grown_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GrownPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError>;
```

**Why this shape.** The signature is the other steps', and the step ends almost as they do: it turns the patch with `tilt_patch`, then writes the keypoints of the reference and of every joined view, then reads the track back, so a caller can swap one step for another and compare `NormalReport`s. The write comes after the turn because `tilt_patch` itself moves every keypoint, and would carry the written ones off the pixels the step found. The settings are `*Options` because a caller chooses them. The report is per observation because a wrong normal is diagnosed by asking which views joined, in what order and from which start, and why the others were left out, and because a person checking the write needs each view's keypoint before and after it. Its exact fields are an open question. The step changes the normal and writes keypoints whenever at least one view joins, however low the plane's precision, and reports that precision (`GrownPlanePrecision`); it leaves the track as given only when no view joins. This differs from `grid_plane_normal`, which leaves the track as given on every ending that is not `Turned` (§ [Turn and report](#turn-and-report)). It refuses a track at infinity and one with fewer than two `in` observations through `normal_preconditions`, and it reports progress and is cancelled per view through `progress`, since it can run longer than a second.

**Example.** `grown_plane_normal(&track, &edited, &images, &GrownPlaneNormalOptions::default(), &Progress::none())` returns the turned track and a report saying, for example, "16 of 17 views joined, 1 left out; σ 0.38°; 16 keypoints written".

## Theory

### One plane shared by all views

The model is one plane, the same for every view. The warp from the reference view's working grid into any other view is the plane-induced map: each sample of the reference tile is followed along its reference ray to the plane, and that point is projected into the view's photograph through the camera's lens model, where the photograph is sampled. There is no free per-view affine map; the plane predicts each view's local affine map from its tilt alone.

The plane has two unknowns, a tilt `(a, b)` of the starting frame's normal `n` along the frame's own axes `u` and `v`:

```text
normal = unit(n + a·u + b·v)
```

through the point where the reference keypoint's ray meets the starting frame. Each view adds its own image shift, two unknowns in image px, applied to the projected positions. The shift absorbs error in the point's position: a wrong depth along the reference ray moves every sample of a wide view by about the same amount, up to about 8 image px on the example below (image 36's shift was (−6.75, −4.25) px), which the tilt cannot express and should not try to.

**The keypoint-implied shift.** Each view's own keypoint already places the patch in that view, since the fit put it there. The shift it implies is the keypoint minus the anchor's projection into the view, the anchor being the point where the reference keypoint's ray meets the starting frame. Every plane the step tries passes through the anchor, so this shift is the same at every tilt.

Every evaluation samples the photograph directly through the current plane, reading the photograph around the projected footprint, rather than resampling a tile rendered once at the starting plane. A tile rendered at a wrong plane has already lost detail along the axis that plane squeezed, and resampling it cannot recover that detail.

### Near to far within a view

Under a plane that is wrong, a sample at distance `r` from the window's centre is displaced in the view by about `‖A − I‖·r`, where `A` is the error in the view's local affine map. A small window therefore tolerates a large error in the map, and a map fitted over radius `r` is accurate enough to bring radius about `2r` within capture of the next fit.

Stage 1 places a window of `spot_window` grid px (7×7, half-width 3) at a spot in the middle cell of the reference tile. It finds the view's shift alone, at the current plane, by an exhaustive search on a lattice of `spot_search_step_px` (0.5) image px about the view's keypoint-implied shift, up to `keypoint_search_px` (3) image px along each axis. It widens the search to `spot_search_px` (12) image px about the same shift only when no shift within the ±3 px range reaches `min_keypoint_zncc` (0.7) or the best one is on the edge of that range. The keypoint start exists because a 7×7 window on repeating texture matches equally well at more than one shift, and the spot rule, which prefers the window with the best stage-1 ZNCC, favours exactly such windows, since repeating texture scores high everywhere; a search over the full ±12 px then picks a wrong peak as readily as the right one (see [Prototype on four points](#prototype-on-four-points-2026-10-16)). Each later stage widens the window about the same spot to the next half-width of the schedule, `window_fractions` of the working grid's side W rounded to grid px, any no wider than 3 dropped: 3, then 6, 10, 16 and 24 on a 48-sample grid, and 3, 5, 8 and 12 on a 24-sample grid, the last the whole grid. It refines the tilt and the view's shift together for the best ZNCC over the window. The prototype used a pattern search; Gauss–Newton on the ZNCC is the obvious alternative, and the choice is open.

**Which spot.** The first round's sequence used one spot, the most contrasted window of the middle cell, contrast being the sum over the window of the squared forward differences in x and y. The proposal is to take several candidate spots instead, the `spot_candidates` (4) most contrasted non-overlapping windows of the middle cell and the geometric centre, grow the view from each, and keep the result that agrees with the shared plane with the best whole-grid ZNCC. The middle cell of a 48-sample grid is 16 samples wide and holds only four non-overlapping 7×7 windows, so the fifth candidate is the centre. The centre window is centred on the reference keypoint, to within half a sample, since the grid's side is even. A cheaper alternative is to grow only from the candidate whose stage-1 match has the best ZNCC; on the first example that candidate converged in both views where it was tried. The grow-every-candidate rule was tested on single views only, not in the sequence (see [Prototype on one point](#prototype-on-one-point-2026-10-15)); the best-stage-1 rule, over the four contrasted windows and the centre, is the one the second round ran in the sequence (see [Prototype on four points](#prototype-on-four-points-2026-10-16)).

### Views in sequence, sharing the plane

The reference is the step's reference view, by the rule `grid_plane_normal` and the fit use: the reference the track holds where its row is `in` with a keypoint, else the reference-view rule's pick. The other `in` views are taken in order of their angle from the reference as seen from the point. Each view starts from the plane found so far. When a view agrees with the shared plane (§ [When a view agrees with the shared plane](#when-a-view-agrees-with-the-shared-plane)) it joins, and the shared tilt is refined over every joined view, for the best mean whole-grid ZNCC, each view's shift being fitted again after. A view that fails is tried again from the starting plane. A view that fails from both goes to the back of the queue once, to be tried again from the shared plane after the views behind it have joined, and is otherwise left out.

### An optional wide-pair stage

Proposed, not prototyped, and off by default (`wide_pair_stage`). No number under [Prototype on one point](#prototype-on-one-point-2026-10-15) includes it.

**Why.** How much a view says about the tilt grows with its angle from the view it is compared against: the further apart two views are, the more a change of tilt changes one view's picture of the patch against the other's. Every comparison in the sequence above is against the reference. When the reference sits among the cameras rather than at one end, two joined views on opposite sides of it are further apart from each other than either is from the reference, so comparing them with each other is more sensitive to the plane than any comparison with the reference.

**Why at the end.** By the time every view has joined or been left out, the shared plane has converged from the reference comparisons, so both views' renders through it are already close to each other and the pair needs only a small correction to the tilt. The reason the sequence takes wide views last, that a wrong plane carries a wide view's picture out of capture, does not arise here.

**What it does.** The stage runs only when at least two views have joined and the widest joined pair, by the angle between the two views' rays at the point, is wider than the widest joined view's angle from the reference; otherwise the pair adds nothing the reference comparisons lack, and the stage is skipped. It takes the joined pair with the widest angle. Both views are rendered through the shared plane onto the reference's working grid: each sample of the grid is followed along its reference ray to the plane and projected into each of the two photographs, each with its own shift. The stage then refines the tilt and the two views' shifts for the best ZNCC between the two renders over the whole grid, with every joined view's whole-grid ZNCC against the reference kept in the objective, so the plane stays anchored to the reference and cannot drift to one that only the two views agree on.

The pair constrains only the difference between its two views' shifts: shifting both views by the same amount changes neither render against the other. The common part is set by each view's own reference comparison, which is why those comparisons stay in the objective. They hold each shift near the value the sequence found rather than fixing it, so the pair's term can still move a shift where it outweighs that view's reference comparison.

**Acceptance.** The stage keeps the plane and shifts it found only when the reference-based scores do not get worse: the proposal is that the mean whole-grid ZNCC over the joined views against the reference is not lower than before the stage, and that every joined view still has a whole-grid ZNCC of at least `min_held_zncc` at the stage's plane. Otherwise the step goes on with the plane and shifts from before the stage. The report records the pair, the plane before and after, and whether it was kept (`GrownPlaneWidePair`).

**Risks.** The two views of a wide pair differ more from each other than either does from the reference: in foreshortening, so the two renders sample the photographs at more different scales; in sampling, so one render can be blurred where the other aliases; and in lighting, since the two look at the surface from further apart. Near an occlusion they can also see slightly different surface. Each of these lowers the ZNCC between the pair's two renders for a reason other than the tilt, which is why the stage is optional and gated by the reference-based scores.

**Where the write happens.** The stage comes after the last view has joined or been left out and before the turn. The turn uses the final shared plane, and the keypoint write that follows it uses that plane and each joined view's final shift: the ones the stage found where it was kept, and those from before it otherwise.

### The keypoints the step writes

That the step writes the keypoints is decided; the rules below are proposed. The step solves for each joined view's shift, and the shift says where that view's photograph places the patch, so the step writes it back to the track rather than discarding it. The keypoint is the track-stage measurement `TrackMeasurement::keypoint` ([bench/track.rs](../../crates/sfmtool-core/src/bench/track.rs)): the pixel a commit writes and the place every reading of the row is anchored at, which the fit sets from its kernels and `sight_observation` sets by hand.

- **A joined view.** Its keypoint is set to the pixel where the final shared plane and the view's final shift put the patch's anchor point, both taken after the wide-pair stage where it ran. The anchor is the point where the reference keypoint's ray meets the final plane; every plane the step tries passes through the point where that ray meets the starting frame, so the anchor is that same point. It is projected into the view's photograph through the camera's lens model, as every sample of the view's window is, and the view's shift is added. That is the pixel the view's whole-grid ZNCC was read around. A pixel off the photograph is not written, and the report says so.
- **The reference.** It keeps the keypoint it had before the step. The anchor lies on the reference keypoint's ray, so it projects back to that keypoint, and the reference's shift is zero by construction: the reference tile is the template every other view is compared against, sampled at the plane's own positions, so it has no shift to fit. The step writes that keypoint back after the turn, because the turn moves it. The reference-view rule can pick a row with no keypoint, which it renders through the patch itself; its ray is then the one through the centre's projection, the anchor is the centre, and the step writes the centre's projection, which is also the keypoint the turn gives that row.
- **Every other row**: a view left out, one never attempted, and every row not `in`. The step writes nothing for them. They keep the keypoint the turn gives them, as under every other normal step: `tilt_patch` keeps each sighting's offset from the centre in the patch's plane, rebuilds that point on the turned plane and projects it (`place_keypoints` in [bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs)).

**Readings at the old keypoint.** Every other field of a row's `TrackMeasurement` was read at its keypoint and says nothing about a new one. The precedent is the two places that write keypoints by geometry or by hand: `place_keypoints`, which `tilt_patch`, the translations and a resize from an edge run, and `sight_observation` each replace the row's measurement with one that carries only the new keypoint, and `sight_observation`'s doc gives the reason, that an evaluation recomputes everything else from the track as it stands. The step does the same for each row it writes, so `walked_px` and `walked_to`, which only a fit sets, go with the rest. The turn has already done this for every row and dropped the bitmap, and the readback that ends the step renders a new bitmap from the reference and reads every row again at the keypoint it now has, so no reading is left that a written keypoint makes stale. The write changes no verdict and pins nothing, as a fit pins nothing: a pixel the step found is not one a person ruled on.

**The centre does not move.** The step turns the patch about its centre, as the other normal steps do. A depth error along the reference ray shows up as the per-view shifts, up to about 8 image px on the example below, so the written keypoints can imply a different position than the patch's centre; the readback's `projection_offset_px` then shows, for each written row, how far its keypoint sits from where the centre projects. Whether the step should also re-triangulate the centre from the written keypoints, or leave that to the next fit, is an open question.

How far the keypoints move was not measured. The prototype's shifts are offsets from where the plane projects the anchor, not from the keypoints the track held, so the 8 px is not the size of the move.

### The plane's precision

A view's information about the tilt is minus the Hessian of its whole-grid ZNCC in the tilt `(a, b)`, at the plane the view reached, with the view's shift fitted again at every sample, so that a shift cannot stand in for a tilt. The Hessian is estimated by finite differences: the prototype used a step of `tilt_step` (0.03) tilt units, a centred second difference along each axis and one mixed sample, and clipped negative eigenvalues to 0. The shared plane's information `I` is the sum over the joined views.

The ZNCC tolerance τ (`zncc_tolerance`) is the ZNCC difference noise can produce; the prototype's 0.01 is a placeholder. A tilt change whose ZNCC cost, `½·δᵀIδ`, is under τ cannot be told from noise, which makes the covariance of the tilt `2τ·I⁻¹`. It is regularised so that a direction with no information gets a large variance rather than none. σ is the plane's uncertainty along its weakest direction, `√(2τ/λ_min)` with `λ_min` the smaller eigenvalue of `I`, in degrees, since a small tilt `(a, b)` is that many radians. The strongest direction's σ comes from the larger eigenvalue in the same way. Since σ grows as `√τ`, the choice of τ scales every σ and every agreement distance by the same factor.

### When a view agrees with the shared plane

A view joins when both hold:

- its own plane, the one it reached growing from its start, lies within `max_agreement_distance` (3) of the shared plane by the Mahalanobis distance, with the view's covariance and the shared plane's added;
- held to the shared plane, with its shift fitted again, its whole-grid ZNCC is at least `min_held_zncc` (0.5).

The first view has no shared plane to agree with and joins on the 0.5 floor alone; its plane becomes the shared plane. The rule replaces convergence bars on the view's own readings (a whole-grid ZNCC of at least 0.7, a window ZNCC never more than 0.1 below its best earlier stage, a stage-1 match not on the edge of its search), which left out views that ended near the stored normal and, on one of the second round's points, let no view join at all. A fixed bar on how much ZNCC a view loses when held to the shared plane was tried in between, and let in a view whose own plane was 26° from the shared plane; scaling the test by precision is what refuses it (see [Prototype on four points](#prototype-on-four-points-2026-10-16)).

Every ZNCC is the mean over the colour channels, over the samples that land on the photograph, and is read only when at least `min_coverage` (60%) of the window's samples do.

### Turn and report

The step turns the patch to the shared plane whenever at least one view joined, and reports σ along the weakest and strongest directions as its confidence, with a verdict of low confidence when the weakest σ is over `low_confidence_sigma_deg` (the prototype marked σ over 2°). It leaves the normal unchanged only when no view joined. Precision is still used inside the sequence, in the agreement rule above; it does not decide whether the plane is kept.

This differs from `grid_plane_normal`, which leaves the normal unchanged when it has no good estimate. The reason is that a rejected plane can still be much better than the start: on two of the four points of the second round, a 2° gate on σ left the patch at its start, 8.5° and 80.1° from the stored normal, where the planes it rejected were 1.36° and 16.8°. A low-precision plane says the views constrain the tilt little near it, not that the start is closer; the local curvature of the ZNCC understates what narrow views can correct from far away, as one narrow view took bull point 236 from 80° to 5.6°. A consumer that needs a precise normal reads σ and chooses its own bar.

### A proposed ordering rule

Not prototyped. The angle from the reference is a proxy for how far the current plane's prediction is from the truth in a view. The plane's uncertainty, carried into each remaining view, predicts that error directly: the map error over a cell, given the plane's covariance (§ [The plane's precision](#the-planes-precision)). The proposal is to take next the view the current plane is predicted to get closest, and to defer a view whose predicted squeeze is past about 1.7×. That bar sits below what was captured on the example: from a single view, image 62, whose start needed a stretch of about 2×, was captured, and images 65 and 66, whose starts were squeezed by about 3× and 4×, were not.

### Relation to `grid_plane_normal`

`grid_plane_normal` reads the plane from where nine cells register, by shift alone, on a render made at the current plane, and gates each cell in each view by its ZNCC, its curvature, the bound and the view's robust affine fit; it can only measure what lands inside the shift bound. This step instead treats the plane as the thing being fitted, and every view's picture as evidence for it. The two share the reference rule, the starting frame and the final turn and readback. They differ on a weak estimate: `grid_plane_normal` leaves the normal unchanged when it has no good estimate, and this step turns to any plane at least one view joined and reports its σ. Where `grid_plane_normal` ends `Turned`, the two should agree; the cases this step is for are the wide views `grid_plane_normal`'s gates refuse at the starting plane.

## Implementation notes

- **Cost is unknown.** The prototype is Python and its time does not transfer. What dominates is one direct render of the window per optimiser evaluation per view, and one render per joined view per evaluation of the shared refinement. Gauss–Newton would cut the evaluations against a pattern search at the cost of a Jacobian of the render with respect to the tilt and the shift.
- The render is the plane-induced map `WarpMap::from_patch` makes for every bench tile, applied over a window rather than the whole tile; the shift is applied in the photograph after projection, not by re-anchoring the patch on the view's keypoint. A view's own keypoint is read once, for the keypoint-implied shift that centres its stage-1 search; a view with no keypoint has none to imply, and how its search is centred is not settled (the prototype's views all had keypoints).
- The information estimate costs, per view, a whole-grid ZNCC with the shift fitted again at each of its finite-difference samples: the centre, two per axis and one mixed. Holding a view to the shared plane costs one more shift fit.
- The write sits between the turn and the readback. The other steps end through `turn_and_read` ([bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs)), which runs `tilt_patch` and then `evaluate_rendering_bitmap` with nothing between them; this step needs its keypoint write between the two. The anchor and each joined view's final shift are kept from the search for it, the wide-pair stage included.
- The wide-pair stage, where it runs, adds two renders per evaluation, one per view of the pair, on top of the joined views' reference renders kept in its objective.
- A track at infinity is refused by `normal_preconditions`, as for the other steps: its patch faces along its own bearing and has no plane to fit.

## Determinism and precision

Photograph samples and ZNCC sums are `f32`, as for every bench tile; the tilt, the shifts and the plane are `f64`. The views are taken in a fixed order, ties in angle broken by observation index, with no randomness, so the output is the same for any thread count. No content hash depends on it.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `working_resolution` | `48` | Side of the working grid, samples; the prototype's value |
| `spot_window` | `7` | Side of the first window, grid px |
| `spot_candidates` | `4` | Contrasted candidate spots per view, the centre tried besides them; the second round's sequence picked among these five by best stage-1 ZNCC, and the first round's sequence used 1 |
| `window_fractions` | `[1/8, 1/5, 1/3, 1/2]` | Half-widths of the later windows as fractions of the grid's side, rounded, any no wider than 3 dropped: 6, 10, 16, 24 on a 48 grid; 5, 8, 12 on a 24 grid |
| `keypoint_search_px` | `3.0` | Bound of the first window's shift search about the keypoint-implied shift, image px |
| `min_keypoint_zncc` | `0.7` | Best stage-1 ZNCC within that bound below which the search widens |
| `spot_search_px` | `12.0` | Bound of the widened search, image px |
| `spot_search_step_px` | `0.5` | Step of both searches, image px |
| `zncc_tolerance` | `0.01` | ZNCC difference noise can produce, τ; a placeholder |
| `tilt_step` | `0.03` | Finite-difference step of the information estimate, tilt units |
| `max_agreement_distance` | `3.0` | Largest Mahalanobis distance of a view's plane from the shared plane |
| `min_held_zncc` | `0.5` | Whole-grid ZNCC a view needs when held to the shared plane; the first view's only bar |
| `low_confidence_sigma_deg` | `2.0` | σ above which the report says low confidence; it does not stop the turn |
| `min_coverage` | `0.6` | Share of a window's samples on the photograph for its ZNCC to be read |
| `wide_pair_stage` | `false` | Run the wide-pair stage after the sequence; a proposal, not prototyped |

Every default but `wide_pair_stage` is the prototype's, set on one point in the first round and four in the second.

## Testing

Synthetic planes rendered through known poses, as `grid_plane_normal`'s tests do:

- A textured plane started 20° off its normal, seen from views out to 80°: every view joins and the normal is within 1° of truth.
- The same scene with the point's depth along the reference ray off by several image px in the wide views: the per-view shifts absorb it and the normal is unchanged within the same tolerance.
- A wide view whose start is squeezed past capture: it fails from the starting plane and joins from the shared plane after the nearer views.
- A flat patch: no view joins, and the normal and every keypoint are left as given, with no precision reported.
- A textured plane seen only from views within a few degrees of the reference: the patch is turned, σ along the weakest direction is large and the report says low confidence.
- A wrong join: a view whose own plane is far from the shared plane but which, held to the shared plane, still scores over `min_held_zncc` is refused by the Mahalanobis distance and does not move the shared plane.
- Repeating texture under the spot: a stripe or checker period of a few image px, where a 7×7 window matches equally at more than one shift. Started from the keypoint-implied shift, stage 1 takes the right peak and the view joins on its first try; a keypoint off by more than `keypoint_search_px`, or a peak under `min_keypoint_zncc`, widens the search and is reported as widened.
- The precision: on a synthetic plane, σ from the finite-difference information matches the spread of the fitted normal over renders with noise added so that ZNCC differences of about τ arise.
- The written keypoints: in the depth-error scene, each joined view's keypoint is, to within a fraction of a px, the projection of the true surface point on the reference keypoint's ray, the reference's keypoint is the one it had before the step, a view left out and a row that is `out` carry the keypoint `tilt_patch` gives them, and every row's readings are those of the readback at its new keypoint.
- The wide-pair stage: with the reference in the middle of the views and the sequence stopped short of convergence, turning the stage on brings the normal closer to truth, and is reported as kept; with the reference at one end of the views it is reported as skipped; with lighting changed in one view of the widest pair so that the ZNCC between the pair's two renders favours a wrong tilt, it is reported as not kept and the normal and keypoints are those the step gives with the stage off.

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

**What the numbers cannot show.** How often the method converges, on which kinds of point it fails, and what it costs. The full sweep has not been run. None of these numbers includes the wide-pair stage, which was not prototyped.

## Prototype on four points (2026-10-16)

**Question.** Does judging a view by its agreement with the shared plane, scaled by the plane's precision, join the views the convergence bars left out without letting in wrong ones; and does starting stage 1 from the keypoint-implied shift keep the first window from picking a wrong peak?

**Setup.** Four of the maintainer's hand-picked points, each started from the mean viewing direction. Angles are to the stored normal, which is an estimate, with the same caveats as in the first round: a stored normal is not ground truth, and four points say what the method can do, not how often.

| Point | Views beside the reference, angle from it | Start |
|-------|-------------------------------------------|-------|
| dino `pt3d_0fdc9e3b_810` | 17, 7° to 104° | 22.8° |
| Kerry Park `pt3d_4ddeabf6_91` (fisheye) | 3, at 3°, 26° and 104° | 32.4° |
| dino `pt3d_0fdc9e3b_933` | 3, at 3°, 8° and 10° | 8.5° |
| Seoul bull `pt3d_e6c46c07_236` | 2, at 2° and 4° | 80.1° |

Every run in this round picked the spot by the best stage-1 ZNCC among the four most contrasted non-overlapping 7×7 windows of the middle cell and the geometric centre.

**Today's steps from the same start.** Today's Grid Plane Normal at 2×2: 20.3°, 25.6°, 1.2° and 32.2°. `grid_plane_normal` (bar 0.6, bound 4 px, one render): first plane 6.5°, 3.3°, 1.3° and none; it kept none, ending `Unstable` on the first three and `Undetermined` on 236.

**Convergence bars.** With the bars of the first round: 0.61° (12 of 17 views joined), 1.39° (2 of 3), 8.53° (0 of 3; nothing joined, so the normal stayed at its start) and 16.8° (2 of 2).

**A fixed ZNCC gap.** A view agreed when holding it to the shared plane, its shift fitted again, cost at most 0.03 of whole-grid ZNCC and left it at least 0.5, the first view needing only the 0.5 floor: 0.47° (17 of 17), 1.12° (3 of 3), 1.36° (3 of 3) and 16.8° (2 of 2). One join was wrong. On point 810, image 27, 14° from the reference, failed from the shared plane, growing to 68°; its retry from the starting plane ended 26° from the shared plane yet cost exactly 0.030 when held to it, so it passed, and it pulled the shared plane from 0.89° to 2.89° before later views brought it back.

**Precision agreement.** With the Mahalanobis rule of § [When a view agrees with the shared plane](#when-a-view-agrees-with-the-shared-plane), τ = 0.01: point 810's image 27 was refused at 3.2σ from the shared plane and at 3.6σ on its retry; 16 of 17 joined, ending 0.54° with σ 0.38°. Kerry Park 91 ended 1.12° with σ 1.43°, dino 933 1.36° with σ 7.1°, and bull 236 16.8° with σ 390°. On point 810 a single view 7° to 9° from the reference has σ of 13° to 17°, and images 62, 65 and 66 each 0.6° to 0.9°; the shared σ fell from 13.8° after two views to 0.37° at the end. With a 2° gate on σ the prototype first left 933 and 236 at their starts, 8.5° and 80.1°; the maintainer reversed that to the rule of § [Turn and report](#turn-and-report).

**The shift diagnosis.** On point 810, at the stored normal: image 27's centre 7×7 window picked a shift of (6.5, 3.0) at ZNCC 0.958, with a second peak at (3.0, 6.5) also at 0.958, where the whole grid's best shift is (−0.5, −0.5), at which the 7×7 window scores 0.937. A 13×13 window scores 0.51 at the wrong shift and 0.91 at the right one. Image 45's centre window picked (−6.5, −6.5) at 0.967 against the right (1.0, 0.5). The centre was the chosen spot for 11 of the 17 views. The keypoint-implied shifts were (−0.44, −0.41) for image 27, (0.82, 0.65) for image 45 and (−6.57, −2.62) for image 36.

**Keypoint start.** With precision agreement and stage 1 started from the keypoint-implied shift: point 810 joined 17 of 17, every view on its first try, ending 0.61° with σ 0.37°; image 27 ended 2.3° from the stored normal and image 45 0.75°. Kerry Park 91 ended 1.11° with σ 1.41°, dino 933 1.45° with σ 7.1°, and bull 236 16.0° with σ 106°. Four views widened to the full search: point 810's images 10, 09 and 62, and bull 236's image 10. Image 62's keypoint implies (2.46, 1.69), where single-view growth earlier settled at about (2.4, −4.7), so either its keypoint is off in that 80° view or the shift at the window's spot depends on the tilt there; the widened search covers it.

**What is still weak.** Points seen only by narrow views get planes of low precision: 933's plane is 1.45° off with σ 7.1°, where the 2×2 Grid Plane Normal reaches 1.2°. Bull 236 drifts in the joint refinement: its first view alone reached 5.6°, the shared plane ends at 16°, and the 2×2 step at 32°.

**What the numbers cannot show.** How often the method succeeds, on more than four points, and what it costs; the cost is still not measured in a form that transfers to Rust. None of these numbers includes the multi-spot rule or the wide-pair stage.

## Non-goals

**Moving the centre.** The step turns the patch about its centre, as the other normal steps do, and does not move it. It writes the keypoints of the views that joined (§ [The keypoints the step writes](#the-keypoints-the-step-writes)); whether it should also re-triangulate the centre from them is an open question.

**Per-cell one-dimensional constraints.** A cell over stripes pins its shift across the stripes only. Using such constraints was discussed but is not part of this method.

## Open questions

- **The optimiser.** Pattern search, as prototyped, or Gauss–Newton on the ZNCC.
- **The window schedule and the spot rule.** Whether these fractions of the grid's side are the right schedule. On the first example the most contrasted window of the middle cell was the wrong first window for images 65 and 66, and other candidates converged; whether to grow every candidate and keep the best agreeing result, or only the candidate with the best stage-1 ZNCC, how many candidates to take, and what to do when the centre is blank. The best-stage-1 rule has been run in the sequence, with the keypoint start; growing every candidate has not.
- **The order of the views** beyond angle from the reference; the predicted-error rule above is the candidate.
- **The agreement numbers.** Whether τ = 0.01 is too cautious: it scales every σ, so it sets how many views agree and how often the report says low confidence. Whether 3σ and the 0.5 held floor are the right bars; they were set on four points.
- **Keeping the best single-view plane.** On bull 236 the first view alone reached 5.6° and the joint refinement drew the shared plane to 16°. Whether the step should keep the best single-view plane when the joint refinement lowers that view's agreement, and how it would tell, without the stored normal, that the refinement made the plane worse.
- **Narrow-view points.** On a point whose views all lie within about 10° of the reference, the plane has low precision (933: 1.45° off, σ 7.1°), where the 2×2 Grid Plane Normal reaches 1.2°. Whether to hand such points to a cell-based step.
- **A view's keypoint in a wide view.** Image 62's keypoint-implied shift differs from where its single-view growth settled by about 6 px in y; whether its keypoint is off in that 80° view or the shift at the window's spot depends on the tilt there is not known, and the widened search is what covers it.
- **Weighting the shared refinement.** Whether views should be weighted, since a wide view carries more information about the tilt per degree than a narrow one.
- **Blur matching the other way.** A view finer than the reference aliases when sampled bilinearly from the full-resolution photograph; sampling it from a coarser pyramid level, or blurring it, may be needed.
- **What the step reports.** σ along the weakest and strongest directions and the low-confidence verdict are decided; the exact fields of the rest are open: each view's outcome and start, its agreement distance and held ZNCC, whether its stage-1 search widened, which joined and which were left out, each written view's keypoint before and after the write, the shared plane, and whether the wide-pair stage ran, on which pair or pairs, the plane before and after it, and whether its result was kept.
- **The wide-pair stage's pair selection.** The widest joined pair only, as proposed; every joined pair whose angle at the point is over some bar; or every pair, weighted by its angle. Whether a pair must lie on opposite sides of the reference, or only be wider than the widest joined view's angle from the reference, as proposed.
- **The wide-pair stage's acceptance rule.** Whether the mean whole-grid ZNCC against the reference not falling, with every joined view still over `min_held_zncc`, is the right gate, or whether it should use the agreement rule's precision; whether a small fall should be allowed when the ZNCC between the pair's two renders rises by much; and how the pair's term is weighted against the reference terms in the objective.
- **The wide-pair stage's grid.** Whether the pair is compared on the reference's working grid, as proposed, or on a grid of its own, for example one centred on the bisector of the two views' rays, where both renders are foreshortened alike.
- **The centre after the write.** A depth error shows up as the per-view shifts, so the written keypoints can imply a different centre than the patch's. Whether the step should re-triangulate the centre from the written keypoints before the readback, or leave it to the next fit, is not decided. The step's plane passes through the anchor and the turned patch's through its centre, so the two planes also differ in depth wherever the anchor is not the centre.
- **A bound on the write.** A fit keeps a sighting at its seed when its kernels would move it further than `max_shift_px` on the patch's plane, and records the refusal in `walked_px` and `walked_to`. Whether this step should bound its write the same way, and what it measures the move from, is open; the size of the moves on real tracks has not been measured.
- **The full sweep** over every point of `seoul_bull_sculpture`, `kerry_park` and `dino_dog_toy` is not done.
