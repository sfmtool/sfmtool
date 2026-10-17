# Grown Plane Normal

**Status:** Draft. A fourth bench normal step, `grown_plane_normal`, beside `fit_normal`, `finite_difference_normal` and `grid_plane_normal`, that fits one plane shared by every view directly to the photographs. It is a separate step and not an option of `grid_plane_normal`, because it measures nothing that step measures: it has no cell shift search on a fixed render and no triangulation.

Decided: the step's existence as a separate step; that it writes each joined view's keypoint back from the shift it solved for (§ [The keypoints the step writes](#the-keypoints-the-step-writes)); that a view joins by its agreement with the shared plane, scaled by the plane's precision, rather than by convergence bars on its own ZNCC readings (§ [When a view agrees with the shared plane](#when-a-view-agrees-with-the-shared-plane)); that the step turns the patch to the shared plane whenever at least one view joined and reports the plane's precision as its confidence, rather than rejecting a plane it found (§ [Turn and report](#turn-and-report)); that stage 1 starts each view from the shift its own keypoint implies, with its window centred on the anchor, and that each view is grown from that shift with no search first, searching only when that growth does not join (§ [Near to far within a view](#near-to-far-within-a-view)); and that every score of the whole grid is the robust combination of the grid's nine cells, with a last stage that fits the whole grid on that score (§ [The robust cell score](#the-robust-cell-score)). Decided too: that the step has no fit of the views together, and combines the planes each view reaches against the reference by their information (§ [Combining the views' planes](#combining-the-views-planes)). Not decided: which of the five strategies that combine the planes is kept (§ [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17)); whether the optimiser built here, Levenberg–Marquardt, is kept; the window schedule; the order views are taken in; the ZNCC tolerance and the other numbers of the agreement rule; the rules of the wide pairs and of the selection; the report's exact fields; whether it also moves the patch's centre; and how a textured spot off the anchor is used. See [Open questions](#open-questions).

Built: the step in `sfmtool-core` ([bench/normal/grown_plane.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane.rs)), with its tests on synthetic planes (§ [Testing](#testing)), and its Python binding, `sfmtool.bench.grown_plane_normal` ([bench/normal.rs](../../crates/sfmtool-py/src/bench/normal.rs)), which the evaluation scripts `grid_plane_eval.py` and `grid_plane_detail.py` in [scripts/track_at_pixel/normal_sources](../../scripts/track_at_pixel/normal_sources) run with `--grown`. The five strategies are built side by side behind one setting, `strategy`, so that each can be measured from the same build; one is to be kept and the others removed. Not built: a Track View entry or a wire method, so the viewer does not call it yet; the multi-spot rule; the ordering of the sequence by information; the pinned-anchor tilt search. A Python prototype, not in the repository, was run on one point of the `dino_dog_toy` candidate ground truth ([Prototype on one point](#prototype-on-one-point-2026-10-15)), and a second round of it on four points ([Prototype on four points](#prototype-on-four-points-2026-10-16)); the step follows the prototype's method in the branches the maintainer approved, with the differences § [Relation to the prototype](#relation-to-the-prototype) lists.

Amends: [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal", which names the step and points here.

Related drafts: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) designs `grid_plane_normal` and records the measurements this step is compared against. [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) defines the predicted precision of a fitted normal, which bears on the proposed ordering rule. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) decides how a view is scored against the reference at matched sharpness, which bears on blur matching here.

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in the world, and its normal says which way the square faces. Each photograph that sees the patch is resampled onto the square through that plane, so a wrong normal stretches and shears every view's picture of the patch against the others, blurs the patch's stored bitmap and misleads every reading taken from it afterwards.

This step estimates the normal by fitting the plane to the photographs themselves. One view is the reference. For every other view, the plane predicts where each part of the reference's picture of the patch lands in that view's photograph. The step finds the plane under which the views' pictures agree with the reference's, by ZNCC. Within a view it registers a small window at the patch's anchor first and grows the window outward, because a small window tolerates a large error in the plane; each view's growth ends at that view's own plane, with its information about the plane. The step combines the views' planes by an information-weighted mean, never by fitting the views together. By default it takes the views closest in angle to the reference first and the wide views last, each grown from the mean of the planes the earlier views reached, because a wide view is where a wrong plane distorts the picture most; it then matches the widest pairs of views against each other and adds their planes to the mean. The step turns the patch to the plane it found whenever at least one view joined, reports how precisely the estimates determine that plane, and writes each joined view's keypoint at the pixel where that plane and the view's own shift put the patch.

### Why a new step

`grid_plane_normal` ([piece-gated-grid-normal.md](piece-gated-grid-normal.md)) measures the normal by parallax: it renders each view once through the current plane, registers each of nine cells by a shift search within a small bound, gates each cell in each view by a robust affine fit, triangulates the cells and fits a plane through them, then turns the patch and repeats. Each part of that chain needs the starting plane to be near enough that every cell of a view lands within the shift bound and that each cell's picture in the view differs from the reference's by little more than a shift, since a cell is registered by shift alone. In a wide view a wrong plane breaks both: the cells far from the centre are displaced past the bound, and the cells whose picture is stretched or turned against the reference's fall under the cell ZNCC bar.

This step replaces the nine-cell shift search, the per-view affine gate and the cell triangulation with a direct fit of the plane to the photographs. It uses the plane's own prediction to reach the views the shift search cannot: a view taken late in the sequence is rendered through the plane the views before it reached, so its picture already arrives close to the reference's. It keeps the nine cells only as the parts a whole-grid score is combined from (§ [The robust cell score](#the-robust-cell-score)).

### Why the name

The glossary's **fit normal** / **finite-difference normal** / **grid-plane normal** entry names each bench normal step for the evidence it reads. *Grown plane normal* says the estimate is a plane and that its support grows: the window over which a view is registered grows from a few samples to the whole grid, and, in sequence, the set of views the plane is combined from grows one view at a time. It pairs with *Grid Plane Normal* in Track View's list. `homography_normal` names the warp but not the schedule that is the method's point, and *homography* is not bench vocabulary. `incremental_plane_normal` says the views are added one at a time but not that each view's window grows. *Register* is not used in the name, because the glossary gives it the meaning of giving an image a pose. When the name is settled, a glossary entry is added with the step.

## Rust API

The step lives beside `grid_plane_normal` in [bench/normal/grown_plane.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane.rs), with the plane-induced render and the scores in [grown_plane/render.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/render.rs) and the optimiser in [grown_plane/solve.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/solve.rs). It has the signature every bench normal step has ([bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs)), so Track View and the wire's `fit_bench_track_normal` can call it the way they call the other three once they are given it. The weighted mean, the robust combination and the agreement distance are in [grown_plane/combine.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/combine.rs), the sequence in [grown_plane/sequential.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/sequential.rs), and the strategies that match each view alone, the pairs and the selection in [grown_plane/independent.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/independent.rs). The Python binding `sfmtool.bench.grown_plane_normal` ([bench/normal.rs](../../crates/sfmtool-py/src/bench/normal.rs)) calls it already: it takes every `GrownPlaneNormalOptions` setting as a keyword of the same name, defaulting to the Rust default (`strategy` as a word: `"sequential_mean"`, `"sequential_pairs"`, `"independent"`, `"independent_pairs"` or `"selected_pairs"`), and `fit`'s four keywords, and gives the sequence back as the report's `estimate`, `kind` `"grown_plane"`, with one dict per view, per try, per pick and per pair; [editable-track.md](../core/bench/editable-track.md) § "Estimating the normal" lists its fields.

```rust
/// How grown_plane_normal combines the views' planes. Kept side by side while
/// they are measured; one is to be kept.
pub enum GrownPlaneStrategy {
    SequentialMean,
    #[default] SequentialPairs,
    Independent,
    IndependentPairs,
    SelectedPairs,
}

/// How grown_plane_normal grows its windows, fits its planes and decides
/// which views it keeps. Defaults in § Parameters.
pub struct GrownPlaneNormalOptions {
    pub strategy: GrownPlaneStrategy,
    pub working_resolution: Option<u32>,
    pub spot_window: u32,
    pub window_fractions: Vec<f64>,
    pub keypoint_search_px: f64,
    pub min_keypoint_zncc: f64,
    pub spot_search_px: f64,
    pub spot_search_step_px: f64,
    pub zncc_tolerance: f64,
    pub max_agreement_distance: f64,
    pub min_held_zncc: f64,
    pub low_confidence_sigma_deg: f64,
    pub max_pairs: usize,
    pub target_sigma_deg: f64,
    pub max_selected_views: usize,
    pub min_coverage: f64,
    pub max_iterations: u32,
    pub initial_damping: f64,
    pub min_improvement: f64,
    pub jacobian_tilt_step: f64,
    pub jacobian_shift_step_px: f64,
    /// The reading the step ends with, and the reference-view rule's grazing
    /// gate and sampler (`localize`).
    pub fit: FitOptions,
}

impl GrownPlaneNormalOptions {
    /// Refuses a setting the step cannot run with, naming it:
    /// NormalError::BadGrownPlaneOption("spot_search_step_px").
    pub fn check(&self) -> Result<(), NormalError>;
}

/// What the step did, carried by NormalEstimate::GrownPlane(Box<_>).
pub struct GrownPlaneSequence {
    pub strategy: GrownPlaneStrategy,
    pub reference: Option<usize>,          // None: no reference, nothing tried
    pub working_resolution: u32,
    pub window_half_widths: Vec<u32>,      // the stages' windows, grid px
    pub views: Vec<GrownPlaneView>,        // one per observation of the track
    pub trace: Vec<GrownPlaneTry>,         // every try, in the order made
    pub picks: Vec<GrownPlanePick>,        // SelectedPairs's picks, in order
    pub pairs: Vec<GrownPlanePair>,        // the pairs matched, widest first
    pub normal: Option<[f64; 3]>,          // the final plane; None: none joined
    pub precision: Option<GrownPlanePrecision>,
    pub written: usize,                    // keypoints written, the reference's among them
}
// joined(), left_out(), shift_only(), matched(), pairs_kept(), turned();
// Display: "5 of 5 views joined, 2 of 2 pairs kept; σ 0.56° weakest, 0.48°
// strongest (0.67° weakest with each view once); 6 keypoints written", or
// "0 of 5 views joined, 5 left out, normal left unchanged".

pub enum GrownPlaneView {
    /// The keypoint written after the turn: the one it had, or the anchor's
    /// projection where it had none; None when nothing was written.
    Reference { keypoint: Option<[f64; 2]> },
    /// from_start and start of the growth that was kept; distance and
    /// held_zncc its agreement when it joined in sequence, else its plane's
    /// distance from the final plane and its score there; shift (moved to the
    /// final plane) and grid_zncc at the final plane; the keypoint before the
    /// step and the one written (None when off the photograph).
    Joined {
        from_start: bool,
        start: GrownPlaneStart,
        shift: [f64; 2],
        grid_zncc: f64,
        distance: Option<f64>,
        held_zncc: Option<f64>,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    /// Not matched by SelectedPairs: its shift alone fitted at the final
    /// plane; keypoint written when grid_zncc reaches min_held_zncc.
    ShiftOnly {
        start: GrownPlaneStart,
        shift: [f64; 2],
        grid_zncc: f64,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    LeftOut { reason: GrownPlaneRefusal, distance: Option<f64>, held_zncc: Option<f64> },
    /// Not in, a second row in one image, or no reference.
    NotAttempted,
}

pub enum GrownPlaneStart { NoSearch, Keypoint, Widened, NoKeypoint }
pub enum GrownPlaneRefusal { UnderFloor, TooFar, HeldUnderFloor, ViewLeftOut }

/// One try of one view, and the shared plane after it.
pub struct GrownPlaneTry {
    pub observation: usize,
    pub image: u32,
    pub angle_deg: f64,
    pub attempts: Vec<GrownPlaneAttempt>,  // shared plane, then start; each unsearched, then searched
    pub outcome: GrownPlaneTryOutcome,     // Joined, Matched, Deferred or LeftOut
    pub shared_normal: Option<[f64; 3]>,
    pub shared_sigma_deg: f64,
}

/// One growth of one view from one plane, and how it was judged.
pub struct GrownPlaneAttempt {
    pub from_start: bool,
    pub start: GrownPlaneStart,
    pub stage_zncc: Vec<f64>,   // stage 1 at its starting shift, each later window, the whole grid's robust score
    pub shift: [f64; 2],
    pub normal: [f64; 3],       // the view's own plane
    pub grid_zncc: f64,
    pub distance: Option<f64>,  // None for the first view and a growth judged alone
    pub held_zncc: Option<f64>, // None for the first view and a growth judged alone
    pub sigma_deg: f64,         // its own weakest σ
    pub refusal: Option<GrownPlaneRefusal>,
}

/// One pair of kept views matched against each other.
pub struct GrownPlanePair {
    pub observations: [usize; 2],
    pub images: [u32; 2],
    pub angle_deg: f64,               // between the two views, from the patch centre
    pub normal: [f64; 3],             // the pair's own plane
    pub relative_shift: [f64; 2],     // the second view's shift against the first's
    pub zncc: f64,                    // the robust score between the two renders
    pub sigma_deg: f64,
    pub distance: Option<f64>,        // from the combined plane
    pub refusal: Option<GrownPlaneRefusal>,
}

/// One view SelectedPairs picked.
pub struct GrownPlanePick {
    pub observation: usize,
    pub image: u32,
    pub angle_deg: f64,
    pub offset_direction: [f64; 2],   // in the starting frame's (u, v)
    pub scale_px: f64,                // photograph px per world unit at the anchor
    pub predicted_weak: f64,          // smallest eigenvalue predicted with it; NaN by angle alone
    pub matched: bool,
    pub sigma_after_deg: f64,
}

pub struct GrownPlanePrecision {
    pub sigma_weak_deg: f64,               // every kept estimate's information summed
    pub sigma_strong_deg: f64,
    pub discounted_sigma_weak_deg: f64,    // each kept pair's information halved
    pub discounted_sigma_strong_deg: f64,
    pub low_confidence: bool,   // discounted_sigma_weak_deg over low_confidence_sigma_deg; turned either way
}

pub fn grown_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GrownPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError>;
```

**Why this shape.** The signature is the other steps', and the step ends almost as they do: it turns the patch with `tilt_patch`, then writes the keypoints of the reference and of every joined view, then reads the track back, so a caller can swap one step for another and compare `NormalReport`s. The write comes after the turn because `tilt_patch` itself moves every keypoint, and would carry the written ones off the pixels the step found; `turn_and_read` in [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs) runs through `read_back`, which takes the write as a closure between the two. The settings are `*Options` because a caller chooses them. The report is per observation because a wrong normal is diagnosed by asking which views joined, in what order and from which start, and why the others were left out, and because a person checking the write needs each view's keypoint before and after it; the trace keeps every growth, so a view retried from the starting plane or deferred shows both tries, and the picks and pairs say what the selection chose and what each pair measured. The strategy is one enum field rather than separate steps because the strategies share every growth and differ only in which planes they combine and how; while it is measured, every strategy runs from the same build. The step changes the normal and writes keypoints whenever at least one view joins, however low the plane's precision, and reports that precision (`GrownPlanePrecision`); it leaves the track as given only when no view joins, and then its sentence ends "normal left unchanged". This differs from `grid_plane_normal`, which leaves the track as given on every ending that is not `Turned` (§ [Turn and report](#turn-and-report)). It refuses a track at infinity and one with fewer than two `in` observations through `normal_preconditions`, and it reports a phase for the whole step and one per try of a view, with finer detail phases when `progress` is detailed (§ [Implementation notes](#implementation-notes)), and is cancelled between stages through `progress`, since it can run longer than a second.

**Example.** `grown_plane_normal(&track, &edited, &images, &GrownPlaneNormalOptions::default(), &Progress::none())` returns the turned track and a report whose sentence reads, for example, "17 of 17 views joined, 3 of 3 pairs kept; σ 0.26° weakest, 0.19° strongest (0.29° weakest with each view once); 18 keypoints written, turned 22.8°, …".

## Theory

### One plane shared by all views

The model is one plane, the same for every view. The warp from the reference view's working grid into any other view is the plane-induced map: each sample of the grid is followed along its reference ray to the plane, and that point is projected into the view's photograph through the camera's lens model, where the photograph is sampled. There is no free per-view affine map; the plane predicts each view's local affine map from its tilt alone.

The working grid is the starting frame re-anchored on the reference keypoint (`OrientedPatch::anchored_at_keypoint`), `W` samples a side, sample `(col, row)` at grid position `(col + ½ − W/2, row + ½ − W/2)`: the reference render's grid in `grid_plane_normal`. `W` is `working_resolution`, or with `None` that step's rule: the reference view's footprint of the patch at the starting frame, about one sample per photograph pixel, clamped to `R ..= 2R` for the patch resolution `R`; at least `spot_window` either way.

The plane has two unknowns, a tilt `(a, b)` of the starting frame's normal `n` along the frame's own axes `u` and `v`:

```text
normal = unit(n + a·u + b·v)
```

through the anchor, the point where the reference keypoint's ray meets the starting frame. Each view adds its own image shift, two unknowns in image px, added to the projected positions. The shift absorbs error in the point's position and in the view's pose: a wrong depth along the reference ray moves every sample of a wide view by about the same amount, up to about 8 image px on the example below (image 36's shift was (−6.75, −4.25) px), which the tilt cannot express and should not try to.

**The keypoint-implied shift.** Each view's own keypoint already places the patch in that view, since the fit put it there. The shift it implies is the keypoint minus the anchor's projection into the view. Every plane the step tries passes through the anchor, so this shift is the same at every tilt.

**The render.** Every evaluation samples the photograph directly through the current plane, rather than resampling a tile rendered once at the starting plane, because a tile rendered at a wrong plane has already lost detail along the axis that plane squeezed. Each view is read from one level of its pyramid for the whole step: the level the bilinear-mip rule picks, `round(log2 σ_major)` clamped to the pyramid's depth, for the larger singular value of the map from the working grid at the starting plane into the view, at the grid's centre (`patch_grid_jacobian`). A sample is a bilinear read at that level, kept as a float, and is on the photograph when all four of its taps are. The bench's tile renders (`render_tile`) choose a level per sample or walk an anisotropic footprint and round to bytes; either would make the score jump as the plane turns, which a forward-difference Jacobian cannot follow, so the step keeps one level per view and the float sample. The reference is rendered the same way at no tilt and no shift; its rays meet every plane where the reference sees it, so its tile is the same at every tilt and is the template every view is scored against. Its level is 0 unless the grid was clamped well under its footprint.

### Near to far within a view

Under a plane that is wrong, a sample at distance `r` from the window's centre is displaced in the view by about `‖A − I‖·r`, where `A` is the error in the view's local affine map. A small window therefore tolerates a large error in the map, and a map fitted over radius `r` is accurate enough to bring radius about `2r` within capture of the next fit.

**Stage 1 at the anchor.** Stage 1 places a window of `spot_window` grid px (7×7, half-width 3) on the grid's centre sample, `(W/2, W/2)` by integer division: on an even grid that sample is half a sample right of and below the anchor. A view is grown first from its keypoint-implied shift with no search: stage 1 reads the window's ZNCC at that shift, which is the stage-1 score `stage_zncc` records, and the growing windows fit the shift from there together with the tilt (`NoSearch`). Only when that growth is refused (§ [Views in sequence, sharing the plane](#views-in-sequence-sharing-the-plane)) is the view grown again from a stage-1 search. The search finds the view's shift alone, at the current plane, exhaustively on a lattice of `spot_search_step_px` (0.5) image px about the view's keypoint-implied shift, up to `keypoint_search_px` (3) image px along each axis, scanned row by row with the first maximum kept (`Keypoint`). It widens to `spot_search_px` (12) image px about the same keypoint-implied shift only when no shift within the ±3 px range reaches `min_keypoint_zncc` (0.7) or the best one is on the edge of that range (`Widened`). A search whose best shift is the keypoint-implied shift itself would repeat the growth just refused, so it is not grown again. A view with no keypoint, or one whose anchor does not project into it, has no implied shift; it is grown from the search alone, over the ±12 px range about zero shift, the anchor's own projection (`NoKeypoint`).

The growth with no search comes first because the 7×7 window can be ambiguous where the keypoint is not. On dino point 933, seen by three views within 10° of the reference, the ±3 px search found no shift reaching 0.7 in any view and widened, and in every view it picked a wrong peak scoring about 0.6. The Python prototype recorded first-window shifts of (−5.0, 11.0), (−10.0, 10.5) and (−7.0, −1.5) px for the three views, against keypoint-implied shifts of (1.74, −1.39), (1.77, −0.99) and (0.18, −1.35) px, so the peaks were about 14, 16 and 7 px from the implied shifts; the fit then collapsed and no view joined, although each keypoint-implied shift was within 0.7 px of the true one. Grown from the implied shifts, all three views join (§ [Measurement on sixteen points](#measurement-on-sixteen-points-2026-10-17)). The search stays as the fallback for a view whose keypoint is off by more than the growing windows recover.

The search is centred on the keypoint-implied shift because a 7×7 window on repeating texture matches equally well at more than one shift, and a search over the full ±12 px picks a wrong peak as readily as the right one (see [Prototype on four points](#prototype-on-four-points-2026-10-16)). The window is centred on the anchor because the keypoint-implied shift is exact only there: every plane passes through the anchor, so at the anchor the right shift does not depend on the plane, but at a spot `r` away it differs by the plane error's displacement there, about `(A − I)·r`. On dino point 528 that displacement carried image 45's match at an off-centre spot outside the ±3 px search, so the search widened and locked onto a wrong shift 10 px away. The multi-spot proposal of the first round and the spot candidates no longer apply to stage 1 under the keypoint start. A textured spot off the anchor needs a different compensation, for example pinning the anchor at the keypoint and searching the tilt at stage 1; that is proposed and not built (see [Open questions](#open-questions)), and the maintainer expects points whose centre lacks texture.

**The growing windows.** Each later stage widens the window about the same sample to the next half-width of the schedule, `window_fractions` of the working grid's side `W` rounded to grid px, any no wider than the first window's half-width dropped: 3, then 6, 10, 16 and 24 on a 48-sample grid, 3, 4, 6, 11 and 16 on a 32-sample grid, and 3, 5, 8 and 12 on a 24-sample grid; the half-width `W/2` covers the whole grid. Each stage fits the tilt and the view's shift together for the best mean-channel ZNCC over the window (§ [The optimiser](#the-optimiser)).

**The whole-grid stage.** After the last window, a final stage fits the tilt and the shift for the best robust cell score over the whole grid (§ [The robust cell score](#the-robust-cell-score)). The view's own plane and its whole-grid score are those this stage ends at.

### The robust cell score

Wherever the step scores a whole grid (the floor and every held score, each view's information, the whole-grid stage, the shift-only fit, the pair fit, and the scores in the report), it scores the grid's nine cells rather than the grid at once. The cells are the step's own three-by-three split, rows and columns at `[0, W/3, W − W/3, W]`. A cell where the reference is flat, its spread under 0.5 grey levels in every channel, is not scored. Each live cell's ZNCC is read as every ZNCC is (§ [When a view agrees with the shared plane](#when-a-view-agrees-with-the-shared-plane)), `−1` under the coverage bar, and the cells are combined by a one-sided Tukey weighting:

- `m` is the median of the live cells' ZNCC;
- the scale is 1.4826 times the median absolute deviation from `m` of the cells at or above `m`, floored at 0.05, and the cut is 4.685 scales;
- a cell at or above `m` weighs 1, and one below weighs `(1 − ((m − z)/cut)²)²`, 0 past the cut;
- the score is the weighted mean of the cells' ZNCC; with no live cell it is `−1`.

The scale is read from the cells at or above the median only, because with several bad cells the spread of all of them is so wide that nothing falls past the cut. The reason for the score: on dino point 528, images 45 and 46 match the reference in the bottom two cell rows (0.67 to 0.81) but not the top row (0.40, −0.06 and 0.06), whose texture the two views do not see, and a whole-grid ZNCC refused them. On that point the robust score did not rescue them in the prototype, because the shared plane had drifted 9° after two nearly uninformative views had joined; that is recorded as open, since ordering the views by information is not decided.

### The optimiser

Every fit is Levenberg–Marquardt on the photometric residual of mean- and spread-normalised samples. For one channel, with `a` the reference's samples and `b` the view's over the same samples, each less its mean, `â = a/‖a‖` and `b̂ = b/‖b‖` are unit vectors and

```text
‖â − b̂‖² = 2 − 2·ZNCC(a, b)
```

so maximising the ZNCC is minimising that residual. A block of samples (a window, or one cell) contributes its channels' residuals scaled by `√(w / C)` for its weight `w` and its `C` scored channels, so its residual sum of squares is `w · (2 − 2z)` for its mean-channel ZNCC `z`. A window is one block of weight 1; a whole grid is its live cells, each weighted by its Tukey weight over the weights' sum. Either way the residual sum of squares is `S = 2 · (1 − score)` for the score being fitted.

The residual is frozen at the parameters it is linearised at: which samples are on the photograph, which channels are scored, and the cells' Tukey weights, which are read again at the start of every iteration, as an iteratively reweighted fit. Only the view's samples move with the parameters. The Jacobian is a forward difference of the frozen residual, with steps of `jacobian_tilt_step` (10⁻³) along each tilt parameter and `jacobian_shift_step_px` (0.05 px) along each shift. An iteration solves `(JᵀJ + λ·diag(JᵀJ) + ε) δ = −Jᵀr`, with `λ` from `initial_damping` (10⁻³) and `ε` 10⁻⁹ of the largest diagonal entry, scales `δ` back so that no tilt parameter moves more than 0.1 and no shift more than 2 px, and reads the true score at the new parameters from a fresh render. The step is kept only when the score rises, and `λ` falls tenfold; otherwise `λ` rises tenfold and the step is solved again, up to eight times. A stage ends after `max_iterations` (20) iterations, when an iteration raises the score by less than `min_improvement` (10⁻⁶), or when no damping finds a rise; a stage that cannot rise keeps the parameters it was given.

The parameters are a view's tilt and shift `[a, b, sx, sy]` within a view, its shift alone with the tilt held for the held score in sequence and for the shift-only fit, and in the pair fit the tilt and the second view's shift against the first's `[a, b, rx, ry]` (§ [Wide pairs](#wide-pairs)). The pair fit's residual compares two renders that both move with the tilt, so it normalises both sides afresh at every evaluation, where a view against the reference normalises the reference once. No fit has more than four parameters.

The stage-1 search stays exhaustive, as above: the 7×7 window on repeating texture has several equal peaks, and a local fit from the wrong one stays there.

### Combining the views' planes

Every growth ends at the view's own plane, the tilt `tₑ` its whole-grid stage reached, with its shift and its information `Iₑ` about the tilt, read at that plane and shift (§ [The plane's precision](#the-planes-precision)). The step combines planes by their information and never fits views together. The combined plane of a set of estimates is their information-weighted mean,

```text
t = (Σ Iₑ + ε)⁻¹ · (Σ Iₑ·tₑ + ε·t̄)        covariance 2τ · (Σ Iₑ)⁻¹
```

the plane that maximises the sum of the estimates' quadratic score models `−½ (t − tₑ)ᵀ Iₑ (t − tₑ)`. `t̄` is the plain mean of the tilts and `ε` is 10⁻⁶ of the summed information's largest eigenvalue, or 1 where that is zero, so that the mean of one estimate is that estimate and a direction no estimate informs takes the plain mean rather than being undetermined.

`strategy` picks which planes are combined and how each is found. There are five; one is to be kept (§ [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17)).

- **`SequentialMean`.** The reference is the step's reference view, by the rule `grid_plane_normal` and the fit use: the reference the track holds where its row is `in` with a keypoint, else the reference-view rule's pick among the `in` rows, one row per image. The other `in` views are taken in order of their angle from the reference as seen from the patch centre, ties by observation index. Each view grows from the plane the views before it share. The first view joins when its own whole-grid score is at least `min_held_zncc`; a later view joins when it agrees with the shared plane (§ [When a view agrees with the shared plane](#when-a-view-agrees-with-the-shared-plane)). After each join the shared plane is the weighted mean of the joined views' own planes, each with the information it was judged with, and its covariance `2τ·(Σ I)⁻¹` is the shared plane's in the next agreement test. One try of a view makes up to four growths, in this order, and stops at the first that joins: from the shared plane at the keypoint-implied shift, from the shared plane by the stage-1 search, from the starting plane at the keypoint-implied shift, and from the starting plane by the search. The first view has no shared plane, so it grows from the starting plane only, in the two ways. A view with no implied shift makes the searched growths alone, and a search that lands on the implied shift is not grown again. A view refused every way goes to the back of the queue once, to be tried again after the views behind it have joined, and is otherwise left out.
- **`SequentialPairs`**, the default. `SequentialMean`, then the joined views' planes combined by the robust combination below, the wide pairs among the views it keeps, and the robust combination of the views' and the pairs' planes. A joined view the combination leaves out is reported left out with the combination's reason.
- **`Independent`.** Every view grows against the reference from the starting plane, from its keypoint-implied shift first and from the stage-1 search where that growth's own whole-grid score is under `min_held_zncc`, and counts when a growth reaches that floor on its own. There is no shared plane while the views grow, so there is no order, no retry from another plane and no deferral. The planes that count are combined by the robust combination. A view the combination leaves out whose kept growth had no search is grown once more from the search, as a refused growth with no search is everywhere else, unless the search lands on its implied shift; where that growth reaches the floor it replaces the view's estimate, and the planes are combined again.
- **`IndependentPairs`.** `Independent`, then the wide pairs among the views kept, and the robust combination of the views' and the pairs' planes.
- **`SelectedPairs`.** `IndependentPairs` on a selection of the views, matched one at a time until the combined plane is precise enough (§ [The selection](#the-selection)); every view not matched has its shift alone fitted at the final plane.

The searched growth from the shared plane comes before the unsearched growth from the starting plane in sequence. On the sixteen points the other order, both unsearched growths first, gave the same error on fifteen and took Kerry Park point 85 from 0.34° to 4.23°: its second view joined from the starting plane with no search where this order joins it from the shared plane by the search. That order also took slightly longer in total, 7.9 s against 7.6 s, before the rule that a search landing on the implied shift is not grown again. Both measurements were made with the joint fit the step then had.

**The robust combination.** Every view estimate, and every pair estimate not already refused under its floor, is in the first mean. While some kept estimate lies further than `max_agreement_distance` (3) from the mean of the kept ones, by the Mahalanobis distance with the estimate's covariance and the mean's added as in § [When a view agrees with the shared plane](#when-a-view-agrees-with-the-shared-plane), the furthest is left out (`TooFar`), with any pair that holds a view left out (`ViewLeftOut`), and the mean is read again. Then each kept view is scored at the mean, at its shift moved there (below), one render each; a view under `min_held_zncc` is left out (`HeldUnderFloor`) with its pairs, and the whole is repeated until no view is left out. The furthest goes first and one at a time so that the result does not depend on the order the estimates are given in, and so that at least one estimate is kept, since an estimate alone is at no distance from its own mean. The distance is read against a mean the estimate is part of, which shortens it; this is the same test the sequence applies when a view joins, with the shared plane then holding the views before it only.

**Moving a shift to another plane.** A view's shift was fitted at its own plane `tₑ`, and the plane it ends at is `t`. With `N` the normal matrix of its whole-grid residual at its own plane and shift over `[a, b, sx, sy]`, the shift the Gauss–Newton model fits at `t` is

```text
s = sₑ − N_ss⁻¹ N_st (t − tₑ)
```

which costs no render. The held score at a mean, the score the report gives at the final plane, and the keypoint the step writes are all read at the shift moved this way; nothing is fitted again.

### Wide pairs

**Why.** How much a view says about the tilt grows with its angle from the view it is compared against: the further apart two views are, the more a change of tilt changes one view's picture of the patch against the other's. Every growth compares a view with the reference. When the reference sits among the cameras rather than at one end, two views on opposite sides of it are further apart from each other than either is from the reference, so comparing them with each other is more sensitive to the plane than either comparison with the reference.

**Which pairs.** Among the views the robust combination kept, every pair is ranked by the angle between the two views' directions from the patch centre, widest first, ties by the two views' order among the candidates, and pairs are taken down that list so that no view is in more than one pair, up to `max_pairs` (3). Taking pairs greedily by angle is a choice: it does not ask that a pair lie on opposite sides of the reference, or that it be wider than the widest view's angle from the reference, so a pair of two views on the same side is taken where it is the widest left.

**The fit.** Both views are rendered through the plane onto the reference's working grid: each sample of the grid is followed along its reference ray to the plane and projected into each of the two photographs. The fit starts from the combined plane, holds each view's shift against the reference at its own shift moved to that plane, and fits the tilt and one relative shift `(rx, ry)` added to the second view's, for the best robust cell score of the ZNCC between the two renders, over the cells the reference is textured in (§ [The robust cell score](#the-robust-cell-score)). The pair constrains only the difference between its views' shifts, since shifting both by the same amount changes neither render against the other, which is why the common part is held at the reference comparisons' value. The pair's information about the tilt is the Schur complement of its normal matrix with the relative shift eliminated, as for a view.

**Acceptance.** A pair whose score between its two renders is under `min_held_zncc` is refused (`UnderFloor`) and never combined. The others are added to the robust combination with the views, under the same distance test, and a pair holding a view the combination leaves out is left out with it.

**Two σ.** A pair compares two photographs each of which is already compared with the reference, so its information is not independent of theirs. The precision is reported both ways: σ (a), `sigma_weak_deg` and `sigma_strong_deg`, from every kept estimate's information summed, and σ (b), `discounted_sigma_weak_deg` and `discounted_sigma_strong_deg`, with each kept pair's information halved, so that a view in a pair counts about once. The rule is the simplest that discounts the reuse; it is not derived from the noise. The plane itself is the mean with the full information of every kept estimate, (a).

**Risks.** The two views of a wide pair differ more from each other than either does from the reference: in foreshortening, so the two renders sample the photographs at more different scales; in sampling, so one render can be blurred where the other aliases; and in lighting, since the two look at the surface from further apart. Near an occlusion they can also see slightly different surface. Each of these lowers the ZNCC between the pair's renders for a reason other than the tilt, which is why a pair is held to the floor and to the distance test.

### The selection

`SelectedPairs` predicts each view's information about the tilt from geometry before it renders the view, and matches only the views the prediction picks.

**The model.** For each candidate: `θ`, its angle from the reference seen from the patch centre; `d`, the unit direction of its offset from the reference, the difference of the two viewing directions written in the starting frame's `(u, v)`; and `s`, its photograph px per world unit at the anchor, the square root of the area scale of the projection of the starting frame there, capped at the working grid's samples per world unit `ρ`, since the grid samples no finer. With `e` the direction across `d` and the geometric gain `g = (s/ρ)² · sin²θ`, the prediction is

```text
P = g · (α · d dᵀ + β · e eᵀ)
```

`α` and `β` are read from the views matched so far: each the median over them of their measured information along `d`, or across it, divided by `g`. With `β` zero this is the rank-one prediction along the offset direction. It is not rank one in practice: a single view at 40° on the tests' scene determines the tilt in both directions, to 1.47° weakest, and a rank-one model, scaled by the ratio of traces, picked 20° views across the first one's offset over the 40° views and needed five views where the two-direction model needs three.

**The picks.** The first pick, and every pick while no view has been matched, is the widest view left. Each later pick maximises the smallest eigenvalue of the measured information of the views the combination keeps plus the view's prediction, ties by the larger eigenvalue, then by the wider view, then by the earlier one. Each pick is grown against the reference as under `Independent`; a pick under the floor is left out, and the next pick replaces it. After each pick that reaches the floor the picks are combined by the robust combination, and the selection stops when the combined weakest σ is at most `target_sigma_deg` (0.5°) or `max_selected_views` (6) picks have reached the floor. A view left out by the combination is grown once more from the search, as under `Independent`, and the wide pairs are taken among the picks kept.

**The views not matched.** Each candidate the selection did not pick has its shift alone fitted at the final plane: stage 1 reads the first window at its keypoint-implied shift, and the whole-grid stage fits the shift with the tilt held. Where that is under the floor, or the view has no keypoint, the stage-1 search runs at the final plane and the shift is fitted from its best, unless it lands on the implied shift. The view is reported `ShiftOnly`; its keypoint is written when its score reaches `min_held_zncc` and is on the photograph. It is not in the plane.

### The keypoints the step writes

The step solves for each joined view's shift, and the shift says where that view's photograph places the patch, so the step writes it back to the track rather than discarding it. The keypoint is the track-stage measurement `TrackMeasurement::keypoint` ([bench/track.rs](../../crates/sfmtool-core/src/bench/track.rs)): the pixel a commit writes and the place every reading of the row is anchored at, which the fit sets from its kernels and `sight_observation` sets by hand.

- **A joined view.** Its keypoint is set to the pixel where the final plane and the view's shift put the anchor, the shift its growth reached moved to the final plane by its own normal matrix (§ [Combining the views' planes](#combining-the-views-planes)). Every plane the step tries passes through the anchor, so that pixel is the anchor's projection through the camera's lens model plus the view's shift: the pixel the view's score at the final plane is read around. A pixel off the photograph, outside `[0, width) × [0, height)`, is not written, and the report's `keypoint` is `None`.
- **A shift-only view** (`SelectedPairs`). Its keypoint is set the same way from the shift fitted at the final plane, when that fit reaches `min_held_zncc`.
- **The reference.** It keeps the keypoint it had before the step. The anchor lies on the reference keypoint's ray, so it projects back to that keypoint, and the reference's shift is zero by construction: the reference tile is the template every other view is compared against, sampled at the plane's own positions, so it has no shift to fit. The step writes that keypoint back after the turn, because the turn moves it. The reference-view rule can pick a row with no keypoint, which it renders through the patch itself; its ray is then the one through the centre's projection, the anchor is the centre, and the step writes the centre's projection.
- **Every other row**: a view left out, a shift-only view under the floor, one never attempted, and every row not `in`. The step writes nothing for them. They keep the keypoint the turn gives them, as under every other normal step: `tilt_patch` keeps each sighting's offset from the centre in the patch's plane, rebuilds that point on the turned plane and projects it (`place_keypoints` in [bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs)).

**Readings at the old keypoint.** Every other field of a row's `TrackMeasurement` was read at its keypoint and says nothing about a new one. The precedent is the two places that write keypoints by geometry or by hand: `place_keypoints`, which `tilt_patch`, the translations and a resize from an edge run, and `sight_observation` each replace the row's measurement with one that carries only the new keypoint, and `sight_observation`'s doc gives the reason, that an evaluation recomputes everything else from the track as it stands. The step does the same for each row it writes, so `walked_px` and `walked_to`, which only a fit sets, go with the rest. The turn has already done this for every row and dropped the bitmap, and the readback that ends the step renders a new bitmap from the reference and reads every row again at the keypoint it now has, so no reading is left that a written keypoint makes stale. The write changes no verdict and pins nothing, as a fit pins nothing: a pixel the step found is not one a person ruled on.

**The centre does not move.** The step turns the patch about its centre, as the other normal steps do. A depth error along the reference ray shows up as the per-view shifts, up to about 8 image px on the example below, so the written keypoints can imply a different position than the patch's centre; the readback's `projection_offset_px` then shows, for each written row, how far its keypoint sits from where the centre projects. Whether the step should also re-triangulate the centre from the written keypoints, or leave that to the next fit, is an open question.

How far the keypoints move on real tracks was not measured.

### The plane's precision

A view's information about the tilt is read from the normal matrix of its whole-grid residual at the plane and shift it is read at: `N = JᵀJ` over `[a, b, sx, sy]`, with the shift eliminated by the Schur complement,

```text
I = N_tt − N_ts · N_ss⁻¹ · N_st
```

which is the information about the tilt with the shift fitted again at every tilt, so that a shift cannot stand in for a tilt. Its negative eigenvalues, which only rounding can make, are set to 0.

**The scaling.** The residual is scaled so that its sum of squares is `S = 2 · (1 − score)` (§ [The optimiser](#the-optimiser)). Near the fitted plane the Gauss–Newton model gives `S(δ) ≈ S₀ + δᵀ I δ` for a tilt change `δ` with the shift refitted, so the score falls by `½ · δᵀ I δ`: `I` is minus the Hessian of the score in the tilt, the quantity the prototype estimated by finite differences of the ZNCC. A ZNCC drop of `τ` is therefore a residual rise of `ΔS = 2τ`, reached on the ellipse `δᵀ I δ = 2τ`. The ZNCC tolerance τ (`zncc_tolerance`) is the ZNCC difference noise can produce; the prototype's 0.01 is a placeholder. A tilt change inside that ellipse cannot be told from noise, which makes the covariance of the tilt `2τ · I⁻¹`, regularised as `2τ · (I + 10⁻⁶)⁻¹` so that a direction with no information gets a large variance rather than none. σ is the plane's uncertainty along its weakest direction, `√(2τ/λ_min)` with `λ_min` the smaller eigenvalue of `I`, in degrees, since a small tilt `(a, b)` is that many radians; the strongest direction's σ comes from the larger eigenvalue in the same way. Since σ grows as `√τ`, the choice of τ scales every σ and every agreement distance by the same factor.

**Which information.** Every growth's information is read when it is judged, at its own plane and shift, the first view's in sequence too, and a pair's at the plane and relative shift its fit ended at. The combined plane's information is the sum of the kept estimates' information as each was read, and the precision the report gives is that sum, σ (a), and the same with each pair's halved, σ (b) (§ [Wide pairs](#wide-pairs)). Nothing is read again at the final plane.

### When a view agrees with the shared plane

A view other than the first joins when both hold:

- its own plane, the one it reached growing from its start, lies within `max_agreement_distance` (3) of the shared plane by the Mahalanobis distance `√(dᵀ C⁻¹ d)`, with `d` the difference of the two tilts and `C` the view's covariance and the shared plane's added (`TooFar` otherwise);
- held to the shared plane, with its shift fitted again from its own, its robust grid score is at least `min_held_zncc` (0.5) (`HeldUnderFloor` otherwise).

This is the test of `SequentialMean` and `SequentialPairs` as each view joins. The first view has no shared plane to agree with and joins on the 0.5 floor alone, on its own robust grid score (`UnderFloor` otherwise); its plane becomes the shared plane. The robust combination applies the same distance and floor to a set of estimates afterwards, the held score there read at the shift moved to the mean rather than fitted again. The rule replaces convergence bars on the view's own readings (a whole-grid ZNCC of at least 0.7, a window ZNCC never more than 0.1 below its best earlier stage, a stage-1 match not on the edge of its search), which left out views that ended near the stored normal and, on one of the second round's points, let no view join at all. A fixed bar on how much ZNCC a view loses when held to the shared plane was tried in between, and let in a view whose own plane was 26° from the shared plane; scaling the test by precision is what refuses it (see [Prototype on four points](#prototype-on-four-points-2026-10-16)).

Every ZNCC is the mean over the colour channels the reference is not flat in (spread at least 0.5 grey levels), over the samples that land on the photograph in both renders, and is read only when at least `min_coverage` (60%) of the window's or cell's samples do; under it the ZNCC is `−1`. A fourth channel, the photograph's alpha, is not read.

### Turn and report

The step turns the patch to the final plane whenever at least one view joined, and reports σ along the weakest and strongest directions as its confidence, both ways (§ [Wide pairs](#wide-pairs)), with a verdict of low confidence when the weakest σ (b) is over `low_confidence_sigma_deg` (the prototype marked σ over 2°). σ (b) is the one judged because it is the larger; with no pair kept the two are the same. It leaves the normal unchanged only when no view joined. Precision is still used inside the sequence, in the agreement rule above; it does not decide whether the plane is kept.

This differs from `grid_plane_normal`, which leaves the normal unchanged when it has no good estimate. The reason is that a rejected plane can still be much better than the start: on two of the four points of the second round, a 2° gate on σ left the patch at its start, 8.5° and 80.1° from the stored normal, where the planes it rejected were 1.36° and 16.8°. A low-precision plane says the views constrain the tilt little near it, not that the start is closer; the local curvature of the ZNCC understates what narrow views can correct from far away, as one narrow view took bull point 236 from 80° to 5.6°. A consumer that needs a precise normal reads σ and chooses its own bar.

### A proposed ordering rule

Not prototyped and not built for the sequence; `SelectedPairs` orders its picks by predicted information instead (§ [The selection](#the-selection)). The angle from the reference is a proxy for how far the current plane's prediction is from the truth in a view. The plane's uncertainty, carried into each remaining view, predicts that error directly: the map error over a cell, given the plane's covariance (§ [The plane's precision](#the-planes-precision)). The proposal is to take next the view the current plane is predicted to get closest, and to defer a view whose predicted squeeze is past about 1.7×. That bar sits below what was captured on the example: from a single view, image 62, whose start needed a stretch of about 2×, was captured, and images 65 and 66, whose starts were squeezed by about 3× and 4×, were not. Dino point 528 is a case for ordering by information as well: two nearly uninformative views joined first and let the shared plane drift 9° before the views that see the tilt were tried (§ [The robust cell score](#the-robust-cell-score)).

### Relation to `grid_plane_normal`

`grid_plane_normal` reads the plane from where nine cells register, by shift alone, on a render made at the current plane, and gates each cell in each view by its ZNCC, its curvature, the bound and the view's robust affine fit; it can only measure what lands inside the shift bound. This step instead treats the plane as the thing being fitted, and every view's picture as evidence for it. The two share the reference rule, the working grid's size rule, the starting frame and the final turn and readback. They differ on a weak estimate: `grid_plane_normal` leaves the normal unchanged when it has no good estimate, and this step turns to any plane at least one view joined and reports its σ. Where `grid_plane_normal` ends `Turned`, the two should agree; the cases this step is for are the wide views `grid_plane_normal`'s gates refuse at the starting plane.

### Relation to the prototype

The step follows the prototype run with `SPOT_RULE=anchor JUDGE=precision KP_START=1 ROBUST=1`, the runs the maintainer approved, and differs from it in these ways:

- **The optimiser.** The prototype's growing windows, whole-grid stage, held score and joint refinement used a pattern search; the step uses Levenberg–Marquardt (§ [The optimiser](#the-optimiser)).
- **No joint refinement.** The prototype refined the shared tilt over every joined view after each join, moving the tilt alone and fitting each view's shift again afterwards. The step fits no views together: the shared plane is the information-weighted mean of the views' own planes (§ [Combining the views' planes](#combining-the-views-planes)).
- **The information.** The prototype estimated each view's information by a centred second difference of the whole-grid score, a step of 0.03 tilt units along each axis and one mixed sample, with the shift fitted again at every sample; the step reads it from the normal matrix with the shift eliminated, which is the same quantity to the Gauss–Newton approximation and is positive semi-definite by construction.
- **The render.** The prototype sampled the full-resolution photograph bilinearly; the step samples one pyramid level per view, chosen by the bilinear-mip rule at the starting plane (§ [One plane shared by all views](#one-plane-shared-by-all-views)), so a view seen at more than about 1.4 photograph px per grid px is read from a coarser level. Its reference tile is its own render at no tilt, where the prototype took `grid_plane_normal`'s byte-valued reference render; the two have the same geometry.
- **The widened search's centre.** The prototype centred the widened ±12 px search on zero shift, the anchor's own projection; the step centres it on the keypoint-implied shift, as it does the ±3 px search, so a view whose keypoint is a few px off the content and whose pose is more than 12 px off still finds its shift. A view with no keypoint searches about zero shift in both.
- **The growth with no search.** The prototype always started stage 1 with the search about the keypoint-implied shift. The step first grows each view from that shift with no search, and searches only when that growth is refused (§ [Near to far within a view](#near-to-far-within-a-view)).
- **The working grid.** The prototype took the working grid's side from the `grid_plane_normal` record it was given, 48 on the dino points; the step applies the same rule itself, or takes `working_resolution`.

## Implementation notes

- **Cost.** What dominates is one render of the window per optimiser evaluation per view: a Jacobian costs one render per parameter, four within a view, and an iteration one more for each damping it tries. Each view keeps the projection of its last tilt, so the renders at one tilt and several shifts (the shift columns of the Jacobian and the stage-1 search) project the grid once and only sample again. On the tests' synthetic scenes, 128-px photographs and a 32-sample grid, a whole step takes well under a tenth of a second in an optimised build. With no joint fit the cost is the growths: about linear in the number of views matched, with one render per kept view for each held check of the robust combination, a few whole-grid renders per iteration of each pair fit, and one stage-1 window and one whole-grid fit with the tilt held per shift-only view. On `dino_dog_toy` points 0 and 810 (34 and 17 views, grids of 28 and 48) `SequentialMean` takes 0.40 and 0.63 s where the step with the joint fit took 1.8 and 1.9 s, and `SelectedPairs` 0.14 and 0.51 s (§ [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17)).
- **Phases.** The step opens the overview phase `grown plane normal` over everything after its settings are checked, and under it a `grown plane view` per try, noted with the image and the try's outcome. A detailed `Progress` adds, under the step, `setup` (the reference, the working grid, the views' pyramid levels and the strategy; the reference-view rule's phases nest in it), a `mean` per robust combination (noted with the views and pairs kept, those held under the floor, the means read, `σ` and the renders), a `selection` per pick (noted with the view, its angle, its scale and its predicted weakest information), `pairs` with a `pair fit` per pair (noted with the two images, their angle, the pair's ZNCC, `σ`, verdict, steps taken and renders), a `shift-only fit` per view not matched (noted with how its first window found its shift, its ZNCC and whether it is written), and `turn and write` (the final shifts and scores, the keypoint writes and `read_back`, whose phases nest in it, noted with the keypoints written); under each try, a `growth` per growth, noted with its plane (shared or starting), how its first window found its shift, its first-window and grid ZNCC, whether it was taken or why it was refused, and its renders, and in sequence a `mean` after a join, noted with the views and `σ`; and under each growth a `keypoint search` or `widened search` where one ran (noted with its bound, its centre, the best ZNCC, the best shift's offset and the shifts tried), a `window stage` per later window and a `whole-grid stage` (each noted with its ZNCC, steps taken and renders), and `judge` (the growth's own information and, in sequence, its held fit, noted with the distance, held ZNCC, `σ` and verdict). A growth whose search lands on the implied shift is noted so and holds no stages. A view counts its renders in a `Cell` beside its projection cache; the notes are built only when a phase records, and nothing a phase does reaches the result, so the step's track and report are the same with detail on and off.
- **Single-threaded.** The step renders one view at a time on the calling thread, as `grid_plane_normal` does; nothing in it is parallel.
- A view's keypoint is read once, for the keypoint-implied shift that its first growth starts from and its stage-1 search is centred on. A view with no keypoint, or whose anchor does not project into it, is grown only from a search about zero shift and is reported `NoKeypoint`.
- **Cost of the order.** A refused growth with no search costs a whole growth before the searched one; the search itself is at most 2401 renders of the 7×7 window, small against a growth. Where the keypoints are poor every view pays for both growths: on Kerry Park point 242 every view's window scores −0.47 to 0.06 at its implied shift (§ [Measurement on sixteen points](#measurement-on-sixteen-points-2026-10-17)).
- The write sits between the turn and the readback, in the closure `read_back` takes. The anchor and each written view's shift at the final plane are kept for it.
- A track at infinity is refused by `normal_preconditions`, as for the other steps: its patch faces along its own bearing and has no plane to fit. The step's geometry assumes a finite anchor and finite reference rays throughout.
- A ray that meets the plane behind the reference camera, or not at all, and a point a camera does not project, give a sample off the photograph, as a sample past the image's edge does.

## Determinism and precision

Photograph samples are `f32`, read bilinearly from `u8` pyramid levels; the ZNCC sums, the residuals, the tilt, the shifts and the plane are `f64`. The views are taken in a fixed order, ties in angle broken by observation index, every search and fit is deterministic, and nothing runs in parallel, so the output is the same on every run. No content hash depends on it.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `strategy` | `SequentialPairs` | How the views' planes are combined: `SequentialMean`, `SequentialPairs`, `Independent`, `IndependentPairs` or `SelectedPairs` |
| `working_resolution` | `None` | Side of the working grid, samples; `None` follows the reference's footprint, clamped to `R ..= 2R`, as `grid_plane_normal` does; at least `spot_window` |
| `spot_window` | `7` | Side of the first window, grid px, centred on the grid's centre sample |
| `window_fractions` | `[1/8, 1/5, 1/3, 1/2]` | Half-widths of the later windows as fractions of the grid's side, rounded, any no wider than the first window's dropped: 6, 10, 16, 24 on a 48 grid; 4, 6, 11, 16 on a 32 grid; 5, 8, 12 on a 24 grid |
| `keypoint_search_px` | `3.0` | Bound of the first window's shift search about the keypoint-implied shift, run when the growth from that shift is refused, image px |
| `min_keypoint_zncc` | `0.7` | Best stage-1 ZNCC within that bound below which the search widens |
| `spot_search_px` | `12.0` | Bound of the widened search about the keypoint-implied shift, and of a keypoint-less view's search about zero shift, image px |
| `spot_search_step_px` | `0.5` | Step of both searches, image px |
| `zncc_tolerance` | `0.01` | ZNCC difference noise can produce, τ; a placeholder |
| `max_agreement_distance` | `3.0` | Largest Mahalanobis distance of an estimate's plane from the shared or combined plane |
| `min_held_zncc` | `0.5` | Robust grid score a view needs when held to the shared or final plane, and on its own where it is judged alone; a pair's score between its two views |
| `low_confidence_sigma_deg` | `2.0` | σ (b) above which the report says low confidence; it does not stop the turn |
| `max_pairs` | `3` | Pairs of views matched against each other, at most |
| `target_sigma_deg` | `0.5` | `SelectedPairs` stops matching once the combined weakest σ is at most this, degrees |
| `max_selected_views` | `6` | `SelectedPairs` stops matching once this many picks have reached the floor |
| `min_coverage` | `0.6` | Share of a window's or cell's samples on the photograph for its ZNCC to be read |
| `max_iterations` | `20` | Levenberg–Marquardt iterations per stage, at most |
| `initial_damping` | `1e-3` | The damping each fit starts from |
| `min_improvement` | `1e-6` | A stage stops once an iteration raises its score by less |
| `jacobian_tilt_step` | `1e-3` | Forward-difference step along each tilt parameter, tilt units |
| `jacobian_shift_step_px` | `0.05` | Forward-difference step along each shift, image px |

The search and agreement defaults are the prototype's, set on one point in the first round and four in the second. The five optimiser settings are this implementation's and were chosen on the tests' synthetic scenes only. `strategy` defaults to `SequentialPairs` on the sixteen points of § [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17); `max_pairs`, `target_sigma_deg` and `max_selected_views` are first values, with two other settings of the last two measured there.

## Testing

The tests in [bench/tests/grown_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs) build synthetic planes seen by 128-px pinholes that look at the point from around it, 4 units away, through `Scene::looking_at`, with the patch spanning 32 px in a facing view; [grown_plane/tests.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/tests.rs) tests the parts that need no photograph. Every test runs the default strategy, `SequentialPairs`, unless it names another. Built and passing:

- A textured plane started 25° off its normal, seen from six views out to 50°, under every strategy: the normal ends within 0.03° of the plane, every joined (or shift-only) view's written keypoint is within 0.5 px of the point's projection (within 0.01 px here), the reference keeps its keypoint, and 6 keypoints are written. Under the default every view joins on its first growth, from its keypoint-implied shift with no search, both pairs are kept, and σ is 0.56° weakest and 0.48° strongest, 0.67° with each pair's information halved; with no pairs it is 0.88° and 0.77°.
- One view at 25° whose top cells see black and white blocks the reference never sees there: it joins, and the normal ends 0.01° from the plane. At the final plane its robust score is 0.997 and its ZNCC over the whole grid 0.19.
- A textured plane seen only from views within 3° of the reference, started 20° off: the patch is turned, here to 0.3° from the plane, with σ 9.6° weakest (10.3° with the pair halved), and the report says low confidence.
- A wrong join: one view at 38° sees the texture on a plane turned 25° from the one the others see. It is left out as `TooFar`, 8.7σ from the shared plane in sequence (its held score is 0.49) and 8.4σ from the combined plane under `IndependentPairs`, no kept pair holds it, and the normal ends within 0.09° of the others' plane. In sequence its searches land on its keypoint-implied shift, so each try makes two growths, both with no search, from the shared and from the starting plane, and it is deferred once and then left out. Matched alone it passes the floor, the combination leaves it out, and the search that follows lands on the implied shift, so it stays out.
- Repeating texture under the spot: a checker of 8 px period, with three views' poses turned so that the photograph's content sits about 6.5 px from where the pose projects the point and each keypoint on the content. Every view joins on its first growth, `NoSearch`, and is written within 0.75 px of the content, and the same holds with `keypoint_search_px` 0, under which the search, had it run, would widen at once to the ±12 px about the keypoint-implied shift and take another peak. With the views' keypoints removed, so that each is grown from the ±12 px search alone, every view joins on another peak of the checker, written more than 4 px from the content.
- Keypoints off the content in views whose poses are further off: three views' poses turned so that the content sits about 16 px from where the pose projects the point, past the ±12 px a search about zero shift reaches, and each keypoint 6 px off the content, past the ±3 px search about it and past what the growing windows recover. Each view's growth with no search is refused; the widened search follows, and every view joins `Widened` and is written within 0.75 px of the content.
- Wide pairs: four views each 20° from a reference that faces the plane, on opposite sides of it, so that the two pairs of opposite views are 40° apart. With the views matched alone or in sequence, both pairs are taken and kept, and the weakest σ falls from 1.62° to 0.95°, or 1.16° with each pair's information halved; the plane ends within 0.02° of the truth.
- The selection: sixteen views in rings at 10° to 40° about a reference that faces the plane, with a target of 1°. The first pick is a 40° view, the second one not along its offset, and the selection stops after three picks, at a weakest σ of 0.85°, and so matches 3 of the 16 views; every other view is fitted for its shift alone and written within 0.5 px of the point's projection, and the plane ends within a degree of the truth.
- A flat wall: no view joins, the normal and every keypoint are left as given, no precision is reported, and the sentence says the normal was left unchanged.
- A bearing is refused as `AtInfinity`, a search step of 0 as `BadGrownPlaneOption`, and a cancelled `Progress` as `Cancelled`.
- On the scene of the wrong join with one view's keypoint removed, under every strategy, the track and report are the same, compared by their `Debug` text, with no sink, with an overview sink and with a detailed one; the overview records the step and one phase per try and nothing finer, and the detailed run records a `growth`, a `judge` and a `whole-grid stage` per growth, a `window stage` per later window of each, a `mean` per join in sequence and per combination otherwise, a `pairs` with a `pair fit` per pair under the strategies with pairs, a `selection` per pick and a `shift-only fit` per view not matched under `SelectedPairs`, no `joint fit` and no `precision`, and one `setup` and `turn and write`, each noted, at the depths above, with at least one `judge` or `mean` noting a refusal.
- Without photographs: the robust score on dino 528's cell readings, the residual identity `‖â − b̂‖² = 2 − 2·ZNCC` with weights, the coverage bar, a flat channel left unscored, the window schedule on 24, 32 and 48 grids, the Schur complement against the profile it stands for, σ from the eigenvalues, the fit reaching the optimum of a quadratic, and the weighted mean, its covariance and the agreement distance against a hand-computed case.

Not built, from the cases this draft proposed: the depth-error scene (a patch centre off along the reference ray, whose shifts the views absorb); a wide view squeezed past capture from the starting plane that joins from the shared plane; σ against the spread of the fitted normal under added noise; a pair that is refused under its floor. A depth error large against the distance to the point changes each view's scale as well as its shift (on the tests' scene, half a unit at 4 units fitted a plane 7° off), so the depth-error scene needs a longer focal length than the tests' 160 px to keep the error small while the shift is several px.

Then the measurement on every point of `seoul_bull_sculpture`, `kerry_park` and the `dino_dog_toy` candidate, against `grid_plane_normal` and today's Grid Plane Normal at 2×2 and 3×3, from the same fitted mean-view start as [piece-gated-grid-normal.md](piece-gated-grid-normal.md)'s sweep, and a hand review in SfM Explorer. The scripts run it through the binding with `--grown`; the sweep has not been run.

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

**The starting spot.** The maintainer suspected the single-view failures came from the spot the contrast rule picked. Single views were run again from five candidate spots: the four non-overlapping 7×7 windows of the middle cell, ranked by contrast, grid (20, 21), (29, 19), (18, 31) and (27, 31), and the geometric centre (24, 24).

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

## Measurement on sixteen points (2026-10-17)

**Question.** Does growing each view from its keypoint-implied shift before any search join the views a wrong stage-1 peak kept out, and at what cost?

**Setup.** The maintainer's sixteen hand-picked points, each started from the mean viewing direction, run through the binding by `grid_plane_detail.py --grown --start mean_view`: `dino_dog_toy` 0, 1, 528, 672, 810 and 933 (the candidate ground truth v006), `seoul_bull_sculpture` 120, 50, 236 and 32, and `kerry_park` 85, 200, 228, 342, 91 and 242. Errors are to each file's stored normal, an estimate as before. "Search first" is the step with stage 1 always searched; "no search first" is the step with the growth from the implied shift first. Both still had the joint fit after each join that § [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17) removes. Times are each step's wall time in one optimised build on one machine, and vary by a few percent between runs.

| Point | Start | 2×2 | Search first | No search first | σ weakest | Confidence | Joined | ms |
|-------|-------|-----|--------------|-----------------|-----------|------------|--------|----|
| dino 0 | 3.34° | 0.36° | 1.42° | 1.34° | 1.02° | ok | 34/34 | 1943 → 1838 |
| dino 1 | 59.39° | 19.60° | 0.46° | 0.48° | 10.21° | low | 5/6 | 899 → 1191 |
| dino 528 | 37.97° | 38.37° | 5.36° | 1.49° | 1.03° | ok | 4/4 | 313 → 304 |
| dino 672 | 64.95° | 40.20° | 2.02° | 2.02° | 3.41° | low | 7/7 | 536 → 541 |
| dino 810 | 22.77° | 20.26° | 0.56° | 0.56° | 0.35° | ok | 17/17 | 1907 → 1923 |
| dino 933 | 8.53° | 1.16° | 8.53° | 2.68° | 4.27° | low | 3/3 | 229 → 103 |
| bull 120 | 47.46° | 28.13° | 3.94° | 3.94° | 1.22° | ok | 3/3 | 87 → 92 |
| bull 50 | 29.09° | 23.77° | 25.77° | 25.69° | 4.92° | low | 3/3 | 70 → 76 |
| bull 236 | 80.07° | 32.19° | 9.36° | 7.10° | 37.14° | low | 2/2 | 71 → 60 |
| bull 32 | 40.78° | 29.68° | 1.29° | 1.30° | 1.81° | ok | 7/7 | 130 → 111 |
| Kerry Park 85 | 54.23° | 49.78° | 0.34° | 0.34° | 2.88° | low | 2/2 | 33 → 56 |
| Kerry Park 200 | 48.31° | 44.74° | 3.24° | 5.91° | 1.90° | ok | 4/4 | 165 → 55 |
| Kerry Park 228 | 26.84° | 24.39° | 2.02° | 2.02° | 0.39° | ok | 5/9 | 503 → 773 |
| Kerry Park 342 | 45.73° | 45.64° | 48.03° | 43.85° | 6.12° | low | 2/3 | 158 → 147 |
| Kerry Park 91 | 32.39° | 25.64° | 1.49° | 1.49° | 0.48° | ok | 3/3 | 62 → 58 |
| Kerry Park 242 | 60.97° | 62.23° | 3.94° | 3.94° | 0.96° | ok | 5/7 | 203 → 377 |

σ, confidence and the joined count are the step with no search first. The median error fell from 2.63° to 2.02°, and the step beat the 2×2 Grid Plane Normal on 13 points where it had on 12. The total time rose from 7.3 s to 7.7 s.

**Where it helped.** On dino 933 no view joined with the search first; grown from the implied shifts all three join, on their first growth, and the plane ends 2.68° off with σ 4.27°, still behind the 2×2 step's 1.16°. On dino 528 the fourth view joins and the error falls from 5.36° to 1.49°. Kerry Park 200 joins all four views where two joined before.

**Where it did worse.** Kerry Park 200 ends 5.91° off where it ended 3.24°: its nearest view, at 24°, now joins first with its own plane 6.57° off, and the three wider views, whose own planes are 1.5° to 4.0° off, agree with it within the agreement distance and leave the shared plane at 5.91°. With the search first that view was refused and the 51° view set the plane. Dino 1 and bull 32 differ by 0.02° or less.

**Cost.** A view whose growth with no search is refused pays for a second growth. On Kerry Park 242 every view's window scores −0.47 to 0.06 at its implied shift, every unsearched growth is refused, and the step takes 1.9 times as long; Kerry Park 228 grows 41 times where it grew 23 and takes 1.5 times as long. Where the implied shift is good, as on dino 933 and Kerry Park 200, the step skips the search and the growths that a wrong peak led to, and takes a third to a half as long.

## Measurement without the joint fit (2026-10-17)

**Question.** Without a fit of the views together, which way of combining the views' planes comes closest to the stored normals, and at what cost?

**Setup.** The sixteen points of § [Measurement on sixteen points](#measurement-on-sixteen-points-2026-10-17), from the mean viewing direction, by `grid_plane_detail.py --grown --start mean_view --grown-opt strategy=…`, once per strategy at the defaults, in one optimised build on one machine. "Joint fit" is the step of that section, growing from the implied shift first, with the joint fit after each join. Errors are to each file's stored normal, an estimate as before. σ is the weakest direction's; (a) sums every kept estimate's information and (b) halves each pair's.

| Point | Start | 2×2 | joint fit | `sequential_mean` | `sequential_pairs` | `independent` | `independent_pairs` | `selected_pairs` |
|---|---|---|---|---|---|---|---|---|
| dino 0 | 3.34° | 0.36° | 1.34° | 0.67° | 0.63° | 0.62° | 0.57° | 1.24° |
| dino 1 | 59.39° | 19.60° | 0.48° | 5.44° | 4.76° | 5.22° | 4.62° | 4.62° |
| dino 528 | 37.97° | 38.37° | 1.49° | 2.20° | 2.40° | 2.20° | 2.40° | 2.40° |
| dino 672 | 64.95° | 40.20° | 2.02° | 2.08° | 1.89° | 2.08° | 1.89° | 1.90° |
| dino 810 | 22.77° | 20.26° | 0.56° | 0.55° | 0.56° | 0.46° | 0.41° | 0.24° |
| dino 933 | 8.53° | 1.16° | 2.68° | 2.23° | 2.60° | 2.23° | 2.62° | 2.62° |
| bull 120 | 47.46° | 28.13° | 3.94° | 3.86° | 3.72° | 3.86° | 3.72° | 3.72° |
| bull 50 | 29.09° | 23.77° | 25.69° | 23.55° | 23.62° | 23.55° | 23.62° | 23.62° |
| bull 236 | 80.07° | 32.19° | 7.10° | 3.19° | 1.78° | 22.53° | 20.86° | 20.86° |
| bull 32 | 40.78° | 29.68° | 1.30° | 2.50° | 2.27° | 3.54° | 3.98° | 4.03° |
| Kerry Park 85 | 54.23° | 49.78° | 0.34° | 0.40° | 0.46° | 6.08° | 6.08° | 6.08° |
| Kerry Park 200 | 48.31° | 44.74° | 5.91° | 3.11° | 2.90° | 3.13° | 2.93° | 2.93° |
| Kerry Park 228 | 26.84° | 24.39° | 2.02° | 2.39° | 2.71° | 2.90° | 2.76° | 2.73° |
| Kerry Park 342 | 45.73° | 45.64° | 43.85° | 45.72° | 46.55° | 45.72° | 46.55° | 46.55° |
| Kerry Park 91 | 32.39° | 25.64° | 1.49° | 1.58° | 1.52° | 1.75° | 1.68° | 1.68° |
| Kerry Park 242 | 60.97° | 62.23° | 3.94° | 4.06° | 4.02° | 3.14° | 3.19° | 3.19° |
| Median error | | | 2.02° | 2.45° | 2.50° | 3.14° | 3.06° | 3.06° |
| Mean error | | | 6.51° | 6.47° | 6.40° | 8.06° | 7.99° | 8.03° |
| Beats the 2×2 | | | 13 | 13 | 13 | 13 | 13 | 13 |
| Within 2° | | | 7 | 4 | 6 | 3 | 4 | 4 |
| Within 5° | | | 12 | 13 | 14 | 11 | 12 | 12 |
| Total time | | | 7.7 s | 4.0 s | 4.4 s | 2.8 s | 3.3 s | 2.6 s |
| Views matched (of 114) | | | 114 | 114 | 114 | 114 | 114 | 70 |
| Error over 2σ (a) | | | 7 | 7 | 8 | 6 | 8 | 8 |
| Error over 2σ (b) | | | 7 | 7 | 7 | 6 | 7 | 7 |

The 114 views are every candidate beside the reference over the sixteen points; `SelectedPairs` grows 70 of them against the reference and fits the shift alone of the other 44. With the selection's target at 1.0° and budget 4, it grows 49 views, takes 1.9 s, and ends at a median of 3.23°; at 0.3° and 10, 2.8 s and 3.06°. On dino 0 the three settings match 4, 6 and 10 of 34 views and end 1.34°, 1.24° and 0.99° off, against 0.57° with every view; on dino 810 they match 1, 7 and 11 of 17 and end 0.39°, 0.24° and 0.43°.

**What changes most.**

- **Dino 0** (34 views, 3.34° start): 1.34° with the joint fit, 0.67° with the sequential mean, 0.62° matching each view alone and 0.57° to 0.63° with pairs, against the 2×2 step's 0.36°. In sequence the step takes 0.40 s where the joint fit took 1.84 s.
- **Dino 1** (6 views within 5°, 59.4° start): 0.48° with the joint fit, 4.6° to 5.4° without. Every view's own plane is poorly determined (σ 15° to 171°) and lies 1.7° to 17.7° from the stored normal; their weighted mean is 5.4° off, and the pairs, two of them 4° and 8.6° apart, bring it to 4.6° to 4.8°. Fitted together, the six views reached 0.48°. Where each view's score is this flat in the tilt, the mean of their quadratic models is a poor stand-in for their summed score. Image 16, 17.6° off, is left out under every strategy, about 6σ from the shared plane in sequence.
- **Dino 528** (4 views): 1.49° with the joint fit, 2.20° with the mean, 2.40° with pairs. Image 24, 5° from the reference, has its own plane 8.1° off with σ 8.6°, and the shared plane after it is 7.4° off with the joint fit and 8.0° with the mean; after the 40° view joins it is 1.45° off with the joint fit and 2.34° with the mean. The pair of images 25 and 45, 40° apart, ends 3.86° off on its own.
- **Dino 933** (3 views within 10°): 2.68° with the joint fit, 2.23° with the mean, 2.60° with the pair, against the 2×2 step's 1.16°. Image 78 alone is 0.47° off; the narrow views' σ of 5.6° to 19° keeps the mean between them.
- **Kerry Park 200** (4 views, 24° to 64°): 5.91° with the joint fit, 3.11° with the mean, 2.90° with pairs. The nearest view, 24° from the reference, joins first with its own plane 6.57° off; with the joint fit the shared plane stays at 5.91° as the three wider views join, and with the mean it moves to 5.2°, 4.8° and 3.11° as they join, the last the 64° view, 1.53° off with σ 3.6°. Of the two pairs, images 9 and 10, 59° apart, end 3.1° off on their own and images 4 and 29, 28° apart, 1.4°.
- **Bull 50** (3 views within 9°): about 24° for every strategy, 25.69° with the joint fit. Every view's own plane is 21° to 28° from the stored normal, so no combination of them is near it; whether the views or the stored normal are wrong is not known.
- **Kerry Park 342** (3 views): 44° to 47° for every strategy. The two near views, 4° from the reference, agree on a plane 44° to 47° off, and the 42° view scores 0.2 to 0.37 at every start and is left out. As under the joint fit, the step turns the patch to a plane its views agree on and the stored normal does not.
- **Bull 236 and Kerry Park 85**, where growing every view from the starting plane fails: bull 236's second view, grown from the shared plane, ends 3.5° off, and grown from the starting plane, 80° off, ends 27.8° off and passes the floor, so `Independent` ends 22.5° off where `SequentialMean` ends 3.19° and `SequentialPairs` 1.78°. Kerry Park 85's 28° view from the starting plane ends 6.13° off at a score of 0.695, which passes the floor, so it is never searched; in sequence its growth from the shared plane is refused held under the floor and the search from the shared plane finds a plane 0.50° off, and the step ends 0.40° off against 6.08°.

**Cost.** Every strategy without the joint fit is faster than the step with it: 4.0 s and 4.4 s in sequence, 2.8 s and 3.3 s matching each view alone, and 2.6 s with the selection, against 7.7 s. The largest savings are on the long tracks: dino 0 from 1.84 s to 0.40 s and dino 810 from 1.92 s to 0.63 s in sequence. Dino 1 is the slowest point, 0.9 s in sequence, from the eight growths of its view left out.

**Calibration of σ.** The error is more than twice σ on 6 to 8 of the sixteen points under every strategy, as with the joint fit (7); halving the pairs' information takes one point back under 2σ, under both strategies with pairs. So σ at τ = 0.01 is too small on about half of these points, and the discount for the pairs' reuse of views changes that little. Two of these points are ones where the views agree with each other and not with the stored normal (bull 50, Kerry Park 342), which no σ read from the views can show.

**Reading.** Combining in sequence keeps what growing near to far gives: a wide view grown from the plane the near views reached arrives within capture, where grown from the starting plane, 30° to 80° off on these points, it can end on a wrong plane that still passes the floor. Matching each view alone against the reference loses that on bull 236 and Kerry Park 85 and gains nothing elsewhere, beyond its speed. The pairs lower σ by a quarter to a third; they lower the error on dino 672, bull 236 and Kerry Park 200 and raise it by 0.2° to 0.4° on dino 528, dino 933 and Kerry Park 228, and cost 0.4 to 0.5 s in all. On bull 236 its one pair, 1.5° apart, 6.7° off on its own and with σ 41°, moves the plane from 3.19° to 1.78°. The selection matches 70 of 114 views for about the same error as matching all of them alone, and is the fastest; on the two long tracks it matches 6 and 7 views and ends 1.24° and 0.24° off. No strategy without the joint fit recovers dino 1, and none reaches the joint fit's median; `SequentialPairs` has the lowest mean error and the most points within 5°, `SequentialMean` the lowest median.

## Non-goals

**Moving the centre.** The step turns the patch about its centre, as the other normal steps do, and does not move it. It writes the keypoints of the views that joined (§ [The keypoints the step writes](#the-keypoints-the-step-writes)); whether it should also re-triangulate the centre from them is an open question.

**Per-cell one-dimensional constraints.** A cell over stripes pins its shift across the stripes only. Using such constraints was discussed but is not part of this method.

## Open questions

- **The optimiser.** Levenberg–Marquardt on the normalised residual is built (§ [The optimiser](#the-optimiser)); the prototype's pattern search took larger first steps, up to 0.08 tilt units and 1 px, and could leave a basin a local fit stays in. Whether the built fit reaches what the pattern search reached on the prototype's points is not measured, and its five settings were chosen on synthetic scenes only.
- **A textured spot off the anchor.** Stage 1 is centred on the anchor because the keypoint-implied shift is exact only there. Where the centre lacks texture, which the maintainer expects on some points, the first window needs a spot elsewhere, and the shift at that spot depends on the plane. One proposal, not built: pin the anchor at the keypoint and search the tilt at stage 1, so that a spot `r` from the anchor is matched at the shift the tilt implies there. Whether to do that, and how a spot is then chosen, is open; the multi-spot rule and the spot candidates of the first round no longer apply under the keypoint start.
- **The window schedule.** Whether these fractions of the grid's side are the right schedule.
- **The order of the views** beyond angle from the reference; the predicted-error rule above is the candidate, and dino point 528, where two nearly uninformative views drew the shared plane 9° off before the views that see the tilt were tried, is a case for ordering by information. Not decided.
- **The agreement numbers.** Whether τ = 0.01 is too cautious: it scales every σ, so it sets how many views agree and how often the report says low confidence. Whether 3σ and the 0.5 held floor are the right bars; they were set on four points.
- **Which strategy to keep.** `SequentialPairs` is the default on the sixteen points: the lowest mean error and the most points within 5°, at 4.4 s against 7.7 s with the joint fit. `SequentialMean` has a slightly lower median and is simpler; `SelectedPairs` is the fastest and matches a third fewer views, at a median of 3.06°, because it grows every pick from the starting plane. A selection that grows each pick from the plane of the picks before it, as the sequence does, is not built. The four strategies not kept are removed with the decision.
- **Points the mean cannot combine.** On dino 1 the six views' own planes are each so poorly determined that their weighted mean is 4.6° to 5.4° off where fitting them together reached 0.48°. Whether such a point needs the views' scores summed rather than their quadratic models, for example by a search over the tilt of the summed robust scores once the views' shifts are fixed, and how the step would tell such a point from one where the mean is right.
- **Narrow-view points.** On a point whose views all lie within about 10° of the reference, the plane has low precision (933: 1.45° off, σ 7.1°), where the 2×2 Grid Plane Normal reaches 1.2°. Whether to hand such points to a cell-based step.
- **A view's keypoint in a wide view.** Image 62's keypoint-implied shift differs from where its single-view growth settled by about 6 px in y; whether its keypoint is off in that 80° view or the shift at the window's spot depends on the tilt there is not known, and the widened search is what covers it. Under the anchor-centred window the second cause no longer applies to stage 1.
- **Blur matching the other way.** A view finer than the reference is read from the pyramid level the bilinear-mip rule picks at the starting plane, once per view. Whether one level chosen at the start is enough for a view whose footprint changes much as the plane turns, and whether the reference should instead be blurred to the view's sampling, is open.
- **What the step reports.** σ along the weakest and strongest directions and the low-confidence verdict are decided; which of the two σ the verdict reads is not. The rest of `GrownPlaneSequence` is what the step knows and records today, and is open to change once a person reading it through the binding says what is missing or unused.
- **The pairs' rules.** Whether the widest pairs, each view in one pair, up to three, are the right pairs, or whether a pair should lie on opposite sides of the reference or be wider than the widest view's angle from the reference; whether halving a pair's information is the right discount for its reuse of views, and whether the plane should be the mean with that discount rather than with the full information.
- **The pairs' grid.** Whether a pair is compared on the reference's working grid, as built, or on a grid of its own, for example one centred on the bisector of the two views' rays, where both renders are foreshortened alike.
- **The selection's model.** The predicted information is `g·(α d dᵀ + β e eᵀ)` with `α` and `β` read from the views matched so far; whether a model that reads the view's texture as well as its geometry, or the reference's information along each direction, picks better, and what the target σ and the budget should be once τ is settled.
- **The calibration of σ.** The error is over twice σ on about half of the sixteen points under every strategy. Whether τ should be raised, read from the data, or σ reported with a floor, and how the pairs' reuse of views should enter it.
- **The centre after the write.** A depth error shows up as the per-view shifts, so the written keypoints can imply a different centre than the patch's. Whether the step should re-triangulate the centre from the written keypoints before the readback, or leave it to the next fit, is not decided. The step's plane passes through the anchor and the turned patch's through its centre, so the two planes also differ in depth wherever the anchor is not the centre.
- **A bound on the write.** A fit keeps a sighting at its seed when its kernels would move it further than `max_shift_px` on the patch's plane, and records the refusal in `walked_px` and `walked_to`. Whether this step should bound its write the same way, and what it measures the move from, is open; the size of the moves on real tracks has not been measured.
- **The full sweep** over every point of `seoul_bull_sculpture`, `kerry_park` and `dino_dog_toy` is not done; `grid_plane_eval.py --grown` runs it.
