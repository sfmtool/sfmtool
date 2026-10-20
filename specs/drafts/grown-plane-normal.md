# Grown Plane Normal

**Status:** Draft. A fourth bench normal step, `grown_plane_normal`, beside `fit_normal`, `finite_difference_normal` and `grid_plane_normal`, that fits one plane shared by every view directly to the photographs. It is a separate step and not an option of `grid_plane_normal`, because it measures nothing that step measures: it has no cell shift search on a fixed render and no triangulation.

Decided: the step's existence as a separate step; that it writes each joined view's keypoint back from the shift it solved for (§ [The keypoints the step writes](#the-keypoints-the-step-writes)); that an estimate is kept by its agreement with the combined plane, scaled by the plane's precision, rather than by convergence bars on its own ZNCC readings (§ [When an estimate agrees with the combined plane](#when-an-estimate-agrees-with-the-combined-plane)); that the step turns the patch to the combined plane whenever at least one view joined and reports the plane's precision as its confidence, rather than rejecting a plane it found, except that it keeps the starting normal when the turn is large and the evidence for it weak, a turn of at least 70° under 3 times its weakest σ (§ [Turn and report](#turn-and-report), § [Judged by eye](#judged-by-eye-2026-10-18)); that stage 1 starts each view from the shift its own keypoint implies, with its window centred on the anchor, and that each view is grown from that shift with no search first, searching only when that growth is refused (§ [Near to far within a view](#near-to-far-within-a-view)); and that every score of the whole grid is the robust combination of the grid's nine cells, with a last stage that fits the whole grid on that score (§ [The robust cell score](#the-robust-cell-score)). Decided too: that the step has no fit of the views together, and combines the planes each view reaches against the reference by their information (§ [Combining the views' planes](#combining-the-views-planes)); and that the views it grows are a selection, with wide pairs and a shift-only fit for the views not picked (§ [The selection](#the-selection)), the method kept after the sweep of § [Sweep of all ground-truth points](#sweep-of-all-ground-truth-points-2026-10-17), which removed the four other ways of combining the planes that were built beside it. The selection picks near to far in rungs of angle from the reference and grows each later pick from the plane of the views kept before it, the method kept after § [A near-to-far selection](#a-near-to-far-selection-2026-10-18), which removed the selection that picked the widest view first and grew every pick from the starting plane. Not decided: whether the optimiser built here, Levenberg–Marquardt, is kept; the window schedule; the ZNCC tolerance and the other numbers of the agreement rule; the weak-turn rule's two thresholds, which were chosen on the points they were scored on; the rules of the wide pairs and of the selection, the ladder's three angles and its ratio among them; the report's exact fields; whether it also moves the patch's centre; and how a textured spot off the anchor is used. See [Open questions](#open-questions).

Built: the step in `sfmtool-core` ([bench/normal/grown_plane.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane.rs)), with its tests on synthetic planes (§ [Testing](#testing)), and its Python binding, `sfmtool.bench.grown_plane_normal` ([bench/normal.rs](../../crates/sfmtool-py/src/bench/normal.rs)), which the evaluation scripts `grid_plane_eval.py` and `grid_plane_detail.py` in [scripts/track_at_pixel/normal_sources](../../scripts/track_at_pixel/normal_sources) run with `--grown`. Not built: a Track View entry or a wire method, so the viewer does not call it yet; the multi-spot rule; the pinned-anchor tilt search. A Python prototype, not in the repository, was run on one point of the `dino_dog_toy` candidate ground truth ([Prototype on one point](#prototype-on-one-point-2026-10-15)), and a second round of it on four points ([Prototype on four points](#prototype-on-four-points-2026-10-16)); the step follows the prototype's method within a view in the branches the maintainer approved, with the differences § [Relation to the prototype](#relation-to-the-prototype) lists.

Amends: [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal", which names the step and points here.

Related drafts: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) designs `grid_plane_normal` and records the measurements this step is compared against. [cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md) defines the predicted precision of a fitted normal. [sharper-patch-bitmap.md](sharper-patch-bitmap.md) decides how a view is scored against the reference at matched sharpness, which bears on blur matching here.

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in the world, and its normal says which way the square faces. Each photograph that sees the patch is resampled onto the square through that plane, so a wrong normal stretches and shears every view's picture of the patch against the others, blurs the patch's stored bitmap and misleads every reading taken from it afterwards.

This step estimates the normal by fitting the plane to the photographs themselves. One view is the reference. For every other view, the plane predicts where each part of the reference's picture of the patch lands in that view's photograph. The step finds the plane under which a view's picture agrees with the reference's, by ZNCC. Within a view it registers a small window at the patch's anchor first and grows the window outward, because a small window tolerates a large error in the plane; each view's growth ends at that view's own plane, with its information about the plane. The step grows only a few views: it picks them one at a time by what their geometry predicts they say about the plane, until the plane is precise enough. It combines the picked views' planes by a robust information-weighted mean, never by fitting the views together, then matches the widest pairs of kept views against each other and adds their planes to the mean. The step turns the patch to the plane it found whenever at least one view joined, unless the turn is large and the evidence for it weak, when it keeps the starting normal; it reports how precisely the estimates determine that plane, writes each joined view's keypoint at the pixel where that plane and the view's own shift put the patch, and fits the shift alone of every view it did not pick, writing its keypoint the same way.

### Why a new step

`grid_plane_normal` ([piece-gated-grid-normal.md](piece-gated-grid-normal.md)) measures the normal by parallax: it renders each view once through the current plane, registers each of nine cells by a shift search within a small bound, gates each cell in each view by a robust affine fit, triangulates the cells and fits a plane through them, then turns the patch and repeats. Each part of that chain needs the starting plane to be near enough that every cell of a view lands within the shift bound and that each cell's picture in the view differs from the reference's by little more than a shift, since a cell is registered by shift alone. In a wide view a wrong plane breaks both: the cells far from the centre are displaced past the bound, and the cells whose picture is stretched or turned against the reference's fall under the cell ZNCC bar.

This step replaces the nine-cell shift search, the per-view affine gate and the cell triangulation with a direct fit of the plane to the photographs. It reaches a wide view by growing the window from the anchor, where the plane's error displaces the picture least, so each stage starts within capture of the next (§ [Near to far within a view](#near-to-far-within-a-view)). It keeps the nine cells only as the parts a whole-grid score is combined from (§ [The robust cell score](#the-robust-cell-score)).

### Why the name

The glossary's **fit normal** / **finite-difference normal** / **grid-plane normal** entry names each bench normal step for the evidence it reads. *Grown plane normal* says the estimate is a plane and that its support grows: the window over which a view is registered grows from a few samples to the whole grid, and the set of views the plane is combined from grows one pick at a time. It pairs with *Grid Plane Normal* in Track View's list. `homography_normal` names the warp but not the schedule that is the method's point, and *homography* is not bench vocabulary. `incremental_plane_normal` says the views are added one at a time but not that each view's window grows. *Register* is not used in the name, because the glossary gives it the meaning of giving an image a pose. When the name is settled, a glossary entry is added with the step.

## Rust API

The step lives beside `grid_plane_normal` in [bench/normal/grown_plane.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane.rs), with the plane-induced render and the scores in [grown_plane/render.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/render.rs) and the optimiser in [grown_plane/solve.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/solve.rs). It has the signature every bench normal step has ([bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs)), so Track View and the wire's `fit_bench_track_normal` can call it the way they call the other three once they are given it. The weighted mean, the robust combination and the agreement distance are in [grown_plane/combine.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/combine.rs), and the selection, the wide pairs and the shift-only fits in [grown_plane/selected.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/selected.rs). The Python binding `sfmtool.bench.grown_plane_normal` ([bench/normal.rs](../../crates/sfmtool-py/src/bench/normal.rs)) calls it already: it takes every `GrownPlaneNormalOptions` setting as a keyword of the same name, defaulting to the Rust default, and `fit`'s four keywords, and gives the sequence back as the report's `estimate`, `kind` `"grown_plane"`, with one dict per view, per try, per pick and per pair; [editable-track.md](../core/bench/editable-track.md) § "Estimating the normal" lists its fields.

```rust
/// How grown_plane_normal grows its windows, fits its planes and decides
/// which views it keeps. Defaults in § Parameters.
pub struct GrownPlaneNormalOptions {
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
    pub weak_turn_deg: f64,
    pub min_turn_sigma_ratio: f64,
    pub max_pairs: usize,
    pub target_sigma_deg: f64,
    pub max_selected_views: usize,
    pub first_rung_deg: f64,
    pub first_rung_min_deg: f64,
    pub first_rung_max_deg: f64,
    pub rung_ratio: f64,
    pub min_coverage: f64,
    pub max_iterations: u32,
    pub initial_damping: f64,
    pub min_improvement: f64,
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
    pub reference: Option<usize>,          // None: no reference, nothing tried
    pub working_resolution: u32,
    pub window_half_widths: Vec<u32>,      // the stages' windows, grid px
    pub views: Vec<GrownPlaneView>,        // one per observation of the track
    pub trace: Vec<GrownPlaneTry>,         // every try, in the order made
    pub picks: Vec<GrownPlanePick>,        // the selection's picks, in order
    pub pairs: Vec<GrownPlanePair>,        // the pairs matched, widest first
    pub outcome: GrownPlaneOutcome,        // Turned, NoneJoined or WeakTurn
    pub normal: Option<[f64; 3]>,          // the final plane, under WeakTurn too; None: none joined
    pub turn_deg: Option<f64>,             // the turn from the start to it, made or not
    pub precision: Option<GrownPlanePrecision>,
    pub written: usize,                    // keypoints written, the reference's among them
}
// joined(), left_out(), shift_only(), matched(), pairs_kept(), turned(),
// turn_sigma_ratio();
// Display: "5 of 5 views joined, 2 of 2 pairs kept; σ 0.56° weakest, 0.48°
// strongest (0.67° weakest with each view once); 6 keypoints written",
// "3 of 3 views joined, 1 of 1 pairs kept; σ 9.58° weakest, …, low
// confidence; normal left unchanged: large turn on weak evidence, 19.8° at
// 2.1σ", or "0 of 5 views joined, 5 left out, normal left unchanged".

/// How the step ended. NoneJoined: no plane, the track as given.
/// WeakTurn: a turn of at least weak_turn_deg under min_turn_sigma_ratio
/// times sigma_weak_deg, not made; the track as given, the plane reported.
pub enum GrownPlaneOutcome { Turned, NoneJoined, WeakTurn }

pub enum GrownPlaneView {
    /// The keypoint written after the turn: the one it had, or the anchor's
    /// projection where it had none; None when nothing was written.
    Reference { keypoint: Option<[f64; 2]> },
    /// start of the growth that was kept; its plane's distance from the
    /// final plane and its score there; shift (moved to the final plane) and
    /// grid_zncc at the final plane, equal to held_zncc; the keypoint
    /// before the step and the one written (None when off the photograph).
    Joined {
        start: GrownPlaneStart,
        shift: [f64; 2],
        grid_zncc: f64,
        distance: Option<f64>,
        held_zncc: Option<f64>,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    /// Not picked: its shift alone fitted at the final plane; keypoint
    /// written when grid_zncc reaches min_held_zncc.
    ShiftOnly {
        start: GrownPlaneStart,
        shift: [f64; 2],
        grid_zncc: f64,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    LeftOut { reason: GrownPlaneRefusal, distance: Option<f64>, held_zncc: Option<f64> },
    /// Not in, a second row in one image, no reference, or not picked by a
    /// selection that joined no view or whose turn was not made.
    NotAttempted,
}

pub enum GrownPlaneStart { NoSearch, Keypoint, Widened, NoKeypoint }
pub enum GrownPlaneTiltStart { Starting, Combined }
pub enum GrownPlaneRefusal { UnderFloor, TooFar, HeldUnderFloor, ViewLeftOut }

/// One try of one view, and the combined plane after it.
pub struct GrownPlaneTry {
    pub observation: usize,
    pub image: u32,
    pub angle_deg: f64,
    pub attempts: Vec<GrownPlaneAttempt>,  // from the combined plane, then from the start unsearched, then searched
    pub outcome: GrownPlaneTryOutcome,     // Matched or LeftOut
    pub combined_normal: Option<[f64; 3]>, // the picks' plane after this pick
    pub combined_sigma_deg: f64,
}

/// One growth of one view against the reference, and how it was judged.
pub struct GrownPlaneAttempt {
    pub start: GrownPlaneStart,
    pub tilt_start: GrownPlaneTiltStart, // Starting or Combined: the plane its windows started from
    pub stage_zncc: Vec<f64>,   // stage 1 at its starting shift, each later window, the whole grid's robust score
    pub shift: [f64; 2],
    pub normal: [f64; 3],       // the view's own plane
    pub grid_zncc: f64,
    pub sigma_deg: f64,         // its own weakest σ
    pub refusal: Option<GrownPlaneRefusal>, // only ever UnderFloor
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

/// One view the selection picked.
pub struct GrownPlanePick {
    pub observation: usize,
    pub image: u32,
    pub angle_deg: f64,
    pub offset_direction: [f64; 2],   // in the starting frame's (u, v)
    pub scale_px: f64,                // photograph px per world unit at the anchor
    pub predicted_weak: f64,          // smallest eigenvalue predicted with it; NaN in the first rung
    pub matched: bool,
    pub rung: u32,                    // the selection's rung, the first 0
    pub sigma_after_deg: f64,
}

pub struct GrownPlanePrecision {
    pub sigma_weak_deg: f64,               // every kept estimate's information summed
    pub sigma_strong_deg: f64,
    pub discounted_sigma_weak_deg: f64,    // each kept pair's information halved
    pub discounted_sigma_strong_deg: f64,
    pub low_confidence: bool,   // discounted_sigma_weak_deg over low_confidence_sigma_deg; does not decide the turn
}

pub fn grown_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GrownPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError>;
```

**Why this shape.** The signature is the other steps', and the step ends almost as they do: it turns the patch with `tilt_patch`, then writes the keypoints of the reference, of every joined view and of every shift-only view that reached the floor, then reads the track back, so a caller can swap one step for another and compare `NormalReport`s. The write comes after the turn because `tilt_patch` itself moves every keypoint, and would carry the written ones off the pixels the step found; `turn_and_read` in [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs) runs through `read_back`, which takes the write as a closure between the two. The settings are `*Options` because a caller chooses them. The report is per observation because a wrong normal is diagnosed by asking which views were picked, which joined and from which start, and why the others were left out, and because a person checking the write needs each view's keypoint before and after it; the trace keeps every growth, so a view grown again from the search after the combination left it out shows both tries, and the picks and pairs say what the selection chose and what each pair measured. The step changes the normal and writes keypoints whenever at least one view joins, however low the plane's precision, unless the turn is a large one on weak evidence, and reports that precision (`GrownPlanePrecision`); it leaves the track as given when no view joins (`GrownPlaneOutcome::NoneJoined`), and then its sentence ends "normal left unchanged", and when it does not make a large turn on weak evidence (`WeakTurn`), and then the sentence says "normal left unchanged: large turn on weak evidence" with the turn and its ratio to σ. The two outcomes are kept apart because a caller acts on them differently: under `NoneJoined` there is no plane, and under `WeakTurn` there is one, reported in `normal`, `turn_deg` and `precision` with its views, picks and pairs, which the step judged too weakly supported to turn to. This differs from `grid_plane_normal`, which leaves the track as given on every ending that is not `Turned` (§ [Turn and report](#turn-and-report)). It refuses a track at infinity and one with fewer than two `in` observations through `normal_preconditions`, and it reports a phase for the whole step and one per try of a view, with finer detail phases when `progress` is detailed (§ [Implementation notes](#implementation-notes)), and is cancelled between stages through `progress`, since it can run longer than a second.

**Example.** `grown_plane_normal(&track, &edited, &images, &GrownPlaneNormalOptions::default(), &Progress::none())` returns the turned track and a report whose sentence reads, on `dino_dog_toy` point 810 from the mean viewing direction, "6 of 7 views joined, 1 left out, 10 shift only, 3 of 3 pairs kept; σ 0.37° weakest, 0.27° strongest (0.42° weakest with each view once); 17 keypoints written, turned 22.7°, …".

## Theory

### One plane shared by all views

The model is one plane, the same for every view. The warp from the reference view's working grid into any other view is the plane-induced map: each sample of the grid is followed along its reference ray to the plane, and that point is projected into the view's photograph through the camera's lens model, where the photograph is sampled. There is no free per-view affine map; the plane predicts each view's local affine map from its tilt alone.

The working grid is the starting frame re-anchored on the reference keypoint (`OrientedPatch::anchored_at_keypoint`), `W` samples a side, sample `(col, row)` at grid position `(col + ½ − W/2, row + ½ − W/2)`: the reference render's grid in `grid_plane_normal`. `W` is `working_resolution`, or with `None` that step's rule: the reference view's footprint of the patch at the starting frame, about one sample per photograph pixel, clamped to `R ..= 2R` for the patch resolution `R`; at least `spot_window` either way.

The reference is the step's reference view, by the rule `grid_plane_normal` and the fit use: the reference the track holds where its row is `in` with a keypoint, else the reference-view rule's pick among the `in` rows, one row per image. The other `in` views, one row per image, are the candidates.

The plane has two unknowns, a tilt `(a, b)` of the starting frame's normal `n` along the frame's own axes `u` and `v`:

```text
normal = unit(n + a·u + b·v)
```

through the anchor, the point where the reference keypoint's ray meets the starting frame. Each view adds its own image shift, two unknowns in image px, added to the projected positions. The shift absorbs error in the point's position and in the view's pose: a wrong depth along the reference ray moves every sample of a wide view by about the same amount, up to about 8 image px on the example below (image 36's shift was (−6.75, −4.25) px), which the tilt cannot express and should not try to.

**The keypoint-implied shift.** Each view's own keypoint already places the patch in that view, since the fit put it there. The shift it implies is the keypoint minus the anchor's projection into the view. Every plane the step tries passes through the anchor, so this shift is the same at every tilt.

**The render.** Every evaluation samples the photograph directly through the current plane, rather than resampling a tile rendered once at the starting plane, because a tile rendered at a wrong plane has already lost detail along the axis that plane squeezed. Each view is read from one level of its pyramid for the whole step: the level the bilinear-mip rule picks, `round(log2 σ_major)` clamped to the pyramid's depth, for the larger singular value of the map from the working grid at the starting plane into the view, at the grid's centre (`patch_grid_jacobian`). A sample is a bilinear read at that level, kept as a float, and is on the photograph when all four of its taps are. The bench's tile renders (`render_tile`) choose a level per sample or walk an anisotropic footprint and round to bytes; either would make the score jump as the plane turns, which no Jacobian follows, so the step keeps one level per view and the float sample. The reference is rendered the same way at no tilt and no shift; its rays meet every plane where the reference sees it, so its tile is the same at every tilt and is the template every view is scored against. Its level is 0 unless the grid was clamped well under its footprint.

### Near to far within a view

Under a plane that is wrong, a sample at distance `r` from the window's centre is displaced in the view by about `‖A − I‖·r`, where `A` is the error in the view's local affine map. A small window therefore tolerates a large error in the map, and a map fitted over radius `r` is accurate enough to bring radius about `2r` within capture of the next fit.

Every view the selection picks is grown against the reference from a plane: the selection's first rung from the starting plane, `(a, b) = (0, 0)`, and every later pick first from the combined plane of the views kept before it and, where that growth is refused, from the starting plane again (§ [The selection](#the-selection)). The growth from the combined plane starts its windows at that plane's tilt and at the view's keypoint-implied shift, which is the same at every tilt since every plane passes through the anchor; a view with no keypoint is grown from the search at that plane. The sequence below is the growth from the starting plane.

**Stage 1 at the anchor.** Stage 1 places a window of `spot_window` grid px (7×7, half-width 3) on the grid's centre sample, `(W/2, W/2)` by integer division: on an even grid that sample is half a sample right of and below the anchor. A view is grown first from its keypoint-implied shift with no search: stage 1 reads the window's ZNCC at that shift, which is the stage-1 score `stage_zncc` records, and the growing windows fit the shift from there together with the tilt (`NoSearch`). Only when that growth is refused, its own whole-grid score under `min_held_zncc` (`UnderFloor`), is the view grown again from a stage-1 search. The search finds the view's shift alone, at the starting plane, exhaustively on a lattice of `spot_search_step_px` (0.5) image px about the view's keypoint-implied shift, up to `keypoint_search_px` (3) image px along each axis, scanned row by row with the first maximum kept (`Keypoint`). It widens to `spot_search_px` (12) image px about the same keypoint-implied shift only when no shift within the ±3 px range reaches `min_keypoint_zncc` (0.7) or the best one is on the edge of that range (`Widened`). A search whose best shift is the keypoint-implied shift itself would repeat the growth just refused, so it is not grown again. A view with no keypoint, or one whose anchor does not project into it, has no implied shift; it is grown from the search alone, over the ±12 px range about zero shift, the anchor's own projection (`NoKeypoint`). A growth that reaches the floor is the view's estimate; whether it is kept is the combination's verdict (§ [Combining the views' planes](#combining-the-views-planes)).

The growth with no search comes first because the 7×7 window can be ambiguous where the keypoint is not. On dino point 933, seen by three views within 10° of the reference, the ±3 px search found no shift reaching 0.7 in any view and widened, and in every view it picked a wrong peak scoring about 0.6. The Python prototype recorded first-window shifts of (−5.0, 11.0), (−10.0, 10.5) and (−7.0, −1.5) px for the three views, against keypoint-implied shifts of (1.74, −1.39), (1.77, −0.99) and (0.18, −1.35) px, so the peaks were about 14, 16 and 7 px from the implied shifts; the fit then collapsed and no view joined, although each keypoint-implied shift was within 0.7 px of the true one. Grown from the implied shifts, all three views join (§ [Measurement on sixteen points](#measurement-on-sixteen-points-2026-10-17)). The search stays as the fallback for a view whose keypoint is off by more than the growing windows recover.

The search is centred on the keypoint-implied shift because a 7×7 window on repeating texture matches equally well at more than one shift, and a search over the full ±12 px picks a wrong peak as readily as the right one (see [Prototype on four points](#prototype-on-four-points-2026-10-16)). The window is centred on the anchor because the keypoint-implied shift is exact only there: every plane passes through the anchor, so at the anchor the right shift does not depend on the plane, but at a spot `r` away it differs by the plane error's displacement there, about `(A − I)·r`. On dino point 528 that displacement carried image 45's match at an off-centre spot outside the ±3 px search, so the search widened and locked onto a wrong shift 10 px away. The multi-spot proposal of the first round and the spot candidates no longer apply to stage 1 under the keypoint start. A textured spot off the anchor needs a different compensation, for example pinning the anchor at the keypoint and searching the tilt at stage 1; that is proposed and not built (see [Open questions](#open-questions)), and the maintainer expects points whose centre lacks texture.

**The growing windows.** Each later stage widens the window about the same sample to the next half-width of the schedule, `window_fractions` of the working grid's side `W` rounded to grid px, any no wider than the first window's half-width dropped: 3, then 6, 10, 16 and 24 on a 48-sample grid, 3, 4, 6, 11 and 16 on a 32-sample grid, and 3, 5, 8 and 12 on a 24-sample grid; the half-width `W/2` covers the whole grid. Each stage fits the tilt and the view's shift together for the best mean-channel ZNCC over the window (§ [The optimiser](#the-optimiser)).

**The whole-grid stage.** After the last window, a final stage fits the tilt and the shift for the best robust cell score over the whole grid (§ [The robust cell score](#the-robust-cell-score)). The view's own plane and its whole-grid score are those this stage ends at.

### The robust cell score

Wherever the step scores a whole grid (the floor and every held score, each view's information, the whole-grid stage, the shift-only fit, the pair fit, and the scores in the report), it scores the grid's nine cells rather than the grid at once. The cells are the step's own three-by-three split, rows and columns at `[0, W/3, W − W/3, W]`. A cell where the reference is flat, its spread under 0.5 grey levels in every channel, is not scored. Each live cell's ZNCC is read as every ZNCC is (§ [When an estimate agrees with the combined plane](#when-an-estimate-agrees-with-the-combined-plane)), `−1` under the coverage bar, and the cells are combined by a one-sided Tukey weighting:

- `m` is the median of the live cells' ZNCC;
- the scale is 1.4826 times the median absolute deviation from `m` of the cells at or above `m`, floored at 0.05, and the cut is 4.685 scales;
- a cell at or above `m` weighs 1, and one below weighs `(1 − ((m − z)/cut)²)²`, 0 past the cut;
- the score is the weighted mean of the cells' ZNCC; with no live cell it is `−1`.

The scale is read from the cells at or above the median only, because with several bad cells the spread of all of them is so wide that nothing falls past the cut. The reason for the score: on dino point 528, images 45 and 46 match the reference in the bottom two cell rows (0.67 to 0.81) but not the top row (0.40, −0.06 and 0.06), whose texture the two views do not see, and a whole-grid ZNCC refused them. On that point the robust score did not rescue them in the prototype, because the prototype's shared plane had drifted 9° after two nearly uninformative views had joined it in sequence; the step fits no views together, and a later pick is only started from the combined plane, its own plane fitted from there against the reference alone.

### The optimiser

Every fit is Levenberg–Marquardt on the photometric residual of mean- and spread-normalised samples. For one channel, with `a` the reference's samples and `b` the view's over the same samples, each less its mean, `â = a/‖a‖` and `b̂ = b/‖b‖` are unit vectors and

```text
‖â − b̂‖² = 2 − 2·ZNCC(a, b)
```

so maximising the ZNCC is minimising that residual. A block of samples (a window, or one cell) contributes its channels' residuals scaled by `√(w / C)` for its weight `w` and its `C` scored channels, so its residual sum of squares is `w · (2 − 2z)` for its mean-channel ZNCC `z`. A window is one block of weight 1; a whole grid is its live cells, each weighted by its Tukey weight over the weights' sum. Either way the residual sum of squares is `S = 2 · (1 − score)` for the score being fitted.

The residual is frozen at the parameters it is linearised at: which samples are on the photograph, which channels are scored, and the cells' Tukey weights, which are read again at the start of every iteration, as an iteratively reweighted fit. Only the view's samples move with the parameters.

The Jacobian is analytic, read at the render the residual is frozen at, and costs no render of its own. A sample's value moves with the view's shift by its image gradient there, divided by the level's scale so that it is per full-resolution px. It moves with the tilt by that gradient times how its pixel moves as the plane turns about the anchor: the sample's reference ray is fixed, so its world point `o + λ·d` slides along the ray with `λ = ((A − o)·n)/(d·n)`, and the pixel follows through the view's pose and its lens model's Jacobian, analytic for the perspective family and the θ-map fisheyes and a central difference of the projection for the others. Each scored channel of a block then goes through its normalisation by the chain rule: for the centred samples `c = v − mean v` and `û = c/‖c‖`, `dû = (dc − û (û·dc))/‖c‖` with `dc = dv − mean dv`, and a channel too flat to normalise, whose normalised samples are zeros, adds nothing. A sample off the photograph is not in the frozen residual, so it has no row. In the pair fit both renders move with the tilt and only the second with the relative shift, and each is normalised over the block afresh, so the Jacobian is the first view's normalised derivative less the second's.

The gradient is read in one of two ways, at the sample's level position and with the value's bilinear weights:

- **For the fit's steps**, from the level's Sobel gradient images: at each pixel, the difference of its two neighbours along the axis, halved, averaged across the other axis with weights `[1, 2, 1]/4`. The bilinear read of those images is continuous across pixel edges, and is the slope of the photograph averaged over about two pixels. It is computed from the 4×4 pixels about the sample, which give the Sobel values at the four taps the value reads, so no gradient image is stored.
- **For the information about the tilt** (§ [The plane's precision](#the-planes-precision)), from the bilinear read's own derivative, from the same four taps as the value: the local slope of the score the fit maximises. It jumps where the position crosses a pixel centre, and along an axis where a sample's two taps are clamped to one edge pixel it is zero.

The steps use the Sobel gradient because the bilinear read's slope, the exact derivative of the render, steps the fits less well on the ground truths (§ [An analytic Jacobian](#an-analytic-jacobian-2026-10-18)). The information keeps the exact derivative because the Sobel gradient is smaller wherever the photograph has detail a pixel or two across, which would inflate every σ and change which views the selection picks and when it stops.

An iteration solves `(JᵀJ + λ·diag(JᵀJ) + ε) δ = −Jᵀr`, with `λ` from `initial_damping` (10⁻³) and `ε` 10⁻⁹ of the largest diagonal entry, scales `δ` back so that no tilt parameter moves more than 0.1 and no shift more than 2 px, and reads the true score at the new parameters from a fresh render. The step is kept only when the score rises, and `λ` falls tenfold; the render that scored the kept step also gives the residual, frozen there, that the next iteration linearises at, so a kept point is rendered once; otherwise `λ` rises tenfold and the step is solved again, up to three tries in all. A stage ends after `max_iterations` (20) iterations, when an iteration raises the score by less than `min_improvement` (10⁻⁶) or by less than 1% of what the stage's first kept step raised it, or when no damping finds a rise; a stage that cannot rise keeps the parameters it was given. A step is kept only when the score rises, so the first step's gain is positive and the 1% rule first applies to the second step: a stage whose first step does nearly all its work stops after the second, which is kept. The 1% is a constant of the optimiser, not a setting, as the three tries are; `min_improvement` stays the absolute floor.

The parameters are a view's tilt and shift `[a, b, sx, sy]` within a view, its shift alone with the tilt held in the shift-only fit, and in the pair fit the tilt and the second view's shift against the first's `[a, b, rx, ry]` (§ [Wide pairs](#wide-pairs)). The pair fit's residual compares two renders that both move with the tilt, so it normalises both sides afresh at every evaluation, where a view against the reference normalises the reference once. No fit has more than four parameters.

The stage-1 search stays exhaustive, as above: the 7×7 window on repeating texture has several equal peaks, and a local fit from the wrong one stays there.

### Combining the views' planes

Every growth ends at the view's own plane, the tilt `tₑ` its whole-grid stage reached, with its shift and its information `Iₑ` about the tilt, read at that plane and shift (§ [The plane's precision](#the-planes-precision)). The step combines planes by their information and never fits views together. The combined plane of a set of estimates is their information-weighted mean,

```text
t = (Σ Iₑ + ε)⁻¹ · (Σ Iₑ·tₑ + ε·t̄)        covariance 2τ · (Σ Iₑ)⁻¹
```

the plane that maximises the sum of the estimates' quadratic score models `−½ (t − tₑ)ᵀ Iₑ (t − tₑ)`. `t̄` is the plain mean of the tilts and `ε` is 10⁻⁶ of the summed information's largest eigenvalue, or 1 where that is zero, so that the mean of one estimate is that estimate and a direction no estimate informs takes the plain mean rather than being undetermined.

The step runs in this order:

1. **The selection** picks views one at a time near to far, growing the first from the starting plane and each later one from the combined plane so far (§ [The selection](#the-selection)). After each pick that reaches the floor, the picks so far are combined by the robust combination below, and the selection stops once that plane is precise enough.
2. **The robust combination** of every pick that reached the floor.
3. **The search fallback for a view left out.** A view the combination leaves out whose kept growth had no search is grown once more from the search, as a refused growth with no search is everywhere else, unless the search lands on its implied shift; where that growth reaches the floor it replaces the view's estimate, and the planes are combined again.
4. **The wide pairs** among the views kept, fitted from the combined plane, and the robust combination of the views' and the pairs' planes (§ [Wide pairs](#wide-pairs)). That plane is the final plane.
5. **The shift-only fits** of every candidate the selection did not pick, at the final plane (§ [The selection](#the-selection)).

**The robust combination.** Every view estimate, and every pair estimate not already refused under its floor, is in the first mean. While some kept estimate lies further than `max_agreement_distance` (3) from the mean of the kept ones, by the Mahalanobis distance of § [When an estimate agrees with the combined plane](#when-an-estimate-agrees-with-the-combined-plane), the furthest is left out (`TooFar`), with any pair that holds a view left out (`ViewLeftOut`), and the mean is read again. Then each kept view is scored at the mean, at its shift moved there (below), one render each; a view under `min_held_zncc` is left out (`HeldUnderFloor`) with its pairs, and the whole is repeated until no view is left out. The furthest goes first and one at a time so that the result does not depend on the order the estimates are given in, and so that at least one estimate is kept, since an estimate alone is at no distance from its own mean. The distance is read against a mean the estimate is part of, which shortens it.

**Moving a shift to another plane.** A view's shift was fitted at its own plane `tₑ`, and the plane it ends at is `t`. With `N` the normal matrix of its whole-grid residual at its own plane and shift over `[a, b, sx, sy]`, the shift the Gauss–Newton model fits at `t` is

```text
s = sₑ − N_ss⁻¹ N_st (t − tₑ)
```

which costs no render. The held score at a mean, the score the report gives at the final plane, and the keypoint the step writes are all read at the shift moved this way; nothing is fitted again. The score at the final plane is not rendered again either: the combination's last held check scored every kept view at the final plane and that shift and left none out, so a joined view's `grid_zncc` is its `held_zncc`.

### Wide pairs

**Why.** How much a view says about the tilt grows with its angle from the view it is compared against: the further apart two views are, the more a change of tilt changes one view's picture of the patch against the other's. Every growth compares a view with the reference. When the reference sits among the cameras rather than at one end, two views on opposite sides of it are further apart from each other than either is from the reference, so comparing them with each other is more sensitive to the plane than either comparison with the reference.

**Which pairs.** Among the views the robust combination kept, every pair is ranked by the angle between the two views' directions from the patch centre, widest first, ties by the two views' order among the candidates, and pairs are taken down that list so that no view is in more than one pair, up to `max_pairs` (3). Taking pairs greedily by angle is a choice: it does not ask that a pair lie on opposite sides of the reference, or that it be wider than the widest view's angle from the reference, so a pair of two views on the same side is taken where it is the widest left.

**The fit.** Both views are rendered through the plane onto the reference's working grid: each sample of the grid is followed along its reference ray to the plane and projected into each of the two photographs. The fit starts from the combined plane, holds each view's shift against the reference at its own shift moved to that plane, and fits the tilt and one relative shift `(rx, ry)` added to the second view's, for the best robust cell score of the ZNCC between the two renders, over the cells the reference is textured in (§ [The robust cell score](#the-robust-cell-score)). The pair constrains only the difference between its views' shifts, since shifting both by the same amount changes neither render against the other, which is why the common part is held at the reference comparisons' value. The pair's information about the tilt is the Schur complement of its normal matrix with the relative shift eliminated, as for a view.

**Acceptance.** A pair whose score between its two renders is under `min_held_zncc` is refused (`UnderFloor`) and never combined. The others are added to the robust combination with the views, under the same distance test, and a pair holding a view the combination leaves out is left out with it.

**Two σ.** A pair compares two photographs each of which is already compared with the reference, so its information is not independent of theirs. The precision is reported both ways: σ (a), `sigma_weak_deg` and `sigma_strong_deg`, from every kept estimate's information summed, and σ (b), `discounted_sigma_weak_deg` and `discounted_sigma_strong_deg`, with each kept pair's information halved, so that a view in a pair counts about once. The rule is the simplest that discounts the reuse; it is not derived from the noise. The plane itself is the mean with the full information of every kept estimate, (a).

**Risks.** The two views of a wide pair differ more from each other than either does from the reference: in foreshortening, so the two renders sample the photographs at more different scales; in sampling, so one render can be blurred where the other aliases; and in lighting, since the two look at the surface from further apart. Near an occlusion they can also see slightly different surface. Each of these lowers the ZNCC between the pair's renders for a reason other than the tilt, which is why a pair is held to the floor and to the distance test.

### The selection

The step predicts each candidate's information about the tilt from geometry before it renders the view, and grows only the views the prediction picks.

**The model.** For each candidate: `θ`, its angle from the reference seen from the patch centre; `d`, the unit direction of its offset from the reference, the difference of the two viewing directions written in the starting frame's `(u, v)`; and `s`, its photograph px per world unit at the anchor, the square root of the area scale of the projection of the starting frame there, capped at the working grid's samples per world unit `ρ`, since the grid samples no finer. With `e` the direction across `d` and the geometric gain `g = (s/ρ)² · sin²θ`, the prediction is

```text
P = g · (α · d dᵀ + β · e eᵀ)
```

`α` and `β` are read from the views matched so far: each the median over them of their measured information along `d`, or across it, divided by `g`. With `β` zero this is the rank-one prediction along the offset direction. It is not rank one in practice: a single view at 40° on the tests' scene determines the tilt in both directions, to 1.47° weakest, and a rank-one model, scaled by the ratio of traces, picked 20° views across the first one's offset over the 40° views and needed five views where the two-direction model needs three.

**The picks.** The selection picks near to far by angle from the reference, in rungs, because a wide view grown from a start far off fails or ends on a wrong plane more often than a narrow one, and a wide view grown from a plane the narrower views reached is within capture (§ [A near-to-far selection](#a-near-to-far-selection-2026-10-18)).

- **The first rung** takes the view nearest `first_rung_deg` (15°) within `first_rung_min_deg` (10°) to `first_rung_max_deg` (20°), ties by the earlier view, grown from the starting plane. A refused pick is followed by the next nearest view in that range. Where no view in the range is left and none has passed, the first rung takes the view nearest `first_rung_deg` among all those left, so the widest view under 10° where nothing is wider. A view under 10° gives a poor plane on its own (σ 13° to 40° in the evidence), but the ladder climbs from it.
- **Each later rung** aims at `rung_ratio` (1.6) times the widest angle among the views the combination keeps, and takes the views within a factor √1.6 of that aim: from a first view at 15°, the aims are about 24°, 38° and 61° where each rung keeps a view near its aim. A rung with no view left in its range aims 1.6 times further out, and one with an aim of zero, where every view kept is 0° from the reference, aims at the widest candidate. The rung whose range reaches the widest candidate is the last: it takes every view from the smaller of its aim and the widest candidate's angle, over √1.6, outward, and the selection stays on it, so it ends with the widest views. Once no view is left there, one more rung takes every view left, the narrower ones among them.
- **Within a later rung** the pick maximises the smallest eigenvalue of the measured information of the views the combination keeps plus the view's prediction, which spreads the picks' offset directions about the normal, ties by the larger eigenvalue, then by the view nearer the aim, then by the earlier view.
- **Growth.** Each pick of a later rung is grown from the combined plane of the views kept so far, from its keypoint-implied shift; where that growth is refused, the view is grown from the starting plane as the first rung grows it, from its keypoint-implied shift and then from the search. A rung waits until one of its views is kept: a refused pick is followed by another view in the same range, and the ladder moves out only once the widest kept angle widens or no view is left in the range.
- **Stopping.** After each pick that reaches the floor the picks are combined by the robust combination, and the selection stops when the combined weakest σ is at most `target_sigma_deg` (1.5°) or `max_selected_views` (6) picks have reached the floor, or no candidate is left. A candidate whose angle is not finite is never picked. The trace records each pick's try with the plane each growth started from (`tilt_start`) and the combined plane after it, and each pick its rung.

**The views not picked.** Each candidate the selection did not pick has its shift alone fitted at the final plane: stage 1 reads the first window at its keypoint-implied shift, and the whole-grid stage fits the shift with the tilt held. Where that is under the floor, or the view has no keypoint, the stage-1 search runs at the final plane and the shift is fitted from its best, unless it lands on the implied shift. The view is reported `ShiftOnly`; its keypoint is written when its score reaches `min_held_zncc` and is on the photograph. It is not in the plane. When no pick joined there is no final plane, and the views not picked are reported `NotAttempted`.

### The keypoints the step writes

The step solves for each joined view's shift, and the shift says where that view's photograph places the patch, so the step writes it back to the track rather than discarding it. The keypoint is the track-stage measurement `TrackMeasurement::keypoint` ([bench/track.rs](../../crates/sfmtool-core/src/bench/track.rs)): the pixel a commit writes and the place every reading of the row is anchored at, which the fit sets from its kernels and `sight_observation` sets by hand.

- **A joined view.** Its keypoint is set to the pixel where the final plane and the view's shift put the anchor, the shift its growth reached moved to the final plane by its own normal matrix (§ [Combining the views' planes](#combining-the-views-planes)). Every plane the step tries passes through the anchor, so that pixel is the anchor's projection through the camera's lens model plus the view's shift: the pixel the view's score at the final plane is read around. A pixel off the photograph, outside `[0, width) × [0, height)`, is not written, and the report's `keypoint` is `None`.
- **A shift-only view.** Its keypoint is set the same way from the shift fitted at the final plane, when that fit reaches `min_held_zncc`.
- **The reference.** It keeps the keypoint it had before the step. The anchor lies on the reference keypoint's ray, so it projects back to that keypoint, and the reference's shift is zero by construction: the reference tile is the template every other view is compared against, sampled at the plane's own positions, so it has no shift to fit. The step writes that keypoint back after the turn, because the turn moves it. The reference-view rule can pick a row with no keypoint, which it renders through the patch itself; its ray is then the one through the centre's projection, the anchor is the centre, and the step writes the centre's projection.
- **Every other row**: a view left out, a shift-only view under the floor, one never attempted, and every row not `in`. The step writes nothing for them. They keep the keypoint the turn gives them, as under every other normal step: `tilt_patch` keeps each sighting's offset from the centre in the patch's plane, rebuilds that point on the turned plane and projects it (`place_keypoints` in [bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs)).

**Without the turn.** When the step does not make a large turn on weak evidence (§ [Turn and report](#turn-and-report)), it writes no keypoint, and fits no view's shift alone: the track keeps the keypoints it was given, which are the readings at the starting normal it keeps. The joined views' shifts were measured at the plane found, at least 70° from the start under the defaults; moving them back to the starting plane through each view's normal matrix, as the turn moves them to the final plane, would carry that first-order model across the whole turn, far outside where it was read, and no score at the starting plane would check the result, where every keypoint the turn writes was scored at the plane it is written for.

**Readings at the old keypoint.** Every other field of a row's `TrackMeasurement` was read at its keypoint and says nothing about a new one. The precedent is the two places that write keypoints by geometry or by hand: `place_keypoints`, which `tilt_patch`, the translations and a resize from an edge run, and `sight_observation` each replace the row's measurement with one that carries only the new keypoint, and `sight_observation`'s doc gives the reason, that an evaluation recomputes everything else from the track as it stands. The step does the same for each row it writes, so `walked_px` and `walked_to`, which only a fit sets, go with the rest. The turn has already done this for every row and dropped the bitmap, and the readback that ends the step renders a new bitmap from the reference and reads every row again at the keypoint it now has, so no reading is left that a written keypoint makes stale. The write changes no verdict and pins nothing, as a fit pins nothing: a pixel the step found is not one a person ruled on.

**The centre does not move.** The step turns the patch about its centre, as the other normal steps do. A depth error along the reference ray shows up as the per-view shifts, up to about 8 image px on the example below, so the written keypoints can imply a different position than the patch's centre; the readback's `projection_offset_px` then shows, for each written row, how far its keypoint sits from where the centre projects. Whether the step should also re-triangulate the centre from the written keypoints, or leave that to the next fit, is an open question.

How far the keypoints move on real tracks was not measured.

### The plane's precision

A view's information about the tilt is read from the normal matrix of its whole-grid residual at the plane and shift it is read at: `N = JᵀJ` over `[a, b, sx, sy]`, with `J` read through the bilinear read's own derivative (§ [The optimiser](#the-optimiser)), with the shift eliminated by the Schur complement,

```text
I = N_tt − N_ts · N_ss⁻¹ · N_st
```

which is the information about the tilt with the shift fitted again at every tilt, so that a shift cannot stand in for a tilt. Its negative eigenvalues, which only rounding can make, are set to 0.

**The scaling.** The residual is scaled so that its sum of squares is `S = 2 · (1 − score)` (§ [The optimiser](#the-optimiser)). Near the fitted plane the Gauss–Newton model gives `S(δ) ≈ S₀ + δᵀ I δ` for a tilt change `δ` with the shift refitted, so the score falls by `½ · δᵀ I δ`: `I` is minus the Hessian of the score in the tilt, the quantity the prototype estimated by finite differences of the ZNCC. A ZNCC drop of `τ` is therefore a residual rise of `ΔS = 2τ`, reached on the ellipse `δᵀ I δ = 2τ`. The ZNCC tolerance τ (`zncc_tolerance`) is the ZNCC difference noise can produce; the prototype's 0.01 is a placeholder. A tilt change inside that ellipse cannot be told from noise, which makes the covariance of the tilt `2τ · I⁻¹`, regularised as `2τ · (I + 10⁻⁶)⁻¹` so that a direction with no information gets a large variance rather than none. σ is the plane's uncertainty along its weakest direction, `√(2τ/λ_min)` with `λ_min` the smaller eigenvalue of `I`, in degrees, since a small tilt `(a, b)` is that many radians; the strongest direction's σ comes from the larger eigenvalue in the same way. Since σ grows as `√τ`, the choice of τ scales every σ and every agreement distance by the same factor.

**Which information.** Every growth's information is read when it is judged, at its own plane and shift, and a pair's at the plane and relative shift its fit ended at. The combined plane's information is the sum of the kept estimates' information as each was read, and the precision the report gives is that sum, σ (a), and the same with each pair's halved, σ (b) (§ [Wide pairs](#wide-pairs)). Nothing is read again at the final plane.

### When an estimate agrees with the combined plane

A growth counts as an estimate when its own whole-grid robust score reaches `min_held_zncc` (0.5) (`UnderFloor` otherwise). The robust combination keeps an estimate while both hold:

- its own plane lies within `max_agreement_distance` (3) of the combined plane by the Mahalanobis distance `√(dᵀ C⁻¹ d)`, with `d` the difference of the two tilts and `C` the estimate's covariance and the combined plane's added (`TooFar` otherwise);
- for a view, its robust grid score at the combined plane, at its shift moved there, is at least `min_held_zncc` (`HeldUnderFloor` otherwise); for a pair, its score between its two renders is (`UnderFloor` otherwise).

The rule replaces convergence bars on the view's own readings (a whole-grid ZNCC of at least 0.7, a window ZNCC never more than 0.1 below its best earlier stage, a stage-1 match not on the edge of its search), which left out views that ended near the stored normal and, on one of the second round's points, let no view join at all. In the prototype, which took the views in sequence, a fixed bar on how much ZNCC a view loses when held to the plane the views before it shared was tried in between, and let in a view whose own plane was 26° from that plane; scaling the test by precision is what refuses it (see [Prototype on four points](#prototype-on-four-points-2026-10-16)).

Every ZNCC is the mean over the colour channels the reference is not flat in (spread at least 0.5 grey levels), over the samples that land on the photograph in both renders, and is read only when at least `min_coverage` (60%) of the window's or cell's samples do; under it the ZNCC is `−1`. A fourth channel, the photograph's alpha, is not read.

### Turn and report

The step turns the patch to the final plane when at least one view joined, unless the turn is a large one on weak evidence, and reports σ along the weakest and strongest directions as its confidence, both ways (§ [Wide pairs](#wide-pairs)), with a verdict of low confidence when the weakest σ (b) is over `low_confidence_sigma_deg` (the prototype marked σ over 2°). σ (b) is the one judged because it is the larger; with no pair kept the two are the same. The low-confidence verdict does not by itself decide the turn. Precision is used inside the step, in the agreement rule above and to stop the selection, and in the weak-turn rule.

**The weak-turn rule.** The turn is the angle from the starting normal to the final plane's normal, the `TiltReport::degrees` the turn gives, or would give: `tilt_patch`'s arc, stopped where it would stop at `MAX_TILT_DEG` from a camera (`tilt_degrees` in [bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs) computes it with no track built). A turn of at least `weak_turn_deg` (70°) whose ratio to σ (a), the weakest-direction σ of every kept estimate's information summed with no pair halved, is under `min_turn_sigma_ratio` (3) is a large turn on weak evidence. The step does not make it: the outcome is `GrownPlaneOutcome::WeakTurn`, the normal and every keypoint are left as given, no view not picked has its shift fitted, and the track is read back. The report still gives the plane found (`normal`), the turn (`turn_deg`), the precision, and the views, trace, picks and pairs. A σ that is not finite counts as no evidence, a ratio of 0, so a large turn with no finite σ is not made, as a σ that is not a number is reported low confidence. Every other turn is made, a turn under 70° however weak its evidence. The rule reads σ (a), not σ (b), because the judgements that set it (§ [Judged by eye](#judged-by-eye-2026-10-18)) were scored on σ (a). Setting `min_turn_sigma_ratio` to 0, or `weak_turn_deg` to infinity, turns the rule off. The step leaves the normal unchanged on two outcomes only: when no view joined (`NoneJoined`), and on a large turn on weak evidence (`WeakTurn`).

This differs from `grid_plane_normal`, which leaves the normal unchanged when it has no good estimate. The reason a low precision alone does not stop the turn is that a rejected plane can still be much better than the start: on two of the four points of the second round, a 2° gate on σ left the patch at its start, 8.5° and 80.1° from the stored normal, where the planes it rejected were 1.36° and 16.8°. A low-precision plane says the views constrain the tilt little near it, not that the start is closer; the local curvature of the ZNCC understates what narrow views can correct from far away, as one narrow view took bull point 236 from 80° to 5.6°. A consumer that needs a precise normal reads σ and chooses its own bar. The weak-turn rule is narrower than a gate on σ: it holds back only a turn of at least 70° that is under three times its σ, the case where the maintainer, judging 62 points by eye, found most turns wrong: of the turns of 65° or more under three times their σ, 9 of 13 were wrong (§ [Judged by eye](#judged-by-eye-2026-10-18)); under 70° the turn was right on almost every point, however weak its σ.

### Relation to `grid_plane_normal`

`grid_plane_normal` reads the plane from where nine cells register, by shift alone, on a render made at the current plane, and gates each cell in each view by its ZNCC, its curvature, the bound and the view's robust affine fit; it can only measure what lands inside the shift bound. This step instead treats the plane as the thing being fitted, and every view's picture as evidence for it. The two share the reference rule, the working grid's size rule, the starting frame and the final turn and readback. They differ on a weak estimate: `grid_plane_normal` leaves the normal unchanged when it has no good estimate, and this step turns to any plane at least one view joined and reports its σ, unless the turn is a large one on weak evidence. Where `grid_plane_normal` ends `Turned`, the two should agree; the cases this step is for are the wide views `grid_plane_normal`'s gates refuse at the starting plane.

### Relation to the prototype

The step follows the prototype run with `SPOT_RULE=anchor JUDGE=precision KP_START=1 ROBUST=1`, the runs the maintainer approved, within a view, and differs from it in these ways:

- **The optimiser.** The prototype's growing windows, whole-grid stage, held score and joint refinement used a pattern search; the step uses Levenberg–Marquardt (§ [The optimiser](#the-optimiser)).
- **The views and how they are combined.** The prototype took every view in order of angle from the reference, grew each from the plane the views before it shared, and refined the shared tilt over every joined view after each join. The step grows only the views the selection picks, near to far as the prototype did, the first from the starting plane and each later one from the combined plane of the views kept before it, fits no views together, and combines their planes by a robust information-weighted mean with the wide pairs added (§ [Combining the views' planes](#combining-the-views-planes)).
- **The information.** The prototype estimated each view's information by a centred second difference of the whole-grid score, a step of 0.03 tilt units along each axis and one mixed sample, with the shift fitted again at every sample; the step reads it from the normal matrix with the shift eliminated, which is the same quantity to the Gauss–Newton approximation and is positive semi-definite by construction.
- **The render.** The prototype sampled the full-resolution photograph bilinearly; the step samples one pyramid level per view, chosen by the bilinear-mip rule at the starting plane (§ [One plane shared by all views](#one-plane-shared-by-all-views)), so a view seen at more than about 1.4 photograph px per grid px is read from a coarser level. Its reference tile is its own render at no tilt, where the prototype took `grid_plane_normal`'s byte-valued reference render; the two have the same geometry.
- **The widened search's centre.** The prototype centred the widened ±12 px search on zero shift, the anchor's own projection; the step centres it on the keypoint-implied shift, as it does the ±3 px search, so a view whose keypoint is a few px off the content and whose pose is more than 12 px off still finds its shift. A view with no keypoint searches about zero shift in both.
- **The growth with no search.** The prototype always started stage 1 with the search about the keypoint-implied shift. The step first grows each view from that shift with no search, and searches only when that growth is refused (§ [Near to far within a view](#near-to-far-within-a-view)).
- **The working grid.** The prototype took the working grid's side from the `grid_plane_normal` record it was given, 48 on the dino points; the step applies the same rule itself, or takes `working_resolution`.

## Implementation notes

- **Cost.** What dominates is one render of the window per optimiser evaluation per view: an iteration renders once for each damping it tries, and a fit stops once a step gains under 1% of its first step's gain. The point an iteration starts from was rendered when its step was scored, and its Jacobian is read at that render: it renders nothing, and makes one pass over the samples that reads each sample's Sobel gradient, 16 pixels per channel, and, for a fit with the tilt free, goes through the lens model's Jacobian, which reads no photograph. The gradient is read only for a point a Jacobian is taken at, so a damped step that is refused costs its render alone. Each view keeps the projection of its last tilt, so the renders at one tilt and several shifts (the stage-1 search) project the grid once and only sample again, and a Jacobian reads the projection its point's render made. On the tests' synthetic scenes, 128-px photographs and a 32-sample grid, a whole step takes well under a tenth of a second in an optimised build. The cost is the growths, about linear in the number of views picked, with one render per kept view for each held check of the robust combination, a few whole-grid renders per iteration of each pair fit, and one stage-1 window and one whole-grid fit with the tilt held per shift-only view. On the 1609 points of § [Sweep of all ground-truth points](#sweep-of-all-ground-truth-points-2026-10-17) the step took 745 s, a median of 331 ms a point. Run alone on the same points, the step with the fewer renders of § [Fewer renders per fit](#fewer-renders-per-fit-2026-10-18) took 356 s, a median of 155 ms a point, and with the σ target of 1.5° 231 s, a median of 103 ms. With the analytic Jacobian and the Sobel gradient it takes 37% less time than the forward difference did, timed as a matched pair (§ [An analytic Jacobian](#an-analytic-jacobian-2026-10-18)).
- **Phases.** The step opens the overview phase `grown plane normal` over everything after its settings are checked, and under it a `grown plane view` per try, noted with the image and whether it was matched or left out. A detailed `Progress` adds, under the step, `setup` (the reference, the working grid, the views and the windows; the reference-view rule's phases nest in it), a `selection` per pick (noted with the view, its angle, its scale, its rung and its predicted weakest information), a `mean` per robust combination (noted with the views and pairs kept, those held under the floor, the means read, `σ` and the renders), `pairs` with a `pair fit` per pair (noted with the two images, their angle, the pair's ZNCC, `σ`, verdict, steps taken and renders), a `shift-only fit` per view not picked (noted with how its first window found its shift, its ZNCC and whether it is written), and `turn and write` (the final shifts, the keypoint writes and `read_back`, whose phases nest in it, noted with the keypoints written); under each try, a `growth` per growth, noted with the plane it started from, how its first window found its shift, its first-window and grid ZNCC, whether it was taken, and its renders; and under each growth a `keypoint search` or `widened search` where one ran (noted with its bound, its centre, the best ZNCC, the best shift's offset and the shifts tried), a `window stage` per later window and a `whole-grid stage` (each noted with its ZNCC, steps taken and renders), and `judge` (the growth's own information, noted with its grid ZNCC, `σ` and verdict). The shift-only fit's searches nest under it. A growth whose search lands on the implied shift is noted so and holds no stages. A view counts its renders in a `Cell` beside its projection cache; the notes are built only when a phase records, and nothing a phase does reaches the result, so the step's track and report are the same with detail on and off.
- **Single-threaded.** The step renders one view at a time on the calling thread, as `grid_plane_normal` does; nothing in it is parallel.
- A view's keypoint is read once, for the keypoint-implied shift that its first growth starts from and its stage-1 search is centred on. A view with no keypoint, or whose anchor does not project into it, is grown only from a search about zero shift and is reported `NoKeypoint`.
- **Cost of the order.** A refused growth with no search costs a whole growth before the searched one; the search itself is at most 2401 renders of the 7×7 window, small against a growth. Where the keypoints are poor every picked view pays for both growths: on Kerry Park point 242 every view's window scores −0.47 to 0.06 at its implied shift (§ [Measurement on sixteen points](#measurement-on-sixteen-points-2026-10-17)).
- The write sits between the turn and the readback, in the closure `read_back` takes. The anchor and each written view's shift at the final plane are kept for it.
- The weak-turn rule is judged after the pairs and before the shift-only fits, so a turn that is not made costs no shift-only fit; the turn it judges is `tilt_patch`'s, computed by `tilt_degrees` without building the turned track. The rule changes nothing on a point whose turn it makes: the selection, the combination, the pairs and the shift-only fits are the same, in the same order.
- A track at infinity is refused by `normal_preconditions`, as for the other steps: its patch faces along its own bearing and has no plane to fit. The step's geometry assumes a finite anchor and finite reference rays throughout.
- A ray that meets the plane behind the reference camera, or not at all, and a point a camera does not project, give a sample off the photograph, as a sample past the image's edge does.

## Determinism and precision

Photograph samples are `f32`, read bilinearly from `u8` pyramid levels; the ZNCC sums, the residuals, the tilt, the shifts and the plane are `f64`. The candidates are held in order of their angle from the reference, ties by observation index; the selection's picks and the pairs break their ties by fixed rules (§ [The selection](#the-selection), § [Wide pairs](#wide-pairs)); every search and fit is deterministic, and nothing runs in parallel, so the output is the same on every run. No content hash depends on it.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `working_resolution` | `None` | Side of the working grid, samples; `None` follows the reference's footprint, clamped to `R ..= 2R`, as `grid_plane_normal` does; at least `spot_window` |
| `spot_window` | `7` | Side of the first window, grid px, centred on the grid's centre sample |
| `window_fractions` | `[1/8, 1/5, 1/3, 1/2]` | Half-widths of the later windows as fractions of the grid's side, rounded, any no wider than the first window's dropped: 6, 10, 16, 24 on a 48 grid; 4, 6, 11, 16 on a 32 grid; 5, 8, 12 on a 24 grid |
| `keypoint_search_px` | `3.0` | Bound of the first window's shift search about the keypoint-implied shift, run when the growth from that shift is refused, image px |
| `min_keypoint_zncc` | `0.7` | Best stage-1 ZNCC within that bound below which the search widens |
| `spot_search_px` | `12.0` | Bound of the widened search about the keypoint-implied shift, and of a keypoint-less view's search about zero shift, image px |
| `spot_search_step_px` | `0.5` | Step of both searches, image px |
| `zncc_tolerance` | `0.01` | ZNCC difference noise can produce, τ; a placeholder |
| `max_agreement_distance` | `3.0` | Largest Mahalanobis distance of an estimate's plane from the combined plane |
| `min_held_zncc` | `0.5` | Robust grid score a growth needs on its own, a view needs at the combined plane, a pair needs between its two views, and a shift-only view needs for its keypoint to be written |
| `low_confidence_sigma_deg` | `2.0` | σ (b) above which the report says low confidence; it does not by itself stop the turn |
| `weak_turn_deg` | `70.0` | Smallest turn from the starting normal, degrees, that the weak-turn rule judges: a turn of at least this under `min_turn_sigma_ratio` times σ (a) is not made; at least 0, infinite turns the rule off |
| `min_turn_sigma_ratio` | `3.0` | Ratio of the turn to σ (a) that a turn of at least `weak_turn_deg` needs to be made; a σ that is not finite counts as a ratio of 0; finite and at least 0, 0 turns the rule off |
| `max_pairs` | `3` | Pairs of views matched against each other, at most |
| `target_sigma_deg` | `1.5` | The selection stops once the combined weakest σ is at most this, degrees |
| `max_selected_views` | `6` | The selection stops once this many picks have reached the floor |
| `first_rung_deg` | `15.0` | The selection's first pick is the view nearest this angle from the reference within the first rung's range, degrees; within that range itself |
| `first_rung_min_deg` | `10.0` | The narrowest angle of the first rung's range, degrees; finite, at least 0 |
| `first_rung_max_deg` | `20.0` | The widest angle of the first rung's range, degrees; finite, at least `first_rung_min_deg` |
| `rung_ratio` | `1.6` | Each later rung aims at this times the widest kept angle and takes views within a factor of its square root of that aim; finite, above 1 |
| `min_coverage` | `0.6` | Share of a window's or cell's samples on the photograph for its ZNCC to be read |
| `max_iterations` | `20` | Levenberg–Marquardt iterations per stage, at most |
| `initial_damping` | `1e-3` | The damping each fit starts from |
| `min_improvement` | `1e-6` | A stage stops once an iteration raises its score by less |

The search and agreement defaults are the prototype's, set on one point in the first round and four in the second. The three optimiser settings are this implementation's and were chosen on the tests' synthetic scenes only. `max_pairs` and `max_selected_views` are first values, with two other settings of `max_selected_views` measured in § [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17). `target_sigma_deg` was 0.5° at first and is 1.5° since the sweep of § [A looser σ target](#a-looser-σ-target-2026-10-18). The ladder's settings were proposed from the angle evidence of § [A near-to-far selection](#a-near-to-far-selection-2026-10-18) and swept there, which moved `first_rung_deg` from 12° to 15°.

## Testing

The tests in [bench/tests/grown_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs) build synthetic planes seen by 128-px pinholes that look at the point from around it, 4 units away, through `Scene::looking_at`, with the patch spanning 32 px in a facing view; [grown_plane/tests.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/tests.rs) tests the parts that need no photograph, and [grown_plane/jacobian_tests.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/jacobian_tests.rs) the analytic Jacobian. Every test runs the defaults unless it names a setting. The tests on the scenes with up to six views that need every view picked set a target σ of 0.5°, which the combined σ there stays over. Built and passing:

- A textured plane started 25° off its normal, seen from six views out to 50°: every view is picked and joins on its first growth, from its keypoint-implied shift with no search, the first from the starting plane and every later one from the combined plane; the normal ends within 0.02° of the plane, every written keypoint is within 0.5 px of the point's projection (within 0.01 px here), the reference keeps its keypoint, and 6 keypoints are written. Both pairs are kept, and σ is 0.56° weakest and 0.48° strongest, 0.67° with each pair's information halved.
- One view at 25° whose top cells see black and white blocks the reference never sees there: it joins under the robust cell score, and the normal ends 0.03° from the plane.
- A textured plane seen only from views within 3° of the reference, started 20° off: the patch is turned, here to 0.3° from the plane, with σ 9.6° weakest (10.3° with the pair halved), and the report says low confidence.
- A disagreeing view: one view at 38° sees the texture on a plane turned 25° from the one the others see. It passes the floor on its own, and the combination leaves it out as `TooFar`, 8.4σ from the combined plane; its search after that lands on its keypoint-implied shift, so it is not grown again and stays out. No kept pair holds it, and the normal ends within 0.09° of the others' plane.
- Repeating texture under the spot: a checker of 8 px period, with three views' poses turned so that the photograph's content sits about 6.5 px from where the pose projects the point and each keypoint on the content. Every view is matched on its first growth, `NoSearch`, joins and is written within 0.75 px of the content, and the same holds with `keypoint_search_px` 0, under which the search, had it run, would widen at once to the ±12 px about the keypoint-implied shift and take another peak. With the views' keypoints removed, so that each is grown from the ±12 px search alone, every view joins on another peak of the checker, written more than 4 px from the content.
- Keypoints off the content in views whose poses are further off: three views' poses turned so that the content sits about 16 px from where the pose projects the point, past the ±12 px a search about zero shift reaches, and each keypoint 6 px off the content, past the ±3 px search about it and past what the growing windows recover. Each view's growth with no search is refused, from the combined plane and from the starting plane alike; the widened search from the starting plane follows, and every view joins `Widened` and is written within 0.75 px of the content.
- Wide pairs: four views each 20° from a reference that faces the plane, on opposite sides of it, so that the two pairs of opposite views are 40° apart. Both pairs are taken and kept, and the weakest σ falls from 1.62° with `max_pairs` 0 to 0.95°, or 1.16° with each pair's information halved; the plane ends within 0.02° of the truth.
- The selection: sixteen views in rings at 10° to 40° about a reference that faces the plane, with a target of 1°. The first pick is a 20° view, of the first rung, and the next two are 40° views, of the last rung, grown from the combined plane and not along one offset; the selection stops after three picks, at a weakest σ of 0.99°, and so matches 3 of the 16 views; every other view is fitted for its shift alone and written within 0.5 px of the point's projection, and the plane ends within a degree of the truth.
- The ladder's order: views at 5°, 12°, 19°, 31° and 50° with the first rung at 12° and a target no scene reaches. The picks are 12°, 19°, 31°, 50° and then 5°, in rungs 0 to 4: one view a rung outward, then the view under the first rung. The first is grown from the starting plane and every later one from the combined plane, each on its one growth, and the plane ends within a degree of the truth.
- A refused rung: views at 12°, 17°, 22° and 35°, the 22° view seeing texture no plane matches, across the first pick's offset so that the second rung picks it first. It is refused from the combined plane and grown again from the starting plane, where it is refused too; the next pick is the 17° view of the same rung, which joins, and only then does the ladder move out.
- The end at the widest view that passes: views at 12°, 20°, 32°, 48° and 56°, the 56° view seeing unrelated texture. The last rung holds the 48° and 56° views and is reached after every narrower view; the 56° view is left out, and the 48° view, the widest that passes, joins.
- A view at the reference's own centre, 0° from it, beside one at 35°: the first rung takes the 0° view, and the selection then picks the 35° view rather than aiming at 0° again.
- A flat wall: no view joins, the outcome is `NoneJoined`, the normal and every keypoint are left as given, no plane, turn or precision is reported, and the sentence says the normal was left unchanged, with no mention of weak evidence.
- The narrow views' scene at the defaults: the turn of about 20° is under three times its σ, weak evidence, but under 70°, so it is made (`Turned`), and the turn the report gives equals the tilt's degrees.
- A large turn on weak evidence: the narrow views' scene with `weak_turn_deg` 10. The turn of 19.8° on a σ (a) of 9.58° is 2.1σ, so the step keeps the start (`WeakTurn`): the patch and every keypoint are as given, the tilt is 0°, no keypoint is written and no view is fitted for its shift alone, and the report gives the plane found, within 2° of the truth, its turn, its precision and the three joined views; the sentence says "normal left unchanged: large turn on weak evidence". With `min_turn_sigma_ratio` 0 the same scene turns.
- A large turn on strong evidence: the wide scene started 25° off with `weak_turn_deg` 10. The turn is about 45 times its σ of 0.56°, so it is made, to within a degree of the plane, with 6 keypoints written.
- A bearing is refused as `AtInfinity`, a search step of 0 as `BadGrownPlaneOption`, and a cancelled `Progress` as `Cancelled`. `check` refuses a `weak_turn_deg` that is NaN or negative and a `min_turn_sigma_ratio` that is NaN, infinite or negative, and takes an infinite `weak_turn_deg`, a `weak_turn_deg` of 0 and a ratio of 0; it refuses a first rung whose bounds are not finite, are negative or are out of order, a `first_rung_deg` outside its bounds, and a `rung_ratio` of 1 or less or not finite.
- On the scene of the disagreeing view with one view's keypoint removed, with four picks and with the default six, the track and report are the same, compared by their `Debug` text, with no sink, with an overview sink and with a detailed one; the overview records the step and one phase per try and nothing finer, and the detailed run records a `growth`, a `judge` and a `whole-grid stage` per growth, a `window stage` per later window of each, a `selection` per pick, a `mean` per combination, a `pairs` with a `pair fit` per pair, a `shift-only fit` per view not picked, at least one `widened search`, no `joint fit` and no `precision`, and one `setup` and `turn and write`, each noted, at the depths above; with every view picked, at least one `mean` notes a view left out.
- The analytic Jacobian, on renders of a textured plane through a pinhole, a radial lens and a fisheye, from a reference facing it and views 25° and 35° off, with the starting frame 10° off the plane. With the bilinear read's own derivative, for a view fit over the whole grid, a window, a shift-only fit and a pair, every column lies within 3% of a central difference of renders (steps of 10⁻⁴ in the tilt and 10⁻³ px in the shift), and the worst columns differ by 0.1% to 1.7%. The difference itself is no closer: a sample's position is read in `f32`, about 8·10⁻⁶ px near 100 px, 0.8% of those steps, and larger steps cross more of the pixel edges where a bilinear read's slope jumps. The forward differences the fits used before, with steps of 10⁻³ and 0.05 px, differ from it by 9% to 12%. With the Sobel gradient the worst columns differ from it by 26% to 31% on this texture, whose detail is a few px across; a wrong sign or a lost term would differ by 100% or more. A grid shifted until about a quarter of it lies off the photograph agrees within the same 3%, its off-photograph samples having no row. Without a photograph read, each piece is checked to 10⁻⁶ of its size: how a pixel moves with the tilt against a central difference of the projection, and a block's derivative through its normalisation, for a view and for a pair, with a channel the view is flat in adding nothing. The bilinear derivative matches a difference of the read's values inside the photograph and past its edges at pyramid levels 0 and 1, and is zero along an axis clamped to the edge and for a sample with no pixel. The Sobel read matches a bilinear read of Sobel images built pixel by pixel, inside the photograph and at its edges, reads a linear ramp's slope exactly, and moves by under 0.01 across a pixel centre where the bilinear derivative jumps by more than 1.
- Without photographs: the robust score on dino 528's cell readings, the residual identity `‖â − b̂‖² = 2 − 2·ZNCC` with weights, the coverage bar, a flat channel left unscored, the window schedule on 24, 32 and 48 grids, the Schur complement against the profile it stands for, σ from the eigenvalues, the fit reaching the optimum of a quadratic, and the weighted mean, its covariance and the agreement distance against a hand-computed case.

Not built, from the cases this draft proposed: the depth-error scene (a patch centre off along the reference ray, whose shifts the views absorb); σ against the spread of the fitted normal under added noise; a pair that is refused under its floor. A depth error large against the distance to the point changes each view's scale as well as its shift (on the tests' scene, half a unit at 4 units fitted a plane 7° off), so the depth-error scene needs a longer focal length than the tests' 160 px to keep the error small while the shift is several px.

The step has been measured on every finite ground-truth point of `seoul_bull_sculpture`, `kerry_park` and the `dino_dog_toy` candidate (§ [Sweep of all ground-truth points](#sweep-of-all-ground-truth-points-2026-10-17)); a hand review in SfM Explorer has not been made.

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

**Precision agreement.** With the Mahalanobis rule of § [When an estimate agrees with the combined plane](#when-an-estimate-agrees-with-the-combined-plane), τ = 0.01: point 810's image 27 was refused at 3.2σ from the shared plane and at 3.6σ on its retry; 16 of 17 joined, ending 0.54° with σ 0.38°. Kerry Park 91 ended 1.12° with σ 1.43°, dino 933 1.36° with σ 7.1°, and bull 236 16.8° with σ 390°. On point 810 a single view 7° to 9° from the reference has σ of 13° to 17°, and images 62, 65 and 66 each 0.6° to 0.9°; the shared σ fell from 13.8° after two views to 0.37° at the end. With a 2° gate on σ the prototype first left 933 and 236 at their starts, 8.5° and 80.1°; the maintainer reversed that to the rule of § [Turn and report](#turn-and-report).

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

## Sweep of all ground-truth points (2026-10-17)

**Question.** Over every ground-truth point rather than sixteen, how does the step compare with and without the joint fit, how do the five ways of combining the views' planes then built compare with each other, and what does each cost?

**Setup.** All 1609 points of the three ground truths, `dino_dog_toy` (the candidate v006) 961, `kerry_park` 382 and `seoul_bull_sculpture` 266, each started from the mean viewing direction and fitted once, as in § [Measurement on sixteen points](#measurement-on-sixteen-points-2026-10-17). Errors are to each file's stored normal, an estimate as before. The step was run in six forms: the former joint fit, the step with the fit of every joined view together after each join, and the five strategies of § [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17) at their defaults. The 2×2 Grid Plane Normal and `grid_plane_normal` were run from the same fitted track. Each row gives the median and mean error, the share of points within 2°, 5° and 10°, the share on which the step beats the 2×2 Grid Plane Normal, the share it reports confident (not low confidence), and the total time.

| Form | Median | Mean | < 2° | < 5° | < 10° | Beats 2×2 | Confident | Total time |
|---|---|---|---|---|---|---|---|---|
| start | 31.8° | 33.1° | 2% | 5% | 9% | – | – | – |
| 2×2 Grid Plane Normal | 24.7° | 27.0° | 6% | 13% | 22% | – | – | 353 s, median 101 ms |
| `grid_plane_normal` | 30.9° | 31.3° | 7% | 12% | 16% | – | – | 93 s |
| joint fit | 2.77° | 11.6° | 41% | 61% | 74% | 79% | 60% | 4470 s, median 810 ms |
| `sequential_mean` | 2.74° | 11.0° | 41% | 63% | 74% | 81% | 61% | 1731 s |
| `sequential_pairs` | 2.70° | 10.9° | 43% | 63% | 75% | 81% | 64% | 1806 s |
| `independent` | 2.75° | 11.1° | 42% | 62% | 74% | 81% | 59% | 1140 s |
| `independent_pairs` | 2.70° | 11.0° | 43% | 63% | 74% | 82% | 62% | 1223 s |
| `selected_pairs` | 2.74° | 11.2° | 42% | 62% | 74% | 81% | 62% | 745 s, median 331 ms |

**By dataset.** The median error with the joint fit, and the range over the five strategies: dino 1.46°, against 1.40° to 1.46°; Kerry Park 8.0°, against 7.5° to 7.8°; Seoul bull 21.5°, against 18.9° to 20.8°.

**Paired against the joint fit.** On the same points, the median of each strategy's error less the joint fit's is −0.01° to −0.03°. Each strategy is more than 1° better than the joint fit on 15% to 19% of the points and more than 1° worse on 11% to 16%.

**Calibration of σ.** Among the points each form reports confident, the error is more than twice σ on 51% to 86% of them, so σ at τ = 0.01 is too small.

**Cost.** All 18 runs, the six forms on the three datasets, shared one 32-core machine, so the absolute times are inflated, and only the ratios between forms within a run hold. By those, `selected_pairs` takes about a sixth of the joint fit's time, two thirds of `independent`'s and under half of either sequence's.

**Reading.** Without the joint fit the step is no worse on the whole: every strategy has a median within 0.07° of the joint fit's and a slightly lower mean, and the five strategies lie within 0.05° of each other in median and 0.3° in mean. What separates them is the cost, and `selected_pairs` is the fastest by a clear margin while it grows only the views it picks. The maintainer chose `selected_pairs` as the method, and the other four strategies and the setting that chose between them were removed after this sweep; the step as this draft describes it is that method.

## Judged by eye (2026-10-18)

**Question.** On the points where the step's normal and the stored or starting normal disagree, which does a person judging the photographs find right, and is there a rule on the step's own readings that picks the right one more often than turning every time?

**Setup.** The maintainer was shown 70 points across four review rounds, each with the grown normal beside the start or the stored normal, and decided 62 of them; the other 8 were left undecided. The judgement is by eye on the photographs, not against the stored normals: the stored normals were not used as truth here, and some of them were set to fit the point's position rather than the surface. The quantities scored are the step's turn from the start, as the report's tilt degrees give it, and the turn's ratio to σ (a), the weakest-direction σ with no pair halved, as the binding reports it in `sigma_weak_deg`.

| Points | Grown normal right |
|---|---|
| All 62 decided | 49 |
| Turn under 65° | 30 of 32 |
| Turn of 65° or more | 19 of 30 |
| Turn of 65° or more, ratio under 3 | 4 of 13 |
| Turn of 65° or more, ratio of 3 or more | 15 of 17 |

**Rules tried.**

- A rule that keeps the start on the ratio alone, whatever the turn, never beats turning every time, at any bar tried.
- Enlarging small patches to 12 px before the step made it worse: right on 12 of the 27 points it changed, against 19 with the patches as they were.
- The combined rule, keeping the start when the turn is at least a bound and the ratio is under a bar, was scored on all 62 points for bounds of 60°, 65° and 70° and bars from 2 to 3.5. It beats turning every time in every combination. At 70° and 3 it is right on 56 of 62, against 49: it fixes 8 wrong turns and loses 1 good one.

**On every ground-truth point.** With the rule at its defaults, the sweep of § [Sweep of all ground-truth points](#sweep-of-all-ground-truth-points-2026-10-17) run again keeps the start on 8 of 266 Seoul bull points, 12 of 382 Kerry Park points and 1 of 961 dino points, and every other point ends where it ended without the rule. Against the stored normals the median error goes from 19.96° to 19.04° on the bull, 7.83° to 6.87° on Kerry Park and stays 1.46° on the dino, and the share within 5° from 15.0% to 16.2%, 38.0% to 40.8% and 85.1% to 85.1%. The start is nearer the stored normal than the plane found on 6 of the 8 bull points, all 12 Kerry Park points and not on the dino point; the stored normals are not the truth here, as above. Of the judged points, the rule keeps the start on bull 180, 207, 221, 236 and 237, Kerry Park 58, 248 and 298 and dino 507, and turns bull 193, 199, 238, 242 and 268, Kerry Park 59 and 250 and dino 19.

**Caveat.** The thresholds were chosen on the same points they are scored on, so 56 of 62 is an optimistic figure for points not yet judged. That every combination near 70° and 3 beats turning every time is the stronger evidence that the rule helps.

**Reading.** Turns under about 65° were right almost every time, however weak their σ, so a gate on σ alone would hold back good turns. The wrong turns were large ones on weak evidence: the views were narrow, the score was flat in the tilt, and the growth drifted toward a plane seen edge on. The step keeps the start on that case alone, as `weak_turn_deg` 70 and `min_turn_sigma_ratio` 3 (§ [Turn and report](#turn-and-report)).

## Fewer renders per fit (2026-10-18)

**Question.** The Levenberg–Marquardt fits make about 94% of the step's renders. Can they make fewer without changing what the step finds?

**Changes.** Three changes, each measured on all 1609 points of § [Sweep of all ground-truth points](#sweep-of-all-ground-truth-points-2026-10-17) with the defaults, the weak-turn rule included.

- The render that scores a kept step also gives the residual the next iteration linearises at, where before the next iteration rendered the same parameters again (§ [The optimiser](#the-optimiser)). The residual is a function of that render and the frozen blocks alone, so every point's result is the same, bit for bit, on every point of the three datasets.
- An iteration tries at most three damped steps, where it tried eight. More than half of all fits end on an iteration whose steps all fail to raise the score, and such an iteration cost 11.5 renders on average, its four Jacobian columns and every try.
- A fit stops once a kept step raises the score by less than 1% of what its first kept step raised it, as well as under the absolute floor `min_improvement` (§ [The optimiser](#the-optimiser)). The step that falls under the share is kept, so a fit whose first step does nearly all its work ends after its second.

**Renders.** Counted over the whole sweep, the fits made 69.1 renders each before, 58.0 with the residual reused and 52.7 with three tries as well; the step made 5.11 million renders in all, then 4.31 million and 3.92 million, 23% fewer. The renders under the 1% stop were not counted.

**Others measured.** With three tries the fits ending without a rise cost 6.5 renders each, and two tries gave the same results within 0.02° in every median for 1% less time. Four tries moved the bull's median to 19.59°. Against the residual reused with eight tries, in the same runs: a `min_improvement` of 10⁻⁵ lost a judged point (54 of 63), and 10⁻⁴ took 23% less time but brought the bull to 15.4% within 5° and Kerry Park's median to 6.99°. The same stop at 0.1% of the first step, with eight tries, lost a judged point.

**Results.** The judged points are the 63 of § [Judged by eye](#judged-by-eye-2026-10-18) as the judgement script scores them against the normals judged right; "moved" counts the points more than 1° better or worse than before.

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Time alone | Time shared |
|---|---|---|---|---|---|---|---|
| before | 55 of 63 | 19.04°, 16.2% | 6.87°, 40.8% | 1.46°, 85.1% | – | 460 s | 926 s, 874 s |
| residual reused | 55 of 63 | 19.04°, 16.2% | 6.87°, 40.8% | 1.46°, 85.1% | 0 / 0 | – | 895 s, 840 s |
| and three tries | 55 of 63 | 19.06°, 16.2% | 6.88°, 40.8% | 1.45°, 85.1% | 24 / 18 | 387 s | 829 s, 772 s |
| and the 1% stop | 55 of 63 | 18.85°, 15.8% | 6.94°, 41.1% | 1.46°, 85.1% | 56 / 51 | 356 s | –, 671 s |

"Time alone" is the sum of the step's times over the 1609 points with the three datasets run at once on one machine; "time shared" is the same sum in two runs that each ran six or seven forms at once, so only the ratios within a column hold. Alone, the step takes 16% less time, a median of 167 ms a point against 206 ms; shared, the residual reused takes 3% to 4% less and three tries 7% to 8% less again, 10% to 12% in all. The 1% stop was timed alone in a run of its own, after a fresh run of three tries alone that took 391 s, a median of 171 ms: against that run it takes 9% less time, a median of 155 ms, and 23% less than before; in the shared run that holds both, it takes 13% less than three tries. The time falls less than the renders do because the render the reuse saves is at a tilt whose projection is kept, which samples the photograph again but does not project the grid.

**Reading.** Reusing the residual changes nothing and saves the cheapest renders. Three tries changes the fits that needed a fourth to eighth try to rise; they end where the fit stops a little earlier, and the points that move do so in both directions in about equal numbers, with every measure of the bar the same within 0.02°. The 1% stop ends fits that are still rising slowly. Against three tries it moves 43 points more than 1° better and 50 worse; the judged points stay at 55 of 63, the bull's median falls 0.21° and its share within 5° by 0.4 points, and Kerry Park's median rises 0.06°.

## A near-to-far selection (2026-10-18)

**Question.** The selection then picked the widest view first and the rest by predicted information, and grew every pick from the starting plane. On slow points the wide picks failed one after another: dino 100 refused 17 picks before 6 narrower views passed, and dino 321 matched 29 views to keep 6. Does picking near to far, growing each later pick from the plane the narrower views reached, keep the bar with fewer refused picks and less time?

**The angle evidence.** Over 9029 picks of the sweep of § [Sweep of all ground-truth points](#sweep-of-all-ground-truth-points-2026-10-17), every one grown from the starting plane, by its angle from the reference:

| Angle from the reference | Passes the floor | Own-plane error, median | σ, median |
|---|---|---|---|
| 0–10° | 97–98% | 9–30° | 13–40° |
| 10–15° | 94% | 5.2° (dino 2.3°) | 6.7° |
| 15–20° | 88% | 3.6° | 3.9° |
| 20–60° | 74–82% | 2–3.6° | 1.6–3.2° |
| 60°+ | 65% | 1.3° | 1.2° |

A wide view fails mainly when the start is far off: with a start within 15° of the stored normal, views at 20° to 60° pass 92% to 98% of the time, and with a start 30° or more off, 61% to 76%. A view near 10° to 20° passes almost always and gives a plane good to a few degrees, which is within capture for the next, wider view.

**Setup.** The selection of § [The selection](#the-selection) was built beside the old one, behind a temporary setting, and both were run on all 1609 points of the three ground truths, from the mean viewing direction, with every other setting at its default. The ladder ran at the proposed settings (first rung 12° within 10° to 20°, ratio 1.6) and at three settings around them: the first rung at 15°, and the ratio at 1.4 and at 2.0. "Judged right" is the 63 points of § [Judged by eye](#judged-by-eye-2026-10-18) as the judgement script scores them, and "moved" counts the points more than 1° better or worse than the old selection. "Time shared" is the sum of the step's times with all five forms run at once on one machine; "time alone" is each form run by itself, its three datasets at once.

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Time shared | Time alone |
|---|---|---|---|---|---|---|---|
| widest first, from the start | 55 of 63 | 18.85°, 15.8% | 6.94°, 41.1% | 1.46°, 85.1% | – | 535 s | 335 s, median 144 ms |
| ladder, 12°, 1.6 | 54 of 63 | 18.73°, 16.5% | 6.64°, 42.7% | 1.38°, 85.7% | 98 / 85 | 517 s | 321 s, median 143 ms |
| ladder, 15°, 1.6 | 55 of 63 | 18.73°, 16.2% | 6.60°, 42.9% | 1.37°, 86.5% | 102 / 79 | 520 s | 324 s, median 141 ms |
| ladder, 12°, 1.4 | 54 of 63 | 18.73°, 16.2% | 6.64°, 42.7% | 1.43°, 86.0% | 101 / 91 | 512 s | – |
| ladder, 12°, 2.0 | 54 of 63 | 18.75°, 16.2% | 6.49°, 44.5% | 1.38°, 86.3% | 98 / 78 | 511 s | – |

**Picks.** Per point, the median number of picks is 6 in every form; the 90th percentile falls from 8 to 7 and the most from 29 to 17 or 18. Over the sweep the old selection made 8473 picks, 17.7% of them refused; the ladder at 15° made 8249, 11.1% refused (12°: 8271, 10.9%; ratio 1.4: 8269, 10.3%; ratio 2.0: 8164, 11.0%). By rung, at 15° and 1.6, the share refused is 7.9% in the first rung, 8.5% in the second, 10.5% in the third, 16.3% in the fourth, 12.5% in the fifth and 20.4% beyond; at 12° the first five are 6.4%, 7.6%, 10.0%, 15.9% and 14.8%, and 18.5% beyond. Of the later picks grown from the combined plane, 83% passed there; of the 1081 refused there, 300 passed when grown again from the starting plane.

**The slow points.** Old selection against the ladder at 15°, run alone: dino 100 from 23 picks (17 refused), 1.01° and 2.45 s to 9 picks (3 refused), 0.75° and 0.90 s; dino 321 from 29 (23 refused), 2.67° and 3.06 s to 17 (11 refused), 2.31° and 2.35 s; dino 83 from 24 (18 refused), 2.16° and 2.75 s to 5 (none refused), 1.98° and 0.42 s; dino 85 from 15 (10 refused), 0.89° and 2.04 s to 10 (4 refused), 0.96° and 1.46 s. The slowest points under the ladder are dino 321, 716, 954, 99 and 85, at 1.5 to 2.4 s, where under the old selection they were dino 321, 83, 100, 85 and 87, at 1.8 to 3.1 s.

**The judged point lost.** At 12° the ladder is wrong on dino 10, whose judged-right and alternative normals are 3.8° apart: the old selection's first pick, a 13.7° view grown from the starting plane, was refused, where the ladder grows it from the plane of an 11.8° view and keeps it, and the plane ends 2.3° from the judged-right normal and 1.5° from the stored one, with σ 1.7°. At 15° the 13.7° view is again picked first from the starting plane and refused, and the point is as before.

**Reading.** The ladder holds the judged points at 15°, lowers every dataset's median, by 0.12° on the bull, 0.34° on Kerry Park and 0.09° on the dino, and raises every share within 5°; it refuses about 40% fewer picks and caps the worst points' picks at 17 rather than 29. It saves 2% to 4% of the time over the whole sweep, much less than on the slow points, because on most points both selections make about six picks. The ratio changes little between 1.4 and 2.0. The ladder at 15° and 1.6 replaced the old selection, which was removed, with the setting that chose between them.

## A looser σ target (2026-10-18)

**Question.** The selection stopped once the combined weakest σ was at most 0.5°. Does a looser target keep the bar while growing fewer views?

**Setup.** The near-to-far selection of § [The selection](#the-selection) was run on all 1609 points of the three ground truths, from the mean viewing direction, with `target_sigma_deg` at 0.5°, 1.0° and 1.5° and every other setting at its default. The three runs ran at once on one machine, so only the ratios between their times hold. "Within 0.1°", "better" and "worse" compare each point's error against its stored normal with its error under the 0.5° target.

**Results.**

- **Judged points.** The points of § [Judged by eye](#judged-by-eye-2026-10-18) judged right stay at 55 of 63 under all three targets.
- **Time.** The step takes 423 s at 0.5°, 357 s at 1.0° and 303 s at 1.5°: 28% less at 1.5° than at 0.5°.
- **`seoul_bull_sculpture`.** Every quantile is the same at 1.5° as at 0.5°, and 95% of the points are within 0.1° of their error at 0.5°.
- **`kerry_park`.** The quartiles and the 90th percentile are within 0.08° of those at 0.5°. 82% of the points are within 0.1°; 4% are better by more than 0.5° and 5% worse by more than 0.5°.
- **`dino_dog_toy`.**

  | Target | 25th percentile | Median | 75th percentile | 90th percentile | Under 1° |
  |---|---|---|---|---|---|
  | 0.5° | 0.75° | 1.37° | 2.81° | 6.4° | 35% |
  | 1.5° | 0.86° | 1.55° | 2.90° | 6.7° | 30% |

  Per point, 45% are within 0.1°; 7% are better by more than 0.5° and 13% worse by more than 0.5°.

**Two other stops, simulated from the traces.** Each pick's combined plane in the sweep's traces gives where the selection would have ended had it stopped there.

- **An oracle.** About half of the points are already within 0.5° of their final plane after their first join. A stop that knew when each point had reached its final plane would grow only 32% to 44% of the views the selection grew.
- **A small move of the combined plane.** Stopping once a join moves the combined plane by less than a given angle, set to save as many growths as the σ target does, costs about as much error as the σ target on `dino_dog_toy` and more on `kerry_park`.

**Reading.** At 1.5° the step takes 28% less time, keeps the judged points and the bull's quantiles, and moves Kerry Park's by less than 0.1°. On `dino_dog_toy` each quantile shown is 0.08° to 0.33° higher, and more points are worse by over 0.5° than better. The dino's stored normals come from a candidate ground truth, so its shifts of 0.1° to 0.3° may lie within the stored normals' own error. `target_sigma_deg` is 1.5° since this measurement.

## An analytic Jacobian (2026-10-18)

**Question.** The fits' Jacobian was a forward difference, one render per parameter per iteration, and the fits take about three quarters of the step's time. Can the Jacobian be read from the render the residual is frozen at instead, with no render of its own, and keep what the step finds?

**The form measured.** A sample's value moves with its view's shift by its image gradient: the derivative of the bilinear read at the sample's level position, from the same four taps as the value, divided by the level's scale so that it is per full-resolution px. It moves with the tilt by that gradient times how its pixel moves as the plane turns about the anchor: the sample's world point slides along its reference ray to the turned plane, and the pixel follows through the view's pose and its lens model's Jacobian. Each scored channel of a block goes through its normalisation by the chain rule, `dû = (dc − û (û·dc))/‖c‖` for the centred samples `c` with `û = c/‖c‖` and `dc = dv − mean dv`. A sample off the photograph has no row, and a tap clamped to the edge pixel along an axis gives zero gradient along it. The pair fit takes the first view's normalised derivative less the second's, the first moving with the tilt only. On synthetic planes seen through a pinhole, a radial lens and a fisheye, the worst column of each fit's Jacobian lay 0.1% to 1.7% from a central difference of renders. The difference itself is no closer than about 1%, because a sample's position is read in `f32`. The pixel's tilt derivative and the derivative through the normalisation each matched a difference to 10⁻⁶. On the view fits of 12 points of each dataset, the tilt columns lie a median of 3% to 4% from the forward difference the fits use. The shift columns lie a median of 1% to 2% from a central difference with a 10⁻³ px step, and 14% to 19% from the fits' forward difference with its 0.05 px step: that step is a secant across the pixel edges where a bilinear read's slope jumps.

**Setup.** All 1609 points of the three ground truths, from the mean viewing direction, at the defaults. The step before and the analytic form were each timed alone, one after the other. To show how far the bar moves under a change that should not matter, the forward difference was also run with its steps 0.8 and 1.2 times as long, with its tilt step 1.5 times as long, and with `initial_damping` 10% higher and 10% lower. "Moved" counts the points more than 1° better or worse than before.

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Time alone |
|---|---|---|---|---|---|---|
| before | 55 of 63 | 18.73°, 16.2% | 6.58°, 43.5% | 1.55°, 86.1% | – | 231 s |
| the five changes of the forward difference | 55 of 63 in each | 18.42° to 18.66°, 16.2% to 16.9% | 6.44° to 6.63°, 42.4% to 44.0% | 1.54° to 1.57°, 85.5% to 86.1% | 42 to 78 / 35 to 81 | – |
| analytic | 53 of 63 | 18.11°, 15.4% | 6.53°, 41.6% | 1.55°, 84.9% | 127 / 138 | 171 s |
| analytic, damping 10% higher, lower | 54, 53 of 63 | 18.11°, 15.4% | 6.54°, 41.4% to 41.6% | 1.54° to 1.55°, 84.9% | 126 to 130 / 136 to 140 | – |
| forward-difference tilt, analytic shift | 54 of 63 | 18.18°, 15.4% | 6.54°, 41.9% | 1.57°, 85.0% | 129 / 138 | – |
| analytic tilt, forward-difference shift (0.05 px) | 54 of 63 | 18.55°, 15.8% | 6.60°, 41.4% | 1.55°, 85.5% | 56 / 56 | 193 s |
| the same at 0.04 px, 0.06 px | 55, 53 of 63 | 18.36° to 18.45°, 15.8% to 16.5% | 6.56° to 6.60°, 42.1% to 42.9% | 1.55° to 1.56°, 85.5% to 85.8% | 78 to 79 / 72 | – |
| analytic, gradient over ±0.25 px, ±0.5 px | 53, 52 of 63 | 18.65° to 18.87°, 16.9% to 17.3% | 6.38° to 6.42°, 42.4% to 43.5% | 1.45° to 1.50°, 85.6% | 182 to 205 / 166 to 189 | – |

The last row reads each sample's gradient as a central difference of the bilinear read over that distance, a smoothed slope, in place of the read's own derivative. "Time alone" is the sum of the step's times over the 1609 points with the three datasets run at once on one machine: the step before takes a median of 103 ms a point, the analytic form 76 ms and the analytic tilt with the forward-difference shift 88 ms.

**Renders.** On 60 points of each dataset, a window stage made 50 to 57 renders before, 11 to 12 with the analytic form and 30 to 35 with the analytic tilt and the forward-difference shift; a whole-grid stage 37 to 38, 9 to 10 and 22 to 25; a pair fit 88 to 93, 18 to 22 and 52 to 57. The time falls by less than the renders do: the window stages' own time on those dino points was 5.1 s before and 3.1 s with the analytic form. The analytic form reads each sample's gradient at every render a fit scores, and each Jacobian still makes one pass over the samples, through the lens model's Jacobian and the normalisation's chain rule.

**Reading.** The analytic form takes 26% less time and its Jacobian is accurate, but it moves about twice as many points by more than 1° as any of the five changes of the forward difference. On all three datasets it brings the share within 5° under the range those changes span, and it loses two judged points: bull point 207 moves from 1.3° from the normal judged right to 69° from it, and dino point 507 to within 3° of the normal judged wrong. The shift columns cause the difference. Forward-difference tilt with analytic shift behaves as the analytic form does. Analytic tilt with forward-difference shift moves no more points than the five changes, but at the default step it has 54 judged points and Kerry Park's share within 5° is under their range; it takes 16% less time. The fits do better with the forward difference's secant over 0.05 px than with the bilinear read's exact slope. A smoothed slope keeps each share within 5° at or over the range's low end and lowers the dino's median to 1.45° to 1.50°, but it has only 52 or 53 judged points. 49 of the 63 judged points are ones on which the judged run's own grown normal was right, so a form that moves many points loses more of them than it gains whether or not it is better; the shares within 5° are read over every point.

**A gradient continuous across pixel edges.** The bilinear read's slope jumps at every pixel centre, so the forms below read the gradient another way, with the analytic tilt columns in every one, on all 1609 points with the same setup. The standard Lucas–Kanade gradient is a bilinear read, at the sample's position and with the value's weights, of gradient images of the pyramid level: central differences (`[−1, 0, 1]/2`), Sobel (that, averaged across the other axis by `[1, 2, 1]/4`) or Scharr (by `[3, 10, 3]/16`). Each is computed from the 4×4 pixels about the sample, which give the gradient image's values at the four taps, so it equals a read of stored gradient images and stores none. The smoothed slope reads a central difference of the bilinear read over ±h px. The blurred render reads the value itself from the level blurred by `[1, 6, 1]/8` or `[1, 2, 1]/4` in each axis, with the exact slope of that read.

Used for the information as well as the steps, every smoother gradient lowers the information, since it is smaller wherever the photograph has detail a pixel or two across: the median weakest σ on the bull rises from 6.7° to 11.4° to 12.6°, and the dino's picks per point from 3.5 to 4.6 to 4.9. The step then picks more views and stops later, which costs time and moves the points the selection reaches. In the other forms the information is read through the bilinear read's slope, as the forward difference's 0.05 px step nearly read it, and σ and the picks stay near those before: on the bull a median weakest σ of 6.6° against 6.7°, and on the dino 3.4 picks per point against 3.5.

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse |
|---|---|---|---|---|---|
| forward difference, before | 55 of 63 | 18.73°, 16.2% | 6.58°, 43.5% | 1.55°, 86.1% | – |
| bilinear slope (the analytic form above) | 53 of 63 | 18.11°, 15.4% | 6.53°, 41.6% | 1.55°, 84.9% | 127 / 138 |
| central differences, Sobel, Scharr, for steps and information | 50, 52, 51 of 63 | 16.87° to 18.50°, 15.8% to 16.9% | 6.36° to 6.53°, 42.1% to 42.7% | 1.42° to 1.43°, 85.8% to 86.3% | 256 to 278 / 229 to 243 |
| **Sobel for steps, bilinear slope for information** | **52 of 63** | **18.77°, 16.5%** | **6.15°, 43.7%** | **1.54°, 86.0%** | **220 / 222** |
| central differences for steps, bilinear for information | 50 of 63 | 17.40°, 16.2% | 6.33°, 42.7% | 1.55°, 86.4% | 223 / 188 |
| Scharr for steps, bilinear for information | 51 of 63 | 18.71°, 15.8% | 6.39°, 42.1% | 1.55°, 85.7% | 216 / 221 |
| Sobel for the shift columns only, bilinear for the tilt's | 46 of 63 | 18.92°, 15.8% | 6.52°, 42.4% | 1.54°, 86.6% | 215 / 187 |
| Sobel for the tilt columns only, bilinear for the shift's | 51 of 63 | 18.72°, 16.2% | 7.00°, 40.8% | 1.54°, 85.3% | 212 / 219 |
| smoothed slope over ±0.1, ±0.15 px | 53, 53 of 63 | 17.86° to 18.34°, 15.8% to 16.9% | 6.31° to 6.42°, 42.9% to 43.2% | 1.55°, 85.1% to 85.6% | 148 to 150 / 133 to 141 |
| smoothed slope over ±0.25 px | 54 of 63 | 18.76°, 16.2% | 6.26°, 43.2% | 1.55°, 85.4% | 168 / 161 |
| smoothed slope over ±0.35, ±0.5 px | 50, 50 of 63 | 18.96° to 19.49°, 16.5% to 17.3% | 6.42°, 42.4% to 42.9% | 1.52° to 1.53°, 85.7% | 176 to 186 / 179 to 183 |
| blurred render, `[1, 6, 1]/8`, `[1, 2, 1]/4` | 47, 44 of 63 | 17.95° to 18.87°, 15.8% to 16.2% | 6.59° to 7.71°, 39.3% to 42.1% | 1.51° to 1.55°, 85.7% to 85.8% | 245 to 312 / 243 to 328 |

The same forms by their shares under 1°, 2° and 5° and their quartiles:

| Form | Bull < 1°, < 2°, < 5°; 25th–75th | Kerry Park | Dino |
|---|---|---|---|
| forward difference, before | 1.5%, 3.8%, 16.2%; 7.73°–37.44° | 8.6%, 17.0%, 43.5%; 2.70°–26.18° | 30.3%, 61.4%, 86.1%; 0.86°–2.90° |
| bilinear slope | 1.1%, 3.0%, 15.4%; 7.70°–36.11° | 7.9%, 17.8%, 41.6%; 2.71°–25.62° | 30.4%, 62.1%, 84.9%; 0.87°–2.93° |
| Sobel for steps, bilinear for information | 1.1%, 4.1%, 16.5%; 7.37°–37.53° | 8.4%, 18.3%, 43.7%; 2.43°–22.01° | 30.2%, 61.3%, 86.0%; 0.89°–2.86° |
| central differences for steps | 1.1%, 3.0%, 16.2%; 7.56°–35.51° | 8.6%, 18.3%, 42.7%; 2.45°–22.21° | 30.4%, 62.1%, 86.4%; 0.86°–2.82° |
| Scharr for steps | 1.5%, 3.4%, 15.8%; 7.44°–34.82° | 8.6%, 17.3%, 42.1%; 2.78°–19.49° | 31.2%, 62.0%, 85.7%; 0.87°–2.90° |
| smoothed slope over ±0.25 px | 1.1%, 3.4%, 16.2%; 7.63°–37.79° | 8.4%, 17.8%, 43.2%; 2.64°–24.92° | 30.8%, 61.9%, 85.4%; 0.87°–2.93° |

**The optimiser's settings under the Sobel gradient.** With the Sobel gradient for steps the fits were run with `initial_damping` at 10⁻⁴, 10⁻² and 10% higher, with two and five damped tries, with the relative stop at 0.5% and 2%, with `min_improvement` at 10⁻⁵, and with the step caps at 0.05 in the tilt and 1 px in the shift. Every one has 52 of 63 judged right except damping 10⁻² (51) and the tilt cap of 0.05 (49); every share within 5° stays within 16.5% to 16.9% on the bull, 42.4% to 44.2% on Kerry Park and 85.8% to 86.1% on the dino, and Kerry Park's median within 6.05° to 6.41°. Against the Sobel form at the defaults, damping 10% higher moves 2 points by more than 1° better and none worse, `min_improvement` at 10⁻⁵ 3 and none, the shift cap of 1 px 1 and 3, two tries 10 and 6, damping 10⁻⁴ 13 and 9, the 0.5% stop 15 and 6, the 2% stop 22 and 15, five tries 31 and 27, damping 10⁻² 35 and 26, and the tilt cap of 0.05 119 and 82; the five changes of the forward difference moved 42 to 78 and 35 to 81 against it. None of these settings is better than the defaults on every measure, so the defaults stay.

**The judged points.** Of the 63, 49 are points on which the judged run's own grown normal was judged right and 13 on which the stored normal was; the forward difference has 48 and 7 of them, the Sobel form 44 and 8. The Sobel form gains Kerry Park 62 (stored right; 25.1° from the stored normal before, 17.2° after) and loses four grown-right points. On dino 10, dino 389 and Kerry Park 144 it moves to the stored normal: 3.8° from it before and 1.5° after, 2.4° and 1.2°, 15.9° and 4.7°, with dino 10's two candidate normals 3.8° apart and the Sobel form's 2.4° from the one judged right and 1.5° from the other. On Kerry Park 309, whose stored normal is wrong, the forward difference ends near the normal judged right with two views joined and σ 15.9°; the Sobel form joins one view, with σ 93°, and ends 82° from it and 65° from the stored normal; both report low confidence. The smoothed slope over ±0.25 px loses Kerry Park 309 alone.

**Time.** The Sobel form and the step before were timed as a matched pair, each run alone with its three datasets at once, in the order before, Sobel, before, Sobel, with other work using five of the machine's 32 threads throughout: the step before took 369 s and 360 s, a median of 157 ms and 155 ms a point, and the Sobel form 226 s and 228 s, a median of 101 ms and 99 ms, 37% less. In a run with both at once, where only the ratio holds, the Sobel form took 9% less time than the bilinear slope. The other forms were not timed alone.

**Reading.** No form reaches 55 judged points with every share within 5° inside the range of the five changes of the forward difference. The Sobel gradient for the steps, with the information read through the bilinear slope, keeps every share within 5° inside that range, lowers Kerry Park's median from 6.58° to 6.15° and its 75th percentile from 26.2° to 22.0°, leaves the bull's and the dino's medians within 0.04°, and is stable under small changes of the optimiser's settings, which move a few to a few tens of points where the five changes of the forward difference moved 35 to 81; it takes 37% less time than the forward difference. It has 52 judged points right, three fewer, losing grown-right points by moving toward the stored normal on three of them. The smoothed slope over ±0.25 px has 54 judged points and the dino's share within 5° just under the range, at 85.4%. The analytic tilt with the forward-difference shift (above) has 54 and Kerry Park's share under the range, and takes 16% less time than before. The fits use the Sobel gradient for their steps and the bilinear slope for the information.

## Non-goals

**Moving the centre.** The step turns the patch about its centre, as the other normal steps do, and does not move it. It writes the keypoints of the views that joined and of the shift-only views that reached the floor (§ [The keypoints the step writes](#the-keypoints-the-step-writes)); whether it should also re-triangulate the centre from them is an open question.

**Per-cell one-dimensional constraints.** A cell over stripes pins its shift across the stripes only. Using such constraints was discussed but is not part of this method.

## Open questions

- **The calibration of σ.** Among the points the sweep reports confident, the error is over twice σ on 51% to 86% of them, and on the sixteen points it is over twice σ on about half. Whether τ should be raised, read from the data, or σ reported with a floor, and how the pairs' reuse of views should enter it.
- **Points where every view is weak.** On dino 1 the six views' own planes are each so poorly determined (σ 15° to 171°) that their weighted mean is 4.6° off, where fitting them together reached 0.48° (§ [Measurement without the joint fit](#measurement-without-the-joint-fit-2026-10-17)). One idea, not built: once each view's shift is fixed, fit the tilt alone to the sum of the views' robust scores, so that the views' scores are summed rather than their quadratic models. Whether to do that, and how the step would tell such a point from one where the mean is right, is open.
- **The retry from the starting plane.** A later pick refused from the combined plane is grown again from the starting plane; on the sweep of § [A near-to-far selection](#a-near-to-far-selection-2026-10-18) that retry passed on 300 of 1081 such picks, 28%, and costs up to two growths each time. Whether the retry is worth its cost, or whether a refused pick should go straight to the next view of its rung, is not measured.
- **The ladder's settings.** The first rung at 15° held the judged points where 12° lost one, and the ratios 1.4 and 2.0 moved the dataset medians by 0.15° or less; whether a ratio read from the spread of the track's angles, or a rung that ends at a fixed set of angles, does better is open.
- **The optimiser.** Levenberg–Marquardt on the normalised residual is built (§ [The optimiser](#the-optimiser)); the prototype's pattern search took larger first steps, up to 0.08 tilt units and 1 px, and could leave a basin a local fit stays in. Whether the built fit reaches what the pattern search reached on the prototype's points is not measured, and its three settings were chosen on synthetic scenes only.
- **A textured spot off the anchor.** Stage 1 is centred on the anchor because the keypoint-implied shift is exact only there. Where the centre lacks texture, which the maintainer expects on some points, the first window needs a spot elsewhere, and the shift at that spot depends on the plane. One proposal, not built: pin the anchor at the keypoint and search the tilt at stage 1, so that a spot `r` from the anchor is matched at the shift the tilt implies there. Whether to do that, and how a spot is then chosen, is open; the multi-spot rule and the spot candidates of the first round no longer apply under the keypoint start.
- **The window schedule.** Whether these fractions of the grid's side are the right schedule.
- **The agreement numbers.** Whether τ = 0.01 is too cautious: it scales every σ, so it sets how many estimates agree, when the selection stops and how often the report says low confidence. Whether 3σ and the 0.5 floor are the right bars; they were set on four points.
- **Narrow-view points.** On a point whose views all lie within about 10° of the reference, the plane has low precision (933: 2.62° off, σ 3.5°, against the 2×2 Grid Plane Normal's 1.16°). Whether to hand such points to a cell-based step.
- **A view's keypoint in a wide view.** Image 62's keypoint-implied shift differs from where its single-view growth settled by about 6 px in y; whether its keypoint is off in that 80° view or the shift at the window's spot depends on the tilt there is not known, and the widened search is what covers it. Under the anchor-centred window the second cause no longer applies to stage 1.
- **Blur matching the other way.** A view finer than the reference is read from the pyramid level the bilinear-mip rule picks at the starting plane, once per view. Whether one level chosen at the start is enough for a view whose footprint changes much as the plane turns, and whether the reference should instead be blurred to the view's sampling, is open.
- **The weak-turn rule's thresholds.** 70° and 3 were chosen on the same 62 points that score them (§ [Judged by eye](#judged-by-eye-2026-10-18)), and every combination near them also beat turning every time; whether they hold on points not yet judged, and whether σ (a) is the right σ once its calibration is settled, is open.
- **What the step reports.** σ along the weakest and strongest directions and the low-confidence verdict are decided; which of the two σ the verdict reads is not. The rest of `GrownPlaneSequence` is what the step knows and records today, and is open to change once a person reading it through the binding says what is missing or unused.
- **The pairs' rules.** Whether the widest pairs, each view in one pair, up to three, are the right pairs, or whether a pair should lie on opposite sides of the reference or be wider than the widest view's angle from the reference; whether halving a pair's information is the right discount for its reuse of views, and whether the plane should be the mean with that discount rather than with the full information.
- **The pairs' grid.** Whether a pair is compared on the reference's working grid, as built, or on a grid of its own, for example one centred on the bisector of the two views' rays, where both renders are foreshortened alike.
- **The selection's model.** The predicted information is `g·(α d dᵀ + β e eᵀ)` with `α` and `β` read from the views matched so far; whether a model that reads the view's texture as well as its geometry, or the reference's information along each direction, picks better, and what the target σ and the budget should be once τ is settled.
- **The centre after the write.** A depth error shows up as the per-view shifts, so the written keypoints can imply a different centre than the patch's. Whether the step should re-triangulate the centre from the written keypoints before the readback, or leave it to the next fit, is not decided. The step's plane passes through the anchor and the turned patch's through its centre, so the two planes also differ in depth wherever the anchor is not the centre.
- **A bound on the write.** A fit keeps a sighting at its seed when its kernels would move it further than `max_shift_px` on the patch's plane, and records the refusal in `walked_px` and `walked_to`. Whether this step should bound its write the same way, and what it measures the move from, is open; the size of the moves on real tracks has not been measured.
