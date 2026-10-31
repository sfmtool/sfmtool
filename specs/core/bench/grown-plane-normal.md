# Grown Plane Normal

A patch in a patch-based reconstruction is a small oriented square standing in
the world at a 3D point, and its normal says which way the square faces. Each
photograph that sees the point is resampled onto the square through that
plane, so a wrong normal stretches and shears every photograph's picture of the
patch against the others, blurs the patch's stored bitmap and misleads every
reading taken from it afterwards. The grown-plane normal estimates the normal by
fitting the plane to the photographs themselves. One photograph is the
reference. For each of a few other photographs, the step finds the plane under
which that photograph's picture of the patch agrees with the reference's, by
normalised cross-correlation (ZNCC), starting from a small window at the point
and growing it to the whole patch. It combines those photographs' planes into
one, turns the patch to it, and reports how precisely the photographs determine
it. It moves neither the patch's centre nor any of the point's image positions.

The step is a bench step: it takes an editable track at the track stage, as
the bench's other normal steps do ([editable-track.md](editable-track.md)
§ "Estimating the normal"), and gives the track back turned with a report. It is
Track View's *Grown Plane Normal* ([../../gui/track-view.md](../../gui/track-view.md)
§ "The toolbar") and the wire's `fit_bench_track_normal` with `method`
`grown_plane`, both at the default options, and it is the per-point step of
`bench::refine_normals` ([refine-normals.md](refine-normals.md)), which turns
every finite point of a reconstruction and which the viewer's *Refine All
Normals*, `sfm xform --refine-normals`, `inspect --strips`, `compare --strips`
and `sfm embed-patches` run. The measurements behind its design and its
defaults are in
[grown-plane-normal-measurements.md](grown-plane-normal-measurements.md).

In outline, for the views it picks:

- **Within a view** it registers a small window at the patch's anchor first,
  where the view's own keypoint places the patch, and grows the window
  outward, fitting the plane's tilt and the view's image shift at each size,
  because a small window tolerates a large error in the plane. Each view's
  growth ends at that view's own plane, with its information about the plane.
- **It trusts the keypoints.** It does not search a view for another place, so
  a caller whose keypoints may be off fits them first with a separate step.
- **It grows only a few views**, picked one at a time near to far by what their
  geometry predicts they say about the plane, until the plane is precise
  enough. It combines their planes by a robust information-weighted mean,
  never by fitting the views together, then matches the widest pairs of kept
  views against each other and adds their planes to the mean.
- **It turns the patch** to the plane found whenever at least one view joined,
  unless the turn is large and the evidence for it weak, when it keeps the
  starting normal. It fits the shift alone of every view it did not pick, and
  reports each view's shift and score.

## Why a separate step

`grid_plane_normal` ([../../drafts/piece-gated-grid-normal.md](../../drafts/piece-gated-grid-normal.md))
measures the normal by parallax: it renders each view once through the current
plane, registers each of nine cells by a shift search within a small bound,
gates each cell in each view by a robust affine fit, triangulates the cells and
fits a plane through them, then turns the patch and repeats. Each part of that
chain needs the starting plane to be near enough that every cell of a view
lands within the shift bound and that each cell's picture in the view differs
from the reference's by little more than a shift. In a wide view a wrong plane
breaks both: the cells far from the centre are displaced past the bound, and
the cells whose picture is stretched or turned against the reference's fall
under the cell ZNCC bar.

This step replaces the nine-cell shift search, the per-view affine gate and the
cell triangulation with a direct fit of the plane to the photographs. It
reaches a wide view by growing the window from the anchor, where the plane's
error displaces the picture least, so each stage starts within capture of the
next (§ [Near to far within a view](#near-to-far-within-a-view)). It keeps the
nine cells only as the parts a whole-grid score is combined from
(§ [The robust cell score](#the-robust-cell-score)). It is a step of its own
and not an option of `grid_plane_normal` because it measures nothing that step
measures: it has no cell shift search on a fixed render and no triangulation.

The two share the reference rule, the working grid's size rule, the starting
frame and the final turn and readback. They differ on a weak estimate:
`grid_plane_normal` leaves the normal unchanged when it has no good estimate,
and this step turns to any plane at least one view joined and reports its σ,
unless the turn is a large one on weak evidence. Where `grid_plane_normal` ends
`Turned`, the two should agree; the cases this step is for are the wide views
`grid_plane_normal`'s gates refuse at the starting plane.

The name follows the glossary's **finite-difference normal** /
**grid-plane normal** / **grown-plane normal** entry
([../../GLOSSARY.md](../../GLOSSARY.md)), which names each bench normal step
for the evidence it reads: the estimate is a plane, and its support grows, both
the window each view is registered over and the set of views the plane is
combined from.

## Rust API

The step is in
[bench/normal/grown_plane.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane.rs),
re-exported from `sfmtool_core::bench`, with the plane-induced render and the
scores in
[grown_plane/render.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/render.rs),
the optimiser in
[grown_plane/solve.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/solve.rs),
the weighted mean, the robust combination and the agreement distance in
[grown_plane/combine.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/combine.rs),
and the selection, the wide pairs and the shift-only fits in
[grown_plane/selected.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/selected.rs).
It has the signature every bench normal step has
([bench/normal.rs](../../../crates/sfmtool-core/src/bench/normal.rs)), so
Track View and the wire call it the way they call the other normal steps.

```rust
/// How grown_plane_normal grows its windows, fits its planes and decides
/// which views it keeps. Defaults in § Parameters.
pub struct GrownPlaneNormalOptions {
    pub working_resolution: Option<u32>,
    pub spot_window: u32,
    pub window_fractions: Vec<f64>,
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
    /// Refuses a setting the step cannot run with, naming the first that
    /// fails: NormalError::BadGrownPlaneOption("zncc_tolerance").
    pub fn check(&self) -> Result<(), NormalError>;
}

/// What the step did, carried by NormalEstimate::GrownPlane(Box<_>).
pub struct GrownPlaneSequence {
    pub reference: Option<usize>,          // None: no reference, nothing tried
    pub working_resolution: u32,           // 0 with no reference
    pub window_half_widths: Vec<u32>,      // the stages' windows, grid px
    pub views: Vec<GrownPlaneView>,        // one per observation of the track
    pub trace: Vec<GrownPlaneTry>,         // every try, in the order made
    pub picks: Vec<GrownPlanePick>,        // the selection's picks, in order
    pub pairs: Vec<GrownPlanePair>,        // the pairs matched, widest first
    pub outcome: GrownPlaneOutcome,        // Turned, NoneJoined or WeakTurn
    pub normal: Option<[f64; 3]>,          // the final plane, under WeakTurn too; None: none joined
    pub turn_deg: Option<f64>,             // the turn from the start to it, made or not
    pub precision: Option<GrownPlanePrecision>,
}
// joined(), left_out(), shift_only(), matched(), pairs_kept(), agreement(),
// turned(), turn_sigma_ratio(); Display (below).

/// How the step ended. NoneJoined: no plane, the track as given.
/// WeakTurn: a turn of at least weak_turn_deg under min_turn_sigma_ratio
/// times sigma_weak_deg, not made; the track as given, the plane reported.
pub enum GrownPlaneOutcome { Turned, NoneJoined, WeakTurn }

pub enum GrownPlaneView {
    Reference,
    /// Its kept growth's plane's distance from the final plane and its score
    /// there; shift moved to the final plane (its growth's own shift under
    /// WeakTurn), and grid_zncc at the final plane, equal to held_zncc.
    Joined {
        shift: [f64; 2],
        grid_zncc: f64,
        distance: Option<f64>,
        held_zncc: Option<f64>,
    },
    /// Not picked: its shift alone fitted at the final plane from its
    /// keypoint-implied shift, and its score there.
    ShiftOnly { shift: [f64; 2], grid_zncc: f64 },
    /// reason NoKeypoint: no keypoint-implied shift, never grown.
    LeftOut { reason: GrownPlaneRefusal, distance: Option<f64>, held_zncc: Option<f64> },
    /// Not in, a second row in one image, no reference, or not picked by a
    /// selection that joined no view or whose turn was not made.
    NotAttempted,
}

pub enum GrownPlaneTiltStart { Starting, Combined }
pub enum GrownPlaneRefusal { UnderFloor, TooFar, HeldUnderFloor, StoppedEarly, ViewLeftOut, NoKeypoint }
pub enum GrownPlaneTryOutcome { Matched, LeftOut }

/// One try of one view, and the combined plane after it.
pub struct GrownPlaneTry {
    pub observation: usize,
    pub image: u32,
    pub angle_deg: f64,
    pub attempts: Vec<GrownPlaneAttempt>,  // from the combined plane, then from the starting plane
    pub outcome: GrownPlaneTryOutcome,
    pub combined_normal: Option<[f64; 3]>, // the picks' plane after this pick; None when left out
    pub combined_sigma_deg: f64,
}

/// One growth of one view against the reference, and how it was judged.
pub struct GrownPlaneAttempt {
    pub tilt_start: GrownPlaneTiltStart, // the plane its windows started from
    pub stage_zncc: Vec<f64>,   // stage 1 at the keypoint-implied shift, each later window, the whole grid's robust score
    pub shift: [f64; 2],
    pub normal: [f64; 3],       // the view's own plane
    pub grid_zncc: f64,         // NaN when it stopped early
    pub sigma_deg: f64,         // its own weakest σ; NaN when it stopped early
    pub refusal: Option<GrownPlaneRefusal>, // UnderFloor or StoppedEarly
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
    pub distance: Option<f64>,        // from the combined plane; None when refused under the floor
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
    pub sigma_after_deg: f64,         // NaN when not matched
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

`GrownPlaneSequence::agreement()` is how well the views agree at the final
plane: the mean `grid_zncc` of the views that joined it (`Joined`) or had their
shift fitted at it (`ShiftOnly`), a `NaN` score left out; the reference has no
score of its own and does not count. It is `NaN` when no view has a score, as
when no view joined. Under `WeakTurn` it is the joined views' mean at the plane
found and not turned to, since no view has its shift fitted then.
`bench::refine_normals` reports it per point and its mean over the turned
points. `turn_sigma_ratio()` is the turn over `precision.sigma_weak_deg`, 0
where that σ is not finite and `None` when no view joined, the quantity the
weak-turn rule bounds.

The report's sentence (`Display`) reads, for example, "5 of 5 views joined, 2
of 2 pairs kept; σ 0.56° weakest, 0.48° strongest (0.67° weakest with each view
once)"; the parenthesis appears only when halving the kept pairs' information
changes the weakest σ (§ [Wide pairs](#wide-pairs)). It adds ", N left out"
and ", N shift only" when there are such views, ", low confidence" when the
precision is low, and, under `WeakTurn`, "; normal left unchanged: large turn
on weak evidence, 19.8° at 1.5σ". When no view joined it ends ", normal left
unchanged", as in "0 of 3 views joined, 3 left out, normal left unchanged".

### Why it is shaped this way

The signature is the other normal steps', and the step ends as they do: it
turns the patch with `tilt_patch`, which moves no keypoint, and reads the track
back through `turn_and_read` in
[bench/normal.rs](../../../crates/sfmtool-core/src/bench/normal.rs), so a
caller can swap one step for another and compare `NormalReport`s. The settings
are `*Options` because a caller chooses them.

The report is per observation because a wrong normal is diagnosed by asking
which views were picked, which joined and from which plane, and why the others
were left out, and because a person checking the fit needs each view's shift
and score. The trace keeps every growth, so a pick grown again from the
starting plane shows both growths, and the picks and pairs say what the
selection chose and what each pair measured.

`NoneJoined` and `WeakTurn` are kept apart because a caller acts on them
differently: under `NoneJoined` there is no plane, and under `WeakTurn` there is
one, reported in `normal`, `turn_deg` and `precision` with its views, picks and
pairs, which the step judged too weakly supported to turn to. The precision is
a report rather than a gate, because a low-precision plane can still be much
better than the start (§ [Turn and report](#turn-and-report)); a consumer that
needs a precise normal reads σ and chooses its own bar.

The step refuses a track at the cluster stage, one with no patch, a track at
infinity and one with fewer than two `in` observations through
`normal_preconditions`, as every normal step does: a track at infinity faces
along its own bearing and has no plane to fit. It takes a `Progress`, reports a
phase for the whole step and one per try of a view, with finer detail phases
when `progress` is detailed (§ [Implementation notes](#implementation-notes)),
and is cancelled between stages, since it can run longer than a second.

### Example

```rust
use sfmtool_core::bench::{grown_plane_normal, GrownPlaneNormalOptions};
use sfmtool_core::progress::Progress;

let (turned, report) = grown_plane_normal(
    &track,
    &edited,
    &images,
    &GrownPlaneNormalOptions::default(),
    &Progress::none(),
)?;
println!("{report}"); // "6 of 7 views joined, 1 left out, 10 shift only, 3 of 3 pairs kept; σ …"
```

### Python binding

`sfmtool.bench.grown_plane_normal(track, edited, images, *, …)`
([bench/normal.rs](../../../crates/sfmtool-py/src/bench/normal.rs)) takes every
`GrownPlaneNormalOptions` setting as a keyword of the same name, defaulting to
the Rust default, and `fit`'s four keywords (`max_seed_offset_px`,
`max_cache_bytes`, `sigma_px`, `depth_likelihood_ratio_threshold`), which set
the reading the step ends with. It returns `(EditableTrack, report)`; the
report's `estimate` has `kind` `"grown_plane"` and carries the sequence's
fields, `agreement`, `turn_sigma_ratio`, the counts, and one dict per view, per
try, per pick and per pair; [editable-track.md](editable-track.md) § "Estimating
the normal" lists its fields. With `record_phases=True` the report also carries
the phases the step recorded. The evaluation scripts `grid_plane_eval.py` and
`grid_plane_detail.py` in
[scripts/track_at_pixel/normal_sources](../../../scripts/track_at_pixel/normal_sources)
run it with `--grown`.

## Theory

### One plane shared by all views

The model is one plane, the same for every view. The warp from the reference
view's working grid into any other view is the plane-induced map: each sample of
the grid is followed along its reference ray to the plane, and that point is
projected into the view's photograph through the camera's lens model, where the
photograph is sampled. There is no free per-view affine map; the plane predicts
each view's local affine map from its tilt alone.

**The reference** is the step's reference view, by the rule
`grid_plane_normal` and the fit use: the reference the track holds where its row
is `in` with a keypoint, else the reference-view rule's pick among the `in`
rows, one row per image. The other `in` views, one row per image, are the
candidates.

**The working grid** is the starting frame re-anchored on the reference keypoint
(`OrientedPatch::anchored_at_keypoint`), `W` samples a side, sample `(col, row)`
at grid position `(col + ½ − W/2, row + ½ − W/2)`: the reference render's grid
in `grid_plane_normal`. `W` is `working_resolution`, or with `None` that step's
rule: the reference view's footprint of the patch at the starting frame, about
one sample per photograph pixel, clamped to `R ..= 2R` for the patch resolution
`R`; at least `spot_window` either way.

**The plane** has two unknowns, a tilt `(a, b)` of the starting frame's normal
`n` along the frame's own axes `u` and `v`:

```text
normal = unit(n + a·u + b·v)
```

through the anchor, the centre of the starting frame slid onto the reference
keypoint. Every other view reads the same tilted patch slid onto its own
keypoint: its centre `q_i` is where the keypoint's ray meets the plane through
the patch centre that faces that view's camera, and each grid sample's place on
it is the sample's world point on the reference's plane moved by
`q_i − anchor`. `q_i` does not depend on the normal, so that move is worked out
once per view and holds at every tilt and every step of the fits; the image
gradients, the tilt Jacobian and the information are read at the moved points.
Reading each view through its own slid patch, rather than through the
reference's plane with the keypoint's gap as a 2D image shift, matches how the
bench cuts each view's tile, and measured better on the ground truths
([measurements](grown-plane-normal-measurements.md#each-view-read-through-its-own-slid-patch-2026-10-24)).

Each view adds its own image shift, two unknowns in image px, added to the
projected positions. The shift absorbs error in the point's position and in the
view's pose: a wrong depth along the reference ray moves every sample of a wide
view by about the same amount, several image px, which the tilt cannot express
and should not try to.

**The keypoint-implied shift.** Each view's own keypoint already places the
patch in that view. The shift it implies is the keypoint minus the projection of
the view's slid centre `q_i`, which is zero up to rounding, since the slid
centre lies on the keypoint's ray; only where the slide refuses the keypoint (a
ray at or past a right angle to the line of sight, as a fisheye can see) does
the view read the patch as it stands and start from the keypoint's gap from the
centre's projection. No tilt moves `q_i`, so the shift is the same at every
tilt. The step trusts it: every growth and every shift-only fit starts from it,
and the step never searches for another shift
(§ [Trusting the keypoints](#trusting-the-keypoints)). A view with no keypoint,
or one whose anchor does not project into it, has no implied shift; it is not a
candidate, is never picked, grown or fitted for its shift alone, and is
reported left out as `NoKeypoint` under every outcome.

**The render.** Every evaluation samples the photograph directly through the
current plane, rather than resampling a tile rendered once at the starting
plane, because a tile rendered at a wrong plane has already lost detail along
the axis that plane squeezed. Each view is read from one level of its pyramid
for the whole step: the level the bilinear-mip rule picks, `round(log2 σ_major)`
clamped to the pyramid's depth, for the larger singular value of the map from
the working grid at the starting plane into the view, at the grid's centre
(`patch_grid_jacobian`); level 0 where the centre does not project. A sample is
a bilinear read at that level, kept as a float, and is on the photograph when
all four of its taps are. The bench's tile renders (`render_tile`) choose a
level per sample or walk an anisotropic footprint and round to bytes; either
would make the score jump as the plane turns, which no Jacobian follows, so the
step keeps one level per view and the float sample. The reference is rendered
the same way at no tilt and no shift; its rays meet every plane where the
reference sees it, so its tile is the same at every tilt and is the template
every view is scored against.

### Near to far within a view

Under a plane that is wrong, a sample at distance `r` from the window's centre
is displaced in the view by about `‖A − I‖·r`, where `A` is the error in the
view's local affine map. A small window therefore tolerates a large error in
the map, and a map fitted over radius `r` is accurate enough to bring radius
about `2r` within capture of the next fit.

Every view the selection picks is grown against the reference from a plane:
the selection's first rung from the starting plane, `(a, b) = (0, 0)`, every
later pick from the combined plane of the views kept before it, and a pick of
the second rung, where that growth is refused, from the starting plane as well
(§ [The selection](#the-selection)). Every growth starts its windows at its
plane's tilt and at the view's keypoint-implied shift, which is the same at
every tilt since every plane passes through the anchor.

**Stage 1 at the anchor.** Stage 1 places a window of `spot_window` grid px
(7×7, half-width 3) on the grid's centre sample, `(W/2, W/2)` by integer
division: on an even grid that sample is half a sample right of and below the
anchor. Stage 1 reads the window's ZNCC at the view's keypoint-implied shift,
which is the first score `stage_zncc` records, and the growing windows fit the
shift from there together with the tilt.

The window is centred on the anchor because the keypoint-implied shift is exact
only there: every plane passes through the anchor, so at the anchor the right
shift does not depend on the plane, but at a spot `r` away it differs by the
plane error's displacement there, about `(A − I)·r`. The step grows from the
keypoint-implied shift rather than from a search for the first window's shift
because a 7×7 window on repeating texture matches equally well at more than one
shift, so a search picks a wrong peak as readily as the right one, where the
keypoint picks the peak it is on. On `dino_dog_toy` point 933, seen by three
views within 10° of the reference, a ±3 px search picked a wrong peak in every
view, 7 to 16 px from the implied shifts, although each implied shift was within
0.7 px of the true one; grown from the implied shifts, all three views join
([measurements](grown-plane-normal-measurements.md#growing-from-the-keypoint-implied-shift-2026-10-16-and-2026-10-17)).

**The growing windows.** Each later stage widens the window about the same
sample to the next half-width of the schedule, `window_fractions` of the
working grid's side `W` rounded to grid px, in increasing order, repeats and
any no wider than the first window's half-width dropped: 3, then 6, 10, 16 and
24 on a 48-sample grid, 3, 4, 6, 11 and 16 on a 32-sample grid, and 3, 5, 8 and
12 on a 24-sample grid; the half-width `W/2` covers the whole grid. Each stage
fits the tilt and the view's shift together for the best mean-channel ZNCC over
the window (§ [The optimiser](#the-optimiser)).

**The whole-grid stage.** After the last window, a final stage fits the tilt
and the shift for the best robust cell score over the whole grid
(§ [The robust cell score](#the-robust-cell-score)). The view's own plane and
its whole-grid score are those this stage ends at. A growth whose whole-grid
score is under `min_held_zncc` is refused (`UnderFloor`), and no search for
another shift follows it. A growth that reaches the floor is the view's
estimate; whether it is kept is the combination's verdict
(§ [Combining the views' planes](#combining-the-views-planes)).

**Stopping early.** A growth stops at a window stage, from the second window
fitted on (index 2 of `stage_zncc`, whose index 0 is the first window's
unfitted score), whose ZNCC is under `min_held_zncc` less 0.25, so under 0.25 at
the defaults. It is refused there (`StoppedEarly`) with no later window, no
whole-grid stage and no information read: its `stage_zncc` ends at the stage it
stopped at, and its `grid_zncc` and `sigma_deg` are `NaN`. It is otherwise
treated as a growth refused under the floor. The first fitted window is not
judged, because its score is often low on a growth that passes. The margin is
set so that almost no growth that passes the floor has a window stage under it:
on the ground-truth sweep, 24 of 5545 growths that passed had one, and 496 of
1562 refused growths stop at one
([measurements](grown-plane-normal-measurements.md#cheaper-growths-2026-10-20-and-2026-10-21)).

### The robust cell score

Wherever the step scores a whole grid (the floor and every held score, each
view's information, the whole-grid stage, the shift-only fit, the pair fit, and
the scores in the report), it scores the grid's nine cells rather than the grid
at once. The cells are a three-by-three split of the grid (`CellLayout`), rows
and columns at `[0, W/3, W − W/3, W]`. A cell where the reference is flat, its
spread under 0.5 grey levels in every channel, is not scored. Each live cell's
ZNCC is read as every ZNCC is
(§ [When an estimate agrees with the combined plane](#when-an-estimate-agrees-with-the-combined-plane)),
`−1` under the coverage bar, and the cells are combined by a one-sided Tukey
weighting:

- `m` is the median of the live cells' ZNCC;
- the scale is 1.4826 times the median absolute deviation from `m` of the cells
  at or above `m`, floored at 0.05, and the cut is 4.685 scales;
- a cell at or above `m` weighs 1, and one below weighs
  `(1 − ((m − z)/cut)²)²`, 0 past the cut;
- the score is the weighted mean of the cells' ZNCC; with no live cell it is
  `−1`.

The scale is read from the cells at or above the median only, because with
several bad cells the spread of all of them is so wide that nothing falls past
the cut. The reason for the score is a view that sees part of the patch's
texture differently: on `dino_dog_toy` point 528, images 45 and 46 match the
reference in the bottom two cell rows (0.67 to 0.81) but not the top row (0.40,
−0.06 and 0.06), whose texture the two views do not see, and a whole-grid ZNCC
refused them.

### The optimiser

Every fit is Levenberg–Marquardt on the photometric residual of mean- and
spread-normalised samples. For one channel, with `a` the reference's samples
and `b` the view's over the same samples, each less its mean, `â = a/‖a‖` and
`b̂ = b/‖b‖` are unit vectors and

```text
‖â − b̂‖² = 2 − 2·ZNCC(a, b)
```

so maximising the ZNCC is minimising that residual. A block of samples (a
window, or one cell) contributes its channels' residuals scaled by `√(w / C)`
for its weight `w` and its `C` scored channels, so its residual sum of squares
is `w · (2 − 2z)` for its mean-channel ZNCC `z`. A window is one block of
weight 1; a whole grid is its live cells, each weighted by its Tukey weight
over the weights' sum. Either way the residual sum of squares is
`S = 2 · (1 − score)` for the score being fitted.

The residual is frozen at the parameters it is linearised at: which samples are
on the photograph, which channels are scored, and the cells' Tukey weights,
which are read again at the start of every iteration, as an iteratively
reweighted fit. Only the view's samples move with the parameters.

**The Jacobian** is analytic, read at the render the residual is frozen at, and
costs no render of its own. A sample's value moves with the view's shift by its
image gradient there, divided by the level's scale so that it is per
full-resolution px. It moves with the tilt by that gradient times how its pixel
moves as the plane turns about the anchor: the sample's reference ray is fixed,
so its world point `o + λ·d` slides along the ray with
`λ = ((A − o)·n)/(d·n)`, and the pixel follows through the view's pose and its
lens model's Jacobian, analytic for the perspective family and the θ-map
fisheyes and a central difference of the projection for the others. Each scored
channel of a block then goes through its normalisation by the chain rule: for
the centred samples `c = v − mean v` and `û = c/‖c‖`,
`dû = (dc − û (û·dc))/‖c‖` with `dc = dv − mean dv`, and a channel too flat to
normalise adds nothing. A sample off the photograph is not in the frozen
residual, so it has no row. In the pair fit both renders move with the tilt and
only the second with the relative shift, and each is normalised over the block
afresh, so the Jacobian is the first view's normalised derivative less the
second's.

**The gradient** is read from the level's Sobel gradient images, at the
sample's level position and with the value's bilinear weights, for the fit's
steps and for the information about the tilt
(§ [The plane's precision](#the-planes-precision)) alike: at each pixel, the
difference of its two neighbours along the axis, halved, averaged across the
other axis with weights `[1, 2, 1]/4`. The bilinear read of those images is
continuous across pixel edges, and is the slope of the photograph averaged over
about two pixels. It is computed from the 4×4 pixels about the sample, which
give the Sobel values at the four taps the value reads, so no gradient image is
stored.

The bilinear read's own derivative, from the same four taps as the value, is
the exact derivative of the render, but it jumps where the position crosses a
pixel centre, and along an axis where a sample's two taps are clamped to one
edge pixel it is zero. The steps do not use it because, with it, the fits moved
twice as many ground-truth points by more than a degree as small changes of the
earlier forward difference did, and lost judged points; with the Sobel gradient
they keep every dataset's share within 5° and take 37% less time than the
forward difference
([measurements](grown-plane-normal-measurements.md#the-sobel-gradient-for-the-steps-2026-10-18)).
The information does not use it because detail a pixel or two across adds to it
information the fits cannot use, which makes σ too small
([measurements](grown-plane-normal-measurements.md#sobel-information-and-the-σ-target-2026-10-21)).
The tests read it to check the Jacobian against a central difference of renders
(§ [Testing](#testing)).

**An iteration** solves `(JᵀJ + λ·diag(JᵀJ) + ε) δ = −Jᵀr`, with `λ` from
`initial_damping` and `ε` 10⁻⁹ of the largest diagonal entry, scales `δ` back so
that no tilt parameter moves more than 0.1 and no shift more than 2 px, and
reads the true score at the new parameters from a fresh render. The step is
kept only when the score rises, and `λ` falls tenfold, to no less than 10⁻⁹;
the render that scored the kept step also gives the residual, frozen there,
that the next iteration linearises at, so a kept point is rendered once.
Otherwise `λ` rises tenfold and the step is solved again, up to three tries in
all. A stage ends after `max_iterations` iterations, when a kept step raises
the score by less than `min_improvement` or by less than 1% of what the stage's
first kept step raised it, or when no damping finds a rise; a stage that cannot
rise keeps the parameters it was given. The first step's gain is positive, so
the 1% rule first applies to the second step: a stage whose first step does
nearly all its work stops after the second, which is kept. The three tries and
the 1% are constants of the optimiser, not settings; together with reusing the
kept render they cut the step's renders by 23% and its time by about a quarter
with the judged points unchanged
([measurements](grown-plane-normal-measurements.md#fewer-renders-per-fit-2026-10-18)).

The parameters are a view's tilt and shift `[a, b, sx, sy]` within a view, its
shift alone with the tilt held in the shift-only fit, and in the pair fit the
tilt and the second view's shift against the first's `[a, b, rx, ry]`
(§ [Wide pairs](#wide-pairs)). The pair fit's residual compares two renders
that both move with the tilt, so it normalises both sides afresh at every
evaluation, where a view against the reference normalises the reference once.
No fit has more than four parameters.

### Combining the views' planes

Every growth ends at the view's own plane, the tilt `tₑ` its whole-grid stage
reached, with its shift and its information `Iₑ` about the tilt, read at that
plane and shift (§ [The plane's precision](#the-planes-precision)). The step
combines planes by their information and never fits views together. The
combined plane of a set of estimates is their information-weighted mean,

```text
t = (Σ Iₑ + ε)⁻¹ · (Σ Iₑ·tₑ + ε·t̄)        covariance 2τ · (Σ Iₑ)⁻¹
```

the plane that maximises the sum of the estimates' quadratic score models
`−½ (t − tₑ)ᵀ Iₑ (t − tₑ)`. `t̄` is the plain mean of the tilts and `ε` is 10⁻⁶
of the summed information's largest eigenvalue, or 10⁻⁶ where that is not
positive, so that the mean of one estimate is that estimate and a direction no
estimate informs takes the plain mean rather than being undetermined.

Combining the planes of a selection of views, rather than fitting the shared
tilt over every joined view after each join, gave about the same error over the
1609 ground-truth points, a median 0.03° lower and a slightly lower mean, in
about a sixth of the time
([measurements](grown-plane-normal-measurements.md#combining-the-planes-and-the-selection-2026-10-17)).

The step runs in this order:

1. **The selection** picks views one at a time near to far, growing the first
   from the starting plane and each later one from the combined plane so far
   (§ [The selection](#the-selection)). After each pick that reaches the floor,
   the picks so far are combined by the robust combination below, and the
   selection stops once that plane is precise enough.
2. **The robust combination** of every pick that reached the floor.
3. **The wide pairs** among the views kept, fitted from the combined plane, and
   the robust combination of the views' and the pairs' planes
   (§ [Wide pairs](#wide-pairs)). That plane is the final plane.
4. **The weak-turn rule** judges the turn to the final plane
   (§ [Turn and report](#turn-and-report)).
5. **The shift-only fits** of every candidate the selection did not pick, at
   the final plane, when the turn is made (§ [The selection](#the-selection)).

**The robust combination.** Every view estimate, and every pair estimate not
already refused under its floor, is in the first mean. While some kept estimate
lies further than `max_agreement_distance` from the mean of the kept ones, by
the Mahalanobis distance of
§ [When an estimate agrees with the combined plane](#when-an-estimate-agrees-with-the-combined-plane),
the furthest is left out (`TooFar`), with any pair that holds a view left out
(`ViewLeftOut`), and the mean is read again. Then each kept view is scored at
the mean, at its shift moved there (below), one render each; a view under
`min_held_zncc` is left out (`HeldUnderFloor`) with its pairs, and the whole is
repeated until no view is left out. The furthest goes first and one at a time so
that the result does not depend on the order the estimates are given in, and so
that at least one estimate is kept, since an estimate alone is at no distance
from its own mean. The distance is read against a mean the estimate is part of,
which shortens it.

**Moving a shift to another plane.** A view's shift was fitted at its own plane
`tₑ`, and the plane it ends at is `t`. With `N` the normal matrix of its
whole-grid residual at its own plane and shift over `[a, b, sx, sy]`, the shift
the Gauss–Newton model fits at `t` is

```text
s = sₑ − N_ss⁻¹ N_st (t − tₑ)
```

which costs no render. The held score at a mean, the score the report gives at
the final plane, and a joined view's reported shift are all read at the shift
moved this way; nothing is fitted again. The score at the final plane is not
rendered again either: the combination's last held check scored every kept view
at the final plane and that shift and left none out, so a joined view's
`grid_zncc` is its `held_zncc`.

### Wide pairs

**Why.** How much a view says about the tilt grows with its angle from the view
it is compared against: the further apart two views are, the more a change of
tilt changes one view's picture of the patch against the other's. Every growth
compares a view with the reference. When the reference sits among the cameras
rather than at one end, two views on opposite sides of it are further apart from
each other than either is from the reference, so comparing them with each other
is more sensitive to the plane than either comparison with the reference.

**Which pairs.** Among the views the robust combination kept, every pair is
ranked by the angle between the two views' directions from the patch centre,
widest first, ties by the two views' order among the candidates, and pairs are
taken down that list so that no view is in more than one pair, up to
`max_pairs`. Taking pairs greedily by angle does not ask that a pair lie on
opposite sides of the reference, or that it be wider than the widest view's
angle from the reference, so a pair of two views on the same side is taken
where it is the widest left.

**The fit.** Both views are rendered through the plane onto the reference's
working grid: each sample of the grid is followed along its reference ray to the
plane and projected into each of the two photographs. The fit starts from the
combined plane, holds each view's shift against the reference at its own shift
moved to that plane, and fits the tilt and one relative shift `(rx, ry)` added
to the second view's, for the best robust cell score of the ZNCC between the
two renders, over the cells the reference is textured in. The pair constrains
only the difference between its views' shifts, since shifting both by the same
amount changes neither render against the other, which is why the common part
is held at the reference comparisons' value. The pair's information about the
tilt is the Schur complement of its normal matrix with the relative shift
eliminated, as for a view.

**Acceptance.** A pair whose score between its two renders is under
`min_held_zncc` is refused (`UnderFloor`) and never combined. The others are
added to the robust combination with the views, under the same distance test,
and a pair holding a view the combination leaves out is left out with it.

**Two σ.** A pair compares two photographs each of which is already compared
with the reference, so its information is not independent of theirs. The
precision is reported both ways: σ (a), `sigma_weak_deg` and
`sigma_strong_deg`, from every kept estimate's information summed, and σ (b),
`discounted_sigma_weak_deg` and `discounted_sigma_strong_deg`, with each kept
pair's information halved, so that a view in a pair counts about once. The rule
is the simplest that discounts the reuse; it is not derived from the noise. The
plane itself is the mean with the full information of every kept estimate, (a).

**Risks.** The two views of a wide pair differ more from each other than either
does from the reference: in foreshortening, so the two renders sample the
photographs at more different scales; in sampling, so one render can be blurred
where the other aliases; and in lighting, since the two look at the surface from
further apart. Near an occlusion they can also see slightly different surface.
Each of these lowers the ZNCC between the pair's renders for a reason other than
the tilt, which is why a pair is held to the floor and to the distance test.

### The selection

The step predicts each candidate's information about the tilt from geometry
before it renders the view, and grows only the views the prediction picks.

**The model.** For each candidate: `θ`, its angle from the reference seen from
the patch centre; `d`, the unit direction of its offset from the reference, the
difference of the two viewing directions written in the starting frame's
`(u, v)`; and `s`, its photograph px per world unit at the anchor, the square
root of the area scale of the projection of the starting frame there, capped at
the working grid's samples per world unit `ρ`, since the grid samples no finer.
With `e` the direction across `d` and the geometric gain
`g = (s/ρ)² · sin²θ`, the prediction is

```text
P = g · (α · d dᵀ + β · e eᵀ)
```

`α` and `β` are read from the views matched so far: each the median over them of
their measured information along `d`, or across it, divided by `g`. A candidate
with no offset direction is predicted `g · (α + β)/2` in every direction. The
model has two directions rather than one because a single view determines the
tilt in both directions: on the tests' scene a rank-one model picked 20° views
across the first one's offset over the 40° views and needed five views where
the two-direction model needs three.

**The picks.** The selection picks near to far by angle from the reference, in
rungs, because a wide view grown from a start far off fails or ends on a wrong
plane more often than a narrow one, and a wide view grown from a plane the
narrower views reached is within capture: with a start 30° or more off, views at
20° to 60° pass the floor 61% to 76% of the time, and views at 10° to 20° pass
almost always with a plane good to a few degrees
([measurements](grown-plane-normal-measurements.md#the-near-to-far-ladder-2026-10-18)).

- **The first rung** takes the view nearest `first_rung_deg` within
  `first_rung_min_deg` to `first_rung_max_deg`, ties by the earlier view, grown
  from the starting plane. A refused pick is followed by the next nearest view
  in that range. Where no view in the range is left and none has passed, the
  first rung takes the view nearest `first_rung_deg` among all those left, so
  the widest view under the range where nothing is wider. A view under 10°
  gives a poor plane on its own (σ 13° to 40° on the ground truths), but the
  ladder climbs from it.
- **Each later rung** aims at `rung_ratio` times the widest angle among the
  views the combination keeps, and takes the views within a factor
  `√rung_ratio` of that aim: from a first view at 15°, the aims are about 24°,
  38° and 61° where each rung keeps a view near its aim. A rung with no view
  left in its range aims `rung_ratio` times further out, and one with an aim of
  zero, where every view kept is 0° from the reference, aims at the widest
  candidate. The rung whose range reaches the widest candidate is the last: it
  takes every view from the smaller of its aim and the widest candidate's
  angle, over `√rung_ratio`, outward, and the selection stays on it, so it ends
  with the widest views. Once no view is left there, one more rung takes every
  view left, the narrower ones among them.
- **Within a later rung** the pick maximises the smallest eigenvalue of the
  measured information of the views the combination keeps plus the view's
  prediction, which spreads the picks' offset directions about the normal, ties
  by the larger eigenvalue, then by the view nearer the aim, then by the earlier
  view.
- **Growth.** Each pick of a later rung is grown from the combined plane of the
  views kept so far, from its keypoint-implied shift. Where that growth is
  refused, a pick of the second rung, rung 1 in the picks' count, is grown again
  from the starting plane as the first rung grows it, at the same shift; a pick
  of any later rung is left out after its one growth. The second growth is kept
  on the second rung because a wrong first pick makes a wrong combined plane,
  which refuses the second rung's picks, and grown from the starting plane they
  can pass and pull the plane back; on later rungs it cost more than it found. A
  rung with no view left in its range counts as a rung, so the first rung a pick
  is grown from the combined plane on can be past the second, and then no pick is
  grown again. A rung waits until one of its views is kept: a refused pick is
  followed by another view in the same range, and the ladder moves out only once
  the widest kept angle widens or no view is left in the range.
- **Stopping.** After each pick that reaches the floor the picks are combined by
  the robust combination, and the selection stops when the combined weakest σ
  (a) is at most `target_sigma_deg` or `max_selected_views` picks have reached
  the floor, or no candidate is left. A candidate whose angle is not finite is
  never picked. The trace records each pick's try with the plane each growth
  started from (`tilt_start`) and the combined plane after it, and each pick its
  rung.

**The views not picked.** When the turn is made, each candidate the selection
did not pick has its shift alone fitted at the final plane: stage 1 reads the
first window at its keypoint-implied shift; the shift is fitted with the tilt
held over the window of the second fitted stage, half-width `1/5` of the grid's
side under the default fractions (10 on a 48 grid; the widest window on a grid
with fewer), to the optimiser's usual stop; and then two Levenberg–Marquardt
iterations over the whole grid on the robust cell score, whose score is the one
reported. A fit under the floor is kept as it is, and no search for another
shift follows it. The view is reported `ShiftOnly`, with its shift and its
score, which the progress note judges against `min_held_zncc`; its keypoint is
not written. It is not in the plane. The window and the two iterations keep the
score of a full whole-grid fit at about 40% less time on the shift-only views
([measurements](grown-plane-normal-measurements.md#cheaper-growths-2026-10-20-and-2026-10-21)).
When no pick joined, or the turn is not made, the views not picked are reported
`NotAttempted`.

### The plane's precision

A view's information about the tilt is read from the normal matrix of its
whole-grid residual at the plane and shift it is read at: `N = JᵀJ` over
`[a, b, sx, sy]`, with `J` read through the Sobel gradient the fit's steps use,
with the shift eliminated by the Schur complement,

```text
I = N_tt − N_ts · N_ss⁻¹ · N_st
```

which is the information about the tilt with the shift fitted again at every
tilt, so that a shift cannot stand in for a tilt. Its negative eigenvalues,
which only rounding can make, are set to 0.

**The scaling.** The residual is scaled so that its sum of squares is
`S = 2 · (1 − score)`. Near the fitted plane the Gauss–Newton model gives
`S(δ) ≈ S₀ + δᵀ I δ` for a tilt change `δ` with the shift refitted, so the score
falls by `½ · δᵀ I δ`: `I` is minus the Hessian of the score in the tilt. A ZNCC
drop of `τ` is therefore a residual rise of `ΔS = 2τ`, reached on the ellipse
`δᵀ I δ = 2τ`. The ZNCC tolerance τ (`zncc_tolerance`) is the ZNCC difference
noise can produce. A tilt change inside that ellipse cannot be told from noise,
which makes the covariance of the tilt `2τ · I⁻¹`, regularised as
`2τ · (I + 10⁻⁶)⁻¹` so that a direction with no information gets a large
variance rather than none. σ is the plane's uncertainty along its weakest
direction, `√(2τ/λ_min)` with `λ_min` the smaller eigenvalue of `I`, in degrees,
since a small tilt `(a, b)` is that many radians; the strongest direction's σ
comes from the larger eigenvalue in the same way. Since σ grows as `√τ`, the
choice of τ scales every σ and every agreement distance by the same factor.

**Which information.** Every growth's information is read when it is judged, at
its own plane and shift, and a pair's at the plane and relative shift its fit
ended at. The combined plane's information is the sum of the kept estimates'
information as each was read, and the precision the report gives is that sum,
σ (a), and the same with each pair's halved, σ (b). Nothing is read again at the
final plane.

**Calibration.** Read through the Sobel gradient, σ is somewhat large in the
middle of the errors and still small in their tail: at the 2.75° target, 63% of
the `dino_dog_toy` points and 54% of the points whose stored normal was judged
right by eye are within σ of it, against 39% for a calibrated σ, and 87% and 75%
within 2σ, against 86%
([measurements](grown-plane-normal-measurements.md#sobel-information-and-the-σ-target-2026-10-21)).

### When an estimate agrees with the combined plane

A growth counts as an estimate when its own whole-grid robust score reaches
`min_held_zncc` (`UnderFloor` otherwise), and when it did not stop early at a
window stage (`StoppedEarly`). The robust combination keeps an estimate while
both hold:

- its own plane lies within `max_agreement_distance` of the combined plane by
  the Mahalanobis distance `√(dᵀ C⁻¹ d)`, with `d` the difference of the two
  tilts and `C` the estimate's covariance and the combined plane's added
  (`TooFar` otherwise);
- for a view, its robust grid score at the combined plane, at its shift moved
  there, is at least `min_held_zncc` (`HeldUnderFloor` otherwise); for a pair,
  its score between its two renders is (`UnderFloor` otherwise).

The test is scaled by precision rather than set as fixed bars on a view's own
readings (a whole-grid ZNCC bar, a bound on how far a window's ZNCC may fall,
or a fixed bound on the ZNCC a view loses when held to the combined plane):
fixed bars left out views that ended near the stored normal, and a fixed loss
bound let in a view whose own plane was 26° from the plane the others shared,
which the precision-scaled distance refuses at 3.2σ
([measurements](grown-plane-normal-measurements.md#the-agreement-rule-2026-10-16)).

Every ZNCC is the mean over the colour channels the reference is not flat in
(spread at least 0.5 grey levels), over the samples that land on the photograph
in both renders, and is read only when at least `min_coverage` of the window's
or cell's samples do, and at least two; under it the ZNCC is `−1`, and with no
channel scored it is 0. A fourth channel, the photograph's alpha, is not read.

### Turn and report

The step turns the patch to the final plane when at least one view joined,
unless the turn is a large one on weak evidence, and reports σ along the
weakest and strongest directions as its confidence, both ways
(§ [Wide pairs](#wide-pairs)), with a verdict of low confidence when the weakest
σ (b) is over `low_confidence_sigma_deg`, or is not a number. σ (b) is the one
judged because it is the larger; with no pair kept the two are the same. The
low-confidence verdict does not by itself decide the turn. Precision is used
inside the step in the agreement rule above, to stop the selection, and in the
weak-turn rule.

**The weak-turn rule.** The turn is the angle from the starting normal to the
final plane's normal, the `TiltReport::degrees` the turn gives, or would give:
`tilt_patch`'s arc, stopped where it would stop at `MAX_TILT_DEG` from a camera
(`tilt_degrees` in
[bench/steps.rs](../../../crates/sfmtool-core/src/bench/steps.rs) computes it
with no track built). A turn of at least `weak_turn_deg` whose ratio to σ (a),
the weakest-direction σ of every kept estimate's information summed with no
pair halved, is under `min_turn_sigma_ratio` is a large turn on weak evidence.
The step does not make it: the outcome is `GrownPlaneOutcome::WeakTurn`, the
normal and every keypoint are left as given, no view not picked has its shift
fitted, and the track is read back. The report still gives the plane found
(`normal`), the turn (`turn_deg`), the precision, and the views, trace, picks
and pairs. A σ that is not finite counts as no evidence, a ratio of 0, so a
large turn with no finite σ is not made. Every other turn is made, a turn under
`weak_turn_deg` however weak its evidence. The rule reads σ (a), not σ (b),
because the judgements that set it were scored on σ (a). Setting
`min_turn_sigma_ratio` to 0, or `weak_turn_deg` to infinity, turns the rule off.
The step leaves the normal unchanged on two outcomes only: when no view joined
(`NoneJoined`), and on a large turn on weak evidence (`WeakTurn`).

A low precision alone does not stop the turn because a plane of low precision
can still be much better than the start: a 2° gate on σ left two of four
hand-picked points at their starts, 8.5° and 80.1° from the stored normal, where
the planes it rejected were 1.36° and 16.8°. A low-precision plane says the
views constrain the tilt little near it, not that the start is closer; the
local curvature of the ZNCC understates what narrow views can correct from far
away. The weak-turn rule is narrower than a gate on σ. Judging 62 points by eye,
the maintainer found the turn right on 30 of the 32 under 65°, however weak its
σ, and wrong on 9 of the 13 turns of 65° or more under three times their σ; at
70° and 3, the rule is right on 56 of the 62 where turning every time is right
on 49
([measurements](grown-plane-normal-measurements.md#the-weak-turn-rule-2026-10-18)).

### The keypoints after the turn

The step writes no keypoint. Each view's image shift is an unknown of the tilt
fit, a nuisance it solves for so that the view's samples can agree with the
reference's, and it is reported (`GrownPlaneView::Joined::shift`,
`GrownPlaneView::ShiftOnly::shift`) and then discarded. The keypoint is the
track-stage measurement `TrackMeasurement::keypoint`
([bench/track.rs](../../../crates/sfmtool-core/src/bench/track.rs)), the pixel
a commit writes and the place every reading of the row is anchored at; placing
it is the fit's job, run by the caller before this step where the keypoints may
be off and after it where they should be relocalized against the turned patch.
Written as keypoints, the shifts were worse than the localizer's and the refit
moved them straight back
([measurements](grown-plane-normal-measurements.md#writing-no-keypoint-2026-10-27)).

- **Every row keeps its keypoint**, under every outcome. The turn is
  `tilt_patch`'s, about the patch centre, which does not move, and it keeps each
  keypoint's gap from the centre's projection, so no keypoint moves
  (`follow_centre` in
  [bench/steps.rs](../../../crates/sfmtool-core/src/bench/steps.rs)), as under
  every other normal step. A row never localized takes the centre's projection,
  as under every patch step.
- **The reference** needs nothing of its own. The anchor lies on the reference
  keypoint's ray, so it projects back to that keypoint, and the reference's
  shift is zero by construction: the reference tile is the template every other
  view is compared against, so it has no shift to fit. The reference-view rule
  can pick a row with no keypoint, which it renders through the patch itself;
  its ray is then the one through the centre's projection, the anchor is the
  centre, and the turn gives the row the centre's projection.

**The readings.** The turn drops every row's measurement but its keypoint, and
the bitmap, as every tilt does; the readback that ends the step renders a new
bitmap from the reference and reads every row again at its keypoint, so every
reading the step returns was taken at the keypoint the row holds.

**The centre does not move.** A depth error along the reference ray shows up as
the per-view shifts; the readback's `projection_offset_px` shows, for each row,
how far its keypoint sits from where the centre projects, and a fit after the
step is what moves the keypoints and the centre.

### Trusting the keypoints

The step grows every view, and fits every shift-only view, from the shift its
keypoint implies, and has no search for another shift and no setting to bring
one back. A view whose keypoint is off by more than the growing windows recover
is refused and left out; fitting its keypoint is the caller's separate step,
run before this one. A search after a refused growth, within ±3 px of the
implied shift and widened to ±12 px, cost 9.5% of the step's time, left the
judged points and the dataset medians within 0.45° of where they were, and
pulled some points wrong as often as it recovered others whose keypoints were
off
([measurements](grown-plane-normal-measurements.md#trusting-the-keypoints-2026-10-23)).

## Implementation notes

- **Cost.** What dominates is one render of the window per optimiser
  evaluation per view: an iteration renders once for each damping it tries. The
  point an iteration starts from was rendered when its step was scored, and its
  Jacobian is read at that render: it renders nothing, and makes one pass over
  the samples that reads each sample's Sobel gradient, 16 pixels per channel,
  and, for a fit with the tilt free, goes through the lens model's Jacobian,
  which reads no photograph. The gradient is read only for a point a Jacobian is
  taken at, so a damped step that is refused costs its render alone. Each view
  keeps the projection of its last tilt, so renders at one tilt and several
  shifts project the grid once and only sample again, and a Jacobian reads the
  projection its point's render made. The cost is the growths, about linear in
  the number of views picked, with one render per kept view for each held check
  of the robust combination, a few whole-grid renders per iteration of each
  pair fit, and one stage-1 window, one window fit and two whole-grid
  iterations per shift-only view. A growth that stops early skips its later
  windows, its whole-grid stage and the reading of its information. On the
  tests' synthetic scenes a whole step takes well under a tenth of a second in
  an optimised build; on the ground truths a point costs about 6 to 8 ms of one
  thread per observation ([refine-normals.md](refine-normals.md) § "Run time").
- **Phases.** The step opens the overview phase `grown plane normal` over
  everything after its settings are checked, and under it a `grown plane view`
  per try, noted with the image and whether it was matched or left out. A
  detailed `Progress` adds, under the step, `setup` (the reference, the working
  grid, the views, those with no keypoint and the windows; the reference-view
  rule's phases nest in it), a `selection` per pick (noted with the view, its
  angle, its scale, its rung and its predicted weakest information), a `mean`
  per robust combination (noted with the views and pairs kept, those held under
  the floor, the means read, `σ` and the renders), `pairs` with a `pair fit` per
  pair (noted with the two images, their angle, the pair's ZNCC, `σ`, verdict,
  steps taken and renders), a `shift-only fit` per view not picked (noted with
  its first-window and grid ZNCC, whether that reaches the floor, and its
  renders), and `turn` (`turn_and_read`, whose phases nest in it, noted with the
  turn); under each try, a `growth` per growth, noted with the plane it started
  from, its first-window and grid ZNCC, whether it was taken, and its renders;
  and under each growth a `window stage` per later window and a
  `whole-grid stage` (each noted with its ZNCC, steps taken and renders), and
  `judge` (the growth's own information, noted with its grid ZNCC, `σ` and
  verdict, or with the verdict alone for a growth that stopped early). A view
  counts its renders in a `Cell` beside its projection cache; the notes are
  built only when a phase records, and nothing a phase does reaches the result,
  so the step's track and report are the same with detail on and off.
- **Single-threaded.** The step renders one view at a time on the calling
  thread, as `grid_plane_normal` does; `bench::refine_normals` runs points in
  parallel, one step per task.
- A view's keypoint is read once, when the candidates are built, for the
  keypoint-implied shift that every growth and the shift-only fit start from. A
  view with no keypoint, or whose anchor does not project into it, is not a
  candidate: it is recorded `LeftOut` as `NoKeypoint` before the selection, so it
  does not enter the selection's widest angle, the prediction or the pairs.
- The weak-turn rule is judged after the pairs and before the shift-only fits,
  so a turn that is not made costs no shift-only fit; the turn it judges is
  `tilt_patch`'s, computed by `tilt_degrees` without building the turned track.
  The rule changes nothing on a point whose turn it makes: the selection, the
  combination, the pairs and the shift-only fits are the same, in the same
  order.
- The step's geometry assumes a finite anchor and finite reference rays
  throughout, which `normal_preconditions` guarantees by refusing a track at
  infinity.
- A ray that meets the plane behind the reference camera, or not at all, and a
  point a camera does not project, give a sample off the photograph, as a sample
  past the image's edge does.

## Determinism and precision

Photograph samples are `f32`, read bilinearly from `u8` pyramid levels; the
ZNCC sums, the residuals, the tilt, the shifts and the plane are `f64`. The
candidates are held in order of their angle from the reference, ties by observation index;
the selection's picks and the pairs break their ties by fixed rules
(§ [The selection](#the-selection), § [Wide pairs](#wide-pairs)); every fit is
deterministic, and nothing runs in parallel, so the output is the same on every
run. No content hash depends on it.

## Parameters

The defaults are `GrownPlaneNormalOptions::default()` in
[bench/normal/grown_plane.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane.rs);
`check` refuses a setting outside the range given.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `working_resolution` | `None` | Side of the working grid, samples; `None` follows the reference's footprint, clamped to `R ..= 2R`, as `grid_plane_normal` does; at least `spot_window` |
| `spot_window` | `7` | Side of the first window, grid px, centred on the grid's centre sample; at least 1 |
| `window_fractions` | `[1/8, 1/5, 1/3, 1/2]` | Half-widths of the later windows as fractions of the grid's side, rounded, any no wider than the first window's dropped: 6, 10, 16, 24 on a 48 grid; 4, 6, 11, 16 on a 32 grid; 5, 8, 12 on a 24 grid; each finite and above 0 |
| `zncc_tolerance` | `0.01` | ZNCC difference noise can produce, τ; finite and above 0 |
| `max_agreement_distance` | `3.0` | Largest Mahalanobis distance of an estimate's plane from the combined plane |
| `min_held_zncc` | `0.5` | Robust grid score a growth needs on its own, a view needs at the combined plane, a pair needs between its two views, and that a shift-only view's score is reported against |
| `low_confidence_sigma_deg` | `3.67` | σ (b) above which the report says low confidence; it does not by itself stop the turn |
| `weak_turn_deg` | `70.0` | Smallest turn from the starting normal, degrees, that the weak-turn rule judges; at least 0, infinite turns the rule off |
| `min_turn_sigma_ratio` | `3.0` | Ratio of the turn to σ (a) that a turn of at least `weak_turn_deg` needs to be made; a σ that is not finite counts as a ratio of 0; finite and at least 0, 0 turns the rule off |
| `max_pairs` | `3` | Pairs of views matched against each other, at most |
| `target_sigma_deg` | `2.75` | The selection stops once the combined weakest σ is at most this, degrees; finite and above 0 |
| `max_selected_views` | `6` | The selection stops once this many picks have reached the floor; at least 1 |
| `first_rung_deg` | `15.0` | The selection's first pick is the view nearest this angle from the reference within the first rung's range, degrees; within that range itself |
| `first_rung_min_deg` | `10.0` | The narrowest angle of the first rung's range, degrees; finite, at least 0 |
| `first_rung_max_deg` | `20.0` | The widest angle of the first rung's range, degrees; finite, at least `first_rung_min_deg` |
| `rung_ratio` | `1.6` | Each later rung aims at this times the widest kept angle and takes views within a factor of its square root of that aim; finite, above 1 |
| `min_coverage` | `0.6` | Share of a window's or cell's samples on the photograph for its ZNCC to be read; in `0 ..= 1` |
| `max_iterations` | `20` | Levenberg–Marquardt iterations per stage, at most |
| `initial_damping` | `1e-3` | The damping each fit starts from; finite, at least 0 |
| `min_improvement` | `1e-6` | A stage stops once a kept step raises its score by less |
| `fit` | `FitOptions::default()` | The reading the step ends with, and the reference-view rule's grazing gate and sampler |

Five numbers are constants of the step rather than settings: the early stop's
margin of 0.25 under `min_held_zncc` (`EARLY_REFUSAL_MARGIN`) and its first
stage, the second fitted window (`EARLY_REFUSAL_FROM_STAGE`); the two rungs
whose refused picks are grown again from the starting plane
(`STARTING_PLANE_RETRY_RUNGS`); and the shift-only fit's window, the second
fitted stage's (`SHIFT_ONLY_WINDOW`), and its two whole-grid iterations
(`SHIFT_ONLY_GRID_ITERATIONS`). The optimiser's three damping tries
(`DAMPING_TRIES`), its 1% relative stop (`MIN_RELATIVE_GAIN`) and its step caps
of 0.1 in the tilt and 2 px in the shift are constants of
[grown_plane/solve.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/solve.rs).
The step has no setting for a shift search, since it makes none.

Where the defaults come from: the agreement numbers (`zncc_tolerance`,
`max_agreement_distance`, `min_held_zncc`) are a Python prototype's, set on one
point and then four. The three optimiser settings were chosen on the tests'
synthetic scenes; none of the changes tried around them on the ground truths was
better on every measure. `max_pairs` and `max_selected_views` are first values.
`target_sigma_deg` and `low_confidence_sigma_deg`, 4/3 of it, were set when the
information came to be read through the Sobel gradient. The ladder's settings
were swept on the ground truths, which moved `first_rung_deg` from 12° to 15°.
`weak_turn_deg` and `min_turn_sigma_ratio` were chosen on the points judged by
eye. The measurements file gives each
([grown-plane-normal-measurements.md](grown-plane-normal-measurements.md)).

## Testing

The tests in
[bench/tests/grown_plane_normal.rs](../../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs)
build synthetic planes seen by 128-px pinholes that look at the point from
around it, 4 units away, through `Scene::looking_at`, with the patch spanning
32 px in a facing view;
[grown_plane/tests.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/tests.rs)
tests the parts that need no photograph, and
[grown_plane/jacobian_tests.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/jacobian_tests.rs)
the analytic Jacobian. Every test runs the defaults unless it names a setting;
the scenes with up to six views that need every view picked set a target σ of
0.5°, which the combined σ there stays over.

- **A textured plane started 25° off its normal, seen from six views out to
  50°:** every view is picked and joins on its first growth, the normal ends
  within a degree of the plane, every joined view's shift is under 0.5 px, the
  patch turns more than 20°, the centre does not move, and every keypoint, the
  reference's among them, is kept bit for bit. Both pairs are kept, and σ (b) is
  larger than σ (a).
- **A view whose top cells see other texture** joins under the robust cell
  score, and the normal ends within a degree of the plane.
- **Narrow views,** within 3° of the reference, started 20° off: the patch is
  turned, the report says low confidence, and the turn, under 70° though under
  three times its σ, is made and equals the tilt's degrees.
- **A large turn on weak evidence:** the narrow views' scene with
  `weak_turn_deg` 10 keeps the start (`WeakTurn`): the patch and every keypoint
  are as given, no view is fitted for its shift alone, and the report gives the
  plane found, within 2° of the truth, its turn and its ratio to σ, and an
  agreement between 0.5 and 1; with `min_turn_sigma_ratio` 0 the same scene
  turns. **A large turn on strong evidence**, the wide scene with
  `weak_turn_deg` 10, is made.
- **A disagreeing view,** one view at 38° seeing the texture on a plane turned
  25° from the others', passes the floor on its own and is left out as `TooFar`
  at more than 3σ, with one try and no kept pair holding it; the normal ends
  within a degree of the others' plane.
- **Repeating texture under the spot:** a checker of 8 px period, with three
  views' poses turned so the photograph's content sits more than 5 px from where
  the pose projects the point and each keypoint on the content. Every view joins
  on its first growth with a shift under 0.75 px. With the keypoints removed,
  each view is left out as `NoKeypoint` with no try and no pick, and the
  outcome is `NoneJoined`.
- **A keypoint off past the growth:** the 30° view of three, its keypoint 6 px
  and 2 px off the content, is grown from the combined plane and, on the second
  rung, once more from the starting plane, both refused, and left out; no phase
  searches, and the other two views' plane ends within a degree of the truth.
  **A shift-only fit from a keypoint off past the fit** ends nearer the
  keypoint-implied shift than the content, and every keypoint is kept.
- **Wide pairs:** four views 20° from a reference that faces the plane, on
  opposite sides of it. Both pairs, 40° apart, are taken and kept; the weakest σ
  is lower than with `max_pairs` 0, and still is with each pair's information
  halved; the plane ends within a degree of the truth.
- **The selection:** sixteen views in rings at 10° to 40° with a target of 2°.
  The first pick is in the first rung's range, the next are 40° views grown from
  the combined plane, the selection stops at the target having matched far fewer
  views than there are, and every other view is fitted for its shift alone,
  reaches the floor with a shift under 0.5 px and keeps its keypoint.
- **The ladder:** views at 5°, 12°, 19°, 31° and 50° are picked one rung
  outward each and then the view under the first rung; a refused second-rung
  pick is grown again from the starting plane and followed by another view of
  the same rung before the ladder moves out; a refused pick past the second rung
  is not grown again; the last rung ends at the widest view that passes; a view
  at the reference's own centre does not hold the ladder at 0°.
- **A growth stopped early:** a first-rung pick seeing unrelated texture stops
  at a window stage under 0.25, from index 2 of `stage_zncc` on and before the
  whole-grid stage, with `grid_zncc` and `sigma_deg` `NaN`, and is left out as
  `StoppedEarly`; every view that passes is grown through every stage.
- **The shift-only fit on keypoints off the content** by 1.5 px along each axis
  finds the shift that carries the patch to the content, within 0.5 px, and
  keeps every keypoint bit for bit.
- **A flat wall:** no view joins, the outcome is `NoneJoined`, the normal and
  every keypoint are left as given, and no plane, turn or precision is reported.
- **The agreement** is the mean score of the joined and shift-only views, the
  reference, the views left out and a `NaN` score not counted, and `NaN` with
  no score.
- **Refusals:** a bearing is refused as `AtInfinity`, a ZNCC tolerance of 0 as
  `BadGrownPlaneOption`, and a cancelled `Progress` as `Cancelled`; `check`
  refuses a `weak_turn_deg` that is NaN or negative, a `min_turn_sigma_ratio`
  that is NaN, infinite or negative, a first rung whose bounds are not finite,
  are negative or are out of order, a `first_rung_deg` outside its bounds and a
  `rung_ratio` of 1 or less or not finite, and takes an infinite
  `weak_turn_deg`, a `weak_turn_deg` of 0 and a ratio of 0.
- **Detail phases change nothing:** with no sink, an overview sink and a
  detailed one, the track and report are the same, compared by their `Debug`
  text; the overview records the step and one phase per try and nothing finer,
  and the detailed run records the phases of
  § [Implementation notes](#implementation-notes) at their depths.
- **The analytic Jacobian,** on renders of a textured plane through a pinhole, a
  radial lens and a fisheye, from a reference facing it and views 25° and 35°
  off: with the bilinear read's own derivative, every column of a view fit over
  the whole grid, a window, a shift-only fit and a pair lies within 3% of a
  central difference of renders, also with a quarter of the grid off the
  photograph; how a pixel moves with the tilt, and a block's derivative through
  its normalisation, match differences to 10⁻⁶ of their size; the Sobel read
  matches Sobel images built pixel by pixel, reads a linear ramp's slope
  exactly and is continuous across a pixel centre; and a view fit's information
  is the normal matrix of its Jacobian read through the Sobel gradient.
- **Without photographs:** the robust score on dino 528's cell readings, the
  residual identity `‖â − b̂‖² = 2 − 2·ZNCC` with weights, the coverage bar, a
  flat channel left unscored, the window schedule on 24, 32 and 48 grids, the
  Schur complement against the profile it stands for, σ from the eigenvalues,
  the fit reaching the optimum of a quadratic, and the weighted mean, its
  covariance and the agreement distance against a hand-computed case.

No test covers a patch centre off along the reference ray, σ against the spread
of the fitted normal under added noise, or a pair refused under its floor;
[grown-plane-normal-open-questions.md](../../drafts/grown-plane-normal-open-questions.md)
proposes them.

## Non-goals

- **Moving the centre or the keypoints.** The step turns the patch about its
  centre and moves neither the centre nor any keypoint
  (§ [The keypoints after the turn](#the-keypoints-after-the-turn)).
  Relocalizing the keypoints against the turned patch, and placing the centre
  from them, is a fit's job, which the caller runs after the step.
- **Searching for a view's shift.** The step trusts the keypoints it is given
  (§ [Trusting the keypoints](#trusting-the-keypoints)); fitting them is a
  separate step.
- **Per-cell one-dimensional constraints.** A cell over stripes pins its shift
  across the stripes only; the step does not use such constraints.
- **A textured spot off the anchor.** Stage 1 is always centred on the anchor,
  so a point whose centre lacks texture starts each view from an untextured
  window. Starting from a textured spot elsewhere, with the anchor pinned at the
  keypoint and the tilt searched at stage 1, is proposed in
  [grown-plane-normal-off-anchor-spot-amendment.md](../../drafts/grown-plane-normal-off-anchor-spot-amendment.md).

## Open questions

The questions still open about the step's settings and rules (the calibration
of σ, the agreement numbers, the weak-turn thresholds, the ladder, the pairs,
the selection's model, the window schedule, the cheaper growths' constants, the
optimiser, points where every view is weak or narrow, blur matching and the
report's fields) are collected, with the evidence on each, in
[grown-plane-normal-open-questions.md](../../drafts/grown-plane-normal-open-questions.md).
