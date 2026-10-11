# Reference View

A point in a patch-based reconstruction carries a small square bitmap of the
surface around it, and every photograph that sees the point can be resampled
into that square: one tile per view. The views are not equally good. A far view
holds less detail, an oblique one depends most on the patch's normal being
right, a tile that crosses the photograph's border holds only part of the
patch, a blown-out highlight holds no texture, and a view with an occluder in
front of one corner disagrees with the others there. This spec describes the
measurements that say what each view's tile can contribute, and the
**reference-view rule**, which picks the one view whose tile is stored as the
point's patch bitmap. It gives each view a cost in grid px: the tile's
sharpness, combined with a term for how oblique the view is, and multiplied by
penalties for a view that agrees less with the others, differs from them in
part of the tile, falls partly off its photograph or is clipped. The view with
the lowest cost is picked; a view that renders no tile, has less than 90% of it on
the photograph or sees the patch edge on or from behind never is. The bench
measures every track it evaluates this
way and reports the pick, and every operation that renders a point's bitmap
stores the picked view's tile as it is and records which observation it is
([§ "The stored bitmap"](#the-stored-bitmap)).

The design behind the measurements, and the case for storing the view that
holds the most detail rather than a mean of the views, is in
[../../drafts/sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md)
(Parts 4 and 5). How each observation then scores against the stored bitmap is
in [blur-matched-zncc.md](blur-matched-zncc.md) § "Scores against the stored
bitmap".

## Rust API

The measurements and the rule live in
[reference_view.rs](../../../crates/sfmtool-core/src/patch/reference_view.rs)
with its submodules [tile.rs](../../../crates/sfmtool-core/src/patch/reference_view/tile.rs),
[agreement.rs](../../../crates/sfmtool-core/src/patch/reference_view/agreement.rs),
[rule.rs](../../../crates/sfmtool-core/src/patch/reference_view/rule.rs)
and [track.rs](../../../crates/sfmtool-core/src/patch/reference_view/track.rs);
the stored bitmap in
[stored_bitmap.rs](../../../crates/sfmtool-core/src/patch/stored_bitmap.rs).
The viewing angle is in normal refinement's
[obliquity.rs](../../../crates/sfmtool-core/src/patch/normal_refine/obliquity.rs),
whose obliquity priors read the same direction. The bench's evaluation runs
them ([evaluate.rs](../../../crates/sfmtool-core/src/bench/evaluate.rs)) and
writes the readings into each row's `TrackMeasurement`
([editable-track.md](../bench/editable-track.md)).

```rust
// patch::reference_view
pub struct ViewTile {
    pub samples: Array3<u8>,             // (R, R, C), black off the photograph
    pub valid: Vec<bool>,                // per sample: on the photograph
    pub placement: OrientedPatch,        // anchored on the keypoint where it can be
    pub jacobian: Option<[[f64; 2]; 2]>, // image px per grid px at the centre
    pub sampler: Sampler,                // the sampler rule's choice
    pub coverage: f64,                   // share of valid samples
    pub clipped_share: Option<f64>,
    pub viewing_angle: Option<ViewingAngle>,
}
pub fn render_view_tile(patch: &OrientedPatch, view: &ProjectedImage<'_>,
    keypoint: Option<[f64; 2]>, resolution: usize, sampler: SamplerChoice,
    progress: &Progress<'_>) -> ViewTile;
pub fn clipped_share(pyramid: &ImageU8Pyramid, outline: &[[f64; 2]]) -> Option<f64>;
impl ViewTile { pub fn planes(&self) -> TilePlanes; } // what blur matching reads

pub fn pair_zncc_grid(a: &ViewTile, b: &ViewTile) -> [[f64; 3]; 3];
pub struct CellAgreement {
    pub pair_zncc_grid: Vec<[[f64; 3]; 3]>, // per view, median over the others
    pub typical: [[f64; 3]; 3],             // per cell, median over the views
    pub deficit: Vec<f64>,                  // per view, the cell deficit
}
pub fn cell_agreement(tiles: &[&ViewTile]) -> CellAgreement;

pub struct ReferenceReadings {
    pub coverage: Option<f64>, pub clipped_share: Option<f64>,
    pub viewing_angle_deg: Option<f64>, pub cell_deficit: Option<f64>,
    pub pair_zncc: Option<f64>, pub semi_major: Option<f64>, pub semi_minor: Option<f64>,
}
pub struct ReferenceViewOptions {      // impl Default: the values below
    pub normal_error_deg: f64,          // σ_n
    pub agreement_band: f64, pub agreement_scale: f64,
    pub cell_deficit_band: f64, pub cell_deficit_scale: f64,
    pub coverage_band: f64, pub coverage_scale: f64,
    pub clipped_band: f64, pub clipped_scale: f64,
    pub min_coverage: f64,              // the hard limit
    pub unread_radius: f64,             // r where no view has a radius
}
impl ReferenceViewOptions { pub fn angle_scale(&self, resolution: usize) -> f64; } // c
pub enum ReferenceTerm { Sharpness, Angle, Agreement, Cells, Coverage, Clipped }
pub enum ReferenceExclusion { NoTile, Coverage, BackFacing }
pub struct ReferenceCost {
    pub radius: f64, pub radius_read: bool, // r, and whether it is the view's own
    pub angle_term: f64,                    // c·tan θ, grid px
    pub agreement: f64, pub cells: f64, pub coverage: f64, pub clipped: f64,
    pub cost: f64,
}
impl ReferenceCost {
    pub fn effective_radius(&self) -> f64;                 // √(r² + angle_term²)
    pub fn ln_parts(&self) -> [(ReferenceTerm, f64); 6];   // ln cost, split by term
}
pub struct ReferenceChoice {
    pub reference: Option<usize>,
    pub costs: Vec<Result<ReferenceCost, ReferenceExclusion>>,
}
pub struct ReferenceStanding {
    pub cost: Result<ReferenceCost, ReferenceExclusion>,
    pub reference: bool,
    pub pick_cost: Option<f64>,
    pub costliest_term: Option<ReferenceTerm>,
}
impl ReferenceStanding { pub fn is_reference(&self) -> bool; pub fn cost_ratio(&self) -> Option<f64>; }
impl ReferenceChoice { pub fn standing(&self, i: usize) -> Option<ReferenceStanding>; }
pub fn choose_reference_view(views: &[ReferenceReadings], resolution: usize,
    options: &ReferenceViewOptions) -> ReferenceChoice;

// The rule over one track's tiles: the pair ZNCC from member coherence's
// matrix, the cells from the tiles, and the pick.
pub struct TrackReading {
    pub pair_zncc: Vec<Option<f64>>,
    pub cells: CellAgreement,
    pub readings: Vec<ReferenceReadings>,
    pub choice: ReferenceChoice,
}
pub fn tile_semi_axes(tile: &ViewTile) -> Option<[f64; 2]>;
pub fn readings_of(tile: &ViewTile, pair_zncc: Option<f64>, cell_deficit: Option<f64>,
    semi_axes: Option<[f64; 2]>) -> ReferenceReadings;
pub fn read_track(patch: &OrientedPatch, views: &[ProjectedImage<'_>], members: &[u32],
    keypoints: &[Option<[f64; 2]>], tiles: &[&ViewTile], semi_axes: &[Option<[f64; 2]>],
    sampler: SamplerChoice, options: &ReferenceViewOptions,
    progress: &Progress<'_>) -> TrackReading;

// patch::stored_bitmap
pub struct PatchBitmap { pub rgba: Vec<u8>, pub reference: usize } // the pick
pub fn bitmap_from_tile(tile: &ViewTile) -> Vec<u8>; // alpha 255 on the photograph
pub struct ReferenceRender { pub tiles: Vec<ViewTile>, pub semi_axes: Vec<Option<[f64; 2]>>,
    pub reading: TrackReading }
pub fn render_reference(patch: &OrientedPatch, views: &[ProjectedImage<'_>], view_set: &[u32],
    keypoints: &[Option<[f64; 2]>], resolution: u32, sampler: SamplerChoice,
    options: &ReferenceViewOptions, progress: &Progress<'_>) -> ReferenceRender;
pub fn render_patch_bitmap(patch: &OrientedPatch, views: &[ProjectedImage<'_>],
    view_set: &[u32], keypoints: &[[f64; 2]], params: &KeypointSubpixelParams,
    progress: &Progress<'_>) -> Option<PatchBitmap>; // None where no view is eligible
pub struct PatchBitmapColumn { pub bitmaps: Array4<u8>, pub reference_observations: Vec<i32> }
pub enum UnreferencedPoints { Pick, Skip } // what a point at -1 gets
pub fn render_patch_cloud_bitmaps(cloud: &PatchCloud, recon: &SfmrReconstruction,
    views: &[Option<ProjectedImage<'_>>], params: &KeypointSubpixelParams,
    unreferenced: UnreferencedPoints, done: Option<&AtomicUsize>,
    progress: &Progress<'_>) -> Result<PatchBitmapColumn, Cancelled>;

// patch::normal_refine
pub struct ViewingAngle { pub angle_deg: f64, pub tilt_direction_deg: Option<f64> }
pub fn surface_to_camera(patch: &OrientedPatch, cam_from_world: &RigidTransform) -> Option<Vector3<f64>>;
pub fn viewing_angle(patch: &OrientedPatch, cam_from_world: &RigidTransform) -> Option<ViewingAngle>;
```

**Why it is shaped this way.** The rule is a pure function over a slice of
per-view readings rather than a function of tiles and a reconstruction, so the
cost and its terms can be tested on constructed numbers, and a caller that
holds its readings from elsewhere (the slots of a bench row a verdict moved
under) runs the same rule. It takes the tiles' resolution because the angle
term is in grid px of that grid. Each reading is an `Option` because the views
of a real track do not all have every one: a tile that misses the photograph
has no clipped share, and a view member coherence could not score has no pair
ZNCC. The answer gives, for every view, its cost and the terms it is the
product of, or why the view was left out, because that is what a person
reading a table of views needs to see beside each row. A `ReferenceStanding` adds the pick's
cost and the **costliest term**, the term that adds most to the view's cost
over the pick's, so a row can be read on its own. The parameters are a
`ReferenceViewOptions` rather than constants so that tuning can sweep them
without rebuilding; every caller passes the `Default`, the bench
through `EvaluateOptions::reference_view` and the stored bitmap's render
through `KeypointSubpixelParams::reference_view`.

`read_track` takes the readings across one track's tiles and runs the rule, and
both the bench's evaluation and the stored bitmap call it, so the view a bench
row is marked as the reference and the view whose tile a fit stores are picked
from the same readings. It takes the tiles and their self-similarity semi-axes
rather than rendering them because the bench has already rendered every row's
tile and read its self-similarity for its own columns.

`render_view_tile` returns the tile and its readings together because they
share the render's placement and warp map: the outline the clipped share reads
is the warp map's border, and the angle is read at the anchored placement's
centre. The bench, the stored bitmap and the Python binding all call it, so a
tile rendered from Python is the tile the bench measured, and the stored bitmap
is the tile the rule read.

The viewing angle sits in normal refinement because its obliquity priors read
the same direction from a patch to a camera (`surface_to_camera`); one
function serves both, so the angle the bench reports and the cosine the prior
weights by cannot disagree.

```rust
use sfmtool_core::patch::reference_view::{
    choose_reference_view, ReferenceReadings, ReferenceViewOptions,
};

let readings: Vec<ReferenceReadings> = /* one per view */;
let choice = choose_reference_view(&readings, 24, &ReferenceViewOptions::default());
if let Some(i) = choice.reference {
    println!("view {i} is the reference");
}
for i in 0..readings.len() {
    let standing = choice.standing(i).unwrap();
    match (standing.cost, standing.cost_ratio(), standing.costliest_term) {
        (Err(why), _, _) => println!("view {i}: left out ({})", why.name()),
        (Ok(_), Some(ratio), Some(term)) => println!("view {i}: ×{ratio:.2}, mostly {term}"),
        _ => {}
    }
}
```

## The measurements

Each is read on the view's own `R×R` tile, `R` the reconstruction's patch
resolution, rendered through the patch re-anchored on the observation's keypoint
with the sampler the sampler rule picks ([image-warping.md](../camera/image-warping.md)
§ "Choosing the sampler per view"), the same tile the bench's self-similarity
reading takes and the stored bitmap is. Nothing coarser than `R`, and no pixel
of the photograph outside the tile, is read, except by the clipped share, which
reads the photograph's own pixels under the tile.

- **Coverage**: the share of the tile's samples the warp places on the
  photograph. A sample past the photograph's border carries no data.
- **Clipped share**: the share of the photograph's own full-resolution pixels
  inside the tile's outline that are `0` or `255` in any of their first three
  channels. The outline is the polygon through the photograph positions of the
  centres of the samples on the grid's border, taken in order round the grid.
  A border sample off the photograph is left out, so where the tile crosses
  the photograph's edge a straight edge joins the last border sample on the
  photograph to the next one. A pixel is inside when its centre is, by the
  even-odd rule. It is read from the photograph rather than the tile because the
  sampler's blending moves a clipped value off the limit. Specular highlights
  and blown-out sky clip, and a clipped region has neither texture nor its true
  colour.
- **Viewing angle** `θ`: the angle between the patch's outward normal and the
  direction from the anchored patch's centre to the camera, `0` facing the
  patch and `90` edge on. The anchored centre lies on the keypoint's ray, so
  this is the angle at the keypoint. For a patch at infinity the direction is
  minus the patch's bearing.
- **Tilt direction**: the direction in the patch's plane of the ray from the
  camera, as an angle from the patch's `u` axis towards its `v` axis. The view
  foreshortens the patch by `cos θ` along it and not at all across it, and an
  error in the normal shears the tile along it. None within
  `MIN_TILT_ANGLE_DEG = 0.1°` of facing the patch. The tile's rows run along
  `−v`, so in the tile as drawn `v` points up and an angle from `u` towards `v`
  turns counter-clockwise: the opposite sign to an angle measured in image
  coordinates, whose `y` runs down the rows.
- **Pair ZNCC**: the median of the view's pairwise ZNCCs with the track's other
  views, from member coherence's `k×k` matrix
  ([member-coherence-validation.md](member-coherence-validation.md)), rendered
  at the same resolution and sampler, anchored at each view's keypoint, over the
  views' common support with member coherence's Gaussian disk window. The
  tiles are correlated as rendered; none is blurred.
- **Pair ZNCC grid**: for each cell of the ZNCC grid's three-by-three split
  (rows and columns cut at `R/3` and `R − R/3`), the median over the other views
  of the two tiles' ZNCC in that cell. A pair's cell is correlated over the
  samples with data in both tiles, every sample weighted equally, per colour
  channel with the channels averaged; a channel flat in either tile is left out,
  and a cell with fewer than `REFERENCE_MIN_CELL_SAMPLES` shared samples or no
  channel left has no reading.
- **Cell deficit**: per cell, the track's **typical agreement** is the median
  of the views' pair ZNCC grid in that cell. A view's cell deficit is the
  largest amount by which its own cell falls below the typical agreement, over
  the cells whose typical agreement is at least
  `REFERENCE_MIN_JUDGED_CELL_ZNCC`; `0` when no cell is judged.

The rule also reads each tile's self-similarity ellipse, whose semi-major axis
is the ZNCC self-similarity radius
([zncc-self-similarity-radius.md](zncc-self-similarity-radius.md)): how far the
tile can slide over itself and still match itself, short on a sharp tile. It
is the whole tile's reading over its samples on the photograph
(`tile_semi_axes`).

## The rule

Each view the pick is made from gets a cost in grid px, and the view with the
lowest cost is the reference (`choose_reference_view`):

```
cost = r_eff · A · K · C · P
r_eff = √(r² + (c · tan θ)²)
c = (R / √12) · σ_n
```

- `r` is the self-similarity semi-major axis of the view's `R×R` tile, in grid
  px.
- `θ` is the viewing angle, under `90°` for every view the pick is made from
  ([§ "Hard limits"](#hard-limits)).
- `A`, `K`, `C` and `P` are the penalties for agreement, the ninths, coverage
  and clipping ([§ "The penalties"](#the-penalties)), each `1` inside its band.

A tie on the cost goes to the smaller semi-minor axis, a missing one counting
as infinite, and then to the earlier view.

**Why a product on an effective radius.** The cost stays in the unit of the
radius, so a cost reads as "this view counts as if its tile were this blurry".
The radius depends on the texture as well as on the view and is only compared
within one track; a penalty that multiplies makes the same relative trade-off
on a track of fine texture as on one of coarse texture, where a penalty added
in grid px would weigh more on the first. A factor of `2` reads as "this view
counts as twice as blurry".

### The angle term

An error `σ_n` in the patch's normal, in radians, tilts the patch's plane
about its centre. A sample at distance `d` from the centre, along the
direction the plane tilts, moves out of the plane by about `d · σ_n`, and a
view at angle `θ` sees that step as a shift of about `d · σ_n · tan θ` along
its tilt direction ([§ "The measurements"](#the-measurements)). The tile is
anchored on the keypoint, so its centre does not move. Over an `R×R` tile the
root mean square of `d` along one axis is `R/√12`, the standard deviation of a
uniform spread over `R`, so the root mean square shift is `c · tan θ` grid px
(`ReferenceViewOptions::angle_scale`). It is added to the radius in
quadrature, as two independent spreads in the same unit: a view facing the
patch costs its radius alone, and the angle term overtakes the radius only
where `c · tan θ` approaches it. With the default `σ_n = 3°` and `R = 24`,
`c ≈ 0.36` grid px, and `c · tan θ` is about `0.21` at 30°, `0.36` at 45°,
`0.78` at 65° and `2.06` at 80°. With `σ_n = 0` the term is `0` and the rule
ranks by the radius and the penalties alone.

A view at `90°` or more is back-facing and has no cost
([§ "Hard limits"](#hard-limits)), so every angle term the rule ranks by is
finite, though it grows without bound as `θ` nears `90°`.

### The penalties

Each penalty is a **hinge**, `1` up to a band `x₀` and growing linearly past
it with a scale `s`:

```
F(x) = 1 + max(0, x − x₀) / s
```

so `s` is the excess over the band at which the view counts as twice as
blurry. Inside the band the factor is exactly `1`, so a view within every band
is ranked by `r_eff` alone; past it the factor grows without a step, so a view
slightly past a band is slightly worse rather than out.

| Factor | Reading `x` |
|---|---|
| `A`, agreement | the shortfall of the view's pair ZNCC below the best pair ZNCC among the views the pick is made from |
| `K`, the ninths | the cell deficit |
| `C`, coverage | the shortfall of the coverage below `coverage_band`, with a band of `0` |
| `P`, clipping | the clipped share |

**A sharp view is not penalised for detail the others lack.** A sharp view
correlates worse with blurrier views, since its detail is missing from them.
The agreement penalty reads the shortfall below the best view's pair ZNCC
rather than an absolute level, and its band keeps a view within `0.15` of the
best unpenalised. The cell deficit measures a view's ninths against the
track's typical agreement, so a sharp view is not penalised by it either.

### Hard limits

A view that fails any of three limits is never the reference, whatever its
cost, and is left out of the views the pick is made from
(`ReferenceExclusion`):

- **No tile** (`no_tile`): the view has no coverage reading.
- **Coverage under `min_coverage`** (`coverage`), `0.9`: a bitmap taken from
  the view's tile would have holes over a tenth of the patch or more. This
  floor is set by decision and is not tuned. Between it and `coverage_band`
  the coverage penalty applies.
- **Back-facing** (`back_facing`): the view sees the patch at a viewing angle
  of `90°` or more, edge on or from behind. No patch is drawn on such a view:
  the 3D viewer culls a patch whose outward normal faces away from the camera
  ([`patch.wgsl`](../../../crates/sfm-explorer/src/shaders/patch.wgsl)), Image
  Detail draws no ghost outline of a bench track's patch in a view that sees
  its back face (`OrientedPatch::is_front_facing`, in
  [`image_detail/bench_track.rs`](../../../crates/sfm-explorer/src/image_detail/bench_track.rs)),
  and view selection admits no candidate view the patch does not face
  ([`view_selection.rs`](../../../crates/sfmtool-core/src/patch/view_selection.rs)).
  So such a view is never the reference, whatever the other views are; the
  limit is set by decision. A view with no angle reading is not back-facing
  ([§ "Missing readings"](#missing-readings)).

A view meeting all three is **eligible**, and the pick is made from the
eligible views. A track whose every view is back-facing has no eligible view,
whatever else its views read.

### Missing readings

A reading the view does not have never makes it look better than a view that
has it:

- **No radius**: `r` is the largest radius among the views the pick is made
  from, so the view ranks as no sharper than the bluntest view that was read
  (`ReferenceCost::radius_read` is `false`). Where no view has one, every view
  takes `unread_radius`, and the angle and the penalties pick.
- **No pair ZNCC**: the shortfall is the largest among the views that have
  one. Where none has one, `A = 1` for every view.
- **No viewing angle**: the angle term is the largest among the views that
  have an angle. Where none has one, it is `0` for every view.
- **No clipped share or cell deficit**: the factor is `1`, since nothing is
  there to count against the view.

Where no view is eligible the rule picks nothing.

### The costliest term

`ln cost` splits exactly into one part per term,
`ln r + ln(r_eff / r) + ln A + ln K + ln C + ln P` (`ReferenceCost::ln_parts`,
the terms `sharpness`, `angle`, `agreement`, `cells`, `coverage` and
`clipped`), so a view's `ln(cost / pick cost)` splits into the differences of
those parts. The **costliest term** is the term with the largest difference:
the one that adds most to the view's cost over the pick's. Track View and the
wire name it beside each row. A difference that is not a number, which only a
radius of `0` gives, is passed over.

### Each parameter and its reason

The values of `ReferenceViewOptions` were measured against the maintainer's
hand picks of the reference view, on the two ground truths and on a pool of
tracks from ten datasets ([§ "How the values were measured"](#how-the-values-were-measured)).
The held-out half confirmed none of the values the tuning half of the hand
picks preferred, so every default is the value the cost was first built
with. The hand picks decide only some of the values, and for the rest the
table says what they rule out.
`min_coverage` is set by decision and was not swept.

| Field | Default | Reason |
|---|---|---|
| `normal_error_deg`, `σ_n` | `3` | The normal error the angle term expects. An oblique tile depends most on the patch model, since an error in the normal shears it by an amount that grows with `tan θ`. Measured: the term is worth having. Against `σ_n = 0` it changes 85 of 679 pool picks, the `σ_n = 0` pick losing on its angle term in 77 of them, matches the hand pick within 10° of its viewing angle on 63 of 77 hand picks against 60, picks above 65° on 76 of 1,318 tracks against 93 (the previous rule with its 65° gate: 75), and on the 129 ground-truth tracks whose pick it changes the other views align with a mean error of 0.335 px against 0.407. The hand picks do not tell `2` to `6` apart (lenient 69 of 77, exact 26 to 28), and lose from `8` up (`12`: 61 lenient); at `8` the angle term at 45° is `0.97` grid px at `R = 24`, as large as a typical radius. |
| `agreement_band` | `0.15` | Sharp views correlate worse with blurrier ones, so a narrow band penalises exactly the sharp views. The previous rule's margin, `0.15`, was set against hand picks. Measured: the hand picks do not tell `0.075` to `0.4` apart (exact 28 or 29, lenient 69 to 71 of 77) and lose exact matches below `0.05` (`0.025`: 23); the tuning half's best, `0.05`, matched 2 fewer held-out picks exactly. `0.075` matches one more pick exactly and two more leniently, one on each half, which is within what one case moves. The ground truths favour a narrower band: the mean error over the 639 tracks is 0.414 px at `0.075` and 0.403 at `0.05`, against 0.434 at `0.15`. |
| `agreement_scale` | `0.15` | A view `0.30` below the best counts as twice as blurry. Measured: from `0.015` to `1.5` it changes no more than one hand pick. |
| `cell_deficit_band` | `0.3` | A view that agrees overall but not in one part of the tile: an occluder, a shadow edge, parallax within the tile. The previous rule's bar, `0.3`, was set against hand picks. Measured: `0.25` and `0.3` match 28 hand picks of 77 exactly and 69 or 70 leniently; `0.1`, the tuning half's best, matches 3 more leniently but 4 fewer exactly, all 4 of them on the held-out half; from `0.35` up the lenient count falls (`0.4`: 67). |
| `cell_deficit_scale` | `0.3` | A cell deficit of `0.6` counts as twice as blurry. Measured: `0.03` to `0.6` give the same exact and lenient counts over the 77 hand picks; from `0.9` up the lenient count falls (`1.5`: 65). |
| `coverage_band` | `0.99` | A partial tile holds only part of the patch; `0.99` lets a sample or two at a corner fall off the photograph. Measured: no hand pick turns on it, from `0.95` to `1`; 15 of the 679 pool tracks have an eligible view covering between `0.9` and `0.99`. |
| `coverage_scale` | `0.03` | A view covering `0.96` of its tile counts as twice as blurry, and one at the floor of `0.9` four times. Measured: no hand pick turns on it, from `0.003` to `0.3`. |
| `clipped_band` | `0.05` | A clipped region holds neither texture nor its true colour. Measured: below `0.05` the hand picks lose (`0`: 23 exact and 66 lenient of 77, against 28 and 69); above it they hardly change. 95 of the 679 pool tracks have an eligible view clipped past `0.05`. |
| `clipped_scale` | `0.05` | A view with a tenth of its pixels clipped counts as twice as blurry. Measured: below `0.05` one exact match is lost; from `0.05` to `0.5` the hand picks change by at most one. |
| `min_coverage` | `0.9` | The hard limit ([§ "Hard limits"](#hard-limits)). Set by decision. |
| `unread_radius` | `1` grid px | The radius every view takes where no view has one. The per-dataset medians of the stored bitmap's radius run from 0.40 to 1.06 grid px ([§ "Cost of the stored bitmap"](#cost-of-the-stored-bitmap)). Not measured: no eligible view of the 679 pool tracks lacks a radius, so no track reads it. |

Two constants are part of how the cell deficit is read rather than of the
rule:

| Constant | Value | Reason |
|---|---|---|
| `REFERENCE_MIN_JUDGED_CELL_ZNCC` | `0.5` | A cell whose views do not agree, because it holds no texture they share, says nothing about any one view. |
| `REFERENCE_MIN_CELL_SAMPLES` | `16` | Below it a cell's ZNCC is read off a handful of samples. |

**Why one cost rather than a series of tests.** The rule the cost replaced
passed the views through all-or-nothing tests (coverage at least `0.99`,
clipped share at most `0.05`, viewing angle at most `65°`, cell deficit at
most `0.3`, pair ZNCC within `0.15` of the best candidate's, the tests
dropped in turn where no view passed) and then took the smallest radius. The
angle counted only as a gate, so among the views under 65° it had no say: a
view at 20° with a radius of 0.62 grid px lost to one at 55° with 0.60, though
the first depends far less on the patch's normal being right. Each other test
behaved the same way: a view just past a threshold was out however good it was
on the rest, and a view just inside one counted the same as one far inside
it. Where every test was dropped, the point stored a mean of its views and
named no reference. The cost weighs every reading against the others, and the
bands of its penalties start at those thresholds.

**Why a hinge.** Each penalty is `1` exactly inside its band, so a view that
passes every one of the previous rule's tests is ranked by its effective
radius alone; it grows past the band without a step, so a view slightly past
a band is slightly worse rather than out; and it has two parameters, each with
a plain meaning, the fewest that set both where the penalty starts and how
fast it grows. A quadratic past the band, a softplus or an exponential would
each give up one of these.

**Back-facing views were once ranked.** An earlier design picked from the
views behind the patch where every view saw it from behind, with the angle
measured against the flipped normal (`θ' = 180° − θ`). Back-facing views are
left out instead because no patch is drawn on them
([§ "Hard limits"](#hard-limits)); none of the 679 pool tracks has one.

### How the values were measured

With the harness in [`scripts/reference_view/`](../../../scripts/reference_view/README.md),
on 2026-10-11:

- **The hand picks.** The maintainer picked, on 2026-10-06, the view whose
  tile should stand for the point on 79 tracks of nine datasets (the two
  ground truths, 15 each, and seven of each of seven solves) and marked two
  as bad tracks, so 77 are scored. Each case's point was found again by its
  views' images and keypoints, put on the bench with exactly those views and
  evaluated, and the rule run in Python over the readings: the Python rule
  gives the Rust rule's pick on all 1,318 tracks measured, and its costs to
  within a relative `4e-16`. The cases were split in half by dataset with a
  seeded split, 39 for tuning and 38 held out. A pick matches **exactly**
  when it is the hand-picked view, and **leniently** when it is at most 0.1
  grid px blurrier and at most 0.05 less typical (pair ZNCC) than the hand
  pick, so a view as sharp and as typical as it; chance is about one in nine
  for the first. The lenient count reads no angle, so it is also counted with
  the two views' viewing angles within 10° of each other.
- **The sweep.** Each parameter was swept alone from the starting values:
  `σ_n` from `0` to `20`, each band around the previous rule's threshold and
  each scale from a tenth of its band to several times it. The candidate for
  each parameter is the middle of the run of values tied best on the tuning
  half, by lenient count and then exact count, and it is kept only where the
  held-out half scores at least as well on both counts and better on one.
  Four parameters had a candidate other than their starting value on the
  tuning half (`σ_n` `4`, `agreement_band` `0.05`, `cell_deficit_band` `0.1`,
  `cell_deficit_scale` `0.03`), each one or two picks better there, and the
  held-out half confirmed none of them; for the other six the starting
  value is in the tuning half's best run.
- **The pool**: the 79 review cases and 60 tracks of three or more
  observations sampled from each of ten datasets with a fixed seed (the nine
  of the hand picks and the OmniCoast 360° rig), 679 tracks.
- **The ground truths**: every point of three or more observations of the
  seoul_bull and kerry_park ground truths, 639 tracks. For each eligible view
  of each track, the other views were aligned to its tile by the localizer
  with its gates off, and each view's error is its distance from the point's
  projection at the stored pose. A pick's error is the mean error of the other
  views aligned to it; where two rules pick differently, the two are compared
  on the views that are neither pick. A few tracks whose views align badly to
  any tile carry the means, so the median difference is given beside the
  mean.

| Hand picks: exact / lenient / lenient within 10° | Tuning (39) | Held out (38) | All (77) |
|---|---|---|---|
| **The defaults** | **14 / 34 / 31** | **14 / 35 / 32** | **28 / 69 / 63** |
| `σ_n = 0` | 13 / 34 / 30 | 14 / 34 / 30 | 27 / 68 / 60 |
| The previous rule of gates, on the same readings | 12 / 35 / 30 | 15 / 35 / 31 | 27 / 70 / 61 |

The cost matches the hand picks as well as the rule of gates it replaced: one
more exactly, one fewer leniently, two more within the angle bound, on the
same readings. Over the pool it picks differently from the previous rule on
141 of 679 tracks, the previous rule's pick losing on the angle term in 76 of
them, on sharpness in 32 and on clipping in 22. On the ground truths those
changes come to a mean error of 0.371 px against the previous rule's 0.329 on
152 tracks, better on 74 and worse on 78, a median difference of `+0.002`
px: the mean is carried by one kerry_park track, 4.4 px worse, without which
the mean difference is `+0.013` px. The angle term holds the oblique picks to
what the 65° gate allowed: 76 picks above 65° on 1,318 tracks, against the
previous rule's 75 and 93 with `σ_n = 0`.

**What was measured on the previous rule.** The thresholds the bands start at
were set against the same hand picks with the same split procedure on the
readings of 2026-10-06: on the held-out half that rule picked the hand-picked
view on 10 of 38 tracks and a view as sharp and as typical as it on 35 of 38.
Cell bars from `0.2` to `0.4` gave the same agreement on the tuning half, and
margins from `0.10` to `0.20` were within one case of each other; where an
earlier margin of `0.05` missed, the hand-picked view sat `0.06` to `0.21`
below the best. The `65°` gate changed no pick on the review cases and 0.9% of
picks over the whole datasets; on the two ground truths, removing it made the
picks it changed localize the other views 0.12 px worse on average (15
tracks). Two signals were tried and left out: a brightness and colour gate,
which turned away views the hand picks chose, and a contrast floor, which the
agreement margin already covered.

### Why the agreement is read plain

The agreement penalty and the cell readings could read blur-matched
agreement instead ([blur-matched-zncc.md](blur-matched-zncc.md)): before two
views' tiles are correlated, a tile sharper than the other along every
direction is blurred by a round Gaussian to the other's sharpness along its
sharpest direction, so a sharp view is counted as disagreeing less for detail
a blurrier view lacks.

**On the cost.** Each measured track's pair ZNCC was read again from its tiles
with blur matching: each pair correlated with the sharper tile as the bitmap
(`score_against_bitmap`), blurred where the score against the stored bitmap
would blur it, the median taken over the other views. Read the same way
without blur matching, the tiles' pair ZNCC is the bench's to within `1e-3` on
all but 27 of 5,798 rows. Blur matching moves 120 of the 859 rows' pair ZNCC,
by `+0.017` on average, and changes the pick on none of the 77 hand-picked
cases and on 2 of the 670 pool tracks that have the reading (a track whose
patch is at infinity is not read); swept on blur-matched readings, the
agreement band and scale give the same counts as on plain ones within one
case. The band already lets a sharp view sit `0.15` below the best unpenalised,
the protection blur matching would give it, so the rule reads the plain pair
ZNCC. The cell deficit was not measured blur-matched on the cost.

**On the previous rule.** There the agreement and cell tests were gates, and
were measured the same way against the same hand picks, with the thresholds
tuned on the readings of 2026-10-06:

| Readings | Margin, cell bar | Tune exact / lenient (39) | Held-out (38) | All (77) |
|---|---|---|---|---|
| **Plain** | **0.15, 0.3** | **18 / 35** | **10 / 35** | **28 / 70** |
| Blur-matched agreement only | 0.15, 0.3 | 18 / 35 | 10 / 35 | 28 / 70 |
| Blur-matched cells only | 0.15, 0.3 | 18 / 35 | 10 / 35 | 28 / 70 |
| Both blur-matched, every difference | 0.15, 0.3 | 18 / 36 | 10 / 35 | 28 / 71 |
| Both blur-matched, ratio 1.25 | 0.15, 0.3 | 18 / 36 | 10 / 35 | 28 / 71 |
| Both blur-matched, ratio 1.25 | 0.08, 0.3 | 18 / 35 | 11 / 36 | 29 / 71 |
| Both blur-matched, ratio 1.25 | 0.12, 0.35 | 19 / 36 | 10 / 34 | 29 / 70 |

Blur matching picked the hand pick exactly on as many tracks as the plain
readings, within the lenient bounds on one more, changed the pick on 4 of 661
pool tracks (0.6%), and added 0.13 ms (2%) to a track's evaluation. Blur
matching is read where it changed the result: in each observation's score
against the stored bitmap ([blur-matched-zncc.md](blur-matched-zncc.md)
§ "Scores against the stored bitmap").

## The stored bitmap

A point's patch bitmap, the row of `points3d/patch_bitmaps_y_x_rgba` in an
`.sfmr` file ([sfmr-file-format.md](../../formats/sfmr-file-format.md)), is the
tile of its reference observation, which the rule picks where the point has
none, stored as the rule read it: rendered through
the point's patch re-anchored on that observation's keypoint, at the
reconstruction's patch resolution `R`, with the sampler the sampler rule picks
for that view (`render_patch_bitmap`, `render_reference`). Its colour is the
tile's, a grey tile's value in all three channels, and its alpha is `255` on
the samples on the photograph and `0` on the rest (`bitmap_from_tile`), so a
reader that takes a sample whose alpha is `0` as carrying no data (the
renderer's coverage discard, the self-similarity culls) reads exactly the
samples the rule read. The file records which observation it is, as an index
into the point's own track, in `tracks/reference_observations`.

- **Why not a mean.** The photographs differ in exposure and white balance.
  The ZNCC is blind to those differences, but a mean of the tiles is not:
  without a model of each photograph's brightness and colour shift, a mean over
  differently exposed photographs mixes colours that never appeared together on
  the surface, and its detail is blurred by every view that does not line up
  exactly. A single view has neither problem.
- **What a single view costs.** It keeps that view's noise, and any highlight or
  occluder the measurements missed. The bitmap also changes all at once when
  another view comes to rank higher.
- **Every pick is stored.** The rule's pick is always the view whose tile is
  stored, so the row Track View marks as the rule's pick is the stored
  bitmap's tile wherever the bitmap was rendered by the rule.
- **Where the rule picks no view.** That happens only where no view is
  eligible: none renders a tile, none has `min_coverage` of its tile on the
  photograph, or every one is back-facing
  ([§ "Hard limits"](#hard-limits)). The point then gets **no bitmap** and
  names no reference observation (`-1`): `render_patch_bitmap` returns `None`,
  the kernels report no bitmap as they do for a point with fewer than two
  views ([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md#outputs)
  § "Outputs"), a whole-cloud render (`render_patch_cloud_bitmaps`) gives the
  point a zero row and `-1`, and `sfm embed-patches` drops it as it drops any
  point with no bitmap. The floor is set so that a tile under it is not a
  reference, so the rule does not fall back to the most covered view under
  it, nor to a mean of the views. A point with fewer than two views gets no
  bitmap either. With every stored reference cleared, the rule finds no
  eligible view for 57 of the 1,171 points of the BadlandPanorama solve, every
  one of whose views has less than 90% of its tile on the photograph, for 1
  of the 18,992 points of a `dino_dog_toy` solve, and for none of the points
  of the eight other datasets of
  [§ "How the values were measured"](#how-the-values-were-measured).
- **A fused mean already in a file** stays readable. A file older than
  `.sfmr` version 12, or one written before the stored bitmap stopped being a
  mean of the views, can hold a fused mean beside `-1`; readers treat it as
  any bitmap beside `-1`, and the next render replaces it with the rule's
  pick (on the bench, at the first evaluation that picks a row).

**The reference says what to render.** `tracks/reference_observations`
names, per point, the observation its bitmap is, or is to be, rendered from,
with or without stored bitmaps. Where a reconstruction stores a point's bitmap
beside a reference `≥ 0`, the bitmap is that observation's tile as of the last
render. `-1` means the point has no reference observation in use -- none in
its track, or a stored one that a writer aligning keypoints and rendering
bitmaps could no longer render usefully and had no pick to replace (below) --
and a stored bitmap beside it is not the render of
one of its observations: a fused mean an older file stores, or the render of
an observation that an edit has since removed from the point (an image subset or filter keeps the
bitmap and writes `-1`). The next render picks a new reference by the rule.
Dropping the bitmaps keeps the column, and moving geometry
(keypoints, the frame, a bundle adjustment) keeps it too, except in a writer
that aligns keypoints and renders the bitmaps from them
([sfmr-file-format.md](../../formats/sfmr-file-format.md) § "9. Tracks").

**A render reads the reference.** `render_patch_cloud_bitmaps` renders each
point that stores a reference `≥ 0` from that observation's tile at its
keypoint and keeps the reference, so dropping the bitmaps and adding them
again gives the same bitmaps. Where that observation's photograph is not to
hand, the point gets a zero row and still keeps its reference; the rule does
not pick another view for it. Only a point at `-1` runs the rule
(`UnreferencedPoints::Pick`), and its pick is recorded;
`UnreferencedPoints::Skip` gives such a point a zero row and `-1`, for a
caller that has bitmaps for those points already (Python:
`PatchCloud.render_bitmaps(..., referenced_only=True)`). The keypoint
refiners align every view to a given reference, the point's stored one, and
run the rule over the views they keep only where there is none or it cannot
be used. `xform --refine-keypoints` with `bitmaps` and `sfm embed-patches`
record the reference the views were aligned to, so the keypoints, the bitmap
and the reference agree: a point keeps its stored reference wherever it
renders usefully, and a stored reference whose core does not render at its
keypoint (out of frame), or that is past the obliquity cut (`embed-patches`'
`--max-obliquity-deg`, or the localizer's grazing pre-filter), is replaced by
the rule's pick, which is recorded, or by `-1` with no bitmap where the rule
finds no eligible view, though the observation stays in the track.
A writer that aligns keypoints and writes no bitmap (`xform
--localize-keypoints`, `--refine-keypoints bitmaps=false`) keeps each stored
reference instead, since no bitmap is written for it to disagree with. The
writers that render then render every point with a
reference again from that observation through the stored value, through
`render_from_references` in
[`_patch_compaction.py`](../../../src/sfmtool/_patch_compaction.py). That
render reads the `f32` keypoints and frame the file stores rather than the
refiner's `f64` working values, so dropping and adding the bitmaps gives the
same bytes.

**On the bench the pin of the reference row holds the reference.** A track
on the bench holds a reference observation (`TrackPayload::reference`), read
from the point's stored reference when it is put on the bench, with every row
pinned. Every bench render -- the live evaluation's, a fit's, a normal
step's -- renders from that reference while its row is `in`, has a keypoint
and is pinned, and keeps it, whichever row the rule would pick. Where the
reference is undefined (the point stored `-1`, or a step deleted the
reference row, split it off or turned it `out`, which drops the bitmap and the
reference together) or its row is unpinned, the render takes the rule's pick
and sets the reference with the bitmap. So unpinning the reference row hands
the reference to the rule at the next evaluation, which renders the bitmap
from the pick and scores every row against it, and while the row is unpinned
the reference follows the pick. *Set as reference* (`set_reference`) makes a
row the reference and pins it. A pick only the viewer's display render made
reaches the bench like a stored reference and is held the same way; it is the
rule's pick on the file's track, so holding it is the same as the bench
setting it from the rule. A commit writes the reference the bench holds, so
committing a point adopts a display pick
([editable-track.md](../bench/editable-track.md) § "The stored bitmap's
reference").

**The rule's pick and the reference in use are two things.** Every
evaluation still runs the rule over the `in` rows and reports each row's
standing: information about the current readings. The reference in use is the
row the bitmap is the render of. The two differ only while the reference row
is pinned. Track View's *Reference* column marks the reference row green where
it is the rule's pick and red where it is not, with the pick marked in its own
cell, and the wire names the reference in use as `reference_observation` and
the rule's pick as `reference_view_observation`
([../../gui/track-view.md](../../gui/track-view.md)).

Every operation that renders the stored bitmap renders it this way:

| Operation | Where |
|---|---|
| `sfm xform --add-patch-bitmaps` | `render_patch_cloud_bitmaps`, through `PatchCloud.render_bitmaps`: each point from its stored reference, the rule only for a point at `-1` |
| `sfm embed-patches` and `sfm xform --refine-keypoints bitmaps=…` | the sub-pixel refiner with `render_bitmaps`, at the final keypoints; a point that stores a reference is then rendered from it (`render_from_references`) |
| `sfm xform --refine-normals bitmaps=…` | normal refinement with `render_bitmap`, through the refined patch; a point that stores a reference is then rendered from it (`render_from_references`) |
| A bench fit, `render_bitmap_in_place` and `evaluate_rendering_bitmap` | the tile of the track's reference where its row holds it, `in`, keyed and pinned; `render_patch_bitmap` over the `in` rows where it does not ([editable-track.md](../bench/editable-track.md)) |
| The viewer's display patch bitmaps | `render_patch_cloud_bitmaps`, through `render_patch_bitmap_column`: each point from the file's reference; a point at `-1` gets the render's pick, held in the value's references and marked `PointSet::display_only_references`, so Track View marks the row the display bitmap is the tile of, while a save writes neither the bitmaps nor those picks (it writes the file's `-1`) |
| SfM Explorer's conversion to embedded patches | `render_patch_cloud_bitmaps`, through `render_patch_bitmap_column`; `to_embedded_patches` keeps the input's references (a display-only pick goes back to `-1`; an input without the column gets every row `-1`) and drops the old bitmaps; a point with a reference is rendered from it, a point at `-1` gets the rule's pick, and the bitmaps are the converted value's own, so the references are written with them |

The localizer, the sub-pixel refiner and Add Image to Tracks align each view
to the same reference render: the tile of the point's reference observation
at its keypoint, or the rule's pick where the point stores none; where the
rule finds no eligible view there is no reference render, and every view keeps
its starting keypoint, unscored
([patch-keypoint-localization.md](patch-keypoint-localization.md)). Normal
refinement keeps its own weighted consensus of the views, since it chooses a
normal rather than placing views
([patch-normal-refinement.md](patch-normal-refinement.md)).

**The culls read a sharper bitmap.** `embed-patches`' and
`--filter-by-zncc-self-similarity-radius`' cull on a point's bitmap reads its
self-similarity radius, which is the reference view's own and so shorter than a
mean's was. Fewer points are culled at the same bar; the bar was not
re-measured.

## Implementation notes

**Where the cell readings come from.** The pair ZNCC is read from member
coherence's matrix, but the pair ZNCC grid is read from the tiles already
rendered, with each pair's own support and every sample weighted equally.
Member coherence renders only the samples inside its window's disk and common
to every view, so its corner cells would hold only the part of their square
inside the disk, and the cell readings were measured on whole cells. The cost is
one extra member-coherence render per view and `k(k − 1)/2` pairs of nine
cells; on the review tracks (6 to 26 views) the bench's `reference view` phase
takes 0.4 to 2.4 ms, about 9% of an evaluation.

**What the stored bitmap costs.** Rendering a point's bitmap renders each
view's tile once, reads its self-similarity, builds member coherence's matrix
and correlates the cells, where the fused mean it replaced rendered each view
twice (once on the support for the weights and once whole) and blended. The
timings over real reconstructions are in
[§ "Cost of the stored bitmap"](#cost-of-the-stored-bitmap).

**The clipped share reads each pixel once.** A tile far from its camera covers
hundreds of thousands of photograph pixels. `clipped_share` walks the
outline's bounding box one row of pixel centres at a time, finds where the row
crosses the outline's edges, and reads the pixels between alternate crossings,
so it counts the same pixels a test of every centre against every edge counts,
which a test checks on concave, self-crossing and off-photograph outlines.

**A keypoint whose ray misses the patch's plane.** Anchoring fails when the
keypoint's ray does not meet the patch's plane in front of the camera, and the
tile is then rendered through the patch where it stands; the viewing angle is
read at the same placement, so it is the angle at the patch's centre rather
than at the keypoint. On the review cases this happens to one view of one track
(a 360° rig), whose angle reads 10° at the patch's centre where the keypoint's
ray reads 111°.

**The self-similarity of a tile that crosses the border.** The bench reads a
tile's self-similarity over its samples on the photograph, so a partial tile
is judged on what it holds. A view with up to 1% of its tile off the photograph
pays no coverage penalty.

## Cost of the stored bitmap

Measured when the stored bitmap was still the fused mean of the views, against
the reference render that replaced it, on the 661 pool tracks of the ten
datasets (the review cases and 60 of each dataset's pool), each put on the
bench from its stored point and evaluated, one thread, release build, best of
three:

| | Median | p90 | Total |
|---|---|---|---|
| Fused mean, per track | 1.30 ms | 3.52 ms | 1.14 s |
| Reference render (`render_patch_bitmap`), per track | 1.24 ms | 4.37 ms | 1.33 s |
| Ratio, reference to mean | 0.94 | 1.15 | |

Over whole clouds, with every thread (32), on samples of up to 3,600 points:

| Dataset | Points | Fused mean | Reference render |
|---|---|---|---|
| OmniCoast (360° rig) | 1,095 | 0.14 s | 0.15 s |
| dino_dog_toy | 3,166 | 0.17 s | 0.18 s |
| MossyRailing | 1,352 | 0.09 s | 0.10 s |
| ChristmasTreeWithPresents | 3,631 | 0.24 s | 0.24 s |

The render reads every view once at `R` where the mean read it twice, and pays
instead for each tile's self-similarity and member coherence's matrix; the two
come out within 10% of each other. `sfm xform --add-patch-bitmaps` over the
1,095 points of OmniCoast takes 2.1 s, the photographs' decode included.

**What it stores.** Measured on the previous rule of gates
([§ "Each parameter and its reason"](#each-parameter-and-its-reason)): on the
same 661 tracks the evaluation's pick, the view whose
tile `render_patch_bitmap` stored and the bench payload's reference were the
same row on every track, and the rule picked a view on every one. The stored
bitmap's self-similarity semi-major axis fell from a median of 1.21 grid px for
the fused mean to 0.66 for the reference view's tile (p10 0.48 to 0.27, p90
3.0 to 1.9), shorter on 624 of 654 tracks, on every dataset:

| Dataset | Fused mean | Reference render |
|---|---|---|
| Altona (oblique) | 0.82 | 0.40 |
| BadlandPanorama (distant) | 1.16 | 1.04 |
| dino_dog_toy | 2.37 | 1.06 |
| KerryPark480 (fisheye solve) | 1.25 | 0.68 |
| kerry_park ground truth | 1.43 | 0.73 |
| MossyRailing | 1.07 | 0.48 |
| OmniCoast | 1.02 | 0.54 |
| OmniTemple1 | 1.08 | 0.70 |
| seoul_bull ground truth | 1.22 | 0.73 |
| ChristmasTreeWithPresents | 1.21 | 0.90 |

(Median semi-major axis of the stored bitmap, grid px.)

## Parameters

The rule's parameters are the fields of `ReferenceViewOptions`, defined in
[rule.rs](../../../crates/sfmtool-core/src/patch/reference_view/rule.rs), and
the cell readings' constants are in
[reference_view.rs](../../../crates/sfmtool-core/src/patch/reference_view.rs)
([§ "Each parameter and its reason"](#each-parameter-and-its-reason)). The
bench's evaluation runs the rule with `EvaluateOptions::reference_view`, the
stored bitmap's render and the sub-pixel refiner with
`KeypointSubpixelParams::reference_view`, and
every other caller (the localizer, normal refinement, and the bench's
restatement of the rule after a verdict moves) with the `Default`, which
`EvaluateOptions` and `KeypointSubpixelParams` also hold by default.
`MIN_TILT_ANGLE_DEG = 0.1` is in
[obliquity.rs](../../../crates/sfmtool-core/src/patch/normal_refine/obliquity.rs).
The bench renders the tiles at the reconstruction's patch resolution with the
localizer's sampler choice (`EvaluateOptions::localize.sampler`, the sampler
rule by default), and runs member coherence with its default window
(`GaussianDisk { sigma: 0.6 }`) at the same resolution and sampler. The stored
bitmap renders at the caller's `KeypointSubpixelParams::resolution` (the
reconstruction's patch resolution) with its `sampler` (the sampler rule by
default).

## Python bindings

The bench's readings cross on each observation's `"track"` dict
([editable-track.md](../bench/editable-track.md) § "Python bindings"):
`viewing_angle_deg`, `tilt_direction_deg`, `coverage`, `clipped_share`,
`pair_zncc` and `cell_deficit` as floats, `pair_zncc_grid` as a `(3, 3)`
float64 array, and `reference_view` as a dict `{"is_reference", "cost",
"cost_ratio", "terms", "costliest_term", "excluded_by"}`: `cost` the
row's cost in grid px and `cost_ratio` its cost over the pick's, floats or
`None` for a row left out; `terms` a dict `{"radius", "radius_read",
"angle_term", "agreement", "cells", "coverage", "clipped"}` of the radius
(`radius_read` whether it is the row's own), the angle term and the four
penalties, or `None` for a row left out; `costliest_term` one of
`"sharpness"`, `"angle"`, `"agreement"`, `"cells"`, `"coverage"`, `"clipped"`,
or `None` for the pick and a row left out; `excluded_by` one of `"no_tile"`,
`"coverage"`, `"back_facing"` or `None`. Each reading is absent where the row
has no such reading.
`EditableTrack.reference_observation` is the row the stored bitmap is the tile
of, the reference in use, and `EditableTrack.reference_view_observation` the
row the rule picked.

`OrientedPatch.render_view_tile(camera, cam_from_world, image, *,
image_index=None, keypoint=None, resolution=24, sampler="per_view")` renders one
view's tile (`render_view_tile`) from a numpy photograph or from an
`ImagePyramidSet` with `image_index`, and returns a dict: `samples` `(R, R, C)`
uint8, `valid` `(R, R)` bool, `sampler`, `coverage`, `clipped_share`,
`viewing_angle_deg`, `tilt_direction_deg`, `jacobian` `(2, 2)` or `None`, and
`placement`, the `OrientedPatch` the tile was rendered through. It raises
`ValueError` for a `resolution` outside 2 to 1024, a keypoint that is not
finite, an `ImagePyramidSet` without `image_index`, or an `image_index` given
with a plain array, which holds one photograph.

```python
tile = patch.render_view_tile(camera, pose, photo, keypoint=(812.4, 377.9))
print(tile["coverage"], tile["clipped_share"], tile["viewing_angle_deg"])
```

`PatchCloud.render_bitmaps(recon, images, *, resolution=24, sampler="per_view", referenced_only=False, progress=None)`
returns `(bitmaps, reference_observations)`: the `(P, R, R, 4)` bitmap column
and the `(P,)` int32 reference observation of each point, which
`clone_with_changes(patch_bitmaps=…, reference_observations=…)` takes.
`refine_keypoints(render_bitmaps=True)` adds `reference_image` to each point's
dict, and `refine_normals(render_bitmaps=True)` adds `reference_images`, `(P,)`
with `-1` for none; `sfmtool._patch_compaction.reference_observations_from_images`
turns those images into the column.

## Testing

[reference_view/tests.rs](../../../crates/sfmtool-core/src/patch/reference_view/tests.rs)
checks the viewing angle and tilt direction on cameras placed round a patch at
known angles, a view facing the patch, one of its back and a patch at infinity;
the clipped share on photographs with known clipped columns and colour pixels,
and that the row spans count what a test of every pixel centre counts; coverage
on a tile inside and one over the photograph's edge; the cell readings on
identical tiles, flat and short cells, an occluder in one cell and cells the
track does not agree on; the clipped share of a tile rendered through a
fisheye camera, against a count of every pixel centre inside the polygon of its
border samples projected one at a time, and of a tile cut by the photograph's
edge. On constructed readings it checks the rule: a view at 20° with a radius
of 0.62 grid px picked over one at 55° with 0.60, and the sharper one picked
with no normal error; each penalty `1` inside its band and growing at its
scale past it; a view at the coverage floor eligible and one just under it and
one with no tile left out; a track whose every view is behind the patch with
no eligible view, and one with views on both sides picked among the
front-facing ones; a view at exactly 90° left out and one at 89.9° eligible;
each kind of missing reading, including a track where no view has a radius
and one whose views with an angle are all back-facing; the tie breaks; and the
costliest term's parts summing to the log of a view's cost over the pick's.
[stored_bitmap/tests.rs](../../../crates/sfmtool-core/src/patch/stored_bitmap/tests.rs)
checks, on three synthetic views of a painted plane, that the stored bitmap is
the tile of the view the rule picks (the sharp one), with alpha marking the
samples on the photograph; that a patch every view sees from behind has no
eligible view and no bitmap; that a patch whose best view covers between
0.9 and 0.99 of its tile stores that view's tile and names it; and that a patch
no view covers to 0.9, also one whose tiles' middles all stay on their
photographs, has no pick and no bitmap.
[display_bitmaps/tests.rs](../../../crates/sfmtool-core/src/patch/display_bitmaps/tests.rs)
checks, on the seoul_bull ground truth, that the whole-cloud render names a
reference within each point's own track, also when some images have no view
(so the index is the observation's, not its place among the views to hand),
and that a point at `-1` whose patch is turned away from every camera gets a
zero row and keeps `-1` while every other point renders as before.
[bench/tests/reference_view.rs](../../../crates/sfmtool-core/src/bench/tests/reference_view.rs)
evaluates a track of the seoul_bull ground truth and checks that the rule picks
exactly one view, with coverage of at least 0.9 and facing the patch, at the
lowest cost, and that every other row with a cost names its costliest term;
that a point no view of which covers its tile whole stores the rule's pick, and
that the sub-pixel refiner, the localizer and normal refinement name a
reference for it, and that a track at `-1` on it is rendered from the pick at
its first rendering evaluation; that each row's
self-similarity readings are those of its tile rendered directly; that each
row's pair ZNCC is the median of its row of `member_zncc_matrix` called
directly; that rendering the bitmap where the track stands stores the tile of
the row the evaluation picked and names it, and that a commit writes its place
in the stored track; that sighting the reference observation elsewhere, or
a patch step, drops the bitmap and keeps the reference, that turning it `out`
by hand or by the thresholds drops both, while the same step on another row
keeps both; that the bench reads a stored reference with or without a
bitmap, that a display pick reaches the bench, is written as `-1` by a save
and becomes the point's own on a commit; that a track opened from a file whose
reference is not the rule's pick renders from the file's reference, and keeps
rendering from it after a patch step, a sighting of another row or of the
reference row and a fit, while the evaluation reports the rule's pick, that
the render reusing the evaluation's tiles matches the separate calls there,
and that a commit saves that reference; and that turning the reference row
`out` or deleting its image lets the next render set the rule's pick. `display_bitmaps/tests.rs` checks
that a render reads the stored references: storing the rule's picks and
rendering again draws the same column, and storing other observations draws
their tiles under those references.
The Python tests are in
[test_bench_rust_bindings.py](../../../tests/rust_bindings/bench/test_bench_rust_bindings.py)
and
[test_view_tile_rust_bindings.py](../../../tests/rust_bindings/patches/test_view_tile_rust_bindings.py).

## Non-goals

Normal refinement's template stays its own weighted consensus of the views;
the kernels that place views align them to the reference render (§ "The
stored bitmap"). Replacing a point's defined reference when a sharper observation is
added or fitted is not done: a render renders from the defined reference, an
operation that only adds an observation (Add Image to Tracks) keeps the bitmap
and its reference, and the refiners keep a stored reference. Only the bench
replaces a defined reference with the rule's pick, when the person unpins the
reference row, or with another row, by *Set as reference* (§ "The stored
bitmap").
