# Reference View

A point in a patch-based reconstruction carries a small square bitmap of the
surface around it, and every photograph that sees the point can be resampled
into that square: one tile per view. The views are not equally good. A far view
holds less detail, an oblique one depends most on the patch's normal being
right, a tile that crosses the photograph's border holds only part of the
patch, a blown-out highlight holds no texture, and a view that sees part of
the tile at another depth, through parallax, disagrees with the others there.
This spec describes the
measurements that say what each view's tile can contribute, and the
**reference-view rule**, which picks the one view whose tile is stored as the
point's patch bitmap. It gives each view a cost in grid px: the tile's
sharpness, multiplied by a step where that reading is at its cap and so only
a lower bound, by a factor for how oblique the view is and by penalties
for a view that agrees less with the others, differs from them in part of the
tile, falls partly off its photograph or is clipped. The view with
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
    pub typical: [[f64; 3]; 3],             // per cell, median over the views on the photograph
    pub judged: Vec<[[bool; 3]; 3]>,        // per view, the cells the check judges
    pub deficit: Vec<f64>,                  // per view, the cell deficit
}
pub fn cell_agreement(tiles: &[&ViewTile], semi_axes: &[Option<TileSemiAxes>]) -> CellAgreement;

pub struct ReferenceReadings {
    pub coverage: Option<f64>, pub clipped_share: Option<f64>,
    pub viewing_angle_deg: Option<f64>, pub cell_deficit: Option<f64>,
    pub pair_zncc: Option<f64>, pub semi_major: Option<f64>, pub semi_minor: Option<f64>,
    pub max_radius: Option<f64>,        // the cap semi_major was read with
}
pub struct ReferenceViewOptions {      // impl Default: the values below
    pub angle_weight: f64,              // k
    pub capped_radius_factor: f64,      // the step on a radius at the cap
    pub agreement_band: f64, pub agreement_scale: f64,
    pub cell_deficit_band: f64, pub cell_deficit_scale: f64,
    pub coverage_band: f64, pub coverage_scale: f64,
    pub clipped_band: f64, pub clipped_scale: f64,
    pub min_coverage: f64,              // the hard limit
    pub unread_radius: f64,             // r where no view has a radius
}
impl ReferenceViewOptions { pub fn angle_factor(&self, viewing_angle_deg: f64) -> f64; } // 1 + k·tan θ
pub enum ReferenceTerm { Sharpness, CappedRadius, Angle, Agreement, Cells, Coverage, Clipped }
pub enum ReferenceExclusion { NoTile, Coverage, BackFacing }
pub struct ReferenceCost {
    pub radius: f64, pub radius_read: bool, // r, and whether it is the view's own
    pub capped_radius: f64,                 // the step, capped_radius_factor or 1
    pub angle: f64,                         // the angle factor, 1 + k·tan θ
    pub agreement: f64, pub cells: f64, pub coverage: f64, pub clipped: f64,
    pub cost: f64,
}
impl ReferenceCost { pub fn ln_parts(&self) -> [(ReferenceTerm, f64); 7]; } // ln cost, split by term
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
pub fn choose_reference_view(views: &[ReferenceReadings],
    options: &ReferenceViewOptions) -> ReferenceChoice;

// The rule over one track's tiles: the pair ZNCC from member coherence's
// matrix, the cells from the tiles, and the pick.
pub struct TrackReading {
    pub pair_zncc: Vec<Option<f64>>,
    pub cells: CellAgreement,
    pub readings: Vec<ReferenceReadings>,
    pub choice: ReferenceChoice,
}
pub struct TileSemiAxes {
    pub whole: [f64; 2],              // [major, minor] of the whole tile, grid px
    pub grid: [[[f64; 2]; 3]; 3],     // each cell's, NaN for a cell with no data
}
pub fn tile_semi_axes(tile: &ViewTile) -> Option<TileSemiAxes>;
pub fn tile_max_radius() -> f64; // the cap of tile_semi_axes' reading and the bench's, 3
pub fn readings_of(tile: &ViewTile, pair_zncc: Option<f64>, cell_deficit: Option<f64>,
    semi_axes: Option<[f64; 2]>) -> ReferenceReadings;
pub fn read_track(patch: &OrientedPatch, views: &[ProjectedImage<'_>], members: &[u32],
    keypoints: &[Option<[f64; 2]>], tiles: &[&ViewTile], semi_axes: &[Option<TileSemiAxes>],
    sampler: SamplerChoice, options: &ReferenceViewOptions,
    progress: &Progress<'_>) -> TrackReading;

// patch::stored_bitmap
pub struct PatchBitmap { pub rgba: Vec<u8>, pub reference: usize } // the pick
pub fn bitmap_from_tile(tile: &ViewTile) -> Vec<u8>; // alpha 255 on the photograph
pub struct ReferenceRender { pub tiles: Vec<ViewTile>, pub semi_axes: Vec<Option<TileSemiAxes>>,
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
under) runs the same rule. It does not take the tiles' resolution: the radius
is read in grid px of the tile, and every other term is a ratio. Each reading
is an `Option` because the views
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

Each view's readings carry the cap its radius was read with,
`ReferenceReadings::max_radius`, rather than the rule's options holding one.
The cap is a property of the self-similarity reading
(`SelfSimilarityParams::max_radius`), not a choice of how the rule weighs its
readings, so it travels with the radius it qualifies: a sweep of the options
cannot set it out of step with the reading, and a caller that read with
another cap says so on each view. The rule compares the radius with the cap
itself, so the callers do not each write the test. Every caller today reads
with the default parameters: `readings_of` and the bench's restatement of the
rule after a verdict moves fill it with `tile_max_radius()`.

`read_track` takes the readings across one track's tiles and runs the rule, and
both the bench's evaluation and the stored bitmap call it, so the view a bench
row is marked as the reference and the view whose tile a fit stores are picked
from the same readings. It takes the tiles and their self-similarity semi-axes
rather than rendering them because the bench has already rendered every row's
tile and read its self-similarity for its own columns. The semi-axes come
whole and per cell (`TileSemiAxes`) because one reading of the tile
(`zncc_self_similarity_parts`) gives both: the rule ranks by the whole tile's,
and the cell check reads each cell's to tell whether the view holds texture
there.

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
let choice = choose_reference_view(&readings, &ReferenceViewOptions::default());
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
  of the views' pair ZNCC grid in that cell, over the views whose tile has
  every sample of the cell on the photograph. A view's cell deficit is the
  largest amount by which its own cell falls below the typical agreement, over
  the cells it is **judged** in; `0` when it is judged in none. A view is
  judged in a cell whose typical agreement is at least
  `REFERENCE_MIN_JUDGED_CELL_ZNCC`, unless its own tile, in that cell:
  - **holds no texture**: the cell's self-similarity ellipse reads the cap of
    its reading, `max_radius` (3 grid px), along both axes, so the cell
    matches itself shifted anywhere in the search, and its ZNCC with another
    view reads noise or shading. A cell along a straight edge reads at the cap
    along the edge only and is judged, since a view that disagrees across the
    edge is told apart there. The view's reading still counts in the typical
    agreement;
  - **has a sample off the photograph**: its ZNCCs are read over part of the
    cell, and the coverage penalty already charges the view for the samples
    it lacks. Its reading is also left out of the typical agreement.

  What the deficit reads, and why these cells are not judged, is in
  [§ "What the cell check reads"](#what-the-cell-check-reads).

The rule also reads each tile's self-similarity ellipse, whose semi-major axis
is the ZNCC self-similarity radius
([zncc-self-similarity-radius.md](zncc-self-similarity-radius.md)): how far the
tile can slide over itself and still match itself, short on a sharp tile. It
is the whole tile's reading over its samples on the photograph
(`tile_semi_axes`), which reads each cell's ellipse in the same call, for the
cell check.

## The rule

Each view the pick is made from gets a cost in grid px, and the view with the
lowest cost is the reference (`choose_reference_view`):

```
cost = r · S · T · A · K · C · P
S = capped_radius_factor where r reads at the cap, else 1
T = 1 + k · tan θ
```

- `r` is the self-similarity semi-major axis of the view's `R×R` tile, in grid
  px.
- `S` is the step on a radius read at the cap, `capped_radius_factor` where
  `r` is at the cap of its reading (`r ≥ max_radius`, 3 grid px), else `1`
  ([§ "The step on a radius at the cap"](#the-step-on-a-radius-at-the-cap)).
- `T` is the angle factor, `θ` the viewing angle, under `90°` for every view
  the pick is made from ([§ "Hard limits"](#hard-limits)), and `k` its weight,
  `angle_weight` ([§ "The angle factor"](#the-angle-factor)).
- `A`, `K`, `C` and `P` are the penalties for agreement, the ninths, coverage
  and clipping ([§ "The penalties"](#the-penalties)), each `1` inside its band.

A tie on the cost goes to the smaller semi-minor axis, a missing one counting
as infinite, and then to the earlier view.

**Why a product.** The cost stays in the unit of the radius, so a cost reads
as "this view counts as if its tile were this blurry". The radius depends on
the texture as well as on the view and is only compared within one track; a
factor that multiplies makes the same relative trade-off on a track of fine
texture as on one of coarse texture, where a term added in grid px would weigh
more on the first. A factor of `2` reads as "this view counts as twice as
blurry".

### The step on a radius at the cap

The self-similarity reading searches shifts up to `max_radius` (3 grid px),
and a radius that reaches it is capped there and reads "3 or further"
([zncc-self-similarity-radius.md](zncc-self-similarity-radius.md) § "Lower
bounds"): a straight edge, a flat patch and a texture that repeats beyond the
square all read 3. Taken as exactly 3, such a view would beat a view measured
at 2.95 with the same other factors, though the second is the one known to be
sharper. So a view whose radius reads at the cap has its cost multiplied by
`capped_radius_factor`, `1.5`:

```
S = 1.5   where r ≥ max_radius
S = 1     otherwise
```

**Why a step and not a term that grows towards the cap.** A radius below the
cap is a measurement, and 2.95 grid px is as good a reading as 0.5. Only the
reading at the cap is different in kind: it is a bound, and nothing in it says
by how much the true radius is larger. A term that grew as the radius neared
the cap would penalise measured radii for a doubt they do not carry. The step
says only "this view's radius is at least the cap, and counts as blurrier than
that".

**Why the cap and not the lower-bound flag.** The ellipse's
`radius_is_at_least` flag is also set below the cap, where the region of the
surface runs off the square of shifts searched or a gap with no reading could
hide part of it. Such a radius may be larger too, but it is still read from
the region's own extent, so two such readings still order by their length.
At the cap the reading saturates: every true radius from 3 up reads 3, and the
rule cannot order the views among themselves or against an uncapped one. The
step is for that case alone.

**A view with no radius.** It takes the largest radius among the eligible
views ([§ "Missing readings"](#missing-readings)), and with it the step where
that radius reads at the cap: the stand-in is a lower bound exactly when the
reading it is copied from is, so a view with no reading still ranks no better
than the bluntest view that has one. Where no view has a radius every view
takes `unread_radius` with no step; nothing was read, and a step on every view
would change no pick.

**Where it decides.** On the 679 pool tracks, 454 of the 5,845 eligible views
read at the cap; on 19 tracks every eligible view does, and there the step
multiplies every cost alike and changes nothing. With no step, 4 pool tracks
pick a capped view over an uncapped one ([§ "How the values were
measured"](#how-the-values-were-measured)). Each flips to the uncapped view
when the factor passes the uncapped view's cost over the capped pick's:
ChristmasTreeWithPresents pool point 7858 above 1.01 (a view at 2.955 grid px)
and Altona pool point 353 above 1.08 (2.899), and two hand-picked tracks,
altona_p1153 above 2.05 and seoul_bull_p135 above 2.46. On those two the
maintainer picked a capped view as well, and the uncapped views are outliers:
pair ZNCC 0.86 and 0.85 against the pick's 0.98 and 0.95, and cell deficits of
1.13 and 0.83. Any factor from 1.08 to 2.04 changes the first two picks and
keeps the hand picks; `1.5` is near the middle of that range on a log scale
(`√(1.08 · 2.04) = 1.48`).

| `capped_radius_factor` | Hand picks, tuning (39) | Held out (38) | All (77) | Pool picks changed from no step | Ground-truth picks changed: error no step → step, better / worse |
|---|---|---|---|---|---|
| 1 (no step) | 14 / 34 / 32 | 15 / 34 / 30 | 29 / 68 / 62 | 0 | 0 |
| 1.01 | 14 / 34 / 32 | 15 / 34 / 30 | 29 / 68 / 62 | 1 | 0 |
| 1.05 | 14 / 34 / 32 | 15 / 34 / 30 | 29 / 68 / 62 | 1 | 2: 0.231 → 0.275, 0 / 2 |
| 1.1 | 14 / 34 / 32 | 15 / 34 / 30 | 29 / 68 / 62 | 2 | 3: 0.183 → 0.407, 0 / 3 |
| 1.25 | 14 / 34 / 32 | 15 / 34 / 30 | 29 / 68 / 62 | 2 | 3: 0.183 → 0.407, 0 / 3 |
| **1.5** | **14 / 34 / 32** | **15 / 34 / 30** | **29 / 68 / 62** | **2** | **3: 0.183 → 0.407, 0 / 3** |
| 2 | 14 / 34 / 32 | 15 / 34 / 30 | 29 / 68 / 62 | 2 | 3: 0.183 → 0.407, 0 / 3 |
| 2.25 | 14 / 33 / 32 | 15 / 34 / 30 | 29 / 67 / 62 | 3 | 3: 0.183 → 0.407, 0 / 3 |
| 2.5 | 14 / 32 / 31 | 15 / 34 / 30 | 29 / 66 / 61 | 4 | 4: 0.309 → 0.497, 0 / 4 |

(Hand picks against the first review, exact / lenient / lenient within 10°,
the other parameters at their defaults; against the second review every
factor from 1 to 2 matches 72 / 74 / 73 of 74, [§ "The second
review"](#the-second-review). A changed ground-truth pick reads the other
views' mean error, px, aligned to the pick with no step and to the pick with
it, over the views that are neither pick.)

The data does not choose a value between 1.1 and 2. The hand picks are the
same from 1 to 2, so the harness's procedure keeps the starting value, `1.5`,
the maintainer's; they do not tell the step from no step either. The step is
kept for what the reading means, not for a measured gain. The ground truths
lean the other way, on three tracks: from 1.1 to 2 the step changes the pick
on kerry_park points 41 and 125 and seoul_bull point 26, each from a capped
view to an uncapped one (radius 2.46, 2.91 and 2.80 grid px), and the other
views align to the new pick worse on all three, by 0.003, 0.083 and 0.382 px
in the mean over every view but the pick, the measure of the mean over all
tracks below (0.012, 0.077 and 0.584 px over the views that are neither pick,
the table's measure); the last is one view that aligns 1.3 px off. Three
tracks do not establish a direction, and over all 639 ground-truth tracks the
mean error moves from 0.431 to 0.432 px.

### The angle factor

The reference is the template every other view of the point is compared
with: the localizer, the sub-pixel refiner and Add Image to Tracks align each
view to it ([§ "The stored bitmap"](#the-stored-bitmap)). So besides being
sharp it should differ as little as it can from the other views in what the
patch model does not describe, and two such differences grow with the viewing
angle. Surface that stands off the patch's plane by a height `h` shifts
against the plane by about `h · tan θ` along the view's tilt direction
([§ "The measurements"](#the-measurements)), a parallax a view facing the
patch does not have. And an error `σ` in the patch's normal, in radians, tilts
the plane about its centre, so a sample at distance `d` from the centre along
the direction it tilts moves out of the plane by about `d · σ`, which the view
sees as a shift of about `d · σ · tan θ`. Both go like `tan θ`, so the factor
does:

```
T = 1 + k · tan θ
```

`1` facing the patch, and with the default `k = 0.25` about `1.09` at 20°,
`1.14` at 30°, `1.25` at 45°, `1.54` at 65° and `2.42` at 80°. With `k = 0`
the factor is `1` and the rule ranks by the radius and the penalties alone.

**Why `tan θ` and not `1 / cos θ`.** The angle's other effect is
foreshortening: the view sees the patch shortened by `cos θ` along the tilt
direction. The radius already reads most of that, since a foreshortened tile
holds fewer of the photograph's pixels along that direction and so is blurrier
along it, which lengthens its self-similarity semi-major axis. A factor that
follows `1 / cos θ` would count it a second time. Measured, `1 / cos^p θ` does
no better than `1 + k · tan θ` ([§ "The form of the angle
factor"](#the-form-of-the-angle-factor)).

**Why a factor, not a term added in quadrature.** The angle was first a term
in grid px, `c · tan θ` with `c = (R/√12) · σ_n`, the root mean square shift
across an `R×R` tile that a normal error of `σ_n` causes, added to the radius
in quadrature as `√(r² + (c · tan θ)²)`. In quadrature the angle counts only
where the term approaches the radius, so how much it counted depended on the
texture. On a blurry track it had almost no say: on seoul_bull point 10, a
view at 19° with a radius of 1.584 grid px lost to one at 29° with 1.572
(1.589 against 1.585), though the maintainer picked the first; with the factor
they cost 1.720 and 1.792. On a sharp track it counted far more: at
`σ_n = 3°` and `R = 24`, a view at 65° with a radius of 0.3 grid px counted as
2.8 times as blurry, where the factor counts any view at 65° as 1.54 times.
The factor makes the same trade-off between angle and radius whatever the
texture.

A view at `90°` or more is back-facing and has no cost
([§ "Hard limits"](#hard-limits)), so every angle factor the rule ranks by is
finite, though it grows without bound as `θ` nears `90°`.

### The penalties

Each penalty is a **hinge**, `1` up to a band `x₀` and growing linearly past
it with a scale `s`:

```
F(x) = 1 + max(0, x − x₀) / s
```

so `s` is the excess over the band at which the view counts as twice as
blurry. Inside the band the factor is exactly `1`, so a view within every band
is ranked by `r · T` alone; past it the factor grows without a step, so a view
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
  from, with the step where that radius reads at the cap, so the view ranks as
  no sharper than the bluntest view that was read
  (`ReferenceCost::radius_read` is `false`). Where no view has one, every view
  takes `unread_radius` with no step, and the angle and the penalties pick.
- **No pair ZNCC**: the shortfall is the largest among the views that have
  one. Where none has one, `A = 1` for every view.
- **No viewing angle**: the angle factor is the largest among the views
  that have an angle. Where none has one, it is `1` for every view.
- **No clipped share or cell deficit**: the factor is `1`, since nothing is
  there to count against the view.

Where no view is eligible the rule picks nothing.

### The costliest term

`ln cost` splits exactly into one part per term,
`ln r + ln S + ln T + ln A + ln K + ln C + ln P` (`ReferenceCost::ln_parts`,
the terms `sharpness`, `capped_radius`, `angle`, `agreement`, `cells`,
`coverage` and `clipped`), so a view's `ln(cost / pick cost)` splits into the differences of
those parts. The **costliest term** is the term with the largest difference:
the one that adds most to the view's cost over the pick's. Track View and the
wire name it beside each row. A difference that is not a number, which only a
radius of `0` gives, is passed over.

### Each parameter and its reason

The values of `ReferenceViewOptions` were measured against the maintainer's
hand picks of the reference view, on the two ground truths and on a pool of
tracks from ten datasets ([§ "How the values were measured"](#how-the-values-were-measured)).
The held-out half confirmed none of the values the tuning half of the hand
picks preferred, so every default but `angle_weight` and
`capped_radius_factor` is the value the cost was first built with,
`angle_weight` is the value whose picks differ least from those of the angle
term it replaced, and `capped_radius_factor` is the maintainer's value, which
the measurements do not separate from its neighbours. The hand picks decide
only some of the values, and for the rest the table says what they rule out.
`min_coverage` is set by decision and was not swept. The measurements in the
table, except `capped_radius_factor`'s, were made before the step on a radius
at the cap, with `capped_radius_factor = 1`
([§ "How the values were measured"](#how-the-values-were-measured)). The
table's counts are against the first review of the hand picks. The maintainer
then reviewed again the cases where the rule and that review disagree, and the
procedure run again against the second review proposes no change to any value
([§ "The second review"](#the-second-review)).

| Field | Default | Reason |
|---|---|---|
| `angle_weight`, `k` | `0.25` | The angle factor's weight: a view at 45° counts as 1.25 times as blurry as its radius says ([§ "The angle factor"](#the-angle-factor)). Measured: the factor is worth having. Against `k = 0` it changes 68 of 679 pool picks, the `k = 0` pick losing on its angle factor in 65 of them, matches the hand pick within 10° of its viewing angle on 62 of 77 hand picks against 60, picks above 65° on 83 of 1,318 tracks against 93 (the previous rule with its 65° gate: 75), and on the 84 ground-truth tracks whose pick it changes the other views align with a mean error of 0.359 px against 0.478. `0.25` is the value whose picks differ least from the quadrature's at `σ_n = 3°` (63 of 679 pool tracks). The hand picks do not tell `0.05` to `1` apart (lenient 67 or 68 of 77, exact 28 to 30); the tuning half's best, `0.75`, matched one fewer held-out pick leniently. The ground truths' mean error over every track is lowest from `0.25` to `0.4` (0.431 px) and rises past `1` (`1.5`: 0.440). Against the second review `0.25` and `0.3` are best (72 of 74 exact), `0.2` and `0.35` match 71 and 69, and `0` 66. |
| `capped_radius_factor` | `1.5` | A view whose radius reads at the cap counts as 1.5 times as blurry as the cap, since the reading is a lower bound ([§ "The step on a radius at the cap"](#the-step-on-a-radius-at-the-cap)). Measured: from `1.08` to `2.04` it changes the same two pool picks, to an uncapped view at 2.90 or 2.96 grid px, and no hand pick; from `2.05` the lenient count falls (`2.5`: 66 of 77). `1.5` is near the middle of that range on a log scale. The ground truths do not favour it: the three picks it changes there align the other views worse. Against the second review, too, every factor from `1` to `2` matches the same picks (72 of 74 exact, 74 lenient). |
| `agreement_band` | `0.15` | Sharp views correlate worse with blurrier ones, so a narrow band penalises exactly the sharp views. The previous rule's margin, `0.15`, was set against hand picks. Measured: the hand picks do not tell `0.1` to `0.4` apart (exact 29, lenient 68 or 69 of 77); below `0.1` the lenient count is 72 and the exact count 26 to 31. The tuning half's best, `0`, matched 4 fewer held-out picks exactly. `0.075` matches 2 more picks exactly and 4 more leniently, better on both halves (tuning 14 / 35 against 14 / 34, held out 17 / 37 against 15 / 34), but it is not the tuning half's best, so the procedure does not test it. The ground truths favour a narrower band: the mean error over the 639 tracks is 0.426 px at `0.075`, 0.411 at `0.05` and 0.399 at `0`, against 0.431 at `0.15`. The second review decides the hand picks' side against `0.075`: it matches 69 of 74 exactly and 72 leniently, against 72 and 74 at `0.15`, and on one case it goes back to the hand pick the second review replaced ([§ "The second review"](#the-second-review)). `0.15` is kept. |
| `agreement_scale` | `0.15` | A view `0.30` below the best counts as twice as blurry. Measured: from `0.015` to `1.5` it changes no more than one hand pick. |
| `cell_deficit_band` | `0.3` | A view that agrees overall but not in one part of the tile: mostly parallax within the tile, then blur, a misaligned tile and lighting, rarely an occluder ([§ "What the cell check reads"](#what-the-cell-check-reads)). The previous rule's bar, `0.3`, was set against hand picks. Measured: `0.25` to `0.35` match 29 hand picks of 77 exactly and 68 or 69 leniently; `0.1`, the tuning half's best, matches 3 more leniently but 5 fewer exactly, 4 of them on the held-out half; from `0.4` up the lenient count falls (`0.4`: 66). |
| `cell_deficit_scale` | `0.3` | A cell deficit of `0.6` counts as twice as blurry. Measured: `0.03` to `0.9` give the same exact and lenient counts over the 77 hand picks; from `1.5` up the lenient count falls (`3`: 65). Both cell parameters were swept again after the cell check stopped judging cells with no texture or off the photograph, and the procedure proposes no change ([§ "What the cell check reads"](#what-the-cell-check-reads)). |
| `coverage_band` | `0.99` | A partial tile holds only part of the patch; `0.99` lets a sample or two at a corner fall off the photograph. Measured: no hand pick turns on it, from `0.95` to `1`; 15 of the 679 pool tracks have an eligible view covering between `0.9` and `0.99`. |
| `coverage_scale` | `0.03` | A view covering `0.96` of its tile counts as twice as blurry, and one at the floor of `0.9` four times. Measured: no hand pick turns on it, from `0.003` to `0.3`. |
| `clipped_band` | `0.05` | A clipped region holds neither texture nor its true colour. Measured: below `0.05` the hand picks lose (`0`: 25 exact and 66 lenient of 77, against 29 and 68); above it they hardly change. 95 of the 679 pool tracks have an eligible view clipped past `0.05`. |
| `clipped_scale` | `0.05` | A view with a tenth of its pixels clipped counts as twice as blurry. Measured: below `0.03` one exact match is lost; from `0.03` to `0.5` the hand picks change by at most one. |
| `min_coverage` | `0.9` | The hard limit ([§ "Hard limits"](#hard-limits)). Set by decision. |
| `unread_radius` | `1` grid px | The radius every view takes where no view has one. The per-dataset medians of the stored bitmap's radius run from 0.40 to 1.06 grid px ([§ "Cost of the stored bitmap"](#cost-of-the-stored-bitmap)). Not measured: no eligible view of the 679 pool tracks lacks a radius, so no track reads it. |

Two constants are part of how the cell deficit is read rather than of the
rule:

| Constant | Value | Reason |
|---|---|---|
| `REFERENCE_MIN_JUDGED_CELL_ZNCC` | `0.5` | A cell whose views do not agree, because it holds no texture they share, says nothing about any one view. |
| `REFERENCE_MIN_CELL_SAMPLES` | `16` | Below it a cell's ZNCC is read off a handful of samples. |

The test for a cell with no texture adds no constant: it reads the cap of the
cells' self-similarity reading (`tile_max_radius`, the default
`SelfSimilarityParams::max_radius`), the same cap the step on a radius reads.

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
passes every one of the previous rule's tests is ranked by its radius and
angle factor alone; it grows past the band without a step, so a view slightly past
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
  gives the Rust rule's pick and costs on all 1,318 tracks measured. The cases were split in half by dataset with a
  seeded split, 39 for tuning and 38 held out. A pick matches **exactly**
  when it is the hand-picked view, and **leniently** when it is at most 0.1
  grid px blurrier and at most 0.05 less typical (pair ZNCC) than the hand
  pick, so a view as sharp and as typical as it; chance is about one in nine
  for the first. The lenient count reads no angle, so it is also counted with
  the two views' viewing angles within 10° of each other. Every count in this
  section but those of [§ "The second review"](#the-second-review) is
  against this first review (`analyze.py --first-review`).
- **The sweep.** Each parameter was swept alone from the starting values:
  `angle_weight` from `0` to `2`, each band around the previous rule's threshold and
  each scale from a tenth of its band to several times it. The candidate for
  each parameter is the middle of the run of values tied best on the tuning
  half, by lenient count and then exact count, and it is kept only where the
  held-out half scores at least as well on both counts and better on one.
  Four parameters had a candidate other than their starting value on the
  tuning half (`angle_weight` `0.75`, `agreement_band` `0`,
  `cell_deficit_band` `0.1`, `cell_deficit_scale` `0.03`), each one to three
  picks better there, and the held-out half confirmed none of them; for the
  other six the starting value is in the tuning half's best run.
- **Before the step.** Every table and count below, and those of § "Each
  parameter and its reason" but `capped_radius_factor`'s, was made with no
  step on a radius at the cap (`capped_radius_factor = 1`, `analyze.py
  --capped-radius-factor 1`). The step moves them by a few tracks. At the
  defaults: the hand picks not at all (29 / 68 / 62); the pool picks changed
  from `angle_weight = 0` from 68 to 66, from the quadrature from 63 to 60
  and from the previous rule from 139 to 137; the ground-truth picks changed
  from `angle_weight = 0` from 84 to 81 (0.368 px against 0.482), from the
  quadrature from 93 to 90, on which the angle factor no longer aligns
  better than the quadrature (0.276 px against 0.275, better on 45, worse on
  45), and from the previous rule from 117 to 116 (0.440 px against 0.385);
  the quadrature's picks above 65° from 76 to 77; and the mean error over
  every ground-truth track from 0.431 to 0.432 px. In the sweeps,
  `agreement_band` `0` and `0.025` match one more hand pick leniently (73),
  and `cell_deficit_scale` keeps the same exact and lenient counts only from
  `0.03` to `0.45`: `0.6` matches 67 leniently, `0.9` and `1.5` 66, and `3`
  64. Its own sweep is in [§ "The step on a radius at the
  cap"](#the-step-on-a-radius-at-the-cap).
- **The forms of the angle factor**: four forms, each swept with the other
  parameters at their defaults
  ([§ "The form of the angle factor"](#the-form-of-the-angle-factor)).
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
| **The defaults** | **14 / 34 / 32** | **15 / 34 / 30** | **29 / 68 / 62** |
| `angle_weight = 0` | 13 / 34 / 30 | 14 / 34 / 30 | 27 / 68 / 60 |
| The angle in quadrature, `σ_n = 3°` | 14 / 34 / 31 | 14 / 35 / 32 | 28 / 69 / 63 |
| The previous rule of gates, on the same readings | 12 / 35 / 30 | 15 / 35 / 31 | 27 / 70 / 61 |

The cost matches the hand picks as well as the rule of gates it replaced: two
more exactly, two fewer leniently, one more within the angle bound, on the
same readings. Over the pool it picks differently from the previous rule on
139 of 679 tracks, the previous rule's pick losing on the angle factor in 65
of them, on sharpness in 41 and on clipping in 22. On the ground truths those
changes come to a mean error of 0.435 px against the previous rule's 0.387 on
117 tracks, better on 58 and worse on 59, a median difference of `+0.002`
px: the mean is carried by one kerry_park track, 4.4 px worse, without which
the mean difference is `+0.011` px. The angle factor holds most oblique picks
to what the 65° gate allowed: 83 picks above 65° on 1,318 tracks, against the
previous rule's 75, the quadrature's 76 and 93 with `angle_weight = 0`.

#### The form of the angle factor

Four forms of the angle's part of the cost were compared on the same
readings, each swept with the other parameters at their defaults: the angle
in quadrature the rule used before, `√(r² + (c · tan θ)²)` with
`c = (R/√12) · σ_n`; the rule's `r · (1 + k · tan θ)`; `r / cos^p θ`; and
`r · (1 + k · tan φ)`, where `φ` is the median angle between the view's ray,
from the patch's centre to its camera centre, and each other eligible view's
ray: the angle to the other views rather than to the normal. For a patch at
infinity every ray is minus the patch's bearing, so `φ = 0` and that factor is
`1` on every view (48 of the 679 pool tracks, 20 of the 639 ground-truth
tracks). A form's starting value is the one whose picks differ least from the
quadrature's at `σ_n = 3°` over the pool; its candidate is the tuning half's
best, kept only where the held-out half confirms it against the starting
value, as for the parameters. No candidate was confirmed, so each form is at
its starting value: `σ_n = 3°`, `k = 0.25`, `p = 0.75`, and `k = 0` for `φ`,
whose tuning half is best with no factor at all.

| Form | Value | Hand picks, tuning (39) | Held out (38) | All (77) | Ground truths' mean error, px | Ground-truth picks changed from the previous rule | From the quadrature | Pool picks changed from the previous rule | From the quadrature | Picks above 65° |
|---|---|---|---|---|---|---|---|---|---|---|
| `√(r² + (c · tan θ)²)`, `σ_n` | 2 | 13 / 34 / 30 | 14 / 35 / 32 | 27 / 69 / 62 | 0.436 | 111: 0.336 → 0.397 | 48: 0.307 → 0.314 | 114 | 32 | 79 |
| | 3 | 14 / 34 / 31 | 14 / 35 / 32 | 28 / 69 / 63 | 0.434 | 152: 0.329 → 0.371 | | 141 | | 76 |
| | 4 | 15 / 34 / 32 | 13 / 35 / 31 | 28 / 69 / 63 | 0.434 | 192: 0.319 → 0.350 | 50: 0.285 → 0.277 | 167 | 30 | 76 |
| `r · (1 + k · tan θ)`, `k` | 0.2 | 13 / 34 / 32 | 15 / 34 / 30 | 28 / 68 / 62 | 0.434 | 106: 0.389 → 0.451 | 96: 0.272 → 0.275 | 129 | 68 | 84 |
| | **0.25** | **14 / 34 / 32** | **15 / 34 / 30** | **29 / 68 / 62** | **0.431** | **117: 0.387 → 0.435** | **93: 0.278 → 0.271** | **139** | **63** | **83** |
| | 0.3 | 14 / 34 / 32 | 15 / 34 / 30 | 29 / 68 / 62 | 0.431 | 122: 0.384 → 0.428 | 89: 0.278 → 0.267 | 143 | 65 | 83 |
| | 0.75 | 15 / 35 / 34 | 15 / 33 / 30 | 30 / 68 / 64 | 0.435 | 169: 0.384 → 0.432 | 100: 0.319 → 0.337 | 183 | 90 | 81 |
| `r / cos^p θ`, `p` | 0.5 | 13 / 34 / 32 | 15 / 34 / 30 | 28 / 68 / 62 | 0.431 | 118: 0.388 → 0.438 | 83: 0.279 → 0.275 | 126 | 59 | 82 |
| | 0.75 | 13 / 34 / 32 | 15 / 34 / 30 | 28 / 68 / 62 | 0.431 | 142: 0.370 → 0.410 | 75: 0.290 → 0.281 | 136 | 56 | 81 |
| | 1 | 15 / 35 / 33 | 15 / 34 / 30 | 30 / 69 / 63 | 0.431 | 163: 0.354 → 0.391 | 71: 0.330 → 0.325 | 157 | 57 | 81 |
| `r · (1 + k · tan φ)`, `k` | 0 | 13 / 34 / 30 | 14 / 34 / 30 | 27 / 68 / 60 | 0.446 | 54: 0.472 → 0.763 | 129: 0.335 → 0.407 | 72 | 85 | 93 |
| | 0.05 | 13 / 33 / 30 | 15 / 34 / 31 | 28 / 67 / 61 | 0.452 | 105: 0.424 → 0.591 | 155: 0.385 → 0.460 | 100 | 96 | 105 |
| | 0.25 | 13 / 33 / 31 | 16 / 35 / 33 | 29 / 68 / 64 | 0.460 | 147: 0.382 → 0.547 | 175: 0.383 → 0.488 | 127 | 113 | 119 |

(Hand picks against the first review, exact / lenient / lenient within 10°.
A changed ground-truth pick reads the count of tracks and the other views'
mean error aligned to the old pick and to the new, over the views that are
neither pick. Picks above 65° are over the pool and the ground truths, 1,318
tracks.)

The hand picks do not tell the forms apart: at each form's value the exact
counts are 27 to 29 and the lenient 68 or 69 of 77, and every form's tuning-half
best (`σ_n = 4`, `k = 0.75`, `p = 1`, `k = 0` for `φ`) gains nothing on the
held-out half. The ground truths separate the angle to the other views from
the angle to the normal: any weight on `φ` aligns the other views worse than
no angle factor at all (0.446 px), while any factor on `θ` from `k = 0.1` to
`1` aligns them better, and the `φ` factor's picks are more often oblique to
the patch (105 to 119 above 65° at `k` from 0.05 to 0.25, against 81 to
84). A view in the middle of
the others' directions need not face the patch. Between `1 + k · tan θ` and
`1 / cos^p θ` the data does not choose (0.431 px for both, the hand picks
within two), and `1 + k · tan θ` is kept for the reasons of
[§ "The angle factor"](#the-angle-factor). Against the quadrature it replaced
it matches one more hand pick exactly, one fewer leniently and one fewer
within 10°; on the 93 ground-truth tracks whose pick it changes, the other
views align with a mean error of 0.271 px against 0.278 (better on 48, worse
on 45, a median difference of `−0.002` px). Of the 63 pool tracks whose pick
it changes, the quadrature's pick loses on the angle factor in 21, the more
oblique view of a blurry track such as seoul_bull point 10, and on sharpness
in 42, the blurrier view of a sharp track, where the angle in quadrature
counted most.

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

#### The second review

On 2026-10-11 the maintainer reviewed again the 48 cases where the rule at the
defaults (with the step on a radius at the cap) picked another view than the
first review's hand pick, with every view's tile shown, and chose again: the
rule's pick on 27, a tie between the hand pick and the rule's pick on 16, no
single best view on 3 (kerry_park points 202 and 346 and badland point 362),
the hand pick on 1 (altona point 1153) and a third view on 1 (kerry480 point
166). The harness's `cases.json` keeps both reviews
([README](../../../scripts/reference_view/README.md)). Against the second
review each case has a set of **accepted** views: the view chosen, both views
of a tie, and the hand pick on the 29 cases the review did not show, where
the rule already picked it. A pick matches **exactly** when it is an accepted
view and **leniently** when it is within the lenient bounds of one. The three
cases with no single best view are left out, like the two bad tracks, so 74
are scored, split by the same seeded procedure into 37 for tuning and 37
held out (other halves than those of the 77).

| Exact / lenient / lenient within 10° | Tuning (37) | Held out (37) | All (74) | Shown (45) | Not shown (29) | Picks no review accepted or was shown |
|---|---|---|---|---|---|---|
| **The defaults** | **35 / 37 / 36** | **37 / 37 / 37** | **72 / 74 / 73** | **43 / 45 / 44** | **29 / 29 / 29** | **0** |
| No step, `capped_radius_factor = 1` | 35 / 37 / 36 | 37 / 37 / 37 | 72 / 74 / 73 | 43 / 45 / 44 | 29 / 29 / 29 | 0 |
| `agreement_band = 0.075` | 34 / 36 / 34 | 35 / 36 / 36 | 69 / 72 / 70 | 41 / 43 / 41 | 28 / 29 / 29 | 3 |
| `angle_weight = 0` | 33 / 37 / 36 | 33 / 37 / 33 | 66 / 74 / 69 | 40 / 45 / 42 | 26 / 29 / 27 | 7 |
| The angle in quadrature, `σ_n = 3°` | 33 / 37 / 36 | 33 / 37 / 35 | 66 / 74 / 71 | 39 / 45 / 43 | 27 / 29 / 28 | 6 |
| The previous rule of gates | 29 / 36 / 34 | 30 / 36 / 32 | 59 / 72 / 66 | 34 / 43 / 39 | 25 / 29 / 27 | 14 |

(Shown and not shown split the 74 by whether the second review showed the
case. The last column counts, over the 74, the picks that are none of the
first hand pick, the two views the second review compared and the accepted
views.)

The two cases the defaults miss are the two scored cases where the maintainer
did not take the rule's pick. On altona point 1153 nine of the eleven views
read at the cap, so the step multiplies their costs alike and the angle
factor picks the view at 30° over the hand pick at 44°; the first review's
note says the track holds surfaces at different depths, with parallax and
changing shadows, which no reading separates. Kerry480 point 166 is the parallax
below.

**Re-tuning.** Every parameter was swept again by the procedure of [§ "How the
values were measured"](#how-the-values-were-measured), on the same readings.
Every starting value is in its tuning half's best run, so the procedure
proposes no candidate and the defaults are kept. The best runs on the tuning
half: `angle_weight` `0.25` to `0.35` (on all 74, `0.25` and `0.3` match 72
exactly and `0.35` 69); `capped_radius_factor` `1` to `2` (from `2.05`
altona point 1153 moves to an uncapped view at 63° and the lenient count
falls); `agreement_band` `0.15` to `0.4`; `agreement_scale` `0.075` to
`1.5`; `cell_deficit_band` `0.3` to `0.35`; `cell_deficit_scale` `0.3` to
`0.45`; `clipped_band` `0.03` to `0.05`; `clipped_scale` `0.02` to `0.1`;
and the two coverage parameters and `unread_radius` over their whole sweeps.
Each form of the angle factor keeps its starting value too: on all 74,
`1 + k · tan θ` matches 72 exactly at `k = 0.25`, `1 / cos^p θ` 71 at
`p = 0.75`, the quadrature 66 at `σ_n = 3°`, and the `φ` factor no more
than 68 at any weight. The ground truths do not depend on the hand picks
and are as in [§ "Each parameter and its
reason"](#each-parameter-and-its-reason).

**What the second review can show.** It showed only the cases where the rule
at the defaults picked another view than the hand pick, so the views it gave a
second look are the defaults' picks. A setting that picks a third view on a
case the defaults already got right is scored against the first review alone,
and a tie on a view other than the defaults' pick was never offered. The
defaults have no pick outside the reviewed views by construction, and most
other settings have some (the last column). The scores therefore favour the
defaults, and the agreement of 72 of 74 is not a measure of how often the
rule picks the best view on new tracks. The bias is bounded: counting every
unreviewed pick as correct, no swept value of any of the rule's parameters
reaches more than 73 exact, one more than the defaults, so the second review
cannot show any of them better than the defaults by more than one case; of the
other forms of the angle factor, only the `φ` factor at `k = 1` reaches 74
(60 exact and 14 unreviewed). It does show
losses where a setting goes back to a view the second review turned down:
`agreement_band = 0.075` returns seoul_bull point 174 to the hand pick the
maintainer replaced with the rule's pick, and `angle_weight = 0.35` does the
same on kerry_park point 64.

**Known limitations.** The maintainer's notes on four cases name what the
rule's readings do not see:

- **Exposure.** On kerry_park point 202 the rule's pick is the sharpest view
  (radius 0.43 grid px, against 0.47 for the next and 0.67 or more for the
  rest), but it is darker, and so shows less texture across the whole patch.
  The rule reads no exposure. The maintainer found no single best view.
- **Saturation.** On dino point 7423, a tie, the rule's pick is less
  saturated than the hand pick, and the note asks whether the cost should
  read colour saturation. The rule reads no colour.
- **Parallax the cell check reads weakly.** On kerry480 point 166 the
  rule's pick has a pole down the left side of the tile, and the maintainer
  chose a third view. A later look found the top third of the tile lies
  behind the rest, so the views differ there by parallax rather than an
  occluder ([§ "What the cell check reads"](#what-the-cell-check-reads)). The
  pick's cell deficit, 0.309, is just past the band of `0.3`, so that
  penalty adds 3%, and its pair ZNCC, 0.155 below the best, adds 4%; its
  radius is 0.39 grid px against the chosen view's 0.51.
- **A track over two surfaces.** On badland point 362 the patch spans a
  ridge and the darker background behind it, which are different surfaces.
  Every view reads at the cap with the same cost, and the maintainer found no
  single best view. The fault is in the track, which no choice of view
  mends.

#### What the cell check reads

The cell check reads where in the tile a view disagrees with the others. It
was built to catch an occluder in one part of the tile, but on 2026-10-11 the
maintainer surveyed what the cells it flags show, and found that it reads
mostly **parallax within the tile**, the part of the surface at another depth
shifting against the rest from view to view, and then blur, a misaligned tile
and lighting; an occluder is rare. Parallax relative to the other views is
worth reading for the reference: the maintainer prefers the view that shows
least parallax against the others, since the other views are aligned to it.
The survey's record is the harness's `cell_survey.json`
([README](../../../scripts/reference_view/README.md)).

- **Occluders.** On 15 tracks, kerry480 point 166, 5 tracks with a cell
  deficit of `0.5` or more and 9 with one near the band of `0.3`, each view's
  tile was shown with its nine cells, 1,121 judged view-cells in all, and the
  maintainer marked the cells with an occluder. None of the 30 cells over the
  band was marked; the one mark was on a cell whose deficit was `0.01`. On
  kerry480 point 166 the top third of the tile lies behind the rest, so the
  views differ there by parallax, not by the pole in front.
- **Causes.** The maintainer then named the cause of each of the 46 cells
  with a deficit of `0.25` or more on those tracks:

  | Cause | Cells | Over the band |
  |---|---|---|
  | Parallax within the tile | 24 | 14 |
  | Blur | 7 | 3 |
  | The tile shifted or misaligned | 4 | 4 |
  | Little texture | 3 | 1 |
  | Exposure or colour | 3 | 3 |
  | Shadow | 2 | 2 |
  | Off the photograph | 2 | 2 |
  | An occluder | 1 | 1 |

  Of the little-texture cells, the notes on two say that the self-similarity
  radius of each cell would judge that the cell holds no texture. One
  parallax cell is marked as slight, and one is vegetation, leaves at many
  depths.

Two of those causes say nothing about the view's agreement with the others,
so the check does not judge them:

- **A cell with no texture in the view.** A cell whose self-similarity ellipse
  reads the cap along both axes matches itself shifted anywhere in the
  search, so its ZNCC with another view reads noise or shading. The test
  reads the minor axis because a straight edge reads the cap along the edge
  only, and a disagreement across an edge is parallax or a misaligned tile,
  which the check is for; it reads the cap because that adds no constant. The
  view's reading stays in the typical agreement: a view whose cell shows
  smooth shading still says how far the views agree there.
- **A cell with a sample off the photograph.** Its ZNCCs are read over part
  of the cell, near the photograph's border. The coverage penalty already
  charges the view once for the samples it lacks, so the check does not
  charge it again. Any sample off the photograph counts, which adds no
  constant; leaving out only cells with more than a quarter of their samples
  off gives the same counts. Its reading is left out of the typical
  agreement too, so a reading over part of the cell does not set the others'
  baseline.

**Measured.** With the harness on the same pool, cases and ground truths as
[§ "How the values were measured"](#how-the-values-were-measured), read on a
build with both exclusions, and the cell deficit computed again from each
row's cell readings with each exclusion and with none (`analyze.py` section
11):

| View-cells | Pool (679 tracks) | Ground truths (639) |
|---|---|---|
| All | 53,100 | 44,820 |
| Judged before (typical agreement at the floor) | 45,900 | 36,450 |
| Not judged: no texture | 1,895 (726 judged before, 73 over the band) | 1,114 (461, 39) |
| Not judged: off the photograph | 316 (293, 5) | 18 (16, 1) |
| Not judged: either | 2,173 (982, 78) | 1,132 (477, 40) |
| Cell deficits changed | 263 rows, 37 across the band | 111 rows, 17 across the band |

(The off-photograph counts leave out the 335 rows of patches at infinity,
whose tiles the harness does not render again; the cell deficits changed are
the bench's own, which read every tile.)

The exclusions change the pick on 3 of the 679 pool tracks (kerry480 point
480, omni_temple points 157 and 1811), all through the cells with no
texture: each new pick's deficit had come from a cell that holds no texture
in that view (sky over a roof, a flat wall) where most other views see an
edge, and the new picks are about
as sharp as the old (1.23 against 1.31 grid px, 1.05 against 1.10, 1.74
against 1.79). The off-photograph exclusion changes no pick. No hand-picked
case changes its pick (72 of 74 exact, 74 lenient against the second review;
29 / 68 / 62 of 77 against the first), and no ground-truth pick changes, so
the mean error over the ground truths stays 0.431 px.

Of the survey's 46 cells, the two off the photograph (badland point 135, 42
and 45 of 64 samples off) and one of the three of little texture (dino point
7676, both axes at the cap) are no longer judged. The other two
little-texture cells read texture along one direction at least (kerry480
point 166, `[3, 1.37]` grid px; seoul_bull point 182, `[0.43, 0.36]`) and
are judged. Every parallax, blur, misaligned, exposure, shadow and occluder
cell is still judged, and the 14 parallax cells over the band stay over it.

Two other forms were measured and not taken:

- **No texture left out of the typical agreement as well.** It changes 4
  pool picks, and moves seoul_bull point 135 from a view the second review
  accepted to one it did not show (71 of 74 exact): there four views see
  only smooth shading in the top middle cell and agree on it, and leaving
  them out drops the cell under the floor. Of the ground truths it changes 2
  picks, both worse, the other views' mean error from 0.415 px to 0.458.
- **Other tests for no texture.** A semi-minor axis of at least 2, 1.5 or 1
  grid px instead of the cap changes 3 to 5 pool picks, no hand pick, and 1
  or 2 ground-truth picks, by under 0.02 px either way, but adds a constant.
  A semi-major axis at the cap, which also leaves out every cell along an
  edge, loses a hand pick (71 of 74) and changes 3 ground-truth picks, 2 of
  them worse (0.328 px to 0.382).

The band and scale were swept again on these readings by the procedure of
[§ "The second review"](#the-second-review): the tuning half's best runs are
`cell_deficit_band` `0.3` to `0.35` and `cell_deficit_scale` `0.3` to `0.45`,
as before, so both stay. Add Image to Tracks finds the same 67 of the 85
removed observations of its test.

**Cost.** The check reads nothing new: the bench and `tile_semi_axes` already read each
cell's ellipse in the same call as the whole tile's
(`zncc_self_similarity_parts`), so the check only reads the cells' minor
axes and each tile's `R²` flags of samples on the photograph.

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
- **What a single view costs.** It keeps that view's noise, and any highlight,
  parallax or occluder the measurements missed. The bitmap also changes all at once when
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
takes 0.4 to 2.4 ms, about 9% of an evaluation. Which cells are judged reads
each cell's self-similarity ellipse, which the reading of the whole tile
already gives (`TileSemiAxes::grid`), and each tile's flags of samples on the
photograph.

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
"capped_radius", "angle", "agreement", "cells", "coverage", "clipped"}` of the
radius (`radius_read` whether it is the row's own), the step on a radius at
the cap (`1.5` or `1`), the angle factor and the four penalties, or `None` for
a row left out; `costliest_term` one of `"sharpness"`, `"capped_radius"`,
`"angle"`, `"agreement"`, `"cells"`, `"coverage"`, `"clipped"`,
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
identical tiles, flat and short cells and cells the track does not agree on;
that a view that disagrees in one cell holding texture and on the photograph
carries the deficit, while one whose cell there holds no texture (both axes of
the cell's ellipse at the cap) or has samples off the photograph is not judged
there; that a cell with no texture stays in the typical agreement and one off
the photograph leaves it; that a cell along an edge, with its ellipse at the
cap along one axis only, holds texture; and that a cell is on the photograph
only with every sample on it; the clipped share of a tile rendered through a
fisheye camera, against a count of every pixel centre inside the polygon of its
border samples projected one at a time, and of a tile cut by the photograph's
edge. On constructed readings it checks the rule: a view at 20° with a radius
of 0.62 grid px picked over one at 55° with 0.60, and the sharper one picked
with an angle weight of `0`; the readings of seoul_bull point 10, the view at
19° with a radius of 1.584 grid px picked over the one at 29° with 1.572; the
angle factor `1` facing the patch and at a negative angle, growing with
`tan θ`, and making the same trade-off on a sharp track as on a blurry one;
each penalty `1` inside its band and growing at its scale past it; the step
on a radius at the cap: a capped view losing to an uncapped one at 2.95 grid
px with the other factors equal, and to one at 25° that it beats with no
step, the step its costliest term; a capped view still picked over an
uncapped one whose cell penalty exceeds the step; a track whose every view is
capped picking and ranking as with no step; a radius under the cap or read
with no cap taking no step; a view with no radius taking the step with the
largest radius where that one is capped, and none where it is not or where no
view has a radius; a view at the coverage floor eligible and one just under it and
one with no tile left out; a track whose every view is behind the patch with
no eligible view, and one with views on both sides picked among the
front-facing ones; a view at exactly 90° left out and one at 89.9° eligible;
each kind of missing reading, including a track where no view has a radius
and one whose views with an angle are all back-facing; the tie breaks; and the
costliest term's parts summing to the log of a view's cost over the pick's,
each part the log of the view's factor over the pick's.
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
