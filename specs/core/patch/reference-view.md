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
point's patch bitmap: a view that covers the patch, is not clipped or grazing,
agrees with the other views everywhere in the tile and nearly as well as the
best of them overall, and among those is the sharpest. The bench measures every
track it evaluates this way and reports the pick, and every operation that
renders a point's bitmap stores the picked view's tile as it is and records
which observation it is ([§ "The stored bitmap"](#the-stored-bitmap)).

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
[agreement.rs](../../../crates/sfmtool-core/src/patch/reference_view/agreement.rs)
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
pub enum ReferenceTest { Coverage, Clipped, Angle, Cells, Agreement, Sharpness }
pub enum ReferenceFallback { None, WithoutAngle, WithoutAngleOrCells, WithoutAny }
impl ReferenceFallback {
    pub fn applies(self, test: ReferenceTest) -> bool; // the angle test always applies
}
pub struct ReferenceChoice {
    pub reference: Option<usize>,
    pub fallback: ReferenceFallback,
    pub rejected_by: Vec<Option<ReferenceTest>>,
}
pub struct ReferenceStanding { pub rejected_by: Option<ReferenceTest>, pub fallback: ReferenceFallback }
pub fn choose_reference_view(views: &[ReferenceReadings]) -> ReferenceChoice;

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
    sampler: SamplerChoice, progress: &Progress<'_>) -> TrackReading;

// patch::stored_bitmap
pub struct PatchBitmap { pub rgba: Vec<u8>, pub reference: Option<usize> }
pub fn bitmap_from_tile(tile: &ViewTile) -> Vec<u8>; // alpha 255 on the photograph
pub struct ReferenceRender { pub tiles: Vec<ViewTile>, pub semi_axes: Vec<Option<[f64; 2]>>,
    pub reading: TrackReading }
impl ReferenceRender {
    pub fn stored_reference(&self) -> Option<usize>; // the pick, unless reached by WithoutAny
}
pub fn render_reference(patch: &OrientedPatch, views: &[ProjectedImage<'_>], view_set: &[u32],
    keypoints: &[Option<[f64; 2]>], resolution: u32, sampler: SamplerChoice,
    progress: &Progress<'_>) -> ReferenceRender;
pub fn render_patch_bitmap(patch: &OrientedPatch, views: &[ProjectedImage<'_>],
    view_set: &[u32], keypoints: &[[f64; 2]], params: &KeypointSubpixelParams,
    progress: &Progress<'_>) -> Option<PatchBitmap>;
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
thresholds and the fallbacks can be tested on constructed numbers, and a caller
that holds its readings from elsewhere (the slots of a bench row a verdict
moved under) runs the same rule. Each reading is an `Option` because the views
of a real track do not all have every one: a tile that misses the photograph
has no clipped share, and a view member coherence could not score has no pair
ZNCC. The answer names, for every view, the first test that turned it away,
because that is what a person reading a table of views needs to see beside
each row, and the fallback once, because it is a fact about the track.

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
use sfmtool_core::patch::reference_view::{choose_reference_view, ReferenceReadings};

let readings: Vec<ReferenceReadings> = /* one per view */;
let choice = choose_reference_view(&readings);
if let Some(i) = choice.reference {
    println!("view {i} is the reference ({:?})", choice.fallback);
}
for (i, rejected) in choice.rejected_by.iter().enumerate() {
    if let Some(test) = rejected {
        println!("view {i}: turned away by {test}");
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

1. **Candidates** are the views that pass every one of:
   - **coverage** at least `REFERENCE_MIN_COVERAGE`;
   - **clipped** share at most `REFERENCE_MAX_CLIPPED_SHARE`;
   - **angle** at most `REFERENCE_MAX_VIEWING_ANGLE_DEG`;
   - **cells**: cell deficit at most `REFERENCE_MAX_CELL_DEFICIT`.

   When no view passes, the angle test's `REFERENCE_MAX_VIEWING_ANGLE_DEG`
   limit is dropped; when still none passes, the cell check too; and then
   coverage and clipping, so every view that faces the patch is a candidate. A
   test dropped for the track is dropped for every view of it. The angle test
   keeps a limit of `REFERENCE_FACING_LIMIT_DEG` (`90°`) under every fallback:
   a view at `90°` sees the patch edge on and one past it sees the patch's back,
   so its tile is not a picture of the patch's face, and it is turned away by
   `angle` whichever tests are dropped. Where every view is at `90°` or more,
   the rule picks nothing.
2. **Agreement**: of the candidates, those whose pair ZNCC is within
   `REFERENCE_AGREEMENT_MARGIN` of the best candidate's. A view with no pair
   ZNCC fails this test, unless no candidate has one, when the test passes
   every candidate.
3. **Sharpness**: of those, the one with the smallest self-similarity
   semi-major axis; a tie goes to the smaller semi-minor axis, then to the
   earlier view.

A missing coverage or viewing angle fails its test, since a candidate has to
show that it meets it; a missing clipped share or cell deficit passes, since
nothing is there to count against the view. Once the 65° limit is dropped a
missing viewing angle passes, since only an angle shown to be at or past the
facing limit turns the view away. When no view left after the
agreement test has a self-similarity reading, the rule picks nothing.

### Each threshold and its reason

| Constant | Value | Reason |
|---|---|---|
| `REFERENCE_MIN_COVERAGE` | `0.99` | A partial tile holds only part of the patch, so a bitmap taken from it would be missing the rest. Not tuned. |
| `REFERENCE_MAX_CLIPPED_SHARE` | `0.05` | A clipped region holds neither texture nor its true colour. Not tuned. |
| `REFERENCE_MAX_VIEWING_ANGLE_DEG` | `65°` | An oblique tile depends most on the patch model. It changed no pick on the review cases and 0.9% of picks over the whole datasets; on the two ground truths, removing it made the picks it changes localize the other views 0.12 px worse on average (15 tracks). Kept for that reason rather than fitted. |
| `REFERENCE_FACING_LIMIT_DEG` | `90°` | A view at `90°` sees the patch edge on and one past it sees its back, so its tile is not a picture of the patch's face. Kept under every fallback. A geometric limit, not tuned. |
| `REFERENCE_MAX_CELL_DEFICIT` | `0.3` | Catches a view that agrees overall but not in one part of the tile: an occluder, a shadow edge, parallax within the tile. Values from `0.2` to `0.4` gave the same agreement with the hand picks on the tuning half; `0.3` is the middle. |
| `REFERENCE_MIN_JUDGED_CELL_ZNCC` | `0.5` | A cell whose views do not agree, because it holds no texture they share, says nothing about any one view. |
| `REFERENCE_MIN_CELL_SAMPLES` | `16` | Below it a cell's ZNCC is read off a handful of samples. |
| `REFERENCE_AGREEMENT_MARGIN` | `0.15` | Sharp views correlate worse with blurrier ones, since the detail they carry is missing from the others, so a narrow margin turns away exactly the sharp views. Where the earlier margin of `0.05` missed, the hand-picked view sat `0.06` to `0.21` below the best. `0.10` to `0.20` were within one case of each other on the tuning half. |

The values were set against hand picks of the reference view on 77 tracks from
ten datasets, split in half by dataset with the thresholds chosen on one half:
on the held-out half the rule picks the hand-picked view on 10 of 38 tracks and
a view as sharp and as typical as it, within 0.1 px of semi-major axis and 0.05
of pair ZNCC, on 35 of 38 (chance is about one in nine for the first). Two
signals were tried and left out: a brightness and colour gate, which turned away
views the hand picks chose, and a contrast floor, which the agreement margin
already covers.

### Why the agreement is read plain

Both tests could read blur-matched agreement instead
([blur-matched-zncc.md](blur-matched-zncc.md)): before two views' tiles are
correlated, a tile sharper than the other along every direction is blurred by a
round Gaussian to the other's sharpness along its sharpest direction, so a sharp
view is counted as disagreeing less for detail a blurrier view lacks. That was
measured against the same hand picks, with the thresholds tuned the same way
(seeded split by dataset, best lenient then exact, the middle of a tied run),
from the bench's own readings:

| Readings | Margin, cell bar | Tune exact / lenient (39) | Held-out (38) | All (77) |
|---|---|---|---|---|
| **Plain (the rule)** | **0.15, 0.3** | **18 / 35** | **10 / 35** | **28 / 70** |
| Blur-matched agreement only | 0.15, 0.3 | 18 / 35 | 10 / 35 | 28 / 70 |
| Blur-matched cells only | 0.15, 0.3 | 18 / 35 | 10 / 35 | 28 / 70 |
| Both blur-matched, every difference | 0.15, 0.3 | 18 / 36 | 10 / 35 | 28 / 71 |
| Both blur-matched, ratio 1.25 | 0.15, 0.3 | 18 / 36 | 10 / 35 | 28 / 71 |
| Both blur-matched, ratio 1.25 | 0.08, 0.3 | 18 / 35 | 11 / 36 | 29 / 71 |
| Both blur-matched, ratio 1.25 | 0.12, 0.35 | 19 / 36 | 10 / 34 | 29 / 70 |

Blur matching picked the hand pick exactly on as many tracks as the plain
readings, within the lenient bounds on one more, changed the pick on 4 of 661
pool tracks (0.6%), and added 0.13 ms (2%) to a track's evaluation. The
agreement test is a gate on whether a candidate agrees with the track rather
than a ranking of the candidates, and the rule ranks by the radius, so taking
away a sharp view's penalty for detail the others lack rarely changes which
view passes. The rule reads plain readings, and blur matching is read where it
changes the result: in each observation's score against the stored bitmap
([blur-matched-zncc.md](blur-matched-zncc.md) § "Scores against the stored
bitmap").

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
- **Where the rule picks no view.** That happens only where no candidate has a
  self-similarity reading, or every view sees the patch edge on or from behind.
  The bitmap is then the fused mean of the views
  ([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md) § "The
  fused mean"), and the point names no reference observation (`-1`). A point
  with fewer than two views gets no bitmap.
- **Where the rule's pick is not stored.** A pick the rule reaches only by its
  last fallback, `without_any`, is one no view passed the coverage and clipping
  tests for: every candidate's tile has too little of the patch on its
  photograph, has no coverage reading, or has too large a share of clipped
  (blown-out) pixels. Such a tile can hold few samples or little texture, so
  the stored bitmap is the fused mean instead,
  which fills every sample some view covers, and the point names no reference
  (`ReferenceRender::stored_reference`). Where no fused mean renders either,
  because no view has the patch in frame, the pick is stored after all, since
  a part of the patch is better than none. On a solve of `dino_dog_toy`, 559
  of 18,991 points (3%) take the fused mean by this rule. The evaluation's
  pick, which Track View and the wire show, is the rule's and is not changed
  by it, so for these points the row Track View marks as the reference is not
  the stored bitmap's tile.

**The reference says what to render.** `tracks/reference_observations`
names, per point, the observation its bitmap is, or is to be, rendered from,
with or without stored bitmaps. Where a reconstruction stores a point's bitmap
beside a reference `≥ 0`, the bitmap is that observation's tile as of the last
render. `-1` means the point has no reference observation in its track, and
a stored bitmap beside it is not the render of one of its observations: a
fused mean, or the render of an observation that an edit has since removed
from the point (an image subset or filter keeps the bitmap and writes `-1`).
The next render picks a new reference by the rule. Dropping the bitmaps keeps
the column, and moving geometry
(keypoints, the frame, a bundle adjustment) keeps it too
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
`PatchCloud.render_bitmaps(..., referenced_only=True)`). The refiners render
every point by the rule over the views they keep; the `xform` steps and `sfm
embed-patches` that write their bitmaps keep each point's stored reference and
give only a point at `-1` the refiner's pick, then render every point with a
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
at its keypoint, or the rule's pick where the point stores none, and the fused
mean where the rule picks none it would store
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
inside the disk, and the cell check was measured on whole cells. The cost is
one extra member-coherence render per view and `k(k − 1)/2` pairs of nine
cells; on the review tracks (6 to 26 views) the bench's `reference view` phase
takes 0.4 to 2.4 ms, about 9% of an evaluation.

**What the stored bitmap costs.** Rendering a point's bitmap renders each
view's tile once, reads its self-similarity, builds member coherence's matrix
and correlates the cells, where the fused mean rendered each view twice (once
on the support for the weights and once whole) and blended. The timings over
real reconstructions are in [§ "Cost of the stored bitmap"](#cost-of-the-stored-bitmap).

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
still passes the coverage test.

## Cost of the stored bitmap

Measured on the 661 pool tracks of the ten datasets (the review cases and 60
of each dataset's pool), each put on the bench from its stored point and
evaluated, one thread, release build, best of three:

| | Median | p90 | Total |
|---|---|---|---|
| Fused mean (`fuse_patch_bitmap`), per track | 1.30 ms | 3.52 ms | 1.14 s |
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

**What it stores.** On the same 661 tracks the evaluation's pick, the view whose
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

The rule's thresholds are the constants above, defined in
[reference_view.rs](../../../crates/sfmtool-core/src/patch/reference_view.rs).
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
float64 array, and `reference_view` as a dict `{"is_reference",
"rejected_by", "fallback"}`, with `rejected_by` one of `"coverage"`,
`"clipped"`, `"angle"`, `"cells"`, `"agreement"`, `"sharpness"` or `None`, and
`fallback` one of `"none"`, `"without_angle"`, `"without_angle_or_cells"`,
`"without_any"`. Each is absent where the row has no such reading.
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
edge; and each of the rule's tests, its inclusive thresholds, the margin's
reference point, the tie breaks, every fallback, the facing limit under every
fallback, and missing readings.
[stored_bitmap/tests.rs](../../../crates/sfmtool-core/src/patch/stored_bitmap/tests.rs)
checks, on three synthetic views of a painted plane, that the stored bitmap is
the tile of the view the rule picks (the sharp one), with alpha marking the
samples on the photograph; that where every view sees the patch from behind
the bitmap is the fused mean and names no reference; and that a pick reached
only by the `without_any` fallback is not what `stored_reference` returns,
with the pick stored after all where no fused mean renders.
[display_bitmaps/tests.rs](../../../crates/sfmtool-core/src/patch/display_bitmaps/tests.rs)
checks, on the seoul_bull ground truth, that the whole-cloud render names a
reference within each point's own track, also when some images have no view
(so the index is the observation's, not its place among the views to hand).
[bench/tests/reference_view.rs](../../../crates/sfmtool-core/src/bench/tests/reference_view.rs)
evaluates a track of the seoul_bull ground truth and checks that the rule picks
exactly one view, with coverage of at least 0.99 and under the angle limit, and
that every view it turned away for sharpness is no sharper; that each row's
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
