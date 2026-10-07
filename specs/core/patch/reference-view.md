# Reference View

A point in a patch-based reconstruction carries a small square bitmap of the
surface around it, and every photograph that sees the point can be resampled
into that square: one tile per view. The views are not equally good. A far view
holds less detail, an oblique one depends most on the patch's normal being
right, a tile that crosses the photograph's border holds only part of the
patch, a blown-out highlight holds no texture, and a view with an occluder in
front of one corner disagrees with the others there. This spec describes the
measurements that say what each view's tile can contribute, and the
**reference-view rule**, which picks the one view whose tile could stand as the
point's patch bitmap: a view that covers the patch, is not clipped or grazing,
agrees with the other views everywhere in the tile and nearly as well as the
best of them overall, and among those is the sharpest. The bench measures every
track it evaluates this way and reports the pick. The stored bitmap is the
fused mean of the views; computing it from the reference view is proposed in
the draft below (see [Non-goals](#non-goals)).

The design behind the measurements, and the case for computing the bitmap from
the views that hold the most detail, is in
[../../drafts/sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md)
(Parts 4 and 5).

## Rust API

The measurements and the rule live in
[reference_view.rs](../../../crates/sfmtool-core/src/patch/reference_view.rs)
with its submodules [tile.rs](../../../crates/sfmtool-core/src/patch/reference_view/tile.rs)
and [agreement.rs](../../../crates/sfmtool-core/src/patch/reference_view/agreement.rs).
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

pub fn pair_zncc_grid(a: &ViewTile, b: &ViewTile) -> [[f64; 3]; 3];
pub struct CellAgreement {
    pub pair_zncc_grid: Vec<[[f64; 3]; 3]>, // per view, median over the others
    pub typical: [[f64; 3]; 3],             // per cell, median over the views
    pub deficit: Vec<f64>,                  // per view, the cell deficit
}
pub fn cell_agreement(tiles: &[&ViewTile]) -> CellAgreement;
pub fn cell_agreement_from_pairs(pairs: &[[[f64; 3]; 3]], k: usize) -> CellAgreement;

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

// patch::normal_refine
pub struct ViewingAngle { pub angle_deg: f64, pub tilt_direction_deg: Option<f64> }
pub fn surface_to_camera(patch: &OrientedPatch, cam_from_world: &RigidTransform) -> Option<Vector3<f64>>;
pub fn viewing_angle(patch: &OrientedPatch, cam_from_world: &RigidTransform) -> Option<ViewingAngle>;
```

**Why it is shaped this way.** The rule is a pure function over a slice of
per-view readings rather than a function of tiles and a reconstruction, so the
thresholds and the fallbacks can be tested on constructed numbers, and a caller
that holds its readings from elsewhere (a stored self-similarity radius, a
member-coherence matrix it already built) runs the same rule. Each reading is an
`Option` because the views of a real track do not all have every one: a tile
that misses the photograph has no clipped share, and a view member coherence
could not score has no pair ZNCC. The answer names, for every view, the first
test that turned it away, because that is what a person reading a table of views
needs to see beside each row, and the fallback once, because it is a fact about
the track.

`render_view_tile` returns the tile and its readings together because they
share the render's placement and warp map: the outline the clipped share reads
is the warp map's border, and the angle is read at the anchored placement's
centre. The bench and the Python binding both call it, so a tile rendered from
Python is the tile the bench measured.

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
reading and every stored patch bitmap are rendered as.

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
  views' common support with member coherence's Gaussian disk window.
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
tile can slide over itself and still match itself, short on a sharp tile.

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
   `REFERENCE_AGREEMENT_MARGIN` of the best candidate's. A view with no pair ZNCC
   fails this test, unless no candidate has one, when the test passes every
   candidate.
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

## Implementation notes

**Where the cell readings come from.** The pair ZNCC is read from member
coherence's matrix, but the pair ZNCC grid is read from the tiles the bench has
already rendered, with each pair's own support and every sample weighted
equally. Member coherence renders only the samples inside its window's disk and
common to every view, so its corner cells would hold only the part of their
square inside the disk, and the cell check was measured on whole cells. The
cost is one extra member-coherence render per view and `k(k − 1)/2` pairs of
nine cells; on the review tracks (6 to 26 views) the bench's `reference view`
phase takes 0.4 to 2.4 ms, about 9% of an evaluation.

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
tile's self-similarity with every sample as data, including the black samples
off the photograph ([editable-track.md](../bench/editable-track.md) § "The ZNCC
self-similarity radius"), so the edge of the photograph reads as texture and a
partial tile reads sharper than its content is. The rule reads that radius. A
view with up to 1% of its tile off the photograph still passes the coverage
test, so on tracks whose views all sit at the photograph's border the rule can
pick by the black edge. On the review cases this is the one difference between
the rule as built and the reference rule it was tuned with, which read those
samples as missing (one track of 77, on a distant panorama).

## Parameters

The rule's thresholds are the constants above, defined in
[reference_view.rs](../../../crates/sfmtool-core/src/patch/reference_view.rs).
`MIN_TILT_ANGLE_DEG = 0.1` is in
[obliquity.rs](../../../crates/sfmtool-core/src/patch/normal_refine/obliquity.rs).
The bench renders the tiles at the reconstruction's patch resolution with the
localizer's sampler choice (`EvaluateOptions::localize.sampler`, the sampler
rule by default), and runs member coherence with its default window
(`GaussianDisk { sigma: 0.6 }`) at the same resolution and sampler.

## Python bindings

The bench's readings cross on each observation's `"track"` dict
([editable-track.md](../bench/editable-track.md) § "Python bindings"):
`viewing_angle_deg`, `tilt_direction_deg`, `coverage`, `clipped_share`,
`pair_zncc` and `cell_deficit` as floats, `pair_zncc_grid` as a `(3, 3)` float64
array, and `reference_view` as a dict `{"is_reference", "rejected_by",
"fallback"}`, with `rejected_by` one of `"coverage"`, `"clipped"`, `"angle"`,
`"cells"`, `"agreement"`, `"sharpness"` or `None` and `fallback` one of
`"none"`, `"without_angle"`, `"without_angle_or_cells"`, `"without_any"`. Each
is absent where the row has no such reading.

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
fallback, and missing readings. [bench/tests/reference_view.rs](../../../crates/sfmtool-core/src/bench/tests/reference_view.rs)
evaluates a track of the seoul_bull ground truth and checks that the rule picks
exactly one view, with coverage of at least 0.99 and under the angle limit, and
that every view it turned away for sharpness is no sharper; that each row's
self-similarity readings are those of its tile rendered directly; and that each
row's pair ZNCC is the median of its row of `member_zncc_matrix` called
directly. The Python tests are in
[test_bench_rust_bindings.py](../../../tests/rust_bindings/bench/test_bench_rust_bindings.py)
and
[test_view_tile_rust_bindings.py](../../../tests/rust_bindings/patches/test_view_tile_rust_bindings.py).

## Non-goals

The rule reports a view. The stored patch bitmap and every template a kernel
scores against are the fused mean of the views. Computing them from the
reference view, or from a mean of a few of the best views, is proposed in
[../../drafts/sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md)
Part 5.
