# Removing an observation from a track

A reconstruction's track is the record of which photographs saw one 3D point and
where. A sighting that is wrong -- a keypoint on the wrong surface, a member a
matcher put in a track it does not belong to -- is a claim the rest of the
toolkit believes: it pulls the triangulation, it feeds the bundle adjustment,
and it colours every per-point metric read off the result. This spec describes
the function that takes one such sighting out and leaves a reconstruction the
rest of the toolkit cannot distinguish from one the sighting was never in: the
row goes, the point is re-solved from what remains, and what the remainder can
no longer state -- a depth, or the point itself -- it stops stating.

The function is pure. It takes the reconstruction value, the point and the
image, and returns a new value plus a report; the base behind the input's shared
pointer is never written, and the caller's value is left exactly as it was
whether the edit succeeds or is refused.

It needs no photographs. Adding an observation has to find a pixel, which takes
a photometric fit and so takes the images; removing one names a row that is
already there, and the rays the re-triangulation needs are the poses and lenses
the value carries. For the same reason it is defined in **both** observation
modes: taking a row out invents no feature index and no keypoint, so nothing
about it depends on which of the two backs an observation
([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)).

Related specs:
[`edited-reconstruction.md`](edited-reconstruction.md) (the value, `PointRecord`,
`replace_point`, `delete_point`, and why a modification is delete-and-re-add),
[`add-observation.md`](add-observation.md) (the edit this one inverts, including
the frame resize it inverts),
[`create-point.md`](create-point.md) (what a one-observation point is), and
[`batch-triangulation-api.md`](batch-triangulation-api.md) (the solve the
re-triangulation runs).

## Rust API

The function lives in
[remove_observation.rs](../../../crates/sfmtool-core/src/reconstruction/remove_observation.rs),
re-exported as
`sfmtool_core::{remove_observation, RemoveObservationReport,
RemoveObservationError}`.

```rust
pub fn remove_observation(
    edited: &EditedReconstruction,
    point: u32,
    image: u32,
) -> Result<(EditedReconstruction, RemoveObservationReport), RemoveObservationError>;

pub struct RemoveObservationReport {
    pub point: Option<u32>,
    pub replaced: u32,
    pub image: u32,
    pub observation_count: usize,
    pub deleted: bool,
    pub to_infinity: bool,
    pub retriangulated: bool,
    pub position: [f64; 3],
    pub position_shift: f64,
    pub condition_number: f64,
}

pub enum RemoveObservationError {
    NoSuchPoint(u32),
    ImageOutOfRange { image: u32, image_count: usize },
    ImageNotInTrack(u32),
    Triangulation,
    Edit(EditError),
}
```

### Why it is shaped this way

**A value in, a value out.** The edit is a function from one reconstruction
value to the next, which is what lets a caller keep both and undo by pointing at
the one it had ([`edited-reconstruction.md`](edited-reconstruction.md)). There is
no `&mut` on the input, so a refusal cannot leave a half-applied edit behind, and
the returned value shares the input's base `Arc`: the cost of the edit is the
size of the one track it touched.

**Three arguments and no options.** The two edits beside this one carry an
options struct because they run kernels with parameters. This one runs no
kernel: the only decisions it makes are forced by how many sightings are left,
so there is nothing for a caller to tune.

**`point` is an `Option`.** The removal of a track's last row deletes the point,
and a report that named an index in that case would be naming an index that is
no longer live. `None` is the outcome, `deleted` says the same thing in the
positive, and `replaced` always names the index the caller passed in.

**The report names what the remainder can still state.** `observation_count`,
`to_infinity` and `retriangulated` are three different claims about the value
that comes back -- how much of a track is left, whether it still fixes a
distance, and whether the position in it was re-solved -- and a caller that
reports the edit in one line needs all three to write a true sentence.

### Example

```rust
use sfmtool_core::{remove_observation, EditedReconstruction};

let edited = EditedReconstruction::new(base);
let (next, report) = remove_observation(&edited, 42, 7)?;

assert_eq!(report.replaced, 42);
assert!(std::sync::Arc::ptr_eq(&edited.base, &next.base));
match report.point {
    Some(index) => println!("point {index} keeps {} sightings", report.observation_count),
    None => println!("that was the last sighting, and the point is gone"),
}
```

## What the call does

### 1. The refusals

The call refuses, producing nothing and touching nothing, when the image index
is past the image table, when the edited index names no live point, and when
that image does not observe the point. The order is the cheap checks first, so a
caller greying a menu entry gets the same answers without reading a track.

The fourth refusal is the re-triangulation's, below.

### 2. The row

The observation is one entry of the record the point view hands back, and its
per-observation columns -- the feature index, the keypoint, the confidence --
are fields of that entry. Removing the entry removes them together, so no
parallel column is left holding a value for a sighting that is gone. Every other
row keeps exactly what it held: this edit moves no keypoint.

So do the point's colour, patch bitmap, stored error and constraint, and its
normal and patch frame except where the outcomes below say otherwise. The point
is the same point, known by the same appearance, with one fewer photograph
saying where it is.

### 3. Two or more left: the position is re-solved

A finite point is re-triangulated from the observations that remain, by
[`triangulate_batch`](batch-triangulation-api.md) over the world rays their
keypoints unproject to at the stored poses. This is the solve the patch spawn
runs and the one [`add-observation.md`](add-observation.md) runs, refused on the
same three signals: a non-finite position, an infinite condition number (the
depth is not observable), or a solution behind one of the cameras that observe
it. No bundle adjustment runs, and the frame, the normal and the bitmap are the
ones the point had.

Two cases skip the solve and keep the position exactly as it stands, and the
report's `retriangulated` says which value came back either way:

- A point **at infinity** with two or more sightings left. Its stored coordinate
  is a direction rather than a place, and re-solving it into a position would be
  a different edit than removing a row.
- A track carrying **no inline keypoints**, which is a `sift_files` track without
  the format's optional inline copy. A row with only a feature index behind it
  states no coordinate in the value, so there is no ray to solve with.

### 4. One left: the point becomes a bearing

One sighting fixes a direction and no distance. That is what a point at infinity
is, and it is what [`create-point.md`](create-point.md) writes for a point made
from a single click, so a track worn down to one row is stored the same way: `w`
becomes 0 and the coordinate becomes the remaining observation's unit world ray,
through that image's own camera model and pose.

The patch frame is **divided** by the placement distance the point stood at --
the distance from the camera-cloud centroid to its position, the same reference
`SfmrReconstruction::materialize_points_at_infinity` measures from -- which is
exactly the resize [`add-observation.md`](add-observation.md) applies when a
point crosses the other way, run backwards. The stored half-vectors go back to
being angular extents tangent to the direction sphere, so the patch keeps the
apparent size it had; leaving them alone would leave a patch a radian wide on a
bearing.

The bitmap is kept: it is the appearance the point is known by, and resizing a
frame does not change what the tile shows. The normal becomes zero, and the
normal confidence with it where the column exists, which is what the format
states for a `w = 0` row and what a created point carries.

A point that was already a bearing stays the one it is. It has no depth to give
up, and its stored direction is not the remaining sighting's ray to re-derive.

### 5. None left: the point goes

A track with one observation loses its only one, and nothing sees the point any
more. It is deleted through `delete_point` rather than re-added, `deleted` is
set, and `point` is `None`. A point kept at a position no photograph supports
would be a claim about the scene with nothing behind it.

### 6. What the record becomes

Except for the deletion, the record is produced through `replace_point`, so the
point is deleted from the base and re-added whole with its shorter track,
`replaces` records the index it came from, and a materialisation puts it back in
that slot. The point takes a new index and is the same point throughout, which
is what `point` and `replaced` in the report are both for.

## Implementation notes

**The rays come from the image table, not from decoded views.** Every other
per-point kernel in this crate takes a `ProjectedImage` per image, because it
samples pixels. The triangulation here needs the camera model and the pose and
nothing else, both of which the value's own
[`ImageTable`](../../../crates/sfmtool-core/src/reconstruction/data/image_table.rs)
carries, so the signature stays three arguments and the offline caller needs no
photographs on disk to run the edit.

**The placement scale is measured over every image of the table.** The centroid
is the mean of all the camera centres, not of the ones that observe the point.
That is the reference the materialisation of points at infinity uses, and the
one add-observation multiplies by, so the two conversions cancel to the same
frame the point started with rather than to one scaled by which images happen to
be in the track.

**The demotion reads the record's remaining row.** The bearing is built from the
observation left in the record after the removal, so there is no second place
where "which sighting survived" has to be remembered.

## Testing

`crates/sfmtool-core/src/reconstruction/remove_observation/tests.rs` builds four
pinhole cameras looking down world `+z` and a point whose stored keypoints are
the exact projections, so the re-triangulation's truth is known; the fixture
needs no pixels, because the edit reads none. What it pins:

- Each refusal by its own variant: a point that was deleted, an image past the
  table, and an image that does not observe the point, the last leaving the
  input value's track exactly as long as it was.
- Removing one of four sightings: the report's counts and flags, the position
  landing on the truth, the surviving rows keeping their own keypoints, and the
  frame, colour, normal and normal confidence coming through untouched.
- The returned value sharing the input's base by pointer, and the input value
  unchanged.
- The shortened track materialising back into the base index it replaced, with
  its per-observation confidence column following the rows that remain.
- Removing down to one sighting: `w = 0`, the coordinate equal to the remaining
  image's unit ray to the point, the frame's angular extent equal to the world
  extent it had divided by the depth it stood at, a zeroed normal and confidence,
  and the bitmap unchanged.
- Removing the last sighting: the point deleted, the report saying so, and a
  materialisation with no point in it.
- The round trip: a track taken down to a bearing and given a sighting back by
  [`add-observation.md`](add-observation.md) comes home to a finite position near
  the one it started at.

## Non-goals

- Moving an observation the track already holds, or re-fitting the ones that
  remain. This edit takes one row out and touches no other.
- Refitting the patch frame, the normal or the bitmap. They are what the point
  is known by, and the sighting that left them is the one being removed for
  being wrong about the geometry rather than about the appearance. The frame of a
  point crossing to infinity is rescaled, which is a change of units rather than
  a refit.
- Re-solving a point at infinity that keeps two or more sightings.
- Bundle adjustment after the re-triangulation.
- Deciding whether a sighting deserves to go. The caller names the row.
