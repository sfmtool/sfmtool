# A reconstruction as a value: the image table and the point set

A Structure-from-Motion result in memory is two different kinds of thing bundled
into one: a list of photographs with the camera poses and lens models that place
them in the world, and a cloud of 3D points with the record of which photograph
saw each point at which pixel. Almost every operation touches one kind and not
the other. Dropping a photograph, moving a pose, or refitting a lens rewrites the
first; deleting a point, adding an observation to a track, or adjusting a point's
position rewrites the second. This spec describes how that division is expressed
in the type every pipeline in this repository passes around, so that an operation
can name the half it works on, and so that two reconstructions that agree on the
expensive parts can share them instead of copying.

The type is `SfmrReconstruction`, the in-memory form of a `.sfmr` file. It holds
three things about the file itself (where the workspace is, the metadata, the
content hashes) and then two owned halves: an **image table** and a **point set**.
It is a plain value with no interior identity: cloning it gives an independent
reconstruction, and an operation that changes one is written as a function from a
reconstruction to a reconstruction rather than as a mutation of a shared object.
Two columns dominate its memory by an order of magnitude, the per-image
thumbnails and the per-point patch bitmaps, and those two sit behind a shared
pointer so that a clone which does not change them costs nothing for them.

## Rust API

The three types live in
[data.rs](../../../crates/sfmtool-core/src/reconstruction/data.rs) and its two
children [image_table.rs](../../../crates/sfmtool-core/src/reconstruction/data/image_table.rs)
and [point_set.rs](../../../crates/sfmtool-core/src/reconstruction/data/point_set.rs),
re-exported as `sfmtool_core::{SfmrReconstruction, ImageTable, PointSet}`. The
reconstruction reaches Python as `sfmtool._sfmtool.reconstruction.SfmrReconstruction`,
which exposes columns and never the struct layout, so the split is invisible
there.

```rust
pub struct SfmrReconstruction {
    pub workspace_dir: PathBuf,
    pub metadata: SfmrMetadata,
    pub content_hash: ContentHash,
    pub image_table: ImageTable,
    pub point_set: PointSet,
}

pub struct ImageTable {
    pub cameras: Vec<CameraIntrinsics>,
    pub images: Vec<SfmrImage>,
    pub thumbnails_y_x_rgb: Arc<Array4<u8>>,
    pub depth_statistics: DepthStatistics,
    pub depth_histogram_counts: Vec<Vec<u32>>,
    pub rig_frame_data: Option<RigFrameData>,
}

pub struct PointSet {
    pub points: Vec<Point3D>,
    pub tracks: Vec<TrackObservation>,
    pub observation_counts: Vec<u32>,
    pub observations: ObservationSource,
    pub patch_u_halfvec_xyz: Option<Array2<f32>>,
    pub patch_v_halfvec_xyz: Option<Array2<f32>>,
    pub patch_bitmaps_y_x_rgba: Option<Arc<Array4<u8>>>,
    pub has_normals: bool,
    pub normal_confidence: Option<Vec<u8>>,
    pub point_constraints: Option<PointConstraintColumns>,
    pub observation_confidence: Option<Vec<u8>>,
    // Derived from the fields above and the image count.
    pub observation_offsets: Vec<usize>,
    pub image_feature_to_point: Vec<HashMap<u32, u32>>,
    pub max_track_feature_index: Vec<u32>,
    pub infinity_point_count: usize,
}

impl PointSet {
    pub fn point_count(&self) -> usize;
    pub fn observation_count(&self) -> usize;
    pub fn observations_for_point(&self, point_idx: usize) -> &[TrackObservation];
    pub fn observation_row(&self, image_index: usize, point_index: u32, feature_index: u32)
        -> Option<usize>;
    pub fn feature_indexes(&self) -> Option<&[u32]>;
    pub fn keypoints_xy(&self) -> Option<&Array2<f32>>;
    pub fn rebuild_derived_fields(&mut self, image_count: usize);
    pub fn validate_observation_columns(&self, image_count: usize) -> Result<(), String>;
}

impl ImageTable {
    pub fn image_count(&self) -> usize;
    pub fn camera_count(&self) -> usize;
    pub fn camera_for_image(&self, image_index: usize) -> &CameraIntrinsics;
}
```

`SfmrReconstruction` keeps a forwarding method for each of the reads a caller
makes most often, so `recon.point_count()`, `recon.image_count()`,
`recon.observations_for_point(p)`, `recon.keypoints_xy()` and
`recon.rebuild_derived_fields()` mean what they have always meant and a caller
that only wants a count does not have to know which half answers it.

### Why it is shaped this way

**Two owned structs rather than two `Arc`s.** A reconstruction is a value, and
the halves are parts of that value, not separately shared objects. Putting the
whole image table behind a pointer would make two reconstructions able to observe
each other's edits, which is exactly what the value model is for avoiding. The
sharing this design does want is narrower than a half: it is the two heavy
columns, and it is expressed at those two fields.

**The split is by what an index means, not by what an algorithm wants.**
Everything in the image table is addressed by image index or reached from one
(`cameras` through `SfmrImage::camera_index`, `rig_frame_data` through the frame
grouping). Everything in the point set is addressed by point index or by
observation row. That rule decides every field without a judgement call, and it
is why a per-point algorithm can take a `&PointSet` and an image count rather
than the whole reconstruction: an addition set of points that shares an image
table is then the same type as the base's point side.

**The two heavy columns are shared, and shared is not copy-on-write.** On the
largest real reconstruction measured (530 674 points, 8.1 million observations,
500 images, 24 px patch bitmaps) the whole value is 1 354 MB, of which the
thumbnails and patch bitmaps are 1 189 MB, and a full clone costs 355 ms against
32 ms for a clone without them. `Arc<Array4<u8>>` for the thumbnails and
`Option<Arc<Array4<u8>>>` for the bitmaps let a producer that does not change
them hand the new value its input's pointer. Nothing is ever written through
either pointer: a producer that changes the bitmaps builds a new array and wraps
it (see "The sharing rule" below). Reads deref transparently, so a caller writes
`recon.image_table.thumbnails_y_x_rgb.index_axis(Axis(0), i)` exactly as before.

**The derived indexes belong to the point set even though two of them are sized
by the image count.** `observation_offsets`, `image_feature_to_point`,
`max_track_feature_index` and `infinity_point_count` are all a function of the
tracks and the points; the image count only says how long two of the vectors are.
So `PointSet::rebuild_derived_fields` takes the count as an argument, and
`SfmrReconstruction::rebuild_derived_fields` supplies it from the image table.

### Example

```rust
use sfmtool_core::SfmrReconstruction;

let recon = SfmrReconstruction::load(path)?;

// Counts and per-point reads, through the forwarding methods.
println!("{} images, {} points", recon.image_count(), recon.point_count());
for obs in recon.observations_for_point(0) {
    println!("image {} sees point 0", obs.image_index);
}

// Naming the half when an operation belongs to one of them.
let pose = &recon.image_table.images[0];
let colour = recon.point_set.points[0].color;

// A point-side edit, with the derived indexes restored afterwards.
let mut edited = recon.clone();          // shares the thumbnails and bitmaps
edited.point_set.points.truncate(100);
edited.point_set.observation_counts.truncate(100);
edited.point_set.tracks.retain(|t| t.point_index < 100);
edited.rebuild_derived_fields();
```

## Implementation notes

**The sharing rule.** A shared column is read through its pointer and never
written through it. A pass that changes the patch bitmaps takes an owned array
out (`Arc::unwrap_or_clone`, which copies only when the pointer is shared),
edits that array, and wraps a new pointer back in, once for the pass rather than
once per point. The finite/infinity conversions in
[analysis/infinity/convert.rs](../../../crates/sfmtool-core/src/analysis/infinity/convert.rs)
are where this matters: they clear or rescale a patch row per touched point, so
taking the array out around the loop is what keeps a hundred touched points from
costing a hundred array copies. A row selection (the subset and mask edits) is a
new array by construction, so it simply wraps its result. `Arc::make_mut` is
deliberately not used anywhere: it would make the columns copy-on-write, and the
value model wants a producer to be explicit about building a new array.

**The derived-index obligation.** `observation_offsets` is the prefix sum of
`observation_counts`, `image_feature_to_point` and `max_track_feature_index` are
built by walking the tracks against the feature indexes (and stay empty for an
`embedded_patches` reconstruction, which has no feature indexes), and
`infinity_point_count` counts the points with `w == 0`. Any code that edits
`tracks`, `observation_counts`, `points`, or a point's `w` in place owes a
`rebuild_derived_fields()` before the value is handed on: nothing recomputes them
lazily, and a stale `observation_offsets` indexes into the wrong run of tracks
rather than failing. The two in-place `w` mutators refresh
`infinity_point_count` themselves, because they change no track structure and so
have no other reason to rebuild.

**`ObservationSource` straddles the split, and stays whole in the point set.**
The enum carries both the per-observation pixel column (`feature_indexes` and
`keypoints_xy`) and the per-image hash vectors (`feature_tool_hashes`,
`sift_content_hashes`, `image_file_hashes`), because a `.sfmr` file is wholly one
mode and one discriminator selects both sets of columns at once. Splitting the
per-image half into the image table would put a second discriminator there that
has to agree with this one, and a pair of enums that can disagree is a worse
invariant than one field that is measured per image while living on the point
side. So the enum stays whole in the point set, and it is the single exception to
the index rule above. The consequence is visible in
`PointSet::validate_observation_columns`, which takes the image count as an
argument in order to check the hash vectors' lengths.

**The Python side sees the sharing rule as read-only views.** Every column
getter on the `SfmrReconstruction` binding hands Python its own copy, so a
write to the returned array never reaches the value. The one zero-copy getter,
`thumbnails_y_x_rgb`, is a view of the shared buffer, and it is returned with
numpy's writeable flag cleared: a write raises, and a caller that wants to edit
takes `.copy()`. Without that flag a Python write would land in every
reconstruction sharing the buffer at once, which is the one thing the rule
forbids.

**Validation is split the same way.** `validate_observation_columns` checks the
per-observation columns against the track count and the per-image hashes against
the image count; `validate_point_columns` checks the constraint triple against
the point count and its reference images against the image count. Both are
reachable from `SfmrReconstruction`, which supplies the count the point set
cannot see, and both are what the `.sfmr` conversion and the kwargs-driven
Python editor run before handing back a value.

## Testing

`crates/sfmtool-core/src/reconstruction/data/tests.rs` covers the round trip
through `SfmrData` for both observation sources with every optional column
present, which is what pins that each column lands in the right half and comes
back unchanged; `crates/sfmtool-core/src/reconstruction/edit/tests.rs` covers the
three whole-value edits (transform, image subset, point mask) including the patch
frame and bitmap columns; and
`crates/sfmtool-core/src/analysis/infinity/convert/tests.rs` covers the passes
that edit the bitmaps in place, which is the sharing rule's only exercise inside
the crate. The Python side is exercised across `tests/rust_bindings/`, where the
binding surface is the contract: no Python-visible name, dtype or shape depends
on which half a column lives in.

## Non-goals

- An **edited reconstruction**, a shared immutable base plus the point edits made
  on top of it, is not built here. This spec describes the plain value the base
  would be; the base-plus-edits representation, the delete-and-re-add rule for
  point edits, and materialisation are proposed in
  [`../../drafts/sfm-explorer-editing-overlay.md`](../../drafts/sfm-explorer-editing-overlay.md).
- Sharing between reconstructions beyond the two heavy columns. The track
  columns are most of the remaining bytes and are untouched by every bulk edit
  but an image deletion, so they could be shared too; they are not, until there
  is a workload measuring the difference.
- Interior mutability or observation of one reconstruction's edits by another.
  A reconstruction is a value; an operation that changes one returns a new one.
