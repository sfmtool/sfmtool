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
expensive parts can share them instead of copying. It then describes the
**edited** form of that value, a shared immutable base plus the handful of point
edits made on top of it, which is what makes an edit that changes one track cost
the size of that track rather than the size of the reconstruction.

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

## The overlay: a base plus its edits

A reconstruction the viewer targets is up to a million points and ten million
track observations, stored as sorted columns with a prefix-sum index. The tracks
are sorted by point then image, so adding one observation to one track is an
insertion in the middle of a ten-million-row column, and a value that is a plain
clone pays the whole column for it. An `EditedReconstruction` is that value in
two parts instead: an immutable base, and the small set of edits made on it.

- **The base** is a shared `SfmrReconstruction` behind an `Arc`. It is what was
  loaded, or the last materialisation. Every version in a run of edits shares
  it, and nothing is ever written through it.
- **The deleted set** is which indexes are gone, as a hash set. A set rather
  than a mask over the base: the overlay exists for individual point edits, so
  it holds a handful of indexes against a million-point base and its size is the
  size of the edit.
- **The additions** are the points this version holds and the base does not, as
  a `PointSet` of their own with exactly the base's columns, whose observations
  index the base's image table.

The images and the cameras are the base's. An edit to the image table is not an
overlay edit.

### Two kinds of edit

A **point edit** lives in the overlay and is always delete-and-re-add. A point
that changes in any way, its position, its constraint, its normal or frame, its
track, is deleted from the base and re-added to the additions with its whole
record and whole track. Adding an observation to a track, removing one,
splitting a track, merging two, refitting a point, setting a constraint: each is
tens of rows. Deleting a point is one index in the set. There is one code path
for "a point that changed" whatever changed about it, and no partial state in
which a track and a per-observation column disagree.

A **bulk edit** is a function from a plain reconstruction to a plain
reconstruction: it materialises the current version, runs, and its output is the
next version's base with an empty overlay. Deleting an image, moving a pose,
bundle adjustment and baking a transform are bulk, because each touches an
image's worth of structure or more. The three whole-value edits above
(`apply_se3_transform`, `subset_by_image_indices`, `filter_points_by_mask`) are
bulk edits in this sense. The two compose: a run of point edits on a base, then
a bulk edit that materialises them into the next base, then more point edits on
that.

### Indexes are stable

A base point keeps its index while the base lives; a deleted index is a hole
that resolves to nothing and is never reused; an added point takes the next
index at or after the base's point count. So a point edit shifts no index, and
a selection, a panel, a `pt3d_<hash>_<index>` id and a GPU instance buffer all
survive an edit unchanged. Materialisation is the one operation that renumbers,
and it produces the map that says how.

A point re-added after a modification takes a new index, and the edit records
the base index it replaces against it, which is what puts it back in its place
at materialisation.

### Rust API

The type lives in
[edited.rs](../../../crates/sfmtool-core/src/reconstruction/edited.rs),
re-exported as
`sfmtool_core::{EditedReconstruction, PointRecord, RecordObservation, PointView,
RowMap, EditError}`, and reaches Python as
`sfmtool._sfmtool.reconstruction.EditedReconstruction`.

```rust
pub struct EditedReconstruction {
    pub base: Arc<SfmrReconstruction>,
    pub deleted_points: HashSet<u32>,
    pub added: PointSet,
    /// Per added point: the base index it replaces, or `None` if it is new.
    pub replaces: Vec<Option<u32>>,
}

impl EditedReconstruction {
    pub fn new(base: Arc<SfmrReconstruction>) -> Self;

    // Counts, all O(1).
    pub fn point_count(&self) -> usize;
    pub fn image_count(&self) -> usize;
    pub fn base_point_count(&self) -> usize;
    pub fn index_bound(&self) -> u32;
    pub fn is_deleted(&self, index: u32) -> bool;
    pub fn live_indexes(&self) -> impl Iterator<Item = u32> + '_;

    // Column presence, answered from the base.
    pub fn feature_source(&self) -> &str;
    pub fn has_feature_indexes(&self) -> bool;
    pub fn has_keypoints(&self) -> bool;
    pub fn has_observation_confidence(&self) -> bool;
    pub fn has_patch_frames(&self) -> bool;
    pub fn has_patch_bitmaps(&self) -> bool;
    pub fn has_normal_confidence(&self) -> bool;
    pub fn has_point_constraints(&self) -> bool;

    // Reading one point without materialising.
    pub fn point(&self, index: u32) -> Option<PointView<'_>>;

    // The point edits.
    pub fn delete_point(&mut self, index: u32) -> Result<(), EditError>;
    pub fn replace_point(&mut self, index: u32, record: PointRecord)
        -> Result<u32, EditError>;
    pub fn add_point(&mut self, record: PointRecord) -> Result<u32, EditError>;

    // Hashes and the commit.
    pub fn base_content_hash(&self) -> Result<&ContentHash, SfmrError>;
    pub fn point_edit_hash(&self, records: &[PointRecord]) -> Result<String, SfmrError>;
    pub fn materialize(&self) -> (SfmrReconstruction, RowMap);
}

pub struct PointRecord {
    pub point: Point3D,
    pub observations: Vec<RecordObservation>,
    pub patch_u_halfvec: Option<[f32; 3]>,
    pub patch_v_halfvec: Option<[f32; 3]>,
    pub patch_bitmap: Option<Array3<u8>>,
    pub normal_confidence: Option<u8>,
    pub constraint: Option<(u8, f64, u32)>,
}

pub struct RecordObservation {
    pub image_index: u32,
    pub feature_index: Option<u32>,
    pub keypoint_xy: Option<[f32; 2]>,
    pub confidence: Option<u8>,
}

impl RowMap {
    pub fn forward(&self, edited: u32) -> Option<u32>;
    pub fn inverse(&self, new: u32) -> Option<u32>;
    pub fn forward_dense(&self, index_bound: u32) -> Vec<Option<u32>>;
    pub fn inverse_dense(&self, point_count: u32) -> Vec<u32>;
}
```

**Why it is shaped this way.** The base's point side and the additions are the
same `PointSet` type, which is what the split above buys: a per-point algorithm
takes a point set and an image count, so the addition set needs no schema of its
own to drift from the base's. `PointRecord` is the unit an edit trades in
because delete-and-re-add wants a whole point, not a diff, and it names no point
index: an observation's point is the record it belongs to, and the index that
identifies that point differs between the overlay and the materialisation, so
carrying one would be a second answer to a question the record already answers.
A record must carry **exactly** the base's optional columns, no more and no
fewer, which is what lets column presence be answered from the base alone and
never from a walk; `add_point` refuses otherwise, with an `EditError` naming the
column. Every edit validates before it pushes, so a refused record leaves the
addition set exactly as it was.

There is no copy-on-write and no interior mutability. `&mut self` on an edit is
a write to the *overlay*, which the version owns; the base behind the `Arc` is
never written, and a caller can assert that with `Arc::ptr_eq` across any run of
point edits.

### Example

```rust
use std::sync::Arc;
use sfmtool_core::{EditedReconstruction, SfmrReconstruction};

let base = Arc::new(SfmrReconstruction::load(path)?);
let mut edited = EditedReconstruction::new(Arc::clone(&base));

// Add an observation to point 42's track: read the whole record, extend it,
// and re-add. The base is untouched, and 42 stops resolving.
let mut record = edited.point(42).unwrap().to_record();
record.observations.push(sfmtool_core::RecordObservation {
    image_index: 7,
    feature_index: None,
    keypoint_xy: Some([120.5, 88.25]),
    confidence: Some(200),
});
let moved = edited.replace_point(42, record)?;

edited.delete_point(9)?;

// The plain value every algorithm consumes, plus where each index went.
let (plain, row_map) = edited.materialize();
assert_eq!(row_map.forward(moved), Some(42)); // back in its place
assert_eq!(row_map.forward(9), None);         // gone
```

## Materialisation

Materialising produces a plain `SfmrReconstruction` in which **every point keeps
its place**: a modified point goes back to the base index it replaced, carrying
its new record and track; a deleted point's slot closes up, shifting the points
after it down by one; and only a point that is new to this base is appended,
after the last base point, in the order the additions were made. The derived
indexes are rebuilt at the end. The image table is the base's, thumbnails shared
rather than copied.

Keeping places is what makes the materialised file readable beside the one it
came from: a point that was edited is at the same row in both, a point that was
not is at the same row unless something before it was deleted, and a diff of the
two files is the edit. It is also what makes the **row map** cheap. For base
points the map is the identity less a prefix count of the slots that emptied, so
it is stored as the sorted list of those holes and inverted by the same count;
only the additions, a handful, need an entry each. `RowMap::forward` and
`RowMap::inverse` are each a binary search, and `forward_dense` /
`inverse_dense` are there for a caller crossing a language boundary.

The tracks come out of one merge pass over the base's already-sorted runs, with
the deleted runs skipped and the modified ones swapped in at their place, then
the new runs appended: never a re-sort of eight million rows. The patch bitmaps,
which are the heaviest column, are shared with the base outright when the
materialisation neither moves a row nor changes a patch, and are the base's with
those rows replaced when only patches changed.

Materialisation is deterministic (the same value materialises to the same value,
with the same row map) and idempotent (materialising the result, whose overlay
is empty, changes nothing and gives the identity map). It fixes the metadata
counts a save would write and stamps nothing else: the operation and the tool
are the base's, and the write timestamp is the writer's business, not a value's.
It clears `content_hash`, the field that records the hashes of the file a value
was read from, to the empty state a never-written reconstruction carries: a
materialised value holds the base's points only where the edits left them alone,
so it is not the file the base came from and must not claim to be.
`content_xxh128()` is the live answer for what it is.

The result becomes the base of the next version, with empty edits. Older
versions keep their own base and their own edits, so the history's budget counts
one copy of the light columns per materialisation rather than per edit.

## Reading through the overlay

`EditedReconstruction::point` resolves one index without materialising: below
the base's point count it is a base index, at or above it an addition, and the
caller does not learn which. It hands back a `PointView`, a borrow into whichever
point set holds the point, with accessors for the geometry, the whole track and
every column, plus `to_record` for the owned form. The panel, the track rays,
the point picker and Go to Point read that one accessor.

The counts are O(1) from `base.point_count() - deleted.len() + added.len()`, and
column presence is answered from the base, since an addition set never
introduces or removes a column. A reader that wants the whole reconstruction
takes the materialisation: that is everything in `sfmtool-core` taking
`&SfmrReconstruction`, which is most of it.

## Hashes

`SfmrReconstruction::content_xxh128` gives the content hashes a save of a value
would write, computed from the value without touching the filesystem. A `.sfmr`
section hash is defined over the **uncompressed** bytes of that section's
entries, so those bytes are the whole of what a hash needs and the archive
container and its zstd frames are the whole of what it does not. The writer
serialises each section entry through one code path with two consumers: a save,
which hashes each entry's bytes and then compresses and stores them, and
`sfmr_format::content_hash_of`, which hashes them and drops them. So there is
one serialisation rule and one hashing rule, and the returned `content_xxh128`
is the one a save of that value stores, including the normalisations a write
performs on its way (tracks sorted, format version and infinity count refreshed,
depth statistics and missing normals recomputed).

Nothing about the act of saving is in the hash. The write timestamp lives in the
top-level `written.json` entry, which like `content_hash.json` sits outside every
section digest
([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md),
version 8), so two saves of one unchanged value write the same content hash and
a value hashed now still matches the file it becomes later. What a
reconstruction asserts about itself, its operation, its tool, its workspace
configuration, is inside the hash, so a caller that stamps a new operation onto
a value changes the value's hash, which is right. A base that was never written
has the hash a write of it would produce, which is what lets a point id name a
point in a value that has no file yet.

`EditedReconstruction::base_content_hash` computes that once, on first request,
and keeps it. The cost is one serialisation of the value plus one XXH128 pass
over it, with no compression; the patch bitmaps dominate both. It is affordable
on an event and not on a frame.

`EditedReconstruction::point_edit_hash` hashes a point edit that creates points.
It covers the base's `content_xxh128`, then each record's observations as the
content hash of the image that sees the point and the pixel it is seen at, then
the point that pixel set triangulates to. So two different additions on one base
hash differently, and the same addition made twice, in two sessions or after an
undo, hashes the same, which is right, since it is the same point. The image's
content hash is the base's own per-image hash (`image_file_hashes` for
`embedded_patches`, `sift_content_hashes` for `sift_files`, and the image's name
when the reconstruction carries neither); the pixel is the inline keypoint when
the base carries one and the `.sift` feature index otherwise, which is what
identifies a sighting in each mode. An edit that only modifies or deletes
creates no point and needs no hash of its own.

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

**The row map's inversion has a run, not a point.** For base points the forward
map is `slot - holes_below(slot)`, and inverting it means solving
`slot - holes_below(slot) == new`. That equation has more than one solution: a
hole and the live slot after it both satisfy it, because a hole contributes
nothing to the count below itself. Writing `slot = new + m`, the difference
`holes_below(new + m) - m` falls by nought or one per step of `m`, so it is
non-increasing and a binary search finds where it crosses zero; the answer is
the **last** `m` of the crossing run, which is the only one that is not a hole.
Taking the first instead returns a deleted slot, and every read through the map
then reads a neighbour's point.

**A materialisation's addition entries need two orders.** The pairs come out in
emission order, which is base-slot order for the modified points and addition
order for the appended ones, so the list is sorted by neither index on its own.
`forward` and `inverse` each binary-search, so the map keeps the pairs twice,
once sorted by the edited index and once by the new one. They are the size of
the edit, so the second copy is free.

**A record's `NaN` is not a difference.** A free point's constraint distance is
`NaN` by definition, and `NaN != NaN`, so a structural comparison reports two
identical reconstructions as different for no reason other than that neither
constrains anything. `EditedReconstruction`'s `PartialEq` reads two `NaN`
distances as agreeing, since both say the point is at no distance from anything.

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

The overlay is covered by
`crates/sfmtool-core/src/reconstruction/edited/tests.rs` and, from Python, by
`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`. Both build on
a reconstruction carrying every optional column, so a record that omits or
invents one is a visible failure. What they pin:

- Every read through the overlay equals the same read on the materialised value
  under the row map, over pseudo-random sequences of delete, replace and add.
- A run of point edits leaves the base the same `Arc` (`Arc::ptr_eq`).
- Indexes are stable: an index that resolved before an edit and was neither
  deleted nor replaced resolves to the same record after it, and a deleted or
  replaced index resolves to nothing.
- Materialisation is deterministic and idempotent, its row map is a bijection
  from the surviving indexes onto the materialised rows, a modified point lands
  back at the base index it replaced, and a version that changed no patch shares
  the base's bitmap array by pointer.
- A record is refused, with the addition set untouched, when it carries the
  wrong columns, an image the base does not hold, a bitmap of the wrong
  resolution, or no observations at all.
- A materialised value carries no file's hashes: its `content_hash` is the empty
  state.
- The hash a materialised value reports equals the `content_xxh128` a save of it
  writes, read back off the file, and two saves a moment apart write that same
  hash and two different timestamps.

`crates/sfmr-format/src/tests.rs` pins the hashing itself: that
`content_hash_of` agrees section for section with what a write of the same data
stores, over a reconstruction carrying every optional column; that two saves
differ only in the timestamp; and that a file authored in the pre-version-8
layout (timestamp in `metadata.json`, no `written.json`) loads with the
timestamp it stored and verifies against the hashes it stored.

## Non-goals

- Row-level surgery on the base's columns. The base is never written, and
  `Arc::make_mut` appears nowhere.
- Branching, or an edit applied to a version other than the one in hand. The
  overlay is one version's worth of edits on one base.
- Persisting the overlay. A file is always a materialisation.
- The policy that decides *when* to materialise, the GPU-side consequences of
  the deleted set and the additions buffer, and the point-id version graph that
  tracks a point's identity across versions. Those are proposed in
  [`../../drafts/sfm-explorer-editing-overlay.md`](../../drafts/sfm-explorer-editing-overlay.md).
- Sharing between reconstructions beyond the two heavy columns. The track
  columns are most of the remaining bytes and are untouched by every bulk edit
  but an image deletion, so they could be shared too; they are not, until there
  is a workload measuring the difference.
- Interior mutability or observation of one reconstruction's edits by another.
  A reconstruction is a value; an operation that changes one returns a new one.
