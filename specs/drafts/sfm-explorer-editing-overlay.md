# Edited reconstructions: a base plus its edits

**Status:** Draft

Answers Part 4 of [`sfm-explorer-editing.md`](sfm-explorer-editing.md), the
umbrella draft for editing a loaded reconstruction in place, which poses the
row-level edit problem without solving it. That draft's Part 4 links here and
shrinks to a pointer; when this ships, its content files into
`core/reconstruction/shared-columns.md` beside the column model it builds on,
and the umbrella's step 6 is deleted.

A reconstruction in memory is a million points and ten million track
observations, stored as sorted columns with a prefix-sum index (CSR), and the
value-semantics model of the umbrella draft keeps one such value per history
version. Copying a column to insert one row into it costs the whole column, so
any edit that changes the track structure, which is nearly every interesting
edit, is unaffordable under a plain column-sharing model. This draft proposes
that an edited reconstruction is an **immutable base plus a small set of
edits**, that every edit reduces to deleting points from the base and adding
points to a side set, and that the plain CSR form is produced only when
something needs it.

Decided: the two-part representation, the delete-and-re-add rule, stable
indexes across the base's lifetime, and materialisation as the commit. Not
decided: the materialisation policy's thresholds, and how many read paths look
through the overlay rather than materialising.

---

## The representation

An edited reconstruction has three parts:

- **The base**: a shared, immutable `SfmrReconstruction`. It is what was
  loaded, or the last materialisation. Every version in a run of edits shares
  it.
- **The deleted set**: which of the base's points are gone, as a bitset over
  the base's point indexes, and which of its images are gone, as a bitset over
  its image indexes.
- **The additions**: the points that exist in this version and not in the base,
  as a point set of their own with the same columns the base's point side has:
  points, CSR tracks, the per-observation columns, the patch frames and
  bitmaps, the constraints. Their observations refer to the base's image table.

The images and cameras are the base's, with per-version overrides for the
small columns (a pose that moved). Those columns are copied whole on edit, as
the umbrella's column model says, because a thousand images is nothing.

```rust
/// A reconstruction value that is a shared base plus what this version
/// changed. Equal bases and equal edits are equal values.
pub struct EditedReconstruction {
    pub base: Arc<SfmrReconstruction>,
    pub deleted_points: BitSet,
    pub deleted_images: BitSet,
    /// Poses, cameras and names that differ from the base's, copied whole.
    pub images: Option<Arc<[SfmrImage]>>,
    pub added: PointSet,
}
```

Under it, `SfmrReconstruction` splits into an image table and a `PointSet`,
so the base's point side and the additions are one type and every per-point
algorithm takes a `PointSet` and an image table rather than the whole. That
split is the structural change this draft asks for; the alternative, an
additions type that mirrors the point columns by hand, would be a second copy
of the schema that drifts.

### Every edit is delete-and-re-add

A point that changes in any way, its position, its constraint, its normal or
frame, its track, is deleted from the base and re-added to the additions with
its whole record and whole track. The cost of an edit is the size of the
records it touches, never the size of the base:

- **Add an observation to a track**: delete the point, re-add it with its
  observations plus one. Tens of rows.
- **Remove an observation, split a track, merge two tracks, refit a point**:
  the same.
- **Delete a point**: one bit.
- **Move a pose**: the image column is overridden, and every point the image
  observes is re-triangulated at the new pose, so all of those points are
  deleted and re-added with their new positions. Thousands of rows for a
  well-observed image, still far below the base.
- **Delete an image**: one bit, and the same re-triangulation of what it
  observed, minus its observations; a point left with fewer than two
  observations is deleted and not re-added.
- **Bundle adjust**: every point moves, so this is a materialisation by
  construction (below).

There is no third category. An edit never writes a row into the base's
columns, so the base's identity, and everything keyed on it, holds for as long
as the base does.

### Indexes are stable

A base point keeps its index while the base lives; a deleted index is a hole
that resolves to nothing. An added point takes an index at or after the base's
point count, assigned once and never reused within the base's lifetime. So an
edit shifts no index: the selection, the Point Track Detail panel, a
`pt3d_<hash>_<index>` id, the MCP addressing, and the GPU instance buffers all
survive an edit unchanged, and the umbrella's open question about selection
remapping is closed by construction rather than by a rule. Materialisation is
the one operation that renumbers, and it is where a remap is produced.

A point re-added after a modification gets a new index. Whether the panel
follows a modified point across that (select the re-added record when the
selected one was modified) is a panel rule, and the edit knows both indexes,
so it is cheap either way.

---

## Materialisation

Materialising an edited reconstruction produces a plain `SfmrReconstruction`:
the base minus its deleted points and images, with the additions appended and
the whole re-sorted into CSR, derived indexes rebuilt, image indexes
compacted if images were deleted. It is one full copy and it produces a **row
map** from the edited indexes to the new ones, which the selection and the GPU
buffers consume.

It happens when:

- an algorithm that takes a plain reconstruction runs: bundle adjustment, the
  census, alignment, a save;
- the edits have grown past a fraction of the base, so that looking through
  the overlay no longer pays;
- explicitly, as a "flatten" the user or an agent asks for.

The result becomes the **base of the next version**, with empty edits. Older
versions keep their old base and their own edits, so a walk backwards through
the history still shares, and the history budget counts one full copy per
materialisation rather than per edit. Bundle adjustment is the common case: it
reads a materialised value and its output is the next base.

The fraction and the explicit trigger are the open policy. The measurement in
the umbrella's step 1 (a full clone of the largest real reconstruction) sets
what a materialisation costs, which bounds how often one is acceptable.

---

## Reading through the overlay

A reader of an edited reconstruction sees the base's points minus the deleted
ones, then the additions. Two kinds of reader exist:

- **Per-point readers** resolve one index: base index below the base's count
  and not deleted, addition index otherwise. The Point Track Detail panel, the
  track rays, the point picker, a single-point photometric refit, Go to Point.
  These learn the overlay, through one accessor that hides which side an index
  came from, and never materialise.
- **Whole-reconstruction readers** take a plain value and get the
  materialisation. Everything in `sfmtool-core` that takes
  `&SfmrReconstruction` today, which is most of it.

How many readers in the viewer are of the first kind, and whether any hot
per-frame path in it is of the second, is the census question the umbrella's
step 1 already asks. The design holds if the per-frame readers are all
per-point or per-column, which is what the rendering path is today.

### The GPU side

The base's buffers keep their identity, so the umbrella's identity-based
upload sees no change on the base and uploads nothing for it. The deleted set
uploads as a small per-point mask the point shader reads. The additions upload
as a second instance buffer drawn after the base's, with the same per-node
uniforms. A materialisation replaces both with one buffer, through the row
map. No edit re-uploads a million points.

### The file side

A save materialises, then writes as any writer does. An observation added from
a clicked pixel has a keypoint and a patch but no feature index, so the first
track edits are built on `embedded_patches` reconstructions, where that is
what an observation is. On a `sift_files` reconstruction such an edit refuses
until the format carries an observation without a feature; that format
decision is not made here.

---

## Testing

- An edited reconstruction and its materialisation agree: every per-point
  read through the overlay equals the same read on the materialised value
  under the row map, for random edit sequences.
- Delete-and-re-add is total: every edit family produces a value whose base is
  the same `Arc` as before the edit.
- Stable indexes: after any edit that is not a materialisation, every index
  that resolved before and was not deleted or modified resolves to the same
  record.
- Materialisation is deterministic and idempotent, and its row map is a
  bijection from the surviving edited indexes onto the new ones.
- A pose edit's re-triangulation matches the batch triangulation of the same
  points at the same poses.

## Non-goals

- Row-level surgery on the base's CSR columns. The base is never written.
- Branching, or edits applied to a version other than the cursor's.
- Persisting the overlay. A file is always a materialisation.

## Open questions

- The materialisation fraction, and whether it is measured in points, in
  observations, or in unshared bytes.
- Whether a modified point keeps its index (re-add in place, with a
  per-version "modified" set beside "deleted") rather than taking a new one.
  It would spare the panel rule above at the cost of a third structure; the
  first track edit decides it.
- Whether the `PointSet` split lands as part of the shared-columns step or as
  its own step before the first row-level edit.
