# Edited reconstructions: a base plus its edits

**Status:** Draft

Answers Part 4 of [`sfm-explorer-editing.md`](sfm-explorer-editing.md), the
umbrella draft for editing a loaded reconstruction in place, which poses the
row-level edit problem without solving it. That draft's Part 4 links here and
shrinks to a pointer; when this ships, its content files into
`core/reconstruction/edited-reconstruction.md`, and the umbrella's steps 2
and 3 are deleted.

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
- **The deleted set**: which of the base's points are gone, as a hash set of
  the base's point indexes. A set, not a mask over the base: the overlay
  exists for individual point edits, so the set holds a handful of indexes
  against a million-point base, and its size is the size of the edit rather
  than the size of the base.
- **The additions**: the points that exist in this version and not in the base,
  as a point set of their own with the same columns the base's point side has:
  points, CSR tracks, the per-observation columns, the patch frames and
  bitmaps, the constraints. Their observations refer to the base's image table.

The images and cameras are the base's, unchanged. An edit to the image table
is not an overlay edit (below).

```rust
/// A reconstruction value that is a shared base plus what this version
/// changed. Equal bases and equal edits are equal values.
pub struct EditedReconstruction {
    pub base: Arc<SfmrReconstruction>,
    pub deleted_points: HashSet<u32>,
    pub added: PointSet,
    /// For each added point, the base index it replaces (a modified point),
    /// or `None` for a point that is new. What puts a modified point back in
    /// its place at materialisation, and what the version graph's point map
    /// is read from.
    pub replaces: Vec<Option<u32>>,
}
```

Under it, `SfmrReconstruction` splits into an image table and a `PointSet`,
so the base's point side and the additions are one type and every per-point
algorithm takes a `PointSet` and an image table rather than the whole. That
split is the structural change this draft asks for; the alternative, an
additions type that mirrors the point columns by hand, would be a second copy
of the schema that drifts.

### Two kinds of edit

An edit is either a **point edit**, which lives in the overlay, or a **bulk
edit**, which produces a new immutable base. The line between them is the
footprint: a point edit touches a handful of points, and a bulk edit touches
an image's worth of structure or more.

A **point edit** is delete-and-re-add. A point that changes in any way, its
position, its constraint, its normal or frame, its track, is deleted from the
base and re-added to the additions with its whole record and whole track. The
cost of the edit is the size of the records it touches, never the size of the
base:

- **Add an observation to a track**: delete the point, re-add it with its
  observations plus one. Tens of rows.
- **Remove an observation, split a track, merge two tracks, refit a point,
  set a constraint**: the same.
- **Delete a point**: one index in the set.

A point edit never writes a row into the base's columns, so the base's
identity, and everything keyed on it, holds for as long as the base does.

A **bulk edit** is a function from a plain reconstruction to a plain
reconstruction, a plain value in and a plain value out:
it materialises the current version if it has to, runs, and its output is the
next version's base with an empty overlay.

- **Delete an image**: the image row goes, its observations go, every point it
  observed is re-triangulated without it, and a point left under two
  observations goes with them. Image indexes after it shift, which is the
  renumbering the overlay is built to avoid and a bulk edit is allowed.
- **Move a pose**: the image column changes and every point the image observes
  is re-triangulated at the new pose. Thousands of points for a well-observed
  image, so it is bulk, though it renumbers nothing.
- **Bundle adjust**: every point moves.
- **Bake the node transform**: every point and every pose moves.

There is no third category, and the two compose: a run of point edits on a
base, then a bulk edit that materialises them into the next base, then more
point edits on that.

### Indexes are stable

A base point keeps its index while the base lives; a deleted index is a hole
that resolves to nothing. An added point takes an index at or after the base's
point count, assigned once and never reused within the base's lifetime. So an
edit shifts no index: the selection, the Point Track Detail panel, a
`pt3d_<hash>_<index>` id, the MCP addressing, and the GPU instance buffers all
survive an edit unchanged, and the umbrella's open question about selection
remapping is closed by construction rather than by a rule. Materialisation is
the one operation that renumbers, and then only the points after a deletion
and the points that are new; it is where a remap is produced.

A point re-added after a modification gets a new index, and the edit records
the old one against it: the point's identity continues across the
modification (§ "The version graph" under Point ids), so the selection and
the panel follow it, and its id does not change.

---

## Materialisation

Materialising an edited reconstruction produces a plain `SfmrReconstruction`
in which **every point keeps its place**: a modified point goes back to the
base index it replaced, carrying its new record and track; a deleted point's
slot closes up, shifting the points after it down by one; and only a point
that is new to this base is appended, after the last base point, in the
order the additions were made. The tracks are re-sorted into CSR around
that order and the derived indexes rebuilt. The image table is the base's,
untouched.

Keeping places is what makes the materialised file readable beside the one
it came from: a point that was edited is at the same row in both, a point
that was not is at the same row unless something before it was deleted, and
a diff of the two files is the edit. It is also what makes the **row map**
cheap: for base points it is the identity minus a prefix count of
deletions, monotone, so it is stored as the sorted deleted set and inverted
by the same count; for additions it is one index each. The selection and
the GPU buffers consume it. A merge of two tracks keeps the lower index and
deletes the higher; a split keeps the first half in place and appends the
second.

It is one copy of the light columns: the base's tracks are already sorted
and the additions are a handful, so the track column is a merge pass over
the base with the deleted tracks skipped and the modified ones swapped in at
their place, then the new tracks appended, never a re-sort. The bitmap and
thumbnail columns are shared with the base unless an addition changed a
patch, in which case the new bitmap column is the base's with those rows
replaced. On the largest real reconstruction (530 674 points, 8.1 million
observations) the light columns are 165 MB and clone in 32 ms; the umbrella
draft's "Step 1's numbers" has the measurements.

It happens when:

- a bulk edit or an algorithm that takes a plain reconstruction runs: bundle
  adjustment, deleting an image, the census, alignment, a save;
- the edits have grown past a fraction of the base, so that looking through
  the overlay no longer pays;
- explicitly, as a "flatten" the user or an agent asks for.

The result becomes the **base of the next version**, with empty edits. Older
versions keep their old base and their own edits, so a walk backwards through
the history still shares, and the history budget counts one full copy per
materialisation rather than per edit. Bundle adjustment is the common case: it
reads a materialised value and its output is the next base.

The fraction and the explicit trigger are the open policy. A
materialisation costs tens of milliseconds on the largest real
reconstruction, so it is affordable on any event and never on a frame.

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

The census
([`reports/2026-09-07-editing-read-path-census.md`](../../reports/2026-09-07-editing-read-path-census.md))
found no per-frame reader of the second kind: the eight per-frame "whole"
reads are counts and the content hash, which an overlay answers in O(1)
from `base.point_count() - deleted.len() + added.len()` and the cached base
hash. Every real whole walk is event-driven. Three of them need a design
rather than a materialisation: the derived aggregates the point upload
computes (auto point size, camera scale, scene bounds), which a point edit
must either leave stale until the next materialisation or update
incrementally, and whose staleness shows in the clip planes; the Image
Detail panel's embedded-features overlay, which walks every observation on
exactly the `embedded_patches` files the first track edits target and must
iterate base-minus-deleted plus additions instead; and column presence
(`feature_indexes`, `keypoints_xy`, the patch frames), probed every frame
in six places, which an overlay answers from the base alone, since an
addition set never introduces or removes a column.

### The GPU side

The base's buffers keep their identity, so the umbrella's identity-based
upload sees no change on the base and uploads nothing for it. The deleted set
reaches the point shader as a per-point mask built from the set, a few bytes
written into a buffer that is otherwise zero. The patch atlas is the one
piece of GPU state the stable-index rule does not already cover: its slot
assignment is a compaction over the points that carry a bitmap, so a point's
atlas slot is not its index, and a point edit that changes a bitmap needs
the additions' patches in a second atlas, or a slot map the base's atlas
keeps across edits. The additions upload
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
- Delete-and-re-add is total for point edits: every point-edit family
  produces a value whose base is the same `Arc` as before the edit, and
  every bulk-edit family produces a value with an empty overlay.
- Stable indexes: after any edit that is not a materialisation, every index
  that resolved before and was not deleted or modified resolves to the same
  record.
- Materialisation is deterministic and idempotent, and its row map is a
  bijection from the surviving edited indexes onto the new ones.
- A pose edit's and an image deletion's re-triangulation matches the batch
  triangulation of the same points at the same poses, and an image deletion's
  output equals the offline image-drop transform on the same input.

## Non-goals

- Row-level surgery on the base's CSR columns. The base is never written.
- Branching, or edits applied to a version other than the cursor's.
- Persisting the overlay. A file is always a materialisation.

---

## Point ids

A point id today is a coordinate in a file: `pt3d_{hash}_{index}`, the first
eight hex digits of the file's `content_xxh128` and the point's row in it
([the format spec's Point ID
section](../formats/sfmr-file-format.md#point-id-portable-3d-point-references)).
An edited reconstruction is a chain of bases with point edits between them,
so an id has to say which of those its index belongs to and has to keep
naming the same point as the chain grows. Two rules do that: the hash in an
id is always derived from the content of the thing that created the point,
a base or a point edit, never from a counter or a session; and an id is
minted against the **earliest** such thing the point's identity reaches. The
file form stays exactly what it is; a **session form** adds the node.

```
pt3d_{hash}_{index}            file form
pt3d_{hash}_{index}_n{node}    session form
```

| Part | Content | Example |
|------|---------|---------|
| `{hash}` | First 8 hex digits of the `content_xxh128` of the base the id is minted against, whether or not that base has been written to a file (below); or, for a point a point edit created, of that edit's content hash. | `a1b2c3d4` |
| `{index}` | The point's index in that base; or, for a point a point edit created, its index among the points that edit created. | `12345` |
| `{node}` | The node's session id (its `ReconId`), decimal. | `3` |

The session form stays in the `[a-zA-Z0-9_]` class, so it double-click
selects like the file form, and the file form is its prefix, so a session id
truncates to a file id by dropping the last field.

**Every base has a hash.** The section hashes are defined over the
uncompressed bytes of the section entries, and `content_xxh128` over those,
so the hash is a function of the value and not of a file: a base that was
never saved has the same hash a save of it would write. A materialisation
fixes the metadata a save would write (operation, provenance, timestamp) at
the moment it produces the base, and computes the hash from the serialised
sections without compressing or writing them, lazily on the first request
for an id or for the hash itself. A later save writes exactly that metadata
and those bytes, so the file's hash equals the base's, and an id minted
before the save names the same point in the file after it. The cost is one
serialisation and one XXH128 pass over the value, which is a fraction of the
materialisation that produced it; the bitmaps dominate both. Demo data is
hashed the same way, so `00000000` is no longer a state a node can be in.

**Every point edit has a hash.** A point edit that creates points (a new
track, a split's second half) is hashed over its content: the hash of the
base it applies to, and the records it adds, which are the observations with
their image content hashes and pixels, and the point they triangulate to.
That is a function of what was added and where, and of nothing else: two
different additions on the same base hash differently, and the same
addition made twice, in two sessions or after an undo, hashes the same,
which is right, since it is the same point. An edit that only modifies or
deletes creates no points and needs no hash of its own.

### The version graph

A node's history is a chain of versions, and each step between two versions
carries a **point map**: a materialisation's row map from the old indexes to
the new, a point edit's record of which base index a modified point was
re-added under, and which indexes were deleted. A point's identity is
preserved across a modification: adding an observation to a track changes
the record, not which point it is, so the map says old index to new index
and the panel follows it. Deletion ends an identity, and materialisation
gives a modified point its base index back, shifts the points after a
deletion, and numbers the new ones.

The maps are kept for every version ever minted, including versions an undo
followed by a new edit discarded. They are tiny (the size of the edits, or of
a materialisation's survivors as one array), so the node holds the whole
graph of its session even when it has dropped the values. Every version's
serial is minted once and never reused.

**Resolving an id** finds the base or the point edit the hash names among
the node's versions, then walks the graph to the cursor's version. When the minting version is an
ancestor of the cursor's, the walk is forward through the maps in order. When
it is on a discarded branch, the walk is backward from it to the last common
ancestor, inverting each map (every map is a bijection on the points that
survive it, so this is well defined), then forward to the cursor. The walk
refuses, naming the version it stopped at, when the point was deleted along
either leg, or when it was added on the discarded leg and so never existed
on the surviving one. So an id copied at any point in the session, undone
past, edited over, and pasted back, still lands on its point if that point is
still there.

**Minting an id** walks the graph the other way. The id a panel shows for a
point is not its coordinate in the cursor's base but its coordinate in the
**earliest** base it existed in: the walk goes backward from the cursor
through the maps as far as the point's identity reaches, and mints against
the base, or the point edit, it stops at. For most points that is the file the node was loaded
from, so the id shown is the one that file's readers already use, and it
survives every edit of the session, every undo, and every save. The
earliest id is the one with the largest set of versions it resolves in, so
it is the one with the best chance of surviving whatever the user does next,
which is the point of copying an id.

**A point that is in no base yet.** A track added in the session exists only
in an overlay, so the earliest thing its identity reaches is the point edit
that created it, and it mints against that edit: `pt3d_{edit hash}_{k}`,
where `k` is its index among the points the edit created (zero for a single
new track; a split's second half is index one of its edit). Nothing about
that name comes from the session. Add a track, undo, and add a different
track, and the two ids differ, because the edits' contents differ, so the
first id resolves through the version graph to a miss on the discarded
branch rather than to the second track. Close without saving, reopen the
file tomorrow, and add a track: it collides with yesterday's id only if it
is the same observations at the same pixels, in which case it is the same
point. Two sessions adding different tracks on the same file mint different
ids.

The edit hash names a row of no file, which is what the out-of-range rule is
for, and a reader that meets it in a constraints file finds it among no
file's `content_xxh128`. It is found in **lineage**: when the overlay is
materialised the point gets a row in the new base and the row map records
the edit hash and index against that row, so the id keeps resolving, the
earliest rule keeps minting it afterwards, and the saved file's lineage
metadata carries the pair for any later session. An id that names an
addition never carried into a saved file is the one kind that dies with the
session, which is what never saving means.

The two walks are inverses, so an id minted this way and pasted back
resolves in one forward walk, and the same point always shows the same id
however many times it has been modified in between. Without the earliest
rule, two copies of one point's id taken before and after a materialisation
would differ, and the later one would die on an undo past that
materialisation.

**What the file form of a session id means.** For a point that existed
unchanged in a file the node was loaded from, the file form is exactly the
point's id in that file, so every id written down against that file (the
ground-truth table, a constraints file) keeps resolving through any number
of edits, and every id the panel shows for such a point is one the file's
readers can use. For a point added in the session the index is at or past
its base's count, so the file form is out of range in the file that base
was, or will be, saved as: a tool reading it detects that the id names a
point the file does not contain, rather than silently reading another point.

**What the suffix adds.** The node id makes the same file opened as two
nodes, with different edits, unambiguous, and lets Go to Point skip the hash
search. Go to Point accepts both forms; on the file form it searches every
node's version graph for the hash and prefers the selected node, as it does
today; a closed node is a miss, named as such.

**What the panels copy.** *Copy Point ID* copies the session form, minted by
the earliest rule. A constraints file wants file ids, and a tool consuming
one accepts a session id by taking its file-form prefix and applying the
out-of-range rule above.

**What a save does.** A save materialises if the overlay is not empty, which
is a new base with a new hash and a row map from the old indexes, and writes
the current base. Saving a base that is already materialised writes the
bytes its hash was computed from and mints nothing. Within the session the
version graph resolves any older id as above. Across sessions the graph is
gone unless the file carries it, so the saved file's metadata records its
lineage: the ancestor base's hash and the point map from it, the hashes of
the point edits whose points the materialisation carried in, chained back
through every base the session materialised, which is what lets an id from
last week's file land in this week's, and what lets the earliest rule keep
minting last week's ids after a save. That metadata entry is the one
addition this draft asks of the format spec, alongside a sentence in its
Point ID section that a reader may meet the session form and takes its
prefix.

Files into: `gui/goto-point.md` (both forms, the version graph, both walks),
the Point Track Detail's copy action, `gui/edit-history.md` (the graph as
part of the history), and the format spec's Point ID section (the one
sentence) and metadata (the lineage entry).

## Open questions

- The materialisation fraction, and whether it is measured in points, in
  observations, or in unshared bytes.
- Whether the version graph's maps are stored per version as they are minted
  or compacted into one map per base once a base is superseded; the walk is
  the same either way, and the first is simpler.
