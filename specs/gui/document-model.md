# The document model: a loaded reconstruction as a value

A reconstruction loaded into the viewer can be changed: a point deleted, an
image dropped, and in time a pose moved or a track edited. What the viewer holds
for each loaded file is therefore not one reconstruction but a **sequence of
them** -- the one that was read off disk, and one for every change made since --
with a cursor saying which of them is on screen. An edit adds the next one and
moves the cursor forward; undo moves it back. Nothing is ever modified in place,
so undo does not have to reverse anything: it points at a value that is still
there.

That would be unaffordable if each version were a whole copy. A reconstruction
the viewer targets is up to a million points, ten million observations, and
per-point patch bitmaps that are most of its memory, and a run of small edits
must not cost a copy of that per edit. So a version is a **base plus its edits**:
an immutable reconstruction behind a shared pointer, plus the handful of point
changes made on top of it. A point edit shares the base with the version before
it and costs the size of the edit; a bulk edit -- one that changes the image
table or the structure wholesale -- produces a new base, sharing the two heavy
columns with the old one when it did not touch them.

This spec describes what a node holds, the two kinds of edit, and how the
renderer notices a change. The cursor's own semantics -- undo, redo, truncation,
and how a selection follows an edit -- are in
[edit-history.md](edit-history.md); the value type under all of it is
[`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md).

## What a node holds

The node is [`SceneNode`](../../crates/sfm-explorer/src/scene.rs)
([scene-graph.md](scene-graph.md)). Its identity -- the `ReconId`, the label, the
path, the eyes, the tint, the transform -- belongs to the node and survives every
edit; what changes is the value it shows. That value lives in a `History`, in
[document.rs](../../crates/sfm-explorer/src/document.rs).

```rust
pub struct History { /* versions, cursor, maps */ }

pub struct Version {
    pub serial: VersionSerial,
    pub label: String,
    pub at: jiff::Timestamp,
    /// `None` once the budget has released it.
    pub value: Option<EditedReconstruction>,
    pub unshared_bytes: u64,
}

impl History {
    pub fn new(base: SfmrReconstruction, label: impl Into<String>) -> Self;
    pub fn current(&self) -> &EditedReconstruction;
    pub fn current_version(&self) -> &Version;
    pub fn versions(&self) -> &[Version];
    pub fn push(&mut self, value: EditedReconstruction, map: PointMap,
                label: impl Into<String>) -> VersionSerial;
    pub fn can_undo(&self) -> bool;
    pub fn can_redo(&self) -> bool;
    pub fn undo(&mut self) -> Option<(VersionSerial, VersionSerial)>;
    pub fn redo(&mut self) -> Option<(VersionSerial, VersionSerial)>;
}
```

Node identity is the node's, not the value's: the `ReconId` names the node
across every version, and the tint, the eyes, the transform, the solo, the
selected reconstruction and the MCP label addressing all keep working through an
edit without re-pointing. Nothing re-reads a node in place, either: `Open` on a
path that is already loaded appends a second node for it, with a history of its
own, so two nodes over one file are two independent documents.

A **version serial** is minted from a process-wide counter and never reused, by
any node, including by a version a discarded redo tail took with it. It is what
the Action Log names a step by, and what a map is keyed on.

The node offers the reads every panel makes:

```rust
impl SceneNode {
    /// The value: base plus this version's point edits.
    pub fn edited(&self) -> &EditedReconstruction;
    /// The base of that value.
    pub fn recon(&self) -> &SfmrReconstruction;

    pub fn point_count(&self) -> usize;
    pub fn image_count(&self) -> usize;
    pub fn infinity_point_count(&self) -> usize;
    pub fn is_point_deleted(&self, index: u32) -> bool;
}
```

`recon` is the read path for anything addressed by an index the overlay keeps
stable: an image, a camera, one point, one point's track, whether a column is
present. It is a borrow of the base, so a run of point edits hands back the same
reference every time, and that identity is what the upload path keys on.

`point_count`, `image_count` and `infinity_point_count` are the counts the
viewer *shows* -- the Scene panel's row, the viewport's stats overlay, the MCP
scene reply -- and they read the overlay rather than the base, in O(1) or in the
size of the edit. Nothing on a frame path materialises: the counts are
subtraction, and every per-frame read is per-point or per-image.

### Why the value is not simply a reconstruction

A node could have held one `SfmrReconstruction` and replaced it per edit. It
does not, for two reasons that pull the same way. The first is cost: a version
per edit is a version per keystroke, and a whole value is a copy of a
million-point cloud. The second is that undo of an in-place change needs an
inverse, and an inverse that disagrees with what the operation did is a class of
bug that cannot be tested away. Storing the previous value has no inverse to
disagree with.

The value model is enforced rather than merely intended: the base is never
written through, `Arc::make_mut` appears nowhere, and there is no interior
mutability in the document. An edit's `&mut` reaches the *overlay* a version
owns, never anything a second version can see.

## Two kinds of edit

Both edits live in
[state/edits.rs](../../crates/sfm-explorer/src/state/edits.rs), as methods on
`AppState`, and both are functions from the value at the cursor to the next
value.

```rust
impl AppState {
    pub fn delete_selected_point(&mut self) -> Result<(), String>;
    pub fn delete_point(&mut self, point: PointRef) -> Result<(), String>;
    pub fn delete_image(&mut self, image: ImageRef) -> Result<(), String>;
    pub fn undo(&mut self, id: ReconId) -> Result<(), String>;
    pub fn redo(&mut self, id: ReconId) -> Result<(), String>;
    pub fn can_undo(&self, id: ReconId) -> bool;
    pub fn can_redo(&self, id: ReconId) -> bool;
}
```

**Delete point is the point edit.** It clones the current value -- which clones
the overlay and shares the base pointer -- calls
`EditedReconstruction::delete_point`, and pushes the result. The base is the same
`Arc`, every other index still means what it meant, and the version's unshared
cost is one `u32`.

**Delete image is the bulk edit.** It materialises the current value when the
overlay is not empty, runs `SfmrReconstruction::subset_by_image_indices` over the
remaining images with orphaned points dropped, and pushes the output as the next
version's base with an empty overlay. The image table is renumbered, so every
image index at or past the deleted one moves down by one, and the points the
deleted image was the only witness of are gone.

An edit that cannot run changes nothing and returns the sentence saying why: no
point selected, a point already deleted, an image already gone, or the node's
last image, which cannot be deleted because a reconstruction with no images is
not one.

### What a bulk edit owes the rest of the viewer

A point edit shifts no index, so nothing outside the node has to be told. A bulk
edit renumbers the image table, and a cached decode or a selected index keyed by
an image is then a statement about a different photo. So `delete_image`, undo and
redo each drop the state-level caches for the node (the SIFT and full-resolution
caches) and clear its image, camera and hover selections, and the caller drops
the panel-local texture caches -- the same three-part release closing a node
performs. The node keeps its `ReconId` through all of it, which is what makes an
edit different from closing and re-opening: the tint, the transform, the solo
and the MCP addressing carry over untouched.

## Change detection by identity

The 3D viewport draws from GPU buffers filled from the node: a point instance
buffer, frustum and image-quad geometry, a thumbnail atlas, and patch instances
with their bitmap atlas. Each node's buffers live in one bundle, and the bundle
remembers **the base its buffers were built from**.

The frame's upload phase, in
[app.rs](../../crates/sfm-explorer/src/app.rs), asks per node:

```rust
if renderer.base_changed(id, &base) {
    renderer.upload_points(device, id, recon);
    renderer.upload_thumbnails(device, queue, id, recon);
    renderer.upload_patches(device, queue, id, recon);
    renderer.set_uploaded_base(id, base);
}
renderer.update_deleted_mask(queue, id, &node.edited().deleted_points);
```

`base_changed` is a pointer comparison against the stored `Arc`
([upload/overlay.rs](../../crates/sfm-explorer/src/scene_renderer/upload/overlay.rs)).
A load and a bulk edit each hand the node a base the renderer has not seen, and
the three uploads run. A point edit, an undo of one, and a redo of one
all leave the base the same allocation, so none of them does.

**The deleted set reaches the shaders as a mask.** Each bundle carries, beside
its point instance buffer, a second instance buffer of one `u32` per point: `1`
alive, `0` deleted. `update_deleted_mask` writes only the entries the set moved,
so an edit costs four bytes per point it touched rather than a rewrite of the
node. The point shader emits a clipped vertex for a masked point, so it draws
nothing, occludes nothing and answers no pick. Patches carry the same mask, one
entry per surfel; because the patch instances and the atlas are compacted over
the points that carry a bitmap, the bundle keeps the point-index-to-slot map the
mask write needs.

Undoing to a version whose base is the one on the GPU therefore writes the mask
and nothing else, and the GPU cost of an undo is the size of what it undid.

**Node transforms are detected the same way.** Mirroring a node's transform onto
its bundle reports whether it differs from the one the bundle held, and that
answer is what re-derives the global `length_scale`, re-sizes frustum geometry
and rebuilds the CPU-space track rays. What was last drawn is what is compared
against, so no counter has to be kept in step with the field it describes.

## The history budget

A node holds at most `HISTORY_BUDGET_BYTES` -- 4 GiB -- of unshared bytes across
the versions whose values it still has. Past that, the oldest values are
released, oldest first, never the version at the cursor. A released version keeps
its place, its label, its serial and its map; it is simply no longer a version
the cursor can reach, and undo refuses at it rather than stepping onto it.

The unshared cost of a version is what it holds that its predecessor did not: the
overlay alone when the two share a base, and otherwise the base's light columns
plus the thumbnail and patch-bitmap arrays only when this base does not point at
the same allocations. On the largest reconstruction measured -- 1 354 MB in
memory, of which 1 189 MB are those two columns -- that is some 165 MB for a
bulk edit, so the budget holds twenty-odd of them, or any number of point edits.

It is a constant, not a setting: it is a ceiling that keeps a session from
exhausting memory, not a quantity anyone has a reason to tune from the window.

## Implementation notes

**Derived aggregates are as of the last base change.** The auto point size, the
inter-camera distance and the node's bounding sphere are computed over the whole
point cloud during the point upload, which a point edit does not run. So after a
point deletion they describe the base rather than the version: a splat size and a
bounding sphere that include a point no longer drawn. Both are visual scale
hints, both move by less than a point's worth on a deletion, and both are
re-derived at the next bulk edit. An incremental update is not worth an
approximation that has to be kept honest.

**The bulk edit's map is derived by scanning, not restated.**
`subset_by_image_indices` returns the subset and nothing about which points it
dropped, so `delete_image` calls `RowMap::by_scan` on that call's input and
output, with the image map its keep list already describes
([`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md),
"The scanned row map"). The scan walks both point lists in order and reads off
what happened between them, so the viewer restates neither the rule for which
points a dropped image orphans nor any other edit's, and the next bulk edit gets
its map for free. The version's map is that row map, chained after the
materialisation's when there was one.

**A deleted point still resolves through the base.** Indexes are stable, so a
panel that bounds-checks against the base's point count can still reach a
deleted point's record. The selection cannot land on one (the edit clears it, and
the map clears it after) and the pick buffer cannot return one (the shader clips
it), and the track-highlight paths check the overlay. Routing every panel read
through the overlay accessor is what the first track edit needs, and is proposed
in [`../drafts/sfm-explorer-editing.md`](../drafts/sfm-explorer-editing.md).

## Testing

`crates/sfm-explorer/src/document/tests.rs` covers the history in isolation: the
cursor, truncation, the maps that outlive a truncation, the budget releasing
values and never maps, and each `PointMap` case forward and inverse.

`crates/sfm-explorer/src/state/edits/tests.rs` covers the two edits end to end --
that a point edit leaves the base the same `Arc` and moves the count by one, that
a bulk edit produces a new base with an empty overlay under the same `ReconId`,
that a refused edit leaves the history alone, and what the Action Log records.

`crates/sfm-explorer/src/scene_renderer/upload/tests.rs` covers the upload path
against a real `wgpu` device on the `noop` backend: that an unchanged base
uploads nothing, that a point edit uploads nothing and moves only the mask, that
undoing back to the loaded value clears the mask without an upload, and that a
new base re-uploads exactly once.

`crates/sfm-explorer/tests/ui_basic.rs` covers the menu bar in a real window: the
Edit menu is present, and its four items are in the accessibility tree even when
every one of them is greyed.

## Non-goals

- Editing across nodes in one version. An edit names one node.
- Undo of display state -- the eyes, the tint, the node transform, the panel
  layout. Those are not versions of the reconstruction.
- Persisting the history. Saving a node is proposed in
  [`../drafts/sfm-explorer-editing.md`](../drafts/sfm-explorer-editing.md); a
  save writes the value, not the versions.
- Re-reading a node's file. Nothing reloads a node in place: `Open` on a path
  that is already loaded appends a second node for it, with a history of its
  own, and a node's value changes only through that history.
