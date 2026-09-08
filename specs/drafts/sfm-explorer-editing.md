# SfM Explorer editing

**Status:** Draft

The viewer today is read-only. Every operation that changes a reconstruction
runs offline (`sfm xform`, the seed scripts, the Python bindings), and the one
in-viewer operation that produces a different reconstruction, `Resect Image`,
shows its answer as a second node beside the original rather than changing the
original ([`../gui/resect-image.md`](../gui/resect-image.md)). This draft
proposes that a loaded reconstruction becomes editable in place, with a full
edit history the user can walk in both directions and see in a panel, and a way
to write the result back to disk.

It is an arc, not one change. This document is the umbrella: it fixes the
document model every step shares, poses the design questions that are still
open, and lists the steps in order, each naming the standing spec it files
into. A step that ships is deleted from here and written in the present tense
where it belongs, so this file is always exactly the unbuilt remainder, and it
is retired with the last step.

Decided: the document is a value (below), history stores values rather than
commands, sharing is per column, change detection is by identity, node
identity survives edits, the Action Log stays text and the history is the
replayable thing, and the history is per node. Not decided: the history
memory bound's value, and what the row-level edit draft leaves open.

Related standing specs, which will each carry a present-tense sentence pointing
here once the first step lands: [`../gui/scene-graph.md`](../gui/scene-graph.md)
(the node, and the invariant that a node transform never touches the
reconstruction, which one edit here deliberately breaks),
[`../gui/action-log.md`](../gui/action-log.md) (whose non-goals list "an undo
stack"), [`../gui/mcp-server.md`](../gui/mcp-server.md) (the wire surface the
edits join), and
[`../formats/sfmr-file-format.md`](../formats/sfmr-file-format.md) (what a
consumer that rewrites a file must preserve, which is what a save is).

---

## Part 1: the document model

### The reconstruction is a value

The model is the one in Sean Parent's
[*Value Semantics and Concept-based Polymorphism*](https://sean-parent.stlab.cc/presentations/2013-09-24-value-semantics/value-semantics.pdf)
(2013): the document is a regular type, an edit is a function
from one document value to the next, and undo is a vector of document values
with a cursor. Nothing in the model has identity that an edit mutates in place;
the *node* has identity (its `ReconId`, its label, its display state), and it
holds a current value that is replaced whole. This removes the class of undo
bug where a command's inverse disagrees with what the command did: there is no
inverse, only the previous value.

Storing a value per version is affordable only when a copy shares what did not
change. A reconstruction the viewer targets is up to a million points, ten
million observations, and patch bitmaps and thumbnails that dominate memory by
an order of magnitude over everything else. A whole-struct clone per edit is
out.

### Sharing is per column

`SfmrReconstruction` in
[`data.rs`](../../crates/sfmtool-core/src/reconstruction/data.rs) is already
columnar: `points`, `tracks`, `observation_counts`, the observation-source
columns, `patch_u_halfvec_xyz` / `patch_v_halfvec_xyz`, `patch_bitmaps_y_x_rgba`,
`thumbnails_y_x_rgb`, `normal_confidence`, `point_constraints`,
`observation_confidence`, and the derived `observation_offsets`,
`image_feature_to_point`, `max_track_feature_index`. Each of these becomes a
shared column: a newtype over `Arc<…>` that dereferences to the inner type for
reading, so no read site in the workspace changes, and exposes an explicit
make-mut for writing, which clones the column only when it is shared. A version
that moved one image's pose owns a fresh `images` column and shares every other
column with its predecessor.

This lives in `sfmtool-core`, not the viewer, because the struct is core's and
is what the PyO3 bindings and every pipeline hold. The write sites are the cost
of the step: every `recon.points.push`, `recon.tracks[i] = …`, and
`clone_with_changes` in the workspace goes through the make-mut. The census in
step 1 sizes that. The bindings' behaviour does not change: a Python caller
still receives arrays and hands back arrays; whether the Rust side shares or
copies underneath is invisible from Python.

The derived indexes are columns like any other and are shared like any other.
They are part of the value: a version whose `tracks` column is unchanged shares
its offsets and feature maps too, and only an edit that touches the track
structure pays to rebuild them. `rebuild_derived_indexes` remains the single
place that knows how, and the make-mut on `tracks` or `observation_counts` is
where the obligation to call it attaches (§ Implementation notes, when filed).

Files into: `specs/core/reconstruction/shared-columns.md` (new).

### Change detection by identity

The 3D viewport draws from GPU buffers the app fills from the node's
reconstruction: the point instance buffer, the frustum and image-quad
geometry, the thumbnail atlas, and the patch instances with their bitmap
atlas. A node today carries one `needs_upload` boolean and, when it is set,
the app's per-frame upload phase rebuilds all of those from the reconstruction
at once; a transform change is noticed through a separate epoch counter. With
shared columns the upload phase keeps, per GPU resource, the identity of the
column it last uploaded from, and compares identities each frame. Undoing a point edit re-uploads the point
buffer and nothing else; a pose edit re-uploads frustums; undoing to a version
whose columns are all the ones already on the GPU uploads nothing. The boolean
and the epoch both go, replaced by one mechanism, and the GPU-side cost of an
undo is proportional to what the undo changed.

Files into: `specs/gui/document-model.md` (new), amending the upload sections
of [`../gui/scene-graph.md`](../gui/scene-graph.md).

### Node identity survives edits

The `ReconId` names the node across every version. Solo, tint, eyes, the
transform, the selected reconstruction, and the MCP addressing all keep
working through an edit without re-pointing, which is not what re-opening a
loaded file does today (it mints a fresh id and re-points the solo).

Re-reading a file is the same thing. `Open` on a path that is already loaded
re-reads it into the node it already has, as a new version at the cursor
whose base is the file's content, so the id survives and an undo returns to
the edited state. That makes the Scene Graph's `Reload from Disk` entry the
same action as `Open` on the node's path, and it goes.

The selection is the exception that needs a rule. A selected point or image is
an index into the current value, and an edit that deletes rows shifts the
indexes after it. Options: clear the selection on any structural edit; remap
it through the edit's row map; or key the selection by something stable. The
history already knows what each structural edit removed, so remapping is
cheap, and clearing is what a user would notice as a bug. The overlay draft
([`sfm-explorer-editing-overlay.md`](sfm-explorer-editing-overlay.md))
makes indexes stable across every edit but a materialisation, so remapping
is needed only there, where the materialisation's row map supplies it.

---

## Part 2: history

### Values with a cursor

A node's history is a vector of reconstruction values and a cursor. Undo moves
the cursor back, redo moves it forward, a new edit at a cursor that is not at
the end truncates the versions after it (the Photoshop rule; a branching
history is a non-goal). Each version carries a label (the sentence the Action
Log recorded for the edit), a timestamp, and the unshared bytes it holds
relative to its neighbours, which is what the memory bound is measured in.

The history is bounded by a memory budget, not by a version count: a hundred
pose edits share everything but the images column and cost nothing, while one
edit that touches the bitmaps costs the bitmaps. When the budget is exceeded
the oldest versions are dropped from the front. The budget's value is open;
the measurement in step 1 informs it.

Coalescing: an interactive edit that produces intermediate values (a drag) is
one version, committed when the gesture ends. This is the same rule the Action
Log applies to sliders, applied to the history instead of to text.

The history is **per node**: each loaded reconstruction carries its own
versions and cursor. Edits do not cross nodes, and a scene-wide cursor would
make undoing an edit in one node silently undo the last edit in another. The
Edit menu, the shortcuts and the History panel act on the selected
reconstruction, and the panel's header names it. An edit that spans nodes is
a non-goal.

### The log and the history

The Action Log stays text, per session, and complete: `Undo: Deleted 12 points
in global` is an entry. The history is the replayable record and is per node.
The two are not the same thing and neither is derived from the other; the log
non-goal "an undo stack" stays true of the log.

### The History panel

A dock tab, registered with the panel layout like every other panel
([`../gui/panel-layout.md`](../gui/panel-layout.md)), listing the selected
node's versions oldest first with the cursor marked. Clicking a row jumps the
cursor there, in either direction, in one step. Each row shows the label, the
time, and the unshared size; the row at the disk state (the last saved
version, or the loaded one) is marked, which is the dirty indicator in the
panel's vocabulary. Keyboard: the platform's undo and redo shortcuts act on
the selected node from any panel that does not consume them for its own text
editing.

Files into: `specs/gui/edit-history.md` (new).

---

## Part 3: saving

Save writes the current value over the node's path; Save As writes it to a
chosen path and re-points the node; Revert jumps the cursor to the disk state,
which differs from `Open` on the same path only when the file changed on disk
since it was read.
A node that came from no file (demo data, a derived resection) has only Save
As. The title and the tree row carry a dirty marker when the cursor is not at
the disk state. Save recomputes the content hash and appends provenance to the
metadata the way every writer does, and it preserves whatever the format spec
says a rewriting consumer must preserve. Closing a dirty node, or the viewer
with one open, asks.

Files into: `specs/gui/saving.md` (new).

---

## Part 4: the design challenge: row-level edits under value semantics

Column sharing solves the easy edits: a pose moves, a point moves, a constraint
changes, a column is replaced whole and the rest is shared. It does not, on its
own, solve the edit that is the reason to build this at all.

Consider the first track edit we want. Select a point in the Point Track Detail
panel, then select an image that is not in its track. Right-click in that image
and choose *add a keypoint here to the track*. The viewer places the
observation, runs the photometric fit against the track's patch (the same
kernel the embed pass uses), and the track now has one more observation.

What that edit does to the value:

- `tracks` gains one row, and the rows are sorted by point then image, so it is
  an insertion in the middle of a ten-million-row column, not an append.
- `observation_counts[p]` increments, and `observation_offsets` shifts by one
  for every point after `p`.
- The observation-source columns (`feature_indexes` or `keypoints_xy`,
  `observation_confidence`) each gain a row at the same position.
- `image_feature_to_point[i]` and `max_track_feature_index[i]` change for that
  image.
- The point's position, error, normal, and patch frame may all move after the
  fit.

A make-mut on `tracks` clones ten million rows to insert one. That is tens of
milliseconds and eighty megabytes per keystroke, and a hundred such edits in a
history hold eight gigabytes of tracks that differ by a hundred rows in total.
The naive column model fails exactly on the edit that matters.

The answer is proposed in
[`sfm-explorer-editing-overlay.md`](sfm-explorer-editing-overlay.md): an
edited reconstruction is an immutable base plus a deleted set and an addition
set, every edit reduces to deleting points from the base and re-adding them
to the additions, indexes stay stable while the base lives, and the plain CSR
form is materialised only when an algorithm, a save, or a size threshold asks
for it. What that draft leaves open is the materialisation policy and which
read paths look through the overlay. The first track edits are built on
`embedded_patches` reconstructions, where an observation is a pixel and a
patch and nothing else; an added observation on a `sift_files` reconstruction
has no feature index behind it, and whether the format grows to carry one is
a format decision neither draft makes.

---

## Part 5: edits

Each edit family is a small standing spec in the shape of
[`../gui/resect-image.md`](../gui/resect-image.md): invocation, mechanism
pointing at the core function it wraps, what the version's label says, testing,
non-goals. The core function is always a pure function of a reconstruction
value plus named inputs, bound through `sfmtool-py` so the same edit is
available offline; the viewer adds only the invocation and the history entry.

Families, in the proposed order:

- **Delete**: selected points, an observation from a track, an image (with its
  observations and any track left under two views). The first structural edit,
  and the one the selection-remapping rule is tested against.
- **Point constraints**: set a selected point free, ranged, or held, with the
  reference image and distance, from the Point Track Detail panel. The
  ground-truth workflow does this today in scripts against a CSV.
- **Bake transform**: apply the node's `Align to…` transform to the value and
  reset the transform to identity. This is the edit that breaks the scene-graph
  invariant, on purpose and only when asked.
- **Resect in place**: the existing resection applied to the node as a
  version rather than landing a derived node. The derived-node variant stays as
  the comparison affordance.
- **Track edits**: add an observation to a track from a pixel (Part 4), remove
  one, split a track, merge two. Gated on Part 4's answer.
- **Bundle adjust**: run the adjustment on the node's value with the
  constraints it carries, as one version.

Files into: `specs/gui/edits/<family>.md`, one each, plus an
`edits/README.md` index; `resect-image.md` gains its in-place variant.

---

## Part 6: the wire

`undo`, `redo`, `get_history` (the version list with the cursor, so an agent
can read what a human did and where they are), `save`, and one tool per edit
family, each applied on the GUI thread at the same point in the frame as every
other tool. An agent's edit is a version like any other, attributed in the
history's label the way the Action Log attributes it, and a human can undo it.

Files into: [`../gui/mcp-server.md`](../gui/mcp-server.md).

---

## The plan

Each step is one PR, has its own spec change, and is verifiable without the
steps after it.

1. **Census and measurement.** List every write site of a reconstruction
   column across the workspace and every read path that walks `tracks` as CSR;
   time a full clone of the largest real reconstruction with and without
   bitmaps; measure a ten-million-row insertion. The numbers go into this
   draft (the memory budget, the overlay decision) before step 2 starts.
2. **Shared columns in core.** The column newtype, every write site converted,
   the derived indexes as columns, bindings unchanged. Byte parity on the full
   Python and Rust suites is the acceptance test. Files
   `core/reconstruction/shared-columns.md`.
3. **Document model, undo and redo, one edit.** History as values on the node,
   identity-based upload replacing `needs_upload` and the transform epoch, Edit
   menu, shortcuts, Action Log entries, delete-selected-point as the one edit
   that proves the loop. Files `gui/document-model.md` and
   `gui/edit-history.md`, amends `gui/scene-graph.md` and `gui/action-log.md`.
4. **History panel.** Files into `gui/edit-history.md`, amends
   `gui/panel-layout.md`.
5. **Saving.** Files `gui/saving.md`.
6. **Row-level model.** The base-plus-edits representation of
   [`sfm-explorer-editing-overlay.md`](sfm-explorer-editing-overlay.md), on
   `embedded_patches` files, with the add-observation edit as the proof.
   Amends `core/reconstruction/shared-columns.md`.
7. **Edit families**, one PR each in Part 5's order, `gui/edits/`.
8. **Wire surface.** Amends `gui/mcp-server.md`.

Steps 2 and 3 are the groundwork; 4 and 5 are independent of each other and of
6; 7 interleaves with 6 as each family's needs are met.

## Non-goals

- Branching history. A new edit after an undo discards the redo tail.
- History persistence across sessions. A save writes the value, not the
  versions.
- Editing across nodes in one version.
- Undo of display state (eyes, tint, transform, layout). Those stay outside
  the history; bake-transform is the one bridge and it is an edit.

## Open questions

- The history memory budget's value (Part 2), after step 1's numbers.
- The overlay draft's open questions (materialisation policy, how the
  version graph's maps are stored, when the point-set split lands).
