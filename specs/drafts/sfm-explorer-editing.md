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
commands, a version is a base plus its point edits, change detection is by
identity, node identity survives edits, the Action Log stays text and the history is the
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

### A base plus its edits

A version is an **edited reconstruction**
([`sfm-explorer-editing-overlay.md`](sfm-explorer-editing-overlay.md)): a
shared, immutable base `SfmrReconstruction` in
[`data.rs`](../../crates/sfmtool-core/src/reconstruction/data.rs), held by
`Arc`, plus the small set of point edits made on top of it, a set of deleted
base indexes and an addition point set. A point edit (a track gains an
observation, a point is deleted, a constraint is set) touches the edits and
never the base, so a run of point edits shares one base across every version
and costs the size of the edits. A bulk edit (delete an image, move a pose,
bundle adjust, bake a transform) is a function from a plain reconstruction to
a plain reconstruction whose output is the next version's base. The plain
CSR form that every algorithm consumes is **materialised** from base plus
edits only when a bulk edit, a save or a size threshold asks for it, and the
materialisation keeps every point in its place.

Under it, `SfmrReconstruction` splits into an image table and a point set,
so the base's point side and the addition set are one type and per-point
algorithms take a point set rather than the whole. That split lives in
`sfmtool-core`, because the struct is core's and is what the PyO3 bindings
and every pipeline hold; the bindings' behaviour does not change.

There is no copy-on-write anywhere in this. The base is one immutable value
behind one `Arc`, shared whole by every version in a run of point edits, and
the thing a point edit updates is the edit part of the version, which is its
own. A bulk edit produces a new base, which is a full copy including the
patch bitmaps and thumbnails that dominate memory; the history budget
(Part 2) is what bounds how many such bases a node holds. Whether two bases
should share those two columns when a bulk edit did not touch them is an
optimisation to decide from step 1's numbers, not part of the model.

Files into: `specs/core/reconstruction/edited-reconstruction.md` (new).

### Change detection by identity

The 3D viewport draws from GPU buffers the app fills from the node's
reconstruction: the point instance buffer, the frustum and image-quad
geometry, the thumbnail atlas, and the patch instances with their bitmap
atlas. A node today carries one `needs_upload` boolean and, when it is set,
the app's per-frame upload phase rebuilds all of those from the reconstruction
at once; a transform change is noticed through a separate epoch counter. With
a version being a base plus its edits, the upload phase keeps, per node, the
identity of the base it last uploaded from and compares it each frame. A
point edit leaves the base's buffers alone: the deleted set reaches the point
shader as a small mask, and the additions are a second instance buffer drawn
after the base's. A bulk edit changes the base, and re-uploads what changed
in it. Undoing to a version whose base is the one on the GPU uploads the mask and
the additions and nothing else. The boolean and the epoch both go, replaced
by one mechanism, and the GPU-side cost of an undo is proportional to what
the undo changed.

Files into: `specs/gui/document-model.md` (new), amending the upload sections
of [`../gui/scene-graph.md`](../gui/scene-graph.md).

### Node identity survives edits

The `ReconId` names the node across every version. Solo, tint, eyes, the
transform, the selected reconstruction, and the MCP addressing all keep
working through an edit without re-pointing.

There is no reload within a node. `Open` always adds a node, so opening a
path that is already loaded opens it a second time, as a second node with
its own history, and the Scene Graph's `Reload from Disk` entry goes, along
with today's rule that opening a loaded path reloads it in place. A node's
value changes only through its own history.

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
point edits on one base cost the size of the edits, a bulk edit costs a copy
of the light columns, and only an edit that touches the bitmaps costs the
bitmaps. When the budget is exceeded
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
chosen path and re-points the node; Revert jumps the cursor to the disk state.
A node that came from no file (demo data, a derived resection) has only Save
As. The title and the tree row carry a dirty marker when the cursor is not at
the disk state. Save recomputes the content hash and appends provenance to the
metadata the way every writer does, and it preserves whatever the format spec
says a rewriting consumer must preserve. Closing a dirty node, or the viewer
with one open, asks.

Files into: `specs/gui/saving.md` (new).

---

## Part 4: the design challenge: row-level edits under value semantics

A plain value per version, with the heavy columns shared, solves the bulk
edits: a pose moves, an image goes, a bundle adjustment runs, and the version
is a copy of the light columns. It does not, on its own, solve the edit that
is the reason to build this at all.

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

A version that is a plain value clones ten million rows of `tracks` to
insert one. That is tens of
milliseconds and eighty megabytes per keystroke, and a hundred such edits in a
history hold eight gigabytes of tracks that differ by a hundred rows in total.
A plain value per version fails exactly on the edit that matters.

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

1. **Census and measurement.** List every read path in the viewer that walks
   a reconstruction, and whether it reads one point or the whole; time a full
   clone of the largest real reconstruction, and how much of it is bitmaps
   and thumbnails; time a materialisation of a base with a handful of edits.
   The numbers go into the two drafts (the memory budget, the materialisation
   threshold, whether bases share their heavy columns) before step 2 starts.
2. **The point-set split, in core.** `SfmrReconstruction` becomes an image
   table plus a point set. Bindings unchanged; byte parity on the full Python
   and Rust suites is the acceptance test. Files the first half of
   `core/reconstruction/edited-reconstruction.md`.
3. **The edited reconstruction, in core.** The base-plus-edits value, the
   point edits as delete-and-re-add, the per-point accessor that looks through
   the overlay, materialisation with every point in its place and its row map,
   the base and edit hashes, bound so an offline caller can build and
   materialise one. Files the rest of
   `core/reconstruction/edited-reconstruction.md`.
4. **Document model, undo and redo, one edit of each kind.** History as
   versions on the node, the version graph and its point maps, base-identity
   upload with the deleted mask and the additions buffer replacing
   `needs_upload` and the transform epoch, Edit menu, shortcuts, Action Log
   entries. Delete-selected-point is the point edit and delete-image the bulk
   edit that together prove the loop. Files `gui/document-model.md` and
   `gui/edit-history.md`, amends `gui/scene-graph.md` and `gui/action-log.md`.
5. **History panel.** Files into `gui/edit-history.md`, amends
   `gui/panel-layout.md`.
6. **Saving and point ids.** Save, Save As, Revert, the dirty marker, the
   lineage metadata; the session id form, the earliest rule, Go to Point over
   the version graph. Files `gui/saving.md`, amends `gui/goto-point.md` and
   the format spec.
7. **Edit families**, one PR each in Part 5's order, `gui/edits/`. The
   add-observation track edit, on `embedded_patches` files, is the first,
   since it is what the overlay is for.
8. **Wire surface.** Amends `gui/mcp-server.md`.

Steps 2 to 4 are the groundwork; 5 and 6 are independent of each other; 7
follows 4 and interleaves with 5 and 6.

## Non-goals

- Branching history. A new edit after an undo discards the redo tail.
- History persistence across sessions. A save writes the value, not the
  versions.
- Editing across nodes in one version.
- Undo of display state (eyes, tint, transform, layout). Those stay outside
  the history; bake-transform is the one bridge and it is an edit.

## Open questions

- The history memory budget's value (Part 2), after step 1's numbers, and
  whether a bulk edit's base shares the bitmap and thumbnail columns with its
  predecessor when it did not touch them, which the same numbers decide.
- The overlay draft's open questions (materialisation policy, how the
  version graph's maps are stored, when the point-set split lands).
