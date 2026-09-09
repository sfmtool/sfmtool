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

The document model itself is built and standing:
[`../gui/document-model.md`](../gui/document-model.md) describes the node's
versions, the two kinds of edit and the identity-based upload, and
[`../gui/edit-history.md`](../gui/edit-history.md) the cursor, the maps and the
Edit History panel. What is left here is what is built on top of it: saving, the
edit families and the wire surface. Not decided: what the row-level edit draft
leaves open.

Related standing specs: [`../gui/scene-graph.md`](../gui/scene-graph.md)
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
([`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)): a
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

Under it, `SfmrReconstruction` is already an image table plus a point set
([`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)),
so the base's point side and the addition set are one type and per-point
algorithms take a point set rather than the whole, and the two heavy columns
are behind their own `Arc` inside those halves.

There is no copy-on-write anywhere in this. The base is one immutable value
behind one `Arc`, shared whole by every version in a run of point edits, and
the thing a point edit updates is the edit part of the version, which is its
own. A bulk edit produces a new base, a copy of the light columns; the patch
bitmaps and thumbnails, which are 88 % of the bytes, sit behind their own
`Arc` inside the point set and the image table, and a bulk edit that did not
touch them (every one but a patch refit) points at its input's. Nothing is
ever written through those `Arc`s, so it is sharing, not copy-on-write. The
history budget (Part 2) bounds how many bases a node holds; § "Step 1's
numbers" below has the measurements behind both decisions.

Filed:
[`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)
describes the value, the point edits, the overlay accessor, materialisation
and the hashes.

### Step 1's numbers

Measured on the largest real reconstruction to hand (530 674 points,
8 147 206 observations, 500 images, `embedded_patches` with 24 px bitmaps)
with [`scripts/measure_edit_costs.py`](../../scripts/measure_edit_costs.py),
medians of three:

| Quantity | Value |
|----------|-------|
| In-memory size | 1 354 MB |
| Of which patch bitmaps and thumbnails | 1 189 MB (88 %) |
| Full clone | 355 ms |
| Clone without the two heavy columns | 32 ms |
| Materialisation with 8 modified, 4 deleted, 2 new points (numpy, full re-sort of the tracks) | 1 800 ms, of which the re-sort 745 ms |
| XXH128 over every column | 131 ms |

The read-path census is
[`reports/2026-09-07-editing-read-path-census.md`](../../reports/2026-09-07-editing-read-path-census.md):
116 read sites, of which 53 run every frame, and **no per-frame read walks
the point set**. The eight per-frame "whole" reads are counts and the
content hash, answerable over an overlay in O(1). The ten real whole walks
are all event-driven: the point and patch uploads (gated on
`needs_upload`), the derived aggregates computed with them (auto point size,
camera scale, scene bounds), the embedded-features overlay of the Image
Detail panel, the MCP per-image observation counts, zoom-to-fit, and the two
core calls (align, resect).

What the numbers decide:

- **Bases share their heavy columns.** A bulk edit that copies the whole
  base costs 1.35 GB and 355 ms per version, so a 4 GB budget holds three;
  one that shares the bitmaps and thumbnails costs 165 MB and 32 ms, and the
  same budget holds twenty-four. Every bulk edit in Part 5 but a patch refit
  leaves those two columns untouched. So the point set holds them behind
  their own `Arc`, and a bulk edit's output points at its input's. This is
  not copy-on-write: nothing is ever written through those `Arc`s, and a
  refit produces new ones. The base itself stays one immutable value.
- **The materialisation is a merge, not a sort.** The base's tracks are
  already sorted, and the additions are a handful of tracks, so the
  materialised track column is one merge pass over the base (a copy with
  the deleted tracks skipped and the modified ones swapped in at their
  place) plus the appended new tracks, never a re-sort of eight million
  rows. Its cost is then the light-column copy, some tens of milliseconds,
  plus the bitmap copy unless the bitmaps are shared, which they are.
- **The hash is lazy and affordable**: 131 ms once per base, on first
  request.
- **The history budget** starts at 4 GB of unshared bytes per node, which
  the numbers above put at twenty-odd bulk edits or any number of point
  edits on the largest file; it is a setting, not a constant.

### Change detection by identity

Built and standing: [`../gui/document-model.md`](../gui/document-model.md),
"Change detection by identity". A node's GPU bundle remembers the base its
buffers were built from, the upload phase compares pointers, and the deleted
set reaches the point and patch shaders as a per-instance mask. What is still
to come is the **additions** buffer -- a second instance buffer drawn after the
base's with the same per-node uniforms -- and the patch atlas's slot
assignment, which is a compaction over the points that carry a bitmap and so is
not index-stable under an overlay that adds one. Neither is exercised until a
point edit adds a point, which is the track edits in Part 5.

### Node identity survives edits

Built and standing: [`../gui/document-model.md`](../gui/document-model.md) for
what the node holds, what a bulk edit owes the caches, and the rule that a
node's value changes only through its own history, and
[`../gui/edit-history.md`](../gui/edit-history.md) for the selection following
a step's map.

---

## Part 2: history

### Values with a cursor

Built and standing: [`../gui/edit-history.md`](../gui/edit-history.md) for the
cursor, the truncation and the maps, and
[`../gui/document-model.md`](../gui/document-model.md) for the budget, which is
a constant of 4 GiB of unshared bytes per node rather than a setting.

Not built: **coalescing**. An interactive edit that produces intermediate
values (a drag) is one version, committed when the gesture ends -- the rule the
Action Log applies to sliders, applied to the history instead of to text. No
edit produces intermediate values yet, so there is nothing to coalesce; the
first one that does, a pose drag in Part 5, is where it lands.

### The log and the history

Built and standing: [`../gui/edit-history.md`](../gui/edit-history.md), "The
Action Log", and the log spec's own non-goal.

### The Edit History panel

Built and standing: [`../gui/edit-history.md`](../gui/edit-history.md), "The
Edit History panel" for the tab and its rows and "Jumping to a version" for what a
click does, and [`../gui/panel-layout.md`](../gui/panel-layout.md) for the tab
itself. The row it marks as the disk state is the version the node was loaded
at, which is what a save moves.

---

## Part 3: saving

Built and standing: [`../gui/saving.md`](../gui/saving.md) for Save and Save As,
the materialisation a save of an edited value performs and the provenance and
lineage stamped onto it before it is hashed, the disk serial the Edit History
panel marks, the dirty marker in the tree row and the window title, and the
unsaved-changes prompt a close puts up. The lineage a save writes is the format
spec's ([`../formats/sfmr-file-format.md`](../formats/sfmr-file-format.md),
version 9), and the point ids it keeps resolving are
[`../gui/goto-point.md`](../gui/goto-point.md).

---

## Part 4: row-level edits under value semantics

Built and standing. The edit this part posed -- select a point, select an image
that is not in its track, right-click there and add a keypoint to the track -- is
[`../gui/edits/add-observation.md`](../gui/edits/add-observation.md), over the
core function in
[`../core/reconstruction/add-observation.md`](../core/reconstruction/add-observation.md).
The representation that makes it cost the size of one track rather than the size
of the reconstruction is
[`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md),
and what the viewer does with it on the GPU is
[`../gui/document-model.md`](../gui/document-model.md), "Change detection by
identity". What is left of the question is
[`sfm-explorer-editing-overlay.md`](sfm-explorer-editing-overlay.md): the
materialisation policy.

---

## Part 5: edits

Each edit family is a small standing spec in the shape of
[`../gui/resect-image.md`](../gui/resect-image.md): invocation, mechanism
pointing at the core function it wraps, what the version's label says, testing,
non-goals. The core function is always a pure function of a reconstruction
value plus named inputs, bound through `sfmtool-py` so the same edit is
available offline; the viewer adds only the invocation and the history entry.

Families, in the proposed order:

- **Delete**: selected points and an image (with its observations and any track
  left under two views) are built and standing with the document model,
  [`../gui/document-model.md`](../gui/document-model.md) § "Two kinds of edit";
  removing an observation from a track is standing as
  [`../gui/edits/remove-observation.md`](../gui/edits/remove-observation.md).
  The first structural edits, and the ones the selection-remapping rule is
  tested against.
- **Point constraints**: set a selected point free, ranged, or held, with the
  reference image and distance, from the Point Track Detail panel. The
  ground-truth workflow does this today in scripts against a CSV.
- **Bake transform**: apply the node's `Align to…` transform to the value and
  reset the transform to identity. This is the edit that breaks the scene-graph
  invariant, on purpose and only when asked.
- **Resect in place**: the existing resection applied to the node as a
  version rather than landing a derived node. The derived-node variant stays as
  the comparison affordance.
- **Track edits**: add an observation to a track from a pixel is built and
  standing, [`../gui/edits/add-observation.md`](../gui/edits/add-observation.md),
  and so is creating a point from a pixel,
  [`../gui/edits/create-point.md`](../gui/edits/create-point.md), which is the
  edit an added observation then places, and so is removing an observation,
  [`../gui/edits/remove-observation.md`](../gui/edits/remove-observation.md),
  which inverts the first. Split a track and merge two remain.
- **Bundle adjust**: run the adjustment on the node's value with the
  constraints it carries, as one version.

Files into: `specs/gui/edits/<family>.md`, one each, in the
[`../gui/edits/`](../gui/edits/README.md) directory the first family opened;
`resect-image.md` gains its in-place variant.

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

1. **Census and measurement.** Done: the census is
   [`reports/2026-09-07-editing-read-path-census.md`](../../reports/2026-09-07-editing-read-path-census.md),
   the script is
   [`scripts/measure_edit_costs.py`](../../scripts/measure_edit_costs.py),
   and the numbers and what they decided are in Part 1.
7. **Edit families**, one PR each in Part 5's order, `gui/edits/`. The
   add-observation track edit, on `embedded_patches` files, is done: it was
   first because it is what the overlay is for, and it is what routed every
   per-point read through the overlay accessor and put the additions on the
   GPU. Create a point from a pixel is done beside it, and is what exercises the
   rest of that machinery: a point in no base, `push_creating`, the point-edit
   hash an id is minted against, and an addition the GPU draws that replaces
   nothing. Removing an observation is done as well,
   [`../gui/edits/remove-observation.md`](../gui/edits/remove-observation.md),
   and is the edit that inverts add-observation, including the patch frame's
   crossing to and from infinity. Remaining: point constraints, bake transform,
   resect in place, the other track edits (split, merge), and bundle adjust.
8. **Wire surface.** Amends `gui/mcp-server.md`.

## Non-goals

- Branching history. A new edit after an undo discards the redo tail.
- History persistence across sessions. A save writes the value, not the
  versions.
- Editing across nodes in one version.
- Undo of display state (eyes, tint, transform, layout). Those stay outside
  the history; bake-transform is the one bridge and it is an edit.

## Open questions

- The overlay draft's open question: the materialisation policy.
- Finer sharing between bases than the two heavy columns. The track
  structure (keypoints, image and point indexes, observation confidence) is
  most of the light bytes and is untouched by every bulk edit but
  delete-image, so a bulk edit could point at its input's track columns too.
  Deliberately not decided until large reconstructions are being edited live
  and the history's memory can be measured on them.
