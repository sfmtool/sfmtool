# Create 3D Point

An action in the Image Detail panel's context menu that creates a 3D point at
the pixel the user pointed at, with a one-observation track in the image on
screen. The point is created **at infinity**: one sighting fixes a bearing and
no distance, so what is stored is the pixel's own ray. Adding a second
observation to it, the edit next door
([`add-observation.md`](add-observation.md)), triangulates it into a place. The
result is a version of the node, so it is in the Edit History, it can be undone,
and a save writes it.

This is the first edit that creates a **point**. Add-observation grows a track
the reconstruction already holds; this one puts a point in the value that no
base has a row for, which is what the overlay's addition set, the additions
instance buffer and the point-edit hash exist for.

Related specs: [`../document-model.md`](../document-model.md) (the version, and
how an addition reaches the GPU),
[`../edit-history.md`](../edit-history.md) (the cursor, the per-step maps, and
the created-points list a version carries),
[`../../core/reconstruction/create-point.md`](../../core/reconstruction/create-point.md)
(the core function this wraps),
[`../goto-point.md`](../goto-point.md) (the id a point in no base is named by),
[`../multi-panel-image-browser.md`](../multi-panel-image-browser.md) (the panel
the menu lives in), [`../point-track-detail.md`](../point-track-detail.md), and
[`../saving.md`](../saving.md).

---

## Purpose

A reconstruction's points are what the matcher and the triangulation agreed on.
A person looking at a photograph beside the cloud can see something the cloud
does not hold -- a corner every matcher dropped, a mark on a wall the texture is
too flat around -- and until now had no way to put it there. The gesture is the
one the panel already has: point at the thing.

What a pointing hand cannot supply is a **distance**, so the edit does not
invent one. It stores what the click actually determines, a bearing, and leaves
the second sighting to fix the rest. That keeps the value honest at every step:
there is never a moment where the file holds a position nobody measured.

---

## Invocation

Right-click inside the image in the **Image Detail** panel: `Create 3D Point
here...`.

The pixel is where the right-click landed, in source-image coordinates, recorded
on the frame the menu opens, because the menu's entries are laid out on later
frames by which time the pointer has moved off the place the user named. A right
**drag** in this panel is its zoom, so the menu opens on a secondary *click*,
and a zoom gesture never puts a menu up
([`../multi-panel-image-browser.md`](../multi-panel-image-browser.md)).

The entry needs no selection and is never greyed: a pixel on the sensor is all
it takes. It is **absent**, replaced by a line saying why, on a `sift_files`
reconstruction, where an observation is a `.sift` feature index and a clicked
pixel is not one -- the same line the add-observation entry is replaced by, since
neither edit is defined there.

The ellipsis is a promise: the entry opens a prompt rather than running the edit,
because one thing the click cannot say has to be said first.

### The prompt, and the radius

A small popup opens at the click, holding a `Radius (px)` field and `Create` and
`Cancel` buttons. `Enter` commits, `Escape` cancels, and a click anywhere else
cancels: the prompt is a step in a gesture, not a window to leave open. While it
is up, the panel draws a **preview circle** of that radius at the clicked pixel,
in the image's own scale, updating as the field changes -- the radius is a size
in this photograph, and the only place it means anything is on the photograph.

The radius is the patch's half-extent in this image's pixels, and the core
function turns it into the angle those pixels subtend
([`../../core/reconstruction/create-point.md`](../../core/reconstruction/create-point.md)).
The value offered is:

1. **The radius the last point created in this session was given**, when there is
   one. Someone sizing one patch by hand is usually about to size the next one
   the same way.
2. Otherwise the **median pixel radius of the patches this image's own
   observations project to**: each observed point's patch frame projected into
   this image through the affine-shape accessor the feature overlay already
   draws its ellipses with, and the median of those pixel extents. That is the
   scale the reconstruction works at in this view, which is the scale a point
   added to it should be read at, and it comes out right whether the capture is
   a 480-pixel thumbnail or a 4000-pixel frame without anything having to be
   told which.
3. Otherwise the same median over **every** observation of the node, for an image
   that has none of its own.
4. Otherwise **8 px**, a constant named in the code, for a node that carries no
   patch frames at all.

The walk stops after a few hundred samples: a median of that many is the median.

No keyboard shortcut for the action itself. It names a pixel, and there is no
keyboard way to name one.

---

## Mechanism

Everything below the invocation is
[`../../core/reconstruction/create-point.md`](../../core/reconstruction/create-point.md):
`sfmtool_core::create_point`, a pure function from the version's value plus the
image, the pixel, the radius and the decoded views to the next value and a
report. The viewer adds the invocation, the prompt, the image and the history
entry, in [state/edits.rs](../../../crates/sfm-explorer/src/state/edits.rs).

### The image

The colour and the patch bitmap come out of the photograph, which a
reconstruction value does not carry, so the clicked image is decoded on demand
through the node's existing full-resolution cache and built into a pyramid for
the call -- one image per edit, and one a panel has already shown is not read
twice. The other entries of the view slice are one-pixel placeholders, which
nothing samples. An image that cannot be read refuses the edit, naming it.

### The version

A point edit. The value is the node's current one with a point appended to the
overlay, so the base is the same `Arc` and the version costs the size of one
record.

The point takes an index at or past the base's point count, and every index the
version already held is unchanged, so the version's map is a `PointMap::Created`
naming the one new index: the forward direction is the identity, and the inverse
has no answer for the created index. That is what makes an undo **clear** a
selection sitting on the point rather than carry it back to an index that held
nothing.

The version is pushed with `History::push_creating`, carrying a `CreatedPoints`:
the point edit's own content hash (`EditedReconstruction::point_edit_hash` over
the created record) and the index it took. A point no ancestor holds cannot be
named by a base's hash and a row in it, because there is no such row, so its id
is `pt3d_{edit hash}_{k}` -- the hash of the edit that made it, and its place
among that edit's creations ([`../goto-point.md`](../goto-point.md)). This is the
first edit to use that machinery.

The version's label is

`Created point in <image name> (<node label>), radius <r> px`

The radius is in the label because it is the one parameter of the edit that was
not read off the screen, and a history of several created points is otherwise
unreadable.

### The selection

The selection moves to the created point. It is what the user is now looking at
and what the next edit -- the second observation -- acts on, so leaving the
selection where it was would make the obvious next gesture wrong.

### The Action Log

One entry, of kind `Edit`, the label plus the version serials:

`Created point in image_007.jpg (bull), radius 11.5 px (v3 → v4)`

A refusal is one **failed** entry carrying the sentence the core function's
error writes, or the viewer's own for an image it could not read.

---

## What the viewport shows

The point appears in the 3D viewport as points at infinity appear
([`../point-cloud-rendering.md`](../point-cloud-rendering.md)) and its patch as
infinity patches do
([`../patch-rendering.md`](../patch-rendering.md)), because that is what it is.
On the GPU it is the overlay's **additions**: a second point instance buffer and
a second, small patch atlas, drawn after the base's in the same passes. Nothing
of the base is masked -- this point replaces no base instance -- and an undo
drops the additions. See [`../document-model.md`](../document-model.md),
"Change detection by identity".

The Point Track Detail panel shows it with its one observation and its
edit-hash id, the Image Detail panel draws its keypoint in the
embedded-features overlay, and that image's frustum lights as a track member:
each of those reads the point through the overlay accessor. The panel also
carries a hint saying how the gesture is made and that a second observation is
what places the point, beside the add-observation hint it already carries.

---

## Testing

Core (`sfmtool-core`, headless): the refusals, the bearing, the radius becoming
the frame's angular size, the columns, and the second observation that makes it
finite. See
[`../../core/reconstruction/create-point.md`](../../core/reconstruction/create-point.md).

Bindings (`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real `embedded_patches` reconstruction and the workspace's
own photographs -- the refusals, a created point that is a bearing with one
observation at the clicked pixel, and `w` crossing from 0 to 1 when a second
observation is added.

Explorer (`sfm-explorer` lib tests, headless):

- `state/edits/tests.rs`: the edit appends a bearing and selects it, over a node
  with a photograph cached rather than on disk; a `sift_files` node refuses it;
  the id it mints carries the edit's hash rather than a base's and resolves back
  to the point; an undo drops the point and the selection; the offered radius is
  derived from the node's own patches, the typed one is remembered after an
  edit, and a node with no patch frames falls back to the named constant.
- `state/save/tests.rs`: a save materialises the created point into the file's
  own base, the value at the cursor still holds it as a `w = 0` row, the written
  lineage carries the point edit's hash mapping its creation to that row, and the
  id minted before the save resolves after it.
- `scene_renderer/upload/tests.rs`, against a real `wgpu` device on the `noop`
  backend: a created point uploads its own buffers, re-uploads none of the base's,
  masks nothing, and gets a slot in its own atlas keyed on its edited index.

There is no windowed `ui_basic` test, for the reason add-observation has none:
the gesture is a right-click inside an image, this panel's right **drag** is its
zoom, and the accessibility tree carries no stable node for a pixel inside an
image. The prompt's own arithmetic and the edit's effect are covered headlessly
above.

---

## Non-goals

- Placing the point at a finite depth from the one click. A single ray does not
  determine one.
- Choosing the radius without being asked. The prompt offers a value; the person
  pointing decides.
- Creating a point on a `sift_files` reconstruction.
- Creating several points in one gesture. Each click is its own version, which is
  what makes each one undoable on its own.
- Refining the clicked pixel photometrically. There is no second view to register
  it against.
