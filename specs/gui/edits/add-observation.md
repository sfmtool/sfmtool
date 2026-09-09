# Add Observation to Track

An action in the Image Detail panel's context menu that adds one observation of
the selected 3D point to the image on screen, at the pixel the user pointed at.
The clicked pixel is a seed rather than the answer: the point's stored patch is
registered photometrically into that image by the kernel `sfm embed-patches`
places every observation with, and the track is re-triangulated with the new
sighting in it. The result is a version of the node, so it is in the Edit
History, it can be undone, and a save writes it.

This is the first edit that **creates** structure. Deleting a point takes an
index out of the value; this one puts a point back in with a longer track, which
is what the overlay's addition set and the additions instance buffer exist for.

Related specs: [`../document-model.md`](../document-model.md) (the version, the
two kinds of edit, and how an addition reaches the GPU),
[`../edit-history.md`](../edit-history.md) (the cursor and the per-step maps the
selection follows),
[`../../core/reconstruction/add-observation.md`](../../core/reconstruction/add-observation.md)
(the core function this wraps),
[`../multi-panel-image-browser.md`](../multi-panel-image-browser.md) (the panel
the menu lives in), [`../point-track-detail.md`](../point-track-detail.md) (where
the track being grown is read), and [`../saving.md`](../saving.md).

---

## Purpose

A photograph that plainly shows a point, and is missing from its track, is a gap
a human eye sees and a matcher did not close. The two panels that make it visible
are already side by side: the Point Track Detail panel lists the images that see
the point, and the Image Detail panel shows one that does not. What was missing
was the step from seeing the gap to closing it.

Closing it by hand would be worth little if the observation landed where the hand
pointed: a track's value is in its keypoints being placed to a fraction of a
pixel, and a click is worth two or three. So the click names the point's
neighbourhood and the photometric fit names the pixel.

---

## Invocation

Right-click inside the image in the **Image Detail** panel: `Add observation to
track here`.

The pixel is where the right-click landed, in source-image coordinates. It is
recorded on the frame the menu opens, because the menu's entries are laid out on
later frames, by which time the pointer has moved off the place the user named.

The entry is **greyed**, with a hover explanation, when no point is selected, or
when the image on screen already observes the selected point. It is **absent**,
replaced by a line saying why, on a `sift_files` reconstruction: there an
observation is a `.sift` feature index, and a clicked pixel is not one, so the
edit is not defined rather than merely unavailable.

A right **drag** in this panel is its zoom
([`../multi-panel-image-browser.md`](../multi-panel-image-browser.md)), so the
menu opens on a secondary *click* -- press and release inside egui's drag
threshold -- and a zoom gesture never puts a menu up.

The **Point Track Detail** panel carries a hint saying the same thing, under its
stored-patch tile, on an `embedded_patches` node: the track is what a user is
reading when they notice the gap, and the gesture happens in the other panel.
The hint quotes the menu entry's own label, from one constant, so the two cannot
drift.

No keyboard shortcut. The action names a pixel, and there is no keyboard way to
name one.

---

## Mechanism

Everything below the invocation is
[`../../core/reconstruction/add-observation.md`](../../core/reconstruction/add-observation.md):
`sfmtool_core::add_observation`, a pure function from the version's value plus
the point, the image, the pixel and the decoded views to the next value and a
report. The viewer adds the invocation, the images and the history entry, in
[state/edits.rs](../../../crates/sfm-explorer/src/state/edits.rs).

### The images

The photometric fit needs the photographs, and a reconstruction value carries
poses, lenses and thumbnails rather than pixels. The edit is an event rather
than a frame, so they are decoded on demand: the track's images and the clicked
one, through the node's existing full-resolution cache, built into pyramids for
the call. A handful of images per edit, and an image a panel has already shown is
not read twice. Nothing pre-decodes the image table.

The kernels index their view slice by image index, so the slice has one entry per
image of the node; the entries outside that handful are one-pixel placeholders,
which nothing samples. An image that cannot be read refuses the edit, naming it.

### The version

A point edit. The value is the node's current one with the point deleted from the
base and re-added to the overlay with the new sighting in its track, so the base
is the same `Arc` and the version costs the size of one track.

The point takes a **new index**, because a modification is delete-and-re-add
([`../../core/reconstruction/edited-reconstruction.md`](../../core/reconstruction/edited-reconstruction.md)),
and it is the same point throughout. The version's map is therefore a
`PointMap::Replaced`, one pair saying which index became which; every index not
named is unchanged. The selection follows that map, so it stays on the point,
and an undo brings it back to the index it had.

The version's label is

`Added observation of point <index> in <image name> (<node label>)`

where `<index>` is the index the point held before the edit, which is the one the
user was looking at.

### The Action Log

One entry, of kind `Edit`, the label plus what the fit found:

`Added observation of point 42 in image_007.jpg (bull): ZNCC 0.973, 1.84 px from
the click (v3 → v4)`

A refusal is one **failed** entry carrying the sentence the core function's error
writes, or the viewer's own for an image it could not read.

---

## What the viewport shows

The point moves to where the re-triangulation put it, and its patch with it. On
the GPU that is the overlay's **additions**: a second point instance buffer and a
second, small patch atlas, drawn after the base's in the same passes, while the
base's buffers keep their identity and the base instance the modification
replaced is masked. So the point is drawn once, the node's million base instances
are not re-uploaded, and an undo drops the additions and clears the mask. See
[`../document-model.md`](../document-model.md), "Change detection by identity",
[`../point-cloud-rendering.md`](../point-cloud-rendering.md) and
[`../patch-rendering.md`](../patch-rendering.md).

The Point Track Detail panel gains a row for the new image, the 3D viewport gains
a track ray to it, that image's frustum lights as a track member, and the Image
Detail panel draws the new keypoint in its embedded-features overlay: every one
of those reads the point through the overlay accessor.

---

## Testing

Core (`sfmtool-core`, headless): the refusals, the fit landing on a known truth,
the re-triangulation moving the point toward it, and the base left alone. See
[`../../core/reconstruction/add-observation.md`](../../core/reconstruction/add-observation.md).

Bindings (`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real `embedded_patches` reconstruction and the workspace's
own photographs, with the click placed at the point's projection; the refusals;
and that the returned value shares the base and materialises the modification
back into its place.

Explorer (`sfm-explorer` lib tests, headless):

- `state/edits/tests.rs`: the gate, in the same call the menu asks -- a
  `sift_files` node refuses, an image already in the track refuses, no selected
  point greys the entry, and an image outside the track offers it -- and a
  refusal leaves the history alone.
- `point_track_detail/tests.rs`: the panel prepares the version's track rather
  than the base's for a modified point, and takes the empty state for an index
  the version has deleted.
- `image_detail/tests.rs`: the embedded-features walk is over the version's live
  indexes, so a deleted point draws nothing and a modified one draws under its
  new index.
- `scene_renderer/upload/tests.rs`, against a real `wgpu` device on the `noop`
  backend: an addition uploads its own buffers and re-uploads none of the base's,
  it gets a slot in its own atlas keyed on its edited index, the pick range
  covers it and resolves back to it, an undo drops the buffers without touching
  the base, and a new base clears them.

There is no windowed `ui_basic` test. The gesture is a right-click inside an
image, and this panel's right **drag** is its zoom, so a windowed test of the
menu would be asserting on a pointer-gesture threshold rather than on the edit;
the accessibility tree also carries no stable node for a pixel inside an image.
The menu's gating and its effect are covered headlessly above.

---

## The point at infinity

A point created by [`create-point.md`](create-point.md) is a bearing with one
sighting, and the observation added here is what gives it a depth. The fit takes
two passes there: the click first triangulates the track provisionally, and the
photometric fit then runs against the finite patch that depth gives, because a
bearing projects into a second camera as the parallel ray rather than as the
place the surface is. The track is then re-triangulated from the fitted keypoint
and the angular patch frame rescaled at that final depth. The
viewer's side of it is unchanged -- the same menu entry, the same version shape,
the same one log line -- and
[`../../core/reconstruction/add-observation.md`](../../core/reconstruction/add-observation.md)
is where the mechanism is.

---

## Non-goals

- Moving or removing an observation a track already holds.
- Adding an observation on a `sift_files` reconstruction.
- Refitting the point's patch frame, normal or bitmap. Those are what the added
  observation is measured against.
- A keyboard path. The action names a pixel.
