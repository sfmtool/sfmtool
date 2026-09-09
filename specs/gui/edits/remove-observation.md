# Remove Observation from Track

An action that takes one observation out of the selected point's track: a row of
the Point Track Detail panel's observation table, or the selected point's
sighting in the image the Image Detail panel is showing. The result is a version
of the node, so it is in the Edit History, it can be undone, and a save writes
it.

This is the edit that takes structure out of a track without taking the track's
point with it, and the one whose outcome depends on what is left: a shorter
track is re-triangulated, a track worn down to one sighting is a bearing, and a
track with nothing left is a point that goes.

Related specs: [`../document-model.md`](../document-model.md) (the version, the
two kinds of edit, and how a modification reaches the GPU),
[`../edit-history.md`](../edit-history.md) (the cursor and the per-step maps the
selection follows),
[`../../core/reconstruction/remove-observation.md`](../../core/reconstruction/remove-observation.md)
(the core function this wraps),
[`../point-track-detail.md`](../point-track-detail.md) (the table whose rows are
the observations),
[`../multi-panel-image-browser.md`](../multi-panel-image-browser.md) (the panel
the context menu lives in), [`add-observation.md`](add-observation.md) (the edit
this one inverts), and [`../saving.md`](../saving.md).

---

## Purpose

A wrong sighting is the thing a human eye finds fastest and a solver defends
longest: the Point Track Detail panel puts every observation of a point side by
side, with its reprojection error, its ray angle and the patch each view sees,
and one row that disagrees with the others is visible at a glance. What was
missing was the step from seeing it to taking it out.

Unlike the two edits beside it, this one is available whatever backs an
observation. Adding one has to invent a keypoint, which a `sift_files` value has
no room for; removing one names a row that is already there.

---

## Invocation

Two, one in each panel that shows an observation.

**The Point Track Detail table.** A secondary click on an observation row opens
a context menu with one entry, `Remove this observation`. The row is the
observation, so the entry needs no further selection and is never greyed: on a
track's only row it carries a hover text saying that removing it deletes the
point, because that outcome is worth stating before it happens rather than
after.

**The Image Detail context menu.** `Remove observation from track`, beside the
two entries that create structure. It removes the selected point's observation
in the image on screen. It is **greyed**, with a hover explanation, when no point
is selected and when the image on screen does not observe the selected point. It
is present on a `sift_files` node, where the other two entries are replaced by a
line saying they need an `embedded_patches` reconstruction; that line names only
those two, and this entry sits under it, enabled.

Nothing in either invocation names a pixel, so unlike add-observation neither
depends on where in the image the menu was opened.

No keyboard shortcut, and no Edit-menu entry. The menu carries the two edits
that act on the selection alone -- delete the selected point, delete the
selected image -- and this one acts on a pair, an observation being a point and
an image together; the two places that show such a pair are the two above.

---

## Mechanism

Everything below the invocation is
[`../../core/reconstruction/remove-observation.md`](../../core/reconstruction/remove-observation.md):
`sfmtool_core::remove_observation`, a pure function from the version's value plus
the point and the image to the next value and a report. The viewer adds the
invocation and the history entry, in
[state/edits.rs](../../../crates/sfm-explorer/src/state/edits.rs).

No images are decoded. The rays the re-triangulation needs are the poses and
lenses the value already carries, so this edit reads nothing off disk and its
cost is the size of the one track it touched.

### The version

A point edit. The value is the node's current one with the point deleted from
the base and re-added to the overlay with the sighting gone, so the base is the
same `Arc`.

The point takes a **new index**, because a modification is delete-and-re-add
([`../../core/reconstruction/edited-reconstruction.md`](../../core/reconstruction/edited-reconstruction.md)),
and it is the same point throughout. The version's map is therefore a
`PointMap::Replaced`, one pair saying which index became which. The selection
follows that map, so it stays on the point, and an undo brings it back to the
index it had.

When the removed sighting was the track's last, the point is deleted instead and
the map is a `PointMap::Removed` naming the index it held, which is the map
delete-point pushes: the selection clears, and an undo does not guess it back.

The version's label is

`Removed observation of point <index> in <image name> (<node label>)`

where `<index>` is the index the point held before the edit, which is the one the
user was looking at.

### The Action Log

One entry, of kind `Edit`, the label plus what is left of the track:

`Removed observation of point 42 in image_007.jpg (bull): 3 observations left
(v3 → v4)`

with `one observation left, so the point is a bearing at infinity` and `the point
had no other observation and is deleted` in place of the count for the other two
outcomes, because those are what a reader most needs told.

A refusal is one **failed** entry carrying the sentence the core function's
error writes, or the viewer's own for a menu gate that does not hold.

---

## What the viewport shows

The point moves to where the re-triangulation put it, or onto the direction
sphere when it becomes a bearing, or out of the picture when it is deleted. On
the GPU that is the overlay's **additions** for the first two -- the modified
point drawn from the additions buffers while the base instance it replaced is
masked -- and the mask alone for the third. See
[`../document-model.md`](../document-model.md), "Change detection by identity".

The Point Track Detail panel loses the row, the 3D viewport loses that track
ray, that image's frustum stops lighting as a track member, and the Image Detail
panel stops drawing the keypoint: every one of those reads the point through the
overlay accessor.

---

## Testing

Core (`sfmtool-core`, headless): the refusals, the three outcomes, the frame's
angular size across the crossing to infinity, and the round trip back through
add-observation. See
[`../../core/reconstruction/remove-observation.md`](../../core/reconstruction/remove-observation.md).

Bindings (`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real `embedded_patches` reconstruction, the refusals, the
modification materialising back into its place, a track removed row by row to a
deleted point, and the round trip -- a row taken out and put back at the keypoint
it held brings the point home.

Explorer (`sfm-explorer` lib tests, headless):

- `state/edits/tests.rs`: the three outcomes, each with what the version's map
  does to the selection and what the Action Log says; that an undo puts the
  observation and the position back; that a `sift_files` node runs the edit; and
  the menu gate, in the same call the entry asks -- no selected point, an image
  outside the track, and the pair that works.
- `point_track_detail/tests.rs`: a row's context menu reports its own image back
  to the dock, and an ordinary frame asks for nothing.

There is no windowed `ui_basic` test, for the reason the two edits beside it have
none: the gesture is a secondary click inside a panel whose rows and images carry
no stable accessibility node, so a windowed test would be asserting on pointer
geometry rather than on the edit. The gating and the effect are covered
headlessly above.

---

## Non-goals

- Moving an observation, or re-fitting the ones that remain.
- Removing several observations, or the same observation from several points, in
  one version.
- Re-solving a point at infinity that keeps two or more sightings.
- Bundle adjustment after the re-triangulation.
- Deciding which sighting is the wrong one. The panels show what there is to see,
  and the user names the row.
