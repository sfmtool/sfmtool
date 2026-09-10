# Bundle Adjust

An action that refines the selected reconstruction as a whole: every pose and
every point move together until they agree with the observations as well as they
can. The result is a version of the node, so it is in the Edit History, it can be
undone, and a save writes it.

This is the edit that touches everything at once. The others move one point, one
observation or one image; this one is the reconstruction's own solver run over
the value on screen, and the reason it belongs in the viewer at all is that the
edits beside it change what the solver is looking at. Take a wrong observation
out, add one the matcher missed, re-pose an image that drifted, and the answer to
"is it better now" is what the adjustment says.

Related specs: [`../document-model.md`](../document-model.md) (the version, the
two kinds of edit, and what a bulk edit owes the caches),
[`../edit-history.md`](../edit-history.md) (the cursor, the Edit menu, and the
per-step maps the selection follows),
[`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md)
(the core function this wraps),
[`../../core/geometry/bundle-adjustment.md`](../../core/geometry/bundle-adjustment.md)
(the kernel under that), [`../resect-image.md`](../resect-image.md) (the edit
that re-poses one image rather than all of them), and
[`../saving.md`](../saving.md).

---

## Purpose

A reconstruction the viewer opens is the output of a solve that has already run,
and every edit made to it in the viewer leaves it slightly out of step with its
own evidence: a track with an observation removed is a point standing where a
sighting that is gone helped put it. Re-running the adjustment is how that is
settled, and doing it in the viewer rather than offline is what makes the
sequence -- look, edit, re-solve, look again -- one sitting instead of three.

---

## Invocation

The **Edit** menu, `Bundle Adjust...`, below the two delete entries and separated
from them: those act on a selection, this acts on the whole node.

It is **greyed**, with a hover explanation, when there is no selected
reconstruction, and when the selected one cannot be adjusted:

- its observations are `.sift` feature indexes with no inline keypoints, so
  there is no pixel to reproject against;
- its posed images are taken through more than one lens, and the adjustment
  solves one shared camera;
- none of its images carries a pose.

The gate is the edit's own, in
[bundle_adjust_prompt.rs](../../../crates/sfm-explorer/src/bundle_adjust_prompt.rs),
so the entry and the edit cannot disagree about when the adjustment can run.

No keyboard shortcut. It is an action with a dialog in front of it and a solve
behind it, not one a hand should be able to fire by accident.

### The dialog

`Bundle Adjust...` opens a small window rather than running immediately, because
there is one decision to take:

- **Release focal length**, a checkbox, clear by default. A focal that moves is a
  different claim about the capture than a pose that does, so the smaller claim
  is the default. It is **disabled**, with a hover explanation naming the camera
  model, where the adjustment's focal column is not exact for that model.
- **Run** and **Cancel**. `Enter` runs, `Escape` cancels, and clicking the
  window's close button cancels: the vocabulary of the Create 3D Point prompt,
  because this is a step in a gesture rather than a window to leave lying open.

Asking twice while it is up does not stack a second dialog; the question already
on screen is the one that gets answered.

---

## Mechanism

Everything below the dialog is
[`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md):
`sfmtool_core::bundle_adjust`, a pure function from the version's value plus the
options to the next value and a report. The viewer adds the invocation, the
version and the history entry, in
[state/edits.rs](../../../crates/sfm-explorer/src/state/edits.rs).

The checkbox is the only option the dialog sets. The schedule, the iteration
budget and the two floors are the core function's defaults, which are the
kernel's.

No images are decoded. The adjustment reprojects points through the poses and
the lens the value already carries, so this edit reads nothing off disk.

It runs **synchronously**, on the GUI thread, on the frame `Run` was pressed. The
window is unresponsive while it solves. Running it in the background is a
non-goal, below.

### The version

A bulk edit. The value is the whole new base the core function produced, with an
empty overlay; the current value is materialised first when its overlay is not
empty. The version's map is the chain of that materialisation's map and the
`RowMap::by_scan` read off the core call's input and output, because the
adjustment deletes the points it left unsupported and says how many rather than
which. The selection follows that map: it stays on the point it was on, and
clears if that point is one of the deleted.

The image table does not move, so image indexes, the image and camera selections
and the decoded pixels keyed by them all still mean what they meant. What the
panels cached *about* the geometry -- rendered patches, prepared track rows,
per-camera derived quantities -- is dropped, because it describes a geometry the
node no longer holds.

The version's label is

`Bundle adjusted <node label>`

with `, focal released` appended when the checkbox was ticked.

### The Action Log

One entry, of kind `Edit`, the label plus what the solve did:

`Bundle adjusted bull: 17 images, 4210 points, 19882 observations, median
residual 1.402 → 0.631 px (v3 → v4)`

with `, focal 2803.5 → 2794.1` appended when the focal was released and
`, 12 points deleted` when the solve left points unsupported. The three counts
are what went **into** the solve, which is not always the whole node: an unposed
image is not in it, and neither is a point nothing posed observes.

A refusal is one **failed** entry, `Bundle adjust of <node> refused: <reason>`,
carrying the sentence the core function's error writes, or the viewer's own for a
menu gate that does not hold.

---

## What the viewport shows

Every frustum and every point moves. On the GPU that is a new base, which the
upload phase notices by pointer and re-uploads whole
([`../document-model.md`](../document-model.md), "Change detection by
identity") -- unlike a point edit, there is nothing incremental about it.

---

## Testing

Core (`sfmtool-core`, headless): the convergence, the write-back, the deletions
and every refusal. See
[`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md).

Bindings (`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real reconstruction, and the refusals.

Explorer (`sfm-explorer` lib tests, headless):

- `state/edits/tests.rs`: the version pushed and its new base, the image table
  standing still, the perturbed camera coming back, the label with and without
  the focal, the log entry's counts and serials, an undo putting every pose back,
  the selection following the map, and both gated refusals -- no inline
  keypoints, and images that disagree about the lens -- pushing no version and
  logging a failure.
- `bundle_adjust_prompt/tests.rs`: the dialog's default (the focal held), the
  keys that run and cancel it, an ordinary frame answering nothing, and a second
  ask not stacking a second dialog.

There is no windowed `ui_basic` test beyond the Edit menu's own, for the reason
the edits beside it have none: what a windowed test could assert is that a
button exists, and that is what the menu test already does.

---

## Non-goals

- Running in the background, with the viewer live while it solves. The viewer's
  edits are synchronous, and an adjustment that could be interrupted, undone or
  edited over while it ran would be a second document model rather than a longer
  one.
- Releasing the distortion parameters. The dialog offers the focal and nothing
  else; a caller staging a distortion release runs the kernel offline.
- Adjusting a selection -- one image's pose, one region's points. The edit is the
  whole node.
- Choosing the schedule, the iteration budget or the trim floors from the dialog.
  They are the core function's defaults, and a value that needs different ones
  needs a different tool.
- Reporting progress, or a per-round trace. One entry, after it has run.
