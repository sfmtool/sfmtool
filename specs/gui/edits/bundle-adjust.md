# Bundle Adjust

An action that refines one reconstruction as a whole: every pose and
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
(the kernel under that), [`resect-image.md`](resect-image.md) (the edit
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

`Bundle Adjust...` on the **reconstruction row's context menu** in the Scene
Graph panel ([`../scene-graph.md`](../scene-graph.md)), first of the
whole-reconstruction edits and directly above `Retriangulate All Points`. It
acts on the node whose row was right-clicked, whatever the selection is: the
edit is of the whole reconstruction, and the row is where one is addressed. The
row reports the choice as `SceneGraphResponse::bundle_adjust`, and `dock.rs`
answers it with `AppState::open_bundle_adjust`, which puts up the dialog below.

It is **greyed**, with a hover explanation, while an operation is running on
that node, and when the node cannot be adjusted:

- its observations are `.sift` feature indexes with no inline keypoints, so
  there is no pixel to reproject against;
- none of its images carries a pose.

How many cameras the posed images are taken through is not a reason: the
adjustment solves each of them through its own lens.

The gate is the edit's own, in
[bundle_adjust_prompt.rs](../../../crates/sfm-explorer/src/bundle_adjust_prompt.rs),
so the entry and the edit cannot disagree about when the adjustment can run.

No keyboard shortcut. It is an action with a dialog in front of it and a solve
behind it, not one a hand should be able to fire by accident.

### The dialog

`Bundle Adjust...` opens a small window rather than running immediately, because
there is one decision to take, how much of the lens may move and in what form:

- **Release focal length**, a checkbox, clear by default. Ticked, it releases
  the focal of every camera the posed images use, each its own. A focal that
  moves is a different claim about the capture than a pose that does, so the
  smaller claim is the default. It is **disabled** unless every one of those
  cameras has a model the adjustment's focal column is exact for, with a hover
  explanation naming the first camera that does not, by its table index, and its
  model.
- **Release lens distortion**, a checkbox under it, clear by default. Ticked, it
  releases the lens distortion of every camera the posed images use whose model
  the adjustment can free it on, each its own: `k1` on `SIMPLE_RADIAL_FISHEYE`,
  the radial spline on `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`. Every other
  camera keeps its distortion where it is. It is **disabled** unless at least
  one of those cameras has such a model, with a hover explanation saying none
  does and naming the three models to switch a camera to first, and it is
  disabled while **Release focal length** is clear, and cleared with it: neither
  `k1` nor the spline can change the scale at the centre of the image, which is
  the focal's job, so the distortion is released only together with the focal
  ([`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md)).
- **Spline coefficients**, a row under it: a **Keep** checkbox, ticked by
  default, a count from 2 to 32, and `now 8` (or `now 6, 8` when the node's
  spline cameras differ) naming the counts the spline cameras of the posed
  images have. The count starts at the largest of them. Editing the count
  clears **Keep**. With **Keep** clear, every spline camera whose count differs
  is refitted to the count over its whole spline domain before the solve, which
  then fits the new coefficients to the observations; a count equal to the one
  every spline camera already has asks for nothing. The row is **disabled**,
  with a hover explanation, when no camera of the posed images is a spline
  model, and while **Release lens distortion** is clear: a new coefficient
  scheme only approximates the old curve until the solve fits it, so the core
  function refuses the count without the release.
- **Spline domain (°)**, a row under it, built the same way: **Keep**, ticked
  by default, the domain end in degrees (1 to 180), and `now 150.1°` (or several)
  naming the domain ends the node's spline cameras have; the value starts at
  the largest of them. With **Keep** clear, every spline camera whose domain
  differs is refitted on the new domain, in the same refit as the count. Under
  it is the **outermost keypoint** of the node's spline cameras
  ([`../../core/reconstruction/outermost-keypoint.md`](../../core/reconstruction/outermost-keypoint.md)),
  the one at the largest incidence angle over all of them: `outermost keypoint:
  230.3 px, 95.8° observed; 259.2 px, 108.8° detected`, and a **Use 108.8°**
  button that sets the domain to that angle and clears **Keep**. The button
  takes the detected keypoint, and the observed one where no `.sift` file could
  be read, in which case the text names only the observed one. The default
  domain is not changed: it stays the model's own reach, the far image corner,
  and a circular fisheye is trimmed to its image circle by choice. The row is
  disabled for the coefficient row's two reasons.

  The keypoints are read once, when the dialog opens, on the GUI thread: the
  observed one from the node's base value, and the detected one from the
  positions entry of each image's `.sift` file. On the `kerry_park` rig's 48
  images that is about 8 ms, short enough not to need a worker.
- **Run** and **Cancel**. `Enter` runs, `Escape` cancels, and clicking the
  window's close button cancels, because this is a step in a gesture rather than
  a window to leave lying open.

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

The two checkboxes and the two spline rows are the only options the dialog
sets, as `opt_f`, `opt_distortion`, `spline_coeff_count` and
`spline_domain_deg`. The schedule, the iteration
budget and the two floors are the core function's defaults, which are the
kernel's.

No images are decoded. The adjustment reprojects points through the poses and
the lens the value already carries, so this edit reads nothing off disk.

It runs **on a worker thread**, so the window stays usable while it solves and
the node it is running on refuses every edit until it lands
([../background-tasks.md](../background-tasks.md)). The frame that
started it goes on drawing the value at the cursor, which is the value the
worker was handed and which nothing is mutating.

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

with `, focal released` appended when the focal was released, or `, focal and
lens distortion released` when a camera's distortion was released too, and then
`, spline refitted to 12 coefficients` when a spline camera's coefficient count
or domain was changed before the solve, with ` on a 108.8° domain` when the
domain moved.

### The Action Log

One entry, of kind `Edit`, the label plus what the solve did:

`Bundle adjusted bull: 17 images, 4210 points, 19882 observations, median
residual 1.402 → 0.631 px (v3 → v4)`

with each camera's focal change appended when the focal was released, and
`, 12 points deleted` when the solve left points unsupported. A solve over one
camera reads `, focal 2803.5 → 2794.1`; one over several names each camera by
its table index, `, camera 0 focal 2803.5 → 2794.1, camera 1 focal 1401.2 →
1399.8`, because a list of numbers alone would not say which lens moved. A
spline refitted to a new count or domain before the solve adds `, spline 8 → 12
coefficients, domain 150.1° → 108.8° (refit max 0.013 px)` after the focal
clause (the domain only when it moved), with the same
`camera N ` prefix when the solve holds several cameras: the refit's largest
pixel distance from the old curve says how much of the lens change was the
refit rather than the solve. Where the refit's monotonicity constraint bound, the
parenthesis adds where, `(refit max 1.519 px, monotone constraint bound at 1
angle, 113.2°)`, since over that range the refit is the closest invertible curve
rather than the old one. The three counts
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
  the selection following the map, the gated refusal for no inline keypoints
  pushing no version and logging a failure, a node whose images are taken
  through two cameras adjusted rather than refused, each released camera named
  in the entry, the focal gate naming the first camera that cannot release its
  focal, and the distortion gate closed on a node with no spline and open, with
  the label naming the release, once a camera is a spline model or a
  `SIMPLE_RADIAL_FISHEYE`, and the spline counts the coefficients row shows.
- `bundle_adjust_prompt/tests.rs`: the dialog's default (the focal held), the
  distortion released only with the focal, the keys that run and cancel it, an
  ordinary frame answering nothing, a second ask not stacking a second dialog,
  and the coefficient count: kept by default and starting at the largest count,
  asked for only when it changes some camera's count, and never without the
  distortion release or a spline camera; the domain under the same rules; the
  outermost keypoint's button taking the detected angle, the observed one
  without a detected, and nothing without either; and the keypoint text
  labelled by its source.
- `scene_graph/tests.rs`: the context-menu entry live on an adjustable node,
  directly above `Retriangulate All Points`, reporting the node it was opened
  on; and drawn but dead on a node with no inline keypoints and on a busy one.

The one windowed `ui_basic` check is that the entry is present on the
reconstruction row's context menu after a real right-click (Windows only), for
the reason the edits beside it have no more: what a windowed test could assert
is that a button exists.

---

## Non-goals

- Editing the node while the adjustment runs. An adjustment that could be
  undone or edited over while it ran would be a second document model rather
  than a longer one, so the busy node refuses every edit until it lands
  ([../background-tasks.md](../background-tasks.md)).
- Releasing the distortion of the multi-coefficient models (`RADIAL`,
  `OPENCV_FISHEYE`, …). The adjustment has no exact rung for them; a camera is
  switched to a spline model first.
- Adjusting a selection -- one image's pose, one region's points. The edit is the
  whole node.
- Choosing the schedule, the iteration budget or the trim floors from the dialog.
- Reading the bench: the solve takes the reconstruction's tracks alone. Feeding
  it hand-verified bench tracks, exempt from trimming, is proposed in
  [`bench-inconsistent-fit-amendment.md`](../../drafts/bench-inconsistent-fit-amendment.md).
  They are the core function's defaults, and a value that needs different ones
  needs a different tool.
- Reporting progress, or a per-round trace. One entry, after it has run; live
  progress arrives with the background panel above.
