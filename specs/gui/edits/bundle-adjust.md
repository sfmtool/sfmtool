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
there is one decision to take, camera by camera: how much of each lens may move,
and in what form.

- **One row per camera** the posed images use, in table order: `Camera 0
  SFMTOOL_FISHEYE  24 images`, then two checkboxes, both clear by default. A
  camera only unposed images use is not in the solve and has no row. A lens that
  moves is a different claim about the capture than a pose that does, so the
  smaller claim is the default, and a camera left clear is held.
  - **Release focal length** releases that camera's focal. It is **disabled**
    when the camera's model is not one the adjustment's focal column is exact
    for, with a hover explanation naming the camera by its table index and its
    model.
  - **Release lens distortion** releases that camera's lens distortion: `k1` on
    `SIMPLE_RADIAL_FISHEYE`, the radial spline on `SFMTOOL_FISHEYE` and
    `SFMTOOL_PINHOLE`. It is **disabled** when the camera's model is none of
    those, with a hover explanation naming the camera and the three models to
    switch it to first. It is also disabled while the same row's **Release focal
    length** is clear, and cleared with it: neither `k1` nor the spline can
    change the scale at the centre of the image, which is the focal's job, so a
    camera's distortion is released only together with its own focal
    ([`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md)).

  So a rig that mixes an `OPENCV_FISHEYE` camera with spline cameras releases
  the spline cameras and holds the other, whose row is greyed.

  A spline camera's coefficient count and domain end are not in this dialog:
  they choose how the lens curve is parameterized, which the adjustment only
  refines, and changing them is a refit of that one camera, the Camera
  Intrinsics panel's `Refit spline…` ([`switch-camera-model.md`](switch-camera-model.md)),
  taken before the adjustment that fits the new coefficients to the
  observations.
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

The camera rows are the only option the dialog sets, as `releases`, which holds
one `CameraRelease` per camera of the node's table: each row's two checkboxes
for its camera, less anything its model cannot take, and held for a camera with
no row. The schedule, the iteration budget and the two floors are the core
function's defaults, which are the kernel's.

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

followed by what each camera of the solve released, read off the report:

- nothing, when every camera was held;
- one phrase when every camera released the same thing: `, focal released` or
  `, focal and lens distortion released` over one camera, with ` on every
  camera` appended over several;
- otherwise each camera by its table index, `, camera 0 focal and lens
  distortion released, camera 1 held`, because the label is the line in the
  Edit History that says which lens moved.

### The Action Log

One entry, of kind `Edit`, the label plus what the solve did:

`Bundle adjusted bull: 17 images, 4210 points, 19882 observations, median
residual 1.402 → 0.631 px (v3 → v4)`

with each camera's focal change appended when the focal was released, and
`, 12 points deleted` when the solve left points unsupported. A solve over one
camera reads `, focal 2803.5 → 2794.1`; one over several names each camera by
its table index, `, camera 0 focal 2803.5 → 2794.1, camera 1 focal 1401.2 →
1399.8`, because a list of numbers alone would not say which lens moved. The
three counts
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
  through two cameras adjusted rather than refused, one row per camera with its
  image count, each released camera named in the entry and the label saying
  `on every camera`, the focal gate naming the camera that cannot release its
  focal, and the distortion gate closed on each camera with no spline and open,
  with the label naming each camera's release, once a camera is a spline model
  or a `SIMPLE_RADIAL_FISHEYE`.
- `bundle_adjust_prompt/tests.rs`: the dialog's default (every camera held),
  one row per camera the posed images use, each row releasing its own camera
  and the rest held, a release the camera's model cannot take never answered,
  a row's distortion released only with its own focal and cleared when a drawn
  row's focal is clear, the refusal text naming the camera, the keys that run
  and cancel it, an ordinary frame answering nothing, and a second ask not
  stacking a second dialog.
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
