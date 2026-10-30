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
and in what form. One more is about the points, and one about which
observations the solve keeps.

- **One row per camera** the posed images use, in camera index order, on two
  lines: `Camera 0 SFMTOOL_FISHEYE  24 images` and the three shared releases on
  the first, and under them the four per-image releases and, while the rolling
  shutter is ticked, the **Scanlines** choice. Every checkbox is clear by
  default. A camera only unposed images use is not in the solve and has no row.
  A lens that moves is a different claim about the capture than a pose that
  does, so the smaller claim is the default, and a camera left clear is held.
  - **Release focal length** releases that camera's focal. It is **disabled**
    when the camera's model is not one the adjustment's focal column is exact
    for, with a hover explanation naming the camera by its camera index and its
    model.
  - **Release lens distortion** releases that camera's lens distortion: `k1` on
    `SIMPLE_RADIAL_FISHEYE`, the radial spline on `SFMTOOL_FISHEYE` and
    `SFMTOOL_PINHOLE`. It is **disabled** when the camera's model is none of
    those, with a hover explanation naming the camera and the three models to
    switch it to first. It is also disabled while the same row releases its
    focal neither shared nor per image, and cleared then: neither `k1` nor the
    spline can change the scale at the centre of the image, which is the focal's
    job, so a camera's distortion is released only together with its own focal
    ([`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md)).
  - **Release principal point** releases that camera's principal point, the
    pixel its optical axis passes through. It is **enabled on every model** and
    needs no focal: every model's projection adds the principal point after
    everything else, so the adjustment's columns for it are exact everywhere.
    Its hover text says what it solves, and that on a narrow field of view the
    data barely tell it from a turn of every image, so its change in the Action
    Log is worth reading.
  - **Release focal length per image** releases a focal length for each image
    of that camera, a small scale of the camera's that autofocus changes from
    one photograph to the next. It is **disabled** where **Release focal
    length** is, for the same reason and with the same hover text.
  - **Release principal point per image** releases a principal point for each
    image of that camera, a small offset of the camera's that a stabilised lens
    changes from one photograph to the next, on every model.
  - **Release radial distortion per image** releases each image's radial
    correction, two coefficients added to the camera's spline for that image
    ([`../../formats/sfmtool-camera-models.md`](../../formats/sfmtool-camera-models.md#per-image-radial-correction)).
    It is **disabled** on every model but `SFMTOOL_PINHOLE`, with a hover
    explanation naming the camera and its model and saying to switch it to
    `SFMTOOL_PINHOLE` first, and on an `SFMTOOL_PINHOLE` whose spline domain end
    is not positive and finite or whose spline has a single coefficient, with
    a hover explanation saying so
    (`image_radial_bspline_is_releasable` in
    [`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md)).
  - **Release rolling shutter per image** releases each image's
    rolling-shutter rates, how far the camera turned while its sensor read the
    image out. It is **disabled** on every model but `SFMTOOL_PINHOLE`, with a
    hover explanation naming the camera and its model
    (`image_rolling_shutter_is_releasable`).
  - **Scanlines**, a drop-down of `Rows`, `Columns` and `Unknown`, shown beside
    **Release rolling shutter per image** while it is ticked and named by its
    `Scanlines` label in the accessibility tree and `get_widgets`. It says
    whether the sensor reads one row or one column at a time, and is the
    camera's entry of the core function's `rolling_shutter_scanlines`
    ([`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md)
    § "The rolling-shutter scanlines"). It opens on the camera's stored
    `rolling_shutter_scanlines`, or on `Unknown` where the camera stores none,
    since an absent setting means the scanlines are not known and the dialog
    assumes none. `Unknown` solves the adjustment once with each setting and
    keeps the one with the lower cost, which takes about twice as long; its
    hover text says so. The setting the rates were solved on is written into
    the camera, so a second adjustment of the node opens on it. Clearing the
    rolling shutter hides the choice and keeps what it held.

  Each image is held toward the mean of its camera's images by a prior whose
  widths the solve derives, and the camera's own value moves to that mean
  ([`../../core/geometry/bundle-adjustment.md`](../../core/geometry/bundle-adjustment.md)
  § "The per-image release"), which each per-image focal and principal-point
  box's hover text says. For the radial correction and the rolling shutter the
  mean over the camera's images is held through each round instead, and the
  radial correction's then moved into the camera's spline, which their hover
  texts say, so neither stands for a shared release and neither greys one. So
  a ticked per-image focal or principal-point box shows the shared box of its
  kind **ticked and greyed**,
  its hover text saying that the per-image release moves the camera's value to
  the mean of its images'. What the user set in the shared box is kept, so
  clearing the per-image box gives it back as it was; the release the row
  answers is the per-image one alone, which the core function reads as the
  same as both.

  So a rig that mixes an `OPENCV_FISHEYE` camera with spline cameras releases
  the spline cameras and holds the other, whose focal and distortion
  checkboxes, shared and per image, and per-image radial and rolling-shutter
  checkboxes are greyed; its principal point can still be released, shared or
  per image.

  A spline camera's coefficient count and domain end are not in this dialog:
  they choose how the lens curve is parameterized, which the adjustment only
  refines, and changing them is a refit of that one camera, made from the
  Camera Intrinsics panel's `Switch Model…`
  ([`switch-camera-model.md`](switch-camera-model.md)), before the adjustment
  that fits the new coefficients to the observations.
- **Store each point as a position or a direction by its rays**, one checkbox
  under the camera rows, **ticked** by default: the core function's default
  `free_points`, under which every free point is solved in inverse depth and
  stored as the storage decision at the end of the solve says
  ([`../../core/geometry/bundle-adjustment.md`](../../core/geometry/bundle-adjustment.md)
  § "Free points: inverse depth and the storage decision"). Cleared, every point
  keeps the representation it has, `FreePointPolicy::NO_CROSS`. Its hover text says
  both.
- **Schedule**, a text box holding the trim schedule, one `(trim px, loss
  scale)` pair per round, with spaces or commas between the rounds. Before each
  round the adjustment leaves out every observation whose reprojection error is
  at or over that round's trim, and solves the rest under a soft-L1 loss of that
  round's scale
  ([`../../core/geometry/bundle-adjustment.md`](../../core/geometry/bundle-adjustment.md)).
  A trim of `inf` leaves out only the observations that do not project. Beside
  the box, a drop-down of presets fills it:
  - **Default (trims outliers)**, `(50, 5) (12, 2) (4, 1)`, the core function's
    `DEFAULT_SCHEDULE`, written out from that constant.
  - **Keep every observation**, `(inf, 1000)`: one round that leaves no
    observation out, under a loss scale wide enough that the solve is close to
    least squares. On a reconstruction whose bad observations have already been
    removed, the default's final 4 px trim drops the observations of distant
    points, which are the ones that hold the focal length against the cameras'
    distance from the scene; with them dropped, alternating Refit All Points
    and Bundle Adjust moves the points back and forth on every step. Under this
    preset the alternation settles.

  The drop-down shows the preset the box states, or `Custom`. The box is
  named by its `Schedule` label in the accessibility tree and `get_widgets`,
  and the drop-down, with no text beside it, is named `Schedule preset`. The box opens
  with the schedule last run on this node in this session, from the dialog or
  the wire, so a second adjustment of the node repeats the first unless it is
  changed; a node not adjusted this session opens with the default. A schedule
  the box does not state greys **Run**, with the reason as its hover text: text
  that is not a sequence of `(number, number)` pairs, and the core function's
  own refusals, `check_schedule` in
  [`../../core/reconstruction/bundle-adjust.md`](../../core/reconstruction/bundle-adjust.md)
  -- no rounds, a trim that is not a positive number or `inf`, and a loss scale
  that is not a positive finite number.
- **Run** and **Cancel**. `Enter` runs, `Escape` cancels, and clicking the
  window's close button cancels, because this is a step in a gesture rather than
  a window to leave lying open. `Enter` does nothing while **Run** is greyed.

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

The dialog sets four options, through `BundleAdjustAnswer::options` in
[bundle_adjust_prompt.rs](../../../crates/sfm-explorer/src/bundle_adjust_prompt.rs).
The camera rows are `releases`, which holds one `CameraRelease` per camera of
the node's image set: each row's seven checkboxes for its camera, less
anything its model cannot take and less a shared release its per-image box of
the same kind stands for, and held for a camera with no row. The Scanlines
choices are `rolling_shutter_scanlines`, one entry per camera of the image
set: the row's choice for a camera whose rolling shutter the row releases, and
`None` for every other camera. The prior on the per-image releases is the
core function's default, derived from the solve. The crossing checkbox is
`free_points`, and the Schedule box is `schedule`. The iteration budget and
the two floors are the core function's defaults, which are the kernel's.

The job hands these to `sfmtool_core::bundle_adjust` with its own `Progress`,
so under `Unknown` the operation's breakdown in the Action Log and the
Background Tasks panel shows both solves of the race, `solve with rows` and
`solve with columns`, and the core function's message stating the two costs
and the setting kept.

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

The image set does not move, so image indexes, the image and camera selections
and the decoded pixels keyed by them all still mean what they meant. What the
panels cached *about* the geometry -- rendered patches, prepared track rows,
per-camera derived quantities -- is dropped, because it describes a geometry the
node no longer holds.

The version's label is

`Bundle adjusted <node label>`

followed by what each camera of the solve released, read off the report, its
released parts named in the order focal, principal point, lens distortion:

- nothing, when every camera was held;
- one phrase when every camera released the same thing: `, focal released`,
  `, focal and principal point released`, `, focal, principal point and lens
  distortion released` or `, principal point released` over one camera, with
  ` on every camera` appended over several;
- otherwise each camera by its camera index, `, camera 0 focal and lens
  distortion released, camera 1 held`, because the label is the line in the
  Edit History that says which lens moved.

The parts a camera released per image follow as a phrase of their own, `,
focal and principal point released per image`, or after its shared parts, `,
lens distortion released, focal released per image`, in the order focal,
radial distortion, principal point, rolling shutter: `, focal, radial
distortion and rolling shutter released per image`. A part released per image
is not named again among the shared ones, though its shared value moves with it
and the Action Log states that change.

### The Action Log

One entry, of kind `Edit`, the label plus what the solve did:

`Bundle adjusted bull: 17 images, 4210 points, 19882 observations, median
residual 1.402 → 0.631 px (v3 → v4)`

with the schedule appended when it is not the default, `, schedule (inf,
1000)`, written as the dialog's box writes it, then each camera's focal change
appended when the focal was released, then each camera's principal-point change
when the principal point was released (a release per image of either moves the
camera's value, so its change is stated too), each per-image release's prior,
what the storage decision did, `, free points decided at 0.412 px: 3 to finite, 25 to
directions` (` before the final round converged` after the level when the final
round stopped on its iteration budget or settled with a distortion held at its
step guard, `, 2 not scored` after the counts when
the final round kept too few observations of some free points to score them,
`, free points not decided` when there was no level, and nothing with the
crossing off), and `, 12 points deleted`
when the solve left points unsupported. A solve over one
camera reads `, focal 2803.5 → 2794.1`; one over several names each camera by
its camera index, `, camera 0 focal 2803.5 → 2794.1, camera 1 focal 1401.2 →
1399.8`, because a list of numbers alone would not say which lens moved. The
principal point's change is written the same way, `, principal point (1020.0,
768.0) → (1013.1, 787.2)` over one camera and `, camera 1 principal point
(1020.0, 768.0) → (1013.1, 787.2)` over several. A camera that released a
per-image parameter adds the widths of the prior its final round ran under,
whether they were derived or fixed, and how many of its images ended more than
two widths from its mean, `, per-image widths focal 0.42 %, principal point
1.31 % of the short side (20.1 px), derived, 2 images beyond two widths`,
naming only the parameters it released per image, and `, camera 1 per-image
widths …` over several cameras. The focal's width is of the log focal scale
`ln(1 + s)` the solve holds, which reads as a percentage of the focal. The
radial correction's width is in units of the radius over the focal, which is a
fraction of the focal, with the pixels it is at the camera's focal, `radial
distortion 0.00120 of the focal (1.7 px)`, and the rolling-shutter rates' in
radians per short side of readout, `rolling shutter 0.00200 rad per short
side`. The derived widths are the solve's own estimate of how far the lens
moves between photographs, and the images beyond two widths are the ones to
look at. A camera that released its images' rolling-shutter rates then adds
the scanlines they were solved on, `, rolling shutter on rows`, and, where the
setting was `Unknown`, the race that chose it: `, rolling shutter on rows
chosen: cost 49350.0 with rows, 52349.0 with columns over 24931 observations,
both solves run to the end`, the costs being the core function's
`ScanlinesRace` and `both solves run to the end` its `stopped_early` being
false (`one solve stopped early` otherwise); `, camera 1 rolling shutter on
columns` over several cameras. The
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

Bindings (`tests/rust_bindings/reconstruction/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real reconstruction, and the refusals.

Explorer (`sfm-explorer` lib tests, headless):

- `state/edits/tests.rs`: the version pushed and its new base, the image set
  standing still, the perturbed camera coming back, the label with and without
  the focal, the log entry's counts and serials, an undo putting every pose back,
  the selection following the map, the gated refusal for no inline keypoints
  pushing no version and logging a failure, a node whose images are taken
  through two cameras adjusted rather than refused, one row per camera with its
  image count, each released camera named in the entry and the label saying
  `on every camera`, the focal gate naming the camera that cannot release its
  focal, and the distortion gate closed on each camera with no spline and open,
  with the label naming each camera's release, once a camera is a spline model
  or a `SIMPLE_RADIAL_FISHEYE`; the label naming each released part in the
  order focal, principal point, lens distortion, over one camera and several;
  the entry's principal-point clause over one camera and several, and none
  when no principal point was released; a principal point released on the
  demo's two-focal `PINHOLE`, named in the label and its change in the entry;
  the entry's free-point clause by default, and
  none with the crossing off; no schedule clause for the default, and for
  `(inf, 1000)` the clause naming it and the dialog reopened on the node
  holding it; the label naming the parts released per image, alone and after
  shared ones, over one camera and several; the per-image widths clause
  derived and fixed, with its count of images beyond two widths, and none
  without a per-image release; and the principal point released per image on
  the demo's two-focal `PINHOLE`, named in the label, the camera's change and
  the widths in the entry, and the camera marked as varying it; the label
  naming the corrections released per image in their order, the entry's
  widths of the radial correction and the rates, and its scanlines clause for
  a given setting, for a race with both costs, over several cameras and with
  no rates released; the correction gates greyed on the demo's `PINHOLE`
  naming the camera and its model, open on an `SFMTOOL_PINHOLE`, the radial
  one closed alone with a reason on a domain end that is not positive, and
  the stored scanlines read into the gate; and a run on the demo switched to
  `SFMTOOL_PINHOLE` releasing both corrections under `Unknown`, its label, the
  entry's both costs and that both solves ran to the end, the breakdown's
  `solve with rows` and `solve with columns`, and the camera marked as
  varying both with a setting stored, beside a run on `Columns` with no race.
- `bundle_adjust_prompt/tests.rs`: the dialog's default (every camera held
  and the points crossing), the crossing checkbox cleared answering
  `FreePointPolicy::NO_CROSS`,
  one row per camera the posed images use, each row releasing its own camera
  and the rest held, a release the camera's model cannot take never answered,
  a row's distortion released only with its own focal and cleared when a drawn
  row's focal is clear, the principal point answered on every model with or
  without the focal, the per-image releases answered without the shared one of
  their kind and the per-image focal refused where the shared one is, a
  per-image focal carrying the distortion, a drawn row keeping its distortion
  beside a per-image focal and leaving its shared boxes as the user set them,
  the Scanlines choice opening on the stored setting or on `Unknown`, the
  corrections answered with the scanlines of each camera that releases its
  rates and never on a camera that cannot take them, the options the answer
  runs with, a Scanlines choice answered only beside a released rolling
  shutter, a drawn row keeping its corrections and its Scanlines choice,
  the refusal text naming the camera, the keys that run
  and cancel it, an ordinary frame answering nothing, and a second ask not
  stacking a second dialog; the schedule box's text read and written for the
  default and each preset, spaces, commas and `inf` accepted, each malformed
  or out-of-range schedule refused with its reason, `Enter` answering the
  schedule the box states and doing nothing while it states none, and the box
  opening with the schedule last run on that node and the default on another.
- `mcp/tests/widgets.rs`: the dialog's camera row listing **Release focal
  length**, **Release lens distortion**, **Release focal length per image**,
  **Release radial distortion per image** and **Release rolling shutter per
  image** greyed, **Release principal point** and **Release principal point
  per image** enabled, and no **Scanlines** choice, on the demo's `PINHOLE`
  camera; and on the demo switched to `SFMTOOL_PINHOLE` the two corrections
  enabled, and a click on **Release rolling shutter per image** showing the
  **Scanlines** drop-down on `Unknown`.
- `mcp/tests/edit.rs`: the wire's `release_principal_point` and the per-camera
  field parsed; the per-image releases, their per-camera fields and
  `image_prior` parsed, a malformed prior refused at the parse, and a run
  releasing the principal point per image answering with its
  `image_adjustments`; the corrections' releases, `rolling_shutter_scanlines`
  call-wide and per camera and the corrections' prior widths parsed, an
  unknown scanlines name refused at the parse; runs on flat-spline
  `SFMTOOL_PINHOLE` cameras answering with the race's costs when the
  scanlines are omitted on a camera storing none, with the given setting and
  no race call-wide, per camera and from the stored setting, and the
  corrections' spreads and widths; and the corrections refused naming the
  camera on the demo's `PINHOLE`; the wire's `schedule` parsed with `null` as the
  unbounded trim, each malformed or out-of-range schedule refused at the
  parse, and a run under `[[null, 1000]]` naming it in the report and
  opening the dialog with it.
- `scene_graph/tests.rs`: the context-menu entry live on an adjustable node,
  directly above `Retriangulate All Points`, reporting the node it was opened
  on; and drawn but dead on a node with no inline keypoints and on a busy one.

The windowed `ui_basic` checks are that the entry is present on the
reconstruction row's context menu after a real right-click (Windows only), and,
over MCP on all three platforms, that it is there and greyed on the demo
reconstruction. What it does is covered headlessly.

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
- Choosing the per-image prior's widths from the dialog. They are derived from
  the solve; a caller that knows them states them on the wire or in Python.
- Choosing the iteration budget or the trim floors (`min_track`, `min_obs`)
  from the dialog. They are the core function's defaults, and a value that needs
  different ones needs a different tool.
- Reading the bench: the solve takes the reconstruction's tracks alone. Feeding
  it hand-verified bench tracks, exempt from trimming, is proposed in
  [`bench-inconsistent-fit-amendment.md`](../../drafts/bench-inconsistent-fit-amendment.md).
- Reporting progress, or a per-round trace. One entry, after it has run; live
  progress arrives with the background panel above.
