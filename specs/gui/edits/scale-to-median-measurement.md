# Scale to Median Measurement

An action in the Scene Graph panel that gives a reconstruction a physical
size. A structure-from-motion solve has no real-world scale, but a
reconstruction that stores distance measurements, each a real length stated
between two places in its photographs, says how long one of its units is. This
action asks for a physical unit (millimetres, centimetres, metres, inches or
feet), shows in a dialog how well the measurements agree on the size of a scene
unit in it, and on `Apply` multiplies every coordinate by the median of their
factors and declares the unit, as one version: listed in the Edit History
panel, stepped back by `Ctrl+Z`, and written by a save. The 3D view is not
touched, so the scene is drawn larger or smaller from where the camera stands.

Related specs: [../../core/reconstruction/measurement-scale.md](../../core/reconstruction/measurement-scale.md)
(the factor and the statistics, `distance_measurement_scale`),
[../measurement-view.md](../measurement-view.md) (the measurements, their
readings and the per-measurement median line), [bake-transform.md](bake-transform.md)
(the closest edit, whose mechanism this shares), [bundle-adjust.md](bundle-adjust.md)
(the closest dialog), [../scene-graph.md](../scene-graph.md) § "The transform"
(the display transform, which this edit leaves as it is),
[../document-model.md](../document-model.md) and
[../edit-history.md](../edit-history.md) (the version it pushes),
[../../cli/reconstruction/xform/scale-to-median-measurement-command.md](../../cli/reconstruction/xform/scale-to-median-measurement-command.md)
(the same median scale at the command line, as an `sfm xform` step),
[README.md](README.md) (the other edit families).

---

## Invocation

`Scale to Median Measurement...` on the **reconstruction row's context menu**
in the Scene Graph panel, directly under `Bake Transform`, in the separator
group of `Align to ▸`, `Reset Transform` and `Bake Transform`. The four are the
entries that set where and how large a reconstruction is: compute a transform,
discard it, keep it, and here give it a physical size. Like the bake, this one
writes a similarity into the reconstruction's own numbers as one version;
unlike the bake, it moves the picture, since it leaves the view where it is.
Its spelling follows the menu's other entry that opens a dialog, `Bundle
Adjust...`.

It acts on the node whose row was right-clicked, whatever the selection is,
because the edit is of the whole reconstruction and the row is where one is
addressed. The row reports the choice as
`SceneGraphResponse::scale_to_median_measurement`, and `dock.rs` answers it with
`AppState::open_measurement_scale`, which puts up the dialog below.

The entry is **greyed**, with the reason as its hover text, in this order:

- while a background task holds the node, on the busy sentence every sibling
  entry that makes a version carries;
- while a camera model proposal is open on the node, on *"Apply or cancel the
  camera model proposal first."*
  ([switch-camera-model.md](switch-camera-model.md));
- when the version on screen stores no distance measurement, on core's
  sentence: *"The reconstruction stores no distance measurement, so no stated
  length gives it a scale."*;
- when no stored measurement gives a factor in any of the five units, on core's
  sentence naming each measurement and why: *"None of the 2 distance
  measurements gives a scale; measurement 0: endpoint B has no position: …"*.

The last two are `measurement_scale_prompt::refusal`, which the edit and the
wire ask too, so the entry, the dialog's `Apply` and a call that asks anyway
give one answer. Whether a measurement gives a factor hardly depends on the
unit: only a factor that overflows or underflows in one unit and not another
does, so the entry is live when any unit gives a scale, and the dialog says
which do not.

No keyboard shortcut, for the reason `Bundle Adjust...` has none: an action
with a dialog in front of it is not one a hand should fire by accident.

---

## The dialog

A window titled *Scale run_b to Median Measurement*, which names the
reconstruction, drawn by
[measurement_scale_prompt.rs](../../../crates/sfm-explorer/src/measurement_scale_prompt.rs)
from `app/modals.rs`. From the top:

- **Unit**, a drop-down of `mm`, `cm`, `m`, `in` and `ft`
  (`LengthUnit::ALL`). It opens on the file's `world_space_unit` when it
  declares one, which is always one of the five, since a load refuses any
  other. With none it opens on the unit most of the usable measurements state
  their distance in, a tie going to the first of the five in that order
  (`measurement_scale_prompt::default_unit`); a measurement that gives no
  factor does not count. Changing the unit recomputes everything below on the
  same frame.
- **The factor**, in words and as the multiplier: *1 scene unit = 2.716 mm*
  and *Multiplies every coordinate by 2.716*. Where the file declares a unit, a
  third line compares the measurements with it: *The file declares m. By these
  measurements its scale is off by −0.20% (×0.998).* The ratio is the factor
  times the length of the chosen unit over the length of the file's, `1` for
  measurements that agree with the file exactly, and it is given as a
  percentage first. It is not the multiplier the scale applies, which is the
  line above: a file in feet whose measurements agree with it, scaled to
  metres, multiplies every coordinate by 0.3048 and is off by +0.00% (×1.000).
  Where the file declares no unit, the line says the coordinates are in scene
  units.
- **A histogram** of each usable measurement's signed deviation from the
  median, in percent, with the median as a solid line at `0` and the
  disagreement bar, `SCALE_DISAGREEMENT_PERCENT` (5%) either side, as dashed
  lines (below).
- **How well they agree**: *Used 3 of 5 distance measurements*; the median and
  the mean; the sample standard deviation and its percentage of the median, or
  *none for a single measurement*; the median absolute deviation as a
  percentage of the median, to two decimals, so measurements that agree to the
  `float32` they are stored in read `0.00%` rather than a factor's rounding
  noise; the smallest and the largest factor; and the largest deviation,
  signed, naming its measurement. Where that deviation exceeds the bar, a warning in the warning
  colour says how many measurements are past it and asks for their endpoints
  to be checked before applying.
- **A table**, one row per usable measurement: its title and label
  (*Distance #4  paw width*), its stated distance in its own unit, its distance
  in scene units (`1.500 su`), its factor, and its signed deviation from the
  median. A row more than 5% off is drawn in the warning colour. The table
  scrolls past 220 points.
- **Left out**: each measurement that gives no factor, with core's reason as a
  sentence (*Distance #2  bare pixel: Endpoint B has no position: no observation's square
  in image 0 holds pixel (5, 5); use Create Track Here …*).
- **Apply** and **Cancel**. `Enter` applies and `Escape` cancels, as in the
  Bundle Adjust dialog; `Enter` does nothing while `Apply` is greyed. `Apply`
  is greyed, with the reason as its hover text, while the node is busy or a
  proposal is open on it, and when the scale in the chosen unit is an error.

Where the scale is an error, the dialog shows the error's sentence in place of
the factor, and under it the measurements left out.

Numbers are printed to four significant digits, rounded first so a value just
under a power of ten prints as `1.000` rather than `0.9999`, and in exponent
notation outside `1e-4` to `1e7` (`measurement_scale_prompt::number_text`).

### Binning

The histogram reuses the Camera Intrinsics panel's painter
([histogram.rs](../../../crates/sfm-explorer/src/intrinsics_detail/histogram.rs))
and its rule for the width of a bin, the painter generalised so a bin
names its items (*measurements* rather than *images*) and so a bin that selects
nothing senses only the hover. It fits because the two ask the same question,
how one quantity spreads over a set, with a centre and two bounds to mark.

What is binned is the deviation from the median, not the factor itself, so the
axis reads the same in every unit and the bar is the same two lines at ±5. A bin
is as wide as that histogram's rule makes it: the step of the 1/2/5 ladder
nearest the values' spread over about `⌈√n⌉` bins, at least 2 and at most 20,
or one percent where every value is the same.

**The bins are centred on the median.** Their edges are at `±w/2, ±3w/2, …`
for a width `w`, not at multiples of `w` as the Camera Intrinsics panel's are
(`measurement_scale_prompt::deviation_histogram` builds them rather than
`Histogram::new`). A deviation of exactly `0` is then in a bar symmetric about
the median line; with edges at multiples of `w`, measurements that agree
exactly would fill a bar from `0` to `w` and read as a spread of up to `w`. A
single measurement, or measurements that agree, make one bin one percent wide
centred on the median. A bin's name and tooltip print its edges with the fewest
decimals that print every edge exactly, and never more than one past the count
the width alone gives (`histogram::edge_decimals`), which for
these half-width edges is one more digit than the width alone gives where the
width starts with 1 or 5: the single bin reads `−0.5 to 0.5 %`, not
`0 to 0 %`. For the Camera Intrinsics panel's edges, at multiples of the width,
it is the width's own count, so those labels read as before. A camera with one
image, whose one bin's edges are half a width either side of an arbitrary value
that no count prints exactly, gets the cap, one digit more than before
(`1134.6 to 1334.6`).

**The axis is symmetric about the median.** It reaches the largest deviation
on either side, and the ±5 bounds where those lie within a spread of the
values past it; short of them, it reaches one spread past the largest
deviation, as the Camera Intrinsics panel's reaches toward its bounds. It
reaches as far below the median as above it, so the median line is in the
middle of the axis even where every deviation is at or above it.

Each deviation is rounded to a thousandth of a percent before it is binned. A
stated distance is stored as a `float32`, so measurements that agree exactly
still deviate by about `1e-6` percent, which unrounded would be a spread of a
millionth of a percent and bins to match.

### Keeping it current

The dialog remembers only the node and the unit. Every frame it reads the node
as the scene holds it, its measurements and its cached readings
(`SceneNode::measurement_readings`, keyed by the version's document serial), so
an undo, a redo or an edit made while it is up, such as a committed measurement,
is shown on the next frame. A node that is no longer loaded closes it. `Apply`
carries only the node and the unit, and the edit reads the readings again, so
nothing the dialog computed is carried into the version.

This differs from the Bundle Adjust dialog, which reads its gates once when it
opens and lets the edit refuse what has changed since. That dialog's controls
are choices the user makes over the gates; this one's body is a report, and a
report of a version no longer on screen would be wrong, at a cost of one median
over the readings per frame.

Asking twice while it is up does not stack a second dialog; the question already
on screen is the one that gets answered.

---

## Mechanism

```rust
impl AppState {
    /// Scale `id` to `unit` by the median measurement factor, as one version.
    pub(crate) fn scale_node_to_median_measurement(
        &mut self,
        id: ReconId,
        unit: LengthUnit,
    ) -> Result<(), String>;
}
```

In [display_transform.rs](../../../crates/sfm-explorer/src/display_transform.rs),
beside the bake, whose push it shares (`AppState::push_similarity`). The dialog
and the wire both call it. A **bulk edit**:

1. Refuse a busy node, a node a camera model proposal is open on, a node that
   is no longer loaded, the menu entry's gate, and a scale that is an error in
   `unit`, each in the words the greyed entry or `Apply` carries; and a scale
   that would carry the node's bounds, in its own coordinates or where its
   display transform draws them, to where their squared distance is not a
   finite number, in the words the transform's own range check uses.
2. Compute the scale with `distance_measurement_scale` over the measurements of
   the version on screen and its cached readings
   (`measurement_scale_prompt::node_scale`), and take its median as `s`.
3. Fold the overlay in when there is one, as the bake does, and run
   `SfmrReconstruction::apply_se3_transform` with `S`, the uniform scale `s`
   with the identity rotation and no translation; set
   `metadata.world_space_unit` to the unit's name; and put the bench's patch
   frames through `S` with `display_transform::bake_bench`.
4. Push one version stating every half through `History::push_pair`: the scaled
   value, the scaled bench, and the display transform the node already has,
   unchanged.

`apply_se3_transform` is the one whole-reconstruction similarity, what `sfm
xform --scale-to-median-measurement` reaches too: point positions, camera centres,
rig sensor translations, patch half-vectors and the per-image depth statistics
all scale, and points at infinity, which are directions, do not. The row map is
the identity, chained after the materialisation's when there was one, for the
reason the bake's is. A measurement names images and pixels, which a scale does
not move, so the stored measurements pass through unchanged and their readings
are the same points `s` times as far apart.

Everything that caches what the scale changes is keyed on what the version
changes. The node's measurement readings and a measurement item's live readings
are keyed by the document serial, so they are resolved again for the new
version and found again for the old one after an undo. Measurement View, the
Scene tree's Measurements rows and the wire read those. The 3D view's and Image
Detail's measurement layers are keyed by the document serial and the display
transform, so they are built again. The GPU bundle sees a new base by pointer
and re-uploads it, as after the bake. The panels' own caches (`forget_recon`)
are dropped after the edit, by `app/modals.rs` and by the MCP drain
(`Command::renumbers`), as after every bulk edit.

---

## The view

The edit does not touch the 3D view. The viewport camera stays where it is,
with its orientation, its orbit distance and its field of view, and the node's
display transform is the one it had, so the scene is drawn `s` times larger
about the transform's origin, from the same camera. A factor near 1000, as for
a reconstruction in metres scaled to millimetres, puts the camera deep inside
the scene; `Z` frames it again, and so does a double-click on the row.

What the viewer does with any new version, it does with this one, and nothing
more:

- The scene's bounds are measured again from the uploaded value, and the near
  plane follows them over the next frames (`ViewportCamera::update_clip_planes`
  in `app.rs`), as it does every frame.
- The scene scale the frustums and the target indicator are drawn at
  (`AppState::length_scale`) is not measured again. It is re-seeded only when a
  node arrives or a display transform changes, and this edit does neither, so
  the frustums keep their size in the world, small against a scene `s` times
  larger. The HUD's *Scene scale* slider sets it.
- Camera view stays on its image but is not snapped to the image's scaled pose:
  only a move of the cursor (undo, redo, a jump) snaps it again
  (`camera_lock::resnap_camera_view`). Until then the viewport is at the
  image's old centre, not its scaled one, so the photograph drawn behind the
  scene no longer lines up with the points drawn over it, though both keep the
  image's orientation. Leaving camera view and entering it again, or stepping
  to another image and back, looks through the scaled camera, where the two
  line up again.

A camera held in hand for Move Camera on the node is ended before the edit, as
a commit where it has moved, as any other edit of the node ends it: its pending
pose is in the old scale.

**Undo** puts back the value and the bench, and the transform half, which the
edit left as it was. The view is left where the edit left it, as every undo
leaves it, so the scene is drawn at its old size again from the same camera.

---

## The version

Recorded as `Kind::Edit` with `ActionLog::record_done`, from the instant the
operation began, in the vocabulary the bake's row uses, naming the node, the
unit, how many measurements gave the median, and the factor, and ending in the
version transition every edit's row carries:

```
Scaled run_b to mm by the median of 4 distance measurements: x2716 (v3 → v4)
```

The factor is printed as the dialog prints it. The version's label, which the
Edit History panel lists, is the same sentence without the transition. A
refusal is an `ActionLog::fail` with the reason.

The node goes dirty, since the value is a new base with a new unit; a save
writes both. The display transform is not written: it never is.

---

## MCP

`get_measurement_scale` reads what the dialog shows, in a unit or in the one
the dialog opens on, and `scale_reconstruction_to_median_measurement` is
`Apply`, through the same `AppState` call, so it leaves the view alone the same
way and refuses in the same words
([../mcp-server.md](../mcp-server.md)). No tool opens the dialog, as none opens
the Bundle Adjust dialog: a right-click on the row through `click` and a click
on the entry put it up for a screenshot.

---

## Testing

`sfm-explorer` lib tests, headless, on the measurement fixture
([measurements/fixture.rs](../../../crates/sfm-explorer/src/measurements/fixture.rs)),
whose three usable measurements imply 1.02 m, 1.02 m and 1.2 m per scene unit,
with one endpoint refused and one at infinity:

- **The gate** (`measurement_scale_prompt/tests.rs`): a node with usable
  measurements is not refused; one with none, and one whose only measurements
  are the refused and the infinite one, are refused in core's sentences.
- **The default unit**: the file's own where it declares one; with none, `m`,
  the unit of two of the three usable measurements; and where the usable ones
  are restated in `ft`, `in` and `cm`, one each, while the two unusable ones
  stay in `m`, `cm`, the first of the tie, which shows that unusable rows do not
  count.
- **The dialog's numbers**: in millimetres the factor is 1020, the mean 1080,
  the paw width's deviation +17.65%, measurements 2 and 3 are left out, the
  correction line says the file's scale is off by +2.00% (×1.020), the median
  absolute deviation reads 0.00% and the measurements disagree; a file in feet
  whose measurements agree with it, read in metres, is off by +0.00% (×1.000)
  while the multiplier is 0.3048, and a median absolute deviation of float
  noise reads 0.00; a whole frame of the
  dialog draws the title, the factor, the correction, the table's rows and the
  reasons left out, and says so when the file declares no unit.
- **The histogram**: one measurement, and two that agree, make one bin; with
  four measurements that agree, one 17.65% long and one left out, the four are
  in a bin centred on 0 and the axis is symmetric about it; and with every
  deviation at or above the median the axis is still symmetric; and a
  single measurement's bin is named `deviation from the median −0.5 to 0.5 %:
  1 measurement`.
- **The dialog's lifetime**: a node no longer loaded closes it, and a dialog
  left open draws the scaled version's factor after an edit and the original's
  after the undo.
- **The edit** (`display_transform/tests.rs`): every finite point and camera
  centre is multiplied by the factor, the unit becomes `mm`, the node goes
  dirty, and the measurements then give a factor of 1; the bench's patch frame
  is scaled; a node under a display transform keeps it bit for bit, and every
  point and camera centre it draws is the factor times as far from the
  transform's translation; an undo puts back the base,
  the unit, the transform bit for bit and the bench; the Action Log row names
  the node, the unit, the count and the factor; and a node with no measurement
  and a busy one are refused, the first in the menu's words, with no version.
- **The menu entry** (`scene_graph/tests.rs`): live on the fixture, directly
  under `Bake Transform`, reporting the node it was opened on while another is
  selected; greyed with its reason as hover text with no measurement, with no
  usable measurement, on a busy node and under a camera model proposal.
- **MCP** (`mcp/tests/measurements.rs`): the read's factor, statistics,
  factors and exclusions, its default unit, the write as one edit after which
  the read gives 1, and the refusal of a node with no measurement in the menu's
  words, with the read's `refusal` the same; and the write leaves the 3D
  camera's position, orientation, orbit distance, near plane, field of view and
  up direction, the scene scale and the display transform exactly as they were.

`ui_basic` finds the entry in the reconstruction row's menu in a real window,
enabled on the demo scene, which stores a distance measurement.

---

## Non-goals

- **Choosing which measurements count.** The median is what keeps one bad
  measurement from moving the scale; the table and the warning show which one
  it is, and Measurement View is where it is corrected or deleted before the
  scale is applied again.
- **A scale other than the median.** The mean is shown beside it for the
  spread, as core reports it, and is not offered: one misplaced endpoint moves
  it, which is the failure the measurements are there to catch.
- **Moving the view with the scale.** Keeping the picture still, through the
  camera or the display transform, was weighed and left out: the view stays
  where it is, and `Z` frames the scaled scene.