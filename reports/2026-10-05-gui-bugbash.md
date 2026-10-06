# SfM Explorer bug bash — 2026-10-05: walking the user docs

Findings from following `docs/index.md` and `docs/tutorials/getting-started.md`
step by step on a release build of `main` at d4a25429, and rebuilding each of
their six screenshots by driving the SfM Explorer through its MCP endpoint. The
viewer was started with `skills/gui-bug-bash/viewer.py launch` on a random port
(54254) and called as JSON-RPC over HTTP with `mcp.py`. The scope is the user
documentation and what an agent needs to reproduce it: the tutorial's CLI
steps, its GUI steps and screenshots, and the MCP tools those steps exercise.
The published walkthrough with the side-by-side screenshots is a private
artifact; this report carries every finding from it.

## Files

| label in the viewer | file | kind |
|---|---|---|
| `dino` | a fresh `sfm solve -g images/ --max-features 2000` of the 85 `test-data/images/dino_dog_toy` images, run in a scratch workspace outside the repo exactly as the tutorial says (9,941 points, 72,901 observations); the viewer opened a copy in `sfmr/bugbash-2026-10-05-54254/dino.sfmr` beside it | `sift_files`, later converted to embedded patches on the copy |
| `seoul_bull_sculpture_ground_truth` | `test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr`, opened in place with `open_reconstruction` and never edited or saved | 17 images, embedded patches, metres, Z up |

Severity is a judgement: **high** loses or corrupts data, **medium** blocks a
feature or gives a wrong answer, **low** is a wrong message, display or schema
detail. Findings 13–18 are documentation and process recommendations rather
than defects; they carry **docs** in place of a severity.

## Summary

| # | Severity | Area | Finding |
|--:|---|---|---|
| 1 | medium | MCP `set_view` | No relative camera moves; turning in place by look-at leaves camera view |
| 2 | medium | Track View | A point from a fresh `sfm solve` shows no per-observation readings, which is the tutorial's main Track View step |
| 3 | medium | MCP input | The target indicator and the point size cannot be set by an agent |
| 4 | medium | MCP input | No drag or scroll input, so no tutorial gesture can be exercised as a gesture |
| 5 | medium | docs | The GUI tutorial's screenshots and panel names no longer match the viewer |
| 6 | low | MCP reads | No way to find points by image, track length or error; finding one track took 9,941 `get_point` calls |
| 7 | low | 3D view / MCP | A solve that is not gravity-aligned gets an unreadable default view, and nothing reports the scene's up direction |
| 8 | low | MCP schema | Pose quaternions are `quaternion_wxyz` in `get_camera_image` and `orientation_wxyz` in `set_view`, and the convention is not stated |
| 9 | low | MCP schema | `convert_to_embedded_patches` requires `reconstruction_label` where other tools default to the selection |
| 10 | low | screenshots | Every screenshot carries the yellow `MCP: …` status line and any stale hover label |
| 11 | low | 3D viewer HUD | The stats line overlaps the `Pos:` readout in a narrow 3D panel |
| 12 | low | data | The longest track in the dino solve holds several observations in the same image |
| 13 | docs | CLI quickstart | Missing dataset download, unexplained `--max-features` values, unmentioned solver log noise |
| 14 | docs | CLI output | `inspect -v` and `sfm motion` output in the docs is out of date, and nothing says counts vary between runs |
| 15 | docs | GUI tutorial | Write each GUI step as a card: gesture, precise target, expected result, agent calls |
| 16 | docs | GUI tutorial | Run the GUI tutorial on a ground-truth reconstruction, and make one for the dino |
| 17 | docs | screenshots | Generate the docs screenshots from stored scene recipes and check them for drift |
| 18 | docs | docs site | Add a page on exploring a reconstruction with an agent |

## 1. No relative camera moves in `set_view` (medium)

> _Status (2026-10-05): **Partially done** — relative `move`, `turn` and
> `orbit` forms and the scene unit in `get_scene` are being implemented on
> branch `mcp-relative-view-moves`._

> _Status (2026-10-06): **Done** — `set_view` takes relative `move`
> (level axes, optional `unit`), `turn` (keeps camera view) and `orbit` forms,
> applied in that order, and `get_scene` reports each file's
> `world_space_unit` and the view's unit (the selected reconstruction's), PR
> #804._

`set_view` takes only absolute forms. To carry out "step back 2 m, rise 1 m,
turn right 35°" the caller has to read `get_scene`'s view block, compute the
new position and a rotated forward vector itself, and send the look-at form.

Repro on `seoul_bull_sculpture_ground_truth`:

1. `set_view {look_through: {camera_image: 5}}`.
2. Compute, client side, camera 5's centre moved −2 m along its forward
   direction projected to the ground, +1 m in Z, and the forward vector yawed
   −35° about +Z.
3. `set_view {position: [-0.18, -4.51, 2.16], target: [2.37, -1.72, 0.85], up: [0, 0, 1]}`.

That works, but it took a hand-written Rodrigues rotation. The same approach in
the tutorial's camera-view step goes wrong: on `dino`,
`set_view {look_through: {camera_image: 19}}` and then a look-at with the same
position and a forward turned 35° right replies with `looking_through: null`.
The tutorial says that after looking around "the 3D viewport is in camera view
mode", and a person's free-look drag keeps camera view
(`specs/gui/camera-views.md`). An agent therefore cannot follow that step.

Expected: `move`, `turn` and `orbit` forms in the view's frame, with `turn`
staying in camera view as free-look does, and lengths in a unit the reply
names.

## 2. Track View shows no per-observation readings for a fresh solve (medium)

> _Status (2026-10-05): **Needs decision** — either give Track View readings
> for a point with no patch frame (reprojection error, ray angle, feature
> index and size need no frame), or have the tutorial convert to embedded
> patches first and describe the table as it is now._
>
> _Status (2026-10-06): **Not done** — decided: the tutorial adds a Convert to
> Embedded Patches step before the Track View steps and describes the table's
> current columns. Track View keeps showing no per-observation readings for a
> point with no patch frame._

The tutorial's "Viewing 3D Point Tracks" step says the Track View table "shows
the SIFT feature sizes and reprojection error in both pixels and degrees".
Every reconstruction `sfm solve` writes has `feature_source: "sift_files"`, and
for such a point Track View shows an orange refusal and `-` in every cell.

Repro on `dino`, straight after the tutorial's solve:

1. `select_point {point: 830}` (18 observations, one of them in image 2).
2. `screenshot {}`. Track View's header reads `pt3d_2db7e1f4_830 | … | track:
   18 obs`, then *"Cannot evaluate pt3d_2db7e1f4_830: the track carries no
   patch frame to read against, as a point put on the bench from a
   sift_files reconstruction has none; convert the reconstruction to
   embedded patches and put the point on the bench again"*. The rows list
   images 2, 3, 4, 12, … with `-` in every Verdict, ZNCC and Self-similarity
   cell and empty Crop and Patch frames.
3. `convert_to_embedded_patches {reconstruction_label: "dino"}`, wait, and
   `select_point {point: 830}` again: the rows now have crops, verdicts, ZNCC
   and self-similarity, but no feature index, size, pixel error or angle
   column.

This is specified (`specs/gui/track-view.md` § "A point with no patch frame",
and § "Non-goals" lists per-observation readings for such a point), so it is a
gap between the spec and the tutorial rather than a bug in the code. The
readings the tutorial describes need no patch frame: `get_point` already
returns each observation's `reproj_error` and pixel.

## 3. The target indicator and the point size cannot be set by an agent (medium)

The tutorial's last GUI screenshot shows the orbit target (double-tap Alt) and
enlarged points. Neither can be set over MCP.

Repro on `dino`:

1. `press_key {key: "Alt"}` is refused: *"press_key does not know the key
   \"Alt\". The key names are egui's: the letters A to Z, …"*.
2. `press_key {key: "F35", modifiers: ["alt"], panel_name: "viewer_3d", at_px: [200, 240]}`
   twice: accepted, but the target indicator does not appear.
3. Open the Display popup (`click` on its gear button) and `click` on the
   *Points* size slider at 70% of its width: the slider's value stays `"0"`
   (`get_widgets`).
4. `click` the spin box beside it, then `type_text {text: "1.5"}`: refused,
   *"The widget with keyboard focus is an unnamed spin button in the 3D Viewer
   panel, which is not a text input, so the text would be dropped"*.

Expected: a display-settings tool (point size, the layer toggles, Maintain
Z-up, target indicator visible) with a matching read, or at least Alt as a key
in `press_key`.

## 4. No drag or scroll input (medium)

The MCP input tools are `click`, `hover`, `press_key` and `type_text`. Every
navigation gesture the tutorial teaches (drag to orbit, Shift+drag to pan,
Alt+drag to look around, scroll and Control to zoom) is a drag or a scroll, so
an agent can reach the result through `set_view` but cannot check that the
gesture does what the docs say. Sliders are unreachable for the same reason
(finding 3).

Expected: `drag {from_px, to_px, mouse_button, modifiers, steps}` and
`scroll {at_px, delta}`, going through the same input path as `click`.

## 5. The GUI tutorial no longer matches the viewer (medium)

What a new user sees after `sfm explorer` differs from every screenshot in
`docs/tutorials/getting-started.md`:

- The menus are File / Edit / Go / Panels, not File / View.
- The stock layout has seven panels (Scene and Background Task on the left,
  Image Detail behind a tab of the 3D Viewer, Track View and Camera
  Intrinsics on the right, Image Browser, Action Log and Edit History at the
  bottom). The tutorial says "arrange SfM Explorer so that the Image Detail
  and Track View panels are visible side by side" without saying how.
- The Display popup is open over the 3D view on first launch (seen with
  `--no-default-layout`; check whether a plain first launch does the same).
- Image Detail draws the intrinsics layer (angle axes and distortion arrows)
  by default; none of the screenshots has it.
- The screenshots name the panel "Point Track"; the text and the viewer say
  "Track View".
- The Track Length heat map runs red (short) to green (long); the screenshot
  shows blue to red.
- The navigation key list (WASD, R/F, Q/E) is correct but leaves out Z (zoom
  to fit), Home (level horizon), `,` and `.` (previous and next camera in
  camera view), and that Q/E turn Maintain Z-up off
  (`specs/gui/viewport-navigation.md`).
- "Double-clicked on the third image" means index 2, file `_03`; the Image
  Browser numbers from 0 and the file names from 1, and the tutorial does not
  say which it means.

All six screenshots were rebuilt for comparison. Five reproduce in substance;
the visible-target one cannot be (finding 3).

## 6. No way to find points by image, track length or error (low)

To reproduce the tutorial's screenshots an agent needs "a track of about 18
observations seen in image 3", "a 4-observation track through image 57" and
"the longest track". No tool answers those, so the walk read every point: a
loop of 9,941 `get_point` calls from one `mcp.Client` took 2 min 11 s on
`dino`.

Expected: a read such as `find_points {camera_image?, min_track?, max_track?,
max_error?, order_by, limit}` that returns ids with the summary fields.

## 7. A solve that is not gravity-aligned gets an unreadable default view (low)

The dino solve's true up direction, taken as the mean of the camera images'
up vectors, is about (0.50, −0.31, 0.81): roughly 36° from the viewer's Z up.
On opening, and after `set_view {fit: null}`, the view looks at the scene from
underneath, through a scattered cloud with no recognisable dino. The grid is
drawn tilted against the table.

Getting a view like the tutorial's took three scripts: sample the camera
centres and orientations over MCP, estimate the up direction and the point the
cameras converge on, and place the camera by look-at. The first estimate had
the wrong sign because of the quaternion convention (finding 8).

Expected: `get_scene` reports an estimated scene up direction and the point
the cameras look at, and `set_view` can level the view to that up. The
viewer's own Level Horizon (Home) could use the same estimate.

## 8. Quaternion field names and convention differ between tools (low)

`get_camera_image` reports a camera's rotation as `quaternion_wxyz` (with
`translation_xyz`); `set_view` and `get_scene`'s view block call the same kind
of value `orientation_wxyz`. Neither description states the camera-axis
convention. Building a rotation matrix from `quaternion_wxyz` and taking
`R^T·(0,0,1)` as forward gives the negation of the forward vector
`get_scene` reports for `set_view {look_through: {camera_image: 10}}` on
`dino` (mine (−0.002, −1.000, −0.011), the viewer's (0.002, 1.000, 0.011)),
while the camera centre `−R^T·t` matches exactly. A caller cannot tell from the
schema which axes the quaternion maps.

Expected: one field name for a world-to-camera rotation across tools, and a
description that says which way the camera looks and which way is up in its
frame.

## 9. `convert_to_embedded_patches` requires `reconstruction_label` (low)

`convert_to_embedded_patches {}` with `dino` selected is refused:
*"convert_to_embedded_patches needs reconstruction_label."* Almost every
other tool that takes `reconstruction_label` documents "Omit for the selected
one". If the requirement is deliberate (it rewrites every observation), its
description should say why; otherwise it should default like the rest.

## 10. Screenshots carry the MCP status line and stale hover labels (low)

Every whole-window or `viewer_3d` screenshot shows the yellow status line of
the last MCP call (`MCP: hover window 1270,700`, `MCP: Camera placed`) under
the stats line, and a hover label left from an earlier `hover`
(`Point3D #1314` at the bottom of the 3D panel). `hud: false` removes them,
but only for the 3D panel, and it removes the stats as well. For documentation
screenshots there is no way to get the window as a person would see it.

Expected: a screenshot option, or a viewer flag, that hides the MCP status
line and parks the pointer outside the window before the frame is drawn.

## 11. The 3D HUD stats overlap the `Pos:` readout in a narrow panel (low)

With the 3D Viewer about 400 px wide (the three-column layout the tutorial
asks for), the stats line `9941 points | 85 images | 60 fps` is drawn over the
`Pos: [x, y, z]` readout, and on `seoul_bull_sculpture_ground_truth`
`280 points (14 at infinity) | 17 images | 60 fps` overlaps it the same way.
The tutorial's own screenshots show the same overlap, so it predates this
build.

Expected: the readout moves to a second line, or is shortened, when the two
would touch.

## 12. The longest dino track holds several observations in one image (low)

> _Status (2026-10-05): **Needs decision** — whether several observations of
> one point in one image are expected from a GLOMAP solve, and whether the
> tutorial's "long track" example should be chosen after
> `prune_covered_observations`._
>
> _Status (2026-10-06): **Not done** — decided: several observations of one
> point in one image are expected from a GLOMAP solve. Cleanup for them is
> still to be built: in the COLMAP import, or as a viewer menu action. One
> option is to have Prune Covered Observations remove them as part of its
> run; its spec (`specs/gui/edits/prune-covered-observations.md`) does not
> cover repeated observations of the same point today._

Point 9873 of `dino`, the longest track at 57 observations, has three
observations in image 3, three in image 12 and three in image 25, and two in
each of images 5, 13, 16, 17, 29 and 40. Its 57 observations cover 45
images. The tutorial picks "a long track with 58 observations" as its
example. A reader would take that as 58 photographs.

## 13. CLI quickstart gaps (docs)

The CLI steps run word for word: 85 of 85 images registered in 51 s. Four
things would help a new reader:

- Step 1 says `cp path/to/dino_dog_toy/*.jpg` and links the GitHub tree, but
  gives no command to download or clone the 85 images.
- Step 2 runs `sfm ws init --max-features 4000` and step 3
  `sfm solve -g images/ --max-features 2000`, without saying why the values
  differ or which applies (the solve records `max_features: 2000`).
- The solve prints several hundred COLMAP and GLOMAP log lines, including
  *"Requested to use GPU for bundle adjustment, but COLMAP was compiled without
  CUDA support"* and *"Less than 50% of cameras have prior focal lengths"*. The
  docs show a clean summary, so a beginner may take the warnings for a
  failure.
- Step 4 runs `sfm explorer` and then opens the file through the menu;
  `sfm explorer sfmr/<file>.sfmr` does it in one step and is what
  `docs/index.md` shows.

## 14. CLI output in the docs is out of date (docs)

- `sfm inspect -v` now also prints *Tool options*, *Depth reliability*,
  *Point or bearing* and *Integrity: OK*.
- `sfm motion`'s summary table now has the columns Dist, Rot, StepR, CovR,
  ObsZ(A/B), SharedPts, Err(A), Err(B) and Signals, with a P/S/C/O legend and a
  single-signal / multi-signal confidence split. The docs show the older
  Dist(prev) / Dist / Dist(next) layout.
- The numbers differ from run to run (9,941 points and 2 flagged images here,
  10,580 and 1 in the docs), and the docs do not say to expect that.

Recommendation: cut each output block to the lines the text discusses, say
that counts vary, and add a docs test that runs the quickstart on
`seoul_bull_sculpture` and checks the section headings of `inspect -v`,
`analyze --metrics`, `analyze --z-range` and `motion`. A new section then fails
a test instead of leaving the docs stale.

## 15. Write each GUI step as a card (docs)

The GUI steps describe a gesture ("I've looked over to the right", "it should
be easy to move the viewport") but not where the reader should end up or how
to tell they got there. Give each step four parts:

- **Do**: the gesture in loose terms, for a person.
- **Precisely**: the same move in a known frame and unit.
- **You should see**: readable on-screen text where possible ("the Track View
  header reads *track: 18 obs*"), which a person or an agent can check.
- **Agent**: the MCP calls, in a collapsed block.

A worked card, run during this bash on
`seoul_bull_sculpture_ground_truth`:

- **Do**: double-click thumbnail 5 to look through that photo, step back
  about 2 m and up about 1 m (hold S, then R), and turn right about 35°.
- **Precisely**: camera 5's centre, −2.0 m along its forward direction
  projected to the ground, +1.0 m in Z, then yawed −35° about +Z; position
  (−0.18, −4.51, 2.16).
- **You should see**: the bull's patches at the lower-left edge and the arc of
  camera frustums across the top.
- **Agent**: `set_view {look_through: {camera_image: 5}}`, then
  `set_view {position: [-0.18, -4.51, 2.16], target: [2.37, -1.72, 0.85], up: [0, 0, 1]}`
  (one `set_view` with `move` and `turn` once finding 1 is done).

## 16. Use a ground-truth reconstruction for the GUI tutorial (docs)

Distances and angles in a step card only mean something in a reconstruction
with a known scale and up direction. `seoul_bull_sculpture_ground_truth.sfmr`
and `kerry_park_ground_truth.sfmr` are in metres with Z up. The dino has no
ground truth, and its solves come out tilted and at an arbitrary scale
(finding 7). Make one: level it with the mean camera up vector (about 20 lines
during this bash) and scale it from one measured length, such as the table
width or the toy's height. Alternatively, move the GUI tutorial to the Seoul
Bull.

## 17. Generate the docs screenshots from stored scene recipes (docs)

All six GUI screenshots drifted and nothing noticed. Store each one's state
(window size, layout, selection, view, Image Detail overlay and view, display
settings) as a JSON recipe beside the image, for example
`docs/recipes/<shot>.json`, and add a task that launches a viewer on a random
port, applies each recipe over MCP and writes the PNGs. Run it before a
release, or in the `ui-test-windows` job with an image-difference threshold,
so a UI change that invalidates a screenshot is reported.

An `apply_recipe` / `capture_recipe` pair of MCP tools would make the recipe
one call; until then the recipe is a list of existing calls
(`set_window_layout`, `select_*`, `set_image_detail_display`,
`set_image_detail_view`, `set_view`).

## 18. Add a page on exploring a reconstruction with an agent (docs)

The user docs never mention the MCP endpoint. A short page should cover
`pixi run gui-mcp <file>.sfmr` (or `sfm explorer --mcp`), registering it with
`claude mcp add --transport http sfm-explorer http://127.0.0.1:8787/mcp`, and a
few example requests built on the step cards ("open the bull, look through
image 5, show me the longest track"). The material is in AGENTS.md and
`specs/gui/mcp-server.md` but not on the docs site.

## Not yet reproduced

- `set_image_detail_view {camera_image: 2, point: 830, zoom: 3}`, sent while
  Image Detail showed image 47, replied with `zoom: 1.0` and the whole
  photograph in view. The same call without `camera_image`, sent next,
  applied zoom 3 centred on the point. Seen once; a fresh-state repro is
  needed before it is reported.

## Ruled out as by design

- Track View's refusal for a `sift_files` point (`specs/gui/track-view.md`
  § "A point with no patch frame"); kept as finding 2 because the tutorial
  depends on it.
- After Convert to Embedded Patches, the 4-observation track of point 2032
  has three observations judged *out* at the default 70% ZNCC thresholds.
  Verdicts are evaluated against the thresholds; nothing says a solver's
  track must pass them.

## Checked and found working

- The tutorial's CLI steps: `sfm ws init --max-features 4000`,
  `sfm solve -g images/ --max-features 2000`, `sfm inspect -v`,
  `sfm analyze --metrics`, `sfm analyze --z-range`, `sfm motion`.
- `set_window_layout` with a custom three-column arrangement, including
  closing every panel it does not mention.
- `set_view` in its `look_through`, look-at, `fit` and `fov_short_axis_deg`
  forms; `screenshot` with `panel_name`, `hud: false` and whole-window.
- `select_camera_image`, `select_point`, `clear_selection`, `set_solo`,
  `open_reconstruction`, `list_camera_images`, `get_camera_image`,
  `get_point` over every point of a 9,941-point reconstruction.
- `set_image_detail_display` (overlay mode, tracked only, intrinsics layer
  off) and `set_image_detail_view` with `point` + `zoom` and with `fit`.
- `hover` over a tracked feature in Image Detail: the tooltip and the
  cross-panel highlight of the point in the 3D view.
- `click` by `widget` id and by `at_px`, including the Display popup's close
  and gear buttons.
- `convert_to_embedded_patches` with `wait`, on a 9,941-point, 85-image
  `sift_files` reconstruction (1.5 s).
