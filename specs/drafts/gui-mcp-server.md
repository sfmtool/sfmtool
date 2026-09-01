# SfM Explorer MCP Server

> **Draft.** This is a change proposal, not a standing spec: nothing described
> here is implemented yet. Convert it to `specs/gui/mcp-server.md` — rewritten
> in the present tense, with the phasing section dropped — when the first cut
> lands, and add the row to `specs/gui/README.md` then.

## Purpose

SfM Explorer is a viewer you drive by hand: you open reconstructions, click a
camera, look at where its rays land, notice that one image is in the wrong
place, and go and fix it somewhere else. An AI coding agent working on this
repository cannot do any of that. It can read `.sfmr` files with `sfm inspect`
and it can read the source, but it cannot see the picture, and it cannot ask
the picture a question — which means the one tool that would tell it fastest
whether a solve is wrong, and *where*, is the one tool it cannot reach.

This proposes an opt-in control surface for the running viewer, speaking the
[Model Context Protocol][mcp] (MCP) over a loopback HTTP endpoint. Started with
`sfm-explorer --mcp`, the viewer hosts a small server; an agent connects to it
and can then enumerate the loaded scene graph, open and close `.sfmr` files,
move the selection and the 3D camera, and take a screenshot of the viewport.
The human keeps the window in front of them the whole time and watches it
change.

The design is deliberately narrow at the start. The viewer's own invariants
(§ "Addressing", § "Threading") are what the surface is shaped around, not the
other way round, and editing reconstruction data — the intrinsics case in
particular — is held back to a second phase with its own answer (§ "Phase 2").

[mcp]: https://modelcontextprotocol.io/

## Enabling it

Off unless asked for. The flag is on the Rust binary and is forwarded by the
Python CLI:

```bash
sfm-explorer --mcp scene.sfmr          # default port, 127.0.0.1:8787
sfm-explorer --mcp 9000 scene.sfmr     # explicit port
sfm-explorer --mcp 0 scene.sfmr        # ephemeral port, printed at startup
sfm explorer --mcp 9000 scene.sfmr     # same, through the Python CLI
```

`crates/sfm-explorer/src/lib.rs` currently treats every argument as a path
(`std::env::args().skip(1)`). It grows a small hand-rolled parse — `--mcp`,
`--mcp=PORT`, `--mcp PORT`, `--help`, everything else a path. That is a dozen
lines and keeps the binary dependency-free; reach for `clap` if a third flag
ever appears, not before.

On startup the server binds and prints one line to stdout, which is what a
human pastes into a client config:

```
SfM Explorer MCP endpoint: http://127.0.0.1:8787/mcp
```

**A bind failure is fatal and loud.** Two viewers on one port is the common
mistake, and a viewer that silently came up without the endpoint the agent was
told to use is worse than one that refused to start. The message names the port
and suggests `--mcp 0`.

**The window says so.** While the server is live the title carries a suffix —
`SfM Explorer - scene.sfmr [MCP :8787]` — and the Scene panel grows a header
line with the endpoint and a live request count. A window that something else
can drive should never look like one that nothing can. This extends
`AppState::window_title`, which already composes the title from the scene.

### Connecting an agent

For Claude Code, with the viewer already running:

```bash
claude mcp add --transport http sfm-explorer http://127.0.0.1:8787/mcp
```

## The tool surface

Sixteen tools in the first cut. Five read, ten write, one that closes the loop
by handing back a picture.

| Tool | Kind | What it does |
|------|------|--------------|
| `get_scene` | read | The whole scene graph, the selection, and the view state |
| `list_camera_images` | read | One reconstruction's camera images, paginated |
| `get_camera_image` | read | One camera image: pose, intrinsics, observation stats |
| `get_camera_intrinsics` | read | One intrinsics record and the camera images that use it |
| `get_point` | read | One 3D point: position, colour, error, full track |
| `open_reconstruction` | write | Load an `.sfmr` into the scene (reload if already open) |
| `close_reconstruction` | write | Close one reconstruction, or all of them |
| `select_reconstruction` | write | Make one the reconstruction the file- and sequence-shaped panels follow |
| `select_camera_image` | write | Select a camera image — and with it the intrinsics it was shot through |
| `select_camera_intrinsics` | write | Select an intrinsics record |
| `select_point` | write | Select a 3D point |
| `clear_selection` | write | Drop the selection, wholly or one kind of it |
| `set_reconstruction_display` | write | One reconstruction's eyes, tint, interactivity |
| `set_solo` | write | Draw only one reconstruction, or end the solo |
| `set_view` | write | Frame the scene, look through a camera image, or set the viewport camera outright |
| `screenshot` | observe | PNG of the 3D viewport |

Every tool is annotated: the five reads carry `readOnlyHint: true`, the writes
`destructiveHint: false` (nothing here touches a file on disk —
`close_reconstruction` unloads, it does not delete), and every one of them
`openWorldHint: false`.

### The wire vocabulary

One rule, applied without exception: **one entity, one spelled-out word, in tool
names, arguments and reply fields alike.** No abbreviations, and no word that
names two things.

That costs a few characters per call and buys the only thing that matters here
— an agent reading `tools/list` can tell what a tool addresses without reading
prose, and a human reading the agent's transcript can tell what it did. Tool
names and argument names *are* the API: they live in client configs and in the
prompts people write against them, so a name is far more expensive to change
than to choose.

#### `reconstruction`

One loaded `.sfmr` is a **reconstruction**, spelled in full everywhere on the
wire — `open_reconstruction`, `close_reconstruction`, `select_reconstruction`,
`set_reconstruction_display`, and the `"reconstruction_label"` argument every
reconstruction-scoped tool takes. It is the word the domain uses, and it stays
unambiguous however the scene graph grows.

The Scene Graph panel calls each one a *node*, and that word stays in this
spec's prose wherever the subject really is that tree. On the wire it would name
a presentation rather than the thing, and a nested scene graph would want it
back. The Rust side likewise keeps its own names (`ReconId`, `SceneNode`,
`AppState::scene`); only the wire is normalized.

#### `camera_image` and `camera_intrinsics`

Two things in a `.sfmr` answer to the everyday word "camera", and the wire
vocabulary gives each its own name:

| Wire name | What it is | Where it lives |
|-----------|------------|----------------|
| `camera_image` | One **posed view** — extrinsics, a frustum, an image quad | inside a reconstruction (`SfmrImage`) |
| `camera_intrinsics` | One **lens** — model, size, distortion; shared by any number of camera images | inside a reconstruction (`CameraIntrinsics`) |
| `image` | *Reserved.* A picture that is not in a reconstruction | — not implemented — |

The first two are the Scene Graph panel's own two sibling groups under a node,
**Camera Images** and **Camera Intrinsics**; `SceneNode::show_camera_images` is
named for that distinction on purpose. The wire vocabulary follows the
**panel's** words rather than the Rust type names (`ImageRef`, `CameraRef`,
`AppState::selected_camera`), because the panel is what the human is looking at
while the agent works, and the two of them have to be able to talk about the
same thing.

**The third row is a reservation.** The flow this viewer is built toward is:
open a reconstruction, load a folder of *further* images that are not in it,
browse them, and resect the good ones in — and the machinery is half there
already, since `Resect Image from Matches…` estimates a pose against a match
graph read from outside the reconstruction. Once loose images are loadable, two
kinds of picture live in the same panel at the same time, and each needs its own
word.

So the posed one takes the compound: `list_camera_images`, `get_camera_image`,
`select_camera_image`. The bare `list_images` / `get_image` / `select_image` are
held for the pictures with no pose yet (§ "Phase 2"), and cost six characters to
keep free.

#### `<entity>` for the thing, `<entity>_<attribute>` for a reference to it

A field holding a whole entity is named for the entity; a field holding *one
attribute that identifies* an entity is named for both:

| Field | Holds |
|-------|-------|
| `camera_intrinsics` | the expanded record — model, size, params |
| `camera_intrinsics_index` | just the index, whether as a cross-reference from a camera image or as the argument naming which record to act on |
| `reconstruction_label` | just the label, identifying which reconstruction |

This is why the argument is `reconstruction_label` and not `reconstruction`. It
carries a label, not a reconstruction, and a reader comparing
`"reconstruction_label": "seoul_bull"` against a sibling
`"camera_image": { … }` should not have to infer that one is a handle and the
other an object. It is also why a camera image's cross-reference to its lens is
`camera_intrinsics_index` rather than `camera_index` — which would reintroduce
the bare `camera` the previous rule just removed, and read as plausibly "the
index of this camera image".

**A reply always qualifies; an argument qualifies only when it has one
spelling.** A reply knows exactly which form it is emitting, so it always says
— `camera_image_index` in a track observation, `camera_image_indices` for the
set of images using an intrinsics record, `label` on a scene entry. An argument
is named for what it accepts, which is not always one thing:

```jsonc
select_camera_image       { …, "camera_image": 3 }
select_camera_intrinsics  { …, "camera_intrinsics_index": 0 }
```

An intrinsics record has one handle, its index, so the argument names it. A
camera image has two — an index and a name — and this surface hands out both: a
track observation reports an index, a `list_camera_images` row reports both, and
an agent arrives holding whichever it read. That field takes the entity's name
and its schema carries the union, so either spelling is a valid call
(§ "One spelling per handle").

#### One spelling per handle, except where two spell one identity

A reconstruction has one handle, its label, so `reconstruction_label` takes a
label and the field says so (§ "Addressing" for why the label is the handle).

A camera image and a point each have two — an index or a name, an id or a bare
index — and an agent arrives holding whichever the surface last handed it. Those
fields take the entity's name and their schema carries the union, so both
spellings are valid calls. A union of *representations* reads as one question
with two phrasings; a union of *intents* is the thing
§ "select_reconstruction / …" keeps out of a single tool.

#### Where the GUI has no word, the code's word wins

The rules above take their vocabulary from the Scene Graph panel, because those
entities are ones the human is looking at while the agent works. Plenty of what
the surface reports has no panel label at all — the viewport camera is internal
state, not a named thing in the UI — and there the default is the **field name
in the code**, not a term of art imported from elsewhere.

So the view block reports `position`, `orientation`, `target_distance`,
`world_up` and `near` — every one of them a `Camera` or `ViewportCamera` field,
so every one of them greppable from the wire. The same default gives a camera
image's `center` (`SfmrImage::camera_center`) and a point's `position`
(`Point3D::position`): two words for two positions, because the code uses two,
and each is unambiguous inside the object that carries it.

### `get_scene`

The first call an agent makes, and the one that makes every other call
addressable. No arguments.

```jsonc
{
  "scene": [
    {
      "label": "seoul_bull",              // the handle everything else takes
      "path": "C:/work/seoul_bull.sfmr",  // null for demo and derived ones
      "content_hash": "a1b2c3d4",         // 8-hex prefix, as in point ids
      "counts": { "points": 4210, "points_at_infinity": 0, "camera_images": 17,
                  "camera_intrinsics": 1, "observations": 19844 },
      "display": { "visible": true, "interactive": true, "show_points": true,
                   "show_camera_images": true, "show_patches": true,
                   "show_points_at_infinity": true, "tint": null },
      "transformed": false,               // SceneNode::has_transform
      "has_patch_data": false
    }
  ],
  "selection": {
    "reconstruction_label": "seoul_bull",
    "camera_image": { "reconstruction_label": "seoul_bull", "index": 3,
                      "name": "images/IMG_0042.jpg" },
    "camera_intrinsics": { "reconstruction_label": "seoul_bull", "index": 0 },
    "point": null
  },
  "solo": null,
  "view": {                             // see § "The view block"
    "position": [1.20, -4.40, 2.00],    // Camera::position
    "orientation_wxyz": [0.92, 0.39, 0.00, 0.00],   // world→camera attitude
    "target_distance": 4.60,            // along forward, i.e. camera −Z
    "world_up": [0.0, 0.0, 1.0],        // navigation up; carries roll
    "fov_short_axis_deg": 45.0,         // the *shorter* viewport dimension
    "near": 0.081,                      // adaptive, recomputed every frame
    "derived": {                        // all recoverable from the above
      "target": [0.00, 0.00, 0.30],     // position + forward · target_distance
      "forward": [-0.26, 0.87, -0.42],
      "up": [-0.12, 0.40, 0.91],
      "viewport_px": [1280, 720],
      "fov_horizontal_deg": 73.7,
      "fov_vertical_deg": 45.0
    },
    "looking_through": null             // a camera image, in camera-view mode
  },
  "status_message": null,
  "window_title": "SfM Explorer - seoul_bull.sfmr [MCP :8787]"
}
```

`counts` is summary only. A 4 M-point cloud must never cross this boundary as
JSON, and no tool here returns bulk arrays — that is what the `.sfmr` file and
`sfm inspect` are for. The agent reads the file for the data and asks the
viewer for the *state*.

### `list_camera_images` / `get_camera_image` / `get_camera_intrinsics` / `get_point`

```jsonc
// list_camera_images { "reconstruction_label": "seoul_bull",
//                      "offset": 0, "limit": 50 }
{ "reconstruction_label": "seoul_bull", "total": 17, "offset": 0,
  "camera_images": [ { "index": 0, "name": "images/IMG_0039.jpg",
                       "camera_intrinsics_index": 0,
                       "center": [0.31, -1.90, 0.42],
                       "observations": 1204 } ] }

// get_camera_image { "reconstruction_label": "seoul_bull",
//                    "camera_image": "images/IMG_0042.jpg" }
{ "index": 3, "name": "images/IMG_0042.jpg",
  "camera_intrinsics": { "index": 0, "model": "OPENCV",
                         "width": 270, "height": 480 },
  "quaternion_wxyz": [0.98, 0.01, -0.17, 0.04],
  "translation_xyz": [0.10, -1.88, 0.51],
  "center": [0.28, -1.85, 0.49],
  "observations": 1187,
  "reproj_error": { "mean": 0.61, "median": 0.48, "p95": 1.72 } }

// get_camera_intrinsics { "reconstruction_label": "seoul_bull",
//                         "camera_intrinsics_index": 0 }
{ "index": 0, "model": "OPENCV", "width": 270, "height": 480,
  "params": { "fx": 402.1, "fy": 402.1, "cx": 135.0, "cy": 240.0,
              "k1": -0.031, "k2": 0.004, "p1": 0.0, "p2": 0.0 },
  "camera_image_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] }

// get_point { "point": "pt3d_a1b2c3d4_1207" }
{ "id": "pt3d_a1b2c3d4_1207", "reconstruction_label": "seoul_bull",
  "index": 1207,
  "position": [0.02, 0.41, 0.88], "color": [173, 141, 96], "error": 0.53,
  "at_infinity": false,
  "track": [ { "camera_image_index": 3, "name": "images/IMG_0042.jpg",
               "xy": [131.4, 208.9], "reproj_error": 0.44 } ] }

// get_point { "point": 1207 }   // bare index, in the selected reconstruction
```

The reprojection figures come from
`SfmrReconstruction::compute_observation_reprojection_errors` — the same source
the Image Detail panel's error heatmap already reads, so a number the agent is
told matches the colour the human is looking at.

`params` is a name→value map rather than the model's positional parameter
vector, because a positional vector is unreadable without also shipping the
model's parameter order and an agent will get it wrong. `CameraModel` already
knows the names.

`get_point` accepts either shape `crate::goto_point::parse_point_query` already
accepts — a bare index against the selected node, or a full
`pt3d_<hash>_<index>` id that names its own node — and resolves it through
`resolve_point_query`. One parser, one set of error messages, and a point id
copied out of the Point Track panel by a human pastes straight into a tool
call.

### `open_reconstruction` / `close_reconstruction`

```jsonc
// open_reconstruction { "path": "C:/work/global.sfmr" }
// -> the new reconstruction's entry, exactly as get_scene would render it
{ "label": "global", "reloaded": false, /* … */ }

// close_reconstruction { "reconstruction_label": "global" }  or  { "all": true }
{ "closed": ["global"] }
```

`open_reconstruction` is `AppState::load_file` unchanged, including its
already-loaded rule: opening a path that is already open **reloads that node in
place**, keeping its label, display state and transform. `reloaded: true` says
which happened. The returned `label` may differ from the file stem —
`unique_label` disambiguates a collision as `global (2)` — so the agent must
read the label back rather than assume it.

A load failure is a tool error, not a silent status line (§ "Errors").

### The five selection tools

```jsonc
// select_reconstruction     { "reconstruction_label": "global" }
// select_camera_image       { "reconstruction_label": "seoul_bull",
//                             "camera_image": 3 }
// select_camera_image       { "reconstruction_label": "seoul_bull",
//                             "camera_image": "images/IMG_0042.jpg" }
// select_camera_intrinsics  { "reconstruction_label": "seoul_bull",
//                             "camera_intrinsics_index": 0 }
// select_point              { "point": "pt3d_a1b2c3d4_1207" }  // or bare index
// clear_selection           { "scope": "all" }
//   scope: "all" | "camera_image" | "camera_intrinsics" | "point"
```

All five return the same thing — the resulting `selection` block, exactly as
`get_scene` renders it — so the agent sees what the invariants did to its
request rather than assuming it got what it asked for:

```jsonc
{ "selection": { "reconstruction_label": "seoul_bull", "camera_image": {...},
                 "camera_intrinsics": {...}, "point": null } }
```

A call carries one target, because `AppState`'s setters are coupled and each
coupling is what keeps two panels from ever showing two different files'
selections:

- `select_camera_image` also selects the intrinsics that image was shot through,
  and the reconstruction that owns it.
- `select_camera_intrinsics` clears the camera image, unless that image uses
  those intrinsics.
- `select_reconstruction` drops finer selections belonging to other
  reconstructions.

The joint states are reached by composing calls. A camera image and a point can
be selected together when they belong to one reconstruction:
`select_camera_image` then `select_point` gets there, since the second filters
the image only on the reconstruction and so keeps it. Either order works.

`clear_selection` exists because deselecting is not `select_*` with a null, and
because selection is *visible*: a selected image tints its frustum cyan and a
selected point turns its track orange, so an agent that wants a clean render
before `screenshot` needs a way to say so. Its `scope` follows the viewer's own
Esc behaviour — `"camera_image"` drops the camera image and keeps the intrinsics
(dismissing a photograph says nothing about the lens), `"all"` drops everything.

### `set_reconstruction_display`

```jsonc
{ "reconstruction_label": "global", "visible": false }
{ "reconstruction_label": "global", "tint": "Sky Blue" }  // TINT_PALETTE name, or null
{ "reconstruction_label": "global", "show_points": true, "show_camera_images": false }
```

Every field is one of `SceneNode`'s own, and every omitted one is left alone.
`tint` takes a palette name from `scene::TINT_PALETTE` — an unknown name is an
error listing the seven, rather than a free colour, for the reason the palette
is fixed in the first place.

### `set_solo`

```jsonc
{ "reconstruction_label": "global" }   // draw only this one
{ "reconstruction_label": null }       // end the solo
```

Solo is one scene-level value — `AppState::solo`, an `Option<ReconId>` — so the
tool sets one scene-level value. At most one reconstruction is soloed at a time,
and soloing a second **moves** the solo: "show only this one" has one answer.
Naming the reconstruction to solo, or `null` to end it, says that directly, and
keeps a scene-wide effect out of a call that names a single reconstruction.

Two properties an agent has to know, both of them deliberate in the viewer:

- **Solo is independent of selection.** It changes what is *drawn* and nothing
  else, so it leaves `selected_reconstruction` and every finer selection exactly
  as they were. Soloing `global` while a camera image of `seoul_bull` is
  selected is a normal state, and the Image Detail panel goes on showing that
  image while the 3D viewport shows only `global`.
- **Solo does not touch the eyes.** `SceneNode::visible` is never written by
  soloing, so ending a solo restores precisely the visibility the user had —
  including reconstructions they had already hidden by hand. Effective
  visibility is the composition `visible && (solo is none or solo is me)`, which
  `scene::is_visible` owns and `get_scene` reports per reconstruction as
  `display.drawn`.

`set_solo` sets rather than toggles, unlike the GUI's `AppState::toggle_solo`. A
toggle is fine for a click, where the user can see the current state; an agent
issuing one cannot know the outcome without reading the scene first, and a
retried call would undo itself.

### The view block

The block reports the viewport camera's stored state, field for field, and puts
everything computable from it under `derived`.

`ViewportCamera` holds a `sfmtool_core::Camera` — a `position`, an `orientation`
quaternion (world→camera), and a `target_distance` along the camera's forward
(−Z) axis — plus `world_up`, `fov` and `near`. Those six are the state; `target`,
`forward` and `up` all fall out of them, as `position + forward · target_distance`
and two rotations of the quaternion's axes. Keeping the two groups apart tells a
reader which fields to write back and which are there to save them the
arithmetic.

**`world_up` carries the roll.** The navigation code maintains `orientation` as a
function of the forward direction and `world_up`
(`set_orientation_from_forward`), and `ViewportCamera::tilt` rolls the view by
rotating `world_up`. So `world_up` is live view state that changes as the user
tilts, and it is what makes `position` + `target` + `up` a complete description
of the camera.

**`fov_short_axis_deg` is the field of view of the shorter viewport dimension** —
vertical in a landscape window, horizontal in a portrait one — which is how
`ViewportCamera::fov` keeps the amount of scene on screen steady as the window is
reshaped. The name states the axis because the value moves between axes; the two
fixed ones a person thinks in are under `derived`, next to the `viewport_px` they
are computed against.

`near` is reported and **not settable**: `update_clip_planes` recomputes it every
frame from the scene bounds.

### `set_view`

The tool an agent calls immediately before `screenshot`.

```jsonc
{ "fit": "scene" }                          // Z / zoom-to-fit, everything visible
{ "fit": "global" }                         // frame one reconstruction, by label
{ "look_through": { "reconstruction_label": "seoul_bull", "camera_image": 3 } }
{ "exit_camera_view": true }

// look-at form: intuitive, and enough to determine the camera
{ "position": [2, -3, 1], "target": [0, 0, 0], "up": [0, 0, 1] }

// exact form: the stored state, so a view read from get_scene round-trips
{ "position": [2, -3, 1], "orientation_wxyz": [0.92, 0.39, 0, 0],
  "target_distance": 4.0, "world_up": [0, 0, 1] }

// either form may carry it, and it may also be sent alone
{ "fov_short_axis_deg": 50 }
```

The two explicit forms are exclusive. The look-at form takes `up` as the roll,
defaulting to the current `world_up`; a different one re-rolls the view exactly
as `tilt` does. The exact form restores a view verbatim, which is what
`orientation_wxyz` and `target_distance` are reported for.

`fit` and `look_through` go through the same paths the keyboard and
double-click use (`ViewportCamera::zoom_to_fit` over `scene::world_points`,
`Viewer3D::camera_view`), so the agent's framing is the framing a human gets.

**The animated transition is skipped for MCP-driven view changes.** `Viewer3D`
eases the camera over several frames; an agent that sets the view and screenshots
immediately would photograph the middle of the ease. MCP view commands jump.

### `screenshot`

```jsonc
// { "max_dimension": 1024 }   optional; omit for the viewport's native size
```

Returns an MCP `ImageContent` block: base64 PNG, `mimeType: "image/png"`, plus a
text block naming the pixel size and what is in frame (node count, selection).

**It is the 3D viewport, not the window.** The viewport already renders into an
offscreen texture that MCP can copy from; the window is a surface texture, and
copying that back requires configuring the surface with
`TextureUsages::COPY_SRC` and compositing egui's output — a bigger change for a
picture whose interesting part is the 3D view. Full-window capture is a later
question (§ "Open questions").

Mechanics: `scene_renderer::sizing` creates the final `edl output` texture with
`RENDER_ATTACHMENT | TEXTURE_BINDING`; it gains `COPY_SRC`. The copy-to-buffer
and map-read follow `scene_renderer/readback.rs`, including its 256-byte
row-alignment rule, and `image` (already a dependency) encodes the PNG. The
texture is `Rgba8UnormSrgb`, so the bytes are already display-ready and no
conversion is needed.

`screenshot` is the one tool that cannot answer in the frame it arrives in
(§ "Threading").

## Addressing

The wire vocabulary is chosen so that an id an agent reads in one reply is an id
it can send in the next, and so that ids stay meaningful when the human does
something in the GUI in between.

**Reconstructions are addressed by label.** `scene::unique_label` already
guarantees labels are unique across the scene — that is why it exists — and a
label survives `Reload from Disk`, which mints a fresh `ReconId`. An agent
holding `"global"` still holds `"global"` after the human refreshes the file; an
agent holding `ReconId(4)` holds nothing.

The label is the whole of it: unique across the scene, stable across a reload,
and the only reconstruction handle the wire carries in either direction. The
`ReconId` stays inside the process.

To tell whether the data under a label changed, an agent compares
`content_hash`, which `get_scene` reports for every reconstruction. That
identifies the *contents*, which is the question worth asking — a reload of the
same file leaves it untouched, and a different file changes it.

**Camera images are addressed by index or by name**, and `list_camera_images`
returns both. The name is the `.sfmr` relative path and is what appears in every
other tool's output, so an agent that grep'd a log for a filename can use it
directly.

**Points are addressed by `pt3d_<hash>_<index>` or by bare index**, per
`goto_point`.

**Intrinsics records are addressed by index only** — a `CameraIntrinsics` has no
name to address it by. Being the one handle it has is why the argument can be
`camera_intrinsics_index` at all: with a single spelling, the field can say
which attribute it carries, where `camera_image` cannot.

A ref to a reconstruction that has closed, or an index past the end, is a tool error
naming what is actually loaded — never a silent no-op. Refs go stale here for
exactly the reason they go stale inside the viewer, and the reply says so.

## Threading

The central constraint, and the reason the surface is a *command vocabulary*
rather than a view onto `AppState`:

> **Application state stays single-threaded on the GUI thread.** No
> `Arc<Mutex<AppState>>`, no shared read handle. Every MCP command is applied at
> exactly one point in the frame.

`AppState`, `Viewer3D` and the renderer are threaded through the frame as
`&mut`, and the existing panels rely on that — `TabContext` hands out seven
simultaneous `&mut` borrows, the SIFT and full-res caches are split-borrowed
against the scene on purpose. Wrapping any of it in a lock to let an HTTP thread
peek would either deadlock against those borrows or force a redesign of half the
crate. It would also introduce the one bug class this design refuses to have: an
agent observing the scene mid-mutation.

So the server never touches app state. It builds a command, hands it to the GUI
thread, and waits for the answer.

```
 HTTP thread (tokio)                    GUI thread (winit)
 ───────────────────                    ──────────────────
 tools/call arrives
   │
   ├─ Command + oneshot::Sender ──────► mpsc::Receiver
   ├─ proxy.send_event(McpRequest) ───► event loop wakes, request_redraw
   │                                      │
   │                                    run_ui_and_paint
   │                                      ├─ drain_mcp ◄── applies every queued command
   │                                      ├─ prepare_uploads
   │                                      ├─ render_scene
   │                                      ├─ run_egui_pass
   │                                      └─ process_readback ── resolves deferred (screenshot)
   ◄── oneshot reply ──────────────────────┘
   │
 tools/call returns
```

Four things this buys, each load-bearing:

- **Commands land before uploads.** `drain_mcp` runs first in
  `run_ui_and_paint`, ahead of `prepare_uploads`, so a command's effect is in
  the very frame the agent's request woke. An agent can `set_view` then
  `screenshot` and get the new view.
- **Waking is already solved.** `UserEvent` grows one variant, `McpRequest`;
  `EventLoopProxy` is `Send + Clone`, and `App::user_event` already calls
  `request_redraw`. An idle viewer that renders a couple of frames and stops
  wakes on the first request and goes back to sleep after it.
- **A reply is a snapshot.** It was built at one instant with exclusive access,
  so a `get_scene` can never straddle a load.
- **Screenshots defer honestly.** `drain_mcp` returns
  `Outcome::Deferred` for a command whose answer needs the frame to have
  happened; `App` holds it until the readback phase and replies then. One extra
  frame of latency, stated rather than papered over.

**Every reply is timeout-bounded** — 10 s on the HTTP side. The GUI thread can
legitimately stop pumping (a modal `rfd` file dialog is open, the user is
dragging the window on Windows), and an agent must get "the viewer is busy"
rather than a hung connection.

**Every mutating command writes `AppState::status_message`**, prefixed:
`MCP: opened global.sfmr`. The human watching the window sees what the agent
did, in the place the viewer already reports what it did.

## The Rust seam

One new module, `crates/sfm-explorer/src/mcp/`, split so the interesting half is
testable without a window:

```rust
/// Everything the MCP surface can ask the viewer to do. One variant per tool.
///
/// A reconstruction is named by its label, so these carry a `String` that
/// `apply` resolves against `AppState::scene`. `Option` means "the selected
/// reconstruction if omitted".
pub enum Command {
    GetScene,
    ListCameraImages {
        reconstruction_label: Option<String>,
        offset: usize,
        limit: usize,
    },
    GetCameraImage {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    GetCameraIntrinsics {
        reconstruction_label: Option<String>,
        camera_intrinsics_index: usize,
    },
    GetPoint { point: PointSel },
    OpenReconstruction { path: PathBuf },
    CloseReconstruction { target: CloseTarget },
    SelectReconstruction { reconstruction_label: String },
    SelectCameraImage {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    SelectCameraIntrinsics {
        reconstruction_label: Option<String>,
        camera_intrinsics_index: usize,
    },
    SelectPoint { point: PointSel },
    ClearSelection { scope: SelectionScope },
    SetReconstructionDisplay {
        reconstruction_label: String,
        change: DisplayChange,
    },
    SetView { view: ViewCommand },
    Screenshot { max_dimension: Option<u32> },
}

/// A tool's answer: the JSON body, or a message for `isError: true`.
pub type Reply = Result<serde_json::Value, ToolError>;

/// Whether `apply` finished the job, or needs the frame to complete first.
pub enum Outcome {
    Done(Reply),
    Deferred(Deferred),
}

/// Apply one command. **Takes no `App` and no GPU handle** — which is what
/// makes fourteen of the fifteen tools testable in a headless `cargo test`.
pub fn apply(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome;

/// Start the server. Returns once it is bound and listening, or with the bind
/// error; the runtime lives on its own thread from here.
pub fn serve(port: u16, tx: Sender<Request>, proxy: EventLoopProxy<UserEvent>)
    -> Result<SocketAddr, ServeError>;
```

`apply` taking `(&mut AppState, &mut Viewer3D)` rather than `&mut App` is the
one signature worth defending. `App` owns a `wgpu::Device`, a surface and a
window; a test that could construct one would need a GPU and a display, and the
crate's headless lib tests deliberately need neither. Splitting the GPU-shaped
command out into `Outcome::Deferred` — handled in `app.rs`, where the device
already is — keeps the whole command vocabulary, its error messages and its JSON
shapes under headless test, and leaves exactly one tool (`screenshot`) needing a
window.

`App` grows three fields: `mcp_rx: Option<Receiver<Request>>`, a
`Vec<Deferred>`, and the request counter the Scene panel shows.

## Transport and protocol

**Streamable HTTP, via the official Rust SDK ([`rmcp`][rmcp], 3.x), mounted on
`axum`, on a `tokio` current-thread runtime on one dedicated thread.**

The SDK is here to absorb the protocol's revision history. MCP has run
2024-11-05 → 2025-03-26 → 2025-06-18 → 2025-11-25 → 2026-07-28, adding and then
removing sessions, the GET stream, `Last-Event-ID` resumability and
server-initiated requests along the way, and clients in the field speak several
of those. Keeping up with that is `rmcp`'s job, and it is what buys a viewer that
still works after somebody upgrades their client.

The current revision is genuinely simple — a single POST endpoint, no sessions,
no `initialize` handshake, and no SSE needed at all for a server that answers
every request with `application/json` — so the first cut implements exactly that
shape and lets the SDK handle the older eras behind it.

The cost is a dependency tree: `tokio`, `hyper`, `axum`, `rmcp`, in a workspace
with no async runtime today, compiled from the PyPI **sdist** on the user's own
rustc (wheels ship for Linux and Windows only, so macOS users build it). Beside
`wgpu`, `winit`, `egui` and `image`, which `sfm-explorer` already pulls in, it is
a modest addition — and it goes behind a Cargo feature so it can be dropped:

```toml
[features]
default = ["mcp"]
mcp = ["dep:rmcp", "dep:tokio", "dep:axum", "dep:serde", "dep:serde_json"]
```

Default on, so the flag works in a stock build; `--no-default-features` drops it
for anyone trading the feature against build time. In a build without it, `--mcp`
is rejected at startup with a message naming the flag it needs.

Pin `rmcp` to an exact minor and check at implementation time which protocol
revisions that version negotiates; its coverage of the pre-2026 eras is the
whole reason it is here.

[rmcp]: https://crates.io/crates/rmcp

### HTTP on loopback, not stdio

The value of this surface is attaching to the window the human already has open,
which is what a listening socket gives: the viewer runs, an agent connects and
disconnects as it likes, and the human watches the same window throughout. That
is the shape desktop applications with in-process MCP servers use — the Figma
desktop app hosts `http://127.0.0.1:3845/mcp`.

MCP's other transport, stdio, has the client launch the server as a child
process. It suits a tool with no life of its own, where the agent's session and
the server's are the same session; a GUI the user started, and whose stdout is
shared with logging, is the other case.

## Security

The endpoint hands out read access to any `.sfmr` path the process can read
(`open_reconstruction` takes a path) and control of a window on the user's
desktop. Both are appropriate for a tool the user explicitly started with a
flag, and neither is appropriate for anything reachable from outside the
machine.

1. **Off by default.** No flag, no listener, no port. This is the primary gate
   and the reason the rest can stay simple.
2. **Bind `127.0.0.1` only.** Not `0.0.0.0`, and there is no `--mcp-host` flag
   to make it otherwise. Adding one is a deliberate future decision with its own
   authentication story, not a convenience.
3. **Validate `Origin`.** Present and not a loopback origin → `403 Forbidden`,
   per the transport spec. This is what stops a web page the user has open from
   driving their viewer through DNS rebinding.
4. **No write path to disk.** No tool in this surface saves an `.sfmr`, exports
   anything, or deletes a file. `close_reconstruction` unloads a reconstruction.
5. **The window announces it.** Title suffix and Scene panel header, always,
   while the server is live.

`open_reconstruction` reads any path the process can, matching the viewer's own
File ▸ Open. The gate on that is the flag the user typed, and the docs say so.

## Errors

Two levels, and the distinction matters to a client:

- **Protocol errors** (JSON-RPC `error`): malformed request, unknown method,
  arguments that fail the tool's `inputSchema`. The SDK produces these.
- **Domain errors** (`CallToolResult` with `isError: true`): everything the
  viewer refuses. An unknown reconstruction label, a camera image index out of
  range, an unreadable `.sfmr`, an unknown tint name, an unrecognized
  `clear_selection` scope, a viewport that has not rendered yet so there is
  nothing to screenshot, and the 10 s apply timeout.

Domain errors get a message an agent can act on, in the style the viewer already
uses for its status line: what was asked, what is actually there. *"No loaded
reconstruction is labelled `globl` — loaded: `seoul_bull`, `global`."*

## Testing

`crates/sfm-explorer/src/mcp/tests.rs`, headless, no GPU, no window — which is
what the `apply(&mut AppState, &mut Viewer3D, …)` signature is for. The existing
`test_support` fixtures build scenes already.

- **Every command against a two-reconstruction scene**: the JSON shape, the ids, and the
  `selection` block each of the five `select_*` / `clear_selection` tools
  returns.
- **The selection invariants survive the boundary**: selecting a camera image
  sets its intrinsics; selecting an intrinsics record the selected camera image
  does not use clears that camera image; selecting a reconstruction drops
  another's point. These are `AppState`'s guarantees and the point of the test
  is that MCP cannot route around them.
- **`select_camera_image` then `select_point` in one reconstruction leaves both
  selected** — the composition the joint selection states are reached by.
- **Stale and out-of-range refs** produce errors naming what is loaded, for
  every ref-taking tool.
- **Label addressing survives a reload**: open, reload (new `ReconId`), and the
  same label still resolves.
- **Schema round-trip**: every `Command` deserializes from the tool's advertised
  `inputSchema`, so a schema and its parser cannot drift.

Two things need more than that. A **protocol conformance test** starts the
server on port 0 against a scene, drives it with an HTTP client, and checks the
handshake, `tools/list`, one `tools/call`, and the `Origin` rejection.
**`screenshot`** needs a real frame and belongs in `ui_basic` (Windows/macOS,
`pixi run ui-test`) — assert a decodable PNG of the expected size, not its
pixels.

## Phase 2: editing the reconstruction

The ask that motivated this includes changing camera intrinsics. It is
deliberately not in the first cut, and the reason is a viewer-wide invariant
worth stating before anything breaks it:

> **Nothing in the viewer mutates a loaded `SfmrReconstruction`.** Per-node
> `transform` is view state that reaches the GPU as a model matrix and never
> touches the reconstruction. Even `Align to…` only writes the node's transform,
> and `Resect Image` — which genuinely computes new geometry — publishes it as a
> *new derived node* beside the source rather than editing it.

Editing intrinsics in place would break that, and the breakage is not
theoretical: the content hash stops describing the contents, so every
`pt3d_<hash>_<index>` id in flight starts naming a reconstruction that no longer
exists; every frustum, distorted mesh and image quad built from those intrinsics
needs re-upload; and the human has no way to tell an edited node from a loaded
one, or to get back.

The answer the codebase already has is `resect_image`'s. A phase-2
`set_camera_intrinsics` **produces a derived node**:

```jsonc
// { "reconstruction_label": "seoul_bull", "camera_intrinsics_index": 0,
//   "params": { "fx": 410.0, "k1": -0.028 } }
{ "label": "seoul_bull (intrinsics 0 edited)", "replaced": false }
```

— named for its provenance, inheriting the source's transform so it lands
exactly on top, replaced in place when the same edit is repeated, and leaving
the source untouched. The agent then flips `set_reconstruction_display` between the two,
or screenshots both. Saving it is `sfm xform`'s job, as it already is.

That is a real design with real work behind it (partial-parameter merge against
`CameraModel`, re-upload invalidation, the derived-node lifecycle) and it should
land as its own change once the read-and-navigate surface has proven itself.

### Loose images, and the names held for them

The other reason the first cut is careful about vocabulary. The flow worth
building toward is: open a reconstruction, load a folder of images that are
*not* in it, browse them against the structure, and resect the ones that fit —
which is `Resect Image`'s existing question asked of a picture that has no pose
yet rather than one whose pose is in doubt.

Nothing here implements it. What this spec does is **not spend the names**:

| Tool | What it would do |
|------|------------------|
| `open_images` | Load a folder or a list of files as loose images |
| `list_images` | The loose set — no pose, no reconstruction, no track |
| `get_image` | One loose image: path, dimensions, EXIF, any `.sift` beside it |
| `select_image` | Select one for the Image Detail panel |
| `resect_image` | Estimate a pose against a named reconstruction, producing a derived reconstruction the way `set_camera_intrinsics` does |

Each of those is the short name of an entity that genuinely has no
reconstruction behind it, sitting beside the `camera_*` tool that names the
posed thing. Spend `list_images` on camera images and every one of them has to
be named around the collision instead — or the surface has to break
compatibility to reclaim it.

Other phase-2 candidates, in rough order of value:

- **MCP resources**, one per loaded reconstruction
  (`sfmr://seoul_bull/summary`), so a client can attach scene state as context
  without a tool call.
- **`subscriptions/listen` notifications** when the human changes the selection
  or opens a file, so an agent watching alongside a human stays in step. This is
  the one feature that needs SSE, which is why the first cut can answer every
  request with `application/json`.
- **Full-window screenshot**, including the panels (§ "screenshot").
- **`run_align`**, exposing the Scene panel's `Align to…` — a real operation
  with a real answer, and the natural next verb after the read surface.

## Non-goals

- **Not a data channel.** No tool returns point arrays, descriptors, thumbnails
  or track tables in bulk. The agent reads `.sfmr` files with `sfm inspect`; it
  asks the viewer about *state*.
- **Not a replacement for the CLI.** Solving, matching, transforming and
  exporting stay `sfm` subcommands. The viewer's MCP surface drives the viewer.
- **No remote access.** Loopback only, no auth, no TLS. A viewer on another
  machine is a different problem.
- **No headless mode.** The window is the point. An MCP server with no window
  behind it would be a worse `sfm inspect`.
- **No persistence.** The endpoint is not remembered between runs, and neither
  is anything an agent did through it.

## Open questions

- **The default port.** 8787 is arbitrary. A fixed default makes client config
  static (Figma's 3845 is the precedent); the cost is that two viewers cannot
  both take it, hence `--mcp 0`. Is a fixed default worth it, or should `--mcp`
  require a port?
- **Discovery for multiple viewers.** With ephemeral ports, an agent has no way
  to find the running instances. A registry file — `~/.sfmtool/explorer-mcp.json`,
  written on bind and removed on exit — would solve it, at the cost of a stale
  file after a crash. Worth it only once running two viewers at once is common.
- **Should the human be able to turn it off?** A `Help ▸ MCP` menu item showing
  the endpoint and a Stop button costs little and is the honest counterpart to
  the title-bar announcement. Deferred only because it is not needed to prove
  the design.
- **Screenshot resolution policy.** Native viewport size can be a 4K PNG, which
  is a lot of tokens. `max_dimension` defaulting to something like 1280 might
  serve agents better than defaulting to native — but a downscaled screenshot is
  a worse answer to "is this point cloud noisy". Decide with a real agent in the
  loop.
- **Whether `set_view` should expose the HUD's display controls** (point size,
  EDL thickness, patch opacity). They change what a screenshot shows, so an
  agent evaluating a reconstruction may want them; they are also a long tail of
  knobs that would double the tool's surface. Left out until something asks.
