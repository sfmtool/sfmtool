# SfM Explorer MCP Server

## Purpose

SfM Explorer is a viewer you drive by hand: you open reconstructions, click a
camera, look at where its rays land, notice that one image is in the wrong
place, and go and fix it somewhere else. An AI coding agent working on this
repository cannot do any of that. It can read `.sfmr` files with `sfm inspect`
and it can read the source, but it cannot see the picture, and it cannot ask
the picture a question — which means the one tool that would tell it fastest
whether a solve is wrong, and *where*, is the one tool it cannot reach.

This is an opt-in control surface for the running viewer, speaking the
[Model Context Protocol][mcp] (MCP) over a loopback HTTP endpoint. Started with
`sfm-explorer --mcp`, the viewer hosts a small server; an agent connects to it
and can then enumerate the loaded scene graph, open and close `.sfmr` files,
move the selection and the 3D camera, and take a screenshot of the viewport.
The human keeps the window in front of them the whole time and watches it
change.

The surface is deliberately narrow. The viewer's own invariants
(§ "Addressing", § "Threading") are what it is shaped around, not the other way
round, and it edits no reconstruction data — the intrinsics case in particular
runs into a viewer-wide invariant with an answer of its own (§ "Editing
reconstruction data").

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

`crates/sfm-explorer/src/cli.rs` recognizes `--mcp`, `--mcp=PORT`, `--mcp PORT`,
`--help`, and treats everything else as a path. Hand-rolled rather than `clap`:
one flag and a list of paths is a dozen lines, and it keeps the binary's
dependency tree as it was. The following-argument form has to look at what comes
next, because `--mcp scene.sfmr` is the common invocation and means the default
port and a file — so a next argument that is not a port is left alone rather
than consumed.

On startup the server binds and prints one line to stdout, which is what a
human pastes into a client config:

```
SfM Explorer MCP endpoint: http://127.0.0.1:8787/mcp
```

**A bind failure is fatal and loud.** Two viewers on one port is the common
mistake, and a viewer that silently came up without the endpoint the agent was
told to use is worse than one that refused to start. The message names the port
and suggests `--mcp 0`. Binding happens on the calling thread, before the server
thread is spawned, which is what lets the error reach the caller at all.

**The window says so.** While the server is live the title carries a suffix —
`SfM Explorer - scene.sfmr [MCP :8787]` — and the Scene panel grows a header
line with the endpoint and a live request count, above the tree and above its
empty state. A window that something else can drive should never look like one
that nothing can. `AppState::mcp` holds the port and the count, so
`AppState::window_title` and the panel read the same thing without either
knowing the transport exists.

### Connecting an agent

For Claude Code, with the viewer already running:

```bash
claude mcp add --transport http sfm-explorer http://127.0.0.1:8787/mcp
```

**The endpoint has to be listening when the client starts, and only then.** A
client does its handshake and fetches `tools/list` once, at startup; a viewer
that was not up yet registers as a failed server with no tools. But because the
transport is stateless — no session, no standing connection, every call a fresh
POST — a viewer restarted afterwards is picked up with no reconnection at all,
since the next call simply lands in the new process. So an agent that has the
tools can own the viewer's lifecycle from then on: kill it, rebuild, relaunch on
the same port, keep calling. Launching it detached from the agent's own session
is what keeps it alive across the client restart that registered it in the first
place.

## The tool surface

Sixteen tools. Five read, ten write, one that closes the loop by handing back a
picture.

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

Every tool is annotated: the five reads and `screenshot` carry
`readOnlyHint: true`, the ten writes `destructiveHint: false` (nothing here
touches a file on disk — `close_reconstruction` unloads, it does not delete),
and every one of them `openWorldHint: false`. Every `inputSchema` is closed
(`additionalProperties: false`).

### The wire vocabulary

One rule, applied without exception: **one entity, one spelled-out word, in tool
names, arguments and reply fields alike.** No abbreviations, and no word that
names two things.

That costs a few characters per call and buys the only thing that matters here
— an agent reading `tools/list` can tell what a tool addresses without reading
prose, and a human reading the agent's transcript can tell what it did. Tool
names and argument names *are* the API: they live in client configs and in the
prompts people write against them, so a name is far more expensive to change
than to choose. `mcp::tests::the_wire_vocabulary_holds_across_the_catalog`
asserts the rule over the whole catalog rather than leaving it to review.

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
held for the pictures with no pose yet (§ "Loose images, and the names held for
them"), and cost six characters to keep free.

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
and its schema carries the union, so either spelling is a valid call.

#### One spelling per handle, except where two spell one identity

A reconstruction has one handle, its label, so `reconstruction_label` takes a
label and the field says so (§ "Addressing" for why the label is the handle).

A camera image and a point each have two — an index or a name, an id or a bare
index — and an agent arrives holding whichever the surface last handed it. Those
fields take the entity's name and their schema carries the union, so both
spellings are valid calls. A union of *representations* reads as one question
with two phrasings; a union of *intents* is the thing the five separate
selection tools, and `set_view`'s exclusive forms, keep out of a single tool.

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
      "display": { "visible": true, "drawn": true, "interactive": true,
                   "show_points": true, "show_camera_images": true,
                   "show_patches": true, "show_points_at_infinity": true,
                   "tint": null },
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
  "solo": null,                           // a reconstruction label, or null
  "view": {                               // see § "The view block"
    "position": [1.20, -4.40, 2.00],      // Camera::position
    "orientation_wxyz": [0.92, 0.39, 0.00, 0.00],   // world→camera attitude
    "target_distance": 4.60,              // along forward, i.e. camera −Z
    "world_up": [0.0, 0.0, 1.0],          // navigation up; carries roll
    "fov_short_axis_deg": 45.0,           // the *shorter* viewport dimension
    "near": 0.081,                        // adaptive, recomputed every frame
    "derived": {                          // all recoverable from the above
      "target": [0.00, 0.00, 0.30],       // position + forward · target_distance
      "forward": [-0.26, 0.87, -0.42],
      "up": [-0.12, 0.40, 0.91],
      "viewport_px": [1280, 720],
      "fov_horizontal_deg": 73.7,         // null before the panel is laid out
      "fov_vertical_deg": 45.0
    },
    "looking_through": null               // a camera image, in camera-view mode
  },
  "status_message": null,
  "window_title": "SfM Explorer - seoul_bull.sfmr [MCP :8787]"
}
```

`counts` is summary only. A 4 M-point cloud must never cross this boundary as
JSON, and no tool here returns bulk arrays — that is what the `.sfmr` file and
`sfm inspect` are for. The agent reads the file for the data and asks the
viewer for the *state*. `points_at_infinity` is read the way
`scene::visible_stats` reads it, so this number and the one in the viewport's
stats overlay are the same number.

**`display` reports `visible` and `drawn` both.** `visible` is the node's own
master eye; `drawn` is the composition `visible && (no solo, or the solo is me)`
that `scene::is_visible` owns and the draw loop uses. A reconstruction hidden by
hand and one hidden by another's solo look identical in the viewport, and only
the pair says which is which — which matters because ending a solo restores the
first and not the second.

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
{ "reconstruction_label": "seoul_bull", "index": 3,
  "name": "images/IMG_0042.jpg",
  "camera_intrinsics": { "index": 0, "model": "OPENCV",
                         "width": 270, "height": 480 },
  "quaternion_wxyz": [0.98, 0.01, -0.17, 0.04],
  "translation_xyz": [0.10, -1.88, 0.51],
  "center": [0.28, -1.85, 0.49],
  "observations": 1187,
  "reproj_error": { "mean": 0.61, "median": 0.48, "p95": 1.72 } }

// get_camera_intrinsics { "reconstruction_label": "seoul_bull",
//                         "camera_intrinsics_index": 0 }
{ "reconstruction_label": "seoul_bull", "index": 0,
  "model": "OPENCV", "width": 270, "height": 480,
  "params": { "focal_length_x": 402.1, "focal_length_y": 402.1,
              "principal_point_x": 135.0, "principal_point_y": 240.0,
              "radial_distortion_k1": -0.031, "radial_distortion_k2": 0.004,
              "tangential_distortion_p1": 0.0, "tangential_distortion_p2": 0.0 },
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

`list_camera_images` defaults to 50 rows and caps at 500 (`read::MAX_LIMIT`).
An `offset` past the end is an empty page and not a refusal: a caller walking a
reconstruction should learn it has reached the end from `total` and an empty
array, not from an error it has to tell apart from a real one.

**The reprojection figures come from the same source the panels read**, so a
number the agent is told matches the colour the human is looking at.
`get_camera_image`'s summary is `compute_observation_reprojection_errors`, which
the Image Detail panel's error heatmap uses; a track observation's is
`point_track_detail::compute_observation_metrics`, widened to `pub(crate)` for
exactly this reason.

Both can be absent, and say so rather than inventing a number.
`get_camera_image`'s `reproj_error` is `null` when the errors cannot be
computed — the source reads the image's `.sift` file, which an
`embedded_patches` reconstruction does not have, and which a `sift_files` one
whose workspace has moved cannot find. A track observation's `reproj_error` is
`null` where the point falls behind that camera, which the metric reports as
`NaN` and JSON cannot carry.

`params` is a name→value map rather than the model's positional parameter
vector, because a positional vector is unreadable without also shipping the
model's parameter order and an agent will get it wrong. The order is
`CameraIntrinsics::parameters`' declaration order — the order `sfm inspect`
prints and the Intrinsics panel shows, so the three can be diffed against each
other.

`get_point` accepts either shape `goto_point::parse_point_query` accepts — a
bare index against the selected reconstruction, or a full `pt3d_<hash>_<index>`
id that names its own — and resolves it through `resolve_point_query`. One
parser, one set of error messages, and a point id copied out of the Point Track
panel by a human pastes straight into a tool call. A bare JSON integer is
accepted as the index form, since that is what a caller reading an index out of
a track will naturally send.

### `open_reconstruction` / `close_reconstruction`

```jsonc
// open_reconstruction { "path": "C:/work/global.sfmr" }
// -> the new reconstruction's entry, as get_scene renders it, plus `reloaded`
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
`load_file` reports failure by writing `AppState::status_message`, so that is
where the message is read back from and turned into a refusal.

`close_reconstruction` takes `reconstruction_label` or `all: true`, and refuses
both at once: "close this one" and "close everything" are different requests,
and a call carrying both has no answer.

### The six selection tools

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

All six return the same thing — the resulting `selection` block, exactly as
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
rule — `"camera_image"` drops the camera image and keeps the intrinsics
(dismissing a photograph says nothing about the lens, per
`AppState::select_image`), `"all"` drops everything.

### `set_reconstruction_display`

```jsonc
{ "reconstruction_label": "global", "visible": false }
{ "reconstruction_label": "global", "tint": "Sky Blue" }  // TINT_PALETTE name, or null
{ "reconstruction_label": "global", "show_points": true, "show_camera_images": false }
```

Every field is one of `SceneNode`'s own, and every omitted one is left alone.
The reply is the reconstruction's scene entry, so the agent reads back the whole
display state rather than the fields it happened to set.

`tint` takes a palette name from `scene::TINT_PALETTE`, or `null` for the
reconstruction's own colours; an unknown name is an error listing the seven,
rather than a free colour, for the reason the palette is fixed in the first
place. The tint is resolved **before** any field is written, so a refused call
has not applied the rest of itself on the way out.

### `set_solo`

```jsonc
{ "reconstruction_label": "global" }   // draw only this one
{ "reconstruction_label": null }       // end the solo, as does omitting it
```

Solo is one scene-level value — `AppState::solo`, an `Option<ReconId>` — so the
tool sets one scene-level value. At most one reconstruction is soloed at a time,
and soloing a second **moves** the solo: "show only this one" has one answer.
Naming the reconstruction to solo, or `null` to end it, says that directly, and
keeps a scene-wide effect out of a call that names a single reconstruction. The
reply carries the new `solo` and the whole `scene`, because a solo changes
`display.drawn` on every entry at once.

Two properties an agent has to know, both of them deliberate in the viewer:

- **Solo is independent of selection.** It changes what is *drawn* and nothing
  else, so it leaves `selected_reconstruction` and every finer selection exactly
  as they were. Soloing `global` while a camera image of `seoul_bull` is
  selected is a normal state, and the Image Detail panel goes on showing that
  image while the 3D viewport shows only `global`.
- **Solo does not touch the eyes.** `SceneNode::visible` is never written by
  soloing, so ending a solo restores precisely the visibility the user had —
  including reconstructions they had already hidden by hand.

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
are computed against. Both are `null` before the 3D panel has been laid out
once, since there is no aspect ratio to compute them against.

`near` is reported and **not settable**: `update_clip_planes` recomputes it every
frame from the scene bounds.

### `set_view`

The tool an agent calls immediately before `screenshot`.

```jsonc
{ "fit": null }                             // frame everything drawn
{ "fit": "global" }                         // frame one reconstruction, by label
{ "look_through": { "reconstruction_label": "seoul_bull", "camera_image": 3 } }
{ "exit_camera_view": true }

// look-at form: intuitive, and enough to determine the camera
{ "position": [2, -3, 1], "target": [0, 0, 0], "up": [0, 0, 1] }

// partial forms: what a call does not carry is preserved
{ "target": [0, 0, 0] }                          // re-centre on a point, orientation kept
{ "target": [0, 0, 0], "forward": [0, 0, -1] }   // view a point from a direction
{ "forward": [0, 0, -1] }                        // view the standing target from a direction
{ "position": [2, -3, 1] }                       // move the camera, orientation kept
{ "target_distance": 12.0 }                      // dolly: same target, new distance

// exact form: the stored state, so a view read from get_scene round-trips
{ "position": [2, -3, 1], "orientation_wxyz": [0.92, 0.39, 0, 0],
  "target_distance": 4.0, "world_up": [0, 0, 1] }

// either explicit form may carry it, and it may also be sent alone
{ "fov_short_axis_deg": 50 }
```

The reply is `{ "view": … }`, the same block `get_scene` embeds.

The forms are **exclusive**, one per call, and the check is up front. These are
*intents* rather than representations: a call carrying both `fit` and `position`
has no answer, and guessing one would move the camera somewhere the agent did
not ask for. The look-at form takes `up` as the roll, defaulting to the current
`world_up`; a different one re-rolls the view exactly as `tilt` does. The exact
form restores a view verbatim, which is what `orientation_wxyz` and
`target_distance` are reported for.

The explicit camera is a position, an orientation and a target distance, and
the explicit family also takes them **a piece at a time**: what a call does not
carry is preserved. The orientation comes from `forward` (a view direction,
named for the derived field the view block reports, with the roll taken from
`up` defaulting to the current `world_up`), from `position` and `target`
together (the look-at form), from `orientation_wxyz` (the exact form), or it
stands. The distance comes from `target_distance`, from the separation of
`position` and `target` when both are given, or it stands. The view is then
anchored: at `target` where the call names one, else at `position` where the
call names one, else at the standing orbit target -- so `forward` alone orbits
the camera around what it is looking at rather than turning it in place, and
`target_distance` alone is a dolly. The remaining end of the view follows from
the anchor, the orientation and the distance.

A call that over- or under-determines the camera is refused rather than
guessed at: `orientation_wxyz` accepts only its exact-form companions,
`position` with `target` fixes the distance so `target_distance` may not ride
along, `forward` may not accompany the pair, and `up` steers a roll only where
the orientation is being derived (`forward`, or the look-at pair). Every form
refuses the arguments it does not read -- an argument silently ignored leaves
the agent believing it asked for something it did not.

`fit` and `look_through` go through the same paths the keyboard and
double-click use (`ViewportCamera::zoom_to_fit` over `scene::world_points`,
`Viewer3D::jump_to_camera_view`), so the agent's framing is the framing a human
gets. Fitting is over *world* points — the reconstruction's own positions put
through its transform — so an aligned reconstruction is framed where it is
drawn. A fit also leaves camera view, exactly as the Z key's fit does at
the end of its animated transition: framing is a statement about the free
camera, and a fit that left the render looking through a camera image would
frame nothing the caller can see.

**The animated transition is skipped for MCP-driven view changes.** `Viewer3D`
eases the camera over roughly 200 ms; an agent that sets the view and screenshots
immediately would photograph the middle of the ease. MCP view commands jump, and
cancel any ease already running, so a change the human started does not slide
over the top of the one the agent asked for. `Viewer3D::jump_to_camera_view` is
`enter_camera_view`'s end state assigned rather than eased toward; the two share
one derivation (`compute_camera_view`) so they cannot drift apart on where a
camera looks from.

### `screenshot`

```jsonc
// { "max_dimension": 1024 }   optional; omit for the viewport's native size
```

Returns an MCP `ImageContent` block — base64 PNG, `mimeType: "image/png"` — plus
a text block naming the pixel size and what is in frame (which reconstructions
are drawn, the point and camera-image counts, and the camera image being looked
through, if any). That caption is built during the *apply* phase, while
`AppState` is still borrowed, so the picture and the description of it are of
the same instant.

**It is the 3D viewport, not the window.** The viewport already renders into an
offscreen texture that can be copied from; the window is a surface texture, and
copying that back requires configuring the surface with `TextureUsages::COPY_SRC`
and compositing egui's output — a bigger change for a picture whose interesting
part is the 3D view. Full-window capture is a later question
(§ "Open questions").

Mechanics, in `scene_renderer/capture.rs`: `scene_renderer::sizing` creates the
final `edl output` texture with `COPY_SRC` alongside its render-attachment and
texture-binding usages, and keeps the texture (not only its view, which cannot
be a copy source). The copy-to-buffer and map-read follow
`scene_renderer/readback.rs`, including its 256-byte row-alignment rule, and
`image` — already a dependency — encodes the PNG. The texture is
`Rgba8UnormSrgb`, so the bytes are already display-ready and no colour
conversion is needed. `max_dimension` downscales with a Lanczos3 filter: what a
downscaled point cloud is asked to answer is "is this noisy", and a cheap
filter's aliasing invents exactly that.

`screenshot` is the one tool that cannot answer during the apply phase
(§ "Threading").

## Addressing

The wire vocabulary is chosen so that an id an agent reads in one reply is an id
it can send in the next, and so that ids stay meaningful when the human does
something in the GUI in between.

**Reconstructions are addressed by label.** `scene::unique_label` guarantees
labels are unique across the scene — that is why it exists — and a label
survives `Reload from Disk`, which mints a fresh `ReconId`. An agent holding
`"global"` still holds `"global"` after the human refreshes the file; an agent
holding `ReconId(4)` holds nothing.

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

A ref to a reconstruction that has closed, or an index past the end, is a tool
error naming what is actually loaded — never a silent no-op. Refs go stale here
for exactly the reason they go stale inside the viewer, and the reply says so.

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
   ├─ Command + oneshot::Sender ──────► mpsc::UnboundedReceiver
   ├─ proxy.send_event(McpRequest) ───► event loop wakes, request_redraw
   │                                      │
   │                                    run_ui_and_paint
   │                                      ├─ drain_mcp ◄── applies every queued command
   │                                      ├─ prepare_uploads
   │                                      ├─ render_scene
   │                                      ├─ run_egui_pass
   │                                      ├─ submit + present
   │                                      ├─ process_pick_readback
   │                                      └─ resolve_mcp_deferred ── screenshots
   ◄── oneshot reply ──────────────────────┘
   │
 tools/call returns
```

Four things this buys, each load-bearing:

- **Commands land before uploads.** `drain_mcp` runs first in
  `run_ui_and_paint`, ahead of the title sync and `prepare_uploads`, so a
  command's effect is in the very frame the agent's request woke. An agent can
  `set_view` then `screenshot` and get the new view.
- **Waking is already solved.** `UserEvent` grows one variant, `McpRequest`;
  `EventLoopProxy` is `Send + Clone`, and `App::user_event` already calls
  `request_redraw`. An idle viewer that renders a couple of frames and stops
  wakes on the first request and goes back to sleep after it. The variant
  carries nothing — the request travels over the channel, and one wake covers
  however many requests arrived alongside it.
- **A reply is a snapshot.** It was built at one instant with exclusive access,
  so a `get_scene` can never straddle a load.
- **Screenshots defer honestly.** `drain_mcp` gets `Outcome::Deferred` back for
  a command whose answer needs the frame to have happened; `App` holds it until
  the readback phase, after the present, and replies there — so the picture is
  the frame the request woke, including whatever else was applied alongside it.
  A frame can bail before that — GPU state not up yet, a surface that could not
  be presented — so a redraw is requested again while anything is still
  deferred. An idle viewer asks for no frames of its own, and without that the
  screenshot would sit until the caller's timeout rather than until the next
  frame.

**Every reply is timeout-bounded** — 10 s on the HTTP side. The GUI thread can
legitimately stop pumping (a modal `rfd` file dialog is open, the user is
dragging the window on Windows), and an agent must get "the viewer is busy"
rather than a hung connection.

**Every mutating command writes `AppState::status_message`**, prefixed:
`MCP: opened global.sfmr`. The human watching the window sees what the agent
did, in the place the viewer already reports what it did, and can tell it from
something they did themselves.

## The Rust seam

`crates/sfm-explorer/src/mcp/`, in three layers so the interesting one is
testable without a window:

| Module | What it owns |
|--------|--------------|
| `tools` | The tool table and the wire parse: names, descriptions, `inputSchema`, and JSON arguments to a `Command` |
| `mod` + `read` / `write` / `view` / `render` | The command vocabulary, applied to `(&mut AppState, &mut Viewer3D)` |
| `frame` | The two phases `run_ui_and_paint` calls: the drain, and the deferred screenshot |
| `server` | The `rmcp` handler and the `axum` / `tokio` plumbing |

```rust
/// Everything the MCP surface can ask the viewer to do. One variant per tool.
///
/// A reconstruction is named by its label, so these carry a `String` that
/// `apply` resolves against `AppState::scene`. `Option` means "the selected
/// reconstruction if omitted".
pub(crate) enum Command {
    GetScene,
    ListCameraImages { reconstruction_label: Option<String>, offset: usize, limit: usize },
    GetCameraImage { reconstruction_label: Option<String>, camera_image: CameraImageSel },
    GetCameraIntrinsics { reconstruction_label: Option<String>, camera_intrinsics_index: usize },
    GetPoint { point: goto_point::PointQuery },
    OpenReconstruction { path: PathBuf },
    CloseReconstruction { target: CloseTarget },
    SelectReconstruction { reconstruction_label: String },
    SelectCameraImage { reconstruction_label: Option<String>, camera_image: CameraImageSel },
    SelectCameraIntrinsics { reconstruction_label: Option<String>, camera_intrinsics_index: usize },
    SelectPoint { point: goto_point::PointQuery },
    ClearSelection { scope: SelectionScope },
    SetReconstructionDisplay { reconstruction_label: String, change: DisplayChange },
    SetSolo { reconstruction_label: Option<String> },
    SetView { view: ViewCommand },
    Screenshot { max_dimension: Option<u32> },
}

/// What a tool produced: JSON, or the one tool that answers with a picture.
pub(crate) enum ToolOutput {
    Json(Value),
    Png { bytes: Vec<u8>, width: u32, height: u32, caption: String },
}

/// A tool's answer: what it produced, or a message for `isError: true`.
pub(crate) type Reply = Result<ToolOutput, ToolError>;

/// Whether `apply` finished the job, or needs the frame to complete first.
pub(crate) enum Outcome {
    Done(Reply),
    Deferred(Deferred),
}

/// Apply one command. **Takes no `App` and no GPU handle** — which is what
/// makes fifteen of the sixteen tools testable in a headless `cargo test`.
pub(crate) fn apply(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome;

/// Start the server. Returns once it is bound and listening, or with the bind
/// error; the runtime lives on its own thread from here.
pub(crate) fn serve(port: u16, tx: UnboundedSender<Request>, proxy: EventLoopProxy<UserEvent>)
    -> Result<SocketAddr, ServeError>;
```

`apply` taking `(&mut AppState, &mut Viewer3D)` rather than `&mut App` is the
one signature worth defending. `App` owns a `wgpu::Device`, a surface and a
window; a test that could construct one would need a GPU and a display, and the
crate's headless lib tests deliberately need neither. Splitting the GPU-shaped
command out into `Outcome::Deferred` — handled in `mcp::frame`, where the device
is passed in — keeps the whole command vocabulary, its error messages and its
JSON shapes under headless test, and leaves exactly one tool (`screenshot`)
needing a window.

`ToolOutput` has two shapes rather than one because `screenshot` answers with a
picture and the other fifteen answer with JSON; squeezing an image through a
JSON field would mean a magic key the transport has to know to look for. The
fifteen return a plain `Result<Value, ToolError>` and are widened at the `apply`
dispatch, so nothing below it has to name the shape it is not.

`App` grows two fields: `mcp_rx: Option<UnboundedReceiver<Request>>` and
`mcp_deferred: Vec<(Deferred, oneshot::Sender<Reply>)>`. The request counter the
Scene panel shows lives on `AppState::mcp`, beside the port, so the panel and
the window title read one thing.

**Read tools take `&mut AppState`.** That looks wrong for a read and is not:
resolving a `.sfmr` observation to a pixel means reading the `.sift` file it
points into, and the viewer memoizes that in `AppState::sift_cache`. Reading
through the cache is what makes the number reported here the same number the
Point Track panel shows, rather than a second implementation of it.

## Transport and protocol

**Streamable HTTP, via the official Rust SDK ([`rmcp`][rmcp] `~3.2`), mounted on
`axum` 0.8, on a `tokio` current-thread runtime on one dedicated thread.**

The SDK is here to absorb the protocol's revision history. MCP has run
2024-11-05 → 2025-03-26 → 2025-06-18 → 2025-11-25 → 2026-07-28, adding and then
removing sessions, the GET stream, `Last-Event-ID` resumability and
server-initiated requests along the way, and clients in the field speak several
of those. Keeping up with that is `rmcp`'s job, and it is what buys a viewer that
still works after somebody upgrades their client.

The current revision is genuinely simple — a single POST endpoint, no sessions,
no `initialize` handshake, and no SSE needed at all for a server that answers
every request with `application/json` — so the server is configured for exactly
that shape (`legacy_session_mode: false`, `json_response: true`) and the SDK
handles the older eras behind it. `rmcp` is pinned to an exact minor: which
revisions a given version negotiates is the reason it is a dependency at all,
and a silent minor bump could change it.

The cost is a dependency tree: `tokio`, `hyper`, `axum`, `rmcp`, in a workspace
with no async runtime otherwise, compiled from the PyPI **sdist** on the user's
own rustc (wheels ship for Linux and Windows only, so macOS users build it).
Beside `wgpu`, `winit`, `egui` and `image`, which `sfm-explorer` already pulls
in, it is a modest addition — and it sits behind a Cargo feature so it can be
dropped:

```toml
[features]
default = ["mcp"]
mcp = ["dep:rmcp", "dep:tokio", "dep:axum", "dep:serde", "dep:serde_json", "dep:base64"]
```

Default on, so the flag works in a stock build; `--no-default-features` drops it
for anyone trading the feature against build time. In a build without it, `--mcp`
is rejected at startup with a message naming the flag it needs, rather than
ignored.

A tool's JSON reply crosses the wire twice: as `structuredContent` for a client
that reads it as data, and as a text block for one that reads it as text. Both,
because which of the two a client surfaces to its model is the client's
decision.

**`tools/list` must carry its cache hints.** SEP-2549 added `ttlMs` and
`cacheScope` to list results and made them mandatory in the current revisions;
`rmcp` models both as `Option` so one type can also serve the older ones, which
means a handler that leaves them unset compiles fine, passes a conformance check
against an older revision, and is then rejected outright by a current client —
the server shows as *connected*, its tool list fails schema validation, and its
tools are silently absent for that whole session.

`ttl_ms` is **0**, "do not cache". The catalog is a compile-time constant and
cannot change while a viewer runs, so a long TTL would be defensible — but it
changes across a *rebuild*, which is the normal state of affairs for a tool
whose purpose is being iterated on, and a client holding a cached list across a
relaunch would call tools the new binary no longer has. Sixteen tools are cheap
to re-fetch; a stale list is not cheap to debug. `cache_scope` is `private`:
there are no authorization contexts to share a result across.

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
3. **Validate `Origin`.** The allowlist is the loopback origins on the endpoint's
   own port; anything else present in an `Origin` header gets `403 Forbidden`,
   per the transport spec. This is what stops a web page the user has open from
   driving their viewer through DNS rebinding. A real MCP client sends no
   `Origin` at all and is unaffected. `rmcp`'s `Host` allowlist — loopback by
   default — is left as it is.
4. **No write path to disk.** No tool in this surface saves an `.sfmr`, exports
   anything, or deletes a file. `close_reconstruction` unloads a reconstruction.
5. **The window announces it.** Title suffix and Scene panel header, always,
   while the server is live.

`open_reconstruction` reads any path the process can, matching the viewer's own
File ▸ Open. The gate on that is the flag the user typed, and the docs say so.

## Errors

Two levels, and the distinction matters to a client:

- **Protocol errors** (JSON-RPC `error`): a malformed request, an unknown tool
  name, or arguments the tool cannot make sense of. The SDK produces the first;
  `tools::parse` produces the rest as `invalid_params`.
- **Domain errors** (`CallToolResult` with `isError: true`): everything the
  viewer refuses. An unknown reconstruction label, a camera image index out of
  range, an unreadable `.sfmr`, an unknown tint name, a degenerate `set_view`, a
  viewport that has not rendered yet so there is nothing to screenshot, and the
  10 s apply timeout.

The line is whose problem it is: a request that does not fit the advertised
schema is the client's, and a request the viewer will not carry out is the
viewer's.

Domain errors get a message an agent can act on, in the style the viewer already
uses for its status line: what was asked, what is actually there. *"No loaded
reconstruction is labelled `globl` — loaded: `seoul_bull`, `global`."*

**An unknown argument is refused by name rather than ignored.** The schemas say
`additionalProperties: false`, but a schema binds only the clients that enforce
it, and an ignored typo would leave an agent believing it asked for something it
did not — which is the failure the whole return-your-resulting-state design is
shaped to avoid.

## Testing

`crates/sfm-explorer/src/mcp/tests.rs`, headless, no GPU, no window — which is
what the `apply(&mut AppState, &mut Viewer3D, …)` signature is for. The fixture
is a two-reconstruction scene whose reconstructions each resolve to **two**
intrinsics records, because a one-camera reconstruction cannot tell the
camera-image and camera-intrinsics selections apart and every coupling rule
looks like a no-op against it.

- **Every command against that scene**: the JSON shape, the ids, and the
  `selection` block each of the six selection tools returns.
- **The selection invariants survive the boundary**: selecting a camera image
  sets its intrinsics; selecting an intrinsics record the selected camera image
  does not use clears that camera image; selecting the one it *does* use keeps
  it; selecting a reconstruction drops another's finer selection. These are
  `AppState`'s guarantees, and the point of the test is that MCP cannot route
  around them.
- **`select_camera_image` then `select_point` in one reconstruction leaves both
  selected**, in either order — the composition the joint selection states are
  reached by.
- **The view block round-trips**: a view read out of `get_scene`, sent back
  through `set_view`'s exact form after moving elsewhere, restores the same six
  stored fields *and* everything under `derived`.
- **Stale and out-of-range refs** produce errors naming what is loaded, for
  every ref-taking tool; `get_scene` survives a selection that has gone stale
  rather than panicking on it.
- **Label addressing survives a reload**: a node replaced in place with a fresh
  `ReconId` still answers to the same label.
- **Solo**: it moves rather than accumulating, leaves the selection alone, and
  keeps `visible` and `drawn` distinguishable.
- **A refusal is atomic**: an unknown tint lists the palette and leaves the
  call's other fields unapplied.
- **Schema and parser cannot drift**: every property any tool advertises is one
  the parser accepts, walked over the whole catalog rather than tool by tool, so
  a tool added later is covered by construction. The vocabulary rule is asserted
  the same way.
- **The tool list carries its cache hints at the revision a real client
  negotiates.** The test initializes the way a current client does and then uses
  whatever version comes back, rather than hard-coding one — the SDK negotiates
  down from the newest revision, and pinning a version would make the test
  assert the SDK's default instead of the server's obligation.

**The protocol is tested over a real socket, in the same headless module.**
`serve` takes a wake *closure* rather than an `EventLoopProxy`, so nothing in
`server` depends on winit and a test can start a genuine server on port 0 with
an ordinary thread standing in for the GUI — one owner of the state, applying
one command at a time, which is exactly the discipline the real frame keeps.
Requests go out as hand-written HTTP/1.1 rather than through an HTTP client
dev-dependency: a POST with a JSON body is a dozen lines, and the bytes on the
wire are the point. That covers the `initialize` handshake, `tools/list`
matching the catalog, a `tools/call` reaching the stand-in GUI and returning
`structuredContent`, a viewer refusal arriving as `isError: true`, a malformed
argument arriving as a JSON-RPC error instead, and a foreign `Origin` getting
`403` while the endpoint's own origin and a missing one both pass.

`crates/sfm-explorer/src/cli.rs` carries its own tests for the flag, including
the case that matters most: `--mcp scene.sfmr` must not eat the path as a port.

One thing is not covered. **`screenshot`** needs a real frame and belongs in
`ui_basic` (Windows/macOS, `pixi run ui-test`) — asserting a decodable PNG of
the expected size, not its pixels. What the headless tests do assert about it is
that it defers, and that its caption is built while the state is still borrowed.

## Editing reconstruction data

The surface changes what is loaded, what is selected and how it is drawn. It
changes no reconstruction data, and the reason is a viewer-wide invariant worth
stating:

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

The answer the codebase already has is `resect_image`'s. A `set_camera_intrinsics`
would **produce a derived node**:

```jsonc
// { "reconstruction_label": "seoul_bull", "camera_intrinsics_index": 0,
//   "params": { "focal_length_x": 410.0, "radial_distortion_k1": -0.028 } }
{ "label": "seoul_bull (intrinsics 0 edited)", "replaced": false }
```

— named for its provenance, inheriting the source's transform so it lands
exactly on top, replaced in place when the same edit is repeated, and leaving
the source untouched. The agent then flips `set_reconstruction_display` between
the two, or screenshots both. Saving it is `sfm xform`'s job, as it already is.

That is a real design with real work behind it (partial-parameter merge against
`CameraModel`, re-upload invalidation, the derived-node lifecycle) and it should
land as its own change.

### Loose images, and the names held for them

The other reason the vocabulary is careful. The flow worth building toward is:
open a reconstruction, load a folder of images that are *not* in it, browse them
against the structure, and resect the ones that fit — which is `Resect Image`'s
existing question asked of a picture that has no pose yet rather than one whose
pose is in doubt.

Nothing here implements it. What this surface does is **not spend the names**:

| Tool | What it would do |
|------|------------------|
| `open_images` | Load a folder or a list of files as loose images |
| `list_images` | The loose set — no pose, no reconstruction, no track |
| `get_image` | One loose image: path, dimensions, EXIF, any `.sift` beside it |
| `select_image` | Select one for the Image Detail panel |
| `resect_image` | Estimate a pose against a named reconstruction, producing a derived reconstruction the way `set_camera_intrinsics` would |

Each of those is the short name of an entity that genuinely has no
reconstruction behind it, sitting beside the `camera_*` tool that names the
posed thing. Spend `list_images` on camera images and every one of them has to
be named around the collision instead — or the surface has to break
compatibility to reclaim it.

Other candidates, in rough order of value:

- **MCP resources**, one per loaded reconstruction
  (`sfmr://seoul_bull/summary`), so a client can attach scene state as context
  without a tool call.
- **`subscriptions/listen` notifications** when the human changes the selection
  or opens a file, so an agent watching alongside a human stays in step. This is
  the one feature that needs SSE, which is why the server can answer every
  request with `application/json` today.
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

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--mcp PORT` | `8787` (`cli::DEFAULT_MCP_PORT`) | The loopback port to bind. `0` takes an ephemeral one, printed at startup. |
| `list_camera_images` `limit` | `50` (`read::DEFAULT_LIMIT`) | Rows per page. |
| `list_camera_images` `limit` cap | `500` (`read::MAX_LIMIT`) | The most one call will return, whatever it asked for. |
| `screenshot` `max_dimension` | none — native viewport size | Longest side of the returned PNG. |
| Apply timeout | `10 s` (`server::APPLY_TIMEOUT`) | How long a tool call waits for the GUI thread. |
| `set_view` `fov_short_axis_deg` | `5`–`160` degrees (`view::MIN_FOV_DEG`, `view::MAX_FOV_DEG`) | Accepted range, matching what interactive FOV zoom clamps to. |

## Open questions

- **Discovery for multiple viewers.** With ephemeral ports, an agent has no way
  to find the running instances. A registry file — `~/.sfmtool/explorer-mcp.json`,
  written on bind and removed on exit — would solve it, at the cost of a stale
  file after a crash. Worth it only once running two viewers at once is common.
- **Should the human be able to turn it off?** A `Help ▸ MCP` menu item showing
  the endpoint and a Stop button costs little and is the honest counterpart to
  the title-bar announcement.
- **Screenshot resolution policy.** Native viewport size can be a 4K PNG, which
  is a lot of tokens. `max_dimension` defaulting to something like 1280 might
  serve agents better than defaulting to native — but a downscaled screenshot is
  a worse answer to "is this point cloud noisy". Decide with a real agent in the
  loop.
- **`screenshot` has no test that renders a frame** (§ "Testing"). It belongs in
  `ui_basic`, which runs on Windows and macOS only.
- **Whether `set_view` should expose the HUD's display controls** (point size,
  EDL thickness, patch opacity). They change what a screenshot shows, so an
  agent evaluating a reconstruction may want them; they are also a long tail of
  knobs that would double the tool's surface. Left out until something asks.
