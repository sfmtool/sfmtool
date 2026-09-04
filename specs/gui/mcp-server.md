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
move the selection and the 3D camera, choose what the Image Detail panel draws
over its photograph, arrange the panels, size and place the window, read back
everything that has happened in the viewer, and photograph the window or any
panel in it. The human keeps the window in front of them the whole
time and watches it change.

The window itself is part of what the surface drives, because the window is
shared. The agent wants the 3D viewport to fill it before a screenshot and the
Action Log out of the way; the human wants the Action Log in front to see what
the agent just did, and the window back to the size they had it. Both ask for
that in the same vocabulary — one document carrying the window's placement and
the panel arrangement, which is also the file the Panels menu saves — and both
see the result in the Action Log.

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
`--no-default-layout` ([panel-layout.md](panel-layout.md) § "The default layout
file"), `--help`, and treats everything else as a path. Hand-rolled rather than
`clap`: two flags and a list of paths is a dozen lines, and it keeps the
binary's dependency tree as it was. The following-argument form has to look at
what comes next, because `--mcp scene.sfmr` is the common invocation and means
the default port and a file — so a next argument that is not a port is left
alone rather than consumed.

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

Twenty-three tools. Eight read, fourteen write, one that closes the loop by
handing back a picture.

| Tool | Kind | What it does |
|------|------|--------------|
| `get_scene` | read | The whole scene graph, the selection, and the view state |
| `list_camera_images` | read | One reconstruction's camera images, paginated |
| `get_camera_image` | read | One camera image: pose, intrinsics, observation stats |
| `get_camera_intrinsics` | read | One intrinsics record and the camera images that use it |
| `get_point` | read | One 3D point: position, colour, error, full track |
| `get_action_log` | read | What has happened in the viewer, from a revision onward, filtered by who did it |
| `get_window_layout` | read | The window's placement and the panel arrangement as one document, the live window block, and each panel's open state |
| `get_image_detail_display` | read | The Image Detail panel's controls — the feature overlay and its filters, and the intrinsics layer — as one document |
| `open_reconstruction` | write | Load an `.sfmr` into the scene (reload if already open) |
| `close_reconstruction` | write | Close one reconstruction, or all of them |
| `select_reconstruction` | write | Make one the reconstruction the file- and sequence-shaped panels follow |
| `select_camera_image` | write | Select a camera image — and with it the intrinsics it was shot through |
| `select_camera_intrinsics` | write | Select an intrinsics record |
| `select_point` | write | Select a 3D point |
| `clear_selection` | write | Drop the selection, wholly or one kind of it |
| `set_reconstruction_display` | write | One reconstruction's eyes, tint, interactivity |
| `set_solo` | write | Draw only one reconstruction, or end the solo |
| `set_image_detail_display` | write | Change any of the Image Detail panel's controls, leaving the rest alone |
| `set_view` | write | Frame the scene, look through a camera image, or set the viewport camera outright |
| `set_window_layout` | write | Apply a window layout document: the window portion, the panel portion, or both |
| `show_panel` | write | Open a panel at its home position, or raise it if it is open |
| `hide_panel` | write | Close a panel |
| `screenshot` | observe | PNG of the window, or of one panel |

Every tool is annotated: the eight reads and `screenshot` carry
`readOnlyHint: true`, the fourteen writes `destructiveHint: false` (nothing here
touches a file on disk — `close_reconstruction` unloads, it does not delete;
`set_window_layout` changes the window and the dock, not the layout file the
menu saves), and every one of them `openWorldHint: false`. Every `inputSchema`
is closed (`additionalProperties: false`).

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

#### `panel`

One docked view is a **panel** — the word the Panels menu and
[panel-layout.md](panel-layout.md) use, several hundred times over, against a
handful of "pane". A panel has one handle, its name, so the argument spells
both: **`panel_name`**, by the rule below that makes a reconstruction's
argument `reconstruction_label`.

The seven names are the layout file's, and there is no second spelling of them
anywhere: `scene`, `viewer_3d`, `image_browser`, `image_detail`, `point_track`,
`camera_intrinsics`, `action_log` (`Tab::wire_name`). An unknown name is
refused with a message listing all seven (`Tab::all_wire_names`). `Tab` stays
the Rust name — it is `egui_dock`'s word for the thing in a node, and the code
is not the wire.

**`window_layout`** is the whole document — the window's placement and the panel
arrangement — and **`layout`** is its panel section, so the fields holding them
are named for the entities and not `layout_something`: they carry the things,
not handles to them. **`image_detail_display`** is the same kind of thing for
one panel: the Image Detail panel's controls as a document, with the feature
overlay and its filters at the top level and the intrinsics layer as an
**`intrinsics`** sub-block, which is how the panel's toolbar draws them — the
feature controls in a row, the layer behind one checkbox and a gear.

#### `<entity>` for the thing, `<entity>_<attribute>` for a reference to it

A field holding a whole entity is named for the entity; a field holding *one
attribute that identifies* an entity is named for both:

| Field | Holds |
|-------|-------|
| `camera_intrinsics` | the expanded record — model, size, params |
| `camera_intrinsics_index` | just the index, whether as a cross-reference from a camera image or as the argument naming which record to act on |
| `reconstruction_label` | just the label, identifying which reconstruction |
| `revision` | the Action Log's clock, on the log and on each entry |
| `since_revision` | just the revision, identifying where a read of the log starts — "the revision I have seen since" |

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

**The window has no GUI word either, so `winit`'s wins**: `outer_position`,
`outer_size`, `inner_size`, `scale_factor` and `has_focus` are
`winit::window::Window` method names, and an agent reading one on the wire can
grep for what produced it. The four window states are winit's flags spelled as
adjectives — `maximized`, `minimized`, `fullscreen` — plus `normal` for none of
them, which is the word the user and Win32's `SW_SHOWNORMAL` both use.

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
  "action_log_revision": 530,             // the Action Log's clock — see § "get_action_log"
  "window_title": "SfM Explorer - seoul_bull.sfmr [MCP :8787]",
  "window": { "state": "normal", … }      // see § "The window block"
}
```

`counts` is summary only. A 4 M-point cloud must never cross this boundary as
JSON, and no tool here returns bulk arrays — that is what the `.sfmr` file and
`sfm inspect` are for. The agent reads the file for the data and asks the
viewer for the *state*. `points_at_infinity` is read the way
`scene::visible_stats` reads it, so this number and the one in the viewport's
stats overlay are the same number.

**`window` is carried here as well as by `get_window_layout`**, everything but
the monitor list (§ "The window block"). "Can the human see this window, and how
much of the desktop is it" is a question an agent asks before deciding whether
a screenshot is worth taking at all, and a second call for it every time is a
call too many. It is `null` only before the window exists, which no tool call
can be answered ahead of.

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
`metrics::compute_observation_metrics`, the same function the Point Track
Detail table tabulates. `metrics` sits at the crate root rather than inside
that panel for exactly this reason: no one surface owns a number three of them
quote.

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
prints and the Camera Intrinsics panel shows, so the three can be diffed
against each other.

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
`load_file` returns `Result<ReconId, String>` and writes no Action Log entry on
`Err`, so the failure is simply propagated as the refusal. The caller is what
words it: the File menu records `Failed to load …`, the drain records
`open_reconstruction failed: …`, and either way one failure produces one entry.

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

### `get_image_detail_display` / `set_image_detail_display`

The Image Detail panel is the picture that says *where* a solve is wrong: a
reprojection-error heatmap over the photograph, the distortion field under it,
the principal point marked. What it draws is decided by the panel's toolbar —
seven feature overlay modes, three filters on the features, and the intrinsics
layer with its own sub-toggles — and every one of those is scene-level state on
`AppState` (`feature_display`, `intrinsics_display`), not a property of any
image or reconstruction. Which is why they are one document and one pair of
tools rather than a field on `set_reconstruction_display`: they describe how the
panel looks at *whatever* is selected. A `screenshot` of `image_detail` shows
whichever mode the human last picked; these two tools let the agent pick.

`get_image_detail_display` takes no arguments and returns the document:

```jsonc
{
  "image_detail_display": {
    "overlay_mode": "features",       // none | features | reproj_error | track_length
                                      // | max_track_angle | depth_reliability | condition_number
    "max_features": null,             // an integer ≥ 1, or null for all of them
    "feature_size_px": null,          // { "min": 0.0, "max": 50.0 }, or null for no size filter
    "tracked_only": true,             // only features with a 3D point behind them
    "intrinsics": {
      "enabled": true,                // the layer at all; on by default
      "axes": true,                   // angular axes through the principal point
      "rings": false,                 // iso-angle rings
      "distortion": true,             // the displacement field, where the model has any
      "distortion_scale": null,       // 1 | 2 | 3 | 5 | 10 | 20 | 50, or null for auto
      "grid_cols": 16                 // 8 | 12 | 16 | 24 | 32
    }
  }
}
```

`set_image_detail_display` takes any subset of the same fields, at either
level, and leaves every omitted one alone:

```jsonc
{ "overlay_mode": "reproj_error" }
{ "overlay_mode": "features", "tracked_only": false, "max_features": 500 }
{ "feature_size_px": { "min": 2.0, "max": 40.0 } }
{ "feature_size_px": null }                                  // the filter off; the values it held persist
{ "intrinsics": { "enabled": true, "rings": true, "distortion_scale": 10 } }
{ "intrinsics": { "distortion_scale": null } }               // back to auto
```

The reply is the whole document, exactly as `get_image_detail_display` would
return it, so the agent reads back the state rather than the fields it happened
to set — the same rule as `set_reconstruction_display`.

**The wire spellings are the code's.** `overlay_mode` takes the snake-cased
`OverlayMode` variant (`OverlayMode::wire_name`, `from_wire_name`,
`all_wire_names`, exactly as `Tab` spells the seven panels), because the GUI's
labels — `Reproj Error`, `Max Track Angle` — are display text with spaces in
it, and § "Where the GUI has no word, the code's word wins" applies to a word
the GUI has only as a label. An unknown mode is refused with a message listing
all seven. `distortion_scale` and `grid_cols` are refused off their ladders
(`IntrinsicsDisplaySettings::SCALE_LADDER`, `GRID_LADDER`) with a message
listing the ladder, for the reason `tint` refuses a free colour: those are the
values the gear popup offers, and a value the popup cannot show is a value the
human cannot see they are looking at. `max_features` is refused below `1` —
"show no features" is `overlay_mode: "none"`'s job, and `0` would be a second
spelling of it that the `Max:` dropdown cannot display. A `feature_size_px` is
refused when either bound is negative or not finite, or when `min` exceeds
`max`.

**`feature_size_px` is one thing on the wire because it is one checkbox in the
toolbar.** `FeatureDisplaySettings` keeps two `Option<f32>` — `min_feature_size`
and `max_feature_size` — beside two persisted drag values, and the toolbar
re-derives the pair from its single `Min/max size:` checkbox **every frame**:
ticked, both options are written from the drag values; unticked, both are
cleared. So the two are never independently `Some` while the panel is open, and
a tool that let an agent set one without the other would have its half undone
by the next frame. The document therefore reports the pair as one object or
`null`, and setting it writes **all four fields** — both options *and* both
drag values — so that the toolbar's next frame re-derives exactly what the
agent asked for. Setting it to `null` clears the two options and leaves the drag
values where they were, which is what unticking the checkbox does.

**Refusals are atomic, and they happen at the parse.** Every vocabulary here
is static — the seven modes, the two ladders, the bounds on a size filter — so
the whole call is validated in `tools` before a `Command` exists, and `apply`
cannot fail. A call naming a good field and a bad one changes nothing. Being
turned away at the parse also makes it a *protocol* error rather than a domain
one (§ "Errors"), so it never reaches the viewer and leaves no Action Log row;
`Kind::Display`, which `Command::kind` gives the command, is where the entries
a **successful** call writes go.

**Every field the call changed is one Action Log entry**, under `Display` and
in the words the HUD's own controls use — `{Control} {on|off}`, `{Control}
{value}` — because each is its own control to the person watching the window
(the same reasoning as `set_reconstruction_display`, [action-log.md](action-log.md)
§ "Catalogue"). An unchanged field records nothing. Each field is its own
run ([action-log.md](action-log.md) § "Coalescing"), so a call that changes
three fields leaves three rows, and only a repeat of the *same* field inside
the window — an agent stepping `distortion_scale` up the ladder, a human
dragging the size filter — folds into one line. The texts:

| Field | Text |
|-------|------|
| `overlay_mode` | `Overlay {label}` — the GUI label, e.g. `Overlay Reproj Error` |
| `max_features` | `Max features {n}` / `Max features all` |
| `feature_size_px` | `Feature size {min:.1}–{max:.1} px` / `Feature size filter off` |
| `tracked_only` | `Tracked only {on|off}` |
| `intrinsics.enabled` | `Intrinsics {on|off}` |
| `intrinsics.axes` / `rings` / `distortion` | `Intrinsics axes {on|off}` / `Intrinsics rings {on|off}` / `Intrinsics distortion {on|off}` |
| `intrinsics.distortion_scale` | `Distortion scale ×{n}` / `Distortion scale auto` |
| `intrinsics.grid_cols` | `Grid density {n}` |

**The human's changes to the same controls record the same texts**, as `User`.
Before this pair existed the Image Detail toolbar, its gear popup and the `I`
key wrote straight into the two settings structs and logged nothing, which was
tolerable while only a human touched them and is not once an agent can: the
Action Log's premise is that both actors see what the other did in one place,
and a heatmap mode the agent switched on has to be as visible in the log as the
grid the human switched off. The record is one function — the diff of the two
settings structs before and after a change, one `Kind::Display` entry per field
that differs, in the table's words — and **both actors call it**: the tool
around its write, the dock around the panel's frame. One function is what
guarantees the texts are identical; a second catalogue in the toolbar would
drift.

The settings hold whether or not the Image Detail panel is open, so a
`set_image_detail_display` against a closed panel is not refused: it is what
the panel will show when `show_panel` opens it. Nothing here selects an image —
the panel shows the selected camera image of the selected reconstruction, as it
always has, and `select_camera_image` is how the agent chooses which
photograph the overlay is drawn on.

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

### `get_action_log`

Everything else on this surface lets the agent *act* and lets the human *see*
what it did. This is the other direction: what the human did, read as a
transcript.

```jsonc
{ "since_revision": 512 }                       // entries recorded or changed after revision 512
{ "since_revision": 0, "limit": 50 }            // from the start, at most 50
{ "since_revision": 512, "actors": ["user"] }   // what the human did since 512
{}                                              // everything kept, every actor
```

```jsonc
{
  "revision": 530,                    // the log's current revision — send it back next time
  "oldest_revision": 3,               // of the entries the log still holds (CAPACITY, Clear)
  "truncated": false,                 // true when `limit` cut the reply short
  "entries": [
    {
      "revision": 513,
      "at": "2026-09-02T12:41:07.123-07:00",   // RFC 3339, the zone the panel formats in
      "actor": "user",                          // "user" | "mcp" | "viewer"
      "kind": "selection",                      // Kind::wire_name; "query" carries `tool`
      "failed": false,
      "text": "Selected image images/IMG_0007.jpg"
    },
    {
      "revision": 519,
      "at": "2026-09-02T12:41:09.004-07:00",
      "actor": "mcp",
      "kind": "layout",
      "failed": true,
      "text": "set_window_layout failed: layout.main: unknown key \"fracton\""
    },
    {
      "revision": 522,
      "at": "2026-09-02T12:41:12.870-07:00",
      "actor": "mcp",
      "kind": "query",
      "tool": "get_scene",
      "failed": false,
      "text": "get_scene"
    }
  ]
}
```

**A revision is the log's clock.** `ActionLog` keeps a counter that goes up by
one on every record *and on every coalescing replacement*, and stamps the entry
it just wrote or replaced. So `since_revision: N` returns exactly the entries
whose current text the agent has not seen: a new entry once, and an entry that
a run folded into again, with its newest text and time. A timestamp could not
do this — two entries can share an instant, and a fold changes an entry's time
to the newest of the run, which is what the agent needs to be told about.
`revision` in the reply is the counter now, and sending it back as the next
`since_revision` reads the gap and nothing else; `since_revision: 0`, or an
omitted one, is the start.

**Order is log order**, oldest first, because the agent reads it as a
transcript. `limit` defaults to `200` and is capped at `1000`
(`read::ACTION_LOG_DEFAULT_LIMIT`, `read::ACTION_LOG_MAX_LIMIT`); past it,
`truncated: true` and the agent continues from the last entry's `revision`.

**`actors` filters by who did it.** A set of the Action Log's three actors
(`Actor::wire_name`: `user`, `mcp`, `viewer`), and an entry is returned when its
actor is in the set. Omitted, it is all three, which is the whole log; an empty
set is refused at the parse, since a call that can return nothing by
construction has not asked a question. The read an agent makes most is
`actors: ["user"]`: what the human did, with none of the agent's own rows in it
— and since every query is the agent's, that filter is also what keeps a polling
loop's `get_scene` rows out of the answer without a switch of their own.
`actors: ["mcp"]` is the agent auditing itself, queries and refusals included,
`tool` beside `kind` on the query rows. `viewer` is the small third set the log
already keeps apart in its actor column: the session lines, and an animation
running out of images ([action-log.md](action-log.md)).

No `kinds` filter, for the same reason the panel has none: at one second of
coalescing, a session's log is readable whole.

**`oldest_revision`** is the revision of the oldest entry still held. An agent
whose `since_revision` is older than it has missed entries — dropped past
`CAPACITY`, or gone with the toolbar's Clear — and can say so rather than
assuming the gap was quiet. The counter never resets, Clear included.

**`get_action_log` is a read and is recorded as one**, a `Query` entry
`get_action_log since 512`, coalescing per tool like every other read but
`screenshot` — so a poll is one row, however often it asks — and out of its
own reply under
`actors: ["user"]`. A read of the log that the log did not record would be the
one action the human could not see. The row is written after the reply is built,
so a call never reports itself and always reports the one before it.

**`get_scene` carries the revision.** One field, `action_log_revision`, beside
`status_message`, so an agent that already reads `get_scene` knows whether
anything happened since its last `get_action_log` without a second call.
`status_message` stays: it is the status line, which is a different thing from
the log.

### `screenshot`

```jsonc
{}                                              // the whole window
{ "panel_name": "viewer_3d" }                   // one panel's body
{ "panel_name": "image_detail", "max_dimension": 1024 }
{ "panel_name": "viewer_3d", "hud": false }     // the 3D render alone, nothing drawn over it
```

Returns an MCP `ImageContent` block — base64 PNG, `mimeType: "image/png"` — plus
a text block naming the pixel size and what was photographed: `The window,
1920×1129.` or `The Image Detail panel, 640×480.` For the window and for the 3D
Viewer it keeps the frame description too (which reconstructions are drawn, the
point and camera-image counts, and the camera image being looked through, if
any), because those are the pictures the 3D view is in. That caption is built
during the *apply* phase, while `AppState` is still borrowed, so the picture and
the description of it are of the same instant.

**Without `panel_name` it is the window**: the frame the human is looking at, at
`inner_size` in physical pixels, menu bar and status line included. It is the
surface texture that was presented, copied before the present, so the picture is
of the frame the request woke and nothing composited differently.

**With `panel_name` it is that panel's body**, cropped from the same frame:
`LeafNode::viewport`, the tab body below the tab bar, in logical points from the
dock's layout of *this* frame, scaled to pixels and rounded. A panel in a
floating dock window is still inside the surface and crops the same way. A panel
that is **closed** is refused, naming `show_panel`; one that is **behind another
tab** in its node is refused, naming the tab in front and `show_panel` — a
screenshot of a tab that is not drawn would be a picture of the tab in front of
it. Both checks are against `AppState::dock` at apply time, so a `show_panel`
earlier in the same batch satisfies them, and both are headless.

**`viewer_3d` is a crop by default, and the render target on request.** The
default for every panel is a crop of the one presented frame, so the viewport's
comes with what is drawn over it — the HUD, the stats, the status line — because
that is what the human sees. But an agent judging a point cloud does not want a
status line across it, and the clean picture already exists: the `edl output`
texture the viewport renders into. **`hud: false`** returns that texture instead
of the crop, through the capture path in `scene_renderer/capture.rs`, so the two
pictures of the viewport differ only in what egui painted over it.

The two are very nearly the same size and not exactly: the crop is the tab
*body*, which `egui_dock` insets by its own `tab_body.inner_margin` before the
viewport allocates what is left, so the render is the crop less that margin on
each side — twelve logical points in each axis at the stock style. Each picture
says its own size in the caption.

`hud` is accepted with `panel_name: "viewer_3d"` only. With another panel or
with no panel it is refused at the parse — *"hud applies to the 3D Viewer only;
the other panels have no picture underneath what is drawn on them"* — rather
than read as a request to draw the frame differently. The Image Detail panel's
overlays are content its own toggles control, not chrome, and a screenshot flag
that quietly changed what the panel drew would give the agent a picture the
human never saw; an agent that needs the raw photograph says so where the human
would, with `set_image_detail_display { "overlay_mode": "none", "intrinsics":
{ "enabled": false } }`, and the panel then shows what the screenshot
returns. `hud` is the
GUI's own word for the overlay ([viewport-hud.md](viewport-hud.md)), which is
why the initialism stands where the vocabulary rule would otherwise want a
spelled-out word.

**`max_dimension`** applies after the crop, Lanczos3: what a downscaled point
cloud is asked to answer is "is this noisy", and a cheap filter's aliasing
invents exactly that.

**A minimized window is refused rather than photographed:**

> The window is minimized, so nothing is being rendered to photograph. Send
> `set_window_layout { "window": { "state": "normal" } }` first.

Whether a minimized window's swapchain still presents is platform-dependent,
and a picture of a window the human cannot see answers nothing an agent asked
of a shared viewer. The check is in `apply_with_window`, against the window
snapshot's `state` (§ "The window block"), so it is under headless test with
the rest of the vocabulary rather than only on a machine with a window.

`screenshot` is the one tool that cannot answer during the apply phase
(§ "Threading").

#### Mechanics

The surface is configured with `TextureUsages::COPY_SRC` added to the default
when `Surface::get_capabilities(&adapter).usages` allows it — DX12, Vulkan and
Metal all do in practice, and DX12 hands out a `Bgra8UnormSrgb` swapchain that
takes the usage. `lib.rs` ors the flag in and records whether it got it on `App`
(`surface_readable: bool`); the same configuration is what `resize` reconfigures
with, so the flag survives a resize.

In a frame with a deferred window or panel screenshot, and only then, the
frame's encoder copies the surface texture into a readback buffer **after the
egui pass and before `present`** (`App::encode_screenshot_copy`, the same
submit; the texture is not readable once it has been handed back to the
presentation engine), following `scene_renderer/readback.rs`'s 256-byte
row-alignment rule. `App::resolve_mcp_deferred` maps the buffer after the
present and answers every screenshot of that frame from the one copy — they are
all pictures of the same pixels, differing only in where they are cropped. The
surface format is whatever `get_default_config` chose, so the readback swizzles
`Bgra8Unorm[Srgb]` to RGBA — `image` has no BGRA8 colour type to encode from —
and passes `Rgba8Unorm[Srgb]` through; any other format is refused naming it. A
copy per screenshot frame, and no cost in any other frame.

Where the surface does not allow `COPY_SRC`, a screenshot of the window or of a
panel is refused: *"This platform's window surface cannot be read back, so there
is nothing to photograph."* — and the refusal points at `hud: false`, which
reads the render target and does not go through the surface. The alternative —
rendering every frame through an intermediate texture and blitting — is a
per-frame cost for a case no supported platform has shown; it is the fallback if
one does (§ "Open questions").

**The crop rectangle is resolved at readback, not at apply.** A panel opened by
a `show_panel` earlier in the batch has no rectangle until this frame's egui
pass has laid the dock out, so `Deferred::Screenshot` carries the `Tab` and the
frame reads `viewport` from `AppState::dock` after the pass. A rectangle that is
still `Rect::NOTHING` there — a panel that was not drawn after all — is refused
rather than cropped to nothing. Points become pixels through the frame's own
`egui::Context::pixels_per_point`, which is the window's scale factor composed
with egui's zoom and the only number the dock's rectangles are in step with.

**In the Action Log** the `Query` text names the target: `screenshot window
1920×1129`, `screenshot image_detail 640×480`, `screenshot viewer_3d 1280×720
without HUD`. The size is the size the picture will be, after `max_dimension`;
for a panel, the crop's size is not known until the frame, so the text records
the panel's *last* laid-out size, which is the right size in every frame but the
one that opened it.

### `get_window_layout`

Where the window is, and which panels are where. No arguments.

```jsonc
{
  "window_layout": {                // the file, verbatim: what Save Layout… writes right now
    "sfm_explorer_layout": 2,
    "window": { "state": "maximized", "outer_position": [120, 64], "inner_size": [1280, 720],
                "monitor": { "position": [0, 0], "size": [3840, 2160] } },
    "layout": { "main": { /* … */ }, "windows": [] }
  },
  "window": { /* the window block, live, with `monitors` — null with no window */ },
  "panels": {                       // one entry per panel, always all seven
    "scene":             { "open": true,  "active": true },
    "viewer_3d":         { "open": true,  "active": true },
    "image_browser":     { "open": true,  "active": true },
    "image_detail":      { "open": true,  "active": true },
    "point_track":       { "open": true,  "active": false },
    "camera_intrinsics": { "open": true,  "active": false },
    "action_log":        { "open": true,  "active": false }
  }
}
```

**`window_layout` is the file.** Not a rendering of it, not a subset: the object
`WindowLayout::to_json` writes, parsed. An agent that saves it has a file the
Panels menu loads and the viewer reads at startup, and a file the human saved is
an argument `set_window_layout` takes. One schema, one parser, one set of
validation messages ([panel-layout.md](panel-layout.md) § "The window layout
file").

**The `window` block beside it is the observation.** Its `window_layout.window`
sibling is the *settable* placement — the state plus the **normal** rectangle,
what the window restores to — while the block is focus, scale factor, the
current (not normal) position and sizes, the monitor, `derived`, and `monitors`
read live from the host. The two agree for a normal window and differ for a
maximized one, and that difference is the information: the agent sees both what
the window is showing as and what it would come back to.

**`panels` is the arrangement indexed the other way.** "Is the Action Log open"
should not cost the agent a tree walk. `open` is whether the panel appears
anywhere in the document's `layout`; `active` is whether it is the front tab of
its node — a panel alone in a node is active, and the default layout's two
multi-tab nodes leave three of the seven behind a sibling. A closed panel is
`active: false`.

Where there is no window — a headless `AppState` — `window` is `null` and the
document has no `window` section, rather than the tool refusing: the panels half
of the answer is still an answer. The snapshot is refreshed from the host first,
so a read is also the freshest observation anyone has.

### `set_window_layout`

The argument **is the document**, version tag optional:

```jsonc
{ "window": { "state": "maximized" } }
{ "window": { "state": "normal", "inner_size": [1600, 900] } }
{ "layout": { "main": { /* … */ }, "windows": [] } }
{ "layout": "default" }                                     // the stock seven-panel grid
{ "window": { "state": "maximized" }, "layout": "default" } // both, in that order
{ "sfm_explorer_layout": 2, "window": { /* … */ }, "layout": { /* … */ } }  // a file, or a whole reply, sent back
```

The reply is `get_window_layout`'s.

The whole argument goes through `WindowLayout::from_value` — the file's parser,
so the wire and the file cannot differ about what a document is, and a violation
is a **domain error** in the parser's words with its path (`layout.main.second:
unknown key "fracton"`, `window.state: unknown window state "big"; …`). It
reaches the human too, as a failed Action Log entry, exactly as a refused
Panels ▸ Load Layout… does. The version tag is accepted so a reply or a file can
be sent back without editing, and checked when present. Two things are the
tool's own:

- **An empty call is refused at the parse**: `{}`, `{ "window": {} }` and
  `{ "sfm_explorer_layout": 2 }` all ask for nothing, and a call is a request.
  What counts as empty is the parser's own `WindowLayout::is_empty`, so the two
  halves cannot come to disagree about it.
- **Where there is no window, a `window` portion is refused** with the "no
  window" message; a call with only a `layout` portion succeeds there.

Application is `AppState::apply_window_layout`: **window portion first, panel
portion second**, so a call reads "make the window like this, then arrange the
panels" and a panel tree is laid out into the window it was meant for. A
platform refusal in the window portion stops the call before the panels; a
validation refusal applies nothing, because the document is parsed and validated
whole before any of it is applied.

The window portion **preserves what it does not carry**, and its rectangle is
the window's *normal* rectangle — so a size sent to a maximized window changes
what it restores to and leaves it maximized ([panel-layout.md](panel-layout.md)
§ "How a `window` section is applied"). A `window` carrying `monitor` is fitted
exactly as a file's is (§ "Fitting a rectangle to the desktop" there), which is
what lets a file saved at another desk be sent through the tool unchanged; a
rectangle sent without one is applied as written, because an agent that sends a
bare rectangle means that rectangle. The reply is a read-back rather than an
echo: what the platform does with a requested size is not knowable in advance —
the window's `with_min_inner_size` of `800 × 600` logical is the minimum the
*user's drag* respects, and on Windows a programmatic resize goes straight past
it — so an agent that needs to know what it got reads the reply. `focus` may
likewise be declined by a platform that does not let applications steal focus;
`focused` in the reply says whether it worked.

The panel portion **replaces** the arrangement: a panel the document does not
mention is closed. Every panel keeps its state either way (a re-opened Image
Detail shows the image it had), because panel structs live for the process and
the dock only decides which of them draw. `"default"` is a named layout rather
than a separate `reset_layout` tool because it *is* setting the layout, to the
one arrangement that has a name; an agent that has just made a mess of the window
wants one call back, not a document it has to reconstruct.

**The change is applied at the top of the frame**, in the drain, before egui
reads the window size for that frame's layout — so the frame the request woke is
laid out at the new size and a `screenshot` in the *next* call sees it. The reply
is read back from the window immediately after the change, by which time the OS
has processed it on Windows (the calls are synchronous there). On a platform that
animates or defers window changes the read-back may be a frame early; an agent
that needs certainty confirms with `get_window_layout`.

### `show_panel` / `hide_panel`

```jsonc
// show_panel  { "panel_name": "action_log" }
// hide_panel  { "panel_name": "action_log" }
```

Both reply with `get_window_layout`'s block, so the four tools that touch the
arrangement answer alike and the agent sees where the panel landed rather than
assuming.

`show_panel` is `AppState::show_panel`: the three home-position rules of
[panel-layout.md](panel-layout.md) § "Home positions", so a panel the agent
opens appears where the menu would have put it. On a panel that is already open
it raises — makes it the active tab of its node — which is the call an agent
makes to put the Action Log in front of the Image Browser for the human without
moving anything.

`hide_panel` is `AppState::hide_panel`, and **is idempotent**: hiding a panel
that is closed succeeds and changes nothing, exactly as the method does. Both
tools *set* rather than toggle, for the reason `set_solo` does — an agent
issuing a toggle cannot know the outcome without reading first, and a retried
call would undo itself.

Neither takes a position. Where a panel goes is the home rule's decision; an
agent that wants a panel *there* sends the tree through `set_window_layout`.

**In the Action Log**, the panel writes are the Panels menu's own rows with
`MCP` in the actor column, because they go through the same `AppState` methods
the menu does — `Opened Action Log panel`, `Raised Action Log panel`, `Closed
Action Log panel`. A `set_window_layout` records **one row per portion it
carried**: the window portion under `Kind::Window`, composed from the pieces in
application order, and the panel portion under `Kind::Layout` as `Set layout` for
a document or `Reset layout` for `"default"` — the menu's own words, since
`AppState::apply_window_layout` records nothing itself and its three callers word
the entry differently. None of them coalesce, so two panels closed in a row are
two rows. `get_window_layout` is a `Query` entry and never reaches the status
line. The texts are listed in [action-log.md](action-log.md).

### The window block

`get_scene` carries this beside `window_title`, and `get_window_layout` returns
it with one addition.

```jsonc
{
  "window": {
    "state": "normal",                // "normal" | "maximized" | "minimized" | "fullscreen"
    "focused": true,                  // Window::has_focus
    "scale_factor": 1.5,              // Window::scale_factor — physical px per logical pt
    "outer_position": [120, 64],      // Window::outer_position — physical px, desktop coordinates; null where the platform cannot say
    "outer_size": [1936, 1119],       // Window::outer_size — physical px, frame included
    "inner_size": [1920, 1080],       // Window::inner_size — physical px, the drawable area
    "monitor": {                      // Window::current_monitor, or null
      "name": "\\\\.\\DISPLAY1",
      "position": [0, 0],             // physical px, desktop coordinates
      "size": [3840, 2160],
      "scale_factor": 1.5
    },
    "derived": {
      "inner_size_logical": [1280, 720],
      "monitor_fraction": [0.504, 0.518],   // outer_size / monitor.size, per axis; null without a monitor
      "monitor_area_fraction": 0.261
    },
    "monitors": [ /* the four layout tools only: every monitor, same shape as `monitor`, the current one first */ ]
  }
}
```

**Physical pixels throughout, with the scale factor beside them.** They are what
winit reports natively, what the view block's `viewport_px` and a screenshot's
dimensions are in, and what a monitor's size is in — so an agent comparing "the
window" against "the picture I was handed" compares like with like. The logical
size is under `derived`, next to the scale factor it is computed from, for the
agent that wants to reason in the units the window was created in (`1280 × 720`
logical, `800 × 600` minimum).

**`state` is one word for four flags.** winit exposes `is_minimized`,
`is_maximized` and `fullscreen` separately, and a window can be minimized *and*
maximized at once (a maximized window the user minimized; restoring it brings
the maximized one back). The block reports the one that governs what the user
sees, in the order of precedence `minimized` > `fullscreen` > `maximized` >
`normal`, because that is the question an agent asks — "can the human see this
window, and how much of the desktop is it" — and the flags underneath it are the
viewer's business. `is_minimized` returning `None` (a platform that cannot say)
reads as not minimized.

**`outer_position` can be `null`.** Wayland does not tell a window where it is.
The field says so rather than reporting `[0, 0]`, and an `outer_position` in a
window layout fails on such a platform for the same reason.

**`derived.monitor_fraction` is the answer to "how much of the desktop"**, per
axis, with the area under it; both are `null` when there is no current monitor
to compute against. A window straddling two monitors reports the one winit calls
current.

**Read `state` before believing the geometry.** A minimized window on Windows
reports an inner size of `0 × 0` and a position of `(-32000, -32000)`, because
that is what Win32 says about it; the block passes both through rather than
inventing plausible numbers, and `state` is the field that explains them.

**The block a tool returns is a snapshot, `AppState::window`** (§ "Threading"),
which is why `get_scene` can carry it without a window handle and why the
minimized check in `screenshot` is headless. `monitors` is the exception: it is
read from the window at the moment the tool runs, which is why `get_scene` does
not carry it.

**The block is the window's *current* geometry.** What a layout document's
`window` section carries is the rectangle the window restores to, which is a
different number for a maximized window and the same one for a normal window
(§ "`get_window_layout`").

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
   │                                      ├─ drain_mcp ◄── observes the window, applies every queued command
   │                                      ├─ prepare_uploads
   │                                      ├─ render_scene
   │                                      ├─ run_egui_pass
   │                                      ├─ encode_screenshot_copy ── the surface, if one was asked for
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
  deferred, and a deferred screenshot simply waits for the next frame. An idle
  viewer asks for no frames of its own, and without that the screenshot would
  sit until the caller's timeout rather than until the next frame.
- **The surface copy is the one thing that happens mid-frame.** A window or
  panel screenshot cannot be answered after the present, because the texture is
  gone by then; the copy into a readback buffer is encoded between the egui pass
  and the present, into the frame's own encoder, and only in a frame that has
  such a screenshot waiting (§ "screenshot", "Mechanics"). Everything after that
  — the map, the crop, the PNG — is still in the readback phase.

**The commands that must reach `winit` go through a trait**,
`crate::window::WindowHost`, for the same reason `screenshot` leaves through
`Outcome::Deferred`: to keep the command vocabulary applicable to
`(&mut AppState, &mut Viewer3D)` and nothing else. But a window change is not
deferred, and the difference is worth stating. A screenshot's answer does not
exist until the frame has been rendered; a window change can be applied on the
spot, its effect is wanted in *this* frame's layout, and its answer is a
read-back that a fake can produce as well as a real window can. Deferring it
would apply it after the present — a frame late — and would leave a
`set_window_layout` followed by a `get_window_layout` in one batch answering with
the old window. The trait is not the MCP surface's: it is where every window
change goes, the Panels menu's and the startup load's included
([panel-layout.md](panel-layout.md) § "The window").

**`AppState::window` is refreshed twice.** Once at the top of
`run_ui_and_paint`, *before* the drain, by `AppState::observe_window` — so
`get_scene`'s window block and a `screenshot`'s minimized check see this frame's
window — and again inside `apply_window_layout` after a change, so a later call
in the same batch, and the reply itself, see it. The first refresh happens on
**every** frame rather than only the ones the endpoint is live for, because Save
Layout… reads the same snapshot and the normal rectangle it remembers has to be
current at the moment the window is maximized. The cost is a handful of platform
calls a frame; an idle viewer renders none at all.

**Every reply is timeout-bounded** — 10 s on the HTTP side. The GUI thread can
legitimately stop pumping (a modal `rfd` file dialog is open, the user is
dragging the window on Windows), and an agent must get "the viewer is busy"
rather than a hung connection.

**Every applied command records an Action Log entry as actor `MCP`**, and the
viewport status line shows the most recent entry that is not a successful
query, prefixed `MCP: `
when an agent took it. The human watching the window sees what the agent did, in
the place the viewer already reports what it did, and can tell it from something
they did themselves — and can scroll back through the rest of the session in the
Action Log panel. See [action-log.md](action-log.md).

The entries are mostly not composed here. A mutating tool calls the same
`AppState` and `Viewer3D` methods the GUI calls, and *those* record, so two rows
with the same text did the same thing whoever asked for it. What the drain adds
is the two things no state method can know: a **read** is recorded from the
command (a `Query` entry, which never reaches the status line — an agent polling
`get_scene` must not read its own polling back as the viewer's status), and a
**refusal** is recorded as a failed entry, `{tool} failed: {message}`, in the
same words the agent receives.

One tool words its own entries, and for one reason — no state method owns the
change. `set_window_layout` goes through `AppState::apply_window_layout`, which
deliberately records nothing because its three callers word it differently: the
menu and the startup load say which *file* they loaded, and the tool says what
each of its two portions did. Its texts are listed in
[action-log.md](action-log.md).

The actor is ambient rather than an argument: `mcp::apply_as_agent` moves the
Action Log's actor to `Mcp` for the frame's batch and restores `User` after, so
not one `AppState` method signature carries an `Actor`.

## The Rust seam

`crates/sfm-explorer/src/mcp/`, in three layers so the interesting one is
testable without a window:

| Module | What it owns |
|--------|--------------|
| `tools` | The tool table and the wire parse: names, descriptions, `inputSchema`, and JSON arguments to a `Command` |
| `mod` + `read` / `write` / `view` / `render` | The command vocabulary, applied to `(&mut AppState, &mut Viewer3D)` |
| `layout` | The four layout tools and their shared reply, over `AppState`'s own document and panel operations |
| `display` | The `image_detail_display` document: its render, the parse of a change into `ImageDetailDisplayChange`, and the apply — over the two settings structs and the diff-and-record function in `crate::state` that the toolbar shares, unconditional because the human's changes are logged in every build |
| `window` | The `window` block renderer, and nothing else: what a window *is*, how a placement is applied, and the `WindowHost` seam are `crate::window`'s, unconditional because Panels ▸ Save Layout… needs them in every build |
| `frame` | The three phases `run_ui_and_paint` calls: the drain, the surface copy, and the deferred screenshot |
| `mod::apply_as_agent` | The drain's application phase without the channel: the Action Log's actor switch, one `apply` per command, and the query and refusal entries |
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
    /// `actors` is never empty: the parse refuses `[]` and fills an omitted
    /// field with every actor.
    GetActionLog { since_revision: u64, limit: usize, actors: Vec<action_log::Actor> },
    OpenReconstruction { path: PathBuf },
    CloseReconstruction { target: CloseTarget },
    SelectReconstruction { reconstruction_label: String },
    SelectCameraImage { reconstruction_label: Option<String>, camera_image: CameraImageSel },
    SelectCameraIntrinsics { reconstruction_label: Option<String>, camera_intrinsics_index: usize },
    SelectPoint { point: goto_point::PointQuery },
    ClearSelection { scope: SelectionScope },
    SetReconstructionDisplay { reconstruction_label: String, change: DisplayChange },
    SetSolo { reconstruction_label: Option<String> },
    GetImageDetailDisplay,
    /// Every field an `Option`, `None` meaning "leave it": the parse has
    /// already resolved the mode name, checked the ladders and the size
    /// bounds, so `apply` only writes and records.
    SetImageDetailDisplay { change: ImageDetailDisplayChange },
    SetView { view: ViewCommand },
    GetWindowLayout,
    /// The document as it arrived, unparsed, so that one the viewer will not
    /// accept is a domain error in the layout parser's own words — path and
    /// all — rather than a protocol error.
    SetWindowLayout { document: serde_json::Value },
    ShowPanel { panel: Tab },
    HidePanel { panel: Tab },
    /// `hud: false` is only reachable with `panel: Some(Tab::Viewer3D)`; the
    /// parse refuses it elsewhere.
    Screenshot { panel: Option<Tab>, hud: bool, max_dimension: Option<u32> },
}
```

`Command::kind` for `SetWindowLayout` is `Kind::Layout` when the object carries a
`layout` key and `Kind::Window` otherwise, which is where a refusal of it is
filed. `SetImageDetailDisplay` is `Kind::Display` — the kind the HUD's own
controls record under, since the Image Detail toolbar is the same sort of thing
on a different panel — and `GetImageDetailDisplay` a `Kind::Query` like every
other read. Everything the window portion is made of — `WindowChange`, `WindowState`,
`WindowInfo`, `MonitorInfo`, `NormalRect`, `fit_to_monitor` and the `WindowHost`
trait — lives in `crate::window` and is spelled out in
[panel-layout.md](panel-layout.md) § "The window"; the document itself, and
`AppState::apply_window_layout`, in § "The document" there. What is left here is
the wire:

```rust
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

/// A command whose answer cannot exist until this frame has been rendered.
pub(crate) enum Deferred {
    Screenshot { source: ScreenshotSource, max_dimension: Option<u32>, caption: String },
}

/// Which pixels a deferred screenshot reads.
pub(crate) enum ScreenshotSource {
    /// The presented surface, whole.
    Window,
    /// The presented surface, cropped to a panel's body. The rectangle is not
    /// resolved here: the frame reads it after the egui pass.
    Panel(Tab),
    /// The viewport's own render target — `viewer_3d` with `hud: false`.
    ViewportRender,
}

// read::get_action_log(state, since_revision, limit, &actors) -> JsonReply
// read::ACTION_LOG_DEFAULT_LIMIT: usize = 200;  read::ACTION_LOG_MAX_LIMIT: usize = 1000;

impl App {
    /// Encode the surface copy into this frame's encoder when a window or
    /// panel screenshot is deferred, and hand back the buffer to map. Called
    /// between the egui pass and `present`; `None` in every other frame.
    fn encode_screenshot_copy(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder,
                              surface: &wgpu::Texture) -> Option<SurfaceCopy>;
    /// Map the copy, crop to each named panel's body, encode. Called after the
    /// present.
    fn resolve_mcp_deferred(&mut self, device: &wgpu::Device, queue: &wgpu::Queue,
                            surface: Option<SurfaceCopy>);
}

/// The crop rectangle's derivation, pure over the dock and the frame's scale,
/// which is what puts it under headless test.
fn panel_crop(dock: &DockState<Tab>, panel: Tab, pixels_per_point: f32,
              surface: [u32; 2]) -> Option<[u32; 4]>;

/// Apply one command. **Takes no `App` and no GPU handle** — which is what
/// makes twenty-two of the twenty-three tools testable in a headless
/// `cargo test`.
pub(crate) fn apply_with_window(state: &mut AppState, viewer: &mut Viewer3D,
                                host: &mut dyn WindowHost, command: Command) -> Outcome;

/// The same against a `NoWindow` host, which refuses a window portion with
/// "no window" and leaves every other tool as it is. Compiled for the tests,
/// its only callers: the frame always has a window by the time it drains.
#[cfg(test)]
pub(crate) fn apply(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome;

/// Apply a frame's worth of commands **as the agent**: the Action Log's actor
/// moved to `Mcp` for the batch, a `Query` entry per read, a failed entry per
/// refusal. The drain is this plus the channel, which is what puts the
/// attribution under the same headless test as the vocabulary.
pub(crate) fn apply_as_agent(state: &mut AppState, viewer: &mut Viewer3D,
                             host: &mut dyn WindowHost, commands: Vec<Command>) -> Vec<Outcome>;

/// Start the server. Returns once it is bound and listening, or with the bind
/// error; the runtime lives on its own thread from here.
pub(crate) fn serve(port: u16, tx: UnboundedSender<Request>, proxy: EventLoopProxy<UserEvent>)
    -> Result<SocketAddr, ServeError>;
```

`apply_with_window` taking `(&mut AppState, &mut Viewer3D)` and a trait object
rather than `&mut App` is the one signature worth defending. `App` owns a
`wgpu::Device`, a surface and a window; a test that could construct one would
need a GPU and a display, and the crate's headless lib tests deliberately need
neither. Splitting the GPU-shaped command out into `Outcome::Deferred` —
handled in `mcp::frame`, where the device is passed in — and the window-shaped
one out behind `WindowHost` keeps the whole command vocabulary, its error
messages and its JSON shapes under headless test, and leaves exactly one tool
(`screenshot`) needing a window.

`ToolOutput` has two shapes rather than one because `screenshot` answers with a
picture and the other twenty-two answer with JSON; squeezing an image through a
JSON field would mean a magic key the transport has to know to look for. The
twenty-two return a plain `Result<Value, ToolError>` and are widened at the
`apply_with_window` dispatch, so nothing below it has to name the shape it is
not.

`App` carries three fields for this: `mcp_rx: Option<UnboundedReceiver<Request>>`,
`mcp_deferred: Vec<(Deferred, oneshot::Sender<Reply>)>`, and
`surface_readable: bool` — whether the swapchain took `COPY_SRC`, which is what
a window screenshot's refusal reads. `App::drain_mcp`
takes the frame's `&Arc<Window>` and hands a clone of it to `apply_as_agent` as
the host, which is the whole of what `WindowHost` costs the caller. The request counter the
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
relaunch would call tools the new binary does not have. Twenty-three tools are cheap
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
  layout document that does not validate, a size for a window that is maximized,
  a screenshot of a minimized window, of a panel that is not drawn, or of a
  viewport that has not rendered yet, and the 10 s apply timeout.

The line is whose problem it is: a request that does not fit the advertised
schema is the client's, and a request the viewer will not carry out is the
viewer's.

Domain errors get a message an agent can act on, in the style the viewer already
uses for its status line: what was asked, what is actually there. *"No loaded
reconstruction is labelled `globl` — loaded: `seoul_bull`, `global`."*

**A domain error is also recorded in the Action Log**, as a failed entry reading
`{tool} failed: {message}` — so the human at the window sees the refusal in the
status line and in the panel, where before only a success reached them. It is
recorded by the drain rather than by the method that produced it, which is why
every `AppState` method the MCP layer calls returns its failure instead of
logging it: one failure, one entry. Protocol errors are **not** logged — they
never reach the viewer, and a request the GUI thread never saw belongs in the
agent's own transcript.

**An unknown argument is refused by name rather than ignored.** The schemas say
`additionalProperties: false`, but a schema binds only the clients that enforce
it, and an ignored typo would leave an agent believing it asked for something it
did not — which is the failure the whole return-your-resulting-state design is
shaped to avoid.

## Testing

`crates/sfm-explorer/src/mcp/tests.rs`, headless, no GPU, no window — which is
what the `apply_with_window(&mut AppState, &mut Viewer3D, &mut dyn WindowHost,
…)` signature is for. The fixture is a two-reconstruction scene whose
reconstructions each resolve to **two** intrinsics records, because a
one-camera reconstruction cannot tell the camera-image and camera-intrinsics
selections apart and every coupling rule looks like a no-op against it. A
`FakeWindow` implements `WindowHost` over the three flags a real window keeps
apart, records what it was asked in the order it was asked, and clamps a size
to a minimum the way `with_min_inner_size` does; the fixture seeds
`AppState::window` from one, so the window block has something to report even
where a test hands no host over.

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
- **`get_image_detail_display` returns the defaults on a fresh state**, with
  `intrinsics.enabled` true, `feature_size_px` null and `overlay_mode`
  `"features"`; **`set_image_detail_display`** changes exactly the fields it
  names at either level and the reply equals the next `get`; every one of the
  seven mode names round-trips; a `feature_size_px` set through the tool
  survives one Image Detail panel frame (the toolbar's per-frame re-derivation
  finds the drag values it also wrote); `null` clears the filter and leaves the
  drag values; an unknown mode lists the seven, an off-ladder `distortion_scale`
  or `grid_cols` lists its ladder, `max_features: 0` and a size filter with
  `min > max` are refused, and each refusal leaves the call's good fields
  unapplied — and, being a protocol error, records nothing. The log gets **one
  `Display` entry per changed field** in the table's words, as `Mcp`, and
  nothing for a field set to the value it had; a call that changed three fields
  is read off the revision clock, which ticks once per write whether or not the
  run folds. The texts themselves are asserted one field at a time against the
  differ in `crate::state`, which is where both actors' entries come from.
- **The human's Image Detail changes are logged the same way**: a
  `Context::run_ui` frame that presses `I` over the panel records
  `Intrinsics off` as `User`, and a frame that changes nothing records nothing —
  the differ, not the widget, decides.
- **`get_window_layout` returns the file**: its `window_layout`, parsed back
  through `WindowLayout::from_json`, equals `state.window_layout()`; the `window`
  block beside it is the live one with `monitors`, current first; `panels` has
  all seven, with the default layout's three behind-a-sibling tabs inactive and
  the rest active. A maximized fake makes the two disagree on purpose: the block
  reports the monitor-sized rectangle and the document the one it restores to.
  With no host, `window` is `null` and the document has no `window` section,
  while the panels are still answered.
- **`set_window_layout`** takes back each form the spec lists, including a whole
  `get_window_layout` reply sent back unedited and a file's text parsed and sent
  as the argument; `{}`, `{ "window": {} }` and `{ "sfm_explorer_layout": 2 }`
  are refused at the parse; a `window` portion with no host is refused with the
  "no window" message while a `layout`-only call succeeds there; a document that
  does not validate is refused with the parser's path-carrying message and
  neither the window nor the dock moves. A string other than `"default"` is
  refused, and so is a bad window state, in the parser's own words.
- **The window portion behaves as the table says**: it moves between all four
  states, from each of them to each of them; `normal` from a
  minimized-and-maximized fake clears both flags; a size against a maximized
  window changes what it restores to and leaves it maximized; a piece a call does
  not carry is preserved, so `{ "focus": true }` leaves the state, size and
  position as they were.
- **Both portions in one call**: the window is applied first, and a position
  refusal from the fake stops the call with the dock untouched.
- **The reply is a read-back**: a fake that clamps to its minimum replies with
  the clamped size, while the Action Log row says what was asked for.
- **`show_panel` / `hide_panel`**: hiding closes and reports `open: false`;
  hiding a closed panel succeeds and changes nothing; showing after hiding lands
  the panel in its default group-mate's node and in front; showing an open panel
  raises it and moves nothing else; an unknown name lists the seven.
- **The panel writes record the menu's own entries** — `Closed …`, `Opened …`,
  `Raised …`, and `Reset layout` for `"default"` — each under `Kind::Layout` as
  actor `MCP`, and a document records `Set layout`. A call carrying both portions
  records two rows, the window one first and composed in application order; a
  refusal is one failed row, filed under the kind of the portion the call
  carried. The read records a `Query` that never reaches the status line.
- **`get_scene` embeds the window block** without `monitors`, and carries
  `action_log_revision` equal to the log's clock at the moment it answered.
- **`get_action_log` reads what happened**: the reply shape, row for row;
  `since_revision` returns the gap and only the gap, after a mutating call and
  after a coalesced run; `actors: ["user"]` returns the human's rows and none of
  the agent's, `["mcp"]` the agent's including its queries with `tool` beside
  `kind`, an omitted filter gives every actor, and `[]` and an unknown actor
  name are refused at the parse with the three names listed; `limit` truncates
  with `truncated: true` and the continuation reads the rest; a `limit` above
  the cap is capped; the call itself appears as a `Query` on the next read, and
  a repeat of it coalesces into one row.
- **`screenshot` while minimized is refused** with the message naming
  `set_window_layout`; not minimized, it still defers.
- **`screenshot` defers with what it will photograph**: no panel with
  `ScreenshotSource::Window` and a `window` caption; a panel with
  `Panel(tab)`, the panel's name and its last laid-out size in the caption and
  in the query text; `viewer_3d` keeping the frame description. A closed panel
  is refused naming `show_panel`, one behind another is refused naming the tab
  in front, an unknown name lists the seven, and a `show_panel` followed by a
  `screenshot` of that panel in one batch is accepted.
- **`hud: false` with `viewer_3d`** defers with `ScreenshotSource::ViewportRender`
  and a query text ending `without HUD`; with another panel or with no panel it
  is refused at the parse with its message; `hud: true` is accepted anywhere and
  changes nothing.
- **The crop rectangle**: a dock with known `viewport` rectangles and a scale
  factor of 1.5 gives the expected pixel rectangle, clipped to the frame, and
  `None` for a panel the dock has never laid out. The readback's unpad and its
  BGRA swizzle are tested over a synthetic padded buffer, in `mcp::frame`.
- **A window portion through plain `apply`** — no host — is refused with "no
  window", so a caller that forgets the host fails loudly.
- **The catalog is twenty-three tools**, eight of them reads;
  `set_window_layout`'s schema advertises `sfm_explorer_layout`, `window` and
  `layout`, with the `window` section's five keys under it, and `screenshot`'s
  advertises `panel_name`, `hud` and `max_dimension`.
- **Schema and parser cannot drift**: every property any tool advertises is one
  the parser accepts, walked over the whole catalog rather than tool by tool, so
  a tool added later is covered by construction. The vocabulary rule is asserted
  the same way, including that a panel argument is `panel_name` and never
  `panel`, and that `hud` — the one allowed initialism — is on `screenshot` and
  nowhere else, so a second one cannot arrive quietly.
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

**`screenshot` is the one tool with a real frame behind it**, so it also has
tests in `ui_basic` (`pixi run ui-test`, on all three desktop platforms), which
launch a viewer with `--mcp 0`, read the endpoint off its stdout and speak the
same hand-written HTTP to it. Those assert the size and decodability of the PNG
rather than its pixels, since what a frame *looks* like is not a stable thing
to assert: a
`screenshot` with no arguments decodes to the window's `inner_size`, one of a
panel decodes smaller in both axes, `max_dimension` bounds the longer side, the
render target is inside the `viewer_3d` crop and within a body margin of it, and
a panel that is closed or behind a sibling is refused by the real viewer with
the message the headless tests specify. What the headless tests assert about the
tool is everything before the pixels: that it defers, what it defers with, that
it refuses a minimized window, and that its caption is built while the state is
still borrowed.

**A `set_window_layout` window portion against a real window** is not covered:
maximize, read back `maximized`, restore, read back the original inner size —
the round trip, not pixel positions, since where a window manager actually puts
a window is its business. What `ui_basic` does cover of the document is the
startup load ([panel-layout.md](panel-layout.md) § "Testing").

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
- **No persistence of what an agent did.** The endpoint is not remembered
  between runs, and neither is anything an agent set through it. The viewer does
  restore a layout at startup — the one *the human* saved to
  `~/.sfm-explorer-default-layout.json` ([panel-layout.md](panel-layout.md)
  § "The default layout file")— and an agent that wants its arrangement to
  survive a restart asks the human to save it; `get_window_layout` hands over the
  document to point at.
- **No positioning a panel.** `show_panel` has no `where`: the home rule
  decides, and `set_window_layout` takes the whole tree. A `move_panel` with a
  destination vocabulary is a later question, if the tree turns out to be too
  blunt an instrument.
- **No choosing a monitor by name.** A window portion takes a position; an agent
  that wants the other monitor reads `monitors` and sends a position on it.
- **No exclusive fullscreen.** `fullscreen` is `Fullscreen::Borderless(None)`,
  on the current monitor. Video modes are a different feature.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--mcp PORT` | `8787` (`cli::DEFAULT_MCP_PORT`) | The loopback port to bind. `0` takes an ephemeral one, printed at startup. |
| `list_camera_images` `limit` | `50` (`read::DEFAULT_LIMIT`) | Rows per page. |
| `list_camera_images` `limit` cap | `500` (`read::MAX_LIMIT`) | The most one call will return, whatever it asked for. |
| `screenshot` `max_dimension` | none — the native size of whatever was photographed | Longest side of the returned PNG. |
| `get_action_log` `limit` | `200` (`read::ACTION_LOG_DEFAULT_LIMIT`) | Entries per call. |
| `get_action_log` `limit` cap | `1000` (`read::ACTION_LOG_MAX_LIMIT`) | The most one call will return, whatever it asked for. |
| Apply timeout | `10 s` (`server::APPLY_TIMEOUT`) | How long a tool call waits for the GUI thread. |
| `set_view` `fov_short_axis_deg` | `5`–`160` degrees (`view::MIN_FOV_DEG`, `view::MAX_FOV_DEG`) | Accepted range, matching what interactive FOV zoom clamps to. |
| `set_image_detail_display` `intrinsics.distortion_scale` | `1, 2, 3, 5, 10, 20, 50` (`IntrinsicsDisplaySettings::SCALE_LADDER`), or `null` for auto | The only exaggerations accepted, being the ones the gear popup offers. |
| `set_image_detail_display` `intrinsics.grid_cols` | `8, 12, 16, 24, 32` (`IntrinsicsDisplaySettings::GRID_LADDER`) | The only densities accepted, for the same reason. |
| `set_image_detail_display` `max_features` | `≥ 1`, or `null` for all | `0` is refused: "no features" is `overlay_mode: "none"`. |

## Open questions

- **Discovery for multiple viewers.** With ephemeral ports, an agent has no way
  to find the running instances. A registry file — `~/.sfmtool/explorer-mcp.json`,
  written on bind and removed on exit — would solve it, at the cost of a stale
  file after a crash. Worth it only once running two viewers at once is common.
- **Should the human be able to turn it off?** A `Help ▸ MCP` menu item showing
  the endpoint and a Stop button costs little and is the honest counterpart to
  the title-bar announcement.
- **Screenshot resolution policy.** A native-size window on a 4K display is a
  lot of tokens. `max_dimension` defaulting to something like 1280 might
  serve agents better than defaulting to native — but a downscaled screenshot is
  a worse answer to "is this point cloud noisy". Decide with a real agent in the
  loop.
- **Should `get_scene` carry `window` at all?** It costs a few lines in every
  `get_scene` reply and saves an agent one call before deciding whether to
  screenshot. Kept for now, alongside `window_title`, which it subsumes; drop it
  if `get_scene` replies grow noisy.
- **`focus: true` and the human.** An agent that steals focus while the human is
  typing in another application is being rude. The platform usually prevents it;
  if it does not, the window section's `focus` may want to become "request
  attention" (`Window::request_user_attention`) instead.
- **Reporting the default layout file in `get_window_layout`** (`default_file:
  { path, exists }`), so an agent can tell the human where to save the document
  it just read. Two fields; add if an agent asks.
- **Window and layout notifications.** The human resizing the window or dragging
  a tab reaches the agent only where the Action Log records it, which the dock
  rearrangements deliberately are not ([action-log.md](action-log.md)); a resize
  is still something the agent learns by asking. Same gap as selection changes,
  same answer.
- **An intermediate render target** if a platform's surface refuses `COPY_SRC`.
  None of the three supported backends has, and a per-frame blit for a case that
  has not arisen is the wrong trade until it does.
- **Should a panel screenshot include its tab bar?** The body is what the panel
  shows; the tab bar is the same seven words every time. Excluded; include it if
  an agent needs to see which tab is in front, which `get_window_layout`'s
  `panels` already says.
- **Whether `set_view` should expose the HUD's display controls** (point size,
  EDL thickness, patch opacity). They change what a screenshot shows, so an
  agent evaluating a reconstruction may want them; they are also a long tail of
  knobs that would double the tool's surface. Left out until something asks.
  The Image Detail panel's controls got their own pair
  (`get_image_detail_display` / `set_image_detail_display`) because that panel
  is where "where is the solve wrong" is answered and its controls are a closed
  set; if the HUD's are exposed, a `viewer_3d_display` document of the same
  shape is the precedent to follow, not more fields on `set_view`.
