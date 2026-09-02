# MCP Layout and Window Tools

> **Draft.** This is a change proposal against
> [`specs/gui/mcp-server.md`](../gui/mcp-server.md). It is written as the
> sections that spec gains, so that filing it is a merge rather than a
> rewrite; § "Filing" at the end says where each part goes. Everything else
> in `mcp-server.md` — the vocabulary rules, addressing, threading, transport,
> security, errors — holds unchanged and is the frame this fits into.

## Purpose

An agent driving SfM Explorer over MCP can already read the scene, move the
selection and the 3D camera, and take a picture of the viewport. It cannot
touch the window those things happen in: which panels are open and how they
are arranged, and how big the window is and where it sits on the desktop. A
human and an agent sharing one viewer keep running into that. The agent wants
the 3D viewport to fill the window before a screenshot and the Action Log out
of the way; the human wants the Action Log in front to see what the agent just
did, and the window back to the size they had it. Six tools close the gap:
four for the panel layout, carrying exactly the JSON document the Panels menu
saves and loads ([panel-layout.md](../gui/panel-layout.md)), and two for the
window's state, size and position.

Both halves follow the surface's existing shape. A write replies with the
resulting state rather than an acknowledgement, a piece the call does not
carry is preserved, a combination with no answer is refused up front, and
everything an agent does is in the Action Log under the `MCP` actor where the
human can see it.

## The tool surface (additions)

Twenty-two tools: seven read, fourteen write, one observe.

| Tool | Kind | What it does |
|------|------|--------------|
| `get_layout` | read | The panel arrangement, as the layout file spells it, plus each panel's open state |
| `get_window` | read | The window's state, size, position, scale factor, and the monitors |
| `set_layout` | write | Replace the whole arrangement with a layout document, or with the default |
| `show_panel` | write | Open a panel at its home position, or raise it if it is open |
| `hide_panel` | write | Close a panel |
| `set_window` | write | Change the window's state, size, position, or focus, a piece at a time |

The four layout writes and `set_window` carry `destructiveHint: false` like
every other write; nothing here touches a file on disk.

### Vocabulary

**`panel`** is the entity, spelled as the Panels menu and
[panel-layout.md](../gui/panel-layout.md) spell it. A panel has one handle,
its name, so the argument is **`panel_name`** — the `<entity>_<attribute>`
rule, exactly as `reconstruction_label` carries a label and
`camera_intrinsics_index` an index. The seven names are the layout file's:
`scene`, `viewer_3d`, `image_browser`, `image_detail`, `point_track`,
`intrinsics`, `action_log` (`Tab::wire_name`). An unknown name is refused with
a message listing the seven (`Tab::all_wire_names`).

**`layout`** is the whole document, so the argument holding one is `layout`.

**The window has no GUI word, so the code's word wins**: `outer_position`,
`outer_size`, `inner_size`, `scale_factor` are `winit::window::Window`'s
method names, greppable from the wire. The four states are winit's flags
spelled as adjectives — `maximized`, `minimized`, `fullscreen` — plus
`normal` for none of them, which is the word the user and Win32's
`SW_SHOWNORMAL` both use.

## Layout

### `get_layout`

No arguments.

```jsonc
{
  "layout": {                       // the layout file, verbatim (panel-layout.md § "The layout file")
    "sfm_explorer_layout": 1,
    "main": { "split": "left_right", "fraction": 0.18,
              "first": { "tabs": ["scene"], "active": "scene" },
              "second": { /* … */ } },
    "windows": []
  },
  "panels": {                       // derived from `layout`: one entry per panel, always all seven
    "scene":         { "open": true,  "active": true },
    "viewer_3d":     { "open": true,  "active": true },
    "image_browser": { "open": true,  "active": true },
    "image_detail":  { "open": true,  "active": true },
    "point_track":   { "open": true,  "active": false },
    "intrinsics":    { "open": true,  "active": false },
    "action_log":    { "open": false, "active": false }
  }
}
```

**`layout` is the file.** Not a rendering of it, not a subset: the object
`Layout::to_json` writes, parsed. An agent that saves it to disk has a file the
Panels menu loads, and a file the human saved is an argument `set_layout`
takes. One schema, one parser, one set of validation messages.

**`panels` is the same information, indexed the other way.** "Is the Action
Log open" should not cost the agent a tree walk. `open` is whether the panel
appears anywhere in `layout`; `active` is whether it is the front tab of its
node — every panel alone in a node is active, and in the default layout the
two multi-tab nodes each have one. A closed panel is `active: false`.

### `set_layout`

```jsonc
{ "layout": { "sfm_explorer_layout": 1, "main": { /* … */ }, "windows": [] } }
{ "layout": "default" }           // the stock seven-panel grid, as Reset Layout
```

The reply is `get_layout`'s.

The document form goes through `Layout::from_json`'s rules unchanged — the
version tag is required, every panel at most once, no empty leaves, no
unknown keys — and a violation is a domain error carrying the parser's own
message with its path: *"main.second.first: unknown key `fracton`"*. A
refused document leaves the dock untouched, as a refused Load Layout does.

`"default"` is the one string the field accepts. It is a named layout rather
than a separate `reset_layout` tool because it *is* setting the layout, to
the one arrangement that has a name; an agent that has just made a mess of the
window wants one call back, not a document it has to reconstruct.

`set_layout` **replaces** the arrangement — a panel absent from the document
is closed. Every panel keeps its state either way (a re-opened Image Detail
shows the image it had), because panel structs live for the process and the
dock only decides which of them draw.

### `show_panel` / `hide_panel`

```jsonc
// show_panel  { "panel_name": "action_log" }
// hide_panel  { "panel_name": "action_log" }
```

Both reply with `get_layout`'s block, so the agent sees where the panel
landed rather than assuming.

`show_panel` is `AppState::show_panel`: the three home-position rules of
[panel-layout.md](../gui/panel-layout.md) § "Home positions", so a panel the
agent opens appears where the menu would have put it. On a panel that is
already open it raises — makes it the active tab of its node — which is the
call an agent makes to put the Action Log in front of the Image Browser for
the human without moving anything.

`hide_panel` is `AppState::hide_panel`, and **is idempotent**: hiding a panel
that is closed succeeds and changes nothing, exactly as the method does. Both
tools *set* rather than toggle, for the reason `set_solo` does — an agent
issuing a toggle cannot know the outcome without reading first, and a retried
call would undo itself.

Neither takes a position. Where a panel goes is the home rule's decision;
an agent that wants a panel *there* sends the tree through `set_layout`.

### In the Action Log

The three writes go through the same `AppState` methods the menu uses, so
the entries are the menu's — `Opened Action Log panel`, `Raised Action Log
panel`, `Closed Action Log panel`, `Reset layout` — under actor `MCP`.
`set_layout` with a document records `Set layout` (`AppState::apply_layout`
records nothing itself, because its two callers word the entry differently:
the menu says which file, the tool says which tool). All of them are
`Kind::Layout`, non-coalescing. `get_layout` is a `Query` entry, `get_layout`,
and never reaches the status line.

## Window

### The window block

`get_scene` gains a `window` field beside `window_title`, and `get_window`
returns the same block with one addition.

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
    "monitors": [ /* get_window only: every monitor, same shape as `monitor`, the current one first */ ]
  }
}
```

**Physical pixels throughout, with the scale factor beside them.** They are
what winit reports natively, what the view block's `viewport_px` and a
screenshot's dimensions are in, and what a monitor's size is in — so an agent
comparing "the window" against "the picture I was handed" compares like with
like. The logical size is under `derived`, next to the scale factor it is
computed from, for the agent that wants to reason in the units the window
was created in (`1280 × 720` logical, `800 × 600` minimum).

**`state` is one word for four flags.** winit exposes `is_minimized`,
`is_maximized` and `fullscreen` separately, and a window can be minimized
*and* maximized at once (a maximized window the user minimized; restoring it
brings the maximized one back). The block reports the one that governs what
the user sees, in the order of precedence `minimized` > `fullscreen` >
`maximized` > `normal`, because that is the question an agent asks — "can the
human see this window, and how much of the desktop is it" — and the flags
underneath it are the viewer's business. `is_minimized` returning `None`
(a platform that cannot say) reads as not minimized.

**`outer_position` can be `null`.** Wayland does not tell a window where it is.
The field says so rather than reporting `[0, 0]`, and `set_window`'s
`outer_position` is refused on such a platform for the same reason.

**`derived.monitor_fraction` is the answer to "how much of the desktop"**, per
axis, with the area under it; both are `null` when there is no current
monitor to compute against. A window straddling two monitors reports the
one winit calls current.

### `set_window`

```jsonc
{ "state": "maximized" }                     // "normal" | "maximized" | "minimized" | "fullscreen"
{ "state": "normal" }                        // restore: un-minimize, un-maximize, leave fullscreen — all three
{ "inner_size": [1600, 900] }                // physical px; the drawable area
{ "outer_position": [100, 50] }              // physical px, desktop coordinates
{ "focus": true }                            // bring to the front, where the platform allows it

// pieces combine when each has an answer
{ "state": "normal", "inner_size": [1600, 900], "outer_position": [100, 50] }
```

The reply is `{ "window": … }`, the block `get_window` returns.

**What a call does not carry is preserved**, as in `set_view`. The pieces are
applied in a fixed order — `state`, then `outer_position`, then `inner_size`,
then `focus` — so that a call carrying several reads as one sentence: "make
it normal, put it here, this big, and in front".

**Geometry needs a normal window, and the call is refused otherwise.** A
maximized, minimized or fullscreen window's size and position belong to the
window manager, and asking for a size the OS will immediately overrule is a
call with no honest answer. So `inner_size` and `outer_position` are accepted
when the window is `normal` after the `state` piece — either it already was,
or the call says `"state": "normal"` — and refused, naming the state, when the
call would leave it anything else. `{ "state": "minimized", "inner_size":
[…] }` is refused up front; `{ "inner_size": […] }` against a maximized
window is refused with *"The window is maximized; send `state: "normal"` in
the same call to resize it."*

**`normal` means all three flags off**, applied minimized → fullscreen →
maximized, because restoring a minimized window can bring a maximized one
back and the agent asked for normal, not for whatever was underneath.

**`inner_size` is clamped by the window's own minimum** (`800 × 600` logical,
`WindowAttributes::with_min_inner_size`) and by the platform; the reply
reports what the window actually became, which is why the reply is a
read-back and not an echo. `focus` may be declined by a platform that does
not let applications steal focus; `focused` in the reply says whether it
was.

**The change is applied at the top of the frame**, in the drain, before egui
reads the window size for that frame's layout — so the frame the request
woke is laid out at the new size and a `screenshot` in the *next* call sees
it. The reply is read back from the window after the present, by which time
the OS has processed the change on Windows (the calls are synchronous there).
On a platform that animates or defers window changes the read-back may be a
frame early; an agent that needs certainty confirms with `get_window`.

### `screenshot` while minimized

A `screenshot` with `state: "minimized"` is refused rather than attempted:

> The window is minimized, so nothing is being rendered to photograph. Send
> `set_window { "state": "normal" }` first.

Whether a minimized window's swapchain still presents is platform-dependent,
and a picture of a window the human cannot see answers nothing an agent asked
of a shared viewer. The check is in `apply`, against the window block's
`state`, so it is under headless test with the rest of the vocabulary.

### In the Action Log

One entry per `set_window` call, `Kind::Window`, non-coalescing, composed
from the pieces the call carried in application order: `Maximized window`,
`Restored window`, `Minimized window`, `Made window fullscreen`, `Moved window
to (100, 50)`, `Resized window to 1600×900`, `Focused window` — joined with
`; ` when a call carries more than one, so `Restored window; resized window to
1600×900` is one row for one call. A refusal is the usual failed entry,
`set_window failed: …`. `get_window` is a `Query` entry.

## The Rust seam (additions)

```rust
pub(crate) enum Command {
    // … the sixteen …
    GetLayout,
    GetWindow,
    SetLayout { layout: LayoutTarget },
    ShowPanel { panel: Tab },
    HidePanel { panel: Tab },
    SetWindow { change: WindowChange },
}

/// `set_layout`'s argument: a parsed, validated document, or the default.
pub(crate) enum LayoutTarget { Document(Layout), Default }

/// `set_window`'s pieces. `None` is "leave it".
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct WindowChange {
    pub(crate) state: Option<WindowState>,
    pub(crate) outer_position: Option<[i32; 2]>,
    pub(crate) inner_size: Option<[u32; 2]>,
    pub(crate) focus: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowState { Normal, Maximized, Minimized, Fullscreen }

/// The window as last observed: the block `get_window` renders, minus
/// `monitors`. Lives on `AppState` (`AppState::window`, `None` until the
/// window exists), refreshed at the top of every frame and again after every
/// applied `WindowChange`, so reads are plain and headless.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowInfo {
    pub(crate) state: WindowState,
    pub(crate) focused: bool,
    pub(crate) scale_factor: f64,
    pub(crate) outer_position: Option<[i32; 2]>,
    pub(crate) outer_size: [u32; 2],
    pub(crate) inner_size: [u32; 2],
    pub(crate) monitor: Option<MonitorInfo>,
}
pub(crate) struct MonitorInfo { name: Option<String>, position: [i32; 2], size: [u32; 2], scale_factor: f64 }

/// What `apply` needs from the window, and all it needs: the one command
/// that must reach `winit` goes through here. Implemented for `Arc<Window>`
/// in `mcp::frame`, and for a fake in the tests.
pub(crate) trait WindowHost {
    /// Apply the pieces in the fixed order. `Err` only for a piece the
    /// platform cannot do at all (a position on Wayland); a clamp is not an
    /// error, the read-back reports it.
    fn apply(&mut self, change: &WindowChange) -> Result<(), ToolError>;
    /// The window as it is now, including every monitor.
    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)>;
}

/// Unchanged signature; `set_window` through it is refused with "no window",
/// which is the headless case.
pub(crate) fn apply(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome;
/// The real dispatch. `apply` is this with a `NoWindow` host.
pub(crate) fn apply_with_window(state: &mut AppState, viewer: &mut Viewer3D,
                                window: &mut dyn WindowHost, command: Command) -> Outcome;
pub(crate) fn apply_as_agent(state: &mut AppState, viewer: &mut Viewer3D,
                             window: &mut dyn WindowHost, commands: Vec<Command>) -> Vec<Outcome>;
```

**Why a host trait and not a third `Outcome` variant.** `screenshot` defers
because its answer does not exist until the frame has been rendered; a window
change is different — it can be applied on the spot, its effect is wanted in
*this* frame, and its answer is a read-back that a fake can produce as well
as a real window. Routing it through `Deferred` would apply it after the
present, a frame late, and would leave the ordering of `set_window` then
`get_window` in one batch wrong. A trait with two methods keeps `apply`
free of `winit` exactly as it is free of `wgpu`, and puts `set_window`'s
refusals, its application order and its log line under headless test against
a `FakeWindow` that records what it was asked and reports what it then is.

**`AppState::window` is refreshed twice.** Once at the top of
`run_ui_and_paint`, before the drain, from `WindowHost::observe` — so
`get_scene`'s block and a `screenshot`'s minimized check see this frame's
window — and again inside `apply_with_window` after a successful
`WindowChange`, so a `get_window` later in the same batch, and the
`set_window` reply itself, see the change. The refresh is a handful of
`Win32` calls a frame and the viewer renders no frames while idle.

**`Command::kind`** maps the three layout writes to `Kind::Layout`,
`set_window` to a new `Kind::Window` (non-coalescing, label `Window`), and
the two reads to `Query`.

**`tools::parse`** builds `LayoutTarget::Document` through `Layout::from_json`
on the re-serialized argument object, so the wire and the file share one
parser and one set of messages; the `"default"` string is matched before
that. `panel_name` resolves through `Tab::from_wire_name`. `set_window`'s
state-versus-geometry rule is checked in `parse` where the call's own fields
are visible (`minimized` with a size), and in `apply_with_window` where the
window's current state is (a size against a maximized window with no
`state: "normal"`).

## Testing

`crates/sfm-explorer/src/mcp/tests.rs`, headless, extending the existing
module. A `FakeWindow` implements `WindowHost` over a `WindowInfo` it
mutates on `apply` and returns on `observe`; `two_reconstructions()` seeds
`AppState::window` from it.

- **`get_layout` returns the file.** Its `layout` parsed back through
  `Layout::from_json` equals `state.layout()`; `panels` has all seven, with
  the default's two inactive tabs (`point_track`, `intrinsics`) and the
  rest active.
- **`set_layout` with a document** applies it and replies with it; **with
  `"default"`** after a `hide_panel` restores all seven; **with a bad
  document** is refused with the parser's path-carrying message and leaves
  `state.layout()` unchanged. A string other than `"default"` is a protocol
  error from `parse`.
- **`show_panel` / `hide_panel`**: hiding closes and reports `open: false`;
  showing after hiding lands the panel in its group-mate's node; showing an
  open panel reports `active: true` and moves nothing else; hiding a closed
  panel succeeds and changes nothing; an unknown name lists the seven.
- **Action Log**: the four layout writes record their `Kind::Layout` entries
  under actor `MCP`; `get_layout` records a `Query`.
- **`get_scene` embeds `window`**, and `get_window` adds `monitors` with the
  current monitor first.
- **`set_window` state transitions** through the fake: each of the four
  states from each other; `normal` from a minimized-and-maximized fake clears
  both.
- **Geometry rules**: a size with `minimized` is refused in `parse`; a size
  against a maximized fake without `state: "normal"` is refused naming the
  state; with it, applied in order (state, position, size, focus) — the fake
  records the order.
- **Preservation**: a `set_window { "focus": true }` leaves size, position
  and state as they were.
- **The reply is a read-back**: a fake that clamps sizes to its minimum
  replies with the clamped size, not the requested one.
- **`screenshot` while minimized is refused** with the message naming
  `set_window`; not minimized, it still defers.
- **`set_window` through plain `apply`** (no host) is refused with "no
  window", so a test that forgets the host fails loudly.
- **The catalog**: the vocabulary test passes with the six new tools (no
  `panel` argument, only `panel_name`); the schema-versus-parser walk covers
  them by construction; `only_the_reads_are_annotated_read_only` gains
  `get_layout` and `get_window`.

A real `set_window` against a real window belongs in `ui_basic` (Windows and
macOS, `pixi run ui-test`): maximize, read back `maximized`, restore, read
back the original inner size — asserting the round trip, not pixel
positions.

## Non-goals

- **Positioning a panel.** `show_panel` has no `where`; the home rule
  decides, and `set_layout` takes the tree. A `move_panel` with a destination
  vocabulary is a later question, if the tree turns out to be too blunt.
- **Choosing a monitor by name.** `set_window` takes a position; an agent
  that wants the other monitor reads `monitors` and sends a position on it.
- **Exclusive fullscreen.** `fullscreen` is `Fullscreen::Borderless(None)`,
  the current monitor. Video modes are a different feature.
- **Window notifications.** The human resizing the window is still something
  the agent learns by asking. This is the same gap as selection changes, and
  has the same answer (§ "Open questions" in `mcp-server.md`).
- **Full-window screenshot.** Now that an agent can arrange panels it will
  want to see them; that is the renderer change already listed as a
  candidate, not this one.

## Open questions

- **Should `get_scene` carry `window` at all?** It costs a few lines in every
  `get_scene` reply and saves an agent one call before deciding whether to
  screenshot. Kept in for now, alongside `window_title`, which it subsumes;
  drop it if `get_scene` replies grow noisy.
- **`focus: true` and the human.** An agent that steals focus while the human
  is typing in another application is being rude. The platform usually
  prevents it; if it does not, the tool may want to become "request
  attention" (`Window::request_user_attention`) instead.

## Filing

When this is implemented, fold it into `specs/gui/mcp-server.md` rather than
filing it beside it:

- § "The tool surface": the count becomes twenty-two, the table gains the six
  rows above, and the annotation sentence's counts follow.
- § "The wire vocabulary": a `#### panel` subsection from § "Vocabulary"
  above, beside `reconstruction` and `camera_image`; the window sentence
  joins "Where the GUI has no word, the code's word wins".
- § "`get_scene`": the example gains `window`, with a pointer to the new
  § "The window block".
- New sections after § "`screenshot`": § "`get_layout`", § "`set_layout`",
  § "`show_panel` / `hide_panel`", § "The window block", § "`set_window`",
  with the minimized rule added to § "`screenshot`".
- § "Threading": the `WindowHost` paragraph and the twice-refreshed snapshot;
  the sentence "fifteen of the sixteen tools" becomes "twenty-one of the
  twenty-two".
- § "The Rust seam": the `Command` variants, `WindowChange`, `WindowInfo`,
  `WindowHost`, `apply_with_window`; the module table gains a row for where
  the layout and window commands live (`mcp/layout.rs` and `mcp/window.rs`,
  or one module — implementer's call, say which).
- § "Testing": the bullets above.
- § "Non-goals" and § "Open questions": the entries above, merged.
- [panel-layout.md](../gui/panel-layout.md) § "Non-goals": the last bullet
  becomes a pointer to the filed sections.
- [action-log.md](../gui/action-log.md) § "What gets logged": the `Window`
  kind.
- Delete this file.
