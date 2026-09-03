# Panel Layout

## Purpose

SfM Explorer's window is a set of docked panels — the scene tree, the 3D
viewport, the image strip, the detail views, the action log — that the user can
drag into any arrangement, in a window they put where they want it on the
desktop. This spec covers the lifecycle of that arrangement: closing a panel and
getting it back, resetting to the stock layout, and saving the whole thing —
window and panels — to a JSON file so it comes back, in this session, in another,
or at the next start. The file's schema is the viewer's own, written to be read
and written by hand, and it is the one description of a layout the viewer has:
the same schema an agent driving the viewer over MCP reads and writes
([mcp-server.md](mcp-server.md) holds the tools; this spec holds the
shape).

The window's placement is in the same document as the panels because the two are
one thing to the person sitting at it: "my viewer, maximized on the left
monitor, with the Action Log along the bottom" is one arrangement, and saving
half of it is not saving it.

The reason a panel can be closed at all is that the window is increasingly
shared. A human and an agent looking at the same viewer want different things
from it at different moments — the agent wants the 3D viewport as large as
possible before a screenshot, the human wants the Action Log in front to see
what the agent just did — and a fixed seven-panel grid serves neither. What
makes closing safe to allow is that there is a way back: a panel with no way
back is a trap. The **Panels** menu is that way back — it lists all seven,
ticks the open ones, and re-opens a closed one with a click.

## What the user sees

### Closing a panel

Every tab carries a close button, and every panel can be closed, the 3D Viewer
included — a window showing only the Image Detail panel is a legitimate way to
study a feature overlay. Closing the last tab in a dock node removes the node and
its neighbour takes the space, as `egui_dock` does by default. Closing every
panel leaves an empty dock and an intact menu bar; the Panels menu is how
anything comes back.

A closed panel keeps its state. Panel structs (`ImageBrowser`, `ImageDetail`,
`PointTrackDetail`, `IntrinsicsDetail`, `SceneGraphPanel`, `Viewer3D`) live on
`App` for the life of the process and are not touched by closing; the dock
merely stops asking them to draw. Re-opening the Image Detail panel shows the
image and zoom it had when it closed. This is the same mechanism that already
covers a tab behind another tab in the same node.

### The Panels menu

Between **Go** and the right-hand end of the menu bar:

```
Panels
  ✓ Scene
  ✓ 3D Viewer
  ✓ Image Browser
  ✓ Image Detail
    Point Track
    Camera Intrinsics
    Action Log
  ─────────────
    Reset Layout
  ─────────────
    Save Layout...
    Load Layout...
```

The seven entries are in default-layout order, each a checkbox reading the
panel's open state. Clicking a **ticked** entry closes that panel, exactly as
its tab's close button does. Clicking an **unticked** entry opens it, at its
home position (§ "Home positions"), and makes it the active tab of its node so
the click has something to show for itself. There is no "raise" entry: a ticked
panel is one the user can see the tab of.

**Reset Layout** replaces the whole panel arrangement — main surface and any
floating windows — with `Layout::default()`, the seven-panel grid
[multi-panel-image-browser.md](multi-panel-image-browser.md) § "Default
Layout" describes. Panel state survives, as it does for a close. It leaves the
*window* where the user put it: their window is theirs, and a panel reset is not
a reason to move it.

**Save Layout...** opens a save dialog (filter `*.json`) and writes the window's
placement and the panel arrangement as § "The window layout file". The dialog
starts in the home directory with `.sfm-explorer-default-layout.json` filled in,
so the common case — "keep it like this" — is Save Layout..., Enter, and the
viewer comes up that way next time (§ "The default layout file"). Another name
or another directory is the uncommon case and works as any save dialog does.

**Load Layout...** opens an open dialog with the same filter, starting in the
home directory, and applies the whole document: the window first, then the
panels. A file that does not parse, does not validate, or asks for something the
platform will not do is refused as a whole — the window and the layout on screen
are untouched — and the reason is recorded in the Action Log as a failed
`Layout` entry, which puts it on the viewport status line.

### Home positions

A panel opened from the menu lands where the default layout would have put it,
as nearly as the current arrangement allows. Three rules, tried in order:

1. **Already open:** raise it — make it the active tab of the node it is in,
   and focus that node. Nothing moves. (The menu never shows this case as a
   click target, since the entry is ticked, but the same operation serves
   `Go ▸ Go to Point`'s raise of the Point Track panel and, later, an MCP
   `show_panel`.)
2. **A default group-mate is open:** push it into that node, behind the
   current tabs, then make it active. The default layout has two multi-tab
   groups — Image Detail / Point Track / Camera Intrinsics, and Image Browser /
   Action Log — and a panel from either goes home to whichever of its
   group-mates is still there.
3. **Otherwise, split the main surface's root** along the panel's home edge,
   at its home fraction, and put the panel in the new node:

   | Panel | Home edge | New node's share |
   |-------|-----------|------------------|
   | Scene | left | 0.18 |
   | 3D Viewer | *(takes the root)* | — |
   | Image Browser, Action Log | below | 0.20 |
   | Image Detail, Point Track, Camera Intrinsics | right | 0.33 |

   "Takes the root" for the 3D Viewer means: if the dock is empty it becomes
   the root leaf; if not, it is pushed into the root's first leaf as a new
   active tab, because the viewport is what everything else is arranged
   around and there is no edge that is home to it. Any panel opened into an
   **empty** dock becomes its root leaf, whatever the table says.

Rule 3 approximates the default layout rather than reproducing it — a bottom
strip re-opened this way runs under the Scene panel as well, where the default
has it under the rest — and that is deliberate. Reproducing the default would
mean reasoning about what the user has since done to the other nodes, and a
user who wants the exact grid has Reset Layout. What rule 3 guarantees is that
the panel appears somewhere predictable, of a sensible size, in one click.

The home table is checked by `layout::tests`, so the numbers here are the
numbers in the code (§ "Parameters").

### In the Action Log

Every menu action records an entry of its own kind, `Kind::Layout`,
non-coalescing — two panels closed in a row are two events:

| Action | Entry |
|--------|-------|
| close (tab button or menu) | `Closed Action Log panel` |
| open (menu) | `Opened Action Log panel` |
| raise | `Raised Point Track panel` |
| Reset Layout | `Reset layout` |
| Save Layout... | `Saved layout to C:/Users/mark/.sfm-explorer-default-layout.json` |
| Load Layout..., and the startup load | `Loaded layout from C:/Users/mark/.sfm-explorer-default-layout.json` |
| a load refused | *failed:* `Load layout from …: <reason>` |

A load is one row whatever the file carried — the file is the action, and a
document that moved the window and rearranged the panels did one thing the
person asked for. (An MCP `set_window_layout` is the other case: nobody asked
for a *file*, so it records a row per portion. See
[mcp-server.md](mcp-server.md).)

Drag rearrangements the user makes with the mouse — moving a tab, resizing a
split, floating a node — are **not** logged. `egui_dock` reports none of them as
events, and diffing the tree every frame to synthesize entries would log a
split drag as hundreds of resizes. The Action Log records what was *asked for*,
by menu or, later, by tool call; the arrangement itself is what
`Layout::from_dock` reads back at any moment.

## The window layout file

One JSON document, version **2**, with two sections after the version tag:
`window`, the window's placement, and `layout`, the panel arrangement. Either
may be absent. The default arrangement, as the viewer writes it when it has not
observed a window (a headless `AppState`):

```json
{
  "sfm_explorer_layout": 2,
  "layout": {
    "main": {
      "split": "left_right",
      "fraction": 0.18,
      "first": {
        "tabs": ["scene"],
        "active": "scene"
      },
      "second": {
        "split": "top_bottom",
        "fraction": 0.8,
        "first": {
          "split": "left_right",
          "fraction": 0.67,
          "first": {
            "tabs": ["viewer_3d"],
            "active": "viewer_3d"
          },
          "second": {
            "tabs": ["image_detail", "point_track", "camera_intrinsics"],
            "active": "image_detail"
          }
        },
        "second": {
          "tabs": ["image_browser", "action_log"],
          "active": "image_browser"
        }
      }
    },
    "windows": []
  }
}
```

That is the file byte for byte: two-space indentation, one key to a line, a
leaf's `tabs` kept on one line because a list of panel names reads as a list,
and a trailing newline. `layout::tests` compares
`WindowLayout::default().to_json()` against exactly this document, so the two
cannot drift. With a window observed, the `window` section comes between the tag
and `layout`:

```json
{
  "sfm_explorer_layout": 2,
  "window": {
    "state": "maximized",
    "outer_position": [120, 64],
    "inner_size": [1280, 720],
    "monitor": { "position": [0, 0], "size": [3840, 2160] }
  },
  "layout": {
    "main": { "…": "as above" },
    "windows": []
  }
}
```

### The `window` section

| Key | Type | Meaning |
|-----|------|---------|
| `state` | `"normal"` \| `"maximized"` \| `"minimized"` \| `"fullscreen"` | What the window is shown as — the one word the window block reports. |
| `outer_position` | `[x, y]`, physical px, desktop coordinates | The top-left corner of the window's **normal rectangle** (below). Omitted where the platform cannot say (Wayland). |
| `inner_size` | `[width, height]`, physical px | The drawable area of the window's normal rectangle. |
| `monitor` | `{ "position": [x, y], "size": [width, height] }`, physical px | The monitor the normal rectangle was measured on: the current monitor's rectangle as the window block reports it, without its name or scale factor. Written whenever the snapshot has a monitor; read as § "Fitting a rectangle to the desktop" says. |
| `focus` | `true` | Wire-only in practice: bring the window to the front. Never written by the viewer; accepted on read so the file and the MCP argument have one parser. `false` is refused. |

Every key is optional. A `window` section that is absent, or `null`, leaves the
window alone — which is how a file saved by a headless `AppState`, or a
`set_window_layout` that only carries panels, reads.

**The rectangle is the window's *normal* rectangle**: the one it has when it is
`normal`, and the one it comes back to when a maximized, fullscreen or minimized
window is restored. That is the rectangle that survives a maximize/restore round
trip, and it is the one a saved layout needs — "maximized on the left monitor"
is a normal rectangle on that monitor, plus the word `maximized`. `winit` does
not report a non-normal window's restored rectangle, so the viewer **remembers
the last normal rectangle it observed** (`AppState::window_normal_rect`,
refreshed every frame the window reads as `normal`), and that is what the section
carries. For a normal window it is the current rectangle, exactly as the window
block reports it.

**The units are physical pixels**, as the window block's are, and for the same
reason: they are what `winit` reports, what a monitor's size is in, and what a
screenshot's dimensions are in. A layout saved on a 150 % monitor and loaded on a
100 % one comes back at the same pixel size, hence larger in points; that is the
honest reading of "the rectangle it occupied on the desktop", and the scale
factor is in the window block for an agent that wants to correct for it.

### Fitting a rectangle to the desktop

A layout saved at one desk and loaded at another — a laptop undocked from its 4K
monitor, a file an agent wrote on a different machine — carries a rectangle that
may be nowhere the human can see. `monitor` is what makes that recoverable: it
says which rectangle of the desktop the window's rectangle was measured against,
and so what *share* of a monitor the window occupied.

**When a section carries geometry and a `monitor`, the rectangle is fitted
before it is applied**, by one rule with two exits:

1. **The rectangle is used as saved** when a current monitor has the saved
   monitor's `position` and `size` — the desktop is the one the file was written
   at, snapped and straddling windows included — **or** when the rectangle
   (`outer_position` and `inner_size`) lies entirely within some current monitor.
   In either case it is visible where it was.
2. **Otherwise it is mapped proportionally** from the saved monitor onto the
   **target monitor**, the one the window is currently on: each edge keeps its
   fraction of the monitor, per axis. With `s` the saved monitor and `t` the
   target,

   ```
   x' = t.x + (x − s.x) · t.width  / s.width        width'  = width  · t.width  / s.width
   y' = t.y + (y − s.y) · t.height / s.height       height' = height · t.height / s.height
   ```

   rounded to whole pixels, `width'` and `height'` at least 1. A window that took
   the left half of a 3840 × 2160 monitor takes the left half of a 1920 × 1080
   one. The aspect ratio follows the monitors' rather than being preserved,
   because "the same share of the desktop" is the thing being restored; a
   rectangle that stuck partly off its saved monitor sticks off the target by the
   same fraction, which is left to the window manager as it was.

The fit is applied to `outer_position` and `inner_size` together when both are
present; a section with only one of them and a `monitor` fits that one (a size
alone is scaled; a position alone is mapped). A section **without** `monitor` is
applied as written — there is nothing to fit against, and an agent that sends a
bare rectangle over MCP means that rectangle. The target monitor is the
snapshot's `monitor`; where the snapshot has none, the rectangle is applied as
written.

The fit is a pure function over the saved section and the monitor list
(`window::fit_to_monitor`), computed in `AppState::apply_window_layout` before
the change reaches the host, so it is under headless test with the rest and the
host's `apply` never sees a `monitor`. What the window became is read back
afterwards as always, and the Action Log row says the rectangle was fitted
([action-log.md](action-log.md)).

### How a `window` section is applied

**Geometry first, on a normal window; then the state; then focus.** When the
section carries `outer_position` or `inner_size`, the window is made normal (all
three of minimized, fullscreen and maximized cleared, minimized first), moved,
and sized — so the geometry sets the normal rectangle whatever the window was
showing as. Then the state piece is applied: the one the section names, or, when
it names none and geometry was applied, **the state the window was in before**,
put back. Then `focus`, last.

So every combination reads as one sentence:

| Section | Reads as |
|---------|----------|
| `{ "state": "maximized" }` | maximize where it is |
| `{ "state": "normal" }` | restore — all three flags off |
| `{ "inner_size": [1600, 900] }` against a normal window | resize |
| `{ "inner_size": [1600, 900] }` against a maximized window | its restored size is now 1600 × 900; it stays maximized |
| `{ "state": "maximized", "outer_position": [120, 64], "inner_size": [1280, 720] }` | the normal rectangle is this, shown maximized — on the monitor that rectangle is on |
| `{ "state": "minimized", "inner_size": [1280, 720] }` | restored size 1280 × 720, and minimized |

Applying the geometry to the *normal* rectangle is what lets a maximized layout
be saved and restored at all: a size sent to a window the window manager owns
would otherwise be a request with no honest answer.

**A platform refusal stops the call.** `outer_position` on a platform that does
not let an application place its own window (Wayland), or any window piece where
there is no window (a headless `AppState`), is refused with the message the host
gives, and — since the window portion is applied before the panel portion — the
panel portion is not applied. The document is validated whole before anything is
applied, so a *validation* refusal touches nothing; a *platform* refusal can only
come from the host, which applies the pieces in order and stops at the one it
cannot do.

### The `layout` section

An object with `main` and `windows`, both optional: the split tree of the main
surface and the floating windows. Or the string **`"default"`**, the stock
seven-panel grid — a file that says `"layout": "default"` is a reset. The viewer
never writes that form. The section may be absent or `null`, meaning the panels
are left alone.

#### Vocabulary

**Panel names** are the panel titles, lower-cased and joined with underscores,
which makes them greppable against the `Tab` enum and readable in a diff:

| Name | Tab | Title |
|------|-----|-------|
| `scene` | `SceneGraph` | Scene |
| `viewer_3d` | `Viewer3D` | 3D Viewer |
| `image_browser` | `ImageBrowser` | Image Browser |
| `image_detail` | `ImageDetail` | Image Detail |
| `point_track` | `PointTrackDetail` | Point Track |
| `camera_intrinsics` | `IntrinsicsDetail` | Camera Intrinsics |
| `action_log` | `ActionLog` | Action Log |

The word is **panel**, on the wire as in the specs, which use it several hundred
times against a handful of "pane". `Tab` stays the Rust name: it is
`egui_dock`'s word for the thing in a node, and the code is not the wire.

**A split names its arrangement, not its divider.** `"split": "left_right"` puts
`first` on the left and `second` on the right; `"top_bottom"` puts `first` on
top. `egui_dock` calls those `Node::Horizontal` and `Node::Vertical`, words that
read either way — a horizontal *split line* is a top/bottom arrangement — and the
file does not repeat the ambiguity. `fraction` is `first`'s share, in `(0, 1)`
exclusive, which is `SplitNode::fraction`'s meaning exactly.

**A leaf is its tabs, in tab-bar order, and which one is in front.** `active`
names one of `tabs`; omitted on read, it is the first, and the viewer always
writes it. A leaf with no tabs is invalid — `egui_dock` removes such nodes on
sight, so a file could never round-trip one.

**`windows`** is the floating surfaces, each a `tree` (the same node shape) and a
`rect` in logical points (`x`, `y`, `width`, `height`), screen-anchored as
`egui_dock`'s `WindowState::rect` reports it. A window that has never been laid
out has no rect to report, and the field is omitted rather than invented; loading
a window without one lets `egui_dock` place it. A rect that is off-screen when
loaded — saved on a larger monitor, say — is left to egui's own window
constraint, which drags it into view. These are logical points and screen-
anchored because that is what `egui_dock` reports and takes; the *outer* window's
rectangle is the `window` section's, in physical pixels, and the two do not mix.

### Validation

The file is validated as a whole before any of it is applied, and a refusal names
what was wrong and **where**: every message below a node carries the path to it,
so a refusal reads `layout.main.second.first: unknown key "fracton"`. The rules,
each with its message:

- `sfm_explorer_layout`, when present, is a number and equals `2`: `Layout
  version 3 is newer than this viewer reads (2)`, or `Layout version 1 is not one
  this viewer reads (2)`. There is no upgrade path from version 1: it is the
  panel-only document, a different shape rather than a different key.
- The tag is **optional**, so that a document carrying only what it wants
  changed is a document. A JSON file that carries no tag, no `window` and no
  `layout` and yet has keys of its own is not a layout at all: `Not a layout
  file` — checked before anything else, so such a file says so rather than
  complaining about its own perfectly good keys.
- Top-level keys are `sfm_explorer_layout`, `window`, `layout`; anything else is
  `unknown key "…"`.
- `window` is an object or `null` (`window: must be an object or null`), and its
  keys are the five above. `state` is one of the four names (`window.state:
  unknown window state "big"; the states are normal, maximized, minimized,
  fullscreen`); `outer_position` two whole numbers (`window.outer_position: must
  be two whole numbers`); `inner_size` two whole numbers with neither zero
  (`window.inner_size: must be two whole numbers greater than zero`); `focus`
  must be `true` (`window.focus: can only ask for the foreground; omit it to
  leave focus alone`); `monitor` an object with exactly `position` and `size`,
  shaped as those two are (`window.monitor: must be an object with "position"
  (two whole numbers) and "size" (two whole numbers greater than zero)`), and a
  `monitor` with no rectangle to fit is `window.monitor: has nothing to fit —
  send outer_position or inner_size with it`.
- `layout` is an object, `null`, or the string `"default"`; any other string is
  `layout: the only named layout is "default"`, and anything else `layout: must
  be an arrangement, null, or "default"`.
- Inside `layout`: every panel name is one of the seven (`unknown panel
  "viewer3d"; the panels are scene, viewer_3d, image_browser, image_detail,
  point_track, camera_intrinsics, action_log`); **every panel appears at most
  once** across `main` and every window (`panel "scene" appears more than once`),
  because a `Tab` is a singleton — one struct draws it — and two tabs with one
  identity would draw one panel twice and confuse egui's widget ids, while a
  panel that appears nowhere is simply closed; every leaf has at least one tab
  (`a leaf must have at least one tab`) and `active`, if present, is one of them
  (`active "camera_intrinsics" is not one of this leaf's tabs`); every `fraction`
  is strictly between 0 and 1 (`fraction must be strictly between 0 and 1, not
  1.5`); a node carries `tabs` or `split` and is read as a leaf or a split
  accordingly, one that carries neither being `a node must have either "tabs" (a
  leaf) or "split" (a split)`; and `main` may be absent or `null`, meaning no
  panel is docked in the main surface — which, with an empty `windows`, is the
  all-closed state, and valid.
- **No unknown keys**, at any level: `unknown key "fracton"`. A typo silently
  applying a default would leave the author believing the file says something it
  does not — the same rule the MCP surface applies to its arguments.
- A document with neither section is valid — it describes no change — and loads
  as a no-op. The MCP `set_window_layout` refuses it
  ([mcp-server.md](mcp-server.md)), because a call is a request and this one asks
  for nothing.

Validation is structural, not geometric. A split whose `fraction` gives a panel a
sliver is legal; `egui_dock` enforces its own minimum sizes on draw. A normal
rectangle off every monitor is legal too, and left to the window manager, which
on Windows and macOS drags a window back on screen and on X11 may not.

## The default layout file

**`~/.sfm-explorer-default-layout.json`**, in the user's home directory
(`std::env::home_dir()`). The name is fixed; the directory is the one place the
viewer can rely on without knowing a workspace.

**At startup, if the file exists, the viewer loads it** — window and panels —
through the same path as Panels ▸ Load Layout…, and records `Loaded layout from
C:/Users/mark/.sfm-explorer-default-layout.json` as a `Layout` entry, actor
`User`: the human put the file there. A file that does not parse or validate is
refused whole, exactly as a menu load is, the viewer starts with the stock grid,
and the failed entry `Load layout from …: <reason>` goes on the viewport status
line — the human sees *why* their layout did not come back rather than wondering.
A file that is absent is nothing: no entry, no log line.

**The load happens after the window is created and before it is shown.** The
window is created with `with_visible(false)` — it already is, so that AccessKit
can register its UIAutomation provider first — the file is applied, and the
window is then made visible. So a saved "maximized on the left monitor" comes up
that way rather than appearing at 1280 × 720 in the middle and jumping.
`ui_basic`'s attach path waits for a window with the base title, and a hidden
window has one.

**`--no-default-layout`** on the viewer's command line skips the startup load —
for a test that must start from the stock grid whatever the developer has saved,
for CI, and for a human whose saved layout has become the problem. `ui_basic`
passes it wherever it launches the binary, except in the one test that is *about*
the startup load. `sfm explorer --no-default-layout` forwards it.

The viewer restores **the one file the human saved to the default location**,
never its last state, and never anything an agent did through MCP. An agent that
wants its arrangement to survive a restart asks the human to save it;
`get_window_layout` gives it the document to point at.

## Rust API

Two modules. `crates/sfm-explorer/src/layout.rs` holds the document, the panel
schema, the conversions and the operations; `crates/sfm-explorer/src/window.rs`
holds what a window *is* and how a placement reaches one. Both are
unconditional: Panels ▸ Save Layout… carries the window's placement in every
build, not only where the MCP feature is compiled in.

The dock state and the window snapshot live on `AppState` rather than on `App`,
so the operations that log are `AppState` methods and the MCP layer calls the
same ones the menu does.

### The document

```rust
/// One document's worth of panels: an arrangement, or the stock grid by name.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayoutSection { Layout(Layout), Default }

/// The layout document: the window's placement and the panel arrangement,
/// either of them optional. `None` is "leave it".
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowLayout {
    pub(crate) window: Option<WindowChange>,
    pub(crate) layout: Option<LayoutSection>,
}

/// The stock grid, and nothing about the window.
impl Default for WindowLayout { fn default() -> Self; }

impl WindowLayout {
    /// Parse and validate a layout file. Refused as a whole: a caller that
    /// gets an `Err` has a window and a layout it can leave as they were.
    pub(crate) fn from_json(text: &str) -> Result<Self, LayoutError>;
    /// The half below the JSON parse — which is how a document arrives over
    /// MCP, as the `set_window_layout` argument object. So the wire and the
    /// file cannot differ about what a document is.
    pub(crate) fn from_value(value: &serde_json::Value) -> Result<Self, LayoutError>;
    /// Pretty-printed, keys in the order § "The window layout file" shows
    /// them, trailing newline.
    pub(crate) fn to_json(&self) -> String;
    /// Neither section, or a window section that is empty.
    pub(crate) fn is_empty(&self) -> bool;
}

/// Where the default file lives, or `None` without a home directory.
pub(crate) fn default_layout_path() -> Option<PathBuf>;
pub(crate) const DEFAULT_LAYOUT_FILE_NAME: &str = ".sfm-explorer-default-layout.json";

/// What the window portion of a document did: the change as it reached the
/// window — the *fitted* rectangle where one was fitted — and the monitor it
/// was fitted from. The two things an Action Log row needs to report what
/// happened rather than what was asked.
pub(crate) struct AppliedWindow { pub change: WindowChange, pub fitted: Option<MonitorRect> }
```

### The panel arrangement

```rust
/// A dock arrangement, as the layout file's `layout` section spells it.
/// Serializable, buildable by hand, and independent of `egui_dock`'s node
/// indices.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Layout {
    pub main: Option<LayoutNode>,
    pub windows: Vec<LayoutWindow>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayoutNode {
    Split { split: SplitDirection, fraction: f32, first: Box<LayoutNode>, second: Box<LayoutNode> },
    Leaf  { tabs: Vec<Tab>, active: Tab },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitDirection { LeftRight, TopBottom }

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayoutWindow { pub tree: LayoutNode, pub rect: Option<LayoutRect> }

/// Logical points, screen-anchored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LayoutRect { pub x: f32, pub y: f32, pub width: f32, pub height: f32 }

pub(crate) const LAYOUT_VERSION: u64 = 2;

/// The stock seven-panel grid. `Layout::default().to_dock()` is what the
/// viewer starts with, and what Reset Layout restores.
impl Default for Layout { fn default() -> Self; }

impl Layout {
    /// Read the arrangement out of a live dock.
    pub(crate) fn from_dock(dock: &DockState<Tab>) -> Self;
    /// Check every rule in § "Validation" that survives parsing, naming the
    /// first violation. `path` is where the arrangement sits in the document —
    /// `"layout"` — so a violation reads `layout.main.second: …`.
    pub(crate) fn validate(&self, path: &str) -> Result<(), LayoutError>;
    /// Build a dock from a validated layout.
    pub(crate) fn to_dock(&self) -> DockState<Tab>;
    /// Validate the `layout` section of a document. The JSON is read into a
    /// `serde_json::Value` and walked by hand rather than
    /// `Deserialize`-derived: a node is a leaf or a split by which keys it
    /// carries, and serde does not honour `deny_unknown_fields` on an untagged
    /// enum, so a derive could not refuse a typo. Walking by hand is also what
    /// lets an error carry its path — `layout.main.second: unknown key
    /// "fracton"`.
    pub(crate) fn from_value(value: &serde_json::Value, path: &str) -> Result<Self, LayoutError>;
    /// The section as the file writes it, indented for a parent at `depth`.
    /// Called by `WindowLayout::to_json` for its `layout` key.
    pub(crate) fn write_json(&self, out: &mut String, depth: usize);
}

/// One violation, with the path to it. `Display` writes `<path>: <message>`,
/// or just the message for a violation of the document itself — the text the
/// Action Log records and the MCP tool returns. A window refusal from the host
/// arrives as `LayoutError::at("window", …)`, so every refusal a load or a tool
/// can produce is one type with one `Display`.
pub(crate) struct LayoutError { pub path: String, pub message: String }

/// Where a panel goes when nothing of its default group is open.
pub(crate) enum Home { Root, Edge { edge: egui_dock::Split, share: f32 } }

/// `Tab`'s wire spelling, both ways, and its home. Hand-written rather than
/// `serde`-derived: the file never passes through a `Deserialize`, and these
/// two functions are also what the MCP tools will spell panels with.
impl Tab {
    pub(crate) const ALL: [Tab; 7];           // default-layout order, the menu's order
    pub(crate) fn wire_name(self) -> &'static str;
    pub(crate) fn from_wire_name(name: &str) -> Option<Tab>;
    /// All seven, comma-joined, as the unknown-panel message lists them.
    pub(crate) fn all_wire_names() -> String;
    pub(crate) fn home(self) -> Home;
}

impl AppState {
    pub(crate) dock: DockState<Tab>,
    /// The window as last observed, `None` before there is one.
    pub(crate) window: Option<WindowInfo>,
    /// The rectangle it had when it last read as `normal`.
    pub(crate) window_normal_rect: Option<NormalRect>,

    pub(crate) fn is_panel_open(&self, tab: Tab) -> bool;
    /// Rules 1–3 of § "Home positions". Records `Opened …` or `Raised …`.
    pub(crate) fn show_panel(&mut self, tab: Tab);
    /// Records `Closed …`; a no-op on a panel that is not open.
    pub(crate) fn hide_panel(&mut self, tab: Tab);
    /// Records `Reset layout`. The panels only.
    pub(crate) fn reset_layout(&mut self);
    /// Validates, then replaces the whole dock. Records nothing itself: the
    /// caller words the entry (`Loaded layout from …`, or the MCP tool name).
    pub(crate) fn apply_layout(&mut self, layout: &Layout) -> Result<(), LayoutError>;
    pub(crate) fn layout(&self) -> Layout;

    /// Refresh the snapshot from the host; when it reads `normal`, remember
    /// the rectangle. Called at the top of every frame.
    pub(crate) fn observe_window(&mut self, host: &dyn WindowHost);
    /// The document Save Layout… writes: the placement from the snapshot and
    /// the remembered rectangle (no `window` section without a snapshot), and
    /// the panel tree.
    pub(crate) fn window_layout(&self) -> WindowLayout;
    /// Window portion first — fitted through `window::fit_to_monitor` against
    /// the host's monitors and the snapshot's current one — then panels.
    /// Records nothing: the menu, the startup load and the MCP tool each word
    /// their own entry. `LayoutSection::Default` goes through
    /// `apply_layout(&Layout::default())` and so records nothing either;
    /// `reset_layout` is the recording wrapper the menu calls.
    pub(crate) fn apply_window_layout(&mut self, host: &mut dyn WindowHost,
                                      document: &WindowLayout)
        -> Result<Option<AppliedWindow>, LayoutError>;
    /// Load and apply one file, recording `Loaded layout from …` or the failed
    /// entry. Shared by Load Layout… and the startup load.
    pub(crate) fn load_layout_file(&mut self, host: &mut dyn WindowHost, path: &Path);
}

/// The body of the Panels menu, drawn into an open `ui.menu_button("Panels", …)`.
/// A function rather than inline in `app.rs` so a headless frame can draw it.
/// It takes the host because Save and Load carry the window: the frame passes a
/// clone of its `Arc<Window>`, the headless test passes a fake.
pub(crate) fn panels_menu(ui: &mut egui::Ui, state: &mut AppState, host: &mut dyn WindowHost);
```

### The window

`crates/sfm-explorer/src/window.rs`, unconditional — `winit` is not gated.

```rust
pub(crate) enum WindowState { Normal, Maximized, Minimized, Fullscreen }

/// One monitor, as the wire reports it.
pub(crate) struct MonitorInfo {
    name: Option<String>, position: [i32; 2], size: [u32; 2], scale_factor: f64,
}

/// The window as last observed: the block a reply renders, minus `monitors`.
pub(crate) struct WindowInfo {
    state: WindowState, focused: bool, scale_factor: f64,
    outer_position: Option<[i32; 2]>, outer_size: [u32; 2], inner_size: [u32; 2],
    monitor: Option<MonitorInfo>,
}

/// A monitor's rectangle on the desktop, physical px — the two fields of
/// `MonitorInfo` a fit needs.
pub(crate) struct MonitorRect { pub position: [i32; 2], pub size: [u32; 2] }

/// The window's rectangle when normal. Remembered across maximize, fullscreen
/// and minimize, since `winit` reports only the current rectangle.
pub(crate) struct NormalRect {
    pub outer_position: Option<[i32; 2]>,
    pub inner_size: [u32; 2],
}

/// The `window` section of a document, and the pieces one window portion
/// carries. `None` is "leave it".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct WindowChange {
    pub state: Option<WindowState>,
    pub outer_position: Option<[i32; 2]>,
    pub inner_size: Option<[u32; 2]>,
    /// Read by `fit_to_monitor` and never by a host: `apply_window_layout`
    /// resolves it into a plain rectangle before the change is applied.
    pub monitor: Option<MonitorRect>,
    pub focus: bool,
}

impl WindowChange {
    pub(crate) fn is_empty(&self) -> bool;
    pub(crate) fn has_geometry(&self) -> bool;
    /// The `window` section, with the document's rules and path-carrying
    /// messages.
    pub(crate) fn from_json(value: &Value, path: &str) -> Result<Self, LayoutError>;
    /// The section as the file writes it; never `focus`.
    pub(crate) fn write_json(&self, out: &mut String, depth: usize);
    /// The Action Log phrase, pieces in application order joined with `; `,
    /// naming the saved monitor when `fitted` says the rectangle was fitted.
    pub(crate) fn log_text(&self, fitted: Option<MonitorRect>) -> String;
}

/// § "Fitting a rectangle to the desktop": the change with its rectangle
/// fitted onto `target` and `monitor` cleared, and the saved monitor when a fit
/// happened; the change unchanged (but for `monitor`) otherwise.
pub(crate) fn fit_to_monitor(change: &WindowChange, monitors: &[MonitorInfo],
                             target: Option<&MonitorInfo>) -> (WindowChange, Option<MonitorRect>);

/// A refusal from the window itself: something the platform cannot do.
pub(crate) struct WindowError(pub String);

/// What a window read or change says when there is no window behind it.
pub(crate) const NO_WINDOW: &str = "This viewer has no window to read or change — …";

/// What applying a window change needs from the window, and all it needs.
///
/// Five primitives, and one provided method that holds the application order —
/// written once, so the fake in the tests exercises the real rule rather than a
/// copy of it.
pub(crate) trait WindowHost {
    fn set_state(&mut self, state: WindowState);   // Normal = all three flags off, minimized first
    fn set_outer_position(&mut self, position: [i32; 2]) -> Result<(), WindowError>;
    fn set_inner_size(&mut self, size: [u32; 2]);
    fn focus(&mut self);
    /// The window as it is now, and every monitor, the current one first.
    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)>;

    /// § "How a `window` section is applied": geometry on a normal window,
    /// then the named state or the previous one, then focus.
    fn apply(&mut self, change: &WindowChange) -> Result<(), WindowError> { /* provided */ }
}

impl WindowHost for Arc<winit::window::Window> { /* the real window */ }
```

**Why a host trait rather than reaching for the window.** A window change has to
be appliable from `AppState` — the menu loads a document, the startup load
applies one, an MCP tool sends one — and none of those has a `winit::Window` in
reach that a headless test could stand up. `Arc<Window>::set_outer_position`
answers `Err` where `outer_position()` does, which is the one platform refusal
there is. `apply`'s provided body reads the state through `observe` before
touching anything, so "the state the window was in before" is the observed one
and not a guess; a host that observes nothing has nothing to apply to and
returns `NO_WINDOW`.

The MCP layer keeps only what is the wire's: the `window` block renderer
([mcp-server.md](mcp-server.md) § "The window block").

**Why a schema of its own rather than `egui_dock`'s `serde` feature.** The
crate can serialize `DockState` directly, and the result is its internal
representation: a flat `Vec<Node>` in heap order with `Empty` placeholders,
every node's last-frame `rect` and `viewport`, scroll offsets, collapse flags.
Nobody can write that by hand, an agent could not be asked to, and it changes
whenever `egui_dock` does — the 0.19 → 0.21 move that already bit the fraction
semantics would have invalidated every saved file. The tree above is five
concepts, and `from_dock` / `to_dock` are the one place the crate's
representation is known.

**Why `Layout` and not a list of "open" panels.** Ticking panels open and
closed is the menu's job and needs only `show_panel` / `hide_panel`. The file
exists so an arrangement — *this* panel at *this* width, beside *that* one —
survives. That is a tree.

**Why `apply_layout` records nothing.** Two callers word the same operation
differently — the menu says where the file came from, an MCP tool says which
tool — and a method that logged would have to be told. `show_panel` and
`hide_panel` log themselves because they have one wording each.

**Why the dock lives on `AppState`.** So that a layout operation is an
`AppState` method with the Action Log in reach, and so that the MCP `apply`
seam — `(&mut AppState, &mut Viewer3D)`, deliberately without `App` — drives the
layout headlessly. `DockState<Tab>` is plain data with no GPU or window behind
it, and the window reaches the same methods through a trait object, which is why
`layout::tests` can build and read whole arrangements, and apply whole
documents, with no window at all.

### Example

```rust
use crate::layout::{default_layout_path, WindowLayout};

// Save Layout...
std::fs::write(&path, state.window_layout().to_json())?;

// Load Layout..., and the startup load
state.load_layout_file(&mut host, &path);

// At startup, in `resumed`, between creating the window and showing it
if !no_default_layout {
    if let Some(path) = default_layout_path().filter(|path| path.is_file()) {
        state.load_layout_file(&mut window.clone(), &path);
    }
}
window.set_visible(true);
```

## Implementation notes

**The borrow during the dock pass.** `TabContext` holds `&mut AppState` while
`DockArea::new(&mut dock).show_inside(…)` needs the dock mutably at the same
time, which `AppState::dock` would make a conflict. The frame takes the dock
out of `AppState` for the duration of the `DockArea` call with
`std::mem::replace` against an empty `DockState::new(vec![])` and puts it back
straight after — a pointer swap each way. Nothing inside a tab body reads
`state.dock`, and nothing may: a tab that needs a layout operation reports it
in its response struct, the way `SceneGraphResponse` already carries
everything the Scene panel asks of the app, and the frame applies it after the
dock is back.

**Building a tree by splitting.** `egui_dock` has no constructor that takes a
subtree; a `Tree` is grown by `split(parent, Split, fraction, Node)`, which
moves the existing node to the first child and installs the new one as the
second, and **asserts the new node has at least one tab**. So `to_dock` places
a node recursively: a `Leaf` is written straight into its index; a `Split`
first splits its index with a one-tab placeholder leaf as the second child,
then recurses into both children, each of which overwrites its own index.
`Split::Right` for `left_right`, `Split::Below` for `top_bottom`, so that
`fraction` keeps its "first child's share" meaning in both — the `Left` /
`Above` variants swap the children and would invert it.

**Reading a tree.** `Tree` indexes as a binary heap (`NodeIndex::left`,
`right`, `parent`); `from_dock` walks from `NodeIndex::root`, mapping
`Node::Horizontal` to `left_right` and `Node::Vertical` to `top_bottom`, and
must tolerate `Node::Empty` under a leaf (the heap is padded to a full level).
The all-closed dock is the other edge: removing the root leaf's last tab
*clears* the node vector, so `from_dock` reads no root at all and writes
`main: null`. `to_dock` spells the same state the same way, with
`remove_leaf(NodeIndex::root())` on a one-placeholder tree, rather than leaving
behind a tabless leaf `egui_dock` would then have to remove itself.

**A split with one readable child collapses to that child.** It cannot arise
from a dock `egui_dock` is maintaining; tolerating it costs one match arm and
means a layout describes the panel that is there rather than a split around
nothing.

**Floating windows.** `DockState::add_window(tabs)` creates a surface with a
one-leaf tree; a window whose tree is a split is built by splitting that
surface's tree the same way as the main one. `WindowState::set_position` /
`set_size` queue the rect for the next frame; `rect()` returns `Rect::NOTHING`
until the window has been laid out, which is the case `LayoutWindow::rect` is
`None` for.

**Writing the file.** `to_json` is a hand-written pretty-printer rather than
`serde_json::to_string_pretty`, for two reasons: JSON objects have no key order
a `Map` would preserve without `serde_json`'s `preserve_order` feature, and a
leaf's `tabs` reads far better on one line than as seven. Each piece writes
itself at a depth its parent gives it, so the `layout` section indents inside
the document exactly as it would at any other nesting, and the default's output
is asserted verbatim against the document in § "The window layout file".

**`serde_json` is not optional.** The layout file needs it in every build, so it
is a plain dependency rather than one of the `mcp` feature's. `serde` itself is
not needed: the file is read and written through `serde_json::Value`, and
`Tab`'s wire spelling is `wire_name` / `from_wire_name`, which the MCP tools use
as well.

**Observing the window every frame is cheap and load-bearing.** Save Layout…
reads the snapshot, and the normal rectangle has to be *remembered* — it cannot
be read once the window is maximized — so the memory has to be current at the
moment of the maximize, which means every frame rather than only the frames an
MCP endpoint is live for. The cost is a handful of platform calls per rendered
frame; an idle viewer renders none.

**A rectangle applied to a window that is not normal is remembered as asked.**
`observe_window` can only remember what it can read, and a window that comes out
of an apply maximized does not report the rectangle underneath — but the viewer
just *set* that rectangle, so `apply_window_layout` records it. Without that,
the document would go on describing the rectangle from before the call as the
one the window restores to. A normal window needs none of this: the observation
after the apply read the truth, clamps included.

**The remembered rectangle and the frame after a maximize.** On Windows the
state and geometry calls are synchronous, so a frame never reads `normal` with a
maximized rectangle. On a platform that animates, a frame could in principle read
the flag before the rectangle; if that shows up, the fix is to require two
consecutive normal frames before trusting a rectangle, not to give up
remembering.

**Order inside the provided `apply`.** Observe; if geometry: `set_state(Normal)`,
then the position (an `Err` returns here, with nothing else touched), then the
size; then `set_state` for the named state, or for the previous one when
geometry made a non-normal window normal; then `focus`. The fake records each
primitive call in order, so the tests assert the sequence rather than a fake's
re-implementation of it.

**Closing is `egui_dock`'s, and `on_close` records the entry.** `TabViewer`
has two hooks here and only one of them is live in 0.21.1: `closeable` is
deprecated and never called by `DockArea`, while `is_closeable` decides, and
defaults to `true`. So the viewer overrides neither — the close button and the
removal are the crate's — and implements `on_close`, which logs `Closed …` and
returns `OnCloseResponse::Close`. Anyone reaching for `closeable` to make one
tab un-closeable (§ "Open questions") will find it has no effect; the hook that
works is `is_closeable`.

**`Go ▸ Go to Point` raises the Point Track panel through `show_panel`**, so
a jump to a point with that panel closed opens it rather than silently finding
nothing to raise.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `layout::LAYOUT_VERSION` | `2` | The `sfm_explorer_layout` value written and the only one read. |
| Scene home | left, `0.18` | § "Home positions" rule 3 (`Tab::home`). |
| Image Browser / Action Log home | below, `0.20` | Same. |
| Image Detail / Point Track / Camera Intrinsics home | right, `0.33` | Same. |
| `layout::DEFAULT_LAYOUT_FILE_NAME` | `.sfm-explorer-default-layout.json` | The file the viewer reads at startup, and the name the save dialog offers. |
| `--no-default-layout` | off | Skip the startup load (`cli::Args::no_default_layout`). |

## Testing

`crates/sfm-explorer/src/layout/tests.rs`, headless — a `DockState<Tab>` needs
no window, and the window goes through `test_support::FakeWindow`. The
default-layout test lives here too, beside the layout it checks.

- **The default round-trips:** `Layout::default().to_dock()` read back with
  `from_dock` equals `Layout::default()`; and `WindowLayout::default()` through
  `to_json` / `from_json` likewise. The JSON matches the document in § "The
  window layout file" verbatim, so the spec and the code cannot drift.
- **A `window` section round-trips** with `outer_position` present and absent,
  sits between the version tag and `layout`, and never carries `focus`;
  `"layout": "default"` round-trips too.
- **Every panel appears exactly once in the default**, and `Tab::ALL` is in
  the menu's order.
- **Wire names round-trip** for all seven, and `from_wire_name` refuses
  `"Scene"`, `"viewer3d"`, `""`.
- **Home positions:** for each panel, close it from the default layout and
  `show_panel` it back — it lands in its group-mate's node when one exists
  (rule 2), and in a root split along its home edge at its home fraction when
  none does (rule 3); `show_panel` on an open panel changes nothing but the
  active tab (rule 1); any panel into an empty dock becomes the root.
- **Hide then show keeps the others where they were.**
- **Each validation rule refuses, with its message:** a document that claims
  nothing, versions `1` and `3`, an unknown key at top level and inside the
  section, `window` not an object, each `window` key's own rule (bad state name,
  zero size, a wrong-length pair, `focus: false`, a malformed `monitor`, a
  `monitor` with nothing to fit), `layout` as a string other than `"default"`,
  and every panel-tree rule under its `layout.` prefix — unknown panel name
  (listing the seven), a panel in two leaves, a panel in `main` and in a window,
  an empty leaf, `active` not in `tabs`, `fraction` of `0`, `1` and `1.5`, an
  unknown key inside a leaf and inside a split, and a node that is neither.
- **A document with neither section is valid and `is_empty`**, as is one whose
  `window` object is empty.
- **A refused load leaves the dock untouched.**
- **All-closed is valid**, loads to an empty main surface, and `layout()` of
  an empty dock writes `main` absent and no `windows`.
- **A floating window round-trips**, with and without a rect.
- **The Action Log** gets `Closed …`, `Opened …`, `Raised …`, `Reset layout`,
  and a failed entry for a refused load, each under `Kind::Layout`, and none
  of them coalesce.
- **`observe_window` remembers the rectangle only from a `normal` reading:**
  observe normal at R1, maximize the fake and observe again, and
  `window_layout()` writes `maximized` with R1 and the monitor it was measured
  on. A windowless `AppState` writes no `window` section at all.
- **The application-order table, row by row against the fake:** the primitive
  sequence and the flags and rectangle each row leaves behind, including that a
  size against a maximized fake leaves it maximized with a new normal rectangle.
  A fake that refuses the position stops the call with the size unapplied, and a
  host with no window refuses with `NO_WINDOW`.
- **`fit_to_monitor`:** the same desktop (a monitor matching the saved one) uses
  the rectangle as saved, even one straddling two monitors; a saved monitor gone
  but the rectangle inside another monitor is used as saved; a rectangle off
  every monitor is mapped onto the target with the formula — the left half of
  3840 × 2160 becomes the left half of 1920 × 1080, and a target at `(1920, 0)`
  shifts it there; a size alone scales and never reaches zero; no `monitor` in
  the section, or no target monitor, means no fit.
- **`apply_window_layout` applies the window then the panels:** the fake's call
  log shows the window primitives and the dock ends up reset; a fake refusing
  the position leaves the dock as it was; a validation failure touches neither.
  It hands the host the *fitted* rectangle and never a `monitor`, and the phrase
  it hands back says `fitted from a 3840×2160 monitor`.
- **`load_layout_file`** records `Loaded layout from …` for a good file, applies
  its window and its panels, and records the failed entry — carrying the
  parser's path — for a bad one, applying nothing.
- **`default_layout_path()`** ends in the file name and is under the home
  directory when there is one.

The menu itself is exercised through `test_support::painted_texts`, which is
why `panels_menu` is a function taking a `Ui`: a headless frame draws the body
and the test asserts it painted all seven panel titles and the three
layout-wide items, and that drawing it changed nothing. What the *click* does
is `show_panel` / `hide_panel`, tested directly against `is_panel_open` and the
resulting `Layout` — synthesizing a pointer press at a widget rect would test
egui's checkbox rather than anything this spec decides.

`crates/sfm-explorer/tests/ui_basic.rs` (Windows/macOS) covers the one thing no
headless test can: a small default-layout file written to the home directory,
the viewer launched *without* `--no-default-layout`, and the panel that file
named found in the accessibility tree — the startup load end to end, through a
real window. It puts the developer's own file back afterwards. Every other test
there passes `--no-default-layout`, so a saved layout on the machine running the
suite cannot make its panel assertions fail.

## Non-goals

- **Remembering the layout the user did not save.** The viewer restores the one
  file the human saved to the default location, never its last state, and never
  anything an agent set through MCP. A layout that silently came back different
  from the one that was saved is a support question.
- **Serializing collapse, scroll, or tab-bar visibility.** `egui_dock` has
  them; the file does not. They are transient, and a layout is an arrangement.
- **Naming, duplicating, or parameterizing panels.** Seven singletons, one
  struct each. A second Image Detail panel is a different design.
- **Logging drag rearrangements** (§ "In the Action Log").
- **The MCP tools.** `get_window_layout`, `set_window_layout`, `show_panel` and
  `hide_panel` are specified in [mcp-server.md](mcp-server.md) §
  "`get_window_layout`" and the sections after it; this spec is what they carry.
  They call the `AppState` methods above and send the document this spec
  defines, so the wire adds no schema of its own — which is why "the panel
  names" and "the validation rules" have one home and not two.

## Open questions

- **Should closing the 3D Viewer be confirmed, or its tab be un-closeable?**
  It is the panel the user is least likely to close on purpose and the one
  whose absence looks most like a broken viewer. The menu makes it a one-click
  recovery; if that proves not enough, `is_closeable` can say no for that one
  tab while the others keep the button.
- **A workspace-relative layout file.** `.sfmtool/layout.json` beside the
  reconstructions would suit the CLI's workspace model, and a viewer opened on a
  workspace could prefer it to the home-directory file. The viewer does not know
  its workspace today.
- **A `--layout PATH` flag** to start from a named file, for an agent that
  launches the viewer and wants a known arrangement without touching the human's
  default. Cheap once `load_layout_file` exists; left out until asked for.
- **Fitting preserves the share, not the aspect.** A 16:9 window from a 16:9
  monitor comes out 16:10 on a 16:10 monitor. Uniform scaling by the smaller
  ratio, centred, is the alternative; switch if the per-axis result looks wrong
  in practice.
- **A rectangle that is off-screen without a `monitor` to fit from** — an
  agent's bare rectangle, or a hand-edited file — is left to the window manager.
  A clamp to the nearest monitor is the obvious fallback if it bites.
