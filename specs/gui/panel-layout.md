# Panel Layout

## Purpose

SfM Explorer's window is a set of docked panels — the scene tree, the 3D
viewport, the image strip, the detail views, the action log — that the user can
drag into any arrangement. This spec covers the lifecycle of that arrangement:
closing a panel and getting it back, resetting to the stock layout, and saving
an arrangement to a JSON file so it can be loaded again, in this session or
another. The file's schema is the viewer's own, written to be read and written
by hand, and it is the one description of a layout the viewer has: the same
schema an agent driving the viewer over MCP reads and writes
([mcp-server.md](mcp-server.md) holds the tools; this spec holds the
shape).

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

**Reset Layout** replaces the whole arrangement — main surface and any floating
windows — with `Layout::default()`, the seven-panel grid
[multi-panel-image-browser.md](multi-panel-image-browser.md) § "Default
Layout" describes. Panel state survives, as it does for a close.

**Save Layout...** opens a save dialog (filter `*.json`, default name
`layout.json`) and writes the current arrangement as § "The layout file".
**Load Layout...** opens an open dialog with the same filter, parses the file, and
replaces the arrangement with it. A file that does not parse or does not
validate is refused as a whole — the layout on screen is untouched — and the
reason is recorded in the Action Log as a failed `Layout` entry, which puts it
on the viewport status line.

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
| Save Layout... | `Saved layout to C:/work/layout.json` |
| Load Layout... | `Loaded layout from C:/work/layout.json` |
| a load refused | *failed:* `Load layout from …: <reason>` |

Drag rearrangements the user makes with the mouse — moving a tab, resizing a
split, floating a node — are **not** logged. `egui_dock` reports none of them as
events, and diffing the tree every frame to synthesize entries would log a
split drag as hundreds of resizes. The Action Log records what was *asked for*,
by menu or, later, by tool call; the arrangement itself is what
`Layout::from_dock` reads back at any moment.

## The layout file

One JSON document, version-tagged, describing the main surface's split tree and
any floating windows. The default layout, as the viewer writes it:

```json
{
  "sfm_explorer_layout": 1,
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
```

That is the file byte for byte: two-space indentation, one key to a line, a
leaf's `tabs` kept on one line because a list of panel names reads as a list,
and a trailing newline. `layout::tests` compares `Layout::default().to_json()`
against exactly this document, so the two cannot drift.

### Vocabulary

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

The word is **panel**, on the wire as in the specs, which use it several
hundred times against a handful of "pane". `Tab` stays the Rust name: it is
`egui_dock`'s word for the thing in a node, and the code is not the wire.

**A split names its arrangement, not its divider.** `"split": "left_right"`
puts `first` on the left and `second` on the right; `"top_bottom"` puts `first`
on top. `egui_dock` calls those `Node::Horizontal` and `Node::Vertical`, words
that read either way — a horizontal *split line* is a top/bottom arrangement —
and the file does not repeat the ambiguity. `fraction` is `first`'s share, in
`(0, 1)` exclusive, which is `SplitNode::fraction`'s meaning exactly.

**A leaf is its tabs, in tab-bar order, and which one is in front.** `active`
names one of `tabs`; omitted on read, it is the first, and the viewer always
writes it. A leaf with no tabs is invalid —
`egui_dock` removes such nodes on sight, so a file could never round-trip one.

**`windows`** is the floating surfaces, each a `tree` (the same node shape) and
a `rect` in logical points (`x`, `y`, `width`, `height`), screen-anchored as
`egui_dock`'s `WindowState::rect` reports it. A window that has never been laid
out has no rect to report, and the field is omitted rather than invented;
loading a window without one lets `egui_dock` place it. A rect that is
off-screen when loaded — saved on a larger monitor, say — is left to egui's own
window constraint, which drags it into view.

### Validation

The file is validated as a whole before any of it is applied, and a refusal
names what was wrong and **where**: every message below a node carries the path
to it, so a refusal reads `main.second.first: unknown key "fracton"`. The rules,
each with its message:

- `sfm_explorer_layout` is present and a number. Otherwise `Not a layout file`
  — checked before anything else, so a JSON file that is not a layout at all
  says so rather than complaining about its own perfectly good keys.
- It equals `1`: `Layout version 2 is newer than this viewer reads (1)`, or
  `Layout version 0 is not one this viewer reads (1)`.
- Every panel name is one of the seven: `unknown panel "viewer3d"; the panels
  are scene, viewer_3d, image_browser, image_detail, point_track,
  camera_intrinsics, action_log`.
- **Every panel appears at most once** across `main` and every window:
  `panel "scene" appears more than once`. A `Tab` is a singleton — one struct
  draws it — and two tabs with one identity would draw one panel twice and
  confuse egui's widget ids. A panel that appears nowhere is simply closed.
- Every leaf has at least one tab (`a leaf must have at least one tab`);
  `active`, if present, is one of them (`active "camera_intrinsics" is not one
  of this leaf's tabs`).
- Every `fraction` is strictly between 0 and 1: `fraction must be strictly
  between 0 and 1, not 1.5`.
- **No unknown keys**, at any level: `unknown key "fracton"`. A typo silently
  applying a default would leave the author believing the file says something
  it does not — the same rule the MCP surface applies to its arguments.
- A node carries `tabs` or `split` and is read as a leaf or a split
  accordingly; one that carries neither is `a node must have either "tabs" (a
  leaf) or "split" (a split)`.
- `main` may be absent or `null`, meaning no panel is docked in the main
  surface. Together with an empty `windows` that is the all-closed state, which
  is valid.

Validation is structural, not geometric. A split whose `fraction` gives a
panel a sliver is legal; `egui_dock` enforces its own minimum sizes on draw.

## Rust API

`crates/sfm-explorer/src/layout.rs`, with the schema types, the conversions,
and the panel operations. The dock state lives on `AppState` rather than on
`App`, so the operations that log are `AppState` methods and the MCP layer can
later call the same ones the menu does.

```rust
/// A dock arrangement, as the layout file spells it. Serializable, buildable
/// by hand, and independent of `egui_dock`'s node indices.
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

pub(crate) const LAYOUT_VERSION: u64 = 1;

/// The stock seven-panel grid. `Layout::default().to_dock()` is what the
/// viewer starts with, and what Reset Layout restores.
impl Default for Layout { fn default() -> Self; }

impl Layout {
    /// Read the arrangement out of a live dock.
    pub(crate) fn from_dock(dock: &DockState<Tab>) -> Self;
    /// Check every rule in § "Validation", naming the first violation.
    pub(crate) fn validate(&self) -> Result<(), LayoutError>;
    /// Build a dock from a validated layout.
    pub(crate) fn to_dock(&self) -> DockState<Tab>;
    /// Parse and validate. The JSON is read into a `serde_json::Value` and
    /// walked by hand rather than `Deserialize`-derived: a node is a leaf or a
    /// split by which keys it carries, and serde does not honour
    /// `deny_unknown_fields` on an untagged enum, so a derive could not refuse
    /// a typo. Walking by hand is also what lets an error carry its path —
    /// `main.second.first: unknown key "fracton"`.
    pub(crate) fn from_json(text: &str) -> Result<Self, LayoutError>;
    /// The same, on an already-parsed document — which is how a layout arrives
    /// over MCP, as a `set_layout` argument object. `from_json` is the JSON
    /// parse and then this, so a document off the wire meets exactly the rules,
    /// and exactly the messages, a file on disk meets.
    pub(crate) fn from_value(value: &serde_json::Value) -> Result<Self, LayoutError>;
    /// Pretty-printed, keys in the order § "The layout file" shows them,
    /// trailing newline.
    pub(crate) fn to_json(&self) -> String;
}

/// One violation, with the path to it. `Display` writes `<path>: <message>`,
/// or just the message for a violation of the document itself — the text the
/// Action Log records and, later, the MCP tool returns.
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
    pub(crate) fn is_panel_open(&self, tab: Tab) -> bool;
    /// Rules 1–3 of § "Home positions". Records `Opened …` or `Raised …`.
    pub(crate) fn show_panel(&mut self, tab: Tab);
    /// Records `Closed …`; a no-op on a panel that is not open.
    pub(crate) fn hide_panel(&mut self, tab: Tab);
    /// Records `Reset layout`.
    pub(crate) fn reset_layout(&mut self);
    /// Validates, then replaces the whole dock. Records nothing itself: the
    /// caller words the entry (`Loaded layout from …`, or the MCP tool name).
    pub(crate) fn apply_layout(&mut self, layout: &Layout) -> Result<(), LayoutError>;
    pub(crate) fn layout(&self) -> Layout;
}

/// The body of the Panels menu, drawn into an open `ui.menu_button("Panels", …)`.
/// A function rather than inline in `app.rs` so a headless frame can draw it.
pub(crate) fn panels_menu(ui: &mut egui::Ui, state: &mut AppState);
```

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
seam — `(&mut AppState, &mut Viewer3D)`, deliberately without `App` — can drive
the layout headlessly when the tools land. `DockState<Tab>` is plain data with
no GPU or window behind it, which is why `layout::tests` can build and read
whole arrangements with no window at all.

### Example

```rust
use crate::layout::Layout;

// Save Layout...
let text = state.layout().to_json();
std::fs::write(&path, text)?;

// Load Layout...
let layout = Layout::from_json(&std::fs::read_to_string(&path)?)?;
state.apply_layout(&layout)?;
state.action_log.record(Kind::Layout, format!("Loaded layout from {}", path.display()));
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
leaf's `tabs` reads far better on one line than as seven. It is thirty lines,
and the default's output is asserted verbatim against the document in § "The
layout file".

**`serde_json` is not optional.** It was behind the `mcp` feature; the layout
file needs it in every build, so it is a plain dependency and the feature list
is one shorter. `serde` itself is not needed: the file is read and written
through `serde_json::Value`, and `Tab`'s wire spelling is `wire_name` /
`from_wire_name`, which the MCP tools will use as well.

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
| `layout::LAYOUT_VERSION` | `1` | The `sfm_explorer_layout` value written and the only one read. |
| Scene home | left, `0.18` | § "Home positions" rule 3 (`Tab::home`). |
| Image Browser / Action Log home | below, `0.20` | Same. |
| Image Detail / Point Track / Camera Intrinsics home | right, `0.33` | Same. |
| Save dialog default name | `layout.json` | `layout::DEFAULT_LAYOUT_FILE_NAME`, Panels ▸ Save Layout... |

## Testing

`crates/sfm-explorer/src/layout/tests.rs`, headless — a `DockState<Tab>` needs
no window. The default-layout test lives here too, beside the layout it checks.

- **The default round-trips:** `Layout::default().to_dock()` read back with
  `from_dock` equals `Layout::default()`; and through `to_json` / `from_json`
  likewise. The JSON matches the document in § "The layout file" verbatim,
  so the spec and the code cannot drift.
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
- **Each validation rule refuses, with its message:** version missing, version
  `2` and version `0`, unknown panel name (listing the seven), a panel in two
  leaves, a panel in `main` and in a window, an empty leaf, `active` not in
  `tabs`, `fraction` of `0`, `1` and `1.5`, an unknown key at top level, inside
  a leaf and inside a split, and a node that is neither.
- **A refused load leaves the dock untouched.**
- **All-closed is valid**, loads to an empty main surface, and `layout()` of
  an empty dock writes `main` absent and no `windows`.
- **A floating window round-trips**, with and without a rect.
- **The Action Log** gets `Closed …`, `Opened …`, `Raised …`, `Reset layout`,
  and a failed entry for a refused load, each under `Kind::Layout`, and none
  of them coalesce.

The menu itself is exercised through `test_support::painted_texts`, which is
why `panels_menu` is a function taking a `Ui`: a headless frame draws the body
and the test asserts it painted all seven panel titles and the three
layout-wide items, and that drawing it changed nothing. What the *click* does
is `show_panel` / `hide_panel`, tested directly against `is_panel_open` and the
resulting `Layout` — synthesizing a pointer press at a widget rect would test
egui's checkbox rather than anything this spec decides.

## Non-goals

- **Remembering the layout between runs.** The viewer persists nothing today
  (the MCP spec's "No persistence" holds here too), and a layout that silently
  came back different from the stock one is a support question. Save and Load
  are explicit. An opt-in "restore last layout" is the natural next step and
  belongs with a decision about where the viewer keeps configuration at all.
- **Serializing collapse, scroll, or tab-bar visibility.** `egui_dock` has
  them; the file does not. They are transient, and a layout is an arrangement.
- **Naming, duplicating, or parameterizing panels.** Seven singletons, one
  struct each. A second Image Detail panel is a different design.
- **Logging drag rearrangements** (§ "In the Action Log").
- **The MCP tools.** `get_layout`, `set_layout`, `show_panel` and `hide_panel`
  are specified in [mcp-server.md](mcp-server.md) § "`get_layout`" and the
  sections after it; this spec is what they carry. They call the `AppState`
  methods above and send the document this spec defines, so the wire adds no
  schema of its own — which is why "the panel names" and "the validation rules"
  have one home and not two.

## Open questions

- **Should closing the 3D Viewer be confirmed, or its tab be un-closeable?**
  It is the panel the user is least likely to close on purpose and the one
  whose absence looks most like a broken viewer. The menu makes it a one-click
  recovery; if that proves not enough, `is_closeable` can say no for that one
  tab while the others keep the button.
- **Where the layout file lives by default.** The save dialog opens wherever
  the platform opens it. A workspace-relative default (`.sfmtool/layout.json`
  beside the reconstructions) would suit the CLI's workspace model, but the
  viewer does not know its workspace today.
