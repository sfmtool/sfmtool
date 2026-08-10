# Go to Point

*Status: Implemented*

This document specifies **Go to Point**: a dialog that takes a typed or pasted
point index — or a whole `pt3d_<hash>_<index>` Point ID — and selects that
point, switching the selected reconstruction when the ID names a different one.

For the panel this most directly serves, see
[gui-point-track-detail.md](gui-point-track-detail.md). For the ID format
itself, see the [Point ID section in the `.sfmr` format
spec](../formats/sfmr-file-format.md#point-id-portable-3d-point-references).

## Motivation

Every existing way to reach a 3D point in the viewer is a **click**: on a splat
in the 3D viewport, on a feature in the Image Detail overlay, on the selection
row in the Scene panel. Each requires already having the point on screen and
being able to tell it apart from its neighbours.

That leaves no way *back in from the outside*, which is precisely what the Point
ID format exists for. The format spec describes the intended workflow — copy an
ID out of the panel header, paste it into a constraints file, a notes document,
a CLI invocation — but the round trip only closed in one direction: the viewer
could emit an ID and never accept one. A user holding
`pt3d_a1b2c3d4_4821` from a ground-truth table had to find that point by eye.

Go to Point closes the loop. It also covers the more mundane case: a point index
printed by an `sfm analyze` run, a log line, or a colleague's message, typed
straight in.

## Design

### Entry Points

Three, all opening the same dialog:

| Entry point | Where | Why there |
|-------------|-------|-----------|
| `Go ▸ Go to Point…` | Menu bar | Discoverable; the conventional home for "jump to a thing by name" (editors put it under *Go* or *Goto*). |
| Ctrl+G / Cmd+G | Anywhere | The conventional shortcut for the same. `COMMAND` rather than `CTRL` so macOS gets Cmd. |
| Button in the Point Track Detail panel | Header, beside *Copy Point ID*; and in the empty state | Copy and Go-to are the two halves of one round trip, so they belong side by side. The empty state is where a user with an ID in hand and no selection actually looks. |

The **Go** menu is new. The menu bar previously held only **File** — the former
View menu having moved into the viewport HUD (see
[gui-user-experience.md](gui-user-experience.md)) — because nothing left was
app-global. Go to Point *is* app-global: it can retarget the selected
reconstruction, so it belongs to no single panel.

Opening an already-open dialog is idempotent — it re-focuses the field — so the
menu item and the shortcut cannot fight over it.

### The Dialog

A modal `egui::Window` anchored to the viewport centre, non-collapsible and
non-resizable, matching the existing *Load Demo Data* dialog:

```
+--------------------------------------------------+
|  Go to Point                                 [x] |
+--------------------------------------------------+
|  Point index, or full ID with hash:              |
|  [ 12345   or   pt3d_a1b2c3d4_12345           ]  |
|  A bare index refers to the selected             |
|  reconstruction; a full ID selects the one it    |
|  names.                                          |
|                                                  |
|  [ Go ]  [ Cancel ]                              |
+--------------------------------------------------+
```

| Interaction | Effect |
|-------------|--------|
| Text field | Focused on open. Contents persist across opens so a mistyped index is corrected, not retyped. |
| Enter, or **Go** | Submit. **Go** is disabled while the field is blank. |
| Esc, **Cancel**, or the window's ✕ | Close without selecting. |
| A query that does not resolve | The dialog **stays open** with the reason in the error colour under the field, and focus returns to the field. |

Staying open on failure is the whole point of the error handling: a mistyped
hash is one character away from a correct one, and a dialog that closed would
throw the other 18 characters away with it.

### Accepted Input

Two shapes, plus tolerance for how they get pasted:

| Input | Meaning |
|-------|---------|
| `12345` | Point index in the **currently selected** reconstruction. |
| `pt3d_a1b2c3d4_12345` | Point 12345 in the reconstruction whose `content_xxh128` starts with `a1b2c3d4`. |

Tolerated without complaint:

- **Surrounding whitespace** — pasting rarely trims.
- **A leading `#`** — the Image Detail hover tooltip prints `Point3D #12345`,
  so `#12345` is a natural thing to copy or retype.
- **Any case in the Point ID** — `PT3D_A1B2C3D4_7` is the same reference. The
  hash is lowercased at the parse, once, so nothing downstream compares case.
- **A full 32-character hash** — the format spec offers the whole
  `content_xxh128` for exact disambiguation, so an ID built from one resolves
  by prefix match like any other.

Anything else is rejected with a message naming both accepted shapes rather
than guessing at intent. A malformed ID says *which half* is wrong — a
non-hex hash and a non-numeric index get different messages, because "that
isn't a point ID" doesn't tell you where to look.

### Resolution

A bare index resolves against the selected reconstruction; a qualified ID
resolves against whichever loaded node carries that content hash. That
difference is what lets a pasted ID move the whole session to a *different*
loaded file — the behaviour that makes an ID copied in one session useful in the
next.

**Ambiguity.** Several loaded nodes can match one hash: the same file opened
from two paths shares a `content_xxh128`, and any two reconstructions carrying
*no* hash both display as `00000000`. The selected node wins if it is among the
matches; otherwise the first in scene order does. Every match holds the same
content by definition, so the index means the same thing in each, and preferring
the selected node just keeps the answer where the user is already looking. The
preference only breaks ties among real matches — it never turns a miss into a
hit.

**A reconstruction with no hash** resolves by the `00000000` it displays. Demo
data and any pre-hash file are in this state, and an ID copied out of the panel
has to paste back in.

**Bounds.** The index is checked against the resolved node's point count before
any selection happens, and the failure names the node and its count
(`run_b has 60 points — index 60 is out of range.`). Checking here rather than
leaving it to the panels matters: a selection pointing past the end of its own
reconstruction would render as an empty Point Track panel with nothing to say
why.

### What "Go" Does

On a query that resolves:

1. `AppState::select_point` selects the point — which also selects its owning
   reconstruction and clears any image selection belonging to a different one,
   per the finer-selection invariant in
   [gui-scene-graph.md](gui-scene-graph.md).
2. The **Point Track Detail** tab is raised
   (`DockState::find_tab` + `set_active_tab`), so the jump has something to show
   for itself. In the default layout that panel is tabbed *behind* Image Detail,
   and without this the only visible effect of a successful jump would be a
   recoloured splat somewhere in the 3D viewport.
3. The dialog closes.

Everything else follows from the ordinary selection propagation: track rays
appear in the 3D viewer, observing thumbnails gain orange borders in the Image
Browser, and the Scene panel's `selected:` row updates.

Note what Go to Point deliberately does **not** do: it does not move the camera.
Framing the viewport on the point would fight the user's navigation state, and
the selection highlight plus the now-visible track table already answer "where
did I land". Zoom-to-fit remains an explicit action.

## Implementation

`crates/sfm-explorer/src/goto_point.rs`, with tests in `goto_point/tests.rs`.

The parse and the scene lookup are plain functions over the scene slice, so the
interesting behaviour is testable without running a frame:

```rust
/// What the user typed, once recognized as a point reference.
pub enum PointQuery {
    /// A bare index — resolves against the selected reconstruction.
    Index(usize),
    /// A full `pt3d_<hash>_<index>` id — the hash names the reconstruction.
    Qualified { hash: String, index: usize },
}

pub fn parse_point_query(input: &str) -> Result<PointQuery, String>;

pub fn resolve_point_query(
    scene: &[SceneNode],
    selected: Option<ReconId>,
    query: &PointQuery,
) -> Result<PointRef, String>;
```

`GotoPointDialog` is the thin egui shell around them. It holds the open flag,
the text buffer, the last error and a pending-focus flag, lives on `AppState`
beside the demo dialog's state, and its `show` **returns** the resolved
`PointRef` rather than applying it — the caller (`app.rs`) owns what "go there"
means, which keeps the module free of `AppState`.

The Point Track Detail panel reports its button through
`PointTrackDetailResponse::request_goto_point`, which `dock.rs` turns into
`state.goto_point.open()` — the same shape every other cross-panel action in
that response uses.
