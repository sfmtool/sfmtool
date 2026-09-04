# Go to Point

**Go to Point** is a dialog in the viewer for reaching a 3D point by naming it
instead of finding it on screen. It takes a typed or pasted point index, or a
whole portable point ID of the form `pt3d_<hash>_<index>`, and selects that
point — switching the selected reconstruction when the ID names a different one.

For the panel this most directly serves, see
[point-track-detail.md](point-track-detail.md). For the ID format
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
[user-experience.md](user-experience.md)) — because nothing left was
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
|  [ ▓pt3d_a1b2c3d4_4821▓                       ]  |
|  A bare index refers to the selected             |
|  reconstruction; a full ID selects the one it    |
|  names.                                          |
|                                                  |
|  [ Go ]  [ Cancel ]                              |
+--------------------------------------------------+
```

(Shaded = selected. With no point selected the field shows the previous query,
or the hint `12345   or   pt3d_a1b2c3d4_12345` when there is none.)

| Interaction | Effect |
|-------------|--------|
| Text field | Prefilled with the selected point's ID and **fully selected**, focused on open — see below. |
| Enter, or **Go** | Submit. **Go** is disabled while the field is blank. |
| Esc, **Cancel**, or the window's ✕ | Close without selecting. |
| A query that does not resolve | The dialog **stays open** with the reason in the error colour under the field, and focus returns to the field — but the text is *not* re-selected. |

Staying open on failure is the whole point of the error handling: a mistyped
hash is one character away from a correct one, and a dialog that closed would
throw the other 18 characters away with it. For the same reason the failure
path re-focuses without re-selecting: the user is about to fix one character,
and a selected field would delete the whole query on the next keystroke.

#### Prefill and Selection on Open

Opening the dialog puts the **currently selected point's ID** in the field, in
the same `pt3d_<hash>_<index>` form the Point Track header displays, and selects
the whole thing.

Both halves matter, and the selection is the load-bearing one. A field that
opens prefilled but unselected is actively worse than one that opens empty:
typing or pasting lands beside the existing text and produces
`pt3d_aaaa1111_57` or `pt3d_aaaa1111_5pt3d_cccc3333_42` — a query that can only
fail, and one the user has to notice and clear before they can do what they came
for. Selecting the contents makes the first keystroke or paste *overwrite*,
which is the behaviour every address bar and Go-to-line box already trains for.

Selecting also makes the dialog a place to **read and copy** the current point's
ID: Ctrl+C on open copies it without a trip to the panel header, and the field
shows which point you are on before you replace it. That is a second, smaller
reason to prefill — but it only works because the text is selected.

Two cases deliberately do not prefill:

- **Nothing selected** — the previous query stays in the field (still selected),
  so a failed or repeated lookup is edited rather than retyped.
- **A selection pointing past the end of its own reconstruction**, which a
  reload can produce. Prefilling an ID that no longer resolves would hand the
  user a query that fails the instant they press Enter.

Re-opening an **already-open** dialog only re-focuses: it neither re-prefills
nor re-selects, because the text there may be half-typed and the menu item
racing the shortcut must not throw it away.

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
   [scene-graph.md](scene-graph.md).
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

The dialog lives in [goto_point.rs](../../crates/sfm-explorer/src/goto_point.rs),
with tests in
[goto_point/tests.rs](../../crates/sfm-explorer/src/goto_point/tests.rs).

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

/// The selected point's ID, for prefilling — `None` when there is no
/// selection or it has gone stale against its reconstruction.
pub fn selected_point_id(
    scene: &[SceneNode],
    selected_point: Option<PointRef>,
) -> Option<String>;
```

`GotoPointDialog` is the thin egui shell around them. It holds the open flag,
the text buffer, the last error and two pending flags — focus and select-all,
kept apart precisely because the failure path wants one without the other. It
lives on `AppState` beside the demo dialog's state, and its `show` **returns**
the resolved `PointRef` rather than applying it — the caller (`app.rs`) owns
what "go there" means, which keeps the module free of `AppState`.

All three entry points go through `AppState::open_goto_point`, which computes
the prefill and calls `GotoPointDialog::open`. One entry point rather than three
call sites, so none of them can forget the prefill.

Selecting the field's contents is done by writing directly into the widget's
stored `egui::text_edit::TextEditState` — a `TextEdit` offers no "select all on
focus" of its own. That needs a stable widget id, so the field is given an
explicit one rather than egui's auto-generated id. The store lands after the
widget has already run for the frame, so the selection takes effect on the next
frame — which is also the first frame the field is focused, leaving no window in
which the user could type into an unselected field.

The Point Track Detail panel reports its button through
`PointTrackDetailResponse::request_goto_point`, which `dock.rs` turns into
`state.open_goto_point()` — the same shape every other cross-panel action in
that response uses.
