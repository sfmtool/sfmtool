# Go to Point

**Go to Point** is a dialog in the viewer for reaching a 3D point by naming it
instead of finding it on screen. It takes a typed or pasted point index, or a
whole portable point ID of the form `pt3d_<hash>_<index>`, and selects that
point -- switching the selected reconstruction when the ID names a different
one.

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
|  Go to Point                                 [×] |
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

Opening the dialog puts the **currently selected point's ID** in the field, as
the Point Track header displays it, and selects the whole thing.

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
- **A selection pointing past the end of its own reconstruction**, which an
  edit can produce. Prefilling an ID that no longer resolves would hand the
  user a query that fails the instant they press Enter.

Re-opening an **already-open** dialog only re-focuses: it neither re-prefills
nor re-selects, because the text there may be half-typed and the menu item
racing the shortcut must not throw it away.

### Accepted Input

Two shapes, plus tolerance for how they get pasted:

| Input | Meaning |
|-------|---------|
| `12345` | Point index in the **currently selected** reconstruction, as it stands: a coordinate in the value on screen, not a name to be resolved. |
| `pt3d_a1b2c3d4_12345` | A **point ID**: the point numbered `12345` in the content `a1b2c3d4`, wherever a loaded node has held that content. |

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
resolves against whichever loaded node has held that content, and then walks
that node's version graph to the version on screen. That difference is what lets
a pasted ID move the whole session to a *different* loaded file -- the behaviour
that makes an ID copied in one session useful in the next -- and what lets one
copied before a run of edits still land after them.

**Which node.** Nothing in an ID names one. A point's identity is its content
hash and its row in that content, which is the same pair in every node that
holds that content, so which node to show is a question about the session rather
than about the ID, and it is answered by the selection. The **selected node is
tried first**, and then every other loaded node in scene order; the first that
has held the content wins. Every match held the same content by definition, so
the index means the same thing in each, and going to the selected node first
just keeps the answer where the user is already looking. **A closed node is a
miss**: the message names what was searched, so the answer says the ID's content
is in none of these three loaded reconstructions rather than only that it was not
found.

**A bare index is still a coordinate**, not a name. It is used as it stands
against the selected node's current value, with no hash to resolve and no walk
to make, which is what makes it the right thing to type when the number came off
this reconstruction a moment ago.

**Bounds.** A bare index is checked against the resolved node's point count before
any selection happens, and the failure names the node and its count
(`run_b has 60 points — index 60 is out of range.`). Checking here rather than
leaving it to the panels matters: a selection pointing past the end of its own
reconstruction would render as an empty Point Track panel with nothing to say
why.

### The ID forms and the version graph

A point ID names a point by the **content that created it** and its place in
that content, never by a counter or a session. Two forms carry that:

```
pt3d_{hash}_{index}
```

| Part | Content | Example |
|------|---------|---------|
| `{hash}` | First 8 hex digits of the content hash of the base the ID is minted against, whether or not that base has been written to a file; or, for a point a point edit created, of that edit's content hash. | `a1b2c3d4` |
| `{index}` | The point's index in that base; or, for a point a point edit created, its index among the points that edit created, numbered from zero in creation order. | `12345` |

There is one form, and it carries nothing about the session it was copied in.
A point's identity **is** its content hash and its row there, and that pair is
the same in every node that holds that content, so a field naming a node would
add nothing to the identity and would only say where one copy of the point
happened to be looked at. Which node to show is a question about the session,
and it is answered by the selection (§ "Resolution") rather than by the ID. The
whole ID stays inside the `[a-zA-Z0-9_]` class, so it double-click selects. IDs
are minted and resolved in
[point_ids.rs](../../crates/sfm-explorer/src/point_ids.rs), over the version
graph a node's `History` keeps
([document.rs](../../crates/sfm-explorer/src/document.rs)).

**The version graph** is the history's step list: one entry per version ever
minted, holding the version, the version it was made from, the point map between
them, and the points that step created
([edit-history.md](edit-history.md) § "The version graph"). It is never pruned,
neither by the history budget nor by the truncation a new edit after an undo
performs, so a discarded redo tail is still walkable and an ID minted on it
still has somewhere to resolve.

**Minting** walks back from the cursor, inverting each step's map as far as the
point's identity reaches, and then chooses which content on that walk to name.
The rule is **the version on disk first, and the earliest content otherwise**.

- **The version on disk** is the version the node was loaded at or last saved as
  (`History::disk_serial`, [saving.md](saving.md)). If the point's identity
  reaches it and the point is a row of its base, that base's hash and that row
  are the ID. This is the ID a reader of the file on disk uses as it stands,
  with no lineage to consult and no other file to find, and reaching the file on
  disk is what someone copying an ID almost always wants it for.
- **The earliest content** otherwise. A step that says it created the point ends
  the walk, and the ID is that edit's content hash with the point's place among
  that edit's creations. Otherwise the ID is the base content hash of the oldest
  version on the trail that still holds its value, with the index the point has
  in it; a version the budget released is skipped, since it has no columns to
  hash.

Three things fall to the second rule: a point created since the last save, which
is a row of no base at all; a cursor on a branch the disk version is not an
ancestor of, such as after an undo past a save; and a disk version the budget has
released. None of those is a broken ID, only a weaker one, because the lineage a
save records keeps an earlier content's IDs resolving in every file written
afterwards ([the format spec's Lineage
section](../formats/sfmr-file-format.md#lineage-version-9)). That is also why
preferring the disk version costs nothing: the earlier ID a user already wrote
down goes on working after the save that moved the displayed ID off it.

**Resolving** finds the hash first and walks second. The hash is looked for
among, in order: the point edits' hashes, the bases of the versions that still
hold values (the computed hash, or the stored one a loaded file carries), and
those bases' recorded lineage. From wherever it was found, `History::follow`
walks to the cursor -- back to the last version the two share, inverting each
step's map, then forward -- and `Err` carries the version the walk stopped at.
So there are **two distinct failures**, and the messages separate them: the node
has never held content with that hash, and the row the ID named is not in the
value at the cursor, which names the version the walk stopped at.

`point_ids::holds_hash` asks about the hash alone. That is what keeps the two
failures apart across nodes: a node that held the content but no longer holds
the row is still the node the query belongs to, and it gets to say what happened
to the row rather than being passed over for a node that never saw the hash at
all.

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
    /// A full `pt3d_<hash>_<index>` id -- the hash names the content the point
    /// was minted against, and the index its row there.
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
