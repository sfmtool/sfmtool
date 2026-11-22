# Interaction rules for SfM Explorer

**Status:** Draft

Decided by the maintainer: SfM Explorer's pointer gestures follow one set of
written rules, so a click, a double-click, a hover and a right-click mean the
same kind of thing in every panel. The rules are a standing spec, every panel
spec points at the rules it follows rather than restating them, and the audit
skills check the code and the specs against them. Six cases where the viewer
broke the first version of these rules were decided as follows, and the rules
below are written to include them:

- A right-click on a point selects it before its menu opens (G3).
- A click on an Edit History row moves the cursor to that version (G11).
- A double-click on a point or a feature puts its track on the bench, which
  pushes a version (G2).
- A click on a recent-items chip focuses that item; a chip is a control (G10).
- A click on empty space in the 3D view, or on an empty part of the photograph
  in Image Detail, clears the selection it would have made (G1).
- In a list with a multi-selection, Shift-click selects a range and
  Ctrl+Shift-click adds a range (G12).

Nothing in this draft is still open.

SfM Explorer has many panels that show the same things: an image is a
thumbnail in the Image Browser, a row in the Scene tree, a frustum in the 3D
view and a row in Track View, and a point is a dot, a feature, a bench row and
a Track Table row. A person learns what a click does in one panel and expects
it everywhere. Until now each panel's spec has said what its own gestures do,
and most of them agree, because each new panel copied an older one. This
draft writes down what they agree on as rules, lists where the viewer does not
follow them, and proposes how the rules are kept: a standing spec, an entry in
`specs/STANDARDS.md`, and a check in each of the three audit skills that read
the viewer.

The inventory this is drawn from, every gesture of every panel with its spec
and its code, was taken on 2026-11-21 from `main` at `e6aac050`.

## The rules

Each rule has a number so a spec, a finding or a commit can cite it.

**G1. A click selects what it is on, and nothing more.** The selection is the
viewer's one selected image, point, camera, measurement or bench row of that
kind. Other panels follow it: the Image Browser and the Scene tree scroll it
into view, the Track Table scrolls its row into view, and Image Detail reveals
a pixel it names under G5. A click never moves the 3D view, never raises a
panel and never pushes a version. A click on the item already selected keeps
it selected; nothing toggles off. A click on empty space in the 3D view, or
on a part of the photograph in Image Detail where nothing is drawn, clears the
selection a click there could have made: the image and the point in the 3D
view, the point in Image Detail.

**G2. A double-click does what the target stands for, which a click does
first.** It is the click, then one action chosen by what the target denotes:

| The target denotes | The double-click |
|---|---|
| An image (thumbnail, image row, frustum) | Enter camera view through it. |
| A pixel of an image (an observation row, a measurement endpoint) | Enter camera view through its image, turned until the pixel is inside the middle 1/2 of the viewport (`Viewer3D::look_through_toward_feature`). |
| A point (a dot in the 3D view, a feature in Image Detail) | Put its track on the bench when it is not there already, which pushes a version, raise Track View on it, and move the 3D view onto it under G4. |
| An item on the bench (a Bench row) | Focus it, raise Track View on it, and move the 3D view onto its point under G4. |
| A set of things (a reconstruction, a camera's images, a measurement's endpoints) | Frame the set in the 3D view. |
| A position inside a picture with nothing on it (empty image in Image Detail) | Zoom the picture about the pointer. |

A double-click raises a panel only through `show_panel_beside`, so the raised
panel never covers the one the gesture was made in (G9).

**G3. A right-click opens the target's menu.** On a point, a dot in the 3D
view or a feature in Image Detail, it first selects the point, as a click
would: a point is one of many small things close together, and the selection's
highlight is what shows which of them the menu acts on. On anything else, a
row, a thumbnail, a frustum or a patch square, which the pointer already
names without doubt, it selects nothing. The menu's entries are the ones that
fit the target as it stands; an entry that needs something the target lacks is
drawn greyed, as below. Every menu is opened through
`context_menu::on_secondary_click`, so a long left press never opens one and a
right drag is never a menu. An entry that cannot run
is drawn greyed with hover text that says why.

**G4. Moving the 3D view onto a point named outside the viewport.** A point
inside the middle 2/3 of the viewport on both axes is panned to; any other
point, including one behind the camera, first has the camera turn in place
until it is inside the middle 1/2, and is then panned to
(`Viewer3D::turn_and_move_target_to`). A point at infinity is turned toward
(`Viewer3D::turn_toward_bearing`). Nothing moves while a camera is held for
Move Camera. A double-click on a point in the 3D view itself pans only, since
that point is on screen.

**G5. Image Detail reveals a pixel by panning, never zooming,** and only when
the pixel is outside the middle 90% of the view and the view is not at fit
(`Look::Reveal`).

**G6. The panel under the pointer owns the hover.** It sets the hover field it
produces and clears the other; leaving a panel clears both. Hover never
changes the selection, and every panel that draws the hovered thing
highlights it.

**G7. A mark or handle drawn over other content owns its press.** A click on
it selects or moves what the mark stands for, never what lies underneath.

**G8. One gesture is one code path.** The same gesture on the same kind of
target, in two panels, calls the same function, so the two cannot drift
apart. A menu offered in two places is one menu (the image menu).

**G9. A gesture that raises a panel never covers the panel it was made in.**

**G10. A control is not a selection target.** A switch, a pin, a button, a
field, a chip such as Track View's recent items, and a tool such as the eraser
do what their label or their hover text says on a single click, which may
push a version. G1 and G2 apply to the things the panels show, not to these.

**G11. In a list of states, a click moves to the state.** A row of the Edit
History panel is a version of the document rather than a thing in it, so
selecting the row and moving the cursor to that version are one act: a click
moves the cursor, and nothing else in the viewer is selected by it.

**G12. A list with a multi-selection.** A click selects the row alone and makes
it the anchor. Ctrl-click adds the row to the selection, or takes it out, and
makes it the anchor. Shift-click selects the rows from the anchor to the row,
in the order the list shows, in place of the selection. Ctrl+Shift-click adds
those rows to the selection. On macOS, Cmd stands for Ctrl. Track View's
observation table is such a list today.

## How the rules are kept

1. **The standing spec.** The rules above become
   `specs/gui/interaction-rules.md`, listed in `specs/gui/README.md` beside
   `user-experience.md`. A panel's spec says which rule each of its gestures
   follows ("Click: G1. Double-click: G2, enter camera view") and states only
   what the rule leaves to it, such as which panel a double-click raises.
2. **`specs/STANDARDS.md`** gains a standard: "A viewer gesture follows
   `specs/gui/interaction-rules.md`." Like every standard it has no documented
   exceptions: where a gesture and a rule disagree, the gesture changes, or the
   maintainer changes the rule for every panel at once.
3. **`audit-hygiene`** gains a check: for each viewer unit it reads, list the
   gestures the code defines (the `clicked`, `double_clicked`,
   `secondary_clicked`, `hovered` and `context_menu::on_secondary_click`
   handlers) and check each against the rules and against the panel's spec.
4. **`audit-specs`** gains a cross-spec check: the gesture sentences of every
   `specs/gui/` spec in the sample are compared with the rules, so a spec that
   restates a rule differently, or describes a gesture the rules do not allow,
   is a finding.
5. **`gui-bug-bash`** gains a probe list drawn from the rules: for each panel,
   click, double-click, right-click and hover its targets over MCP and check
   the outcome against the rule (the selection block, `get_view`, the Image
   Detail view, the version count, which panel is raised). A gesture the rules
   allow is not a bug even if surprising; one they do not allow is, even if
   its own spec describes it.

## Where the viewer does not follow the rules

Found by the inventory, and each is fixed in the code, or in the spec where
the code is right and the spec drifted. They are tracked, with their code
locations, as the findings of
[`reports/2026-11-21-gui-interaction-audit.md`](../../reports/2026-11-21-gui-interaction-audit.md).

**Likely accidental:**

1. The Scene tree's `hovered:` row cannot be clicked: the pointer over the tree
   clears the point hover, so the row disappears as the pointer reaches it
   (`scene_graph/mod.rs:373`, `dock.rs:1228`). G6.
2. The Track Table's hover never clears `hovered_image`, and leaving the 3D view
   may leave a frustum or thumbnail highlighted until the pointer leaves the
   window (`track_table_panel.rs:1032`, `app.rs:1254`). G6. The latter is read
   from the code, not seen.
3. `cross-panel-hover.md`'s table of hover sources omits the Scene tree, Track
   View and Track Table, and `track-table.md` says the Scene tree sets the point
   hover, which it does not. Spec drift.
4. Image Detail's single click on a feature, which selects its point, has no
   spec section, and `multi-panel-image-browser.md` calls the panel
   "display-only (no selection input)" in two places. Spec drift.
5. A Track Table row and the Scene tree's `selected:` row have no double-click,
   though every other target that denotes a point moves the view onto it. G2.
6. A frustum has no right-click menu, though the image menu is on the
   thumbnail and the image row. G3, G8.
7. Alt+click in the 3D view selects what it is on as well as setting the
   target; the spec mentions only the target.
8. `Viewer3D::zoom_to_fit_points` has no Move Camera check, so a double-click
   on a reconstruction or camera row may move a held camera. G4. Read from the
   code, not seen.
9. A single click that selects another node's image does not end a held Move
   Camera, though `edits/move-camera.md` says selecting another node does.
10. A double-click on a Track View recent-items chip may focus two different
    items, since the strip reorders on the first click. Not seen.
11. The Camera Intrinsics panel's image-name link selects the image that is
    already selected, so the click does nothing.
12. A click in the Image Browser stops animation playback, which no spec says.
13. Shift-click on a Track View row adds the row, as Ctrl-click does, rather
    than selecting a range (`table.rs:1744`). G12.
14. A click on an empty part of the photograph in Image Detail keeps the
    selected point. G1.
15. A right-click on a feature in Image Detail selects nothing, where a
    right-click on a point in the 3D view selects it. G3.

## Steps

1. File `specs/gui/interaction-rules.md` with the rules as decided, point every
   panel spec's gesture section at it, add the standard, and add the checks to
   the three skills. Spec and skill text only.
2. One PR per finding in the report, through `fix-report-findings`.
