# Interaction Rules

SfM Explorer, sfmtool's desktop viewer for reconstructions, shows the same
things in many panels: an image is a thumbnail in the Image Browser, a row in
the Scene tree, a frustum in the 3D view and a row in Track View, and a point
is a dot in the 3D view, a feature in Image Detail, a bench row and a Track
Table row. A person learns what a click does in one panel and expects it to do
the same kind of thing everywhere. This spec holds the rules every pointer
gesture in the viewer follows (a click, a double-click, a right-click, a hover,
a press on a mark, and the modifier clicks of a list with a multi-selection),
so that a click, a double-click, a hover and a right-click mean the same kind
of thing in every panel.

Each rule has a number, G1 to G12, so a spec, an audit finding or a commit can
cite it. A panel's spec says which rule each of its gestures follows ("Click:
G1. Double-click: G2, enter camera view") and states only what the rule leaves
to the panel, such as which panel a double-click raises. `specs/STANDARDS.md`
§ ["A viewer gesture follows the interaction
rules"](../STANDARDS.md#a-viewer-gesture-follows-the-interaction-rules) holds
every gesture to them, with no documented exceptions: where a gesture and a
rule disagree, the gesture changes, or the rules change for every panel at
once. The `audit-hygiene`, `audit-specs` and `gui-bug-bash` skills check the
code, the specs and the running viewer against them.

The viewer does not follow every rule in every panel. The gestures that
differ are the findings of
[`reports/2026-11-21-gui-interaction-audit.md`](../../reports/2026-11-21-gui-interaction-audit.md),
and the work that brings each into line is
[drafts/gui-interaction-rules.md](../drafts/gui-interaction-rules.md).

## The rules

**G1. A click selects what it is on, and nothing more.** The selection is the
viewer's one selected item of each kind: a reconstruction, an image, a camera,
a point, and the bench observation Track View edits. Other panels follow it: the Image Browser and the Scene tree scroll
it into view, the Track Table scrolls its row into view, and Image Detail
reveals a pixel it names under G5. A click never moves the 3D view, never
raises a panel and never pushes a version. A click on the item already
selected keeps it selected; nothing toggles off. A click on empty space in the
3D view, or on a part of the photograph in Image Detail where nothing is
drawn, clears the selection a click there could have made: the image and the
point in the 3D view, the point in Image Detail.

**G2. A double-click does what the target stands for, which a click does
first.** It is the click, then one action chosen by what the target denotes:

| The target denotes | The double-click |
|---|---|
| An image (a thumbnail, an image row, a frustum) | Enter camera view through it. |
| A pixel of an image (a Track View observation row) | Enter camera view through its image, turned until the pixel is inside the middle 1/2 of the viewport (`Viewer3D::look_through_toward_feature`). |
| A point (a dot in the 3D view, a feature in Image Detail) | Put its track on the bench when it is not there already, which pushes a version, raise Track View on it, and move the 3D view onto it under G4. |
| An item on the bench (a Bench row) | Focus it, raise Track View on it, and move the 3D view onto its point under G4. |
| A set of things (a reconstruction, a camera's images) | Frame the set in the 3D view. |
| A position inside a picture with nothing on it (empty photograph in Image Detail) | Zoom the picture about the pointer. |

A double-click raises a panel only through `AppState::show_panel_beside`, so
the raised panel never covers the one the gesture was made in (G9).

**G3. A right-click opens the target's menu.** On a point, a dot in the 3D
view or a feature in Image Detail, it first selects the point, as a click
would: a point is one of many small things close together, and the selection's
highlight is what shows which of them the menu acts on. On anything else, a
row, a thumbnail, a frustum or a patch square, which the pointer already names
without doubt, it selects nothing. The menu's entries are the ones that fit
the target as it stands, and an entry that needs something the target lacks,
or that cannot run for another reason, is drawn greyed with hover text that
says why. Every menu is opened through `context_menu::on_secondary_click`, or
its `context_menu::on_secondary_click_where` form for a widget only part of
which carries the menu, so a long left press never opens one and a right drag
is never a menu.

**G4. Moving the 3D view onto a point named outside the viewport.** A point
inside the middle 2/3 of the viewport on both axes is panned to; any other
point, including one behind the camera, first has the camera turn in place
until it is inside the middle 1/2, and is then panned to
(`Viewer3D::turn_and_move_target_to`). A point at infinity is turned toward
(`Viewer3D::turn_toward_bearing`). Nothing moves while a camera is held for
Move Camera. A double-click on a point in the 3D view itself pans only
(`Viewer3D::move_target_to`), since that point is on screen.

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
observation table is such a list.

## Where the rules live in the code

The functions the rules name are the ones every panel calls, which is how a
rule stays one code path (G8):

- [context_menu.rs](../../crates/sfm-explorer/src/context_menu.rs):
  `on_secondary_click` and `on_secondary_click_where`, which open a menu on a
  right-click and on nothing else (G3).
- [layout.rs](../../crates/sfm-explorer/src/layout.rs):
  `AppState::show_panel_beside`, the raise that never covers the panel a
  gesture was made in (G2, G9); [panel-layout.md](panel-layout.md) § "A raise
  from a gesture" describes it.
- [viewer_3d/mod.rs](../../crates/sfm-explorer/src/viewer_3d/mod.rs):
  `Viewer3D::move_target_to`, `Viewer3D::turn_and_move_target_to` and
  `Viewer3D::turn_toward_bearing`, the moves onto a point (G4), whose
  fractions are the constants `PAN_ONLY_FRACTION` (2/3) and
  `TURN_INTO_FRACTION` (1/2); and `Viewer3D::look_through_toward_feature`,
  camera view toward a pixel (G2). [viewport-navigation.md](viewport-navigation.md)
  § "The point context menu" describes the moves.
- [image_detail/view.rs](../../crates/sfm-explorer/src/image_detail/view.rs):
  `Look::Reveal`, Image Detail's reveal (G5), whose margin is the constant
  `REVEAL_MARGIN` (0.05 of the panel on each side) in
  [image_detail/mod.rs](../../crates/sfm-explorer/src/image_detail/mod.rs).
- [image_menu.rs](../../crates/sfm-explorer/src/image_menu.rs): `image_menu::show`,
  the one image menu (G8); [scene-graph.md](scene-graph.md) § "Tree rows"
  describes it.
