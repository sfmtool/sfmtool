# GUI interaction audit — 2026-11-21

An inventory of every pointer gesture SfM Explorer defines (click,
double-click, right-click, hover, modifiers and marks, in every panel), each
read in its spec and in its code, at `main` `e6aac050`. The findings are the
places where the viewer does not follow the interaction rules proposed in
[`specs/drafts/gui-interaction-rules.md`](../specs/drafts/gui-interaction-rules.md),
cited by rule number (G1 to G12), or where a spec and its code disagree. The
code was read, not driven: findings marked **read from the code** were not seen
in a running viewer and need a repro before a fix. Paths are under
`crates/sfm-explorer/src/` for code and `specs/gui/` for specs.

## Summary

| # | Rule | Area | Finding |
|---|---|---|---|
| 1 | G6 | Scene tree | The `hovered:` row under Points cannot be clicked |
| 2 | G6 | Track Table, 3D view | A hover does not clear the image hover it does not produce (read from the code) |
| 3 | — | Specs | `cross-panel-hover.md`'s table of hover sources is out of date |
| 4 | G1 | Image Detail | The click on a feature has no spec, and the spec calls the panel display-only |
| 5 | G2 | Track Table, Scene tree | A row that denotes a point has no double-click |
| 6 | G3, G8 | 3D view | A frustum has no right-click menu |
| 7 | G1 | 3D view | Alt+click selects as well as setting the target, which the spec does not say |
| 8 | G4 | 3D view | Zoom to fit has no Move Camera check (read from the code) |
| 9 | — | Move Camera | Selecting another node's image does not end a held camera |
| 10 | G10 | Track View | A double-click on a recent-items chip may focus two items (read from the code) |
| 11 | G1 | Camera Intrinsics | The image-name link re-selects the selected image |
| 12 | — | Image Browser | A click stops animation playback, which no spec says |
| 13 | G12 | Track View | Shift-click adds one row instead of selecting a range |
| 14 | G1 | Image Detail | A click on an empty part of the photograph keeps the selected point |
| 15 | G3 | Image Detail | A right-click on a feature does not select its point |

## Findings

### 1. The Scene tree's `hovered:` row cannot be clicked

The Points group shows a `hovered:` row while a point is hovered, and the spec
says a click on it selects that point. With the pointer over the tree, the tree
sets `has_pointer` (`scene_graph/mod.rs:373`) and reports no point hover, so
the dock clears `hovered_point` (`dock.rs:1228`) and the row disappears as the
pointer reaches it. Either the row keeps the hover it shows while the pointer
is over it, or the row goes.

### 2. A hover does not clear the image hover it does not produce

G6: the panel under the pointer sets the hover it produces and clears the
other. The Track Table sets `hovered_point` and never clears `hovered_image`
(`track_table_panel.rs:1032-1034`). The panels that own no hover (Edit History,
Action Log, Background Tasks) clear neither, and the 3D view updates the hover
only while `hover_pixel` is set (`app.rs:1254`), so leaving it may leave a
frustum or thumbnail highlighted until the pointer leaves the window.
`cross-panel-hover.md` says leaving a panel clears both. Read from the code.

### 3. `cross-panel-hover.md`'s table of hover sources is out of date

It omits the Scene tree, Track View and the Track Table. `track-table.md:683`
says the Scene tree sets the point hover, which it does not (finding 1).

### 4. Image Detail's click on a feature has no spec

A click on a tracked feature selects its point (`image_detail/overlay.rs:536-546`,
`dock.rs:877-883`), which `cross-panel-hover.md` and `track-view.md:296` rely
on. `multi-panel-image-browser.md:88` calls the panel "display-only (no
selection input)" and line 156 lists "Selection input (3D Viewer only)", and no
section says what the click does.

### 5. A row that denotes a point has no double-click

G2: a double-click on a point moves the 3D view onto it. A double-click on a
point in the 3D view, a feature in Image Detail and a Bench row all do; a Track
Table row (`track_table_panel.rs:1855-1920`) and the Scene tree's `selected:`
row (`scene_graph/mod.rs:1308-1320`) do nothing.

### 6. A frustum has no right-click menu

G3 and G8: the image menu is one menu offered wherever an image is the target.
A right-click on a thumbnail or an image row opens it; a right-click on a
frustum in the 3D view does nothing (`viewer_3d/mod.rs:565-570`).

### 7. Alt+click in the 3D view also selects

Alt+click sets the orbit target from the depth under the pointer and also
selects what is there (`app.rs:1283-1336`). `viewport-navigation.md` mentions
only the target. Either the spec says it selects, or the click stops selecting.

### 8. Zoom to fit has no Move Camera check

G4: nothing moves the 3D view while a camera is held for Move Camera.
`Viewer3D::turn_and_move_target_to`, `move_target_to` and `turn_toward_bearing`
check the lock; `Viewer3D::zoom_to_fit_points` (`viewer_3d/mod.rs:1397-1418`)
does not, so a double-click on a reconstruction or camera row may move a held
camera. Read from the code.

### 9. Selecting another node's image does not end a held camera

`edits/move-camera.md:77-78` says selecting another node ends the lock. A click
on a reconstruction row of another node does (`dock.rs:1194-1206`), and so does
any double-click that enters camera view; a single click on another node's
image row or frustum does not (the callers of `camera_lock::exit_implicitly`
are `app.rs:1301` and `dock.rs:642, 1202, 1425`).

### 10. A double-click on a recent-items chip may focus two items

A click on a chip focuses its item, which leaves the strip, and the item just
left is put first, so the second click of a double-click can land on a
different chip (`track_view/recent.rs:124-146`). Read from the code.

### 11. The Camera Intrinsics image-name link does nothing

The link in the pose header selects the image that is already selected
(`intrinsics_detail/extrinsics.rs:141-144`, `intrinsics_detail/mod.rs:343-345`).

### 12. An Image Browser click stops animation playback

`image_browser.rs` stops playback on a thumbnail click. No spec says so; G1
says a click selects and does nothing more. Either the spec states it as part
of selecting during playback, or the click leaves playback alone.

### 13. Shift-click on a Track View row is not a range

G12: Shift-click selects the range from the anchor, and Ctrl+Shift-click adds
it. Both Ctrl-click and Shift-click on a Track View row add or remove the one
row (`track_view/body/table.rs:1744`), as `bench.md:771-772` specifies.

### 14. A click on empty photograph keeps the selected point

G1: a click on empty space clears the selection a click there could have
made. In the 3D view it does. In Image Detail, a click on an untracked
keypoint or on empty photograph reports `select_point = None`, which the dock
ignores, so the selected point stays.

### 15. A right-click on a feature selects nothing

G3: a right-click on a point selects it before its menu opens. In the 3D view
it does (`viewer_3d/mod.rs:572-577`); on a feature in Image Detail it does not
(`multi-panel-image-browser.md:402`, "A right click selects nothing").

## Checked and found following the rules

- G1: clicks on thumbnails, image rows, camera rows, frustums, points, Bench
  rows, observation rows, histogram bins and bench marks select and move no
  view; none toggles off.
- G2: image targets enter camera view; reconstruction and camera rows frame
  their set; observation rows look through toward the feature; Bench rows and
  points move the view under G4; empty photograph in Image Detail zooms.
- G3: every menu opens through `context_menu::on_secondary_click`; tree rows
  and thumbnails select nothing.
- G4 and G5: `turn_and_move_target_to` and `Look::Reveal` as described.
- G7: bench handles and marks own their press in the 3D view and Image Detail.
- G9: every raise goes through `show_panel_beside`.
- G11: an Edit History click jumps the cursor.
