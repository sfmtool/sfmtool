# Bringing SfM Explorer's gestures into line with the interaction rules

**Status:** Draft

Amends [gui/interaction-rules.md](../gui/interaction-rules.md), the standing
spec of the rules G1 to G13 that every pointer gesture in SfM Explorer
follows. The rules are decided; this draft holds only the work that makes the
viewer follow them everywhere, and is deleted when that work is done.

An inventory of every gesture of every panel, with its spec and its code,
taken on 2026-11-21 from `main` at `e6aac050`, found fifteen places where the
viewer or its specs do not follow the rules, and filing the rules found a
sixteenth. The sixteen are tracked, with their code locations, as the findings
of
[`reports/2026-11-21-gui-interaction-audit.md`](../../reports/2026-11-21-gui-interaction-audit.md):

| Finding | Rule | What changes |
|---|---|---|
| 1 | G6 | The Scene tree's `hovered:` row keeps the hover it shows while the pointer is over it, or the row goes. |
| 2 | G6 | Every panel clears the hover it does not produce, and leaving the 3D view clears its hover. |
| 3 | — | `cross-panel-hover.md` lists every hover source; `track-table.md` stops saying the Scene tree sets the point hover. |
| 4 | G1 | `multi-panel-image-browser.md` specifies Image Detail's click on a feature. |
| 5 | G2 | A Track Table row and the Scene tree's `selected:` row move the 3D view onto their point on a double-click. |
| 6 | G3, G8 | A right-click on a frustum opens the image menu. |
| 7 | G1 | Alt+click in the 3D view and its spec agree on whether it selects. |
| 8 | G4 | Zoom to fit moves nothing while a camera is held for Move Camera. |
| 9 | — | A click that selects another node's image ends a held Move Camera, as `edits/move-camera.md` says. |
| 10 | G10 | A double-click on a Track View recent-items chip focuses one item. |
| 11 | G1 | The Camera Intrinsics panel's image-name link does something, or goes. |
| 12 | — | An Image Browser click and animation playback agree with their spec. |
| 13 | G12 | Shift-click on a Track View row selects a range, and Ctrl+Shift-click adds one. |
| 14 | G1 | A click on an empty part of the photograph in Image Detail clears the selected point. |
| 15 | G3 | A right-click on a feature in Image Detail selects its point. |
| 16 | G2, G9 | A double-click on a Scene tree Bench row raises Track View through `show_panel_beside`, and `panel-layout.md` § "A raise from a gesture" says so. |

Each is fixed in the code, or in the spec where the code is right and the spec
drifted, as its own change through `fix-report-findings`. A panel spec whose
gesture is fixed then cites the rule it follows, as the other gestures' specs
do. When the last finding is fixed this draft is deleted, and the standing
spec's sentence that links it goes with it.
