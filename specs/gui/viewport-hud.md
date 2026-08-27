# Viewport HUD

**Status: implemented** in `viewer_3d/hud.rs`, tests in
`viewer_3d/hud/tests.rs`. The display controls have left the menu bar, and the
**View** menu is gone with them — **File** is now the only menu.

This document specifies moving the 3D-viewport display controls out of the
menu bar and onto a heads-up display drawn inside the viewport itself.

---

## Motivation

Every control that used to sit in the View menu (`app.rs`) is 3D-viewport
state. None of it affects the Image Browser, Image Detail, or Point Track
panels. Two problems followed from housing it in an app-global menu:

1. **It contradicts the panel model.** The dock already establishes that a
   panel owns its own controls: Image Detail has an overlay/filter toolbar
   (`dock.rs::show_overlay_toolbar`), Image Browser has a playback minibar. The
   3D Viewer is the only panel that outsources its display controls.

2. **It breaks the tuning loop.** Point size, patch opacity, length scale and
   FOV are all tune-until-it-looks-right sliders. The menu popup is anchored to
   the top-left menu bar — far from where you are looking, and covering the
   viewport while you drag. Each adjustment costs a menu round-trip.

---

## Scope

**Moved to the HUD** — all twelve former View-menu controls, plus the four
parameters that were plumbed to the GPU but had no widget at all
(`edl_line_thickness`, `frustum_size_multiplier`, `target_size_multiplier`,
`target_fog_multiplier`), plus the two previously unbuilt point-cloud controls
(["Show points at infinity" toggle and count
readout](point-cloud-rendering.md#ui--shipped)).

**Stays in the menu bar** — File (Open / Load Demo Data / Quit), and nothing
else. With the display controls gone there is no app-global viewport state left
for a **View** menu to hold, and the dock panels are permanent
(`TabViewer::closeable` is false), so it is deleted rather than repurposed. A
menu kept alive for one synthetic entry is worse than no menu.

**Out of scope** — the Image Detail and Image Browser toolbars. They already
follow the panel-owns-its-controls rule and are not touched.

---

## Layout

The HUD is **open by default**. The controls are the point of the panel, and a
viewport that starts by hiding them just trades a menu round-trip for a click.
It collapses to a single gear glyph for when the full-bleed viewport matters
more (design principle #4, *Dark, Cinematic Aesthetic*) — the translucent fill
keeps it from fighting the scene while open.

```
┌────────────────────────────────────────────────────────────┐
│ 12,043 pts | 17 imgs | 60 fps   Pos: [1.2, -3.4, 0.8]   ⚙ │  collapsed
│                                                            │
│                                                            │
│  ⟂                          Point3D #841 | depth: 2.13     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ 12,043 pts | 17 imgs | 60 fps   Pos: [1.2, -3.4, 0.8]   ✕ │  expanded
│                                      ┌───────────────────┐ │
│                                      │ ▾ Layers          │ │
│                                      │   ☑ Points        │ │
│                                      │   ☑ Camera Images │ │
│                                      │   ☑ Grid          │ │
│                                      │   ☑ Patches       │ │
│                                      │ ▾ Size            │ │
│                                      │   Points  ──●──   │ │
│                                      │   ∞ (px)  ─●───   │ │
│                                      │   Scene   ──●──   │ │
│                                      │ ▸ Patches         │ │
│                                      │ ▾ Camera          │ │
│                                      │   FOV 45°  ─●──   │ │
│                                      │   [Reset FOV]     │ │
│                                      │ ▸ Advanced        │ │
│                                      │ ▸ Debug           │ │
│                                      └───────────────────┘ │
│  ⟂                          Point3D #841 | depth: 2.13     │
└────────────────────────────────────────────────────────────┘
```

**Anchor**: top-right, inset from the viewport edge. The existing overlays
occupy all four corners plus the top centre
([user-experience.md](user-experience.md#information-overlays)), and
top-right is the cheapest to vacate — it holds only the touchpad diagnostics,
which move into the HUD's Debug section (they are developer instrumentation and
should not be permanently burned into the viewport anyway).

**Position is relative to the viewport rect, recomputed every frame.** The HUD
lives in a dock tab that the user can resize, re-dock, or tab away from. An
`Area` at a fixed screen position would detach from the panel; the fixed
position must be derived from the viewport rect on each frame. In practice the
HUD is built by `Viewer3D::show_hud`, called from `dock.rs` immediately before
`Viewer3D::show` and from the same `Ui`, and takes its viewport rect from
`ui.available_rect_before_wrap()` — which is exactly the rect `show` then hands
to `allocate_painter`. The split exists for borrows: the HUD needs `&mut
AppState` whole, while `show` borrows `reconstruction` and `selected_image` as
separate fields. The `Area` is anchored with an `Align2::RIGHT_TOP` pivot, so
the collapsed gear and the wider expanded panel share a right edge without
either needing to know its own width.

**Expanded width** is fixed (220 pt) so slider tracks do not jitter as
labels change. If the viewport is too small to show the expanded HUD without
covering more than about a third of it, the HUD stays collapsed and the gear
is the only affordance. Concretely: the expanded panel needs a viewport of at
least 472 × 300 pt (twice the panel's width-plus-insets, and enough height for
a few sections). At the minimum width the panel spans half the viewport
horizontally but only part of it vertically, so the area it covers stays under
a third. A refused open is remembered, not discarded — widen the panel and the
HUD expands without a second click.

**Every section is visible at once — the panel never scrolls.** Its height is
whatever its content needs, and `constrain_to(viewport)` keeps it on screen. A
scroll container inside a floating panel this small costs more than it saves:
it hides controls behind a gesture that the viewport underneath also consumes.

**The panel is slightly translucent** (88% fill opacity), so a little of the
scene shows through and it reads as floating over the viewport rather than
bolted to it. Not lower: slider tracks and checkmarks have to stay legible
against a bright point cloud.

---

## Sections

Collapsible (`egui::collapsing_header::CollapsingState` under explicit,
`Ui`-independent ids — `hud::section_id`), with open/closed state remembered for
the session. Defaults: Layers, Size, and Camera open; Patches, Advanced, and
Debug closed.

| Section | Contents |
|---------|----------|
| **Layers** | Show Points, Show Camera Images, Show Grid, Show Patches, Show Points at Infinity |
| **Size** | Point Size (log₂, −3…+3) + reset, Infinity Point Size (1–16 px), Length Scale (0.001–100, log) |
| **Patches** | Patch Opacity, Patch Size, Patch Edge Cutoff |
| **Camera** | Field of View (10°–120°) + reset |
| **Advanced** | EDL Line Thickness (0.5–8 px), Frustum Size (0.05–5, log), Target Size (0.05–5, log), Target Fog (0.5–100, log) |
| **Debug** | Controls-help toggle, fps toggle, touchpad diagnostic counters |

The **count readout** for points at infinity is not a HUD widget: it belongs
with the point/image counts already painted top-left, so the scene stats line
reads `N points (M at infinity) | K images | F fps`, dropping the parenthetical
when the reconstruction has no `w = 0` points and the fps when the Debug toggle
is off. Since [scene-graph.md](scene-graph.md) the counts are summed
over the *visible* nodes and the line leads with `R reconstructions | ` once
more than one is contributing.

The Layers toggles are **master switches** across every loaded node: effective
visibility of a layer is the AND of the toggle here, the node's master eye, and
the node's group eye, both of the latter living in the Scene panel. Show Patches
is greyed only when *no* loaded node carries patch data, and the Patches section
appears when *any* of them does. The Grid stays global-only — it belongs to the
world, not to a reconstruction.

Three changes in behaviour from the menu:

- **Patch controls are hidden, not greyed.** In the menu the four patch
  controls were wrapped in `add_enabled_ui(has_patches, …)` and showed as dead
  widgets for the common `sift_files` reconstruction. In the HUD the whole
  Patches section is omitted when the reconstruction carries no patch frames +
  bitmaps. Show Patches stays in Layers, greyed, so the capability remains
  discoverable.
- **The always-on overlays become togglable.** Controls help and the fps
  readout get toggles in Debug, satisfying the intent the UX spec previously
  claimed for the controls help. Both default to on, so the move changes
  nothing until the user asks it to.
- **Points at infinity get a visibility toggle.** Hiding them is a
  `show_infinity` flag in `PointUniforms` that makes the vertex shader emit a
  clipped vertex, not a filtered upload: `instance_index` has to stay equal to
  the global `recon.points` index or picking, hover and selection break.

---

## Input arbitration

This is the part that needs care. `Viewer3D::show` allocates the **entire**
available rect with `Sense::click_and_drag` and then runs orbit / pan / zoom /
click handling across it. A HUD floating inside that rect means every input
path needs a rule.

**The HUD rect is authoritative.** Build the HUD first, capture the union of
its rects (gear when collapsed, gear + panel when expanded) into a
`hud_rect: Option<Rect>` on `Viewer3D`, and have every input path below consult
it. Building first is safe: an `egui::Area` is a separate layer and still
paints above the scene texture regardless of call order, so the HUD is
constructed before input handling but drawn after it.

| Path | Rule |
|------|------|
| **Scroll / zoom** | `handle_scroll` is gated on `platform::pointer_in_rect(ctx, rect)`. That helper is a raw geometric containment test — on Windows it reads the OS cursor position directly and knows nothing about egui layers. It must become `pointer_in_rect(rect) && !pointer_in_rect(hud_rect)`, computed in the same logical coordinate space. |
| **Gestures** | Same gate, same fix — `handle_gestures` and `handle_pinch` take `pointer_over` from the same helper. |
| **Drag / orbit** | An `Area` on a higher layer claims the pointer, leaving `response.dragged()` false. **Verified against egui 0.34, re-verified on 0.36**: `hit_test` keeps only the top-most layer covering the search area, so the painter's `WidgetRect` never reaches `hits.click` / `hits.drag`, and `dragged()`, `clicked()` and `hovered()` are all false. No `hud_rect` fallback is needed; `hud/tests.rs` pins the behaviour so a future egui change surfaces as a test failure rather than a viewport that orbits while a slider is dragged. |
| **Click / pick** | Excluded by the same mechanism as drag. A click that starts on the HUD must never deselect the current entity. `hover_pixel` follows for free: `Response::hover_pos` returns `None` when the response is not hovered, so the HUD never feeds the depth/pick readback either. |
| **Fly keys** | `handle_fly_keys` reads `key_down(W/A/S/D/…)` unconditionally. Any text-entry widget in the HUD (a `DragValue`) would fly the camera while being typed into. Gate all keyboard handling — `handle_fly_keys` *and* `handle_keyboard`'s Z / `,` / `.` / Home shortcuts — on `!ctx.egui_wants_keyboard_input()` (`wants_keyboard_input` is the deprecated spelling, from egui 0.34 on). That reports focus only for widgets that actually consume text, so clicking a HUD checkbox does not disarm the fly keys. |

There is no such guard anywhere in `sfm-explorer` today — the viewport has
never had to share its rect with a widget. These are new rules, not
adjustments to existing ones.

---

## State ownership

| State | Home | Rationale |
|-------|------|-----------|
| The setting values themselves | `AppState` (unchanged) | The renderer already reads them there; moving them would be churn for no gain |
| `hud_open`, `hud_rect` | `Viewer3D` | Per-viewport UI state, same place as `camera_view` and `hover_pixel` |
| Per-section collapsed flags | egui's own `CollapsingState` memory, under the stable ids from `hud::section_id` | Already exactly session-scoped, with per-section defaults via `load_with_default_open`; duplicating it into `Viewer3D` fields would only add a sync step. The explicit ids keep it addressable from outside the HUD |

Nothing is persisted across runs; the HUD opens collapsed each launch.

---

## Staging

Two commits, so the risky part lands on its own.

**Phase 1** — *done.* HUD shell with Layers, Size, Patches, and Camera
sections, plus the full input-arbitration rule set. The View menu is
**retained as a duplicate** through this phase, so a HUD regression cannot make
the controls unreachable.

**Phase 2** — *done.* Fold in Advanced and Debug (including the diagnostics
move and the overlay toggles) and delete the View menu outright, leaving File
as the only menu.

Phase 2 first repurposed View to dock-panel visibility rather than deleting it.
That was dropped after live use: the panels are permanent, so the menu existed
only to justify itself.

---

## Open questions

- **Does the HUD auto-hide during navigation?** Fading it out while a drag or
  fly key is active would keep the cinematic look, but flicker during the
  tune-adjust-look loop would be worse than a static panel. Leaning static.
- **Should Length Scale sit under Size or Advanced?** It is a scene-wide
  parameter that also drives frustum and target-indicator size, so it is
  arguably not a "size" control in the same sense as point size.
- **Docked variant?** If the HUD grows past ~8 sections, an
  `egui_dock`-native "Viewport Settings" tab would be a better home than a
  floating panel. Not worth it at the current control count.
- **Keyboard shortcuts for the visibility toggles?** Deliberately out of scope
  here — this spec covers pointer operation only. Direct bindings for the
  frequently-flipped toggles are a plausible follow-up once the HUD has been
  used enough to show which ones those actually are.

---

## Alternative considered: a toolbar row

A plain toolbar row above the viewport, exactly like
`dock.rs::show_overlay_toolbar`. It is consistent with the existing panel
precedent and sidesteps **every** rule in
[Input arbitration](#input-arbitration), because it is laid out beside the
painter rather than over it.

Rejected because it consumes vertical viewport space permanently, does not
scale to ~18 controls in one row, and breaks the full-bleed viewport that the
UX spec treats as a design principle. Worth revisiting if the input
arbitration proves harder than expected — it is the natural fallback.

---

## Implementation Status

All of it is built (`viewer_3d/hud.rs`, tests in `viewer_3d/hud/tests.rs`).

- [x] HUD shell: `Area`, viewport-relative anchoring, collapsed/expanded gear
- [x] `hud_rect` capture and the scroll/gesture exclusion
- [x] Drag/click layer arbitration (verified against egui 0.34 and again on
      0.36 — layering holds, no geometric fallback needed)
- [x] `egui_wants_keyboard_input` gate on fly keys and viewport shortcuts
- [x] Layers / Size / Patches / Camera sections
- [x] Advanced section (exposes `edl_line_thickness`, `frustum_size_multiplier`,
      `target_size_multiplier`, `target_fog_multiplier` for the first time)
- [x] Debug section: diagnostics move, controls-help and fps toggles
- [x] "Show points at infinity" toggle + count readout
      ([point-cloud-rendering.md](point-cloud-rendering.md#ui--shipped))
- [x] Remove the View menu outright, leaving File as the only menu

Still open, and deliberately so: everything under
[Open questions](#open-questions) — auto-hide during navigation, where Length
Scale belongs, a docked variant, and keyboard shortcuts for the visibility
toggles.
