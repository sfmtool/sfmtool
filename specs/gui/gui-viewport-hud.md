# Viewport HUD

**Status: proposed.** Nothing in this document is implemented. The controls it
describes currently live in the **View** menu — see
[gui-user-experience.md](gui-user-experience.md#ui-controls) for the accurate
description of what ships today.

This document specifies moving the 3D-viewport display controls out of the
menu bar and onto a heads-up display drawn inside the viewport itself.

---

## Motivation

Every control in the View menu (`app.rs`) is 3D-viewport state. None of it
affects the Image Browser, Image Detail, or Point Track panels. Two problems
follow from housing it in an app-global menu:

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

**Moves to the HUD** — all twelve current View-menu controls, plus the four
parameters that are plumbed to the GPU but have no widget at all
(`edl_line_thickness`, `frustum_size_multiplier`, `target_size_multiplier`,
`target_fog_multiplier`), plus the two unbuilt point-cloud controls
(["Show points at infinity" toggle and count
readout](gui-point-cloud-rendering.md#remaining-ui-work)).

**Stays in the menu bar** — File (Open / Load Demo Data / Quit). Once the
display controls leave, "View" is repurposed to dock-panel visibility — which
tabs are shown — which is a genuinely app-global concern and leaves the menu
meaningful rather than deleted.

**Out of scope** — the Image Detail and Image Browser toolbars. They already
follow the panel-owns-its-controls rule and are not touched.

---

## Layout

The HUD is collapsed by default to a single gear glyph, so the default
experience remains a full-bleed viewport (design principle #4, *Dark,
Cinematic Aesthetic*).

```
┌────────────────────────────────────────────────────────────┐
│ 12,043 pts | 17 imgs | 60 fps   Pos: [1.2, -3.4, 0.8]   ⚙ │  collapsed
│                                                            │
│                                                            │
│  ⟂                          Point3D #841 | depth: 2.13     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ 12,043 pts | 17 imgs | 60 fps   Pos: [1.2, -3.4, 0.8]   ✕ │  expanded
│                                        ┌─────────────────┐ │
│                                        │ ▾ Layers        │ │
│                                        │   ☑ Points      │ │
│                                        │   ☑ Cameras     │ │
│                                        │   ☑ Grid        │ │
│                                        │   ☑ Patches     │ │
│                                        │ ▾ Size          │ │
│                                        │   Points  ──●── │ │
│                                        │   ∞ (px)  ─●─── │ │
│                                        │   Scene   ──●── │ │
│                                        │ ▸ Patches       │ │
│                                        │ ▾ Camera        │ │
│                                        │   FOV 45°  ─●── │ │
│                                        │   [Reset view]  │ │
│                                        │ ▸ Advanced      │ │
│                                        │ ▸ Debug         │ │
│                                        └─────────────────┘ │
│  ⟂                          Point3D #841 | depth: 2.13     │
└────────────────────────────────────────────────────────────┘
```

**Anchor**: top-right, inset from the viewport edge. The existing overlays
occupy all four corners plus the top centre
([gui-user-experience.md](gui-user-experience.md#information-overlays)), and
top-right is the cheapest to vacate — it holds only the touchpad diagnostics,
which move into the HUD's Debug section (they are developer instrumentation and
should not be permanently burned into the viewport anyway).

**Position is relative to the viewport rect, recomputed every frame.** The HUD
lives in a dock tab that the user can resize, re-dock, or tab away from. An
`Area` at a fixed screen position would detach from the panel; the fixed
position must be derived from the `rect` returned by `allocate_painter` on
each frame.

**Expanded width** is fixed (~220 pt) so slider tracks do not jitter as
labels change. If the viewport is too small to show the expanded HUD without
covering more than about a third of it, the HUD stays collapsed and the gear
is the only affordance.

---

## Sections

Collapsible (`egui::CollapsingHeader`), with open/closed state remembered for
the session. Defaults: Layers, Size, and Camera open; Patches, Advanced, and
Debug closed.

| Section | Contents |
|---------|----------|
| **Layers** | Show Points, Show Camera Images, Show Grid, Show Patches, Show Points at Infinity |
| **Size** | Point Size (log₂, −3…+3) + reset, Infinity Point Size (1–16 px), Length Scale (0.001–100, log) |
| **Patches** | Patch Opacity, Patch Size, Patch Edge Cutoff |
| **Camera** | Field of View (10°–120°) + reset |
| **Advanced** | EDL Line Thickness, Frustum Size, Target Size, Target Fog |
| **Debug** | Touchpad diagnostic counters, controls-help toggle, fps toggle |

Two changes in behaviour from the menu:

- **Patch controls are hidden, not greyed.** Today the four patch controls are
  wrapped in `add_enabled_ui(has_patches, …)` and show as dead widgets for the
  common `sift_files` reconstruction. In the HUD the whole Patches section is
  omitted when the reconstruction carries no patch frames + bitmaps. Show
  Patches stays in Layers, greyed, so the capability remains discoverable.
- **The always-on overlays become togglable.** Controls help and the fps
  readout get toggles in Debug, satisfying the intent the UX spec previously
  claimed for the controls help.

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
| **Drag / orbit** | An `Area` on a higher layer should claim hover, leaving `response.dragged()` false. This is the one rule that depends on egui internals rather than our own geometry, so it must be **verified against egui 0.34 rather than assumed**; if it does not hold, fall back to the same `hud_rect` exclusion. |
| **Click / pick** | Excluded by the same mechanism as drag. A click that starts on the HUD must never deselect the current entity. |
| **Fly keys** | `handle_fly_keys` reads `key_down(W/A/S/D/…)` unconditionally. Any text-entry widget in the HUD (a `DragValue`) would fly the camera while being typed into. Gate all keyboard handling on `!ctx.wants_keyboard_input()`. |

There is no such guard anywhere in `sfm-explorer` today — the viewport has
never had to share its rect with a widget. These are new rules, not
adjustments to existing ones.

---

## State ownership

| State | Home | Rationale |
|-------|------|-----------|
| The setting values themselves | `AppState` (unchanged) | The renderer already reads them there; moving them would be churn for no gain |
| `hud_open`, per-section collapsed flags, `hud_rect` | `Viewer3D` | Per-viewport UI state, same place as `camera_view` and `hover_pixel` |

Nothing is persisted across runs; the HUD opens collapsed each launch.

---

## Staging

Two commits, so the risky part lands on its own.

**Phase 1** — HUD shell with Layers, Size, Patches, and Camera sections, plus
the full input-arbitration rule set. The View menu is **retained as a
duplicate** through this phase, so a HUD regression cannot make the controls
unreachable.

**Phase 2** — fold in Advanced and Debug (including the diagnostics move and
the overlay toggles), delete the display controls from the View menu, and
repurpose View to dock-panel visibility.

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

None of this is built.

- [ ] HUD shell: `Area`, viewport-relative anchoring, collapsed/expanded gear
- [ ] `hud_rect` capture and the scroll/gesture exclusion
- [ ] Drag/click layer arbitration (verify egui 0.34 behaviour first)
- [ ] `wants_keyboard_input` gate on fly keys
- [ ] Layers / Size / Patches / Camera sections
- [ ] Advanced section (exposes `edl_line_thickness`, `frustum_size_multiplier`,
      `target_size_multiplier`, `target_fog_multiplier` for the first time)
- [ ] Debug section: diagnostics move, controls-help and fps toggles
- [ ] "Show points at infinity" toggle + count readout
      ([gui-point-cloud-rendering.md](gui-point-cloud-rendering.md#remaining-ui-work))
- [ ] Remove display controls from the View menu; repurpose it to panel visibility
