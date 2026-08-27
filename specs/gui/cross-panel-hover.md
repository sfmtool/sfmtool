# Cross-Panel Hover Tracking

*Status: Implemented*

This document specifies the cross-panel hover feedback system for the sfmtool
GUI. It extends the selection model (click-to-select) with transient hover
highlighting that propagates across the 3D Viewer, Image Browser, and
Image Detail panels.

## Motivation

The GUI supports click-to-select for images (frustum click or thumbnail click)
and points (3D point click or feature click in detail panel). Selection
propagates across all panels via `AppState::selected_image` and
`AppState::selected_point`.

Cross-panel hover feedback adds lightweight, transient visual cues that help the
user understand spatial relationships before committing to a selection. When
hovering over a 3D point, the user can see which images observe it. When
hovering over a thumbnail, the corresponding frustum highlights in the 3D view.

## Design

### Hover State

Two fields on `AppState`:

```rust
pub hovered_image: Option<usize>,  // Image index under cursor
pub hovered_point: Option<usize>,  // Point3D index under cursor
```

These are updated every frame from the panel that currently has pointer focus.
Only one panel has pointer focus at a time, so there is no conflict.

Both fields are cleared in `load_file()` alongside `selected_image` and
`selected_point` when a new reconstruction is loaded.

### Hover Sources

| Panel | Produces `hovered_image` | Produces `hovered_point` |
|-------|--------------------------|--------------------------|
| 3D Viewer | From GPU pick buffer (frustum tag) | From GPU pick buffer (point tag) |
| Image Browser | From thumbnail hit-test | — |
| Image Detail | — | From nearest-feature hit-test (8px radius) |

When the pointer leaves a panel or moves to empty space, both hover fields
are cleared to `None`.

**Panel pointer ownership:** Each panel reports a `has_pointer` flag in its
response. When a panel has the pointer, it "owns" hover state and sets both
fields (the one it produces plus clearing the other). This prevents stale
hover from a previously-focused panel from persisting.

**Frame timing:** Hover state persists across frames (NOT cleared at frame
start). This is essential because uniform writing happens before the egui
panel pass — the uniforms use the previous frame's hover state, which is
set either by the readback (for 3D viewer) or by panel responses (for
browser/detail). The readback runs after the egui pass, setting hover for
the next frame's uniform writing.

**One-frame readback delay:** The GPU pick buffer readback is one frame behind.
When the pointer moves from the 3D viewer to another panel, the readback
result from the previous frame may still arrive. The readback-to-hover
conversion is gated on whether the 3D viewer currently has pointer focus
(check `viewer_3d.hover_pixel.is_some()` for the *current* frame).

### Hover Consumers

#### 3D Viewer

| Hover State | Visual Effect |
|-------------|---------------|
| `hovered_image` (from browser) | Brightness boost on hovered frustum wireframe (via uniform in fragment shader) |
| `hovered_point` (from detail) | Brightness boost on hovered 3D point (via uniform in fragment shader) |

The 3D viewer also renders hover info in the bottom-left overlay. This
is independent of the `hovered_*` state fields — the overlay shows info for
whatever entity is under the cursor in the 3D viewport itself.

#### Image Browser

| Hover State | Visual Effect |
|-------------|---------------|
| `hovered_point` (from 3D/detail) | Semi-transparent orange fill on thumbnails of images that observe the hovered point |
| `hovered_image` (from 3D viewer) | Soft highlight border on the hovered frustum's thumbnail |

The browser computes `hover_track_images` from `hovered_point` using the same
`compute_track_images()` helper used for selection. Displayed as a softer
version of the existing orange border (semi-transparent orange fill rather than
a border stroke).

The browser communicates hover back via `hovered_image: Option<usize>` on
`ImageBrowserResponse`, set each frame based on which thumbnail the pointer
is over.

#### Image Detail

| Hover State | Visual Effect |
|-------------|---------------|
| `hovered_point` (from 3D viewer) | White outline circle at the feature's location in the current image |

When `hovered_point` is `Some` and differs from `selected_point`, the
observation of that point in the currently displayed image's feature list is
found and drawn with a white outline circle (slightly larger than normal
feature circles).

The detail panel communicates hover back via `hovered_point: Option<usize>` on
`ImageDetailResponse`. The hit-test uses an 8px radius for nearest-feature
search.

### Performance Considerations

**Frustum hover highlight via uniform (NOT re-upload):** `hovered_image` is
passed as a uniform to the frustum shader, which boosts brightness in the
fragment shader. This is zero-cost per frame — no frustum buffer re-upload.

The `hovered_image_index: u32` field is in `FrustumUniforms`. The sentinel
value `0xFFFFFFFF` means "no hover". The `FrustumUniforms` struct is shared
with track ray uniforms (`update_track_ray_uniforms`), which always sets
`hovered_image_index` to `0xFFFFFFFF`.

**Point hover highlight via uniform:** `hovered_point_index: u32` in
`PointUniforms` follows the existing `selected_point_index` pattern. The
point fragment shader highlights the hovered point with a distinct color
(bright cyan) when it differs from the selected point. Sentinel: `0xFFFFFFFF`.

**Hover track images:** Computing track images requires scanning the track
observations array for a point index. This is O(track_length), typically
< 100 observations. Negligible at 60 fps.

**Detail feature lookup:** Finding a feature by point index in the current
image's feature list is O(N) where N is the number of features (typically
< 10,000). Negligible per frame.

### State Lifetime

Hover state is transient but persists across frames:
- Set every frame by the panel with pointer focus via `has_pointer` ownership
- Cleared when a different panel takes pointer focus (each panel clears the
  other panel's hover field when it has the pointer)
- Cleared on reconstruction load (`load_file()`)
- Never persisted or saved
- Does not affect GPU buffer uploads (frustums, points, track rays)
- Does not trigger SIFT cache loading or image loading

### Interaction with Selection

- Hover and selection are independent: hovering does not change selection
- When `hovered_point == selected_point`, the hover highlight is suppressed
  (the selection highlight is already stronger)
- When `hovered_image == selected_image`, the hover highlight is suppressed
- Click converts hover to selection (existing behavior, unchanged)

### Point Index Mapping

The point shader uses `instance_index` as the point3d_index, and the frustum
shader uses `frustum_index` from instance data. Both match the reconstruction
array indices because points and frustums are uploaded in order. The hover
uniform comparison relies on this assumption.
