# Multi-Panel GUI with Image Browser and Detail Pane

## Overview

The viewer's window is a dockable, tabbed workspace rather than a single
full-window 3D view: panels can be resized, re-tabbed and dragged into new
splits, and they stay in step with one another through a shared selection.
Four panel types:

1. **3D Viewer** — the existing viewport (point cloud, frustums, navigation)
2. **Image Browser** — bottom strip of 128×128 thumbnails for browsing the image sequence
3. **Image Detail** — full-resolution image view for the selected camera
4. **Point Track Detail** — per-observation diagnostics for the selected 3D point
   (see `specs/gui/point-track-detail.md`); shares the right-side tab
   region with Image Detail.

A fifth panel, **Scene**, was added by
[scene-graph.md](scene-graph.md); it takes a narrow left split of the
root, and everything below describes the arrangement to its right.

## Default Layout

```
┌───────┬──────────────────────────┬──────────────┐
│  File                            │  (menu bar)  │
├───────┼──────────────────────────┼──────────────┤
│       │                          │              │
│       │                          │    Image     │
│ Scene │        3D Viewer         │    Detail    │
│       │                          │              │
│       │                          │              │
│       ├──────────────────────────┴──────────────┤
│       │ [Image Browser] [Action Log]            │
│       │ ◀ [img01] [img02] [img03] [img04] ... ▶ │
└───────┴─────────────────────────────────────────┘
```

- **Scene**: left, ~18% width. The tree of loaded reconstructions.
- **3D Viewer**: top-left of the rest, ~2/3 of its width. Point cloud,
  frustums, navigation.
- **Image Detail**: top-right, ~1/3 width, sharing a tab group with Point Track
  and Camera Intrinsics. Full-resolution image of the selected camera.
- **Image Browser**: bottom strip, full width, ~20% of the height.
  Horizontally-scrollable strip of 128×128 thumbnails. It shares its tab group
  with the **Action Log** ([action-log.md](action-log.md)) and is the active
  member, so the viewer opens on the strip.

Since we use `egui_dock`, the user can re-dock any panel anywhere (float, reorder tabs,
resize splits, etc.), and close any of them — the **Panels** menu is what
re-opens one and what puts this grid back
([panel-layout.md](panel-layout.md)).

## Panel Interaction Model

### Image Selection

All four panels share `AppState::selected_image` as the central image selection state:

```
   Image Browser ──click──▶ selected_image ◀──click── 3D Viewer (frustum pick)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
             Image Browser   3D Viewer    Image Detail
             (cyan border)  (cyan frustum) (loads full-res)
```

**Selection flow:**
- **Image Browser → others**: Click a thumbnail to select. The 3D viewer highlights the
  corresponding frustum (cyan). The detail pane loads and displays the full-resolution image.
- **3D Viewer → others**: Click a frustum to select (existing GPU pick behavior). The image
  browser highlights the corresponding thumbnail and scrolls it into view. The detail pane
  loads the full-resolution image.
- **Image Detail → others**: The detail pane is display-only (no selection input — it
  shows whatever is selected). `,`/`.` keys on the 3D viewport step the selection
  back/forward (wrapping at the ends) even when not in camera view mode.
- **Deselect**: Clicking background in the 3D viewer clears `selected_image` and the
  detail pane shows "No image selected." Re-clicking an already-selected thumbnail keeps
  it selected (no toggle-off) — thumbnail clicks always set the selection to that image.

**What changes when `selected_image` changes:**
- Image Browser: cyan highlight border moves to the new thumbnail
- 3D Viewer: frustum re-upload with new selection color (already implemented via
  `prev_selected_image` change detection)
- Image Detail: loads the new full-resolution image from disk (same path as `upload_bg_image`
  in camera view mode, but rendered to an egui texture instead of a wgpu background pass)

### 3D Point Selection

All four panels also share `AppState::selected_point: Option<usize>` for 3D point selection.
A selected 3D point implies its track — the set of `(image_index, feature_index)` observations
from `SfmrReconstruction::tracks`.

**Data model**: Tracks are stored sorted by `(point_index, image_index)` in
`SfmrReconstruction::tracks`, with `observation_counts[i]` giving the number of observations
for point `i`. To find observations for a point, compute the offset from the prefix sum of
`observation_counts` and read `observation_counts[point_idx]` entries.

**Derived state**: When `selected_point` changes, compute the set of track images:
```rust
/// Set of image indices that participate in the selected point's track.
fn track_images(recon: &SfmrReconstruction, point_index: usize) -> HashSet<usize>
```
This derived set drives the cross-panel highlighting described below.

**Cross-panel effects of point selection:**

```
                          selected_point
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                   ▼
       Image Browser       3D Viewer          Image Detail
       (highlight track    (highlight point    (highlight SIFT
        images)             + track frustums)   feature keypoint)
```

- **3D Viewer**:
  - The selected point is outlined in a distinct highlight color (e.g., yellow or magenta)
    to distinguish it from the existing cyan frustum selection color, while preserving
    the original point color.
  - Frustums for images in the track set are highlighted with a secondary color (e.g.,
    a subtler tint or outline) to indicate they observe the selected point. This is
    distinct from the primary `selected_image` cyan highlight.
- **Image Browser**:
  - Thumbnails for images in the track set receive a secondary highlight (e.g., a
    colored dot, border tint, or subtle overlay) distinct from the cyan `selected_image`
    border. This shows which images observe the selected 3D point.
- **Image Detail**:
  - If the currently `selected_image` is in the track set, the SIFT feature keypoint
    corresponding to the selected point's observation is highlighted on the image. The
    feature index comes from `TrackObservation::feature_index` for the observation where
    `image_index` matches `selected_image`. The keypoint location (x, y, scale,
    orientation) is read from the `.sift` file for that image.
  - If the selected image is not in the track set, no feature highlighting is shown.

**Selection input (3D Viewer only)**:
- Point picking in the 3D viewer uses the existing GPU pick buffer (or a depth-based
  approach). Clicking a point sets `selected_point` to that point.
- Clicking empty space clears `selected_point`.
- Point selection and image selection are independent — both can be active simultaneously.

### 3D Point Hover

A single hovered 3D point provides live feedback as the mouse moves over the point cloud.
This complements the persistent `selected_point` with a transient, softer highlight.

**State**: `AppState::hovered_point: Option<usize>`. Updated every frame from the 3D
viewer's existing GPU pick buffer (`SceneRenderer::hover_pick_id`). Currently
`hover_pick_id` lives only in `SceneRenderer` and is passed as a parameter to the
status text overlay in `viewer_3d/overlay.rs`. To enable cross-panel hover, the
resolved point index is promoted to `AppState`:

```rust
// In AppState:
/// Transiently hovered 3D point index from the 3D viewer's pick buffer.
/// Updated every frame; None when the cursor is not over a point.
pub hovered_point: Option<usize>,
```

Each frame, after `SceneRenderer::read_readback_result()`, app.rs extracts the point index:
```rust
let hover_pick_id = self.scene_renderer.hover_pick_id();
let tag = hover_pick_id & PICK_TAG_MASK;
let index = (hover_pick_id & PICK_INDEX_MASK) as usize;
state.hovered_point = if tag == PICK_TAG_POINT { Some(index) } else { None };
```

**Cross-panel effects**: The hover point drives the same track-based highlighting as
point selection, but with a visually softer treatment:

```
                          hovered_point
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                   ▼
       Image Browser       3D Viewer          Image Detail
       (soft highlight     (soft highlight    (soft highlight
        track images)       hovered point)     feature keypoint)
```

- **3D Viewer**:
  - The hovered point is rendered with a soft highlight (e.g., brighter or with a subtle
    glow/outline) distinct from both the normal color and the selection highlight.
  - The existing status text ("Point3D #N | depth: X.XXXX") remains in the bottom-left.
  - Frustums for images in the hover point's track are not highlighted (too noisy when
    moving the mouse rapidly). Only the selected point's track highlights frustums.

- **Image Browser**:
  - Thumbnails for images in the hover point's track receive a soft highlight (e.g., a
    dimmed border, subtle background tint, or small indicator dot). This should be
    visually lighter than the selection highlight — enough to notice but not distracting
    as the mouse moves.
  - The highlight updates every frame as `hovered_point` changes. Since the track lookup
    is O(observation_count) for that point, this is cheap (typical tracks have 2–20
    observations).

- **Image Detail**:
  - If the currently `selected_image` is in the hover point's track, the corresponding
    SIFT feature keypoint is highlighted with a soft indicator (e.g., a thin circle
    outline or a translucent highlight ring) distinct from the stronger selection
    highlight.
  - This gives immediate feedback: "this feature on the current image corresponds to the
    point you're hovering over in 3D."
  - If the selected image is not in the hover point's track, no hover highlight is shown
    on the detail panel.

**Visual hierarchy**: The three highlighting tiers from strongest to softest:

| Tier | Source | 3D point color | Frustum highlight | Browser highlight | Detail feature |
|------|--------|---------------|-------------------|-------------------|----------------|
| **Selection** | `selected_point` | Bold (yellow/magenta) | Yes (secondary color) | Strong border/tint | Bold color/ring |
| **Hover** | `hovered_point` | Soft (brighter/glow) | No | Soft border/dot | Thin outline |
| **Normal** | — | Original point color | Normal | No highlight | Normal (if overlay on) |

When hover and selection overlap (hovering the selected point, or hovering a point whose
track shares images with the selected point's track), both are visible. The hover adds an
additional visual cue on top of the selection styling — e.g., a second outer ring, a
brightness boost, or a size pulse — so the user always sees which specific element the
cursor is over.

**Performance**: Track lookup for a single hover point is trivial — prefix-sum into the
tracks array to find the observation range, read a handful of `(image_index,
feature_index)` pairs. This runs every frame but costs negligible time.

## Architecture

The `egui_dock` layout lives in
[dock.rs](../../crates/sfm-explorer/src/dock.rs) and the three non-viewport
panels beside it —
[image_browser.rs](../../crates/sfm-explorer/src/image_browser.rs),
[image_detail/](../../crates/sfm-explorer/src/image_detail),
[point_track_detail/](../../crates/sfm-explorer/src/point_track_detail) — while
the cross-panel selection state they share is `AppState` in
[state.rs](../../crates/sfm-explorer/src/state.rs).

### Tab Model

```rust
// dock.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    SceneGraph,
    Viewer3D,
    ImageBrowser,
    ImageDetail,
    PointTrackDetail,
    IntrinsicsDetail,
    ActionLog,
}
```

The three panels this spec is about are `Viewer3D`, `ImageBrowser` and
`ImageDetail`. The rest are the tabs later panels added to the same enum, each
specified on its own:

| Tab | Title | Spec |
|---|---|---|
| `PointTrackDetail` | Point Track Detail | [point-track-detail.md](point-track-detail.md) |
| `SceneGraph` | Scene | [scene-graph.md](scene-graph.md) |
| `IntrinsicsDetail` | Camera Intrinsics | [camera-intrinsics.md](camera-intrinsics.md) |
| `ActionLog` | Action Log | [action-log.md](action-log.md) |

`IntrinsicsDetail` shares the Image Detail / Point Track tab group as its third
and non-active member; `ActionLog` shares the Image Browser's as its second and
non-active member.

### TabContext and TabViewer

```rust
struct TabContext<'a> {
    state: &'a mut AppState,
    viewer_3d: &'a mut Viewer3D,
    image_browser: &'a mut ImageBrowser,
    image_detail: &'a mut ImageDetail,
    // ... scene_texture_id, gesture_events, etc.
}

impl egui_dock::TabViewer for TabContext<'_> {
    type Tab = Tab;
    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText { ... }
    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) { ... }
}
```

### DockState Initialization

The default grid is `Layout::default()`, in
`crates/sfm-explorer/src/layout.rs`, and the viewer starts on
`Layout::default().to_dock()` — the same value Panels ▸ Reset Layout restores.
Written as a tree of panel names rather than as a sequence of splits, it is:

```rust
Split { split: LeftRight, fraction: 0.18,
    first:  Leaf { tabs: [SceneGraph] },
    second: Split { split: TopBottom, fraction: 0.8,
        first:  Split { split: LeftRight, fraction: 0.67,
            first:  Leaf { tabs: [Viewer3D] },
            second: Leaf { tabs: [ImageDetail, PointTrackDetail, IntrinsicsDetail] } },
        second: Leaf { tabs: [ImageBrowser, ActionLog] } } }
```

`fraction` is the share of the **first** child in layout order.
(`egui_dock` 0.19's doc comment says "the old node" for both directions of a
split, which is true only of Right and Below; 0.21 gave the Scene panel four
fifths of the window when it was read the other way. `Layout::to_dock` only
ever splits Right and Below for exactly that reason.) A leaf opens on its
**first** tab, which is what puts Image Detail and the Image Browser in front
of the tabs they share a node with.

The panels are **closeable**, so the grid above is the layout they start in
rather than the only one they have. [panel-layout.md](panel-layout.md) carries
the Panels menu, the home position a re-opened panel lands at, and the layout
file the viewer saves and loads.

### Integration in app.rs

The dock fills the central panel, under the menu bar:

```rust
egui::CentralPanel::default().show(ctx, |ui| {
    DockArea::new(&mut dock).show_inside(ui, &mut tab_context);
});
```

`dock` is taken out of `AppState` for the duration of the call and put back
straight after, because `TabContext` holds the state mutably at the same time —
see [panel-layout.md](panel-layout.md) § "Implementation notes".

## Panel Specifications

### Image Browser

A horizontally-scrollable strip of 128×128 thumbnails.

**Thumbnails**: The image browser displays the same 128×128 thumbnails used on the 3D
viewer's frustum far planes. These are loaded from disk via the `image` crate, resized to
128×128, and cached as egui textures (separate from the GPU texture atlas in
`SceneRenderer`, since egui has its own texture management).

**Aspect ratio**: The 128×128 thumbnails are square, but the source images are typically
not. The image browser must display thumbnails at the correct aspect ratio. The aspect
ratio is obtained from the camera intrinsics:
`recon.cameras[image.camera_index as usize]` → `CameraIntrinsics { width, height }`.
The 128×128 pixel data is drawn at the correct aspect ratio to fit within the strip
height (e.g., a 640×360 source produces a 128×72 drawn region). This is the same
approach the 3D viewer uses for frustum image quads — the square texture is UV-mapped
to the correct proportions.

**Layout**: Manual offset-based horizontal panning (not `ScrollArea`, to support
DirectManipulation gesture-driven scrolling on Windows). Each thumbnail is rendered
via `egui::Image` sized to the correct aspect ratio derived from camera intrinsics.
When the panel is resized (changing thumbnail height), the scroll offset is rescaled
so that the image at the center of the viewport stays anchored in place.

**Selection**:
- Click to select. Clicking an already-selected thumbnail keeps it selected
  (no toggle). Deselection happens by clicking background in the 3D viewer.
- Selected thumbnail gets a cyan border (matching the 3D viewer's selection color).
- When `selected_image` changes externally (e.g., frustum click in 3D viewer), the
  browser auto-scrolls to keep the selected thumbnail visible.

**Thumbnail loading**:
- Cache: `HashMap<usize, egui::TextureHandle>` in `ImageBrowser`.
- Lazy: load a few thumbnails per frame to avoid stalling. Prioritize visible thumbnails.
- Path: `reconstruction.workspace_dir.join(&img.name)`, resized to 128×128 with the
  `image` crate (same as `SceneRenderer::upload_thumbnails`).

**Label**: Image index or filename shown below each thumbnail.

### Image Detail

Full-resolution image display for the selected camera, with SIFT feature overlays.

#### Base Image

**Image loading**: When `selected_image` changes, load the full-resolution image from
`workspace_dir.join(&img.name)` into an egui texture. This is the same image path used by
`SceneRenderer::upload_bg_image` for camera view mode, but rendered as an egui `Image`
widget instead of a wgpu background pass.

**Display**: The image is shown fitted to the panel dimensions (maintaining aspect ratio)
using `egui::Image` with `fit_to_exact_size` or `max_size`. Pan/zoom within the detail
pane is a future enhancement.

**Empty state**: When no image is selected, show "No image selected" centered in the panel.

**Cache**: Store a single `Option<(usize, egui::TextureHandle)>` — the currently loaded
image index and its texture. Only reload when `selected_image` changes.

#### Feature Overlays

The Image Detail panel supports drawing SIFT feature overlays on top of the image. These
correspond to the CLI visualization commands (`sfm sift --draw`, `sfm heatmap`) but
rendered interactively via egui rather than baked into an output image.

**Overlay modes** (selectable via a dropdown or toolbar at the top of the panel):

| Mode | What it shows | Corresponds to |
|------|--------------|----------------|
| **None** | Clean image, no overlays | — |
| **Features** | SIFT keypoint ellipses + center dots | `sfm sift --draw` |
| **Reproj Error** | Colored circles by reprojection error | `sfm heatmap --metric reproj` |
| **Track Length** | Colored circles by observation count | `sfm heatmap --metric tracks` |
| **Max Track Angle** | Colored circles by max pairwise ray angle (triangulation angle) | `sfm heatmap --metric angle` |
| **Depth Reliability** | Colored circles by inverse-depth z-score (low ⇒ near-infinity) | `sfm analyze --depth-reliability` |
| **Condition Number** | Colored circles by `log10` of the normal-matrix condition number | the same diagnostic |

> _An **intrinsics layer** — principal point, angular axes, iso-angle rings,
> distortion field — is drawn on this panel **independently of the mode above**,
> composing with any of them (including `None`). It is not an `OverlayMode`
> variant: the enum, the filters below, and the exclusivity among the seven
> modes are all unchanged. Its own state lives in a sibling
> `IntrinsicsDisplaySettings`, and it draws beneath the feature layers in a
> haloed near-white so it survives an arbitrary colormap underneath — except its
> principal-point marker, which draws last, over everything. It contributes one
> checkbox and one gear to the toolbar row, and `I` toggles it while the pointer
> is over the panel (beside the panel's existing `Z`). It also contributes text
> to **this panel's one hover tooltip**, appended below a painted rule to
> whatever the feature layer produced; with the layer off that tooltip is
> unchanged. See [camera-intrinsics.md](camera-intrinsics.md) § "Image
> Detail: the Intrinsics overlay layer"._

#### Feature Filtering

Features in `.sift` files are sorted by decreasing size (largest first). The Image Detail
panel provides scene-level filtering controls to limit which features are displayed. Both
filters produce a prefix of the sorted array, so they compose naturally.

**Scene-level settings** (in `AppState`):

```rust
struct FeatureDisplaySettings {
    /// Maximum number of features to display per image. None = unlimited.
    /// Since features are sorted by decreasing size, this shows the N largest.
    max_features: Option<usize>,
    /// Minimum feature size threshold (in pixels). None = no threshold.
    /// Feature size = average of column norms of the 2×2 affine shape matrix
    /// (same as `_sift_utils.py:feature_size()`).
    min_feature_size: Option<f32>,
    /// Maximum feature size threshold (in pixels). None = no threshold.
    /// Features larger than this are excluded.
    max_feature_size: Option<f32>,
    /// Drag value for the min size slider (persists when checkbox is unchecked).
    min_feature_size_value: f32,
    /// Drag value for the max size slider (persists when checkbox is unchecked).
    max_feature_size_value: f32,
    /// If true, only show features that participate in a track (have an
    /// associated 3D point). Equivalent to CLI `--filter-sfm`.
    tracked_only: bool,
}
```

**Defaults**: `overlay_mode: Features`, `max_features: None` (all), `tracked_only: true`,
size filters disabled with persisted slider values (min: 0.0, max: 50.0).

These settings, together with the intrinsics layer's, are also readable and
writable over MCP as one `image_detail_display` document
([mcp-server.md](mcp-server.md) § "`get_image_detail_display` /
`set_image_detail_display`"), and every change to them — from the toolbar or
from the tool — records a `Display` entry in the Action Log. The two size
options are one `feature_size_px` object on the wire, because the toolbar's
single checkbox re-derives both from the persisted slider values every frame.

**Effective feature set per image**: The actual number of features displayed varies per
image. Filtering is applied in order:

1. **max_features** — Truncates to the N largest features (prefix of sorted array).
2. **min_feature_size** — Scans backward from the truncation point to exclude features
   smaller than the threshold (produces a shorter prefix).
3. **max_feature_size** — Excludes individual features larger than the threshold from
   within the prefix (per-feature check, since large features are at the start).
4. **tracked_only** — Excludes features without an associated 3D point.

**I/O optimization**: `sift_format::read_sift_partial(path, count)` already skips reading
the tail of the arrays at the file level, so `max_features` saves real I/O for large
`.sift` files. The size threshold requires reading affine shapes to check, but since
features are sorted, only `max_features` entries need to be read before truncating further.

**UI controls**: A toolbar row at the top of the Image Detail panel, alongside the overlay
mode selector:

```
[Overlay: Features ▾]  [Max: All ▾]  [☐ Min/max size: 0.0  50.0]  [☑ Tracked only]
```

- **Max features**: dropdown with common presets (100, 500, 1000, 5000, All). Changing
  this may trigger a re-read of SIFT data if the new value is larger than what was
  previously loaded (since `read_sift_partial` was used with the old count).
- **Min/max size**: single checkbox + two drag values (min: 0.0–1000 px, max: 0.5–1000 px). The drag values
  are always visible and editable; the checkbox controls whether both size filters are
  applied. Values persist when unchecked so users can toggle without losing their settings.
- **Tracked only**: checkbox. When enabled, only features with a track observation are
  drawn. In Features mode, untracked features are hidden. In heatmap modes, untracked
  features are always hidden (they have no metric). This is independent of the size
  filters — applied after the size-based prefix truncation.

#### Data Loading

When an overlay mode is active and `selected_image` changes, load:

1. **SIFT data** for the image via `sift_format::read_sift_partial()`:
   - `positions: Array2<f32>` (N×2) — keypoint (x, y) locations
   - `affine_shapes: Array3<f32>` (N×2×2) — oriented affine shape matrices
   - Read count: `max_features.unwrap_or(total_feature_count)`
   - Descriptors are not needed for visualization and can be discarded.
   - Cache per image index: `HashMap<usize, (Vec<[f32; 2]>, Vec<[[f32; 2]; 2]>)>`
     keyed by `(image_index, read_count)`. Invalidate if `max_features` increases
     beyond the cached read count.

2. **Track mapping** for the image from `SfmrReconstruction::tracks`:
   - Build `image_idx → Vec<(feature_idx, point_idx)>` mapping (same approach as the
     heatmap command, lines 161-174 of `_commands/heatmap.py`)
   - Only features that participate in a track have associated 3D points and metrics
   - Features not in any track are untracked (drawn differently or omitted depending on mode)

3. **Per-point metrics** looked up by `point_idx`:
   - Reprojection error: `SfmrReconstruction::points[point_idx].error`
   - Track length: `SfmrReconstruction::observation_counts[point_idx]`
   - Max track angle (triangulation angle): max pairwise angle (degrees) between
     world-space rays from observing cameras to the 3D point. Computed on
     demand when the Max Track Angle overlay is active, cached per-feature in
     the overlay state for the duration of the current mode.

**Drawing (egui painter)**:

All feature overlays are drawn using `egui::Painter` on top of the image widget. Feature
positions in image pixel coordinates are transformed to panel coordinates using the
image-to-panel transform (accounting for the fitted image size and offset within the panel).

- **Features mode**: For each SIFT feature:
  - Draw an oriented ellipse (green stroke) from the 2×2 affine shape matrix. Decompose
    via SVD to get semi-axis lengths and rotation angle — same math as
    `sift_file.py:draw_sift_features()` (lines 830-860).
  - Draw a small filled circle (red) at the center position.
  - Only draw features within the visible panel region for performance.
  - The "Tracked only" checkbox controls whether untracked features are shown.

- **Heatmap modes** (Reproj Error / Track Length / Max Track Angle): For each tracked feature:
  - Draw a filled circle at the feature position.
  - Color is mapped from the metric value using the same colormap definitions as
    `visualization/_colormap.py` for error and tracks. Max Track Angle uses a
    red→yellow→green gradient (low angle = weak triangulation = red,
    high = well-triangulated = green).
  - Circle radius is a fixed size in image pixels (default ~5px, configurable via the
    overlay toolbar).
  - Show a small colorbar legend in the corner of the panel with min/max range labels.
  - Untracked features are not drawn in heatmap modes (they have no associated metric).

**Interaction with 3D point selection**:

When `selected_point` is set and the selected image participates in that
point's track, the corresponding feature keypoint is highlighted with an additional
visual indicator (regardless of the current overlay mode):

- In **None** mode: draw just the selected feature's ellipse (or circle) so the user
  can see which feature on this image corresponds to the selected 3D point.
- In **Features** mode: the selected feature gets a distinct color (e.g., yellow or
  magenta) instead of the default green, making it stand out.
- In **Heatmap** modes: the selected feature gets an additional outline ring or border
  to distinguish it from the surrounding heatmap circles.

The feature index comes from `TrackObservation::feature_index` for the observation where
`image_index == selected_image` and `point_index == selected_point`.

**Performance considerations**:
- Images can have 10K+ SIFT features. The `max_features` setting (default 500) is the
  primary performance control — it limits both I/O (via `read_sift_partial`) and drawing.
  At 500 features, egui painter handles ellipses comfortably every frame.
- SIFT data loading is done lazily when the overlay mode is first activated or when the
  selected image changes. Descriptors (N×128 u8) are not needed and can be discarded.
- The track mapping is built once when the reconstruction loads and cached in `AppState`
  or `ImageDetail`.

### 3D Viewer

No changes to the 3D viewer's rendering or interaction. Its existing behavior already
supports the selection model:

- Frustum click sets/clears `selected_image` via GPU pick buffer.
- Frustum re-upload on `selected_image` change applies cyan highlight.
- Camera view mode (Z key) is independent — it sets the viewport pose and loads a wgpu
  background texture. The detail pane's full-res image display is separate and doesn't
  conflict.

## Relationship to Camera View Mode

The 3D viewer's camera view mode (Z key with a frustum selected) and the image detail pane
both display a full-resolution image, but they serve different purposes and don't conflict:

| Aspect | Camera View Mode | Image Detail Pane |
|--------|-----------------|-------------------|
| **Purpose** | Navigate the 3D scene from a camera's perspective | Inspect the image itself |
| **Trigger** | Z key with frustum selected | Automatic on selection |
| **Rendering** | wgpu background pass behind point cloud | egui `Image` widget in its own panel |
| **Navigation** | Overrides viewport pose + FOV | No 3D navigation effect |
| **Exits when** | Any navigation input (orbit, pan, zoom) | Never (always shows selected) |

They coexist naturally: you can be in camera view mode in the 3D viewer while the detail
pane also shows the same image in its own panel.

### Entering Camera View Mode

Camera view mode can be entered in three ways:

- **Z key** (existing): with a frustum selected, press Z to view through that camera.
- **Double-click a frustum** in the 3D viewer: selects the image and immediately enters
  camera view mode.
- **Double-click a thumbnail** in the image browser: selects the image and enters camera
  view mode in the 3D viewer.

All three paths set `selected_image` and activate `CameraViewMode` on the `Viewer3D`. The
3D viewer then snaps to the camera's pose with best-fit FOV and loads the full-resolution
background image (existing behavior).

## Image Detail: 2D pan and zoom navigation

The image detail panel supports pan and zoom to inspect the full-resolution
image, similar to how the 3D viewer navigates the point cloud but in 2D.

**State** (in `ImageDetail`):

- `pan: egui::Vec2` — offset of image center from panel center, in panel pixels
- `zoom: f32` — zoom level (1.0 = fit to panel, max 32×)
- `last_display_size: Option<egui::Vec2>` — the displayed image extent `pan` was
  measured against, recorded at the end of each frame that drew an image and
  cleared by a view reset. See "View persistence" below.

**Navigation controls** (sign conventions match the 3D viewer):

| Action | Input | Behavior |
|--------|-------|----------|
| Pan | Left/middle button drag | Translate the image (grab-and-drag) |
| Pan | Trackpad two-finger scroll | Translate the image (push convention) |
| Pan | DM Pan gesture | Translate the image (push convention) |
| Zoom | Scroll wheel | Zoom toward cursor position |
| Zoom | Right button drag (vertical) | Zoom toward cursor position |
| Zoom | Pinch | Zoom toward cursor position |
| Zoom | Ctrl + two-finger scroll | Zoom toward cursor position |
| Zoom | Ctrl + DM Pan gesture | Zoom |
| Zoom | DM Zoom gesture | Zoom toward cursor position |
| Fit | Z key / Double-click | Reset pan and zoom to fit image in panel |

- **Sign conventions**: Mouse drag uses "grab the content" convention (content
  follows cursor). DM gestures and trackpad scroll use "push/scroll viewport"
  convention (opposite direction), matching the 3D viewer's shift+scroll→pan
  and shift+DM→pan mappings.
- **Zoom-to-cursor**: Zoom is anchored at the cursor position so the point
  under the cursor stays fixed: `pan = pan * ratio + cursor_rel * (1 - ratio)`.
- **Zoom limits**: Minimum = 1.0 (fit-to-panel). Maximum = 32×.
- **Pan limits**: Clamped so the image overlaps the panel by at least 50px.
- **View persistence**: The view outlives the image it was set on — see below.

**Rendering** (`image_detail/`):
- `base_scale = min(panel_w / tex_w, panel_h / tex_h)` fits the image to panel
- `effective_scale = base_scale * zoom`, `image_center = panel_center + pan`
- Image drawn via `egui::Painter::image()` with clip to panel rect
- Feature overlays use `image_to_panel(px, py)` and `panel_to_image(pos)`
  transforms derived from `image_rect` and `effective_scale` each frame
- Features outside the visible panel are culled for performance when zoomed in
- The panel rect is `ui.available_rect_before_wrap()`, which is only the panel's
  own if nothing above it has overflowed — see "The toolbar may not widen the
  panel" below

### The toolbar may not widen the panel

egui grows a `Ui`'s `max_rect` to include any widget that overflowed it
(`Placer::advance_after_rects`). The overlay toolbar is a single unwrapped row
of controls — roughly 730 px with a feature mode active — so in a dock cell
narrower than that it overflows, and the `available_rect_before_wrap()` the
panel reads below it then describes a rectangle reaching into the
**neighbouring** dock cell. A 400 px cell reported 726 px.

That rect is load-bearing twice over: the image is fitted and centred in it, and
`platform::pointer_in_rect` tests it to decide whether a trackpad gesture is
addressed to this panel (see
[viewport-navigation.md](viewport-navigation.md#which-panel-a-gesture-is-addressed-to)).
An overhang therefore both mis-lays-out the image and steals gestures aimed at
whatever sits to the right — scrolling the Camera Intrinsics panel beside it
panned the image, for exactly as far into that panel as the overhang reached,
which is why widening the Image Detail panel made the symptom disappear.

So `show_overlay_toolbar` draws its row into a child `Ui` and allocates back only
the space it was *offered*, leaving the parent's `max_rect` alone. What does not
fit stays clipped by the `ScrollArea` egui_dock wraps every tab body in. The
panel is the only one that reads its rect *after* drawing something — every other
tab takes `available_rect_before_wrap()` as its first act, which is pristine by
construction.

### View persistence across image, reconstruction and panel changes

The panel is used to *compare*: flipping between two images with `,` / `.`,
between reconstructions with `[` / `]`, clicking a thumbnail in the strip, or
selecting a different image in the Scene Graph. Comparing a detail — a feature
that moves, a blur, a mis-registered edge — means being zoomed in on it while
the switch happens, so the view is **not** reset on any of those. Only the
explicit `Z` / double-click Fit resets it. Animation playback needs no special
case for this: it is the same image switch as any other.

What is held fixed is the *region of the image*, not the raw `pan`. `pan` is in
panel pixels, so the same value frames a different part of an image of another
resolution. The invariant is the normalized image coordinate at the panel
centre:

```
anchor = 0.5 - pan / display_size          (per axis)
```

Each frame computes `display_size` from the current texture, panel rect and
zoom, then rescales `pan` by the ratio against `last_display_size` before
anything else uses it, and re-clamps to the pan limits. A change of image
resolution, of aspect ratio, or of panel size therefore all keep the same
region framed; two images of equal size in an unchanged panel give a ratio of
1 and carry the view over untouched, so a `,` / `.` flip is pixel-stable.

`zoom` is relative to fit, so it needs no rescale: at 8× two images of different
resolutions each show one eighth of their own frame. The rescale runs *before*
input handling and `last_display_size` is recorded *after* it, so a zoom gesture
within a frame is never mistaken for a change of extent. `reset_view` clears
`last_display_size` for the same reason — a fit view has nothing to carry.

## Navigation minibar

A thin navigation minibar below the thumbnail strip that provides
at-a-glance position awareness and fast random-access navigation across the
full image sequence. This is essential for large datasets (100s–10Ks of
images) where the visible thumbnails represent a tiny fraction of the total
and mouse-drag panning alone is too slow for long-distance jumps.

The minibar is analogous to VS Code's minimap — a compressed visual
representation of the full content that doubles as an interactive navigation
control.

### Visual design

The minibar is ~20px tall, rendered directly below the thumbnail strip,
spanning the full width of the Image Browser panel. It has three layers,
bottom to top:

1. **Color barcode** (background): Each image in the sequence is represented
   as a narrow vertical stripe with 8 pixels of height. Each pixel is the
   average color of the corresponding vertical eighth of the 128×128
   thumbnail (top eighth, second eighth, etc.), giving a rough sense of each
   image's vertical color layout. The full sequence is mapped proportionally
   to the bar width, so the bar always represents the entire image set
   regardless of count.

   At high image counts (e.g., 10K images on a 1500px bar), multiple images
   share pixels and their colors blend together. This is intentional — the
   blended pattern creates recognizable visual landmarks. Scene changes,
   lighting shifts, indoor/outdoor transitions, and camera repositions all
   produce visible color boundaries that give the user spatial memory of the
   sequence ("the bright outdoor section is in the middle, the dark hallway
   starts near the end").

2. **Viewport indicator** (overlay): A 1px white border rectangle showing
   which portion of the sequence is currently visible in the thumbnail strip
   above. Width is proportional to
   `visible_thumbnail_count / total_image_count`. For large sequences this
   becomes a thin sliver, immediately communicating how much content exists
   beyond the visible window.

3. **Selection markers** (top): Thin vertical tick marks drawn over the color
   barcode:
   - **Cyan tick**: Currently selected image (`selected_image`). Visible
     even when the selected thumbnail is scrolled off-screen, so the user
     always knows where their selection is relative to the current view.
   - **Secondary color ticks**: Images in the track set from
     `selected_point` (if any). Uses the same secondary highlight color as
     the thumbnail track highlighting.

### Interaction

| Action | Input | Behavior |
|--------|-------|----------|
| Jump | Click on bar | Set strip `offset_x` so the clicked position is centered in the thumbnail strip |
| Scrub | Drag on bar | Continuously update `offset_x` as the pointer moves horizontally |
| Scrub | Click outside indicator, then drag | Jump to click position, then scrub from there |

- Click position maps to image index via `(click_x / bar_width) * num_images`.
  The strip `offset_x` is set to center that image index in the visible
  thumbnail region.
- Dragging anywhere on the bar (whether starting on the viewport indicator or
  not) scrubs smoothly. There is no need to precisely grab the indicator —
  clicking anywhere jumps first, then dragging continues from the new
  position.
- The minibar does not consume pan gestures that start in the thumbnail strip
  area above. Only pointer events within the minibar's own rect trigger
  navigation.

### Data model

```rust
struct NavigationMinibar {
    /// Texture: width = num_images, height = 8, RGBA.
    /// Each column has 8 pixels representing the average color of each
    /// vertical eighth of the thumbnail.
    color_barcode: Option<egui::TextureHandle>,
    /// Number of images when the barcode was last built (for invalidation).
    cached_image_count: usize,
}
```

- The color barcode texture is built once all thumbnails are loaded. For each
  image, the 128×128 thumbnail is divided into 8 horizontal bands (16 rows
  each), and the average color of each band becomes one pixel in the column.
- The barcode texture is invalidated and rebuilt when the reconstruction
  changes (same trigger as `ImageBrowser::thumbnail_cache` invalidation).
- egui stretches the texture to fill the bar rect, so rendering cost is
  independent of image count.

### Rendering

- Paint the color barcode as a textured mesh stretched to the minibar rect.
- Paint the viewport indicator as a `rect_stroke` with a 1px white border
  over the corresponding horizontal span. The span is computed from
  `offset_x` and the visible thumbnail count.
- Paint selection ticks as thin `rect_filled` calls (1–2px wide, full
  minibar height) at the proportional x-position of each marked image.

### Performance

- The barcode is a single texture upload, built once all thumbnails are
  loaded. No per-frame cost scales with image count.
- Navigation hit-testing is a simple `rect.contains(pointer_pos)` check.
- Position-to-index mapping is O(1): `index = (x / width) * num_images`.

## Non-goals

- **Co-track point highlighting.** Selecting a point does not highlight the
  other 3D points its track images observe. That set is potentially enormous —
  a single point's track images may see thousands of others — so the highlight
  would read as "most of the cloud" rather than as an answer, and there is no
  obvious rule for trimming it that is not itself a new feature.
- **A grid mode in the image browser.** The strip is one row, horizontally
  scrolled. A multi-row thumbnail grid would show more of a large sequence at
  once, at the cost of the position-along-a-sequence reading the strip and its
  minibar are built around.
- **Epipolar lines from the selected image** as a feature-overlay mode on the
  Image Detail panel. `sfm epipolar` draws them offline; nothing in the viewer
  does.
