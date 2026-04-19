# Plan: Multi-Panel GUI with Image Browser and Detail Pane

## Overview

Introduce `egui_dock`-based multi-panel layout to the GUI, replacing the current single
`CentralPanel` with a dockable tab system. Add three panel types:

1. **3D Viewer** — the existing viewport (point cloud, frustums, navigation)
2. **Image Browser** — bottom strip of 128×128 thumbnails for browsing the image sequence
3. **Image Detail** — full-resolution image view for the selected camera

## Default Layout

```
┌──────────────────────────────────┬──────────────┐
│  File  View                      │  (menu bar)  │
├──────────────────────────────────┼──────────────┤
│                                  │              │
│                                  │    Image     │
│           3D Viewer              │    Detail    │
│                                  │              │
│                                  │              │
├──────────────────────────────────┴──────────────┤
│ ◀ [img01] [img02] [img03] [img04] [img05] ... ▶│
│              Image Browser (~160px)             │
└─────────────────────────────────────────────────┘
```

- **3D Viewer**: top-left, ~2/3 width. Point cloud, frustums, navigation.
- **Image Detail**: top-right, ~1/3 width. Full-resolution image of the selected camera.
- **Image Browser**: bottom strip, full width, ~160px. Horizontally-scrollable strip
  of 128×128 thumbnails.

Since we use `egui_dock`, the user can re-dock any panel anywhere (float, reorder tabs,
resize splits, etc.).

## Panel Interaction Model

### Image Selection

All three panels share `AppState::selected_image` as the central image selection state:

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
- **Deselect**: Clicking background in 3D viewer or clicking the selected thumbnail again
  clears `selected_image`. The detail pane shows "No image selected."

**What changes when `selected_image` changes:**
- Image Browser: cyan highlight border moves to the new thumbnail
- 3D Viewer: frustum re-upload with new selection color (already implemented via
  `prev_selected_image` change detection)
- Image Detail: loads the new full-resolution image from disk (same path as `upload_bg_image`
  in camera view mode, but rendered to an egui texture instead of a wgpu background pass)

### 3D Point Selection

All three panels also share `AppState::selected_point: Option<usize>` for 3D point selection.
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

**State**: `AppState::hovered_point_index: Option<usize>`. Updated every frame from the 3D
viewer's existing GPU pick buffer (`SceneRenderer::hover_pick_id`). Currently
`hover_pick_id` lives only in `SceneRenderer` and is passed as a parameter to the
status text overlay in `viewer_3d.rs` (line ~1280). To enable cross-panel hover, the
resolved point index is promoted to `AppState`:

```rust
// In AppState:
/// Transiently hovered 3D point index from the 3D viewer's pick buffer.
/// Updated every frame; None when the cursor is not over a point.
pub hovered_point_index: Option<usize>,
```

Each frame, after `SceneRenderer::read_back_pick()`, main.rs extracts the point index:
```rust
let hover_pick_id = self.scene_renderer.hover_pick_id();
let tag = hover_pick_id & PICK_TAG_MASK;
let index = (hover_pick_id & PICK_INDEX_MASK) as usize;
state.hovered_point_index = if tag == PICK_TAG_POINT { Some(index) } else { None };
```

**Cross-panel effects**: The hover point drives the same track-based highlighting as
point selection, but with a visually softer treatment:

```
                          hovered_point_index
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
  - The highlight updates every frame as `hovered_point_index` changes. Since the track lookup
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
| **Hover** | `hovered_point_index` | Soft (brighter/glow) | No | Soft border/dot | Thin outline |
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

### Tab Model

```rust
// main.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Viewer3D,
    ImageBrowser,
    ImageDetail,
}
```

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

```rust
let mut dock_state = DockState::new(vec![Tab::Viewer3D]);
let surface = dock_state.main_surface_mut();
// Split bottom strip for image browser (80/20 vertical)
let [top, _browser] = surface.split_below(NodeIndex::root(), 0.8, vec![Tab::ImageBrowser]);
// Split top area for detail pane (67/33 horizontal)
let [_viewer, _detail] = surface.split_right(top, 0.67, vec![Tab::ImageDetail]);
```

### Integration in main.rs

Replace the current `egui::CentralPanel` block with:

```rust
egui::CentralPanel::default().show(ctx, |ui| {
    DockArea::new(&mut dock_state)
        .style(egui_dock::Style::from_egui(ui.style().as_ref()))
        .show_inside(ui, &mut tab_context);
});
```

The menu bar (`TopBottomPanel::top`) remains unchanged above the dock area.

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

## Implementation Plan

Each step is a self-contained commit. Verify `pixi run cargo-check` after each step.
Run `pixi run cargo-fmt-check && pixi run cargo-clippy` before each commit. Run
`pixi run cargo-test` at the end.

### Step 1: egui_dock integration with 3D viewer only — DONE

`egui_dock`-based layout with `Tab` enum (`Viewer3D`, `ImageBrowser`, `ImageDetail`),
`DockState`, `TabContext`, and `TabViewer` implementation in `main.rs`. Default layout:
3D Viewer top-left (67%), Image Detail top-right (33%), Image Browser bottom (20%).

### Step 2: Image browser — thumbnail loading and strip layout — DONE

`image_browser.rs` fully implemented with:
- Manual offset-based horizontal panning (not `ScrollArea`, for DirectManipulation
  gesture support)
- Lazy thumbnail loading (up to 8 per frame)
- Aspect-ratio-correct thumbnail display from camera intrinsics

### Step 3: Image browser — selection and sync — DONE

- Click to select/deselect with cyan border
- Auto-scroll to selected thumbnail on external selection change
- `ImageBrowserResponse` with `selection_changed` and `request_camera_view`

### Step 4: Image detail — full-resolution image display — DONE

`image_detail.rs` implemented with:
- Lazy full-res image loading on selection change
- Aspect-ratio-preserving fit to panel
- "No image selected" empty state
- Texture handle caching
- 2D pan/zoom navigation (see "Image Detail — 2D pan and zoom navigation"
  section below)

### Step 5: Double-click to enter camera view mode — DONE

- Double-click thumbnail in browser → selects image and requests camera view
- Double-click frustum in 3D viewer → selects image and enters camera view

### Phase A: Single Point Selection — DONE

Select one 3D point at a time by clicking it in the 3D viewer. Deselect by
clicking background or clicking the point again. This phase establishes the selection mechanics, track lookup, and all
cross-panel visual effects for a single point.

#### Step 6: Click-to-select a single point — DONE

Wire up the existing GPU pick buffer for point clicks. The pick buffer already
writes `PICK_TAG_POINT | point3d_index` per point splat — this step reads it.

**`main.rs`** — extend the click handler (around line 654):

Currently only `PICK_TAG_FRUSTUM` is handled. Add a `PICK_TAG_POINT` branch:

```rust
if tag == scene_renderer::PICK_TAG_FRUSTUM {
    // ... existing frustum selection logic ...
} else if tag == scene_renderer::PICK_TAG_POINT {
    let idx = index as usize;
    state.selected_point = Some(idx);   // always set (even if same point)
}
```

Clicking the same point again keeps it selected — it does not toggle off.
Deselection is only possible by clicking empty background.

Point click should not clear `selected_image`, and image click should not
clear `selected_point`. They are independent selections.

Clicking empty background (the existing `else if !pending_click_is_alt`
branch) clears both `selected_image` and `selected_point`.

**`state.rs`**:

Use `selected_point: Option<usize>`.

Clear `selected_point` on reconstruction change (in `load_reconstruction`).

**Verification**: Load a reconstruction, click a point in the 3D viewer.
The hover overlay already shows "Point3D #N" — confirm the click sets
`selected_point`. Click the same point again — stays selected. Click a
different point — selection moves. Click background to deselect.

#### Step 7: Track lookup helper — DONE

Build a helper to look up a point's track — the set of `(image_index,
feature_index)` observations. This is used by all subsequent visualization
steps.

**`sfmtool-core/src/reconstruction.rs`**:

Add a precomputed prefix sum of `observation_counts` to
`SfmrReconstruction`. This makes track lookups O(1) for any point,
which matters for hover (called every frame).

```rust
pub struct SfmrReconstruction {
    // ... existing fields ...

    /// Prefix sum of `observation_counts`: `observation_offsets[i]` is the
    /// index into `tracks` where point `i`'s observations begin.
    /// Length: `points.len() + 1` (last element = total observation count).
    pub observation_offsets: Vec<usize>,
}
```

Compute `observation_offsets` in `SfmrReconstruction::from(SfmrData)`:

```rust
let mut offsets = Vec::with_capacity(observation_counts.len() + 1);
offsets.push(0);
for &count in &observation_counts {
    offsets.push(offsets.last().unwrap() + count as usize);
}
```

Add methods:

```rust
impl SfmrReconstruction {
    /// Return the observations for a given 3D point. O(1) lookup.
    pub fn observations_for_point(&self, point_idx: usize)
        -> &[TrackObservation]
    {
        let start = self.observation_offsets[point_idx];
        let end = self.observation_offsets[point_idx + 1];
        &self.tracks[start..end]
    }

    /// Return the image indices that observe a given 3D point.
    pub fn track_image_indices(&self, point_idx: usize) -> Vec<usize> {
        self.observations_for_point(point_idx)
            .iter()
            .map(|obs| obs.image_index as usize)
            .collect()
    }
}
```

**Verification**: Write a `#[test]` in `reconstruction.rs` that constructs
a small reconstruction with known tracks and verifies `observations_for_point`
returns the correct observations.

#### Step 8: Highlight the selected point in the 3D viewer — DONE

When a point is selected, it should be visually distinct from all other
points. The approach is to re-upload the point buffer with the selected
point's color changed.

**`scene_renderer/upload.rs`**:

Add a `selected_point: Option<usize>` parameter to the point upload. When
set, override the color of that point to a highlight color (yellow:
`0xFF_FF_FF_00` packed ABGR, or experiment with magenta `0xFF_FF_00_FF`).

Track the previous `selected_point` in `SceneRenderer` (same pattern as
`prev_selected_image` for frustums). Only re-upload when it changes.

**Performance**: Re-uploading the full point buffer (~160 MB for 10M points)
on every selection change is not ideal. Two alternatives:

1. **Uniform-based highlight** (preferred): Pass `selected_point_index` as
   a uniform to the point vertex shader. The shader compares
   `instance_index == selected_point_index` and overrides the color for
   that one point. This avoids any buffer re-upload and costs one
   comparison per vertex.

   ```wgsl
   // In points.wgsl vertex shader:
   if (uniforms.selected_point_index == in.point3d_index) {
       out.color = vec4(1.0, 1.0, 0.0, 1.0);  // yellow highlight
   }
   ```

   Add `selected_point_index: u32` to the point uniforms struct (use
   `0xFFFFFFFF` as "none" sentinel since point indices are 24-bit).

2. **Buffer re-upload**: Simpler but slower for large point clouds. Only
   viable if selection changes are infrequent. Acceptable as a first pass
   if the uniform approach complicates the shader too much.

Start with approach 1 (uniform-based).

**Visual treatment**: The selected point should be clearly visible but not
overwhelming. Options to evaluate:

- Bright yellow color override (simple, high contrast against most scenes)
- Larger point size (e.g., 2× radius) via shader
- Both color + size

Start with color-only, evaluate after seeing it in practice.

**Verification**: Select a point, confirm it turns yellow. Deselect, confirm
it returns to its original color. The highlight should be visible against
both dark and bright point clouds.

#### Step 9: Highlight track frustums in the 3D viewer — DONE

When a point is selected, the frustums for cameras that observe it should
get a secondary highlight color — distinct from the primary cyan used for
`selected_image`.

**Color choice**: Orange or warm yellow for track frustums. This creates a
clear visual hierarchy:
- Cyan = "I selected this image"
- Orange = "These cameras see the point I selected"
- White = normal, unselected

**`scene_renderer/upload.rs`**:

Extend `upload_frustums()` to accept `track_image_set: &HashSet<usize>`
(or a slice). When generating frustum edge colors:

```rust
let color = if Some(i) == selected_image {
    SELECTED_COLOR   // cyan (existing)
} else if track_image_set.contains(&i) {
    TRACK_COLOR      // orange (new)
} else {
    NORMAL_COLOR     // white (existing)
};
```

**`main.rs`** or **`state.rs`**:

Compute `track_image_set` when `selected_point` changes. Cache it in
`AppState` as `track_image_set: HashSet<usize>`. Recompute on
`selected_point` or reconstruction change. Pass to `upload_frustums()`.

Frustum re-upload is already triggered by `prev_selected_image` change
detection. Add `prev_selected_point` tracking alongside it. When either
changes, re-upload frustums.

**Verification**: Select a point, confirm that a subset of frustums turn
orange. These should be the cameras that actually observe the point. Select
an image that is in the track set — it should be cyan (image selection
overrides track highlight). Deselect the point — frustums revert to white
(or cyan for the still-selected image).

#### Step 10: Highlight track images in the browser strip — DONE

When a point is selected, thumbnails in the image browser that are part of
the selected point's track get a secondary highlight.

**`image_browser.rs`**:

`ImageBrowser::show()` already receives `selected_image`. Add a
`track_image_set: &HashSet<usize>` parameter.

For each thumbnail, check membership:

```rust
let is_selected = selected_image == Some(i);
let is_in_track = track_image_set.contains(&i);

// Draw highlight:
if is_selected {
    // cyan border (existing)
} else if is_in_track {
    // orange border (thinner or different shade than cyan)
}
```

**Visual treatment**: A 2px orange border around track thumbnails. Thinner
than the 3px cyan selection border to maintain hierarchy. Alternatively, a
small colored dot in the corner of each track thumbnail (less intrusive).

Start with the orange border approach — it mirrors the frustum highlighting
and is easy to see at a glance.

**Verification**: Select a point in the 3D viewer. Confirm the image browser
shows orange borders on the thumbnails corresponding to the track images.
Scroll to see if off-screen track images are also correctly highlighted when
scrolled into view. Deselect the point — borders disappear.

#### Step 10b: Track ray visualization in the 3D viewer — DONE

When a point is selected, draw semi-transparent "glow" rays along each
observing camera's true observation direction. Each ray starts at the
camera center and extends along the ray through the 2D feature keypoint
(unprojected via camera intrinsics) to the point on the ray nearest to
the 3D point. This means the rays do **not** converge exactly on the 3D
point — the gap between the ray endpoint and the 3D point is the
reprojection error visualized in 3D space. This makes tracks physically
visible and provides immediate visual feedback about reconstruction
accuracy.

**Rendering approach**: Post-EDL pass (Pass 2.75), following the same
pattern as the target indicator (Pass 2.5). Renders onto `edl_output` with
alpha blending, sampling the linear depth texture for depth-aware occlusion.

**Geometry**: Same ribbon-quad technique as frustum wireframes. Each ray is
a camera-facing quad stretched between two endpoints (camera center →
nearest point on the observation ray to the 3D point). Reuses the
`EdgeInstance` GPU struct (`endpoint_a`, `endpoint_b`).

**Shader** (`track_ray.wgsl`): Vertex shader identical to frustum shader.
Fragment shader samples depth buffer for occlusion, outputs orange glow
with UV-based falloff.

**Pipeline** (`pipelines/track_ray.rs`): Single color target on
`edl_output` with alpha blending, no depth stencil (shader-based occlusion).

**Data flow** (`upload_track_rays()` in `scene_renderer/upload.rs`):
For each `TrackObservation`:
1. Look up the camera center `C` and camera-to-world rotation `R^T`
2. Look up the feature position `(px, py)` from the shared SIFT cache
3. Unproject via `CameraIntrinsics::pixel_to_ray(px, py)` → camera-local ray
4. Rotate to world space: `d_world = R^T * d_cam`
5. Project the 3D point onto the ray: `t = dot(P - C, d_world)`,
   `nearest = C + t * d_world`
6. Emit `EdgeInstance` with `endpoint_a = C`, `endpoint_b = nearest`

The SIFT positions come from the shared `AppState::sift_cache` (see
"Shared SIFT cache" below). The caller pre-populates the cache for all
track images before calling `upload_track_rays()`.

#### Step 11: Feature overlays and feature-click selection on the detail image — DONE

Draw all tracked features for the selected image on the image detail panel,
make them clickable to select the corresponding 3D point, and highlight the
feature(s) belonging to the currently selected point.

##### Step 11a: Draw all tracked features on the detail image — DONE

**Per-image track index** (`sfmtool-core/src/reconstruction.rs`):
`image_feature_to_point: Vec<HashMap<u32, u32>>` and
`max_track_feature_index: Vec<u32>` computed at reconstruction load time.

**SIFT data loading**: Via the shared SIFT cache (see below). When
`selected_image` changes, `ensure_sift_cached()` loads positions and
affine shapes from the `.sift` file using `read_sift_partial(path,
max_track_feature_index[img_idx] + 1)`. `ImageDetail` builds a local
`Vec<TrackedFeature>` + KD-tree from the cached data.

**Drawing**: Green oriented ellipses (1px stroke, 32-segment polygons via
SVD decomposition of the 2×2 affine shape matrix) with red center dots
(2px). Features outside the visible panel are culled for performance.

##### Step 11b: Click a feature to select the corresponding 3D point — DONE

**Hit testing**: KD-tree (`kiddo::KdTree<f32, 2>`) for O(log n) nearest
feature lookup. 8px hit radius in image coordinates. Returns
`ImageDetailResponse::select_point` which main.rs wires to
`state.selected_point`.

**Tooltip**: On hover within hit radius, shows "Point3D #N | err: X.XXXpx"
with a dark background rect for readability.

##### Step 11c: Highlight the selected point's feature on the detail image — DONE

When `selected_point` is set and is observed by the current image, its
feature gets yellow ellipse (2px stroke) + yellow center dot (4px),
drawn on top of the default green features.

#### Shared SIFT cache — DONE

`AppState::sift_cache: HashMap<usize, CachedSiftFeatures>` stores
positions and affine shapes (no descriptors) per image index. Loaded
lazily via the free function `ensure_sift_cached()` which takes disjoint
borrows of `sift_cache` and `reconstruction` to avoid borrow conflicts.
Cleared on reconstruction load.

Used by:
- **ImageDetail**: caller ensures cache is populated for the selected
  image, passes `&CachedSiftFeatures` to `show()`
- **Track ray upload**: caller pre-populates cache for all track images,
  passes `&sift_cache` to `upload_track_rays()`

#### Step 12: Highlight co-track points in the 3D viewer

When a point is selected, other 3D points that share images with the
selected point (co-visible points) could be highlighted. However, this is
potentially a very large set (a single point's track images may observe
thousands of other points), so this step is deferred to Phase B evaluation.

Do not implement this in Phase A. Evaluate the need after using single-point
selection in practice.

---

### Phase B: Evaluation and Refinement — NOT STARTED

After Phase A is complete and tested with real reconstructions, use the
single-point selection to identify what additional visualization or
functionality would be most valuable. This phase is intentionally
open-ended.

**Evaluation questions:**

- Is the selected point clearly visible in the 3D viewer? Does it need
  to be larger, brighter, or have a glow effect?
- Are the orange track frustums useful? Are there too many or too few for
  typical points?
- Is the SIFT feature marker on the detail image at the right location?
  Is the ellipse useful or is a simple crosshair sufficient?
- Should we draw *all* features on the detail image (the full overlay mode
  system from the original Step 8 spec) or is the single selected feature
  marker sufficient?
- Is the track information in the browser strip (orange borders) useful
  for navigating? Would a minibar (original Step 10) help more?
- Should clicking a track-highlighted frustum in the 3D viewer both select
  that image *and* keep the point selected? (Current design: independent.)
- Should clicking a track-highlighted thumbnail in the browser auto-scroll
  the 3D viewer to show that camera's frustum?
- Do we need co-track point highlighting (Step 12)?
- Do we need hover-based track highlighting (transient highlight while
  the mouse is over a point, without clicking)?
- Image detail pan/zoom is now implemented — is it sufficient for
  inspecting features at pixel level?

**Potential additions based on evaluation:**

- **Point hover**: Promote `hovered_point_index` to `AppState` (from the
  existing hover overlay text). Show soft highlights in all panels for the
  hovered point's track — lighter than selection, updates every frame.
  This was part of the original spec but may be more complexity than value
  for Phase B.

- **Feature overlay modes**: The full overlay system (Features / Reproj
  Error / Track Length / Max Track Angle) from the original Step 8 spec. Only
  add this if evaluation shows the single-feature marker is insufficient.

- **Image detail pan/zoom**: Implemented (see "Image Detail — 2D pan and
  zoom navigation" section below).

- **Navigation minibar**: Implemented. See the
  [Navigation Minibar](#navigation-minibar) section below.

- **Co-track point highlighting**: Step 12 from Phase A. Only add if
  evaluation shows it would help understand reconstruction connectivity.

---

### Image Browser Navigation Minibar — DONE

The navigation minibar is implemented below the thumbnail strip, providing
at-a-glance position awareness and fast random-access navigation. See the
[Navigation Minibar](#navigation-minibar) section below for the full design.

---

### Image Detail — 2D pan and zoom navigation — DONE

The image detail panel supports pan and zoom to inspect the full-resolution
image, similar to how the 3D viewer navigates the point cloud but in 2D.

**State** (in `ImageDetail`):

- `pan: egui::Vec2` — offset of image center from panel center, in panel pixels
- `zoom: f32` — zoom level (1.0 = fit to panel, max 32×)
- `prev_selected_image: Option<usize>` — for detecting changes and resetting

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
- **Auto-reset**: When `selected_image` changes, pan and zoom reset to fit.

**Rendering** (`image_detail.rs`):
- `base_scale = min(panel_w / tex_w, panel_h / tex_h)` fits the image to panel
- `effective_scale = base_scale * zoom`, `image_center = panel_center + pan`
- Image drawn via `egui::Painter::image()` with clip to panel rect
- Feature overlays use `image_to_panel(px, py)` and `panel_to_image(pos)`
  transforms derived from `image_rect` and `effective_scale` each frame
- Features outside the visible panel are culled for performance when zoomed in

### Navigation minibar

A thin navigation minibar below the thumbnail strip that provides
at-a-glance position awareness and fast random-access navigation across the
full image sequence. This is essential for large datasets (100s–10Ks of
images) where the visible thumbnails represent a tiny fraction of the total
and mouse-drag panning alone is too slow for long-distance jumps.

The minibar is analogous to VS Code's minimap — a compressed visual
representation of the full content that doubles as an interactive navigation
control.

#### Visual design

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

#### Interaction

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

#### Data model

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

#### Rendering

- Paint the color barcode as a textured mesh stretched to the minibar rect.
- Paint the viewport indicator as a `rect_stroke` with a 1px white border
  over the corresponding horizontal span. The span is computed from
  `offset_x` and the visible thumbnail count.
- Paint selection ticks as thin `rect_filled` calls (1–2px wide, full
  minibar height) at the proportional x-position of each marked image.

#### Performance

- The barcode is a single texture upload, built once all thumbnails are
  loaded. No per-frame cost scales with image count.
- Navigation hit-testing is a simple `rect.contains(pointer_pos)` check.
- Position-to-index mapping is O(1): `index = (x / width) * num_images`.
