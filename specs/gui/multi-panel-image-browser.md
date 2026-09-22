# Multi-Panel GUI with Image Browser and Detail Pane

## Overview

The viewer's window is a dockable, tabbed workspace rather than a single
full-window 3D view: panels can be resized, re-tabbed and dragged into new
splits, and they stay in step with one another through a shared selection.
Four panel types:

1. **3D Viewer** — the existing viewport (point cloud, frustums, navigation)
2. **Image Browser** — bottom strip of 128×128 thumbnails for browsing the image sequence
3. **Image Detail** — full-resolution image view for the selected camera
4. **Track View** — the selected 3D point's observations, or the bench's active
   track while its *Edit* box is ticked (see [track-view.md](track-view.md));
   the right-hand column beside the two pictures.

A fifth panel, **Scene**, was added by
[scene-graph.md](scene-graph.md); it takes a narrow left split of the
root, and everything below describes the arrangement to its right.

## Default Layout

```
┌───────┬──────────────────────────┬──────────────┐
│  File                            │  (menu bar)  │
├───────┼──────────────────────────┼──────────────┤
│       │[3D Viewer][Image Detail] │ [Track View] │
│       │                          │              │
│ Scene │        3D Viewer         │  Track View  │
│       │                          │              │
├───────┤                          │              │
│Backgr.├──────────────────────────┴──────────────┤
│       │[Image Browser][Action Log][Edit History]│
│       │ ◀ [img01] [img02] [img03] [img04] ... ▶ │
└───────┴─────────────────────────────────────────┘
```

- **Scene**: left, ~18% width, upper ~72% of that column. The tree of loaded
  reconstructions.
- **Background Task**: under the tree, the rest of the left column. What a long
  operation running off the GUI thread is doing, and what the last one cost
  ([background-tasks.md](background-tasks.md)).
- **3D Viewer**: top-left of the rest, ~2/3 of its width. Point cloud,
  frustums, navigation. It is the active member of its tab group.
- **Image Detail**: behind the 3D Viewer in that same group, one click away.
  Full-resolution image of the selected camera. The two are the large pictures
  of one selection and each wants the width, so they take turns in the middle
  rather than halving it.
- **Track View**: top-right, ~1/3 width, sharing a tab group with Camera
  Intrinsics, and the active member of it. The column beside the
  pictures is where the tables about the selection go.
- **Image Browser**: bottom strip, full width, ~20% of the height.
  Horizontally-scrollable strip of 128×128 thumbnails. It shares its tab group
  with the **Action Log** ([action-log.md](action-log.md)) and the **Edit
  History** panel ([edit-history.md](edit-history.md)), and is the active
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
  shows whatever is selected). The `,`/`.` keys step the selection back/forward
  (wrapping at the ends) even when not in camera view mode; they are handled for
  the whole window rather than by one panel, so they work whichever tab is in
  front ([camera-views.md](camera-views.md#navigating-between-cameras)).
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
from `PointSet::tracks`.

**Data model**: Tracks are stored sorted by `(point_index, image_index)` in
`PointSet::tracks`, with `observation_counts[i]` giving the number of observations
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
[track_view/](../../crates/sfm-explorer/src/track_view) — while
the cross-panel selection state they share is `AppState` in
[state.rs](../../crates/sfm-explorer/src/state.rs).

### Tab Model

```rust
// dock.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    SceneGraph,
    BackgroundTask,
    Viewer3D,
    ImageBrowser,
    ImageDetail,
    TrackView,
    IntrinsicsDetail,
    ActionLog,
    EditHistory,
}
```

The three panels this spec is about are `Viewer3D`, `ImageBrowser` and
`ImageDetail`. The rest are the tabs later panels added to the same enum, each
specified on its own:

| Tab | Title | Spec |
|---|---|---|
| `SceneGraph` | Scene | [scene-graph.md](scene-graph.md) |
| `BackgroundTask` | Background Task | [background-tasks.md](background-tasks.md) |
| `TrackView` | Track View | [track-view.md](track-view.md) |
| `IntrinsicsDetail` | Camera Intrinsics | [camera-intrinsics.md](camera-intrinsics.md) |
| `ActionLog` | Action Log | [action-log.md](action-log.md) |
| `EditHistory` | Edit History | [edit-history.md](edit-history.md) |

`IntrinsicsDetail` shares Track View's tab group as its second and non-active
member; `ActionLog` and `EditHistory` share the Image Browser's as its second
and third.

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
    first:  Split { split: TopBottom, fraction: 0.72,
        first:  Leaf { tabs: [SceneGraph] },
        second: Leaf { tabs: [BackgroundTask] } },
    second: Split { split: TopBottom, fraction: 0.8,
        first:  Split { split: LeftRight, fraction: 0.67,
            first:  Leaf { tabs: [Viewer3D, ImageDetail] },
            second: Leaf { tabs: [TrackView, IntrinsicsDetail] } },
        second: Leaf { tabs: [ImageBrowser, ActionLog, EditHistory] } } }
```

`fraction` is the share of the **first** child in layout order.
(`egui_dock` 0.19's doc comment says "the old node" for both directions of a
split, which is true only of Right and Below; 0.21 gave the Scene panel four
fifths of the window when it was read the other way. `Layout::to_dock` only
ever splits Right and Below for exactly that reason.) A leaf opens on its
**first** tab, which is what puts the 3D Viewer, Track View and the Image
Browser in front of the tabs they share a node with.

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

#### The bench layer

The active track of the node's **bench** ([`bench.md`](bench.md)) is drawn on
this panel as a second layer, independent of the mode above in the same way the
intrinsics layer is, and **last**: it is the one thing here that is not about
the reconstruction, so no mode turns it off and nothing draws on top of it. It
is drawn only in the images that track observes, and only for the *active*
item: the bench holds several and this panel shows the one being worked on.

The colours are the bench's own, one violet per verdict (`in`, `candidate`,
`out`), used nowhere else in the panel, so a mark on the bench is never read as
committed structure; the strokes are thicker than a feature ellipse's.

What is drawn is the track's own geometry rather than a symbol for it, and
differs by stage:

- At the **track stage**, the patch's square boundary sampled and each sample
  pushed through the camera's own forward projection, drawn as a closed
  polyline. The patch is first re-anchored on this image's keypoint
  (`OrientedPatch::anchored_at_keypoint`), as the tile in Track View is
  rendered, so the outline sits where the sighting is in this photograph; the
  patch's own projection is the hollow centre the offset segments run to. The outline is therefore the curve a distorting lens really maps
  that square to, rather than the quadrilateral through its four corners:
  `OrientedPatch::boundary` supplies the samples
  ([`../core/patch/patch-cloud.md`](../core/patch/patch-cloud.md)), eight per
  edge and more when the projection is large. A sample behind the camera or
  outside the lens model's domain **breaks** the polyline there, so an outline
  that leaves the model's field is drawn as the arcs that are defined rather
  than closed across a chord that means nothing. Beside it, each observation's
  own keypoint as a filled dot and, for **every** observation whatever its
  verdict, in that observation's own colour, the segment from that dot to the
  patch's own projection with a hollow circle at the projection: the gap is
  the *Proj. off* column, drawn. Where the two coincide the segment has no
  length and is not seen, which needs no special casing and is the answer as
  much as a long segment is. It is drawn for a judged observation as well as a
  proposed one because where this image's feature sits relative to the
  projection is what the layer is read for, and an observation already voted
  *in* is exactly the one whose answer is worth having in view.
- At the **cluster stage** there is no geometry, so each observation in this
  image contributes the parallelogram its refined 2x2 affine shape maps the
  template's square to, at the refined position, with the seed's own
  parallelogram dashed behind it. How far the two are apart is how far the
  refinement moved and how much it turned. The square is `[-radius, radius]` at
  the cluster's own radius, which it carries from the moment it is started
  ([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
  § "The cluster stage's units"), so a seed clicked at a radius in pixels is
  drawn at that many pixels before anything has evaluated it.

Clicking a mark selects that observation's row in Track View's edit mode
([`track-view.md`](track-view.md) § "Edit mode"): the mark and the row are one observation, so
clicking either is the one gesture. The layer is on top, so a click it catches
does not also select a feature underneath.

##### The handles

**What the layer draws, it edits.** Each mark is a handle, so the geometry a
person is looking at is the geometry they take hold of, and there is no second
picture of the patch to keep in step with the first.

- **The dot moves the patch.** At the track stage a track has one patch and
  every observation is a view of it, so dragging a mark slides that patch
  across its own plane until its centre sits under the pointer: the
  half-vectors and the normal are kept, and **every** observation's keypoint
  moves by the same displacement along the plane, keeping its own offset, so
  the outline moves in every image at once. At the **cluster** stage there is
  no shared geometry, so the mark is that sighting's own seed and nothing else
  moves. With Track View's *Lock* cleared the track stage's mark is that
  sighting's own keypoint as well (below). The cursor is `Move` on hover and
  `Grabbing` while it is held.
- **An edge resizes the patch, with the opposite edge left where it is.**
  Dragging one of the outline's four edges is "put this edge here", and what a
  person expects is the other three where the geometry puts them rather than the
  far edge running away: with the dragged edge at `+h` from the centre and the
  far one at `-h`, a pointer naming the offset `p` gives the new half-length
  `(p + h) / 2` and moves the centre by `h' - h` along the drag. Patch frames are
  square, so that is **one** scale and not two ([`../core/patch/patch-cloud.md`](../core/patch/patch-cloud.md)).
  The cursor comes from the edge's orientation **on screen**, which is what says
  which edge is under the pointer when a lens has bent the square: near
  horizontal takes `ResizeVertical`, near vertical `ResizeHorizontal`, and
  oblique the diagonal its slope names.
- **A corner turns it.** Dragging a corner rotates the patch in its own plane --
  about the patch's outward normal at the track stage, about the sighting at
  the cluster stage -- keeping its place and its size. egui has no cursor for a
  rotation, so a corner takes the resize cursor lying along the way it travels:
  the tangent of the circle it spins on, the perpendicular of its radius from
  the sighting it spins about. Running the pointer along an edge and onto a
  corner therefore turns the cursor from across the edge to along the arc.

**The pointer is read against the patch, not against the screen.** A pixel is a
ray, the ray meets the patch's own plane, and what the pointer named is the
offset of that meeting on the patch's axes
(`OrientedPatch::keypoint_plane_offset`). So an edge put under the pointer
*reprojects onto the pointer*, through whatever distortion the lens has, and a
turn is the angle swept on the patch's own surface rather than the foreshortened
one swept on screen. The frame the pointer is read against is the one drawn: the
patch re-anchored on the observation whose outline it is.

At the track stage the patch is shared, so all three gestures change the
outline in **every** image. The two that move its centre -- the slide and the
resize -- also move the track's point and **carry every sighting along the
plane by the same displacement**, which keeps each one's own offset from where
the centre projects: that offset is where the photograph sees the patch's
content against where the geometry puts its middle, and it is what the tiles
are cut on, so resetting the keypoints to the centre would scramble the
correlation the next reading scores. The sighting the gesture came through
therefore lands under the pointer -- its plane point plus the displacement *is*
the plane point under the pixel -- and the others move with the patch. A turn
moves the centre nowhere, so every sighting stays where it is. None of the three pins anything: where the
patch is says nothing about whether a sighting belongs to it
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Placing,
sizing and turning by hand"). At the cluster stage each sighting has its own
affine shape and its own seed, so every gesture touches only the observation
whose mark or outline was grabbed.

**Track View's *Lock* decides what the track stage's dot is**
([`track-view.md`](track-view.md) § "The toolbar"). Ticked, which is how it
starts, the dot slides the patch as above. Cleared, the dot moves **that one
sighting's keypoint** and nothing else: the patch keeps its place, its size and
its turn, every other sighting keeps its own, and the step is the one the
cluster stage's dot already is (`sight_observation`), so the sighting is pinned
and the measurements read at its old pixel are dropped. This is how a keypoint
that settled on the wrong detail is put right without dragging the rest of the
track after it. In this photograph the outline follows the dot, being the patch
re-anchored on the keypoint, while the hollow centre stays where the patch
projects, so the segment between them grows by the drag.

A track-stage sighting has a place of its own and no shape of its own; its size
and its turn are the patch's. So while the lock is cleared **the outline's edges
and corners take no drag**: they are drawn, a press on one pans the photograph
as a press off the layer does, and no cursor is offered over them. Offering them
patch-wide would make a resize or a spin move every sighting under a setting
whose whole promise is that a drag here moves one. At the cluster stage the lock
changes nothing, every handle there being one sighting's already, and the box is
greyed. The lock reaches this panel only: the 3D viewer's handles are the
patch's own, and its marks select rather than move.

**The press decides which handle, not the drag.** A pointer press inside the
panel is hit-tested against the geometry that frame starts from -- which is what
the person pressed on, since nothing has panned yet -- and if it lands on a
handle the gesture is that handle's from that moment, before egui would call it
a drag at all. The alternative loses the gesture twice over: egui reports a drag
only once the pointer has left the press by several pixels, while the view pans
on whatever motion it is given with no threshold of its own, so the photograph
moves first, the handle is no longer under the press position a later hit test
would be given, and what the person gets is a pan. So the pan is suppressed from
the **press**, for as long as the button is down over a handle, and a press that
hits no handle leaves the view's own drag exactly as it was. A press on a handle
that never moves is a click: it selects that observation's row and edits
nothing.

The reach is nine panel pixels for a dot or a corner and eight from an edge's
polyline, generous against the marks they draw, because the two misses do not
cost the same: a handle missed by two pixels pans the photograph, which the
person then has to undo by eye, while one caught a little early is released
without motion and does nothing.

**One version per drag.** While the pointer is down nothing is pushed: the
layer draws from a transient copy, the track the release would produce, built
by the same core step and read under the same lock. The release applies it through
`AppState::edit_bench_patch`, which is the call the wire's eight patch tools
make, so one gesture is one version, one Action Log row of kind `Bench`, and one
Undo. A drag that ends where it started pushes nothing, the way a verdict an
observation already holds does. **Escape abandons the drag**: the preview goes
and nothing is pushed. The photograph does not pan for the whole of a handle
drag, cancelled or not, until the button comes up: the pointer means one thing
at a time, and what it means was decided where the button went down.

The layer is [image_detail/bench_track.rs](../../crates/sfm-explorer/src/image_detail/bench_track.rs),
and what a pointer means against a patch is
[bench/geometry.rs](../../crates/sfm-explorer/src/bench/geometry.rs), which the
wire's patch tools read it through too.
Like the selection, what the panel is told about the bench is passed in by the
dock rather than read by the panel: one value carrying the task holding the node,
the active track and Track View's *Lock*, which both this layer and the menu's
two bench entries read, so what is offered and what is drawn cannot disagree
about which track is active or what its dot does.

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

**I/O optimization**: `sfmtool_sift_format::read_sift_partial(path, count)`
already skips reading the tail of the arrays at the file level, so
`max_features` saves real I/O for large `.sift` files. The size threshold
requires reading affine shapes to check, but since features are sorted, only
`max_features` entries need to be read before truncating further.

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

1. **SIFT data** for the image via `sfmtool_sift_format::read_sift_partial()`:
   - `positions: Array2<f32>` (N×2) — keypoint (x, y) locations
   - `affine_shapes: Array3<f32>` (N×2×2) — oriented affine shape matrices
   - Read count: `max_features.unwrap_or(total_feature_count)`
   - Descriptors are not needed for visualization and can be discarded.
   - Cache per image index: `HashMap<usize, (Vec<[f32; 2]>, Vec<[[f32; 2]; 2]>)>`
     keyed by `(image_index, read_count)`. Invalidate if `max_features` increases
     beyond the cached read count.

2. **Track mapping** for the image from `PointSet::tracks`:
   - Build `image_idx → Vec<(feature_idx, point_idx)>` mapping (same approach as the
     heatmap command, lines 161-174 of `_commands/heatmap.py`)
   - Only features that participate in a track have associated 3D points and metrics
   - Features not in any track are untracked (drawn differently or omitted depending on mode)

3. **Per-point metrics** looked up by `point_idx`:
   - Reprojection error: `PointSet::points[point_idx].error`
   - Track length: `PointSet::observation_counts[point_idx]`
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
| Zoom | Double-click off a feature | One step of √2 in, toward the cursor position |
| Fit | Z key | Reset pan and zoom to fit image in panel |

- **Sign conventions**: Mouse drag uses "grab the content" convention (content
  follows cursor). DM gestures and trackpad scroll use "push/scroll viewport"
  convention (opposite direction), matching the 3D viewer's shift+scroll→pan
  and shift+DM→pan mappings.
- **Zoom-to-cursor**: Zoom is anchored at the cursor position so the point
  under the cursor stays fixed: `pan = pan * ratio + cursor_rel * (1 - ratio)`.
- **Zoom limits**: Minimum = 1.0 (fit-to-panel). Maximum = 32×.
- **Pan limits**: Clamped so the image overlaps the panel by at least 50px.
- **View persistence**: The view outlives the image it was set on — see below.
- **What a double-click means is decided before the view input runs.** On a
  feature that observes a point it is `Edit on Bench` (see "the context menu"
  below) and the view does not move; anywhere else -- empty image, or a keypoint
  the solve matched to no point -- it is the zoom above. The hit test is the
  click's own, so a double-click stages the point a single click there would
  have selected. Two double-clicks are a doubling of the magnification, and √2
  is what puts the steps on the round zooms a reader thinks in.
- **Fit is the `Z` key**, with the pointer over the panel, and the wire's
  `set_image_detail_view { fit }` ([mcp-server.md](mcp-server.md)).

### Image Detail: the context menu

A right **click** inside the image, press and release within egui's drag
threshold, opens a context menu at the pointer. A right **drag** is the zoom
above, and the threshold is what tells the two apart, so a zoom gesture never
puts a menu up. The menu is opened on egui's own `clicked_by(Secondary)` rather
than on the raw platform button state the pan/zoom handler reads, which is what
makes that distinction available at all. A **left** press opens nothing however
long it is held: like every menu in this window it is built from
`context_menu::on_secondary_click`, which drops the long touch that egui's own
`Popup::context_menu` also opens on, because on Windows the left mouse button
reaches egui as a touch contact
([scene-graph.md](scene-graph.md) § "Panel plumbing").

Its three entries act on the node's **bench** ([`bench.md`](bench.md)) rather
than on the reconstruction, in this order:

| Entry | What it does |
|-------|--------------|
| `Edit on Bench` | Puts the track of the point the feature under the pointer observes on the bench as a track-stage track, and raises Track View on it |
| `Start cluster on the bench here` | Puts a cluster-stage track on the bench seeded at the clicked pixel, with the node's own default patch radius, and raises Track View on it |
| `Add observation to bench track here` | Adds a candidate sighting at that pixel to the bench's active track |

All three are edited afterwards in Track View
([`track-view.md`](track-view.md)), and the commit there is what reaches the
reconstruction. The lower two are the viewer's only way to name a pixel, so this
is where every gesture that needs one lives.

`Edit on Bench` is the same entry the 3D viewport's point menu offers, under the
same name and reporting the same request
([viewport-navigation.md](viewport-navigation.md) § "The point context menu"),
so one gesture is one code path wherever it was made. What it needs is a
**point**, and what names one here is the feature under the place the menu was
opened at. Which features those are is what the observations are backed by: an
`embedded_patches` node keeps a keypoint per observation, so every feature it
draws belongs to a point; a `sift_files` node draws the `.sift` keypoints, and
one the solve matched to nothing is a feature with no point behind it. Both ways
of having nothing to stage grey the entry rather than hide it, in words that say
which it was -- there is no feature here, or that feature belongs to no 3D
point. The hit test is the one a left click selects by, so the entry offers the
point a click there would have selected.

The two lower entries are offered whatever backs the node's observations,
because a bench track is seeds in one image's pixels until it is
committed. Starting a cluster needs nothing but a pixel and a node no background
task is holding; adding to the bench track is greyed until a track is active,
with *"No track is being edited: tick Edit in Track View, or double-click a
Bench item in the Scene tree."*, since a bench with items on it can have none
active. An image the active track already holds a sighting in is not a
refusal -- a second one joins as a candidate and is scored like any other, and
it is the `in` verdict a track cannot hold twice
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)). A busy
node greys all three, carrying the state's own busy sentence.

The pixel is recorded on the frame the menu opens, in source-image coordinates
through the same `panel_to_image` transform the feature hit-testing uses: the
menu's entries are laid out on later frames, by which time the pointer has moved
off the place the user named. It is also what `Edit on Bench` hit-tests at, so
the point the entry stages is the point that was under the pointer when the menu
went up.

`Edit on Bench` and `Start cluster on the bench here` end in a layout
operation, the raise of Track View, which `Add observation` does not, so they
cannot be carried out where the panel's response is read: the frame swaps the
dock out of the state while a tab body draws, and a raise applied there would
land on the placeholder. The panel keeps each request instead
(`ImageDetail::take_point_gesture`, `ImageDetail::take_cluster_start`), as the
viewport keeps its menu's, and `app.rs` drains them once the dock is back,
the cluster through `AppState::start_cluster_here`.

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
explicit `Z` fit resets it. Animation playback needs no special case for this:
it is the same image switch as any other.

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

### Looking at a place: the one function every request goes through

Three things ask this panel to look somewhere: a row click in another panel
(below), the wire's `set_image_detail_view`
([mcp-server.md](mcp-server.md) § "The Image Detail view"), and the panel's own
`Z` fit. What each of them *means* is one pure function over the frame's
geometry, in
[image_detail/view.rs](../../crates/sfm-explorer/src/image_detail/view.rs), so
two callers asking for the same place land in the same pixel:

```rust
/// What the panel was looking at on the frame it last drew: the view, and the
/// frame it is held in. Both halves, because neither is meaningful alone.
pub(crate) struct ViewGeometry {
    pub(crate) image: ImageRef,
    pub(crate) image_size: [f32; 2],   // the photograph's own pixels
    pub(crate) panel_size: [f32; 2],   // the panel body, in points
    pub(crate) pan: [f32; 2],          // image centre off panel centre, in points
    pub(crate) zoom: f32,              // 1.0 = fitted
}

impl ViewGeometry {
    /// Panel points one source pixel spans at zoom 1, and at the standing zoom.
    pub(crate) fn fit_scale(&self) -> f32;
    pub(crate) fn scale(&self) -> f32;
    pub(crate) fn display_size(&self) -> [f32; 2];
    /// The rectangle of the photograph the panel shows, `[x0, y0, x1, y1]`.
    pub(crate) fn visible_rect(&self) -> [f32; 4];
}

/// Where a caller is asking the panel to look.
pub(crate) enum Look {
    Reveal { pixel: [f32; 2] },
    Pixel { pixel: [f32; 2], zoom: Option<f32> },
    Rect([f32; 4]),
    Fit,
}

/// The view `look` settles on, starting from `view`. Only `pan` and `zoom`
/// move.
pub(crate) fn look_at(view: ViewGeometry, look: &Look) -> ViewGeometry;
```

- **`Pixel`** puts that place at the panel centre, at the zoom it names or at
  the standing one.
- **`Rect`** zooms so the rectangle fills the panel on its tighter axis and
  centres it; given its far corner first it frames the same region.
- **`Fit`** is `Z`: zoom 1.0, pan zero.
- **`Reveal`** is the row click's rule, below.

Every outcome obeys the two limits a hand obeys: the zoom is clamped to
`[1, MAX_ZOOM]` and the pan to the rule that keeps `PAN_MARGIN` points of the
image on the panel. `visible_rect` is **not** clipped to the photograph -- at
fit zoom the letterboxed axis runs past both edges -- because the invariant
worth having is that its centre is the image pixel at the centre of the panel,
which is what every `Look` aims.

**The geometry is published, not asked for.** The panel is the only thing that
knows how big its body is, and it knows that only while it is drawing, so each
frame that draws a photograph reports its `ViewGeometry` in
`ImageDetailResponse::view` and the dock puts it on
`AppState::image_detail_view`. That is what the wire's two view tools read; a
frame that drew no photograph publishes nothing and leaves the last reading
standing, and before any frame has drawn there is no reading at all. A view tool
arriving then has nothing to measure against, so it leaves its request standing
and waits for the frame that draws the photograph to publish one
([mcp-server.md](mcp-server.md) § "The Image Detail view"): what is missing is
the reading, not the view.

**A request travels beside the selection**, as `AppState::look:
Option<(ImageRef, Look)>`, written by `AppState::look_at_in_image` (which
selects the image through `select_image` as it goes) and *taken* by the dock
with `AppState::take_look` on the frame this panel shows that image. Taken
rather than read: it asks for a single view, and a request left standing would
re-frame the panel on every later frame.

### Revealing a feature named by another panel

A row of Track View, in either of its modes, is an
*observation*: it names an image **and** a place in it. Clicking one selects
the image, and this panel then shows it at whatever view was left behind, which
by the persistence above can be a corner of the frame the feature is nowhere
near. So the selection carries the feature's pixel with it, and the panel brings
that pixel into view.

The request sits beside the selection rather than in either panel:
`AppState::reveal_in_image` writes a `Look::Reveal` into `AppState::look`
(selecting the image through `select_image` as it goes, so the coupling rules
and the Action Log row are the ones every selection gets), and the dock takes it
on the frame this panel shows that image, handing it to `ImageDetail::show`.
Both panels report the pixel in their response as `reveal_feature` and the dock
turns either into that one call, so the rule below has one implementation
instead of one per panel.

`select_image` clears the field, which is what makes every other way of
selecting an image reveal nothing: the Image Browser, the Scene tree, a frustum
click and `,` / `.` name a photograph and not a place in one, and the view they
arrive at is exactly the one the persistence rule carries over.

What the panel does with the pixel, after the rescale and the pan clamp that
open every frame (so the test is against the view this frame actually starts
from):

- **At fit zoom, nothing.** The whole image is on screen, so there is nothing to
  bring into it, and the margin below would otherwise slide a fitted image
  off-centre for a feature near its edge.
- **When the pixel is already inside the middle `1 - 2 * REVEAL_MARGIN` of the
  panel per axis, nothing.** Walking down a track's rows must not jerk the image
  about for features that are all in one corner of the view. The margin is 5% of
  the panel per axis: a feature a few pixels inside the panel edge is on screen
  but not visible in any useful sense, half its neighbourhood cut off.
- **Otherwise pan so the pixel is at the panel centre**: `pan = display_size / 2
  - pixel * effective_scale`, clamped by the same pan limits as any other pan.
  Centring a pixel of the image asks for a `pan` of at most `display_size / 2`,
  which the limit `(display_size + panel_size) / 2 - 50` allows for any panel
  wider than 100 px, so the clamp bites only in a very small panel; there a
  corner feature ends on screen but off centre.

The **zoom is never touched**. It is the magnification the user chose to inspect
at, and a reveal is a statement about position. That is the whole of what makes
`Look::Reveal` a separate variant rather than a `Look::Pixel` with no zoom: the
two refusals above.

Implemented as `Look::Reveal` in
[image_detail/view.rs](../../crates/sfm-explorer/src/image_detail/view.rs) and
applied by `ImageDetail::look`
([image_detail/mod.rs](../../crates/sfm-explorer/src/image_detail/mod.rs)), and
covered headlessly in
[image_detail/tests.rs](../../crates/sfm-explorer/src/image_detail/tests.rs): a
feature out of view ends centred with the zoom unchanged, one already in view
leaves the pan alone, a fit-zoom reveal moves nothing, a corner feature in a
panel small enough for the limit to bite stops at the clamp, and the view the
frame publishes is centred on the pixel the look named -- which is the reading
`get_image_detail_view` answers with.

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
