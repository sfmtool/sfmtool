# GUI Plan for sfmtool

This document tracks the implementation plan, current status, and future
direction for the sfmtool 3D viewer. For detailed specifications, see the
other documents in this directory:

- [gui-user-experience.md](gui-user-experience.md) — Vision and design
- [gui-architecture.md](gui-architecture.md) — Technology stack and crate structure
- [gui-point-cloud-rendering.md](gui-point-cloud-rendering.md) — Point rendering and EDL
- [gui-camera-views.md](gui-camera-views.md) — Frustum wireframes and image projection
- [gui-viewport-navigation.md](gui-viewport-navigation.md) — Navigation controls

---

## Current CLI Visualization

The CLI already provides several visualization commands that inform the GUI
design:

| Command | Purpose |
|---------|---------|
| `sfm epipolar` | Epipolar lines, matched features, sweep visualization |
| `sfm heatmap` | Per-feature quality metric overlays on images |
| `sfm inspect` | Text-based reconstruction summary with ASCII histograms |
| `sfm compare` | Align and compare two reconstructions |
| `sfm undistort` | Remove lens distortion from images |

---

## Current Implementation Status

*Updated: 2026-03-27*

### 3D Viewer — Functional

The core 3D viewer is working with these features implemented:

- Custom winit + wgpu event loop with egui integration
- `egui_dock`-based multi-panel layout (3D Viewer, Image Browser, Image Detail)
- GPU point splat rendering (billboard quads, instanced)
- Eye-Dome Lighting (EDL) post-processing
- Camera frustum wireframes with image texture projection (128×128 thumbnails)
- Distorted frustum rendering (tessellated mesh for cameras with lens distortion)
- GPU pick buffer for entity selection (frustums and points)
- Full orbit/pan/zoom navigation with Alt-mode target control
- WASD fly navigation with Q/E tilt and Shift sprint
- Target indicator (rotating octahedron) with supernova lighting effect
- Windows precision touchpad via DirectManipulation API
- Ground plane grid, axis lines, orientation gizmo
- `.sfmr` file loading via File > Open dialog
- Hover overlay showing entity info under cursor
- FOV menu slider (10°–120°) with min-dimension FOV convention
- View through selected camera (Z key: pose + best-fit FOV from intrinsics)
- Full-resolution background image in camera view mode (pinhole + distorted)
- `,`/`.` keys to navigate between cameras in camera view mode
- Single 3D point selection with track visualization (Phase A complete)
- Image detail feature overlay modes (None, Features, Reproj Error, Track Length)
- Feature filtering controls (max features, min/max size, tracked only)
- Colorbar legend for heatmap overlay modes

### Core Data — Complete

- `.sfmr` and `.sift` file read/write/verify: fully implemented (separate
  `sfmr-format` and `sift-format` crates)
- `SfmrReconstruction` struct with full conversion to/from columnar I/O format
- `Camera` type with quaternion orientation, view matrix, look-at transforms
- Feature matching, alignment, and spatial indexing modules

### PyO3 Bindings — I/O Complete, GUI Stub

- `.sfmr` and `.sift` I/O bindings with zero-copy numpy arrays
- Feature matching and image pair graph bindings
- Module is `sfmtool._sfmtool`
- The GUI runs as a standalone binary via `pixi run gui` or running `sfm-explorer`

## Next Steps

In rough priority order:

1. **Grid and coordinate systems in 3D** — Currently the grid and the 3D coordinate system are
   overlaid on top of everything. It would be nicer if they integrate inside the 3D scene with
   depth occlusion.
2. **FOV zoom gesture** — Dedicated input binding for FOV adjustment during
   navigation (lower priority now that the menu slider exists; see
   [gui-viewport-navigation.md](gui-viewport-navigation.md#fov-zoom-planned))
3. ~~**Adaptive grid** — Update grid to work at any scale.~~ Done.
4. ~~**Adaptive clip planes** — Reversed-Z infinite far projection with adaptive near plane.~~ Done. See [gui-adaptive-clip-and-grid.md](gui-adaptive-clip-and-grid.md).

---

## Future Direction

### Multi-Panel Interface — Implemented

The viewer uses an `egui_dock`-based dockable, tabbed interface with three
panels:

- **3D Viewer** — Top-left (~67% width). Point cloud, frustums, navigation.
- **Image Detail** — Top-right (~33% width). Full-resolution image of the
  selected camera.
- **Image Browser** — Bottom strip (~20% height). Horizontally-scrollable
  strip of 128×128 thumbnails.

Cross-view synchronization: selecting a camera in any panel (click thumbnail
in browser, click frustum in 3D viewer) updates all others.

### Image Browser — Implemented

The image browser is a horizontal thumbnail strip with:
- Click to select/deselect, double-click to enter camera view
- Gesture-driven horizontal scrolling (DirectManipulation on Windows)
- Auto-scroll to selected image
- Lazy thumbnail loading (up to 8 per frame)

**Planned enhancements**:
- Grid mode (multi-row thumbnail layout)
- Animation mode (play through images as sequence)
- Overlay modes: features, tracks, reprojection error, triangulation angle,
  epipolar lines to selected image

### Performance Targets

| Metric | Target |
|--------|--------|
| Points | 10,000,000+ |
| Cameras | 10,000+ |
| Frame rate | 30+ fps during navigation |

See [gui-architecture.md](gui-architecture.md#performance-design) for GPU
memory budget and design decisions.
