# GUI Architecture

SfM Explorer, the sfmtool 3D viewer, is a native Rust application in the
`sfm-explorer` crate. It opens a `winit` window, draws the 3D scene with `wgpu`,
and draws its panels with `egui` inside the same frame, running its own event
loop rather than eframe's. `pixi run gui` runs the crate's `sfm-explorer`
binary; `sfm explorer` runs `launch-sfm-explorer`, a binary that the
`sfmtool-py` crate builds and the Python wheel ships, as a subprocess. Both
call `sfm_explorer::run`. `lib.rs` owns the window and the event loop, `app.rs`
runs each frame, `dock.rs` routes the panels, `scene_renderer/` owns the GPU
passes and `state.rs` holds the application state. This spec records why the
viewer is built on this stack, what each module is responsible for, the order
of the render passes, how it is built and launched, its performance targets,
what differs per platform, and how it is tested.

For the user experience goals driving these choices, see
[user-experience.md](user-experience.md).

---

## Technology Stack

### Decision: Rust with PyO3 Bindings

The GUI is implemented in Rust and exposed to Python via PyO3. A pure Python
GUI (PyQt + VisPy/Open3D) was considered but discarded due to performance
concerns with 10M+ points and GIL limitations for background image loading.

| Component | Choice | Notes |
|-----------|--------|-------|
| GUI Framework | **egui** | Immediate mode, rendered via wgpu (using `egui_wgpu` from the `eframe` crate) |
| 3D Rendering | **wgpu** | WebGPU API, Vulkan/Metal/DX12 backends |
| Window Management | **winit** | Cross-platform window creation and event loop |
| Python Bindings | **PyO3** | Zero-copy numpy array passing |
| Build Tool | **maturin** | Builds Rust extensions as Python wheels |

### Why This Stack

- **egui + wgpu**: Proven combination (used by Rerun, others)
- **Performance**: Native rendering loop, no GIL, true multithreading
- **Scale**: wgpu handles 10M+ points trivially via GPU instancing
- **Cross-platform**: Single codebase for Windows/macOS/Linux
- **Future**: Can compile to WASM for web deployment

### Key Rust Crates

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (quaternions, matrices, transforms) |
| `image` | Image loading and resizing for thumbnails |
| `rayon` | Parallel iterators |
| `eframe` | Used for its `egui_wgpu` sub-crate (renderer + screen descriptor), not the eframe event loop |
| `egui` + `egui-winit` | Immediate-mode UI + winit integration |
| `egui_dock` | Dockable tab layout for multi-panel interface |
| `winit` | Window creation and event loop |
| `pollster` | Blocking executor for wgpu async operations |
| `rfd` | Native file open dialogs |
| `windows` | Windows API bindings for DirectManipulation (Windows-only) |

### Why Custom Event Loop (Not eframe)

The GUI uses a custom winit + wgpu event loop rather than eframe (egui's
built-in framework) because of a Windows-specific incompatibility:
DirectManipulation for precision touchpad support does not work on windows
created through eframe's `WgpuWinitApp::resumed()` code path. The symptom is
that `DM_POINTERHITTEST` is never generated, preventing trackpad gesture
recognition.

The custom event loop creates the winit window directly and integrates egui
via `eframe::egui_wgpu` (the wgpu renderer from the eframe crate, used
standalone without eframe's event loop), giving full control over
DirectManipulation initialization order. See [viewport-navigation.md](viewport-navigation.md#windows-precision-touchpad-support)
for the DirectManipulation details.

---

## Crate Structure

```
sfmtool/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── sfmtool-sfmr-format/      # .sfmr file read/write/verify
│   │   └── src/
│   │       ├── types.rs          # SfmrCamera, SfmrData, SfmrMetadata
│   │       ├── read.rs           # .sfmr archive reading
│   │       ├── write.rs          # .sfmr archive writing
│   │       └── verify.rs         # .sfmr integrity verification
│   │
│   ├── sfmtool-sift-format/      # .sift file read/write/verify
│   │   └── src/
│   │       ├── types.rs          # SiftData, SiftMetadata
│   │       ├── read.rs           # .sift archive reading
│   │       ├── write.rs          # .sift archive writing
│   │       └── verify.rs         # .sift integrity verification
│   │
│   ├── sfmtool-core/             # Core data structures and algorithms
│   │   └── src/
│   │       ├── camera.rs         # Camera struct (position, orientation, target_distance)
│   │       ├── reconstruction.rs # SfmrReconstruction, SfmrImage, Point3D
│   │       ├── frustum.rs        # Frustum corner computation
│   │       └── ...               # Feature matching, alignment, spatial indexing
│   │
│   ├── sfm-explorer/              # GUI application
│   │   ├── examples/             # DirectManipulation reference examples (feature-gated)
│   │   └── src/
│   │       ├── main.rs           # Thin entry-point shim (calls into lib.rs)
│   │       ├── lib.rs            # Winit event loop, wgpu setup, window/DPI plumbing
│   │       ├── app.rs            # Per-frame pipeline: uploads, render passes, egui pass
│   │       ├── dock.rs           # egui_dock Tab enum and TabViewer impl (4 tabs)
│   │       ├── viewer_3d/        # ViewportCamera, interaction, fly navigation
│   │       │   ├── mod.rs        # Viewer3D state and per-frame `show`
│   │       │   ├── camera.rs     # ViewportCamera + clip-plane / FOV control
│   │       │   ├── input.rs      # Mouse / scroll / pinch / keyboard dispatch
│   │       │   └── overlay.rs    # Status text + grid overlay
│   │       ├── scene_renderer/   # GPU rendering pipeline
│   │       │   ├── mod.rs        # SceneRenderer struct, initialization
│   │       │   ├── render.rs     # Multi-pass render method
│   │       │   ├── readback.rs   # GPU pick + depth readback
│   │       │   ├── sizing.rs     # Texture creation and resize
│   │       │   ├── uniforms.rs   # Uniform buffer updates
│   │       │   ├── gpu_types.rs  # GPU data struct definitions, pick tags, constants
│   │       │   ├── compass.rs          # Orientation-compass mesh generation
│   │       │   ├── distorted_mesh.rs   # Tessellated mesh for distorted cameras
│   │       │   ├── upload/       # Per-resource GPU data upload
│   │       │   │   ├── mod.rs          # Upload module wiring
│   │       │   │   ├── points.rs       # Point cloud instance buffer
│   │       │   │   ├── frustums.rs     # Frustum edges, image quads, color buffer
│   │       │   │   ├── thumbnails.rs   # Camera thumbnail atlas
│   │       │   │   ├── patches.rs      # Patch instances + bitmap atlas
│   │       │   │   ├── bg_image.rs     # Camera-view background image
│   │       │   │   └── track_rays.rs   # Selected-point observation rays
│   │       │   └── pipelines/    # Per-pass pipeline creation
│   │       │       ├── mod.rs          # Pipeline module re-exports
│   │       │       ├── points.rs       # Point splat pipeline
│   │       │       ├── edl.rs          # Eye-Dome Lighting post-process pipeline
│   │       │       ├── frustum.rs      # Frustum wireframe pipeline
│   │       │       ├── image_quad.rs   # Pinhole image quad pipeline (instanced)
│   │       │       ├── distorted_quad.rs # Distorted image quad pipeline (indexed)
│   │       │       ├── bg_image.rs     # Shared background-image bind group resources (layout, uniform, sampler)
│   │       │       ├── bg_distorted.rs # Background image pipeline (distorted; the only BG mesh pipeline)
│   │       │       ├── patch.rs        # Patch/surfel rendering pipeline
│   │       │       ├── target.rs       # Target indicator pipeline
│   │       │       └── track_ray.rs    # Track ray pipeline
│   │       ├── state.rs              # Shared application state (AppState)
│   │       └── state/
│   │           └── ops.rs        # AppState's reconstruction operations (load, align)
│   │       ├── layout.rs             # The layout document: window placement + panel arrangement
│   │       ├── window.rs             # WindowHost, the window snapshot, and the `window` section
│   │       ├── image_browser.rs      # Thumbnail strip with horizontal scrolling
│   │       ├── image_menu.rs         # The image menu, shared by image rows and thumbnails
│   │       ├── add_image_to_tracks.rs # The image menu's Add Image to Tracks: gate, background job, log line
│   │       ├── image_detail/         # Full-resolution image display panel
│   │       │   ├── mod.rs            # Panel state and per-frame `show`
│   │       │   ├── input.rs          # Pan / zoom / hover dispatch
│   │       │   └── overlay.rs        # Feature + reprojection overlays
│   │       ├── track_view/           # Track View: the Edit checkbox and its one body
│   │       │   ├── mod.rs            # The box, the empty state, and the choice of mode
│   │       │   └── body/             # One track, Viewed (Edit clear) or Edited (Edit ticked)
│   │       │       ├── mod.rs        # Body state, header, toolbar, threshold boxes
│   │       │       ├── table.rs      # Observation table, rows, Keep or Verdict column
│   │       │       ├── tile.rs       # Per-observation tiles
│   │       │       ├── crop.rs       # Per-observation crops of the photograph
│   │       │       ├── surface_plot.rs # Self-similarity surface plots
│   │       │       └── patch.rs      # The tile's warp and the stored bitmap's picture
│   │       ├── bench.rs              # Every bench step as a version of the node
│   │       ├── bench/live.rs         # The live evaluation of every bench track
│   │       ├── colormap.rs           # Shared colour ramps for overlays
│   │       ├── context_menu.rs       # The context-menu builder every panel opens with
│   │       ├── metrics.rs            # Reprojection error, ray angles, triangulation diagnostics
│   │       ├── platform/
│   │       │   ├── mod.rs
│   │       │   └── windows.rs    # DirectManipulation touchpad integration
│   │       └── shaders/
│   │           ├── points.wgsl         # Point splat rendering
│   │           ├── edl.wgsl            # Eye-Dome Lighting post-process
│   │           ├── frustum.wgsl        # Frustum wireframe rendering
│   │           ├── image_quad.wgsl     # Image texture on frustum far plane (pinhole)
│   │           ├── distorted_quad.wgsl # Tessellated image quad (distorted cameras)
│   │           ├── bg_image_distorted.wgsl # Background image rendering (distorted)
│   │           ├── patch.wgsl           # Patch/surfel rendering
│   │           ├── target_indicator.wgsl # Rotating 3D compass / world-up indicator at target
│   │           └── track_ray.wgsl       # Track ray visualization
│   │
│   ├── sfmtool-colmap/           # COLMAP format read/write
│   │   └── src/
│   │
│   └── sfmtool-py/               # PyO3 bindings
│       └── src/
│           └── lib.rs            # Python module (sfmtool._sfmtool)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `main.rs` | Thin entry-point shim (~6 lines) that calls into `lib::run`. |
| `lib.rs` | Window creation, wgpu device/surface, DirectManipulation init, `winit` event loop, Windows DPI awareness, `egui_dock` `DockState` initialization from `Layout::default()` ([panel-layout.md](panel-layout.md)). |
| `app.rs` | The per-frame pipeline (`App::run_ui_and_paint`): per-frame uploads (points, frustums, frustum colors, thumbnails, track rays, bg image, clip planes), camera-uniform updates, scene encoder + render passes, egui pass with the dock UI, AccessKit propagation, surface acquire/present, pick-result dispatch. |
| `app/menu.rs`, `app/modals.rs`, `app/save.rs` | The menu bar and keyboard shortcuts, modal answers and deferred closes, and File-menu save handling, respectively. The egui pass calls them in that order before drawing the dock. |
| `dock.rs` | `Tab` enum (4 variants) and `egui_dock::TabViewer` implementation: routes each tab to its panel's `show()` and threads the cross-panel `*Response` shape into `AppState`. |
| `viewer_3d/` | `ViewportCamera` (wraps `Camera` from sfmtool-core with FOV/clip planes), orbit/pan/zoom math, WASD fly navigation with Q/E tilt, input handling (mouse/trackpad/keyboard), Alt-mode target control, camera view mode, grid drawing. `mod.rs` orchestrates; `camera.rs`/`input.rs`/`overlay.rs` own the sub-concerns. |
| `scene_renderer/` | All GPU pipeline management across ~14 modules: texture creation/resize, point upload, frustum upload (pinhole + distorted), thumbnail texture array atlas loading, uniform updates, multi-pass rendering, GPU readback, per-pass pipeline creation (`pipelines/` subdirectory). |
| `state.rs` | `AppState` struct: reconstruction data, visibility toggles, selected image/points, hover state, rendering parameters, the dock, and the window snapshot — plus the small selection and lookup accessors the panels call every frame. |
| `state/ops.rs` | `AppState`'s reconstruction *operations*, as a second `impl` block: `load_demo` and `align_node`. These are the methods that make demo data or run a fit, build a `SceneNode` out of the answer or move one, and write the sentence the user reads about it. |
| `state/open.rs` | Opening a file as a background task: `start_open` and `open_files`, the worker's `open_job` (the read, then the thumbnails and patch bitmaps a file lacks, built for display), and `append_opened`, which makes the nodes when the task lands. See [background-tasks.md](background-tasks.md) § "Opening a file". |
| `layout.rs` | The layout document — the window's placement and the panel arrangement, its JSON, and the `AppState` operations the Panels menu and the MCP tools share. See [panel-layout.md](panel-layout.md). |
| `window.rs` | What the window is (`WindowInfo`, `WindowState`, `MonitorInfo`), the `window` section of a layout document (`WindowChange`, `MonitorRect`, `fit_to_monitor`), and the `WindowHost` seam every `winit` window call goes through — five primitives and the provided `apply` that orders them. |
| `image_browser.rs` | Horizontally-scrollable thumbnail strip with click-to-select, double-click to enter camera view, a right click that opens the image menu on a thumbnail, gesture-driven panning, lazy thumbnail loading, navigation minibar + animation playback. |
| `image_menu.rs` | The image menu: `show`, which lays out an image's context-menu entries (`Resect Image`, `Add Image to Tracks`, `Move Camera`, `Delete Image`) with their greyed states and gives back the one chosen, and `AppState::image_menu` / `resect_image_refusal`, which say why `Resect Image` is greyed. A Scene tree image row and an Image Browser thumbnail both call `show`, and the dock carries the entry out through one function. See [scene-graph.md](scene-graph.md) § "Image menu". |
| `image_detail/` | Full-resolution image display for the selected camera, with lazy loading, aspect-ratio-preserving fit, pan/zoom that persists across image and reconstruction switches, and 7 overlay modes. |
| `track_view/` | Track View ([track-view.md](track-view.md)). `mod.rs` draws the *Edit* checkbox, which reads the focused item, and the empty state, and hands the rest to `body/`, the one body that draws a track in two modes: with *Edit* clear the viewed track, the selected point read as an editable track off every bench, read-only, with the point's summary in the header and a *Verdict* column the session's read-only bars colour; with *Edit* ticked the focused item of the selected node's bench, with its verdict switches, the threshold boxes that apply to it, and the toolbar that fits, moves the stage, splits and commits. Both show the crop, the rendered tile, the ZNCC and self-similarity readings and the evaluation's state. `table.rs` draws the table, `tile.rs` and `crop.rs` the two pictures per row and their hover views, `patch.rs` the warp and the stored bitmap's conversion. The header's triangulation numbers are `metrics.rs`, at the crate root. |
| `bench.rs` | Every step on a node's bench, as a version of that node: the `AppState` methods that call the pure `sfmtool_core::bench` steps, push one version and write one Action Log row, and the two that read photographs as background tasks. `bench/live.rs` keeps every track evaluated against its current inputs, on a worker of its own, with no version and no row. See [bench.md](bench.md). |
| `goto_point.rs` | Go to Point: parses a typed point index or `pt3d_<hash>_<index>` ID, resolves it against the loaded scene (bare index → selected node, hash → the node carrying it), and owns the modal that collects it. Parse and lookup are plain functions over the scene slice; the dialog returns a `PointRef` rather than applying it. See [goto-point.md](goto-point.md). |
| `save_minimal_prompt.rs` | The workspace path `File > Save As Minimal...` asks about once its file dialog has named a file: a `SaveMinimalPrompt` holding the chosen file and the text of the field, prefilled with the path the save would measure, and reporting a `SaveMinimalAnswer` for the caller to write with rather than writing anything itself. See [saving.md](saving.md) § "The workspace path prompt". |
| `colormap.rs` | The two color ramps — `ERROR_COLORMAP` and `QUALITY_COLORMAP` — one `ramp(value, vmin, vmax, &Colormap)` that samples either, and the colorbar legend the heatmap overlays draw. |
| `context_menu.rs` | `on_secondary_click(&response)`, the builder every context menu in the window is opened with. It is `egui::Popup::context_menu` restricted to `clicked_by(Secondary)`, and `on_secondary_click_where(&response, here)` for a widget that hit-tests its own parts, such as the Image Browser strip: egui's own builder also opens on a long touch, and on Windows the left mouse button reaches egui as a touch contact, so a left press rested for 0.8 s would otherwise put the menu up. See [scene-graph.md](scene-graph.md) § "Panel plumbing". |
| `metrics.rs` | Triangulation numerics: per-observation reprojection error and ray angle, whole-track condition number and inverse-depth z-score, and the widest pairwise ray angle. At the crate root because three surfaces quote the same numbers — Track View's table, the Image Detail overlay's heatmaps, and the MCP `get_point` tool. |

---

## Rendering Pipeline

The renderer uses a multi-pass architecture with three color render targets
plus hardware depth. All scene geometry shares the same depth buffer for
correct mutual occlusion.

### Render Targets

| Target | Format | Purpose |
|--------|--------|---------|
| Color | Rgba8UnormSrgb | Visible scene color |
| Linear depth | R32Float | EDL shading + depth readback for Alt+click |
| Pick ID | R32Uint | Entity identification (hover + click) |
| HW depth | Depth32Float | Z-test during rendering |

These four formats are declared once, in `scene_renderer/gpu_types.rs`, as
`GBUFFER_COLOR_FORMAT` / `GBUFFER_LINEAR_DEPTH_FORMAT` / `GBUFFER_PICK_FORMAT` /
`HW_DEPTH_FORMAT`, alongside the reversed-Z `GBUFFER_DEPTH_STATE` and the
`gbuffer_targets(color_blend)` helper that builds the three-target array. Both
sides of the contract read them: `scene_renderer/sizing.rs` allocates the
textures and every pass-1 pipeline in `scene_renderer/pipelines/` declares its
targets. `pipelines/tests.rs` binds all five pass-1 pipelines inside a pass
built from the same constants, so a producer that drifts from the consumer
fails headlessly rather than at the first frame on a real GPU.

### Pass Order

```
Pass 0:  Background image (camera view mode only)
  Input:  Full-resolution camera photograph
  Output: EDL output texture (cleared to BG_COLOR, image quad drawn)

Pass 1a: Point splats
  Input:  Point instance buffer (position + color)
  Output: Color + linear depth (positive) + pick ID + hw depth

Pass 1b: Frustum wireframes
  Input:  Frustum edge buffer (endpoints + color + index)
  Output: Color + linear depth (0.0) + pick ID + hw depth

Pass 1c: Image quads (pinhole instanced + distorted indexed)
  Input:  Image quad instances + thumbnail texture array atlas
  Output: Color + linear depth (0.0) + pick ID + hw depth

Pass 2:  EDL post-process
  Input:  Color + linear depth textures
  Output: Final display color (edge darkening + supernova glow)

Pass 3:  Target indicator
  Input:  Compass edge instances + star mesh + scene depth texture
  Output: EDL output color (additive blending, reads depth for occlusion)

Pass 4:  Track rays (when a 3D point is selected)
  Input:  Edge instance buffer (camera center → nearest ray point) + hw depth
  Output: EDL output color (premultiplied alpha blending, reads depth for occlusion)
```

For detailed specifications of each pass, see:
- [point-cloud-rendering.md](point-cloud-rendering.md) — Points, EDL,
  target indicator, supernova
- [camera-views.md](camera-views.md) — Frustum wireframes, image quads,
  pick buffer

### GPU Readback

A 5x5 pixel region around the cursor is read back from both the linear depth
and pick ID textures every frame using `wgpu::Buffer::map_async`. This serves:

- **Hover overlay**: Shows entity info under cursor (point index, camera name,
  depth value)
- **Click picking**: Alt+click reads depth to set orbit target; regular click
  reads pick ID to select/deselect cameras
- **Fuzzy matching**: Center-first search (center pixel → 3x3 → 5x5) makes it
  easy to click on thin wireframe lines

### Track Ray Visualization

When a 3D point is selected, track rays show the observation rays from each
camera that contributed to that point. Each ray is a semi-transparent orange
glow line drawn from the camera center to the nearest point on the camera's
observation ray to the selected 3D point. The gap between the ray endpoint
and the actual point visualizes reprojection error in 3D space.

**Data flow**: `upload_track_rays` computes an `EdgeInstance` per observation —
it looks up the cached SIFT feature position, unprojects it through the camera
intrinsics to get a world-space ray direction, then projects the selected 3D
point onto that ray. Each `EdgeInstance` stores `endpoint_a` (camera center) and
`endpoint_b` (nearest point on ray).

**When they are rebuilt**: whenever what they were built from changes, which is
the selected point *and the version of the node that owns it*, plus that node's
transform. The point alone is not enough: a bundle adjustment that renumbers
nothing leaves the selection exactly where it was and replaces every position
under it, and an undo of one does the same in reverse. A `VersionSerial` is
minted once and never reused, so holding the last frame's against this one's
asks whether the rays on screen were built from the value on screen.

That pair -- point and version serial -- is `app::selected_point_source`, and
everything the frame draws *for the selection* is gated on it, the rays and the
frustum colours of the images that observe the point alike. A commit of a bench
track moves both halves at once: the selection lands on the row the commit
wrote, above the node's point count where it replaced one, and the version under
it is the one the commit pushed.

**When they are cleared**: whenever there is nothing to draw: no selection, a
node that has gone, a point this version does not have, or a node that is not
visible. Visibility belongs here because the rays are a singleton with no node
of their own in the draw loop: `render_track_rays` draws whatever the buffer
holds, so nothing else would stop a hidden or solo'd-out node's rays hanging in
the air over the node still shown.

**Rendering**: Track rays reuse the shared `QuadVertex` buffer (same as point
splats). The vertex shader expands each edge instance into a screen-space ribbon
quad with a configurable pixel width (`line_half_width: 1.5px`), keeping line
thickness constant regardless of zoom. The fragment shader samples the hardware
depth texture for per-fragment occlusion against scene geometry, and applies a
`smoothstep` glow falloff from center to edge at 40% peak opacity with
premultiplied alpha blending. The orange color (`rgb(1.0, 0.647, 0.0)`) matches
frustum highlight coloring.

**Pipeline details**: Triangle strip topology, no culling, no depth write. Uses
`FrustumUniforms` (view-projection matrix, screen size, line half-width). Bind
group includes the uniform buffer and the hardware depth texture for occlusion
reads.

### Scene-to-egui Integration

The 3D scene renders to an offscreen texture, which is then displayed as an
egui `Image` filling the central panel. egui renders its own UI (menu bar,
overlays, controls) on top. This avoids the complexity of mixing egui and wgpu
render passes.

---

## Build System

### Pixi Integration

The Rust crates are built using pixi to manage the Rust toolchain and system
dependencies. This follows the pattern used by
[rattler](https://github.com/conda/rattler) and its
[py-rattler](https://github.com/conda/rattler/tree/main/py-rattler) Python
bindings.

**Why pixi for Rust?**
- Reproducible Rust toolchain version across all developers
- Handles system dependencies (OpenSSL, pkg-config, compilers)
- Unified environment for both Python and Rust development
- Single `pixi run` command to build, test, and develop

### Development Workflow

```bash
# Build and test Rust code
pixi run cargo-build
pixi run cargo-test

# Run the standalone GUI binary (release mode)
pixi run gui

# Format and lint Rust code
pixi run cargo-fmt
pixi run cargo-clippy

# Check for build errors without producing binaries
pixi run cargo-check
```

### Python Integration

The GUI runs as a standalone binary (`sfm-explorer`). It can be launched via
`pixi run gui` or `sfm explorer` from the CLI. The `sfmtool-py` crate includes
a `launch-sfm-explorer` binary that the Python wheel ships, and the
`sfm explorer` CLI command runs it as a subprocess.

---

## Performance Design

### Target Scale

| Metric | Target |
|--------|--------|
| Points | 10,000,000+ |
| Cameras | 10,000+ |
| Frame rate | 30+ fps during navigation |

### GPU Memory Budget

| Item | Size (10K cameras, 10M points) |
|------|-------------------------------|
| Point instance buffer | 160 MB |
| Frustum edge buffer | 2.5 MB |
| Image quad instances | 640 KB |
| Thumbnail texture array atlas (128x128) | 625 MB |
| Pick buffer (1920x1080) | 8.3 MB |
| Depth textures | ~16 MB |

Thumbnail memory is the main concern at scale. The atlas uses a
`texture_2d_array` with multiple pages to stay within GPU texture dimension
limits (e.g. 8192px → 64×64 = 4096 cells per page, multiple pages for larger
datasets). Compressed texture formats (BC7/ASTC) would reduce memory ~4x.
For 10K+ cameras, async loading and an LRU texture cache are planned.

### Design Decisions for Performance

- **GPU instancing**: Points and frustum edges use instanced draw calls, not
  individual draw calls per entity
- **Buffer re-upload on change**: Frustum edges are re-uploaded (~2.5 MB) when
  selection changes, rather than maintaining a separate selection uniform buffer.
  This is simpler and fast enough.
- **Single depth readback**: One 5x5 region per frame, not full-screen readback
- **Lazy image loading**: Thumbnails loaded on reconstruction open; full-res
  photographs decoded on demand, off the GUI thread, into a shared cache under a
  byte budget ([../core/camera/photograph-cache.md](../core/camera/photograph-cache.md))

---

## Platform-Specific Details

### Windows

- **DirectManipulation API**: Precision touchpad gesture recognition (pan,
  pinch, inertia). Requires specific initialization order relative to winit.
  See [viewport-navigation.md](viewport-navigation.md#windows-precision-touchpad-support).
- **Trackpad scroll in `ScrollArea`s**: DM claims the touchpad contacts for the
  whole window, so Windows never synthesises a `WM_MOUSEWHEEL` for a two-finger
  scroll and egui's own scroll areas — the scene graph tree and its inner
  lists, the Camera Intrinsics panel, Track View's table — would sit still
  under one. `platform::gesture_scroll_events` converts each frame's DM pan back
  into a `Point`-unit `Event::MouseWheel` on the raw input, which is what makes
  them scroll. X is negated on the way through: DM reports a horizontal pan
  with the opposite sign to the way egui reads a wheel's X, which is why the
  image strip adds `dx` to its scroll offset where it subtracts a wheel
  `delta.x` from it.
  The panels that read DM gestures directly (3D viewport, Image Detail,
  Image Browser) do not double-handle it: they take wheel input through
  `platform::ScrollInput`, which suppresses `Point` scroll for exactly the
  frames DM was active. A pan under Ctrl/Cmd is dropped instead of forwarded,
  since those frames are already a zoom gesture for the panels that handle DM
  and egui reads a Ctrl+wheel as a request to zoom the whole UI.
- **DPI awareness**: `SetProcessDpiAwarenessContext` for per-monitor DPI
- **Graphics backend**: DirectX 12 via wgpu

### macOS (Planned)

- Trackpad gestures via native NSEvent / egui's built-in `zoom_delta`
- Metal backend via wgpu

### Linux

- **Graphics backend**: Vulkan via wgpu, on X11 or Wayland through winit. It is
  the only backend the crate compiles in for this platform (`wgpu`'s features
  in `crates/sfm-explorer/Cargo.toml` name `dx12`, `vulkan` and `metal`), so
  there is nothing to fall back to: a machine carrying the Vulkan loader with
  no ICD behind it — a bare CI runner is the usual one — panics on
  `Failed to create wgpu surface` at startup rather than degrading to software
  GL. Mesa's lavapipe is enough to run the viewer, and is what the
  `ui-test-linux` CI job installs.
- **Accessibility**: AT-SPI2 over D-Bus, published by AccessKit's Unix adapter,
  which egui-winit sets up. Unlike UI Automation and the AX API, this is not
  part of the OS: the tree exists only where a session bus and the
  `at-spi-bus-launcher` / `at-spi2-registryd` daemons are running. `ui_basic`
  does not read this tree, so its Linux job starts none of it; see "Testing"
  below.
- **Touchpad gestures**: none of their own. The precision-touchpad handling
  under Windows is DirectManipulation-specific, and Linux gets whatever winit
  reports as scroll and egui's built-in pinch handling.

## Testing

The crate's tests split by what they need underneath them. The **lib** tests
are headless and run anywhere: `scene_renderer/upload/tests.rs` drives real
`wgpu` uploads on the `noop` backend, which validates in wgpu-core while
stubbing the driver, and `track_view/tests.rs` and
`track_view/body/tests.rs` run whole egui frames through `Context::run_ui`.
Everything decidable without an OS is decided there, because it is decidable in
milliseconds and on every platform.

The **`ui_basic`** integration tests are the other half: a real window and a
real GPU surface. They exist for the defects that live below egui and that a
headless frame cannot show — a menu item wired to a command the event loop
never reads, a HUD check box that never reaches the window, a `screenshot` of a
frame that was actually presented, a right mouse button that never reaches
egui. They run on all three desktop platforms via `pixi run ui-test`, one
window at a time (a process-wide mutex, so a plain `cargo test` behaves like
`--test-threads=1`).

Because they need a window, they are **off by default**. `ui_basic` is declared
in [`crates/sfm-explorer/Cargo.toml`](../../crates/sfm-explorer/Cargo.toml) as an
explicit `[[test]]` target with `required-features = ["ui-tests"]`, so
`cargo test --workspace` builds and runs the lib tests and nothing windowed,
while `pixi run ui-test` expands to
`cargo test -p sfm-explorer --features ui-tests --test ui_basic --
--test-threads=1 --nocapture`. Cargo *silently skips* a target whose required
features are off rather than reporting anything, so every invocation that is
supposed to see this file — including the clippy gate that type-checks it on
Linux — names the feature explicitly.

### Two ways of reading the window

**Most of the tests read the window through the viewer's own MCP endpoint.**
Each launches the viewer with `--mcp 0` (an ephemeral port, since a developer
running the suite very likely has a viewer of their own on 8787) and reads the
port it prints. `get_widgets` lists every widget egui drew in a frame, with its
role, name, rectangle, enabled and toggled state, and reports open dialogs and
menus as their own blocks; `click` presses a widget by id, the way a person's
mouse would; `screenshot` and the command vocabulary do the rest
([mcp-server.md](mcp-server.md) § "`get_widgets`", § "`click` / `hover`").
That is the information the platform's accessibility tree carries, read inside
the viewer's process instead of across the operating system's accessibility
bridge, and the bridge is the expensive part: on a GitHub-hosted Windows runner
one walk of the viewer's tree through it can take about 18.6s. Read
in-process, the listing also says what the platform trees do not report the
same way on every platform: whether a menu item is enabled, which widget owns a
context menu, and which items open a submenu.

An MCP test never reads the accessibility API. `McpViewer` waits on
the endpoint instead: it polls `get_widgets` until the menu bar's `File` button
is listed, which says the window exists and has drawn a frame, and that point
ends the test's launch time. Every reply comes after a frame the viewer drew,
and a `click` reply after the frame in which a menu or dialog it opened
appears, so a test asserts on a menu straight from the click that opened it.
Anything that can take more frames — a background load, a layout change —
is polled for with `McpViewer::wait_for`, a `get_widgets` loop with a deadline.
Names repeat across roles (the HUD has a `Points` check box, slider and
label), so lookups take a role and a name. A menu item drawn with a keyboard
shortcut carries it in its name (`Save Ctrl+S`, `Save ⌘S` on macOS), and a
shortcut has no space in it, so `menu_item` matches the label alone or the
label followed by one word, and no test spells a platform's shortcut.

**One smoke test reads the tree through the platform, on Windows only,**
because what it checks is outside the viewer's process. `window_appears` checks
that the tree reaches UI Automation with the menu bar's four buttons in it, so
a published but empty tree fails; this is what a screen reader needs. It runs
on Windows alone because the viewer's own accessibility logic is
Windows-specific: the viewer creates its window hidden and shows it only after
AccessKit has registered its UI Automation provider. On macOS and Linux the
AccessKit adapters are egui-winit's, with no code of the viewer's around them,
and on every platform the MCP tests show that the tree has content, because
`get_widgets` reads the same AccessKit update the adapters publish. The
`#[ignore]`d `dump_widgets` prints the whole window's `get_widgets` listing for
a person debugging one.

**One test sends real OS input.**
`a_real_right_click_opens_the_reconstruction_rows_context_menu` (Windows)
presses the right mouse button with `SendInput` on the Scene panel's `demo`
row. A synthetic `click` goes into egui's input and never passes through
winit, so it cannot see a defect in how the operating system's input reaches
egui. On Windows `EnableMouseInPointer` makes every mouse button arrive as a
touch, and a right click reaches egui only because
`platform::windows::restore_mouse_button` rewrites it
([scene-graph.md](scene-graph.md)). The input is real;
where to press and what happened are both read over MCP. The point is the
window block's `inner_position`, the drawable area's top-left corner on the
desktop, plus the centre of the row label's `rect_px` in a `get_widgets`
listing of the whole window, which is measured from that corner
([mcp-server.md](mcp-server.md) § "The window block"). Both are physical
pixels. The viewer is per-monitor DPI aware, and `SetCursorPos` takes
coordinates in the *calling* process's DPI context, so the test makes its own
process per-monitor aware (`SetProcessDpiAwarenessContext` with
`DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`) before it aims and checks that the
awareness took; without that, on a scaled display the cursor would land at the
logical point rather than the physical one. After the press it polls
`get_widgets` until a `context_menu` owned by a widget in the `scene` panel is
listed, checks that the menu's `owner.at_px` (where egui received the button)
is inside the label it aimed at, and checks a few of the menu's entries. Its
MCP twin, `the_reconstruction_rows_context_menu_lists_every_entry`, runs on all
three platforms and asserts everything the menu holds.

Setup goes through the **command line** rather than the UI: `--demo` appends
the node File > Load Demo Data… makes, at the dialog's default point count and
with the same `None` path, after any files named on the line. Exactly one test
drives the route that shortcut replaces —
`the_scene_panel_lists_the_loaded_reconstruction` clicks through the menu and
the dialog, then asserts on what arrived — so the route stays covered. Two
tests start the viewer on a saved default layout file, which is the one piece
of setup that is not a flag.

### The smoke test's tree walks

`window_appears` calls UI Automation directly through the `windows` crate. It
launches the viewer without an MCP endpoint and finds its window by process id:
the visible, titled top-level window `EnumWindows` reports for that pid (winit's
own helper windows are invisible and untitled). The viewer makes that window
visible only after its UI Automation provider is registered, so the window
appearing ends the test's launch time.

It then joins the multithreaded COM apartment, creates a `CUIAutomation`
client, and takes the window's element with `ElementFromHandle`. **One walk is
one `FindAllBuildCache(TreeScope_Subtree)` against that element**, with a true
condition and a cache request for the name and control type properties: one
cross-process call that returns the window and every node beneath it with
those two properties already fetched. Rooting at the window loses nothing to
look at, because the viewer runs a single egui viewport and paints its menus
and popups inside that one window.

The test polls that walk with a deadline until one snapshot holds a node of
control type Button named exactly each of `File`, `Edit`, `Go` and `Panels`,
and then asserts that the same snapshot holds no `View` button. Polling is
what keeps it honest on a slow runner, where the menu bar can land in the tree
a walk or two after the window appears; a failed walk counts as "not yet"
until the deadline.

### What the suite costs

**The suite reports what it costs, in every log.** As each test's `Guard`
drops — so a panicking test reports too — it prints a line, and after it the
running total; the last `UIPROBE TOTAL` is the run's:

```text
UIPROBE test=file_menu_items launch_ms=369 window_ms=0 ops=0 op_ms=0 walks=0 walk_ms=0 calls=2 call_ms=12 total_ms=431
UIPROBE TOTAL tests=20 launch_ms=10438 window_ms=16 ops=1 op_ms=35 walks=2 walk_ms=67 calls=54 call_ms=660 total_ms=13672 mean_launch_ms=521 mean_op_ms=35 mean_walk_ms=33 mean_call_ms=12
```

`launch_ms` is the process spawn and GPU init up to the point the viewer is
usable: for the smoke test its window becoming visible, for an MCP test the
first `get_widgets` that lists the menu bar. It is work no change to what the
tests *ask* makes cheaper, so `mean_launch_ms` moving between two runs means
the *machine* moved. `window_ms`, `ops`, `op_ms`, `walks` and `walk_ms` are the
accessibility bridge's share and are zero for every test but the Windows smoke
test: `window_ms` is creating the UI Automation client and taking the window's
element; `ops` is how many *waits* on the tree the test made — the wait for the
menu bar is one op however many walks it polls, so `ops` moves only when the
tests change — and `op_ms` the time inside them; `walks` is how many subtree
queries those waits took (plus the one `UIPROBE TREE` makes), and `walk_ms` the
time inside those calls and nothing else, so `walk_ms` close to `op_ms` says the
platform's query is what costs and a gap says the suite was waiting for the app
to draw. `calls` and `call_ms` are the
MCP tool calls a test made after its launch, polls included, and the time spent
waiting for their replies; each reply follows at least one drawn frame, so
`mean_call_ms` is close to the price of a few frames. `total_ms` is the guard's
whole life, teardown included, and alone distinguishes none of this, which is
why one log carries all of it. The counters are process-wide statics reset per
guard, which is sound only because `UI_TEST_LOCK` keeps exactly one guard alive
at a time. The `ui-test` pixi task, and its Linux form, pass `--nocapture`,
since libtest discards a passing test's stdout and these lines are wanted on
green runs above all.

`window_appears` prints the denominator the walk averages need, once per run:

```text
UIPROBE TREE nodes=52 walk_ms=31
```

`nodes` and `walk_ms` come from one more walk of the same kind the wait makes,
so it prices the same work, taken after the menu bar has been seen, against the
viewer's empty state. The window itself is counted among the nodes. `walk_ms /
nodes` is the per-node price of a UI Automation subtree query.

The Windows job also brackets the suite with
[`ci_windows_ui_snapshot.ps1`](../../scripts/ci_windows_ui_snapshot.ps1). Its
`UIENV` lines record the hosted image and OS, CPU model and topology, current
clock and load, memory and disk, interactive session and desktop processes,
display mode, power plan, effective Defender state, and UI Automation version.
These are observations rather than setup: an unavailable probe reports itself
and cannot suppress the tests. The same snapshot after the suite distinguishes
a runner that arrived slow from resource pressure accumulated by the build.
Runners on the same image differ in speed, and these properties are what show
which host attribute moves with the cost.

### What the lock covers

**Whatever a test puts outside its own process is the lock's business too.** The
`Guard` that holds the mutex owns the viewer process *and* anything the test
placed in the developer's home directory — the two tests that start the viewer
on a saved default layout write one path there — and it drops them in that
order, so the file is back before the next test can take the lock. Restored
after the lock is released, it would race the next test's own `rename` of it
under a plain multi-threaded `cargo test`, and that test would fail with
"Access is denied".

### Platforms

The MCP tests run the same way on all three platforms, and need only what the
viewer itself needs: a display to open a window on and a GPU surface. Neither
they nor the smoke test need anything set up for the platform's accessibility
API — UI Automation is always live on Windows, and the macOS and Linux runs do
not read the accessibility tree at all, so macOS needs no Accessibility (TCC)
grant and Linux no AT-SPI daemons.

A headless Linux box has no display, so the Linux `ui-test` task goes through
[`scripts/display_env.sh`](../../scripts/display_env.sh), which starts Xvfb
when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set and otherwise only runs the
command. The `ui-test-linux` CI job apt-installs `xvfb` for it, with the X
client libraries winit loads at run time.

Every test needs a working GPU surface, since each draws real frames and the
screenshot tests decode the PNG a presented frame produced — on Linux that
means a Vulkan ICD, per "Linux" above. The right-click test is Windows-only:
it presses the button through `SendInput` to check the `WM_POINTER` routing,
which exists only there.

**A test that synthesizes OS input aims first.** `SendInput` presses a button
wherever the cursor happens to be, on whatever window is under it, and neither
of those belongs to the test process: the viewer has just launched, another
application can be in front of it, the point can be off-screen, and
`SetCursorPos` can be clamped or refused outright. Every one of those looks
identical from inside the test — the menu never opens, a widget lookup burns
its full budget, and the failure reads like a product regression. So
`ui_basic`'s `aim_at` raises the viewer's own window, moves the cursor, and
checks that the window under it belongs to that process before a button is
pressed, retrying the *aim* and never the assertion; `mouse_event` checks that
`SendInput` actually inserted the event rather than being refused. Raising
takes two calls, because the obvious one does not carry: `SetForegroundWindow`
is refused whenever the caller is not already the foreground process, which a
test runner launched from a terminal is not, so `aim_at` follows it with a
`SetWindowPos` to `HWND_TOPMOST` — no such restriction, and Z order is all
`WindowFromPoint` reads — and lets the left click do the activating. Nothing
raises a window above a locked session's lock screen, so on a locked desktop
the aim fails and says which process is on top (`explorer.exe`, whose
`LockScreenBackstopFrame` covers every monitor).

In CI the three suites are three jobs — `ui-test-windows`, `ui-test-macos`,
`ui-test-linux` — each running `pixi run -e ui ui-test`, which passes
`--features ui-tests`, and separate from the coverage job, which excludes
`sfm-explorer` entirely so that uninstrumented artifacts never land in its
target directory. The lib tests run instead in `test-os-rust`, whose single
`cargo test --workspace` reaches them precisely because `ui_basic` is gated out
of it.
