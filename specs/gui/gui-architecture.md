# GUI Architecture

This document describes the technology stack, crate structure, rendering
pipeline architecture, and build system for the sfmtool 3D viewer.

For the user experience goals driving these choices, see
[gui-user-experience.md](gui-user-experience.md).

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
| `kiddo` | KD-tree for nearest-neighbor distance (auto point sizing) |
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
DirectManipulation initialization order. See [gui-viewport-navigation.md](gui-viewport-navigation.md#windows-precision-touchpad-support)
for the DirectManipulation details.

---

## Crate Structure

```
sfmtool/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── sfmr-format/              # .sfmr file read/write/verify
│   │   └── src/
│   │       ├── types.rs          # SfmrCamera, SfmrData, SfmrMetadata
│   │       ├── read.rs           # .sfmr archive reading
│   │       ├── write.rs          # .sfmr archive writing
│   │       └── verify.rs         # .sfmr integrity verification
│   │
│   ├── sift-format/              # .sift file read/write/verify
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
│   │   └── src/
│   │       ├── main.rs           # Entry point, winit event loop, wgpu setup, egui_dock layout
│   │       ├── viewer_3d.rs      # ViewportCamera, interaction, fly navigation
│   │       ├── scene_renderer/   # GPU rendering pipeline (~14 modules)
│   │       │   ├── mod.rs        # SceneRenderer struct, initialization
│   │       │   ├── render.rs     # Multi-pass render method
│   │       │   ├── upload.rs     # Point/frustum/thumbnail upload
│   │       │   ├── readback.rs   # GPU pick + depth readback
│   │       │   ├── sizing.rs     # Texture creation and resize
│   │       │   ├── uniforms.rs   # Uniform buffer updates
│   │       │   ├── gpu_types.rs  # GPU data struct definitions, pick tags, constants
│   │       │   ├── auto_point_size.rs  # Median NN distance computation
│   │       │   ├── distorted_mesh.rs   # Tessellated mesh for distorted cameras
│   │       │   └── pipelines/    # Per-pass pipeline creation
│   │       │       ├── mod.rs          # Pipeline module re-exports
│   │       │       ├── points.rs       # Point splat pipeline
│   │       │       ├── edl.rs          # Eye-Dome Lighting post-process pipeline
│   │       │       ├── frustum.rs      # Frustum wireframe pipeline
│   │       │       ├── image_quad.rs   # Pinhole image quad pipeline (instanced)
│   │       │       ├── distorted_quad.rs # Distorted image quad pipeline (indexed)
│   │       │       ├── bg_image.rs     # Background image pipeline (pinhole)
│   │       │       ├── bg_distorted.rs # Background image pipeline (distorted/fisheye)
│   │       │       ├── target.rs       # Target indicator pipeline
│   │       │       └── track_ray.rs    # Track ray pipeline
│   │       ├── state.rs          # Shared application state (AppState)
│   │       ├── image_browser.rs  # Thumbnail strip with horizontal scrolling
│   │       ├── image_detail.rs   # Full-resolution image display panel
│   │       ├── bin/              # DirectManipulation test binaries
│   │       ├── platform/
│   │       │   ├── mod.rs
│   │       │   └── windows.rs    # DirectManipulation touchpad integration
│   │       └── shaders/
│   │           ├── points.wgsl         # Point splat rendering
│   │           ├── edl.wgsl            # Eye-Dome Lighting post-process
│   │           ├── frustum.wgsl        # Frustum wireframe rendering
│   │           ├── image_quad.wgsl     # Image texture on frustum far plane (pinhole)
│   │           ├── distorted_quad.wgsl # Tessellated image quad (distorted cameras)
│   │           ├── bg_image.wgsl       # Background image rendering (pinhole)
│   │           ├── bg_image_distorted.wgsl # Background image rendering (distorted)
│   │           ├── target_indicator.wgsl # Rotating octahedron at target
│   │           └── track_ray.wgsl       # Track ray visualization
│   │
│   ├── sfmr-colmap/              # COLMAP format read/write
│   │   └── src/
│   │
│   └── sfmtool-py/               # PyO3 bindings
│       └── src/
│           └── lib.rs            # Python module (sfmtool._sfmtool)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `main.rs` | Window creation, wgpu device/surface, DirectManipulation init, event dispatch, egui integration, `egui_dock` layout with three tabs (3D Viewer, Image Browser, Image Detail) |
| `viewer_3d.rs` | `ViewportCamera` (wraps `Camera` from sfmtool-core with FOV/clip planes), orbit/pan/zoom math, WASD fly navigation with Q/E tilt, input handling (mouse/trackpad/keyboard), Alt-mode target control, camera view mode, grid drawing |
| `scene_renderer/` | All GPU pipeline management across ~14 modules: texture creation/resize, point upload, frustum upload (pinhole + distorted), thumbnail texture array atlas loading, uniform updates, multi-pass rendering, GPU readback, per-pass pipeline creation (`pipelines/` subdirectory) |
| `state.rs` | `AppState` struct: reconstruction data, visibility toggles, selected image/points, rendering parameters |
| `image_browser.rs` | Horizontally-scrollable thumbnail strip with click-to-select, double-click to enter camera view, gesture-driven panning, lazy thumbnail loading |
| `image_detail.rs` | Full-resolution image display for the selected camera, with lazy loading and aspect-ratio-preserving fit |

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
  Input:  Octahedron edges + scene depth texture
  Output: EDL output color (additive blending, reads depth for occlusion)

Pass 4:  Track rays (when a 3D point is selected)
  Input:  Edge instance buffer (camera center → nearest ray point) + hw depth
  Output: EDL output color (premultiplied alpha blending, reads depth for occlusion)
```

For detailed specifications of each pass, see:
- [gui-point-cloud-rendering.md](gui-point-cloud-rendering.md) — Points, EDL,
  target indicator, supernova
- [gui-camera-views.md](gui-camera-views.md) — Frustum wireframes, image quads,
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

**Data flow**: When point selection changes, `upload_track_rays` computes an
`EdgeInstance` per observation — it looks up the cached SIFT feature position,
unprojects it through the camera intrinsics to get a world-space ray direction,
then projects the selected 3D point onto that ray. Each `EdgeInstance` stores
`endpoint_a` (camera center) and `endpoint_b` (nearest point on ray).

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
  images loaded on demand (planned)

---

## Platform-Specific Details

### Windows

- **DirectManipulation API**: Precision touchpad gesture recognition (pan,
  pinch, inertia). Requires specific initialization order relative to winit.
  See [gui-viewport-navigation.md](gui-viewport-navigation.md#windows-precision-touchpad-support).
- **DPI awareness**: `SetProcessDpiAwarenessContext` for per-monitor DPI
- **Graphics backend**: DirectX 12 via wgpu

### macOS (Planned)

- Trackpad gestures via native NSEvent / egui's built-in `zoom_delta`
- Metal backend via wgpu

### Linux (Planned)

- Vulkan backend via wgpu
- Touchpad support via libinput (through winit)
