# Camera Views: Frustum Wireframes and Image Projection

This document specifies how cameras from the SfM reconstruction are visualized
in the 3D viewer. Each camera is rendered as a wireframe frustum pyramid showing
its position and field of view, with the camera's photograph projected as a
texture onto the frustum's far plane. Clicking on a frustum selects that camera.

For the point cloud rendering pipeline that these integrate with, see
[gui-point-cloud-rendering.md](gui-point-cloud-rendering.md). For navigation
controls including Alt+click target picking, see
[gui-viewport-navigation.md](gui-viewport-navigation.md).

## Goal

Show each registered camera as a wireframe frustum pyramid in the 3D scene,
with its photograph projected onto the far plane. Support click-to-select via
GPU pick buffer, hover identification, and selection highlighting.

---

## Data Model (`sfmtool-core`)

Each registered camera pose lives in `reconstruction::SfmrImage`:

```rust
pub struct SfmrImage {
    pub name: String,                          // relative path to image file
    pub camera_index: u32,                     // index into cameras[] for intrinsics
    pub quaternion_wxyz: UnitQuaternion<f64>,  // world-to-camera rotation (COLMAP convention)
    pub translation_xyz: Vector3<f64>,         // world-to-camera translation
}
```

Intrinsics are stored separately in `SfmrCamera`:

```rust
pub struct SfmrCamera {
    pub model: String,                         // e.g. "PINHOLE", "SIMPLE_RADIAL"
    pub width: u32,
    pub height: u32,
    pub parameters: HashMap<String, f64>,      // focal_length or focal_length_x/y, etc.
}
```

The relationship is: `recon.images[i]` (an `SfmrImage`) references
`recon.cameras[image.camera_index]`.

There is already a `frustum.rs` module with `compute_frustum_corners()` that takes
raw arrays (`&[f64; 3]` center, `&[f64; 9]` rotation, individual intrinsic floats).
It returns `[f64; 24]` (8 corners × 3 coords: near TL/TR/BR/BL then far TL/TR/BR/BL).

### Viewport Camera (`sfmtool-core`)

The viewport navigation camera is `Camera` (in `sfmtool-core`), wrapped by
`ViewportCamera` (in `sfmtool-gui`) which adds FOV and clip planes:

```rust
pub struct Camera {
    pub position: Point3<f64>,
    pub orientation: UnitQuaternion<f64>,  // world-to-camera rotation (same convention)
    pub target_distance: f64,
}
```

Both `SfmrImage.quaternion_wxyz` and `Camera.orientation` store the **same thing**:
a world-to-camera rotation quaternion. The difference is that `Camera` stores
the world-space position directly, while `SfmrImage` stores it implicitly as
`C = -R^T * t`.

---

## Convention Bridge: Image to World-Space Pose

To render a frustum, we need the camera's world-space position and a
camera-to-world rotation (the inverse of what's stored). The bridge is:

```
world_position   = -R^T * t           (already: SfmrImage::camera_center())
R_world_from_cam = R^T = quat.inverse().to_rotation_matrix()
```

`SfmrImage::camera_to_world_rotation_flat()` produces the rotation as
a row-major `[f64; 9]` that `compute_frustum_corners()` expects.

---

## Frustum Geometry

### World-space rotation

`SfmrImage` provides a helper to produce the camera-to-world rotation in the
format that `compute_frustum_corners()` expects. Defined in `reconstruction.rs`:

```rust
impl SfmrImage {
    pub fn camera_to_world_rotation_flat(&self) -> [f64; 9] {
        let r = self.quaternion_wxyz.inverse().to_rotation_matrix();
        let m = r.matrix();
        [m[(0,0)], m[(0,1)], m[(0,2)],
         m[(1,0)], m[(1,1)], m[(1,2)],
         m[(2,0)], m[(2,1)], m[(2,2)]]
    }
}
```

### Intrinsic access

`SfmrCamera` provides a helper (in `sfmr-format/types.rs`) that handles both
shared-focal-length models (e.g. SIMPLE_RADIAL uses `focal_length`) and
split-focal-length models (e.g. PINHOLE uses `focal_length_x` /
`focal_length_y`):

```rust
impl SfmrCamera {
    pub fn pinhole_params(&self) -> (f64, f64, f64, f64) {
        let (fx, fy) = if let Some(&f) = self.parameters.get("focal_length") {
            (f, f)
        } else {
            (self.parameters["focal_length_x"], self.parameters["focal_length_y"])
        };
        let cx = self.parameters["principal_point_x"];
        let cy = self.parameters["principal_point_y"];
        (fx, fy, cx, cy)
    }
}
```

### GPU data layout for frustum wireframes

With `near_z = 0` the four near-plane corners collapse to a single apex at
the camera center. This gives a 4-sided pyramid with 8 edges:

```
Apex → far TL, Apex → far TR, Apex → far BR, Apex → far BL   (4 side edges)
Far face: TL-TR, TR-BR, BR-BL, BL-TL                          (4 base edges)
```

Each edge is an instance with two 3D endpoints, expanded to a screen-space-width
ribbon quad in the vertex shader (same technique as the target indicator).

```rust
#[repr(C)]
struct FrustumEdge {
    endpoint_a: [f32; 3],
    color_packed: u32,      // R8G8B8A8 — white default, cyan selected
    endpoint_b: [f32; 3],
    frustum_index: u32,     // image index for pick buffer output
}
```

For N cameras: `N * 8` edge instances. At 32 bytes each, 10K cameras = 2.5 MB.

**Open question**: The frustum wireframe doesn't currently indicate which way
is "up" for the camera. Some applications add a small triangle wireframe on
top of the frustum (like an arrowhead pointing up from the TL–TR edge),
making the camera's orientation immediately readable. This would add 2–3
extra edges per frustum.

### Selection highlighting

Selection is encoded directly in the `color_packed` field of each edge during
`upload_frustums()`. When selection changes, the entire edge buffer is
re-uploaded. This is cheap (< 2.5 MB for 10K cameras) and avoids the
complexity of a separate `FrustumMeta` storage buffer.

| State | Color | Alpha |
|-------|-------|-------|
| Default | White (255, 255, 255) | 0.7 |
| Selected | Cyan (0, 255, 255) | 1.0 |

### Frustum stub sizing

```
near_z = 0                                          (apex at camera center)
far_z  = length_scale * frustum_size_multiplier     (stub depth)
```

The intrinsics (fx, fy, width, height) determine the opening angle and
aspect ratio of each pyramid. `frustum_size_multiplier` defaults to `0.5`.

Frustum geometry is re-uploaded when any of these change:
- `length_scale` (UI slider)
- `frustum_size_multiplier` (planned UI slider)
- `selected_image` (click pick)

This is tracked via `prev_frustum_length_scale`, `prev_frustum_size_multiplier`,
and `prev_selected_image` on the `App` struct.

---

## Frustum Shader (`frustum.wgsl`)

Frustums are **scene geometry**, not overlay UI. They write to the hardware
depth buffer and are occluded by points just like any other object in the
scene. This is a deliberate contrast with the target indicator, which uses
glow, depth-aware transparency, and animation.

### Vertex shader

- Reads edge endpoints (world space) and `color_packed` from instance buffer
- Selects endpoint A or B based on quad corner `x` (-1 or 1)
- Projects both endpoints to clip space
- Computes perpendicular direction in NDC for ribbon-quad expansion
- Expands by `line_half_width` pixels (default 1.0 px)
- Passes `frustum_index` as `@interpolate(flat)` for the pick buffer

### Fragment shader

Three outputs:

```wgsl
@location(0) color: vec4<f32>      // premultiplied RGBA from vertex stage
@location(1) linear_depth: f32     // always 0.0 (frustums don't affect EDL)
@location(2) pick_id: u32          // PICK_TAG_FRUSTUM | frustum_index
```

### Three render targets

The render pass uses three color attachments plus hardware depth:

| Attachment | Format | Written by points | Written by frustums / image quads | Purpose |
|------------|--------|-------------------|-----------------------------------|---------|
| `@location(0)` color | Rgba8UnormSrgb | Yes | Yes | Visible color |
| `@location(1)` linear depth | R32Float | Yes (positive) | Yes (**writes 0.0**) | EDL shading + mouse depth readback |
| `@location(2)` pick ID | R32Uint | Yes (`PICK_TAG_POINT \| point3d_index`) | Yes | Entity picking |
| hw depth | Depth32Float | Yes | Yes | Z-test during rendering |

**Frustums and image quads write 0.0 to linear depth:**

Frustums and image quads actively write `0.0` to the linear depth texture.
This is critical: when frustum geometry occludes a point, the point's positive
depth value must be cleared — otherwise the EDL shader would apply shading
to what is visually frustum geometry. Writing `0.0` ensures the EDL shader's
`depth <= 0` check correctly identifies these pixels and skips EDL.

Mouse depth readback also benefits: `depth == 0` means Alt+click snaps to
points, not wireframes or image quads.

### EDL shader compatibility

The EDL post-process shader (`edl.wgsl`) uses a three-way check:

```
alpha == 0    → background (no points or frustums rendered here)
depth == 0    → frustum/image quad pixel (composite over background without EDL)
depth > 0     → point cloud pixel (apply full EDL shading)
```

### Render order

```
Pass 0:  Background image   (camera view only; writes edl_output, see below)
Pass 1a: Point splats        (writes color + hw depth + linear depth + pick ID)
Pass 1b: Frustum wireframes  (writes color + hw depth + 0.0 depth + pick ID)
Pass 1c: Image quads         (writes color + hw depth + 0.0 depth + pick ID)
Pass 2:  EDL post-process    (reads color + linear depth, writes edl_output)
Pass 3:  Target indicator    (reads linear depth, writes edl_output color, no pick)
```

Pass 0 only runs when in camera view mode. It renders the selected camera's
full-resolution photograph into `edl_output_view`, cleared to `BG_COLOR` so
that letterbox/pillarbox bars match the normal background. The EDL pass then
uses `LoadOp::Load` (instead of `Clear`) to preserve this background, and
`discard`s empty pixels so the image shows through. See
[Full-resolution image background](#full-resolution-image-background) for
details.

---

## Pick Buffer

### Pick ID encoding

The pick buffer stores a single `u32` per pixel:

```
Bits 31..24:  entity type tag (8 bits → 256 types)
Bits 23..0:   entity index   (24 bits → 16M entities per type)
```

| Tag | Constant | Entity type | Index meaning |
|-----|----------|-------------|---------------|
| `0x00` | `PICK_TAG_NONE` | None (background) | — |
| `0x01` | `PICK_TAG_FRUSTUM` | Frustum / camera | Image index into `recon.images` |
| `0x02` | `PICK_TAG_POINT` | 3D point | Point3D index into `recon.points` |

### Unified hover + click readback

A single pair of 5×5 staging buffers serves both hover (every frame) and click.
Every frame, the pixel under the cursor is read back:

1. `copy_readback_region(encoder, cx, cy)` — copies a 5×5 pixel region from
   **both** the linear depth texture and the pick buffer to staging buffers.
2. `read_readback_result(device)` — maps both staging buffers, searches for
   valid values using center → 3×3 → 5×5 priority, caches results for hover
   overlay, and returns a `ReadbackResult` for click handling:

```rust
pub struct ReadbackResult {
    pub depth: Option<f32>,          // nearest valid depth in 5x5 region
    pub pick: Option<(u32, u32)>,    // (tag, index) of picked entity
}
```

Cached accessors for the hover overlay:
- `hover_depth() -> Option<f32>` — depth under cursor
- `hover_pick_id() -> u32` — raw pick ID (tag | index) under cursor

The 5×5 fuzzy search makes it easy to hover over and click on thin wireframe lines.
Both hover tooltip and click selection use the same readback result.

### Click handling in Viewer3D

A single `pending_click: Option<[u32; 2]>` field with a `pending_click_is_alt: bool`
flag tracks whether a click needs to be resolved.

On click result:
- **Alt+Click**: Uses `depth` to set orbit target via `apply_pick_result()`
- **Any click**: Uses `pick` to select frustums
  - Clicking a frustum selects it (`selected_image = Some(idx)`).
    Clicking an already-selected frustum keeps it selected (no toggle).
  - Double-clicking a frustum enters camera view (or switches camera if
    already in camera view) and selects the frustum.
  - Clicking background (non-Alt) deselects

### Why GPU picking over CPU ray testing

- **Occlusion-correct**: Depth test ensures you pick what you see.
- **Scales to all entity types**: Each shader just writes its own tag
  (points write `PICK_TAG_POINT`, frustums write `PICK_TAG_FRUSTUM`).
  New entity types only need a new tag constant.
- **Cheap**: One extra `R32Uint` render target adds ~8 MB at 1080p.

---

## AppState Fields

```rust
pub struct AppState {
    // ... existing fields ...

    /// Whether to show camera frustum wireframes and images.
    pub show_camera_images: bool,     // checkbox: "Show Camera Images"

    /// Currently selected image index (set by click-picking a frustum).
    pub selected_image: Option<usize>,

    /// Frustum visualization depth multiplier.
    /// far_z = length_scale * frustum_size_multiplier.
    pub frustum_size_multiplier: f32,  // default 0.5
}
```

---

## Image Textures in Frustums (Step 3)

### Concept

For each camera, render its image as a textured quad on the frustum's far plane.
The four far-plane corners from `compute_frustum_corners()` define the quad
vertices. UV coordinates are derived from the quad corner positions.

### Rendering approach

A separate sub-pass (Pass 1c) from the wireframe edges:

```
Per-frustum textured quad:
  - 4 vertices (triangle strip, reuses point quad vertex buffer)
  - 1 instance per camera (4 world-space corners + frustum_index)
  - Texture: 2D texture array atlas with thumbnails packed in grid pages
  - Rendered opaque (alpha 1.0, no blending)
```

### GPU data layout

```rust
#[repr(C)]
struct ImageQuadInstance {
    corner_tl: [f32; 3],
    frustum_index: u32,     // atlas grid index + pick ID
    corner_tr: [f32; 3],
    _pad0: u32,
    corner_bl: [f32; 3],
    _pad1: u32,
    corner_br: [f32; 3],
    _pad2: u32,
}
```

64 bytes per instance. For N cameras: N × 64 bytes (640 KB for 10K cameras).

### Image quad pipeline

- **Depth test**: ON (occluded by points in front)
- **Depth write**: ON (occludes points behind)
- **Linear depth**: Writes 0.0 (same as frustum wireframes — no EDL effect)
- **Pick ID**: Writes `PICK_TAG_FRUSTUM | frustum_index` (same entity as wireframes)
- **Blending**: None (rendered opaque)

### Shader (`image_quad.wgsl`)

Bind group:
- `@binding(0)` Uniforms (view_proj, atlas_cols, atlas_rows, images_per_page)
- `@binding(1)` `texture_2d_array<f32>` — thumbnail texture array atlas (multi-page grid)
- `@binding(2)` Sampler (linear filtering)

Vertex shader: bilinear interpolation of 4 corners based on quad position.
Fragment shader: computes page (layer) and within-page atlas UV from
frustum_index, images_per_page, and grid dimensions, then samples the
texture array atlas. Outputs opaque color to 3 targets.

### Thumbnail loading

On reconstruction load, `upload_thumbnails()` synchronously:

1. Computes atlas grid dimensions constrained by the GPU's
   `max_texture_dimension_2d` limit. When one page isn't enough (e.g.
   8192px limit → 64×64 = 4096 cells, but >4096 images), creates a
   `texture_2d_array` with multiple pages (layers). Each page has the
   same cols×rows grid layout.
2. For each image, reads the embedded 128×128 thumbnail from the `.sfmr`
   file, converts RGB → RGBA, and uploads to the correct page and grid
   cell (page = `i / images_per_page`, cell within page = `i % images_per_page`)
3. Creates the image quad bind group (uniforms + texture array view + sampler)

The shader receives `images_per_page` as a uniform so it can compute the
texture array layer and within-page UV from the flat `frustum_index`.

Image quad instances are generated alongside frustum edges in `upload_frustums()`,
but only when `thumbnail_texture` exists. They are re-uploaded on
length_scale / frustum_size_multiplier changes (same as frustum wireframes).

---

## Full-Resolution Image Loading Enhancements (future)

The basic synchronous full-resolution loader (`upload_bg_image()`) is
implemented. Future enhancements:

1. Implement LRU texture cache with memory budget.
2. Load full-res images on demand for nearby/selected cameras.
3. Swap thumbnail with full-res texture when ready.
4. Async thumbnail loading on a background thread.

---

## Viewing Through a Camera

When a frustum is selected and the user presses Z, the viewport transitions to
view through that camera — showing exactly what the camera saw, with its
photograph as the background.

### Viewing camera N (common operation)

Several actions cause the viewport to view through a specific camera. The
behavior differs depending on whether the viewport is already in camera view.

**Entering from non-camera-view** (Z key, first double-click):

1. **Set the viewport camera pose** to match camera N's extrinsics
   (position from `camera_center()`, orientation from `quaternion_wxyz`)
2. **Set `target_distance`** to the median depth of points visible to this
   camera, computed from the per-image depth histogram stored in the `.sfmr`
3. **Set FOV** to `best_fit_fov()` for perspective cameras; leave unchanged
   for fisheye cameras (see [Step 9 FOV handling](#fov-handling-on-enter))
4. **Set camera view mode** — stores the camera's `image_index` and
   world-from-camera rotation
5. **Trigger full-res image load** for camera N (via `upload_bg_image()`)

**Switching between cameras** (`,`/`.` keys, double-click while in camera view):

1. **Keep the current FOV** unchanged
2. **Compute the relative orientation** of the viewport with respect to the
   outgoing SfM camera, then apply it to the new SfM camera (see
   [Step 9 camera-to-camera transitions](#camera-to-camera-transitions))
3. **Set the viewport position** to the new camera's center
4. **Set `target_distance`** from the new camera's median depth
5. **Transform `world_up`** through the same relative rotation
6. **Update `CameraViewMode`** with the new image index and rotation
7. **Trigger full-res image load** for camera N

This preserves the user's relative viewing direction across camera switches.

The `,`/`.` keys also update `selected_image` to match the new camera,
keeping the image browser strip and 3D viewport highlight in sync with the
viewed camera.

### Entering camera view

When Z is pressed with a frustum selected, the [view camera N](#viewing-camera-n-common-operation)
operation is performed with N = the selected frustum's image index.

### FOV best-fit

The camera's intrinsics define two fields of view:

```
vfov_cam = 2 * atan(height / (2 * fy))
hfov_cam = 2 * atan(width  / (2 * fx))
```

While in camera view mode, the viewport FOV is computed each frame so that
the camera's entire field of view fits within the viewport — the image is as
large as possible without any part being cropped. This is the same logic as
fitting a rectangle inside another rectangle: match the constraining axis.
Recomputing each frame keeps the fit correct when the window is resized.

The viewport has its own aspect ratio (`viewport_width / viewport_height`).
For a given vertical FOV, the viewport's horizontal FOV is determined by:
`hfov_viewport = 2 * atan(tan(vfov/2) * viewport_aspect)`. The best-fit
chooses the vertical FOV that makes the camera's image fit exactly on the
constraining axis:

```
# Try fitting by vertical FOV
vfov = vfov_cam
hfov_at_vfov = 2 * atan(tan(vfov/2) * viewport_aspect)

if hfov_at_vfov >= hfov_cam:
    # Image fits horizontally — use vfov_cam
    # (image fills full height, pillarbox bars on sides if wider viewport)
    viewport_vfov = vfov_cam
else:
    # Image is wider than viewport can show at vfov_cam —
    # fit by horizontal FOV instead
    # (image fills full width, letterbox bars top/bottom)
    viewport_vfov = 2 * atan(tan(hfov_cam/2) / viewport_aspect)
```

The resulting `viewport_vfov` is the vertical FOV for the projection matrix.
To store it in the `fov` field (which uses the min-dimension convention), we
convert: in landscape viewports `fov = viewport_vfov` directly; in portrait
viewports `fov = 2 * atan(tan(viewport_vfov/2) * viewport_aspect)`.

No special rendering mode is needed — the standard projection matrix with
the viewport's own aspect ratio is used. The image naturally fills the
constraining axis and has bars on the other. 3D geometry aligns with the
photograph because the FOVs match on the fitted axis and the projection
is correct on both axes.

### Full-resolution image background

The selected camera's photograph is drawn as a full-resolution textured quad
behind the 3D scene. The quad is positioned and sized so that it exactly fills
the image's portion of the viewport (the area inside the letterbox/pillarbox
bars). This requires loading the image at full resolution (not the 128×128
thumbnail used for frustum far-plane textures).

#### Image loading

`upload_bg_image()` on `SceneRenderer` handles full-resolution image loading.
Only one full-res image is loaded at a time (the viewed camera). The loader
is synchronous — it decodes the image from disk and creates the GPU texture
in the same frame. This is simple and works well for typical image sizes;
async loading may be added later for very large images (see
[enhancements](#full-resolution-image-loading-enhancements-future)).

The loader state:

```
bg_image_texture: Option<Texture>       // GPU texture with full-res image
bg_image_loaded_index: Option<usize>    // which camera index is loaded
```

When viewing camera N:
- If `bg_image_loaded_index == Some(N)`: texture is already loaded, skip
- Otherwise: load from disk, decode to RGBA, create GPU texture at full
  resolution, rebuild bind group

When exiting camera view: `clear_bg_image()` releases the texture and bind
group. (Future enhancement: retain the texture for faster re-entry.)

The GPU texture is sized to the actual image dimensions — no power-of-2
padding required on modern GPUs.

#### Rendering the background image

The background image renders into the EDL output texture (`edl_output_view`)
in a new pass that runs before the EDL post-process. The EDL pass then
preserves this pre-rendered background for pixels where no geometry was
rendered.

**Background pass** (new, only runs when `camera_view` is active):
- Target: `edl_output_view`
- Clear: `BG_COLOR` (the standard dark background — fills the letterbox/
  pillarbox bars)
- Draw: a quad textured with the full-res image (or thumbnail placeholder),
  positioned to cover exactly the image's NDC region
- No depth test, no depth write — this pass only touches `edl_output_view`
- Uses a simple shader: vertex positions the quad in NDC, fragment samples
  the image texture

**EDL pass changes:**
- When `camera_view` is active: `LoadOp::Load` (preserves the background
  rendered in the previous pass)
- When `camera_view` is inactive: `LoadOp::Clear(BG_COLOR)`
- EDL pipeline uses **premultiplied alpha blending** on the color attachment:
  `src + dst × (1 − src_alpha)`. This composites the EDL output over
  whatever is already in the render target — either the pre-rendered image
  (camera view) or `BG_COLOR` (normal view).
- EDL shader outputs premultiplied color + alpha instead of compositing
  over `BG_COLOR` itself:
  - Empty pixels (`alpha <= 0`): `discard`. Preserves the render target
    contents (image or clear color).
  - Frustum pixels (`depth <= 0`): `vec4(color.rgb, alpha)`. The
    `color.rgb` from Pass 1 is already premultiplied.
  - Point pixels (`depth > 0`): apply EDL shading and supernova, then
    output `vec4((color.rgb * shade_factor + supernova) * alpha, alpha)`.
    The blend hardware adds `background × (1 − alpha)`, reproducing the
    same composite formula but against the actual background content.

The `BG_COLOR` constant is not used for compositing in the shader.
It remains only in the `LoadOp::Clear` value for normal (non-camera-view)
rendering.

The complete render pass order in camera view mode becomes:

```
Pass 0:  Background image   (writes edl_output; clear BG_COLOR + draw image quad)
Pass 1a: Point splats        (writes color + hw depth + linear depth + pick ID)
Pass 1b: Frustum wireframes  (writes color + hw depth + 0.0 depth + pick ID)
Pass 1c: Image quads         (writes color + hw depth + 0.0 depth + pick ID)
Pass 2:  EDL post-process    (reads color + linear depth, writes edl_output via Load)
Pass 3:  Target indicator    (reads linear depth, writes edl_output color, no pick)
```

When not in camera view, Pass 0 is skipped and the render order is unchanged.

#### Background mesh shader

The background uses a tessellated mesh with vertices on the unit sphere
(ray directions from `pixel_to_ray`). The shader projects through a
`view_proj` matrix and forces depth to the far plane:

```wgsl
struct BgUniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> bg: BgUniforms;
@group(0) @binding(1) var bg_texture: texture_2d<f32>;
@group(0) @binding(2) var bg_sampler: sampler;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    let clip = bg.view_proj * vec4<f32>(position, 0.0);
    out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);  // depth = far plane
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(bg_texture, bg_sampler, in.uv);
}
```

Single render target (`edl_output_view`), no depth, no blending (opaque).
Using `w=0` transforms the position as a direction (no translation), so the
mesh appears infinitely far away. Vertices are pre-rotated to world space
during mesh generation, so `view_proj` is the standard `projection * view`
used by all other pipelines. Free-look navigation works automatically
because the viewport camera's orientation is captured in the view matrix.

### Navigating between cameras

While in camera view mode, the `,` and `.` keys step through camera images
sequentially:

| Key | Action |
|-----|--------|
| `,` | Switch to the previous camera image (by index in `recon.images`) |
| `.` | Switch to the next camera image (by index in `recon.images`) |

Navigation wraps around: pressing `.` on the last image goes to the first,
and pressing `,` on the first image goes to the last.

Each step performs the [view camera N](#viewing-camera-n-common-operation)
operation with the new image index. The transition is immediate (no
animation), matching the quick scrubbing intent of `,`/`.` for reviewing
a sequence of images.

These keys are chosen to match the `<`/`>` mnemonics (same physical keys
without Shift) commonly used for previous/next in video editors and image
viewers. They are also the Vim keys for paragraph navigation, reinforcing
the "step through a sequence" mental model.

Outside camera view mode, `,`/`.` still step through images — they move
`selected_image` back/forward with the same wrap-around — but the 3D
viewport is left alone. This lets you scrub through the image browser /
image detail panel without the viewport jumping around.

### Exiting camera view

Camera view exits when navigation moves the camera center — orbiting, panning,
dolly zoom, or WASD fly movement. Orientation-only navigation (nodal pan /
free-look, tilt, FOV changes, target distance changes) keeps camera view
active. See [Step 9](#step-9-persistent-camera-view-with-free-look-navigation)
for the full specification of which inputs keep vs exit camera view.

When camera view exits, the mode flag is cleared and the image background is
hidden. The camera's pose and FOV are kept as-is. Navigation continues
smoothly from that state.

There's no special exit action; you just start moving the camera center.
Pressing Z again (with no frustum selected) performs zoom-to-fit as usual.
Pressing Home also exits camera view (explicit reset).

---

## Distorted Frustum Rendering

### Motivation

The current frustum visualization assumes a pinhole camera: the image quad is a
flat rectangle and the wireframe base is four straight edges. Real cameras have
lens distortion (barrel, pincushion, or more complex), which means the actual
field of view is not rectangular. For cameras solved with distortion models
(SIMPLE_RADIAL, RADIAL, OPENCV, etc.), the frustum should show the true
projected shape of the image boundary.

The approach is to tessellate the frustum far plane into a grid mesh, with each
vertex placed along the ray corresponding to its pixel after accounting for
distortion. For pinhole cameras (no distortion), this produces the same flat
quad as today. For distorted cameras, the mesh curves to show the actual FOV
shape.

---

### Distortion Convention (COLMAP)

COLMAP's camera model projects a 3D point to pixel coordinates in three steps:

```
1. Normalize:    x = X/Z,  y = Y/Z
2. Distort:      (x_d, y_d) = distort(x, y, k1, k2, ...)
3. Pixel coords: u = fx * x_d + cx,  v = fy * y_d + cy
```

To go from a pixel (u, v) back to a 3D ray direction, we reverse this:

```
1. Normalized distorted:  x_d = (u - cx) / fx,  y_d = (v - cy) / fy
2. Undistort:             (x, y) = undistort(x_d, y_d, k1, k2, ...)
3. Ray direction:         normalize(x, y, 1)
```

The undistortion step (step 2) requires iterative solving because the distortion
function is not analytically invertible. Newton's method converges in 5-10
iterations for typical lens parameters.

The current `compute_frustum_corners()` only performs steps 1 and 3 (skipping
undistortion), which is correct for PINHOLE/SIMPLE_PINHOLE but wrong for
distorted models — it treats distorted normalized coordinates as if they were
undistorted ray directions.

---

### Undistortion by Camera Model

All iterative undistortion uses the same pattern: given distorted coordinates
(x_d, y_d), find undistorted (x, y) such that `distort(x, y) == (x_d, y_d)`.
Initialize with (x, y) = (x_d, y_d) and iterate:

```
for _ in 0..max_iterations:
    (x_d_est, y_d_est) = distort(x, y)
    x += x_d - x_d_est
    y += y_d - y_d_est
    if |x_d - x_d_est| + |y_d - y_d_est| < 1e-10:
        break
```

This is a fixed-point iteration (equivalent to Newton's method for the
identity-plus-correction form of these distortion models). It converges
reliably for typical distortion magnitudes.

**PINHOLE / SIMPLE_PINHOLE**: No distortion. (x, y) = (x_d, y_d).

**SIMPLE_RADIAL** (k1):

```
r² = x² + y²
x_d = x * (1 + k1 * r²)
y_d = y * (1 + k1 * r²)
```

**RADIAL** (k1, k2):

```
r² = x² + y²
x_d = x * (1 + k1 * r² + k2 * r⁴)
y_d = y * (1 + k1 * r² + k2 * r⁴)
```

**OPENCV** (k1, k2, p1, p2):

```
r² = x² + y²
radial = 1 + k1 * r² + k2 * r⁴
x_d = x * radial + 2 * p1 * x * y + p2 * (r² + 2 * x²)
y_d = y * radial + p1 * (r² + 2 * y²) + 2 * p2 * x * y
```

**FULL_OPENCV** (k1, k2, p1, p2, k3, k4, k5, k6): Same structure as
OPENCV but with a rational radial factor `(1 + k1*r² + k2*r⁴ + k3*r⁶) /
(1 + k4*r² + k5*r⁴ + k6*r⁶)`. Can be added straightforwardly when needed.

#### Fisheye Models

COLMAP defines 7 fisheye camera models, all sharing the same
**equidistant base projection** via `BaseFisheyeCameraModel`. The shared
projection pipeline is:

```
Forward (3D → pixel):
  1. Perspective normalize:  (u, v) = (X/Z, Y/Z)
  2. Equidistant mapping:    r = sqrt(u² + v²), θ = atan(r)
                             (uu, vv) = (u, v) * θ/r
  3. Distortion (additive):  (uu_d, vv_d) = (uu + duu, vv + dvv)
  4. Pixel coordinates:      pixel_x = fx * uu_d + cx
                             pixel_y = fy * vv_d + cy

Inverse (pixel → 3D ray):
  1. Remove focal/center:    (uu_d, vv_d) = ((px - cx)/fx, (py - cy)/fy)
  2. Undistort:              (uu, vv) = undistort(uu_d, vv_d)
  3. Inverse equidistant:    θ = sqrt(uu² + vv²)
                             scale = sin(θ) / (θ * cos(θ))  [= tan(θ)/θ]
                             (u, v) = (uu, vv) * scale
  4. Ray direction:          (u, v, 1)
```

Step 3 (inverse equidistant) produces `(u, v)` in perspective image-plane
coordinates where `u = tan(θ)`. This is why all fisheye models have the
flat far-plane degeneracy described in
[Fisheye Camera Rendering](#fisheye-camera-rendering).

Our `distort_fisheye()` in `distortion.rs` combines steps 2+3 of the forward
path into a single function operating on perspective coordinates `(x, y)`:
`theta_d = theta * (1 + k1*θ² + ...)`, `scale = theta_d / r`. This is
mathematically equivalent to COLMAP's separated approach.

The 5 COLMAP fisheye models differ only in the distortion applied in step 3
(in equidistant/theta space) and in focal length parameterization:

| COLMAP Model | Focal | Distortion Params | Distortion Formula |
|---|---|---|---|
| `SIMPLE_RADIAL_FISHEYE` | f | k | radial: `k*θ²` |
| `RADIAL_FISHEYE` | f | k1, k2 | radial: `k1*θ² + k2*θ⁴` |
| `OPENCV_FISHEYE` | fx, fy | k1, k2, k3, k4 | radial: `k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸` |
| `THIN_PRISM_FISHEYE` | fx, fy | k1,k2,p1,p2,k3,k4,sx1,sy1 | radial + tangential + thin prism (see below) |
| `RAD_TAN_THIN_PRISM_FISHEYE` | fx, fy | k0..k5,p0,p1,s0..s3 | 6th-order radial + tangential + thin prism (see below) |

**SIMPLE_RADIAL_FISHEYE** (f, cx, cy, k): Equidistant projection with one
radial distortion coefficient. The fisheye analog of SIMPLE_RADIAL. Distortion
in theta-space: `duu = uu * k*θ²`, `dvv = vv * k*θ²` where `θ² = uu² + vv²`.

**RADIAL_FISHEYE** (f, cx, cy, k1, k2): Equidistant projection with two
radial distortion coefficients. The fisheye analog of RADIAL. Distortion:
`duu = uu * (k1*θ² + k2*θ⁴)`.

**OPENCV_FISHEYE** (fx, fy, cx, cy, k1, k2, k3, k4): Equidistant projection
with four radial distortion coefficients. Currently the only fisheye model
implemented in sfmtool. Distortion:
`duu = uu * (k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)`.

**THIN_PRISM_FISHEYE** (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1):
Equidistant projection with radial, tangential, and thin-prism distortion.
Distortion in theta-space:

```
θ² = uu² + vv²;  θ⁴ = θ²²;  θ⁶ = θ⁴*θ²;  θ⁸ = θ⁴²
radial = k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸
duu = uu*radial + 2*p1*uu*vv + p2*(θ² + 2*uu²) + sx1*θ²
dvv = vv*radial + 2*p2*uu*vv + p1*(θ² + 2*vv²) + sy1*θ²
```

**RAD_TAN_THIN_PRISM_FISHEYE** (fx, fy, cx, cy, k0..k5, p0, p1, s0..s3):
Meta/Aria fisheye model with 6th-order radial, tangential, and thin-prism
distortion. 16 total parameters. The distortion is more complex: radial
coefficients apply as `th_radial = 1 + k0*θ² + k1*θ⁴ + ... + k5*θ¹²`, then
tangential and thin-prism terms are applied to the radially-distorted
coordinates. See COLMAP source or the
[Aria documentation](https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models)
for the full formula.

---

### Tessellated Image Quad

#### Grid geometry

The image quad for each camera is tessellated into an `N × N` vertex grid
(`(N-1) × (N-1)` cells, each split into 2 triangles). `N` is a compile-time
constant, recommended **N = 5** (4 subdivisions per edge).

For each camera, grid vertex `(i, j)` where `i, j ∈ [0, N-1]` corresponds to
pixel coordinates:

```
u = (i / (N-1)) * width
v = (j / (N-1)) * height
```

The vertex's world-space position is computed by:

1. Normalized distorted coords: `x_d = (u - cx) / fx`, `y_d = (v - cy) / fy`
2. Undistort to get the true ray direction: `(x, y) = undistort(x_d, y_d)`
3. Camera-space point on far plane: `p_cam = (x, y, 1.0) * (far_z / 1.0)`
   (no normalization; scale so z = far_z)
4. Transform to world space: `p_world = camera_center + R_cam_to_world * p_cam`

UV coordinates for texture lookup:

```
uv = (i / (N-1), j / (N-1))
```

This maps the full thumbnail texture across the mesh. The UVs are regular even
though the geometry is warped — the distortion is "baked into" the vertex
positions, and the texture (which is the actual distorted photograph) maps
onto the curved surface correctly.

#### Why N = 5

Lens distortion is smooth and low-frequency. A 4-subdivision grid (25 vertices)
captures the curvature well while keeping triangle counts low for scalability
to large camera counts. Comparison:

| N | Vertices/camera | Triangles/camera | Edges (boundary) |
|---|-----------------|------------------|-------------------|
| 2 | 4 | 2 | 4 (same as current) |
| 5 | 25 | 32 | 16 |
| 9 | 81 | 128 | 32 |
| 17 | 289 | 512 | 64 |

N = 5 is the sweet spot — visually smooth curves with minimal geometry
overhead. N = 2 reproduces the current flat-quad behavior exactly.

#### GPU data layout

The instanced `ImageQuadInstance` with 4 corners is replaced by a shared vertex
buffer + index buffer approach:

```rust
#[repr(C)]
struct DistortedQuadVertex {
    position: [f32; 3],   // world-space position
    uv: [f32; 2],         // texture coordinate (0..1)
    frustum_index: u32,   // image index for atlas grid lookup + pick ID
}
// 24 bytes per vertex
```

All cameras' meshes are concatenated into a single vertex buffer and a single
index buffer. Each camera contributes `N * N` vertices and `(N-1) * (N-1) * 6`
indices.

The index buffer uses `u32` indices. For camera `k`, its vertices start at
offset `k * N * N` in the vertex buffer, and its indices are pre-offset to
reference the correct vertices.

A single `draw_indexed()` call renders all image quads.

#### Memory estimates (N = 5)

| Item | Per camera | 10K cameras |
|------|-----------|-------------|
| Vertices (25 × 24 bytes) | 600 B | 5.9 MB |
| Indices (32 × 3 × 4 bytes) | 384 B | 3.8 MB |
| **Total** | ~1 KB | **~9.7 MB** |

Compare to current: 64 bytes per camera (640 KB for 10K). The tessellated
approach uses ~15× more memory for the mesh, but this is still very modest
compared to the thumbnail texture atlas (625 MB for 10K cameras at 128×128
RGBA).

#### Shader changes (`distorted_quad.wgsl`)

The shader simplifies — no more bilinear corner interpolation in the vertex
shader. Vertex positions and UVs come directly from the vertex buffer:

```wgsl
struct Uniforms {
    view_proj: mat4x4<f32>,
    atlas_cols: u32,
    atlas_rows: u32,
    images_per_page: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var thumbnail_texture: texture_2d_array<f32>;
@group(0) @binding(2) var thumbnail_sampler: sampler;

const PICK_TAG_FRUSTUM: u32 = 0x01000000u;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) frustum_index: u32,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) frustum_index: u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = uniforms.view_proj * vec4(in.position, 1.0);
    out.uv = in.uv;
    out.frustum_index = in.frustum_index;
    return out;
}

// Fragment shader computes page and atlas UV from frustum_index,
// images_per_page, and grid dimensions, samples texture array, outputs pick ID
```

The pipeline vertex layout changes from per-vertex `quad_pos` + per-instance
4-corner attributes to a single per-vertex layout with position, UV, and
frustum index.

---

### Tessellated Frustum Wireframe

The frustum wireframe edges are updated to follow the distorted boundary:

**Side edges** (4 total): Unchanged — straight lines from the camera center
(apex) to the four corner vertices of the tessellated grid. These are rays
through specific pixels, which are straight lines in 3D regardless of
distortion.

**Base edges**: Instead of 4 straight edges around the far face, the base
outline follows the tessellated grid boundary. For an `N × N` grid, the
boundary has `4 * (N - 1)` edge segments:

```
Top edge:    (0,0)→(1,0)→...→(N-1,0)           N-1 segments
Right edge:  (N-1,0)→(N-1,1)→...→(N-1,N-1)     N-1 segments
Bottom edge: (N-1,N-1)→(N-2,N-1)→...→(0,N-1)   N-1 segments
Left edge:   (0,N-1)→(0,N-2)→...→(0,0)          N-1 segments
```

For N = 5: 4 side edges + 16 base segments = **20 edges per camera** (vs 8
currently).

The existing `FrustumEdge` struct and frustum shader are reused without change.
The only difference is that `upload_frustums()` emits more edges per camera,
with the base edge endpoints taken from the tessellated grid's boundary
vertices.

#### Memory estimate (N = 5)

| Item | Per camera | 10K cameras |
|------|-----------|-------------|
| Edges (20 × 32 bytes) | 640 B | 6.3 MB |

Compare to current: 8 × 32 = 256 bytes per camera (2.5 MB for 10K).

---

### Pinhole Fast Path

Cameras with no effective distortion use the existing untessellated rendering
path: a flat 4-vertex image quad (via `ImageQuadInstance`) and 8 straight
wireframe edges. This saves memory and draw complexity for the common case
where many or all cameras are pinhole.

A camera is considered "effectively pinhole" when either:

1. **Model type**: `Pinhole` or `SimplePinhole` (no distortion parameters).
2. **Zero distortion parameters**: Any distortion model (e.g. `SimpleRadial`,
   `Radial`, `OpenCV`) where all distortion coefficients are zero (or below
   a small epsilon like `1e-12`). This handles cases where a solver outputs
   a distortion-capable model but converges to zero distortion.

`CameraModel` provides an `has_distortion() -> bool` method that implements
this check. It returns `false` for pinhole models and for distortion models
with all-zero coefficients.

During `upload_frustums()`, each camera is checked via `has_distortion()`:

- **No distortion**: Use the current code path — `compute_frustum_corners()`
  for the 4 corner positions, `ImageQuadInstance` for the image quad, and
  8 `FrustumEdge` entries for the wireframe.
- **Has distortion**: Use `compute_distorted_frustum_grid()` for the N×N
  tessellated grid, `DistortedQuadVertex` entries for the image quad mesh,
  and 4 + 4*(N-1) `FrustumEdge` entries for the wireframe.

Both types of image quads are rendered in the same draw pass. The pinhole
quads use the existing instanced pipeline, and the distorted quads use the
vertex/index buffer pipeline. Two `draw` calls in Pass 1c (one instanced,
one indexed) handle the mixed case. If all cameras are pinhole, the indexed
draw is skipped entirely (and vice versa).

---

### CPU-Side Computation

All tessellation and undistortion happens on CPU during `upload_frustums()`.
The GPU receives pre-computed world-space vertex positions — no distortion
math in shaders.

#### Why CPU, not GPU

- **Runs once per upload** (not per frame) — when length_scale, frustum_size,
  or selection changes. Cost is negligible.
- **Avoids shader complexity**: No branching on camera model type in WGSL.
  No iterative Newton's method on GPU.
- **Any distortion model**: New models only require a Rust function, not
  shader changes.
- **Shared vertex buffer**: All cameras' meshes are pre-built into one draw
  call.

#### New Rust API (`sfmtool-core`)

```rust
/// Undistort normalized image coordinates to recover the true ray direction.
///
/// Given distorted normalized coordinates (x_d, y_d) and camera model
/// parameters, returns undistorted (x, y) such that distort(x, y) ≈ (x_d, y_d).
///
/// Returns (x_d, y_d) unchanged for pinhole models (no distortion).
pub fn undistort_point(
    x_d: f64,
    y_d: f64,
    model: &str,
    params: &HashMap<String, f64>,
) -> (f64, f64);

/// Compute a tessellated grid of world-space positions for a camera's
/// far-plane image quad, accounting for lens distortion.
///
/// Returns `N * N` world-space positions as `[f64; N*N*3]` and
/// boundary vertex positions for wireframe edges.
pub fn compute_distorted_frustum_grid(
    camera_center: &[f64; 3],
    r_world_from_cam: &[f64; 9],
    camera: &SfmrCamera,
    far_z: f64,
    subdivisions: usize,   // N-1 (e.g., 8 for an N=9 grid)
) -> DistortedFrustumGrid;

pub struct DistortedFrustumGrid {
    /// World-space positions, row-major: grid[j * N + i] = vertex at (i, j).
    /// Length: N * N * 3.
    pub positions: Vec<f64>,
    /// Grid dimension (N = subdivisions + 1).
    pub grid_size: usize,
}
```

The `DistortedFrustumGrid` provides all the data needed to build both the
image quad vertex/index buffers and the wireframe edge buffer.

#### Integration into `upload_frustums()`

The existing `upload_frustums()` method changes as follows:

1. For each camera, call `compute_distorted_frustum_grid()` instead of
   `compute_frustum_corners()`.
2. **Wireframe edges**: Extract the 4 corner vertices for side edges (apex to
   grid corners at `(0,0)`, `(N-1,0)`, `(N-1,N-1)`, `(0,N-1)`). Walk the
   grid boundary to emit base edge segments.
3. **Image quad mesh**: Iterate the grid to build `DistortedQuadVertex` entries
   (position + UV + frustum_index). Build triangle indices for `(N-1)*(N-1)*2`
   triangles per camera.

For pinhole cameras, `undistort_point()` is a no-op, and the resulting mesh
is a flat quad identical to the current behavior.

---

### Effect on Background Image (Camera View Mode)

For perspective-like cameras (SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV),
the existing distorted background mesh (`generate_bg_distorted_mesh`) works
correctly — the distortion is modest enough that `unproject` returns bounded
image-plane coordinates, and the tessellated mesh renders properly.

For fisheye cameras, the background image requires the spherical mesh approach
described in [Fisheye Camera Rendering](#fisheye-camera-rendering).

---

### Effect on Pick Buffer

No change to pick buffer behavior. All vertices in a camera's tessellated mesh
write the same `PICK_TAG_FRUSTUM | frustum_index`, so clicking anywhere on the
distorted quad selects the same camera as before.

---

## Fisheye Camera Rendering

### Problem: Flat Far-Plane Degeneracy

The existing distorted frustum pipeline places grid vertices on a **flat plane
at z = far_z** in camera space:

```rust
let (x, y) = camera.unproject(u, v);
let p_cam = Vector3::new(x * far_z, y * far_z, far_z);
```

For perspective-like cameras (SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV),
the unprojected `(x, y)` values are bounded because FOV is typically < 120°.
The flat far-plane works well.

For **OPENCV_FISHEYE** cameras, `unproject` returns image-plane coordinates
`(x, y)` where the ray direction is `(x, y, 1)` and `x = tan(theta)`. At
wide angles:

| θ (degrees) | tan(θ)  | Lateral offset at far_z |
|-------------|---------|-------------------------|
| 45°         | 1.0     | 1.0 × far_z             |
| 70°         | 2.7     | 2.7 × far_z             |
| 80°         | 5.7     | 5.7 × far_z             |
| 85°         | 11.4    | 11.4 × far_z            |
| 89°         | 57.3    | 57.3 × far_z            |
| 90°         | ∞       | ∞                        |

A fisheye camera with 150° FOV has corner pixels at θ ≈ 75°, producing
`tan(75°) ≈ 3.7`. The frustum extends 3.7× farther sideways than it is deep
— a comically flat, wide shape that looks nothing like the camera's actual
field of view.

At θ ≥ 90° (full hemisphere or beyond), `tan(θ)` becomes infinite or
negative, causing the geometry to explode or flip.

### Solution: Spherical Far-Surface

For fisheye cameras, place grid vertices on a **sphere of radius far_z**
centered at the camera, rather than on a flat plane. Each vertex lies along
the correct ray direction at a fixed distance from the camera:

```rust
let (x, y) = camera.unproject(u, v);
let dir = Vector3::new(x, y, 1.0).normalize();
let p_cam = dir * far_z;
```

This produces a **dome-shaped** frustum that accurately represents the
camera's wide FOV. The wireframe boundary traces a curve on the dome surface,
and the image quad texture maps onto the dome correctly.

**Properties of the spherical approach:**

- All vertices are equidistant from the camera center (radius = far_z)
- Works for **any FOV**, including > 180° — rays beyond the hemisphere simply
  point backward, and the mesh wraps around the sphere accordingly. There is
  no angular limit or clamping needed.
- For narrow FOV cameras (θ < 30°), the dome is nearly flat — visually
  indistinguishable from the flat-plane result
- Side edges (apex to corners) remain straight lines (they are rays)
- Base edges between adjacent grid vertices are short chords of the sphere,
  which appear curved when there are enough subdivisions
- From the outside, the frustum of an ultra-wide (e.g., 220°) fisheye
  camera looks like a nearly-closed sphere with a small opening at the back

**When to use spherical vs flat placement:**

The spherical approach is used when `CameraModel::is_fisheye()` returns true
(all 7 fisheye variants). All perspective-like distortion models continue to
use the flat far-plane. This avoids changing the visual appearance of existing
perspective camera frustums.

### Detection: `CameraModel::is_fisheye()`

Add a method to distinguish fisheye from perspective-like models:

```rust
impl CameraModel {
    /// Returns true for camera models that use a fisheye (equidistant)
    /// projection, which requires spherical far-surface placement in the GUI.
    pub fn is_fisheye(&self) -> bool {
        matches!(
            self,
            CameraModel::SimpleFisheye { .. }
            | CameraModel::Fisheye { .. }
            | CameraModel::SimpleRadialFisheye { .. }
            | CameraModel::RadialFisheye { .. }
            | CameraModel::OpenCVFisheye { .. }
            | CameraModel::ThinPrismFisheye { .. }
            | CameraModel::RadTanThinPrismFisheye { .. }
        )
    }
}
```

This is distinct from `has_distortion()` — a fisheye camera with all-zero k
values (or no distortion parameters, like `SimpleFisheye` and `Fisheye`)
still needs spherical placement because its projection model is fundamentally
different (equidistant vs perspective). The detection is by model type, not
distortion magnitude.

### Changes to `compute_distorted_frustum_grid()`

The grid computation in `frustum.rs` is modified to use spherical placement
for fisheye cameras:

```rust
for j in 0..n {
    for i in 0..n {
        let u = (i as f64 / (n - 1) as f64) * w;
        let v = (j as f64 / (n - 1) as f64) * h;
        let (x, y) = camera.unproject(u, v);

        let p_cam = if camera.model.is_fisheye() {
            // Spherical: place on sphere of radius far_z
            let dir = Vector3::new(x, y, 1.0).normalize();
            dir * far_z
        } else {
            // Flat: perspective projection, place on plane at z = far_z
            Vector3::new(x * far_z, y * far_z, far_z)
        };

        let p_world = center + r * p_cam;
        // ...
    }
}
```

No other changes needed — the wireframe edge extraction, image quad mesh
generation, and pick buffer all work unchanged because they operate on the
grid positions regardless of how those positions were computed.

### Increased Subdivisions for Fisheye

The curvature of the spherical far-surface is much higher than the subtle
warp from radial distortion. The default `DISTORTION_SUBDIVISIONS` (N-1 = 4,
giving a 5×5 grid) is sufficient for perspective cameras but produces
visible faceting on fisheye domes.

Use a higher subdivision count for fisheye cameras:

```rust
const DISTORTION_SUBDIVISIONS: usize = 4;      // perspective distortion
const FISHEYE_SUBDIVISIONS: usize = 16;         // fisheye dome

let subdivisions = if camera.model.is_fisheye() {
    FISHEYE_SUBDIVISIONS
} else {
    DISTORTION_SUBDIVISIONS
};
```

A 17×17 grid (16 subdivisions) has 289 vertices and 512 triangles per camera
— still modest. The wireframe boundary has 64 edge segments, producing smooth
curves.

### Changes to Background Image (Camera View Mode)

The background image uses a unified tessellated mesh pipeline for all camera
models. Mesh vertices are unit ray directions on the sphere, computed via
`CameraIntrinsics::pixel_to_ray()`. This avoids the `tan(theta)` singularity
entirely and correctly handles fisheye cameras with FOV at and beyond 180
degrees.

#### Unit-sphere mesh generation

`generate_bg_distorted_mesh` calls `pixel_to_ray` for each grid vertex to
get a unit ray direction in camera local space:

```rust
let ray = camera.pixel_to_ray(s * w, t * h);
vertices.push(BgDistortedVertex {
    position: [ray[0] as f32, ray[1] as f32, ray[2] as f32],
    uv: [s as f32, t as f32],
});
```

For pinhole cameras, the rays point into the forward hemisphere. For fisheye
cameras, the rays can span beyond 90 degrees from the optical axis, wrapping
past the equator of the unit sphere.

#### Shader projection

The vertex shader projects unit-sphere positions through a `view_proj` matrix.
Using `w=0` transforms the position as a direction (no translation effect),
then depth is forced to the far plane:

```wgsl
let clip = bg.view_proj * vec4<f32>(position, 0.0);
out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);
```

For rays with negative z (beyond 90 degrees from optical axis), the GPU
clips triangles that straddle the boundary. Content within the viewport's
perspective FOV renders correctly.

#### Subdivision counts

```rust
const BG_PINHOLE_SUBDIVISIONS: usize = 1;       // no distortion, 4 vertices
const BG_DISTORTION_SUBDIVISIONS: usize = 32;   // perspective + distortion
const BG_FISHEYE_SUBDIVISIONS: usize = 64;      // fisheye (highly non-linear)
```

All cameras use the same pipeline — the subdivision count is the only
difference.

### FOV Handling for Fisheye Cameras

For fisheye cameras, the FOV is **not changed** on entering camera view.
The fisheye image is projected onto a unit sphere and the viewport provides
a perspective window into that sphere. The user can free-look to explore the
full fisheye field of view and adjust the viewport FOV with the slider to
see more or less of the image at once.

This differs from perspective cameras, where `best_fit_fov()` is computed
once on entry so the background image fills the viewport exactly. See
[Step 9 FOV handling](#fov-handling-on-enter) for details.

---

## Implementation Status

### Step 1: Frustum wireframe rendering — DONE

1. ✓ `SfmrImage::camera_to_world_rotation_flat()` helper
2. ✓ `SfmrCamera::pinhole_params()` helper (handles both focal length conventions)
3. ✓ Frustum edge buffer upload (`upload_frustums()` on `SceneRenderer`)
4. ✓ `frustum.wgsl` shader (ribbon-quad expansion with per-edge color)
5. ✓ Frustum render pipeline with three color targets + hw depth
6. ✓ Hooked into render pass (Pass 1b, after points)
7. ✓ EDL shader updated for frustum-only pixels (`color.a` check)
8. ✓ Re-upload on `length_scale` or `frustum_size_multiplier` change

### Step 2: Pick buffer + selection + hover overlay — DONE

1. ✓ `R32Uint` pick texture (created/resized in `ensure_size()`)
2. ✓ Cleared to 0 (`PICK_TAG_NONE`) at frame start
3. ✓ `@location(2)` pick ID output in both `frustum.wgsl` and `points.wgsl`
4. ✓ Points write `PICK_TAG_POINT | instance_index` via `@builtin(instance_index)`
5. ✓ Unified 5×5 readback — shared staging buffers for hover + click
6. ✓ Decode tag + index; select/deselect frustum on click
7. ✓ Selection highlighting via per-edge `color_packed` on re-upload
8. ✓ Hover overlay: bottom-left text shows entity under cursor
   - Point: "Point3D #N | depth: X.XXXX"
   - Frustum: "Camera: image_name"
   - Background with depth: "depth: X.XXXX"

### Step 3: Image textures on frustums — DONE

1. ✓ `ImageQuadInstance` data type (4 corners + frustum_index, 64 bytes)
2. ✓ `image_quad.wgsl` shader (bilinear corner interpolation, texture array atlas sampling)
3. ✓ Image quad pipeline (opaque, depth test ON, depth write ON)
4. ✓ Thumbnail loading (`upload_thumbnails` — reads embedded thumbnails from `.sfmr`, multi-page texture array atlas)
5. ✓ Instance generation in `upload_frustums` (one quad per camera)
6. ✓ Hooked into render pass (Pass 1c, after frustum wireframes)
7. ✓ Pick ID output matches frustum wireframes (same `PICK_TAG_FRUSTUM | index`)

### Step 4: Full-resolution image loading — DONE

1. ✓ Single-image texture loader (`upload_bg_image()` on `SceneRenderer`)
2. ✓ Synchronous image load (decodes from disk, creates GPU texture at full resolution)
3. ✓ Skip reload if same image index already loaded (`bg_image_loaded_index` check)
4. ✓ `clear_bg_image()` releases texture when leaving camera view

### Step 5: View through a camera — DONE

1. ✓ `CameraViewMode` struct with `image_index` and `best_fit_fov()` method
2. ✓ `ViewportCamera::vertical_fov()` helper for min-dimension FOV convention
3. ✓ Z key enters camera view when frustum selected (sets pose, target_distance, intrinsic FOVs)
4. ✓ Per-frame FOV recomputation from stored intrinsic FOVs and viewport aspect
5. ✓ Camera view exits on any navigation input (drag, scroll, gesture, Home key)
6. ✓ Full-resolution image background display
7. ✓ `,`/`.` keys to navigate to previous/next camera image in camera view mode

### Step 6: Background image rendering — DONE

1. ✓ Background mesh pipeline (tessellated unit-sphere mesh, `bg_image_distorted.wgsl`)
2. ✓ Background render pass (Pass 0: clear `BG_COLOR`, draw mesh)
3. ✓ `view_proj` uniform projects unit-sphere ray directions to clip space
4. ✓ EDL pass conditional `LoadOp` (`Load` in camera view, `Clear(BG_COLOR)` otherwise)
5. ✓ EDL shader `discard` for empty pixels (replaces `return vec4(BG_COLOR, 1.0)`)
6. ✓ Bind group for background mesh (uniforms + image texture + sampler)
7. ✓ Integration in `render()` — pass `camera_view` state to control Pass 0 and EDL LoadOp

### Step 7: Distorted frustum rendering — DONE

1. ✓ `CameraModel::has_distortion()` for pinhole fast-path detection
2. ✓ `undistort_point()` with iterative solving for SIMPLE_RADIAL, RADIAL, OPENCV
3. ✓ `compute_distorted_frustum_grid()` for tessellated N×N grid
4. ✓ `DistortedQuadVertex` GPU type and `distorted_quad.wgsl` shader
5. ✓ Mixed pinhole (instanced) + distorted (indexed) rendering in Pass 1c
6. ✓ Distorted background image rendering (`bg_image_distorted.wgsl`)
7. ✓ Pinhole fast path: cameras without distortion use untessellated pipeline

### Step 8: Fisheye camera rendering — DONE

All 7 COLMAP fisheye models share the same equidistant base projection and
need the same spherical far-surface fix. The work is:

1. ✓ Add all fisheye `CameraModel` variants to the enum (currently only
       `OpenCVFisheye` exists):
       `SimpleFisheye`, `Fisheye`, `SimpleRadialFisheye`, `RadialFisheye`,
       `ThinPrismFisheye`, `RadTanThinPrismFisheye`
2. ✓ Implement `distort`/`undistort` for each new variant (the distortion
       formulas operate in theta-space; the equidistant base mapping
       `FisheyeFromNormal`/`NormalFromFisheye` is shared)
3. ✓ `CameraModel::is_fisheye()` detection method (matches all 7 variants)
4. ✓ Spherical far-surface in `compute_distorted_frustum_grid()` for fisheye
5. ✓ Higher subdivision count for fisheye frustums (N=17 vs N=5)
6. ✓ `pixel_to_ray` API on `CameraIntrinsics` — converts pixel coordinates
       to unit ray directions without the `tan(theta)` singularity, supporting
       fisheye FOV at and beyond 180 degrees
7. ✓ Frustum grid and BG mesh both use `pixel_to_ray` for vertex placement
8. ✓ Unified BG mesh pipeline — all camera models (pinhole, distorted,
       fisheye) use the same tessellated unit-sphere mesh and shader. Removed
       the separate pinhole quad pipeline (`bg_image.wgsl`)
9. ✓ BG mesh projected through `view_proj` matrix (prepared for free-look)
10. ✓ Higher subdivision count for fisheye background mesh (N=65 vs N=33)

### Step 9: Persistent camera view with free-look navigation

Camera view mode currently exits on any navigation input. This step makes the
background image persist during orientation-only navigation, enabling the user
to free-look around a fisheye image with a narrower viewport FOV.

#### Background: unit-sphere BG mesh (prerequisite, done)

The background image mesh was unified into a single pipeline for all camera
models. Mesh vertices are unit ray directions computed via `pixel_to_ray`
and transformed to world space by the camera-to-world rotation during mesh
generation. This matches the coordinate convention of frustum wireframes and
image quads, so the BG shader uses the same `view_proj = projection * view`
transform pipeline. The shader uses `w=0` so only the view rotation affects
the directions. This supports fisheye cameras with FOV at and beyond 180°.

#### Design: which navigation keeps camera view

Camera view stays active for any input that does **not** move the camera
center. The background image remains visible and the `CameraViewMode` struct
is preserved. Inputs that translate the camera position exit camera view as
before.

| Input | Camera method | Moves center? | Keeps camera view? |
|-------|---------------|---------------|--------------------|
| Left-drag / two-finger drag / gesture | `nodal_pan()` | No | **Yes** |
| Alt-drag / Alt+two-finger / Alt+gesture | `orbit()` | Yes | **No** |
| Shift-drag / middle-drag (pan) | `pan()` | Yes | No |
| Ctrl-drag / right-drag / scroll wheel / pinch (zoom) | `zoom_fov()` | No | **Yes** |
| Alt+Ctrl scroll/drag (target push/pull) | `target_push_pull()` | No | Yes |
| WASD (fly move) | `fly_move()` | Yes | No |
| Q/E (tilt) | `tilt()` | No | Yes |
| FOV slider | sets `camera.fov` | No | Yes |
| Alt+pinch | `target_push_pull()` | No | Yes |
| Home | resets orientation | N/A | No (explicit reset) |

**Swapped bindings in camera view:** The default (unmodified) drag/scroll/gesture
performs **nodal pan** (free-look) instead of orbit, keeping camera view active.
Alt+drag performs **orbit** instead of nodal pan, exiting camera view. This swap
only applies while in camera view mode — outside camera view, the standard
bindings apply (unmodified = orbit, Alt = nodal pan). The rationale is that
free-look is the primary navigation in camera view (exploring a fisheye image),
while orbit is a deliberate exit.

**FOV zoom in camera view:** All zoom controls (scroll wheel, Ctrl+drag,
right-drag, pinch, gesture zoom) adjust the viewport FOV via `zoom_fov()`
instead of dollying the camera. This keeps camera view active and lets the user
zoom in/out on the background image. FOV is clamped to 5°–160°. Outside camera
view, zoom controls dolly the camera as before.

For gesture events (Windows DirectManipulation), the same swapped logic applies:
unmodified pan gesture calls `nodal_pan` (keeps camera view), Alt+pan gesture
calls `orbit` (exits), zoom gestures call `zoom_fov` (keeps camera view), etc.

#### FOV handling on enter

When entering camera view, the FOV behavior depends on the camera model:

- **Perspective cameras**: set `camera.fov` to `best_fit_fov()` as today.
  This ensures the background image fills the viewport.
- **Fisheye cameras**: do **not** change `camera.fov`. The viewport keeps
  whatever FOV it had before entering camera view. The fisheye image covers
  a much wider area than the viewport's perspective projection can show, so
  best-fit is not meaningful. The user sees a perspective window into the
  fisheye sphere and can free-look to explore the full field of view.

#### FOV handling while in camera view

Stop overriding `camera.fov` every frame. Currently `best_fit_fov()` is
recomputed per-frame to handle window resizes. With persistent camera view,
the user may have changed the FOV via the slider or zoom, so the per-frame
override must stop.

For the initial enter from non-camera-view, the FOV is set once (perspective
cameras only, as described above). After that, the FOV is under user control.

#### Camera-to-camera transitions

When switching from one camera view to another (`,`/`.` keys, double-click
on image in the strip or viewport), the transition preserves the user's
relative viewing direction:

1. **Keep the current FOV** — do not recompute `best_fit_fov()`.
2. **Compute the relative orientation** of the viewport camera with respect
   to the outgoing SfM camera:
   ```
   R_old_cam_from_world = R_world_from_old_cam.inverse()
   R_relative = R_old_cam_from_world * R_viewport
   ```
   where `R_viewport` is the viewport camera's current orientation (which may
   differ from the outgoing SfM camera if the user has been free-looking).
3. **Apply the relative orientation to the new SfM camera**:
   ```
   R_viewport_new = R_world_from_new_cam * R_relative
   ```
4. **Set the viewport position** to the new camera's center.
5. **Set `target_distance`** from the new camera's median depth.
6. **Update `world_up`** by transforming through the same relative rotation:
   ```
   up_in_old_cam = R_old_cam_from_world * world_up
   world_up_new = R_world_from_new_cam * up_in_old_cam
   ```
7. **Load the new camera's background image**.

This means if the user is looking 30 degrees to the right of camera A's
optical axis when they press `.`, they will be looking 30 degrees to the
right of camera B's optical axis after the switch. The view feels stable
relative to the cameras rather than jumping to each camera's optical axis.

When entering camera view from non-camera-view (Z key, first double-click),
the current behavior is preserved: set the viewport pose to exactly match
the SfM camera, set FOV to best-fit (perspective only), etc.

#### BG mesh view_proj update

Because BG mesh vertices are pre-rotated to world space during mesh
generation (same convention as frustum wireframes), the BG uniform's
`view_proj` is simply the standard `projection * view` — the same
transform used by every other pipeline.

```
view_proj = projection * view
```

The shader uses `w=0` to transform positions as directions (no
translation), so only the view matrix's rotation affects the mesh.
Free-look navigation works automatically because the viewport camera's
orientation is captured in the view matrix.

The `CameraViewMode` struct stores the SfM camera's `r_world_from_cam`
rotation for the camera-to-camera transition math (not for the BG
uniform computation).

#### Changes to `CameraViewMode`

Add fields to support the new behavior:

```rust
pub struct CameraViewMode {
    pub image_index: usize,
    /// World-from-camera rotation of the SfM camera being viewed.
    /// Used to compute the relative view rotation for the BG mesh.
    pub r_world_from_cam: UnitQuaternion<f64>,
}
```

The `vfov_cam` and `hfov_cam` fields are removed since the FOV is no longer
force-updated per frame. The `best_fit_fov()` method moves to a standalone
function used only at initial entry.

#### Implementation tasks

1. Store `r_world_from_cam` in `CameraViewMode`; remove `vfov_cam`/`hfov_cam`
2. On enter from non-camera-view: set pose, set FOV (perspective only),
   set `world_up`, load BG image
3. On camera-to-camera transition: compute relative orientation, apply to new
   camera, keep FOV, transform `world_up`, load BG image
4. Stop per-frame FOV override (remove the `best_fit_fov` call in
   `handle_3d_panel`)
5. Per-action camera view exit: replace blanket `self.camera_view = None`
   with per-action checks (only exit for center-moving operations)
6. Update `update_bg_image_uniforms` to use standard `projection * view`
   (BG mesh vertices are pre-rotated to world space during generation)
7. Test: enter camera view, free-look with unmodified drag, verify BG image rotates; Alt-drag should orbit and exit camera view
   correctly and 3D points remain aligned at the optical axis
8. Test: navigate between cameras with `,`/`.`, verify stable relative view
9. Test: FOV slider in camera view, verify BG image scales correctly
10. Test: Q/E tilt in camera view, verify BG and horizon rotate together

### Future Enhancements

- [ ] Camera up indicator on frustum wireframes (see [open question](#gpu-data-layout-for-frustum-wireframes))
- [ ] Async thumbnail loading
- [ ] FULL_OPENCV distortion model (rational radial)
- [ ] Compressed textures (BC7/ASTC) for thumbnail memory reduction
- [ ] Fisheye viewport projection (equidistant shader to show full fisheye FOV without perspective re-projection)

---

## Size Estimates

| Item | Size |
|------|------|
| Frustum edges — pinhole (8 × 32 bytes × 10K cameras) | 2.5 MB |
| Frustum edges — distorted N=5 (20 × 32 bytes × 10K cameras) | 6.3 MB |
| Frustum edges — fisheye N=17 (68 × 32 bytes × 10K cameras) | 21.8 MB |
| Image quad instances — pinhole (64 bytes × 10K cameras) | 640 KB |
| Image quad mesh — distorted N=5 (vertices + indices × 10K cameras) | ~10 MB |
| Image quad mesh — fisheye N=17 (vertices + indices × 10K cameras) | ~130 MB † |
| Pick buffer (R32Uint, 1920×1080) | 8.3 MB |
| Readback staging buffers (2 × 256 × 5 bytes) | 2.5 KB |
| Thumbnail texture array atlas (128×128 RGBA × 10K cameras) | 625 MB (across multiple pages if needed) |
| Full-res background image (e.g. 4624×3472 RGBA) | ~61 MB |
| Background uniform buffer (view_proj mat4) | 64 bytes |

† Fisheye cameras at 10K scale are unlikely in practice — fisheye is
typically used with camera rigs that have a few fisheye sensors. A
reconstruction with 500 fisheye cameras would use ~6.5 MB for the image
quad mesh.

Square thumbnails (128×128) store images with non-square pixels. The
distortion is compensated on the GPU: each frustum quad has the correct
aspect ratio from camera intrinsics, and UV 0→1 mapping stretches the
texture back to the original proportions. For 10K+ cameras, consider
compressed formats (BC7/ASTC) for ~4× reduction.
