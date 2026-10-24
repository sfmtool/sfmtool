# Point Cloud Rendering

The SfM Explorer's 3D viewer draws each reconstructed point as a small round
splat that always faces the camera, using GPU instancing to draw one shared quad
once per point. A point has no surface normal to shade by, so a full-screen
Eye-Dome Lighting (EDL) pass then darkens each point pixel whose neighboring
pixels hold nearer points or no point at all. That outlines surfaces and depth
edges and makes the 3D shape of a sparse cloud readable. While Alt is held, or
while the HUD's Target indicator checkbox is on, a rotating compass glyph marks
the orbit camera's target point: cyan where it is in front of the scene, a faded orange where scene
geometry hides it. The EDL pass also brightens points near the target, with
pulses moving outward from it, so the points around the pivot stand out even
when the glyph itself is hidden. The code calls this brightening the supernova
effect.

For navigation controls that interact with the target (Alt-mode), see
[viewport-navigation.md](viewport-navigation.md#target-control-alt-mode).

---

## Length Scale

`length_scale` is a characteristic length that captures the scale of what
you're interacting with — not the overall size of the reconstruction, but the
local density and spacing of the data around you. For example, in a city-scale
reconstruction where you're inspecting a park bench, `length_scale` should
reflect the bench, not the city.

It is used throughout the viewer to scale things that should adapt to the
scene: target indicator size, extent of the target light echoes, frustum stub depth,
and fly-mode movement speed.

Currently, `length_scale` is auto-computed from the point cloud as a multiple
of the trimmed median nearest-neighbor distance (same value as auto point
sizing). This is a rough approximation — it doesn't adapt as you navigate to
different parts of the scene, and can be off significantly when point
density varies across the reconstruction.

The computed value is held to the range of the HUD's Scene slider, 0.001 to
100, and rounded to the slider's three decimals before it is stored, so the
value the viewer sets is one the slider and `set_viewer_3d_display` both
accept ([mcp-server.md](mcp-server.md) § "`get_viewer_3d_display` /
`set_viewer_3d_display`").

---

## Point Splat Rendering

### Approach

Each 3D point is rendered as a camera-facing **billboard quad** — a small square
that always faces the viewer. The GPU expands each point into a quad using the
camera's right and up vectors, then the fragment shader discards pixels outside
a unit circle to produce a round splat.

This is more efficient than CPU-based rendering and scales to millions of points
via GPU instancing.

### Per-Point Data

```rust
#[repr(C)]
struct PointInstance {
    position: [f32; 3],     // world-space XYZ (unit direction when at infinity)
    color_packed: u32,      // R8G8B8A8 (alpha: 255 = finite, 0 = at infinity)
}
```

16 bytes per point. 10M points = 160 MB of GPU buffer. The alpha byte carries
the points-at-infinity flag (see [Points at Infinity](#points-at-infinity)).

A second instance buffer steps alongside it, one `u32` per point, the **mask
word**. That is how an edit's deleted set, and the sets of points the viewport
calls out, reach the pass without rewriting the instance buffer:

| Word | Constant | The point is | Drawn as |
|------|----------|--------------|----------|
| `0` | `MASK_DELETED` | deleted by the version in view | nothing: a clipped vertex, which occludes nothing and answers no pick |
| `1` | `MASK_ALIVE` | live | its own colour |
| `2` | `MASK_HIGHLIGHTED` | live, and observed by a camera being moved (the Move Camera lock, `crate::camera_lock`) | the hover cyan, `(0, 1, 1)` |
| `3` | `MASK_PAINTED` | live, and painted by the eraser brush's stroke being held | red, `(1.0, 0.1, 0.1)` |
| `4` | `MASK_UNDER_BRUSH` | live, and under the eraser brush's circle before a press | pale red, `(1.0, 0.6, 0.6)` |

The words are declared in
[upload/overlay.rs](../../crates/sfm-explorer/src/scene_renderer/upload/overlay.rs)
and in [common.wgsl](../../crates/sfm-explorer/src/shaders/common.wgsl), with the
two brush colours, and a test in `scene_renderer/pipelines/tests.rs` checks that
the two sides agree. A point in more than one set takes one word, chosen by
`mask_word`: deleted, then painted, then under the brush, then highlighted. The
colours replace the point's own outright and are not tinted, as the selection's
yellow and the hover's cyan are not. The two reds come before the selection and
the hover in the fragment shader, because they say what releasing the stroke
will do to the point. They share a hue so the circle's preview reads as what a
press would paint, and the preview's is the paler one because nothing has been
done yet. The eraser brush that sets `3` and `4` is specified in
[edits/eraser-brush.md](edits/eraser-brush.md).

A live point the Track Table's *Filter the 3D view* rejects carries one more
bit over whichever word it takes, `MASK_REJECTED` (`0x100`), so a rejected
point under the brush keeps its red; the shaders read the word as
`alive & MASK_WORD` (`0xFF`). A deleted point's word never carries it. See
§ "The Track Table's filter".

`SceneRenderer::update_point_mask` writes the words; see
[document-model.md](document-model.md) § "Change detection by identity".

### The Track Table's filter

The Track Table's *Filter the 3D view* box ([track-table.md](track-table.md)
§ "Filtering the 3D view") dims or hides the points of the reconstruction it
shows that its rows do not list. The panel publishes the set it keeps,
`KeptPoints` in [filter_3d.rs](../../crates/sfm-explorer/src/filter_3d.rs),
one bit per point index, for one version of the node. The frame's mask upload
hands it to `update_point_mask` in `PointMarks::kept`, with
`PointMarks::filter_hides`, only for the node the table shows and only when the
set was built for the version at the node's cursor; otherwise the node gets no
filter, and every point is drawn.

- **The set is a bit of the mask word**, not a buffer of its own. The word is
  already there, one per point and one per surfel slot, stepping beside the
  instances, so the filter costs no GPU memory and no vertex slot, and it
  reaches the surfels and the overlay's additions by the writer that already
  covers them. The bundle keeps the set it last wrote (`masked_kept`) and
  compares the next one by pointer. A new set, the filter coming or going, or
  a new base writes every word of the node once; an unchanged one writes
  nothing. The panel builds a new set only when its row list is computed
  again, so a frame costs one pointer comparison.
- **Dim or hide is a uniform.** `ReconUniforms::filter_mode`, in the padding
  after `preview_s`, is `FILTER_DIM` (0) or `FILTER_HIDE` (1); switching
  between them writes no word. It is read only where a word carries the bit,
  which is only on the filtered node.
- **Two draws.** The point and patch pipelines read their corners from the
  splat corner buffer (`SPLAT_CORNERS` in `gpu_types.rs`), the unit quad's
  four corners twice, each carrying which draw it is of at `@location(15)`.
  The point pass and the surfel pass draw every node with corners `0..4`
  (`FIRST_DRAW` in
  [render.rs](../../crates/sfm-explorer/src/scene_renderer/render.rs)), then,
  after every node's surfels, the nodes the filter dims again with corners
  `4..8` (`DIMMED_DRAW`). The draw is vertex data rather than read off
  `vertex_index`, because DX12's vertex id does not count the draw's first
  vertex, while every backend fetches vertex data from it; the first capture
  run, which read `vertex_index`, drew no dimmed point on Windows. The vertex
  shaders decide with `in_this_draw` in
  [common.wgsl](../../crates/sfm-explorer/src/shaders/common.wgsl): the first
  draws every point but the rejected ones, the second the rejected ones alone,
  at `DIM_ALPHA` (0.25), and a rejected point under Hide is a clipped vertex
  in both, so it draws nothing and answers no pick. Drawing the dimmed points
  last means a kept point is never behind a dimmed one that was drawn first
  and won the depth test: a kept point behind a dimmed one shows through it,
  and a dimmed one behind a kept one fails the test. Both draws write depth
  and pick ids, so a dimmed point is picked where it is in front. The second
  draw costs a vertex pass over the filtered node's points while Dim is on.
- **Selected and hovered points are exempt.** A rejected point whose global
  pick index is the selected or the hovered one is drawn by the first draw at
  full strength. The splats read the two indexes from `PointUniforms`; the
  surfels from `PatchUniforms::selected_point_index` and
  `hovered_point_index`, in the padding before `camera_pos`.
- **A point at infinity** is drawn from the same instance buffer by the same
  vertex shader, which reads the word before it branches on the kind, so it is
  dimmed or hidden as a finite point is.
- **The EDL pass keeps the dim alpha.** Pass 1's colour target holds
  premultiplied colour, and the EDL pass returns the shaded colour as it is,
  with the glow scaled by the alpha, rather than multiplying by the alpha
  again, which would draw a dimmed point at its square.

The constants are written once on each side, `MASK_REJECTED` in
[upload/overlay.rs](../../crates/sfm-explorer/src/scene_renderer/upload/overlay.rs),
`FILTER_*` in `gpu_types.rs`, `DIMMED_DRAW` in `render.rs` and `DIM_ALPHA` in
`filter_3d.rs`, and `the_wgsl_filter_values_match_the_rust_ones` in
`scene_renderer/pipelines/tests.rs` holds them to `common.wgsl`. The upload
tests in `scene_renderer/upload/tests.rs` hold that the bit marks exactly the
live points the set leaves out, a deleted one never, a rejected brush mark
keeping its word; that the same set writes nothing, Hide writes nothing and
sets the uniform, a set built again rewrites every word and dropping the
filter clears them; that another node's words and uniforms are untouched; that
a point at infinity takes the bit; and that a new base and a fresh additions
buffer take it. The `noop` backend runs no draw, so which draw a word lands in
and the pick through a hidden point were checked in a release build on the
Seoul Bull and Kerry Park ground truths.

### The overlay's additions

A node draws **two** point instance buffers: its base's, and the points its
current version added on top of it. The second is drawn immediately after the
first in the same pass, with the same global uniforms and its own liveness
buffer, and it is rebuilt only when the addition set moves; the base's is what a
run of point edits shares and is never rebuilt for one. The two occupy one
contiguous pick range, so a point's global pick id is
`point_pick_base + edited index` whichever buffer holds it. See
[document-model.md](document-model.md), "Change detection by identity".

### The camera model proposal's preview

While a camera model proposal's **Preview retriangulation** has shown an answer
([edits/switch-camera-model.md](edits/switch-camera-model.md) § "The
preview"), its node has a **preview buffer** beside each point buffer
([`upload/preview.rs`](../../crates/sfm-explorer/src/scene_renderer/upload/preview.rs)):
one `vec4<f32>` per row, `VERTEX | COPY_DST`, stepping per instance beside
`PointInstance` and the liveness buffer. Its `xyz` is the row's target (a unit
direction for a point at infinity) and its `w` a flag: `0` for a row that does
not move, `1` for one that moves to its target, and `2` for one that stays
where it stands and takes the failure colour. A row that does not move is
marked by its flag, not by a zeroed position. The additions get a preview
buffer of their own by the same rule. `PointInstance` stays 16 bytes and
`VERTEX` only: the target is not written into it, so ending the preview
restores nothing, and drops the preview buffers.

The node is drawn with a variant of the point pipeline, `vs_preview`, which
reads the preview buffer in a fourth vertex slot; every other node is drawn
with the pipeline it always is. The blend `s` goes in the node's
`ReconUniforms::preview_s`, in the padding that follows `show_infinity`, so the
struct's size does not change; it is declared once, in `common.wgsl`, and the
additions' copy of the block carries it too. `vs_preview` draws a moving row at
`stored + s · (target − stored)`, normalised for a point at infinity so it
turns along the shorter arc, and a row flagged `2` at its stored position in
`mix(colour, FAILURE_COLOR, s)`, after the tint, so a node tinted Vermillion
still shows its failures. `FAILURE_COLOR` is the linear value of `FAILURE_RGB`
(`[235, 30, 30]`, beside `TINT_PALETTE` in `scene.rs`), since the colour target
is `Rgba8UnormSrgb` and encodes what the shader writes, so the screen shows
(235, 30, 30), the red the proposal banner draws in; a pipeline test checks it
against `common.wgsl` with the three flags. At `s = 1` a moving row is drawn at
its target itself rather than at `mix`'s `stored + 1 · (target − stored)`,
which can be an ulp off it, so the end of the motion is drawn from the
coordinates the version Apply pushes is drawn from.

The rows are built on the proposal's job thread, laid out by the node's two
point buffers, so a new answer is one upload of each preview buffer and a
frame of motion writes nothing but `s`. A buffer whose row count no longer
matches the point buffer it steps beside (a new base, a changed addition set)
gets no preview rather than one that moves the wrong points. The pick target
is written by the same vertex shader, so a click on a moving point picks it
where it is drawn, and the depth readback reads the drawn position.

### Point Sizing

Point size is determined by two factors:

1. **Auto-size from data**: On point cloud upload, a nearest-neighbor distance
   is collected for each of a seeded random subsample of up to 10,000 finite
   points (using a KD-tree). The base `point_size` is 1.1 times the *trimmed
   median* of those distances, which adapts to the spacing of the
   reconstruction's coherent structure rather than to the whole cloud:

   - Take the median of the distance set.
   - Drop every distance above 2x that median. Such a point is isolated by the
     cloud's own scale, not by any distance fixed in scene units.
   - Take the median of what is left and repeat, stopping at the first pass
     that drops nothing, or after 8 passes.

   A cloud carries a scattered sub-population beside its surfaces
   (mis-triangulated points strung out along rays through empty space, for
   one). Those distances form a broad upper tail that pulls the plain median
   several-fold above the spacing of the dense structure, and splats sized off
   it swell until the surfaces vanish underneath. Each pass cuts only from
   above, so the medians are non-increasing and settle in a handful of passes;
   on a tight distribution nothing exceeds the bar and the trim is a no-op.

2. **User adjustment**: A log2-scale slider (`point_size_log2`, range -3 to +3)
   multiplies the base size. This gives intuitive control — each slider step
   doubles or halves the point size.

The final world-space radius sent to the shader is:
```
radius = point_size * 2^(point_size_log2)
```

### Vertex Shader

The vertex shader receives a unit quad (4 corners: ±1, ±1) and expands it in
world space:

```
world_pos = point_position + (quad_x * camera_right + quad_y * camera_up) * radius
```

The quad is then projected to clip space. UV coordinates (`quad_x`, `quad_y`)
are passed to the fragment shader for circle clipping.

### Fragment Shader

The fragment shader:
1. Discards fragments where `length(uv) > 1.0` (circle clipping)
2. Replaces the point's colour, first match wins: the brush's red or pale red
   for mask word `3` or `4`, the selection's yellow for the selected point, the
   hover's cyan for the hovered point or mask word `2` (see "Per-Point Data")
3. Outputs premultiplied RGBA color to the color target, at alpha 1, or at
   the Track Table filter's `DIM_ALPHA` in its dimmed draw (see "The Track
   Table's filter")
4. Outputs positive linear view-space depth to the depth target
5. Outputs `PICK_TAG_POINT | instance_index` to the pick target

### Three Render Targets

The point splat pass writes to three color attachments plus hardware depth:

| Target | Format | Content | Purpose |
|--------|--------|---------|---------|
| Color | Rgba8UnormSrgb | Premultiplied RGBA | Visible color |
| Linear depth | R32Float | Positive view-space depth | EDL shading + mouse depth readback |
| Pick ID | R32Uint | `PICK_TAG_POINT \| index` | Entity picking |
| HW depth | Depth32Float | Standard depth | Z-test during rendering |

The linear depth texture stores *positive* values for points. This is critical
for the EDL shader and depth readback — background pixels have alpha = 0 and
depth = 0, while frustum/image pixels write depth = 0 explicitly (see
[camera-views.md](camera-views.md#edl-shader-compatibility)).

---

## Eye-Dome Lighting (EDL)

### Concept

Eye-Dome Lighting is a post-processing technique that creates depth perception
in point clouds without requiring surface normals. It darkens pixels at depth
discontinuities — wherever a point is near the edge of a surface or in front
of a more distant surface — producing an effect similar to ambient occlusion.

The result gives sparse point clouds a sense of solidity and makes it much
easier to perceive the 3D structure.

### Algorithm

The EDL shader runs as a fullscreen post-process after the point splat pass.
For each pixel:

1. Read the center pixel's linear depth
2. Sample 8 neighbors (cardinal + diagonal directions) at two radii:
   - Inner ring at `radius` pixels
   - Outer ring at `2 × radius` pixels (weighted 0.5×)
3. For each neighbor, compute a depth response:
   ```
   response = max(0, (center_depth - neighbor_depth) / point_size)
   ```
4. Average all responses and apply exponential falloff:
   ```
   shade = exp(-average_response * strength)
   ```
5. Multiply the original color by the shade factor

### Parameters

| Parameter | Default | UI Control | Effect |
|-----------|---------|------------|--------|
| `edl_strength` | 0.7 | — | Controls how dark the edges get |
| `edl_radius` | 2.4 | "EDL width" slider, 0.5–8 px (viewport HUD, Advanced) | Sampling distance in pixels |
| `point_size` | auto | — | Normalizes depth differences |

`edl_radius` is wired end-to-end: `AppState::edl_line_thickness` flows into the
EDL uniforms every frame and the HUD's Advanced section drives it.

### Background and Frustum Handling

The EDL shader uses a three-way check to handle different pixel types:

```
alpha == 0    → background (no geometry rendered here)
depth == 0    → frustum/image quad pixel (composite without EDL)
depth > 0     → point cloud pixel (apply full EDL shading)
```

Background pixels receive the dark background color. Frustum and image quad
pixels pass through without EDL darkening (they write 0.0 to the linear depth
texture to signal this). Only point cloud pixels receive the full EDL treatment.

---

## Target Indicator

The target indicator is a 3D glyph rendered at the orbit camera's target point.
It makes the otherwise-invisible pivot point visible and provides spatial
context for understanding the 3D structure around it.

For the navigation controls that activate and move the target, see
[viewport-navigation.md](viewport-navigation.md#target-control-alt-mode).

### Activation

The target indicator and supernova effect activate in two ways, and the
indicator alone also flashes for 0.3 s after the target moves:

| Trigger | Behavior |
|---------|----------|
| **Hold Alt** | Indicator appears while Alt is held, fades on release |
| **Target indicator** (HUD checkbox, Layers) | Keeps the indicator visible without holding Alt |

Activation fades in/out over ~300ms for smooth transitions.

### 3D Shape: Rotating Compass

The indicator is a 3D compass rendered at the target point. It combines a
filled compass-rose star polygon with wireframe vertical spikes and a
circular ring. The compass shape clearly shows the current `world_up`
direction (important when tilt/roll has been applied) and provides
horizontal orientation via the star pattern.

**Geometry**:

The compass has three layers, all sharing the same rotation and scale:

- **Vertical axis** (wireframe): A tall top spike at (0, 0, 1.5) and a
  shorter bottom spike at (0, 0, −0.7), both connected to the center
  (0, 0, 0). The asymmetry (1.5 vs 0.7) makes "up" unambiguous from any
  viewing angle. Rendered as ribbon quads.
- **Filled star polygon**: An 8-point compass-rose star in the horizontal
  (z = 0) plane, triangulated as a center fan (16 triangles). The star
  alternates between outer tips and inner indentations:
  - *Cardinal tips* (N/E/S/W): extend to 1.25× the ring radius (0.75)
  - *Intercardinal tips* (NE/SE/SW/NW): extend to 0.8× the ring radius
    (0.48)
  - *Inner indentations*: 8 vertices on a small circle at 1/5 the ring
    radius (0.12), positioned 2/3 of the way from each cardinal direction
    toward the adjacent intercardinal. This gives cardinal tips wider bases
    and intercardinal tips narrower bases.
- **Circular ring** (wireframe): A 32-segment circle at radius 0.6 in the
  horizontal plane, providing a clean round outline for the compass rose.

**Properties**:

- **World-space radius**: `target_size_multiplier × length_scale`
- **Continuous rotation**: Spins around `world_up` at 30°/sec, conveying
  that it is a live interactive element
- **Two render pipelines**: The filled star uses a triangle-list pipeline;
  the vertical axis and ring use instanced ribbon-quad edges (same
  technique as frustum wireframes). The star is drawn first, then the
  wireframe on top, both with additive blending.

### Depth-Aware Transparency

The indicator is rendered *after* the EDL pass and samples the scene's hardware
depth buffer to determine whether each fragment is in front of or behind
scene geometry. It is the hardware buffer and not the linear depth attachment
because a patch, a frustum and an image quad all write zero to the latter, so
geometry that meets one of those has to be read against the buffer they do
write ([`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md) § "Depth against
the scene"):

| Depth Relationship | Color | Opacity | Description |
|-------------------|-------|---------|-------------|
| In front of geometry | Cyan | 100% | Fully visible, unoccluded |
| Behind geometry | Warm orange | Fades with depth | Color shift signals occlusion |

The opacity when occluded falls linearly from 20% to a 5% floor
(`shaders/target_indicator.wgsl`):
```
t = clamp(depth_behind / fog_distance, 0, 1)
opacity = mix(0.2, 0.05, t)
```
where `depth_behind` is the reversed-Z NDC depth difference between the scene
and the indicator fragment, and
`fog_distance = 0.1 × target_fog_multiplier / DEFAULT_TARGET_FOG_MULTIPLIER`
(0.1 at the default multiplier of 10.0, `scene_renderer/uniforms.rs`). It does
not scale with `length_scale`. The 5% floor keeps the indicator from becoming
fully invisible, so it can always be found.

### Glow Effect

Each ribbon-quad fragment applies a center-weighted glow using
`smoothstep(1.0, 0.4, dist_from_center)`. Combined with **additive blending**,
this creates a soft glow that brightens surrounding pixels without darkening
anything. This works well against dark backgrounds but can become invisible
against bright or white regions of the point cloud.

**Open question**: How should the indicator render to stay visible over all
backgrounds? Possible approaches include outline/contrast edge, color
inversion, or switching between additive and subtractive blending based on
background luminance.

---

## Supernova Lighting Effect

The supernova effect creates a spherical illumination centered at the target
point, giving the surrounding point cloud a "star in a nebula" feel. It reveals
the local 3D structure around the target by making nearby points glow brighter.

Inspired by Hubble imagery of supernova light echoes — see
[SN 2014J in M82](https://science.nasa.gov/asset/hubble/light-echo-around-sn-2014j-in-m82/).

### Purpose

When the user holds Alt to see where the target is, the supernova effect
answers a deeper question: "what is the 3D structure *around* the target?"
In a dense point cloud, the target indicator alone might be occluded or hard
to locate. The spherical glow illuminates nearby points in all directions,
creating a localized "lantern" that reveals the spatial neighborhood.

### Implementation

The supernova effect is computed in the EDL shader (Pass 2) as an additive
post-process:

1. **Reconstruct view-space position**: Each fragment's view-space XY is
   reconstructed from its UV coordinates using `tan_half_fov` and `aspect`
   ratio uniforms. Combined with the linear depth, this gives the fragment's
   full 3D position in view space.

2. **Compute 3D distance**: The true Euclidean distance from the fragment to
   the target point (also in view space) is computed.

3. **Inverse-square falloff**: The glow intensity follows
   `r² / (dist² + r²)` where `r` is the target indicator radius (derived
   from `length_scale`). Values below 0.005 are culled early.

4. **Radiating waves**: Sine-wave pulses traverse outward from the target
   over time. The raw sine is raised to the 4th power to create narrow
   spikes with wide gaps:
   ```
   wave = pow(max(sin(phase), 0.0), 4.0)
   intensity = envelope * (0.3 + 0.7 * wave)
   ```

5. **Additive blending**: The glow is added to the EDL-shaded color, so it
   only brightens — it never darkens the scene.

### Uniforms

The EDL shader receives these additional uniforms for the supernova:

| Uniform | Type | Purpose |
|---------|------|---------|
| `target_view_pos` | `vec3<f32>` | Target position in view space |
| `supernova_active` | `f32` | 0.0–1.0 fade for smooth activation |
| `tan_half_fov` | `f32` | For view-space position reconstruction |
| `aspect` | `f32` | For view-space position reconstruction |
| `time` | `f32` | For wave animation |

---

## Render Pipeline Summary

The full rendering pipeline for the point cloud and its post-processing:

```
Pass 1a: Point splats
  → writes: color, linear depth (positive), pick ID, hw depth
  → technique: billboard quads, circle-clipped, instanced

Pass 1b: Frustum wireframes  (see camera-views.md)
  → writes: color, linear depth (0.0), pick ID, hw depth

Pass 1c: Image quads  (see camera-views.md)
  → writes: color, linear depth (0.0), pick ID, hw depth

Pass 1d: Patch surfels
  → writes: color, linear depth (0.0), pick ID, hw depth

Pass 1e: The Track Table filter's dimmed splats and surfels
  → only while the filter dims; vertices 4..8 of the same pipelines
  → writes: color at DIM_ALPHA, linear depth, pick ID, hw depth

Pass 2:  EDL post-process
  → reads: color + linear depth
  → applies: edge darkening + supernova glow
  → writes: final display color

Pass 3:  Target indicator
  → reads: linear depth (for occlusion testing)
  → writes: EDL output color (additive blending, no depth write)
  → no pick ID (not a selectable entity)
```

All passes share the same hardware depth buffer for correct occlusion between
points, frustums, and image quads. The target indicator reads scene depth but
does not write to it, allowing it to render with depth-aware transparency.

---

## Points at Infinity

Render `w = 0` points — directions, not locations — so a distant
skyline/ridge/cloud track contributes to the visual reconstruction instead of
being dropped or scattered to arbitrary far coordinates. The defining
property: **a point at infinity has no parallax.** As the camera translates, a
`w = 0` point holds its angular position; only camera *rotation* moves it —
exactly the `w = 0` homogeneous transform the background-image mesh already
relies on (see [camera-views.md](camera-views.md)). Materialising
infinity points at a finite far distance is explicitly rejected: it
reintroduces parallax and is wrong. The format model and classifier live in
the [sfmr file format](../formats/sfmr-file-format.md) (§7) and
[batch-triangulation-api.md](../core/reconstruction/batch-triangulation-api.md); points are
normalised on read so the renderer can rely on `w ∈ {0, 1}`, with `position`
holding a unit direction when `w = 0`.

### Design

1. **One instance buffer for all points.** `instance_index` stays equal to the
   global `recon.points` index, so picking, hover, and selection work
   unchanged for infinity points. The kind flag is packed into the otherwise
   unused alpha byte of `color` (finite → `255`, infinity → `0`), keeping
   `PointInstance` at 16 B; the shader tests
   `is_inf = ((color >> 24) & 0xFFu) == 0u`.

2. **Shader branch on kind** (`points.wgsl` `vs_main`). Finite points keep
   today's world-space billboard path. For a direction `d`:
   project with `clip_c = view_proj * vec4(d, 0.0)` (translation drops out —
   no parallax); cull when `clip_c.w <= 0` (behind camera); expand the quad in
   **screen space** at a fixed pixel radius (world-space size is meaningless
   for a direction); bias depth to `ndc_z = INF_EPSILON` — a raw `w = 0`
   transform yields `ndc_z = 0`, which fails `Greater` against the reversed-Z
   `0.0` clear, so the tiny positive bias keeps infinity splats visible while
   losing to all finite geometry; and write `view_depth = 0.0`, the existing
   "skip EDL" sentinel, so infinity points composite unshaded like
   frustums/quads instead of generating EDL halos. The fragment shader is
   unchanged (circle clip, selection/hover override, pick id).

3. **Uniforms.** `PointUniforms` gains `screen_size` (pixel-radius → NDC
   conversion) and `infinity_point_px` (on-screen splat radius for infinity
   points, default 3 px; a direction has no distance, so a fixed pixel size is
   the only meaningful sizing).

4. **Data-pipeline exclusions** — sites that iterate `p.position` as a 3-D
   location must skip or branch on `w = 0` points, or they silently corrupt
   (unit directions cluster at the origin):
   `sfmtool_core::analysis::scene_scale::compute_scene_bounds` and `compute_auto_point_size`
   skip them (protecting scene center/radius, clip planes, and auto splat
   size); `upload/track_rays.rs::upload_track_rays` draws infinity observations as
   fixed-length bearing rays from each camera center (no finite endpoint to
   converge on); Track View's header (`track_view/body/`) shows the
   **direction** with the ∞ mark and "at infinity" instead of bogus xyz, and
   `metrics.rs` projects it rotation-only (`R·d`, no depth divide); `app.rs` Alt+click does not move the orbit target from an
   infinity-point pick (a point with no location has no pivot — selection and
   hover still work via the pick id).

There is **no visual distinction** between finite and infinity splats beyond
the (free) absence of EDL shading. In camera-view mode an infinity point's
splat lands on the same pixel as its observed feature regardless of the viewed
camera's translation, since the background image mesh uses the same `w = 0`
transform.

### UI — shipped

- **Visibility toggle** — "Points at ∞" in the HUD's Layers section, default
  on (`AppState::show_points_at_infinity`). It is a `show_infinity` flag in
  `PointUniforms` that makes the vertex shader emit a clipped vertex, *not* a
  filtered upload: `instance_index` has to stay equal to the global
  `recon.points` index or picking, hover and selection break.
- **Count readout** — the top-left scene stats read
  `N points (M at infinity) | K images | F fps`, dropping the parenthetical
  when `metadata.infinity_point_count` is zero. While the Track Table filters
  the 3D view on a drawn node, `(S shown)` follows the point count: every
  drawn point less the filtered node's rejected ones. Until the table has rows
  for the version in view it reads `(all shown until the Track Table
  updates)` (`viewer_3d/overlay.rs::scene_stats_text`).
- **Infinity point size slider** — `infinity_point_px` in the HUD's Size
  section ("∞ (px)"), range 1–16 px, default 3. Separate from the world-unit
  `point_size_log2` slider. The 1–16 px range proved fine in practice; the
  carried-over open question is resolved.

## Non-goals

- **Point colouring by quality.** Points carry their RGB colour and nothing
  else; there is no mode that colours the cloud by reprojection error, track
  length, triangulation angle or any other track table column. The cloud is
  filtered by the Track Table's query instead (§ "The Track Table's filter"),
  which dims or hides the points it does not list and leaves the colours of
  the rest alone. The equivalent per-feature overlays exist on the Image
  Detail panel, where a metric can be read against the image it came from.
- **Level of detail.** Every point in a loaded reconstruction is uploaded and
  drawn every frame. Nothing decimates the cloud at distance, so the practical
  ceiling is what one instanced draw sustains rather than what an LOD scheme
  could.
- **Distance attenuation of point size.** A point's world-unit size is constant;
  splats shrink on screen only through perspective.
- **A `length_scale` that tracks the camera.** It is computed once from the
  scene bounds on upload (see [Length Scale](#length-scale)) and does not adapt as
  you navigate into a dense corner of a large scene.

## Open questions

- **Target-indicator visibility over bright backgrounds.** The additive glow
  described under [Glow Effect](#glow-effect) disappears against bright regions
  of the cloud. Outline or contrast edge, colour inversion, and luminance-driven
  switching between additive and subtractive blending are all plausible; none has
  been tried.
