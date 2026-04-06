# Point Cloud Rendering

This document specifies the point cloud rendering pipeline for the sfmtool 3D
viewer, including point splat rendering, Eye-Dome Lighting (EDL) post-processing,
and the target indicator with its supernova lighting effect.

For navigation controls that interact with the target (Alt-mode), see
[gui-viewport-navigation.md](gui-viewport-navigation.md#target-control-alt-mode).

---

## Length Scale

`length_scale` is a characteristic length that captures the scale of what
you're interacting with — not the overall size of the reconstruction, but the
local density and spacing of the data around you. For example, in a city-scale
reconstruction where you're inspecting a park bench, `length_scale` should
reflect the bench, not the city.

It is used throughout the viewer to scale things that should adapt to the
scene: target indicator size, extent of the target light echoes, frustum stub depth, fog
falloff distances, and fly-mode movement speed.

Currently, `length_scale` is auto-computed from the point cloud as a multiple
of the median nearest-neighbor distance (the same value used for auto point
sizing). This is a rough approximation — it doesn't adapt as you navigate to
different parts of the scene, and can be off significantly when point
density varies across the reconstruction.

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
    position: [f32; 3],     // world-space XYZ
    color_packed: u32,      // R8G8B8A8 (alpha unused, set to 255)
}
```

16 bytes per point. 10M points = 160 MB of GPU buffer.

### Point Sizing

Point size is determined by two factors:

1. **Auto-size from data**: On point cloud upload, the median nearest-neighbor
   distance is computed from a random subsample of up to 10,000 points (using
   a KD-tree). This provides a base `point_size` that adapts to the
   reconstruction's scale.

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
2. Outputs premultiplied RGBA color to the color target
3. Outputs positive linear view-space depth to the depth target
4. Outputs `PICK_TAG_POINT | instance_index` to the pick target

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
[gui-camera-views.md](gui-camera-views.md#edl-shader-compatibility)).

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
| `edl_radius` | 2.4 | "EDL Line Thickness" slider | Sampling distance in pixels |
| `point_size` | auto | — | Normalizes depth differences |

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
[gui-viewport-navigation.md](gui-viewport-navigation.md#target-control-alt-mode).

### Activation

The target indicator and supernova effect activate in two ways:

| Trigger | Behavior |
|---------|----------|
| **Hold Alt** | Indicator appears while Alt is held, fades on release |
| **Double-tap Alt** | Toggles indicator to stay visible without holding Alt |

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

The indicator is rendered *after* the EDL pass and samples the scene's linear
depth texture to determine whether each fragment is in front of or behind
scene geometry:

| Depth Relationship | Color | Opacity | Description |
|-------------------|-------|---------|-------------|
| In front of geometry | Cyan | 100% | Fully visible, unoccluded |
| Behind geometry | Warm orange | Fades with depth | Color shift signals occlusion |

The opacity when occluded follows a fog-like falloff:
```
opacity = base_opacity * exp(-depth_behind / fog_distance)
```
where `fog_distance = target_fog_multiplier × length_scale` (default multiplier:
10.0). The indicator never becomes fully invisible — a minimum floor ensures
it can always be found.

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
The planned UI label for this effect is **"Target Light Echoes"**.

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

Pass 1b: Frustum wireframes  (see gui-camera-views.md)
  → writes: color, linear depth (0.0), pick ID, hw depth

Pass 1c: Image quads  (see gui-camera-views.md)
  → writes: color, linear depth (0.0), pick ID, hw depth

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

## Implementation Status

### Implemented

- [x] GPU point splat rendering (billboard quads, instanced)
- [x] Per-point RGB color
- [x] Auto point sizing from median nearest-neighbor distance
- [x] User-adjustable point size (log2 slider)
- [x] EDL post-processing (8-neighbor, two-radius sampling)
- [x] Adjustable EDL line thickness
- [x] Three render targets (color + linear depth + pick ID)
- [x] Target indicator (rotating wireframe octahedron)
- [x] Depth-aware transparency with color shift
- [x] Supernova lighting effect with inverse-square falloff and radiating waves
- [x] Smooth activation/deactivation fade

### Future Enhancements

- [ ] Adaptive `length_scale` that updates as you navigate to different parts of the scene
- [x] Target indicator redesign: 3D compass shape with filled star rose showing `world_up` (see [3D Shape](#3d-shape-rotating-compass))
- [ ] Target indicator visibility over bright backgrounds (see [open question](#glow-effect))
- [ ] Color points by reprojection error, track length, or triangulation angle
- [ ] Filter points by quality metrics
- [ ] LOD (level of detail) for 10M+ point performance
- [ ] Point size attenuation with distance
