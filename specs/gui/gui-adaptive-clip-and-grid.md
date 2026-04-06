# Adaptive Clip Planes and Ground Grid

**Status**: Implemented (2026-04-05)

## Problem

The 3D viewer originally used hardcoded near/far clip planes (0.1 / 1000.0) and a fixed ground grid
(extent ±10, step 1.0). SfM reconstructions have arbitrary scale — a room-scale scene might
span 0.5–5 units while an outdoor scene might span 50–5000. The fixed values caused:

1. **Z-fighting**: With a 10,000:1 depth ratio, large scenes get visible depth artifacts
2. **Near-plane clipping**: Small-scale scenes clip geometry near the camera when orbiting close
3. **Far-plane clipping**: Distant geometry disappears at the fixed 1000-unit cutoff
4. **Invisible grid**: The grid is only useful when scene scale happens to be ~1–20 units
5. **Grid doesn't occlude**: Grid is drawn as a 2D egui overlay, so it floats on top of geometry

## Solution: Reversed-Z Infinite Far Plane

The viewer uses **reversed-Z with an infinite far plane** projection. This eliminates two of
the five problems entirely (far-plane clipping and z-fighting) and simplifies the adaptive
clip plane logic to only managing the near plane.

### Why Reversed-Z

Standard Z-buffer projection maps near→0, far→1 with a hyperbolic distribution that wastes
most of the depth precision on geometry close to the far plane. For SfM scenes spanning
arbitrary distances, this produces z-fighting on nearby surfaces while the far plane clips
distant geometry.

Reversed-Z maps near→1, far→0 (with `CompareFunction::Greater`). Combined with an infinite
far plane, this gives:

- **No far-plane clipping**: Geometry at any distance is rendered. There is no far plane.
- **Optimal depth precision**: The hyperbolic distribution is inverted, concentrating
  precision near the camera where it matters most. With `Depth32Float`, this gives usable
  precision across the entire depth range from `near` to infinity.
- **One clip plane to manage**: Only `near` needs adaptive adjustment.

### Projection Matrix

The projection matrix uses the reversed-Z infinite far plane form:

```
| f/aspect  0    0      0    |
| 0         f    0      0    |
| 0         0    0      near |
| 0         0   -1      0    |
```

where `f = 1 / tan(vfov/2)`. This maps `z_view = -near` to `ndc_z = 1` and
`z_view = -∞` to `ndc_z = 0`.

### Depth Buffer Configuration

| Setting | Value |
|---------|-------|
| Format | `Depth32Float` |
| Compare function | `Greater` (closer fragments have larger depth values) |
| Clear value | `0.0` (reversed-Z: 0 = infinitely far) |

## Adaptive Near Plane

With no far plane to worry about, the only adaptive behavior needed is adjusting the near
clip plane based on the camera's position relative to the scene.

### Algorithm

1. On reconstruction load, compute the **scene bounding sphere** (center + radius) from the
   point cloud. The center is the component-wise median position (robust to outliers).
   The radius is the 80th percentile distance from center.

2. Each frame, compute the distance from camera to the bounding sphere center:
   `d = ||camera_pos - sphere_center||`

3. Set near plane:
   - `target_near = max((d + scene_radius) / 1000.0, 0.0001)`
   - Smooth transitions: exponential decay with `dt` to avoid frame-rate dependency.
     Use `lerp_factor = 1.0 - (-dt * 8.0).exp()` (~120ms settling at any frame rate).

### Rationale

- The near plane is set to 1/1000th of the distance to the far edge of the scene,
  giving a comfortable margin for close-up inspection
- With reversed-Z and `Depth32Float`, there is no practical depth precision concern
  even with a very small near plane — the precision is concentrated near the camera
- Camera distance awareness means the near plane recedes when you're far away and
  advances when you're close
- Time-based smoothing prevents jarring changes during fly navigation regardless of fps
- The floor at 0.0001 prevents the near plane from reaching zero

### Edge Cases

- Empty reconstruction (no points): keep default near = 0.1
- Camera inside the bounding sphere: use same formula (d is small, so
  target_near ≈ scene_radius / 1000)
- Camera view mode: no special handling needed (the bounding sphere still covers the scene)

## Adaptive Grid

Scale the ground grid's extent and step size based on `length_scale`, which is already
computed from the point cloud's nearest-neighbor statistics.

### Algorithm

1. Compute grid step as the nearest power of 10 to `length_scale * 5`:
   `step = 10^round(log10(length_scale * 5))`
2. Grid extent = `step * 10` (always 10 major grid lines in each direction)
3. Axis lines: scale to `step * 2` length (proportional to grid)

This makes the grid meaningful at any scale — if your scene spans 500 units, the grid
shows 50-unit major lines; if it spans 0.5 units, the grid shows 0.05-unit lines.

Note: The grid renders before any reconstruction is loaded using the fallback `length_scale`.
The adaptive values only become meaningful after point upload.

### Rationale

- Power-of-10 stepping gives clean round numbers at any scale
- Extent proportional to step keeps a consistent visual density
- `length_scale` is already computed and represents the characteristic spacing

### Grid Rendering

The grid currently renders as an egui overlay (no depth occlusion). Moving the grid to
GPU rendering (with proper depth testing against the point cloud) is a separate, larger
task. The adaptive scaling is independent of the rendering method.

## Implementation

### Scene Bounding Sphere

`compute_scene_bounds()` in `auto_point_size.rs` computes the bounding sphere on point
upload. It uses component-wise median for a robust center and the 80th percentile distance
for a robust radius. The result is stored as `scene_center` and `scene_radius` on
`SceneRenderer` and exposed via accessor methods.

### Near Plane Update

`ViewportCamera::update_clip_planes()` in `camera.rs` is called each frame from `app.rs`.
It computes the target near plane and applies exponential smoothing:

```rust
pub fn update_clip_planes(&mut self, scene_center: Point3<f64>, scene_radius: f64, dt: f64) {
    let d = (self.position() - scene_center).norm();
    let target_near = ((d + scene_radius) / 1000.0).max(0.0001);
    let alpha = (1.0 - (-dt * 8.0).exp()).clamp(0.0, 1.0);
    self.near += (target_near - self.near) * alpha;
}
```

`ViewportCamera` stores only `near` — there is no `far` field. The projection matrix
is built directly from `near` using the reversed-Z infinite form.

### Adaptive Grid

`draw_grid()` in `overlay.rs` accepts `length_scale` and computes the grid step as the
nearest power of 10 to `length_scale * 5`. The grid extent is `step * 10` and axis lines
are `step * 2`.

## Testing

### Unit Tests

- `compute_scene_bounds`: verify center/radius on known point distributions
  (clustered, uniform, outlier-heavy)
- Grid step computation: verify power-of-10 snapping for various `length_scale` values
  (0.001, 0.1, 1.0, 10.0, 1000.0)

### Manual Testing

1. **Seoul Bull** (~1-unit scale): Grid should show ~0.1-unit steps, near plane tight
2. **Dino Dog Toy** (~10-unit scale): Grid should show ~1-unit steps
3. **Verify smooth transitions**: fly through the scene rapidly, confirm no popping
4. **Verify no far clipping**: orbit far away from the scene, confirm distant points
   remain visible at any distance
