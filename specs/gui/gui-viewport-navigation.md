# Viewport Navigation

This document specifies the viewport navigation behavior for the sfmtool 3D
viewer — the orbit camera model, input controls, and the Alt-mode target
control system.

For the visual rendering of the target indicator and supernova effect, see
[gui-point-cloud-rendering.md](gui-point-cloud-rendering.md#target-indicator).
For the overall user experience goals, see
[gui-user-experience.md](gui-user-experience.md).

## Coordinate System

- **Z-up**: The world coordinate system uses Z as the up direction
- **Right-handed**: The coordinate system is right-handed.
- **Origin**: The world origin (0, 0, 0) is the primary reference point
- **Ground plane**: The XY plane at Z=0 serves as the ground plane grid

## Navigation Model

The viewport uses an **orbit camera** model with these properties:

| Property | Description |
|----------|-------------|
| `position` | Camera location in world space |
| `orientation` | Quaternion rotation from world to camera coordinates |
| `target_distance` | Distance from camera to the target/pivot point |
| `world_up` | The world up direction, default (0, 0, 1). Modified by tilt/roll controls (see [Dolly / Fly Navigation](#dolly--fly-navigation)). |
| `fov` | Vertical field of view in radians, adjustable via FOV zoom (see [FOV and aspect ratio](#fov-and-aspect-ratio)) |

### Target Point

The **target point** is the point `target_distance` units in front of the camera
(along the forward direction). This point serves as:
- The orbit pivot during rotation
- The focus point for zooming
- The point the camera is always looking at

The target point moves with the camera during panning, and is recalculated
during Zoom to Fit to center on the point cloud.

## Controls

### Mouse Controls

| Action | Input | Behavior |
|--------|-------|----------|
| Orbit | Left button drag | Rotate camera around the pivot point |
| Pan | Middle button drag | Translate camera and target parallel to view plane |
| Pan | Shift + Left button drag | Translate camera and target parallel to view plane |
| Zoom | Right button drag | Move camera toward/away from target |
| Zoom | Scroll wheel | Move camera toward/away from target |
| Nodal pan | Alt + Left button drag | Pivot camera in place; target orbits around camera (see [Target Control](#target-control-alt-mode)) |
| Target push/pull | Alt + Scroll | Move target forward/backward along view direction |
| Pan | Alt + Shift + Left button drag | Pan camera and target (same as Shift+drag, target visible) |
| Set target | Alt + Click | Set target to the point under the cursor (depth pick) |
| Enter camera view | Double-click frustum | Select frustum and enter camera view mode (see [gui-camera-views.md](gui-camera-views.md)) |

### Trackpad Controls

| Action | Input | Behavior |
|--------|-------|----------|
| Orbit | Two-finger drag | Rotate camera around the pivot point |
| Pan | Shift + two-finger drag | Translate camera and target parallel to view plane |
| Zoom | Ctrl/Cmd + two-finger drag | Move camera toward/away from target |
| Zoom | Two-finger pinch | Move camera toward/away from target |
| Nodal pan | Alt + two-finger drag | Pivot camera in place; target orbits around camera |
| Target push/pull | Alt + pinch / Alt + Ctrl + two-finger drag | Move target forward/backward along view direction |
| Pan | Alt + Shift + two-finger drag | Pan camera and target (same as Shift+drag, target visible) |

> **Camera view mode override:** When in camera view mode, the default
> (unmodified) drag/scroll/gesture performs **nodal pan** (free-look) instead of
> orbit, and Alt+drag performs **orbit** instead of nodal pan. Additionally, all
> zoom controls (scroll wheel, Ctrl+drag, right-drag, pinch) adjust the **FOV**
> instead of dollying the camera. This makes free-look and FOV zoom the primary
> gestures in camera view, while orbit deliberately exits it. See
> [gui-camera-views.md](gui-camera-views.md) Step 9 for details.

### Keyboard Shortcuts

| Key | Action | Behavior |
|-----|--------|----------|
| Z | Zoom to Fit | Frame all visible points (or view through selected camera if one is selected) |
| Home | Level Horizon | Reset `world_up` to Z-up without moving the camera |
| Shift+Home | Reset View | Reset to default camera position looking at origin with Z-up |
| Alt (hold) | Target Reveal | Show target indicator and supernova lighting effect |
| Alt (double-tap) | Target Toggle | Toggle target indicator to stay visible without holding Alt |
| WASD | Fly Movement | W=forward, S=back, A=left, D=right (camera-relative) |
| R / F | Fly Up / Down | Camera-relative up/down movement |
| Q / E | Tilt / Roll | Rotate horizon left/right around view axis |
| , | Previous Camera | In camera view mode: switch to previous camera image index |
| . | Next Camera | In camera view mode: switch to next camera image index |

## Default Behavior

### Initial Camera Position

When no data is loaded or after Shift+Home reset:
- Position: (0, -5, 2) - behind and above the origin, looking forward
- Target: (0, 0, 0) - the origin
- Target distance: 5.0 (approximately)

### Auto-Framing on Load

When a reconstruction is first loaded:
- Camera moves to frame all points (using the same zoom-to-fit logic)
- The center of the outlier-trimmed bounding box becomes the new target point
- Camera distance is calculated to fit that bounding box in view
- Camera orientation is preserved (default initial orientation on first load)

## Orbit Behavior

Orbiting uses spherical coordinates around the **target point** (the point
`target_distance` in front of the camera).

### Drag Direction Convention

The orbit follows a "camera moves with drag" convention:
- **Drag right**: Camera orbits clockwise (to the right) around the Z axis
- **Drag left**: Camera orbits counter-clockwise (to the left) around the Z axis
- **Drag up**: Camera moves up (decreases polar angle from Z axis)
- **Drag down**: Camera moves down (increases polar angle from Z axis)

This feels like grabbing the camera and moving it around the scene.

### Spherical Coordinates

1. **Horizontal orbit**: Changes the azimuthal angle (phi) around Z axis
   - `phi = phi - delta_x * sensitivity` (negated for correct direction)
2. **Vertical orbit**: Changes the polar angle (theta) from Z axis
   - `theta = theta - delta_y * sensitivity`

### Constraints

- Theta is clamped to (0.01, PI - 0.01) to prevent gimbal lock at poles
- The camera always maintains `world_up` orientation (no roll). `world_up`
  defaults to Z-up but can be tilted via fly mode controls.
- The target distance is preserved during orbiting

### Sensitivity

- Default orbit sensitivity: 0.01 radians per pixel of drag
- Sensitivity should be configurable in settings (future)

## Pan Behavior

Panning translates the camera parallel to the view plane.

### Drag Direction Convention

Pan follows a "grab and drag the scene" convention:
- **Drag right**: Scene moves right (camera moves left)
- **Drag left**: Scene moves left (camera moves right)
- **Drag up**: Scene moves up (camera moves down)
- **Drag down**: Scene moves down (camera moves up)

This feels like grabbing the scene and dragging it around.

### Implementation

1. Calculate the right and up vectors of the view plane
2. Move the camera position in the opposite direction of the drag
3. The target point moves with the camera (target distance preserved)

### Behavior

- The target point moves with the camera during panning
- The view direction and target distance are preserved
- After panning, orbiting happens around the new target point location

## Zoom Behavior

Zooming moves the camera toward or away from the target point:

1. Calculate new target distance by scaling current distance
2. Move camera to maintain the same target point in space
3. Minimum distance is clamped to 0.1 to prevent going through the target

### Zoom Sensitivity

- Scroll delta is multiplied by 0.1 to get a smooth zoom factor
- Zoom follows exponential scaling (multiply/divide by factor)

### Zoom-to-Cursor (Planned)

Zoom toward the point under the cursor rather than toward the target:
- Uses the existing GPU depth readback (same as Alt+click target picking)
- Useful for focusing on specific areas without explicitly setting the target

### FOV Zoom (Planned)

A separate zoom operation that adjusts the camera's field of view instead of
moving the camera. This is distinct from the standard dolly zoom (which
changes `target_distance`) — FOV zoom changes `fov` while the camera stays
in place.

FOV is already adjustable via the View menu slider (10°–120°). A dedicated
gesture binding would make it faster to adjust during navigation. When
viewing through a camera (see
[gui-camera-views.md](gui-camera-views.md#viewing-through-a-camera)),
the FOV is set automatically from camera intrinsics — FOV zoom is not needed
for that use case.

**Open question**: What input binding should FOV zoom use? It needs to be
distinct from the existing dolly zoom (scroll wheel / Ctrl+two-finger drag).
Most 3D applications treat FOV as a camera property rather than a viewport
gesture (see survey below), so this is lower priority now that the menu slider
exists.

Precedent from other applications:
- **Blender**: No default shortcut for viewport FOV. The viewport focal
  length is only adjustable via the N-panel sidebar (View tab → Focal
  Length). Blender uses Shift+Ctrl+MMB drag for dolly (moving the camera
  through the target point), which is distinct from scroll-wheel zoom.
- **Cinema 4D**: Uses the **2 key** + Ctrl+drag for dolly zoom. FOV is
  controlled through camera properties, not a viewport gesture.
- **Maya**: The `\` key toggles a mode where zoom/track don't adjust camera
  parameters — but FOV itself is a camera property, not a viewport control.
- **Houdini**: Alt+RMB drag for zoom. No dedicated FOV gesture; focal
  length is a camera parameter.

## Dolly / Fly Navigation

WASD keys for first-person movement through the scene, like a video game fly
camera. The keys sit right next to the Shift/Ctrl/Alt modifiers that are
already used for navigation, so they're easy to reach without changing hand
position.

| Key | Action |
|-----|--------|
| W | Move forward (along view direction) |
| S | Move backward |
| A | Strafe left |
| D | Strafe right |
| R | Move up (camera-relative up) |
| F | Move down (camera-relative down) |
| Q | Tilt (roll) left |
| E | Tilt (roll) right |
| Home | Level horizon (reset `world_up` to Z-up) |

The target point moves with the camera during fly movement (same as pan), so
that releasing WASD and orbiting still feels natural — you orbit around
whatever is in front of you at the current target distance.

Speed scales with target distance — moving fast when far from things, slow
when close. Shift acts as a sprint multiplier (3×) for quick repositioning.

Any fly key press also exits camera view mode, so you can start flying
immediately from a camera view.

### Tilt / Roll

Q and E rotate the camera's `world_up` direction around the view axis. This
tilts the horizon — orbit and pan will then operate relative to the new up
direction. The target indicator (visible when holding Alt) shows the current
up orientation, so you can see how tilted you are.

Home levels the horizon by resetting `world_up` to (0, 0, 1) without moving
the camera. Shift+Home does a full view reset (position, orientation, and
`world_up`).

**Open question**: Should there also be a mouse-drag binding for tilt/roll?
A natural candidate would be Ctrl+drag (or Ctrl+left-drag), since Ctrl is
already the zoom modifier for trackpad and tilt is a less common operation.
Another option is middle-click drag without Shift. Needs experimentation to
see what feels right.

While any WASD/R/F/Q/E key is held, mouse drag and two-finger drag switch to
nodal pan (same as Alt+drag — camera stays fixed, view direction changes,
target slides to new look-at point). This matches video game conventions where
mouse-look rotates your view while moving. Fly mode does not affect the target
indicator or supernova visibility — if the target is visible (via Alt hold or
Alt double-tap lock), it stays visible while flying.

The navigation mode is locked when a fly-key-initiated drag starts and held
until the drag ends. If you release the fly keys while still dragging, the
drag completes as nodal pan — it does not switch to orbit mid-gesture. The
next drag after that returns to normal orbit. This does not apply to Alt —
pressing and releasing Alt mid-drag switches between orbit and nodal pan
immediately, which is core to the fluid exploration workflow described in
[Target Control](#target-control-alt-mode).

This complements orbit navigation well: orbit to examine something from
different angles, then fly to reposition to a completely different part
of the scene.

## Zoom to Fit Behavior

The Zoom to Fit operation (Z key) adjusts the camera to show all content while
**preserving the current viewing angle**. This feels like a "zoom to fit" that
keeps your orientation, rather than snapping to a fixed viewpoint.

### Algorithm

1. Transform all visible points into **camera space** (right / up / forward axes)
2. Sort each axis independently and compute a **percentile bounding box**
   using the 20th–80th percentile range, which automatically ignores outliers
3. Find the center of the percentile box in camera space (cx, cy, cz)
4. Measure the view-plane extent: `view_size = max(sx, sy)` where sx/sy are
   the right/up widths of the percentile box
5. Calculate the required camera distance: `distance = (view_size * 1.2) / tan(fov/2)`
   (clamped to a minimum of 1.0)
6. Convert the bounding box center back to world space
7. Position the camera at `world_center - forward * distance`
8. Set `target_distance` to the computed distance

The 1.2× margin provides comfortable framing with some space around the points.

### View Through Selected Camera

When a camera frustum is selected, pressing Z sets the viewport camera to
match that frustum's pose — position, orientation, and field of view —
effectively "looking through" that camera. The camera's photograph is
displayed as a full-resolution background image. Pressing Z with nothing
selected returns to the normal zoom-to-fit behavior.

See [gui-camera-views.md](gui-camera-views.md#viewing-through-a-camera)
for the full camera view specification including FOV best-fit, background
image rendering, and `,`/`.` navigation between cameras.

## Rendering Details

### Near-Plane Line Clipping

Grid lines and axis lines are clipped against the camera's near plane to prevent
them from disappearing when one endpoint is behind the camera.

When projecting a line segment:
1. Transform both endpoints to view space
2. Check if each point is in front of the near plane (z < -near in view space)
3. If one point is behind the near plane, compute the intersection point
4. Project the clipped line segment to screen coordinates

This ensures grid lines remain visible when the camera is close to the ground plane.

### FOV and Aspect Ratio

`fov` is the field of view of the **shorter** viewport dimension. In
landscape, `fov` applies to the vertical axis. In portrait, `fov` applies to
the horizontal axis, and the vertical FOV is derived as
`atan(tan(fov/2) / aspect) * 2`. This way the same `fov` value gives a
consistent sense of "how much you can see" regardless of window shape.

The FOV is adjustable at runtime via the View menu slider (10°–120°, default
45°). It can also be adjusted via FOV zoom (see
[FOV Zoom](#fov-zoom-planned)) — this is needed both for viewing through a
camera at its native FOV and for general-purpose field of view adjustment.

## Target Control (Alt Mode)

The orbit target is arguably the most important hidden state in orbit-camera navigation.
When it drifts away from what the user is looking at, orbiting feels broken — the camera
swings around an invisible point in empty space, zoom stops working ("zoom plateau"),
and the user loses spatial orientation. This section specifies explicit target control
via the **Alt modifier key**.

### Industry Context

Most 3D applications treat the orbit pivot as invisible state that users control
indirectly (frame selection, orbit-around-selection preference, etc.). A few do more:

| Application | Approach | Visual Feedback |
|-------------|----------|-----------------|
| Houdini | **Set Pivot on Tumble** (default) — auto-sets pivot to geometry under cursor when orbiting begins. Explicit set via Space+Z click. | Togglable crosshair at pivot location |
| 3ds Max | **Orbit Point of Interest** mode — orbit center follows cursor raycast. | Green dot (steering wheel), yellow trackball circle |
| Fusion 360 | Shift+MMB click to set orbit point. | Red dot at set location |
| CloudCompare | Dedicated **Pick Rotation Center** tool — click any point to set pivot. | Reticle icon indicates mode |
| Potree | **Double-click** on point cloud to set orbit center and zoom to it. | Colored circle at pivot (Earth Controls mode) |
| RealityCapture | **Double-click** to set scene pivot on point cloud. | Pivot widget at location |

### Design: Alt as the Target Control Modifier

**Alt** is chosen because:
- It is the standard 3D navigation modifier in the Maya/Cinema 4D/Houdini tradition
  (Alt+LMB = orbit, Alt+MMB = pan, Alt+RMB = dolly)
- It is ergonomically accessible as a thumb key while the hand is on the mouse
- It creates a clean mental model: "Alt = I'm thinking about the target"

### Activation and Visual Feedback

#### Holding Alt: Target Reveal

While Alt is held, the viewport renders a persistent **target indicator** at the current
target point. This serves two purposes: confirming where the target is, and providing a
spatial anchor for understanding the 3D structure around it.

#### Double-Tap Alt: Target Toggle

Double-tapping Alt (two presses within 300ms) toggles the target indicator and supernova
effect to stay visible without holding Alt. Double-tap again to turn it off. This is
useful when you want to keep the target visible while navigating normally.

**3D Shape at Target**:
- A small **rotating 3D compass** at the target point, slowly spinning to
  convey that it is a live interactive element
- The vertical axis is elongated/spiky (top spike 1.5, bottom spike 0.7)
  to clearly show the current `world_up` direction. The horizontal part
  is a filled compass-rose star with a 32-segment circular ring outline.
  Cardinal tips (N/S/E/W) extend to 1.25× the ring radius; intercardinal
  tips extend to 0.8×. See [gui-point-cloud-rendering.md](gui-point-cloud-rendering.md#3d-shape-rotating-compass)
  for full geometry details.
- **World-space size**: radius = tunable multiplier × `length_scale` (currently
  using point size as a temporary proxy for `length_scale`)
- Color: bright cyan or white with slight glow, distinct from point cloud colors
- Rendered with depth testing against the scene (not overlaid), with depth-aware
  transparency:

  | Depth Relationship | Opacity | Description |
  |-------------------|---------|-------------|
  | In front of all geometry | 100% | Fully visible, unoccluded |
  | Just behind occluding geometry | 20% | Immediate drop on occlusion |
  | Further behind | 20% → 5% | Fog-like falloff based on world-space depth distance behind the occluder, scaled as a tunable multiplier on `length_scale` |
  | Deep behind geometry | 5% (floor) | Never fully invisible |

- The fog falloff distance is a tunable multiplier on `length_scale`, allowing
  experimentation with different falloff rates

**Supernova Lighting Effect**:
- Create a spherical illumination effect centered at the target point that gives
  the surrounding point cloud a "supernova" or "star in a nebula" feel
- Each fragment's view-space 3D position is reconstructed from its UV coordinates
  and linear depth, then the true 3D distance to the target is computed. Points
  closer to the target in 3D space glow brighter, creating a real spherical
  falloff that reveals the local 3D structure of the point cloud around the target.
- The effect uses an inverse-square falloff: `r² / (dist² + r²)` where `r` is
  the target indicator radius (derived from `length_scale`). This creates a
  localized "lantern" illumination that naturally adapts to the scene scale.
- Implementation: pass the target's view-space position (3 floats), plus
  `tan_half_fov` and `aspect` ratio to the EDL shader as additional uniforms.
  The fragment shader reconstructs view-space XY from UV + depth to compute
  the 3D distance. The result is additively blended after EDL shading.

### Target Movement Controls

While **Alt** is held, the standard navigation inputs are reinterpreted to move the
target instead of the camera. The core idea is a **dual orbit** symmetry:

#### The Dual Orbit Model

| Mode | What Moves | What's Fixed | Mental Model |
|------|-----------|--------------|--------------|
| Normal (no Alt) | Camera orbits around target | Target stays fixed | "I'm circling around this thing" |
| Alt held | Target orbits around camera | Camera stays fixed | "I'm standing still, looking around" |

This is the key insight: **normal orbit and Alt orbit are exact duals of each other**.
In normal mode, the camera moves on a sphere centered at the target. In Alt mode,
the target moves on a sphere centered at the camera. Both use the same spherical
coordinate math, just with the roles of camera and target swapped.

The Alt orbit is equivalent to a **nodal pan** (also called "look around" or "tripod
pan") — the camera pivots in place while the view direction sweeps across the scene.
The target slides to wherever the camera is now looking, at the current target distance.

**Why this pairing is powerful for navigation:**
- Switching between the two modes lets you fluidly explore a 3D scene without
  ever losing your bearings
- Normal orbit: "I've found something interesting, let me circle around it"
- Alt orbit: "I want to stay here and look around to find something interesting"
- The transition is seamless — release Alt and you're orbiting around wherever
  you were just looking. Hold Alt again and you're looking around from wherever
  you just orbited to.
- In a point cloud, this solves the common problem of the orbit pivot drifting:
  hold Alt, look at the area you care about, release Alt, orbit naturally

#### Full Control Mapping

| Input | Normal Mode | Alt Mode (Target Control) |
|-------|-------------|---------------------------|
| Two-finger drag / LMB drag | Orbit camera around target | **Nodal pan**: Target orbits around camera (camera pivots in place, view direction changes, target slides to new look-at point at current distance) |
| MMB drag | Pan camera and target together | (same as normal mode) |
| RMB drag / Scroll wheel / pinch | Zoom: move camera toward/away from target | **Target Push/Pull**: Move the target forward/backward along the view direction. Camera stays fixed, target distance changes. |
| Shift + two-finger drag / Shift + LMB drag | Pan camera and target together | **Pan**: Same as normal pan (camera and target translate together). The target indicator is visible because Alt is held. |

**Target Push/Pull** (Alt + Scroll) deserves specific attention: it directly solves the
"zoom plateau" problem. When you've zoomed in close and normal scrolling slows to a
crawl (because zoom speed is proportional to target distance), Alt+Scroll lets you
push the target deeper into the scene, effectively resetting the zoom range.

**Alt + Ctrl + Scroll**: An alternative binding for target push/pull, in case Alt+Scroll
conflicts with OS-level shortcuts on some platforms.

#### Nodal Pan Implementation

The nodal pan reuses the existing spherical orbit math with swapped roles:

1. Current orbit computes camera position on a sphere centered at the target:
   - `camera_pos = target_pos + sphere_to_cartesian(theta, phi, target_distance)`
2. Nodal pan computes target position on a sphere centered at the camera:
   - `target_pos = camera_pos + sphere_to_cartesian(theta', phi', target_distance)`
   - where `theta'` and `phi'` are updated by the drag delta (same sensitivity)
3. The camera orientation is updated to look at the new target position
4. Target distance is preserved

The drag direction convention should feel consistent: dragging right in normal mode
swings the camera right around the target; dragging right in Alt mode swings the
view direction right (target moves right relative to camera), which is the same
screen-space direction. This maintains the "grab and move" metaphor in both modes.

### Depth-Under-Cursor Auto-Targeting

The most common way users will want to set the target is "make the target be *that
point*" — the point they're looking at. This is the "click-to-set-pivot" pattern seen
in Houdini (Space+Z), CloudCompare (Pick Rotation Center), and Potree/RealityCapture
(double-click).

**Alt + Click** (single click, no drag):
1. Read the **linear depth buffer** at the cursor position
2. If a point is found at the cursor pixel (depth ≠ background):
   - Unproject the cursor position at that depth to get a 3D world-space point
   - Animate the target to the new point over ~200ms with ease-in/ease-out.
     Use spherical linear interpolation (slerp) for the camera orientation and
     smooth interpolation for position and target distance, so the transition
     feels like a fluid camera move rather than a snap.
   - Flash the target indicator at the destination to confirm the new location
3. If no point is at the exact cursor pixel, expand to a **3×3 pixel neighborhood**
   (or up to a configurable radius, e.g., 5px). Take the nearest valid depth within
   that window. This accommodates the sparse nature of point clouds where the cursor
   may fall between points.
4. If no point is found within the search window, do nothing (ignore the click).
   Do not move the target to an arbitrary depth — this would be confusing.

**Why depth buffer rather than kd-tree query?**
- The depth buffer is already rendered and available — no additional computation
- It naturally handles occlusion (you pick what you *see*, not what's behind it)
- It works at interactive frame rates regardless of point cloud size
- The existing `linear_depth` texture in the EDL pipeline provides exactly the
  data needed

### Interaction Flow Examples

#### Workflow 1: Click-to-target

1. **Load reconstruction** → camera frames all points, target is at the centroid
2. **Orbit and zoom** to get close to an area of interest
3. **Hold Alt** → target indicator appears, supernova effect reveals the target is
   still at the centroid, far behind the points the user is looking at
4. **Alt + Click** on a point in the area of interest → target animates smoothly
   to that point, flash confirms the new location, supernova effect now
   illuminates the local 3D structure around the clicked point
5. **Release Alt** → indicator and supernova fade, but the target remains at the
   new location
6. **Orbit normally** → camera now orbits smoothly around the point of interest

#### Workflow 2: Nodal pan to reorient

1. Orbiting around a building corner, want to look at a different face
2. **Hold Alt + drag** (nodal pan) → camera stays put, view sweeps across the scene
   to the other face of the building. Target slides to the new look-at point.
3. **Release Alt** → now orbiting around the new face
4. **Orbit normally** to examine the new area from different angles

#### Workflow 3: Depth adjustment

1. Zoomed into a dense area, but zoom has plateaued (target is too close)
2. **Hold Alt + scroll forward** → target pushes deeper into the scene
3. **Release Alt** → zoom range is restored, scrolling feels responsive again

#### Workflow 4: Fluid exploration (combining modes)

1. **Orbit** to circle around current target
2. **Hold Alt + drag** to look around from current position (nodal pan)
3. **Still holding Alt, click** on a distant feature → target jumps there
4. **Release Alt + orbit** around the new feature
5. Repeat — the alternation between "circle around" and "look around" gives
   complete spatial control without ever needing to hunt for a menu or tool

### Edge Cases and Fallbacks

| Scenario | Behavior |
|----------|----------|
| Alt held but no point cloud loaded | Show target indicator at current target (origin or last set). Supernova effect has nothing to illuminate. |
| Alt+Click on empty space (no depth) | Expand search to 3×3 neighborhood. If still nothing, do nothing — don't move the target. |
| Target pushed behind camera (negative distance) | Clamp target distance to a minimum of 0.1 (same as zoom minimum). |
| Target pulled very far away | Allow it — the user may want a very distant pivot for wide orbits. |
| Alt+Drag near edge of viewport | Standard behavior, no special casing needed. |
| OS intercepts Alt key (e.g., Alt activates menu bar on Windows) | May need to consume the Alt key event before it reaches the OS. Investigate `winit` key handling. This is a known issue in many 3D apps on Windows. |

### Implementation Notes

**Shader changes (EDL pass)**:
- Add uniforms to the EDL shader: `target_view_pos: vec2<f32>` (XY),
  `target_view_z: f32`, `target_active: f32` (0.0 or 1.0),
  `tan_half_fov: f32`, `aspect: f32`
- When `target_active > 0.0`, reconstruct each fragment's view-space position
  from its UV and linear depth, compute the 3D distance to the target, and
  apply an inverse-square falloff for the glow intensity
- The result is additively blended onto the EDL-shaded color

**Depth buffer readback**:
- The `linear_depth` texture is already available for HUD/diagnostic purposes
- For Alt+Click, read back a small region (e.g., 5×5 pixels) around the click
  position. This can be done asynchronously with `wgpu::Buffer::map_async` to
  avoid stalling the GPU pipeline.
- Readback runs every frame for hover overlay; click handling uses the same result

**Target indicator rendering**:
- Render as a small set of instanced lines (wireframe octahedron = 12 edges) in
  a separate mini-pass or as part of the existing line rendering
- World-space size: radius = `target_indicator_size_multiplier` × `length_scale`
  (default: 3.0, tunable)
- Apply a slow rotation (e.g., 30°/sec around the world up axis) for visual life
- Use additive blending for the glow effect
- Depth-aware transparency: render with depth testing enabled, then compare
  fragment depth against the scene depth buffer. Opacity is 100% when unoccluded,
  drops to 50% when just behind geometry, fades via fog falloff to a 10% floor.
  Fog falloff distance = `target_indicator_fog_multiplier` × `length_scale`
  (tunable, experiment to find a good default)

## Implementation Status

### Implemented

- [x] Orbit around target point (spherical coordinates)
- [x] Pan (Shift+drag / middle-drag) with correct drag direction
- [x] Scroll wheel zoom toward/away from target
- [x] Trackpad two-finger orbit, Shift+two-finger pan, pinch-to-zoom
- [x] Ctrl+two-finger drag zoom (Blender convention)
- [x] Zoom to Fit (Z key) — percentile-based framing preserving view angle
- [x] Reset View (currently Home, moving to Shift+Home)
- [x] Near-plane clipping for grid and axis lines
- [x] Windows precision touchpad integration via DirectManipulation
- [x] Target control (Alt mode):
  - [x] Alt+drag nodal pan (dual orbit)
  - [x] Alt+scroll target push/pull
  - [x] Alt+click depth-pick to set target
  - [x] Alt hold / double-tap to reveal target indicator
  - [x] Target indicator with supernova lighting (see [gui-point-cloud-rendering.md](gui-point-cloud-rendering.md#target-indicator))

### Future Enhancements

- [x] Remap Zoom to Fit from F to Z key
- [x] Remap keys: Home→Shift+Home for Reset View, Home for Level Horizon
- [x] Orbit and nodal pan use `world_up` (no longer hardcoded to Z-up)
- [x] Camera view sets `world_up` to match the camera's actual up direction
- [x] Dolly / fly navigation (WASD + R/F + Shift sprint)
- [x] Tilt / roll (`world_up` modification via Q/E)
- [x] FOV based on shorter viewport dimension (see [FOV and aspect ratio](#fov-and-aspect-ratio))
- [x] View through selected camera (Z with frustum selected)
- [x] Camera image navigation (`,`/`.` to step through cameras in camera view mode)
- [ ] Mouse-drag binding for tilt/roll (open question — Ctrl+drag? See [Tilt / Roll](#tilt--roll))
- [x] Animated target transitions (slerp + ease-in/ease-out over ~200ms on Alt+click)
- [x] Target indicator redesign: 3D compass shape with filled star rose showing `world_up` (see [Activation and Visual Feedback](#activation-and-visual-feedback))
- [ ] FOV zoom (adjust field of view without moving camera; see [FOV Zoom](#fov-zoom-planned))
- [ ] Configurable sensitivity settings
- [ ] Zoom-to-cursor (depth readback to zoom toward point under cursor)
- [ ] Inertial scrolling / smooth animation
- [ ] Save/restore camera positions

---

## Windows Precision Touchpad Support

### Problem Statement

On Windows, trackpad viewport navigation works poorly compared to Blender. The winit
library does not properly expose precision touchpad gestures to applications.

### Solution: DirectManipulation API

Microsoft's DirectManipulation API provides hardware-accelerated gesture recognition for
precision touchpads. This is the same approach used by Blender and Firefox.

**Key benefits:**
- Automatic pan vs. pinch gesture detection
- Noise filtering and smoothing
- Works with all Windows Precision Touchpads

### Implementation

The implementation is in `crates/sfm-explorer/src/platform/windows.rs`:

1. Create DirectManipulation manager via `CoCreateInstance`
2. Create a viewport configured for pan/pinch gestures with inertia
3. Set viewport to `MANUALUPDATE` mode (we drive updates each frame)
4. Implement `IDirectManipulationViewportEventHandler` COM interface
5. Subclass the window to intercept `DM_POINTERHITTEST` messages
6. On touchpad contact, call `viewport.SetContact(pointerId)`
7. Each frame, call `update_manager.Update()` to process gesture state
8. Extract pan/scale deltas from the transform matrix in `OnContentUpdated`

### Using DirectManipulation with Winit

Winit's window creation and event loop interact poorly with DirectManipulation out of the box.
Three issues must be worked around:

1. **`DM_POINTERHITTEST` is never generated** if the DM manager is created after winit's
   `EventLoop` initialization. DirectManipulation installs internal hooks when the manager is
   activated; something about winit's prior initialization prevents those hooks from observing
   precision touchpad input.

2. **`DM_POINTERHITTEST` is delivered via `SendMessage`**, which bypasses winit's message queue
   and `msg_hook`. Winit's wndproc receives the message but has no knowledge of
   DirectManipulation, so `SetContact()` is never called and gestures never begin.

3. **`WM_TIMER` is not delivered** when a winit-created window exists, so the standard pattern
   of calling `update_manager.Update()` from a timer callback does not work. Updates must be
   driven manually.

#### Workaround

The following initialization order resolves all three issues:

```
OleInitialize()
EnableMouseInPointer(true)
CoCreateInstance(DirectManipulationManager)   ← BEFORE winit
manager.GetUpdateManager()                    ← BEFORE winit

EventLoopBuilder::default().build()           ← winit EventLoop
event_loop.create_window(...)                 ← winit window

SetWindowSubclass(hwnd, dm_subclass_proc)     ← intercept DM_POINTERHITTEST
manager.CreateViewport(hwnd)                  ← attach DM to the winit HWND
viewport.ActivateConfiguration(...)
viewport.SetViewportOptions(MANUALUPDATE)
viewport.AddEventHandler(hwnd, handler)
viewport.SetViewportRect(...)
manager.Activate(hwnd)
viewport.Enable()

// In the winit event loop (ControlFlow::WaitUntil at ~16ms):
update_manager.Update(None)                   ← drive DM manually each frame
```

#### Subclass Procedure

The subclass procedure intercepts `DM_POINTERHITTEST` (0x0250) and calls `SetContact()`
synchronously before returning. All other messages are forwarded to winit via `DefSubclassProc`.

```rust
unsafe extern "system" fn dm_subclass_proc(
    hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM,
    _uid_subclass: usize, _dw_ref_data: usize,
) -> LRESULT {
    if msg == 0x0250 {  // DM_POINTERHITTEST
        let pointer_id = (wparam.0 & 0xFFFF) as u32;
        viewport.SetContact(pointer_id);  // must be called synchronously
        return LRESULT(0);
    }
    DefSubclassProc(hwnd, msg, wparam, lparam)
}
```

(In practice, the viewport is accessed via a global or passed through `dw_ref_data`.
See `winit_directmanipulation.rs` for the full implementation.)

`SetContact()` **must** be called synchronously inside the subclass proc. DirectManipulation
uses `SendMessage` precisely because it expects an immediate response. Deferring the call
(e.g., posting a message) causes DM to assume the app declined to track that contact.

#### Why Early DM Manager Creation Matters

When `CoCreateInstance(DirectManipulationManager)` is called **before** winit's `EventLoop`
initialization, `DM_POINTERHITTEST` is generated with the correct pointer type (type=5,
PT_TOUCHPAD). When created **after** winit initialization, the message is never generated —
DirectManipulation's internal hooks fail to observe precision touchpad input.

The exact mechanism is not fully understood, but a series of tests working to isolate the problem
narrowed it down: winit's `EventLoop::new()` or window creation changes something about how
Windows routes pointer input such that DirectManipulation's observation hooks no longer fire.
Creating the DM manager first avoids this interaction.

#### Why Manual Update Calls Are Needed

DirectManipulation in `MANUALUPDATE` mode expects the application to call
`update_manager.Update()` periodically so it can process accumulated gesture state and fire
`OnContentUpdated` callbacks. The standard approach uses `SetTimer` / `WM_TIMER`, but
`WM_TIMER` messages are not delivered when a winit-created window exists (the cause is
unknown but was consistently observed across tests of winit).

The workaround is to call `Update()` from winit's event loop using `ControlFlow::WaitUntil`
with a ~16ms interval, triggered from `new_events` when `StartCause::ResumeTimeReached` fires.

#### Reference Implementation

See `crates/sfm-explorer/src/bin/winit_directmanipulation.rs` for a minimal working
example of all three workarounds combined. Compare with
`crates/sfm-explorer/src/bin/win32_directmanipulation.rs` for a minimal working
example directly using Win32.

Additional test binary:

- `winit_wgpu_directmanipulation.rs` — winit + wgpu + DM (working). Tests that wgpu/DXGI
  surface creation does not interfere with DirectManipulation.

All test binaries require the `directmanipulation` Cargo feature:

```
cargo run --bin win32_directmanipulation --features directmanipulation
cargo run --bin winit_directmanipulation --features directmanipulation
cargo run --bin winit_wgpu_directmanipulation --features directmanipulation
```

### Known Limitation: Eframe Window Creation

Extensive investigation confirmed that DirectManipulation does not work on windows created through
eframe's `WgpuWinitApp::resumed()` code path. The symptom is that `DM_POINTERHITTEST` is never
generated — all pointer events arrive as `PT_MOUSE` (4) instead of `PT_TOUCHPAD` (5), meaning
DM's internal observation hooks fail to install.

The root cause was never fully identified. Every individual component of eframe's initialization
(window creation, wgpu setup, egui context, event dispatch wrapper, etc.) works correctly when
tested in isolation. The failure only occurs when running through eframe's actual
`WinitAppWrapper<WgpuWinitApp>` as the `ApplicationHandler`. The process is not globally
poisoned — a second window created after eframe initialization receives DM events correctly.

The working solution bypasses this issue by setting up DirectManipulation directly on the winit
window handle, using the three workarounds documented above. Since sfm-explorer creates its own
winit event loop and window (not using `eframe::run_native`), this limitation does not apply.
