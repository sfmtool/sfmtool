# GUI User Experience

This document describes the vision, goals, and design for the sfmtool 3D
viewer — the interactive GUI for exploring Structure-from-Motion
reconstructions.

## Vision

Interacting with an SfM reconstruction should be **fun**.

> **Load a reconstruction. Move around it freely. Build spatial intuition.**

A reconstruction is a rich 3D artifact — thousands of camera positions tracing
a path through space, millions of colored points forming a ghostly model of a
real place. The sfmtool viewer is designed so that the moment you open a
reconstruction, you can fluidly navigate the 3D space, see the structure from
any angle, and understand how the cameras observed the scene. Everything
else — selection, filtering, analysis overlays — builds on top of that
foundation.

## Design Principles

### 1. Start with View Navigation

The single most important feature is the ability to move through the 3D space
easily and naturally. If navigation feels good, you explore freely. If it
feels bad, nothing else matters.

This means:
- **Smooth with any input device**: Whether you have a mouse or a trackpad,
  orbit, zoom, and pan should feel as smooth as scrolling a web page. For
  Windows trackpads, this required using the DirectManipulation APIs to map gestures
  to 3D motion directly.
- **Understand and control the camera target**: The orbit pivot (target point)
  is critical hidden state. Making it visible and directly controllable via the
  Alt modifier — combining visual feedback, nodal pan, and depth-under-cursor
  auto-targeting — keeps you oriented while exploring.
- **Continuous, fluid motion**: Orbit, pan, and zoom are always available
  through modifier keys and gestures — no mode switches needed for basic
  navigation. The transition between "circle around something" (orbit) and
  "look around from here" (Alt nodal pan) is seamless.

See [gui-viewport-navigation.md](gui-viewport-navigation.md) for the full
navigation spec.

### 2. Spatial Understanding Through Rendering

SfM co-solves two things: the **structure** (3D points) and the **motion**
(camera poses). The viewer shows both, plus spatial reference, as layers that
reinforce each other:

| Layer | What It Shows | SfM Role |
|-------|---------------|----------|
| **Point cloud** | 3D points colored from photographs | Structure |
| **Camera frustums** | Camera positions, orientations, and images | Motion |
| **Ground plane grid** | Reference frame with axes | Spatial context |

The point cloud uses Eye-Dome Lighting (EDL), a technique popularized by
CloudCompare and Potree, to create depth perception without requiring surface
normals — edges of depth discontinuities are darkened, giving the sparse point
cloud a sense of solidity.

Camera frustums are rendered as wireframe pyramids with the actual photograph
projected onto their far plane as a thumbnail image. This lets you see both
where a camera was and what it saw, directly in the 3D space.

See [gui-point-cloud-rendering.md](gui-point-cloud-rendering.md) for the point
cloud and EDL specification, and [gui-camera-views.md](gui-camera-views.md) for
camera frustum and image rendering.

### 3. Interact Directly with the Scene

Input should map naturally to what you see. When you drag to pan, the scene
moves with your cursor. When you scroll to zoom, the speed scales with
distance to the target so it feels proportional whether you're far away or
close up. Operations are tuned so that the visual result on screen matches
what your hands are doing.

This extends to picking — click a frustum to select a camera, Alt+click a
point to set the orbit target there. The pick buffer (GPU-based entity
identification) makes it easy to click on thin wireframe lines and individual
points.

### 4. Dark, Cinematic Aesthetic

The viewer uses a dark theme:
- Dark background for high contrast with colored points
- Subtle grid lines that don't compete with the data
- Cyan/white accents for selection and the target indicator
- The supernova effect creates a "star in a nebula" feel when
  revealing the target point in the point cloud

## How It Works

### What You See When You Open a Reconstruction

1. The point cloud fills the viewport, automatically framed to show all points
2. Camera frustums appear as small wireframe pyramids scattered through the
   scene, each showing a thumbnail of its photograph
3. A ground plane grid provides spatial reference at Z=0
4. An axis gizmo in the corner shows the current orientation (X=red, Y=green,
   Z=blue)

### Core Interactions

| Goal | Trackpad | Mouse |
|------|----------|-------|
| Look from a different angle | Two-finger drag | Left-drag |
| Move sideways | Shift + two-finger drag | Middle-drag |
| Get closer or further | Ctrl + two-finger drag | Scroll wheel or right-drag |
| Select a camera | Click on a frustum | Click on a frustum |
| Orbit around a point of interest | Alt+click on it, then orbit | Alt+click on it, then orbit |
| Look around from here | Alt + two-finger drag | Alt + left-drag |
| Move the target forward or back | Alt + Ctrl + two-finger drag | Alt + scroll |
| See where the orbit target is | Hold Alt | Hold Alt |
| Keep the target visible | Double-tap Alt | Double-tap Alt |
| Fly through the scene | WASD + R/F | WASD + R/F |
| Zoom to fit everything | Press Z | Press Z |
| View through a selected camera | Select frustum, then Z (or double-click frustum) | Select frustum, then Z (or double-click frustum) |

### Information Overlays

Contextual information shows up without cluttering the scene:

- **Hover overlay** (bottom-left): Shows what's under the cursor
  - Over a point: "Point3D #N | depth: X.XXXX"
  - Over a frustum: "Camera: image_name"
  - Over background with depth: "depth: X.XXXX"
- **Controls help** (top-right, togglable): Quick reference for navigation
- **Point count and camera info**: Basic reconstruction statistics

### UI Controls

A minimal set of controls in the View menu:

| Control | Purpose |
|---------|---------|
| Show Points | Toggle point cloud visibility |
| Show Camera Images | Toggle frustum and image quad visibility |
| Show Grid | Toggle ground plane grid |
| Point Size | Adjust point rendering size (log₂ scale, -3 to +3) |
| Length Scale | Adjust scene length scale (affects target indicator, frustum size) |
| Field of View | Adjust viewport FOV (10°–120°) |

## Design Influences

The interaction design draws from several sources:

- **Blender**: Viewport grid, orbit/pan/zoom control scheme, trackpad gesture
  handling approach (DirectManipulation on Windows)
- **Houdini**: Alt-key modifier for view control, explicit pivot setting
- **Potree/RealityCapture**: Click-to-set-pivot on point clouds
- **CloudCompare**: Point cloud navigation patterns

## Scene Scalability

The viewer should handle real-world reconstruction sizes:

| Metric | Target |
|--------|--------|
| Points | 10,000,000+ |
| Cameras | 10,000+ |
| Frame rate during navigation | 30+ fps |

This drives the technology choices: GPU-based rendering via wgpu, instanced
drawing, and lazy image loading. See [gui-architecture.md](gui-architecture.md)
for the technical architecture.

## Multi-Panel Layout

The viewer uses an `egui_dock`-based dockable, tabbed interface with three
panels. Panels can be re-docked, reordered, and resized:

- **3D Viewer** — Top-left (~67% width). Point cloud, frustums, navigation.
- **Image Detail** — Top-right (~33% width). Full-resolution image of the
  selected camera.
- **Image Browser** — Bottom strip (~20% height). Horizontally-scrollable
  thumbnails with click-to-select and gesture-driven panning.

Selecting a camera in any panel updates all others — clicking a frustum in the
3D viewer highlights it in the browser and loads the image in the detail panel,
and vice versa.

## Future Directions

- **Feature overlays**: Visualize tracked features, reprojection error, and
  track length on images
- **Point coloring modes**: Color by reprojection error, track length, or
  triangulation angle
- **Animation playback**: Fly through the camera path as a sequence
- **Image browser grid mode**: Multi-row thumbnail layout
