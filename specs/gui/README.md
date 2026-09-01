# GUI Specifications

This directory contains the design specifications for the SfM Explorer 3D viewer —
an interactive GUI for exploring Structure-from-Motion reconstructions.

## Specification Documents

| Document | Description |
|----------|-------------|
| [user-experience.md](user-experience.md) | Vision, design principles, and product design. Start here to understand what the viewer is and why it's built this way. |
| [viewport-navigation.md](viewport-navigation.md) | Orbit camera model, input controls (mouse, trackpad, keyboard), and the Alt-mode target control system. Includes Windows DirectManipulation touchpad integration. |
| [point-cloud-rendering.md](point-cloud-rendering.md) | Point splat rendering, Eye-Dome Lighting post-processing, target indicator (rotating compass), and supernova lighting effect. |
| [patch-rendering.md](patch-rendering.md) | Embedded-patch (surfel) rendering: textured oriented quads in the 3D viewport, one per 3D point carrying a patch frame. Mirrors the camera image-quad pipeline; front-face culled. |
| [camera-views.md](camera-views.md) | Camera frustum wireframes, image texture projection onto frustum far planes, GPU pick buffer, selection/hover interaction, and distorted frustum rendering. |
| [multi-panel-image-browser.md](multi-panel-image-browser.md) | Multi-panel layout (egui_dock), image browser strip, image detail pane, cross-panel selection model, and feature overlay design. |
| [camera-intrinsics.md](camera-intrinsics.md) | Camera intrinsics: a Camera Intrinsics scene-graph group beside a renamed Camera Images group, the independently-toggled intrinsics overlay layer on the Image Detail panel (principal point, angular axes, distortion field, composing with any feature/heatmap mode), and the Intrinsics panel (parameter table, projection plot, per-image extrinsics). |
| [point-track-detail.md](point-track-detail.md) | Point Track Detail panel: per-point track inspector showing observation thumbnails, per-observation reprojection error, and cross-panel navigation. |
| [goto-point.md](goto-point.md) | Go to Point: type or paste a point index or a `pt3d_<hash>_<index>` ID to select that point, switching reconstructions when the ID names a different one. |
| [cross-panel-hover.md](cross-panel-hover.md) | Cross-panel hover tracking: transient hover highlighting across 3D Viewer, Image Browser, and Image Detail panels via GPU uniforms. |
| [viewport-hud.md](viewport-hud.md) | The in-viewport HUD that owns every 3D display control: layout, sections, and the input-arbitration rules a floating panel inside the 3D viewport requires. |
| [scene-graph.md](scene-graph.md) | Multi-reconstruction support: several `.sfmr` files loaded at once as scene-graph nodes, per-node visibility/tint/solo and similarity transforms with an in-GUI "Align to…" operation, per-reconstruction GPU resource bundles, widened pick encoding, and the Scene Graph tree panel. |
| [resect-image.md](resect-image.md) | Resect Image: a per-image Scene Graph action that re-estimates one image's pose against held-out structure and shows the result as a derived node beside the original. |
| [mcp-server.md](mcp-server.md) | The `--mcp` control surface: a loopback Model Context Protocol endpoint an agent drives the running viewer through — the scene graph, the selection, the 3D camera, and a screenshot of the viewport — applied on the GUI thread at one point in the frame. |
| [adaptive-clip-and-grid.md](adaptive-clip-and-grid.md) | Reversed-Z infinite far projection, adaptive near plane, and adaptive ground grid scaling. |
| [image-animation.md](image-animation.md) | Image animation playback: play through image sequence with keyboard/UI controls, camera view fly-through. |
| [architecture.md](architecture.md) | Technology stack (Rust, wgpu, egui, egui_dock, winit, PyO3), crate structure, multi-pass rendering pipeline, build system, and performance design. |

## Planning and Reference

| Document | Description |
|----------|-------------|
| [plan.md](plan.md) | Roadmap, milestone definitions, and current implementation status. |
| [blender-viewport-navigation-implementation-overview.md](blender-viewport-navigation-implementation-overview.md) | Reference analysis of how Blender implements precision touchpad navigation on Windows via DirectManipulation. Used during development of our own touchpad support. |
