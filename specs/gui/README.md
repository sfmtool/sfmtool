# GUI Specifications

This directory contains the design specifications for the SfM Explorer 3D viewer —
an interactive GUI for exploring Structure-from-Motion reconstructions.

## Specification Documents

| Document | Description |
|----------|-------------|
| [gui-user-experience.md](gui-user-experience.md) | Vision, design principles, and product design. Start here to understand what the viewer is and why it's built this way. |
| [gui-viewport-navigation.md](gui-viewport-navigation.md) | Orbit camera model, input controls (mouse, trackpad, keyboard), and the Alt-mode target control system. Includes Windows DirectManipulation touchpad integration. |
| [gui-point-cloud-rendering.md](gui-point-cloud-rendering.md) | Point splat rendering, Eye-Dome Lighting post-processing, target indicator (rotating compass), and supernova lighting effect. |
| [gui-camera-views.md](gui-camera-views.md) | Camera frustum wireframes, image texture projection onto frustum far planes, GPU pick buffer, selection/hover interaction, and distorted frustum rendering. |
| [gui-multi-panel-image-browser.md](gui-multi-panel-image-browser.md) | Multi-panel layout (egui_dock), image browser strip, image detail pane, cross-panel selection model, and feature overlay design. |
| [gui-point-track-detail.md](gui-point-track-detail.md) | Point Track Detail panel: per-point track inspector showing observation thumbnails, per-observation reprojection error, and cross-panel navigation. |
| [gui-cross-panel-hover.md](gui-cross-panel-hover.md) | Cross-panel hover tracking: transient hover highlighting across 3D Viewer, Image Browser, and Image Detail panels via GPU uniforms. |
| [gui-adaptive-clip-and-grid.md](gui-adaptive-clip-and-grid.md) | Reversed-Z infinite far projection, adaptive near plane, and adaptive ground grid scaling. |
| [gui-image-animation.md](gui-image-animation.md) | Image animation playback: play through image sequence with keyboard/UI controls, camera view fly-through. |
| [gui-architecture.md](gui-architecture.md) | Technology stack (Rust, wgpu, egui, egui_dock, winit, PyO3), crate structure, multi-pass rendering pipeline, build system, and performance design. |

## Planning and Reference

| Document | Description |
|----------|-------------|
| [gui-plan.md](gui-plan.md) | Roadmap, milestone definitions, and current implementation status. |
| [blender-viewport-navigation-implementation-overview.md](blender-viewport-navigation-implementation-overview.md) | Reference analysis of how Blender implements precision touchpad navigation on Windows via DirectManipulation. Used during development of our own touchpad support. |
