# Workspace Specifications

The workspace layout and the config files that live in it. Implemented across
`src/sfmtool/_workspace.py`, `src/sfmtool/camera/config.py`,
`src/sfmtool/rig/config.py` and `crates/sfmtool-core/src/workspace/`.

| Document | Description |
|----------|-------------|
| [workspace.md](workspace.md) | Workspace layout, the files `sfm ws init` creates, and how commands locate them. |
| [camera-config.md](camera-config.md) | Per-directory `camera_config.json` intrinsics, and closest-ancestor-wins resolution capped at the workspace root. |
| [named-image-lists.md](named-image-lists.md) | `.sfm-named-image-lists.json`: named sets of the workspace's images by image pattern and frame range, their validation, order and resolution to files or to a reconstruction's image names. |
| [rig-config.md](rig-config.md) | `rig_config.json`: multi-sensor rig ingestion, frame grouping, and its relationship to `.camrig`. |
