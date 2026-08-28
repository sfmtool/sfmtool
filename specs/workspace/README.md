# Workspace Specifications

The workspace layout and the config files that live in it. Implemented across
`src/sfmtool/_workspace.py`, `src/sfmtool/camera/config.py`, and
`src/sfmtool/rig/config.py`.

| Document | Description |
|----------|-------------|
| [workspace.md](workspace.md) | Workspace layout, the files `sfm ws init` creates, and how commands locate them. |
| [camera-config.md](camera-config.md) | Per-directory `camera_config.json` intrinsics, and closest-ancestor-wins resolution capped at the workspace root. |
| [rig-config.md](rig-config.md) | `rig_config.json`: multi-sensor rig ingestion, frame grouping, and its relationship to `.camrig`. |
