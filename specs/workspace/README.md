# Workspace Specifications

The workspace layout, the config files that live in it and the records
written into it. Implemented across `src/sfmtool/_workspace.py`,
`src/sfmtool/camera/config.py`, `src/sfmtool/rig/config.py`,
`src/sfmtool/_video_frames.py` (the `video-frames.json` record) and
`crates/sfmtool-core/src/workspace/`.

| Document | Description |
|----------|-------------|
| [workspace.md](workspace.md) | Workspace layout, the files `sfm ws init` creates, the two versions of `.sfm-workspace.json` and `sfm ws upgrade` between them, the rule that `.sift` versions never mix, and how commands locate them. |
| [camera-config.md](camera-config.md) | Per-directory `camera_config.json` intrinsics, and closest-ancestor-wins resolution capped at the workspace root. |
| [named-image-lists.md](named-image-lists.md) | `.sfm-named-image-lists.json`: named sets of the workspace's images by image pattern and frame range, their validation, order and resolution to files or to a reconstruction's image names, and the slice that takes part of a list by position (`walk1[:30]`). |
| [rig-config.md](rig-config.md) | `rig_config.json`, COLMAP's rig config: the input `sfm camrig cp` converts to a `.camrig`, and the refusal of one left at a workspace root. |
| [video-frames.md](video-frames.md) | `video-frames.json`, the record beside a video's extracted frames: the video, the extraction's options, and each frame's time in the video. |
