# CLI Specifications

One spec per `sfm` subcommand, in the directory matching the category the
command is registered under in `src/sfmtool/cli.py` — the same grouping
`sfm --help` prints. Implementations live in `src/sfmtool/_commands/`.

`sfm explorer` is the only command without a spec; the viewer it launches is
specced under [../gui/](../gui/README.md).

## Workspace

| Command | Spec |
|---------|------|
| `sfm ws init` | [ws-init-command.md](workspace/ws-init-command.md) |
| `sfm camrig` (`create`, `cp`, `spherical-tiles`) | [camrig-command.md](workspace/camrig-command.md) |
| `sfm pano2rig` | [pano2rig-command.md](workspace/pano2rig-command.md) |
| `sfm insv2rig` | [insv2rig-command.md](workspace/insv2rig-command.md) |

## Image Feature

| Command | Spec |
|---------|------|
| `sfm sift` | [sift-command.md](image-feature/sift-command.md) |
| `sfm match` | [match-command.md](image-feature/match-command.md) |
| `sfm cluster-patches` | [cluster-patches-command.md](image-feature/cluster-patches-command.md) |

## Reconstruction

| Command | Spec |
|---------|------|
| `sfm solve` | [solve-command.md](reconstruction/solve-command.md) |
| `sfm inspect` | [inspect-command.md](reconstruction/inspect-command.md) |
| `sfm analyze` | [analyze-command.md](reconstruction/analyze-command.md) |
| `sfm compare` | [compare-command.md](reconstruction/compare-command.md) |
| `sfm align` | [align-command.md](reconstruction/align-command.md) |
| `sfm merge` | [merge-command.md](reconstruction/merge-command.md) |
| `sfm densify` | [densify-command.md](reconstruction/densify-command.md) |
| `sfm motion` | [motion-command.md](reconstruction/motion-command.md) |
| `sfm embed-patches` | [embed-patches-command.md](reconstruction/embed-patches-command.md) |
| `sfm xform` | [xform/](reconstruction/xform/) — see below |

### `sfm xform` sub-commands

| Sub-command | Spec |
|-------------|------|
| the command and its shared transforms | [xform-command.md](reconstruction/xform/xform-command.md) |
| `--refine-normals` | [refine-normals-command.md](reconstruction/xform/refine-normals-command.md) |
| `--refine-keypoints` | [refine-keypoints-command.md](reconstruction/xform/refine-keypoints-command.md) |
| `--localize-keypoints` | [localize-keypoints-command.md](reconstruction/xform/localize-keypoints-command.md) |
| `--select-by-distribution` | [select-by-distribution-command.md](reconstruction/xform/select-by-distribution-command.md) |
| `--find-points-at-infinity` | [find-points-at-infinity.md](reconstruction/xform/find-points-at-infinity.md) |
| `--scale-by-measurements` | [scale-by-measurements-command.md](reconstruction/xform/scale-by-measurements-command.md) |

## Visualization

| Command | Spec |
|---------|------|
| `sfm epipolar` | [epipolar-command.md](visualization/epipolar-command.md) |
| `sfm heatmap` | [heatmap-command.md](visualization/heatmap-command.md) |
| `sfm render-patches` | [render-patches-command.md](visualization/render-patches-command.md) |
| `sfm panorama` | [panorama-command.md](visualization/panorama-command.md) |

## Image Processing

| Command | Spec |
|---------|------|
| `sfm flow` | [flow-command.md](image-processing/flow-command.md) |
| `sfm undistort` | [undistort-command.md](image-processing/undistort-command.md) |

## COLMAP Interop

| Command | Spec |
|---------|------|
| `sfm to-colmap-bin` | [to-colmap-bin-command.md](colmap-interop/to-colmap-bin-command.md) |
| `sfm to-colmap-db` | [to-colmap-db-command.md](colmap-interop/to-colmap-db-command.md) |
| `sfm from-colmap-bin` | [from-colmap-bin-command.md](colmap-interop/from-colmap-bin-command.md) |
| `sfm to-nerfstudio` | [to-nerfstudio-command.md](colmap-interop/to-nerfstudio-command.md) |
