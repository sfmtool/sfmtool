# CLI Specifications

One spec per `sfm` subcommand, in the directory matching the category the
command is registered under in `src/sfmtool/cli.py` — the same grouping
`sfm --help` prints. Implementations live in `src/sfmtool/_commands/`.

## How `sfm` loads its commands

`sfm` imports a command's module only when that command is looked up: when it
runs, or when its own `--help` is printed. `COMMANDS` in
[cli.py](../../src/sfmtool/cli.py) lists every top-level command but `version`
with its name, its `--help` category, the module in `sfmtool._commands` and the
attribute that define it, and its one-line help, the first sentence of the
command's docstring. `CategoryGroup` in
[_cli_group.py](../../src/sfmtool/_cli_group.py) lists the commands by
category from that table, so `sfm --help` imports no command module, and
`sfm explorer` does not import the modules of the commands that need numpy,
OpenCV or pycolmap. The package root works the same way:
[`sfmtool/__init__.py`](../../src/sfmtool/__init__.py) binds the extension's
three root-level names on import, which takes about 10 ms, and the names of its
Python submodules and the binding modules such as `sfmtool.fileio` on first use,
through a module `__getattr__`. For an unknown
command name `CategoryGroup` suggests the close matches among every command
name (`sfm solv` gives "Did you mean 'solve'?"), not only among the commands
loaded so far.

So a command module does no work at import time beyond defining its command,
and `sfmtool._commands/__init__.py` imports none of them. A new command is a
row in `COMMANDS`, with the first sentence of its docstring as the one-line
help; [test_lazy_loading.py](../../tests/test_lazy_loading.py) checks that each
row names a command defined where the row says, under that name, with that
help and category, that every command defined in a `sfmtool._commands`
module has a row, and that `import sfmtool` and `sfm --help` import none of
numpy, OpenCV, pycolmap and the command modules.

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
| `sfm estimate-intrinsics` | [estimate-intrinsics-command.md](reconstruction/estimate-intrinsics-command.md) |
| `sfm xform` | [xform/](reconstruction/xform/) — see below |

### `sfm xform` sub-commands

| Sub-command | Spec |
|-------------|------|
| the command and its shared transforms | [xform-command.md](reconstruction/xform/xform-command.md) |
| `--refine-normals` | [refine-normals-command.md](reconstruction/xform/refine-normals-command.md) |
| `--refine-keypoints` | [refine-keypoints-command.md](reconstruction/xform/refine-keypoints-command.md) |
| `--localize-keypoints` | [localize-keypoints-command.md](reconstruction/xform/localize-keypoints-command.md) |
| `--include-by-distribution` | [select-by-distribution-command.md](reconstruction/xform/select-by-distribution-command.md) |
| `--find-points-at-infinity` | [find-points-at-infinity.md](reconstruction/xform/find-points-at-infinity.md) |
| `--scale-by-measurements` | [scale-by-measurements-command.md](reconstruction/xform/scale-by-measurements-command.md) |

## Visualization

| Command | Spec |
|---------|------|
| `sfm epipolar` | [epipolar-command.md](visualization/epipolar-command.md) |
| `sfm explorer` | [explorer-command.md](visualization/explorer-command.md) — launches the viewer specced under [../gui/](../gui/README.md) |
| `sfm heatmap` | [heatmap-command.md](visualization/heatmap-command.md) |
| `sfm render-patches` | [render-patches-command.md](visualization/render-patches-command.md) |
| `sfm panorama` | [panorama-command.md](visualization/panorama-command.md) |
| `sfm web-export` | [web-export-command.md](visualization/web-export-command.md) |

## Image Processing

| Command | Spec |
|---------|------|
| `sfm flow` | [flow-command.md](image-processing/flow-command.md) |
| `sfm undistort` | [undistort-command.md](image-processing/undistort-command.md) |

## COLMAP Interop

The COLMAP file reading and writing these commands share is described in
[../formats/colmap-interop.md](../formats/colmap-interop.md).

| Command | Spec |
|---------|------|
| `sfm to-colmap-bin` | [to-colmap-bin-command.md](colmap-interop/to-colmap-bin-command.md) |
| `sfm to-colmap-db` | [to-colmap-db-command.md](colmap-interop/to-colmap-db-command.md) |
| `sfm from-colmap-bin` | [from-colmap-bin-command.md](colmap-interop/from-colmap-bin-command.md) |
| `sfm to-nerfstudio` | [to-nerfstudio-command.md](colmap-interop/to-nerfstudio-command.md) |
