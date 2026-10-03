# `sfm sift` Command

## Overview

`sfm sift` detects SIFT keypoints and descriptors in a set of images and writes
one `.sift` file per image for `sfm match` and `sfm solve`; with `--draw` it
draws those keypoints onto copies of the images. Exactly one action mode must
be specified per invocation.

The command is implemented in
[`src/sfmtool/_commands/sift.py`](../../../src/sfmtool/_commands/sift.py), with
extraction in [`src/sfmtool/sift/extract.py`](../../../src/sfmtool/sift/extract.py).

To inspect an existing `.sift` file, use `sfm inspect <FILE.sift>`.

## Command Syntax

```bash
sfm sift [PATHS...] --extract | --draw <DIR> [OPTIONS...]
```

`PATHS` are image files or directories. When omitted, the current directory is
used. Directories are expanded recursively, keeping only files ending in
`.png`, `.jpg` or `.jpeg` (in any letter case); a file named directly is used
whatever its extension. The command errors if no image remains after expansion
and `--range` filtering.

## Action Modes

Exactly one of these is required:

| Mode | Description |
|------|-------------|
| `--extract / -e` | Extract SIFT features from images, writing `.sift` files |
| `--draw / -d <DIR>` | Draw SIFT features as ellipses on images, saving to `<DIR>` |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter-sfm` | path | | Only draw features used in a `.sfmr` reconstruction (with `--draw`) |
| `--range / -r` | string | | Range expression for file numbers (e.g., `1-100`, `1-100:2`, `5,10,15`) |
| `--num-threads / -t` | int | -1 (all) | Thread count for extraction (`colmap` and `opencv` only) |
| `--tool` | `colmap` \| `opencv` \| `sfmtool` | workspace | Use this feature tool instead of the workspace |
| `--dsp / --no-dsp` | bool | off | Domain size pooling; requires `--tool colmap` |

`--dsp / --no-dsp` is rejected without `--tool`, and with any `--tool` other
than `colmap`. In workspace mode domain size pooling comes from the workspace
configuration; change it by reinitializing the workspace with
`sfm ws init --dsp --force`.

The `sfmtool` tool is the toolkit's own SIFT (the Rust `sfmtool-core`
implementation), calibrated to match COLMAP's keypoint density and descriptor
convention. It needs no external library and parallelizes internally, so it
ignores `--num-threads`. It chooses how many images to decode and extract at
once from the core count and available memory, and the
`SFMTOOL_SIFT_EXTRACT_WORKERS` environment variable overrides that number (see
[Extraction-orchestration pipelining](../../core/features/sift.md#extraction-orchestration-pipelining)).

## Workspace and Tool Modes

The feature tool and its options come from one of two places:

- **Workspace mode** (no `--tool`): the command finds the workspace containing
  the common parent directory of the images and reads the feature tool, its
  options and `feature_prefix_dir` from its `.sfm-workspace.json` (see
  [workspace.md](../../workspace/workspace.md)). If no workspace is found, the
  command errors and asks for either `sfm ws init` or `--tool`.
- **Tool mode** (`--tool` given): the workspace is not consulted at all. The
  tool runs with its default options (plus `--dsp` for `colmap`). Both
  `--extract` and `--draw` use this tool's feature directory, even for images
  inside a workspace.

## Output Location

Each image `<dir>/<name>` gets one `.sift` file:

- In workspace mode: `<dir>/<feature_prefix_dir>/<name>.sift`.
- In tool mode: `<dir>/features/<type>-<xxh128>/<name>.sift`, where `<type>` is
  `sift-colmap` (with `-dsp` and `-max<N>` suffixes when those options differ
  from the defaults), `sift-opencv` or `sift-sfmtool`, and `<xxh128>` is a
  hash of the tool name, type and options.

`--extract` skips an image whose `.sift` file already exists and is at least as
new as the image (by modification time), and reports how many images were
skipped and how many were extracted.

`--draw <DIR>` writes `<DIR>/<name>` for each image. An image whose `.sift` file
is missing is reported as an error and the command continues with the next
image.

## Usage Examples

```bash
# Extract features for all images in workspace
sfm sift --extract

# Extract with specific tool override
sfm sift --extract --tool opencv

# Extract with the sfmtool (Rust) backend
sfm sift --extract --tool sfmtool

# Visualize features on images
sfm sift --draw ./sift_viz

# Visualize only features used in a reconstruction
sfm sift --draw ./sift_viz --filter-sfm sfmr/solve_001.sfmr
```
