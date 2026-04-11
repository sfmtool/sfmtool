# `sfm sift` Command

## Overview

Extracts, inspects, and visualizes SIFT features for images in a workspace. Exactly one
action mode must be specified per invocation.

## Command Syntax

```bash
sfm sift [PATHS...] --extract | --print | --draw <DIR> [OPTIONS...]
```

`PATHS` are image files or directories. If omitted, the workspace root is used.

## Action Modes

Exactly one of these is required:

| Mode | Description |
|------|-------------|
| `--extract / -e` | Extract SIFT features from images, writing `.sift` files |
| `--print / -p` | Print summary of existing `.sift` files (feature count, dimensions) |
| `--draw / -d <DIR>` | Draw SIFT features as ellipses on images, saving to `<DIR>` |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter-sfm` | path | | Only draw features used in a `.sfmr` reconstruction (with `--draw`) |
| `--verbose / -v` | flag | | Print verbose output |
| `--range / -r` | string | | Range expression for file numbers (e.g., `1:100`, `5,10,15`) |
| `--num-threads / -t` | int | -1 (all) | Thread count for extraction |
| `--tool` | `colmap` \| `opencv` | workspace | Override workspace feature tool |
| `--dsp / --no-dsp` | bool | workspace | Override domain size pooling (requires `--tool`) |

## Workspace Integration

The command reads `.sfm-workspace.json` to determine the feature tool and settings. Use
`--tool` to override without modifying the workspace config.

## Usage Examples

```bash
# Extract features for all images in workspace
sfm sift --extract

# Extract with specific tool override
sfm sift --extract --tool opencv

# Print feature summary for a range of images
sfm sift --print --range 1:50

# Visualize features on images
sfm sift --draw ./sift_viz

# Visualize only features used in a reconstruction
sfm sift --draw ./sift_viz --filter-sfm sfmr/solve_001.sfmr
```
