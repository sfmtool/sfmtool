# `sfm from-colmap-bin` Command

## Overview

Imports a COLMAP binary reconstruction into `.sfmr` format. Reads COLMAP's `cameras.bin`,
`images.bin`, and `points3D.bin` files and converts them to a single `.sfmr` file.

## Command Syntax

```bash
sfm from-colmap-bin <COLMAP_DIR> --image-dir <DIR> --output <OUTPUT.sfmr> [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--image-dir` | path | required | Directory containing images (for workspace and `.sift` resolution) |
| `--output / -o` | path | required | Output `.sfmr` file path |
| `--tool-name` | string | `unknown` | Tool provenance tag (e.g., `colmap`, `glomap`) |

## Input Directory

The `COLMAP_DIR` should contain COLMAP's standard binary output (typically a numbered
subdirectory like `colmap_output/0/`):

```
colmap_output/0/
  cameras.bin
  images.bin
  points3D.bin
```

## Usage Examples

```bash
# Import a COLMAP reconstruction
sfm from-colmap-bin colmap_output/0/ --image-dir images/ -o imported.sfmr --tool-name colmap

# Import a GLOMAP result
sfm from-colmap-bin glomap_output/0/ --image-dir images/ -o imported.sfmr --tool-name glomap
```
