# `sfm init` Command

## Overview

Initializes an SfM workspace by creating a `.sfm-workspace.json` configuration file in the
target directory. The workspace config stores feature extraction settings used by subsequent
commands.

## Command Syntax

```bash
sfm init [WORKSPACE_DIR] [OPTIONS...]
```

If `WORKSPACE_DIR` is omitted, the current directory is used.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--feature-tool` | `colmap` \| `opencv` | `colmap` | Feature extraction backend |
| `--dsp / --no-dsp` | bool | `false` | Enable domain size pooling (COLMAP only) |
| `--max-features` | int | 8192 | Maximum features per image (COLMAP only) |
| `--gpu / --no-gpu` | bool | `true` | GPU acceleration for SIFT (COLMAP only) |
| `--affine-shape / --no-affine-shape` | bool | `false` | Affine shape estimation (COLMAP only) |
| `--force / -f` | flag | | Allow creating workspace even if nested or already exists |

## Validation Rules

- COLMAP-specific options (`--dsp`, `--max-features`, `--gpu`, `--affine-shape`) cannot be
  used with `--feature-tool opencv`.
- `--gpu` and `--affine-shape` cannot both be enabled simultaneously.
- Without `--force`, the command errors if the directory is already a workspace or is nested
  inside an existing workspace.

## Usage Examples

```bash
# Initialize with defaults (COLMAP, GPU, 8192 features)
sfm init

# Initialize a specific directory with OpenCV
sfm init ./my_project --feature-tool opencv

# COLMAP with domain size pooling, more features
sfm init --feature-tool colmap --dsp --max-features 16384

# Re-initialize an existing workspace
sfm init --force
```
