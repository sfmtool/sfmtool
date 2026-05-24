# `sfm ws init` Command

## Overview

Initializes an SfM workspace by creating a `.sfm-workspace.json` configuration file in the
target directory. The workspace config stores feature extraction settings used by subsequent
commands.

## Command Syntax

```bash
sfm ws init [WORKSPACE_DIR] [OPTIONS...]
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

## Persisted Settings

The COLMAP feature options — `domain_size_pooling`, `max_num_features`,
`estimate_affine_shape`, and `use_gpu` — are written into the
`feature_options` block of `.sfm-workspace.json` and honored at extraction time
(`use_gpu` maps to `FeatureExtractionOptions.use_gpu`; passing `--no-gpu`
forces CPU SIFT, which is also required to combine with `--affine-shape`).

`use_gpu` is the one option excluded from the feature-cache hash
(`feature_prefix_dir`): GPU vs CPU is a hardware/performance choice, so toggling
it does not change the cache directory, and workspaces written before this key
existed hash identically.

## Usage Examples

```bash
# Initialize with defaults (COLMAP, GPU, 8192 features)
sfm ws init

# Initialize a specific directory with OpenCV
sfm ws init ./my_project --feature-tool opencv

# COLMAP with domain size pooling, more features
sfm ws init --feature-tool colmap --dsp --max-features 16384

# Re-initialize an existing workspace
sfm ws init --force
```
