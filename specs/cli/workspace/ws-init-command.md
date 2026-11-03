# `sfm ws init` Command

## Overview

Initializes an SfM workspace by creating a `.sfm-workspace.json` configuration file in the
target directory. The workspace config stores feature extraction settings used by subsequent
commands. The file is version 2
([workspace.md § Versions](../../workspace/workspace.md#versions)): its
`feature_options` carry the tool's `image_to_gray` object, and its
`feature_prefix_dir` is a new folder named from them. A version 1 workspace is
moved to version 2, keeping hand-edited options, by
[`sfm ws upgrade`](ws-upgrade-command.md).

## Command Syntax

```bash
sfm ws init [WORKSPACE_DIR] [OPTIONS...]
```

If `WORKSPACE_DIR` is omitted, the current directory is used.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--feature-tool` | `sfmtool` \| `colmap` \| `opencv` | `sfmtool` | Feature extraction backend |
| `--dsp / --no-dsp` | bool | `false` | Enable domain size pooling (COLMAP only) |
| `--max-features` | int | 8192 | Maximum features per image (COLMAP and sfmtool) |
| `--gpu / --no-gpu` | bool | `true` | GPU acceleration for SIFT (COLMAP only) |
| `--affine-shape / --no-affine-shape` | bool | `false` | Affine shape estimation (COLMAP only) |
| `--image-to-gray FORMULA` | string | `0.2126*R + 0.7152*G + 0.0722*B` | The formula over `R`, `G` and `B` that converts an image to grey (sfmtool only) |
| `--quantize-bits B` | int, 1 to 16 | none | Round the formula's value to `B` bits (sfmtool only) |
| `--force / -f` | flag | | Allow creating workspace even if nested or already exists |

## Validation Rules

- COLMAP-specific options (`--dsp`, `--gpu`, `--affine-shape`) cannot be used with any
  non-`colmap` feature tool (`opencv` or `sfmtool`). `--max-features` is allowed for
  `colmap` and `sfmtool` (both honour a per-image feature cap), but not `opencv`.
- `--gpu` and `--affine-shape` cannot both be enabled simultaneously.
- `--image-to-gray` and `--quantize-bits` are allowed only for `sfmtool`; the
  `colmap` and `opencv` tools make their own grey conversion, which the file
  records. The formula is checked against the grammar of
  [sift-file-format.md § `feature_options.image_to_gray`](../../formats/sift-file-format.md#feature_optionsimage_to_gray),
  and `B` must be an integer from 1 to 16. Either flag alone is allowed:
  `--quantize-bits` alone rounds the default formula.
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
it does not change the cache directory.

Every tool's `feature_options` carry `image_to_gray`, so the cache directory
always differs from one a version 1 workspace named. `--force` writes fresh
options from the tool's defaults and the flags, dropping any hand-edited option;
the old folder stays on disk, and the named image lists file is not touched.

The command prints the tool, its main options, the `image_to_gray` conversion
and the `feature_prefix_dir`.

## Usage Examples

```bash
# Initialize with defaults (sfmtool, 8192 features)
sfm ws init

# Initialize a specific directory with OpenCV
sfm ws init ./my_project --feature-tool opencv

# Use the COLMAP backend instead
sfm ws init --feature-tool colmap

# COLMAP with domain size pooling, more features
sfm ws init --feature-tool colmap --dsp --max-features 16384

# The green channel, rounded to 8 bits
sfm ws init --image-to-gray G --quantize-bits 8

# Re-initialize an existing workspace
sfm ws init --force
```
