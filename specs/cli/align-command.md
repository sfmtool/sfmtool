# `sfm align` Command

## Overview

Aligns multiple `.sfmr` reconstructions into a common coordinate frame defined by a reference
reconstruction. Two alignment methods are available: feature-based (points) and pose-based
(cameras).

## Command Syntax

```bash
sfm align <REFERENCE.sfmr> <ALIGN1.sfmr> [ALIGN2.sfmr ...] --output-dir <DIR> [OPTIONS...]
```

The reference reconstruction defines the target coordinate frame. All other reconstructions
are transformed to align with it.

## Alignment Methods

| Method | Description |
|--------|-------------|
| `points` (default) | Aligns via shared 3D feature observations with RANSAC outlier rejection |
| `cameras` | Aligns via matched camera poses for images with the same name |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output-dir / -o` | path | required | Output directory for aligned reconstructions |
| `--method` | `cameras` \| `points` | `points` | Alignment method |
| `--confidence` | float (0–1) | 0.7 | Confidence threshold for image matches (cameras only) |
| `--max-error` | float | 0.1 | Maximum acceptable alignment error |
| `--iterative` | flag | | Enable iterative refinement |
| `--visualize` | flag | | Generate visualization data |
| `--ransac / --no-ransac` | bool | `true` | Enable RANSAC outlier rejection (points only) |
| `--ransac-percentile` | float (0–100) | 95.0 | Percentile of distances for RANSAC threshold (points only) |
| `--ransac-iterations` | int | 1000 | RANSAC iterations (points only) |

## Multi-Reconstruction Alignment

When aligning more than one reconstruction, the command builds a connectivity graph of shared
images to determine alignment order, ensuring each reconstruction is aligned through the
shortest path to the reference.

## Usage Examples

```bash
# Align two reconstructions to a reference using point correspondences
sfm align reference.sfmr part_a.sfmr part_b.sfmr -o aligned/

# Camera-based alignment
sfm align reference.sfmr other.sfmr -o aligned/ --method cameras --confidence 0.8

# Fine-tuned RANSAC
sfm align reference.sfmr other.sfmr -o aligned/ --ransac-percentile 90 --ransac-iterations 5000
```
