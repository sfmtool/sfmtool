# `sfm compare` Command

## Overview

Compares two `.sfmr` reconstructions, reporting differences in alignment, camera intrinsics,
image poses, and feature usage.

## Command Syntax

```bash
sfm compare <RECONSTRUCTION1> <RECONSTRUCTION2>
```

`RECONSTRUCTION1` is the reference; `RECONSTRUCTION2` is compared against it.

## Comparison Steps

1. **Alignment** — Computes a similarity transform (rotation, translation, scale) aligning
   the second reconstruction to the first using shared 3D point correspondences.
2. **Camera intrinsics** — Compares focal length, principal point, and distortion parameters
   for matching cameras.
3. **Image poses** — For images present in both reconstructions, reports positional and
   angular differences after alignment.
4. **Feature usage** — Compares which features were triangulated in each reconstruction for
   shared images.

## Usage Examples

```bash
# Compare two reconstructions
sfm compare sfmr/solve_001.sfmr sfmr/solve_002.sfmr

# Compare against ground truth
sfm compare ground_truth.sfmr sfmr/my_result.sfmr
```
