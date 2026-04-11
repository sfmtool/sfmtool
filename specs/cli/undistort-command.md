# `sfm undistort` Command

## Overview

Removes lens distortion from all images in a reconstruction using the solved camera
parameters. Outputs undistorted images with PINHOLE camera models.

## Command Syntax

```bash
sfm undistort <RECONSTRUCTION.sfmr>
```

## Output

- **Directory**: `{stem}_undistorted/` alongside the input `.sfmr` file
- **Camera model**: All output cameras are PINHOLE (zero distortion)
- **Image size**: Same as input (scale fixed to 1.0)
- **Metadata**: JSON file with updated camera parameters

## Usage Examples

```bash
# Undistort all images in a reconstruction
sfm undistort sfmr/solve_001.sfmr
# Output: sfmr/solve_001_undistorted/
```
