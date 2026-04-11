# `sfm to-colmap-bin` Command

## Overview

Exports a `.sfmr` reconstruction to COLMAP binary format for use with the COLMAP GUI or
other COLMAP-compatible tools.

## Command Syntax

```bash
sfm to-colmap-bin <INPUT.sfmr> <OUTPUT_DIR>
```

## Output Files

```
output_dir/
  cameras.bin
  images.bin
  points3D.bin
  rigs.bin       (if rig data present)
  frames.bin     (if rig data present)
```

## Usage Examples

```bash
# Export for COLMAP GUI
sfm to-colmap-bin sfmr/solve_001.sfmr colmap_export/

# Then open in COLMAP
# colmap gui --import_path colmap_export/ --image_path images/
```
