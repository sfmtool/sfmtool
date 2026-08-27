# `sfm merge` Command

## Overview

Merges multiple aligned `.sfmr` reconstructions into a single unified reconstruction. Input
reconstructions should be pre-aligned to a common coordinate frame (e.g., via `sfm align`).

## Command Syntax

```bash
sfm merge <RECONSTRUCTION1.sfmr> <RECONSTRUCTION2.sfmr> [MORE...] --output <OUTPUT.sfmr> [OPTIONS...]
```

At least two reconstructions are required.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output `.sfmr` file path |
| `--merge-percentile` | float | 95.0 | Percentile of correspondence distances for merge threshold |

## Merge Process

1. **Deduplicate cameras** — Merge camera intrinsics across reconstructions.
2. **Merge images** — Combine image sets, handling duplicates.
3. **Find correspondences** — Match 3D points across reconstructions via shared feature
   observations using union-find.
4. **Filter by percentile** — Reject point correspondences with distances above the
   percentile-based threshold.
5. **Merge points and tracks** — Average corresponding point positions, combine observation
   lists.
6. **Refine poses** — Run parallel PnP+RANSAC to refine camera poses in the merged
   reconstruction.
7. **Combine tracks** — Build the final unified track data.

## Usage Examples

```bash
# Merge two aligned reconstructions
sfm merge aligned/part_a.sfmr aligned/part_b.sfmr -o merged.sfmr

# Stricter correspondence threshold
sfm merge aligned/a.sfmr aligned/b.sfmr aligned/c.sfmr -o merged.sfmr --merge-percentile 80
```
