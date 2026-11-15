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
| `points` (default) | Aligns via 3D points matched through their observations in shared images, with RANSAC outlier rejection |
| `cameras` | Aligns via the camera poses of the paired images |

Both methods start by pairing the images of each two reconstructions by the
hash of the image file first, and then by name: an `embedded_patches`
reconstruction stores each image's hash, and a `sift_files` one's are read from
its `.sift` files' metadata. Images renamed or moved to another folder still
pair, and images of the same name pair whether or not their files are the same,
so the same photographs at another resolution pair too. The rule, including a
file held by several images, is in
[reconstruction-alignment.md](../../core/analysis/reconstruction-alignment.md#pairing-the-images).

`points` matches observations by feature index when both reconstructions are `sift_files`.
When either is `embedded_patches`, which has no feature indexes, it matches them by keypoint
position (2 px, at least 2 shared images per point pair): the inline keypoints of an
`embedded_patches` side, the `.sift` positions of a `sift_files` one. The progress line names
the matching used. See
[point-correspondence.md](../../core/reconstruction/point-correspondence.md).

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output-dir / -o` | path | required | Output directory for aligned reconstructions |
| `--method` | `cameras` \| `points` | `points` | Alignment method |
| `--confidence` | float (0–1) | 0.7 | Confidence threshold for image matches. Used only by `--method cameras`; accepted but ignored for `--method points` (not rejected). |
| `--ransac / --no-ransac` | bool | `true` | Enable RANSAC outlier rejection (points only) |
| `--ransac-percentile` | float (0–100) | 95.0 | Percentile of distances for RANSAC threshold (points only) |
| `--ransac-iterations` | int | 1000 | RANSAC iterations (points only) |

## Multi-Reconstruction Alignment

When aligning more than one reconstruction, the command builds a connectivity graph of shared
images to determine alignment order. Alignment expands greedily from the reference: each pass
aligns every not-yet-aligned reconstruction that shares images with one already aligned, and
passes repeat until no reconstruction can be added. A reconstruction that connects to the
reference only through others is aligned through whichever aligned neighbours it has at that
moment, not through a shortest path.

## Output

Under "Analyzing connectivity", after each input's line of neighbours and the
number of images it shares with each, one line per two inputs that pair an
image says how their images were paired, for example `reference.sfmr and
part_a.sfmr: images paired: 15 by file hash, 2 by name, 2 of them different
files`. When some images paired by name are known to be different files, a
second line lists the first three of their names and how many more there are
(`Paired by name, different files: frame_000001.jpg, frame_000002.jpg,
frame_000003.jpg and 14 more`). This is information only: those images are
paired.

The reference and every aligned reconstruction are written into `--output-dir` under their
**input basename** (the reference is copied verbatim; each aligned result is saved alongside it).
Because the output name is just the basename, the command rejects up front any set of inputs in
which two paths share a basename — same-named files from different directories, or the same file
passed twice — since they would otherwise silently overwrite each other and yield fewer outputs
than inputs. Give such inputs distinct file names. Inputs located inside the output directory are
likewise rejected.

## Usage Examples

```bash
# Align two reconstructions to a reference using point correspondences
sfm align reference.sfmr part_a.sfmr part_b.sfmr -o aligned/

# Camera-based alignment
sfm align reference.sfmr other.sfmr -o aligned/ --method cameras --confidence 0.8

# Fine-tuned RANSAC
sfm align reference.sfmr other.sfmr -o aligned/ --ransac-percentile 90 --ransac-iterations 5000
```
