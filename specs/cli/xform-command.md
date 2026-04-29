# `sfm xform` Command Design

## Overview

The `sfm xform` command provides a unified interface for applying transformations and
filters to SfM reconstructions. Multiple operations can be chained in a single pass,
applied sequentially in the order specified on the command line.

## Command Syntax

```bash
sfm xform <input.sfmr> [<output.sfmr>] [OPTIONS...]
```

If `<output.sfmr>` is omitted, the result is written next to the input as
`<stem>-transformed.sfmr`. If that name is already taken, a numeric suffix is
appended starting at 2 (`<stem>-transformed-2.sfmr`, `-3`, ...).

### Order Matters

Transformations are applied sequentially. The output of each becomes the input to the next.

```bash
# Rotate, then translate
sfm xform in.sfmr out.sfmr --rotate 0,1,0,+25deg --translate 3,5,-2

# Translate, then rotate (different result!)
sfm xform in.sfmr out.sfmr --translate 3,5,-2 --rotate 0,1,0,+25deg
```

## Available Operations

### Geometric Transformations

#### `--rotate <axisX>,<axisY>,<axisZ>,<angle>`

Rotates all 3D points and camera poses about the origin. Axis is normalized internally.
Angle requires a unit suffix: `deg`/`degrees` or `rad`/`radians`.

```bash
--rotate 0,1,0,90deg
--rotate 1,1,0,45deg
--rotate 0,0,1,1.5708rad
```

#### `--translate <X>,<Y>,<Z>`

Translates all 3D points and camera centers. Camera orientations unchanged.

```bash
--translate 3,5,-2
```

#### `--scale <S>`

Scales all 3D points and camera positions relative to the origin. Must be positive.

```bash
--scale 2.0
```

### Filtering Operations

#### `--include-range <RANGE_EXPR>` / `--exclude-range <RANGE_EXPR>`

Filters images by file number. Supports single numbers, ranges, and comma-separated
combinations. Also removes 3D points with no remaining observations and remaps indices.

```bash
--include-range 10-50
--exclude-range "23-25,47"
```

#### `--remove-isolated <factor>,<value_spec>`

Removes 3D points whose nearest neighbor distance exceeds `factor * reference_value`.
The reference value is `median` or `<N>percent`/`<N>percentile`.

```bash
--remove-isolated 3.0,median
--remove-isolated 2.0,95percent
```

#### `--remove-short-tracks <size>`

Removes 3D points with track length (observation count) <= size.

```bash
--remove-short-tracks 2
```

#### `--remove-narrow-tracks <angle>`

Removes 3D points whose maximum viewing angle span is less than the threshold.

```bash
--remove-narrow-tracks 5deg
```

#### `--filter-by-reprojection-error <threshold>`

Removes 3D points with reprojection error exceeding the threshold (in pixels).

```bash
--filter-by-reprojection-error 2.0
```

### Optimization

#### `--bundle-adjust`

Applies bundle adjustment via pycolmap to refine camera poses and 3D point positions.

```bash
--remove-short-tracks 2 --bundle-adjust
```

### Scaling to Physical Units

#### `--scale-by-measurements <measurements.yaml>`

Scales the reconstruction to physical units using known real-world distances between pairs
of 3D points. The measurements are specified in a YAML file with point IDs, distances, and
a target unit. With multiple measurements, the median scale factor is used.

See [scale-by-measurements-command.md](scale-by-measurements-command.md) for the full
specification, including the YAML schema, unit conversion, point ID resolution across
reconstructions, and diagnostics output.

```bash
--scale-by-measurements measurements.yaml
```

### Alignment

#### `--align-to <other.sfmr>`

Aligns the current reconstruction to match another using a similarity transformation
(rotation, translation, scale) computed from shared images and 3D point correspondences.

```bash
--align-to ground_truth.sfmr
```

#### `--align-to-input`

Aligns back to the original input reconstruction. Useful after filtering operations
to keep the result in the original coordinate frame.

```bash
sfm xform in.sfmr out.sfmr --remove-short-tracks 2 --align-to-input
```

## Architecture

The implementation uses a `Transform` protocol:

```python
class Transform(Protocol):
    def apply(self, reconstruction: SfmrReconstruction) -> SfmrReconstruction: ...
```

Each operation is a class implementing this interface. The CLI parses arguments into an
ordered list of `Transform` objects and applies them sequentially. The reconstruction is
loaded once, transformed through the pipeline, and written once.

## Usage Examples

```bash
# Clean and align reconstruction
sfm xform raw.sfmr clean.sfmr \
  --remove-short-tracks 2 \
  --remove-isolated 3.0,median \
  --filter-by-reprojection-error 2.0 \
  --align-to ground_truth.sfmr

# Transform to world coordinates
sfm xform model.sfmr world.sfmr \
  --rotate 1,0,0,90deg \
  --translate 0,0,-5

# Iterative refinement
sfm xform raw.sfmr refined.sfmr \
  --remove-short-tracks 2 \
  --bundle-adjust \
  --filter-by-reprojection-error 2.0 \
  --remove-isolated 2.5,median \
  --scale 0.01

# Filter by image range and clean up
sfm xform full.sfmr subset.sfmr \
  --include-range 10-50 \
  --remove-short-tracks 2

# Filter, optimize, then scale to physical units
sfm xform input.sfmr output.sfmr \
  --remove-short-tracks 2 \
  --bundle-adjust \
  --scale-by-measurements measurements.yaml
```
