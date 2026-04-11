# Scale by Measurements

This document specifies the `--scale-by-measurements` option for `sfm xform`, which scales a
`.sfmr` reconstruction to physical units using known real-world distances between pairs of 3D
points.

## Motivation

SfM reconstructions are solved up to an unknown scale factor. To convert reconstruction units to
physical units (e.g., millimeters), the user needs at least one known real-world distance between
two visible points. The typical workflow is:

1. Place a ruler or known-size object in the scene.
2. Solve the reconstruction.
3. Select points on the ruler in the GUI, copy their Point IDs.
4. Provide the Point IDs and real-world distance to a scaling tool.

The `--scale-by-measurements` option on `sfm xform` handles step 4, integrating naturally into the
existing transform pipeline.

## Measurements File Format

The measurements are specified in a YAML file. YAML is chosen over JSON because it supports
comments (useful for documenting what each measurement refers to) and has less syntactic noise for
hand-editing.

### Schema

```yaml
# Unit that 3D world space will be in after this transform.
# Required. Supported values: mm, cm, m, in, ft
unit: mm

# Path to the .sfmr file these Point IDs come from.
# Optional. If omitted, the tool resolves Point IDs against the input .sfmr file
# passed on the command line.
sfmr: sfmr/calib_2000feat_every7th_global_x04_isolated_ba_align.sfmr

measurements:
  - point_a: pt3d_a1b2c3d4_12345
    point_b: pt3d_a1b2c3d4_67890
    distance: 12in
    label: "ruler left to right"

  - point_a: pt3d_a1b2c3d4_11111
    point_b: pt3d_a1b2c3d4_22222
    distance: 152.4
    label: "ruler left to midpoint"
```

### Fields

#### Top-level

| Field | Required | Description |
|-------|----------|-------------|
| `unit` | Yes | The unit that 3D world space will be in after the transform is applied. Supported values: `mm`, `cm`, `m`, `in`, `ft`. |
| `sfmr` | No | Path to the `.sfmr` file that the Point IDs reference, relative to the measurements YAML file. If omitted, Point IDs are resolved against the input `.sfmr` file provided on the command line. See [Point ID Resolution](#point-id-resolution). |
| `measurements` | Yes | List of point-pair distance measurements. Must contain at least one entry. |

#### Per-measurement

| Field | Required | Description |
|-------|----------|-------------|
| `point_a` | Yes | Point ID of the first point (e.g., `pt3d_a1b2c3d4_12345`). |
| `point_b` | Yes | Point ID of the second point (e.g., `pt3d_a1b2c3d4_67890`). |
| `distance` | Yes | Known real-world distance. Either a bare number (interpreted as the target `unit`) or a number with a unit suffix (converted to the target unit). |
| `label` | No | Human-readable description of the measurement. Printed in diagnostics output but otherwise ignored. |

### Distance Values

The `distance` field accepts two forms:

1. **Bare number**: Interpreted as the target unit declared at the top level.
   - `distance: 304.8` with `unit: mm` means 304.8 mm.

2. **Number with unit suffix**: The value is converted to the target unit.
   - `distance: 12in` with `unit: mm` means 304.8 mm.
   - `distance: 0.3048m` with `unit: mm` means 304.8 mm.

Supported unit suffixes:

| Suffix | Unit | To meters |
|--------|------|-----------|
| `mm` | millimeters | x 0.001 |
| `cm` | centimeters | x 0.01 |
| `m` | meters | x 1.0 |
| `in` | inches | x 0.0254 |
| `ft` | feet | x 0.3048 |

The suffix is parsed by matching a trailing alphabetic string after the numeric value. No
whitespace is allowed between the number and the suffix (e.g., `12in` not `12 in`). The numeric
part follows standard floating-point syntax (e.g., `12`, `12.0`, `0.3048`).

### Conversion Logic

All distances are normalized to the target unit through meters as an internal pivot:

```
distance_in_target_unit = distance_in_source_unit * (source_to_meters / target_to_meters)
```

For a bare number, no conversion is needed (source unit = target unit).

## Scale Factor Computation

For each measurement:

1. Resolve both Point IDs to point indices in the input reconstruction.
2. Look up their 3D positions from the reconstruction's point array.
3. Compute the Euclidean distance in reconstruction units: `recon_distance = ||pos_a - pos_b||`.
4. Compute the per-measurement scale factor: `scale_i = real_distance / recon_distance`.

With multiple measurements, the **median** scale factor is used. The median is robust to a single
outlier measurement caused by a misidentified point or reconstruction error.

### Diagnostics Output

The command prints per-measurement diagnostics so the user can assess consistency:

```
Scale by measurements (target unit: mm):
  1. "ruler left to right": recon=0.317, real=304.8mm -> scale=961.5
  2. "ruler left to midpoint": recon=0.159, real=152.4mm -> scale=958.5
  3. "tile diagonal":         recon=0.472, real=452.0mm -> scale=957.6
  4. "ruler 3/4 mark":        recon=0.238, real=228.6mm -> scale=960.5
  5. "box width":             recon=0.314, real=301.2mm -> scale=958.6

Scale factor distribution (5 measurements):
  957.6 |####        2
  958.4 |##          1
  959.2 |
  960.0 |##          1
  960.8 |##          1
  median: 958.6, mean: 959.6, std: 1.6 (0.17%)

Scaling reconstruction by 958.6
```

The histogram uses a fixed number of equally-spaced bins (5 by default) spanning from the
minimum to maximum observed scale factor. Each `#` pair represents one measurement. The bin
edge labels use enough decimal places to distinguish bins. For a single measurement, the histogram
is omitted.

If measurements disagree significantly (e.g., >5% deviation from median), the command prints a
warning. This serves as a cross-check for reconstruction distortion — if two measurements on the
same rigid object give very different scale factors, the reconstruction geometry is locally
distorted.

### Edge Cases

- **Single measurement**: The scale factor is used directly (no median needed).
- **Zero reconstruction distance**: Error — the two points are coincident in the reconstruction.
- **Negative or zero real distance**: Error at parse time.

## Point ID Resolution

Point IDs are resolved following the
[Point ID specification](sfmr-file-format.md#point-id-portable-3d-point-references). The goal is
to find the 3D point position corresponding to each Point ID in the **input reconstruction** (the
one being transformed), even when the Point IDs were copied from a different reconstruction.

### Direct match (same reconstruction)

If the Point ID hash prefix matches the input `.sfmr` file's `content_xxh128`, the point index
is used directly. This is the simplest case — the user picked points from the same file they're
now scaling.

### Cross-reconstruction match via feature observations

If the hash prefix does **not** match the input file, the tool uses the `sfmr` field (or searches
for the matching `.sfmr` file) and resolves the point through shared feature observations. This is
the expected common case: the user picks points in one reconstruction, then applies the scaling to
a re-solved or further-filtered version.

The resolution process:

1. **Load the source reconstruction** — the `.sfmr` file referenced by `sfmr` (resolved relative
   to the YAML file's directory), or found by searching for a file whose `content_xxh128` matches
   the Point ID hash prefix (using the
   [workspace search strategy](sfmr-file-format.md#resolving-a-point-id-to-a-sfmr-file)).
2. **Map point to (image, feature) observations** — look up the point index in the source
   reconstruction's track data to get the list of `(image_name, feature_index)` pairs that
   observe this point.
3. **Find matching observations in the input reconstruction** — for each `(image_name,
   feature_index)` pair, search the input reconstruction's track data for an observation of the
   same image name and feature index. If found, that observation's 3D point index in the input
   reconstruction is the resolved point.
4. **Use the resolved point's position** — look up the 3D position from the input reconstruction's
   point array.

If multiple observations match (the source point's track overlaps the input reconstruction on
several images), they should all map to the same 3D point in the input reconstruction (since
a feature can only belong to one track). If they disagree, this indicates a data inconsistency —
warn and use the most common point index.

### Failure modes

- **Source `.sfmr` not found**: Error if `sfmr` is not provided and no file in the workspace
  matches the hash prefix. The error message should suggest adding the `sfmr` field.
- **No matching observations**: If none of the source point's `(image_name, feature_index)` pairs
  appear in the input reconstruction's tracks, the point was not reconstructed in the input file.
  Error with a message listing which images were checked.
- **Image exists but feature not in any track**: The feature was extracted but not matched/
  triangulated in the input reconstruction. Same error as above.

### Diagnostics

When cross-reconstruction resolution is used, print a summary:

```
Resolving Point IDs from sfmr/calib_x04.sfmr -> input.sfmr
  pt3d_a1b2c3d4_12345 -> pt3d_e5f6g7h8_8901 (via image_003.jpg feat #847)
  pt3d_a1b2c3d4_67890 -> pt3d_e5f6g7h8_3456 (via image_012.jpg feat #1247)
  pt3d_a1b2c3d4_11111 -> pt3d_e5f6g7h8_7722 (via image_015.jpg feat #602)
  pt3d_a1b2c3d4_22222 -> pt3d_e5f6g7h8_9103 (via image_023.jpg feat #2031)
```

This lets the user verify that the right points were found and through which image the match
was made.

### Multiple source reconstructions

Point IDs in a measurements file may reference different source reconstructions. This is useful
when measurements span multiple solves — e.g., one pair of points was picked from an early solve
and another pair from a later re-solve. However, within a single measurement, `point_a` and
`point_b` must share the same hash prefix (they must come from the same reconstruction), since
the reconstruction distance between them is only meaningful within a single coordinate system.

**Validation:** After parsing all Point IDs, group measurements by hash prefix. For each
measurement, verify that `point_a` and `point_b` have the same prefix. If they differ, error
immediately — this is always a mistake.

### Efficiency

Source `.sfmr` files can be tens of MB compressed, so the resolution process should batch
by source file: group measurements by hash prefix, load each source `.sfmr` once, resolve all
Point IDs that reference it, then unload it before loading the next. Measurements whose hash
prefix matches the input reconstruction need no source file load at all. An `image_name -> image_index` map and an
`(image_index, feature_index) -> point_index` index should be built once from the input
reconstruction to make observation lookups fast.

## CLI Integration

### Usage

```bash
pixi run sfm xform input.sfmr output.sfmr --scale-by-measurements measurements.yaml
```

### Composability

The option produces a transform that participates in the standard `sfm xform` pipeline. It can
be combined with other transforms in the usual order-dependent way:

```bash
# Filter, optimize, then scale to physical units
pixi run sfm xform input.sfmr output.sfmr \
  --remove-short-tracks 2 \
  --bundle-adjust \
  --scale-by-measurements measurements.yaml
```

The transform is applied at its position in the argument sequence, like `--scale`. Transforms
that appear after it (e.g., `--align-to-input`) operate on the already-scaled reconstruction.

## World-Space Unit in .sfmr Metadata

When `--scale-by-measurements` is applied, the target unit is recorded in the output `.sfmr`
file's top-level metadata as `world_space_unit`. See the
[sfmr file format spec](sfmr-file-format.md#world-space-unit) for the field definition.

This allows downstream tools and the GUI to display coordinates with appropriate units.
