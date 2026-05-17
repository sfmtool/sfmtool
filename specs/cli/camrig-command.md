# `sfm camrig` Command

## Overview

Builds and inspects `.camrig` camera rig files. `camrig` is a command group
with two subcommands:

- `spherical-tiles` — build a spherical tile rig (a sphere discretised into
  co-centric pinhole tiles) and write it to a `.camrig` file.
- `inspect` — verify a `.camrig` file's integrity and print its metadata.

See `specs/formats/camrig-file-format.md` for the file format and
`specs/core/spherical-tiles-rig.md` for the spherical tile rig.

## `sfm camrig spherical-tiles`

### Syntax

```bash
sfm camrig spherical-tiles <OUTPUT_FILE> --n <N> (--equirect-width <W> | --arc-per-pixel <R>) [OPTIONS...]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `OUTPUT_FILE` | path | required | Path of the `.camrig` file to write |
| `--n` | int | required | Number of tiles in the rig (`>= 2`) |
| `--equirect-width` | int | — | Target equirectangular width in pixels; sets the per-tile pixel size to `2π / width` |
| `--arc-per-pixel` | float | — | Angular size of one tile pixel, in radians |
| `--overlap-factor` | float | 1.15 | Tile FOV safety margin over the worst-case gap between tiles |
| `--centre` | 3 floats | `0 0 0` | Rig optical centre in world space |
| `--atlas-cols` | int | `ceil(sqrt(n))` | Atlas column count |
| `--seed` | int | unseeded | Relaxer seed for reproducible tile placement |
| `--name` | str | output file stem | Rig name stored in the file |

Exactly one of `--equirect-width` and `--arc-per-pixel` must be given — they
are two ways of expressing the same per-tile angular resolution.

### Behaviour

The tile look directions come from a Thomson-relaxed sphere-point set; the
half-FOV, patch size, and atlas packing are derived from the measured tile
spacing (see `specs/core/spherical-tiles-rig.md`). Pass `--seed` for a
bit-for-bit reproducible layout. The written `.camrig` file is a
`spherical_tiles` rig: one shared pinhole camera, all-zero translations, one
`sensor_from_rig` quaternion per tile, and the derived scalars in
`rig_attributes`.

## `sfm camrig inspect`

### Syntax

```bash
sfm camrig inspect <CAMRIG_FILE>
```

Recomputes the file's content hashes, checks its structural constraints, and
prints the rig metadata: format version, name, rig type, sensor and camera
counts, `rig_attributes`, and the content hash. Exits with an error if
verification fails. Works for any `.camrig` file, not only spherical tile
rigs.

## Usage Examples

```bash
# Build a 1280-tile rig sized for a 1024-wide equirectangular target
sfm camrig spherical-tiles tiles.camrig --n 1280 --equirect-width 1024

# Build a reproducible rig with an explicit per-tile pixel size and centre
sfm camrig spherical-tiles tiles.camrig --n 320 --arc-per-pixel 0.012 \
    --overlap-factor 1.2 --centre 0 1.5 0 --seed 42

# Verify and inspect a .camrig file
sfm camrig inspect tiles.camrig
```
