# `sfm xform --scale-to-median-measurement`

A structure-from-motion solve has no real-world scale. `sfm xform
--scale-to-median-measurement UNIT` gives a reconstruction a physical size from
the distance measurements it stores, each a real length stated between two
places in its photographs: every measurement implies a scale factor, its stated
length in `UNIT` over the distance its two ends lie apart in the
reconstruction; the step multiplies every coordinate by the median of those
factors and declares `UNIT` (`mm`, `cm`, `m`, `in` or `ft`) as the
reconstruction's world-space unit. It prints each measurement's factor, the
measurements that give none and why, and how well the rest agree, so a
misplaced endpoint shows before the scaled file is used. It is SfM Explorer's
`Scale to Median Measurement...` run from the command line
([gui/edits/scale-to-median-measurement.md](../../../gui/edits/scale-to-median-measurement.md)),
through the same core operation
([core/reconstruction/measurement-scale.md](../../../core/reconstruction/measurement-scale.md)).

## Command syntax

```
sfm xform <input.sfmr> [<output.sfmr>] --scale-to-median-measurement UNIT [...]
```

The step is `ScaleToMedianMeasurementTransform` in
[_scale_to_median_measurement.py](../../../../src/sfmtool/xform/_scale_to_median_measurement.py),
registered in `_TRANSFORM_OPTIONS` in
[_arg_parser.py](../../../../src/sfmtool/xform/_arg_parser.py) and declared as
the `--scale-to-median-measurement` Click option in
[xform.py](../../../../src/sfmtool/_commands/xform.py). The factor and its
statistics are `SfmrReconstruction.distance_measurement_scale(unit)`
([sfmtool-py reconstruction/measurement_scale.rs](../../../../crates/sfmtool-py/src/reconstruction/measurement_scale.rs)),
which resolves the stored measurements' endpoints with the default options and
calls core's `distance_measurement_scale`
([measurement_scale.rs](../../../../crates/sfmtool-core/src/reconstruction/measurement_scale.rs)).
The Python step parses the unit, calls that binding, prints its result,
applies `Se3Transform(scale=factor) @ recon` and sets `world_space_unit` with
`clone_with_changes`; it computes no factor or statistic of its own.

`UNIT` is one of `mm`, `cm`, `m`, `in` and `ft`, the names of
`sfmtool.fileio.LENGTH_UNIT_NAMES`. Any other value is a usage error before the
reconstruction is loaded:

```
Error: Invalid --scale-to-median-measurement parameter 'furlong': expected a unit, one of mm, cm, m, in or ft
```

```bash
# Give a reconstruction millimetres from its stored measurements
sfm xform in.sfmr out.sfmr --scale-to-median-measurement mm

# Filter, optimize, then scale to physical units
sfm xform in.sfmr out.sfmr \
  --remove-short-tracks 2 \
  --bundle-adjust \
  --scale-to-median-measurement mm
```

The measurements are the `.sfmr` file's own `measurements/distance/` section
([sfmr-file-format.md](../../../formats/sfmr-file-format.md#distance-measurements-measurementsdistance)),
which SfM Explorer's Measurement View writes
([gui/measurement-view.md](../../../gui/measurement-view.md)). The step reads
nothing else: no side file and no Point ID.

## Where it sits in the pipeline

It is an ordinary step: it runs at its position on the command line, on the
reconstruction as the steps before it left it, and the steps after it work on
the scaled result, in `UNIT`. Each measurement's endpoints are resolved against
the reconstruction as it stands at that step
([measurement-endpoint.md](../../../core/reconstruction/measurement-endpoint.md)),
so an earlier step that drops points, observations or images
(`--remove-short-tracks`, `--include-range`, `--filter-by-reprojection-error`,
…) can change which endpoints resolve and where, and so which measurements give
a factor. An image filter (`--include-range`, `--exclude-range`,
`--include-glob`, `--exclude-glob`, `--include-by-distribution`) also removes
every stored measurement with an endpoint on an image it drops, so that
measurement is neither used nor listed as left out, and a filter that removes
them all leaves a reconstruction that stores none (§ "Errors"). An earlier
`--scale`, or an `--align-to` whose similarity scales, changes the scene
distances and not the stated lengths, so the factor this step computes absorbs
it, and a rotation or translation changes neither: `--scale 0.5 --scale-to-median-measurement cm` on a file whose 1 m edge is
1 scene unit long scales by 200, and the result is in centimetres as it would
be without the `--scale`.

The option may be repeated. Each occurrence is its own step and computes its
own median from the stored measurements against the reconstruction as it then
stands; a second occurrence in the same unit right after the first scales by 1.

## What it changes

The scale is `apply_se3_transform` with the identity rotation, no translation
and the median as its scale, the path `--scale` takes
([xform-command.md](xform-command.md#rust-primitives-behind-the-operations)):
finite points, camera centres, rig sensor translations, patch half-vectors and
the per-image depth statistics scale, and points at infinity, which are
directions, do not. `world_space_unit` is then set to `UNIT`, replacing
whatever unit the file declared; the factor is computed from the file's own
coordinates, so it does not depend on that unit
([measurement-scale.md](../../../core/reconstruction/measurement-scale.md#rust-api)
§ "Why it reads no `world_space_unit`"). A measurement names images and
pixels, which a scale does not move, so the stored measurements pass through
unchanged, and in the output each one reads its factor times its old scene
distance. The step's entry in the saved `tool_options` `transforms` list is
`Scale to median measurement: 1000x to mm`.

## Output

The step prints, after the pipeline's `[i/n] Scale to median measurement
(mm)` line, as here on the demo scene with its cube's edge stated three times,
once 20% long, a fourth measurement on a pixel no observation holds, and a
fifth that states no distance:

```
Scale to median measurement (unit: mm):
  Used 3 of 5 distance measurements:
    Distance #0 "cube edge": stated 1000 mm, scene 1 su, factor 1000, deviation +0.00%
    Distance #1 "cube edge again": stated 1000 mm, scene 1 su, factor 1000, deviation +0.00%
    Distance #2 "too long": stated 1200 mm, scene 1 su, factor 1200, deviation +20.00%
  Left out:
    Distance #3 "corner": endpoint A has no position: no observation's square in image 2 holds pixel (0.5, 0.5); use Create Track Here at the endpoint's pixel to build a track observed there
    Distance #4 "ruler": it states no distance, so it reads a length but gives no scale
  median: 1000, mean: 1067, std: 115.5 (11.55% of the median), median absolute deviation: 0.00% of the median
  min: 1000, max: 1200, largest deviation: 20.00%

  WARNING: 1 of 3 measurements deviate more than 5% from the median (Distance #2 "too long"). Check their endpoints, or the reconstruction for distortion, before trusting the scale.

Scaling reconstruction by 1000
```

- **Each used measurement**, in stored order: its title, `Distance #` and its
  stored index, as the viewer's Scene tree and dialog title it; its label in
  quotes; its stated length converted to `UNIT`; its scene
  distance (`su`, scene units: the file's own coordinates); its factor; and its
  signed deviation from the median, as a percentage of the median.
- **Left out**, where any measurement gives no factor: its title and core's
  reason as a sentence, one reason per measurement in core's order
  ([core/reconstruction/measurement-scale.md](../../../core/reconstruction/measurement-scale.md)).
  A measurement that states no distance (a ruler: it reads a length but gives
  no scale), one with an endpoint not placed, an endpoint the resolver
  refuses, an endpoint on a point at infinity (a direction, with no distance
  to it), or endpoints whose distance is zero or not finite is left out with
  its reason, not an error; the step goes on with the rest, as the viewer's
  dialog does.
- **The statistics**: the median, which is the factor applied; the mean; the
  sample standard deviation and its percentage **of the median**, or *none for
  a single measurement*; the median absolute deviation as a percentage of the
  median; the smallest and largest factor; and the largest deviation from the
  median. Every one is a field of core's `ScaleFactorStatistics`, the median
  and the median absolute deviation from `sfmtool_core::numeric`; the step
  only divides the median absolute deviation by the median to print it as a
  percentage, as the viewer's dialog does. The percentages are
  of the median rather than of the mean, as in core and the viewer, because
  the median is the factor applied and the mean is reported only for the
  spread.
- **The warning**, where the largest deviation exceeds core's
  `SCALE_DISAGREEMENT_PERCENT` (5%, the binding's `disagreement_percent`):
  how many measurements are past it, and their titles. Two measurements of one
  rigid object that disagree by that much point at a misplaced endpoint or a
  locally distorted reconstruction.
- **The factor applied**, `Scaling reconstruction by 1000`.

Numbers are printed to four significant digits and percentages to two
decimals.

There is no histogram. Core leaves binning to whoever draws the factors
([measurement-scale.md](../../../core/reconstruction/measurement-scale.md#rust-api)
§ "Why no histogram"), and a text histogram here would either repeat the
viewer's binning, which centres its bins on the median and picks their width
from a 1/2/5 ladder, in Python, or bin differently from it. The per-measurement
deviations and the largest deviation already show the spread line by line, at
the few measurements a reconstruction stores.

## Errors

When no measurement gives a factor, the step fails with core's sentence, which
names each measurement and why, as a Click error, and nothing is written:

```
Error: --scale-to-median-measurement mm: none of the 1 distance measurements gives a scale; measurement 0: endpoint A has no position: …
```

A reconstruction that stores no distance measurement fails the same way:

```
Error: --scale-to-median-measurement mm: the reconstruction stores no distance measurement, so no stated length gives it a scale
```

The binding raises each as a `ValueError`; the step prefixes the option and
its unit, and the command reports it as every failed step is reported, before
the save.

## Testing

In [test_scale_to_median_measurement.py](../../../../tests/xform/test_scale_to_median_measurement.py),
through the CLI on a copy of the demo scene's workspace, whose one measurement
is the cube's 1 m edge in a file in metres, restated where a test needs other
measurements:

- Scaling to `mm` declares `mm`, puts the edge 1000 apart and every finite
  point at 1000 times its position, and prints the measurement's line, *none
  for a single measurement* and the factor, with no warning and nothing left
  out.
- The edge stated as 100 cm and as 3.2808 ft gives 1000 mm for each, so a unit
  misread in either would move the median of the two.
- A measurement whose endpoints no observation holds is printed under *Left
  out* with its label and the resolver's reason, and the other one sets the
  scale.
- A measurement that states no distance is printed under *Left out* with its
  label and core's sentence for it.
- Three statements of the edge, one 20% long, print its +20.00% deviation, a
  standard deviation of 11.55% of the median and the warning naming it, and
  scale by the other two's factor.
- No usable measurement, and no stored measurement, each fail with core's
  sentence and write nothing.
- `--scale 0.5 --scale-to-median-measurement cm --translate 0,0,5` scales by
  200, records the step in `tool_options`, and leaves every finite point at
  100 times its position plus the translation.
- `--scale-by-measurements` is an unknown option, which Click answers by
  suggesting this one.

[test_arg_parser.py](../../../../tests/xform/test_arg_parser.py) checks that
the option requires a value and that an unknown unit is a usage error in the
words above.
