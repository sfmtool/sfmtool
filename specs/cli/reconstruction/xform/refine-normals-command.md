# `sfm xform --refine-normals`

`sfm xform --refine-normals` turns the patch of each finite point of an
embedded-patches reconstruction to its grown-plane normal: one plane fitted,
view by view, to the photographs the point is seen in. It is the viewer's
*Refine All Normals* run from the command line, through
`sfmtool.bench.refine_normals` ([refine-normals.md](../../../core/bench/refine-normals.md)),
with that function's default settings. It moves no point, no camera and no
keypoint, removes no point and no observation, and reads the workspace's source
images.

## Command syntax

```
sfm xform <input.sfmr> [<output.sfmr>] --refine-normals [...]
```

The operation is `RefineNormalsTransform` in
[_refine_normals.py](../../../../src/sfmtool/xform/_refine_normals.py), parsed
by `parse_refine_normals_params` in
[_arg_parser.py](../../../../src/sfmtool/xform/_arg_parser.py) and declared as
the `--refine-normals` Click option in
[xform.py](../../../../src/sfmtool/_commands/xform.py). The refinement is
`bench::refine_normals` in
[bench/refine_normals.rs](../../../../crates/sfmtool-core/src/bench/refine_normals.rs),
reached through its binding in
[sfmtool-py bench/refine_normals.rs](../../../../crates/sfmtool-py/src/bench/refine_normals.rs).

**It takes no keys.** The grown plane's settings are those of Track View's
*Grown Plane Normal*, and none is exposed here. The option is declared with an
optional value (`is_flag=False, flag_value=""`), as the other `xform` options
with `key=value` strings are, so a value given to it reaches the parser rather
than being read as the output path: any token other than blank comma segments
is a `click.UsageError` saying that the option takes no keys, that it turns
each patch to its grown-plane normal, that the keys of the photometric search
it ran before were removed, and that `--add-patch-bitmaps` adds bitmaps. So
`--refine-normals bitmaps=false` and `--refine-normals angular_range_deg=25`
are refused before the reconstruction is loaded. A bare `--refine-normals`
followed by another option leaves that option alone: `--refine-normals
--bundle-adjust` refines and then bundle-adjusts.

## What it requires

- **An `embedded_patches` reconstruction.** `RefineNormalsTransform` sets
  `required_feature_source = "embedded_patches"`, so `xform/_apply.py` checks
  the reconstruction as it stands at that step and refuses `sift_files` with a
  `click.UsageError` naming `sfm xform --to-embedded-patches` (the shared
  `require_embedded_patches` in `_feature_source.py`). A
  `--to-embedded-patches --refine-normals` chain converts first and passes.
  The grown plane reads each observation's stored keypoint and the point's
  stored patch frame, and only an embedded-patches reconstruction stores them.
- **Patch frames.** A reconstruction with no patch frames is refused with a
  `ValueError` naming `--to-embedded-patches` before any image is read.
- **The source images**, one per image, read by `load_workspace_images`
  (`workspace_dir / image_name`, decoded by `read_image_rgb`,
  [reading-and-writing-images.md](../../../core/camera/reading-and-writing-images.md)).
  A missing image or an unresolvable workspace is a `FileNotFoundError`.

Any refusal `sfmtool.bench.refine_normals` makes of the run as a whole (no
posed image, for one) is raised as a `ValueError` whose message starts with
`--refine-normals:` and gives the refusal's sentence; `sfm xform` reports it as
an error and writes nothing.

## What it writes

Every point keeps its index, position, constraint and keypoints, and every
track keeps its images, so the point and observation counts are the input's.
For each finite point the run does what the bench does when the point is put on
it with every observation `in`, *Grown Plane Normal* is pressed and *Commit* is
pressed ([refine-normals.md](../../../core/bench/refine-normals.md) § "What one
point's refinement is"). A point the step turns, whose record differs from the
stored one, is rewritten:

- its normal and patch frame, turned about its centre;
- its bitmap, rendered again at the turned frame from its reference
  observation, **only where the reconstruction stores bitmaps**. The run adds
  no bitmap column, so a reconstruction without bitmaps comes out without
  them; `--add-patch-bitmaps` renders them at the stored frames;
- its reference observation, where the reconstruction has that column: the
  stored one, or the reference-view rule's pick for a point stored at `-1`;
- its colour, the colour at the centre of the bitmap rendered for it;
- the readings and the blur-matched score of each of its observations, read
  again on the turned patch. A reconstruction with no readings column gains
  one, in which only the rewritten points' observations are measured.

Every other point keeps every column it had: a point at infinity, which is
skipped; a point no view joined, or whose turn would be large on weak
evidence; a point the step refused (fewer than two observations, for one); and
a turned point whose record agrees with the stored one. Such a point's bitmap
and readings are the ones it came with, and they describe its stored normal.

The normals the run writes survive the save: `.sfmr` write keeps every stored
normal and computes only the missing ones, the zero vector
(`merge_preserving_normals` in
[write.rs](../../../../crates/sfmtool-sfmr-format/src/write.rs)).

## Console output

Before the run it prints the point and image counts, and after it one line
built from the report:

```
  Refining the normals of 280 points across 17 images...
  Turned 258 of 266 normals (median turn 17.0°, mean agreement 0.78; 168 low-confidence); left unchanged: 2 no view joined, 6 large turn on weak evidence, 0 refused or with no bitmap; 14 at infinity skipped
```

`Turned N of M` is the report's `turned` over `read` (the finite points). The
median turn, the mean agreement (the mean robust grid ZNCC of the views in each
turned point's plane, `n/a` where no turned point has one) and the
low-confidence count are printed only when a point was turned; the
at-infinity count only when it is not zero. The figures above are the seoul_bull
ground truth's. The run reports no progress while it runs; it takes about 0.4 s
on that file and 2 s on the kerry_park ground truth on a 32-thread machine
([refine-normals.md](../../../core/bench/refine-normals.md) § "Run time").

## Ordering and interactions

`--refine-normals` runs at its place in the command line, like every `xform`
operation.

- **After `--bundle-adjust`** it turns the patches against the adjusted poses
  and points; bundle adjustment does not turn them afterward. Both orders are
  allowed.
- **A global similarity** (`--rotate`, `--translate`, `--scale`) moves points
  and poses together, so the photographs agree on the same planes and the order
  does not change the result.
- **Repeating it** is not idempotent: a second `--refine-normals` starts each
  point from the normal the first wrote. It starts from the same keypoints,
  since neither run moves one.
- **The keypoints are trusted.** The grown plane reads the stored keypoints as
  they are, so on a reconstruction whose keypoints may be off,
  `--refine-keypoints` or `--localize-keypoints` comes first.
- **Bitmaps.** To refine and end with bitmaps on a reconstruction that has
  none, follow it with `--add-patch-bitmaps`, or put `--add-patch-bitmaps`
  before it so that the turned points' bitmaps are rendered again by the run.

## Testing

[`tests/xform/test_refine_normals.py`](../../../../tests/xform/test_refine_normals.py)
checks that any key is a usage error naming the change, and that it is refused
on the command line; the summary line; on the seoul_bull ground truth, that the
finite points' normals turn while no position, keypoint or track moves, and
that the points at infinity keep their normals and colours; on an
embedded-patches conversion of the seoul_bull workspace, that a reconstruction
without bitmaps gets none and one with bitmaps has the rewritten points'
bitmaps rendered again and the others kept; a missing image; the
`--to-embedded-patches --refine-normals` chain; a bare option before another;
and the refusals of a `sift_files` input and of the wrong chain order.
[`tests/xform/test_observation_readings_writers.py`](../../../../tests/xform/test_observation_readings_writers.py)
checks that only the rewritten points' observations are measured, and
[`tests/xform/test_stored_references.py`](../../../../tests/xform/test_stored_references.py)
that a stored reference observation is kept and that dropping and adding the
bitmaps gives the rewritten points the same bitmaps.

## Non-goals

- **Settings.** The grown plane's settings are not exposed; a run with other
  settings calls `sfmtool.bench.refine_normals` from Python.
- **A subset of points.** The run refines every finite point; the binding's
  `point_indexes` is not exposed.
- **Moving a keypoint or a point.** Relocalizing the keypoints against the
  turned patches is `--refine-keypoints`, or the viewer's *Refit All Points*.
