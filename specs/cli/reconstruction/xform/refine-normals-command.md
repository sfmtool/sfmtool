# `sfm xform --refine-normals`

`sfm xform --refine-normals` turns the patch of each finite point of an
embedded-patches reconstruction to its grown-plane normal: one plane fitted,
view by view, to the photographs the point is seen in. With
`method=pairwise_flow` it turns each patch to its pairwise-flow normal instead,
the plane fitted to the optical flow between pairs of the point's views
([pairwise-flow-normal.md](../../../core/bench/pairwise-flow-normal.md)). It is
the viewer's *Refine All Normals ▸ Grown Plane* or *▸ Pairwise Flow* run from
the command line, through `sfmtool.bench.refine_normals`
([refine-normals.md](../../../core/bench/refine-normals.md)), with the chosen
step's default settings. It moves no point, no camera and no keypoint, removes
no point and no observation, and reads the workspace's source images.

## Command syntax

```
sfm xform <input.sfmr> [<output.sfmr>] --refine-normals [method=grown_plane|pairwise_flow] [...]
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

**Its one key is `method`**, the binding's `method`, with the values the
wire's `refine_all_normals` takes: `grown_plane`, the default, or
`pairwise_flow`. It is a `key=value` token, as `--to-embedded-patches` chooses
its starting normal with `normal=`, so the option's value has the form of every
other `xform` option that takes `key=value` strings. The steps' other settings
are those of Track View's *Grown Plane Normal* and *Pairwise Flow Normal*, and
none is exposed here.

The pairwise-flow normal costs far more: seconds of one thread per point
against tens of milliseconds for the grown plane
([refine-normals.md](../../../core/bench/refine-normals.md) § "Choosing the
step"), so on the seoul_bull ground truth `method=pairwise_flow` takes about a
minute where the grown plane takes under a second (§ "Console output").

The option is declared with an optional value (`is_flag=False,
flag_value=""`), as the other `xform` options with `key=value` strings are, so
a value given to it reaches the parser rather than being read as the output
path. `parse_refine_normals_params` refuses each of these with a
`click.UsageError`, before the reconstruction is loaded:

- any key other than `method`, saying that the option takes only the key
  `method`, that it turns each patch to its grown-plane normal or with
  `method=pairwise_flow` to its pairwise-flow normal, that the keys of the
  photometric search it ran before were removed, and that
  `--add-patch-bitmaps` adds bitmaps. So `--refine-normals bitmaps=false` and
  `--refine-normals angular_range_deg=25` are refused;
- a `method` other than `grown_plane` or `pairwise_flow`, the hyphenated
  `pairwise-flow` among them, naming the two values. The parser and
  `RefineNormalsTransform` take the names from the binding's
  `REFINE_NORMALS_METHODS`
  ([refine-normals.md](../../../core/bench/refine-normals.md) § "Python
  binding"), so the list is kept once, in Rust;
- a malformed token (`method` with no `=`) or a repeated `method`, with the
  messages every `key=value` option gives.

Blank comma segments are tolerated. A bare `--refine-normals` followed by
another option leaves that option alone: `--refine-normals --bundle-adjust`
refines and then bundle-adjusts.

## What it requires

- **An `embedded_patches` reconstruction.** `RefineNormalsTransform` sets
  `required_feature_source = "embedded_patches"`, so `xform/_apply.py` checks
  the reconstruction as it stands at that step and refuses `sift_files` with a
  `click.UsageError` naming `sfm xform --to-embedded-patches` (the shared
  `require_embedded_patches` in `_feature_source.py`). A
  `--to-embedded-patches --refine-normals` chain converts first and passes.
  Both steps read each observation's stored keypoint and the point's stored
  patch frame, and only an embedded-patches reconstruction stores them.
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
it with every observation `in`, the step's entry (*Grown Plane Normal*, or
*Pairwise Flow Normal* with `method=pairwise_flow`) is pressed and *Commit* is
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
skipped; a point whose normal the step leaves unchanged (for the grown plane,
one no view joined or whose turn would be large on weak evidence; for the
pairwise-flow normal, one whose normal is undetermined or unstable); a point
the step refused (fewer than two observations, for one); and a turned point
whose record agrees with the stored one. Such a point's bitmap
and readings are the ones it came with, and they describe its stored normal.

The normals the run writes survive the save: `.sfmr` write keeps every stored
normal and computes only the missing ones, the zero vector
(`merge_preserving_normals` in
[write.rs](../../../../crates/sfmtool-sfmr-format/src/write.rs)).

## Console output

Before the run it prints the point and image counts, and after it one line
built from the report. With the grown plane:

```
  Refining the normals of 280 points across 17 images...
  Turned 263 of 266 normals (median turn 16.4°, mean agreement 0.79; 164 low-confidence); left unchanged: 1 no view joined, 2 large turn on weak evidence, 0 refused or with no bitmap; 14 at infinity skipped
```

With `method=pairwise_flow`, the pairwise-flow normal's outcomes (the
glossary's **single pair**, **undetermined** and **unstable**) take the place
of the figures only the grown plane has (the mean agreement, the
low-confidence count, no view joined and a large turn on weak evidence):

```
  Turned 180 of 266 normals (median turn 12.9°; 10 from a single pair); left unchanged: 78 undetermined, 8 unstable, 0 refused or with no bitmap; 14 at infinity skipped
```

`Turned N of M` is the report's `turned` over `read` (the finite points). The
figures in parentheses are printed only when a point was turned: the median
turn and, for the grown plane, the mean agreement (the mean robust grid ZNCC of
the views in each turned point's plane, `n/a` where no turned point has one)
and the low-confidence count, or, for the pairwise-flow normal, how many of the
turned points it read from a single pair. The refused count follows the
step's reasons for leaving a normal unchanged, and the at-infinity count is
printed only when it is not zero. The figures above are the seoul_bull ground
truth's. With the grown plane the run takes about 0.4 s on that file and 2 s
on the kerry_park ground truth on a 32-thread machine
([refine-normals.md](../../../core/bench/refine-normals.md) § "Run time");
with the pairwise-flow normal it takes about 57 s on that file on the same
machine.

## Progress and Ctrl-C

While the run goes, `_poll_progress` (in
[_progress.py](../../../../src/sfmtool/_progress.py), the poller
`sfm embed-patches` uses) reads the `ProgressCounter` the transform hands the
binding every 5 s and prints how many of the points it has counted, those at
infinity included, in that helper's words:

```
    50/280 patches (17%)
```

A run shorter than 5 s prints no such line. Ctrl-C stops the run: the binding
runs Python's signal handlers every 50 ms while it waits for the points
([refine-normals.md](../../../core/bench/refine-normals.md) § "Python
binding"), the `KeyboardInterrupt` cancels the run within a part of one
point's step per worker, and `sfm xform` prints `Aborted!`, exits with status
1 and writes nothing. On the seoul_bull ground truth with the pairwise-flow
normal, a SIGINT raised 12 s into the run ended the process 0.2 s later.

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
- **The keypoints are trusted.** Both steps read the stored keypoints as
  they are, so on a reconstruction whose keypoints may be off,
  `--refine-keypoints` or `--localize-keypoints` comes first.
- **Bitmaps.** To refine and end with bitmaps on a reconstruction that has
  none, follow it with `--add-patch-bitmaps`, or put `--add-patch-bitmaps`
  before it so that the turned points' bitmaps are rendered again by the run.

## Testing

[`tests/xform/test_refine_normals.py`](../../../../tests/xform/test_refine_normals.py)
checks the option's value: bare is the grown plane, `method=` chooses either
step, written separated or joined on the command line, and any other key, an
unknown or hyphenated method and a malformed `method` are usage errors, refused
on the command line before the reconstruction is loaded; the summary lines of
both steps; on five points of the seoul_bull ground truth, one of them at
infinity, that the pairwise-flow normal turns finite normals to normals other
than the grown plane's, moves no point, skips the point at infinity and prints
its own summary; that the transform's progress counter reaches every point;
that a SIGINT raised once a pairwise-flow run over the ground truth has
counted a point aborts the command with nothing written; on the seoul_bull
ground truth, that the
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

- **Settings.** The steps' settings, other than which step runs, are not
  exposed; a run with other settings calls `sfmtool.bench.refine_normals` from
  Python.
- **A subset of points.** The run refines every finite point; the binding's
  `point_indexes` is not exposed.
- **Moving a keypoint or a point.** Relocalizing the keypoints against the
  turned patches is `--refine-keypoints`, or the viewer's *Refit All Points*.
