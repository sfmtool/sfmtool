# `sfm pick-images`

**Status:** Draft

Proposes a new command. Decided with the maintainer: the command's name and
its one-command shape (it reports, and writes a list when given a name), that a
written list records nothing about how it was picked, the refusal to overwrite
without `--force`, one list per run across every sequence, the workspace and
`.sift` version 2 requirement, SIFT matching through a KD-forest as the measure
of motion, the detection of a sequence of photographs, which stops the
command, the defaults of 10% and 0.6, and the algorithm in
[The algorithm](#the-algorithm). The smaller choices this draft makes on its
own are listed under [Open questions](#open-questions), and the default values
other than 10% and 0.6 are starting values to be set as
[Settling the defaults](#settling-the-defaults) describes.

## Purpose

A video of a walk around a scene has many more frames than a
Structure-from-Motion solve should use. Neighbouring frames are nearly the same
picture, some frames are blurred, and some stretches of a capture cannot be
solved at all: the camera only turned on the spot, the scene itself moved, or
the camera hardly moved. `sfm pick-images` looks at a capture before it is
solved and chooses the frames a solve should use: frames spaced so that the
image moves by about a tenth of its width from one to the next, sharp ones in
place of blurred ones, more frames where the capture is hard to follow and
fewer where it is easy. It prints what it found, including a verdict on whether
the capture can be solved, and when given a name it writes the frames it chose
as a named image list ([workspace/named-image-lists.md](../workspace/named-image-lists.md)),
which `sfm solve -r NAME` and the other commands that take `-r` then use.

It measures everything from SIFT features. Each frame it reads gets a `.sift`
file, extracted with the workspace's own settings (its grey conversion
`image_to_gray` included) and stored where
`sfm sift --extract` would store it, and the frame's sharpness and gradient
structure tensor come from that file. The motion between two frames is the
median displacement of the SIFT features that match between them. The command
reads only the frames it needs: a sample spread over the whole capture, the
frames it chooses, and the frames it tries on the way.

It chooses each step by measuring it rather than by a stride fixed in advance.

## Command

```bash
sfm pick-images [PATHS...] [OPTIONS...]
```

`PATHS` are image folders or patterns, as for [`sfm sift`](../cli/image-feature/sift-command.md):
directories are expanded recursively and keep the files ending in `.png`,
`.jpg` or `.jpeg` (in any letter case). When omitted, the current directory is
used. The command needs a workspace, found by the upward search from the
images' common parent ([workspace.md § Workspace Discovery](../workspace/workspace.md#workspace-discovery)),
and is an error without one, asking for `sfm ws init`. There is no `--tool`
option: the feature tool and its settings are always the workspace's.

| Option | Default | Meaning |
|--------|---------|---------|
| `--write-list NAME` | | Write the chosen frames as the named image list `NAME`. Without it the command only reports. |
| `--force` | off | Replace a list of that name if the workspace already defines one. |
| `-r/--range` | | A range of file numbers, or the name of a named image list, that narrows the input ([the shared `-r/--range` option](../cli/README.md#the-shared--r--range-option)). |
| `--named-image-list NAME` | | Read the input frames from this named image list, in place of `PATHS`. |
| `--target-step PERCENT` | `10` | The step to aim for between two chosen frames: the median feature displacement, in percent of the image's long side. |
| `--step-band FRACTION` | `0.3` | How far a step may be from the target and still be accepted: within `target × (1 ± FRACTION)`, 7% to 13% at the defaults. |
| `--blur-ratio RATIO` | `0.6` | A frame whose sharpness is below `RATIO` times the capture's median sharpness counts as blurred. |
| `--baseline-frames N` | `200` | How many frames the baseline sample takes, spread evenly over the capture. |
| `--min-inlier-share SHARE` | `0.5` | A step whose matches fit a fundamental matrix with a smaller share of inliers fails. |
| `--json PATH` | | Also write every measurement and decision to a JSON file. |

**Why `--write-list` and not a positional name or `--name`.** On the eight
commands that take it, `--named-image-list NAME` names a list to **read**, and
this command takes that option too, for its input. A bare `--name` beside it
would leave a reader to guess which of the two names is the output. A
positional name would be confused with `PATHS`. `--write-list` says what it
does, and its absence says that the run only reports, which is decision 1: the
same command reports and writes, and a list is written exactly when a name is
given.

**`-r` narrows the input.** The command joins the shared `-r/--range` option as
a ninth command on image directories, beside `sift`, `match`, `motion` and
`solve`: a range keeps the files whose number `number_from_filename` reads is
in it, and a list name replaces `PATHS` with the list's image members. So
`sfm pick-images -r walk1 --write-list walk1_picked` chooses frames from among
an existing list's. The output name may be the input list's own name only with
`--force`; the input is resolved before anything is written.

**Refusing to overwrite.** Before it reads any image, the command reads the
workspace's lists file (`NamedImageLists::read`) and stops with an error when
it already defines `NAME`, naming the file and the list and saying to pass
`--force` to replace it. The same check is made again when the file is written, so that a
list another command added in the meantime is not lost silently.

**No provenance.** A written list holds pattern and range entries only, in the
lists file's version 1 layout, unchanged. Nothing about the run (the options,
the verdict, the readings) goes into the lists file, the `.sift` files or the
workspace file. The printed report and `--json` are the only record of why a
frame was chosen.

**One list per run.** The list covers every frame chosen in the run, across
every folder and sequence it was given, one entry per sequence
([Writing the list](#writing-the-list)).

**Exit status.** The command exits 0 when it finishes, whatever it found, as
`sfm motion` does. When the verdict says a sequence is not a video or the
capture cannot be solved it writes no list, and with `--write-list` it then
exits 1, since the list it was asked for does not exist.

### Examples

```bash
# Report on a capture without writing anything
sfm pick-images frames/

# Choose the frames and write them as the list walk1
sfm pick-images frames/ --write-list walk1
sfm solve -i -r 'walk1[:30]'    # a quick test solve of its first 30 images
sfm solve -i -r walk1

# Larger steps, and only frames 1-3000
sfm pick-images frames/ -r 1-3000 --target-step 15 --write-list walk1_fast

# Several sequences into one list, replacing an earlier one
sfm pick-images walk1/ walk2/ walk3/ --write-list all_walks --force
```

## The algorithm

The command works in six stages: collect the frames, read a baseline sample,
judge whether each sequence is a video, judge whether the capture can be
solved, walk the capture choosing frames, and report. Every stage that reads
frames reads them through the same source, which extracts a frame's `.sift`
files when they are missing and reuses them otherwise.

### Words

- A **frame** is one instant of the capture: one image, or for a rig the images
  of all its sensors at that instant.
- A frame is **read** when the command loads its `.sift` file, extracting it
  first when it is missing or older than the image. A frame gets a `.sift` file
  exactly when the command reads it.
- The **step** between two frames is the median length of the displacement of
  their matched features, in percent of the long side of the image. It does not
  depend on the image's resolution.
- The **rate** is the step per frame: a step divided by the number of frames
  between the two frames, counted by their position in the sequence's order,
  not by the difference of their file numbers, so a sequence numbered in steps
  of 2 has the same rate as one numbered in steps of 1.
- A frame is **blurred** when its sharpness is below `--blur-ratio` times the
  sequence's median sharpness, or when it is blurred along one direction: its
  axis ratio or its coherence lies more than 2 robust spreads above the
  sequence's median ([Blur](#blur)).
- A step **matches well** when it has at least 30 mutual matches and its
  fundamental-matrix inlier share is at least `--min-inlier-share`.

### Collecting the frames

The command expands `PATHS`, or resolves the input list, into image files as
`sfm sift` does. It then groups them into **sequences**: the files of one
folder whose names are the same apart from the file number
`number_from_filename` reads ([`_filenames.py`](../../src/sfmtool/_filenames.py)),
the number `-r` selects a file by. `walk1/frame_0001.jpg` to
`walk1/frame_6120.jpg` are one sequence, and `walk1/a_0001.jpg` beside
`walk1/b_0001.jpg` are two. Each sequence is ordered by file number. A file
with no number, or two files of one sequence with the same number, is an error
naming the files, since `-r` cannot select them and no order exists for them.

When the workspace has a `rig_config.json` ([workspace/rig-config.md](../workspace/rig-config.md)),
the images under one rig's sensor prefixes are grouped into rig frames by
their frame key, the path after the sensor prefix, as `sfm solve` groups them
(`_infer_frame_key` in [`rig/config.py`](../../src/sfmtool/rig/config.py)).
Each rig frame is one frame of one sequence: it is read, judged and chosen with
all its sensors. [Rigs](#rigs) says how its motion and sharpness are judged.

The sequences are walked one after another, and the baseline sample is shared
among them in proportion to their frame counts.

### The baseline

Before choosing anything, the command reads a sample of `--baseline-frames`
frames spread evenly over each sequence, the first and last frames included
(every frame when the sequence is shorter). For each sampled frame `s` it also
reads the frames `s + 1`, `s + 3`, `s + 10` and `s + 30` that exist, about five
frames per sample, and reuses a frame another sample already read. It reads
them in two batches: the sampled frames with the frames 1 and 3 after them
first, which is enough to tell a video from a set of photographs
([Is it a video?](#is-it-a-video)), and the frames 10 and 30 after them only
when every sequence is judged to be a video. Each batch is requested from
the source at once, so extraction runs as one batch.

From the sampled frames' own `.sift` files, per sequence:

- the **median sharpness**, which every later frame is compared with;
- the median and robust spread of the **axis ratio** `|log(Exx/Eyy)|` and the
  **coherence** `sqrt((Exx − Eyy)² + 4·Exy²) / (Exx + Eyy)`, the values
  [sift-file-format.md § Gradient structure tensor](../formats/sift-file-format.md#gradient-structure-tensor)
  defines a reader to derive from the gradient structure tensor
  ([One derivation of the tensor's values](#one-derivation-of-the-tensors-values)).

From the matches between `s` and `s + 1`, and between `s` and `s + 3`:

- the **rate** at each sample: the step from `s` to `s + 3` divided by 3, with
  the step to `s + 1` as a check that the motion is smooth;
- the **flagged regions**: a sampled frame that is blurred, or whose matches to
  `s + 1` or `s + 3` do not match well, flags the frames from the previous
  sampled frame to the next one. The walk goes through a flagged region in
  short steps ([The walk](#the-walk)).

The baseline covers the whole capture independently of the walk, so a problem
the walk could step over in one long step is still seen.

### Is it a video?

The walk assumes that neighbouring frames are close in time, so that the
motion between them is small and changes smoothly. A folder of photographs
numbered in the order they were taken does not behave that way, and the
command judges each sequence separately from its baseline's 1- and 3-frame
matches, so a run can hold a video folder beside a folder of photographs. A
sequence is a video when at least three of these four signs hold, each the
median over its sampled frames:

- **The motion adds up.** The step from `s` to `s + 3` is about three times the
  step from `s` to `s + 1`: their ratio is between `2` and `4`.
- **The step is small.** The step from `s` to `s + 1` is below `5%` of the long
  side.
- **The step changes little.** Between neighbouring sampled frames, the 1-frame
  steps differ by less than a factor of `2`: the median of
  `|log(step_i / step_{i+1})|` is below `log 2`.
- **Neighbours match closely.** The F-inlier share of the 1-frame matches is
  above `0.8`.

A sample whose 1-frame or 3-frame pair has fewer than the minimum matches
counts against every sign it enters, since photographs that barely overlap
give that result. Photographs break the signs together: their 1-frame steps
are large and erratic, the 3-frame motion is not three times the 1-frame
motion, and neighbours barely match. A sequence of fewer than 4 frames has no
3-frame pair and is judged not to be a video.

A sequence judged to be photographs is named in the verdict, and the command
stops before the walk, extracts nothing more and writes no list, with a message
that it picks from video sequences only:

```text
Verdict: not a video: photos/DSC_%04d.JPG: steps 1 frame apart median 18.4% and erratic, 3 frames apart 1.2 times as far; sfm pick-images picks from video sequences only
```

The check comes before [Can it be solved?](#can-it-be-solved), whose gaps of
10 and 30 frames mean something only in a video. Its thresholds are starting
values ([Settling the defaults](#settling-the-defaults)).

### Can it be solved?

From the matches between each sampled frame and the frames 10 and 30 after it,
the command judges three things, each from the median over the samples:

- **A moving scene.** At a gap of 10 frames, the share of matches that fit one
  fundamental matrix (the **F-inlier share**) is below `0.3`: much of what is
  matched moves on its own, so the frames do not fit one camera motion.
- **A panorama, or a camera that only turns.** At a gap of 30 frames, the share
  of the F-inliers that also fit one homography (**H/F**) is above `0.97`, and
  that homography is within `0.01` of a pure rotation (the **rotation error**,
  below). A solve of such a capture can look excellent and still be wrong,
  since without translation there is no depth to measure.
- **A camera that barely moves.** The total motion over the sequence, the
  median rate times the frame count, is below `100%` of the long side.

The **rotation error** is defined this way: for focal lengths `f` from `0.3`
to `3` times the long side, on 80 steps spaced evenly in `log f`, with `K` the
camera matrix of focal `f` and the principal point at the image centre, take
`M = K⁻¹ H K` scaled to determinant 1, and the error is the smallest
Frobenius norm `‖M Mᵀ − I‖` over those focal lengths. It is 0 for a camera that
turned without moving.

The command prints one verdict line, in the shape of the `Verdict:` line
[`sfm motion`](../cli/reconstruction/motion-command.md) prints for a
reconstruction, as the last line of the baseline section:

```text
Verdict: solvable
Verdict: not solvable: panorama: 30 frames apart, 0.991 of the matches fit a homography and it is a pure rotation (error 0.004)
Verdict: not solvable: walk2: moving scene: only 0.21 of the matches 10 frames apart fit one camera motion
```

When one of the checks fails for any sequence, the command stops after the
verdict: it extracts nothing more, writes no list, and the `.sift` files of
the baseline stay.

### The walk

The walk replaces a fixed stride. It starts at the first frame of a sequence
that matches its next frame well, and repeats from the last chosen frame `f`:

1. **Predict.** The next candidate is `c = f + target / rate`, rounded and at
   least `f + 1`, with the rate the median of the last three accepted steps'
   rates, and at the start of a segment the rate of the nearest sampled frame
   of the baseline. Inside a flagged region the target is halved, and a step
   never crosses a flagged region without landing in it.
2. **Measure.** Read `c` and match it with `f`, which gives the measured step
   `m` and its F-inlier share; `c`'s sharpness comes from its `.sift` file.
3. **Accept.** If `m` is within the band and the step matches well and `c` is
   not blurred, accept `c`: it becomes `f`, and its step's rate joins the last
   three.
4. **Re-aim.** If `m` is outside the band, the next candidate is
   `c' = f + (c − f) · target / m`, and the walk measures again. When two
   candidates one frame apart lie on either side of the band, the motion per
   frame is wider than the band, and the walk accepts the one below the band.
   After four re-aims without reaching the band, the step fails.
5. **Swap a blurred frame.** If `c` is in the band but blurred, the walk reads
   its neighbours, nearest first (`c − 1`, `c + 1`, `c − 2`, …), and accepts
   the first that is sharp, matches `f` well and stays in the band. The swap is
   reported.

**A failing step.** A step fails when it does not match well or cannot be
brought into the band. The walk then halves the gap: it tries the frame halfway
between `f` and the failed candidate, and accepts it if it matches well, even
though its step is below the band; if that fails too it halves again, down to
`f + 1`. So the list becomes dense across a bad patch and stays sparse where
nothing goes wrong. Each run of steps accepted below the band is reported as a
gap filled in.

**A blurred run.** When no neighbour of a blurred candidate is sharp within the
band, the walk widens the search to every frame from `f + 1` to the band's
upper edge and accepts a step below the band to land on a sharp one. If none
of those is sharp, it accepts the sharpest frame among them that matches well,
and the report marks it as kept though blurred.

**A break.** When even the step from `f` to `f + 1` does not match well, the
walk records a break after `f`, skips forward to the first frame that matches
its own next frame well, and starts a new segment there. The frames skipped
are in no list. A segment of fewer than 3 chosen frames is reported as a weak
point.

**What is never read.** A frame the walk does not read lies between two chosen
frames whose step matched well, in a stretch where no sampled frame of the
baseline raised a flag. Every other frame of the capture was either read or
lies inside a break.

### Blur

The axis ratio and coherence are judged with a robust z-score against the
baseline: `z = (x − median) / (1.4826 · MAD)`, the median and the median
absolute deviation taken over the sampled frames of the sequence with
`sfmtool_core::numeric::median`. Only a z above 2 counts, since motion blur
raises both readings (it weakens the gradients along the direction of motion).
The sharpness ratio catches a frame blurred in every direction, and the
z-scores catch one blurred along one direction that the sharpness alone misses,
which is how [`sfm inspect`](../cli/reconstruction/inspect-command.md) reads
them.

### Rigs

A chosen rig frame includes the images of all its sensors, and reading a rig
frame extracts the `.sift` files of all of them. The proposal is:

- **Motion** from the reference sensor's images only (the sensor
  `rig_config.json` marks `ref_sensor`), since its displacement and its
  inlier shares describe the rig's motion as well as any sensor's.
- **Sharpness** per sensor, each against that sensor's own median: the sensors
  look at different parts of the scene, so their sharpness is not comparable.
  A rig frame is blurred when any of its sensors is.

Both are listed under [Open questions](#open-questions).

### Writing the list

Each sequence becomes one entry of the list: a pattern made from the sequence's
file names, with the file number replaced by a frame field (`%0Nd` when every
number in the sequence is written with `N` digits, `%d` otherwise, and `%%` for
a `%` in the name), and a range of the chosen frames' numbers, made by
`RangeExpr::from_values` and written with its `Display`, which joins runs into
`a-b` and `a-b:s` spans. A rig sequence becomes one entry per sensor with the
same range, so the list holds every sensor's image of each chosen frame.

```json
{
  "version": 1,
  "named_image_lists": {
    "walk1": [
      { "pattern": "frames/walk1_%04d.jpg", "range": "1-61:15,74,86-98:2,112-1462:15" }
    ]
  }
}
```

The list is written by `NamedImageLists::write_lists`, which rewrites the
lists file whole, keeps every list it already held in its place and order and
with its text, puts a replaced list in the place of the one it replaces and a
new one at the end, and leaves the old file when a write fails
([named-image-lists.md § Writing the file](../workspace/named-image-lists.md#writing-the-file)).

## Report

The report prints the verdict, the motion along the capture, the blurred frames
swapped, the gaps filled in, the breaks and the weak points, and, when it
writes a list, the number of images in it with three slices of 30 images to
try as quick test solves. An example, for a 4K video of a walk with one blurred
run:

```text
Using workspace: /captures/garden
  Feature tool: sfmtool
1 sequence: frames/walk1_%04d.jpg, 6120 frames (1-6120)

Baseline: 200 frames sampled, 984 frames read (972 extracted, 12 reused)  Sharpness: median 0.0123; 9 sampled frames blurred
  Motion per frame: median 0.72%, p90 1.31% of the long side; total 4400%
  A video: 1 frame apart step 0.72%, F-inlier share 0.93; 3 frames apart
    2.9 times as far; neighbouring samples' steps within a factor of 1.3
  10 frames apart: F-inlier share 0.84
  30 frames apart: H/F 0.41, rotation error 0.21
  Flagged regions: 2210-2271 (blurred), 4381-4442 (weak matches)
Verdict: solvable

Walk: target step 10% (band 7-13%), 1 break, 2 segments
  Chosen: 512 frames; read: 1307 frames (1250 extracted, 57 from the baseline)
  Motion along the capture (median step, rate per frame, frames per step):
    frames    1-612     9.9%   0.64%   15
    frames  613-1224   10.1%   0.71%   14
    ...
    frames 5509-6120    9.7%   0.48%   20
  Blurred frames swapped (23): 1415->1417, 1890->1889, 2224->2226, ... and 15 more
  Gaps filled in (2):
    2210-2290: 14 frames, steps 2.1-5.0%, F-inlier share down to 0.41
    4380-4410:  9 frames, steps 1.0-3.3%, F-inlier share down to 0.38
  Kept though blurred (1): 2244, the sharpest of 2236-2251 (0.31 of the median)
  Breaks (1): 4411->4416: frames 4411 to 4415 do not match their next frame
  Weak points (2): 4410->4411 (32 matches), 4416->4421 (F-inlier share 0.52)

Wrote the named image list walk1 (512 images) to /captures/garden/.sfm-named-image-lists.json
  For a quick test solve, try -r 'walk1[:30]', -r 'walk1[241:271]' or -r 'walk1[-30:]'
```

The suggested slices use the syntax of
[named-image-lists.md § Slices](../workspace/named-image-lists.md#slices):
the first 30 images, the 30 from the middle, starting at `(n − 30) // 2` for a
list of `n` images, and the last 30. A list of 60 images or fewer gets no
suggestion line, since its first 30 and last 30 images would overlap or be the
whole list. The slices count image files, not frames, so for a rig list, whose
entries this draft writes one per sensor, `[:30]` would take only the first
sensor's images ([Rigs](#rigs)); the report gives no suggestion for a rig list.

A **weak point** is an accepted step with fewer than twice the minimum match
count, or an F-inlier share within `0.1` of `--min-inlier-share`, and a segment
of fewer than 3 frames. These are where a solve is most likely to split.

`--json PATH` writes the same content, with every frame read and its readings,
every pair matched and its statistics, and every decision of the walk.

## Where the code lives

The walk's decisions live in Rust core, in the module
`sfmtool_core::analysis::pick_images`
([`pick_images.rs`](../../crates/sfmtool-core/src/analysis/pick_images.rs) beside a
`pick_images/` directory), beside the frame pair matching it already holds
([core/analysis/frame-pair-matching.md](../core/analysis/frame-pair-matching.md)),
bound in `sfmtool.analysis`, following
[STANDARDS.md § Logic lives in Rust](../STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers).
Python holds the command, in `src/sfmtool/_commands/pick_images.py` with its
row in `cli.COMMANDS` under *Image Feature*, beside `sift` and `match`. It
collects the frames, builds the source, calls the binding, prints the report
and writes the lists.

### Reused pieces

- **Extraction.** `image_files_to_sift_files` in
  [`sift/extract.py`](../../src/sfmtool/sift/extract.py), called with the
  workspace's `feature_tool`, `feature_options` and `feature_prefix_dir` from
  `load_workspace_config`, as [`_commands/sift.py`](../../src/sfmtool/_commands/sift.py)
  calls it in workspace mode. It writes each `.sift` that is missing or older
  than its image, through the same backends, the same refusals and the same
  `SiftWriteQueue` as `sfm sift --extract`, so a later `sfm sift`, `sfm match`
  or `sfm solve` finds the files and extracts nothing for them.
- **Reading.** `read_sift_metadata` and `read_sift_features` in
  [`sfmtool-sift-format`](../../crates/sfmtool-sift-format/src/read.rs): the
  metadata holds the readings (`SiftMetadata`'s `sharpness` and
  `gradient_structure_tensor_xx_yy_xy`, `None` in a version 1 file), and the
  features give positions and descriptors. Rust reads the files itself, so no
  feature array crosses into Python and back.
- **Matching and geometry.** `FrameFeatures` and `match_frame_pair`
  ([core/analysis/frame-pair-matching.md](../core/analysis/frame-pair-matching.md)):
  one forest per frame, built once and kept while the frame is in a pair still
  to be matched, mutual best matches through the ratio test, and the step, the
  F-inlier share, H/F and the rotation error of each pair, with a fixed seed so
  a run is repeatable. Its inlier bounds of 1 pixel per 960 pixels of the long
  side for the fundamental matrix and 2 per 960 for the homography are starting
  values.
- **Statistics.** Medians from
  [`sfmtool_core::numeric`](../../crates/sfmtool-core/src/numeric.rs), and the
  MAD as the median of the absolute deviations from the median, since the
  module has no MAD of its own.
- **Lists.** `NamedImageLists` in
  [`named_image_lists.rs`](../../crates/sfmtool-core/src/workspace/named_image_lists.rs)
  reads and checks the file, `is_named_image_list_name` checks the output
  name, and `NamedImageLists::write_lists` writes the list
  ([named-image-lists.md § Writing the file](../workspace/named-image-lists.md#writing-the-file)).
- **Input.** `range_option`, `parse_image_selection` and `resolve_list_images`
  in [`_range_options.py`](../../src/sfmtool/_commands/_range_options.py).
- **Progress.** `ProgressCounter` ([py_progress.rs](../../crates/sfmtool-py/src/py_progress.rs)).

### One derivation of the tensor's values

The walk takes the axis ratio and the coherence from
`ImageReadings::axis_ratio` and `ImageReadings::coherence`
([sift.md](../core/features/sift.md)), the methods whose values
`sfm inspect` prints.

### Rust interface

The walk asks a caller-supplied source to make sure frames' `.sift` files
exist, then reads them itself. It matches frames with `FrameFeatures` and
`match_frame_pair`, whose options, error and statistics (`FramePairOptions`,
`FramePairError`, `FramePairStats`) are in
[core/analysis/frame-pair-matching.md § Rust API](../core/analysis/frame-pair-matching.md#rust-api):

```rust
/// Where each image of a capture is and where its `.sift` file goes.
pub struct CaptureImage {
    pub image_path: PathBuf,
    pub sift_path: PathBuf,
}

/// One instant: one image, or one per rig sensor, the reference sensor first.
pub struct CaptureFrame {
    pub number: i64,
    pub images: Vec<CaptureImage>,
}

/// One sequence, in file-number order, and the pattern its list entry uses.
pub struct CaptureSequence {
    pub patterns: Vec<String>, // one per sensor
    pub frames: Vec<CaptureFrame>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameId {
    pub sequence: usize,
    pub frame: usize, // position in the sequence, from 0
}

/// Why a [`FrameSource`] could not make the files exist, in its own words.
pub struct FrameSourceError {
    pub message: String,
}

/// Makes sure the `.sift` files of the given frames exist and are current.
pub trait FrameSource {
    fn ensure_sift_files(
        &mut self,
        frames: &[FrameId],
        progress: &Progress,
    ) -> Result<(), FrameSourceError>;
}

pub struct PickImagesOptions {
    pub target_step_percent: f64,  // 10.0
    pub step_band: f64,            // 0.3
    pub blur_ratio: f64,           // 0.6
    pub baseline_frames: usize,    // 200
    pub min_inlier_share: f64,     // 0.5
    pub forest: KdForestParams,    // KdForestParams::balanced()
    pub pair: FramePairOptions,    // matching and fitting
}

pub fn pick_images(
    sequences: &[CaptureSequence],
    source: &mut dyn FrameSource,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<PickImagesReport, PickImagesError>;
```

`PickImagesReport` holds the verdict and the checks behind it, the baseline's
readings, the chosen frames per sequence, every step the walk
measured with its `FramePairStats`
([frame-pair-matching.md](../core/analysis/frame-pair-matching.md#rust-api)),
the swaps, the gaps filled in, the frames
kept though blurred, the breaks, the weak points and the count of frames read.
`PickImagesError` is an enum: `Options` (a value of `PickImagesOptions` out of
range, such as a target step that is not positive or a blur ratio outside
`(0, 1)`), `Sequence` (an empty sequence, or a frame whose sensor count differs
from its sequence's), `Version1Files` (a folder the capture's `.sift` files go
in that holds version 1 files, with the folder and the files; see
[The version 1 refusals](#the-version-1-refusals)), `Source` (the source's
error, carrying its message), `ReadSift` (with the path and the `SiftError`),
`Version1Sift` (with the path), `Pair` (a `FramePairError` with the two frames
it was raised for, which covers the forest parameters and two frames of one
sensor whose sizes differ) and `Cancelled`.

**Two error types.** `match_frame_pair` returns its own `FramePairError`, for
the reason
[frame-pair-matching.md § Rust API](../core/analysis/frame-pair-matching.md#rust-api)
gives, and `PickImagesError::Pair` wraps it with the frames, so a walk's error
still says where it happened.

**Why it is shaped this way.** The walk decides which frame to read next from
what it read last, so it cannot be handed all its frames in advance; asking a
source keeps the walk's logic in Rust while extraction stays where it is, in
the Python extraction path that handles all three feature tools (Rust's own
`build_sift_files` writes only the `sfmtool` tool's files). The source is asked
only to make the files exist, not to return features, because the Rust reader
already reads and checks `.sift` files and the features then never cross the
binding. It takes several frames per call because the baseline and a blur
search know several frames at once, and extraction in a batch keeps the
extractor's concurrent decodes busy. `pick_images` takes a `&Progress` and
checks it for cancellation between pairs, since it runs for minutes, and
returns an error rather than panicking on any input
([STANDARDS.md § Caller input never panics](../STANDARDS.md#caller-input-never-panics));
the checks on one frame's features and on the pair options are
`FrameFeatures::new`'s and `match_frame_pair`'s.

### Python

```python
from sfmtool.analysis import CaptureFrame, CaptureSequence, pick_images
from sfmtool.sift.extract import image_files_to_sift_files

def ensure_sift_files(frames):
    paths = [image.image_path for f in frames for image in sequences[f.sequence].frames[f.frame].images]
    image_files_to_sift_files(
        paths,
        feature_tool=config["feature_tool"],
        feature_options=config["feature_options"],
        feature_prefix_dir=config["feature_prefix_dir"],
        settings_source=f"The workspace file {workspace_file}",
    )

report = pick_images(sequences, ensure_sift_files, target_step_percent=10.0, progress=counter)
```

The binding implements `FrameSource` over a Python callable, taking the GIL for
each call, and raises the callable's exception unchanged. The callable cannot
raise the version 1 refusals of `image_files_to_sift_files` for a folder,
because `pick_images` has checked every folder before the first call
([The version 1 refusals](#the-version-1-refusals)). Every other error is
a `ValueError` with the Rust message, a `.sift` read error the `OSError` of
[python-bindings.md § Format errors](../python-bindings.md#format-errors), and
a cancelled run `InterruptedError`.

The command writes its list with `NamedImageLists.write_lists` in
`sfmtool.workspace`, passing `replace=True` for `--force`
([named-image-lists.md § Writing the file](../workspace/named-image-lists.md#writing-the-file)).
Its `ListExists` refusal is the second check of
[Refusing to overwrite](#command).

### The version 1 refusals

Every reading comes from a version 2 `.sift` file. The command checks for
version 1 once, before it reads any frame, so that the message it gives names a
fix that works in the case at hand. There are two cases, since
[versions never mix](../workspace/workspace.md#versions-never-mix):

- **A version 1 workspace.** Its `feature_options` carry no `image_to_gray`,
  the test `image_files_to_sift_files` applies to version 1 settings. The
  command refuses with the message `sfm sift --extract` gives for version 1
  settings: it names the workspace file and says to run
  [`sfm ws upgrade`](../cli/workspace/ws-upgrade-command.md), which moves the
  workspace to version 2 and a new features folder. The advice works there:
  `upgrade_workspace` rewrites a version 1 file as version 2 and names a new
  `feature_prefix_dir`, which holds no `.sift` files, so the next run extracts
  version 2 files into it. The check has to be made up front because
  `image_files_to_sift_files` refuses version 1 settings only when it has a file
  to write, and a version 1 workspace whose files are all present would
  otherwise reach the reader.
- **Version 1 files in a version 2 workspace's folder.** A folder named under
  version 2 holds version 1 files only when they were put there by hand, or
  when the workspace file was edited by hand to name a version 1 folder.
  `sfm ws upgrade` cannot help: on a version 2 workspace file it prints
  "nothing to do: already version 2" and changes nothing. `pick_images` lists
  every folder the capture's `.sift` files go in, reads the version of each
  `.sift` file already there, and returns `Version1Files` when one holds
  version 1 files, before anything is extracted. Its message names the folder
  and the version 1 files in it, and says to remove those files from the folder
  and run the command again, which then extracts version 2 files in their
  place. That works because the only check against a version 1 file in the
  folder, in `image_files_to_sift_files` (`_check_versions_never_mix`) and in
  `build_sift_files`, looks at the files the folder holds.

The folder check is `version_1_sift_files` in
[`features/sift/files.rs`](../../crates/sfmtool-core/src/features/sift/files.rs),
which `build_sift_files` and, through its binding, `_check_versions_never_mix`
call, and `pick_images` calls it too, so the three agree on what
counts. `Version1Files` wraps the `Version1SiftFiles` it returns, whose message
is the one those two writers give for a folder
([workspace.md § Versions never mix](../workspace/workspace.md#versions-never-mix)).
Once it has passed, every `.sift` file the run writes is version 2, so the
per-call check in `image_files_to_sift_files` cannot fail on a folder the
command checked. A `.sift` file the walk reads that is still version 1, one put
in the folder while the command runs, is `Version1Sift`, whose message names the
file and gives the same fix: remove it and run again.

## Why SIFT matching rather than optical flow

Optical flow and SIFT matching are two ways to measure the motion between
frames. They were compared on two 4K portrait videos, one with a blurred run,
CPU only, on frame pairs 1, 3, 10 and 30 frames apart starting at four or five
places in each video, with each frame at 960 pixels, 1920 pixels and full size
on its long side. The SIFT matches were mutual best matches through a KD-forest
with the ratio test at `0.8`, with features capped at 8192, and the step was
the median displacement of the matches; the flow's was the median length of
the flow field.

- **At 1 and 3 frames the two agree.** The SIFT step is within 6% of the
  flow's on one video and within about 30% on the other, where the flow reads
  higher. At 960 pixels the flow sampled at each matched keypoint is a median
  0.3 to 0.7 pixels from the SIFT match, and 84% to 99% of the matches lie
  within 2 pixels of it.
- **At 30 frames the flow is wrong and the matches are right.** The flow's step
  is 6% to 43% of the SIFT step: at one place 2.2% to 2.4% of the long side
  against 22% to 23%. The SIFT step grows steadily with the gap there, 1.1%,
  3.7%, 12% and 22% at gaps of 1, 3, 10 and 30. At 10 frames the flow already
  falls well short at 3 of the 9 places (24% to 64% of the SIFT step). On the
  video where the inlier share was recorded, 59% to 85% of the matches 10 and
  30 frames apart fit a fundamental matrix at 960 pixels outside the blurred
  run, so the matches describe one camera motion; on the other video, a separate
  check at 960 pixels found 70% to 98% of the mutual matches fitting a
  fundamental matrix across gaps of 1 to 30. Where SIFT itself fails,
  across the blurred run 30 frames apart, it found 30 matches and 37% of them
  fit, and the inlier share shows that the step is not to be trusted.
- **The matching costs less than the flow, given the features.** Per pair on 32
  cores, the flow (default preset) took 22 to 35 ms at 960 pixels, 62 to
  115 ms at 1920 and 197 to 280 ms at full size. Extracting SIFT took 39 to
  64 ms, 68 to 108 ms and 145 to 290 ms per frame. Building the forests of both
  frames took 2 to 16 ms and the two queries 1.3 to 6.6 ms. The flow on the GPU
  with the default preset takes as long as on the CPU. Since the command
  extracts each frame's `.sift` file anyway, for the readings and for the
  solve, matching adds about 3 to 21 ms per pair, less than the flow.
- **The matches measure the walk's own steps.** A 10% step is between about 8
  and 35 frames on these videos, depending on where in the video, which is the
  range where the flow is unreliable. SIFT matching measures each step the walk
  takes directly; the flow could only measure short gaps and extrapolate.

## Cost

Reading a frame costs one SIFT extraction when its `.sift` file is missing:
145 to 290 ms per 4K frame on 32 cores at the measurements above. The baseline
reads about five frames per sample, about 1000 frames at the default of 200
samples. On a 6000-frame walk at about 15 frames per step, the walk chooses
about 400 frames and reads perhaps 2 to 3 per choice. That is about 2000 of
the 6000 frames, roughly 5 to 10 minutes on 32 cores, after which `sfm solve -r
NAME` finds every `.sift` file it needs already written. A second run reads the
same files and extracts nothing.

`image_files_to_sift_files` checks, on every call that has a file to write,
every `.sift` file already in the target folder for version 1
(`_check_versions_never_mix`), which opens each file. Called once per walk
step, that check grows with the files the run has written. `pick_images`
makes the folder check once, before the first extraction
([The version 1 refusals](#the-version-1-refusals)), so the per-call check
repeats work; the source calls extraction in batches, and
`image_files_to_sift_files` should remember, for the life of the process, a
folder it has found to hold no version 1 file and has since written only
version 2 files into.

## Settling the defaults

The target step of 10% and the blur ratio of 0.6 are decided. Every other
value in this draft is a starting value: the band of ±30%, the inlier share of
0.5, the minimum of 30 matches, the baseline of 200 samples, the solvability
thresholds (an F-inlier share of 0.3, an H/F of 0.97, a rotation error of 0.01
and a total motion of 100%), the z of 2, the ratio test of 0.8, the inlier
bounds of 1 and 2 pixels per 960, and the four signs of a video (a 3-frame to 1-frame ratio between 2 and
4, a 1-frame step below 5%, neighbouring samples' steps within a factor of 2, a
1-frame F-inlier share above 0.8, with three of the four needed). They are to
be set by running the command on the maintainer's collection of captures,
covering at least:

- walking videos, where the walk should choose steps near the target, read few
  frames, and give a list that solves in one piece;
- a video with a blurred run, where the swaps, the gaps filled in and the
  frames kept though blurred should cover the run and nothing else;
- a rotation-only panorama, which the verdict should stop;
- a video with a moving scene, which the verdict should stop;
- a capture of several sequences, one list across them;
- a folder of photographs numbered in the order they were taken, which the
  verdict should name as not a video, beside a video folder in the same run,
  which it should not;
- a rig capture, with every sensor of each chosen frame in the list.

Each value is judged by solving the lists it gives and by looking at the
solves, not by the command's own numbers alone. Where a value moves, the draft
records the captures that moved it, by kind.

## Non-goals

- Choosing among photographs. A set of separate photographs has no motion a
  step can follow. Files with no number are an error naming them
  ([Collecting the frames](#collecting-the-frames)), and a numbered set is
  judged not to be a video by the baseline ([Is it a video?](#is-it-a-video)),
  which stops the command with a message that it picks from video sequences
  only.
- A rotation-only solve. The command stops a panorama capture rather than
  choosing frames for a solver that would accept it.
- Recording why a frame was chosen in the list or in a `.sfmr`.

## Open questions

- **The output option's name.** This draft uses `--write-list NAME`. `--name`
  was the maintainer's example; it reads less clearly beside the input's
  `--named-image-list`.
- **Taking `-r` and `--named-image-list` for the input.** This draft makes the
  command a ninth command of the shared `-r/--range` option, so a list or a
  range can narrow the input. Without them the input would be `PATHS` alone.
- **The default baseline size.** 200 samples, about 1000 frames read. "A few
  hundred" could be 300; the cost grows by five frames per sample.
- **The band and the inlier bar.** ±30% and 0.5, and whether the band should be
  asymmetric (a short step costs only a frame more; a long one can lose the
  match).
- **The blur rule's spread.** A robust z of 2 against the MAD, one-sided. The
  MAD of a capture with very even blur readings is small and makes the rule
  strict; a floor on the spread may be needed.
- **Medians per sequence or per capture.** This draft takes the sharpness and
  tensor medians per sequence, since separate sequences can differ in light and
  scene. For several sequences of one camera in one place, one median over the
  capture may be better.
- **A failing sequence among several.** This draft stops the whole run when any
  sequence fails the verdict. It could instead write a list of the sequences
  that pass.
- **Rigs.** Motion from the reference sensor only, sharpness per sensor against
  its own median, and a frame blurred when any sensor is. Whether a sensor
  pointing at a featureless view should be allowed to make every frame look
  blurred, and whether `.camrig` files, as well as `rig_config.json`, should
  define the rigs.
- **Rig list entries.** One entry per sensor with the same range keeps each
  pattern literal but orders the list by sensor and then frame, so a slice
  of the list ([named-image-lists.md § Slices](../workspace/named-image-lists.md#slices))
  takes the first sensor's images before any other's; one entry with a `*`
  segment keeps the sensors of a frame together but can match folders that
  are not sensors.
- **A range too long to write.** A range of more than 10,000 runs
  (`openjd_expr`'s `MAX_RANGE_EXPR_CHUNKS`) does not parse back, and
  `write_lists` refuses it. The command could split such a sequence into
  several entries of one pattern, or stop with the error.
- **The exit status** when the verdict stops a run that was asked to write a
  list: this draft exits 1. A flag to choose frames anyway, for a person who
  means to solve a panorama with a rotation-only tool, is not proposed.
- **The command's category.** *Image Feature*, beside `sift` and `match`, since
  it reads images and extracts features. *Workspace* is the other candidate,
  since it writes a workspace file.
- **Telling photographs from a video.** The four signs, their thresholds and
  the rule that three of four make a video are starting values. Whether a
  sequence with a few photograph-like stretches (a video paused and resumed,
  say) should be split rather than judged whole.
- **A way to choose among photographs.** How a set of photographs should be
  chosen from, for example by how much of the scene each one's matches cover
  and by sharpness alone, without steps, is a design of its own and would be
  proposed in its own draft. This draft only detects such a sequence and stops.
- **The JSON schema** of `--json`, which this draft does not fix.
