# `sfm pick-images`

**Status:** Draft

Proposes a new command. The walk it runs is built and specified in
[core/analysis/pick-images.md](../core/analysis/pick-images.md): the
baseline, the judgements of a video and of solvability, the walk with its
swaps, gaps, breaks and weak points, blur, rigs, the checks for version 1
files, and the `sfmtool.analysis.pick_images` binding. This draft proposes the
command around it: collecting the frames, the options, the report it prints
and the list it writes. Decided with the maintainer: the command's name and
its one-command shape (it reports, and writes a list when given a name), that a
written list records nothing about how it was picked, the refusal to overwrite
without `--force`, one list per run across every sequence, the workspace and
`.sift` version 2 requirement, SIFT matching through a KD-forest as the measure
of motion, the detection of a sequence of photographs, which stops the
command, the defaults of 10% and 0.6, and the walk. The smaller choices this
draft makes on its own are listed under [Open questions](#open-questions), and
the default values other than 10% and 0.6 are starting values to be set as
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

The command collects the frames into sequences, builds a source over
`image_files_to_sift_files`, and calls `sfmtool.analysis.pick_images`, which
reads the baseline, judges each sequence and the capture, and walks each
sequence ([core/analysis/pick-images.md § The algorithm](../core/analysis/pick-images.md#the-algorithm)).
It then prints the report and, when the verdict is solvable and a name was
given, writes the list. The option values reach `pick_images` unchanged:
`--target-step` as `target_step_percent`, `--step-band` as `step_band`,
`--blur-ratio` as `blur_ratio`, `--baseline-frames` as `baseline_frames` and
`--min-inlier-share` as `min_inlier_share`.

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
all its sensors. [core/analysis/pick-images.md § Rigs](../core/analysis/pick-images.md#rigs)
says how its motion and sharpness are judged.

The sequences are walked one after another, and the baseline sample is shared
among them in proportion to their frame counts.

Each frame becomes one `CaptureFrame`, a rig frame with the reference
sensor's image first, and each sequence one `CaptureSequence` whose patterns
are the ones its list entries use ([Writing the list](#writing-the-list)).

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
sensor's images; the report gives no suggestion for a rig list.

The verdict and its checks, the motion, the swaps, the gaps filled in, the
frames kept though blurred, the breaks and the weak points are the fields of
the report `pick_images` returns
([core/analysis/pick-images.md § The report](../core/analysis/pick-images.md#the-report)),
which names frames by their position in their sequence; the command prints
them as file numbers through the sequences it built.

The verdict line has the shape of the `Verdict:` line
[`sfm motion`](../cli/reconstruction/motion-command.md) prints for a
reconstruction, and is the last line of the baseline section. A sequence judged
not to be a video is named, with a message that the command picks from video
sequences only:

```text
Verdict: not a video: photos/DSC_%04d.JPG: steps 1 frame apart median 18.4% and erratic, 3 frames apart 1.2 times as far; sfm pick-images picks from video sequences only
Verdict: solvable
Verdict: not solvable: panorama: 30 frames apart, 0.991 of the matches fit a homography and it is a pure rotation (error 0.004)
Verdict: not solvable: walk2: moving scene: only 0.21 of the matches 10 frames apart fit one camera motion
```

When the verdict stops the run, the command writes no list, and the `.sift`
files of the baseline stay.

`--json PATH` writes the same content, with every frame read and its readings,
every pair matched and its statistics, and every decision of the walk.

## Where the code lives

The walk lives in Rust core, in `sfmtool_core::analysis::pick_images`
([core/analysis/pick-images.md](../core/analysis/pick-images.md)), bound in
`sfmtool.analysis`, following
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
  or `sfm solve` finds the files and extracts nothing for them. The source the
  command hands `pick_images` calls it with the images of the frames it is
  asked for:

  ```python
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

  The callable cannot raise the version 1 refusals of
  `image_files_to_sift_files` for a folder, because `pick_images` has checked
  every folder before the first call
  ([core/analysis/pick-images.md § Version 1 files](../core/analysis/pick-images.md#version-1-files)).
- **The walk.** `pick_images`, `CaptureSequence`, `CaptureFrame` and
  `CaptureImage` in `sfmtool.analysis`.
- **Lists.** `NamedImageLists` in
  [`named_image_lists.rs`](../../crates/sfmtool-core/src/workspace/named_image_lists.rs)
  reads and checks the file, `is_named_image_list_name` checks the output
  name, and `NamedImageLists::write_lists` writes the list
  ([named-image-lists.md § Writing the file](../workspace/named-image-lists.md#writing-the-file)).
  The command writes it with `NamedImageLists.write_lists` in
  `sfmtool.workspace`, passing `replace=True` for `--force`. Its `ListExists`
  refusal is the second check of [Refusing to overwrite](#command).
- **Input.** `range_option`, `parse_image_selection` and `resolve_list_images`
  in [`_range_options.py`](../../src/sfmtool/_commands/_range_options.py).
- **Progress.** `ProgressCounter` ([py_progress.rs](../../crates/sfmtool-py/src/py_progress.rs)),
  which `pick_images` sets to the number of frames read.

### The version 1 refusals

Every reading comes from a version 2 `.sift` file. `pick_images` refuses
version 1 files in a folder the capture's `.sift` files go in before anything
is extracted, and a version 1 file it reads
([core/analysis/pick-images.md § Version 1 files](../core/analysis/pick-images.md#version-1-files)).
That covers a version 2 workspace whose folder holds version 1 files, put there
by hand or named by a workspace file edited by hand. `sfm ws upgrade` cannot
help there: on a version 2 workspace file it prints "nothing to do: already
version 2" and changes nothing.

The command adds the check of a **version 1 workspace**, made before it reads
any frame, since [versions never mix](../workspace/workspace.md#versions-never-mix).
Its `feature_options` carry no `image_to_gray`, the test
`image_files_to_sift_files` applies to version 1 settings, and the command
refuses with the message `sfm sift --extract` gives for version 1 settings: it
names the workspace file and says to run
[`sfm ws upgrade`](../cli/workspace/ws-upgrade-command.md), which moves the
workspace to version 2 and a new features folder. The advice works there:
`upgrade_workspace` rewrites a version 1 file as version 2 and names a new
`feature_prefix_dir`, which holds no `.sift` files, so the next run extracts
version 2 files into it. The check has to be made up front because
`image_files_to_sift_files` refuses version 1 settings only when it has a file
to write, and a version 1 workspace whose files are all present would
otherwise reach the walk's reader, whose `Version1Sift` advice to remove the
file is the wrong fix for a version 1 workspace.

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
value is a starting value
([core/analysis/pick-images.md § Parameters](../core/analysis/pick-images.md#parameters)): the band of ±30%, the inlier share of
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
  judged not to be a video by the baseline
  ([core/analysis/pick-images.md § Is it a video?](../core/analysis/pick-images.md#is-it-a-video)),
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
  strict, and a MAD of 0 makes every reading above the median a blur; a floor
  on the spread may be needed.
- **Medians per sequence or per capture.** The walk takes the sharpness and
  tensor medians per sequence, since separate sequences can differ in light and
  scene. For several sequences of one camera in one place, one median over the
  capture may be better.
- **A failing sequence among several.** The walk stops the whole run when any
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
  say) should be split rather than judged whole. A camera held still moves by
  sub-pixel jitter, whose 3-frame step is not three times its 1-frame step and
  whose steps vary by more than a factor of 2, so it fails two signs and is
  judged not a video rather than a camera that barely moves.
- **A stretch where every frame is blurred.** The walk then keeps the sharpest
  frame up to the band's upper edge, which can be the next frame, so such a
  stretch is walked in steps far below the band. The walk could instead take
  the farthest frame whose sharpness is within some margin of the sharpest.
- **A way to choose among photographs.** How a set of photographs should be
  chosen from, for example by how much of the scene each one's matches cover
  and by sharpness alone, without steps, is a design of its own and would be
  proposed in its own draft. This draft only detects such a sequence and stops.
- **The JSON schema** of `--json`, which this draft does not fix.
