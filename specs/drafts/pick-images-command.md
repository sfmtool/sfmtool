# `sfm pick-images`

**Status:** Draft

Proposes a new command. Decided with the maintainer: the command's name and
its one-command shape (it reports, and writes a list when given a name), that a
written list records nothing about how it was picked, the refusal to overwrite
without `--force`, one list per run across every sequence, the workspace and
`.sift` version 2 requirement, SIFT matching through a KD-forest as the measure
of motion, the defaults of 10% and 0.6, and the algorithm in
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

It replaces what the out-of-tree `reconstruct-dataset` skill's `probe.py`
script does today: optical flow at 20 positions, a fixed stride, blurred
frames swapped for a sharper neighbour, three test subsets and a verdict. That
script prints a range expression for the user to copy; this command writes the
list itself, and it chooses each step by measuring it rather than by a stride
fixed in advance.

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
| `--write-list NAME` | | Write the chosen frames as the named image list `NAME`, and the test lists beside it. Without it the command only reports. |
| `--force` | off | Replace a list of that name (and the test lists' names) if the workspace already defines one. |
| `-r/--range` | | A range of file numbers, or the name of a named image list, that narrows the input ([the shared `-r/--range` option](../cli/README.md#the-shared--r--range-option)). |
| `--named-image-list NAME` | | Read the input frames from this named image list, in place of `PATHS`. |
| `--target-step PERCENT` | `10` | The step to aim for between two chosen frames: the median feature displacement, in percent of the image's long side. |
| `--step-band FRACTION` | `0.3` | How far a step may be from the target and still be accepted: within `target × (1 ± FRACTION)`, 7% to 13% at the defaults. |
| `--blur-ratio RATIO` | `0.6` | A frame whose sharpness is below `RATIO` times the capture's median sharpness counts as blurred. |
| `--baseline-frames N` | `200` | How many frames the baseline sample takes, spread evenly over the capture. |
| `--min-inlier-share SHARE` | `0.5` | A step whose matches fit a fundamental matrix with a smaller share of inliers fails. |
| `--test-lists / --no-test-lists` | on | With `--write-list`, also write the three test lists. |
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
it already defines `NAME` or, with test lists on, one of the test lists'
names, naming the file and each such list and saying to pass `--force` to
replace them. The same check is made again when the file is written, so that a
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
`sfm motion` does. When the verdict says the capture cannot be solved it writes
no list, and with `--write-list` it then exits 1, since the list it was asked
for does not exist.

### Examples

```bash
# Report on a capture without writing anything
sfm pick-images frames/

# Choose the frames and write them as the list walk1, plus walk1.start,
# walk1.middle and walk1.end
sfm pick-images frames/ --write-list walk1
sfm solve -i -r walk1.start     # a quick test solve first
sfm solve -i -r walk1

# Larger steps, and only frames 1-3000
sfm pick-images frames/ -r 1-3000 --target-step 15 --write-list walk1_fast

# Several sequences into one list, replacing an earlier one
sfm pick-images walk1/ walk2/ walk3/ --write-list all_walks --force
```

## The algorithm

The command works in five stages: collect the frames, read a baseline sample,
judge whether the capture can be solved, walk the capture choosing frames, and
report. Each stage reads frames through the same source, which extracts a
frame's `.sift` files when they are missing and reuses them otherwise.

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
frames per sample, and reuses a frame another sample already read. All these
frames are requested from the source together, so extraction runs as one batch.

From the sampled frames' own `.sift` files, per sequence:

- the **median sharpness**, which every later frame is compared with;
- the median and robust spread of the **axis ratio** `|log(Exx/Eyy)|` and the
  **coherence** `sqrt((Exx − Eyy)² + 4·Exy²) / (Exx + Eyy)`, derived from the
  gradient structure tensor as `sfm inspect` derives them
  (`_image_reading_rows` in [`analyze/summary.py`](../../src/sfmtool/analyze/summary.py),
  defined in [sift-file-format.md § Gradient structure tensor](../formats/sift-file-format.md#gradient-structure-tensor)).

From the matches between `s` and `s + 1`, and between `s` and `s + 3`:

- the **rate** at each sample: the step from `s` to `s + 3` divided by 3, with
  the step to `s + 1` as a check that the motion is smooth;
- the **flagged regions**: a sampled frame that is blurred, or whose matches to
  `s + 1` or `s + 3` do not match well, flags the frames from the previous
  sampled frame to the next one. The walk goes through a flagged region in
  short steps ([The walk](#the-walk)).

The baseline covers the whole capture independently of the walk, so a problem
the walk could step over in one long step is still seen.

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

The **rotation error** is the probe's: for focal lengths `f` from `0.3` to `3`
times the long side, on 80 steps spaced evenly in `log f`, with `K` the camera
matrix of focal `f` and the principal point at the image centre, take
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

### Test lists

With `--write-list NAME` and test lists on, the command also walks three short
stretches of the capture, starting at its start, its middle and its end, with
half the target step, until each holds 30 frames or reaches a break or the end
of its sequence. A shorter step keeps each test list's frames close enough to
match well while the 30 frames still span about 1.5 times the image's width
of motion, enough for a solve to find depth. They are written as the lists
`NAME.start`, `NAME.middle` and `NAME.end`, which the name rule allows, since
`.` is one of its characters. The middle one starts at the middle of the
longest segment; the end one walks backward from the last chosen frame of the
last segment. A test list whose
walk would overlap an earlier one is not written, and the report says so.

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
    ],
    "walk1.start": [
      { "pattern": "frames/walk1_%04d.jpg", "range": "1-204:7" }
    ]
  }
}
```

The lists file is rewritten whole, with every list it already held kept in its
place and order, a replaced list in the place of the one it replaces, and new
lists at the end. It is written to a temporary file beside it and renamed over
it, so a failed write leaves the old file.

## Report

The report prints the verdict, the motion along the capture, the blurred frames
swapped, the gaps filled in, the breaks and the weak points. An example, for a
4K video of a walk with one blurred run:

```text
Using workspace: /captures/garden
  Feature tool: sfmtool
1 sequence: frames/walk1_%04d.jpg, 6120 frames (1-6120)

Baseline: 200 frames sampled, 984 frames read (972 extracted, 12 reused)
  Sharpness: median 0.0123; 9 sampled frames blurred
  Motion per frame: median 0.72%, p90 1.31% of the long side; total 4400%
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

Test lists: walk1.start 1-204 (30 frames), walk1.middle 2950-3141 (30 frames),
  walk1.end 5917-6120 (30 frames)
Wrote 4 named image lists to /captures/garden/.sfm-named-image-lists.json:
  walk1 (512 images), walk1.start, walk1.middle, walk1.end
```

A **weak point** is an accepted step with fewer than twice the minimum match
count, or an F-inlier share within `0.1` of `--min-inlier-share`, and a segment
of fewer than 3 frames. These are where a solve is most likely to split.

`--json PATH` writes the same content, with every frame read and its readings,
every pair matched and its statistics, and every decision of the walk.

## Where the code lives

Pair matching and its statistics, and the walk's decisions, live in Rust core,
in a new module `sfmtool_core::analysis::pick_images`
(`crates/sfmtool-core/src/analysis/pick_images.rs` beside a
`pick_images/` directory), bound in `sfmtool.analysis`, following
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
- **Matching.** `KdForestU8::build` and `search_batch_with_distances` in
  [`features/kdforest/`](../../crates/sfmtool-core/src/features/kdforest/mod.rs)
  ([randomized-kdtree-forest.md](../core/features/randomized-kdtree-forest.md)):
  one forest per frame, built once and kept while the frame is in a pair still
  to be matched; `k = 2` queries both ways, the ratio test at `0.8` on the
  distances (squared distances against `0.8²`), and the mutual best matches.
- **Geometry.** `estimate_fundamental` in
  [`epipolar_estimation.rs`](../../crates/sfmtool-core/src/geometry/epipolar_estimation.rs)
  ([epipolar-estimation.md](../core/geometry/epipolar-estimation.md)) and
  `estimate_homography` in
  [`homography_estimation.rs`](../../crates/sfmtool-core/src/geometry/homography_estimation.rs),
  each with a fixed seed, so a run is repeatable. Their inlier bounds scale
  with the image: 1 pixel per 960 pixels of the long side for the fundamental
  matrix and 2 per 960 for the homography, the bounds `probe.py` used at
  960 pixels.
- **Statistics.** Medians from
  [`sfmtool_core::numeric`](../../crates/sfmtool-core/src/numeric.rs), and the
  MAD as the median of the absolute deviations from the median, since the
  module has no MAD of its own.
- **Lists.** `NamedImageLists` in
  [`named_image_lists.rs`](../../crates/sfmtool-core/src/workspace/named_image_lists.rs)
  reads and checks the file, and `is_named_image_list_name` checks the output
  name. The module has no writer today; the command needs one, below.
- **Input.** `range_option`, `parse_image_selection` and `resolve_list_images`
  in [`_range_options.py`](../../src/sfmtool/_commands/_range_options.py).
- **Progress.** `ProgressCounter` ([py_progress.rs](../../crates/sfmtool-py/src/py_progress.rs)).

### Rust interface

The walk asks a caller-supplied source to make sure frames' `.sift` files
exist, then reads them itself:

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
    pub min_matches: usize,        // 30
    pub test_lists: bool,          // true
    pub forest: KdForestParams,
}

pub fn pick_images(
    sequences: &[CaptureSequence],
    source: &mut dyn FrameSource,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<PickImagesReport, PickImagesError>;

/// One pair's statistics, for a caller that matches two frames on its own.
pub fn match_frame_pair(
    a: &FrameFeatures,
    b: &FrameFeatures,
    options: &FramePairOptions,
) -> Result<FramePairStats, PickImagesError>;

pub struct FramePairStats {
    pub matches: usize,
    pub step_percent: f64,
    pub f_inlier_share: Option<f64>,  // None under min_matches
    pub h_over_f: Option<f64>,
    pub rotation_error: Option<f64>,
}
```

`PickImagesReport` holds the verdict and the checks behind it, the baseline's
readings, the chosen frames per sequence, the test walks, every step the walk
measured with its `FramePairStats`, the swaps, the gaps filled in, the frames
kept though blurred, the breaks, the weak points and the count of frames read.
`PickImagesError` is an enum: `Options` (a value out of range, such as a
target step that is not positive or a blur ratio outside `(0, 1)`), `Sequence`
(an empty sequence, or a frame whose sensor count differs from its sequence's),
`Source` (the source's error, carrying its message), `ReadSift` (with the path
and the `SiftError`), `Version1Sift` (with the path), `ImageSize` (two frames
of one sensor whose sizes differ) and `Cancelled`.

**Why it is shaped this way.** The walk decides which frame to read next from
what it read last, so it cannot be handed all its frames in advance; asking a
source keeps the walk's logic in Rust while extraction stays where it is, in
the Python extraction path that handles all three feature tools (Rust's own
`build_sift_files` writes only the `sfmtool` tool's files). The source is asked
only to make the files exist, not to return features, because the Rust reader
already reads and checks `.sift` files and the features then never cross the
binding. It takes several frames per call because the baseline and a blur
search know several frames at once, and extraction in a batch keeps the
extractor's concurrent decodes busy. `match_frame_pair` is public because the
same statistics are what a later command, or `sfm motion`, could read from a
pair. `pick_images` takes a `&Progress` and checks it for cancellation, since
it runs for minutes, and returns an error rather than panicking on any input
([STANDARDS.md § Caller input never panics](../STANDARDS.md#caller-input-never-panics)):
`KdForest::build` asserts its lengths and its parameters, so `pick_images` and
`match_frame_pair` check the forest parameters as options and a `.sift` file's
feature arrays against each other before building a forest.

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
each call, and raises the callable's exception unchanged. Every other error is
a `ValueError` with the Rust message, a `.sift` read error the `OSError` of
[python-bindings.md § Format errors](../python-bindings.md#format-errors), and
a cancelled run `InterruptedError`.

The lists writer is a new method beside the reader, in Rust and bound in
`sfmtool.workspace`:

```rust
impl NamedImageLists {
    /// Add or replace lists in the workspace's lists file, creating it when
    /// absent. Refuses a name the file already defines unless `replace`.
    pub fn write_lists(
        workspace_dir: &Path,
        lists: &[(String, Vec<(String, Option<RangeExpr>)>)],
        replace: bool,
    ) -> Result<PathBuf, NamedImageListsError>;
}
```

It checks the new names, patterns and ranges by the reader's rules, reads and
checks the existing file, refuses with a new `ListExists` variant naming each
list, and writes as [Writing the list](#writing-the-list) says. The existing
lists' ranges are written back from their text as read, which needs the
reader to keep that text beside the parsed `RangeExpr`.

### The version 1 refusal

Every reading comes from a version 2 `.sift` file. A version 1 workspace file
is refused before anything is read, with the message `sfm sift --extract` gives
for version 1 settings, which says to run
[`sfm ws upgrade`](../cli/workspace/ws-upgrade-command.md): the upgrade moves
the workspace to version 2 and a new features folder, where this command then
extracts version 2 files. This has to be checked up front, because
`image_files_to_sift_files` refuses version 1 settings only when it has a file
to write, and a version 1 workspace whose files are all present would otherwise
reach the reader. A `.sift` file the walk reads that is still version 1 (a
version 2 workspace's folder holds version 1 files only after a hand edit,
since [versions never mix](../workspace/workspace.md#versions-never-mix)) is
`Version1Sift`, whose message names the file and says it records no
sharpness.

## Why SIFT matching rather than optical flow

`probe.py` measured motion with optical flow. Optical flow and SIFT matching
were compared on two 4K portrait videos, one with a blurred run, CPU only, on
frame pairs 1, 3, 10 and 30 frames apart starting at four or five places in
each video, with each frame at 960 pixels, 1920 pixels and full size on its
long side. The SIFT matches were mutual best matches through a KD-forest with
the ratio test at `0.8`, with features capped at 8192, and the step was the
median displacement of the matches; the flow's was the median length of the
flow field.

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
step, that check grows with the files the run has written. The source should
call extraction in batches and make the folder check once per run, or the
check should remember a folder it has found clean.

## Settling the defaults

The target step of 10% and the blur ratio of 0.6 are decided. Every other
value in this draft is a starting value: the band of ±30%, the inlier share of
0.5, the minimum of 30 matches, the baseline of 200 samples, the solvability
thresholds (an F-inlier share of 0.3, an H/F of 0.97, a rotation error of 0.01
and a total motion of 100%, all from `probe.py`), the z of 2, the ratio test
of 0.8, the inlier bounds of 1 and 2 pixels per 960, and the test lists' 30
frames at half the target step. They are to be set by running the command on
the maintainer's collection of captures, covering at least:

- walking videos, where the walk should choose steps near the target, read few
  frames, and give a list that solves in one piece;
- a video with a blurred run, where the swaps, the gaps filled in and the
  frames kept though blurred should cover the run and nothing else;
- a rotation-only panorama, which the verdict should stop;
- a video with a moving scene, which the verdict should stop;
- a capture of several sequences, one list across them;
- a rig capture, with every sensor of each chosen frame in the list.

Each value is judged by solving the lists it gives and by looking at the
solves, not by the command's own numbers alone. Where a value moves, the draft
records the captures that moved it, by kind.

## Non-goals

- Photographs that are not a sequence. A folder of separate photographs has no
  order a step can follow, and its files often carry no number `-r` can select.
  The command stops with an error naming the files that carry no number.
  `probe.py`'s `--photos` mode, which matched photographs neighbouring by file
  number, is not carried over; see Open questions.
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
- **The test lists' names and default.** `NAME.start`, `NAME.middle` and
  `NAME.end`, on by default with `--write-list`. Alternatives: `NAME-test1` and
  so on, or off by default. Whether `--force` should be needed for them
  separately from `NAME`. Their length (30 frames) and step (half the target)
  are starting values.
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
  pattern literal but orders the list by sensor and then frame; one entry with
  a `*` segment keeps the sensors of a frame together but can match folders
  that are not sensors.
- **Writing the lists file.** Rewriting it whole loses formatting a person gave
  it by hand; this draft keeps each list's ranges as their text. A range of more
  than 10,000 runs (`openjd_expr`'s `MAX_RANGE_EXPR_CHUNKS`) would not parse
  back; the writer should check that the range it writes parses, and split the
  entry when it does not.
- **The exit status** when the verdict stops a run that was asked to write a
  list: this draft exits 1. A flag to choose frames anyway, for a person who
  means to solve a panorama with a rotation-only tool, is not proposed.
- **The command's category.** *Image Feature*, beside `sift` and `match`, since
  it reads images and extracts features. *Workspace* is the other candidate,
  since it writes a workspace file.
- **Photographs.** Whether to support separate photographs at all, by matching
  each to its neighbours by file number as `--photos` did, or to leave them to
  `sfm match` and hand-written lists.
- **The JSON schema** of `--json`, which this draft does not fix.
