# `sfm pick-images` Command

A video of a walk around a scene has many more frames than a
Structure-from-Motion solve should use. Neighbouring frames are nearly the same
picture and some frames are blurred. `sfm pick-images` looks at a capture
before it is solved and chooses the frames a solve should use: frames spaced
so that the image moves by about a tenth of its width from one to the next,
sharp ones in place of blurred ones, more frames where the capture is hard to
follow and fewer where it is easy. It prints what it found, including a
verdict on whether each sequence is a video, and when given a name it writes
the frames it chose
as a named image list ([workspace/named-image-lists.md](../../workspace/named-image-lists.md)),
which `sfm solve -r NAME` and the other commands that take `-r` then use.

It measures everything from SIFT features. Each frame it reads, and each it
predicts it will read, gets a `.sift` file, extracted with the workspace's own
settings and stored where `sfm sift --extract` would store it, so a later
`sfm sift`, `sfm match` or `sfm solve` finds the file and extracts nothing for
it; in a workspace whose feature tool is not `sfmtool`, every input frame gets
one ([Extraction](#extraction)). It reads only the frames it needs: a sample
spread over the whole capture, the frames it chooses, and the frames it tries
on the way. The sample, the verdict and the
walk that chooses the frames are
[core/analysis/pick-images.md](../../core/analysis/pick-images.md)'s; this
spec covers the command around them: collecting the frames, the options, the
report it prints and the list it writes.

## Command

```bash
sfm pick-images [PATHS...] [OPTIONS...]
```

`PATHS` are image folders or files, as for [`sfm sift`](sift-command.md):
folders are expanded recursively and keep the files ending in `.png`, `.jpg`
or `.jpeg` (in any letter case). When omitted, the current directory is used.
The command needs a workspace, found by the upward search from the images'
common parent ([workspace.md § Workspace Discovery](../../workspace/workspace.md#workspace-discovery)),
and without one it stops with an error asking for `sfm ws init`. There is no
`--tool` option: the feature tool and its settings are always the workspace's.

| Option | Default | Meaning |
|--------|---------|---------|
| `--write-list NAME` | | Write the chosen frames as the named image list `NAME`. Without it the command only reports. |
| `--force` | off | Replace a list of that name if the workspace already defines one. Only with `--write-list`. |
| `-r/--range` | | A range of file numbers, or the name of a named image list with an optional slice, that narrows the input ([the shared `-r/--range` option](../README.md#the-shared--r--range-option)). |
| `--named-image-list NAME` | | Read the input frames from this named image list, in place of `PATHS`. |
| `--target-step PERCENT` | `10` | The step to aim for between two chosen frames: the median feature displacement, in percent of the image's long side. |
| `--step-band FRACTION` | `0.3` | How far a step may be from the target and still be accepted: within `target × (1 ± FRACTION)`, 7% to 13% at the defaults. |
| `--blur-ratio RATIO` | `0.6` | A frame whose sharpness is below `RATIO` times its sequence's median sharpness counts as blurred. |
| `--neighbour-blur-ratio RATIO` | `0.85` | A frame about to be chosen whose sharpness is below its sequence's median is blurred against its neighbours, and swapped for a sharper neighbour, when its sharpness is below `RATIO` times the sharpest of the frames up to 2 positions away ([core/analysis/pick-images.md § Blur](../../core/analysis/pick-images.md#blur)). |
| `--baseline-frames N` | `200` | How many frames the baseline sample takes, spread evenly over the capture. The default suits long captures of 10,000 frames or more; on a capture of a few hundred frames the baseline reads most of them ([core/analysis/pick-images.md § The baseline](../../core/analysis/pick-images.md#the-baseline)). |
| `--json PATH` | | Also write the whole report to a JSON file ([JSON](#json)). |

The five numeric options reach `sfmtool.analysis.pick_images` unchanged, as
`target_step_percent`, `step_band`, `blur_ratio`, `neighbour_blur_ratio` and
`baseline_frames`, which checks their ranges before
it reads anything; a value out of range is an error with the binding's message
([core/analysis/pick-images.md § Parameters](../../core/analysis/pick-images.md#parameters)).

**Why `--write-list` and not a positional name or `--name`.** On the commands
that take it, `--named-image-list NAME` names a list to **read**, and this
command takes that option too, for its input. A bare `--name` beside it would
leave a reader to guess which of the two names is the output, and a positional
name would be confused with `PATHS`. `--write-list` says what it does, and its
absence says that the run only reports: the same command reports and writes,
and a list is written exactly when a name is given. A name that breaks the
list name rule is a usage error before anything is read.

**`-r` narrows the input.** A range keeps the files whose number
`number_from_filename` reads is in it, and a list name replaces `PATHS` with
the list's image members, resolved in the workspace the upward search from the
current directory finds; a slice after the name keeps part of them by
position. So `sfm pick-images -r walk1 --write-list walk1_picked` chooses
frames from among an existing list's. However the files were selected, the
command orders each sequence by file number ([Collecting the frames](#collecting-the-frames)).

**Refusing to overwrite.** Before it reads any image, the command reads the
workspace's lists file (`NamedImageLists.read`) and stops when it already
defines `NAME`, with an error naming the file and the list and saying to pass
`--force` to replace it. `NamedImageLists.write_lists` makes the same check
again when the file is written, so a list another command added in the
meantime is refused rather than lost. The output name may be the input list's
own name only with `--force`; the input is resolved before anything is
written.

**No provenance.** A written list holds pattern and range entries only, in the
lists file's version 1 layout. Nothing about the run (the options, the verdict,
the readings) goes into the lists file, the `.sift` files or the workspace
file. The printed report and `--json` are the only record of why a frame was
chosen.

**One list per run.** The list covers every frame chosen in the run, across
every folder and sequence it was given ([Writing the list](#writing-the-list)).

**Exit status.** The command exits 0 when it finishes, whatever it found, as
`sfm motion` does. When the verdict says a sequence is not a video, it writes
no list; with `--write-list` it then says
so on standard error and exits 1, since the list it was asked for does not
exist. An error that stops the run exits 1, and a usage error 2.

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

## What the command does

The command is [`_commands/pick_images.py`](../../../src/sfmtool/_commands/pick_images.py),
with its row in `cli.COMMANDS` under *Image Feature*. The grouping into
sequences, the source and the list entries are in
[`_pick_images.py`](../../../src/sfmtool/_pick_images.py), and the report's
text in [`_pick_images_report.py`](../../../src/sfmtool/_pick_images_report.py).
In order, it:

1. Resolves the input files and finds the workspace.
2. Loads the workspace file and refuses version 1 settings
   ([The version 1 refusals](#the-version-1-refusals)).
3. With `--write-list` and no `--force`, refuses a name the lists file already
   defines.
4. Groups the files into sequences and prints them.
5. With the `sfmtool` feature tool, calls `sfmtool.analysis.pick_images`
   with one `CaptureSequence` per sequence and the workspace's feature
   options, which extracts the `.sift` files itself; with another tool,
   extracts every input frame first and calls it to read them
   ([Extraction](#extraction)).
6. Prints the report, writes the list when every sequence is a video and a name
   was given, and writes the JSON file when asked.

### Collecting the frames

The command groups the input files into **sequences**: the files of one folder
whose names are the same apart from the file number `number_from_filename`
reads ([`_filenames.py`](../../../src/sfmtool/_filenames.py)), the number `-r`
selects a file by. `walk1/frame_0001.jpg` to `walk1/frame_6120.jpg` are one
sequence; `walk1/a_0001.jpg` beside `walk1/b_0001.jpg` are two, and so are
`x_0001.jpg` and `x_0002.JPG`. Each sequence is ordered by file number, and the
sequences by their first pattern ([Writing the list](#writing-the-list)). The
command reads the number's place in the name with `file_number_span`, the
function `number_from_filename` reads the number with, so the list pattern and
`-r` agree on which digits are the number.

It stops, before reading any image, with one error that lists every problem
and names the files:

- a file with no number, since `-r` cannot select it and it has no place in an
  order;
- two files of one sequence with the same number (`shot_1.jpg` and
  `shot_01.jpg`), which have no order;
- a file with a `*` in its workspace-relative path, which a list pattern reads
  as a wildcard and cannot hold as text;
- a rig frame that lacks the image of a sensor another frame of its sequence
  has.

**Rigs.** When the workspace has a `rig_config.json`
([workspace/rig-config.md](../../workspace/rig-config.md)), an image under one
of a rig's sensor prefixes belongs to that sensor, and the images of one rig
with the same **frame key**, the path after the sensor prefix, are one rig
frame, as `sfm solve` groups them (`_match_image_to_sensor` and
`_infer_frame_key` in [`rig/config.py`](../../../src/sfmtool/rig/config.py)).
The sequence is then formed from the frame keys, and each rig frame is one
frame of it: read, judged and chosen with all its sensors
([core/analysis/pick-images.md § Rigs](../../core/analysis/pick-images.md#rigs)).
A frame's images are the reference sensor's first, whatever its place in the
file, then the others in the file's order. The sensors are those that appear
in the sequence's files. Images under no sensor prefix form ordinary
sequences. `.camrig` files do not define rigs here.

Each frame becomes one `CaptureFrame` with its file number, each image a
`CaptureImage` with the path its `.sift` file has under the workspace's
`feature_prefix_dir`, and each sequence one `CaptureSequence` whose patterns
are the ones its list entries use. The baseline sample is shared among the
sequences in proportion to their frame counts, and the sequences are walked
one after another.

### Extraction

How the frames get their `.sift` files depends on the workspace's
`feature_tool`.

**The `sfmtool` tool**, the one `sfm ws init` chooses by default, makes one
call: `pick_images` is given the workspace's `feature_options` and
`feature_prefix_dir` and extracts in Rust, as `build_sift_files` does
([core/features/sift.md § Building `.sift` files](../../core/features/sift.md#building-sift-files)):
the same files at the same paths, with the same content, a file at least as
new as its image used as it is, and every file kept. The extraction runs on a
pool of workers, as many as `build_sift_files` runs (one per core, capped
by memory, or `SFMTOOL_SIFT_EXTRACT_WORKERS`), ahead of the walk, on the frames the walk knows it will read:
the baseline, and the frames ahead of it through a break
([core/analysis/pick-images.md § Run-ahead](../../core/analysis/pick-images.md#run-ahead)).
The walk's steps wait for each frame they measure.
The walk decides only from frames it waited for, so the frames chosen are the
same whatever the workers did ahead of it; a predicted frame the walk does not
read keeps its `.sift` file.

**Another tool** (`colmap` or `opencv`) is extracted in Python, so the
command first calls `image_files_to_sift_files`
([`sift/extract.py`](../../../src/sfmtool/sift/extract.py)) once with every
input frame's images and the workspace's `feature_tool`, `feature_options` and
`feature_prefix_dir`, as `sfm sift --extract` calls it in a workspace, with its
own output, and then calls `pick_images` with no feature options, which reads
the files and extracts nothing. Such a workspace gets a `.sift` file for every
input frame, not only for those the walk reads.

A frame whose `.sift` file was written in the run counts as **extracted** in
the report and the JSON file, and one read from a file already up to date as
**reused**: with the `sfmtool` tool from the report's `frames_extracted`, and
with another tool from whether the file's modification time changed in the
extraction made first. Every 10 seconds of the walk the command prints how many
frames have been read so far, from the `ProgressCounter`
([py_progress.rs](../../../crates/sfmtool-py/src/py_progress.rs)) that
`pick_images` sets to the number of frames read. Ctrl+C stops the run and its
workers, as
[core/analysis/pick-images.md § Python bindings](../../core/analysis/pick-images.md#python-bindings)
describes.

**Cost.** Reading a frame costs one SIFT extraction when its `.sift` file is
missing, and nothing more than reading the file otherwise. The baseline reads
each sampled frame and the frames 1 and 3 after it, so up to three
frames per sample, fewer where two samples share a frame. The walk reads each
frame it chooses and the candidates it measures on the way, usually two or
three per choice, and skips the frames between; a choice less sharp than the
sequence's median also reads the frames within 2 positions of it, to judge it
against its neighbours; with the `sfmtool` tool the
frames prefetched past the end of a break are extracted too, which the last
line of the report counts. A later `sfm solve -r NAME` finds every `.sift`
file it needs already written, and a second run of the command reads the same
files and extracts none of them.

### The version 1 refusals

Every reading comes from a version 2 `.sift` file. `pick_images` refuses
version 1 files in a folder the capture's `.sift` files go in before anything
is extracted, and a version 1 file it reads
([core/analysis/pick-images.md § Version 1 files](../../core/analysis/pick-images.md#version-1-files)).
That covers a version 2 workspace whose folder holds version 1 files.

The command adds the check of a **version 1 workspace**, made before it reads
any frame, since [versions never mix](../../workspace/workspace.md#versions-never-mix).
It is the settings check of `image_files_to_sift_files` (`_check_versions_never_mix`
with no folder to list), so its refusals and their messages are
`sfm sift --extract`'s ([sift-command.md § Version refusals](sift-command.md#version-refusals)):
version 1 settings, whose message names the workspace file and says to run
[`sfm ws upgrade`](../workspace/ws-upgrade-command.md), which moves the
workspace to version 2 and a new features folder; a `colmap` or `opencv`
conversion other than the tool's; and the `sfmtool-undistort` tool. The check
is made up front because `image_files_to_sift_files` refuses version 1
settings only when it has a file to write, and a version 1 workspace whose
files are all present would otherwise reach the walk's reader, whose
`Version1Sift` advice to remove the file is the wrong fix for a version 1
workspace.

## Writing the list

Each sequence becomes one entry of the list: a pattern made from the sequence's
file names, with the file number replaced by a frame field and `%%` for a `%`
in the path, and a range of the chosen frames' numbers. The frame field is
`%0Nd` when every number in the sequence is written with the same `N > 1`
digits, and `%d` otherwise. The range is `RangeExpr.from_list` of the numbers,
written as its `str`, which joins runs into `a-b` and `a-b:s` spans. A range
expression holds at most 10,000 sub-ranges, so when the chosen numbers of one
sequence need more, the command splits them into halves, again until each
half's range parses back from its text, and writes one entry of the same
pattern per part, in increasing order of number; the list's members are then
the same, in the same order.

A rig sequence becomes one entry per sensor, the sensor's prefix followed by
the frame key's pattern, each with the same range, so the list holds every
sensor's image of each chosen frame. A list orders its members by entry first,
so such a list holds all of the first sensor's images before any of the
second's.

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

The list is written by `NamedImageLists.write_lists`, with `replace` set by
`--force`, which rewrites the lists file whole, keeps every list it already
held in its place and order and with its text, puts a replaced list in the
place of the one it replaces and a new one at the end, and leaves the old file
when a write fails ([named-image-lists.md § Writing the file](../../workspace/named-image-lists.md#writing-the-file)).
Resolved with `-r NAME`, the list gives exactly the chosen frames' images.

## Report

The command prints the workspace, the sequences it collected, a baseline
section per sequence ending in the verdict, a walk section per sequence when
every sequence is a video, and the list it wrote. For one sequence:

```text
Using workspace: /captures/garden
  Feature tool: sfmtool
Given 1 path(s), 6120 image file(s) in 1 sequence:
  frames/walk1_%04d.jpg, 6120 frames (1-6120)

Baseline: 200 frames sampled, 590 frames read (578 extracted, 12 reused)
  Sharpness: median 0.0123; 9 sampled frames blurred
  Motion per frame: median 0.72%, p90 1.31% of the long side; total 4400%
  A video (3 of 3 signs): 1 frame apart step 0.72%; 3 frames apart 2.9 times as far
    neighbouring samples' steps within a factor of 1.3
  Flagged regions: 2210-2271 (blurred), 4381-4442 (weak matches)
Verdict: video

Walk: target step 10% (band 7-13%), 1 break, 2 segments
  Chosen: 512 frames; read: 1307 frames (1250 extracted, 0 reused, 57 from the baseline)
  Motion along the capture (median step, rate per frame, frames per step):
    frames         1-612     9.9%    0.64%     15
    frames      613-1224    10.1%    0.71%     14
    ...
  Blurred frames swapped (23): 1415->1417, 1890->1889, 2224->2226, ... and 15 more
  Frames blurred against their neighbours swapped (6): 377->376, 1066->1068, ... and 1 more
  Gaps filled in (2):
    2210-2290: 14 frames, steps 2.1-5.0%, down to 41 matches
    4380-4410: 9 frames, steps 1.0-3.3%, down to 33 matches
  Kept though blurred (1):
    2244, the sharpest of 2236-2251 (0.31 of the median)
  Breaks (1): 4411->4416: frames 4411-4415 do not match their next frame
  Weak points (2): 4410->4411 (32 matches), 4416->4421 (51 matches)

Extracted and never read: 12 frames

Wrote the named image list walk1 (512 images) to /captures/garden/.sfm-named-image-lists.json
  For a quick test solve, try -r 'walk1[:30]', -r 'walk1[241:271]' or -r 'walk1[-30:]'
Completed in 6m 12.40s
```

Every line is built from the dict `pick_images` returns
([core/analysis/pick-images.md § The report](../../core/analysis/pick-images.md#the-report)),
which names frames by their position in their sequence; the command prints
them as file numbers through the sequences it built. With several sequences,
each baseline and walk heading names its sequence by its first pattern
(`Baseline of walk2/frame_%04d.jpg: …`). A list of more than eight items
(regions, swaps, gaps, frames kept though blurred, weak points) shows the
first eight and the count of the rest.

- **Baseline.** The samples, the frames the baseline read with how many were
  extracted and reused, each sensor's median sharpness (one per sensor prefix
  for a rig) and the count of blurred samples, the rate's median and 0.9
  quantile and the total motion, the three signs of a video with how many
  hold, and the flagged
  regions with what flagged each. A value the binding could not measure, for
  too few matches, prints as `unmeasured`.
- **Verdict.** The last line of the baseline sections, in the shape of the
  `Verdict:` line [`sfm motion`](../reconstruction/motion-command.md) prints.
  `video`; or `not a video:` naming each sequence that is not one with the
  signs it failed, ending "sfm pick-images picks from video sequences only":

  ```text
  Verdict: not a video: photos/DSC_%04d.JPG: steps 1 frame apart median 18.40%, 3 frames apart 1.2 times as far; sfm pick-images picks from video sequences only
  ```

  A sequence of fewer than 4 frames is named as too short to tell a video from
  photographs.
- **Walk.** The target and band, the breaks and segments; the frames chosen
  and the frames the walk used, split into those it extracted, those it
  reused and those the baseline had read; a table of the sequence in ten
  equal runs of positions, each with the median step, rate and stride of the
  steps that start in it, leaving out a run where no step starts; the swaps
  of blurred frames and, on a line of their own, the swaps of frames blurred
  against their neighbours, each as candidate and chosen frame, or `none`;
  each gap filled in with its span, its steps' range and the fewest matches
  of its steps; each frame kept though blurred with the frames searched and its
  sharpness over the median; each break with where the walk resumed, or that
  the sequence ends there; and each weak point,
  a step with its matches, or a short segment with its span.
- **Extracted and never read.** The last line of the report, whatever the
  verdict: how many frames had a `.sift` file written in the run that the walk
  never read, the report's `frames_extracted_unread`. With the `sfmtool` tool
  these are frames prefetched past the end of a break; with
  another tool, the input frames extracted first that the walk did not read.
- **The list.** When a list is written, its name, its number of images and the
  file. A list of more than 60 images that holds no rig gets a line of three
  slices of 30 images to try as quick test solves, in the syntax of
  [named-image-lists.md § Slices](../../workspace/named-image-lists.md#slices):
  the first 30, the 30 from the middle starting at `(n − 30) // 2` for a list
  of `n` images, and the last 30. A smaller list's first and last 30 would
  overlap or be the whole list, and a rig list's slices would take one sensor's
  images, since its entries go one per sensor. When every sequence is a video and
  no name was given, a line gives the count of frames chosen and says to pass
  `--write-list NAME`.

When the verdict stops the run, the command writes no list, and the `.sift`
files of the baseline stay.

### JSON

`--json PATH` writes the dict `pick_images` returns, as it is (`verdict`,
`sequences`, `frames`, `pairs`, `frames_extracted`, per entry of `frames`
whether a `.sift` file was written for it, and `frames_extracted_unread`;
positions, not numbers), with these keys added:

| Key | Holds |
|-----|-------|
| `workspace` | the workspace folder |
| `options` | the five numeric options, under the binding's names |
| `capture` | per sequence: `patterns`, `sensors` (the sensor prefixes, or `null`), `numbers` (the file number at each position) and `images` (each frame's workspace-relative image paths) |
| `chosen_numbers` | per sequence, the chosen frames' file numbers, or `null` when it was not walked |
| `named_image_list` | the written list's `name`, `path`, `images` and `entries`, or `null` |

The file is written whatever the verdict, after the report and the list.
With a tool other than `sfmtool`, `frames_extracted` and
`frames_extracted_unread` are the command's, from the extraction made first.

## Testing

[test_pick_images_command.py](../../../tests/test_pick_images_command.py)
runs the command with click's runner on scratch workspaces. Its video is a
synthetic pan: crops sliding a few pixels per frame across one checked-in
photograph, so it is a video. It covers a report-only run
that writes nothing but `.sift` files and prints the count of frames
extracted and never read; an `opencv` workspace, whose every input frame gets a
`.sift` file before the walk reads them; a written list that `-r NAME` and
`sfm sift -r NAME` resolve to exactly the chosen frames; the JSON file and a
second run that reuses every `.sift` file; the refusal of a defined name before
any image is read, and `--force` replacing a list in place; a list with a
slice as input, and the input list's own name as output only with `--force`;
a range narrowing a folder; the seoul_bull
photographs, which write no list and exit 1 with `--write-list`; a version 1
workspace; no workspace; files with no number or a shared number; a rig whose
reference sensor is listed second, written one entry per sensor; a rig frame
missing a sensor; an invalid output name; and `--force` without `--write-list`
and paths given with a list, which are usage errors.
[test_pick_images_sequences.py](../../../tests/test_pick_images_sequences.py)
checks the grouping on names alone: folders, names, letter case, `%`,
parenthesized numbers, padded and unpadded numbers, the rig order, every
error, the list entries, a range split in parts, and the slice suggestion.

## Non-goals

- Choosing among photographs. A set of separate photographs has no motion a
  step can follow. Files with no number are an error, and a numbered set is
  judged not to be a video by the baseline
  ([core/analysis/pick-images.md § Is it a video?](../../core/analysis/pick-images.md#is-it-a-video)),
  which stops the command. How photographs could be chosen from is an open
  question of [pick-images-open-questions.md](../../drafts/pick-images-open-questions.md).
- Judging whether a capture's frames describe one camera motion through a
  static scene, or whether its camera moved enough to be solved. The command
  fits no geometry and judges only whether each sequence is a video, so a
  camera that only turned, a scene that moved on its own or a camera that
  moved little is walked like any other video
  ([core/analysis/pick-images.md § Non-goals](../../core/analysis/pick-images.md#non-goals)).
- Recording why a frame was chosen in the list or in a `.sfmr`.

The choices of this command that are still open, among them the size of the
baseline, are in
[pick-images-open-questions.md](../../drafts/pick-images-open-questions.md).
