# Picking Images from a Video

A video of a walk around a scene has many more frames than a
Structure-from-Motion solve should use. Neighbouring frames are nearly the same
picture, some frames are blurred, and some stretches of a capture cannot be
solved at all: the camera only turned on the spot, the scene itself moved, or
the camera hardly moved. Picking images looks at a capture before it is solved
and chooses the frames a solve should use: frames spaced so that the image
moves by about a tenth of its width from one to the next, sharp ones in place
of blurred ones, and more frames where the capture is hard to follow. It reads
a sample spread over the whole capture first, decides from it whether each
sequence is a video and whether the capture can be solved, and only then walks
each sequence frame by frame, measuring each step rather than taking a stride
fixed in advance.

Everything is measured from SIFT features. The motion between two frames is
the median displacement of the features that match between them
([frame-pair-matching.md](frame-pair-matching.md)), and a frame's sharpness
and gradient structure tensor come from its `.sift` file's metadata
([sift-file-format.md § The reading image](../../formats/sift-file-format.md#the-reading-image)).
The walk reads only the frames it needs: the sample, the frames it chooses and
the frames it tries on the way. A caller makes each frame's `.sift` files exist
just before the walk reads them, so a frame gets a `.sift` file exactly when it
is read. The [`sfm pick-images`](../../cli/image-feature/pick-images-command.md)
command collects a workspace's frames, prints the report and writes the chosen
frames as a named image list.

## Rust API

The walk lives in
[pick_images/pick.rs](../../../crates/sfmtool-core/src/analysis/pick_images/pick.rs),
with its inputs in
[capture.rs](../../../crates/sfmtool-core/src/analysis/pick_images/capture.rs)
and its report in
[report.rs](../../../crates/sfmtool-core/src/analysis/pick_images/report.rs),
all re-exported from `sfmtool_core::analysis::pick_images` beside the frame
pair matching it uses. It is bound as `sfmtool.analysis.pick_images`, with
`CaptureImage`, `CaptureFrame`, `CaptureSequence` and `FrameId`.

```rust
/// Where each image of a capture is and where its `.sift` file goes.
pub struct CaptureImage { pub image_path: PathBuf, pub sift_path: PathBuf }

/// One instant: one image, or one per rig sensor, the reference sensor first.
pub struct CaptureFrame { pub number: i64, pub images: Vec<CaptureImage> }

/// One sequence, in file-number order, and the pattern each sensor's list
/// entry uses.
pub struct CaptureSequence { pub patterns: Vec<String>, pub frames: Vec<CaptureFrame> }

/// A sequence's index and a frame's position in it, from 0.
pub struct FrameId { pub sequence: usize, pub frame: usize }

pub struct FrameSourceError { pub message: String }

/// Makes sure the `.sift` files of the given frames exist and are current.
pub trait FrameSource {
    fn ensure_sift_files(
        &mut self,
        frames: &[FrameId],
        progress: &Progress,
    ) -> Result<(), FrameSourceError>;
}

pub struct PickImagesOptions {
    pub target_step_percent: f64,  // 10.0, positive
    pub step_band: f64,            // 0.3, in (0, 1)
    pub blur_ratio: f64,           // 0.6, in (0, 1)
    pub baseline_frames: usize,    // 200, at least 2
    pub min_inlier_share: f64,     // 0.5, in [0, 1]
    pub forest: KdForestParams,    // KdForestParams::balanced()
    pub pair: FramePairOptions,    // FramePairOptions::default()
}

pub fn pick_images(
    sequences: &[CaptureSequence],
    source: &mut dyn FrameSource,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<PickImagesReport, PickImagesError>;

pub enum PickImagesError {
    Options { name: String, reason: String },
    Sequence { reason: String },
    Version1Files(Version1SiftFiles),
    ListFolder { folder: PathBuf, source: std::io::Error },
    Source { message: String },
    ReadSift { path: PathBuf, source: SiftError },
    Version1Sift { path: PathBuf },
    Pair { a: FrameId, b: FrameId, images: Box<[PathBuf; 2]>, error: FramePairError },
    Cancelled,
}
```

**Why a source, and why it only makes files exist.** The walk decides which
frame to read next from what it read last, so it cannot be handed its frames in
advance. Asking a caller-supplied source keeps every decision in Rust while
extraction stays where it is, in the Python extraction path that handles all
three feature tools and the workspace's settings (Rust's own
`build_sift_files` writes only the `sfmtool` tool's files). The source is asked
only to make the files exist, not to return features, because the Rust reader
already reads and checks `.sift` files, and the features then never cross the
binding. It takes several frames per call because the baseline's batches and a
blurred run know several frames at once, and extraction in a batch keeps the
extractor's concurrent decodes busy. The walk asks for each frame once and for
no frame it does not then read.

**Positions, not file numbers.** A `FrameId` names a frame by its position in
its sequence, and every rate is a step per position, so a sequence numbered in
steps of 2 has the same rate as one numbered in steps of 1. The report names
frames the same way; `sequences[id.sequence].frames[id.frame].number` is the
file number.

**A verdict is a report, not an error.** A sequence that is not a video, or a
capture that cannot be solved, is what the caller asked to find out, so
`pick_images` returns a report with that verdict and no walk. An error is
something the caller has to fix before the question can be answered.

**The errors.** `Options` names a field out of range (`forest.` and `pair.`
prefix the nested ones, which are checked here, before anything is read, with
the checks `FrameFeatures::new` and `match_frame_pair` make). `Sequence` covers
no sequences, a sequence with no frames, a frame with no images or with another
number of images than its sequence's first frame, a pattern count that differs
from the sensor count, frame numbers not in increasing order, and two images
with one `.sift` path. `Version1Files` and `ListFolder` come from the check of
the folders ([Version 1 files](#version-1-files)). `Source` carries the
source's own message. `ReadSift` is a `.sift` file that cannot be read, or
whose readings or features break the format's rules, such as a negative
sharpness or a feature count that disagrees with the metadata, each wrapped as
the `SiftError::InvalidFormat` a damaged file is. `Pair` is two frames whose
reference images differ in size, the one `FramePairError` the options checks
leave possible. `Cancelled` is a set cancel flag. No caller input panics
([STANDARDS.md § Caller input never panics](../../STANDARDS.md#caller-input-never-panics)).

**A `Progress`.** A walk over a long video runs for minutes, mostly in the
source's extraction. `pick_images` opens the phases `baseline`,
`baseline far pairs` and `walk`, reports the count of frames read after each
call to the source, reports the count of pairs matched as the value
`pairs_matched` after each pair, and checks for cancellation before every call
to the source and every pair it matches
([STANDARDS.md § Long-running functions take a Progress](../../STANDARDS.md#long-running-functions-take-a-progress)).

**Example.**

```rust
use sfmtool_core::analysis::pick_images::{
    pick_images, CaptureSequence, FrameId, FrameSource, FrameSourceError, PickImagesOptions,
    Verdict,
};
use sfmtool_core::progress::Progress;

struct Extract<'a>(&'a [CaptureSequence]);
impl FrameSource for Extract<'_> {
    fn ensure_sift_files(&mut self, frames: &[FrameId], _: &Progress)
        -> Result<(), FrameSourceError> {
        // Extract the .sift file of every image of `frames` that is missing.
        Ok(())
    }
}

let report = pick_images(&sequences, &mut Extract(&sequences),
                         &PickImagesOptions::default(), &Progress::none())?;
if report.verdict == Verdict::Solvable {
    let walk = report.sequences[0].walk.as_ref().unwrap();
    let numbers: Vec<i64> = walk.chosen.iter()
        .map(|&p| sequences[0].frames[p].number).collect();
}
```

### The report

`PickImagesReport` holds the `verdict` (`Solvable`, `NotAVideo` or
`NotSolvable`), one `SequenceReport` per sequence, every frame read
(`FrameReading`: its id, the stage that read it, and each sensor's readings
with the blur judgement made from them) in the order read, and every pair
matched (`MeasuredPair`, with its `FramePairStats`) in the order matched. A
`SequenceReport` holds:

- `baseline`: the sampled positions, the frames the baseline read, each
  sensor's medians, median absolute deviations and the spreads the z-scores
  divide by, the count of blurred
  samples, the rate at each sample, its median and 0.9 quantile, the total
  motion, and the flagged regions with what flagged them;
- `video`: the four signs of a video, the medians behind them, and
  `is_video`;
- `solvability`: the three checks, their medians and `solvable`, or `None`
  when some sequence is not a video;
- `walk`: or `None` unless the verdict is `Solvable`. The chosen positions,
  the segments between breaks as runs of them, every accepted step (from, to,
  its statistics, its rate, its `StepKind` and whether its band was halved in a region
  flagged for weak matches),
  the swaps, the gaps filled in as runs of steps, the frames kept though
  blurred with the range searched, the breaks with where the walk resumed, the
  weak points, and the counts of frames the walk used and of those the
  baseline had already read.

Every type serializes with `serde`, a number that is not finite as `null`, so
a caller writes the whole report as JSON as it is.

## The algorithm

Picking runs in five stages: check the input, read the baseline's first batch
and judge whether each sequence is a video, read its second batch and judge
whether the capture can be solved, walk each sequence, and assemble the
report. The sequences are walked one after another.

### Words

- A **frame** is one instant of the capture: one image, or for a rig the
  images of all its sensors at that instant. A frame is **read** when its
  `.sift` files are read, just after the source has been asked to make them
  exist.
- The **step** between two frames is the median displacement of their mutual
  matches in percent of the image's long side, from the reference sensor's
  images ([frame-pair-matching.md § The step](frame-pair-matching.md#the-step)).
- The **rate** is a step divided by the number of positions between the two
  frames.
- A pair **matches well** when it has at least `pair.min_matches` mutual
  matches and an F-inlier share of at least `min_inlier_share`.
- The **band** is `target × (1 ± step_band)` around the target step: 7% to
  13% at the defaults.

### The baseline

`baseline_frames` is shared among the sequences in proportion to their frame
counts: a sequence gets `round(baseline_frames × len / total)` samples, at
least 2 and at most its frame count. Its samples are spread evenly over it, at
positions `round(i (len − 1) / (n − 1))`, so the first and last frames are
sampled.

The baseline is read in two batches, each requested from the source in one
call. The first reads the sampled frames and the frames 1 and 3 after each
that exist, and matches each sample with them; that is enough to tell a video
from a set of photographs. The second, read only when every sequence is a
video, reads the frames 10 and 30 after each sample and matches each sample
with them. A frame another sample already read is not read again.

From the first batch, per sequence:

- **Each sensor's baseline**: the median sharpness, and the median and median
  absolute deviation of the axis ratio and of the coherence, which
  `ImageReadings::axis_ratio` and `ImageReadings::coherence` derive from the
  gradient structure tensor, over the sampled frames. A flat image, which has
  no axis ratio, is left out of the axis ratio's.
- **The rate at each sample**: the step to the frame 3 after it divided by 3,
  or, when that pair has fewer than `min_matches` matches, the step to the
  next frame; a sample with neither has no rate. The baseline's rate is the
  median of these, and the total motion that median times the frame count.
- **The flagged regions**: a sampled frame that is blurred, or that does not
  match the frame 1 or 3 after it well, flags the frames from the sampled
  frame before it to the sampled frame after it. Overlapping regions are
  merged, keeping what flagged each. Only a region flagged for weak matches
  changes the walk's steps ([The walk](#the-walk)).

### Is it a video?

The walk assumes that neighbouring frames are close in time, so that the
motion between them is small and changes smoothly. A folder of photographs
numbered in the order they were taken does not behave that way. Each sequence
is judged from the samples that have a frame 3 after them, by four signs,
each a median over those samples. A sequence is a video when the step is
small and at least two of the other three signs hold:

- **The motion adds up**: the step 3 frames apart over the step 1 frame apart
  is between 2 and 4.
- **The step is small**: the step 1 frame apart is below 5%.
- **The step changes little**: the median of `|log(step_i / step_{i+1})|` over
  neighbouring samples' 1-frame steps is below `log 2`.
- **Neighbours match closely**: the F-inlier share 1 frame apart is above 0.8.

The small step is required, not one sign among four, because it is what makes
a sequence worth walking: the walk thins a capture by passing over the frames
between steps of the target, and neighbours already half a target step apart
leave nothing to pass over. Photographs taken one after another around a
subject can show the other three signs, since their motion adds up, changes
little from one photograph to the next and leaves neighbours that match
closely, and only the size of the step tells them from a video. The other
three tolerate one failure, so that a video whose camera barely moves, whose
sub-pixel jitter does not add up over 3 frames, is still judged a video and
then stopped as a camera that barely moves.

A sample whose 1-frame or 3-frame pair has fewer than `min_matches` matches
counts against every sign it enters (its step and ratio as infinite, its share
as 0), since photographs that barely overlap give that result. A sequence of
fewer than 4 frames has no 3-frame pair and is not a video. When a sequence is
not a video, nothing more is read and the verdict is `NotAVideo`.

### Can it be solved?

From the second batch, per sequence, each from the median over the samples
whose pair has the value:

- **A moving scene**: the F-inlier share 10 frames apart is below 0.3.
- **A camera that only turns**: H/F 30 frames apart is above 0.97 and the
  rotation error 30 frames apart is below 0.01
  ([frame-pair-matching.md § The rotation error](frame-pair-matching.md#the-rotation-error)).
  A solve of such a capture can look excellent and still be wrong, since
  without translation there is no depth to measure.
- **A camera that barely moves**: the total motion is below 100% of the long
  side.

A check with no sample to take its median from does not hold. When any check
holds for any sequence, nothing more is read and the verdict is
`NotSolvable`.

### The walk

The walk starts at the first frame that matches its next frame well and
repeats from the last chosen frame `f` until it chooses the sequence's last
frame:

1. **Bounds.** When `f` lies in a region flagged for weak matches, the target
   and the band are halved, so the walk goes through the region in short
   steps. Otherwise the step's **limit** is the first frame of the next region
   flagged for weak matches, which a step lands on rather than passes, so a
   problem the walk could step over is still walked through; the limit is
   never past the last frame. A region flagged for blur alone sets no bound:
   the walk swaps a blurred candidate for a sharp neighbour (steps 6 and 7) and
   measures every step, so a blurred stretch whose frames match well is
   crossed at the full target, and one whose frames do not is filled in by the
   steps that fail there.
2. **Predict.** The candidate is `c = f + target / rate`, rounded, at least
   `f + 1` and at most the limit, with the rate the median of the last three
   accepted steps' rates, or at the start of a segment the rate of the nearest
   sample that has one.
3. **Measure** the step `m` from `f` to `c`.
4. **Decide.** Beyond `f + 1`, a pair with fewer than twice `min_matches`
   matches counts here as not matching well: such a step is a weak point,
   where a solve is most likely to split, and the walk takes one only at the
   next frame or when halving a failed step finds nothing better.
   - `m` in the band and the pair matches well: settle on `c`.
   - `m` above the band: re-aim. When `c` is `f + 1`, the motion per frame is
     wider than the band, and `c` is settled on if it matches well.
   - `m` below the band and the pair matches well: settle on `c` when it is
     the limit; otherwise re-aim.
   - The pair does not match well, in or below the band, or above it at
     `f + 1`: the step fails.
5. **Re-aim** to `c' = f + (c − f) · target / m`, kept strictly between the
   largest candidate measured below the band and the smallest measured above
   it. When no frame is left between them, they lie one frame apart on either
   side of the band, and the one below is settled on. After four re-aims
   without reaching the band, the step fails.
6. **Settle** on a frame that is not blurred by accepting it. A blurred one is
   swapped: the walk reads its neighbours nearest first (`c − 1`, `c + 1`,
   `c − 2`, …) among the frames whose step, at the candidate's measured rate,
   would be in the band, and accepts the first that is sharp, matches well,
   and has a step no longer than the band and no shorter than the band or the
   candidate's own step. The swap is reported.
7. **A blurred run.** When no neighbour qualifies, the walk reads every frame
   from `f + 1` to the band's upper edge at the candidate's rate in one batch,
   and accepts the farthest sharp one that matches well within the band's
   upper edge, which may be a step below the band. When no sharp frame there
   does, it accepts the sharpest of the frames whose step matches well within
   the band, the sharpest being the frame whose least sharp sensor has the
   highest ratio to its median, and reports it as kept though blurred. A
   stretch where every frame is blurred is so crossed at the band's spacing,
   with the least blurred frame at each step, and the chosen frames are no
   denser there than the band makes them elsewhere; taking the sharpest of all
   the frames up to the upper edge would let the next frame win and walk the
   stretch one frame at a time. Only when no frame in the band matches well
   does it take the sharpest that matches well below the band. The next frame
   always qualifies when it matches well, and the candidate itself matched
   well, so the search always accepts a frame.

**A failing step.** When a step fails, the largest candidate measured below the
band that matched well is accepted, since halving would reach it. When there
is none, the walk halves the gap: it measures the frame halfway between `f`
and the failed candidate and accepts it if it matches well, even though its
step is below the band; otherwise it halves again, down to `f + 1`. A blurred
frame accepted this way is swapped for its sharp neighbour on either side
inside the gap when one matches well, and otherwise kept though blurred. Each
run of consecutive steps accepted this way is reported as a gap filled in, so
the chosen frames become dense across a bad patch and stay sparse where
nothing goes wrong.

**A break.** When even the step from `f` to `f + 1` does not match well, the
walk records a break after `f`, reads forward one frame at a time to the first
frame that matches its next frame well, and starts a new segment there with no
recent rates. The frames in between are read and chosen for nothing. When no
such frame remains, the sequence ends at the break.

**Weak points** are where a solve is most likely to split: an accepted step
with fewer than twice `min_matches` matches, or with an F-inlier share within
0.1 of `min_inlier_share`, and a segment of fewer than 3 chosen frames.

**What is never read.** Outside regions flagged for weak matches, a frame the
walk does not read lies between two chosen frames whose step matched well.
Inside such a region the halved steps still pass over frames, which are then
not read either. Every frame of a break is read.

### Blur

A frame's sensor image is **blurred** when its sharpness is below
`blur_ratio` times its sensor's median sharpness, or when its axis ratio or its
coherence has a robust z-score above 2 against its sensor's baseline:
`z = (x − median) / (1.4826 · MAD)`, with the median and the median absolute
deviation from `sfmtool_core::numeric::median` over the sampled frames. Only a
z above 2 counts, one-sided, since motion blur raises both readings by
weakening the gradients along the direction of motion. The sharpness ratio
catches a frame blurred in every direction; the z-scores catch one blurred
along one direction that the sharpness alone misses.

**A MAD of 0** happens when more than half the sampled values equal the median
exactly, as when frames repeat or a reading does not vary. Dividing by it
would make any value above the median, however close, a blur, and so flag the
other part of the capture. The spread is then `1.4826` times the smallest
positive deviation from the median instead: the finest difference the
capture's own readings show, so that a value counts as blurred only when it is
more than twice that far beyond the median, and the readings that make up the
other part, which deviate by about that much, do not. A fraction of the median
would not serve as the floor, since the median axis ratio or coherence of
images with gradients equally strong in every direction is near 0. When every
sampled value equals the median there is no spread to judge against, the
z-score is not made (`None`), and only the sharpness ratio judges that
reading.

### Rigs

A rig frame is read, judged and chosen with all its sensors, and reading it
reads every sensor's `.sift` metadata. Its motion is measured from the
reference sensor's images only, the first of each frame, since their
displacement and inlier shares describe the rig's motion as well as any
sensor's and the sensors' images need not be the same size. Its sharpness is
judged per sensor against that sensor's own baseline, since the sensors look
at different parts of the scene; a rig frame is blurred when any of its
sensors is.

## Version 1 files

Every reading comes from a version 2 `.sift` file, which records the sharpness
and gradient structure tensor a version 1 file lacks
([workspace.md § Versions never mix](../../workspace/workspace.md#versions-never-mix)).
Before it reads anything, `pick_images` lists every folder the capture's
`.sift` files go in with `version_1_sift_files`
([features/sift/files.rs](../../../crates/sfmtool-core/src/features/sift/files.rs)),
the check `build_sift_files` and, through its binding, the Python extraction
make before they write into a folder, so the three agree on what counts. A
folder that holds version 1 files is `Version1Files`, whose message names the
folder and the files and says to remove them and run again, which then
extracts version 2 files in their place. A folder that is there but cannot be
listed is `ListFolder`. A `.sift` file read that is still version 1, one put
in its folder after the check, is `Version1Sift`, with the same advice.

## Implementation notes

- **One forest per frame, held only while it has a pair to match.** A frame's
  features are read from its `.sift` file and its forest built
  (`FrameFeatures::from_sift`) when a pair needs it. The baseline counts each
  frame's pairs and drops its features after its last, matching the pairs in
  sample order, so only the frames near the current sample are held. The walk
  holds the last chosen frame's features and the last candidate's, which
  become the chosen frame's when it is accepted. A frame's readings, a few
  numbers per sensor, are kept for every frame read. A pair's statistics are
  kept, so a pair the baseline matched is not matched again by the walk.
- **The source sees each frame once.** Every frame passes through one reading
  step that asks the source for the frames not yet read and reads all their
  `.sift` metadata at once, which is what makes the report's frame list the
  exact list of frames the source was asked for, in order.
- **No duplicate derivations.** The axis ratio and coherence are
  `ImageReadings`' own, the statistics of a pair `match_frame_pair`'s, the
  medians, MADs and the rate's 0.9 quantile `sfmtool_core::numeric`'s.

## Determinism and precision

The report is the same for the same capture and options: the frame pair
statistics repeat exactly ([frame-pair-matching.md § Determinism and precision](frame-pair-matching.md#determinism-and-precision)),
every decision reads them and the readings in a fixed order, and no hash map is
iterated for an output. The readings are kept as `f32`, as `ImageReadings`
holds them, which is enough for readings compared within a capture to about
three digits; the medians, ratios and z-scores are `f64` because
`sfmtool_core::numeric` takes `f64`, and widening is exact.

## Parameters

| Value | Default | Meaning |
|-------|---------|---------|
| `target_step_percent` | `10` | the step aimed for, percent of the long side |
| `step_band` | `0.3` | the band, `target × (1 ± step_band)` |
| `blur_ratio` | `0.6` | below this times the median sharpness is blurred |
| `baseline_frames` | `200` | the samples, shared among the sequences |
| `min_inlier_share` | `0.5` | the F-inlier share a pair that matches well has |
| `pair.min_matches` | `30` | the matches a pair that matches well has |
| baseline gaps | 1, 3; 10, 30 | the frames after each sample the two batches read |
| robust z | 2 | above this, an axis ratio or coherence is blurred |
| video signs | 2 to 4; 5%; factor 2; 0.8; the small step and 2 of the other 3 | the four signs and which make a video |
| moving scene | 0.3 | the F-inlier share 10 frames apart below which the scene moves |
| panorama | 0.97, 0.01 | the H/F and rotation error 30 frames apart |
| barely moves | 100% | the total motion below which the camera barely moves |
| re-aims | 4 | before a step fails |
| recent rates | 3 | the accepted steps the prediction takes the median of |
| weak point | 2 × `min_matches`; `min_inlier_share` + 0.1; 3 frames | a weak step and a short segment |

The last nine are constants of the module.

## Python bindings

The bindings are in
[analysis/pick_images.rs](../../../crates/sfmtool-py/src/analysis/pick_images.rs).

```python
from sfmtool.analysis import CaptureFrame, CaptureImage, CaptureSequence, pick_images

sequence = CaptureSequence(
    ["frames/walk1_%04d.jpg"],
    [CaptureFrame(n, [CaptureImage(f"frames/walk1_{n:04d}.jpg",
                                   f"frames/features/x/walk1_{n:04d}.jpg.sift")])
     for n in range(1, 6121)],
)

def ensure_sift_files(frames):  # a list of FrameId
    paths = [sequence.frames[f.frame].images[0].image_path for f in frames]
    image_files_to_sift_files(paths, ...)

report = pick_images([sequence], ensure_sift_files, target_step_percent=10.0)
report["verdict"]                         # "solvable"
report["sequences"][0]["walk"]["chosen"]  # positions of the chosen frames
```

`CaptureImage`, `CaptureFrame` and `CaptureSequence` mirror the Rust structs;
a frame's `images` and a sequence's `frames` are tuples, built once, so
`sequence.frames[f.frame]` costs nothing per call. `FrameId` has `sequence`
and `frame`, compares and hashes by them, and is what the callable is given, as
a list. `pick_images` takes the fields of `PickImagesOptions` as keyword-only
arguments, with the pair options under the names `match_frame_pair` uses and
the forest under the names `FrameFeatures` uses (`preset`, `num_trees`,
`leaf_size`, `max_leaf_checks`); one left out takes the Rust default.
`progress`, a `ProgressCounter`, holds the count of frames read. The report is
returned as a dict of plain values, the report's serde form, with `None` for a
number that is not finite.

The GIL is released while the walk runs and taken for each call of the
callable, whose exception is raised unchanged. Each progress event, so each
pair matched and each batch read, also takes it to run Python's signal
handlers; a handler that raises, such as the `KeyboardInterrupt` of Ctrl+C,
cancels the walk, and its exception is raised, so Ctrl+C stops a walk that is
matching as it stops one that is extracting. A cancellation from anywhere else
is `InterruptedError`. A `.sift` file that cannot be read is the `OSError` of
[python-bindings.md § Format errors](../../python-bindings.md#format-errors),
a folder that cannot be listed the `OSError` subclass for its kind, a callable
that is not callable a `TypeError`, and every other error a `ValueError` with
the Rust message.

## Testing

The Rust tests in
[pick_images/tests.rs](../../../crates/sfmtool-core/src/analysis/pick_images/tests.rs)
walk synthetic captures
([tests/capture.rs](../../../crates/sfmtool-core/src/analysis/pick_images/tests/capture.rs)):
a scene of points with random descriptors seen by a pinhole camera along a
path with known motion per frame, with readings set per frame, served by a
source that writes a frame's version 2 `.sift` files only when it is asked for
them and records every request. They cover a steady video chosen at about the
target step, a change of speed that the stride follows, a blurred frame (in
every direction and along one) swapped for a neighbour, a blurred run kept at
its sharpest frames within the band, a long blurred stretch, flagged for blur
alone, crossed with every step in the full band, a spread of 0 floored at the
smallest deviation and readings mostly at the median flagging none of the
rest, a bad patch flagged for weak matches and filled in at halved steps, two
breaks with a short segment and steps with few matches taken only at the next
frame or in a gap filled in, shuffled frames, frames 6% apart that show the
other three signs and a three-frame sequence that are not a video, a camera
that only turns, one that barely moves and a moving scene that are not
solvable, a rig whose second sensor's images match nothing
and whose blur alone swaps a frame, several sequences sharing the baseline,
every error variant, cancellation before and during the run, and that a run
repeats exactly. Every test checks that the frames the source was asked for are
the frames the report lists as read, each once, and that exactly those have
`.sift` files.

[test_pick_images_walk_rust_bindings.py](../../../tests/rust_bindings/analysis/test_pick_images_walk_rust_bindings.py)
checks the bindings on a synthetic capture written by a Python callable, the
options and their errors, a callable that raises, a `KeyboardInterrupt`
delivered while the walk matches, the exceptions of each error, the capture
classes, and the seoul_bull photographs extracted through the workspace's own
`image_files_to_sift_files`, which are judged not a video.

## Non-goals

- Collecting a workspace's frames into sequences, printing the report and
  writing the chosen frames as a named image list, which the
  [`sfm pick-images`](../../cli/image-feature/pick-images-command.md) command
  does. The choices still open are in
  [pick-images-open-questions.md](../../drafts/pick-images-open-questions.md).
- Choosing among photographs. A sequence that is not a video stops the run;
  nothing chooses frames from it.
- Choosing frames for a rotation-only solve. A capture whose camera only turns
  is not solvable here.
