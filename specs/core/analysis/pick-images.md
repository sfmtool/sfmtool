# Picking Images from a Video

A video of a walk around a scene has many more frames than a
Structure-from-Motion solve should use. Neighbouring frames are nearly the same
picture, some frames are blurred, and a capture whose camera hardly moved
cannot be solved at all. Picking images looks at a capture before it is solved
and chooses the frames a solve should use: frames spaced so that the image
moves by about a tenth of its width from one to the next, sharp ones in place
of blurred ones, and more frames where the capture is hard to follow. It reads
a sample spread over the whole capture first, decides from it whether each
sequence is a video and whether its camera moves enough to be solved, and only
then walks each sequence frame by frame, measuring each step rather than
taking a stride fixed in advance. It fits no geometry: whether the frames it
chooses describe one camera motion through a static scene is for the solve to
find out.

Everything is measured from SIFT features. The motion between two frames is
the median displacement of the features that match between them, and how well
they match is how many mutual matches they have
([frame-pair-matching.md](frame-pair-matching.md)); a frame's sharpness
and gradient structure tensor come from its `.sift` file's metadata
([sift-file-format.md § The reading image](../../formats/sift-file-format.md#the-reading-image)).
The walk reads only the frames it needs: the sample, the frames it chooses and
the frames it tries on the way. A source makes each frame's `.sift` files
exist: the extracting source extracts them on a pool of workers that runs
ahead of the walk, on the frames the walk predicts it will read, so a frame
gets a `.sift` file when the walk reads it or predicts it. The walk decides
only from frames it waited for, so the work done ahead changes how long a run
takes and never what it chooses. The [`sfm pick-images`](../../cli/image-feature/pick-images-command.md)
command collects a workspace's frames, prints the report and writes the chosen
frames as a named image list.

## Rust API

The walk lives in
[pick_images/pick.rs](../../../crates/sfmtool-core/src/analysis/pick_images/pick.rs),
with its inputs in
[capture.rs](../../../crates/sfmtool-core/src/analysis/pick_images/capture.rs),
its sources in
[source.rs](../../../crates/sfmtool-core/src/analysis/pick_images/source.rs)
and its report in
[report.rs](../../../crates/sfmtool-core/src/analysis/pick_images/report.rs),
all re-exported from `sfmtool_core::analysis::pick_images` beside the frame
pair matching it uses. It is bound as `sfmtool.analysis.pick_images`, with
`CaptureImage`, `CaptureFrame` and `CaptureSequence`.

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

/// Makes the `.sift` files of a capture's frames exist.
pub trait FrameSource {
    /// How many frames it works on at once; 0 when it does no work ahead.
    fn concurrency(&self) -> usize;
    /// Start on these frames, in place of the last prediction; return at once.
    fn prefetch(&mut self, frames: &[(FrameId, &CaptureFrame)]);
    /// Block until every image of these frames has its `.sift` file.
    fn wait(&mut self, frames: &[(FrameId, &CaptureFrame)], progress: &Progress)
        -> Result<(), FrameSourceError>;
    /// Stop the work no wait asked for; the frames it wrote files for, sorted.
    fn finish(&mut self) -> Vec<FrameId>;
}

/// Extracts missing `.sift` files with the `sfmtool` tool, as `build_sift_files` does.
pub struct ExtractingSource { /* … */ }
impl ExtractingSource {
    pub fn new(options: BuildSiftFilesOptions) -> Result<Self, BuildSiftFilesError>;
}

/// Reads `.sift` files that already exist; a missing one is an error.
pub struct ReadOnlySource;

pub enum FrameSourceError {
    Missing { frame: FrameId, path: PathBuf },
    SiftPath { frame: FrameId, sift_path: PathBuf, expected: PathBuf },
    Extract { frame: FrameId, error: BuildSiftFilesError },
    Panicked { frame: FrameId, message: String },
    Cancelled,
}

pub struct PickImagesOptions {
    pub target_step_percent: f64,  // 10.0, positive
    pub step_band: f64,            // 0.3, in (0, 1)
    pub blur_ratio: f64,           // 0.6, in (0, 1)
    pub neighbour_blur_ratio: f64, // 0.85, in (0, 1)
    pub baseline_frames: usize,    // 200, at least 2
    pub min_matches: usize,        // 30, at least 1
    pub forest: KdForestParams,    // KdForestParams::balanced()
    pub pair: FramePairOptions,    // FramePairOptions::default()
    pub prefetch: bool,            // true; changes no choice, only the work ahead
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
    Source(FrameSourceError),
    ReadSift { path: PathBuf, source: SiftError },
    Version1Sift { path: PathBuf },
    Pair { a: FrameId, b: FrameId, images: Box<[PathBuf; 2]>, error: FramePairError },
    Cancelled,
}
```

**Why a source that prefetches and waits.** The walk decides which frame to
read next from what it read last, so it cannot be handed its frames in
advance, and extraction, not matching, is most of a cold run's time. A source
that is only asked for a frame when the walk needs it leaves the extractor
idle while the walk matches and the walk idle while one frame is extracted.
So the walk tells the source what it knows it will read
([Run-ahead](#run-ahead)) with `prefetch`, which starts the work and returns
at once, and blocks in `wait` only for the frames it is about to read. The
source is asked only to make the files exist, not to return features: the
files are on disk, the reader already reads and checks them, and what stays
in memory is bounded ([Sources](#sources)). `finish` stops the work no wait
asked for, so no worker goes on after a run, and says which frames the source
wrote files for, which the report counts. The trait is public so a test, or
another caller, can supply its own; the two sources here are the extracting
one, for a workspace whose feature tool is `sfmtool`, and the read-only one,
for files another tool already extracted.

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
the folders ([Version 1 files](#version-1-files)). `Source` is a frame the
walk waited for that the source could not make exist, naming the frame: a
missing file for the read-only source, and for the extracting source an image
that cannot be read or a file that cannot be written (`Extract`, with the
`BuildSiftFilesError::Image` or `Write` that names the path), a `sift_path`
other than where the extraction writes it (`SiftPath`), or a worker that
panicked, which is a bug and is reported rather than left to hang the wait
(`Panicked`). A frame that fails but is never waited for is no error, since
whether a worker reached it depends on timing. `ReadSift` is a `.sift` file that cannot be read, or
whose readings or features break the format's rules, such as a negative
sharpness or a feature count that disagrees with the metadata, each wrapped as
the `SiftError::InvalidFormat` a damaged file is. `Pair` is two frames whose
reference images differ in size, the one `FramePairError` the options checks
leave possible. `Cancelled` is a set cancel flag. No caller input panics
([STANDARDS.md § Caller input never panics](../../STANDARDS.md#caller-input-never-panics)).

**A `Progress`.** A walk over a long video runs for minutes, mostly in
extraction. `pick_images` opens the phases `baseline` and `walk`, reports
the count of frames read after each wait on the source,
reports the count of pairs matched as the value `pairs_matched` after each
pair, and checks for cancellation before every wait and every pair it
matches. A wait that blocks reads the cancel flag and reports the value
`source_frames_ended` every 50 ms, so a cancellation stops it within that
time and a caller whose sink checks for a signal, as the binding's does, sees
one while the workers extract
([STANDARDS.md § Long-running functions take a Progress](../../STANDARDS.md#long-running-functions-take-a-progress)).
The workers run on threads of their own, which cannot hold the caller's
`Progress`; each runs its job under a `Progress` whose cancel flag is the
pool's stop flag, which a cancelled wait and `finish` set, so a worker stops
at the extraction's next check, before its extraction or its write.

**Example.**

```rust
use sfmtool_core::analysis::pick_images::{
    pick_images, ExtractingSource, PickImagesOptions, Verdict,
};
use sfmtool_core::features::sift::BuildSiftFilesOptions;
use sfmtool_core::progress::Progress;

// The workspace's feature_options and feature_prefix_dir.
let mut source = ExtractingSource::new(BuildSiftFilesOptions::sfmtool_defaults())?;
let report = pick_images(&sequences, &mut source,
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
with the blur judgement made from them) in the order read, every pair
matched (`MeasuredPair`, with its `FramePairStats`) in the order matched,
`frames_extracted`, per frame read, whether the source wrote its `.sift`
files in this run rather than finding them up to date, and
`frames_extracted_unread`, how many frames the source wrote files for that
the walk never read, the predictions it did not follow. That count is the one
part of the report that depends on how far ahead the walk predicts and how
fast the source works. A `SequenceReport` holds:

- `baseline`: the sampled positions, the frames the baseline read, each
  sensor's median sharpness, the count of blurred samples, the rate at each
  sample, its median and 0.9 quantile, the total motion, and the flagged
  regions with what flagged them;
- `video`: the three signs of a video, the medians behind them, and
  `is_video`;
- `solvability`: the total motion, whether the camera barely moves and
  `solvable`, or `None` when some sequence is not a video;
- `walk`: or `None` unless the verdict is `Solvable`. The chosen positions,
  the segments between breaks as runs of them, every accepted step (from, to,
  its statistics, its rate, its `StepKind` and whether its band was halved in a region
  flagged for weak matches),
  the swaps, each with its `SwapReason` (`Median` for a blurred candidate,
  `Neighbours` for one blurred against its neighbours), the gaps filled in as
  runs of steps, the frames kept though blurred with the range searched, the
  breaks with where the walk resumed, the weak points, and the counts of
  frames the walk used and of those the baseline had already read.

Every type serializes with `serde`, a number that is not finite as `null`, so
a caller writes the whole report as JSON as it is.

## The algorithm

Picking runs in five stages: check the input, read the baseline, judge whether
each sequence is a video and whether its camera moves enough to be solved,
walk each sequence, and assemble the report. The sequences are walked one
after another.

### Words

- A **frame** is one instant of the capture: one image, or for a rig the
  images of all its sensors at that instant. A frame is **read** when its
  `.sift` files are read, just after the walk has waited on the source for
  it. It is **predicted** when the walk asks the source to prefetch it.
- The **step** between two frames is the median displacement of their mutual
  matches in percent of the image's long side, from the reference sensor's
  images ([frame-pair-matching.md § The step](frame-pair-matching.md#the-step)).
- The **rate** is a step divided by the number of positions between the two
  frames.
- A pair **matches well** when it has at least `min_matches` mutual matches.
- The **band** is `target × (1 ± step_band)` around the target step: 7% to
  13% at the defaults.

### The baseline

`baseline_frames` is shared among the sequences in proportion to their frame
counts: a sequence gets `round(baseline_frames × len / total)` samples, at
least 2 and at most its frame count. Its samples are spread evenly over it, at
positions `round(i (len − 1) / (n − 1))`, so the first and last frames are
sampled.

The default of 200 samples is sized for what the command is for, subsampling
long captures of 10,000 to 100,000 frames, where the baseline reads a small
share of the frames. On a short capture of a few hundred frames the samples
and the frames read after them are most of the capture, which is accepted:
fewer samples would put neighbouring samples further apart, and the sign that
the step changes little compares neighbouring samples' steps
([Is it a video?](#is-it-a-video)).

The baseline reads the sampled frames and the frames 1 and 3 after each that
exist, and matches each sample with them; that is enough to tell a video from
a set of photographs and to measure the rate. It is prefetched whole, then
waited for and matched sample by sample in sample order, so that matching one
sample's pairs runs while the source works on the next ones; the frames are
read, and the pairs matched, in the baseline's order. A frame another sample
already read is not read again.

From the baseline, per sequence:

- **Each sensor's baseline**: the median sharpness over the sampled frames.
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
is judged from the samples that have a frame 3 after them, by three signs,
each a median over those samples. A sequence is a video when all three hold:

- **The step is small**: the step 1 frame apart is below 5%.
- **The motion adds up**: the step 3 frames apart over the step 1 frame apart
  is between 2 and 4.
- **The step changes little**: the median of `|log(step_i / step_{i+1})|` over
  neighbouring samples' 1-frame steps is below `log 2`.

The small step is what makes a sequence worth walking: the walk thins a
capture by passing over the frames between steps of the target, and
neighbours already half a target step apart leave nothing to pass over.
Photographs taken one after another around a subject can show the other two
signs, since their motion adds up and changes little from one photograph to
the next, and only the size of the step tells them from a video. A video
whose camera barely moves, whose sub-pixel jitter does not add up over 3
frames, is judged not a video; one whose motion does add up is judged a video
and then stopped as a camera that barely moves.

A sample whose 1-frame or 3-frame pair has fewer than `min_matches` matches
counts against every sign it enters (its step and ratio as infinite), since
photographs that barely overlap give that result. A sequence of fewer than 4
frames has no 3-frame pair and is not a video. When a sequence is not a video,
nothing more is read and the verdict is `NotAVideo`.

### Does the camera move enough?

From the baseline, per sequence: **the camera barely moves** when the total
motion, the baseline's median rate times the frame count, is below 100% of
the long side, so the whole capture moves the image by less than its own
width. When it holds for any sequence, nothing more is read and the verdict
is `NotSolvable`.

Nothing else about solvability is judged. Whether the matches describe one
camera motion through a static scene, as they do not for a camera that only
turns or a scene that moves on its own, is not measured: the solve that
follows finds out.

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
4. **Decide.** Here and in steps 6 and 7, a pair beyond `f + 1` with fewer
   than twice `min_matches` matches counts as not matching well: such a step
   is a weak point, where a solve is most likely to split, and the walk takes
   one only at the next frame or when halving a failed step finds nothing
   better.
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
6. **Settle** on a frame that is neither blurred nor blurred against its
   neighbours ([Blur](#blur)) by accepting it. A frame blurred against its
   neighbours, which the walk has read for that judgement, is swapped for
   one of them: the walk tries, sharpest first, the neighbours the candidate
   is blurred against that lie after `f` and within the limit, are not
   blurred and are not themselves blurred against the candidate on any
   sensor, and accepts the first whose step matches well and is no longer
   than the band and no shorter than the band or the candidate's own step.
   The sharpest is the neighbour whose sharpness over the candidate's, the
   least over the sensors, is highest; ties go to the nearer, then the
   earlier. The swap is reported with the reason `Neighbours`. When no
   neighbour qualifies, the candidate is accepted. A blurred frame is
   swapped: the walk reads its neighbours nearest first (`c − 1`, `c + 1`,
   `c − 2`, …) among the frames whose step, at the candidate's measured rate,
   would be in the band, and accepts the first that is sharp, matches well
   (a neighbour beyond `f + 1` with few matches does not), and has a step no
   longer than the band and no shorter than the band or the candidate's own
   step. The swap is reported with the reason `Median`.
7. **A blurred run.** When no neighbour qualifies, the walk reads every frame
   from `f + 1` to the band's upper edge at the candidate's rate in one batch,
   and accepts the farthest sharp one that matches well, by the same rule,
   within the band's upper edge, which may be a step below the band. When no sharp frame there
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
with fewer than twice `min_matches` matches, and a segment of fewer than 3
chosen frames.

**What is never read.** Outside regions flagged for weak matches, a frame the
walk does not read lies between two chosen frames whose step matched well.
Inside such a region the halved steps still pass over frames, which are then
not read either. Every frame of a break is read. A frame within 2 positions
of a candidate judged against its neighbours is read, and may be measured
only if it is tried in the candidate's place. A frame the walk predicted
and did not read may still have been extracted ([Run-ahead](#run-ahead)).

### Blur

A frame's sensor image is **blurred** when its sharpness is below
`blur_ratio` times its sensor's median sharpness over the sampled frames, the
median from `sfmtool_core::numeric::median`. Sharpness compared within one
sequence tracks blur, since its frames show the same scene through the same
lens, so a frame much less sharp than the median is most often one whose
detail was smeared; a frame that shows mostly a plain surface reads low as
well.

A frame the walk is about to accept in step 6 is also judged against its
**neighbours**, the frames within 2 positions of it on either side. Sharpness
depends on what a frame shows as much as on how sharp it is, so a frame
compared with the median of a whole sequence is judged against frames of
other parts of the scene, and much of the blur in a capture leaves a frame
above `blur_ratio` times the median. A frame a few positions away shows
nearly the same scene, so the less sharp of two such frames is the blurrier
one. A sensor image is **blurred against its neighbours** when its sharpness
is below `neighbour_blur_ratio` times the sharpest neighbour's on that sensor,
and a rig frame when any of its sensors' images is. Only a sensor image whose
sharpness is below its sensor's median is judged this way: a frame sharper
than the median on every sensor is never judged against its neighbours, and
the walk does not read them for it, which keeps the reads to the frames that
could be blurred. Otherwise the walk reads all the neighbours in one wait,
before the swap of step 6 ([The walk](#the-walk)) looks for a sharper one
among them. A frame accepted in a blurred run (step 7) or for a failing step
([A failing step](#the-walk)) is judged against the median alone: the blurred
run has already searched every frame up to the band's upper edge for a sharp
one, and a frame accepted for a failing step is taken for matching where the
steps around it fail.

The comparison is with the sharpest neighbour, not their median, because
blur comes in runs: within a few positions of a blurred frame several
neighbours are often blurred too, their median is then a blurred frame, and
the sharpest is also the frame a swap would take. For the same reason a
stretch whose frames are all equally blurred is not judged blurred against
its neighbours, which share the blur; the comparison with the median, which
the walk keeps, is what finds such a stretch.

The ratio errs towards judging a frame blurred against its neighbours. A
swap moves only to a neighbouring frame that is sharper on a judged sensor
by the ratio and passes the match and band checks, so judging a sharp frame
blurred against its neighbours costs little, while a higher ratio swaps more
of the frames that are blurrier than a neighbour.

The report also carries each image's axis ratio and coherence, which
`ImageReadings::axis_ratio` and `ImageReadings::coherence` derive from the
gradient structure tensor, but blur is not judged from them. They measure the
direction of the gradients, which the scene sets as strongly as motion blur
does: a fisheye's geometry, a slope seen at an angle or a subject made of
lines gives a sharp frame strongly directional gradients, so against a
capture-wide median those readings would call such sharp frames blurred. A
frame smeared along one direction whose sharpness stays near the median is so
not judged blurred.

### Rigs

A rig frame is read, judged and chosen with all its sensors, and reading it
reads every sensor's `.sift` metadata. Its motion is measured from the
reference sensor's images only, the first of each frame, since their
displacement describes the rig's motion as well as any
sensor's and the sensors' images need not be the same size. Its sharpness is
judged per sensor against that sensor's own baseline, since the sensors look
at different parts of the scene, and against each sensor's own images of
its neighbouring frames; a rig frame is blurred, or blurred against its
neighbours, when any of its sensors is.

## Sources

The extracting source is `sfm sift --extract`'s extraction for the `sfmtool`
feature tool, frame by frame: each image's file is the one `build_sift_files`
writes, at `BuildSiftFilesOptions::sift_path_for_image`, with the same
content, skipped when it is at least as new as its image, and kept on disk
([features/sift.md § Building `.sift` files](../features/sift.md#building-sift-files)).
`ExtractingSource::new` refuses options that describe no extraction and
version 1 settings, as `build_sift_files` does, and `pick_images` has checked
the folders for version 1 files before the first frame. An image whose
`sift_path` is not where the extraction writes it is `SiftPath`, since the
walk would otherwise read a file other than the one written.

Its workers follow `build_sift_files`: there are `concurrent_images` of them,
or by default the `SFMTOOL_SIFT_EXTRACT_WORKERS` environment variable or the
core count capped so that the images in flight fit in half of physical memory
at the first image's size, and each decodes, hashes and makes the thumbnail
of its frame's images on its own thread, then extracts and writes on the
rayon pool, which keeps one thread per core. They take the frames a wait is
blocked on first, in its order, then the last prefetch's in its order. A
prefetch replaces the one before: its frames not yet started are dropped
unless the new one lists them too, and a frame started, written or failed is
not started again. A wait puts its frames not yet started ahead of every
prefetched one, and returns the error of the first of its frames, in its
order, that failed.

**Memory.** What stays resident does not grow with the frames extracted:
the files are on disk, and the walk reads a frame's features back from its
file when a pair needs them, holding only the few forests of the pairs near
the current one ([Implementation notes](#implementation-notes)). The pool
holds a state of a few bytes per frame asked for, the error of a frame that
failed, and the image paths of the frames queued, which are at most one
prefetch's (the baseline, at most three frames per sample, or the
`W` frames ahead through a break); each worker holds the one frame it extracts, as many as
`build_sift_files` holds in flight under the same memory cap.

**Cancellation.** A wait that sees the cancel flag sets the pool's stop flag,
drops the queued work, joins the workers and returns `Cancelled`. A worker
stops at the extraction's next check, before its extraction or its write, so
a cancelled run leaves only whole files. `finish` stops the workers the same
way at the end of every run, with a report or an error. A frame counts as
written when any of its images' files was, however its work ended, so a rig
frame stopped or failed between its images counts once among the frames
written, and in `frames_extracted_unread` when the walk never reads it; a
later wait for it extracts only the images still missing. The count is of
frames, as the report's frame list is, not of files.

The read-only source does nothing ahead: its `concurrency` is 0, so the walk
predicts nothing, and its wait checks that each image's `.sift` file exists,
naming the first that does not (`Missing`). A caller whose feature tool is not
`sfmtool` extracts the files first and walks with it.

## Run-ahead

The walk prefetches the frames it knows it will read, so that the source's
workers extract them while it matches the frames already read.

- **The baseline.** It is one prefetch of all its frames, which the
  workers extract while the walk waits for and matches them sample by
  sample.
- **A break.** Once a pair the walk reads forward through does not match
  well, it reads every following frame until one matches its next frame, so
  after each such pair it prefetches the next `W` frames, for a source of `W`
  workers. At most `W` frames past the end of a break are extracted and not
  read.
- **The steps.** A step's candidates are not prefetched: its second candidate
  is aimed from what its first measured, and a frame prefetched on a guess
  takes a worker and cores from the frame the walk is waiting for, so a step
  waits for each frame it measures, with every worker free for it.

**The determinism rule.** The walk decides only from frames it explicitly
waited for, never from what happens to be ready: every decision reads the
readings and pairs of frames read, and a frame is read only after a wait on
it returns. A prediction only starts work. So the frames waited for, the
frames chosen and every decision in the report are the same with
`prefetch` off or on and for any number of workers; only
`frames_extracted_unread` differs.

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
- **Each frame is waited for once.** Every frame passes through one reading
  step that waits on the source for the frames not yet read and reads all
  their `.sift` metadata, which is what makes the report's frame list the
  exact list of frames waited for, in order. The prefetch step only passes on
  the frames not yet read.
- **No duplicate derivations.** The axis ratio and coherence are
  `ImageReadings`' own, the statistics of a pair `match_frame_pair`'s, the
  medians and the rate's 0.9 quantile `sfmtool_core::numeric`'s.

## Determinism and precision

The report is the same for the same capture and options: the frame pair
statistics repeat exactly ([frame-pair-matching.md § Determinism and precision](frame-pair-matching.md#determinism-and-precision)),
every decision reads them and the readings in a fixed order, and no hash map is
iterated for an output. It is the same for any source, lookahead and worker
count too, but for `frames_extracted_unread`, by the
[determinism rule](#run-ahead). The readings are kept as `f32`, as `ImageReadings`
holds them, which is enough for readings compared within a capture to about
three digits; the medians and ratios are `f64` because
`sfmtool_core::numeric` takes `f64`, and widening is exact.

## Parameters

| Value | Default | Meaning |
|-------|---------|---------|
| `target_step_percent` | `10` | the step aimed for, percent of the long side |
| `step_band` | `0.3` | the band, `target × (1 ± step_band)` |
| `blur_ratio` | `0.6` | below this times the median sharpness is blurred |
| `neighbour_blur_ratio` | `0.85` | below this times the sharpest neighbour's sharpness is blurred against the neighbours |
| `baseline_frames` | `200` | the samples, shared among the sequences |
| `min_matches` | `30` | the mutual matches a pair that matches well has |
| baseline gaps | 1, 3 | the frames after each sample the baseline reads |
| video signs | 5%; 2 to 4; factor 2; all three | the three signs, all of which make a video |
| barely moves | 100% | the total motion below which the camera barely moves |
| re-aims | 4 | before a step fails |
| recent rates | 3 | the accepted steps the prediction takes the median of |
| weak point | 2 × `min_matches`; 3 frames | a weak step, which beyond the next frame fails, and a short segment |
| break read-ahead | `W` frames | how far the walk prefetches through a break, for a source of `W` workers |
| neighbours | 2 positions on either side; below the median | the frames a candidate is judged against for blur against its neighbours, and which sensor images are judged |

The last eight are constants of the module.

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

# Extract with the workspace's sfmtool settings, ahead of the walk.
report = pick_images([sequence], feature_options=config["feature_options"],
                     feature_prefix_dir=config["feature_prefix_dir"],
                     target_step_percent=10.0)
# Or read .sift files that already exist.
report = pick_images([sequence])
report["verdict"]                         # "solvable"
report["sequences"][0]["walk"]["chosen"]  # positions of the chosen frames
```

`CaptureImage`, `CaptureFrame` and `CaptureSequence` mirror the Rust structs;
a frame's `images` and a sequence's `frames` are tuples, built once, so
`sequence.frames[id["frame"]]` is the frame a report's id names.
`feature_options` and `feature_prefix_dir`, given together, choose the
extracting source, with `concurrent_images` its worker count; with neither,
the source is the read-only one, and one without the other, or
`concurrent_images` without them, is a `ValueError`. `pick_images` takes the
fields of `PickImagesOptions` as keyword-only arguments, with the pair options
under the names `match_frame_pair` uses and the forest under the names
`FrameFeatures` uses (`preset`, `num_trees`, `leaf_size`, `max_leaf_checks`);
one left out takes the Rust default. `progress`, a `ProgressCounter`, holds
the count of frames read. The report is returned as a dict of plain values,
the report's serde form, with `None` for a number that is not finite.

The GIL is released while the walk runs. Each progress event, so each pair
matched, each wait and every 50 ms of a wait on the workers, takes it to run
Python's signal handlers; a handler that raises, such as the
`KeyboardInterrupt` of Ctrl+C, cancels the walk and stops its workers, and its
exception is raised. A cancellation from anywhere else is `InterruptedError`.
A `.sift` file that cannot be read is the `OSError` of
[python-bindings.md § Format errors](../../python-bindings.md#format-errors),
a folder that cannot be listed the `OSError` subclass for its kind, a missing
file of the read-only source `FileNotFoundError`, an image that cannot be read
or a file that cannot be written while extracting `OSError`, a panicked worker
`RuntimeError`, and every other error a `ValueError` with the Rust message,
among them invalid feature options, named as `params['key']` as
`build_sift_files` names them.

## Testing

The Rust tests in
[pick_images/tests.rs](../../../crates/sfmtool-core/src/analysis/pick_images/tests.rs)
walk synthetic captures
([tests/capture.rs](../../../crates/sfmtool-core/src/analysis/pick_images/tests/capture.rs)):
a scene of points with random descriptors seen by a pinhole camera along a
path with known motion per frame, with readings set per frame, served by a
source whose workers, the extracting source's pool, write a frame's version 2
`.sift` files only when it is prefetched or waited for, and which records
every wait and prefetch. They cover a steady video chosen at about the
target step, a change of speed that the stride follows, a blurred frame
swapped for a neighbour, a frame above the median's blur threshold that is
blurred against its neighbours swapped for the sharpest neighbour whose step
fits, a frame sharper than the median whose neighbours are not read though
it is far less sharp than they are, a uniformly blurred stretch none of
whose frames is swapped against its neighbours, neighbours out of the band
or with few matches not taken and the candidate kept when none qualifies, a
sharp frame with gradients along one direction
kept and not judged blurred, blur judged from the sharpness alone, a blurred
run kept at its sharpest frames within the band, a long blurred stretch,
flagged for blur alone, crossed with every step in the full band, a bad patch
flagged for weak matches and filled in at halved steps, two breaks with a
short segment and steps with few matches taken only at the next frame or in
a gap filled in, a blurred frame whose sharp neighbours, a
blurred run whose sharp frames and a wholly blurred run whose least blurred
frames have few matches, none of them taken, shuffled
frames, frames 6% apart that show the
other two signs and a three-frame sequence that are not a video, a camera
that barely moves, which is not solvable, a rig whose second sensor's images
match nothing
and whose blur alone, against its median and against its neighbours, swaps a
frame, several sequences sharing the baseline,
every error variant, a worker's failure on a frame of each stage, which names
the frame, and on a frame only predicted, which stops nothing, cancellation
before and during the run, that a run repeats exactly, that every frame of
the baseline and every frame read through a break after its first failing
pair was prefetched, with no more than the workers' count extracted and not
read, and that one capture with a change of speed, a swap for each reason, a
blurred run, a gap and a break gives the same report and the same waits with
prefetching off and on and with one and eight workers. Every
test checks that the frames waited for are the frames the report lists as
read, each once, that those have `.sift` files written in the run, and that
every other frame with files was predicted, as many as
`frames_extracted_unread` counts.

[source/tests.rs](../../../crates/sfmtool-core/src/analysis/pick_images/source/tests.rs)
checks the extracting source on seoul_bull images, writing the files
`build_sift_files` writes and nothing for files up to date, extracting a
prefetched frame without a wait, naming the frame of an image that cannot be
read and refusing a `sift_path` elsewhere, invalid options and version 1
settings; the read-only source's missing file; and the pool's order (a wait
ahead of prefetched frames, a prefetch replacing the last), a cancelled wait
that stops its workers promptly, a panicking job reported as an error, a
two-image frame cancelled between its images, which reports the file it
wrote, a frame whose work wrote and was then stopped or failed, which the
pool counts as written, and workers stopped as they start, over many rounds,
every one of which stops.

[test_pick_images_walk_rust_bindings.py](../../../tests/rust_bindings/analysis/test_pick_images_walk_rust_bindings.py)
checks the bindings on a synthetic capture whose files are written first and
read, a missing file, the options and the source options and their errors,
a `KeyboardInterrupt` delivered while the walk runs, the exceptions of each
error, the capture classes, and the seoul_bull photographs extracted by the
walk with the workspace's settings, which are judged not a video, whose files
equal `build_sift_files`', and which a second run reads without extracting.

## Non-goals

- Collecting a workspace's frames into sequences, printing the report and
  writing the chosen frames as a named image list, which the
  [`sfm pick-images`](../../cli/image-feature/pick-images-command.md) command
  does. The choices still open are in
  [pick-images-open-questions.md](../../drafts/pick-images-open-questions.md).
- Choosing among photographs. A sequence that is not a video stops the run;
  nothing chooses frames from it.
- Fitting geometry to the matches. Neither the walk nor the verdict tells a
  camera that moved through a static scene from one that only turned, or from
  a scene that moved on its own; a capture of either is walked like any other
  video, and the solve that follows finds out.
