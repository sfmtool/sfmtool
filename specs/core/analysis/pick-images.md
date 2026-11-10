# Picking Images from a Video

A video of a walk around a scene has many more frames than a
Structure-from-Motion solve should use. Neighbouring frames are nearly the same
picture and some frames are blurred. Picking images looks at a capture before
it is solved and chooses the frames a solve should use: frames spaced so that
the image moves by about a tenth of its width from one to the next, but never
so far apart that two consecutive chosen frames keep fewer mutual matches than
a floor, sharper ones in place of frames blurred against their neighbours,
and more frames where the capture is hard to follow. It walks each sequence
from its first frame, measuring each step rather than taking a stride fixed
in advance. A capture whose neighbouring frames already match with fewer
matches than the floor, such as one subsampled from a video or a set of
photographs taken one after another, keeps every frame.
It makes no judgement of whether a capture can be solved and fits no
geometry: whether the frames it chooses describe one camera motion through a
static scene, and whether the camera moved enough, is for the solve to find
out.

Everything is measured from SIFT features. The motion between two frames is
the median displacement of the features that match between them, and how well
they match is how many mutual matches they have
([frame-pair-matching.md](frame-pair-matching.md)); a frame's sharpness
and gradient structure tensor come from its `.sift` file's metadata
([sift-file-format.md § The reading image](../../formats/sift-file-format.md#the-reading-image)).
The walk reads only the frames it needs: the frames it chooses, the frames it
tries on the way and the frames around each frame it chooses. A source makes each frame's `.sift` files
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
    pub neighbour_blur_ratio: f64, // 0.85, in (0, 1)
    pub min_step_matches: usize,   // 600, at least 1: the match floor
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
reference images differ in size, or an essential split camera of another size
than the frames, the two `FramePairError`s the options checks leave possible. `Cancelled` is a set cancel flag. No caller input panics
([STANDARDS.md § Caller input never panics](../../STANDARDS.md#caller-input-never-panics)).

**A `Progress`.** A walk over a long video runs for minutes, mostly in
extraction. `pick_images` opens the phase `walk`, reports
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
use sfmtool_core::analysis::pick_images::{pick_images, ExtractingSource, PickImagesOptions};
use sfmtool_core::features::sift::BuildSiftFilesOptions;
use sfmtool_core::progress::Progress;

// The workspace's feature_options and feature_prefix_dir.
let mut source = ExtractingSource::new(BuildSiftFilesOptions::sfmtool_defaults())?;
let report = pick_images(&sequences, &mut source,
                         &PickImagesOptions::default(), &Progress::none())?;
let numbers: Vec<i64> = report.sequences[0].chosen.iter()
    .map(|&p| sequences[0].frames[p].number).collect();
```

### The report

`PickImagesReport` holds one `SequenceReport` per sequence, every frame read
(`FrameReading`: its id and each sensor's readings, the sharpness, the
gradient structure tensor and the axis ratio and coherence derived from it)
in the order read, every pair matched (`MeasuredPair`, with its
`FramePairStats`) in the order matched, `frames_extracted`, per frame read,
whether the source wrote its `.sift` files in this run rather than finding
them up to date, and `frames_extracted_unread`, how many frames the source
wrote files for that the walk never read, the predictions it did not follow.
That count is the one part of the report that depends on how far ahead the
walk predicts and how fast the source works. A `SequenceReport` holds the
chosen positions, every accepted step (from, to, its statistics, its rate
and its `StepKind`: `InBand`, `AboveBand`, `Bracketed`, `AtLimit`,
`AtFloor` or `BelowBand`), the swaps of frames blurred against their
neighbours, each as candidate and chosen frame, and the count of frames the
walk read.

Every type serializes with `serde`, a number that is not finite as `null`, so
a caller writes the whole report as JSON as it is.

Each `FramePairStats`, of a pair and of a step, carries the pair's essential
split ([frame-pair-matching.md § The essential split](frame-pair-matching.md#the-essential-split))
when `pair.essential_split` gives a camera, as `{"status": "measured", …}` or
`{"status": "not_measured", "matches": …, "reason": …}`, and `null` when it
gives none. The walk does not read the split, and `PickImagesOptions::default()`
gives no camera, so a default walk measures none and `essential_split` is
`null` in every pair.

## The algorithm

Picking runs in three stages: check the input, walk each sequence, and
assemble the report. The sequences are walked one after another.

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
- The **match floor** is `min_step_matches` mutual matches. A step
  **holds** when it goes to the next frame, whatever its matches, or when it
  goes beyond the next frame and keeps at least the match floor; a step
  beyond the next frame that keeps fewer is **short of the floor**.
- The **band** is `target × (1 ± step_band)` around the target step: 7% to
  13% at the defaults.

### The match floor

A solve follows a capture through the tracks that consecutive chosen frames
share: the more of the scene two consecutive chosen frames show as the same
points, the more tracks link them, and the more views each of those tracks
has. The mutual matches of a pair count those shared points, and they fall
as the frames move apart, at a rate the step alone does not show: two
captures whose neighbouring frames move by the same step can lose matches
with the distance at very different rates, as their scenes' depth and
texture and their cameras' motion differ. So the walk keeps every step beyond
the next frame at or above the match floor, an absolute count of
`min_step_matches` mutual matches, whatever its step. The floor is a count rather than a fraction of
the matches the next frame keeps, since the share of those that a step of the
target keeps differs widely from one capture to another, and a fraction that
leaves a video's steps alone passes over the frames of a capture whose
matches fall fast.

The step to the next frame is not held to the floor, and is always taken: it
is the shortest step there is, and passing over no frame it loses nothing a
shorter step would keep. A capture whose neighbouring frames match with fewer
matches than the floor so keeps every frame, as a capture already subsampled
from a video, or a set of photographs taken one after another, needs. In a
video the floor binds only where its matches fall fast: elsewhere a step of
the target keeps more than the floor, and the floor changes nothing.

Since the next frame is always taken, a stretch of frames that match nothing,
such as frames of a covered lens or of a camera swung too fast, is walked one
frame at a time, and every one of its frames goes into the list. Nothing
breaks the walk or restarts it, and nothing is left out of the list for
matching badly: the solve ignores the frames that do not connect to the
others, and a solve that splits there splits whatever the list holds. The
floor is the walk's one count of matches.

### The walk

The walk chooses the sequence's first frame and repeats from the last chosen
frame `f` until it chooses the sequence's last frame. The **recent rates**
are the rates of the last three accepted steps that kept the match floor.

1. **Predict.** With recent rates, the candidate is `c = f + target / rate`,
   rounded, at least `f + 1` and at most the sequence's last frame, with the
   rate their median. With none, at the start of the sequence and after a
   step to the next frame that kept fewer matches than the floor, the
   candidate is `f + 1`: the walk measures the step to the next frame first,
   and when that step holds and is below the band, its first re-aim (step 4)
   aims from the next frame's rate, which is the step's prediction and does
   not count as a re-aim.
2. **Measure** the step `m` from `f` to `c`.
3. **Decide.**
   - `c` is `f + 1` and keeps fewer matches than the floor: settle on it,
     as stopped at the floor. Matches fall with the distance, so every step
     beyond it would keep fewer too.
   - The step is short of the floor: `c` is too far, whatever its step.
     Search towards the floor (step 4).
   - `m` in the band: settle on `c`.
   - `m` above the band: re-aim. When `c` is `f + 1`, the motion per frame
     is wider than the band, and `c` is settled on.
   - `m` below the band: settle on `c` when it is the last frame; otherwise
     re-aim.
4. **Re-aim** to `c' = f + (c − f) · target / m`, kept strictly between the
   largest candidate measured below the band and the smallest measured
   above the band or short of the floor. After a candidate short of the
   floor, the next is instead the middle of the frames left between them,
   rounded up, so that the walk finds the farthest frame that keeps the floor
   in a few tries rather than one frame at a time; those tries do not count
   as re-aims. When no frame is left between them, they lie one frame apart,
   and the one below is settled on: as stopped at the floor when the one
   above it is short of the floor, and otherwise as bracketed, the two lying
   on either side of the band. After four re-aims without reaching the band,
   the walk settles on the largest candidate measured below the band, or,
   when it measured none, on `f + 1`.
5. **Settle** on the frame by judging it against its neighbours
   ([Blur](#blur)). When it is not blurred against them, it is accepted.
   When it is, the walk tries, sharpest first, the neighbours it is blurred
   against that lie after `f` and before the nearest candidate measured short
   of the floor and are not themselves blurred against it on any sensor, and
   accepts the first whose step holds and is no longer than the band and no
   shorter than the band or the candidate's own step. The sharpest is the
   neighbour whose sharpness over the candidate's, the least over the
   sensors, is highest; ties go to the nearer, then the earlier. The swap is
   reported. When no neighbour qualifies, the candidate is accepted.
6. **Accept** the frame. A step that keeps the floor adds its rate to the
   recent rates; a step to the next frame that keeps fewer clears them, so
   the next step measures its next frame first rather than aiming from a
   rate measured before the matches fell.

**Every step holds.** A step to the next frame always holds, and the search
of step 4 only settles beyond the next frame on a frame whose step keeps the
floor, so a step cannot fail: no step is retried, no gap is filled in by
halving, and the walk never stops before the last frame. Where the matches
fall fast, the floor search shortens the steps; where the next frame itself
keeps fewer than the floor, the walk takes every frame.

**What is never read.** A frame the walk does not read lies between two
chosen frames, inside a step that held. A frame within 2 positions of a
chosen candidate is read, and is measured only if it is tried in the
candidate's place. A frame the walk predicted and did not read may still
have been extracted ([Run-ahead](#run-ahead)).

### Blur

A frame the walk is about to accept is judged against its **neighbours**,
the frames within 2 positions of it on either side, which the walk reads in
one wait. A sensor image is **blurred against its neighbours** when its
sharpness is below `neighbour_blur_ratio` times the sharpest neighbour's on
that sensor, and a rig frame when any of its sensors' images is. Sharpness
depends on what a frame shows as much as on how sharp it is, so a frame
compared with frames of other parts of the scene is judged against other
pictures; a frame a few positions away shows nearly the same scene, so the
less sharp of two such frames is the blurrier one. Every frame about to be
accepted is judged this way, whatever its sharpness.

The comparison is with the sharpest neighbour, not their median, because
blur comes in runs: within a few positions of a blurred frame several
neighbours are often blurred too, their median is then a blurred frame, and
the sharpest is also the frame a swap would take. A stretch whose frames are
all equally blurred is not judged blurred against its neighbours, which
share the blur: the walk crosses it at the band's spacing, and the solve
gets the frames the capture has there.

The ratio errs towards judging a frame blurred against its neighbours. A
swap moves only to a neighbouring frame that is sharper on a sensor by the
ratio and passes the match and band checks, so judging a sharp frame
blurred against its neighbours costs little, while a higher ratio swaps more
of the frames that are blurrier than a neighbour.

The report also carries each image's axis ratio and coherence, which
`ImageReadings::axis_ratio` and `ImageReadings::coherence` derive from the
gradient structure tensor, but blur is not judged from them. They measure the
direction of the gradients, which the scene sets as strongly as motion blur
does: a fisheye's geometry, a slope seen at an angle or a subject made of
lines gives a sharp frame strongly directional gradients, so those readings
would call such sharp frames blurred.

### Rigs

A rig frame is read, judged and chosen with all its sensors, and reading it
reads every sensor's `.sift` metadata. Its motion is measured from the
reference sensor's images only, the first of each frame, since their
displacement describes the rig's motion as well as any
sensor's and the sensors' images need not be the same size. Its sharpness is
judged per sensor against that sensor's own images of its neighbouring
frames, since the sensors look at different parts of the scene; a rig frame
is blurred against its neighbours when any of its sensors is.

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
prediction's, five frames or, for a source of `W` workers, `W`
([Run-ahead](#run-ahead)), and a wait's; each worker holds the one frame it
extracts, as many as `build_sift_files` holds in flight under the same memory
cap.

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

The walk waits for one candidate at a time, or for the frames around a
candidate in one wait. Each frame's extraction already runs on every core,
so the workers the walk can keep busy ahead of it are worth having only for
frames it will read: a frame extracted on a guess that the walk then does
not read takes cores from the frame it is waiting for, and on a capture
whose motion or matches change from step to step most guesses along the
predicted strides are wrong. So the walk predicts only frames it is about to
read in most cases:

- **The next step's first candidate and its neighbours.** When the walk
  settles on a candidate `c`, before it waits for `c`'s neighbours it
  computes the first candidate of the step that would follow if `c` is
  accepted, from the recent rates that accepting `c` would leave (step 1 of
  [The walk](#the-walk)), and prefetches it with the frames within 2
  positions of it, so that they are extracted while the walk reads and
  judges `c`'s neighbours. Unless a neighbour is swapped in, that candidate
  is the next frame the walk measures, and where the motion is steady it is
  accepted and its neighbours read.
- **The frames after a step below the floor.** When accepting `c` would
  leave no recent rates, the walk goes on to read the frames after `c` one
  after another for as long as each keeps fewer matches than the floor with
  the one before, as through a stretch that matches nothing or a set of
  photographs. So it prefetches the `W` frames after `c`, for a source of
  `W` workers; at the end of such a stretch at most `W` of them are
  extracted and not read.
- **The floor search's next tries.** When a candidate is short of the floor
  and the walk moves to the middle of the frames left, it prefetches the
  middles of the two halves on either side of that try, one of which the
  search tries next unless it ends there.

Each prediction replaces the one before. The walk prefetches nothing ahead
of a re-aim, whose candidate depends on the step just measured.

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
  (`FrameFeatures::from_sift`) when a pair needs it. The walk holds the last
  chosen frame's features and the last candidate's, which become the chosen
  frame's when it is accepted. A frame's readings, a few numbers per sensor,
  are kept for every frame read. A pair's statistics are kept, so a pair is
  never matched twice.
- **Each frame is waited for once.** Every frame passes through one reading
  step that waits on the source for the frames not yet read and reads all
  their `.sift` metadata, which is what makes the report's frame list the
  exact list of frames waited for, in order. The prefetch step only passes on
  the frames not yet read.
- **No duplicate derivations.** The axis ratio and coherence are
  `ImageReadings`' own, the statistics of a pair `match_frame_pair`'s, and
  the median of the recent rates `sfmtool_core::numeric`'s.

## Determinism and precision

The report is the same for the same capture and options: the frame pair
statistics repeat exactly ([frame-pair-matching.md § Determinism and precision](frame-pair-matching.md#determinism-and-precision)),
every decision reads them and the readings in a fixed order, and no hash map is
iterated for an output. It is the same for any source, lookahead and worker
count too, but for `frames_extracted_unread`, by the
[determinism rule](#run-ahead). The readings are kept as `f32`, as `ImageReadings`
holds them, which is enough for readings compared within a capture to about
three digits; the sharpness ratios are compared in `f64`, and widening is
exact.

## Parameters

| Value | Default | Meaning |
|-------|---------|---------|
| `target_step_percent` | `10` | the step aimed for, percent of the long side |
| `step_band` | `0.3` | the band, `target × (1 ± step_band)` |
| `neighbour_blur_ratio` | `0.85` | below this times the sharpest neighbour's sharpness is blurred against the neighbours |
| `min_step_matches` | `600` | the match floor, the mutual matches a step beyond the next frame keeps; at least 1 |
| re-aims | 4 | before the walk settles below the band; the tries towards the floor and the aim from the next frame's rate are not counted |
| recent rates | 3 | the accepted steps the prediction takes the median of |
| neighbours | 2 positions on either side | the frames a candidate is judged against for blur against its neighbours |
| prediction below the floor | `W` frames | how far the walk prefetches where it reads every frame, for a source of `W` workers |

The re-aims, the recent rates and the neighbours are constants of the
module; the prediction below the floor is the source's `concurrency`.

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
report["sequences"][0]["chosen"]  # positions of the chosen frames
```

`CaptureImage`, `CaptureFrame` and `CaptureSequence` mirror the Rust structs;
a frame's `images` and a sequence's `frames` are tuples, built once, so
`sequence.frames[id["frame"]]` is the frame a report's id names.
`feature_options` and `feature_prefix_dir`, given together, choose the
extracting source, with `concurrent_images` its worker count; with neither,
the source is the read-only one, and one without the other, or
`concurrent_images` without them, is a `ValueError`. `pick_images` takes the
fields of `PickImagesOptions` but `prefetch`, which it leaves on, as
keyword-only arguments, with the pair options
under the names `match_frame_pair` uses and the forest under the names
`FrameFeatures` uses (`preset`, `num_trees`, `leaf_size`, `max_leaf_checks`);
one left out takes the Rust default. It takes no camera, so its report's
`essential_split` is `None` in every pair. `progress`, a `ProgressCounter`, holds
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
target step, whose first step measures the next frame; a change of speed
that the stride follows; a frame blurred against its neighbours swapped for
the sharpest neighbour whose step fits; a frame sharper than every other
frame of the capture but its neighbours, swapped all the same; a uniformly
blurred stretch none of whose frames is swapped against its neighbours,
crossed at the band's spacing; neighbours out of the band or short of the
floor not taken and the candidate kept when none qualifies; a sharp frame
with gradients along one direction kept; a capture whose neighbouring frames
match below the floor, every frame of which is kept with one pair matched
per frame; a capture whose matches fall with the distance, whose steps stop
at the farthest frame that keeps the floor short of the target; stretches
that match nothing, at the start and in the middle, walked one frame at a
time; a rig whose second sensor's images match nothing and whose blur
against its neighbours alone swaps a frame; several sequences walked in turn
and a lone frame; every error variant; a worker's failure on a frame read,
which names the frame; cancellation before and during the run; that a run repeats exactly; that
every step of a steady walk after the first starts at a predicted frame, that
the floor search's tries are predicted, with at most five frames per step
and one per try extracted and not read, that every frame of a capture walked
frame by frame is predicted, and that a worker failing on a frame predicted
and never read stops nothing; and that one capture
with a change of speed, a frame swapped for a neighbour, steps stopped at the
floor beyond the next frame and a stretch walked one frame at a time, each
named and checked to happen, gives the same report and the same waits with
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
walk with the workspace's settings, every one of which is kept, whose files
equal `build_sift_files`', and which a second run reads without extracting.

## Non-goals

- Collecting a workspace's frames into sequences, printing the report and
  writing the chosen frames as a named image list, which the
  [`sfm pick-images`](../../cli/image-feature/pick-images-command.md) command
  does. The choices still open are in
  [pick-images-open-questions.md](../../drafts/pick-images-open-questions.md).
- Choosing among photographs by anything but their order. Numbered
  photographs are walked as a sequence like a video's frames, and where
  consecutive photographs match with fewer matches than the floor, every
  one is kept.
- Judging whether a capture can be solved. The walk fits no geometry to the
  matches and does not ask whether the camera moved enough, so a
  camera that only turned, a scene that moved on its own and a camera that
  moved little but steadily are walked like any other video, and the solve
  that follows finds out.
