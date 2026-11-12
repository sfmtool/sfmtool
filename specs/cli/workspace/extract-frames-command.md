# `sfm extract-frames` Command

Most captures for Structure-from-Motion start as video: a phone or action
camera filmed while walking around a scene. The solver and every other tool in
sfmtool read still images, so the video's frames first have to be written out
as numbered image files. `sfm extract-frames` does that with ffmpeg: each
video becomes a directory of JPEG frames, sampled at a chosen rate (10 per
second by default), scaled down to a chosen size (1920 pixels on the long side
by default), turned upright, and named so that the commands which group
numbered files into sequences, such as `sfm pick-images`, read each directory
as one. Beside the frames it writes a record of each frame's time in the
video. Without it, a user has to know ffmpeg's options for rotation, variable
frame rate and file naming, and a wrong choice (frames duplicated by ffmpeg's
constant-rate output, or numbers padded to a width that runs out) only shows
up later as a poor solve.

## Command

```bash
sfm extract-frames VIDEO... -o DIR [--fps N|full] [--max-frame-size PX|full]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `-o/--output DIR` | required | The directory the frame directories go in. Created when missing. It need not be in a workspace. |
| `--fps N\|full` | `10` | Frames per second of video time to keep, a positive number; `full` keeps every decoded frame. |
| `--max-frame-size PX\|full` | `1920` | The largest long side of a written frame, in pixels, an integer of at least 64. A larger frame is scaled down keeping its aspect; a smaller one is written at its own size, never scaled up. `full` writes every frame at the video's size. |

An invalid `--fps` or `--max-frame-size` is a usage error. Each `VIDEO` gets
the directory `DIR/<stem>/`, `<stem>` being the video's file name without its
extension, holding `frame_000001.jpg`, `frame_000002.jpg`, … in presentation
order, and the record `video-frames.json`
([workspace/video-frames.md](../../workspace/video-frames.md)).

```bash
# Two walks at the defaults: 10 fps, long side at most 1920
sfm extract-frames walk1.mp4 walk2.MOV -o captures/garden
# -> captures/garden/walk1/frame_000001.jpg ...  captures/garden/walk2/frame_000001.jpg ...

# Every frame, at the video's own size
sfm extract-frames orbit.mov -o frames --fps full --max-frame-size full

# Then, in a workspace, as for any frames
sfm pick-images captures/garden --write-list picked
```

The command is [`_commands/extract_frames.py`](../../../src/sfmtool/_commands/extract_frames.py),
with its row in `cli.COMMANDS` under *Workspace*, beside `sfm insv2rig` and
`sfm pano2rig`, the other commands that turn a capture into a workspace's
images. The work is in [`_video_frames.py`](../../../src/sfmtool/_video_frames.py),
in two steps so that a caller can check every input before it creates
anything: `plan_extraction(videos, out_dir, *, fps, max_frame_size, command)`
finds ffmpeg, probes every video and applies every check of
[Checks](#checks), creating nothing, and returns an `ExtractionPlan`;
`extract_planned(plan)` then extracts its videos in order and returns, for
each, an `ExtractedVideo` with the frame directory, the frame pattern, the
frame count and the record. `parse_fps` and `parse_max_frame_size` parse the
two option values, `FullOr` is the click type that reads them, and `command`
names the calling command in the messages. `problems` takes the problems a
caller found in its own inputs, listed first in the same error.
[`sfm pick-images --new-workspace`](../image-feature/pick-images-command.md#from-videos---new-workspace)
is the other caller: it checks its new workspace's directory, plans the
extraction with those problems, creates the workspace and extracts with
`extract_planned`, at the same defaults.

```python
plan = plan_extraction([Path("walk1.mp4")], Path("frames"), fps=10.0,
                       max_frame_size=1920, command="sfm extract-frames")
for video in extract_planned(plan):
    print(video.frame_dir, video.frame_count)
```

## Naming and numbering

The frames are numbered from 1 with no gaps, in the order they are written,
with six zero-padded digits: `frame_%06d.jpg`, the pattern `sfm insv2rig`
also uses. `number_from_filename` reads the digits next to the extension, so
the number is the frame's, whatever digits the directory name holds, and
pick-images and named image lists read each directory as one sequence with
the frame field `%06d`. Six digits hold 999,999 frames, over 9 hours of 30 fps
video at `--fps full` and 27 hours at the default 10 fps. A longer video goes
on to `frame_1000000.jpg`, which `%06d` still matches, since printf writes a
number wider than its field in full
([pick-images-command.md § Writing the list](../image-feature/pick-images-command.md#writing-the-list)).
The number is a count of written frames, not a source frame index: at 10 fps
from a 30 fps video, `frame_000002.jpg` is the video's fourth frame. The
record holds each frame's time.

## What ffmpeg is asked to do

One `ffmpeg` process per video, with `-nostdin -nostats`, the first video
stream that is not an attached picture (cover art) mapped with `-map 0:v:N`,
`-an -sn -dn` (no audio, subtitle or data stream is read), and this filter
chain:

1. **Rotation.** A phone stores a portrait video as landscape pixels with a
   rotation in its metadata (the display matrix). ffmpeg applies it by
   default (`-autorotate` is on), inserting the turn before the filters
   below, so the frames are upright and the size filter sees the rotated
   size. The record holds the rotation the metadata gave.
2. **Frame rate.** With a number `N`, a `select` filter keeps the first
   decoded frame whose presentation time `t` falls in each interval
   `[k/N, (k+1)/N)`:
   `select='isnan(prev_selected_t)+gt(floor(t*N),floor(prev_selected_t*N))'`.
   This works the same for a constant and a variable frame rate (phones record
   variable rate, slowing down in low light): it never duplicates a frame and
   never invents one, an interval with no frame is skipped, and the numbers
   stay continuous. ffmpeg's own `fps` filter is not used, since it duplicates
   frames to fill gaps in a variable-rate video. When `N` is at or above the
   video's base rate (ffprobe's `r_frame_rate`, the lowest rate that
   represents every timestamp, so no two frames are closer than `1/N`), every
   frame is kept, there is no `select` filter, and the command says so. With
   `full` there is no `select` filter.
3. **Square pixels.** A video whose sample aspect ratio is not 1:1 (some
   camcorder and broadcast formats) is scaled horizontally to square pixels,
   `scale=iw*sar:ih,setsar=1`, since every camera model here assumes square
   pixels.
4. **Size.** With a number `S`, when the displayed frame's long side (the
   stored width times the sample aspect ratio, or the height) is larger than
   `S`, `scale='if(gte(iw,ih),S,-2)':'if(gte(iw,ih),-2,S)':flags=area,setsar=1`
   brings the long side to `S`, the other side following the aspect, rounded
   to even (`-2`). A frame no larger than `S` gets no `scale` filter, so it is
   never scaled up and keeps its own size. Area averaging is the resampling
   sfmtool uses for its own reductions
   ([core/features/sift.md](../../core/features/sift.md)). `setsar=1` marks
   the pixels square after the rounding, which changes the aspect by under a
   pixel.
5. **Times.** `showinfo` logs each kept frame's index, `pts` and size on
   stderr, which the command reads for the record and for its progress line.

Output options: `-fps_mode passthrough`, so the image muxer writes each frame
the filters pass exactly once (constant-rate output would duplicate and drop
frames of a variable-rate video), and `-q:v 2` to ffmpeg's `mjpeg` encoder,
its highest quality setting in practical use (about libjpeg quality 95, the
quality `write_image_rgb` writes with), in the full-range 4:2:0 format ffmpeg
negotiates for JPEG. Video is 4:2:0 already, so 4:4:4 would store no more
colour detail. A 10-bit video is converted to 8 bits by ffmpeg's default
conversion.

## Frame times

Each frame directory gets `video-frames.json`
([workspace/video-frames.md](../../workspace/video-frames.md)), written once
its video's frames are all written. A frame's time is its `pts` from
`showinfo` times the time base `showinfo` logs when it is configured, rounded
to the microsecond; this is exact, where `showinfo`'s printed `pts_time` has
six significant digits, 10 ms after 1000 s. `fps` and `max_frame_size` are the
options as given (`"full"` or a number), and `frame_size` is the size
`showinfo` logs for the first frame.

## Output

```text
ffmpeg: /usr/bin/ffmpeg (7.1)
walk1.mp4: hevc 3840x2160, rotated 90, 29.97 fps (variable), 612.4 s
  -> captures/garden/walk1/ at 10 fps
  2210 of about 6124 frames (10 s)
  6122 frames written, 1080x1920, in 1m 48.2s
walk2.MOV: h264 1920x1080, 30.00 fps, 95.1 s
  -> captures/garden/walk2/ at 10 fps
  951 frames written, 1920x1080, in 9.8s
Wrote 7073 frames from 2 videos to captures/garden
Completed in 1m 58.43s
```

Each video's line gives its codec, its stored size, its rotation when it has
one, its average frame rate, marked `(variable)` when ffprobe's
`avg_frame_rate` and `r_frame_rate` differ, and its duration. The progress
line is printed every 10 seconds while a video is extracted, counting the
`showinfo` lines; "about" is the duration times the rate (the average frame
rate when every frame is kept). Exit status 0 when every video was extracted,
1 on an error, 2 on a usage error.

## Checks

Every check here runs before anything is created, and the error lists every
offending input, not only the first, under "N problems found, so nothing was
created":

- **ffmpeg.** `ffmpeg` and `ffprobe` are found on PATH by `locate_program`,
  the lookup `sfm version` reports ffmpeg by, so the two cannot disagree.
  Missing either, the command stops before any other check: "sfm
  extract-frames needs ffmpeg and ffprobe on PATH, and found no ffprobe.
  Install ffmpeg 5.1 or newer (https://ffmpeg.org/download.html; `winget
  install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, the
  `ffmpeg` package on Linux, or `ffmpeg` from conda-forge) and put the folder
  that holds both ffmpeg and ffprobe on PATH. `sfm version` shows the ffmpeg
  it finds." An ffmpeg older than 5.1, the first release with `-fps_mode`, is
  refused the same way, naming its version. The release is read from the first
  line of `ffmpeg -version` (`7.1.1`, `n6.0`, `4.4.2-0ubuntu0.22.04.1`); a
  development build names a date or a commit there instead, and its libavutil
  version is compared with 57.28, the libavutil of release 5.1. An output
  with neither is refused as a version the command cannot tell.
- **`.insv`.** A file ending `.insv` (any letter case) is refused: "an
  Insta360 .insv file holds two fisheye lenses; extract it with `sfm
  insv2rig`, which writes both lenses and their .camrig".
- **Not a video.** A directory, a missing file, or a file with an image
  extension (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, which ffmpeg
  would open as a one-frame video) is refused. Any other file is probed with
  `ffprobe -v error -show_streams -show_format -of json -o -`; one that ffprobe
  cannot open is refused with the last line ffprobe printed, and one with no
  video stream other than an attached picture is refused as having none.
- **HDR.** A video whose transfer characteristic (ffprobe's `color_transfer`)
  is `arib-std-b67` (HLG, which iPhones record) or `smpte2084` (PQ, as in
  HDR10) is refused: "HDR video (HLG, transfer arib-std-b67), which this
  command does not convert; convert the video to SDR first". Converting it to
  8-bit JPEG without tone mapping would leave HLG frames flat and desaturated
  and PQ frames dark and flat, and the choice of tone mapping changes what
  SIFT sees.
- **Colliding names.** Two videos whose stems are equal ignoring letter case
  (`walk.mp4` and `walk.MOV`, or one file given twice) are refused, naming
  both, since they would share a directory; letter case is ignored because
  Windows and macOS do. A stem with a `*` is refused, since a named image
  list's pattern cannot hold it.
- **Frame directories.** Every video's `DIR/<stem>/` must be missing or an
  empty directory. One that holds anything, or a file of that name, is
  refused, naming it, so a second run never mixes two extractions' frames in
  one sequence. `DIR` itself may hold anything else, such as earlier videos'
  directories; a `DIR` that is a file is refused.

## Failures and Ctrl+C

A non-zero exit of ffmpeg stops the run with the video, the exit status and
ffmpeg's last 20 lines of stderr. A count of `showinfo` lines that does not
match the frame files written (none, a missing last frame, or one past it)
stops it too. On Ctrl+C the command terminates ffmpeg, kills it if it has not
exited after 5 seconds, waits for it and stops. Either way the frames already
written are left, and the video being extracted has no `video-frames.json`;
the videos extracted before it are complete, with theirs.

## Implementation notes

**Why Python.** The work is ffmpeg's: decoding, filtering and JPEG encoding
happen in its process, and the command builds its arguments, reads its stderr
and checks its result, as `sfm insv2rig` does
([`rig/insv2rig.py`](../../../src/sfmtool/rig/insv2rig.py)). A Rust version
would spawn the same process and parse the same text, and no Rust caller
needs it, so this is the pipeline case
[STANDARDS.md § Logic lives in Rust](../../STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers)
puts out of scope. The long call is the child process, so `run_ffmpeg`
prints no progress of its own: it hands each stderr line to its caller, which
prints the progress line, and on any exception from that loop, Ctrl+C
included, it stops ffmpeg as above and re-raises.

**Output lost at exit.** Some ffmpeg builds, conda-forge's win-64 ffmpeg
9.0.2 among them, now and then lose what they printed to stdout through the C
library when they exit, and still exit with status 0. Measured on that build,
about 3 runs in 100 of `ffmpeg -version` printed nothing, and about 2 in 100
of ffprobe's JSON description were cut short at a buffer boundary. So ffprobe
is given `-o -`, which writes the description through ffmpeg's own output
layer, flushed before exit, and `run_for_output`, which runs both programs
for their stdout, repeats a run that exited with status 0 and printed
nothing, up to three runs in all; `ffmpeg -version` prints less than one
buffer, so a run loses all of its output or none. `sfm version` reads
`ffmpeg -version` through the same function. The extraction itself is not
affected: its frames are written to files, and `showinfo`'s lines go to
stderr, which is not buffered.

**The record is not a format of a format crate.** No sfmtool command reads
`video-frames.json`, so it is a record rather than one of the formats
[STANDARDS.md § Every file format has the same I/O shape](../../STANDARDS.md#every-file-format-has-the-same-io-shape)
covers; a command that comes to read it would give it a format crate first.

**Floating point in the `select` expression.** `t*N` for a frame exactly on
an interval boundary can land just below an integer, putting that frame in
the interval before. That moves at most one frame to the next interval and
never keeps two in one, so the rule (at most one frame per interval, the
earliest) holds. The record's times are rounded to the microsecond, so a test
that places them in intervals allows for a time written just below a
boundary.

## Testing

[`tests/test_extract_frames_command.py`](../../../tests/test_extract_frames_command.py).
The videos are made once per test module with ffmpeg's `lavfi` sources
(`testsrc2`, one or two seconds at 320x240 or smaller) and its built-in
`mpeg4` and `aac` encoders, so no codec library is needed. ffmpeg comes from
conda-forge in the `test` and `test-ci` environments, so these tests run in
CI rather than skip.

- At the default 10 fps from a 30 fps video, 20 frames numbered from 1 with
  no gaps, one in each tenth of a second, the whole record, the frame size
  and the output lines; at `--fps full`, 15 and 2.5, the counts and at most
  one frame per interval; at 30, every frame, with the message saying so.
- `--max-frame-size 160` on 320x240 gives 160x120; on a 100x80 video gives
  100x80; `full` keeps 320x240. A video with rotation metadata
  (`-display_rotation 90` when it is made) gives portrait frames and
  `rotation_degrees` 90 in the record.
- A variable-rate video (timestamps rewritten with `setpts`, written with
  `-fps_mode vfr`): 40 frames, no two byte-identical, at the source frames'
  times.
- Two videos in one run, beside an earlier video's directory and an empty
  directory named for one of them.
- Every refusal of [Checks](#checks) in one run, each named, with the output
  directory unchanged: `.insv`, an image, a non-video file, a directory, a
  missing file, a stem with `*`, audio whose only video stream is cover art,
  HLG and PQ video, a non-empty frame directory, a file in its place, and
  colliding stems; one file given twice; no ffmpeg on PATH; an ffmpeg 4.4
  (a script on PATH that prints its version line); the version parsing on
  release, `n`-prefixed and development-build outputs; and invalid option
  values as usage errors.
- Ctrl+C: an exception from the line callback terminates a running ffmpeg and
  is re-raised; raised from the progress line of an extraction, it leaves no
  record.
- `sfm version` and the command find the same ffmpeg, and a run that exits
  with status 0 having printed nothing is repeated.

## Non-goals

- Insta360 `.insv` and other multi-lens files; `sfm insv2rig` handles those.
- Converting HDR video. The command refuses it, and the video is converted
  to SDR before extraction.
- Camera intrinsics from video metadata. Phone videos rarely carry a focal
  length, and pick-images finds a camera by its pilot vote.
- Hardware-accelerated decoding, and extracting several videos at once:
  ffmpeg already decodes with several threads.
- Resuming an interrupted extraction.
- Choosing frames. `sfm pick-images` chooses among the frames on disk, and
  `sfm pick-images --new-workspace DIR VIDEO...` extracts the videos with
  this command's code and chooses among their frames in one call.
