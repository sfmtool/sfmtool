# Extracting Frames from Video

**Status:** Draft. Decided by the maintainer: a reusable command that extracts
video frames into one directory per video, numbered so each directory reads as
one image sequence; `sfm pick-images --new-workspace DIR VIDEO...`, which makes
a workspace, extracts the videos with the same code and picks frames from them;
a selectable frame rate and maximum frame size, each with the value `full`;
unchosen frames kept unless `--prune-images` is given; a refusal of a non-empty
target; the refused inputs (`.insv`, colliding names, unreadable files, no
ffmpeg). Every other choice is this draft's and is listed in
[Decisions for review](#decisions-for-review). When it ships, the command
section is filed as `cli/workspace/extract-frames-command.md` and the
`--new-workspace` section folds into
[cli/image-feature/pick-images-command.md](../cli/image-feature/pick-images-command.md).

## Purpose

Most captures for Structure-from-Motion start as video: a phone or action
camera filmed while walking around a scene. The solver and every other tool in
sfmtool read still images, so the video's frames first have to be written out
as numbered image files. `sfm extract-frames` does that with ffmpeg: each video
becomes a directory of JPEG frames, sampled at a chosen rate (10 per second by
default) and scaled down to a chosen size (1920 pixels on the long side by
default), named so that the commands which group numbered files into sequences
read each directory as one. `sfm pick-images --new-workspace DIR VIDEO...` goes
from videos to a solvable set of frames in one call: it creates a workspace at
`DIR`, extracts the videos into it, and chooses the frames a solve should use,
as `sfm pick-images` does for frames already on disk. Without this, a user has
to know ffmpeg's options for rotation, variable frame rate and file naming, and
a wrong choice (frames duplicated by ffmpeg's constant-rate output, or numbers
padded to a width that runs out) only shows up later as a poor solve.

## `sfm extract-frames`

```bash
sfm extract-frames VIDEO... -o DIR [--fps N|full] [--max-frame-size PX|full]
```

The command is `_commands/extract_frames.py`, with its row in `cli.COMMANDS`
under *Workspace*, beside `sfm insv2rig` and `sfm pano2rig`, the other commands
that turn a capture into a workspace's images. The extraction itself is in
`src/sfmtool/_video_frames.py`, which `sfm pick-images --new-workspace` calls
too.

| Option | Default | Meaning |
|--------|---------|---------|
| `-o/--output DIR` | required | The directory the frame directories go in. Created when missing. It need not be in a workspace. |
| `--fps N\|full` | `10` | Frames per second of video time to keep, a positive number; `full` keeps every decoded frame. |
| `--max-frame-size PX\|full` | `1920` | The largest long side of a written frame, in pixels, an integer of at least 64. A larger frame is scaled down keeping its aspect; a smaller one is written at its own size, never scaled up. `full` writes every frame at the video's size. |

Each `VIDEO` gets the directory `DIR/<stem>/`, `<stem>` being the video's file
name without its extension, holding `frame_000001.jpg`, `frame_000002.jpg`, …
in presentation order, and a `video-frames.json` record
([Frame times](#frame-times)).

```bash
# Two walks at the defaults: 10 fps, long side at most 1920
sfm extract-frames walk1.mp4 walk2.MOV -o captures/garden
# -> captures/garden/walk1/frame_000001.jpg ...  captures/garden/walk2/frame_000001.jpg ...

# Every frame, at the video's own size
sfm extract-frames orbit.mov -o frames --fps full --max-frame-size full

# Then, in a workspace, as for any frames
sfm pick-images captures/garden --write-list picked
```

### Naming and numbering

The frames are numbered from 1 with no gaps, in the order they are written,
with six zero-padded digits: `frame_%06d.jpg`, the pattern `sfm insv2rig`
already uses. `number_from_filename` reads the digits next to the extension, so
the number is the frame's, whatever digits the directory name holds, and
pick-images and named image lists read each directory as one sequence with the
frame field `%06d`. Six digits hold 999,999 frames, over 9 hours of 30 fps
video at `--fps full` and 27 hours at the default 10 fps. A longer video goes
on to `frame_1000000.jpg`, which `%06d` still matches, since printf writes a
number wider than its field in full
([pick-images-command.md § Writing the list](../cli/image-feature/pick-images-command.md#writing-the-list)).
The number is a count of written frames, not a source frame index: at 10 fps
from a 30 fps video, `frame_000002.jpg` is the video's fourth frame. The times
are in [Frame times](#frame-times).

### What ffmpeg is asked to do

One `ffmpeg` process per video, with `-nostdin`, the first video stream that is
not an attached picture (cover art) mapped with `-map 0:v:N`, `-an -sn -dn` (no
audio, subtitle or data stream is read), and this filter chain:

1. **Rotation.** A phone stores a portrait video as landscape pixels with a
   rotation in its metadata (the display matrix). ffmpeg applies it by default
   (`-autorotate` is on), inserting the turn before the user's filters, so the
   frames are upright and the size filter below sees the rotated size. The
   command does not pass `-noautorotate`, and the record holds the rotation the
   metadata gave.
2. **Frame rate.** With a number, a `select` filter keeps the first decoded
   frame whose presentation time `t` falls in each interval
   `[k/N, (k+1)/N)`: `select='isnan(prev_selected_t)+gt(floor(t*N),floor(prev_selected_t*N))'`.
   This works the same for a constant and a variable frame rate (phones record
   variable rate, slowing down in low light): it never duplicates a frame and
   never invents one, an interval with no frame is skipped, and the numbers
   stay continuous. ffmpeg's own `fps` filter is not used, since it duplicates
   frames to fill gaps in a variable-rate video. A value at or above the
   video's rate keeps every frame, and the command says so. With `full` there
   is no `select` filter.
3. **Square pixels.** A video whose sample aspect ratio is not 1:1 (some
   camcorder and broadcast formats) is scaled horizontally to square pixels,
   `scale=iw*sar:ih,setsar=1`, since every camera model here assumes square
   pixels.
4. **Size.** With a number, `scale` with `flags=area` brings the long side to
   at most that size, the other side following the aspect, rounded to even
   (`-2`), and never scales up. Area averaging is the resampling sfmtool uses
   for its own reductions ([core/features/sift.md](../core/features/sift.md)).
5. **Times.** `showinfo` logs each kept frame's index and `pts_time` on
   stderr, which the command reads for the record and for its progress line.

Output options: `-fps_mode passthrough`, so the image muxer writes each frame
the filters pass exactly once (its default, constant rate, would duplicate and
drop frames of a variable-rate video), and `-q:v 2` to ffmpeg's `mjpeg`
encoder, its highest quality setting in practical use (about libjpeg quality
95, the quality `write_image_rgb` writes with), in the full-range 4:2:0 format
ffmpeg negotiates for JPEG. Video is 4:2:0 already, so 4:4:4 would store no
more colour detail. `-fps_mode` needs ffmpeg 5.1 or newer; an older ffmpeg
fails on it, and the error says so ([Errors](#errors)).

Bit depth and colour: a 10-bit video is converted to 8 bits by ffmpeg's
default conversion. An HDR video (transfer `arib-std-b67`, HLG, which iPhones
record, or `smpte2084`, PQ, as in HDR10) is converted with no tone mapping
([HDR video](#hdr-video)).

### Frame times

Beside the frames, `DIR/<stem>/video-frames.json` records where each frame came
from:

```json
{
  "version": 1,
  "video": "/captures/raw/walk1.mp4",
  "video_stream": 0,
  "codec": "hevc",
  "video_size": [3840, 2160],
  "rotation_degrees": 90,
  "color_transfer": "arib-std-b67",
  "fps": 10,
  "max_frame_size": 1920,
  "frame_size": [1080, 1920],
  "frame_pattern": "frame_%06d.jpg",
  "frame_times_s": [0.0, 0.1, 0.2, 0.3002, 0.4]
}
```

`frame_times_s[i]` is the presentation time of frame number `i + 1`, in
seconds from the stream's start, from `showinfo`'s `pts_time`; `fps` and
`max_frame_size` are the options as given (`"full"` or a number). A variable
frame rate makes number and time unrelated by any formula, and once the video
is gone or edited the times cannot be recovered, so they are written now, for
the cost of one small file. They let a later step match frames to another
record of the capture (a GPS track, a second camera) or re-extract one stretch
at a higher rate. No sfmtool command reads the file, so it is a record and not
one of the formats [STANDARDS.md § Every file format has the same I/O shape](../STANDARDS.md#every-file-format-has-the-same-io-shape)
covers; a command that comes to read it would give it a format crate first. Its
`.json` extension keeps it out of every image listing.

### Output

```text
ffmpeg: /usr/bin/ffmpeg (7.1)
walk1.mp4: hevc 3840x2160, rotated 90, 29.97 fps (variable), 612.4 s; HDR (HLG), not tone-mapped
  -> captures/garden/walk1/ at 10 fps, 1080x1920
  2210 of about 6124 frames (10 s)
  6122 frames written in 1m 48.2s
walk2.MOV: h264 1920x1080, 30.00 fps, 95.1 s
  -> captures/garden/walk2/ at 10 fps, 1920x1080
  951 frames written in 9.8s
Wrote 7073 frames from 2 videos to captures/garden
```

The progress line is printed every 10 seconds, as `sfm pick-images` prints
its own, counting the `showinfo` lines; "about" is the duration times the
rate. Exit status 0 when every video was extracted, 1 on an error, 2 on a
usage error.

### Directory rule

Before anything is written, every video's `DIR/<stem>/` must be missing or an
empty directory. One that holds anything, or a file of that name, is refused,
naming it, so a second run never mixes two extractions' frames in one
sequence. `DIR` itself may hold anything else, such as earlier videos'
directories.

## `sfm pick-images --new-workspace`

```bash
sfm pick-images --new-workspace DIR VIDEO... [--fps N|full] [--max-frame-size PX|full]
    [--prune-images] [workspace options] [pick-images options]
```

With `--new-workspace DIR`, the positional arguments are video files, not image
folders. The command:

1. Checks the options, finds ffmpeg and ffprobe, and probes every video
   ([Errors](#errors)), creating nothing until all pass.
2. Refuses a `DIR` that exists and is not an empty directory, and a `DIR`
   inside an existing workspace (`sfm ws init`'s nesting rule, with no
   `--force` to override it here).
3. Creates the workspace at `DIR`, with `init_workspace` and the workspace
   options below, as `sfm ws init DIR` would, and prints its lines.
4. Extracts each video into `DIR/<stem>/` with the same function as
   `sfm extract-frames -o DIR`, at the same defaults.
5. Runs pick-images on the frame directories, in the order the videos were
   given, exactly as `sfm pick-images DIR/<stem1> DIR/<stem2> …` would: `-r`,
   `--write-list`, `--force` (which replaces a list, as always), `--json` and
   the five numeric options apply unchanged. `--named-image-list` is a usage
   error, since a new workspace has no lists.
6. With `--prune-images`, deletes the frames not chosen
   ([Pruning](#pruning)).

```bash
sfm pick-images --new-workspace garden walk1.mp4 walk2.mov --write-list picked
cd garden && sfm solve -i -r picked

# A COLMAP workspace, every frame of a short video, only the chosen frames kept
sfm pick-images --new-workspace orbit orbit.mov --fps full \
    --feature-tool colmap --max-features 4096 --prune-images --write-list picked
```

**Workspace options.** These `sfm ws init` options are taken, under the same
names, with the same defaults and the same checks: `--feature-tool`,
`--max-features`, `--dsp/--no-dsp`, `--gpu/--no-gpu`,
`--affine-shape/--no-affine-shape`, `--image-to-gray` and `--quantize-bits`.
None of them is a pick-images option, so they need no prefix, and each means
exactly what it means to `sfm ws init`. `sfm ws init --force` is not taken: it
would clash with pick-images' `--force`, and its two uses (re-initialising a
workspace, nesting one) do not apply to a new empty directory. The checks move
from the `init` click function in
[`_commands/ws.py`](../../src/sfmtool/_commands/ws.py) into one function that
both commands call. Without `--new-workspace`, each of these options, `--fps`,
`--max-frame-size` and `--prune-images` is a usage error naming
`--new-workspace`.

**Failure part-way.** A failure or Ctrl+C after the workspace is created leaves
`DIR` as it is, with the message naming it and saying that the run can be
repeated once `DIR` is deleted or emptied. Nothing is deleted on a failure. The
frames already extracted are still usable by hand (`sfm pick-images
DIR/<stem>` in the workspace).

### Pruning

`--prune-images` deletes, after pick-images has finished and written the list
and the JSON file when asked for, and only then:

- every frame this run extracted that pick-images did not choose, frames that
  `-r` left out of the input included, and
- the `.sift` file of each of those frames, at the path `sfm sift` gives it
  (`get_sift_path_for_image` in [`sift/file.py`](../../src/sfmtool/sift/file.py)),
  when one exists. pick-images writes these for frames it read or predicted it
  would read.

It never touches the chosen frames or their `.sift` files, the videos,
`.sfm-workspace.json`, `.sfm-named-image-lists.json`, the `video-frames.json`
records (which keep every extracted frame's time, so a kept frame's number
still finds its time) or any file the run did not write. Numbers of the kept
frames are left with their gaps, which the written list's ranges and
`sfm solve DIR/<stem>` both read. The report's last line says how many frames
and `.sift` files were deleted. A failed or cancelled pick deletes nothing.
Pruning frees disk after the run; the peak is still every extracted frame.

## Errors

Every check here runs before anything is created, for both commands, and an
error lists every offending input, not only the first:

- **No ffmpeg.** `ffmpeg` and `ffprobe` are found with `shutil.which`, the
  discovery `sfm version` reports ffmpeg by (`ffmpeg_facts` in
  [`_machine_report.py`](../../src/sfmtool/_machine_report.py)); the helper is
  shared, so the two cannot disagree. Missing either, the command stops: "sfm
  extract-frames needs ffmpeg and ffprobe on PATH, and found no ffprobe. Install
  ffmpeg 5.1 or newer (https://ffmpeg.org/download.html; `winget install
  Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, the `ffmpeg`
  package on Linux, or `ffmpeg` from conda-forge) and put the folder that holds
  both on PATH. `sfm version` shows the ffmpeg it finds."
- **`.insv`.** A file ending `.insv` (any letter case) is refused: "an Insta360
  `.insv` file holds two fisheye lenses; extract it with `sfm insv2rig`, which
  writes both lenses and their `.camrig`".
- **Not a video.** A directory, a missing file, or a file with an image
  extension the commands collect (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`,
  `.bmp`, which ffmpeg would open as a one-frame video) is refused. Any other
  file is probed with `ffprobe -show_streams -show_format -of json`; one that
  ffprobe cannot open, or that has no video stream other than an attached
  picture, is refused naming the file and the last line ffprobe printed.
- **Colliding names.** Two videos whose stems are equal ignoring letter case
  (`walk.mp4` and `walk.MOV`, or one file given twice) are refused, naming
  both, since they would share a directory; letter case is ignored because
  Windows and macOS do. A stem with a `*` is refused, since a named image
  list's pattern cannot hold it (pick-images refuses such a path).
- **Non-empty target.** As in [Directory rule](#directory-rule) and step 2
  above.
- **ffmpeg fails.** A non-zero exit during extraction stops the run with the
  video, the exit status and ffmpeg's last 20 lines of stderr; an
  "Unrecognized option 'fps_mode'" among them adds "this ffmpeg is older than
  5.1".

## HDR video

A phone's HDR video (HLG on iPhones, which also carries Dolby Vision
metadata on an HLG base; PQ on HDR10 recordings) is 10-bit BT.2020. Converting
it to 8-bit JPEG without tone mapping, as this command does, leaves HLG frames
somewhat flat and desaturated (HLG was designed to look acceptable on SDR
screens) and PQ frames dark and flat. SIFT works on intensity gradients and
still finds features in such frames, with fewer above its contrast threshold.
The command detects HDR from ffprobe's `color_transfer` and prints a warning on
the video's line and in the record, naming the transfer and giving the ffmpeg
command that converts the video to SDR first:

```bash
ffmpeg -i in.mov -vf zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p -c:v libx264 -crf 16 out.mp4
```

Tone mapping inside the command is left out of the first version: it needs
ffmpeg built with `libzimg` (`zscale`), which not every build has, or
`libplacebo`, which needs Vulkan, and the choice of operator changes what SIFT
sees. A later `--tone-map` option can add it, refusing an ffmpeg without
`zscale`.

## Implementation notes

**Why Python.** The work is ffmpeg's: decoding, filtering and JPEG encoding
happen in its process, and the command builds its arguments, reads its stderr
and checks its result, as `sfm insv2rig` does
([`rig/insv2rig.py`](../../src/sfmtool/rig/insv2rig.py)). A Rust version would
spawn the same process and parse the same text, and no Rust caller needs it, so
this is the pipeline case [STANDARDS.md § Logic lives in Rust](../STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers)
puts out of scope. The `Progress` standard is about library functions in Rust;
here the long call is the child process, so the Python function prints the
progress line and, on Ctrl+C, terminates ffmpeg (then kills it after 5 seconds
if it has not exited), waits for it, and re-raises, leaving the frames written
so far. `_run_ffmpeg` in `insv2rig.py` does not stop its child on Ctrl+C, and
the two can share the new runner.

**One extraction function.** `extract_video_frames(video, out_dir, *, fps,
max_frame_size, ffmpeg)` in `_video_frames.py` extracts one probed video and
returns the frame count and the record; `sfm extract-frames` and
`--new-workspace` both loop over it, so they cannot differ in naming, filters
or record. Probing (`probe_video`) is separate so both commands can probe
every input before creating anything. The module imports nothing heavy at
module level, so `_machine_report.py` can take the ffmpeg discovery from it.

**The size filter's expression** is
`scale='if(gte(iw,ih),min(iw,S),-2)':'if(gte(iw,ih),-2,min(ih,S))':flags=area`.
`force_original_aspect_ratio=decrease` with an `S×S` box would scale a smaller
video up, which this must not do.

**Floating point in the `select` expression.** `t*N` for a frame exactly on an
interval boundary can land just below an integer, putting that frame in the
interval before. That moves at most one frame to the next interval and never
keeps two in one, so the rule (at most one frame per interval, the earliest)
holds; the test checks counts within one of the expected value, not exact
times.

## Testing

`tests/test_extract_frames_command.py`, and additions to
`tests/test_pick_images_command.py`. The videos are made in the test with
ffmpeg's `lavfi` sources (`testsrc2`, a few seconds at 320x240) and its
built-in `mpeg4` and `mjpeg` encoders, so no codec library is needed; a pan
for pick-images is made by cropping a sliding window of a checked-in
photograph, as the existing pick-images tests do. ffmpeg is added to the
`test` and `test-ci` environments from conda-forge, so these tests run in CI
rather than skip.

- At `--fps 10` from a 30 fps video, a third of the frames, numbered from 1
  with no gaps, `%06d`, and their times in the record 0.1 s apart.
- A variable-rate video (timestamps rewritten with `setpts` and written with
  `-fps_mode vfr`): no duplicated frame (no two frames byte-identical), no
  two frames in one interval, continuous numbers.
- `--fps full` writes as many frames as `ffprobe -count_frames` reads.
- `--max-frame-size 160` on 320x240 gives 160x120; on a 100x80 video gives
  100x80; `full` keeps 320x240.
- A video with rotation metadata (`-display_rotation 90` when it is made) gives
  portrait frames and `rotation_degrees` in the record.
- A video tagged `arib-std-b67` warns and records the transfer.
- An audio track and a cover-art stream are ignored.
- Every refusal, with nothing created: no ffmpeg (`shutil.which` patched to
  return `None`), `.insv`, an image file, a directory, a non-video file,
  colliding stems in two cases, a stem with `*`, a non-empty target directory,
  and for `--new-workspace` a non-empty `DIR` and a `DIR` inside a workspace.
- `--new-workspace` writes the workspace with the given workspace options,
  extracts, and writes a list that `-r NAME` resolves to the chosen frames;
  the workspace options, `--fps` and `--prune-images` without
  `--new-workspace` and `--named-image-list` with it are usage errors.
- `--prune-images` leaves exactly the chosen frames and their `.sift` files,
  and the record, workspace and list files; a run that fails in pick-images
  (an invalid numeric option reaching the binding) deletes nothing.

## Non-goals

- Insta360 `.insv` and other multi-lens files; `sfm insv2rig` handles those.
- Camera intrinsics from video metadata. Phone videos rarely carry a focal
  length, and pick-images finds a camera by its pilot vote.
- Tone mapping HDR video ([HDR video](#hdr-video)).
- Hardware-accelerated decoding, and extracting several videos at once:
  ffmpeg already decodes with several threads.
- Resuming an interrupted extraction.

## Decisions for review

Each is this draft's choice, with the alternative.

1. **Command name `sfm extract-frames`.** A verb and its object, like
   `pick-images`, `embed-patches` and `cluster-patches`, and it says what it
   produces. Alternative: `sfm video2frames`, after `insv2rig` and `pano2rig`.
2. **Output option `-o/--output DIR`, required.** `sfm insv2rig` takes the same
   option for the same job. Alternative: `--into DIR`, which reads well but
   would be the only such option in the CLI.
3. **Category *Workspace*,** beside `insv2rig` and `pano2rig`. Alternative:
   *Image Processing*.
4. **Default `--fps 10`.** pick-images aims for a step of about a tenth of the
   image width and judges blur against the frames up to 2 positions away. A
   handheld walk at 30 fps moves about 0.5 to 1% of the width per frame
   ([pick-images-command.md § Report](../cli/image-feature/pick-images-command.md#report)
   shows 0.64%), so at 10 fps a step spans about 4 to 7 frames, leaving
   candidates to choose among and sharper neighbours to swap in, at a third of
   the disk and encoding of 30 fps and a sixth of 60 fps. Fast motion leaves
   fewer than 2 frames per step, for which `--fps full` (or a higher number) is
   the answer. Alternatives: 15 fps for more room on fast captures, or `full`
   as the default, giving pick-images every frame at the cost of disk.
5. **Default `--max-frame-size 1920`.** The `sfmtool` feature tool has no size
   cap and doubles the image before its first octave, so its cost grows with
   the frame's area; COLMAP caps at 4096 (`max_image_size` in
   [`sift/extract_colmap.py`](../../src/sfmtool/sift/extract_colmap.py)), so no
   tool needs a frame larger than 4096. At 1920 a 4K frame keeps a quarter of
   its pixels, which still holds the 8192 features a workspace extracts by
   default, and a JPEG at `-q:v 2` is about 0.3 to 0.6 MB instead of 1 to 3 MB:
   a 10-minute walk at 10 fps is 6,000 frames, about 2 to 4 GB instead of 6 to
   18 GB. Phone video at 4K is compressed hard enough that its finest detail
   is weaker than a 4K photograph's. The cost is precision: a solve on 1920
   frames has half the pixel resolution of one on 4K frames, and a user who
   wants it passes `full`. Alternatives: 3840, which scales only 5.7K and 8K
   video; 4096, COLMAP's cap.
6. **JPEG written by ffmpeg at `-q:v 2`, 4:2:0.** One process, no decode in
   Python. Alternative: decode raw frames into Python and write them with
   `write_image_rgb` at quality 95, as `insv2rig` writes its halves, at the
   cost of piping every frame through Python.
7. **Six-digit `frame_%06d.jpg` numbering from 1 inside `DIR/<stem>/`.**
   Matches `insv2rig`; lasts 9 hours at 30 fps full. Alternatives: seven digits;
   the stem in the file name (`walk1_%06d.jpg`), which makes a frame's name
   identify its video on its own.
8. **The `select` rule for a set rate,** not ffmpeg's `fps` filter, and
   `-fps_mode passthrough`, which need ffmpeg 5.1. Alternative: the `fps`
   filter, simpler but duplicating frames across gaps.
9. **Frame times recorded** in `DIR/<stem>/video-frames.json`, read by nothing
   in sfmtool yet. Alternative: leave times out; a variable-rate video's times
   are then lost once the frames are written.
10. **HDR converted without tone mapping, with a warning** and the command
    that converts first. Alternatives: refuse HDR video until tone mapping
    exists; tone-map with `zscale` when the ffmpeg has it, warning otherwise.
11. **The extract command's directory rule:** each video's own `DIR/<stem>/`
    must be missing or empty; the rest of `DIR` may hold anything.
    Alternative: refuse a non-empty `DIR`, as `--new-workspace` does, which
    stops adding a third video beside two extracted earlier.
12. **No workspace needed by `sfm extract-frames`.** Frames are plain files;
    the workspace matters only to the commands that extract features.
    Alternative: require one, as `insv2rig` does.
13. **Workspace options passed under `sfm ws init`'s own names,** valid only
    with `--new-workspace`, all but `--force`. Alternative: a `--ws-` prefix
    (`--ws-feature-tool`, `--ws-max-features`), which marks them as the
    workspace's but spells each differently from `sfm ws init`.
14. **`--new-workspace` refuses a `DIR` inside a workspace,** with no
    override. Alternative: a `--nested` flag.
15. **Frame options spelled the same on both commands** (`--fps`,
    `--max-frame-size`), the second spelled out so that among pick-images'
    options it cannot be read as a size of anything else.
16. **`--prune-images` also deletes frames `-r` left out,** since every
    extracted frame not chosen is unused. Alternative: delete only frames
    pick-images was given and did not choose.
17. **A failure after the workspace exists leaves `DIR` as it is.**
    Alternative: delete `DIR` on failure, since the run created everything in
    it, which would also delete hours of extraction on a late Ctrl+C.
