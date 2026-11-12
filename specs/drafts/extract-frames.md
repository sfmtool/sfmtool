# Extracting Frames from Video: `sfm pick-images --new-workspace`

**Status:** Draft, amending
[cli/image-feature/pick-images-command.md](../cli/image-feature/pick-images-command.md).
Its first part, `sfm extract-frames`, is built and filed as
[cli/workspace/extract-frames-command.md](../cli/workspace/extract-frames-command.md),
with the `video-frames.json` record in
[workspace/video-frames.md](../workspace/video-frames.md); this draft holds
the part not yet built, `sfm pick-images --new-workspace DIR VIDEO...` and its
`--prune-images`. Every choice is decided by the maintainer; the decisions are
listed in [Decisions](#decisions). When it ships, this content folds into
pick-images-command.md and this draft is deleted.

## Purpose

`sfm pick-images --new-workspace DIR VIDEO...` goes from videos to a solvable
set of frames in one call: it creates a workspace at `DIR`, extracts the
videos into it with the code `sfm extract-frames` uses, and chooses the frames
a solve should use, as `sfm pick-images` does for frames already on disk.
`--prune-images` then deletes the frames not chosen.

## `sfm pick-images --new-workspace`

```bash
sfm pick-images --new-workspace DIR VIDEO... [--fps N|full] [--max-frame-size PX|full]
    [--prune-images] [workspace options] [pick-images options]
```

With `--new-workspace DIR`, the positional arguments are video files, not image
folders. The command:

1. Checks the options and calls `plan_extraction`, which finds ffmpeg and
   ffprobe, probes every video and applies every check of
   [extract-frames-command.md § Checks](../cli/workspace/extract-frames-command.md#checks),
   creating nothing until all pass.
2. Refuses a `DIR` that exists and is not an empty directory, and a `DIR`
   inside an existing workspace (`sfm ws init`'s nesting rule, with no
   `--force` to override it here).
3. Creates the workspace at `DIR`, with `init_workspace` and the workspace
   options below, as `sfm ws init DIR` would, and prints its lines.
4. Extracts each video into `DIR/<stem>/` with `extract_planned`, the
   function `sfm extract-frames -o DIR` uses, at the same defaults, from the
   plan `plan_extraction` made in step 1.
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

Every check of
[extract-frames-command.md § Checks](../cli/workspace/extract-frames-command.md#checks)
applies, with the command named as `sfm pick-images --new-workspace` in its
messages, and these are added, all before anything is created, the error
listing every offending input:

- **Non-empty or nested `DIR`.** As in step 2 above.
- **Options.** The workspace options, `--fps`, `--max-frame-size` and
  `--prune-images` without `--new-workspace`, and `--named-image-list` with
  it, are usage errors.

## Testing

Additions to `tests/test_pick_images_command.py`, with videos made in the test
as `tests/test_extract_frames_command.py` makes them, and a pan for
pick-images made by cropping a sliding window of a checked-in photograph, as
the existing pick-images tests do, written as a video with ffmpeg.

- `--new-workspace` writes the workspace with the given workspace options,
  extracts, and writes a list that `-r NAME` resolves to the chosen frames;
  the workspace options, `--fps` and `--prune-images` without
  `--new-workspace` and `--named-image-list` with it are usage errors.
- A non-empty `DIR` and a `DIR` inside a workspace are refused, with nothing
  created.
- `--prune-images` leaves exactly the chosen frames and their `.sift` files,
  and the record, workspace and list files; a run that fails in pick-images
  (an invalid numeric option reaching the binding) deletes nothing.

## Non-goals

The non-goals of
[extract-frames-command.md § Non-goals](../cli/workspace/extract-frames-command.md#non-goals)
hold here too, HDR video among them.

## Decisions

Decided by the maintainer on review of this draft. Those that `sfm
extract-frames` needed are built as decided and described in its spec.

1. **Command name `sfm extract-frames`.** Decided; built.
2. **Output option `-o/--output DIR`, required.** Decided; built.
3. **Category *Workspace*,** beside `insv2rig` and `pano2rig`. Decided; built.
4. **Default `--fps 10`.** pick-images aims for a step of about a tenth of the
   image width and judges blur against the frames up to 2 positions away. A
   handheld walk at 30 fps moves about 0.5 to 1% of the width per frame, so at
   10 fps a step spans about 4 to 7 frames, leaving candidates to choose among
   and sharper neighbours to swap in, at a third of the disk and encoding of
   30 fps. Decided; built.
5. **Default `--max-frame-size 1920`.** At 1920 a 4K frame keeps a quarter of
   its pixels, which still holds the 8192 features a workspace extracts by
   default, and a 10-minute walk at 10 fps is about 2 to 4 GB of JPEG instead
   of 6 to 18 GB; a user who wants the full resolution passes `full`.
   Decided; built.
6. **JPEG written by ffmpeg at `-q:v 2`, 4:2:0.** Decided; built.
7. **Six-digit `frame_%06d.jpg` numbering from 1 inside `DIR/<stem>/`.**
   Decided; built.
8. **The `select` rule for a set rate and `-fps_mode passthrough`,** which
   need ffmpeg 5.1. Decided, with ffmpeg 5.1 or newer required: the version
   is checked and an older ffmpeg refused. Built.
9. **Frame times recorded** in `DIR/<stem>/video-frames.json`. Decided;
   built.
10. **HDR video refused** for now, detected from ffprobe's `color_transfer`,
    with a message saying to convert the video to SDR first. Converting
    without tone mapping, and tone mapping in the command, were not chosen;
    tone mapping is to be looked into separately. Built.
11. **The extract command's directory rule:** each video's own `DIR/<stem>/`
    must be missing or empty; the rest of `DIR` may hold anything. Decided;
    built.
12. **No workspace needed by `sfm extract-frames`.** Decided; built.
13. **Workspace options passed under `sfm ws init`'s own names,** valid only
    with `--new-workspace`, all but `--force`. Decided.
14. **`--new-workspace` refuses a `DIR` inside a workspace,** with no
    override. Decided.
15. **Frame options spelled the same on both commands** (`--fps`,
    `--max-frame-size`). Decided.
16. **`--prune-images` also deletes frames `-r` left out,** since every
    extracted frame not chosen is unused. Decided.
17. **A failure after the workspace exists leaves `DIR` as it is.** Decided.
18. **ffmpeg in the test environments.** conda-forge `ffmpeg` is a dependency
    of the `test` and `test-ci` environments, so the tests run in CI. Done.
