# `sfm estimate-intrinsics` Command

## Overview

Estimates a shared focal length and camera-model family for a set of images
from their cluster matches alone -- no reconstruction is run. The command is
the CLI face of the structure-free focal vote
([`focal-vote.md`](../../core/geometry/focal-vote.md)): image pairs vote
through whichever estimator their geometry can observe, the pooled log-median
is the focal consensus, and the camera-model columns arbitrate pinhole
against equidistant fisheye on the certified mass of model-informative scan
votes. Because no structure is estimated, the answer cannot be biased by the
depth/focal (bas-relief) compensation of structure-based estimation.

The output is a report; with `--write-camrig` the command also commits the
estimate to the workspace as a one-sensor `.camrig`
([camrig-file-format.md](../../formats/camrig-file-format.md)), the same
shape `sfm camrig create` writes -- which `sfm solve` auto-discovers as the
intrinsics prior for the images its stored pattern matches, taking
precedence over `camera_config.json`. With `--write-camrig-per-sequence` it
estimates each image sequence of the file from that sequence's images and
writes one `.camrig` per sequence beside it
([One camera per sequence](#one-camera-per-sequence)).

## Command Syntax

```bash
sfm estimate-intrinsics -i MATCHES [OPTIONS...]
```

| Option | Default | Description |
|---|---|---|
| `-i, --input MATCHES` | required | Cluster-bearing `.matches` file (from `sfm match --cluster`) |
| `-r, --range RANGE\|NAME` | all images | Estimate from only the file's images whose file number the range holds, or that are among a named image list's images, optionally with a slice (`'NAME[:30]'`); see [Selecting images](#selecting-images) |
| `--named-image-list NAME` | | The long form for a list; with `-r RANGE`, only its members whose frame number is in `RANGE` |
| `--model [auto\|pinhole\|fisheye]` | `auto` | Which camera-model columns to run: `auto` votes pinhole-only and adds the second column, arbitrating between them, only when that vote comes back weak; the named forms run one column and skip arbitration |
| `--write-camrig PATH` | off | Write the estimate as a one-sensor `.camrig` at PATH (see below) |
| `--pattern PATTERN` | derived | Image pattern stored in the `.camrig`; defaults to the selected images' common directory plus one `*` segment per directory level below it, ending `*<ext>` for the extension their names share (`*` when they mix), narrowed to `<prefix>*<ext>` when it would match images `-r` left out (see [`--write-camrig`](#--write-camrig)) |
| `--write-camrig-per-sequence` | off | Write one one-sensor `.camrig` per image sequence of the file, each estimated from that sequence's images only, in the sequence's folder (see [One camera per sequence](#one-camera-per-sequence)); not with `--write-camrig` or `--pattern` |
| `--force` | off | Allow `--write-camrig` or `--write-camrig-per-sequence` to overwrite existing files |
| `--json` | off | Emit the full vote result as JSON on stdout instead of the report |
| `--seed N` | `0` | RANSAC / pair-sampling seed; same inputs + seed + `--draws` give bit-identical output |
| `--draws N` | `1` | RANSAC draws behind each pair's vote, at least 1 (see [Draws per pair](#draws-per-pair)) |

## Behavior

The command reads the cluster tracks and image dimensions from the
`.matches` file and hands them to the
`sfmtool.geometry.estimate_intrinsics` binding
([estimate-intrinsics.md](../../core/geometry/estimate-intrinsics.md)) with
the column policy implied by `--model` -- the escalating `"auto"` policy, or a
named single column; the verdict, its confirmation, whether the camera-model
columns ran and the focal are read off that result, and the raw vote it nests
under `vote` supplies the report's diagnostics. Without `-r` the whole
file's admission votes: every cluster, on every image, with the members the
format's default selection admits.

The vote assumes one shared camera with a centred principal point. When the
selected images carry more than one distinct `(width, height)`, the command
refuses up front with a message naming the differing dimensions; select one
camera's images with `-r`, or split the images into per-camera match files,
and estimate each.

### Selecting images

`-r` and `--named-image-list` choose which of the file's images the vote runs
on, with exactly the values `sfm solve -r` takes with a `.matches` file and
the same resolution
([solve-command.md § From a `.matches` file](solve-command.md#from-a-matches-file),
[the shared `-r/--range` option](../README.md#the-shared--r--range-option)):

- **A range** keeps the file's images whose file number it holds.
- **A named image list** keeps the file's images that are among the list's
  images. The list is read from the workspace the `.matches` file resolves
  to, with its range, its slice and the `.png` / `.jpg` / `.jpeg` extension
  test applied to the list's members before the file's images are matched
  against them, so `'NAME[:30]'` is the same 30 images here as in
  `sfm solve`. The list's images the file does not hold are ignored, and
  their count is printed on stderr, so `--json` stays parseable.
- A selection that keeps none of the file's images, a malformed value and a
  name the workspace does not define are errors, with the messages `sfm solve`
  gives.

The vote then runs on `select_clusters(restrict_images=...)` of the file
([cluster-selection.md](../../core/features/cluster-selection.md)): only the
members on the selected images vote, and a cluster left spanning fewer than
two of them drops. The report's first line says how many images were
selected, of how many, and by what (`Images:  132 of 160 selected (named
image list 'video1') @ 1080x1920`). A selection whose clusters all drop, such
as a single image, is not an error: the report prints with no clusters and no
consensus, and the command exits 0.

This is what lets one `.matches` file serve a capture of several cameras.
`sfm match --cluster` over every image of the capture finds the matches the
joint solve needs, across the cameras as well as within each, and
`estimate-intrinsics -r` then estimates each camera from its own images in
that file. Without `-r` the vote pools the pairs of every camera into one
focal, which is neither camera's focal when their lenses differ.

`-r` has no form that names a folder. A named image list does: an entry
whose pattern is a folder's glob, such as `{ "pattern": "video1/*.jpg" }`,
selects that folder's images, and a pattern such as `images/video1_*.jpg`
selects one video's frames from a shared folder
([named-image-lists.md](../../workspace/named-image-lists.md#the-file)).

```bash
# One cluster match over both videos' frames, for the joint solve.
sfm match --cluster -r both -o matches/both-clusters.matches
# One .camrig per video, each from its own frames in that file.
sfm estimate-intrinsics -i matches/both-clusters.matches -r video1 \
    --write-camrig video1.camrig
sfm estimate-intrinsics -i matches/both-clusters.matches -r video2 \
    --write-camrig video2.camrig
```

with `.sfm-named-image-lists.json` defining `both`, `video1` and `video2`:

```json
{
  "version": 1,
  "named_image_lists": {
    "both": [
      { "pattern": "video1/*.jpg" },
      { "pattern": "video2/*.jpg" }
    ],
    "video1": [
      { "pattern": "video1/*.jpg" }
    ],
    "video2": [
      { "pattern": "video2/*.jpg" }
    ]
  }
}
```

Written in the workspace root, the two `.camrig` files store the patterns
`video1/*.jpg` and `video2/*.jpg`, so `sfm solve` gives each video's frames
their own camera.

### Draws per pair

`--draws` is passed to the binding as `draws`. Each pair's vote reads that
many RANSAC fits rather than one: draw `k` runs the pair's estimators at seed
`seed + k`, and the pair votes the log-space median of its draws' focals when
more than half of the draws cast a vote, and casts no vote otherwise (see
[Draws per pair](../../core/geometry/focal-vote.md#draws-per-pair) in the
focal-vote spec). The default, `--draws 1`, is the single-draw vote, the
same as the binding's default. More draws make the focal depend less on
`--seed`, and the vote's run time grows with the number of draws. The output
is a function of the inputs, `--seed` and `--draws` together, and a rerun
with the same three is bit-identical. Neither the binding's result nor the report echoes the seed or
the draw count, so `--json` carries no `draws` key.

### When the camera-model columns run

`--model auto` does not mean "always run both columns". The kernel votes
pinhole-only first and pays for the camera-model scans exactly when that vote
comes back weak -- no consensus, a railed rotation family, family
disagreement, or a thin pool (see
[estimate-intrinsics.md](../../core/geometry/estimate-intrinsics.md)). The
report states which way it went, and on a weak vote names the reasons that
fired; `--json` carries the same record as `escalation`. A capture whose
pinhole vote stands therefore has no Columns block and no confirmation
reading: there was no arbitration, and its verdict is Pinhole by
construction.

### The model verdict and its confirmation

Where the columns did run, the binding's verdict (the column with the greater
certified mass of model-informative scan votes) is reported together with a
**confirmation** reading: a Fisheye verdict is CONFIRMED only when the
equidistant column carries nonzero certified rotation-cell mass. The rule is
structural rather than a threshold -- a wrong ray map cannot fake a pure
rotation of rays, so a fisheye verdict without rotation-cell mass is an
arbitration artifact, not a lens. Both the verdict and its confirmation come
from the estimate; the command holds no rule of its own and the kernel runs
once.

An unconfirmed Fisheye verdict still prints, marked UNCONFIRMED, and the
report recommends treating the capture as pinhole.

### Report

The human-readable report leads with the answer and keeps the diagnostics
below it:

- model verdict (with CONFIRMED / UNCONFIRMED under `auto`), the winning
  column's consensus focal in pixels, and the implied diagonal field of view
  under that model's map;
- vote counts (`n_epipolar`, `n_rotation`, `n_pool`), the pool spread, and
  the family disagreement where both families voted;
- under `auto`, whether the camera-model columns ran and, on a weak vote, the
  reasons that made them worth running;
- where they did run, one line per column with its own consensus focal,
  spread, and certified epipolar / rotation masses.

`focal_px = None` (fewer than 2 pooled votes) reports as "no consensus" with
the rejection counters (`n_h_dominated`, `n_estimator_failed`,
`n_band_rejected`), so the user can see whether the capture is match-starved
or parallax-poor. No consensus is still exit code 0 -- the report is the
product; only I/O failures and rejected inputs are errors.

`--json` emits the vote dict verbatim at the top level, for scripting, plus
the image counts, `image_count` (the file's images) and
`selected_image_count` (the images the vote ran on, all of them without
`-r`), and six keys the estimate supplies: `fisheye_confirmed`,
`certified_rotation_mass`, `diagonal_fov_deg`, `escalation` (the weak-vote
reasons that made the columns worth running, `[]` when they did not run, and
`null` under a named `--model`), `screening_vote` (the pinhole-only vote dict
the escalation was decided on, present when the columns ran under `auto` and
`null` otherwise; after an escalation the top-level keys describe the
two-column run, so the pinhole consensus, pool and spread are read here), and
`verdict_votes` (the
winning column's certified scan votes -- the evidence behind this verdict,
which the top-level `epipolar_votes` / `rotation_votes` are not, those always
describing the pinhole closed-form kernel). The vote's keys stay at the top
level rather than nesting under `vote` as the binding does: they are already
the payload, and nesting them as well would duplicate every one.

### `--write-camrig`

Writes a one-sensor `.camrig` whose camera is the verdict model
(`SIMPLE_PINHOLE` or `EQUIDISTANT_FISHEYE`) at the consensus focal, the
selected images' `width` / `height`, and the centred principal point, with
identity sensor extrinsics -- the same shape `sfm camrig create` produces. The
stored image pattern is `--pattern` when given, else derived from the
selected images' names: their common directory, one `*` segment per directory level
between it and the names (a glob's `*` does not cross `/`, so a rig's
per-sensor subdirectories need their own segment), ending in `*<ext>` for
the extension the names share -- so a flat layout's derived pattern does
not sweep up the workspace's own files. Names at mixed depths cannot be one
glob; the derivation then keeps a single level and validation hands the
caller a message naming `--pattern`. The pattern is validated the way
`camrig create` validates one.

The pattern never matches an image of the file that `-r` left out: the
`.camrig` would give that image a camera estimated without it, and
`sfm solve` would use it. The images a pattern matches are found under the
`.camrig`'s folder, as `sfm solve` finds them, and compared with the left-out
images by path. When the derived pattern matches one, it is narrowed: its
final glob starts with the common prefix of the selected names' stems (each
file name less its extension), cut before its first `*`, `?` or `[` (`*` is
always a wildcard, and the filesystem walk that expands a pattern reads `?`
and `[` as wildcards too), less the digits it then ends in, and with each `%`
written `%%`, the pattern language's literal percent. So selected frames
`images/shopA_00001.jpg` ... `images/shopA_01089.jpg` beside
`images/shopB_*.jpg` give `images/shopA_*.jpg`, and frame 1 of a rig,
`rig/cam0/f_01.jpg` and `rig/cam1/f_01.jpg`, gives `rig/*/f_*.jpg`. When
the narrowed pattern still matches a left-out image, or `--pattern` matches
one, the write refuses, naming the pattern, one such image and `--pattern`. Images the file does not hold are not checked: a
video's frames that were not picked for matching are still that video's, and
a pattern covering them is what a later solve on more of its frames needs.
Without `-r` no image is left out, so the derived pattern is never narrowed.

A `SIMPLE_PINHOLE` camera written here has no distortion term, so a solve
that starts from it as written cannot model a lens's radial distortion.
`sfm solve --camera-model SIMPLE_RADIAL` (or another perspective model with
distortion terms) starts instead from this camera refit to that model, at
the same focal and principal point with zero distortion, which the solve
then refines; the `.camrig` is not changed. An `EQUIDISTANT_FISHEYE` camera
written here reaches COLMAP as `SIMPLE_RADIAL_FISHEYE` with zero
distortion, so a solve that starts from it as written already refines a
radial term, and returns a `SIMPLE_RADIAL_FISHEYE` camera; `--camera-model`
with another fisheye model, such as `RADIAL_FISHEYE`, starts from more
terms ([solve-command.md § Camera Intrinsics](solve-command.md#camera-intrinsics)).

The command prints a summary of the written file after the report: the path,
the camera, the focal and the stored pattern. With `--json` the summary goes
to stderr, so stdout holds the JSON alone.

The write refuses -- with the report still printed -- when:

- there is no focal consensus;
- the verdict is an UNCONFIRMED Fisheye (the message says to re-run with
  `--model pinhole` to commit the pinhole estimate instead);
- the output file already exists and `--force` was not given;
- the pattern matches an image of the file that `-r` left out.

### One camera per sequence

`--write-camrig-per-sequence` gives each video of a capture its own camera
in one run, without a named image list per video. The command groups the
file's images into **sequences** exactly as `sfm pick-images` does
([pick-images-command.md § Collecting the frames](../image-feature/pick-images-command.md#collecting-the-frames)):
the files of one folder whose names are the same apart from the file number
`number_from_filename` reads. The vote then runs once per sequence, on
`select_clusters(restrict_images=...)` of that sequence's images, so each
estimate is the one `-r` gives for the same images, and the sequence's
`.camrig` holds it, in the one-sensor shape
[`--write-camrig`](#--write-camrig) writes. Both layouts of a capture of two
videos work:

```bash
# A name prefix per video, in one folder:
#   images/shopA_00001.jpg ... images/shopB_00001.jpg ...
sfm estimate-intrinsics -i matches/clusters.matches --write-camrig-per-sequence
# writes images/shopA.camrig (pattern shopA_%05d.jpg)
#    and images/shopB.camrig (pattern shopB_%05d.jpg)

# A folder per video:
#   shopA/00001.jpg ... shopB/00001.jpg ...
sfm estimate-intrinsics -i matches/clusters.matches --write-camrig-per-sequence
# writes shopA/shopA.camrig (pattern %05d.jpg)
#    and shopB/shopB.camrig (pattern %05d.jpg)
```

`sfm solve` over the whole capture then gives each sequence's images the
camera of its `.camrig`
([solve-command.md § Camera Rig Files](solve-command.md#camera-rig-files-camrig)).

**Where each file goes.** The `.camrig` of a sequence is in the sequence's
folder. Its name is the part of the sequence's file names before the file
number, less the separators it ends in (spaces, `_`, `-`, `.` and `(`), so
`shopA_00001.jpg`, `shopA-00001.jpg` and `shopA (1).jpg` all give
`shopA.camrig`. When nothing is left, as for `00001.jpg`, the file is named
after the folder, and a sequence in the workspace root is named after the
workspace's folder.

**The pattern.** The stored pattern is relative to the `.camrig`'s folder:
the sequence's file names with the file number as a frame field, `%0Nd`
when every number of the sequence is written with the same `N > 1` digits
and `%d` otherwise, and `%%` for a `%` in a name. It is the sequence's list
pattern in `sfm pick-images` without its folder. So it covers every file of
the sequence in the folder, including frames the `.matches` file does not
hold, and no file whose name differs anywhere else: `shot1_%04d.jpg` does
not cover `shot10_0001.jpg`.

**`-r`.** With `-r` only the selected images take part. A sequence with no
selected image is left out of the run; each other sequence votes on its
selected images, and its pattern still names the whole sequence, since the
frames `-r` left out are the same video's.

**Rigs.** The images a multi-sensor `.camrig` covers, found as `sfm solve`
finds it, form one rig sequence, as in `sfm pick-images`. It gets no file,
since its `.camrig` already gives its images their cameras, and the report
says so. A rig of `rig_config.json` among the file's images is refused
before any vote, whatever `-r` selects, with a message naming the sequence:
`sfm solve` ignores `rig_config.json` once any `.camrig` covers an image it
solves
([solve-command.md § Camera Rig Files](solve-command.md#camera-rig-files-camrig)),
so writing any sequence's file would undo the rig. Images that cannot be
grouped (a file with no number, two files of one sequence with the same
number) stop the command, with a message naming the files at fault, as they
stop `sfm pick-images`.

**Refusals.** A sequence gets no file, with the reason in the report, when:

- its vote has no focal consensus, or its verdict is an UNCONFIRMED Fisheye
  (the same two refusals as `--write-camrig`);
- its images have more than one resolution (the message lists them);
- another sequence of the run would get the same file, compared without
  regard to letter case, since the file systems of Windows and macOS do not
  tell `x.camrig` from `X.camrig`. Every such sequence gets none, and the
  message names the others and says to write each one with `-r` and
  `--write-camrig`. `x_0001.jpg` beside `x_0001.png` is one such case;
- a single-sensor `.camrig` other than its own already covers any of its
  images, since `sfm solve` refuses an image two `.camrig` files match;
- its file exists and `--force` was not given;
- its pattern matches an image of the `.matches` file in another sequence,
  as a matcher that ignores letter case can; the message names the pattern
  and one such image.

The other sequences' files are still written. The command exits 0 when it
writes at least one file, and 1, after the report, with
`no .camrig was written.` when it writes none.

**Report.** One block per sequence, in the order of the sequences'
patterns: `Sequence <pattern> (<N> images)`, the sequence's report as above,
then the summary of the written file (its path, camera, focal and stored
pattern) or `No .camrig: <reason>`.

**`--json`** prints one object:

- `workspace_dir`, `image_count` (the file's images) and
  `selected_image_count` (the images `-r` keeps, all of them without `-r`);
- `sequences`, one entry per sequence of the run: `pattern` (the
  workspace-relative list pattern, a rig sequence's patterns joined by
  ` and `), `image_count` (the file's images in the sequence),
  `selected_image_count`, `camrig_path` (workspace-relative, `null` for a
  rig sequence), `camrig_pattern` (the stored pattern, `null` for a rig
  sequence), `written`, `reason` (`null` when written), and `estimate`, the
  payload `--json` prints for a single estimate, here of the sequence's
  images, or `null` when no vote ran.

The report blocks then go to stderr, so stdout holds the JSON alone.

## Implementation Notes

- Registered under the Reconstruction category in `cli.py`; source in
  `src/sfmtool/_commands/estimate_intrinsics.py`.
- `--write-camrig-per-sequence` groups the images with `find_rigs` and
  `collect_sequences` in
  [`_pick_images.py`](../../../src/sfmtool/_pick_images.py), the functions
  `sfm pick-images` uses, so the two commands' sequences are the same.
- The `.matches` read goes through `sfmtool.fileio.MatchesFile`. The
  command passes the selection handle itself (`select_clusters()`,
  restricted to the `-r` images when given) to the kernel's object form,
  which reads the backbone's own CSR index, member images and member
  positions off it; the command derives nothing from those arrays but the
  cluster and observation counts the report prints.
- Keypoint positions come from the file's `member_positions` — the backbone's
  own array, whose content is whatever stage the file is at: the detections of
  a matcher output, the refinement's answer in a cluster-patches one. There is
  no other source and no fallback; a `.matches` file predating format version 6
  is refused by the reader, naming its regeneration. The values are stored
  float32 and widened by the kernel because the vote solves in float64. Member
  exclusion is the default selection's job, by status — the same rule the
  refined path has always used.
- `-r` is resolved by `matches_image_filter` in
  [`_range_options.py`](../../../src/sfmtool/_commands/_range_options.py),
  the function `sfm solve` filters a `.matches` file's images with, and the
  workspace is the opened `MatchesFile`'s `workspace_dir`, the one
  `read_matches` returns to `sfm solve` and `sfm cluster-patches`, so the
  file is read once.
- The shared-camera rule is the kernel's, and the command checks it first
  anyway, on the selected images: both refuse images of more than one
  resolution, and going through the command's own check is what turns the
  refusal into a message listing the resolutions found and an example image
  for each.
- An explicit `--model pinhole` runs no scan, so its result carries no
  per-column block (the closed-form kernel is the whole answer); the same is
  true of an `auto` run whose pinhole vote stood. The report's Columns
  section appears only where a column actually scanned.
- The fisheye diagonal FOV is `2 * theta(r_corner)` under `theta = r / f`;
  the pinhole one is `2 * atan(r_corner / f)`. Both are reported from the
  same corner radius `hypot(width, height) / 2`.
- A real fisheye lens is generally not exactly equidistant, so a calibrated
  reference under a richer distortion model can sit a few percent from this
  estimate. That gap is a property of the estimate's model, not an estimator
  error; the `.camrig` this command writes is the equidistant best-fit,
  which is what the solver-side model can represent.
