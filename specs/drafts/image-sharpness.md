# Image sharpness readings in `.sfmr`

**Status:** Draft. Decided by the maintainer: `.sfmr` version 15 with each
image's sharpness and gradient structure tensor as required per-image columns,
stored as `float32`, and a required `image_to_gray` record for them. The
choices this draft makes inside those decisions are listed in
[Choices made here](#choices-made-here) for review. The readings themselves are
implemented and filed: `.sift` version 2, which records them for every image
at extraction, is in
[`../formats/sift-file-format.md`](../formats/sift-file-format.md), with the
reading image, the sharpness and the gradient structure tensor defined there;
`.sfm-workspace.json` version 2, with the `image_to_gray` object, `sfm ws
upgrade` and the rule that versions never mix, is in
[`../workspace/workspace.md`](../workspace/workspace.md#versions). What remains
here is `.sfmr` version 15, which carries the readings into a reconstruction,
and the measurements behind the readings.

Some frames of a hand-held capture are blurred, by focus or by the camera
moving during the exposure, and a solve does better without them. Each image's
`.sift` file records two cheap numbers about it: a **sharpness**, the variance
of the Laplacian of a 480-pixel grey copy of the image, and a **gradient
structure tensor**, the mean products of that copy's horizontal and vertical
gradients. Both are read against the other frames of the same capture. This
draft carries them into the `.sfmr`, so that a reconstruction can be read and
filtered by them without its `.sift` files. Choosing images by these readings,
and naming the set chosen, is the subject of
[named-image-lists.md](../workspace/named-image-lists.md).

It amends these standing specs, each of which links back here:

- [`../formats/sfmr-file-format.md`](../formats/sfmr-file-format.md): version 15.
- [`../workspace/workspace.md`](../workspace/workspace.md): the readings a
  `.sfmr` written from a workspace carries.

## `.sfmr` version 15

`metadata.version` becomes `15`. Version 15 adds two required per-image
columns and a required `image_to_gray` record for them in
`images/metadata.json`. A `.sfmr` does not reference named image lists at all:
it records no list name and no list entries, and never embeds the workspace's
list file ([named-image-lists.md](../workspace/named-image-lists.md)). When this ships, the `.sfmr` spec defines the
`image_to_gray` object (its inputs, grammar and rounding), the reading image,
the sharpness and the tensor in its own words, repeating the `.sift` format's
definitions, because a format spec does not send its reader to another spec for
what a field means.

### `images/sharpness.{N}.float32.zst`

- **Shape**: `(N,)` where N = image_count.
- **Data type**: `float32` (little-endian).
- **Format**: image `i`'s sharpness, the population variance over the interior
  of the 3 × 3 Laplacian of its reading image ([the `.sift` definition](../formats/sift-file-format.md#the-reading-image), with the
  conversion recorded in `images/metadata.json`'s `image_to_gray`). Finite and
  not negative.
- **Required** from version 15 in both feature sources.

### `images/gradient_structure_tensors_xx_yy_xy.{N}.3.float32.zst`

- **Shape**: `(N, 3)` where N = image_count.
- **Data type**: `float32` (little-endian).
- **Format**: `[Exx, Eyy, Exy]` for image `i`, the means over the interior of
  `Gx²`, `Gy²` and `Gx·Gy`, with `Gx` and `Gy` the 3 × 3 Sobel derivatives
  divided by 8 of its reading image, `x` to the right and `y` down. Not
  normalised. Finite; `Exx` and `Eyy` not negative.
- **Required** from version 15 in both feature sources.

Both columns are for comparing the images of one reconstruction with each other,
and a reader is told so in the format text: they depend on scene content and are
not comparable across scenes.

The values are computed in double precision, as the [`.sift` format](../formats/sift-file-format.md#sharpness)
defines them, and a
writer stores each one rounded to the nearest `float32` (ties to even). A value
copied from a `.sift` file's `metadata.json`, where it is a JSON number holding
the double, is rounded the same way, so a row copied from a `.sift` and a row
computed from the image file by the same arithmetic are the same bytes.

### `images/metadata.json`: `image_to_gray`

A new required key holding the `{"formula": …, "quantize_bits": …}` object the
two columns were computed with, with the [`.sift` format's](../formats/sift-file-format.md#feature_optionsimage_to_gray)
grammar and rounding. One
conversion covers every row, because rows computed through different
conversions are not comparable. It rides inside `images/metadata.json`, which
`images_xxh128` already covers.

### Section hash

Both columns are folded into `images_xxh128` in their lexicographic slots:
`gradient_structure_tensors_xx_yy_xy` after `focal_adjustments` (after
`feature_tool_hashes` in a file without focal adjustments, after
`camera_indexes` in an `embedded_patches` file without them) and before
`image_file_hashes` and `metadata.json`; `sharpness` after
`rolling_shutter_rates_xyz` (after `radial_bspline_adjustments` or
`quaternions_wxyz` when those are absent) and before `sift_content_hashes` (in
an `embedded_patches` file, before `thumbnails_y_x_rgb` and
`translations_xyz`).

They are part of `content_xxh128`, not of the unidentifying `derived/` section,
for the reason thumbnails are: `derived/` holds values recomputable from the
file's other sections, and these are recomputable only from files outside it.
Two writers whose resize or arithmetic differs in the last bits can therefore
write files with different content hashes, as two thumbnail implementations would.

### Why `float32`, and these names

The data type follows from what the values are used for and the range they
take:

- **Use.** They are only compared within one capture: ranks, ratios to the
  capture's median (a frame below 0.6 × the median sharpness, say), z-scores,
  and the coherence and direction derived from the tensor. About three
  significant digits is enough for all of these.
- **Range.** With grey samples in `[0, 1]`, the Laplacian lies in `[−4, 4]`, so
  its variance is at most 16, and under the 1/8 Sobel scale each derivative is
  at most 1/2 in magnitude, so each tensor term is at most 1/4 in magnitude. In two phone
  videos the Laplacian variance on `[0, 1]` grey, at long side 480, had a
  median of 0.008 in one and 0.049 in the other, with 5th percentiles of 0.003
  and 0.019; dark or low-texture captures can be 10 to 100 times lower, and
  nearly flat images lower still. `Exy` is signed and can be near 0.
- **Not `float16`.** It has about 3.3 significant digits, enough for the use,
  but its normal values stop at about 6.1e-5, so the readings of a very flat or
  dark capture would fall into subnormals, with fewer digits, or underflow to 0,
  and lose their ranking exactly where blur detection matters.
- **`float32`** keeps about 7 significant digits and its normal range reaches
  about 1.2e-38, so no realistic reading loses precision or underflows.
- **Size does not decide it.** Four values per image are negligible next to the
  thumbnails, so the narrower type is chosen for being enough, not for the
  bytes it saves.

`sharpness` names what the column is for and has no natural plural, like
`normal_confidence` and `observation_confidence`. The tensor column takes the
plural, as `quaternions_wxyz` and `translations_xyz` do, and its suffix names
the component order, as `principal_point_adjustments_xy` and
`rolling_shutter_rates_xyz` do, since three numbers named only "tensor" do not
say which is the cross term.

### Where a writer gets the values

A conforming version 15 writer stores a row for every image, so it needs each
image's two readings under the file's one `image_to_gray`. It takes them, image
by image, from the first of these that applies:

1. **The input reconstruction**, when the file is made from a version 15 value
   that carries the row and records the same `image_to_gray`. A subset, a
   transform or a merge moves rows with their images.
2. **The image's version 2 `.sift` file**, when its `feature_options.image_to_gray`
   equals the file's and it is verifiably that image's: in a `sift_files` file,
   the `.sift` whose content hash `sift_content_hashes` records; in an
   `embedded_patches` file, a `.sift` at the workspace's `feature_prefix_dir`
   whose `image_file_xxh128` equals `image_file_hashes[i]`. The readings are
   copied from its `metadata.json`, the same way its thumbnail is copied.
3. **The image file**, decoded in stored pixel order and verified against the
   reconstruction's record of it (the `.sift`'s `image_file_xxh128` in a
   `sift_files` file, `image_file_hashes[i]` in an `embedded_patches` file),
   with the readings computed by the [`.sift` definition](../formats/sift-file-format.md#the-reading-image) through the file's
   `image_to_gray`. This is the path for a version 1 `.sift`, an
   `embedded_patches` file with no `.sift`, a `.sift` recorded under a
   different conversion, and an older `.sfmr` being saved as version 15.

When none applies, because the image file is missing, cannot be decoded or no
longer matches the recorded hash, the writer **fails** and writes nothing. Its
error names each such image and says what it looked for: the input's row, the
`.sift` path it tried and why that file was not used, and the image path with
the reason it could not be read. It never stores a placeholder, because a row in
a required column asserts that the image looks that sharp. Converters such as
`sfm from-colmap-bin`, which make a `sift_files` file from `.sift` files, take
step 2 where the `.sift` files are version 2 and step 3 otherwise.

The file's `image_to_gray` is the input reconstruction's when it records one;
otherwise `workspace.contents.feature_options.image_to_gray` when the workspace
record holds one; otherwise the one the `.sift` files record (the first image's version 2 `.sift`,
or, for version 1 files, the object the [`.sift` format's table](../formats/sift-file-format.md#what-each-feature-tool-records) gives for the workspace's
feature tool); and otherwise, in an `embedded_patches` file with no `.sift`
files, `{"formula": "0.2126*R + 0.7152*G + 0.0722*B"}`.

### Reading older files

A file of version 14 or earlier has neither column nor the `image_to_gray` key. A reader reports the readings as absent; it does not
compute them. Saving such a file as version 15 fills them by the rules above, so
an older file whose images are no longer available cannot be saved again until
they are.

### Verification

A version 15 file also fails verification when either column is missing, is not
`N` rows of its shape, or holds a value that is not finite; when `sharpness`,
`Exx` or `Eyy` is negative; when `images/metadata.json` has no `image_to_gray`
or its `formula` does not parse or its `quantize_bits` is not an integer from 1
to 16. A file below
version 15 that carries either column or key fails as well.

### Version 14 → Version 15 (for the migration section)

| Change | Detail |
|---|---|
| `images/sharpness.{N}.float32.zst` | New **required** column: each image's sharpness. Folded into `images_xxh128` after `rolling_shutter_rates_xyz` and before `sift_content_hashes` (`thumbnails_y_x_rgb` in an `embedded_patches` file). |
| `images/gradient_structure_tensors_xx_yy_xy.{N}.3.float32.zst` | New **required** column: each image's `[Exx, Eyy, Exy]`. Folded into `images_xxh128` after `focal_adjustments` (or `feature_tool_hashes`, or `camera_indexes`) and before `image_file_hashes` and `metadata.json`. |
| `images/metadata.json` `image_to_gray` | New required key: the conversion both columns were computed with. |

Migration is not mechanical. A version 14 file reads with no readings; saving it
as version 15 computes them from `.sift` files or image files, which changes
`images_xxh128` and `content_xxh128`, and fails when an image is unavailable.
A version 15 file maps back to version 14 by dropping both columns and the
`image_to_gray` key.

### Consequences for review

- A save of an older file, or of a value built without readings, reads `.sift`
  files or decodes photographs, so its cost rises from writing an archive to
  about one decode per image for files whose `.sift` files are version 1 (an
  earlier measurement decoded a 1920-pixel JPEG to grey in 5.8 ms on one
  thread).
- Saving fails where it succeeds today: an older `.sfmr` copied without its
  images, and a test fixture or synthetic reconstruction with no image files,
  can no longer be written until readings are supplied. In-memory constructors
  that build a reconstruction from arrays need a way to pass them in.
- `--minimal` keeps both columns: a minimal file holds the whole
  reconstruction, and the columns are 16 bytes per image of its identity, where
  thumbnails are optional and 49,152 bytes per image.

## Evidence for the two readings

The definitions come from measurements on two hand-held phone videos, 4K
sources from one phone type. Every sixth frame was graded by eye from a crop of
its most textured 240-pixel block: 0 sharp, 1 soft, 2 clearly blurred (182
graded frames of one video, 199 of the other). Unless a line says otherwise,
the numbers come from a sweep of the reading image's long side over 240 to 3840
pixels, each size area averaged from the 4K source frames' BT.709 grey, with z
the robust z-score of a frame against its whole video (the median and 1.4826
times the median absolute deviation). The scripts and frames are not in this
repository; when this draft is filed, the measurements go into a sibling
`-measurements.md` file. The reading image, the sharpness and the tensor they
support are filed in
[`../formats/sift-file-format.md`](../formats/sift-file-format.md#the-reading-image).

### What the readings catch

- **The Laplacian ranks the blurred frames of a capture.** At long side 480 its
  variance separated clearly blurred from sharp frames with an AUC of 0.858 on
  one video and 0.945 on the other (Spearman ρ with the grade 0.51 and 0.65).
  Flagging frames whose z is above 2 caught 32% and 27% of the clearly blurred
  frames and no sharp frame.
- **It depends on scene content.** A sharp frame 90% replaced by plain wall
  read z +8.6 and +6.4 at 480, more than a Gaussian blur of σ = 1.5 did (+1.2
  and +1.6). This is why the readings are only compared within a capture.
- **The tensor's log ratio sees blur along one axis that the Laplacian misses.**
  In the second video, frames 51 to 75 are smeared horizontally and graded
  clearly blurred. The Laplacian's median z over them was +0.3 at 480 and it
  flagged none of them. The reading that caught them is `|log(Exx/Eyy)|`
  relative to the capture's median: in an earlier measurement on OpenCV's
  reduced JPEG decode to a quarter size (long side 480, 8-bit BT.601 grey), the
  signed log ratio had a median z of −2.2 over the run, and its distance from
  the capture's median was beyond 2 in z for 64% of the frames, the same 64% at
  long side 960. The coherence and the tensor's distance from the capture's
  median normalised tensor caught 0% to 16% of that run at every size in the
  sweep.
- **Coherence and tensor distance add recall over all graded frames.** At 480,
  taking the larger of the Laplacian's z and the coherence's z raised the
  recall of clearly blurred frames from 32% and 27% to 35% and 44%, with no
  sharp frame flagged; with the tensor distance instead it was 42% and 44%,
  with 1% and 0% of sharp frames flagged.
- **Cost.** At 480 the resize, the Laplacian variance and the tensor took 7 ms
  on one thread from an in-memory 4K grey image in double precision.

### Why 480

- **Real blur ranks best at 360 to 640.** The Laplacian's AUC on the graded
  frames (first video / second video) was 0.848/0.948 at 360, 0.858/0.945 at
  480 and 0.851/0.939 at 640, and fell from 960 up: 0.841/0.920 at 960,
  0.838/0.879 at 1920 and 0.832/0.857 on the 4K source.
- **Synthetic blur and the blur that matters for matching rank better at larger
  sizes.** For synthetic copies blurred just enough to cost about 20% of a
  frame's verified matches ([below](#what-blur-costs-matching)), the AUC
  against untouched frames was 0.74/0.81 at 240, 0.90/0.89 at 480, 0.94/0.92 at
  640, 0.98/0.94 at 960 and 1.00/0.96 at 1920. The likely reason, not tested,
  is that synthetic blur also removes sensor noise and compression texture,
  which real blur keeps, so the larger sizes that see that texture separate
  synthetic copies more easily than real blurred frames.
- **Small sizes are pulled more by plain areas.** The sharp frame 90% plain
  wall read z +8.6 at 480 against +5.8 at 960 in the first video, where a
  Gaussian blur of σ 1.5 read +1.2 and +2.8.
- **Frame-to-frame jitter grows with size**: the median change between
  consecutive sharp frames, in units of the capture's spread, was 0.10/0.11 at
  480 and 0.16/0.13 at 1920.
- **The choice within the plateau.** 480 is the middle of 360 to 640; it divides
  3840 and 1920 by whole numbers, so area averaging is exact there; at 7 ms it
  is cheap; and it jitters less than 640 in the first video (0.096 against
  0.108) and about as much in the second (0.114 against 0.113).
- **One reading image for both values.** The coherence is slightly better at
  960: with the Laplacian at 480, the combined recall was 42% and 49% with the
  coherence at 960 against 35% and 44% at 480. That is two or three frames of
  about 40 clearly blurred frames per video, which does not justify a second
  image.

### What blur costs matching

One frame of each of 40 sharp pairs (20 per video) was blurred at the 4K source
and resized to 1920 pixels, and the sfmtool SIFT matches that survive geometric
verification against the unblurred neighbour were counted against the same
pair unblurred. In pixels of the 1920-pixel images SIFT ran on, the pooled
median loss reached 20% at a Gaussian σ of 1.43 px and 50% at 2.39 px, and 20%
at a linear motion of 5.5 px and 50% at 9.4 px. Blur of σ ≤ 1 cost a pooled
median of at most about 6% (16% in the first video at σ = 1), so the mild blur that small reading sizes miss mostly does not
matter for the solve.

### Measured and not chosen

- **Full resolution**: the Laplacian on the 4K source ranked worse (AUC 0.832
  and 0.857 in the sweep) and costs about ten times as much (78 ms against 7 ms).
- **Tenengrad** (the mean of the Sobel gradient energy): AUC 0.795 and 0.887 on
  the earlier quarter decode, and more sensitive to plain content than the
  Laplacian.
- **Sum-modified Laplacian**: AUC 0.852 and 0.932 on the earlier quarter
  decode, no better than the Laplacian and more sensitive to plain content.
- **ZNCC self-similarity on 24-pixel crops**: AUC 0.899 and 0.939 on the
  earlier quarter decode, but its best capture-relative rule flagged no blurred
  frame in the first video, and it took 14.9 ms a frame.

### Weaknesses

The evidence is two videos from one phone type, graded by one person, with
about 40 clearly blurred frames per video, so one frame is about 2.5 percentage
points of recall. Synthetic blur removes noise that real blur keeps. The
match-loss thresholds are interpolated between the blur levels measured, and
they are in pixels of the 1920-pixel images SIFT ran on.

## Implementation

Named once here, as the format specs' Implementations sections will name them:

- **`sfmtool-sfmr-format`** reads and writes version 15, and the `.sfmr`
  writers gather the readings as
  [Where a writer gets the values](#where-a-writer-gets-the-values) says: the
  Python path that copies thumbnails from `.sift` files
  (`src/sfmtool/colmap/io.py`) copies the readings beside them, and the Rust
  save path gains a lookup like `verified_sift_thumbnail` with the photograph
  as the fallback. The readings are computed, where a writer computes them, by
  `image_readings_from_rgb` in `sfmtool-core`, the function every `.sift`
  writer already calls.

## Choices made here

Decisions inside the maintainer's decisions that this draft makes, each with
where it is described and why, for review:

1. **`float32` in `.sfmr`** ([why](#why-float32-and-these-names)): three
   significant digits suffice and `float16` underflows on flat captures.
2. **One `image_to_gray` per `.sfmr`, rows under another recomputed**
   ([`images/metadata.json`](#imagesmetadatajson-image_to_gray),
   [sources](#where-a-writer-gets-the-values)): rows under different conversions
   do not compare.
3. **Readings in `content_xxh128`, not in `derived/`**
   ([section hash](#section-hash)): they are recomputable only from files
   outside the `.sfmr`.

The choices for the `.sift` readings and the workspace (EXIF orientation not
applied, readings in `metadata.json`, the tensor stored raw, long side 480 by
area average, interior pixels only, Sobel divided by 8, version checks at load,
no-mixing checked in every writer, `sfm ws upgrade`, `sfm ws init
--image-to-gray`, `sfm undistort` writing version 2, the viewer's in-memory
conversion, and the removal of `BuildSiftFilesOptions::from_workspace`) are
filed with their specs.

## Not in this draft

Any rule for which frames count as blurred: the readings are recorded so that
such a rule can be written, and it will be designed with the command that
writes named image lists ([named-image-lists.md](../workspace/named-image-lists.md)).

Decoding 16-bit sources at full depth, so that a 16-bit image's readings and
features follow the **Inputs** definition rather than an 8-bit reduction of it,
is future work.
