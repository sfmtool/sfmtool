# Image sharpness readings

**Status:** Draft. Decided by the maintainer: `.sift` version 2 with a required
`image_to_gray` object (with an optional `quantize_bits`), a per-image
sharpness and a per-image gradient structure tensor; `.sfmr` version 15 with
both as required per-image columns, stored as `float32`; `.sfm-workspace.json`
version 2 carrying the same `image_to_gray` object as the `.sift` file, with
the recorded features folder name authoritative; each feature tool recording
the formula closest to its own grey conversion. The choices this draft makes
inside those decisions are listed in [Choices made here](#choices-made-here)
for review. Nothing here is implemented.

Some frames of a hand-held capture are blurred, by focus or by the camera
moving during the exposure, and a solve does better without them. This draft
gives every image two cheap numbers, recorded when its features are extracted:
a **sharpness**, the variance of the Laplacian of a 480-pixel grey copy of the
image, and a **gradient structure tensor**, the mean products of that copy's
horizontal and vertical gradients, from which a reader gets how strongly the
gradients favour one direction and which direction that is. Neither number
means anything across scenes; both are read against the other frames of the
same capture, where a frame much less sharp than the capture's median, or whose
gradients favour one direction more than the median frame's do, is a blurred
frame. Choosing images by these readings, and naming the set chosen, is the
subject of [named-image-lists.md](../workspace/named-image-lists.md).

It amends these standing specs, each of which links back here:

- [`../formats/sift-file-format.md`](../formats/sift-file-format.md): version 2
  (Part 1).
- [`../formats/sfmr-file-format.md`](../formats/sfmr-file-format.md): version 15
  (Part 2).
- [`../workspace/workspace.md`](../workspace/workspace.md): `.sfm-workspace.json`
  version 2, whose `feature_options` carry the `image_to_gray` object (Part 1).
- [`../core/features/sift.md`](../core/features/sift.md) § "Image-to-gray
  conversion", which points here for the structured `image_to_gray` object.

It also renumbers the incremental-extraction proposal: the growable `.sift`
archive in
[`sift-incremental-extraction-amendment.md`](sift-incremental-extraction-amendment.md)
becomes version 3, built on version 2 as defined here, and the `image_to_gray`
object it used to carry is defined here instead.

## Part 1: `.sift` version 2

`metadata.version` becomes `2`. Version 1 files stay valid and readable; a
reader of a version 1 file finds no `image_to_gray`, sharpness or tensor in it.
A conforming writer writes version 2. Extraction never writes a version 2 file
into a directory that holds version 1 files: a version 2 file goes in the new
features directory of a version 2 workspace
([below](#the-workspace-records-the-same-object-sfm-workspacejson-version-2)).
A `.sfmr` writer that meets a version 1 file handles it as
[Part 2](#where-a-writer-gets-the-values) says.

### Summary of differences

| Aspect | version 1 | version 2 |
|--------|-----------|-----------|
| `metadata.version` | `1` | `2` |
| `feature_options.image_to_gray` | absent; the conversion is the tool's own, and the `sfmtool` tool records it as a `gray_formula` string | **required** for every feature tool: the formula and an optional `quantize_bits` |
| `metadata.sharpness` | absent | **required**, a number |
| `metadata.gradient_structure_tensor_xx_yy_xy` | absent | **required**, an array of three numbers |
| Entries, entry names, content-hash order | as listed | unchanged |

### `feature_tool_metadata.json`: `feature_options.image_to_gray`

SIFT and the readings below operate on a single-channel image, and several of
SIFT's parameters, most importantly the contrast threshold, are defined in that
image's value domain. The mapping from the stored image to the grey samples
therefore changes the features, the meaning of the thresholds and the
readings, so a version 2 file records it, for every feature tool, as
`feature_options.image_to_gray`. Being inside `feature_options`, it is part of
`feature_tool_xxh128`.

```json
"image_to_gray": { "formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8 }
```

* `formula` (string, required): an arithmetic expression whose value is the
  grey sample, before any quantisation.
* `quantize_bits` (integer from 1 to 16, optional): when present, the formula's
  value is rounded to this many bits over `[0, 1]`. When absent, the formula's
  value is the grey sample as it is.

**Inputs.** `R`, `G` and `B` are the image's red, green and blue samples, each
normalised to `[0, 1]` by dividing the stored sample by its full-scale value,
`2^d − 1` for `d`-bit samples (255 for 8-bit, 65535 for 16-bit), so a full-scale
sample is exactly `1`. For a single-channel image,
`R = G = B`. An alpha channel is ignored. The pixels are the image's stored
pixels in their stored order: an EXIF orientation tag is **not** applied, so a
keypoint position, a thumbnail pixel and the grey image all address the same
pixel whatever tag the file carries.

**Grammar.** The formula is an expression over the variables `R`, `G`, `B` and
decimal numeric literals, with three operators and parentheses for grouping:

| Operator | Meaning | Precedence | Associativity |
|----------|---------|------------|---------------|
| `**` | power | highest | right |
| `*` | multiply | middle | left |
| `+` | add | lowest | left |

Whitespace is insignificant. A leading `-` on a numeric literal makes a
negative coefficient; there is no subtraction or division operator. The formula
is evaluated in IEEE-754 double precision.

**Output.** Without `quantize_bits`, the grey sample is the formula's value `v`,
not clamped. With `quantize_bits = b`, it is

```
q(v) = round(clamp(v, 0, 1) · (2^b − 1)) / (2^b − 1)
```

where `clamp(v, 0, 1)` is `v` limited to `[0, 1]` and `round` takes the nearest
integer, with an exact half rounded up. Since the argument is never negative,
rounding half up and rounding half away from zero agree. The quantised sample is
a multiple of `1 / (2^b − 1)` in `[0, 1]`; `quantize_bits: 8` expresses the
8-bit grey image a tool makes before it runs SIFT.

Value-domain parameters such as the contrast threshold assume samples on a
`[0, 1]` scale, so a formula keeps typical values in that range. A tool that
works internally in another scale (0 to 255, say) and scales its thresholds to
match still records the conversion on the `[0, 1]` scale.

Examples:

| `formula` | `quantize_bits` | Effect |
|-----------|-----------------|--------|
| `0.2126*R + 0.7152*G + 0.0722*B` | absent | BT.709 luma, unquantised |
| `0.2126*R + 0.7152*G + 0.0722*B` | `8` | BT.709 luma rounded to 8 bits |
| `0.299*R + 0.587*G + 0.114*B` | `8` | BT.601 luma rounded to 8 bits |
| `G` | absent | the green channel |
| `0.2126*R**2.2 + 0.7152*G**2.2 + 0.0722*B**2.2` | absent | weights applied after a gamma-style exponent |

A geometric resize the tool applies before detection, such as a cap on the
image size, is a separate concern with its own `feature_options` key and is not
part of `image_to_gray`.

### `metadata.json`: `sharpness` and `gradient_structure_tensor_xx_yy_xy`

Two new required fields, each a reading of the image the file was extracted
from, made from the same decoded pixels as the thumbnail and through the grey
conversion that `image_to_gray` records:

* `sharpness` (number): the variance of the Laplacian of the reading image,
  defined in [The reading image](#the-reading-image) and
  [Sharpness](#sharpness). Finite and not negative.
* `gradient_structure_tensor_xx_yy_xy` (array of three numbers):
  `[Exx, Eyy, Exy]`, the means of `Gx²`, `Gy²` and `Gx·Gy` over the reading
  image, defined in [Gradient structure tensor](#gradient-structure-tensor).
  Finite; `Exx` and `Eyy` are not negative.

"Finite" is a rule on the writer as well as on the file. An unclamped formula
can produce a NaN or an infinity (`R**-1` at `R = 0` without `quantize_bits`,
for example), and a non-finite grey sample makes the readings non-finite. A
writer whose readings come out non-finite fails, with an error naming the image
and the formula, and writes no file; it does not store the value or replace it.

Both are JSON numbers holding double-precision values, written so they read
back as the same doubles. Both are for comparing the images of one capture
with each other. They depend on what the scene holds (a frame that is mostly
plain wall has a low sharpness however well it is focused), so they are not
comparable between captures of different scenes, and the format gives no
threshold for either.

`metadata.json` is written once and never changes, so the readings are fixed
when the file is written, like the image's hash and size beside them. They
reach `metadata_xxh128` and `content_xxh128` through `metadata.json`'s bytes,
and the content-hash order of the entries is unchanged.

#### The reading image

Both readings are taken on one grey image, the **reading image**, built from the
decoded image of width `W` and height `H` in stored pixel order:

1. **Grey.** Each pixel's grey sample is `image_to_gray` applied to it: the
   formula, then `q` when `quantize_bits` is present. The samples are held in
   double precision from here on.
2. **Size.** The long side 480 is chosen as [Why 480](#why-480) says. Let
   `L = max(W, H)` and `S = min(W, H)`. If `L ≤ 480`, the grey
   image is used at its own size; it is never enlarged. Otherwise the long side
   becomes `480` and the short side becomes
   `S' = max(1, floor((2 · S · 480 + L) / (2 · L)))`, which is `S · 480 / L`
   rounded to the nearest integer with an exact half rounded up, computed in
   integers. Which side is long is kept: a portrait image stays portrait.
3. **Resize.** Each pixel of the `W' × H'` reading image is the area average of
   the grey image over its footprint: treating the grey image as constant over
   each unit pixel square, output pixel `(i, j)` covers
   `[i · W / W', (i + 1) · W / W') × [j · H / H', (j + 1) · H / H')` of it, and
   its value is the integral of the grey image over that rectangle divided by
   the rectangle's area. This is the same coverage-weighted mean the thumbnail
   is made by, computed in double precision on the grey samples rather than on
   8-bit colours.

Write `I(x, y)` for the reading image's sample at column `x` (left to right) and
row `y` (top to bottom). The **interior** is the pixels with
`1 ≤ x ≤ W' − 2` and `1 ≤ y ≤ H' − 2`, so that every 3 × 3 neighbourhood below
lies inside the image; there are `n = (W' − 2)(H' − 2)` of them. Both readings
are means over the interior. No border pixel is padded, reflected or counted.
If `W' < 3` or `H' < 3`, the interior is empty, and the sharpness and all three
tensor values are `0`.

#### Sharpness

The Laplacian at an interior pixel is the 3 × 3 kernel
`[[0, 1, 0], [1, −4, 1], [0, 1, 0]]`:

```
Λ(x, y) = I(x − 1, y) + I(x + 1, y) + I(x, y − 1) + I(x, y + 1) − 4 · I(x, y)
```

and the sharpness is its population variance over the interior, in double
precision:

```
μ = (1/n) Σ Λ(x, y)
sharpness = (1/n) Σ (Λ(x, y) − μ)²
```

The units are grey units squared, the grey unit being the `[0, 1]` scale of
`image_to_gray`, so an image converted with a different formula or with
`quantize_bits` has a different sharpness; images compared with each other must
share one `image_to_gray`.

#### Gradient structure tensor

The derivatives at an interior pixel are the 3 × 3 Sobel kernels divided by 8,
so that on a linear ramp of slope `s` grey units per pixel the derivative is
`s`:

```
Gx(x, y) = ( I(x+1, y−1) + 2·I(x+1, y) + I(x+1, y+1)
           − I(x−1, y−1) − 2·I(x−1, y) − I(x−1, y+1) ) / 8

Gy(x, y) = ( I(x−1, y+1) + 2·I(x, y+1) + I(x+1, y+1)
           − I(x−1, y−1) − 2·I(x, y−1) − I(x+1, y−1) ) / 8
```

`Gx` increases to the right and `Gy` increases downward, the directions of
column and row. The three stored values are means over the interior, in double
precision:

```
Exx = (1/n) Σ Gx²      Eyy = (1/n) Σ Gy²      Exy = (1/n) Σ Gx·Gy
```

in grey units squared per pixel squared of the reading image. They are stored
as they are, not divided by the gradient energy `Exx + Eyy`.

A reader derives from them:

- **Gradient energy** `E = Exx + Eyy`, the image's mean squared gradient.
- **Normalised tensor** `(Exx, Eyy, Exy) / E`, which does not change when the
  image's contrast does.
- **Coherence** `c = sqrt((Exx − Eyy)² + 4·Exy²) / E`, from `0` (gradients
  equally strong in every direction) to `1` (all gradients along one
  direction). Because `Exy² ≤ Exx · Eyy`, `c` does not exceed `1` apart from
  rounding. When `E = 0` the image is flat and `c` is taken as `0`.
- **Dominant gradient direction** `θ = ½ · atan2(2·Exy, Exx − Eyy)`, an angle
  in `(−π/2, π/2]` from the `+x` axis toward `+y` (on screen, clockwise from
  pointing right), meaningful modulo `π` and only when `c` is well above `0`.
  Motion blur weakens the gradients along the direction of motion, so a frame
  blurred by motion reads a `θ` across the motion; the motion's direction is
  near `θ + π/2`.
- **Axis ratio** `|log(Exx / Eyy)|`, how far the gradient energy along `x`
  differs from that along `y`, taken as unbounded when one of the two is `0`.
  Read against the capture's median, it is the reading that catches blur along
  one image axis ([evidence](#what-the-readings-catch)).

All of these come from the three stored values, and each is read against the
capture: a frame whose axis ratio or coherence is well away from the capture's
median, or whose normalised tensor lies far from the capture's median
normalised tensor, is a candidate for motion blur. The
directions are in stored pixel order, so for an image displayed with its
orientation tag applied they turn with the display.

### Example

Against the version 1 example in
[`../formats/sift-file-format.md`](../formats/sift-file-format.md#using-cli-commands-to-pull-apart-a-sift-file),
a version 2 file of the same image differs in `version`, in
`feature_options.image_to_gray` and in the two readings (the values are
illustrative):

```bash
$ jq .feature_options.image_to_gray feature_tool_metadata.json
{
  "formula": "0.2126*R + 0.7152*G + 0.0722*B",
  "quantize_bits": 8
}

$ jq . metadata.json
{
  "version": 2,
  "image_name": "myimage.jpg",
  "image_file_xxh128": "3748f9341bfdcc712beb2c5392664982",
  "image_file_size": 21584,
  "image_width": 270,
  "image_height": 480,
  "feature_count": 2464,
  "sharpness": 0.0041206,
  "gradient_structure_tensor_xx_yy_xy": [0.00093114, 0.00071402, -0.00008351]
}
```

The 270 × 480 image has a long side of 480, so its reading image is the grey
image at its own size.

### Where the readings live, and why

The readings go in `metadata.json` rather than a new entry:

- They describe the input image, which is what `metadata.json` describes (its
  name, hash, size and dimensions), and like those fields they are fixed once
  the file is written.
- Every reader already reads `metadata.json` whole, including the readers that
  read nothing else, so the `.sfmr` writer, which opens each `.sift` for its
  hashes and thumbnail, gets the readings without reading another entry.
- They join `metadata_xxh128` and `content_xxh128` with no new section and no
  change to the content-hash order, so the order a verifier hashes the entries
  in is the same in both versions.
- Under the version 3 proposal, `metadata.json` is one of the immutable entries
  behind the stable `feature_set_xxh128`, so the readings stay part of the
  file's identity across descriptor appends without a new rule.

A small binary entry would hold the same 32 bytes and would need its own slot in
`content_xxh128`; for four numbers per file it buys nothing.

### What each feature tool records

This section names the implementations, as the format spec's Implementations
section will.

| `feature_tool` | `image_to_gray.formula` | `quantize_bits` | Status |
|----------------|-------------------------|-----------------|--------|
| `sfmtool` | the workspace's `image_to_gray.formula`, by default `0.2126*R + 0.7152*G + 0.0722*B` | the workspace's, absent by default | from the code: `gray_from_rgb` in `sfmtool-core` divides each 8-bit channel by 255, evaluates the formula in double precision and stores the result unclamped as `f32` for the detector, with no rounding to a bit depth |
| `colmap` | `0.2126*R + 0.7152*G + 0.0722*B` | `8` | measured with pycolmap 4.2.1: `Bitmap.read(path, as_rgb=False)` gives an 8-bit grey bitmap equal to `clone_as_grey()` of the RGB read, whose value is the BT.709 weights applied in single precision to the 0 to 255 samples and rounded to nearest with half up; that matched all 262,144 pixels of a random RGB image, and `q` of the double-precision formula differs from it by one grey level on 18 of them. A JPEG carrying EXIF orientation 6 read back at its stored width and height, so the tag is not applied. **To verify**: that COLMAP's feature extractor reads through this same `Bitmap` path, and whether the conversion happens before or after its `max_image_size` resize |
| `opencv` | `0.299*R + 0.587*G + 0.114*B` | `8` | from the code: `extract_opencv.py` makes the detector's input with `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)` on the 8-bit decode, which gives 8-bit grey. Measured with OpenCV 5.0.0: the 8-bit conversion is `(9798·R + 19235·G + 3735·B + 16384) >> 15` on the 0 to 255 samples, which matched all 262,144 pixels of a random RGB image; `q` of the exact BT.601 formula differs from it by one grey level on 346 of them (0.13%) |

**Each tool records the closest formula, not always an exact one.** The
formula and `quantize_bits` describe the `sfmtool` backend's grey exactly. For
`colmap`, whose single-precision weights and `floor(v + 0.5)` on 0 to 255
values give its grey, the recorded BT.709 formula with `quantize_bits: 8`
differs by one grey level on about 18 of 262,144 random pixels. For `opencv`,
whose fixed-point weights `(9798·R + 19235·G + 3735·B + 16384) >> 15` give its
grey, the recorded BT.601 formula with `quantize_bits: 8` differs on about
0.13% of pixels. Only the record of the backend's own grey is approximate: the
readings are always computed by sfmtool's one implementation from the recorded
formula, never taken from a backend's grey image, so they can be reproduced
from the file.

All three Python backends decode with `sfmtool.fileio.read_image_rgb`, which
ignores the EXIF orientation, and make their thumbnail from that decode. The
readings are computed from that same decode. The `colmap` backend's detector
reads the image through COLMAP's own decoder instead; its readings still come
from `read_image_rgb`'s decode, beside its thumbnail.

**16-bit images do not yet follow the definition.** Every backend's decoder
reduces a 16-bit image to 8 bits per channel, `round(c · 255 / 65535)`, before
anything else, so for such an image the detector, the thumbnail and the readings
all see 8-bit samples rather than the 16-bit samples divided by 65535 that
**Inputs** defines. For 8-bit images the code follows the definition. Decoding
16-bit sources at full depth is future work.

### The workspace records the same object: `.sfm-workspace.json` version 2

A workspace's `.sfm-workspace.json` defines the directory its `.sift` files go
in, through `feature_options` and `feature_prefix_dir`, and its
`feature_options` are what the extractor writes into each `.sift` file. So the
workspace spells the conversion the way the `.sift` file does: a version 2
workspace file's `feature_options` carries the same `image_to_gray` object, at
its top level, for every feature tool, and the `.sift` file's `feature_options`
are a copy of it.

| `feature_tool` | `feature_options.image_to_gray` in a version 2 workspace |
|----------------|----------------------------------------------------------|
| `sfmtool` | `{"formula": "0.2126*R + 0.7152*G + 0.0722*B"}` by default; a setting, which the `sfmtool` extractor honours for any formula and any `quantize_bits` |
| `colmap` | `{"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8}`, fixed by the tool |
| `opencv` | `{"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8}`, fixed by the tool |
| `sfmtool-undistort` | its source's conversion ([`sfm undistort`](#sfm-undistort)); recorded, not settable |

The `gray_formula` key is gone from version 2: the object replaces it. For
`colmap` and `opencv` the object records what the tool does rather than
choosing it. The file's other fields are unchanged.

**The version number.** `.sfm-workspace.json` becomes version 2 because a
reader has to tell an old file, whose conversion is a `gray_formula` string or
unstated, from a new one, whose conversion is the object. The version says which
without guessing from the keys, and keeps one spelling per version instead of
carrying `gray_formula` forward beside the object.

#### The recorded folder name is authoritative

`feature_prefix_dir` is stored in the workspace file so that nothing downstream
has to compute it: every command, the viewer and every `.sfmr` use the recorded
name as it is, and `load_workspace_config` returns it without recomputing it. A
folder name is computed only when a folder is created: by `sfm ws init`,
`sfm ws upgrade`, `sfm undistort`, `sfm sift --extract --tool` outside a
workspace, and the viewer's conversion of a version 1 record or switch of
another tool's record to the `sfmtool` defaults. It must differ for different options, and the existing name,
`features/{feature_type}-{hash}` with the XXH3-128 hash of the compact JSON of
`feature_tool`, `feature_type` and `feature_options` (without `use_gpu`) that
`get_feature_tool_xxh128` computes, does that. It need not be the same in
Python and Rust: two writers that name the same options differently cache the
features twice, never wrongly. Because `image_to_gray` is part of the options a
folder is named from, every folder created under the new rules is new, for
every tool, even with otherwise identical settings.

What this gives up: a hand edit of `feature_options` does not move the folder,
so it can put `.sift` files made under different settings into one folder. Each
`.sift` file records its own `feature_tool_metadata`, so that is detectable
afterwards. To change the options, run `sfm ws init --force` or
`sfm ws upgrade`, which create a new folder.

The folder hash is still a different value from a `.sift` file's
`feature_tool_xxh128`, which is the XXH128 of the stored bytes of its
`feature_tool_metadata.json`; both inputs now hold the same `image_to_gray`
object, but a reader must not compare one with the other.

#### Loading checks

`load_workspace_config` in `src/sfmtool/_workspace.py`, the one reader of
`.sfm-workspace.json` (core's `BuildSiftFilesOptions::from_workspace`, which
has no caller outside its tests, is removed), refuses at load:

- a file with no `version` key (`init_workspace` has written `"version": 1`
  since the commit that added it, and the workspace spec has required it from
  its first version);
- a `version` other than `1` or `2`;
- a version 2 file whose `feature_options` carry `gray_formula` at the top
  level, carry no top-level `image_to_gray`, or carry one that does not parse;
- a version 1 file whose `feature_options` carry `image_to_gray`, since the
  object belongs to version 2.

The version 2 rules apply to the top level of `feature_options` only;
`sfmtool-undistort`'s nested `source_feature_options` is a copy of its source's
record and may hold a version 1 `gray_formula`.

#### Versions never mix

**No `.sift` file is written from version 1 settings, and no version 2 `.sift`
file goes into a folder that holds version 1 files.** Version 1 settings are a
version 1 workspace file or a record whose `feature_options` carry no
`image_to_gray`, such as a `.sfmr` made from a version 1 workspace. A version 1
workspace moves to version 2 only through `sfm ws upgrade` or `sfm ws init`,
both of which create a new folder. **Reading version 1 files stays allowed
everywhere**: a command that reads `.sift` or `.sfmr` files in a version 1
workspace reads them in its recorded folder as today, a `gray_formula` reading
as `{"formula": <gray_formula>}` and a `colmap` or `opencv` file with no stated
conversion as its tool's row, and a `.sfmr` written there takes its readings
from the image files ([Part 2](#where-a-writer-gets-the-values)). A command that
only needed features already there, and finds every `.sift` present and
current, writes nothing and so does not refuse.

#### The writers

Every `.sift` file is written by one of these, and the rule above is checked in
each, not in one command:

| Writer | Reached from | Reads its settings from | Writes into | Refuses when |
|--------|--------------|-------------------------|-------------|--------------|
| `image_files_to_sift_files` (`src/sfmtool/sift/extract.py`), which writes each `.sift` that is missing or older than its image | `sfm sift --extract`, and implicitly `sfm solve` (`colmap/db_setup.py`), `sfm match` (`feature_match/_run.py`), `sfm to-colmap-db` (`colmap/db_export.py`), the `image_files_to_sift_files_opencv` wrapper | the options and folder its caller passes: the workspace file's, read through `load_workspace_config`, or with `sfm sift --extract --tool` outside a workspace, the tool's defaults | the recorded `feature_prefix_dir`; with `--tool`, a folder named from the tool's defaults | the options carry no `image_to_gray` (as a version 1 workspace file's do); a `.sift` already in the folder is version 1; a `colmap` or `opencv` object differs from its tool's row |
| core's `build_sift_files` (`crates/sfmtool-core/src/features/sift/files.rs`) | its Python binding (`crates/sfmtool-py/src/sift/files.rs`) and the viewer | the options and folder it is handed; it never reads the workspace file | the folder it is handed | the options carry no `image_to_gray`; a `.sift` already in the folder is version 1 |
| the viewer's *Build SIFT Files* (`crates/sfm-explorer/src/sift_files.rs`), through `build_sift_files` | the scene tree and the wire | the node's `.sfmr` record, not the workspace file, so the files go where the node looks for them | a version 2 `sfmtool` record: its recorded folder. A version 1 `sfmtool` record of an embedded-patches node: a new folder named from the record converted in memory by the conversion `sfm ws upgrade` makes. A record naming another tool: the `sfmtool` defaults' folder, as today's switch | a version 1 record on a `sift_files` node |
| `sfm undistort` (`src/sfmtool/_undistort_images.py`), with `write_sift` | `sfm undistort` | the input `.sfmr`'s record and the undistorted images | a new folder in its output | never, for version reasons |

**The viewer.** When every file is built, a converted record (like today's
switch to the `sfmtool` defaults) is pushed as a version, so the node then
looks in the new folder; Undo puts the old record back and a save writes the new
one. The workspace's `.sfm-workspace.json` is neither read nor changed, so it
may still be version 1 and the command-line writers keep refusing there until
`sfm ws upgrade` names its own new folder. The old folder stays, and other
reconstructions still find it through their own records. The Background panel
line and the Action Log entry name the folder written to and, after a
conversion, the folder the record moved from. A version 1 record on a
`sift_files` node is refused for the reason today's switch is: its feature
indexes name features of the old files.

<a id="sfm-undistort"></a>**`sfm undistort`** writes its own workspace
(`feature_tool` `sfmtool-undistort`, `feature_options` holding the source's
`source_feature_tool` and `source_feature_options` and the warp's `fit` and
`filter`) as version 2, with a top-level `image_to_gray` converted from the
source as `sfm ws upgrade` converts one: the source's own top-level object, else
its `gray_formula` as `{"formula": …}`, else the source tool's row, a version 1
`sfmtool-undistort` source being converted from its own nested record the same
way. Its `.sift` files are version 2 with that object, and their readings are
computed by decoding the undistorted image files it wrote, so they describe
those images, not the source images, and a later `.sfmr` writer that recomputes
them from the image files gets the same values. Its folder,
`features/sift-sfmtool-undistort-{hash}`, is named over its whole
`feature_options` when it creates it, so undistorting again under the new rules
writes a new folder beside any older output.

#### Refusals

| Refusal | Raised by | Its message |
|---------|-----------|-------------|
| Version 1 settings | `image_files_to_sift_files`, `build_sift_files` | names the workspace file or the record, says it holds version 1 settings, and says to run `sfm ws upgrade`, which moves the workspace to version 2 and a new folder |
| Version 1 files in the folder | `image_files_to_sift_files`, `build_sift_files` | names the folder, says it holds version 1 `.sift` files, and says to run `sfm ws upgrade` |
| `colmap` or `opencv` conversion differs | `image_files_to_sift_files` | names the tool and the one conversion it makes |
| Version 1 record on a `sift_files` node | the viewer | says the feature indexes name features of the old files, which a new folder would not hold, and suggests converting to embedded patches |
| Invalid workspace file at load | `load_workspace_config` | names the file and what is wrong: no `version`, the version found, or the key that breaks the version's rules |
| `sfm ws upgrade` cannot upgrade | `sfm ws upgrade`, changing nothing | names the file and says which: no workspace file, no `version`, a version above 2, a `feature_tool` other than `sfmtool`, `colmap` or `opencv` (an `sfmtool-undistort` workspace is told to run `sfm undistort` again, since its `.sift` files cannot be re-extracted), or a `gray_formula` that does not parse |

#### Migration

No command rewrites the workspace file unless the user runs one of these:

- **`sfm ws upgrade`** acts on the workspace found by the usual discovery from
  the directory it is given, or the current directory. It rewrites a version 1
  file as version 2 keeping every option, including hand-edited ones such as
  `peak_threshold` or `max_image_size`; converts `sfmtool`'s `gray_formula` into
  `image_to_gray: {"formula": <gray_formula>}` and gives a `colmap` or `opencv`
  workspace its tool's row; names a new `feature_prefix_dir` from the new
  options; prints the old and new folders; and leaves the old folder, its
  `.sift` files and the named image lists file
  ([named-image-lists.md](../workspace/named-image-lists.md)) on disk. On a version 2 file it
  says "nothing to do: already version 2" and changes nothing. `.sfmr` files
  that name the old folder still find their `.sift` files through their own
  record.
- **`sfm ws init` and `sfm ws init --force`** write version 2 with fresh
  options from the tool's defaults and the flags, the tool's `image_to_gray`
  row, and a new folder named from them. `--force` drops any hand-edited option,
  which is why the refusals point at `sfm ws upgrade`. The old folder stays on
  disk, and the named image lists file is not touched.
- **`sfm ws init --image-to-gray FORMULA --quantize-bits B`** sets the
  `sfmtool` tool's conversion before the folder is named. The formula is checked
  against the grammar and `B` must be an integer from 1 to 16; with
  `--feature-tool colmap` or `opencv` either flag is an error.

**`.sfmr` files.** `workspace.contents` copies the workspace's `feature_tool`,
`feature_type`, `feature_options` and `feature_prefix_dir`, so a `.sfmr` written
from a version 2 workspace holds the object in
`workspace.contents.feature_options`, and one written from a version 1 workspace
holds what that file held. The `.sfmr` format treats `feature_options` as
tool-defined, so this needs no format change of its own; the `.sfmr` spec's
example gains the object when version 15 is filed. In a version 15 file written
from a version 2 workspace, `images/metadata.json`'s `image_to_gray` equals
`workspace.contents.feature_options.image_to_gray`.

## Part 2: `.sfmr` version 15

`metadata.version` becomes `15`. Version 15 adds two required per-image
columns and a required `image_to_gray` record for them in
`images/metadata.json`. A `.sfmr` does not reference named image lists at all:
it records no list name and no list entries, and never embeds the workspace's
list file ([named-image-lists.md](../workspace/named-image-lists.md)). When this ships, the `.sfmr` spec defines the
`image_to_gray` object (its inputs, grammar and rounding), the reading image,
the sharpness and the tensor in its own words, repeating Part 1, because a
format spec does not send its reader to another spec for what a field means.

### `images/sharpness.{N}.float32.zst`

- **Shape**: `(N,)` where N = image_count.
- **Data type**: `float32` (little-endian).
- **Format**: image `i`'s sharpness, the population variance over the interior
  of the 3 × 3 Laplacian of its reading image (Part 1's definition, with the
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

The values are computed in double precision, as Part 1 defines them, and a
writer stores each one rounded to the nearest `float32` (ties to even). A value
copied from a `.sift` file's `metadata.json`, where it is a JSON number holding
the double, is rounded the same way, so a row copied from a `.sift` and a row
computed from the image file by the same arithmetic are the same bytes.

### `images/metadata.json`: `image_to_gray`

A new required key holding the `{"formula": …, "quantize_bits": …}` object the
two columns were computed with, with Part 1's grammar and rounding. One
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
   with the readings computed by Part 1's definition through the file's
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
or, for version 1 files, the object Part 1's table gives for the workspace's
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
`-measurements.md` file.

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

- **One function computes the readings.** A Rust function in `sfmtool-core`,
  beside `gray_from_rgb`, takes the decoded 8-bit RGB pixels and the
  `image_to_gray` object and returns the sharpness and the three tensor
  values, building the reading image in double precision. `GrayFormula` gains
  the optional `quantize_bits` (the pair becoming an `ImageToGray` value
  parsed from the object), and the area resize follows the coverage weights of
  `reconstruction::thumbnail::resize_area`, in double precision. It is bound
  for Python beside `extract_sift`. It takes no `Progress`: on a 1920-pixel
  frame it runs in a few milliseconds. OpenCV's `INTER_AREA` applied to the
  double-precision grey image equals the exact area average to about 1e-15 at
  power-of-two factors (2, 4, 8 and 16 were measured), but differs by about
  7e-9 grey (about 1e-8 relative in the readings) at other factors (3, 6 or
  10.64, say), so it is not bit-exact
  with the definition there and the implementation does not use it.
- **The workspace file** is written by `init_workspace` in
  `src/sfmtool/_workspace.py`, and the options it writes come from
  `get_default_sfmtool_feature_options`, `get_colmap_feature_options` and
  `get_default_opencv_feature_options`, which gain the `image_to_gray` object
  (the first losing `gray_formula`); new folder names come from
  `get_feature_tool_xxh128` in `src/sfmtool/sift/file.py`, unchanged, and the
  viewer's from the same rule computed in Rust, as `sfmtool_defaults()` computes its prefix today. `SiftParams::from_feature_options`
  in `sfmtool-core` reads the object instead of `gray_formula`, keeping
  `gray_formula` only for reading version 1 records.
- **The Rust defaults.** `BuildSiftFilesOptions::sfmtool_defaults()` in
  `crates/sfmtool-core/src/features/sift/files.rs` repeats the `sfmtool` default
  options, with `gray_formula`, and the prefix hash computed from them, and a
  test pins it to the prefix `sfm ws init` writes. It gains the `image_to_gray`
  object in place of `gray_formula`; the test stays as a convenience that keeps
  the two default folder names equal, not as a format rule.
- **The version and extraction checks**: `load_workspace_config` in
  `src/sfmtool/_workspace.py` checks the version at load, and each writer in
  [the writers table](#the-writers) checks the no-mixing rule. `sfm ws upgrade`
  is a new subcommand of the `ws` group in `src/sfmtool/_commands/ws.py`, and
  `sfm ws init` gains `--image-to-gray` and `--quantize-bits` there.
- **The three Python backends** (`extract_sfmtool.py`, `extract_colmap.py`,
  `extract_opencv.py` in `src/sfmtool/sift/`) call it on the decode they make
  the thumbnail from, beside `thumbnail_of_rgb` or `cv2.resize`, and put the
  readings and `image_to_gray` into the metadata they yield. The Rust writer
  `build_sift_files` calls it beside `thumbnail_from_rgb`. `sfm undistort`
  calls it on each undistorted image file it wrote, after writing its
  workspace file as version 2 ([`sfm undistort`](#sfm-undistort)); the viewer
  converts a version 1 record through the same core function `sfm ws upgrade`
  binds.
- **`sfmtool-sift-format`** reads and writes version 2: `SiftMetadata` gains the
  two fields, `validate_sift_data` checks them, and `SIFT_FORMAT_VERSION`
  becomes 2.
- **`sfmtool-sfmr-format`** reads and writes version 15, and the `.sfmr`
  writers gather the readings as [Part 2](#where-a-writer-gets-the-values)
  says: the Python path that copies thumbnails from `.sift` files
  (`src/sfmtool/colmap/io.py`) copies the readings beside them, and the Rust
  save path gains a lookup like `verified_sift_thumbnail` with the photograph
  as the fallback.

## Words for the glossary

To add to [`../GLOSSARY.md`](../GLOSSARY.md) when this lands:

- **sharpness** (of an image): the variance of the Laplacian of its reading
  image, `images/sharpness` and the `.sift` `sharpness` field. Scope: a whole
  image. The tile-level sharpness of the bench (`pair_sharpness`, the *sharper*
  tile of blur-matched ZNCC) is a different reading, the ZNCC self-similarity
  ellipse, and keeps its words. Chosen over `laplacian_variance`, which names
  the method and not what the number is for, and `focus`, which leaves out
  motion blur.
- **gradient structure tensor**: the means of `Gx²`, `Gy²` and `Gx·Gy` over an
  image's reading image, stored raw as `[Exx, Eyy, Exy]`; its **coherence**,
  **dominant gradient direction** and **axis ratio** `|log(Exx/Eyy)|` are
  derived. Chosen over `structure tensor`
  alone, which elsewhere often means a smoothed per-pixel field rather than one
  mean per image.
- **reading image**: the 480-pixel area-averaged grey image both readings are
  taken on.

## Choices made here

Decisions inside the maintainer's decisions that this draft makes, each with
where it is described and why, for review:

1. **EXIF orientation not applied**
   ([inputs](#feature_tool_metadatajson-feature_optionsimage_to_gray)): every
   extractor decodes in stored pixel order, so applying it would contradict
   every file written.
2. **Readings in `.sift` `metadata.json`**
   ([why](#where-the-readings-live-and-why)): they describe the image, every
   reader already reads it, and no hash slot changes.
3. **Tensor stored raw** ([tensor](#gradient-structure-tensor)): the normalised
   form is one division away, the energy is not recoverable from it.
4. **`float32` in `.sfmr`** ([why](#why-float32-and-these-names)): three
   significant digits suffice and `float16` underflows on flat captures.
5. **Long side 480, never enlarged, area averaged** ([reading
   image](#the-reading-image), [rationale](#why-480)): the middle of the
   measured plateau, exact at whole-number factors of common sizes.
6. **Interior pixels only, `0` under 3 pixels** ([reading
   image](#the-reading-image)): no padding convention enters the definition.
7. **Sobel divided by 8** ([tensor](#gradient-structure-tensor)): derivatives in
   grey units per pixel.
8. **One `image_to_gray` per `.sfmr`, rows under another recomputed**
   ([`images/metadata.json`](#imagesmetadatajson-image_to_gray),
   [sources](#where-a-writer-gets-the-values)): rows under different conversions
   do not compare.
9. **Version checks at load** ([loading checks](#loading-checks)): a bad file
   fails before any work, not at extraction.
10. **No-mixing checked in every writer** ([writers](#the-writers)): implicit
    extraction in `solve`, `match` and `to-colmap-db` and the viewer write
    `.sift` files too.
11. **Proposed for review: `sfm ws upgrade`** ([migration](#migration)): keeps
    hand-edited options, which `sfm ws init --force` with reconstructed flags
    cannot.
12. **Proposed for review: `sfm ws init --image-to-gray`, `--quantize-bits`**
    ([migration](#migration)): the only way to set the conversion and name its
    folder, since a hand edit does not move the folder.
13. **`sfm undistort` writes version 2 from any source** ([`sfm
    undistort`](#sfm-undistort)): it only reads its source, and its folder is
    new.
14. **The viewer converts a version 1 record in memory** ([writers](#the-writers)):
    it builds where the node looks, and today's switch already works this way.
15. **`BuildSiftFilesOptions::from_workspace` removed**
    ([loading checks](#loading-checks)): one reader of the workspace file,
    with its checks in one place.

## Not in this draft

Any rule for which frames count as blurred: the readings are recorded so that
such a rule can be written, and it will be designed with the command that
writes named image lists ([named-image-lists.md](../workspace/named-image-lists.md)).

Decoding 16-bit sources at full depth, so that a 16-bit image's readings and
features follow the **Inputs** definition rather than an 8-bit reduction of it,
is future work.
