# Image sharpness readings and named image lists

**Status:** Draft. Decided by the maintainer: `.sift` version 2 with a required
`image_to_gray` object (with an optional `quantize_bits`), a per-image
sharpness and a per-image gradient structure tensor; `.sfmr` version 15 with
both as required per-image columns, stored as `float32`; named image lists in their own file in the
workspace root, which no `.sfmr` references; `-r/--range` accepting a list's
name; `.sfm-workspace.json` carrying the same `image_to_gray` object as the
`.sift` file; each feature tool recording the formula closest to its own grey
conversion. The choices this draft makes inside those decisions (entry names,
data types, where the values live in a `.sift` file, the list file's name, the
resize and border rules, the CLI's workspace discovery with a list) are listed in [Choices made here](#choices-made-here)
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
frame. The second half of the draft lets a workspace name a set of images, such
as the sharp frames of one walk, so that commands can be pointed at it by name.

It amends these standing specs, each of which links back here:

- [`../formats/sift-file-format.md`](../formats/sift-file-format.md): version 2
  (Part 1).
- [`../formats/sfmr-file-format.md`](../formats/sfmr-file-format.md): version 15
  (Part 2).
- [`../workspace/workspace.md`](../workspace/workspace.md): `.sfm-workspace.json`
  version 2, whose `feature_options` carry the `image_to_gray` object (Part 1),
  and the `.sfm-named-image-lists.json` file beside it (Part 3).
- [`../cli/README.md`](../cli/README.md): the `-r/--range` option that eight
  commands share (Part 4).
- [`../core/features/sift.md`](../core/features/sift.md) § "Image-to-gray
  conversion", which points here for the structured `image_to_gray` object.

It also renumbers the incremental-extraction proposal: the growable `.sift`
archive in
[`sift-incremental-extraction-amendment.md`](sift-incremental-extraction-amendment.md)
becomes version 3, built on version 2 as defined here, and the `image_to_gray`
object it used to carry is defined here instead.

The command that will write named image lists, a later `sfm survey`-style
command that reads the sharpness and tensor of a capture's frames and proposes
lists of the sharp ones, is the intended producer; it is not designed here.

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
2. **Size.** Let `L = max(W, H)` and `S = min(W, H)`. If `L ≤ 480`, the grey
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

Both `c` and `θ` are read against the capture: a frame whose coherence is well
above the capture's median, or whose normalised tensor lies far from the
capture's median normalised tensor, is a candidate for motion blur. The
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
workspace file's `feature_options` carries the same `image_to_gray` object,
`formula` and optional `quantize_bits`, for every feature tool, and the `.sift`
file's `feature_options` are a copy of it.

| `feature_tool` | `feature_options.image_to_gray` in a version 2 workspace |
|----------------|----------------------------------------------------------|
| `sfmtool` | `{"formula": "0.2126*R + 0.7152*G + 0.0722*B"}` by default; a setting, which the `sfmtool` extractor honours for any formula and any `quantize_bits` |
| `colmap` | `{"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8}`, fixed by the tool |
| `opencv` | `{"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8}`, fixed by the tool |

The `gray_formula` key is gone from version 2: the object replaces it, and a
version 2 file that carries `gray_formula` is invalid. For `colmap` and
`opencv` the object records what the tool does rather than choosing it, so
extraction refuses a workspace whose object differs from the tool's row above,
saying the tool converts only that way.

**The version number.** `.sfm-workspace.json` becomes version 2. The change is
to what `feature_options` must hold, and a reader has to tell an old file,
whose conversion is a `gray_formula` string or unstated, from a new one, whose
conversion is the object; the version says which without guessing from the
keys, and keeps one spelling per version instead of carrying `gray_formula`
forward beside the object. The file's other fields are unchanged.

**The features directory changes.** `feature_prefix_dir` is
`features/{feature_type}-{feature_cache_hash}`, and the hash is computed from
the options, so a workspace written under the new rules gets a new features
directory, for every tool, even with otherwise identical settings. The hash is
the XXH3-128 digest, as 32 lowercase hex digits, of the UTF-8 bytes of the
JSON text of one object with three keys: `feature_tool` (the tool name in lower
case), `feature_type` (the directory's type, such as `sift-sfmtool`,
`sift-colmap`, `sift-colmap-dsp` or `sift-opencv`) and `feature_options` with
the `use_gpu` key removed, serialised with sorted keys, no whitespace (`,` and
`:` as separators). The bytes are the ones Python's
`json.dumps(obj, sort_keys=True, separators=(",", ":"))` produces with its
default escaping of non-ASCII characters. A Rust writer gets the same bytes from
`serde_json`'s compact output, which sorts keys but does not escape non-ASCII,
because every key and string value in a version 2 `feature_options` is ASCII:
the formula is ASCII by its grammar, and loading refuses a version 2 workspace
whose options hold a non-ASCII string (below). The numbers in the default
options are spelled the same by both serialisers, and the test that holds
`BuildSiftFilesOptions::sfmtool_defaults()` to the prefix `sfm ws init` writes
checks that. `image_to_gray` is inside
`feature_options`, so it enters the hash, and since no version 1 workspace's
options hold that object, no version 2 workspace's directory can equal a
version 1 workspace's. That is what keeps version 1 and version 2 `.sift` files
out of the same directory. The same holds for extraction outside a workspace,
which computes the directory from the tool's default options, now including the
object.

The relation the workspace spec states still holds: this cache hash is a
different value from the `.sift` file's `feature_tool_xxh128`, which is the
XXH128 of the stored bytes of the file's `feature_tool_metadata.json` (with
`feature_type` `"sift"`, `use_gpu` included where the options have it, and the
writer's own serialisation). Under the new shape both inputs contain the same
`image_to_gray` object, but the two hashes are still computed from different
bytes and still do not agree; a reader must not compare one with the other.

**Loading checks the version.** `load_workspace_config` in
`src/sfmtool/_workspace.py`, the one reader of `.sfm-workspace.json` that every
command goes through, gains these checks, each an error naming the file:

- A missing `version` key is an error. `init_workspace` has written
  `"version": 1` since the commit that added it, and the workspace spec has said
  the field must be `1` since it was first written, so every file sfmtool made
  has one.
- A `version` other than `1` or `2`, including one above 2, is refused with an
  error naming the file and the version it found.
- A version 2 file whose `feature_options` carry `gray_formula`, or carry no
  `image_to_gray`, or whose `image_to_gray` does not parse, is rejected at load,
  not later at extraction. So is a version 2 file whose `feature_options` hold a
  non-ASCII string.
- A version 1 file whose `feature_options` carry `image_to_gray` is invalid
  too: the object belongs to version 2, and a file holding it under version 1
  would hash to a directory no version 1 reader expects.

`load_workspace_config` never recomputes `feature_prefix_dir`; it returns the
stored one, so editing `image_to_gray` (or any option) by hand does not move the
features directory. A hand edit is detected, not trusted: the extraction check
above recomputes the prefix from the options with `get_feature_tool_xxh128`, and
when it differs from the stored `feature_prefix_dir` it refuses to write, saying
the options no longer match the directory and to run `sfm ws upgrade`, which
brings the directory back in line with the options. Reading is not affected,
since each `.sift` file records the options it was made with.

**Every path that writes `.sift` files refuses to mix versions.** A version 2
`.sift` file never goes into a directory that holds version 1 files, and no
`.sift` file is written for a version 1 workspace. The check is in the two
functions every writer goes through, not in one command:

- `image_files_to_sift_files` in `src/sfmtool/sift/extract.py`, which writes
  each `.sift` that is missing or older than its image. `sfm sift --extract`
  calls it, and so do three commands that extract implicitly: `sfm solve`
  (`src/sfmtool/colmap/db_setup.py`), `sfm match`
  (`src/sfmtool/feature_match/_run.py`) and `sfm to-colmap-db`
  (`src/sfmtool/colmap/db_export.py`), as does the
  `image_files_to_sift_files_opencv` wrapper.
- `build_sift_files` in `sfmtool-core`
  (`crates/sfmtool-core/src/features/sift/files.rs`), which the viewer's
  *Build SIFT Files* (`crates/sfm-explorer/src/sift_files.rs`, writing into the
  `feature_prefix_dir` the `.sfmr` records, which can be a version 1
  directory) and the Python binding over it call.

`sfm undistort`, which writes `.sift` files for the images it undistorts into a
directory of its own, makes the same check before writing.

The check refuses, before writing anything, when the workspace file is version
1, when the feature options it is given carry no `image_to_gray` object (a
version 1 record, such as a `.sfmr` made from a version 1 workspace), or when
any `.sift` file already in the target directory is version 1. Every refusal
carries the same message: it names the workspace file or the directory, says it
holds version 1 settings or version 1 `.sift` files, and says to run
`sfm ws upgrade` (below), which moves the workspace to version 2 and a new
features directory. A command that only needed to read features that are
already there, and finds every `.sift` present and current, writes nothing and
so does not refuse.

**Reading version 1 files stays allowed everywhere.** A command that reads
`.sift` files or `.sfmr` files in a version 1 workspace (`sfm inspect`, the
viewer opening a `.sfmr`, `sfm solve` or `sfm match` on images whose `.sift`
files all exist and are current, a `.sfmr` save) reads its version 1 `.sift`
files in its existing `feature_prefix_dir` as today, and a `.sfmr` written there
takes its readings from the image files
([Part 2](#where-a-writer-gets-the-values)). A `gray_formula` in a version 1
`sfmtool` workspace reads as `{"formula": <gray_formula>}`, and a version 1
`colmap` or `opencv` workspace, which states no conversion, reads as that tool's
row above.

**Migration.** No command rewrites the workspace file unless the user runs
`sfm ws upgrade` or `sfm ws init`:

- **`sfm ws upgrade` moves a workspace to version 2 and keeps its settings.** It
  rewrites a version 1 `.sfm-workspace.json` as version 2 with every existing
  option kept, including hand-edited ones such as `peak_threshold` or
  `max_image_size`. It converts `sfmtool`'s `gray_formula` into
  `image_to_gray: {"formula": <gray_formula>}`, and gives a `colmap` or
  `opencv` workspace its tool's row from the table above. It recomputes
  `feature_prefix_dir` from the new options and writes it, prints the old and
  the new directory, and leaves the old features directory, its `.sift` files
  and the named image lists file on disk. On a version 2 file whose stored
  `feature_prefix_dir` no longer matches its options, it recomputes the
  directory the same way; on a version 2 file that already matches, it changes
  nothing and says so. The `.sfmr` files that name the old directory still
  resolve their `.sift` files through their own record.
- **`sfm ws init` and `sfm ws init --force` write version 2 with fresh
  defaults**: `version: 2`, the tool's default `feature_options` with its
  `image_to_gray` from the table above, and the `feature_prefix_dir` hashed from
  them. `--force` over a version 1 file therefore changes `feature_prefix_dir`
  and drops any hand-edited option, which is why the refusal message points at
  `sfm ws upgrade` instead. The old features directory stays on disk, as every
  features directory does. Neither form touches the named image lists file.
- **`sfm ws init` sets the grey conversion** for the `sfmtool` tool with
  `--image-to-gray FORMULA` and `--quantize-bits B`, written into
  `image_to_gray` before `feature_prefix_dir` is hashed, so the directory
  follows the conversion. The formula is checked against the grammar and `B`
  must be an integer from 1 to 16. With `--feature-tool colmap` or `opencv`
  either option is an error, since those tools convert only one way.

**`.sfmr` files.** `workspace.contents` in a `.sfmr` copies the workspace's
`feature_tool`, `feature_type`, `feature_options` and `feature_prefix_dir`, so a
`.sfmr` written from a version 2 workspace holds the `image_to_gray` object in
`workspace.contents.feature_options`, and one written from a version 1
workspace holds what that file held. The `.sfmr` format treats
`feature_options` as tool-defined, as it does today, so this needs no format
change of its own; the `.sfmr` spec's example of `workspace.contents` gains the
object when version 15 is filed. In a version 15 file written from a version 2
workspace, `images/metadata.json`'s `image_to_gray` equals
`workspace.contents.feature_options.image_to_gray`.

## Part 2: `.sfmr` version 15

`metadata.version` becomes `15`. Version 15 adds two required per-image
columns and a required `image_to_gray` record for them in
`images/metadata.json`. A `.sfmr` does not reference named image lists at all:
it records no list name and no list entries, and never embeds the workspace's
list file (Part 3). When this ships, the `.sfmr` spec defines the
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
  about one decode per image for files whose `.sift` files are version 1 (the
  measurement below decoded a 1920-pixel JPEG to grey in 5.8 ms on one thread).
- Saving fails where it succeeds today: an older `.sfmr` copied without its
  images, and a test fixture or synthetic reconstruction with no image files,
  can no longer be written until readings are supplied. In-memory constructors
  that build a reconstruction from arrays need a way to pass them in.
- `--minimal` keeps both columns: a minimal file holds the whole
  reconstruction, and the columns are 16 bytes per image of its identity, where
  thumbnails are optional and 49,152 bytes per image.

## Part 3: Named image lists in the workspace

### The `.sfm-named-image-lists.json` file

A workspace's named image lists live in their own file, `.sfm-named-image-lists.json`,
in the workspace root beside `.sfm-workspace.json`. The file is optional. Its
name takes the `.sfm-` prefix and the dotfile form of `.sfm-workspace.json`, so
the two sort together and stay out of an image listing, and the rest of it is
the glossary's word for what it holds, *named image list*, which is also its
key; `.sfm-image-lists.json` would drop the *named* that separates these lists
from any other list of images.

```json
{
  "version": 1,
  "named_image_lists": {
    "walk1_sharp": [
      { "pattern": "frames/walk1_%04d.jpg", "range": "1-120,131-400" }
    ],
    "rig_sharp": [
      { "pattern": "rig/*/frame_%05d.jpg", "range": "10-900:2" }
    ],
    "hero_shots": [
      { "pattern": "photos/DSC_0042.JPG" },
      { "pattern": "photos/DSC_0057.JPG" }
    ]
  }
}
```

- **`version`** (integer, required): the file's format version, `1`. A reader
  rejects a version it does not know.
- **`named_image_lists`** (object, required): from a list's name to its list of
  entries, each an object with a `pattern` and an optional `range`.

- **Name.** One or more of the ASCII letters, the digits, `-`, `_` and `.`, with
  at least one letter. Names are compared exactly, case included. The letter is
  what tells a name from a range on the command line (Part 4).
- **`pattern`** (string, required): an image pattern in the `.camrig` syntax,
  relative to the workspace root: a forward-slash path with no leading `/` and
  no `..`, literal text, the glob wildcards `*` (within one path segment) and
  `**` (whole segments), and at most one frame field, `%d` or `%0Nd`, with `%%`
  for a literal `%`. One entry with a wildcard can cover every sensor of a rig,
  as `rig/*/frame_%05d.jpg` does above.
- **`range`** (string, optional): a range expression in the grammar `-r` takes
  (comma-separated integers, `a-b` spans and `a-b:s` stepped spans), applied to
  the number the pattern's frame field captures. An entry whose pattern has no
  frame field takes no `range`; one that gives one is an error. An entry with a
  frame field and no `range` takes every frame number.

**Membership.** A list's members are the union of its entries' matches: the
files under the workspace root whose workspace-relative path matches the
pattern, with the captured frame number in the entry's range. A file that two
entries of one list match is an error naming the file and both entries, since
it would leave the list's order ambiguous. Entries match files, not
directories.

**Case.** A pattern matches case-sensitively, against the paths as the
filesystem lists them or as a reconstruction records them, the same on every
platform. `photos/dsc_0042.jpg` does not match a file listed as
`photos/DSC_0042.JPG`, on Windows and macOS as on Linux, so a list selects the
same images wherever it is resolved.

**Order.** Entry order, then within an entry by frame number, then by
workspace-relative path among files of the same frame number (the sensors of
one rig frame). An entry with no frame field orders its files by
workspace-relative path. This is the order a command receives the images in,
as it receives a directory's sorted files today.

**Validation.** A command that resolves a list checks the whole file (its
version, names, patterns and ranges) and reports every
problem it finds, then matches only the list it was asked for. A list that
matches no file is an error when used. Every error names the lists file's path.
A file that is not valid JSON is reported with the line and column where
parsing failed, and nothing else in it is checked. A `version` other than `1`
is reported as "<path>: named image lists version <n> is not supported; this
sfmtool reads version 1", and nothing else in the file is checked, since its
layout is unknown. A missing `version` or `named_image_lists`, a value of the
wrong JSON type, a top-level key other than those two, and an entry key other
than `pattern` and `range` are each an error naming the key and, inside
`named_image_lists`, the list and the entry's position in it.

**Finding it.** A command first finds the workspace by the existing workspace
discovery, then reads the image lists file that sits beside that workspace's
`.sfm-workspace.json` in the workspace root. The lists file is never searched
for on its own. A command that has no workspace has no lists, so a list name
given to it is an error saying that no workspace was found and so no named
image lists are defined; a lists file with no `.sfm-workspace.json` beside it
is never used. If the workspace has no
lists file, it has no named image lists, so giving a list name is an error that
says so.

**Outside the workspace record and every `.sfmr`.** The lists file adds
nothing to `.sfm-workspace.json`, whose version 2 is about the grey conversion
alone (Part 1). The lists file is not part of
`workspace.contents`, and a `.sfmr` never embeds it, records a list's name or
entries, or refers to it in any other way; a reconstruction made from a list is
the same file it would be if its images had been given by path. `sfm ws init`,
with or without `--force`, writes only `.sfm-workspace.json` and leaves the
lists file untouched.

## Part 4: The CLI's `-r/--range`

Eight commands take `-r/--range` today:

- On image directories, matched against each file's number by
  `number_from_filename`: `sfm sift`, `sfm match`, `sfm motion` and `sfm solve`.
- On a reconstruction's images: `sfm analyze` (with `--metrics`),
  `sfm panorama`, `sfm to-colmap-bin` and `sfm to-nerfstudio`.

Each of them gains the same handling:

- **A value with a letter in it is a list name; any other value is a range.** A
  range expression contains only digits, `-`, `:` and `,` (and whitespace), so
  the two cannot be confused. A value with a letter that is not a valid name
  (`walk 1`, say) is an error saying what a name may contain.
- **`--named-image-list NAME`** is the long form for a list, for scripts that
  want to say which they mean. Its value must be a valid name.
- **One name and one range may be given together**, as two `-r` options or as
  `--named-image-list` with one `-r`. The images are the list's members whose
  frame number the range also selects: the range narrows every entry of the
  list. An entry with no frame field has no frame number to narrow, so narrowing
  a list with such an entry is an error naming the entry. Two names, or two
  ranges, are an error.
- **An unknown name** is an error that names the lists file (or says the
  workspace has none), lists the names it does define, and says: "A --range value with a letter in it is read as a
  named image list; a range contains only digits, '-', ':' and ','."

**Which workspace, and which images.** For the four commands on image
directories, a list replaces the image path arguments: the command takes either
paths or a list, and giving both is an error. The workspace is the one found by
the upward search from the current directory, and the images are the list's
members, in the list's order. For the four commands on a reconstruction, the
list is resolved in the reconstruction's workspace, matched against the
reconstruction's image names rather than the filesystem (so it works where the
image files are absent), and selects the reconstruction's images that are
members, in the reconstruction's order; members the reconstruction does not hold
are ignored, and it is an error, naming the list, when none are left. `--filter-points` keeps its meaning.

A `.sfmr` that `sfm solve` writes from a list records nothing about the list,
as it records nothing about a range today.

## Evidence for the two readings

The definitions come from a measurement on two hand-held phone videos (4K
sources, frames extracted at 1920 pixels on the long side). Every sixth frame was
graded by eye from a crop of its most textured 240-pixel block: 0 sharp, 1 soft,
2 clearly blurred (182 graded frames of one video, 199 of the other). The
experiment's scripts and frames are not in this repository; when this draft is
filed, the measurements go into a sibling `-measurements.md` file.

The numbers below were taken on OpenCV's reduced JPEG decode of the 1920-pixel
frames to a quarter size, a long side of 480, in 8-bit grey (BT.601). The draft
defines an area-averaged 480-pixel resize of the `image_to_gray` image instead,
which does not depend on the JPEG decoder; it has not been measured itself.

- **The Laplacian ranks the blurred frames of a capture.** Its variance at the
  quarter decode separated clearly blurred from sharp frames with an AUC of
  0.858 on one video and 0.945 on the other (Spearman ρ with the grade 0.51 and
  0.65). Flagging frames whose robust z against the capture is above 2 caught
  32% and 27% of the clearly blurred frames and no sharp frame.
- **It depends on scene content.** Replacing 90% of a sharp frame with plain
  wall moved its z to +8.4 and +6.4, more than a Gaussian blur of σ = 1.5 did
  (+1.3 and +1.7). This is why the reading is only compared within a capture.
- **The structure tensor sees blur along one axis that the Laplacian misses.**
  In the second video, frames 51 to 75 are smeared horizontally and graded
  clearly blurred. The Laplacian's median z over them was +0.3 and flagged none
  of them; the log ratio of the x and y gradient energy at the quarter decode
  had a median z of −2.2, and its distance from the capture's median was beyond
  2 in z for 64% of them. Combining the Laplacian with
  the tensor's coherence, taking the larger of their two z-values, raised the
  recall of clearly blurred frames from 32% and 27% to 42% and 49%, with no
  sharp frame flagged. That coherence was measured at a long side of 960
  pixels; at the quarter decode the nearest reading measured, the distance of
  the normalised tensor from the capture's median one, raised recall to 42%
  and 44% with 1% and 0% of sharp frames flagged.
- **Cost.** The Laplacian from an in-memory 1920-pixel grey frame at the quarter
  size took 1.1 ms on one thread, against 5.8 ms to decode the frame.

Measured and not chosen:

- **Full resolution**: the Laplacian on the 4K frames ranked worse (AUC 0.833
  and 0.857) and took 45 ms a frame; at 960 pixels it was 0.839 and 0.920.
- **Tenengrad** (the mean of the Sobel gradient energy): AUC 0.795 and 0.887 at
  the quarter decode, and more sensitive to plain content than the Laplacian.
- **Sum-modified Laplacian**: AUC 0.852 and 0.932 at the quarter decode, no
  better than the Laplacian and more sensitive to plain content.
- **ZNCC self-similarity on 24-pixel crops**: AUC 0.899 and 0.939, but its best
  capture-relative rule flagged no blurred frame in the first video, and it took
  14.9 ms a frame.

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
  frame it runs in a few milliseconds.
- **The workspace file** is written by `init_workspace` in
  `src/sfmtool/_workspace.py`, and the options it writes come from
  `get_default_sfmtool_feature_options`, `get_colmap_feature_options` and
  `get_default_opencv_feature_options`, which gain the `image_to_gray` object
  (the first losing `gray_formula`); the cache hash is `get_feature_tool_xxh128`
  in `src/sfmtool/sift/file.py`, unchanged. `SiftParams::from_feature_options`
  in `sfmtool-core` reads the object instead of `gray_formula`, keeping
  `gray_formula` only for reading version 1 records.
- **The Rust defaults.** `BuildSiftFilesOptions::sfmtool_defaults()` in
  `crates/sfmtool-core/src/features/sift/files.rs` repeats the `sfmtool` default
  options, with `gray_formula`, and the prefix hash computed from them, and a
  test pins it to the prefix `sfm ws init` writes. It gains the `image_to_gray`
  object in place of `gray_formula`, so the test keeps the Rust and Python
  defaults, and their directories, the same.
- **The version and extraction checks**: `load_workspace_config` in
  `src/sfmtool/_workspace.py` checks the workspace version at load; the
  refusal to mix versions is in `image_files_to_sift_files`
  (`src/sfmtool/sift/extract.py`), reached from `sfm sift`, `sfm solve`
  (`src/sfmtool/colmap/db_setup.py`), `sfm match`
  (`src/sfmtool/feature_match/_run.py`) and `sfm to-colmap-db`
  (`src/sfmtool/colmap/db_export.py`), and in core's `build_sift_files`,
  reached from the viewer's *Build SIFT Files*
  (`crates/sfm-explorer/src/sift_files.rs`) and its Python binding
  (`crates/sfmtool-py/src/sift/files.rs`); `sfm undistort`
  (`src/sfmtool/_undistort_images.py`) checks before it writes. `sfm ws upgrade`
  is a new subcommand of the `ws` group in `src/sfmtool/_commands/ws.py`, and
  `sfm ws init` gains `--image-to-gray` and `--quantize-bits` there.
- **The three Python backends** (`extract_sfmtool.py`, `extract_colmap.py`,
  `extract_opencv.py` in `src/sfmtool/sift/`) call it on the decode they make
  the thumbnail from, beside `thumbnail_of_rgb` or `cv2.resize`, and put the
  readings and `image_to_gray` into the metadata they yield. The Rust writer
  `build_sift_files` calls it beside `thumbnail_from_rgb`. `sfm undistort`,
  which writes `.sift` files for the images it undistorts, calls it on each
  undistorted image, as it makes that image's thumbnail.
- **`sfmtool-sift-format`** reads and writes version 2: `SiftMetadata` gains the
  two fields, `validate_sift_data` checks them, and `SIFT_FORMAT_VERSION`
  becomes 2.
- **`sfmtool-sfmr-format`** reads and writes version 15, and the `.sfmr`
  writers gather the readings as [Part 2](#where-a-writer-gets-the-values)
  says: the Python path that copies thumbnails from `.sift` files
  (`src/sfmtool/colmap/io.py`) copies the readings beside them, and the Rust
  save path gains a lookup like `verified_sift_thumbnail` with the photograph
  as the fallback.
- **The named lists** resolve through the `.camrig` pattern matcher in
  `sfmtool-camrig-format` (`pattern_matches`, `pattern_frame_index`) and the
  range type the CLI already parses with (`RangeExpr`, from `openjd-expr`).
  The `-r` handling moves into one shared option in
  `src/sfmtool/_commands/_range_options.py` that all eight commands use.

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
  image's reading image, stored raw as `[Exx, Eyy, Exy]`; its **coherence** and
  **dominant gradient direction** are derived. Chosen over `structure tensor`
  alone, which elsewhere often means a smoothed per-pixel field rather than one
  mean per image.
- **reading image**: the 480-pixel area-averaged grey image both readings are
  taken on.
- **named image list**: a named set of workspace images defined in
  `.sfm-named-image-lists.json` by pattern and range entries. Chosen over `image set`,
  which already names the image half of a reconstruction's data, and over
  `selection`, which the viewer uses for what is picked on screen.

## Choices made here

Decisions inside the maintainer's decisions that this draft makes and that
review should check:

1. **The EXIF orientation is not applied** in `image_to_gray`'s inputs. The
   incremental-extraction draft said it was. Every extractor decodes with
   `read_image_rgb`, which ignores the tag, and keypoints, thumbnails and the
   `.sfmr` thumbnail rule all address stored pixels, so a definition that
   applied the tag would contradict every file the backends write.
2. **The readings live in `metadata.json`**, as two fields, not in a new entry;
   reasons above.
3. **The tensor is stored raw**, not divided by `Exx + Eyy`, because the
   normalised form is one division away while the energy cannot be recovered
   from it, and the energy is itself a contrast reading a reader can use (for
   example to skip near-flat frames, where coherence is unstable).
4. **`float32`** for both `.sfmr` columns, the values computed in double
   precision and rounded to the nearest `float32` when stored: the use needs
   about three significant digits, `float16` would underflow on flat or dark
   captures, and `float32` does not for any realistic value.
5. **The resize rule**: long side to 480, short side rounded to nearest with
   half up, never enlarged; the grey conversion and quantisation happen at full
   resolution before the resize.
6. **The border rule**: interior pixels only, so no padding convention enters
   the definition; readings are `0` for an image under 3 pixels on a side.
7. **The Sobel scale** of 1/8, so the derivative is in grey units per pixel.
8. **The `.sfmr` records one `image_to_gray`** in `images/metadata.json`, and a
   writer recomputes rows recorded under another conversion; for an
   `embedded_patches` file with no `.sift` files it uses unquantised BT.709.
9. **The workspace and the `.sift` file spell the conversion the same way**:
   `.sfm-workspace.json` version 2 carries the `image_to_gray` object in
   `feature_options` for every tool, replacing `gray_formula`, and the `.sift`
   file copies it. A workspace written under the new rules gets a new features
   directory. Every path that writes `.sift` files (`image_files_to_sift_files`
   and core's `build_sift_files`) refuses in a version 1 workspace or a
   directory of version 1 files, and points at `sfm ws upgrade`; reading
   version 1 files stays allowed.
10. **The lists file is named `.sfm-named-image-lists.json`**, after the glossary
    word and the form of `.sfm-workspace.json`, with its own `version` of `1`.
    A key the file does not define, at the top level or in an entry, is an
    error rather than ignored, as `.sfm-workspace.json` has no other keys.
11. **With a list, the image-directory commands take no image paths** and find
    the workspace from the current directory; narrowing a list that has an entry
    with no frame field is an error.
12. **`sfm ws init`, with or without `--force`, writes a version 2
    `.sfm-workspace.json`** with fresh defaults and never touches the lists
    file.
13. **Proposed for review: `sfm ws upgrade`**, a migration command that rewrites
    a version 1 workspace file as version 2 keeping every option, converts the
    grey conversion, recomputes `feature_prefix_dir`, and leaves the old
    features directory and the lists file on disk; the refusal message points at
    it rather than at `sfm ws init --force` with reconstructed flags, which
    cannot reproduce hand-edited options. It also re-derives the directory after
    a hand edit of a version 2 file's options.
14. **Proposed for review: `sfm ws init --image-to-gray FORMULA` and
    `--quantize-bits B`** for the `sfmtool` tool, so the conversion can be set
    and the directory is hashed from it; a hand edit of `image_to_gray` is
    detected as a mismatch with `feature_prefix_dir` when `.sift` files are
    written, not silently honoured.
15. **The workspace file's version is checked at load**: a missing `version`,
    a version other than 1 or 2, a version 2 file with `gray_formula`, without
    `image_to_gray` or with a non-ASCII option string, and a version 1 file with
    `image_to_gray` are all errors there. The cache hash's JSON bytes are
    defined as Python's `json.dumps` output, which a Rust writer matches because
    version 2 options are ASCII.

## Not in this draft

The command that writes named image lists from the readings, and any rule for
which frames count as blurred: the readings are recorded so that such a command
can be written, and it will be designed separately.

Decoding 16-bit sources at full depth, so that a 16-bit image's readings and
features follow the **Inputs** definition rather than an 8-bit reduction of it,
is future work.
