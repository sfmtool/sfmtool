# The SIFT file format

A `.sift` file holds the local image features extracted from one image — the
keypoints, their descriptors, and enough metadata to say which image they came
from and exactly which tool and settings produced them — plus a small thumbnail
of the source image and, from version 2, two readings of how sharp the image is.
One file per image, written once and never modified, so a pipeline can extract
features once and every later stage can find them, check that the image behind
them has not changed, and read only the columns it needs.

## Format design principles

A `.sift` file is an archive-container file, the same container `.matches`,
`.sfmr` and `.camrig` use: a zip file with no zip-level compression, whose
entries are each compressed with zstandard, holding compact JSON for metadata and
little-endian columnar binary for tables, with XXH128 hashes taken over the
uncompressed bytes. [archive-container.md](archive-container.md) specifies it —
the entry layout, the naming that puts a table's shape and data type in its entry
name, and how the hashes are composed and verified. Two things about how `.sift`
uses that container are its own:

1. Metadata about the tool is separated from metadata about the input image file
   and the features produced, so a reader can identify the extraction
   configuration without reading anything about the image.
2. Besides the whole-file content hash — which lets a reference to a `.sift` file
   use the hash stored inside it — the file carries a hash summarizing just the
   feature tool.

The features are derived from one image file, so the metadata records the XXH128 hash and
the size of that file. A tool can compare them with the image on disk to decide whether the
features are still current and extraction can be skipped, which a last-modified timestamp
alone does not establish. XXH128 is not a cryptographic hash, but it is fast to compute and
has strong collision resistance.

This format will produce files larger than necessary due to a few choices made for simplicity.
The affine_shape could be quantized to 16-bit floating point with no expected degradation,
and bit/byte shuffling could be applied [like in blosc](https://www.blosc.org/posts/new-bitshuffle-filter/).

## Format versions

`metadata.version` records the format version. There are two versions, `1` and
`2`, and a conforming writer writes `2`. A reader reads both. It rejects a file
whose version is `0`, which no version of the format has used, or is newer than
the newest version it reads, because it cannot know that layout.

| Aspect | version 1 | version 2 |
|--------|-----------|-----------|
| `metadata.version` | `1` | `2` |
| `feature_options.image_to_gray` | absent; the conversion is the tool's own, and files whose `feature_tool` is `"sfmtool"` record it as a `gray_formula` string | **required** for every feature tool: the formula and an optional `quantize_bits` ([below](#feature_optionsimage_to_gray)) |
| `metadata.sharpness` | absent | **required**, a number ([Sharpness](#sharpness)) |
| `metadata.gradient_structure_tensor_xx_yy_xy` | absent | **required**, an array of three numbers ([Gradient structure tensor](#gradient-structure-tensor)) |
| Entries, entry names, content-hash order | as listed below | unchanged |

A reader of a version 1 file finds no `image_to_gray`, sharpness or tensor in
it. A version 1 file's conversion reads as `{"formula": <gray_formula>}` when
it records a `gray_formula`, and otherwise as the one its feature tool makes
([What each feature tool records](#what-each-feature-tool-records)). A
version 1 file that carries `image_to_gray` or either reading, or a version 2
file without them, is invalid.

Files of the two versions never share a directory: a writer does not write a
version 2 file into a directory that holds version 1 files, so a directory of
`.sift` files is all of one version (see
[workspace.md § Versions never mix](../workspace/workspace.md#versions-never-mix)).

A version 3 built on version 2 is proposed in
[`../drafts/sift-incremental-extraction-amendment.md`](../drafts/sift-incremental-extraction-amendment.md):
it would store descriptors as append-only range chunks so a keypoint pool can be
detected once and described incrementally across several commands, add a
`described_count` coverage entry, and split the whole-file hash into a stable
`feature_set_xxh128` plus a `descriptor_prefix_xxh128` that survives an append.

## File naming and path convention

For an image file `/path/to/myimage.jpg`, a `.sift` file of extracted features goes
in `/path/to/features/{feature_type}-{feature_cache_hash}/myimage.jpg.sift`. The
`features/{feature_type}-{feature_cache_hash}` part is the workspace's `feature_prefix_dir`
when the image is in a workspace, and `{feature_cache_hash}` is a hash of the extraction
configuration; see [Feature Prefix Directory](../workspace/workspace.md#feature-prefix-directory). It is a
different value from the `feature_tool_xxh128` field stored inside the file (see
[Feature tool hash computation](#feature-tool-hash-computation)).
The value of `{feature_type}` encodes the tool and relevant options:

- `sift-colmap` — COLMAP SIFT (default)
- `sift-colmap-dsp` — COLMAP SIFT with domain size pooling
- `sift-colmap-max{N}` — COLMAP SIFT with non-default max features (e.g., `sift-colmap-max500`)
- `sift-colmap-dsp-max{N}` — COLMAP SIFT with DSP and non-default max features
- `sift-opencv` — OpenCV SIFT
- `sift-{tool}` — Generic fallback for other tools

If extending to SURF or other feature types, continue the naming pattern.

This convention provides a predictable way to find the feature file(s) associated with an image,
and ensures that features extracted with different tools or tool options get separated. If a
workspace is using only one feature tool, the features to use are unambiguous as only one
subdirectory of `features` will exist. Otherwise, the mechanism for which features to use is
implementation-defined.

## Specification

A `.sift` file is an [archive-container](archive-container.md) file. It contains
the entries below, all of which are required.

### `feature_tool_metadata.json.zst`

JSON. It contains the following fields (ignore additional fields for future backwards-compatible extension):

* `feature_tool`: (string) The tool used to extract features, e.g. `"colmap"`, `"opencv"`.
* `feature_type`: (string) The type of feature, e.g. `"sift"`. Future feature types
  (SURF, SuperPoint, etc.) would use different values here.
* `feature_options`: (object) All parameters that affect the extracted features. The keys
  are tool-defined — different tools will have different options. Writers should include
  every parameter that affects feature output, and exclude runtime parameters that don't
  (e.g. GPU index, thread count). Because `feature_tool_xxh128` is the hash of this
  entry's bytes, these values determine it
  (see [Feature tool hash computation](#feature-tool-hash-computation)).

  In version 2, `feature_options` carries the image-to-gray conversion as the
  `image_to_gray` object, described next. In version 1 the format does not
  prescribe how the conversion is spelled; files whose `feature_tool` is
  `"sfmtool"` record it as a `gray_formula` string in the formula grammar
  below, with no rounding.

#### `feature_options.image_to_gray`

SIFT and the readings below operate on a single-channel image, and several of
SIFT's parameters, most importantly the contrast threshold, are defined in that
image's value domain. The mapping from the stored image to the grey samples
therefore changes the features, the meaning of the thresholds and the readings,
so a version 2 file records it, for every feature tool, as
`feature_options.image_to_gray`. Being inside `feature_options`, it is part of
`feature_tool_xxh128`.

```json
"image_to_gray": { "formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8 }
```

* `formula` (string, required): an arithmetic expression whose value is the
  grey sample, before any rounding.
* `quantize_bits` (integer from 1 to 16, optional): when present, the formula's
  value is rounded to this many bits over `[0, 1]`. When absent, the formula's
  value is the grey sample as it is.

No other key is defined.

**Inputs.** `R`, `G` and `B` are the image's red, green and blue samples, each
normalised to `[0, 1]` by dividing the stored sample by its full-scale value,
`2^d − 1` for `d`-bit samples (255 for 8-bit, 65535 for 16-bit), so a full-scale
sample is exactly `1`. For a single-channel image, `R = G = B`. An alpha
channel is ignored. The pixels are the image's stored pixels in their stored
order: an EXIF orientation tag is **not** applied, so a keypoint position, a
thumbnail pixel and the grey image all address the same pixel whatever tag the
file carries.

**Grammar.** The formula is an expression over the variables `R`, `G`, `B` and
decimal numeric literals (digits with at most one decimal point, no exponent),
with three operators and parentheses for grouping:

| Operator | Meaning | Precedence | Associativity |
|----------|---------|------------|---------------|
| `**` | power | highest | right |
| `*` | multiply | middle | left |
| `+` | add | lowest | left |

Whitespace is insignificant. A leading `-` on a numeric literal makes a
negative coefficient; there is no subtraction or division operator. A formula
has at most 1000 tokens, each variable, literal, operator and parenthesis
counting as one, and a reader may refuse a longer one.

**Precision.** The formula is evaluated in IEEE-754 single precision
(binary32, `f32`), with every rounding to nearest, ties to even: each input is
the `f32` nearest the normalised sample (the sample and `2^d − 1` are exact,
and their quotient is rounded once), each literal is the `f32` nearest its
decimal text, and the result of each operation in the expression's tree is
rounded to `f32`. `**` is the single-precision power function, which math
libraries do not all round correctly, so for a formula that uses `**` two
implementations can differ in the last bit of a sample; every other operation
gives the same bits in every implementation that follows these rules. Single
precision holds 24 bits, about 7 decimal digits, against the 8 to 16 bits of a
sample, so each rounding is a small fraction of one grey level of the image's
own bit depth; it is twice the values per vector instruction of double
precision and half the memory.

**Output.** Without `quantize_bits`, the grey sample is the formula's value `v`,
not clamped. With `quantize_bits = b`, it is

```
q(v) = round(clamp(v, 0, 1) · (2^b − 1)) / (2^b − 1)
```

where `clamp(v, 0, 1)` is `v` limited to `[0, 1]` and `round` takes the nearest
integer, with an exact half rounded up. Since the argument is never negative,
rounding half up and rounding half away from zero agree. `q` is evaluated in
single precision too: `2^b − 1` is exact, and the product and the quotient are
each rounded to `f32`. The rounded sample is the `f32` nearest a multiple of
`1 / (2^b − 1)` in `[0, 1]`; `quantize_bits: 8` expresses the 8-bit grey image
a tool makes before it runs SIFT. Where the formula's exact value times
`2^b − 1` is a half, as `0.7152 · 14 + 0.0722 · 76 = 15.5` is for the BT.709
weights at `(R, G, B) = (0, 14, 76)` in 8 bits, the level depends on the
roundings before `round`; the single-precision evaluation above decides it
(here 16).

Value-domain parameters such as the contrast threshold assume samples on a
`[0, 1]` scale, so a formula keeps typical values in that range. A tool that
works internally in another scale (0 to 255, say) and scales its thresholds to
match still records the conversion on the `[0, 1]` scale.

| `formula` | `quantize_bits` | Effect |
|-----------|-----------------|--------|
| `0.2126*R + 0.7152*G + 0.0722*B` | absent | BT.709 luma, unrounded |
| `0.2126*R + 0.7152*G + 0.0722*B` | `8` | BT.709 luma rounded to 8 bits |
| `0.299*R + 0.587*G + 0.114*B` | `8` | BT.601 luma rounded to 8 bits |
| `G` | absent | the green channel |
| `0.2126*R**2.2 + 0.7152*G**2.2 + 0.0722*B**2.2` | absent | weights applied after a gamma-style exponent |

A geometric resize the tool applies before detection, such as a cap on the
image size, is a separate concern with its own `feature_options` key and is not
part of `image_to_gray`.

#### What each feature tool records

| `feature_tool` | `image_to_gray` |
|----------------|-----------------|
| `sfmtool` | the conversion the extraction was configured with, by default `{"formula": "0.2126*R + 0.7152*G + 0.0722*B"}`; any formula and any `quantize_bits` |
| `colmap` | `{"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8}` |
| `opencv` | `{"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8}` |
| `sfmtool-undistort` | its source's conversion, since its features are its source's, carried to the undistorted image |

For `colmap` and `opencv` the object records the closest formula to the grey
conversion the tool makes, not always an exact one: the tool's own 8-bit grey
can differ from `q` of the formula by one grey level on a small fraction of
pixels. The readings below are always computed from the recorded object, never
taken from a tool's grey image, so they can be reproduced from the file.

### `metadata.json.zst`

JSON. It should contain the following fields (ignore additional fields for future backwards-compatible extension):

* `version`: (integer) The format version number, `1` or `2` (see
  [Format versions](#format-versions)).
* `image_name`: (string) The image filename without the directory.
* `image_file_xxh128`: (string) The XXH128 sum of the bytes of the image file.
* `image_file_size`: (integer) The number of bytes in the image file.
* `image_width`: (integer) The width of the image, in pixels.
* `image_height`: (integer) The height of the image, in pixels.
* `feature_count`: (integer) The number of features (keypoints).
* `sharpness`: (number, version 2, required) The variance of the Laplacian of
  the reading image, defined in [The reading image](#the-reading-image) and
  [Sharpness](#sharpness). Finite and not negative.
* `gradient_structure_tensor_xx_yy_xy`: (array of three numbers, version 2,
  required) `[Exx, Eyy, Exy]`, the means of `Gx²`, `Gy²` and `Gx·Gy` over the
  reading image, defined in [Gradient structure tensor](#gradient-structure-tensor).
  Finite; `Exx` and `Eyy` are not negative.

`metadata.json`, like every other entry, is written once and never changes: a `.sift` file
is immutable once written. The two readings are therefore fixed when the file is
written, like the image's hash and size beside them, and reach `metadata_xxh128`
and `content_xxh128` through `metadata.json`'s bytes.

Both readings are of the image the file was extracted from, made from the same
decoded pixels as the thumbnail and through the conversion `image_to_gray`
records. Each is computed in single precision, as the next sections define, and
written as a JSON number that reads back as the same value: an `f32` is exactly
a double, so a reader that reads doubles gets the `f32` unchanged. They are for
comparing the images of one capture
with each other: they depend on what the scene holds (a frame that is mostly
plain wall has a low sharpness however well it is focused), so they are not
comparable between captures of different scenes, and the format gives no
threshold for either. Images compared with each other must share one
`image_to_gray`, since the readings are in its grey units.

"Finite" is a rule on the writer as well as on the file. An unclamped formula
can produce a NaN or an infinity (`R**-1` at `R = 0` without `quantize_bits`,
for example), and a non-finite grey sample makes the readings non-finite. A
writer whose readings come out non-finite fails, with an error naming the image
and the formula, and writes no file; it does not store the value or replace it.

#### The reading image

Both readings are taken on one grey image, the **reading image**, built from the
decoded image of width `W` and height `H` in stored pixel order:

1. **Grey.** Each pixel's grey sample is `image_to_gray` applied to it: the
   formula, then `q` when `quantize_bits` is present, in single precision as
   above. Every value from here on is an `f32`, and every product, sum and
   difference below is rounded to `f32`, in the order written.
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
   the rectangle's area. It is computed one axis at a time. Along an axis of
   `n` samples reduced to `m`, in units of `1 / m` of a source pixel, output
   `d` covers `[d · n, (d + 1) · n)` and source `s` covers
   `[s · m, (s + 1) · m)`; the weight of source `s` in output `d` is their
   overlap, an integer, divided by `n`, rounded to the nearest `f32`. Each
   grey row is first reduced across, each of its `W'` outputs being
   `Σ w · g` over its sources in increasing order, added to `0`; then output
   pixel `(i, j)` is `Σ w · a` over the reduced rows `a` under row `j`, in
   increasing order, at column `i`. When the image is used at its own size,
   every weight is `1` and the reading image is the grey image.

Write `I(x, y)` for the reading image's sample at column `x` (left to right) and
row `y` (top to bottom). The **interior** is the pixels with
`1 ≤ x ≤ W' − 2` and `1 ≤ y ≤ H' − 2`, so that every 3 × 3 neighbourhood below
lies inside the image; there are `n = (W' − 2)(H' − 2)` of them. Both readings
are means over the interior. No border pixel is padded, reflected or counted.
If `W' < 3` or `H' < 3`, the interior is empty, and the sharpness and all three
tensor values are `0`.

A **mean over the interior** of a quantity `t(x, y)` is taken in single
precision in a fixed order: each interior row's `t` is summed from `x = 1` to
`x = W' − 2`, starting from `0`; those row sums are summed from `y = 1` to
`y = H' − 2`, starting from `0`; and the total is divided by `n`, which is
exact in an `f32`. Summing by rows bounds the rounding error of the total by
about `(W' + H') · 2^-24` of the sum of the terms' magnitudes, under
`6 · 10^-5`, where a single running sum over all `n` terms would be bounded
only by `n · 2^-24`, about `10^-2`. An implementation that computes these
steps in double precision, or sums in another order, gets readings that differ
from these in the last digits: on photographs, by under `10^-5` of each value
(for `Exy`, of `Exx + Eyy`), or `10^-12` absolute for a reading near `0`. A
sample at an exact or near tie of `q` can be rounded to the neighbouring level
by such an implementation, so an image made mostly of tie colours can read
differently by much more. The readings' use, comparing
the images of one capture, needs about three significant digits.

#### Sharpness

The Laplacian at an interior pixel is the 3 × 3 kernel
`[[0, 1, 0], [1, −4, 1], [0, 1, 0]]`:

```
Λ(x, y) = I(x − 1, y) + I(x + 1, y) + I(x, y − 1) + I(x, y + 1) − 4 · I(x, y)
```

with the additions taken left to right, and the sharpness is its population
variance over the interior, in two passes:

```
μ = (1/n) Σ Λ(x, y)
sharpness = (1/n) Σ (Λ(x, y) − μ)²
```

where both are means over the interior as defined above, and the second pass
computes `Λ − μ` and its square in `f32`.

The units are grey units squared, the grey unit being the `[0, 1]` scale of
`image_to_gray`.

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

with the additions and subtractions taken left to right and the division by
8 exact. `Gx` increases to the right and `Gy` increases downward, the
directions of column and row. The three stored values are means over the
interior, as defined above, of the `f32` products:

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

Each is read against the other images of the capture: a frame whose axis ratio
or coherence is well away from the capture's median, or whose normalised tensor
lies far from the capture's median normalised tensor, is a candidate for motion
blur. The directions are in stored pixel order, so for an image displayed with
its orientation tag applied they turn with the display.

### `content_hash.json.zst`

JSON, containing the following fields:

* `metadata_xxh128`: XXH128 hash of the uncompressed `metadata.json` content bytes.
* `feature_tool_xxh128`: XXH128 hash of the uncompressed `feature_tool_metadata.json`
  content bytes, computed the same way as `metadata_xxh128`.
  See [Feature tool hash computation](#feature-tool-hash-computation).
* `content_xxh128`: the whole-file digest. Every entry is its own one-entry
  section, and the sections contribute in this order:
    1. `feature_tool_metadata.json`
    2. `metadata.json`
    3. `features/positions_xy.{feature_count}.2.float32`
    4. `features/affine_shapes.{feature_count}.2.2.float32`
    5. `features/descriptors.{feature_count}.128.uint8`
    6. `thumbnail_y_x_rgb.128.128.3.uint8`

  See [archive-container.md](archive-container.md) for how those digests are
  taken over the uncompressed bytes and combined.

### `features/positions_xy.{feature_count}.2.float32.zst`

An array of `feature_count` (x, y) coordinate pairs as 32-bit IEEE floating point.
The format follows COLMAP convention that the pixel center of the upper-left pixel is (0.5, 0.5). To convert
to the OpenCV convention of (0, 0) for the center of the upper-left pixel, subtract 0.5 from each coordinate.

### `features/affine_shapes.{feature_count}.2.2.float32.zst`

An array of `feature_count` [[a11, a12], [a21, a22]] affine shape matrices
as 32-bit IEEE floating point. See the [colmap/feature/types.h](https://github.com/colmap/colmap/blob/main/src/colmap/feature/types.h)
file for details including:

From SIFT `scale` and `orientation`, `affine_shape` is [[`scale * cos(orientation)`, `-scale * sin(orientation)`],
[`scale * sin(orientation)`, `scale * cos(orientation)`]].

From `affine_shape`, approximate SIFT `scale = 0.5 * (sqrt(a11 ** 2 + a21 ** 2) + sqrt(a12 ** 2 + a22 ** 2))`
and `orientation = atan2(a21, a11)`.

### Feature ordering

A writer that extracts features from an image writes them — the parallel rows of
`positions_xy`, `affine_shapes`, and the descriptors — in **descending feature size**,
largest first. Feature size is the average of the two affine-shape column norms,
`0.5 * (sqrt(a11² + a21²) + sqrt(a12² + a22²))` (the `scale` formula above). The first `k`
rows are then the `k` largest features, so a consumer that wants at most `k` features per
image reads only that prefix of each entry.

The order is a convention for writers, not a condition of validity. A file whose features
are derived from another `.sift` file keeps the source rows in their source order, so that a
feature index means the same feature in both files, even when the derivation changes the
sizes: undistortion maps each affine shape through the Jacobian of the undistortion, which
can reorder the sizes (see [`sfm undistort`](../cli/image-processing/undistort-command.md)).
A reader or verifier does not reject a file whose rows are out of size order, and a consumer
that reads a prefix gets the first `k` rows, which in such a file are not exactly the `k`
largest.

### Descriptor entries

Each descriptor is an array of 128 unsigned bytes. The meaning of the descriptor is
determined by the `feature_tool` and `feature_options` values — for example, whether the
descriptors use the original SIFT formulation or domain size pooling depends on the
`domain_size_pooling` field in `feature_options`.

All `feature_count` descriptors live in one entry,
`features/descriptors.{feature_count}.128.uint8.zst`, in the same row order as
`positions_xy` and `affine_shapes`. The width is part of the entry's name, so a
reader is never told how wide a descriptor is and a consumer laying several
files' descriptors out end to end can size the buffer before it opens one.

### `thumbnail_y_x_rgb.128.128.3.uint8.zst`

A 128×128 RGB thumbnail of the source image, embedded at feature extraction time so that downstream
consumers (`.sfmr` files, viewers) can display previews without re-reading the source image.

* **Shape**: `(128, 128, 3)`
* **Data type**: `uint8`
* **Format**: Row-major RGB data. 128 rows of 128 pixels, each pixel 3 bytes [R, G, B] in range [0, 255]
* **Dimension order**: `(y, x, channel)` — y is the row (top-to-bottom), x is the column (left-to-right)
* **Size**: Fixed 128×128 square, regardless of the source image aspect ratio. The source image is
  resized to fill the square (stretching if non-square). Consumers restore the correct aspect ratio
  at display time using the image dimensions from `metadata.json.zst`
* **Resize method**: Area-averaging (as OpenCV's `INTER_AREA` does), which antialiases
  better than bilinear when downscaling.

When writing a `.sfmr` file, the thumbnail is copied directly from the `.sift` file rather than
re-reading and re-downscaling the source image. Because it is copied rather than regenerated, the
`.sfmr` thumbnail edge must equal this one; see the `images/thumbnails_y_x_rgb` section of
[`sfmr-file-format.md`](sfmr-file-format.md).

## Feature tool hash computation

`feature_tool_xxh128` is the XXH128 hash of the uncompressed bytes of the
`feature_tool_metadata.json` entry, exactly as stored in the file. A writer computes it from
the bytes it writes, and a verifier recomputes it from the bytes it reads and reports a
mismatch as a hash error, the same as for `metadata_xxh128`. The value identifies the
extraction configuration only as far as the serialized `feature_tool_metadata.json` does: two
writers that serialize the same configuration to different bytes (key order, float
formatting, whitespace) produce different values.

`.sfmr` and `.matches` files record this field per image, copied from each `.sift` file's
`content_hash.json`, so a consumer can confirm which feature configuration it was built from.

The hash in the workspace's `feature_prefix_dir` directory name is a separate value. It is
computed from `feature_tool`, the configured `feature_type` (for example `sift-colmap`) and
`feature_options` minus options that do not affect the output. A workspace computes it once, when
the folder is created, and stores it in `feature_prefix_dir`, which is read back as it is
from then on; for an image outside a workspace it is
computed from the extraction options at extraction time. It is never computed from a `.sift`
file; see
[Feature Prefix Directory](../workspace/workspace.md#feature-prefix-directory). The two hashes
do not in general agree, and a reader must not compare one with the other, although in
version 2 both are taken over inputs that hold the same `image_to_gray` object.

## Implementations

The code that reads, writes and verifies `.sift` files is:

- Rust: `read_sift`, `read_sift_metadata`, `write_sift` and `verify_sift` in
  [`sfmtool-sift-format`](../../crates/sfmtool-sift-format/src/lib.rs), with
  readers for part of a file: `read_sift_partial` (the first `k` features),
  `read_sift_positions`, `read_sift_keypoints`, `read_sift_features` (the three
  feature columns only) and `read_sift_thumbnail`. Every reader rejects an
  unsupported `metadata.version`; `write_sift` refuses to write one; and
  `verify_sift`, which returns `(is_valid, error_messages)`, reports it as an
  error. Every reader also rejects a version 2 file without the two readings
  and a version 1 file with either. None of them checks the feature order. The
  readers ignore a metadata key the format does not define, as the entries
  above ask. `SiftMetadata` holds the readings as `Option`s, `None` in a
  version 1 file, whose `metadata.json` is written without the keys. A writer is
  stricter: `validate_sift_data`, which `write_sift` calls first and
  `verify_sift` applies too, refuses an unsupported version, `feature_options`
  that is not an object, a version 2 file whose `feature_options` carry no
  `image_to_gray` of the right shape (a string `formula`, an optional
  `quantize_bits` from 1 to 16, no other key) or a reading that is not finite
  or is negative where it must not be, a version 1 file with `image_to_gray`,
  and an array whose shape disagrees with `feature_count` or the thumbnail
  edge, and
  `SiftMetadata::from_json_for_write` and
  `FeatureToolMetadata::from_json_for_write` parse metadata given as JSON,
  refusing an unknown, missing or mistyped key. The zstd level a `.sift` is
  written at unless a caller asks otherwise is `SIFT_ZSTD_LEVEL`, 5.
- Python: in `sfmtool.fileio` ([bindings](../../crates/sfmtool-py/src/fileio/sift.rs)),
  `read_sift`, `read_sift_metadata`, `read_sift_partial`, `write_sift` and
  `verify_sift`, which take and return a dict of NumPy arrays and metadata,
  the readings present in a version 2 file's `metadata` dict and absent from a
  version 1 file's;
  `SiftFile`, an open file that reads the metadata when it is opened and the
  columns and thumbnail on request, the whole file or its first `count`
  features, and is what most of the Python package reads through; and
  `SiftWriteQueue`, which writes files on the rayon pool for `sfm sift`
  ([`sfm sift` § Extraction-orchestration
  pipelining](../cli/image-feature/sift-command.md#extraction-orchestration-pipelining)).
  `write_sift(path, data, zstd_level=5)` and `SiftWriteQueue.submit` parse
  and check `data` with the Rust functions above and raise `ValueError` for
  what they refuse, `TypeError` for an array of the wrong dtype and `KeyError`
  for a missing key, all before anything is written. Every `SiftError` raises
  through the shared table in
  [python-bindings.md § Format errors](../python-bindings.md#format-errors):
  a failed filesystem call is the `OSError` subclass for its kind
  (`FileNotFoundError` for a missing file), and a file that cannot be parsed
  or breaks a rule of the format is an `OSError`. A failed save in a
  `SiftWriteQueue` raises from the call that waits for it: `join`,
  `join_oldest`, or a `submit` that waits for room.
  There is no Python wrapper over these bindings; the helpers in
  [`sfmtool.sift.file`](../../src/sfmtool/sift/file.py) find an image's
  `.sift` path and compute the feature-cache hash and affine-shape sizes.

`SIFT_FORMAT_VERSION` in
[`types.rs`](../../crates/sfmtool-sift-format/src/types.rs) is the newest
version the crate reads and writes, `2`. The format crate checks the shape of
`image_to_gray` and not its grammar; the formula is parsed, with the token
limit, by `ImageToGray` in
[`sfmtool-core`](../../crates/sfmtool-core/src/features/sift/gray.rs), and the
readings are computed by one function, `image_readings_from_rgb` in
[`image_readings.rs`](../../crates/sfmtool-core/src/features/sift/image_readings.rs),
bound for Python as `sfmtool.sift.image_readings_from_rgb`. The descriptor width and the thumbnail edge
each have a single declaration there, `DESCRIPTOR_DIM` and `THUMBNAIL_SIZE`,
from which the entry names, the read path, the write path and the shape checks
are derived. A compile-time assertion in `sfmtool-core` keeps `THUMBNAIL_SIZE`
equal to `sfmtool_sfmr_format::THUMBNAIL_SIZE`, and the value is exported to
Python as `sfmtool.THUMBNAIL_SIZE` for the extractors that produce the pixels.

The three extraction backends of [`sfm sift`](../cli/image-feature/sift-command.md)
(`colmap`, `opencv` and `sfmtool`) sort their features by descending size and
resize the thumbnail with OpenCV's `INTER_AREA`. All three decode with
`sfmtool.fileio.read_image_rgb`, which ignores the EXIF orientation, make their
thumbnail from that decode, and compute the readings from the same decode; the
`colmap` backend's detector reads the image through COLMAP's own decoder. The
Rust writer for the `sfmtool` tool, `build_sift_files` in
[`sfmtool-core`](../../crates/sfmtool-core/src/features/sift/files.rs), writes
the same files as that backend, its thumbnail resize following `INTER_AREA`'s
arithmetic (see [SIFT § Building `.sift` files](../core/features/sift.md#building-sift-files)).
`sfm undistort` computes its files' readings by decoding the undistorted images
it wrote.

How each tool's recorded conversion relates to the grey image it makes:

| `feature_tool` | Its grey conversion |
|----------------|---------------------|
| `sfmtool` | exact: `ImageToGray::gray_row` in `sfmtool-core` divides each 8-bit channel by 255, evaluates the formula and applies `q` when `quantize_bits` is present, in single precision, and gives the detector and the readings the same `f32` samples |
| `colmap` | measured with pycolmap 4.2.1: `Bitmap.read(path, as_rgb=False)` gives an 8-bit grey bitmap equal to `clone_as_grey()` of the RGB read, the BT.709 weights applied in single precision to the 0 to 255 samples and rounded to nearest with half up; `q` of the recorded formula differs from it by one grey level on 530 of the 2^24 8-bit colours (0.003%), measured on an image holding each colour once. A JPEG carrying EXIF orientation 6 read back at its stored size, so the tag is not applied |
| `opencv` | `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)` on the 8-bit decode, measured with OpenCV 5.0.0 as `(9798·R + 19235·G + 3735·B + 16384) >> 15` on the 0 to 255 samples; `q` of the recorded BT.601 formula differs from it by one grey level on 20,885 of the 2^24 8-bit colours (0.12%) |

**16-bit images do not yet follow the definition.** Every backend's decoder
reduces a 16-bit image to 8 bits per channel, `round(c · 255 / 65535)`, before
anything else, so for such an image the detector, the thumbnail and the readings
all see 8-bit samples rather than the 16-bit samples divided by 65535 that
**Inputs** defines. For 8-bit images the code follows the definition.

## Using CLI commands to pull apart a .sift file

Here's how to print metadata and the first 5 keypoint positions of a `.sift` file:

```bash
$ unzip myimage.jpg.sift
Archive:  myimage.jpg.sift
 extracting: feature_tool_metadata.json.zst
 extracting: metadata.json.zst
 extracting: features/positions_xy.2464.2.float32.zst
 extracting: features/affine_shapes.2464.2.2.float32.zst
 extracting: features/descriptors.2464.128.uint8.zst
 extracting: thumbnail_y_x_rgb.128.128.3.uint8.zst
 extracting: content_hash.json.zst

$ zstd -d --rm *.zst features/*.zst
7 files decompressed

$ jq . feature_tool_metadata.json
{
  "feature_tool": "colmap",
  "feature_type": "sift",
  "feature_options": {
    "max_image_size": 4096,
    "max_num_features": 8192,
    "estimate_affine_shape": true,
    "domain_size_pooling": true,
    "dsp_min_scale": 0.16666666666666666,
    "dsp_max_scale": 3.0,
    "dsp_num_scales": 10,
    "peak_threshold": 0.006666666666666667,
    "edge_threshold": 10.0,
    "upright": false,
    "normalization": "L1_ROOT",
    "image_to_gray": {
      "formula": "0.2126*R + 0.7152*G + 0.0722*B",
      "quantize_bits": 8
    }
  }
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

$ jq . content_hash.json
{
  "metadata_xxh128": "a7b3c1d2e4f56789abcdef0123456789",
  "feature_tool_xxh128": "8f0d6e2b71c94a35b2e1d07c6a59f3e4",
  "content_xxh128": "5a90164dc1d970770e2a881114ad040a"
}

$ od -Ax --endian=little -tf4 -w8 features/positions_xy.*.2.float32 | head -5
000000       154.56075       277.25787
000008       87.453865       193.33382
000010       182.62576       200.48529
000018        90.68476        96.82127
000020       111.88794       32.918064
```

## Version History

- **Version 2**: the current version, written as the integer `2` in
  `metadata.version`. It makes the `image_to_gray` object required in
  `feature_options` for every feature tool and adds the image's `sharpness` and
  `gradient_structure_tensor_xx_yy_xy` to `metadata.json`. The entries, their
  names and the content-hash order are unchanged.
- **Version 1**: the first version. It has no `image_to_gray` object and no
  readings. Version 1 files stay valid and readable.

A proposed version 3, with incremental chunked descriptors and the identity
hashes that go with them, is drafted in
[`../drafts/sift-incremental-extraction-amendment.md`](../drafts/sift-incremental-extraction-amendment.md).
