# The SIFT file format

## Format design principles

Here are the basic principles used to create the format:

1. The format is a zip file with no compression. With a zip file, you can access
    any file and read the start of the file without reading all data.
2. All data is compressed with zstandard. Each file in the zip has a `.zst` extension.
3. Metadata that occurs just once for the file is stored in JSON format.
4. Metadata about the tool is separated from metadata about the input image file
   and the features produced.
5. While writing the file, a hash summarizing the full contents of the file is created,
   and placed in a content metadata file. References to a .sift file can therefore use
   the content hash stored in the file itself. Additionally, a hash summarizing just the
   feature tool is included.
6. Tables are stored as one file per field. Each of these files
    contains only one primitive data type.
7. Numeric data is stored in little-endian raw binary format, using C/row-major order.
8. Data file extensions contain the tensor shape and primitive data type. E.g.
    `.1044.2.2.float32` means a tensor with shape (1044, 2, 2) and 32-bit IEEE
    floating point data. The file must contain 1044 * 2 * 2 * 4 = 16704 bytes.
9. File names in the zip, field names in the JSON, and field names in tables are
   all selected with an attempt to be self-documenting. Someone encountering a `.sift`
   file without seeing this specification will have an easier time understanding it.

SIFT features are always derived from an image, and if an image is unchanged from when
they were calculated before, it's nice to avoid re-computing the features. The `sfmtool`
command uses last-modified timestamps to do this, but it's nice to have stronger verifiability,
so we also put the XXH128 hash of the source image file in the metadata. While XXH128 is
not a cryptographic hash, it is fast to compute and has strong collision resistance.

This format will produce files larger than necessary due to a few choices made for simplicity.
The affine_shape could be quantized to 16-bit floating point with no expected degradation,
and bit/byte shuffling could be applied [like in blosc](https://www.blosc.org/posts/new-bitshuffle-filter/).

## Format versions

`metadata.version` records the format version. Version 1 is the original layout. Version 2
adds the fields and entries listed below, requires the `image_to_gray` grayscale-conversion
formula (see [Image-to-gray conversion](#image-to-gray-conversion)), stores
descriptors as append-only chunks rather than a single array, and redefines `content_xxh128`;
see [Incremental descriptor extraction](#incremental-descriptor-extraction-version-2).

> **Implementation status.** The `sfmtool` codebase currently reads and writes **version 1
> only** (`sift-format` always writes `metadata.version = 1`, and its reader rejects files
> with a newer `metadata.version`). The version 2 layout described
> below — chunked descriptors, `described_count`, the redefined `content_xxh128`, the
> `feature_set_xxh128` / `descriptor_prefix_xxh128` / `component_xxh128` hashes, and the
> `sfm sift --detect` / `--describe` flow — is a **draft design that is not yet implemented**.
> The `[v2]`-tagged items below describe that planned layout, not current behavior.

**Notation.** In the sections below, items that apply to only one version are tagged
**[v1]** (version 1 only) or **[v2]** (version 2 only); untagged items apply to both.

### Summary of differences

| Aspect | **[v1]** | **[v2]** |
|--------|----------|----------|
| `metadata.version` | `1` | `2` |
| `feature_options.image_to_gray` | absent (conversion implicit/tool-defined) | **required**; the image→gray conversion formula |
| `features/descriptors_metadata.json` | absent (all keypoints described) | present; records `described_count` (prefix length `[0, described_count)`) |
| Descriptor entries | single `features/descriptors.{feature_count}.128.uint8` | append-only chunks `features/descriptors.{start}-{end}.128.uint8` |
| Mutability | write-once, immutable | only descriptors (appended) and `content_hash.json` (rewritten) change; `metadata.json` stays immutable |
| `content_xxh128` | digest-of-digests over all entries | `xxh128(feature_set_xxh128 ‖ descriptor_prefix_xxh128 ‖ described_count)` |
| `feature_set_xxh128` | — | stable id, invariant across appends |
| `descriptor_prefix_xxh128` | — | hash of the described prefix |
| `component_xxh128` | — | cached per-entry digests |

## File naming and path convention

For an image file `/path/to/myimage.jpg`, a `.sift` file of extracted features goes
in `/path/to/features/{feature_type}-{feature_tool_xxh128}/myimage.jpg.sift`, where `{feature_tool_xxh128}`
is the feature tool hash (see [Feature tool hash computation](#feature-tool-hash-computation)).
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

A `.sift` file is a [zip file](https://en.wikipedia.org/wiki/ZIP_(file_format)) using the STORE
method (no ZIP-level compression). It contains the entries below. Most are required in both
versions; entries and fields that are version-specific are tagged **[v1]** / **[v2]** as
described in [Format versions](#format-versions).

### `feature_tool_metadata.json.zst`

This is JSON data compressed with [zstd](https://en.wikipedia.org/wiki/Zstd). It contains
the following fields (ignore additional fields for future backwards-compatible extension):

* `feature_tool`: (string) The tool used to extract features, e.g. `"colmap"`, `"opencv"`.
* `feature_type`: (string) The type of feature, e.g. `"sift"`. Future feature types
  (SURF, SuperPoint, etc.) would use different values here.
* `feature_options`: (object) All parameters that affect the extracted features. The keys
  are tool-defined — different tools will have different options. Writers should include
  every parameter that affects feature output, and exclude runtime parameters that don't
  (e.g. GPU index, thread count). This object, together with `feature_tool` and
  `feature_type`, is used to compute `feature_tool_xxh128`
  (see [Feature tool hash computation](#feature-tool-hash-computation)).
* **[v2]** `feature_options` MUST include an `image_to_gray` object, which lets readers
  reproduce the exact single-channel float values that went into the feature extraction
  algorithm; see [Image-to-gray conversion](#image-to-gray-conversion). Version 1
  files do not carry it (the conversion is implicit and tool-defined).

### `metadata.json.zst`

This is JSON data compressed with zstd. It should contain
the following fields (ignore additional fields for future backwards-compatible extension):

* `version`: (integer) The format version number — `1` or `2` (see
  [Format versions](#format-versions)).
* `image_name`: (string) The image filename without the directory.
* `image_file_xxh128`: (string) The XXH128 sum of the bytes of the image file.
* `image_file_size`: (integer) The number of bytes in the image file.
* `image_width`: (integer) The width of the image, in pixels.
* `image_height`: (integer) The height of the image, in pixels.
* `feature_count`: (integer) The number of features (keypoints).

`metadata.json` is written once and never changes, including across descriptor appends in
version 2. The mutable descriptor-coverage count (`described_count`) lives in
[`features/descriptors_metadata.json`](#featuresdescriptors_metadatajsonzst), not here, so
`metadata.json` (and `metadata_xxh128`) stay constant as descriptors are appended.

### `content_hash.json.zst`

This is JSON data compressed with zstd, that contains the following fields:

* `metadata_xxh128`: XXH128 hash of the uncompressed `metadata.json` content bytes.
* `feature_tool_xxh128`: Hash identifying the feature extraction configuration, derived from
  `feature_tool`, `feature_type`, and `feature_options`. Computed once during workspace
  initialization and propagated from there.
  See [Feature tool hash computation](#feature-tool-hash-computation).
* `metadata_xxh128` and `feature_tool_xxh128` cover the current metadata and the tool
  configuration; both are present in all versions.

**[v1]** Version 1 defines a single whole-file hash:

* `content_xxh128`: Hash of hashes. XXH128 hash of the concatenation of the
    xxh128 binary hash digests (each serialized as 16 bytes in big-endian order) of the
    following entries, in order:
    1. `feature_tool_metadata.json` (uncompressed)
    2. `metadata.json` (uncompressed)
    3. `features/positions_xy.{feature_count}.2.float32` (uncompressed)
    4. `features/affine_shapes.{feature_count}.2.2.float32` (uncompressed)
    5. `features/descriptors.{feature_count}.128.uint8` (uncompressed)
    6. `thumbnail_y_x_rgb.128.128.3.uint8` (uncompressed)

**[v2]** Version 2 splits identity into a part that is stable across descriptor appends and
a part that grows, so that references survive expansion (see
[Stable identity](#stable-identity)). It defines:

* `feature_set_xxh128`: **Stable across the file's entire life.** XXH128 of the
    concatenation of these 16-byte digests, in order — the version 1 `content_xxh128` inputs
    minus the descriptors:
    1. xxh128 of `feature_tool_metadata.json` (uncompressed)
    2. xxh128 of `metadata.json` (uncompressed)
    3. xxh128 of `features/positions_xy.{feature_count}.2.float32` (uncompressed)
    4. xxh128 of `features/affine_shapes.{feature_count}.2.2.float32` (uncompressed)
    5. xxh128 of `thumbnail_y_x_rgb.128.128.3.uint8` (uncompressed)

    Because `metadata.json` is immutable (it no longer carries `described_count`), this digest
    is constant for the file's entire life. It excludes descriptors, so it identifies *which
    keypoints from which image with which tool config* and never changes as descriptors are
    filled in. This is the recommended value for other files to reference.
* `descriptor_prefix_xxh128`: XXH128 of the concatenation of the per-chunk digests of the
    descriptor chunk entries covering `[0, described_count)`, in ascending `start` order
    (the xxh128 of the empty string when `described_count == 0`). For a given prefix
    length it is **reproducible from any later, expanded file** by truncating to that
    length, because chunks are append-only and never rewritten.
* `content_xxh128`: The exact-current-state hash, equal to
    `xxh128(feature_set_xxh128 ‖ descriptor_prefix_xxh128 ‖ u64_be(described_count))`
    (the two hashes as their 16-byte digests). Changes on every append. Because it is a
    pure function of the stable id and the reproducible prefix hash, a consumer that knows
    the `described_count` it pinned can re-derive and verify it against the current file.
* `component_xxh128`: (object) The individual xxh128 digest of each uncompressed entry
    (`feature_tool_metadata.json`, `metadata.json`, `positions_xy`, `affine_shapes`,
    `thumbnail`, and each `descriptors.{start}-{end}` chunk), keyed by entry name. Caches the
    per-entry digests so appending a chunk recomputes the hashes above without re-reading the
    large `positions_xy`, `affine_shapes`, or pre-existing descriptor chunks — see
    [Appending a descriptor chunk](#appending-a-descriptor-chunk).

### `features/positions_xy.{feature_count}.2.float32.zst`

This is a zstd-compressed little-endian array of `feature_count` (x, y) coordinate pairs as 32-bit IEEE floating point.
The format follows COLMAP convention that the pixel center of the upper-left pixel is (0.5, 0.5). To convert
to the OpenCV convention of (0, 0) for the center of the upper-left pixel, subtract 0.5 from each coordinate.

### `features/affine_shapes.{feature_count}.2.2.float32.zst`

This is a zstd-compressed little-endian array of `feature_count` [[a11, a12], [a21, a22]] affine shape matrices
as 32-bit IEEE floating point. See the [colmap/feature/types.h](https://github.com/colmap/colmap/blob/main/src/colmap/feature/types.h)
file for details including:

From SIFT `scale` and `orientation`, `affine_shape` is [[`scale * cos(orientation)`, `-scale * sin(orientation)`],
[`scale * sin(orientation)`, `scale * cos(orientation)`]].

From `affine_shape`, approximate SIFT `scale = 0.5 * (sqrt(a11 ** 2 + a21 ** 2) + sqrt(a12 ** 2 + a22 ** 2))`
and `orientation = atan2(a21, a11)`.

### Feature ordering

Features — the parallel rows of `positions_xy`, `affine_shapes`, and the descriptors — are
ordered by **descending feature size**, largest first. Feature size is the average of the
two affine-shape column norms,
`0.5 * (sqrt(a11² + a21²) + sqrt(a12² + a22²))` (the `scale` formula above).

### Descriptor entries

Each descriptor is an array of 128 unsigned bytes. The meaning of the descriptor is
determined by the `feature_tool` and `feature_options` values — for example, whether the
descriptors use the original SIFT formulation or domain size pooling depends on the
`domain_size_pooling` field in `feature_options`. The two versions store the descriptors
differently:

* **[v1]** `features/descriptors.{feature_count}.128.uint8.zst` — a single zstd-compressed
  array of all `feature_count` descriptors.
* **[v2]** `features/descriptors.{start}-{end}.128.uint8.zst` — one or more append-only
  *chunk* entries that tile the described prefix `[0, described_count)`. See
  [Incremental descriptor extraction](#incremental-descriptor-extraction-version-2) for the
  chunk naming, ordering, and append rules.

### `features/descriptors_metadata.json.zst`

**[v2]** Metadata about descriptor coverage. JSON compressed with zstd, containing:

* `described_count`: (integer) The number of keypoints that have a stored descriptor.
  Descriptors cover the contiguous prefix `[0, described_count)` of the feature list, so
  `0 ≤ described_count ≤ feature_count`.

A descriptor append rewrites this entry (together with `content_hash.json`). `described_count`
is also derivable from the descriptor chunk entries (`last_chunk.end + 1`, or `0` when there
are none); a verifier cross-checks the two. The value is folded into `content_xxh128` as a
`u64` (see [`content_hash.json`](#content_hashjsonzst)), so its integrity is covered there.
Version 1 files do not have this entry (all keypoints are described).

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
* **Resize method**: Area-averaging (OpenCV `INTER_AREA`), which antialiases better than
  bilinear when downscaling. All three extraction backends (colmap, opencv, sfmtool) use it.

When writing a `.sfmr` file, the thumbnail is copied directly from the `.sift` file rather than
re-reading and re-downscaling the source image.

## Incremental descriptor extraction (version 2)

**[v2]** Everything in this section applies only to version 2 files.

Version 2 makes a `.sift` file **growable**: a keypoint pool can be detected once and its
descriptors filled in incrementally, across multiple invocations, without re-reading or
rewriting the bulk data already on disk. This supports a *detect-many, describe-few*
workflow and coarse-to-fine matching (describe the largest keypoints first, finer ones only
where needed). This section is the normative on-disk definition.

### Descriptor coverage

Descriptors cover the contiguous prefix `[0, described_count)` of the feature list. The
format's [descending-size feature ordering](#feature-ordering) makes this prefix exactly the
`described_count` largest features — the natural granularity for coarse-to-fine — so coverage
needs only the single `described_count` integer, with no sparse mask. A partial reader can
pull the top-K features (`positions_xy[0..K]`, `affine_shapes[0..K]`) together with exactly
the descriptor chunks covering `[0, K)`.

`positions_xy` and `affine_shapes` are written **once**, in full (shape `feature_count`);
only descriptors grow.

### Descriptor chunk entries

Descriptors are stored as a sequence of append-only chunk entries, each named by the
**inclusive** keypoint-index range it covers:

```
features/descriptors.0-100.128.uint8.zst      # keypoints 0..=100   (101 rows)
features/descriptors.101-1000.128.uint8.zst   # keypoints 101..=1000 (900 rows)
features/descriptors.1001-4095.128.uint8.zst  # keypoints 1001..=4095
```

The leading `{start}-{end}` token replaces the row-count dimension of the
[extension shape convention](#format-design-principles): the row count is `end - start + 1`,
so an entry must contain `(end - start + 1) * 128 * 1` uncompressed bytes. The remaining
`.128.uint8` is the descriptor dimension and dtype as usual. Constraints:

* The first chunk starts at `0`.
* Chunks are contiguous and gap-free: each chunk's `start` equals the previous chunk's
  `end + 1`.
* The chunks jointly cover `[0, described_count - 1]`; equivalently
  `described_count = last_chunk.end + 1`, or `0` when there are no chunks.
* Chunks never overlap and are never rewritten once written.

A file with `described_count == 0` (detect-only) contains **no** `features/descriptors.*`
entries at all.

### Appending a descriptor chunk

To describe keypoints `[a, b]` (where `a == described_count`):

1. Compute the descriptors for keypoints `a..=b`.
2. **Append** the immutable entry `features/descriptors.{a}-{b}.128.uint8.zst` to the ZIP.
   Existing entries are not touched.
3. Rewrite the two small mutable JSON entries:
   * `features/descriptors_metadata.json.zst`: set `described_count = b + 1`.
   * `content_hash.json.zst`: add the new chunk's digest to `component_xxh128`, then recompute
     `descriptor_prefix_xxh128` and `content_xxh128`. `feature_set_xxh128`, `metadata_xxh128`,
     and `metadata.json` are unchanged.

The ZIP-level mechanics of performing this append in place — how the central directory and
the replaced JSON entries are rewritten — and the locking or coordination needed for safe
concurrent writers are **implementation-defined**. The only requirement is that the resulting
file is a valid `.sift` as specified here, and that concurrent readers always observe a
consistent one.

### Stable identity

A version 2 file's `content_xxh128` changes on every descriptor append, so other files that
pin a `.sift` by its whole-file hash (`.sfmr`, `.matches`, workspace caches) would see their
reference "break" the moment the file is expanded — even though the new file is a strict
superset of what they referenced. Reference `feature_set_xxh128` instead:

**Reference the stable id.** `feature_set_xxh128` is invariant for the file's entire life
and is recomputable from the immutable entries, so it is the correct value to record when a
consumer cares about *which keypoints* (the common case). It never breaks under descriptor
expansion or conversion to the version 1 layout. New and updated references SHOULD use it;
the `.sfmr`/`.matches`/workspace specs that currently store a `.sift` `content_xxh128` should
be updated to store `feature_set_xxh128`.

**Descriptor-dependent consumers verify the prefix.** A `.matches` file built from the first
`M` descriptors remains valid under expansion because chunks `[0, M)` are immutable. Such a
consumer should record `(feature_set_xxh128, M, descriptor_prefix_xxh128@M)` and re-verify by
recomputing `descriptor_prefix_xxh128` over `[0, M)` on the current file — which holds
regardless of how many additional descriptors were appended afterward. (This is also why no
whole-file hash history is needed: identity uses the stable hash, and descriptor state is
verified against the immutable prefix.)

## Image-to-gray conversion

**[v2]** `feature_options.image_to_gray` is a version 2 addition; version 1 files do not
record it.

SIFT operates on a single-channel floating-point image, and several of its parameters are
defined in that value domain — most importantly the contrast threshold (Lowe discards
extrema with `|D(x̂)| < 0.03`, *"assuming image pixel values in the range [0, 1]"*). The
mapping from the stored source image to those float samples therefore changes both the
features produced and the meaning of those thresholds, so it must be pinned for
reproducibility and for cache identity. It is recorded as `feature_options.image_to_gray`,
so it participates in `feature_tool_xxh128` — two conversions yield different features and a
different `features/` subdirectory.

The conversion is a single arithmetic **formula** over the colour channels:

```json
"image_to_gray": { "formula": "0.2126*R + 0.7152*G + 0.0722*B" }
```

**Inputs.** `R`, `G`, `B` are the decoded source image's red, green, and blue channel
values, each normalized to `[0, 1]` (the raw sample divided by its full-scale value — e.g.
255 for 8-bit, 65535 for 16-bit — so the formula is independent of bit depth). For a
single-channel source, `R == G == B`. The source is decoded with its EXIF orientation
applied and any alpha channel ignored.

**Output.** The formula's value is the single-channel sample the detector operates on, used
as-is (not clamped). Value-domain parameters (the `0.03` contrast threshold, the standard
SIFT defaults) assume inputs on a `[0, 1]` scale, so formulas should keep typical values in
that range.

**Grammar.** The formula is an arithmetic expression over the variables `R`, `G`, `B` and
decimal numeric literals, using three operators plus parentheses for grouping:

| Operator | Meaning | Precedence | Associativity |
|----------|---------|------------|---------------|
| `**`     | power   | highest    | right         |
| `*`      | multiply| middle     | left          |
| `+`      | add     | lowest     | left          |

Whitespace is insignificant; evaluation is in IEEE-754 double precision. (A leading `-` on a
numeric literal expresses a negative coefficient; there is no subtraction or division
operator — use negative literals and fractional coefficients.)

Examples:

| `formula` | Effect |
|-----------|--------|
| `0.2126*R + 0.7152*G + 0.0722*B` | BT.709 luma (equals COLMAP `CloneAsGrey`) |
| `0.299*R + 0.587*G + 0.114*B` | BT.601 luma (equals OpenCV `BGR2GRAY`) |
| `0.3333*R + 0.3334*G + 0.3333*B` | equal-weight average |
| `G` | green channel only |
| `0.2126*R**2.2 + 0.7152*G**2.2 + 0.0722*B**2.2` | weighting with a gamma-style exponent |

This single formula subsumes channel weighting, channel selection, and any
gamma/linearization (via `**`); there is no separate notion of "encoded vs linear" —
whatever the expression computes is what the detector sees.

Geometric pre-resizing (e.g. COLMAP's `max_image_size` downscale) also affects features but
is a separate concern, represented by its own `feature_options` keys; it is not part of
`image_to_gray`.

## Feature tool hash computation

The `feature_tool_xxh128` identifies a specific feature extraction configuration. It is a hash
derived from `feature_tool`, `feature_type`, and `feature_options`.

The spec does not prescribe a specific serialization or hashing algorithm. The implementation
computes the hash during workspace initialization (`sfm ws init`) and stores the result in
`.sfm-workspace.json` as part of `feature_prefix_dir`. From that point on, the hash is never
recomputed — it is read from the workspace config and propagated into `.sift`, `.sfmr`, and
`.matches` files as-is.

This avoids cross-implementation issues with floating point serialization.

Implementations should make a best effort to compute the hash deterministically from the
logical configuration values, so that reinitializing a workspace with the same settings
produces the same `feature_prefix_dir` and reuses cached features. For example, an
implementation might serialize the options with sorted keys and a consistent float format,
then hash the result. The important thing is that a single implementation is consistent
with itself — cross-implementation agreement is nice to have but not required.

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
    "normalization": "L1_ROOT"
  }
}

$ jq . metadata.json
{
  "version": 1,
  "image_name": "myimage.jpg",
  "image_file_xxh128": "3748f9341bfdcc712beb2c5392664982",
  "image_file_size": 21584,
  "image_width": 270,
  "image_height": 480,
  "feature_count": 2464
}

$ jq . content_hash.json
{
  "metadata_xxh128": "a7b3c1d2e4f56789abcdef0123456789",
  "feature_tool_xxh128": "c220a90eb516a6654748c328f3403054",
  "content_xxh128": "5a90164dc1d970770e2a881114ad040a"
}

$ od -Ax --endian=little -tf4 -w8 features/positions_xy.*.2.float32 | head -5
000000       154.56075       277.25787
000008       87.453865       193.33382
000010       182.62576       200.48529
000018        90.68476        96.82127
000020       111.88794       32.918064
```

The file above is **[v1]**. A **[v2]** file of the same image, detected with 2464 keypoints
but only the first 1000 described so far, differs as follows — note `version`, the descriptor
*chunk* entries, the `features/descriptors_metadata.json` coverage file, and the extra
`content_hash.json` hash fields. The `metadata.json` is identical to the version 1 file
except for `version`:

```bash
$ unzip -l myimage.jpg.sift
 ...
 feature_tool_metadata.json.zst
 metadata.json.zst
 features/positions_xy.2464.2.float32.zst
 features/affine_shapes.2464.2.2.float32.zst
 features/descriptors.0-499.128.uint8.zst       # first describe-batch
 features/descriptors.500-999.128.uint8.zst     # appended later
 features/descriptors_metadata.json.zst
 thumbnail_y_x_rgb.128.128.3.uint8.zst
 content_hash.json.zst

$ jq . metadata.json
{
  "version": 2,
  "image_name": "myimage.jpg",
  "image_file_xxh128": "3748f9341bfdcc712beb2c5392664982",
  "image_file_size": 21584,
  "image_width": 270,
  "image_height": 480,
  "feature_count": 2464
}

$ jq . features/descriptors_metadata.json
{
  "described_count": 1000
}

$ jq . content_hash.json
{
  "metadata_xxh128": "a7b3c1d2e4f56789abcdef0123456789",
  "feature_tool_xxh128": "c220a90eb516a6654748c328f3403054",
  "feature_set_xxh128": "9f1c…",
  "descriptor_prefix_xxh128": "be77…",
  "content_xxh128": "4d0a…",
  "component_xxh128": {
    "feature_tool_metadata.json": "…",
    "metadata.json": "…",
    "features/positions_xy.2464.2.float32": "…",
    "features/affine_shapes.2464.2.2.float32": "…",
    "features/descriptors.0-499.128.uint8": "…",
    "features/descriptors.500-999.128.uint8": "…",
    "thumbnail_y_x_rgb.128.128.3.uint8": "…"
  }
}
```

## Version History

- **Version 1**: Current format (written as integer `1` in `metadata.version`).
- **Version 2 (draft)**: Incremental chunked descriptors. The single `descriptors` array is
  replaced by append-only `features/descriptors.{start}-{end}.128.uint8` range chunks
  covering the prefix `[0, described_count)` (relies on the existing descending-size feature
  ordering). `metadata.json` stays immutable; the mutable `described_count` lives in a new
  `features/descriptors_metadata.json` entry. `content_hash.json` (hashes only) gains a stable
  `feature_set_xxh128` (invariant across descriptor appends — the recommended value for other
  files to reference; `metadata.json` is part of it), `descriptor_prefix_xxh128`, and
  `component_xxh128` (cached per-entry digests) so the hashes recompute cheaply on append;
  `content_xxh128` is redefined as
  `xxh128(feature_set_xxh128 ‖ descriptor_prefix_xxh128 ‖ described_count)`. Also requires
  `feature_options.image_to_gray`, a formula pinning the source-image→gray conversion. Version 1 files
  remain valid, and a fully-described version 2 file may be written in the version 1 layout
  for consumers that want the plain single-array form. Status: draft.
