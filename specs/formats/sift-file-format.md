# The SIFT file format

A `.sift` file holds the local image features extracted from one image — the
keypoints, their descriptors, and enough metadata to say which image they came
from and exactly which tool and settings produced them — plus a small thumbnail
of the source image. One file per image, written once and never modified, so a
pipeline can extract features once and every later stage can find them, check
that the image behind them has not changed, and read only the columns it needs.

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

`metadata.version` records the format version. There is one version, `1`. A
conforming writer writes `1`. A reader rejects a file whose version is `0`, which
no version of the format has used, or is newer than the newest version it reads,
because it cannot know that layout. Everything below describes version 1.

A version 2 layout is proposed but not implemented: it would store descriptors as
append-only range chunks so a keypoint pool can be detected once and described
incrementally across several commands, add a `described_count` coverage entry, and
split the whole-file hash into a stable `feature_set_xxh128` plus a
`descriptor_prefix_xxh128` that survives an append. See
[`../drafts/sift-incremental-extraction-amendment.md`](../drafts/sift-incremental-extraction-amendment.md).

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

  The image-to-gray conversion belongs in here, because SIFT operates on a single-channel
  float image and several of its parameters — notably the contrast threshold — are defined
  in that value domain, so the mapping changes both the features and the meaning of the
  thresholds. The format does not prescribe how it is spelled; files whose `feature_tool` is
  `"sfmtool"` record it as a `gray_formula` string (`"0.2126*R + 0.7152*G + 0.0722*B"`, BT.709 luma, matching
  COLMAP's `Bitmap::CloneAsGrey`). A structured, reader-evaluable `image_to_gray` object with
  its own formula grammar is part of the proposed version 2; see
  [`../drafts/sift-incremental-extraction-amendment.md`](../drafts/sift-incremental-extraction-amendment.md).

### `metadata.json.zst`

JSON. It should contain the following fields (ignore additional fields for future backwards-compatible extension):

* `version`: (integer) The format version number — `1` (see
  [Format versions](#format-versions)).
* `image_name`: (string) The image filename without the directory.
* `image_file_xxh128`: (string) The XXH128 sum of the bytes of the image file.
* `image_file_size`: (integer) The number of bytes in the image file.
* `image_width`: (integer) The width of the image, in pixels.
* `image_height`: (integer) The height of the image, in pixels.
* `feature_count`: (integer) The number of features (keypoints).

`metadata.json`, like every other entry, is written once and never changes: a `.sift` file
is immutable once written.

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
`feature_options` minus options that do not affect the output. A workspace computes it once at
initialization and stores it in `feature_prefix_dir`; for an image outside a workspace it is
computed from the extraction options at extraction time. It is never computed from a `.sift`
file; see
[Feature Prefix Directory](../workspace/workspace.md#feature-prefix-directory). The two hashes
do not in general agree, and a reader must not compare one with the other.

## Implementations

The code that reads, writes and verifies `.sift` files is:

- Rust: `read_sift`, `read_sift_metadata`, `write_sift` and `verify_sift` in
  [`sfmtool-sift-format`](../../crates/sfmtool-sift-format/src/lib.rs), with
  readers for part of a file: `read_sift_partial` (the first `k` features),
  `read_sift_positions`, `read_sift_keypoints`, `read_sift_features` (the three
  feature columns only) and `read_sift_thumbnail`. Every reader rejects an
  unsupported `metadata.version`; `write_sift` refuses to write one; and
  `verify_sift`, which returns `(is_valid, error_messages)`, reports it as an
  error. None of them checks the feature order.
- Python: `read_sift`, `read_sift_metadata`, `read_sift_partial`, `write_sift`
  and `verify_sift` in `sfmtool.io`, which take and return a dict of
  NumPy arrays and metadata ([bindings](../../crates/sfmtool-py/src/io/sift.rs)),
  and `SiftReader` and `write_sift` in
  [`sfmtool.sift.file`](../../src/sfmtool/sift/file.py), which most of the
  Python package uses.

`SIFT_FORMAT_VERSION` in
[`types.rs`](../../crates/sfmtool-sift-format/src/types.rs) is the newest
version the crate reads and writes. The descriptor width and the thumbnail edge
each have a single declaration there, `DESCRIPTOR_DIM` and `THUMBNAIL_SIZE`,
from which the entry names, the read path, the write path and the shape checks
are derived. A compile-time assertion in `sfmtool-core` keeps `THUMBNAIL_SIZE`
equal to `sfmtool_sfmr_format::THUMBNAIL_SIZE`, and the value is exported to
Python as `sfmtool.THUMBNAIL_SIZE` for the extractors that produce the pixels.

The three extraction backends of [`sfm sift`](../cli/image-feature/sift-command.md)
(`colmap`, `opencv` and `sfmtool`) sort their features by descending size and
resize the thumbnail with OpenCV's `INTER_AREA`.

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

- **Version 1**: the current and only format version, written as the integer `1` in
  `metadata.version`. A proposed version 2 — incremental chunked descriptors and the
  identity hashes that go with them — is drafted in
  [`../drafts/sift-incremental-extraction-amendment.md`](../drafts/sift-incremental-extraction-amendment.md).
