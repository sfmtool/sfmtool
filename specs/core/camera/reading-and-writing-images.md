# Reading and Writing Images

sfmtool decodes an image file into pixels with one reader and encodes pixels
into an image file with one writer, in Rust and in Python alike, so a value
computed from the pixels is the same whichever side computed it, and no image
array is in BGR order on its way to or from a file. They are behind
`ImageU8::read_rgb`, `ImageU8::read_rgba` and `ImageU8::write`: the
`jpeg-decoder` and `jpeg-encoder` crates decode and encode JPEG, and the
`image` crate every other format. Python reaches
the same functions through the bindings `sfmtool.fileio.read_image_rgb`,
`read_image_rgba`, `write_image_rgb` and `write_image_rgba`. Every reader
ignores the EXIF orientation tag, so a pixel is addressed by the row and column
stored in the file. This page states the layout the readers return and the
writers take, the rules for orientation, bit depth, alpha, file format and JPEG
quality, and which code reads and writes through them. OpenCV is used to
process pixels (resampling, colour conversion, drawing), never to read or write
an image file.

## Interface

```rust
pub const DEFAULT_JPEG_QUALITY: u8 = 95;

impl ImageU8 {
    pub fn read_rgb(path: &Path) -> Result<ImageU8, image::ImageError>;  // 3 channels
    pub fn read_rgba(path: &Path) -> Result<ImageU8, image::ImageError>; // 4 channels
    pub fn write(&self, path: &Path, jpeg_quality: u8) -> Result<(), image::ImageError>;
}
pub fn image_has_alpha(path: &Path) -> Result<bool, image::ImageError>; // header only
pub fn image_dimensions(path: &Path) -> Result<(u32, u32), image::ImageError>; // header only
```

```python
from sfmtool.fileio import (
    image_has_alpha,
    read_image_rgb,
    read_image_rgba,
    write_image_rgb,
    write_image_rgba,
)

rgb = read_image_rgb("images/frame_0001.jpg")    # (H, W, 3) uint8, y_x_rgb
rgba = read_image_rgba("masks/frame_0001.png")   # (H, W, 4) uint8, y_x_rgba
keeps_alpha = image_has_alpha("masks/frame_0001.png")

write_image_rgb("out/frame_0001.jpg", rgb)                    # JPEG, quality 95
write_image_rgb("out/frame_0001.jpg", rgb, jpeg_quality=85)
write_image_rgba("out/mask_0001.png", rgba)                   # PNG with alpha
```

The Rust functions are in
[camera/image.rs](../../../crates/sfmtool-core/src/camera/image.rs) and the
bindings in
[fileio/image.rs](../../../crates/sfmtool-py/src/fileio/image.rs). A binding
releases the GIL while it decodes or encodes, so a thread pool reads or writes
several images at once.

The writers are a pair, named like the readers, because the suffix says the
channel order of the array at every call site: an array that reaches
`write_image_rgb` is RGB, and code that converted to BGR for OpenCV would show
it there. `write_image_rgba` has no `jpeg_quality`, because no format it can
write takes one. A caller that writes back what it read picks the pair by
`image_has_alpha`, as it picked the reader.

The reader errors: a missing file raises `FileNotFoundError` and a file that
cannot be decoded raises `OSError`; both messages name the path.
`image_has_alpha` reads the file's header alone, beside `image_dimensions`,
which reads its width and height the same way (core's `image_has_alpha` and
`image_dimensions` in [image.rs](../../../crates/sfmtool-core/src/camera/image.rs)).
Both read a JPEG's markers up to its frame header with the decoder that
reads its pixels, so they refuse the frame headers the reader refuses. A
file whose frame header reads can still be refused by the reader, from a
later segment (a damaged Huffman table, quantization table or scan header)
or from its pixels, such as a 12-bit lossless JPEG: it has dimensions but
cannot be read.

The writer errors, each naming the path: `TypeError` for an array that is not
`uint8`; `ValueError` for an array of the wrong shape or with a side of 0, a
`jpeg_quality` outside 1 to 100 (checked for every format, though only a JPEG
uses it), an extension that names no format the encoder writes 8-bit pixels
to, `write_image_rgba` to a JPEG, or an image the format's encoder refuses,
such as a side over 65535 in a JPEG or over 16384 in a WebP;
`FileNotFoundError` when the parent directory does not exist; and `OSError`
when the file cannot be written otherwise. In Rust these are
`ImageError::Unsupported`, `ImageError::Parameter`, `ImageError::Encoding` (the
encoder's refusal; the image is encoded in memory, so it is never a file
error) and `ImageError::IoError`.

## Layout

The readers return, and the writers take, the layout a `.sfmr` file stores its
pixels in ([sfmr-file-format.md](../../formats/sfmr-file-format.md)): the
thumbnails `images/thumbnails_y_x_rgb`, `(N, 128, 128, 3)`, and the patch
bitmaps `points3d/patch_bitmaps_y_x_rgba`, `(N, R, R, 4)`. An array is indexed
row, column, channel (`y_x_rgb` or `y_x_rgba`), `uint8`, with the channels in
RGB or RGBA order. A reader returns a C-contiguous array; a writer accepts any
strides and copies the array before encoding, so a slice such as the left half
of a frame is written as it is. No reader returns BGR and no writer takes it.
Code that draws on an image with OpenCV gives its colour tuples in RGB order,
since OpenCV draws a tuple in whatever channel order the array has.

## Rules

- **Orientation.** The EXIF orientation is ignored. The array has the width
  and height stored in the file, the ones the camera intrinsics, the SIFT
  extractors' keypoints and the `.sift` thumbnails are in. A tagged photograph
  is not rotated anywhere in sfmtool, and the writers write no orientation tag.
- **Bit depth.** Images are 8 bits per channel. The readers decode a deeper
  source, such as a 16-bit PNG or TIFF or a floating-point image, to 8 bits,
  and the writers write 8 bits per channel in every format, so an image
  `undistort` or `to-nerfstudio` writes from a deeper source has 8 bits per
  channel.
- **Colour.** A grey image has its value repeated in the three colour
  channels.
- **Alpha.** `read_rgb` drops an alpha channel. `read_rgba` keeps the file's
  alpha where it has one and writes 255 where it has none, so `alpha > 0`
  marks pixels with data, as in the patch bitmaps. Its colour channels equal
  `read_rgb`'s. Callers read RGB unless they need alpha. `write_image_rgba`
  writes the alpha, and refuses a format that cannot hold one (JPEG) rather
  than drop it.
- **Formats, reading.** The file's contents, not its extension, choose the
  decoder, so a PNG saved under a `.jpg` name reads as a PNG. The workspace
  `image` dependency is built with the crate's default formats, among them
  JPEG, PNG, TIFF, BMP and WebP, and every format but JPEG is decoded with
  it. A JPEG is decoded with the `jpeg-decoder` crate instead, because the
  `image` crate's JPEG decoder, zune-jpeg 0.5.15, returns wrong pixels
  without an error for a baseline JPEG that stores each component in a scan
  of its own. `jpeg-decoder` reads those as OpenCV does, and on the
  `test-data` photographs its values differ from OpenCV's by 0.04 to 0.08
  grey levels on average. Called from one thread, it decodes the 85
  `dino_dog_toy` photographs in 0.9 s, as zune-jpeg and OpenCV do; it does
  not use rayon, but for an image wider than 128 px it starts one OS thread
  per colour component. A CMYK or YCCK JPEG becomes RGB within one grey level
  of OpenCV's conversion. The reader reads a JPEG into memory once (a file
  over 1 GiB is refused, twice the largest decode the limit below allows)
  and decodes it with `jpeg-decoder`, which reads past the end of the bytes
  only when the file ends before its end-of-image marker: it lacks only the
  marker, was cut part way through, or has segment lengths that run past its
  end. A complete file, including one followed by trailing data or another
  JPEG, never does. A file that does is decoded again, from the same bytes,
  by one of two rules. A sequential (baseline or extended) JPEG with a scan
  that holds fewer components than the frame, such as one with one scan per
  component, is decoded by `jpeg-decoder` with an end-of-image marker
  appended, since zune-jpeg misreads such a file even when it is whole. A
  file of this kind that lacks only its marker reads within a few grey
  levels of the complete file; in one cut part way through a scan, the part
  the scan did not reach decodes from zero bits as a dark textured pattern
  rather than grey. One cut before every component's scan has begun is
  refused, since `jpeg-decoder` requires data for each component (zune-jpeg
  returned wrong pixels for it), as is one with restart markers cut part way
  through a scan. Every other
  such file is read by the `image` crate's decoder, zune-jpeg, as every
  JPEG was before, because `jpeg-decoder` does not fill missing data as
  libjpeg does: it refuses the file at its end, or, given an appended
  marker, fills the missing part with that texture and still refuses a cut
  file with restart markers. zune-jpeg, like libjpeg and so OpenCV, fills
  the missing part with flat grey (128) and reads restart markers, and a
  file that lacks only its marker reads within a few grey levels of the
  complete file. `jpeg-decoder` is in
  maintenance mode (image-rs is
  moving to zune-jpeg; its last release is 0.3.2, 2025-06), so the reader
  goes back to the `image` crate's decoder once a stable zune-jpeg release
  passes `read_decodes_a_jpeg_with_one_scan_per_component`. The decoders have
  limits of their own, each raised as `OSError`: the `image` crate refuses a
  WebP 16384 pixels wide, the largest WebP allows and one OpenCV reads. The
  JPEG reader refuses, from the frame header and before decoding any scan, a
  JPEG that would decode to more than 512 MiB and a lossless JPEG of a
  precision other than 8 bits. Of the JPEG precisions it reads
  only 8 bits: it refuses every DCT frame (baseline, extended or
  progressive) of any other precision, among them 12-bit colour and grey,
  and every lossless frame of a precision other than 8. Only an 8-bit
  lossless JPEG is read.
- **Formats, writing.** The output path's extension, in any case, chooses the
  encoder. `.jpg` and `.jpeg` write a baseline JPEG; `.png` writes a lossless
  PNG; any other extension whose format the `image` crate encodes 8-bit pixels
  to (`.tif`, `.bmp`, `.webp`, ...) is written with that crate's defaults,
  which write WebP lossless and TIFF uncompressed. A format that holds only
  floating-point pixels, OpenEXR (`.exr`) or Radiance HDR (`.hdr`), cannot be
  written at 8 bits: `undistort` and `to-nerfstudio` read such a source, but
  writing its image back out under the same name raises `ValueError`. The
  image is encoded in memory, written to a temporary file beside the path and
  renamed into place, so a failed encode or write leaves neither a partial
  file nor a changed old one. On Windows a file another process holds open,
  as Python's `open` does, cannot be replaced by a rename, though it can be
  written; there the bytes are written to the path directly instead, and a
  failure part way through that write can leave a partial file.
- **JPEG quality.** `jpeg_quality` is 1 to 100 and defaults to
  `DEFAULT_JPEG_QUALITY`, 95, the default of OpenCV's `cv2.imwrite`, so a
  command that names no quality gets the quality OpenCV gave it. Commands
  with a `--jpeg-quality` option (`pano2rig`, `to-nerfstudio`) pass it
  through. The encoder is the `jpeg-encoder` crate's, set as libjpeg's
  defaults are: 4:2:0 chroma subsampling, each chroma sample the average of
  its 2 x 2 block, and the standard Huffman tables. Its optimized Huffman
  tables are off. They make the 85 `dino_dog_toy` photographs 25% smaller
  (49.1 MB against 65.6 MB) and take about 2.5 times as long to write, but
  with them the encoder writes each component in a scan of its own, which
  zune-jpeg 0.5.15, and so any program reading through the `image` crate,
  decodes to wrong pixels without an error.
- **JPEG size, error and time.** Measured on the 85 `dino_dog_toy`
  photographs (2040 x 1536) at quality 95, against `cv2.imwrite` at the same
  quality. The size is the total of the 85 files. The error is the mean of
  `|decoded - source|` over every `uint8` value (H x W x 3) of all 85
  images, where each file is decoded with `read_image_rgb` and the source is
  the photograph as `read_image_rgb` reads it. The time is the best of three
  runs writing all 85 from one thread, the arrays already in memory. The files
  are the size of OpenCV's (65.6 MB against 65.4 MB), with the same error
  (0.210 grey levels against 0.210). One thread takes 2.1 to 2.3 s; OpenCV
  took 1.0 to 2.2 s across runs on a shared machine, so the writer runs at
  about half OpenCV's speed on one thread at worst. With the GIL released,
  eight threads take 0.4 to 0.5 s. `jpeg-encoder` is licensed "(MIT OR Apache-2.0) AND
  IJG", and the credit the IJG licence asks for is in
  [THIRD-PARTY-NOTICES.md](../../../THIRD-PARTY-NOTICES.md).
- **PNG compression.** A PNG is written at the `png` crate's fast compression
  with adaptive row filtering. On the same 85 photographs that is 4% smaller
  than OpenCV's default PNG (381.6 MB against 397.8 MB) and 3.7 times faster
  to encode (1.6 s against 6.0 s).
  The images sfmtool writes as PNG are overlays and montages that are read
  once, so encode time matters more than the last few percent of size.

## Code that reads images

These read through the bindings:

- `read_workspace_image` in
  [_workspace_image.py](../../../src/sfmtool/_workspace_image.py), the
  loader behind `embed-patches`, the strip montages and the `xform` steps that
  read photographs.
- `load_gray` in [_image_load.py](../../../src/sfmtool/_image_load.py), the
  grey decode for optical flow, which converts the RGB with OpenCV's
  `COLOR_RGB2GRAY`.
- `cluster-patches`, whose refinement kernel takes the RGB as read. The
  viewer's cluster run reads with `ImageU8::read_rgb`, so it hands the kernel
  the same pixels.
- The `sfmtool`, OpenCV and COLMAP SIFT backends' decodes and thumbnails, and
  `xform --add-thumbnails`.
- `render_equirect_panorama`, and the drawing in `sift --draw`, the epipolar,
  flow, heatmap and patch overlays.
- `undistort` and `to-nerfstudio`'s downscaled pyramid, which read RGBA when
  `image_has_alpha` says the file has alpha, so the images they write keep it.
- `insv2rig` and `pano2rig`, for the frames and panoramas they resample into
  rig images.
- The scripts, among them `bench_bars`, `add_image_to_tracks`,
  `track_at_pixel`, `patch_crossval` and the benchmarks. They hand the Rust
  kernels the RGB as read.

The viewer reads the background photograph of camera view with
`ImageU8::read_rgba`.

Two readers decode inside other libraries, and both ignore the orientation
tag:

- COLMAP's extractor decodes the photograph inside COLMAP for its features;
  only the thumbnail sfmtool writes beside them is decoded by
  `read_image_rgb`.
- `epipolar --undistort` and `--rectify` read the two images with
  `pycolmap.Bitmap.read`, because the pycolmap undistorter they call takes a
  `Bitmap`. `Bitmap.to_array` returns RGB, so the drawing on it is in RGB as
  everywhere else.

## Code that writes images

Every image file sfmtool's Python code writes goes through `write_image_rgb` or
`write_image_rgba`:

- `undistort`, with alpha when the source has it, and `to-nerfstudio`'s
  downscaled pyramid, likewise, at its `--jpeg-quality`.
- `pano2rig`'s cube faces at its `--jpeg-quality`, and `insv2rig`'s split
  fisheye frames.
- `panorama`'s equirectangular image.
- The overlays of `sift --draw`, `epipolar`, `flow` (and its flow-colour
  image), `heatmap` and `render-patches`, and the flow images `motion` saves.
- The strip montages of `compare --strips` and `inspect --strips`.
- The scripts' montages and crops (`_viz_common`'s at quality 92,
  `patch_crossval`, `kdf_constellation_eval`).

Two Rust writers encode with their own settings rather than through
`ImageU8::write`, because they encode to memory rather than to a file: the
`web-export` atlas pages, JPEGs from the `image` crate's encoder at
`--jpeg-quality` ([web_export/atlas.rs](../../../crates/sfmtool-core/src/web_export/atlas.rs)),
and the viewer's MCP screenshots, PNGs from the `image` crate's PNG encoder.

Tests that need an image no sfmtool writer produces, a 16-bit PNG or a file
written by an encoder independent of the one under test, write it with OpenCV.
