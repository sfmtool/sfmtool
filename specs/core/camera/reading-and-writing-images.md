# Reading and Writing Images

sfmtool decodes an image file into pixels with one decoder and encodes pixels
into an image file with one encoder, in Rust and in Python alike, so a value
computed from the pixels is the same whichever side computed it, and no image
array is in BGR order on its way to or from a file. Both are the `image` crate's, behind
`ImageU8::read_rgb`, `ImageU8::read_rgba` and `ImageU8::write`; Python reaches
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
which reads its width and height the same way.

The writer errors, each naming the path: `TypeError` for an array that is not
`uint8`; `ValueError` for an array of the wrong shape, a `jpeg_quality`
outside 1 to 100, an extension that names no format the encoder writes, or
`write_image_rgba` to a JPEG; `FileNotFoundError` when the parent directory
does not exist; and `OSError` when the file cannot be written otherwise. In
Rust these are `ImageError::Unsupported`, `ImageError::Parameter` and
`ImageError::IoError`.

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
  JPEG, PNG, TIFF, BMP and WebP.
- **Formats, writing.** The output path's extension, in any case, chooses the
  encoder. `.jpg` and `.jpeg` write a baseline JPEG; `.png` writes a lossless
  PNG; any other extension whose format the `image` crate encodes 8-bit pixels
  to (`.tif`, `.bmp`, `.webp`, ...) is written with that crate's defaults,
  which for WebP is lossless. The image is encoded in memory and then written,
  so a failed encode leaves no file behind.
- **JPEG quality.** `jpeg_quality` is 1 to 100 and defaults to
  `DEFAULT_JPEG_QUALITY`, 95, the default of OpenCV's `cv2.imwrite`, so a
  command that names no quality gets the quality OpenCV gave it. Commands
  with a `--jpeg-quality` option (`pano2rig`, `to-nerfstudio`) pass it
  through. The encoder is the `image` crate's own baseline encoder, which
  keeps the chroma at full resolution (no 4:2:0 subsampling). On the 85
  `dino_dog_toy` photographs (2040 x 1536) at quality 95
  its files are 14% larger than OpenCV's (74.8 MB against 65.4 MB), their
  mean absolute error is 0.23 grey levels against OpenCV's 0.20, and one
  thread encodes them in 4.2 s against OpenCV's 1.0 s; with the GIL released,
  eight threads take 0.8 s.
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

Every image file sfmtool writes goes through `write_image_rgb` or
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

Tests that need an image no sfmtool writer produces, a 16-bit PNG or a file
written by an encoder independent of the one under test, write it with OpenCV.
