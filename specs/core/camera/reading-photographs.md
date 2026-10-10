# Reading Photographs

sfmtool decodes a photograph into pixels with one decoder, in Rust and in
Python alike, so a value computed from the pixels is the same whichever side
computed it. The decoder is the `image` crate's, behind `ImageU8::read_rgb`
and `ImageU8::read_rgba`; Python reaches the same two functions through the
bindings `sfmtool.fileio.read_image_rgb` and `read_image_rgba`. Every reader
ignores the EXIF orientation tag, so a pixel is addressed by the row and column
stored in the file. This page states the layout the readers return, the rules
for orientation, bit depth, alpha and file format, and which code reads through
them.

## Interface

```rust
impl ImageU8 {
    pub fn read_rgb(path: &Path) -> Result<ImageU8, image::ImageError>;  // 3 channels
    pub fn read_rgba(path: &Path) -> Result<ImageU8, image::ImageError>; // 4 channels
}
pub fn image_has_alpha(path: &Path) -> Result<bool, image::ImageError>; // header only
```

```python
from sfmtool.fileio import image_has_alpha, read_image_rgb, read_image_rgba

rgb = read_image_rgb("images/frame_0001.jpg")    # (H, W, 3) uint8, y_x_rgb
rgba = read_image_rgba("masks/frame_0001.png")   # (H, W, 4) uint8, y_x_rgba
keeps_alpha = image_has_alpha("masks/frame_0001.png")
```

The Rust functions are in
[camera/image.rs](../../../crates/sfmtool-core/src/camera/image.rs) and the
bindings in
[fileio/image.rs](../../../crates/sfmtool-py/src/fileio/image.rs). A binding
releases the GIL while it decodes, so a thread pool decodes several
photographs at once. A missing file raises `FileNotFoundError` and a file that
cannot be decoded raises `OSError`; both messages name the path.
`image_has_alpha` reads the file's header alone, beside `image_dimensions`,
which reads its width and height the same way. It is how a caller that writes
an image back out picks the reader: `read_image_rgba` when the file has alpha,
so the alpha survives, and `read_image_rgb` otherwise.

## Layout

The readers return the layout a `.sfmr` file stores its pixels in
([sfmr-file-format.md](../../formats/sfmr-file-format.md)): the thumbnails
`images/thumbnails_y_x_rgb`, `(N, 128, 128, 3)`, and the patch bitmaps
`points3d/patch_bitmaps_y_x_rgba`, `(N, R, R, 4)`. An array is indexed row,
column, channel (`y_x_rgb` or `y_x_rgba`), C-contiguous `uint8`, with the
channels in RGB or RGBA order. No reader returns BGR. Code that hands the
pixels to OpenCV for drawing or `imwrite` converts to BGR itself, at that call.

## Rules

- **Orientation.** The EXIF orientation is ignored. The array has the width
  and height stored in the file, the ones the camera intrinsics, the SIFT
  extractors' keypoints and the `.sift` thumbnails are in. A tagged photograph
  is not rotated anywhere in sfmtool.
- **Bit depth.** The readers decode to 8 bits per channel. A deeper source,
  such as a 16-bit PNG or TIFF or a floating-point image, is scaled down to 8
  bits, and an image `undistort` or `to-nerfstudio` writes from it has 8 bits
  per channel.
- **Colour.** A grey image has its value repeated in the three colour
  channels.
- **Alpha.** `read_rgb` drops an alpha channel. `read_rgba` keeps the file's
  alpha where it has one and writes 255 where it has none, so `alpha > 0`
  marks pixels with data, as in the patch bitmaps. Its colour channels equal
  `read_rgb`'s. Callers read RGB unless they need alpha.
- **Formats.** The file's contents, not its extension, choose the decoder, so
  a PNG saved under a `.jpg` name reads as a PNG. The workspace `image`
  dependency is built with the crate's default formats, among them JPEG, PNG,
  TIFF, BMP and WebP.

## Code that reads photographs

Every Python image read in `src/` and `scripts/` goes through the bindings,
except the two below:

- `read_workspace_image` in
  [_workspace_image.py](../../../src/sfmtool/_workspace_image.py), the
  loader behind `embed-patches`, the strip montages and the `xform` steps that
  read photographs.
- `load_gray` in [_image_load.py](../../../src/sfmtool/_image_load.py), the
  grey decode for optical flow, which converts the RGB with OpenCV's
  `COLOR_RGB2GRAY`.
- `cluster-patches`, whose refinement kernel takes BGR, as the viewer's
  cluster run hands it too; it reverses the channels after the read.
- The `sfmtool`, OpenCV and COLMAP SIFT backends' decodes and thumbnails, and
  `xform --add-thumbnails`.
- `render_equirect_panorama`, and the drawing in `sift --draw`, the epipolar,
  flow, heatmap and patch overlays.
- `undistort` and `to-nerfstudio`'s downscaled pyramid, which read RGBA when
  `image_has_alpha` says the file has alpha, so the images they write keep it.
- `insv2rig` and `pano2rig`, for the frames and panoramas they resample into
  rig images.
- The scripts, among them `bench_bars`, `add_image_to_tracks`,
  `track_at_pixel`, `patch_crossval` and the benchmarks. Those that draw or
  render patches in BGR reverse the channels after the read.

Two readers decode inside other libraries, and both ignore the orientation
tag:

- COLMAP's extractor decodes the photograph inside COLMAP for its features;
  only the thumbnail sfmtool writes beside them is decoded by
  `read_image_rgb`.
- `epipolar --undistort` and `--rectify` read the two images with
  `pycolmap.Bitmap.read`, because the pycolmap undistorter they call takes a
  `Bitmap`.

Images are written with OpenCV's `cv2.imwrite`, which takes BGR or BGRA, so
each writer converts just before the write.
