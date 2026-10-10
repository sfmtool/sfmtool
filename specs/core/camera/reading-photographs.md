# Reading Photographs

sfmtool decodes a photograph into pixels with one decoder, in Rust and in
Python alike, so a value computed from the pixels is the same whichever side
computed it. The decoder is the `image` crate's, behind `ImageU8::read_rgb`
and `ImageU8::read_rgba`; Python reaches the same two functions through the
bindings `sfmtool.fileio.read_image_rgb` and `read_image_rgba`. Every reader
ignores the EXIF orientation tag, so a pixel is addressed by the row and column
stored in the file. This page states the layout the readers return, the rule
for orientation, colour depth and alpha, and which Python code reads through
them.

## Interface

```rust
impl ImageU8 {
    pub fn read_rgb(path: &Path) -> Result<ImageU8, image::ImageError>;  // 3 channels
    pub fn read_rgba(path: &Path) -> Result<ImageU8, image::ImageError>; // 4 channels
}
```

```python
from sfmtool.fileio import read_image_rgb, read_image_rgba

rgb = read_image_rgb("images/frame_0001.jpg")    # (H, W, 3) uint8, y_x_rgb
rgba = read_image_rgba("masks/frame_0001.png")   # (H, W, 4) uint8, y_x_rgba
```

The Rust functions are in
[camera/image.rs](../../../crates/sfmtool-core/src/camera/image.rs) and the
bindings in
[fileio/image.rs](../../../crates/sfmtool-py/src/fileio/image.rs). A binding
releases the GIL while it decodes, so a thread pool decodes several
photographs at once. A missing file raises `FileNotFoundError` and a file that
cannot be decoded raises `OSError`; both messages name the path.

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
- **Colour.** A grey image has its value repeated in the three colour
  channels. A 16-bit image is scaled to 8 bits.
- **Alpha.** `read_rgb` drops an alpha channel. `read_rgba` keeps the file's
  alpha where it has one and writes 255 where it has none, so `alpha > 0`
  marks pixels with data, as in the patch bitmaps. Its colour channels equal
  `read_rgb`'s. Callers read RGB unless they need alpha.
- **Formats.** The workspace `image` dependency is built with the crate's
  default formats, among them JPEG, PNG, TIFF, BMP and WebP.

## Python code that reads photographs

These read through the bindings:

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

These keep OpenCV's decode, because they do something the readers do not:

- `undistort` and `to-nerfstudio` read with `IMREAD_UNCHANGED`, to keep a
  16-bit or alpha image's depth and channels in the image they write. That flag
  also ignores the orientation tag.
- `insv2rig` and `pano2rig` read the frames and panoramas they resample into
  new rig images; both pass `IMREAD_IGNORE_ORIENTATION`.

COLMAP's extractor decodes the photograph inside COLMAP for its features; only
the thumbnail sfmtool writes beside them is decoded by `read_image_rgb`.
