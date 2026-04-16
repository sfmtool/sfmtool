# Image Warping for Distortion and Undistortion

## Motivation

The Rust codebase has complete implementations of `distort()` and `undistort()` for all 11
COLMAP camera models, operating on individual pixel coordinates. However, there is no
facility to apply these transforms to entire images — to produce an undistorted image from
a distorted one, or vice versa. The existing Python `sfm undistort` command delegates to
`pycolmap.undistort_image()`, which is a black box that always outputs PINHOLE cameras and
offers no control over interpolation quality or output camera parameters.

This spec proposes two building blocks in the Rust `sfmtool-core` crate:

1. **Warp map generation** — Given source and destination camera intrinsics, produce a
   dense pixel coordinate map describing where each output pixel samples from the input.
2. **Image resampling** — Apply a warp map to an image, with pluggable interpolation:
   bilinear for general use, and area-weighted sampling for downsampling scenarios where
   the distortion field compresses regions of the image.

Together these enable undistortion, re-distortion, and camera-model conversion as
composable, testable operations entirely within Rust. Because the warp map depends
only on camera intrinsics — not on image content — it can be generated once and
reused for all images sharing the same camera. In a typical SfM reconstruction where
dozens or hundreds of images share one camera model, this amortizes the map
generation cost to near zero.

## Coordinate Convention

All coordinates use the sfmtool convention: pixel centers at `(col + 0.5, row + 0.5)`.
This matches the optical flow module, the `.sfmr`/`.sift` formats, and COLMAP.

## Warp Map

### Data Structure

```rust
/// A dense 2D map of source coordinates for each destination pixel.
///
/// For each pixel (col, row) in the destination image, stores the (x, y)
/// coordinates in the source image to sample from. Coordinates use the
/// pixel-center-at-0.5 convention.
pub struct WarpMap {
    width: u32,
    height: u32,
    /// Interleaved (x, y) pairs, row-major. Length = 2 * width * height.
    data: Vec<f32>,
    /// Optional precomputed SVD of the Jacobian at each pixel, for anisotropic
    /// resampling. Computed lazily via `compute_svd()`. See [`WarpMapSvd`].
    svd: Option<WarpMapSvd>,
}
```

Using `f32` is sufficient — sub-pixel precision of ~1/16384 pixel at 4K resolution
is well beyond what interpolation can resolve. This matches the optical flow module's
use of `f32` for coordinates and keeps memory usage at 8 bytes/pixel (vs 16 for `f64`).

**Out-of-bounds pixels** are stored as `(NaN, NaN)`. A destination pixel gets NaN
coordinates when:
- `ray_to_pixel` returns `None` (ray behind camera or outside model domain)
- The computed source coordinates fall outside the source image bounds

NaN propagates naturally through interpolation arithmetic, so the resampler only
needs a single `is_nan()` check per pixel to detect invalid entries — no separate
validity mask needed. Invalid pixels in the output are written as zero (black).

The `WarpMap` exposes a method to query validity:

```rust
impl WarpMap {
    /// Returns true if the source coordinates at (col, row) are valid (not NaN).
    pub fn is_valid(&self, col: u32, row: u32) -> bool;
}
```

### Jacobian / SVD Data

For anisotropic resampling, each pixel needs the SVD of the local 2x2 Jacobian.
This is precomputed and stored on the `WarpMap` rather than recomputed during
resampling, so it can be generated once and reused across multiple remap calls
(e.g. remapping many images with the same camera).

```rust
/// Precomputed SVD of the warp map Jacobian at each pixel.
///
/// For each pixel, stores the two singular values and the major axis
/// direction — the minimum information needed by the anisotropic resampler.
pub struct WarpMapSvd {
    /// Major singular value per pixel. Length = width * height.
    sigma_major: Vec<f32>,
    /// Minor singular value per pixel. Length = width * height.
    sigma_minor: Vec<f32>,
    /// Major axis direction as (dx, dy) unit vectors, interleaved.
    /// Length = 2 * width * height.
    major_dir: Vec<f32>,
}
```

The Jacobian at each pixel is estimated from the warp map using central
differences, then decomposed via 2x2 SVD (closed-form, no iteration needed).
At boundary pixels and NaN pixels, values are set to `(1, 1, (1, 0))` —
the identity, causing the resampler to fall back to a single bilinear sample.

```rust
impl WarpMap {
    /// Compute the SVD of the Jacobian at each pixel and store it.
    /// Subsequent calls to `remap_aniso` will use the precomputed data.
    pub fn compute_svd(&mut self);

    /// Returns true if SVD data has been computed.
    pub fn has_svd(&self) -> bool;
}
```

`remap_aniso` requires the SVD to be precomputed and returns an error (or
panics) if called without it. This keeps the responsibility clear: the caller
decides when to pay the SVD computation cost.

### Construction API

```rust
impl WarpMap {
    /// Create a warp map that undistorts: maps each pixel in the undistorted
    /// (output) image to the corresponding location in the distorted (input) image.
    ///
    /// For each output pixel center (u_out, v_out):
    ///   1. Unproject through dst_camera to get image-plane coords (x, y)
    ///   2. Project through src_camera to get source pixel coords (u_src, v_src)
    ///
    /// This is the "inverse map" convention: for each destination pixel, we
    /// compute where to read from in the source. This is what resampling needs.
    pub fn from_cameras(
        src_camera: &CameraIntrinsics,
        dst_camera: &CameraIntrinsics,
    ) -> Self;
}
```

The key insight is that to **undistort** an image, we need to know where each output
(undistorted) pixel came from in the distorted input. So we unproject through the
output camera model and project through the input camera model:

| Goal | src_camera | dst_camera |
|------|-----------|-----------|
| Undistort | Distorted camera (e.g. OPENCV) | Pinhole camera (no distortion) |
| Re-distort | Pinhole camera | Distorted camera |
| Convert model | Camera model A | Camera model B |

#### Output Camera Construction

For undistortion, the caller constructs a PINHOLE `CameraIntrinsics` as the
destination camera. Typically this preserves the original focal lengths, principal
point, and image dimensions — just with all distortion coefficients removed.

### Parallelization

`from_cameras` is embarrassingly parallel — each output pixel is independent. The
implementation uses `rayon` to parallelize over rows, consistent with the existing
batch operations in `distortion.rs`.

### Fisheye and the `ray_to_pixel` Gap

For fisheye undistortion, pixels near the image boundary may map to very large
image-plane coordinates (approaching infinity at 90° incidence). The existing
`project(x, y)` takes image-plane coordinates where `x = X/Z`, which is `tan(theta)`
— infinite at 90° and undefined beyond. Similarly, `unproject` returns image-plane
coords that blow up for wide-angle fisheye.

The codebase already solved the *inverse* direction with `pixel_to_ray`, which
recovers the incidence angle `theta` directly and builds a unit ray as
`[sin(theta) * x/r, sin(theta) * y/r, cos(theta)]`, sidestepping `tan(theta)`.

The forward direction — ray to pixel — has no equivalent today. The warp map needs
it: for each destination pixel, we call `pixel_to_ray` on the destination camera to
get a ray, then need to project that ray into the source camera to get the source
pixel coordinates.

**New method needed: `ray_to_pixel`**

```rust
impl CameraIntrinsics {
    /// Project a unit ray direction in camera space to pixel coordinates.
    ///
    /// For perspective models, equivalent to `project(rx/rz, ry/rz)`, but
    /// for fisheye models computes the distorted coordinates directly from
    /// the incidence angle `theta = atan2(sqrt(rx² + ry²), rz)`, avoiding
    /// the `tan(theta)` singularity. This is the true inverse of `pixel_to_ray`.
    ///
    /// Returns `None` if the ray falls outside the model's valid domain:
    /// for perspective models, `theta >= pi/2` (ray at or behind the camera
    /// plane); for fisheye models, only when the incidence angle exceeds the
    /// distortion polynomial's representable range (which may be well beyond
    /// 90°, up to ~180° or more for wide-angle fisheye).
    pub fn ray_to_pixel(&self, ray: [f64; 3]) -> Option<(f64, f64)>;

    /// Batch version.
    pub fn ray_to_pixel_batch(&self, rays: &[[f64; 3]]) -> Vec<Option<[f64; 2]>>;
}
```

For perspective models, `ray_to_pixel` divides by `rz` and calls the existing
`distort()` + focal/principal point transform. For fisheye models, it computes:

```
theta = atan2(sqrt(rx² + ry²), rz)
r_xy = sqrt(rx² + ry²)
(dx, dy) = (rx / r_xy, ry / r_xy)   // unit direction in image plane
```

Then applies the fisheye distortion polynomial in theta-space to get `theta_d`,
and produces distorted image-plane coords as `(theta_d * dx, theta_d * dy)` before
applying focal length and principal point. This is exactly the forward path that
`distort_*_fisheye()` already implements internally — `ray_to_pixel` just enters it
from `theta` directly instead of from `tan(theta)`.

Returns `None` when the incidence angle exceeds the model's valid domain.
For perspective models this means `theta >= pi/2`. For fisheye models the
limit depends on the distortion polynomial — many fisheye lenses represent
angles well beyond 90°, so `ray_to_pixel` must handle those correctly. It
only returns `None` when the polynomial becomes non-monotonic or the angle
exceeds the range where `recover_theta_equidistant` converges (the same
limits already computed for `pixel_to_ray`'s fallback logic).

Internally, `ray_to_pixel` computes `theta = atan2(sqrt(rx² + ry²), rz)` and
works in theta-space for fisheye models (applying the distortion polynomial to
theta directly, the same math as `distort_*_fisheye()` but without the `atan(r)`
preamble). This naturally supports angles beyond 90° since the fisheye distortion
polynomials operate in theta-space, not tangent-space. For perspective models it
delegates to the existing `distort()` via `tan(theta)`. Returns `None` only when
`theta >= pi/2` for perspective models, or when theta exceeds the fisheye
polynomial's monotonic range.

### Equirectangular Camera Model

A new `Equirectangular` variant in the `CameraModel` enum provides a natural
target for panoramic output and a lossless representation of full-sphere imagery.
Unlike pinhole projection, equirectangular can represent the full 360° x 180°
field of view without singularities — making it the right output format when
undistorting wide-angle fisheye cameras.

Equirectangular projection maps longitude and latitude linearly to pixel
coordinates. It fits the same `pixel_to_ray` / `ray_to_pixel` framework as
fisheye models, with no distortion parameters — just focal lengths and principal
point controlling the angular-to-pixel scaling.

```rust
CameraModel::Equirectangular {
    focal_length_x: f64,   // pixels per radian (horizontal / longitude)
    focal_length_y: f64,   // pixels per radian (vertical / latitude)
    principal_point_x: f64,
    principal_point_y: f64,
}
```

**`ray_to_pixel`:**
```
longitude = atan2(rx, rz)
latitude  = asin(clamp(ry / |ray|, -1, 1))
u = focal_length_x * longitude + principal_point_x
v = focal_length_y * (-latitude) + principal_point_y
```

Note the negated latitude: `v` increases downward while latitude increases upward.
The forward direction (`ray_to_pixel`) is always valid — every ray maps to a pixel
(returns `Some` for all non-zero rays). This makes equirectangular ideal as an
output format: no pixels are wasted on out-of-bounds regions.

**`pixel_to_ray`:**
```
longitude = (u - principal_point_x) / focal_length_x
latitude  = -(v - principal_point_y) / focal_length_y
ray = [sin(longitude) * cos(latitude), sin(latitude), cos(longitude) * cos(latitude)]
```

**Standard full-sphere panorama** (360° x 180°) at a given resolution:
```
focal_length_x = width / (2 * pi)
focal_length_y = height / pi
principal_point_x = width / 2
principal_point_y = height / 2
```

The focal length and principal point parameterization allows sub-regions of a
panorama to be represented (e.g. a 120° horizontal strip), or non-square pixel
aspect ratios, using the same model.

**Distortion:** `distort()` and `undistort()` are identity operations (no
distortion coefficients). `has_distortion()` returns false.

### Warp Map Pipeline

The warp map construction chooses between two code paths based on whether either
camera is a fisheye model:

**Perspective-to-perspective (both cameras are non-fisheye):**

```
For each destination pixel (u_dst, v_dst):
  (x, y) = dst_camera.unproject(u_dst, v_dst)
  (u_src, v_src) = src_camera.project(x, y)
```

This is the fast path. `unproject` removes the destination distortion to get
image-plane coordinates, and `project` applies the source distortion directly.
No trigonometry beyond what the distortion models themselves need. This covers
the common case of converting between perspective camera models (PINHOLE, OPENCV,
RADIAL, etc.).

**Any fisheye or equirectangular camera involved:**

```
For each destination pixel (u_dst, v_dst):
  ray = dst_camera.pixel_to_ray(u_dst, v_dst)
  (u_src, v_src) = src_camera.ray_to_pixel(ray)  // None → NaN in map
```

The ray path avoids the `tan(theta)` singularity that makes image-plane coordinates
unusable at wide angles. It adds an `atan2` → `tan` round-trip compared to the
direct path, but this is necessary for correctness when either camera has FOV
approaching or exceeding 180°.

The path is selected by checking whether either camera uses a non-perspective
projection (fisheye or equirectangular). Source coordinates that fall outside the source image bounds or where
`ray_to_pixel` returns `None` are stored as NaN so the resampler can skip them.

## Image Resampling

### Bilinear Interpolation

The optical flow module already has `sample_bilinear` for `GrayImage` (f32, single
channel). For image warping we need to support multi-channel `u8` images as used by
the rest of the pipeline.

```rust
/// A multi-channel image stored as packed u8 values.
///
/// Supports 1 (gray), 3 (RGB), or 4 (RGBA) channels.
pub struct ImageU8 {
    width: u32,
    height: u32,
    channels: u32,
    /// Row-major, channels interleaved. Length = width * height * channels.
    data: Vec<u8>,
}
```

The resampling function:

```rust
/// Apply a warp map to an image using bilinear interpolation.
///
/// For each pixel (col, row) in the output:
///   1. Look up source coordinates (sx, sy) from the warp map
///   2. If (sx, sy) is valid (not NaN), bilinearly interpolate from the source image
///   3. If invalid, write zero (black)
///
/// The output image has the same dimensions as the warp map and the same
/// number of channels as the input image.
pub fn remap_bilinear(src: &ImageU8, map: &WarpMap) -> ImageU8;
```

For performance, bilinear interpolation on `u8` data should:
- Compute in integer arithmetic (fixed-point) where possible, or use `f32`
  intermediates and round at the end
- Parallelize over rows with `rayon`
- Clamp source coordinates to image bounds (same as `sample_bilinear` in the
  optical flow module)

### Anisotropic Filtering

When the distortion field compresses regions of the source image — common at the
periphery of barrel-distorted images being undistorted — bilinear interpolation
undersamples and produces aliasing. The compression is often anisotropic: fisheye
undistortion compresses heavily in the radial direction while the tangential direction
stays close to 1:1. An isotropic approach (e.g. selecting a Gaussian pyramid level
based on `sqrt(|det(J)|)`) would over-blur the tangential direction to adequately
filter the radial direction.

#### GPU-Style Anisotropic Filtering

The sampling strategy follows the same principle as GPU hardware anisotropic texture
filtering, using the precomputed SVD data from `WarpMapSvd`:

1. **Look up the precomputed SVD** — `sigma_major`, `sigma_minor`, and `major_dir`
   for this pixel.
2. **Select the pyramid level** based on `sigma_minor` (the minor singular value).
   This is the pre-filtering level that prevents aliasing along the *narrow* axis
   of the elliptical footprint: `level = log2(sigma_minor)`, clamped to `[0, max_level]`.
3. **Sample multiple points along the major axis.** The number of samples is the
   anisotropy ratio `N = ceil(sigma_major / sigma_minor)`, capped at a maximum
   (e.g. 16). The samples are evenly spaced along the major axis direction in
   destination space, mapped to source coordinates, and bilinearly sampled from
   the selected pyramid level.
4. **Average the samples.** The output pixel value is the mean of the N samples.

When `sigma_major <= 1` (no compression in any direction), this reduces to a single
bilinear sample from the base level — the same as `remap_bilinear`.

```
For each destination pixel (col, row):
  (sigma_major, sigma_minor, major_dir) = svd.get(col, row)

  level_f = log2(max(1, sigma_minor))
  level_lo = floor(level_f)
  level_hi = level_lo + 1
  frac = level_f - level_lo                // fractional part for trilinear blend

  N = min(max_aniso, ceil(sigma_major / max(1, sigma_minor)))

  sum_lo = 0, sum_hi = 0
  for i in 0..N:
    t = (i + 0.5) / N - 0.5               // offset along major axis [-0.5, 0.5)
    (sx, sy) = map.get(col, row) + t * sigma_major * major_dir
    sum_lo += sample_bilinear(pyramid[level_lo], sx / 2^level_lo, sy / 2^level_lo)
    sum_hi += sample_bilinear(pyramid[level_hi], sx / 2^level_hi, sy / 2^level_hi)

  output[col, row] = lerp(sum_lo / N, sum_hi / N, frac)
```

This correctly handles the common distortion pattern: at the periphery of a fisheye
undistortion, the radial direction (major axis) may compress 4-8x while the tangential
direction (minor axis) stays near 1:1. The pyramid pre-filters at roughly 1:1 scale
(level 0), and 4-8 samples along the radial direction integrate over the compressed
region — no tangential over-blur.

#### API

```rust
/// Apply a warp map with anisotropic filtering.
///
/// Requires `map.compute_svd()` to have been called first.
///
/// Builds a Gaussian pyramid of the source image. For each output pixel,
/// reads the precomputed SVD, selects the pyramid level from the minor
/// singular value, and takes multiple trilinearly-blended samples along
/// the major axis direction. Falls back to a single bilinear sample where
/// the mapping is non-compressive.
///
/// `max_anisotropy` caps the number of samples along the major axis (default 16).
pub fn remap_aniso(src: &ImageU8, map: &WarpMap, max_anisotropy: u32) -> ImageU8;
```

#### Pyramid Construction

The Gaussian pyramid for `ImageU8` is built by repeated box-filter or Gaussian
downsample-by-2, operating independently per channel. This is a separate lightweight
implementation from the optical flow `ImagePyramid` (which operates on `GrayImage`
f32 single-channel). The number of levels is `floor(log2(min(width, height)))`.

This filtering is primarily useful for undistorting fisheye images where the
periphery represents a much larger field of view per pixel than the center. For
standard perspective cameras with mild radial distortion, `remap_bilinear` alone
is usually sufficient.

## Python Bindings

Expose the warp map and resampling through `sfmtool-py`:

```python
from sfmtool._sfmtool import CameraIntrinsics, WarpMap

# Build a warp map from camera intrinsics
camera = reconstruction.cameras[0]
pinhole = camera.to_pinhole()
warp = WarpMap.from_cameras(src=camera, dst=pinhole)

# Access the raw map data as numpy arrays
map_x, map_y = warp.to_numpy()  # Each is (height, width) float32

# Apply to an image (numpy u8 array, HxWxC)
undistorted = warp.remap_bilinear(image)
undistorted = warp.remap_aniso(image, max_anisotropy=16)

# Properties
warp.width   # int
warp.height  # int
```

The `to_numpy()` method returns two separate arrays matching OpenCV's `cv2.remap`
convention, enabling interop: callers can use the Rust-generated maps with
`cv2.remap()` if desired, or use the built-in Rust resampler.

The `remap_bilinear` and `remap_aniso` methods accept a numpy `ndarray` (HxWx1,
HxWx3, or HxWx4, dtype `uint8`) and return a numpy array of the same shape.

## Module Organization

New files in `sfmtool-core`:

```
crates/sfmtool-core/src/
├── warp_map.rs          # WarpMap struct, from_cameras(), Jacobian estimation
├── remap.rs             # remap_bilinear(), remap_aniso(), ImageU8, ImageU8Pyramid
```

New file in `sfmtool-py`:

```
crates/sfmtool-py/src/
├── py_warp_map.rs       # PyWarpMap Python wrapper
```

Add `ray_to_pixel` / `distort_ray` to the existing `CameraIntrinsics` / `CameraModel`
in `crates/sfmtool-core/src/distortion.rs`.

## Implementation Order

1. **`Equirectangular` camera model** — New `CameraModel` variant with `pixel_to_ray`,
   `ray_to_pixel`, identity `distort`/`undistort`, serialization. Test round-trip at
   center, edges, and poles. Test full-sphere construction.

2. **`ray_to_pixel`** — New method on `CameraIntrinsics`, the inverse of
   `pixel_to_ray`. Test round-trip:
   `ray_to_pixel(pixel_to_ray(u, v))` recovers `(u, v)` for all 12 camera models
   (11 existing + equirectangular).
   Test fisheye at wide angles (80°, 89°, 91°) where `project(unproject(...))` would
   fail. Expose through Python bindings.

3. **`WarpMap` and `from_cameras`** — The core data structure and map generation.
   Test by verifying that identity cameras produce identity maps, and that
   round-tripping (undistort map composed with re-distort map) recovers pixel
   coordinates.

4. **`ImageU8` and `remap_bilinear`** — Multi-channel bilinear resampling. Test
   against known transforms (identity, pure translation, known distortion models
   with pycolmap as reference).

5. **`remap_aniso`** — Anisotropic filtering: Jacobian SVD, `ImageU8` pyramid, and
   multi-sample along the major axis. Test on fisheye cameras where peripheral
   aliasing is visible with bilinear alone.

6. **Python bindings** — `PyWarpMap` wrapper. Test `to_numpy()` interop with
   `cv2.remap()`.

## Testing Strategy

### Unit Tests (Rust)

- **Equirectangular round-trip**: `pixel_to_ray` → `ray_to_pixel` recovers pixel
  coordinates at center, edges, corners, and near the poles. Verify that the
  standard full-sphere construction covers exactly 360° x 180°.
- **Equirectangular as warp target**: `from_cameras(fisheye, equirectangular)`
  produces a valid map with no NaN pixels (every output pixel maps to somewhere
  in the source, since equirectangular has no out-of-domain directions).
- **`ray_to_pixel` round-trip**: `ray_to_pixel(pixel_to_ray(u, v))` recovers `(u, v)`
  for all 12 camera models, at center, corners, and edges of the image.
- **`ray_to_pixel` wide-angle fisheye**: Verify correct results at 80°, 89°, and
  (for fisheye) 91° incidence angles where `project(unproject(...))` would fail.
- **`ray_to_pixel` domain limits**: Returns `None` for `theta >= pi/2` on
  perspective models. For fisheye models, returns valid results for rays
  beyond 90° (tested at 95°, 100°), and `None` only when the distortion
  polynomial's representable range is exceeded.
- **Identity map**: `from_cameras(pinhole, pinhole)` produces coordinates equal to
  pixel centers.
- **Round-trip**: `from_cameras(distorted, pinhole)` composed with
  `from_cameras(pinhole, distorted)` recovers original coordinates (within
  interpolation tolerance).
- **Known distortion**: For SimpleRadial with known k1, verify specific pixel
  mappings analytically.
- **Fisheye boundary**: OpenCVFisheye at 180° FOV — verify that edge pixels produce
  valid (or NaN) source coordinates without panics.
- **Bilinear correctness**: Remap with identity map preserves image exactly.
  Remap with 0.5px shift matches manual bilinear calculation.
- **Anisotropic sampling**: Remap a checkerboard through an anisotropic warp (e.g. 4x
  compression along one axis, 1x along the other). Verify that `remap_aniso` produces
  a smooth result along the compressed axis without over-blurring the other axis,
  while `remap_bilinear` shows aliasing.
- **Anisotropy ratio capping**: Verify that the number of samples along the major
  axis is capped at `max_anisotropy` and that the result degrades gracefully.
- **Pyramid level selection**: For a known 2x isotropic compression, verify that
  `remap_aniso` selects level 1 and takes a single sample (anisotropy ratio ~1).
- **Multi-channel**: Verify RGB and RGBA images remap correctly (each channel
  independent).

### Integration Tests (Python)

- Compare `WarpMap.remap_bilinear()` output against `cv2.remap()` with the same map
  data, verifying pixel values match within ±1 (u8 rounding).
- Undistort a real test image (Seoul Bull dataset) and verify it matches
  `pycolmap.undistort_image()` output.

## Open Questions

1. **GPU acceleration**: The optical flow module has a GPU (wgpu) code path. Should
   warp map resampling also have one? Deferred — CPU with rayon parallelization is
   sufficient for the initial implementation. The warp map + remap pattern is
   naturally GPU-friendly if needed later.
