# Image Warping for Distortion and Undistortion

Image warping resamples a whole image from one camera into another through a
precomputed per-pixel map: for each output pixel, the map holds the position in
the input image to sample. `sfm undistort` uses it to convert each photograph
from its own distorted perspective camera model (such as OPENCV; fisheye and
equirectangular cameras are rejected) to a pinhole camera with square pixels.
The same map, built from the rotation between two cameras or from an oriented
patch instead of from the two cameras' intrinsics alone, renders the panorama
tiles and the small image patches that patch refinement and strip montages
compare and the viewer displays. A third constructor builds it from two poses
and a depth.

This spec describes the two building blocks in the Rust `sfmtool-core` crate:

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

All pixel coordinates use the sfmtool convention: pixel centers at
`(col + 0.5, row + 0.5)`. This matches the optical flow module, the
`.sfmr`/`.sift` formats, and COLMAP.

Camera-space rays (`pixel_to_ray` output, `ray_to_pixel` input) are in the
canonical `.sfmr` camera frame: the camera looks down **−Z** with **+X
right, +Y up**, so a ray through the image has negative z. The distortion
kernels themselves operate internally in COLMAP's y-down / +Z-forward
optical frame, reached through the involutive camera-frame flip
`S = diag(1, −1, −1)` at the camera-model boundary. See the "Coordinate
System Conventions" section of
[`sfmr-file-format.md`](../../formats/sfmr-file-format.md).

## Warp Map

The map and its SVD data live in
[warp_map.rs](../../../crates/sfmtool-core/src/camera/warp_map.rs) (`WarpMap`,
`WarpMapSvd`, `from_cameras` and the pose-aware constructors below) and the
resamplers in [remap.rs](../../../crates/sfmtool-core/src/camera/remap.rs)
(`remap_bilinear`, `remap_aniso` and their variants), which read and write the
image types in [image.rs](../../../crates/sfmtool-core/src/camera/image.rs)
(`ImageU8`, `ImageU8Pyramid`, `ImageF32WithGrad`, also re-exported from
`camera::remap`), with `CameraIntrinsics::ray_to_pixel[_batch]` in
[distortion/projection.rs](../../../crates/sfmtool-core/src/camera/distortion/projection.rs)
and the `Equirectangular` camera model in
[distortion.rs](../../../crates/sfmtool-core/src/camera/distortion.rs); the
PyO3 bindings are in
[warp.rs](../../../crates/sfmtool-py/src/flow/warp.rs).

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
    /// The period, in source px, at which the source image's x wraps:
    /// `Some(2π·fx)` for an equirectangular source camera, `None` otherwise.
    /// See "Jacobian / SVD Data".
    x_period: Option<f64>,
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
At the edge of the map the difference is one-sided; where a neighbour it needs
is NaN, values are set to `(1, 1, (1, 0))` — the identity, causing the
resampler to fall back to a single bilinear sample.

An equirectangular source image's x wraps at the longitude ±180°, a period of
`2π·fx` px. A destination texel beside that seam has neighbours near `x = 0` and
near `x = width`, so a plain difference would give it a singular value about the
panorama's width, and `remap_bilinear_mip` would read the coarsest pyramid level
there, a blurred stripe down the tile. The constructors whose source camera is
equirectangular (`from_cameras`, `from_cameras_with_rotation`,
`from_cameras_with_pose`, `from_patch`) record that period in `x_period`, and
each x difference is brought into `±period/2` before it is divided, the same
rule `patch_grid_jacobian` applies to its four points (one private
`wrap_difference` serves both). `WarpMap::new` and every other source camera
leave `x_period` at `None`, which takes the unchanged `f32` difference, so their
Jacobians and SVDs are the plain differences. The wrap is what samples the
texels beside the seam at their own level wherever a mip or anisotropic remap,
or a reader of `get_jacobian`, uses a map with an equirectangular source: the
Track View tile and its hover view, the bench's `render_bitmap`, and the patch
kernels that render tiles through `from_patch`.

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
destination camera. `sfm undistort` gets it from
`CameraIntrinsics::best_fit_inside_pinhole` or `best_fit_outside_pinhole`
([pinhole_fit.rs](../../../crates/sfmtool-core/src/camera/distortion/pinhole_fit.rs)):
the principal point is the image centre, `fx = fy`, and the focal length is
found by binary search so that the pinhole frame fits inside, or encloses, the
distorted image. Both return an error for fisheye and equirectangular cameras.

### Parallelization

`from_cameras` is embarrassingly parallel — each output pixel is independent. The
implementation uses `rayon` to parallelize over rows, consistent with the existing
batch operations in `camera/distortion.rs`.

### Fisheye and the `ray_to_pixel` Gap

For fisheye undistortion, pixels near the image boundary may map to very large
image-plane coordinates (approaching infinity at 90° incidence). The existing
`project(x, y)` takes image-plane coordinates where `x = X/Z`, which is `tan(theta)`
— infinite at 90° and undefined beyond. Similarly, `unproject` returns image-plane
coords that blow up for wide-angle fisheye.

The codebase already solved the *inverse* direction with `pixel_to_ray`, which
recovers the incidence angle `theta` directly and builds a canonical unit ray as
`[sin(theta) * x/r, −sin(theta) * y/r, −cos(theta)]` (where `(x, y)` are the
y-down image-plane coordinates), sidestepping `tan(theta)`.

The forward direction — ray to pixel — is what the warp map needs: for each
destination pixel, we call `pixel_to_ray` on the destination camera to get a
ray, then project that ray into the source camera to get the source pixel
coordinates.

**The `ray_to_pixel` method**

```rust
impl CameraIntrinsics {
    /// Project a unit ray direction in (canonical, −Z-forward) camera space
    /// to pixel coordinates.
    ///
    /// For perspective models, equivalent to `project(rx/−rz, −ry/−rz)`
    /// (the S-flip into the kernels' y-down optical frame, then divide), but
    /// for fisheye models computes the distorted coordinates directly from
    /// the incidence angle `theta = atan2(sqrt(rx² + ry²), −rz)` off the −Z
    /// optical axis, avoiding the `tan(theta)` singularity. This is the true
    /// inverse of `pixel_to_ray`.
    ///
    /// Returns `None` if the ray falls outside the model's valid domain:
    /// for perspective models, `theta >= pi/2` (ray at or behind the camera
    /// plane); for fisheye models, only when the incidence angle exceeds the
    /// distortion polynomial's representable range (which may be well beyond
    /// 90°, up to ~180° or more for wide-angle fisheye).
    pub fn ray_to_pixel(&self, ray: [f64; 3]) -> Option<(f64, f64)>;

    /// Batch version.
    pub fn ray_to_pixel_batch(&self, rays: &[[f64; 3]]) -> Vec<Option<[f64; 2]>>;

    /// Grid version: project an affine grid of camera-frame rays
    /// (`origin + col·col_step + row·row_step`) to interleaved (sx, sy), with
    /// `(NaN, NaN)` for invalid nodes. Exact for perspective models; bounded
    /// coarse-grid interpolation for fisheye/equirectangular. See
    /// [ray-grid-projection.md](ray-grid-projection.md).
    pub fn ray_to_pixel_grid(
        &self,
        origin: [f64; 3], col_step: [f64; 3], row_step: [f64; 3],
        cols: u32, rows: u32, out: &mut [f32],
    );

    /// Project the homogeneous world point `(xyz, w)` at `cam_from_world`
    /// with no test against the image's bounds: `w = 1` a finite point,
    /// `w = 0` a direction, rotated into the camera frame without the
    /// translation. `None` for a perspective model where the camera-frame
    /// `z >= 0` (the camera looks along −Z), and for a ray-path model only
    /// outside its domain, since it images past 90° off the axis.
    pub fn project_homogeneous(
        &self,
        cam_from_world: &RigidTransform,
        xyz: Vector3<f64>,
        w: f64,
    ) -> Option<[f64; 2]>;
}
```

`project_homogeneous` is the world-to-pixel projection that the keypoint
localizer, the bench's steps, the patch-tile Jacobian
(`warp_map::patch_grid_jacobian`) and the viewer share. It skips the frame test
because a reprojection a pixel outside the frame is a small error, not a missing
measurement, and a tile partly off the photograph still has a geometry.

The bench's neighbourhood search (`ViewCamera::project_homogeneous` in
[`bench/track_at_pixel/neighbourhood.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel/neighbourhood.rs),
used by the track-at-pixel and nearby-point code) keeps its own variant with
different rules: it projects through the rotation matrix and translation it
caches once per view, normalizes the ray before `ray_to_pixel`, takes a
perspective point as behind the camera when `−z ≤ 1e-12` rather than `z ≥ 0`,
and returns `None` for a zero-length ray or a non-finite pixel. Its results were
measured under those rules, so it stays separate rather than calling this one.

`WarpMap::from_patch` builds its grid through `ray_to_pixel_grid`: it forms the
affine ray basis from the patch plane + pose (model-free, infinity-aware) and the
camera owns the projection. See
[ray-grid-projection.md](ray-grid-projection.md) for the seam and the bound on the
coarse-grid path.

For perspective models, `ray_to_pixel` maps the canonical ray through `S`
into the optical frame, divides by the (positive) forward component and
calls the existing `distort()` + focal/principal point transform. For
fisheye models, it computes:

```
theta = atan2(sqrt(rx² + ry²), -rz)   // incidence angle off the −Z optical axis
r_xy = sqrt(rx² + ry²)
(dx, dy) = (rx / r_xy, -ry / r_xy)    // unit direction in the y-down image plane
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

Internally, `ray_to_pixel` computes `theta = atan2(sqrt(rx² + ry²), −rz)` and
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
longitude = atan2(rx, -rz)           # 0 straight ahead (along −Z)
latitude  = asin(clamp(ry / |ray|, -1, 1))   # positive = up (+Y)
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
ray = [sin(longitude) * cos(latitude), sin(latitude), -cos(longitude) * cos(latitude)]
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
- **Compute the corner geometry once per pixel, not once per channel.** The
  half-pixel offset, `floor`/`clamp` edge handling, and stride index depend only
  on `(sx, sy)`, so for a 3-channel (RGB) source they are identical across
  channels. A per-channel sampler re-derives them 3× per pixel; a channel-batched
  gather (`bilinear_corners` → the four corner base indices + blend weights, then
  `data[idx[k] + ch]` per channel) does the address math once and only varies the
  fetch. The batched gather keeps the per-channel path's multiply/add order, so
  its output is **bit-identical** to it (tests
  `sample_bilinear_u8_all_matches_per_channel` and
  `sample_bilinear_with_grad_u8_all_matches_per_channel`). The single geometry
  helper (`bilinear_geometry`) also backs the value+gradient sampler used
  by keypoint-subpixel refinement (`remap_bilinear_with_grad`), so both the value
  and gradient batched gathers stay in lockstep with the per-channel path.
  Opt-in sampler counters live in `camera::remap::prof` (gated on
  `SFMTOOL_PROFILE`).

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

#### Single-Tap Mip Sampling

`remap_bilinear_mip(pyramid, map)` (and its value+gradient twin
`remap_bilinear_mip_with_grad[_into]`) is the middle point between
`remap_bilinear` and the anisotropic path: per output pixel it selects the
pyramid level nearest the warp's local compression —
`level = round(log2(max(sigma_major, 1)))`, clamped to the built levels — and
takes a single bilinear sample there at `(x / 2^level, y / 2^level)`. Using
`sigma_major` (the larger singular value, the GL texture-LOD convention) means
the chosen level never aliases in any direction; on anisotropic footprints it
over-blurs the minor axis, which remains `remap_aniso`'s job. Use it where a
warp compresses the source (e.g. cross-scale patch renders) but the multi-tap
anisotropic cost is not warranted: aliasing stays bounded at ≈ bilinear cost.
Like the anisotropic path it requires `map.compute_svd()` first; on a
non-compressive map the output is bit-identical to `remap_bilinear`. The
gradient twin returns `(∂I/∂x, ∂I/∂y)` rescaled to full-resolution source-pixel
coords (a level-`l` bilinear gradient is per level-pixel, so it is divided by
`2^l`).

The patch kernels choose between this path and the anisotropic one per view,
by the rule in § "Choosing the sampler per view".

#### Pyramid Construction

The Gaussian pyramid for `ImageU8` is built by repeated box-filter or Gaussian
downsample-by-2, operating independently per channel. This is a separate lightweight
implementation from the optical flow `ImagePyramid` (which operates on `GrayImage`
f32 single-channel). The number of levels is `floor(log2(min(width, height)))`.

This filtering is primarily useful for undistorting fisheye images where the
periphery represents a much larger field of view per pixel than the center. For
standard perspective cameras with mild radial distortion, `remap_bilinear` alone
is usually sufficient.

### Choosing the sampler per view

A patch kernel renders each view of a point into the point's `R×R` grid, and
the sampler decides how much of the photograph's detail reaches the tile.
`BilinearMip` reads one mip level for both axes, chosen by the more compressed
one, so a view that sees the patch at an angle, or through a distorting lens,
loses the detail it holds along its less compressed axis before anything reads
the tile. The anisotropic sampler keeps that detail. The **sampler rule**
renders each view with the anisotropic sampler where the single level would
read the less compressed axis too coarsely, and with `BilinearMip` everywhere
else, so every kernel that renders the same observation renders it with the
same sampler.

#### Interface

[camera/sampler.rs](../../../crates/sfmtool-core/src/camera/sampler.rs) holds
the `Sampler` enum (`Bilinear`, `BilinearMip`, `Anisotropic`), the rule and
the one render dispatch every patch kernel goes through.

```rust
pub enum SamplerChoice {
    Fixed(Sampler),                              // every view, one sampler
    PerView { anisotropic_threshold: f64 },      // the rule, with its threshold `a`
}
impl Default for SamplerChoice { /* PerView at DEFAULT_ANISOTROPIC_THRESHOLD = 1.5 */ }

impl SamplerChoice {
    pub fn for_observation(self, patch: &OrientedPatch, camera: &CameraIntrinsics,
        cam_from_world: &RigidTransform, keypoint: Option<[f64; 2]>, resolution: u32) -> Sampler;
    pub fn for_placement(self, placement: &OrientedPatch, /* camera, pose, resolution */) -> Sampler;
    pub fn for_jacobian(self, jacobian: Option<[[f64; 2]; 2]>) -> Sampler;
    pub fn for_singular_values(self, singular_values: Option<[f64; 2]>) -> Sampler;
}

pub fn rule_sampler(singular_values: [f64; 2], threshold: f64) -> Sampler;
pub fn minor_axis_loss(singular_values: [f64; 2]) -> f64;      // L
pub fn render_tile(pyramid: &ImageU8Pyramid, map: &mut WarpMap, sampler: Sampler) -> ImageU8;
pub fn render_tile_with_grad_into(/* … */);
pub fn render_tiles(pyramids: &[&ImageU8Pyramid], maps: &mut [WarpMap],
    samplers: &[Sampler], progress: &Progress<'_>) -> Vec<ImageU8>;
```

```rust
let choice = SamplerChoice::default();
let sampler = choice.for_observation(&patch, view.camera, view.cam_from_world, Some(keypoint), 24);
// The tile itself, rendered through the patch re-anchored on the keypoint.
let anchored = patch
    .anchored_at_keypoint(view.camera, view.cam_from_world, keypoint)
    .unwrap_or_else(|| patch.clone());
let mut map = WarpMap::from_patch(&anchored, view.camera, view.cam_from_world, 24);
let tile = render_tile(view.pyramid, &mut map, sampler);
```

- **Why a choice rather than a sampler.** Every patch kernel's parameters
  (`NormalRefineParams`, `ViewSelectParams`, `KeypointLocalizeParams`,
  `KeypointSubpixelParams`, `MemberCoherenceParams`) carry a `SamplerChoice`,
  the rule by default. `Fixed` keeps one sampler for every view, for comparison
  runs and for a caller that needs one; `Sampler::X.into()` gives it. The
  bindings and the command line spell the rule `per_view` beside the three
  sampler names.
- **Why one placement.** `for_observation` reads the rule from the patch
  re-anchored on the observation's keypoint
  (`OrientedPatch::anchored_at_keypoint`), at the patch resolution `R`, as the
  stored frame has it. The kernels render through frames of their own: the
  localizer a context tile wider than `R` and centred on the projection, the
  sub-pixel refiner a tile padded for its drift, and the refiners an
  orthonormal frame rebuilt around the stored `v` axis, which differs from the
  stored one where the stored axes are not perpendicular. Reading the rule from
  any of those would let two kernels choose differently for one observation,
  so none of them does. The bench's evaluation, Track View
  (`PatchJacobian::sampler`) and the MCP fields read the same Jacobian
  (`patch_grid_jacobian` of the anchored patch at `R`), and Track View and the
  MCP fields apply the bench evaluation's own `SamplerChoice`
  (`EvaluateOptions::localize.sampler`), so they show the sampler the bench
  used.
- **Which keypoint.** Each kernel reads the rule at the keypoint it starts
  from: the fuse, the member gates and view selection's track views at the
  stored keypoint, the bench and the localizer at the seed of the round, the
  sub-pixel refiner at the seed the localizer hands it, and view selection's
  candidates, which have no keypoint, at the projection. Within one pipeline
  run these differ by the localizer's and the refiner's shifts, a few pixels,
  which change the Jacobian by a small fraction. Two kernels can therefore
  choose differently for one observation only when its `L` lies within that
  fraction of `a`, or its `σ_major` within it of a level boundary
  `2^(l + 0.5)`, where `L` doubles. Reading every kernel's choice at one
  keypoint would mean carrying that keypoint through each kernel beside the one
  it renders at, and a kernel that has moved a view should render it as it now
  sits, so the choice is read where each kernel starts.
- **Why a kernel that iterates chooses once.** Normal refinement fixes each
  view's sampler from the patch it starts from (`view_samplers`), since the
  Jacobian changes with the candidate normal and a choice per candidate would
  make `Φ` jump where a candidate carries a view across the threshold. Every
  render of a view from its photograph then uses that sampler. Under the
  default `CacheMode::FrontoParallel` the search does not render candidates
  from the photographs: it scores them from the fronto-parallel cache, whose
  base tiles are rendered with plain bilinear, so the sampler reaches only the
  final scoring of the starting normal and the search's survivors, the
  confidence stencil and the representative bitmap. With `CacheMode::Off` it
  reaches every candidate. The localizer and the sub-pixel refiner choose once
  per view, at its seed keypoint, since their tiles only slide in the patch's
  plane.
- **Timing.** `render_tiles` renders a multi-view stack grouped by sampler, and
  each group runs in a `Progress::detail_phase` named `render bilinear`,
  `render bilinear_mip` or `render anisotropic` whose note gives the number of
  views; the single-view kernels open the same phases per render
  (`render_phase`). The phases record only under `Progress::detailed(true)`.
  The batches that render views (`refine_patch_cloud_normals`,
  `select_patch_cloud_views`, `localize_patch_cloud_keypoints`,
  `refine_patch_cloud_keypoints`, `validate_patch_cloud_member_coherence` and
  `fuse_patch_cloud_bitmaps`) take a `&Progress`, count `patches`, poll for
  cancellation before each patch and return `Cancelled` when it is set, rather
  than the patches they finished; normal refinement then leaves the cloud as
  it was. The bench's `evaluate` and `fit` work on one track; they take a
  `&Progress` too, open the same detail phases and return
  `EvaluateError::Cancelled` / `FitError::Cancelled`.

#### The footprint, and what `BilinearMip` loses

The Jacobian of the patch grid's map into the photograph at the tile's centre
has singular values `σ_major ≥ σ_minor`, in photograph px per grid px.
`BilinearMip` reads level `l = round(log2 max(σ_major, 1))`, whose pixels are
`2^l` photograph pixels wide, so one sample spans `φ_a = max(2^l, 1) / σ_a`
grid px along each singular direction. The view's **footprint**
`φ_v = max(φ_major, φ_minor, 1)` is the scale in grid px below which its tile
holds no detail. A view that shrinks the photograph equally on both axes is
read at the level that matches, so its footprint stays between 1 and √2
however far away it is; a view that magnifies the photograph has a footprint
equal to its zoom. Under the anisotropic sampler the footprint is close to
`max(1, 1/σ_a)` per axis, since it takes the level from `σ_minor` and several
samples along the major axis.

The loss along the minor axis is

```
L = 2^l / max(σ_minor, 1)
```

how much coarser `BilinearMip` reads that axis than its own compression needs.
A minor axis that magnifies the photograph (`σ_minor < 1`) can only lose detail
down to one photograph pixel, hence the floor.

`l` here is not clamped to the pyramid's depth, although `BilinearMip` clamps
it to the top level `K` it has. Past `σ_major = 2^(K − 0.5)` the minor axis is
read at `2^K / max(σ_minor, 1)`, less than `L`, and the major axis aliases by
`σ_major / 2^K` instead. The rule reads `L` unclamped: the anisotropic sampler
is the better of the two there too, since its samples along the major axis
reduce that aliasing, and the depth differs between callers (the command line
builds every level, SfM Explorer six), while the rule has to choose the same
sampler for one observation in every kernel and caller. An infinite `σ_major`
gives an infinite `L`.

#### The rule

A view renders with `Anisotropic` when **`σ_major ≥ √2` and `L ≥ a`**, and with
`BilinearMip` otherwise (`rule_sampler`).

- Below `σ_major = √2` `BilinearMip` reads level 0 for both axes, and a view
  whose lower zoom is above about 0.71× loses nothing to it.
- `L` above 1 also arises from the level rounding alone: a view compressed
  alike on both axes reads `L` up to √2 just under a level boundary. `a` is
  above √2 so that those views stay on `BilinearMip`.
- A view with no Jacobian (its centre does not project) stays on `BilinearMip`.
- The cause of the anisotropy does not enter. A view near the edge of a fisheye
  image and a pinhole view of a patch at 70° with the same Jacobian render the
  same way.

#### How `a = 1.5` was set

The rule was compared against `BilinearMip` everywhere on eleven samples, with
a measurement harness that was written for this comparison and is not in the
repository: up to 400 points per sample, all their views, and 40 tracks of
each on the bench. Two samples are in the repository, the seoul_bull and
kerry_park ground truths in `test-data/images/`. The other nine are local
datasets that are not: a seed reconstruction of dino_dog_toy, two museum
captures (masks, a tree stump exhibit), a gallery sculpture, a mossy railing,
a distant badlands panorama, a 48-image fisheye rig (KerryPark480) and two
12-camera 360° rigs (OmniCoast, OmniTemple1). The 48-image fisheye rig's lenses carry extreme `k2`/`k3` terms and its data near
the edge of the image circle is unreliable, so its views near that edge say
little about the rule; the kerry_park ground truth is the fisheye sample to
read.

| Sample | views | moved at `a = 1.5` | self-similarity semi-major axis, moved views, p50 `BilinearMip` → `Anisotropic` (grid px) | moved view's ZNCC against the track's other views, mean change |
|---|---|---|---|---|
| seoul_bull | 1277 | 15.0% | 0.90 → 0.50 | −0.010 |
| kerry_park | 3767 | 6.6% | 0.86 → 0.49 | −0.005 |
| dino_dog_toy | 2371 | 31.2% | 1.32 → 1.12 | +0.012 |
| museum masks | 3325 | 19.9% | 1.71 → 1.58 | −0.002 |
| gallery sculpture | 1948 | 51.5% | 0.82 → 0.52 | −0.001 |
| tree stump | 3013 | 24.0% | 1.24 → 1.06 | −0.005 |
| fisheye rig | 4931 | 5.3% | 1.27 → 0.96 | −0.011 |
| 360° rig, coast | 4698 | 14.9% | 0.78 → 0.52 | −0.006 |
| 360° rig, temple | 6915 | 9.9% | 0.96 → 0.73 | −0.001 |
| badlands | 3398 | 31.2% | 1.48 → 0.81 | +0.043 |
| mossy railing | 2032 | 4.9% | 1.14 → 1.02 | −0.002 |

- **Every band of `L` above √2 sharpens the moved views.** Split by the views
  each step of the threshold adds, the mean shortening of the semi-major axis
  is 0.21–0.66 grid px for `L ≥ 2` (leaving out the mossy railing's three such views), 0.19–0.44 for `1.75 ≤ L < 2`, 0.11–0.32 for
  `1.5 ≤ L < 1.75` and 0.08–0.23 for `1.42 ≤ L < 1.5`, with the moved view's
  ZNCC against the others changing by under 0.014 on average in the two lower
  bands. `1.5` keeps the bands down to 1.5 and leaves a margin above the √2
  that rounding alone gives a view compressed alike on both axes.
- **The ZNCC drops slightly where the gain is largest.** A sharper tile carries
  detail the other views' tiles lack, which normalized correlation charges for
  (sharper-patch-bitmap draft, § "Why the mean is blurry"); for `L ≥ 2` the mean change of a
  moved view's ZNCC is between −0.018 and +0.014 on the other samples, and
  close to 0 below. On the badlands
  sample, whose far views compress the photograph 10 to 50 times along one
  axis, `BilinearMip` reads a level clamped at the top of the pyramid and
  aliases, and the anisotropic sampler raises the ZNCC by 0.043.
- **A view the rule leaves on `BilinearMip` renders the same tile, bit for
  bit**, so a point none of whose views moves keeps its fused bitmap: none of
  the 2,549 such points across the samples changed. The fused bitmaps of the
  points with a moved view change by a median mean absolute difference of
  0.3–3.2 grey levels.
- **No bench verdict changes at the current bars** (in 0, out 0 over 3,828
  observations). The number of points member coherence does not keep whole
  changes by 0 to 9 of 400 per sample, and view selection admits 0–3% fewer
  views.

#### Cost, and the AVX2 kernel

`remap_aniso_with_pyramid` has an AVX2 kernel
([camera/remap/aniso_avx2.rs](../../../crates/sfmtool-core/src/camera/remap/aniso_avx2.rs)),
chosen at run time with `is_x86_feature_detected!("avx2")`: eight output
pixels of a row at a time wherever they read the same two pyramid levels, each
lane with its own position, direction, sample count and level blend, and the
corners fetched with one 32-bit gather per corner for all channels. It does the
scalar path's `f32` operations in the same order, with no fused multiply-add,
so its output is identical to the scalar path's bit for bit
(`aniso_avx2_matches_scalar_bit_for_bit`, which also checks that the kernel
rendered groups), and the scalar path, which computes a sample's corner
geometry once for all channels, is identical to the per-channel algorithm for
any channel count (`aniso_scalar_matches_the_per_channel_reference`). The
kernel takes 1 to 4 channels; an image of more goes through the scalar path.
The value+gradient anisotropic sampler the sub-pixel refiner's tile reads has
no AVX2 kernel.

**Per render.** Each `R×R` tile was rendered on one thread, one render after
another, with a second local harness that is not in the repository either, at every view of every point of two
local samples (the gallery sculpture, 2,631 points, 12,844 views, and the
badlands panorama, 1,171 points, 9,951 views; `R = 24`, three renders per view
and sampler, the warp map built outside the timed call and its SVD inside it).
Mean µs per render:

| Render | gallery sculpture, moved views | badlands, moved views | badlands, unmoved views |
|---|---|---|---|
| `Bilinear` | 12 | 20 | 13 |
| `BilinearMip` | 33 | 32 | 32 |
| `Anisotropic`, AVX2 kernel | 22 | 50 | 22 |
| `Anisotropic`, scalar path | 54 | 129 | 65 |
| value+gradient `BilinearMip` | 34 | 33 | 33 |
| value+gradient `Anisotropic` (scalar) | 94 | 226 | 94 |

The SVD is about 9 µs of each render that reads it. With the AVX2 kernel an
anisotropic render costs about what a `BilinearMip` one does, and less where
its views take few samples; the badlands sample's moved views compress the
photograph 10 to 50 times along one axis and take up to the full 16 samples.
On the scalar path it costs 1.6 to 4 times a `BilinearMip` render, and the
value+gradient anisotropic render 2.8 to 7 times. Earlier timings of these
renders taken inside the batches' parallel loops, through detail phases
reported to a shared sink, read 41–57 µs for a `BilinearMip` render; that
measured the contention of the loop and the sink, not the render.

**Per batch.** Each batch ran on all 32 threads over the same points, five
times under `Fixed(BilinearMip)` and five times under the rule, alternating
which went first, after one run to warm up; the table gives the best time
under the rule over the best under `BilinearMip`. The rule moves 57% of the
gallery sculpture's views and 37% of the badlands panorama's over these whole
reconstructions, more than the 51.5% and 31.2% of the 400-point subsets in the
table above.

| Batch | gallery sculpture (two runs) | badlands |
|---|---|---|
| view selection | 1.05, 1.05 | 1.24 |
| normal refinement | 1.00, 1.36 | 1.06 |
| localizer | 1.02, 1.00 | 1.12 |
| member coherence | 0.97, 0.98 | 1.19 |
| fuse | 1.00, 0.99 | 1.31 |

The times of one batch spread by 1–24% over its five runs, and by up to 51%:
the badlands fuse spread by 41% under both samplers, so its 1.31 is uncertain
(an earlier run of the code before the view selection fast path, with spreads
of 4–6%, read 1.22), and the 1.36 for normal refinement is one run whose fastest `BilinearMip` time,
1.60 s, was 25% below every other `BilinearMip` run of it (2.16–2.55 s), and
both samplers do the same evaluations and renders per point. Run one after
another on one thread, view selection over the gallery sculpture takes 4.43 s
and 4.46 s under `BilinearMip` and 4.54 s and 4.47 s under the rule (best of
five, two runs; spread 14–42% within a run). On the badlands sample the
moved views' renders take most of their samples, and every batch that renders
them is 6–31% slower under the rule. The bench's evaluation, measured earlier
with three runs on each of the eleven samples, runs within 6% of its
`BilinearMip` time (16% on dino_dog_toy).

View selection scores the views the rule moves through its affine fast path
too ([patch-view-selection.md](../patch/patch-view-selection.md) § "Affine
candidate scoring"). Before that path took the anisotropic sampler, the
gallery sculpture's view selection was 18–20% slower under the rule as a batch
and 14–15% slower on one thread.

#### Storing what a render was made under

The rule reads only the zoom and `a`, so a stored reading needs no record of
its sampler: `sfm embed-patches` records `anisotropic_threshold` beside
`sampler` in the file's `tool_options`, and a reader that knows a render's zoom
works out its sampler. Storing `a` beside per-observation readings in the
`.sfmr` is proposed with those readings in
[sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md) (Part 7).

## Python Bindings

Expose the warp map and resampling through `sfmtool-py`:

```python
from sfmtool._sfmtool.geometry import CameraIntrinsics
from sfmtool._sfmtool.flow import WarpMap

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

In `sfmtool-core`:

```
crates/sfmtool-core/src/camera/
├── warp_map.rs          # WarpMap struct, from_cameras(), Jacobian estimation
├── image.rs             # ImageU8, ImageU8Pyramid, ImageF32WithGrad
├── remap.rs             # remap_bilinear(), remap_aniso() and variants
├── remap/aniso_avx2.rs  # the AVX2 kernel of remap_aniso_with_pyramid()
├── sampler.rs           # Sampler, SamplerChoice, the sampler rule, render_tile()
```

In `sfmtool-py`:

```
crates/sfmtool-py/src/
├── flow/warp.rs       # PyWarpMap Python wrapper
```

`ray_to_pixel` is a method on `CameraIntrinsics` in
`crates/sfmtool-core/src/camera/distortion/projection.rs`, and `distort_ray` a
method on `CameraModel` in `crates/sfmtool-core/src/camera/distortion.rs`.

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

## Pose-Aware Construction

*(Folded in from the implemented `warpmap-pose-extension.md` draft, which this
section supersedes.)*

`WarpMap::from_cameras(src, dst)` assumes both cameras observe the same ray
through the world — correct when they share a world-space pose (the canonical
undistortion / re-projection use case), but wrong whenever the two cameras have
different poses and the map should reflect the scene that lives between them.
Two further construction paths cover that case:

1. **Rotation-aware (at infinity).** "For every destination pixel, take its ray
   in dst-space, rotate it into src-space, call `src.ray_to_pixel(ray)`."
   Models the limit where the scene is infinitely far and only the rotation
   between the two cameras matters.

2. **Pose-and-depth-aware.** "For every destination pixel, take its ray in
   dst-space, trace it to a point at radial distance `r` from the dst camera
   center (expressed in world coordinates), transform that point into
   src-camera coordinates, project." Models a sphere of radius `r` around the
   dst camera — pixels on that sphere land exactly where the pose-aware map
   says they do.

Both paths reuse the existing `ray_to_pixel` machinery and the existing
`RigidTransform` type; neither requires new camera-model arithmetic. They
generalize `from_cameras`.

### API

Rotations and poses use the types the rest of the codebase already uses:
`RotQuaternion` for rotations and `RigidTransform` for world-to-camera poses,
matching the convention in `SfmrImage::{quaternion_wxyz, translation_xyz}`.

```rust
impl WarpMap {
    /// Rotation-only construction. For each dst pixel center,
    ///   d_dst = dst.pixel_to_ray(u, v)
    ///   d_src = rot_src_from_dst * d_dst
    ///   (sx, sy) = src.ray_to_pixel(d_src)
    ///
    /// Equivalent to assuming the scene is infinitely far: only the relative
    /// rotation between the two cameras affects the projection.
    pub fn from_cameras_with_rotation(
        src: &CameraIntrinsics,
        dst: &CameraIntrinsics,
        rot_src_from_dst: &RotQuaternion,
    ) -> Self;

    /// Full pose + depth construction.
    ///
    /// Implemented as a single 3x3 matrix multiply and vector add per dst
    /// pixel:
    ///   d_dst = dst.pixel_to_ray(u, v)                    // unit, dst-cam frame
    ///   p_src = R_sd * (depth * d_dst) + T_sd             // src-cam frame
    ///   (sx, sy) = src.ray_to_pixel(p_src)
    /// where `R_sd = R_sw * R_dw^T` and `T_sd = t_sw - R_sd * t_dw`. This is
    /// the exact formulation, not a small-angle approximation.
    ///
    /// `src_from_world` and `dst_from_world` are world-to-camera extrinsics,
    /// matching `SfmrImage::{quaternion_wxyz, translation_xyz}`.
    ///
    /// `depth` is the radial distance from the dst camera center along the
    /// dst ray. Passing `f64::INFINITY` short-circuits to the
    /// `from_cameras_with_rotation` path (the only pose component that still
    /// matters is the relative rotation).
    pub fn from_cameras_with_pose(
        src: &CameraIntrinsics,
        dst: &CameraIntrinsics,
        src_from_world: &RigidTransform,
        dst_from_world: &RigidTransform,
        depth: f64,
    ) -> Self;
}
```

Both constructors share a single implementation helper
(`build_with_pose_impl`) that iterates dst rows in parallel via rayon,
matching `from_cameras`. The impl uses the collapsed
`p_src = R_sd * p_dst + T_sd` form — no quaternion multiplication, no
`inverse()` call, no small-angle approximation — so it's numerically exact
at all baselines.

### Why depth is radial, not Z

A "Z-depth" plane only makes sense for perspective cameras with a well-defined
optical axis. This method must also work when `dst` is equirectangular or
fisheye, where no single Z direction applies — so depth is expressed as radial
distance from the dst camera center along each dst ray (a sphere, not a plane).
For perspective dst the two agree up to a per-pixel `cos(theta)` factor, and
callers who want a fronto-parallel plane can convert.

### Python bindings

Exposed via `flow/warp.rs`. Rotations accept a `RotQuaternion`; poses accept
a `RigidTransform` built from the same `(quaternion_wxyz, translation_xyz)`
tuple already stored on reconstruction images:

```python
from sfmtool._sfmtool.geometry import RigidTransform
from sfmtool._sfmtool.flow import WarpMap

src_from_world = RigidTransform.from_wxyz_translation(
    recon.quaternions_wxyz[src_idx].tolist(),
    recon.translations[src_idx].tolist(),
)
dst_from_world = RigidTransform.from_wxyz_translation(
    recon.quaternions_wxyz[dst_idx].tolist(),
    recon.translations[dst_idx].tolist(),
)
warp = WarpMap.from_cameras_with_pose(
    src=src_camera, dst=dst_camera,
    src_from_world=src_from_world,
    dst_from_world=dst_from_world,
    depth=scene_radius,
)
```

The PyO3 method signatures use keyword-only style to match the existing
`WarpMap.from_cameras(src=…, dst=…)` convention. `compute_svd()`,
`remap_bilinear()`, `remap_aniso()`, and `to_numpy()` work unchanged on maps
built via either new constructor — the Jacobian is estimated from the map via
central differences and is agnostic to how the map was built.

### Implementation notes

- For `depth = INFINITY` the implementation detects the non-finite depth
  and dispatches to the rotation-only path with the precomputed `R_sd`,
  instead of relying on IEEE arithmetic to cancel the translation cleanly
  (which wouldn't happen — `inf * 0 = NaN`).
- The dst camera bounds are always validated via `ray_to_pixel` plus a
  subsequent `[0, src_w) × [0, src_h)` check, matching the `from_cameras`
  semantics. Rays projecting behind a perspective src camera return `None`
  from `ray_to_pixel` and therefore become `NaN` in the warp map.
- `pixel_to_ray` (not `unproject`) is used on the dst side so
  equirectangular and fisheye destinations work correctly: the new
  constructors always use the ray path (a superset of the image-plane path
  `from_cameras` selects via `needs_ray_path`).
- Tested by Rust unit tests in the `warp_map` tests module (identity / INF /
  coincident-pose / known-depth / equirect cases) and
  `tests/patch/test_warp_map_pose.py`, whose `TestRealReconstruction` cases validate
  the full world-coordinate chain end-to-end against real seoul_bull geometry.

### Non-goals

- No automatic derivation of the pose from reconstruction format metadata —
  that stays in the Python wrapper.
- No per-dst-pixel depth map. For varying-per-pixel depths the caller
  rebuilds the map or uses a depth-aware resampler (future work).

## Open Questions

1. **GPU acceleration**: The optical flow module has a GPU (wgpu) code path. Should
   warp map resampling also have one? Deferred — CPU with rayon parallelization is
   sufficient for the initial implementation. The warp map + remap pattern is
   naturally GPU-friendly if needed later.
