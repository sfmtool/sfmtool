# Per-image radial and rolling-shutter corrections

**Status:** Draft. Amends
[formats/sfmtool-camera-models.md](../formats/sfmtool-camera-models.md),
[formats/sfmr-file-format.md](../formats/sfmr-file-format.md),
[core/camera/sfmtool-pinhole-kernels.md](../core/camera/sfmtool-pinhole-kernels.md),
[core/geometry/bundle-adjustment.md](../core/geometry/bundle-adjustment.md),
[core/reconstruction/bundle-adjust.md](../core/reconstruction/bundle-adjust.md)
and [gui/edits/bundle-adjust.md](../gui/edits/bundle-adjust.md).

Decided by the maintainer: the per-image radial correction is a two-coefficient
spline in the camera's own basis, added to the camera's spline for that image;
each image has its own rolling-shutter rotation rates `ω = (ωx, ωy, ωz)`; each
camera has a rolling-shutter scanlines setting, stored as `ROWS` or `COLUMNS`
and chosen in the bundle adjustment as `ROWS`, `COLUMNS` or `UNKNOWN`, where
`UNKNOWN` solves with each and keeps the lower cost, and a camera that
releases its rates with no stored setting is solved as `UNKNOWN`, never as an
assumed `ROWS`; every new per-image field is zero-mean over its camera's images
with the mean held. Not decided: the prior widths, the early-stopping rule for
`UNKNOWN`, and the questions listed under
[Open questions](#open-questions).

## Purpose

A phone camera does not take every photograph through quite the same lens. Its
autofocus moves the lens, which changes the focal length and, slightly, the
shape of the radial distortion; its stabiliser shifts the lens sideways; and its
sensor reads the image out row by row, so a camera that turns while it takes a
photograph records the bottom of the frame from a slightly different direction
than the top. A reconstruction that gives every photograph the same lens leaves
these per-photograph effects in its residuals, where they look like a lens error
that grows toward the edge of the frame and that no single shared lens can fit.

sfmtool already solves part of this: each image of a camera can have its own
focal scale and principal-point offset, solved by the bundle adjustment under a
prior centred on the mean of the camera's images
([bundle-adjustment.md](../core/geometry/bundle-adjustment.md#the-per-image-release)).
This draft proposes two more per-image corrections of the same kind for the
`SFMTOOL_PINHOLE` camera model:

- **a per-image radial correction**, a small change to the radial distortion
  for each image, for distortion that changes with the focus distance;
- **per-image rolling-shutter rates**, the rate at which the camera turned while
  the sensor read the image out, for the image distortion that turning causes;

and **a per-camera rolling-shutter scanlines setting**, which says whether the
sensor reads one row at a time, rows one after another down the image (`ROWS`,
the usual case), or one column at a time across it
(`COLUMNS`).

On the one capture measured, a handheld Pixel 7 Pro sequence of 85 images, the
two corrections together lower the held-out reprojection error from 0.386 px to
0.230 px, and lower it in every radial band of the image
([Measured results](#measured-results)).

## The model

### Coordinates and conventions

Every quantity of the proposal is in the `.sfmr` camera frame: +X right, +Y up,
the camera looking down −Z. A camera-frame point `P = (Px, Py, Pz)` in front of
the camera (`Pz < 0`) has normalized undistorted coordinates

```
(x, y) = (Px, Py) / (−Pz),        ρ = hypot(x, y)
```

These are the `(x, y)` that `CameraIntrinsics::project` in
[projection.rs](../../crates/sfmtool-core/src/camera/distortion/projection.rs)
takes. The rates `ω` are the camera's angular velocity in this frame.

**The one place y-up meets pixel-down** is the existing pixel mapping, and the
proposal does not change it. `CameraIntrinsics::project` calls the model's
`distort(x, −y)`, because the radial kernels run in the optical frame
`(x, −y, −z)` of
[sfmtool-camera-models.md](../formats/sfmtool-camera-models.md#sfmtool_pinhole),
which looks down +Z with +Y pointing down the image, the way pixel `v` grows.
With the kernel's output written `(x_o, y_o)`, the pixel is then
`u = f·x_o + cx` and `v = f·y_o + cy`. A radial map commutes with the flip, so
`(x_o, y_o) = (x_d, −y_d)` for the `.sfmr` distorted point `(x_d, y_d)`, and in
`.sfmr` coordinates that step reads `u = cx + f·x_d`, `v = cy − f·y_d`. The
new terms are applied in `.sfmr` coordinates before that step, so they add no
sign change of their own.

### The map for image `i`

Image `i` of camera `c` has the camera's spline `δ` (coefficients `c₀..c_{N−1}`
on `[0, ρ_max]`, `ρ_max` being `bspline_rho_max`), its existing focal scale and
principal-point offset (so its focal `f_i` and principal point `(cx_i, cy_i)`
are those of `CameraIntrinsics::with_adjustment`), and the two new fields:

- `e_i = (e_i0, e_i1)`, its radial correction coefficients;
- `ω_i = (ωx, ωy, ωz)`, its rolling-shutter rates.

A ray is projected in three steps.

**1. Radial distortion, with the image's correction.**

```
Δδ_i(ρ) = e_i0·B0(ρ) + e_i1·B1(ρ)
(x_d, y_d) = (x, y) · (ρ + δ(ρ) + Δδ_i(ρ)) / ρ
```

This is rotationally symmetric, so it has the same form in the `.sfmr` frame and
in the kernels' optical frame.

`B0` and `B1` are the camera's centre-anchored cubic basis built with the
minimum coefficient count, two (`MIN_BSPLINE_COEFFS` in
[bspline.rs](../../crates/sfmtool-core/src/camera/distortion/bspline.rs)), on
the camera's own domain `[0, ρ_max]`. With two coefficients the clamped basis
has four functions and no interior knot, so it is the cubic Bernstein basis in
`τ = ρ/ρ_max`, and omitting the gauge-anchored pair leaves

```
B0 = 3τ²(1 − τ),        B1 = τ³          (0 ≤ ρ ≤ ρ_max)
```

so `Δδ_i` spans `{ρ², ρ³}`: it has no constant and no linear part
(`Δδ_i(0) = Δδ_i'(0) = 0`), so it does not overlap the per-image focal scale,
which is the linear part. `Δδ_i(ρ_max) = e_i1`. Past `ρ_max`, `Δδ_i` continues
along its end tangent like the camera's spline,
`Δδ_i(ρ) = e_i1 + 3(e_i1 − e_i0)·(ρ − ρ_max)/ρ_max`.

**2. Rolling shutter.** The readout time `t` is

```
t = y_d    for ROWS      (one row at a time)
t = x_d    for COLUMNS   (one column at a time)
```

the position across the scanlines, in normalized `.sfmr` coordinates, measured
from the scanline through the image's principal point, where `t = 0`, so the
image's stored pose is the camera's pose when that scanline was read. The
readout duration is folded into the rates, so `t` is in normalized image units
and the rates are in radians per normalized image unit of `t`. Then

```
x_r = x_d + t·ẋ,        ẋ =  ωy·(1 + x²) − ωx·x·y + ωz·y
y_r = y_d + t·ẏ,        ẏ = −ωx·(1 + y²) + ωy·x·y − ωz·x
```

`ṗ = (ẋ, ẏ)` is the image-plane velocity of a fixed point under a small rotation
of the camera with angular velocity `ω`. The point's camera coordinates change as
`Ṗ = −ω × P`; with `w = −Pz`, `Ṗx = ωy·w + ωz·Py`, `Ṗy = −ωx·w − ωz·Px` and
`ẇ = ωx·Py − ωy·Px`, and differentiating `x = Px/w`, `y = Py/w` gives
`ẋ = Ṗx/w − x·ẇ/w` and `ẏ = Ṗy/w − y·ẇ/w`, which are the two terms above.

Only whether the scanlines are rows or columns is part of the model, not the
direction the scan advances in. A scan that advances the other way gives the
same fit with every rate negated, and the rates are zero-mean and free, so they
take either sign. For `ROWS`, `t` grows up the image (`+y` is up), while a usual
sensor reads its rows down the image; the fitted rates are then the negative of
the angular velocity times the readout duration per unit, and nothing else in
the model changes.

**3. Pixels.** The existing step, as for the existing adjustments:

```
u = cx_i + f_i·x_r,        v = cy_i − f_i·y_r
```

which is the `.sfmr` reading of the kernels' `distort(x, −y)` followed by
`f·(·) + c` described in [Coordinates and conventions](#coordinates-and-conventions).

Two simplifications are made, and both are deliberate:

- **The velocity is evaluated at the undistorted point** `(x, y)` and scaled
  by the distorted time `t`; the effect of the distortion on the velocity, and
  of the rolling-shutter displacement on `t`, are second order (a few pixels of
  displacement times a percent of distortion).
- **Translation during readout is not modelled.** The image motion a camera
  translation causes depends on each point's depth, so it is not a per-image
  2D field and cannot be a per-image intrinsic. The measured scene was at 0.15
  to 0.5 m, where translation during readout is largest relative to the scene;
  the model fits it there only as far as a rotation can stand in for it.

With `e_i = 0` and `ω_i = 0` each step is the identity and the map is the
existing one. The zero test is exact, as the zero-spline identity's is, so an
image with both fields zero projects, unprojects and differentiates bit for bit
as it does today.

### Why the mean must be held

Every new per-image field is held at zero mean over the camera's images (the
images the round solves), and a prior acts only on each image's deviation from
that mean. Holding the mean is required:

- **The radial correction.** A camera's defined spline contains the whole span of
  `Δδ_i`: the camera's clamped knot vector is a refinement of the
  two-coefficient one, so every `e_0·B0 + e_1·B1` is exactly an `N`-coefficient
  spline in the camera's basis (by knot insertion, a fixed `N×2` matrix `K`
  per camera, `K = I` when `N = 2`), with the same end value and slope and
  therefore the same linear tail. With the shared spline released and the mean
  free, `c + K·ē` is all the data see, and moving `ē` against `c` is a
  direction no observation constrains.
- **The rates.** The mean rate over the camera's images is a near-gauge: a
  common vertical stretch of every image is close to stretching the scene
  vertically, for an orbit capture. With the mean free, the measured mean `ωx`
  grew from −7·10⁻⁴ to −1.7·10⁻³ as the rate prior widened, with no change in
  held-out error (signs in the scratch solver's convention, see
  [Measured results](#measured-results); in this draft's frame that `ωx` is
  positive).

This is the same structure the per-image focal and principal point already have
("Per-image values only" and "Re-centring after every round" in
[bundle-adjustment.md](../core/geometry/bundle-adjustment.md#the-per-image-release)),
except that there the shared value is held during the round and takes the mean
after it. For the radial correction the shared spline itself may be released,
so holding it is not available; the mean of `e` is pinned instead (see
[The bundle adjustment](#core-the-bundle-adjustment)). The rates have no shared
counterpart and are pinned at their mean.

### Invertibility

The shared spline's monotonicity rule ([sfmtool-camera-models.md §
The monotonicity invariant](../formats/sfmtool-camera-models.md#the-monotonicity-invariant))
applies to the image's whole radial correction `δ + Δδ_i`:

```
1 + δ'(ρ) + Δδ_i'(ρ) > 0        for every ρ ∈ [0, ρ_max], and on the tail
```

Because `δ + Δδ_i` is the camera-basis spline with coefficients `c + K·e_i`,
this is `bspline_is_monotone(c + K·e_i, ρ_max, ρ_max)` unchanged, per image.

The rolling-shutter step `(x_d, y_d) → (x_r, y_r)` is invertible where its
Jacobian `I + ∂(t·ṗ)/∂(x_d, y_d)` is close to the identity, which it is for rates of the
measured size (the rates are about 10⁻³ and the image's normalized half-height
about 0.5). A writer keeps it injective over the image's field.

### Unprojection

A pixel is turned into a ray by inverting the three steps. Step 3 is affine:
`(x_r, y_r) = ((u − cx_i)/f_i, (cy_i − v)/f_i)`. For steps 1 and 2 together, a
fixed-point iteration over the rolling-shutter displacement, in `.sfmr`
coordinates, wraps the existing radial inverse (`recover_radial_bspline` in
[kernels/sfmtool_fisheye.rs](../../crates/sfmtool-core/src/camera/distortion/kernels/sfmtool_fisheye.rs),
called by `undistort_sfmtool_pinhole`), run on the image's spline `c + K·e_i`.
The radial inverse is rotationally symmetric, so it reads `.sfmr` coordinates
as they are:

```
(x_d, y_d)₀ = (x_r, y_r)
repeat:
    (x, y)ₖ       = radial_inverse((x_d, y_d)ₖ)              # existing, exact
    (x_d, y_d)ₖ₊₁ = (x_r, y_r) − t((x_d, y_d)ₖ) · ṗ((x, y)ₖ)
until the step is under 1e-12 in normalized units, or a cap of iterations
```

The iteration contracts by about the size of `∂(t·ṗ)/∂(x_d, y_d)`, of order
`|ω|·(1 + ρ²)`, about 10⁻³ on the measured capture, so each pass gains about
three digits and three or four passes reach rounding. A non-converged
iteration is reported the way the radial inverse reports one, and the
step guard below keeps a solve from producing rates where it would happen.
With `ω_i = 0` the loop is skipped and the existing inverse runs alone, bit for
bit.

## Proposed interface

This section names the existing pieces the proposal extends and what changes in
each. Names marked as proposed are open to the glossary
([GLOSSARY.md](../GLOSSARY.md)) when the draft is filed.

### Format: `SFMTOOL_PINHOLE` and `.sfmr`

[sfmtool-camera-models.md](../formats/sfmtool-camera-models.md) gains a section
defining the per-image radial correction in the model's own terms: the closed
form of `B0` and `B1` above, its linear tail, the rule that it is added to the
camera's `δ` for that image, and the monotonicity rule on the sum. A format spec
stands alone, so the closed form is written out there rather than referred to
the kernels.

[sfmr-file-format.md](../formats/sfmr-file-format.md) moves to version 14 and
gains, under
[Per-image intrinsic adjustments](../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13):

| Entry | Proposed content |
|---|---|
| `images/radial_adjustments.{N}.2.float64.zst` | Optional. Per image, `[e_0, e_1]`, its radial correction coefficients, dimensionless. Present exactly when some camera lists `"radial"`. |
| `images/rolling_shutter_rates_xyz.{N}.3.float64.zst` | Optional. Per image, `[ωx, ωy, ωz]`, the angular velocity in the `.sfmr` camera frame (+X right, +Y up, looking down −Z), in radians per normalized image unit of readout time, defined by the formulas in [The map for image `i`](#the-map-for-image-i). Present exactly when some camera lists `"rolling_shutter"`. |
| `cameras/metadata.json` `per_image_adjustments` | May also name `"radial"` (only on an `SFMTOOL_PINHOLE` camera) and `"rolling_shutter"`, in the order `"focal"`, `"principal_point"`, `"radial"`, `"rolling_shutter"`. |
| `cameras/metadata.json` `rolling_shutter_scanlines` | Optional string, `"ROWS"` or `"COLUMNS"`: whether the sensor reads one row at a time (`t = y_d`) or one column at a time (`t = x_d`). Required on a camera that lists `"rolling_shutter"`; allowed on any other camera, so a known setting can be recorded before any rates are solved. Part of `cameras_xxh128`. |

The rules are those of the existing columns: a column is present exactly when
some camera lists its name; a row of an image whose camera does not list it holds
exactly `+0.0`; every value is finite. The two columns fold into
`images_xxh128` in their lexicographic slots, after `quaternions_wxyz`. Their
units are in
[Units of the per-image adjustments](#units-of-the-per-image-adjustments).

#### Units of the per-image adjustments

Each per-image adjustment is stored in a unit that means the same at any
resolution, so a camera resize (which scales the camera's focal and principal
point with the image) leaves every stored value unchanged. The four units are
not the same, on purpose:

| Term | Unit | Why |
|---|---|---|
| focal scale `s` (existing) | dimensionless: the image's focal is its camera's times `1 + s`; the bundle adjustment solves it as `ln(1 + s)` | a scale of the focal is a ratio, so it is the same at any resolution and reads directly as "this image's focal is 1.3% longer"; the log makes a focal `k` times longer and one `k` times shorter equally far from the camera's |
| principal-point offset `a` (existing) | a fraction of the image's short side, `min(width, height)`, the same unit on both axes: the image's principal point is its camera's plus `a·min(width, height)` | it moves the pixel grid after the focal is applied, so it is measured across the image; the short side is one length in x and y, so one offset means the same number of pixels whichever way the lens shifts; it is not a fraction of the focal, which the solve can release, so that an offset does not change in pixels when the focal moves |
| radial correction `e` (new) | units of `ρ`, the normalized radius on the spline's domain `[0, bspline_rho_max]` | it is added to the camera's `δ`, which is in units of `ρ`, before the focal is applied; `e_1` is the correction's value at `ρ_max` |
| rolling-shutter rates `ω` (new) | radians per normalized image unit of `t`, where `t` is `y_d` or `x_d`, in the units of `(Px, Py)/(−Pz)`, the tangent of the angle off the axis | the correction acts in normalized image coordinates, before the focal and principal point; in those units `ω` is the camera's angular velocity times the readout time per unit of `t`, and the velocity formulas are exact; with `t` in short-side units the rates would mean something different for each focal, the image's own focal scale included, so changing an image's focal would change what its stored rates mean |

The units also mean that prior widths of different terms cannot be compared as
numbers. On the measured capture a rate width of 2·10⁻³ is about 2 px of
displacement at the centre of the bottom edge, while the principal point's
derived width of 0.0131 of the short side is about 20 px.

When this draft is filed, this table goes into the `.sfmr` format spec,
[sfmr-file-format.md § Per-image intrinsic adjustments](../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13),
beside its "Why a relative focal scale" and "Why a fraction of the short side,
on both axes" paragraphs, which explain the two existing units in prose today.
No standing spec has a table of these units yet; the format spec is the one
that defines the per-image adjustments.

A file of version 13 or older reads with every radial coefficient and every rate
zero and no camera carrying `rolling_shutter_scanlines`, which is the image read through its
camera exactly as before (no rates means zero corrections). A version 14 file
with none of the new entries maps back to version 13 by its version number
alone. `validate_per_image_adjustments` in
[types.rs](../../crates/sfmtool-sfmr-format/src/types.rs) takes the new names
and columns.

COLMAP export is unaffected: an `SFMTOOL_PINHOLE` camera cannot be exported to
COLMAP at all.

### Core: the adjustment and the kernels

- [`IntrinsicAdjustment`](../../crates/sfmtool-core/src/camera/intrinsics.rs)
  gains `radial: [f64; 2]` and `rolling_shutter: [f64; 3]` (proposed names),
  zero by default, and `is_zero` tests all four fields.
- `CameraIntrinsics::with_adjustment` applies the radial correction by
  replacing the camera's coefficients with `c + K·e_i` (or with `e_i` itself on a
  camera whose spline is empty), so the image's intrinsics remain an ordinary
  `SFMTOOL_PINHOLE` camera and every radial kernel, the inverse and
  `bspline_is_monotone` read it unchanged.
- The rolling shutter is not a radial map, so no `CameraModel` variant carries
  it. The proposal is a field on the image's intrinsics, applied by
  `project`, `ray_to_pixel`, `ray_to_pixel_with_jacobian`, `pixel_to_ray` and
  their batch forms in
  [projection.rs](../../crates/sfmtool-core/src/camera/distortion/projection.rs)
  after the model's `distort` and before the focal and principal point, and
  inverted as in [Unprojection](#unprojection): for example
  `rolling_shutter: Option<RollingShutter>` on `CameraIntrinsics`, with
  `RollingShutter { scanlines: RollingShutterScanlines, rates: [f64; 3] }`, set
  by `ImageSet::image_intrinsics(i)`, which knows the camera's scanlines. Where
  it sits
  is an [open question](#open-questions), because it touches every consumer of
  `CameraIntrinsics`. Whatever its place, `ImageIntrinsicsList` and every
  routine that reads an image through `image_intrinsics(i)` (triangulation, the
  bundle adjustment's re-estimation, patch rendering, image warping) gets the
  corrections without change of its own.
- The per-camera scanlines setting is stored in memory beside the camera's
  [`PerImageAdjustments`](../../crates/sfmtool-core/src/reconstruction/data/image_set.rs),
  which gains `radial`, `rolling_shutter` and
  `rolling_shutter_scanlines: Option<RollingShutterScanlines>`, an enum
  `RollingShutterScanlines { Rows, Columns }` (proposed).
- The ray Jacobian
  ([sfmtool-pinhole-kernels.md § Ray Jacobian](../core/camera/sfmtool-pinhole-kernels.md#ray-jacobian))
  becomes the derivative of the whole map. In `.sfmr` coordinates, with `D` the
  existing radial 2×2 read with the image's spline, `∇t` equal to `(0, 1)` for
  `ROWS` and `(1, 0)` for `COLUMNS` (`∂t/∂(x_d, y_d)`), and
  `J_rs = I + ṗ·∇tᵀ`,

  ```
  ∂(x_r, y_r)/∂(x, y) = J_rs · D + t · ∂ṗ/∂(x, y)
  ∂ṗ/∂(x, y) = [[ 2ωy·x − ωx·y,   −ωx·x + ωz       ],
                [ ωy·y − ωz,      −2ωx·y + ωy·x    ]]
  ```

  followed by the existing pixel step's `f_i·diag(1, −1)`. It is the derivative
  of the map as stated, not of a simplified one, because the bundle
  adjustment's damping ladder and pose refinement compare costs of the map
  itself.
- **How the new terms compose with the existing kernels.** The radial kernels
  keep their optical frame. An implementation calls them as today, with
  `distort(x, −y)`, reads their output back into `.sfmr` coordinates as
  `(x_d, y_d) = (first, −second)`, applies the rolling-shutter step there, and
  forms the pixel as `u = cx_i + f_i·x_r`, `v = cy_i − f_i·y_r`. The radial 2×2
  `D` is the same expression in either frame, since the map is rotationally
  symmetric, so `D` at `(x, y)` is the kernel's radial 2×2 evaluated at
  `(x, y)`. The two `−y` flips are the existing pixel mapping's; the new terms
  add none.

### Core: the bundle adjustment

In [bundle_adjust.rs](../../crates/sfmtool-core/src/geometry/bundle_adjust.rs):

- [`CameraRelease`](../../crates/sfmtool-core/src/geometry/bundle_adjust.rs)
  gains `image_radial: bool` and `image_rolling_shutter: bool` (proposed), and
  `bundle_adjust` two flags `opt_image_radial` and `opt_image_rolling_shutter`,
  narrowed per camera by them as the existing per-image flags are. Both are
  refused on any model but `SFMTOOL_PINHOLE`; `image_radial` also on a camera
  whose spline domain is not positive and finite.
- The kernel takes one fixed scanlines setting per camera
  (`RollingShutterScanlines`, `Rows` or `Columns`), from the caller. `UNKNOWN`
  is the reconstruction-level adjustment's business, since it compares two
  solves ([Choosing rows or columns](#choosing-rows-or-columns)).
- **Columns.** Each image's block of the reduced system grows from
  `[δθ×3, δt×3, t_i, a_x, a_y]` to fourteen wide, `… , e_i0, e_i1, ωx, ωy, ωz`,
  in a solve where some camera releases a new per-image field, with the image's
  unreleased slots pinned. At the image's intrinsics, in `.sfmr` coordinates
  and then through the existing pixel step `F = f_i·diag(1, −1)`:

  ```
  ∂(x_r, y_r)/∂e_ij = J_rs · (Bj(ρ) + Bj'(ρ_max)·(ρ − ρ_max)₊) · û
  ∂(x_r, y_r)/∂ω    = t · [[ −x·y,       1 + x²,  y ],
                           [ −(1 + y²),  x·y,    −x ]]
  ∂(u, v)/∂(·)      = F · ∂(x_r, y_r)/∂(·)
  ```

  with `û = (x, y)/ρ`, `Bj` read at `min(ρ, ρ_max)`, as the shared spline's
  column reads it, and `J_rs = I + ṗ·∇tᵀ` as in the ray Jacobian above (the
  identity without rates).
  The per-observation camera block gains the five columns in further
  instantiations rather than widening the existing ones, for the reason the
  spec already gives for the principal point's columns: the Schur elimination's
  cost grows with the square of the block width. The reduced system is dense
  and factored by LU; at 85 images its width goes from 765 to 1190, about
  3.8 times the factorization's cost.
- **The prior** extends "The prior" of
  [bundle-adjustment.md](../core/geometry/bundle-adjustment.md#the-per-image-release)
  with two more terms per camera, `‖e_i − ē‖²/σ_radial²` and
  `‖ω_i − ω̄‖²/σ_rs²`, isotropic over each field's components, in the same
  `σ_n²`-scaled units. `ImagePriorWidths` gains `radial` and `rolling_shutter`
  (proposed). How the widths are chosen is open
  ([open question](#open-questions)).
- **The mean hold.** The mean of `e` and of `ω` over the camera's solved images
  is held at the value it starts the round at (zero for a fresh release), as an
  exact linear constraint per camera and component. Proposed: a bordered
  system, one Lagrange row per constraint, solved by the dense LU the reduced
  system already uses, since LU does not need the matrix to be positive
  definite. The alternative, a stiff prior on the mean, is simpler but leaves
  the mean held only to the stiffness. After each round, any rounding residue
  of `ē` is moved into the shared spline (`c ← c + K·ē`, `e_i ← e_i − ē`, which
  changes no image's intrinsics to rounding), and of `ω̄` is left as it is.
- **The width derivation** (steps 2 to 5 of "Deriving the widths") reads each
  image's block over all its moving slots, so it extends to the new slots as
  it is.
- **Step guards.** A candidate step is rejected and re-damped when an image's
  `e_i` or `ω_i` is not finite, when `c + K·e_i` fails `bspline_is_monotone`
  over `[0, ρ_max]` (per image, through the existing `bspline_step_admissible`),
  or when the rolling-shutter step's Jacobian at the image's corners is farther
  from the identity than the unprojection's fixed-point iteration tolerates (a
  bound of 0.5 on its distance from `I` is proposed: far from any measured
  value and well inside the iteration's convergence).
- **The spline penalty** (`SplinePenalty`) is unchanged: it smooths the shared
  coefficients `c`, and a two-coefficient correction has no interior
  coefficient for it to act on.
- **Parity.** A solve that releases neither new field, on images whose new
  fields are zero, takes the instantiations it takes today and computes the
  same output bit for bit.

### Choosing rows or columns

The reconstruction-level adjustment
([bundle-adjust.md](../core/reconstruction/bundle-adjust.md),
[bundle_adjust.rs](../../crates/sfmtool-core/src/reconstruction/bundle_adjust.rs))
takes a scanlines choice per camera that releases the rates: `ROWS`, `COLUMNS`
or `UNKNOWN` (proposed `RollingShutterScanlinesChoice`). Where the caller does
not choose, it is the camera's stored `rolling_shutter_scanlines`, used as
stored, or `UNKNOWN` where the camera has none: an absent setting means the
scanlines are not known, and no setting is assumed. `UNKNOWN` is a value of the
option only, never stored. `ROWS` or `COLUMNS` runs one solve. `UNKNOWN` runs
the solve once with each and keeps the one with the lower cost, writing its
setting into the camera.

With several `UNKNOWN` cameras, their choices are made one camera at a time, in
camera index order, each by its own race of two solves. While one camera's race
runs, the `UNKNOWN` cameras already decided use the setting their race chose,
and those not yet decided have their rates held at zero, where the scanlines
setting has no effect, so no setting is assumed for them. A camera whose setting
is `ROWS` or `COLUMNS` releases its rates at that setting in every race.

The first race starts from the caller's reconstruction. Each race after it
starts from the state the previous race's winner reached: its poses, its points,
every intrinsic and rate it released, including the earlier `UNKNOWN` cameras'
rates at the settings their races chose. The earlier races' work is carried forward rather
than thrown away. Both solves of a race start from that same state, with the
raced camera's rates at zero, so the two are compared from one start as in a
single race. In the last race no released rates are held at zero, and the
winner of the last race is the adjustment's result. Starting near convergence
also makes the later races cheaper than the first, though by how much is not
measured. `k` such cameras cost `k` races; the cases with more than one are rare
enough not to justify a combinatorial search over all `2ᵏ` combinations.

A weight between rows and columns is not solved for, because the wrong choice is
a local minimum: a continuous blend `t = λ·y − (1 − λ)·x`, which is rows at
`λ = 1` and columns at `λ = 0`, released from `λ = 0` stays there
([Alternatives considered](#alternatives-considered)).

The cost compared is the robust cost the damping ladder compares (data, priors
and spline penalty), in units of `σ_n²`. Both solves start from the same state
(the rates start at zero, where the scanlines do not matter), so the first round's
trim, kept observations and `σ_n` are the same for both and their costs are
directly comparable through that round. After it each solve trims from its own
state, and the final comparison is over the observations both final rounds
kept, each at its own final state, with one pooled `σ_n`.

**Stopping the trailing solve early.** Two full solves cost twice one. The
proposal is to run them in lockstep, one LM iteration of each at a time (on two
threads, each with its own `Progress`), and to stop the trailing solve once
it cannot plausibly catch up:

- after each pair of iterations, the gap is
  `D = (C_trailing − C_leading)/σ_n²`;
- the trailing solve is stopped, through its `Progress` cancel flag (a stopped
  solve returns the state it reached), when for three consecutive iterations
  both `D > 25` and `D` exceeds what the trailing solve would remove in the rest
  of its iteration budget at its recent rate, the mean of its last three
  iterations' decrease times the iterations the schedule has left;
- the leading solve then runs to the end alone, and the report says the other
  choice was stopped and at what gap;
- if the rule never fires, the final costs decide, as two full solves would.

25 is the likelihood-ratio threshold the point-or-bearing test already uses
(`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD` in
[point_or_bearing.rs](../../crates/sfmtool-core/src/reconstruction/triangulation/point_or_bearing.rs)),
taken as a starting value, not a derived one. On the measured capture the
final costs with rows and with columns differ by about 840 such units, so the rule would fire long
before the end there. The rule assumes that the trailing solve's
per-iteration decrease does not grow again, which a Levenberg–Marquardt solve
does not promise (its damping can fall and its steps grow), and that an early
lead is not reversed while the poses are still settling. **It must be measured
on real solves before it is adopted**: the measurement records both solves'
full cost trajectories, on the measured capture from its stored state and from
a raw `sfm solve` output, and on synthetic scenes with rates planted under
rows and under columns, and reports the first iteration after which the sign of `D` never changes
and whether the rule ever stops the eventual winner. A per-iteration cost needs
to reach the comparator from the kernel; `Progress` reports phases, counts and
messages today, and whether the cost travels as a message or through a small
observer argument is part of the work.

### Bindings, viewer, CLI

- `EditedReconstruction.bundle_adjust` in
  [edited.rs](../../crates/sfmtool-py/src/reconstruction/edited.rs): each
  `releases` dict takes `"image_radial"`, `"image_rolling_shutter"` (bools) and
  `"rolling_shutter_scanlines"` (`"ROWS"`, `"COLUMNS"` or `"UNKNOWN"`; a
  missing key is the camera's stored setting, or `"UNKNOWN"` where it has
  none), with `opt_image_radial=` and `opt_image_rolling_shutter=` for every camera at once;
  `image_prior` dicts take `radial` and `rolling_shutter` widths. The report's
  `CameraAdjustment` gains the scanlines setting used, under `UNKNOWN` the costs
  with rows and with columns and whether one was stopped, and
  `ImageAdjustmentSpread` the new fields' spreads, with `ImageAdjustmentParameter`
  gaining a variant for each so `near_prior` can name them.
- The kernel binding `sfmtool.geometry.bundle_adjust`
  ([bundle_adjust.rs](../../crates/sfmtool-py/src/geometry/bundle_adjust.rs))
  keeps `image_adjustments` as `(n_img, 3)` and takes the new fields as two more
  arrays, `image_radial_adjustments` `(n_img, 2)` and
  `image_rolling_shutter_rates` `(n_img, 3)`, with a `rolling_shutter_scanlines`
  sequence per camera (`"ROWS"` or `"COLUMNS"`, since the kernel runs one
  solve with a known setting and the race is the reconstruction-level
  adjustment's; for a camera that releases no rates and whose images' rates are
  zero the value is ignored), so existing callers are not broken by a wider
  array.
- `read_sfmr` / `write_sfmr` dicts carry `radial_adjustments`,
  `rolling_shutter_rates_xyz` and each camera's `rolling_shutter_scanlines`.
- The viewer's Bundle Adjust dialog
  ([bundle_adjust_prompt.rs](../../crates/sfm-explorer/src/bundle_adjust_prompt.rs),
  [gui/edits/bundle-adjust.md](../gui/edits/bundle-adjust.md)) gains, per
  camera, **Release radial distortion per image** and **Release rolling shutter
  per image** beside the existing **Release focal length per image** and
  **Release principal point per image**, and a **Scanlines** choice (`Rows`,
  `Columns`, `Unknown`) shown while the rolling shutter is ticked, which opens
  on the camera's stored setting, or on `Unknown` where it has none. Both
  checkboxes are greyed with the
  reason on a camera that is not `SFMTOOL_PINHOLE`, through its `CameraGate`.
  The MCP `bundle_adjust` tool
  ([edit.rs](../../crates/sfm-explorer/src/mcp/tools/catalog/edit.rs)) gains
  `release_image_radial`, `release_image_rolling_shutter` and
  `rolling_shutter_scanlines` (`"ROWS"`, `"COLUMNS"` or `"UNKNOWN"`),
  call-wide and per camera, as its existing release keys are; omitted, it is
  the camera's stored setting, or `"UNKNOWN"` where it has none.
- `sfm xform --bundle-adjust` has no key for the existing per-image releases
  or the principal point, and gains none here
  ([open question](#open-questions)).

## Measured results

These numbers come from experiments made for this draft with an independent
bundle adjustment written with scipy in a scratch directory, **not with the
kernel**. They show that the terms predict held-out points better; they do not
show the physical cause.

**Data.** `dino_dog_toy_ground_truth_candidate_v012.sfmr` in the maintainer's
local `dino_dog_toy_ws` (not checked in): 85 handheld Pixel 7 Pro images at
2040×1536, 1176 points, 23731 observations; one `SFMTOOL_PINHOLE` camera with 8
spline coefficients, `bspline_rho_max` 0.84, focal 1447.8 px, with per-image
focal and principal point released. EXIF focus distance 0.15 to 0.48 m,
exposures mostly 1/120 s, every EXIF orientation 1.

**Method.** Plain least squares keeping every observation (no trim, no robust
loss), with the per-image focal and principal-point priors at the kernel's
derived widths for v012 (0.0134 and 0.0131). 5-fold cross-validation by point:
each fold's held-out points are re-triangulated through the trained cameras,
the new terms included, and their reprojection RMS over all observations is the
score. Solves were capped at 150 to 400 evaluations; 100 against 400 changed the
held-out RMS by under 0.0003 px. The scratch solver wrote the rolling-shutter
term in the kernels' optical frame (y down, `t` growing down the image for
rows). That is the same family of maps as the one above, with
`ω = (−ωx, ωy, ωz)` of its rates for rows and `ω = (ωx, −ωy, −ωz)` for columns,
so every fit and score here holds unchanged; only a quoted rate's sign depends on the convention.

**Held-out RMS, px:**

| Terms added to v012 | Held-out RMS |
|---|---|
| none (baseline, as v012) | 0.386 |
| shear alone | 0.344 |
| rolling shutter, rows, alone | 0.331 (better than shear in all 5 folds) |
| 2-coefficient correction (σ 2·10⁻³) + rolling shutter, rows | **0.2304** |
| 2-coefficient correction + rolling shutter, columns | 0.2374 |
| 2-coefficient correction + shear | 0.2384 |
| 2-coefficient correction + shear + rolling shutter, rows | 0.2254 |

Rows beat both columns and shear in all 5 folds. Adding shear on top of
rolling shutter with rows is better in all 5 folds too, but the two terms inflate each
other: 1.55 px and 1.9 px at the image edge together, against 0.76 px and
1.34 px apart, so the proposal does not combine them.

**One coefficient against two.** At a tighter prior (σ 5·10⁻⁴), a
one-parameter quadratic correction, δ ∝ ρ² (a multiple of `(ρ/ρ_max)²` added to
`δ`), with rolling shutter gives 0.257 against 0.255 for the two-coefficient
correction with rolling shutter, the two-coefficient form better in all 5
folds. The `k1`-type one-parameter form, δ ∝ ρ³, was not tested in a re-solve;
both δ ∝ ρ² and δ ∝ ρ³ lie in the span of the two-coefficient basis.

**Prior widths.** For the radial correction, σ is quoted as `δ` at the domain
end (`e_1`, since `B1(ρ_max) = 1`):

| Radial σ | Held-out RMS | Note |
|---|---|---|
| 2.5·10⁻⁴ | 0.281 | rate mean free |
| 5·10⁻⁴ | 0.255 | rate mean free |
| 1·10⁻³ | 0.238 | rate mean free |
| 2·10⁻³ | 0.230 | rate mean free |
| 2·10⁻³ | 0.2384 | mean held, with shear |
| 4·10⁻³ | 0.2374 | mean held, with shear |
| 8·10⁻³ | 0.2374 | mean held, with shear |

| Rate σ | 1·10⁻³ | 2·10⁻³ | 4·10⁻³ | 8·10⁻³ |
|---|---|---|---|---|
| Held-out RMS | 0.2553 | 0.2549 | 0.2548 | 0.2548 |

The held-out error is flat for the radial correction at σ ≥ 4·10⁻³ and for the
rates at σ ≥ 2·10⁻³: past those widths the priors do no work beyond the mean
hold.

**By distance from the centre** (held-out RMS, px):

| ρ | [0, 0.3) | [0.3, 0.5) | [0.5, 0.6) | [0.6, 0.7) | [0.7, 0.75) | [0.75, 0.83) |
|---|---|---|---|---|---|---|
| baseline | 0.330 | 0.395 | 0.436 | 0.549 | 0.670 | 0.612 |
| 2-coefficient + rolling shutter, rows | 0.196 | 0.242 | 0.255 | 0.307 | 0.362 | 0.576 |

The outermost band has about 85 observations.

**Fitted sizes.**

- The rolling-shutter displacement at the centre of the bottom edge has a
  median of 1.3 px and a maximum of 5.5 px; its lag-1 autocorrelation along the
  capture sequence is about 0.4, so it changes from one photograph to the next
  rather than following a slow drift.
- The per-image radial correction is several pixels at ρ = 0.75 and
  correlates at −0.98 with the EXIF focus distance in dioptres, which is what
  focus-dependent distortion would give.
- The poses rotate by a median of 0.15° to 0.24° when the terms are added.
- Earlier residual fits, with the poses and points held, found no missing
  shared lens term: decentering, thin prism, aspect ratio and skew were all at
  the noise level. Only per-image fields remained.

**Caveats.** One capture from one phone. A scratch solver with plain least
squares rather than the kernel's soft-L1 and trim schedule, and capped
evaluations. Cross-validation shows that the terms predict held-out points
better, not what physically causes the residual they remove.

## Alternatives considered

**A per-image shear, with no scanlines setting.** A traceless symmetric linear
map per image, `Δx = a·x + s·y`, `Δy = s·x − a·y`, two parameters. To first
order, rolling shutter with rows or with columns is a linear map of the image
plane; once the part a rotation of the pose and a change of focal absorb is
removed, what remains is exactly this shear, so it does not need to know which.
It reaches 0.2384 with the radial correction against 0.2304 for rolling shutter
with rows, and rolling shutter beats it in every fold. It is not the proposal because the full rolling-shutter field fits
better and has a physical reading; it could serve as a fallback for a camera
whose scanlines are unknown and not worth two solves
([open question](#open-questions)).

**A continuous readout blend.** `t = λ·y − (1 − λ)·x` with one `λ ∈ [0, 1]` per
camera and shared rates. The scratch solver wrote it as `λ·y_o + (1 − λ)·x_o` in
the optical frame, which is the negative of this `t` and so the same fit with
every rate negated. In this draft's frame, `λ·y + (1 − λ)·x` runs along the other
diagonal and is not what was measured between `λ = 0` and `λ = 1`. The solve's cost (px², priors included) at fixed `λ`:

| λ | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| cost | 1190.3 | 1196.1 | 1187.1 | 1150.5 | 1134.0 |

`λ = 1` is rows and `λ = 0` is columns. The cost is not convex in `λ`: the
wrong choice (`λ = 0`, columns, here) is a local minimum. Released from 0, `λ`
stayed at 0; released from 0.5 and from 1, it ended at 1, rows. Rows and
columns differ by about 56 px², about 840 in χ² units at the 0.258 px noise
level. This is why `UNKNOWN` compares two solves rather than
solving for a weight.

**A focus-tied radial correction.** One shared two-coefficient shape per camera,
scaled for each image by its centred EXIF focus distance in dioptres (two
parameters per camera rather than two per image). With shear it gives 0.2467,
and 0.706 in the outermost band, worse than free per-image coefficients there.
With a small free per-image remainder (σ 5·10⁻⁴) and shear it gives 0.2361,
better than free coefficients with shear in all 5 folds. It needs a focus value
for every image: EXIF `SubjectDistance`, or the solved per-image focal scale as
a stand-in, which correlates at 0.96 with the dioptres. It is left for a later
amendment ([open question](#open-questions)).

## Testing

When built, the tests cover at least:

- **Parity.** A camera with no new field listed, and images with zero new
  fields, project, unproject and differentiate bit for bit as today, and a
  solve that releases neither new field gives today's output bit for bit.
- **Knot insertion.** `c + K·e` evaluates to `δ + Δδ` to rounding on and past
  `[0, ρ_max]`, for several `N` including 2.
- **Round trip.** `pixel_to_ray` inverts `ray_to_pixel` to 1e-9 px over the
  image with planted radial corrections and rates of ten times the measured
  size, with rows and with columns.
- **Jacobian.** The analytic ray Jacobian and the five new columns agree with
  central differences.
- **Recovery.** On the per-image tests' synthetic scenes, planted radial
  corrections and rates under rows and under columns are recovered within their noise, the mean
  stays at its start value, and `UNKNOWN` picks the planted scanlines.
- **Format.** Version 14 round-trips the columns and keys bit for bit; a reader
  rejects each broken rule; version 13 files read with zero corrections.

## Non-goals

- Translation during readout: it depends on each point's depth and is not a
  per-image field.
- A per-image correction of any other part of the lens (tangential, thin prism,
  aspect, skew): the residual fits found none needed.
- Rolling shutter on any camera model other than `SFMTOOL_PINHOLE` in this
  draft; the fisheye is an open question.

## Open questions

- **The early-stopping rule for `UNKNOWN`.** The rule in
  [Choosing rows or columns](#choosing-rows-or-columns), with its threshold of 25 and three
  consecutive iterations, is a proposal to be measured on real solves before it
  is adopted. Whether lockstep should be per LM iteration or per schedule round
  (simpler, and needs no per-iteration cost from the kernel, but stops later) is
  part of the same measurement.
- **Several `UNKNOWN` cameras.** The races in
  [Choosing rows or columns](#choosing-rows-or-columns) run one camera at a time,
  each starting from the previous race's winner, with the undecided cameras'
  rates held at zero. An earlier race is then decided without the later
  cameras' rolling shutter, which those cameras' poses absorb in part, and its
  choice is kept once the later cameras' rates are released. Whether releasing
  them could have changed an earlier choice, and whether a second pass over the
  cameras is worth its cost, is untested: every measured capture has one camera.
- **Prior widths.** Whether the new widths are derived per camera by the
  existing empirical Bayes scheme (`ImagePriorOptions` `loose` and `floor`)
  or fixed. Derived, the loose widths cannot be the existing 0.1 (a radial
  correction of 0.1 at the domain end is about 145 px at this focal);
  `loose` of 10⁻² and `floor` of 10⁻⁴ for both new fields would be a starting
  point. Fixed, defaults inside the flat region of the measurements (4·10⁻³ for
  the radial correction, 4·10⁻³ for the rates) would do. The measurements say
  the widths do little work past those values, which argues for fixed defaults;
  the existing release derives its widths because a fixed lens and a
  stabilised one differ, which argues for deriving. Also open: whether one
  isotropic width over `e_0` and `e_1` is right when the two coefficients mean
  different things.
- **How the mean is held.** An exact constraint in a bordered system (proposed)
  or a stiff prior on the mean.
- **Where the rolling shutter sits in memory.** A field on `CameraIntrinsics`,
  read by every projection routine, or something narrower. A field reaches
  every consumer at once; it also means every `CameraIntrinsics` comparison and
  serialisation path has to know about it.
- **`SFMTOOL_FISHEYE`.** Whether it gets the same two corrections. The radial
  correction carries over unchanged in `θ`; the rolling-shutter velocity above
  is the tangent-plane form and would need a ray-space form for a fisheye whose
  field reaches 90°.
- **The shear fallback.** Whether to offer the shear, which needs no scanlines
  setting, as a release, for a camera whose scanlines are unknown and a caller
  that will not pay for two solves.
- **The focus-tied correction.** Whether to add it as a later amendment, and
  where its per-image focus value comes from (EXIF `SubjectDistance`, or the
  solved focal scale).
- **Reporting and display.** How the new fields are summarised by the
  reconstruction-level report and by the CLI (`sfm inspect`, `sfm analyze`),
  for example the rolling-shutter displacement at the edge in pixels and the
  radial correction at `ρ_max` in pixels; how the viewer shows them in an
  image's intrinsics detail
  ([intrinsics/mod.rs](../../crates/sfm-explorer/src/image_detail/intrinsics/mod.rs),
  [gui/camera-intrinsics.md](../gui/camera-intrinsics.md)); and whether its
  distortion overlay draws the image's own field.
- **The format version.** Version 14 as above, with older files reading as zero
  corrections; and whether `rolling_shutter_scanlines` without rates is worth
  allowing, or should come only with the rates.
- **Images whose pixel axes are not the sensor's.** The scanlines setting is per
  camera, in terms of the stored image's rows and columns. A camera whose images
  are stored in both orientations (portrait photographs rotated by their EXIF
  orientation) would need the setting per image, or one camera per orientation.
  Every measured image has orientation 1.
- **The CLI.** Whether `sfm xform --bundle-adjust` gains keys for the per-image
  releases and the scanlines setting.
- **Naming.** `radial` against `distortion` for the per-image correction
  (the shared release is `CameraRelease::distortion`), and the names of the new
  types and columns, to settle in the glossary when the draft is filed. The
  scanlines term is already settled there
  ([GLOSSARY.md](../GLOSSARY.md#camera-intrinsics)).
- **Breadth of the evidence.** One capture from one phone. The other checked-in
  datasets would need promotion to `SFMTOOL_PINHOLE` before they could test the
  terms, and a second phone capture would show whether the scanlines setting
  and the size of the rates generalise.
