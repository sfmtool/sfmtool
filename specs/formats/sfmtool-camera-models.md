# The sfmtool camera models

**Status:** Both models are implemented. The variants and their serialization
live in `crates/sfmtool-core/src/camera/intrinsics.rs`
(`CameraModel::SfmtoolFisheye`, `CameraModel::SfmtoolPinhole`, sharing the
`get_bspline` parameter reader) with the spline basis in
`crates/sfmtool-core/src/camera/distortion/bspline.rs`; serialization tests in
`camera/intrinsics/tests.rs`,
`tests/rust_bindings/test_sfmtool_fisheye_rust_bindings.py` and
`tests/rust_bindings/test_sfmtool_pinhole_rust_bindings.py`. The projection,
inverse and Jacobian kernels are specified per model in
[../core/sfmtool-fisheye-kernels.md](../core/sfmtool-fisheye-kernels.md) and
[../core/sfmtool-pinhole-kernels.md](../core/sfmtool-pinhole-kernels.md).

**Beta:** the parameterization may still change: the basis, the knot
layout, and the parameter names. A `.sfmr` file carrying these models may
therefore need to be regenerated across releases.

## Summary

Two camera models that use a spline for radially symmetric distortion. Each
pairs a distortion-free base model with a monotonic spline correction whose
number of knots is selectable: `SFMTOOL_FISHEYE` on an equidistant base,
`SFMTOOL_PINHOLE` on a pinhole base. Both bases have square pixels and are
parametrized with one focal length. The spline is constrained to be strictly
increasing so that the distortion is invertible.

## Radially symmetric distortion

The distortion is a monotonic spline correction: a **cubic B-spline** `δ` on
an **open-uniform (clamped) knot vector over `[0, d_max]`**, constrained to
keep the distorted radius strictly increasing. Both models share this one
correction machinery: the correction is added to the base model's radial
coordinate before the focal length scales it to pixels,

```
r(d) = f·(d + δ(d)),        δ(d) = Σᵢ cᵢ·Bᵢ(d)
```

where

- `d` is the radial coordinate the base model projects a ray to: the
  incidence angle `θ`, in radians, for the fisheye; the normalized
  image-plane radius `ρ` for the pinhole;
- `δ(d)` is the spline correction, in the units of `d`, not pixels;
- `cᵢ` are the `N` spline coefficients and `Bᵢ` the basis functions of the
  cubic B-spline;
- `f` is the focal length, a scalar: pixels per radian for the fisheye,
  pixels for the pinhole;
- `r(d)` is the pixel radius: the distance from the principal point to the
  projected pixel, along the unit image direction `û`.

Every quantity in the formula is a scalar, and `·` is plain multiplication.

The coefficients are **dimensionless**: units of the radial coordinate, not
pixels. The distorted coordinate `(d + δ(d))·û` therefore never reads `f`, and
`f` is a pure multiplier of it: the same property `EQUIDISTANT_FISHEYE` and
`SIMPLE_RADIAL_FISHEYE` have. That is what makes `∂u/∂f = (u − cx)/f` exact
everywhere in the field (the bundle adjustment's focal column), and what makes
a resolution change a scaling of `f` alone.

### The spline basis and the center-anchored gauge

`d_max`, the end of the spline domain, is the model's domain-end parameter.
For `N` coefficients the full clamped basis has `m = N + 2` functions over the
knot vector of `m + 4` knots:
four at `0`, `m − 4` uniform interior knots, four at `d_max`, giving `m − 3`
knot spans. The **first two** functions of that basis are omitted (their
coefficients are pinned to zero), and coefficient `i` is full-basis function
`i + 2`.

The interior knots are evenly spaced, so every knot span has width
`d_max/(N − 1)` and the interior basis functions are one symmetric bump
translated span by span: identical shapes with evenly spaced peaks. The
repeated end knots reshape the three functions nearest each boundary,
shortening their support and pulling their peaks toward the end; the last
function rises to exactly `1` at `d_max`, so `δ(d_max) = c_{N−1}`.

Omitting exactly that pair pins

```
δ(0) = 0        δ'(0) = 0
```

which is the **center-anchored gauge**: the distorted radius `d + δ(d)` has
slope exactly `1` at the axis, so `r'(0) = f` and `f` is exactly the focal
length the base model would carry, the equidistant focal for the fisheye and
the pinhole focal for the pinhole. The spline can only express how the lens
departs from the base model *away* from the axis. Without the gauge `f` and
the spline would both scale the map near the axis and the pair would be
unidentifiable.

Consequences of the basis that the rest of the models rest on:

- **Local support.** At any `d` at most four basis functions are non-zero, so
  a coefficient derivative has at most four non-zero entries at any
  observation.
- **Minimum length.** A cubic basis needs four functions, so a defined spline
  needs at least two coefficients. A shorter spline (the empty one included)
  evaluates as the identity `δ ≡ 0`, as does a `d_max` that is not positive
  and finite.
- **Held-constant tail.** Beyond `d_max` the correction is held at `δ(d_max)`
  with `δ'(d) = 0`, so the radial distortion map continues linearly,
  `r(d) = f·(d + δ(d_max))`, with slope `f`.

`d_max` places the spline's resolution: the `N` coefficients are spent
uniformly on `[0, d_max]`, and beyond it the map carries only the held
constant. Its intended placement is the radial extent of the imaged field,
the largest `d` any sensor pixel reaches: the incidence angle at the far
corner for the fisheye, the normalized image-plane radius at the far corner
for the pinhole. A smaller `d_max` leaves the outer field without shape
correction; a larger one widens every knot span over the field and leaves the
outer coefficients without observation support. The placement is not
validated: the field's extent in `d` moves with the other parameters during
fitting, so `d_max` is chosen once and held.

### The monotonicity invariant

`d + δ(d)` is required to be strictly increasing:

```
1 + δ'(d) > 0        for every d ∈ [0, d_max]
```

Beyond `d_max` the slope is exactly `1`, so the spline's own domain is the
entire risk region. The invariant is what makes the radial distortion map
injective and therefore exactly invertible. It is a **construction**
invariant: enforced where a spline is produced, and relied on where one is
read. A spline that violates it has left the map's increasing branch wherever
`d + δ(d) ≤ 0` at a positive `d`; projection and derivative are both undefined
there and return nothing. Deserialization is not one of the enforcement sites:
it validates the parameter list and admits whatever coefficients the file
carries. A file holding a violating spline therefore loads, projects on the
fold-gated domain, and inverts to a root of the folded map rather than to the
radius that produced the pixel. The decision procedure and the enforcement
sites are
specified in
[../core/sfmtool-fisheye-kernels.md](../core/sfmtool-fisheye-kernels.md) and
[../core/sfmtool-pinhole-kernels.md](../core/sfmtool-pinhole-kernels.md).

### The zero-spline identity

A spline shorter than two coefficients, the empty one included, or one whose
coefficients are all **exactly** `0.0`, is the identity correction. The test
is exact, not an epsilon: the same convention `SIMPLE_RADIAL_FISHEYE` uses at
`k1 == 0`.

A `d_max` that is not positive and finite is likewise the identity
correction, whatever the coefficients.

With the identity spline each model is exactly its base model, and the
implementation reproduces the base model's projection, unprojection and
derivative **bit for bit**. That is the promotion contract: a base-model
camera rewritten into its sfmtool model with a zero spline of any admissible
length and any positive `d_max` projects, unprojects and differentiates to the
same bits, so a reconstruction promoted before a spline is fitted moves
nothing. The contract covers exactly those three maps; pixel-radius sizing
helpers agree to rounding, not bitwise.

## Parameters

Each model's parameter list is a fixed-length head followed by the spline
coefficients:

```
focal_length, principal_point_x, principal_point_y,
<domain end>, bspline_coeff_count,
bspline_c0 … bspline_c{N−1}
```

- `focal_length`: the focal length `f`, one value for both axes. Pixels per
  radian for `SFMTOOL_FISHEYE`, pixels for `SFMTOOL_PINHOLE`.
- `principal_point_x`, `principal_point_y`: the principal point `(cx, cy)`,
  in pixels.
- The domain end `d_max`, named per model: `bspline_theta_max` for
  `SFMTOOL_FISHEYE`, in radians of incidence angle; `bspline_rho_max` for
  `SFMTOOL_PINHOLE`, in normalized image-plane radius. The correction is held
  constant beyond it.
- `bspline_coeff_count`: the number of spline coefficients `N`, a non-negative
  integer. The parameter count varies with the spline, so the count is itself
  a parameter: the five-parameter head has fixed length, and reading it yields
  the full parameter count, `5 + N`, against which a parameter list is
  validated.
- `bspline_c0 … bspline_c{N−1}`: the spline coefficients, dimensionless.

## `SFMTOOL_FISHEYE`

With the camera-frame ray taken to the optical frame (`S = diag(1, −1, −1)` off
the canonical convention), `ρ = √(rx² + ry²)`, unit image direction
`û = (rx, ry)/ρ` and incidence angle `θ = atan2(ρ, rz) ∈ [0, π]`, the model
projects

```
(u, v) = (cx, cy) + f·(θ + δ(θ))·û
```

There is no perspective divide and no in-front guard anywhere in the map: `θ`
comes from `atan2`, so the periphery past 90°, where a fisheye carries its
model information, is ordinary domain.

The zero-spline base is `EQUIDISTANT_FISHEYE`.

## `SFMTOOL_PINHOLE`

In the same optical frame, restricted to rays in front of the camera
(`rz > 0`), the normalized image-plane radius is `ρ = √(rx² + ry²)/rz`, the
tangent of the incidence angle, and the model projects

```
(u, v) = (cx, cy) + f·(ρ + δ(ρ))·û
```

The domain is the base pinhole's: the map is defined for `θ < 90°` only, and
`ρ` grows without bound toward it, which is why the held-constant tail beyond
`bspline_rho_max` matters: past the calibrated field the map continues as a
pure pinhole with an offset, monotone to the edge of the domain.

The zero-spline base is `SIMPLE_PINHOLE`, the one-focal pinhole.

## Serialization

`SfmrCamera` stores the parameters under the names listed in
[Parameters](#parameters). Because the parameter list is variable-length, these
models are registered as `custom` in the camera model registry — their
serialization is hand-written and intercepts the conversion before the
fixed-arity table is reached, which is what keeps the validation below out of a
generated lookup. See
[../core/camera-model-registry.md](../core/camera-model-registry.md).

The `.sfmr` container is pass-through for camera parameters
(the format stores whatever `parameters` map the model writes), so
the variable-length spline needs no format-side registration. See
[sfmr-file-format.md](sfmr-file-format.md).

Three cameras as they appear in the `cameras/metadata.json.zst` array of a
`.sfmr` file: an `SFMTOOL_FISHEYE` with an eight-coefficient spline, an
`SFMTOOL_PINHOLE` with the two-coefficient minimum, and a freshly promoted
camera whose spline is still empty.

```json
{
  "model": "SFMTOOL_FISHEYE",
  "width": 2880,
  "height": 2880,
  "parameters": {
    "focal_length": 1035.2,
    "principal_point_x": 1440.0,
    "principal_point_y": 1440.0,
    "bspline_theta_max": 2.0,
    "bspline_coeff_count": 8.0,
    "bspline_c0": -0.00041,
    "bspline_c1": -0.0018,
    "bspline_c2": -0.0064,
    "bspline_c3": -0.0135,
    "bspline_c4": -0.0247,
    "bspline_c5": -0.0395,
    "bspline_c6": -0.0563,
    "bspline_c7": -0.0721
  }
}
```

```json
{
  "model": "SFMTOOL_PINHOLE",
  "width": 1920,
  "height": 1080,
  "parameters": {
    "focal_length": 1210.4,
    "principal_point_x": 960.0,
    "principal_point_y": 540.0,
    "bspline_rho_max": 0.9,
    "bspline_coeff_count": 2.0,
    "bspline_c0": 0.0031,
    "bspline_c1": 0.0118
  }
}
```

```json
{
  "model": "SFMTOOL_FISHEYE",
  "width": 480,
  "height": 480,
  "parameters": {
    "focal_length": 130.0,
    "principal_point_x": 240.0,
    "principal_point_y": 240.0,
    "bspline_theta_max": 1.6,
    "bspline_coeff_count": 0.0
  }
}
```

The third camera carries no `bspline_c*` keys and, by the zero-spline
identity, is exactly the `EQUIDISTANT_FISHEYE` camera with focal `130.0`.

Reading back, `bspline_coeff_count` declares the spline's length: it must be
`0` or at least `2` (a defined spline needs at least two coefficients), every
coefficient index below it must be present (`bspline_c0..bspline_c{N−1}`), no
`bspline_c*` key at or beyond it may be present, and the domain end must be
positive and finite. An absent `bspline_coeff_count`, domain end or
coefficient below the declared length is `MissingParameter`; a
`bspline_coeff_count` that is not a finite non-negative integer or is exactly
`1`, a domain end that is not positive and finite, or a coefficient key at or
beyond the declared length, is `InvalidParameter`. An empty spline round-trips
as `bspline_coeff_count = 0` with no coefficient keys.

## Testing requirements

Per model:

- **Serialization**: an `N`-coefficient camera round-trips through
  `SfmrCamera`; an empty spline round-trips as `bspline_coeff_count = 0`; a missing
  coefficient below the declared length, a missing domain end and a missing
  `bspline_coeff_count` are `MissingParameter`; a non-integer, negative,
  non-finite or exactly-one `bspline_coeff_count`, a domain end that is not
  positive and finite, and a coefficient key at or beyond the declared length
  are `InvalidParameter`.
- **COLMAP rejection** on both export paths, with the model name in the error.
- **Python surface** (`tests/rust_bindings/`): construction from a parameter
  dict (including the serialization rejections above), pycolmap export
  rejected, and the model's deliberate absence from `CAMERA_MODEL_NAMES`.
