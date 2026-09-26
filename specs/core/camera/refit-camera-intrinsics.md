# Refitting camera intrinsics to another camera model

A camera model is the function that maps a ray leaving the camera to a pixel.
Two models from different families can describe the same lens, and a lens
calibrated in one family sometimes has to be moved to another: a fisheye first
solved with a COLMAP polynomial, whose inverse fails a little past 90° off the
axis, is better described by the spline model `SFMTOOL_FISHEYE`, which is
defined out to 180°. This spec describes the fit that does the move. Given a
source camera and a target model, it samples rays over the angles where the
source is trusted, projects each with the source, and chooses the target's
parameters so the target puts every ray as close as it can to the same pixel. It
reports how close that is, and names what the target cannot represent.

The fit knows only the lens. Switching the cameras of a reconstruction, which
adds the observations and compares their reprojection errors before and after,
is [`../reconstruction/switch-camera-model.md`](../reconstruction/switch-camera-model.md).
The spline models themselves are specified in
[`../../formats/sfmtool-camera-models.md`](../../formats/sfmtool-camera-models.md).

## Rust API

The fit lives in [refit_intrinsics.rs](../../../crates/sfmtool-core/src/camera/refit_intrinsics.rs),
as `sfmtool_core::camera::refit_intrinsics`, and is bound as `CameraIntrinsics.refit`.

```rust
pub enum RefitTarget {
    SfmtoolFisheye { coeff_count: usize },
    SfmtoolPinhole { coeff_count: usize },
    EquidistantFisheye,
    Colmap(&'static str),
}

impl RefitTarget {
    pub fn from_name(model: &str, coeff_count: Option<usize>) -> Result<Self, RefitError>;
    pub fn model_name(&self) -> &'static str;
    pub fn coeff_count(&self) -> Option<usize>;
    pub fn is_perspective(&self) -> bool;
}

pub struct RefitOptions {
    pub theta_fit_deg: Option<f64>,     // None: trusted bound, else far image corner
    pub spline_domain_deg: Option<f64>, // None: far image corner
}

pub enum ThetaFitSource { TrustedBound, Observations, ImageCorner, Given, SplineDomain }

pub enum DroppedTerm {
    FocalAspect { fy_over_fx: f64 },
    Parameter { name: String, value: f64 },
}

pub struct ModelExtent {
    pub edge_deg: f64,
    pub corner_deg: f64,
    pub source_trusted_deg: Option<f64>,
    pub source_fold_deg: Option<f64>,
}

/// Where a spline fit held the slope at its floor to stay invertible.
pub struct MonotoneConstraint {
    pub active: bool,
    pub active_angles: usize,          // grid angles held at the floor
    pub range_deg: Option<[f64; 2]>,   // the smallest and largest of them
}

pub const MIN_SLOPE: f64 = 0.05;       // the floor, as r′(d) ≥ MIN_SLOPE·f

pub struct CameraIntrinsicsRefit {
    pub camera: CameraIntrinsics,
    pub theta_fit_deg: f64,
    pub theta_fit_source: ThetaFitSource,
    pub spline_domain_deg: Option<f64>,
    pub rms_px: f64,
    pub max_px: f64,
    pub radial_rms_px: f64,
    pub dropped: Vec<DroppedTerm>,
    pub extent: ModelExtent,
    pub monotone_constraint: MonotoneConstraint,
}

pub enum RefitError {
    UnknownTarget { model: String },
    CoeffCount { model: &'static str, count: usize },
    CoeffCountNotApplicable { model: &'static str },
    ThetaFitInvalid { theta_fit_deg: f64 },
    BeyondTrustedBound { theta_fit_deg: f64, trusted_deg: f64 },
    PerspectivePast90 { theta_fit_deg: f64 },
    ObservationsPast90 { max_theta_deg: f64 },
    SourceCannotProject { theta_deg: f64 },
    SplineDomainInvalid { spline_domain_deg: f64 },
    NotMonotone,
    TrustedBoundShort { trusted_deg: f64, theta_fit_deg: f64 },
    NotSplineSource { model: &'static str },
    Degenerate { reason: &'static str },
}

pub fn refit_camera_intrinsics(
    source: &CameraIntrinsics,
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<CameraIntrinsicsRefit, RefitError>;

/// A spline camera as the same spline model with another coefficient count
/// and, optionally, another domain end, fitted over the whole new domain.
pub fn refit_spline(
    source: &CameraIntrinsics,
    coeff_count: usize,
    spline_domain_deg: Option<f64>, // None: the source's domain end, copied exactly
) -> Result<CameraIntrinsicsRefit, RefitError>;
```

The source's trusted bound is
[`trustworthy_max_theta_deg`](../../../crates/sfmtool-core/src/camera/report.rs),
and `ModelExtent::source_fold_deg` is `forward_fold_deg` in the same module: the
angle at which a polynomial fisheye's forward map stops increasing.

### Why it is shaped this way

**The target is an enum, not a model name.** The two spline models need a
coefficient count and every other model must not have one, and the spline fits
and the polynomial fits are different solvers. `RefitTarget::from_name` is the
one place a name becomes a target, so the viewer, MCP and the CLI refuse the
same names with the same sentences.

**Angles are in degrees.** The trusted bound, the fold and every report angle
are in degrees already, and so is every place a person types one. A spline
target's domain end is also given as an incidence angle, whichever radial
coordinate the model stores it in: `SFMTOOL_PINHOLE` stores `tan θ`, and a
caller should not have to know that to say "end the spline at 70°".

**The report names what could not be represented.** A single-focal target fitted
to a lens with `fx ≠ fy` has an irreducible error that grows with the radius and
varies with the azimuth. Without the `dropped` list and the separate
`radial_rms_px`, a reader would see a large overall rms and could not tell a bad
fit from a lost aspect.

**A spline-to-spline refit is its own function.** Changing a spline camera's
coefficient count is not a move between model families, and none of the
defaults of `refit_camera_intrinsics` fit it: the domain end should stay where it
is, exactly, and the fit should cover that whole domain rather than a trusted
bound the source does not have. `refit_spline` states both, and takes only what
can change. The bundle adjustment's coefficient count
([`../reconstruction/bundle-adjust.md`](../reconstruction/bundle-adjust.md)) is
its caller.

**A spline fit is constrained to be monotone, not refused when it is not.** A
lens model must be invertible, since every keypoint's ray comes from the
inverse. The least-squares optimum under that requirement is the closest
invertible curve to the source, which is what a caller asking for the switch or
a new coefficient count wants; refusing instead left the caller to guess another
count or domain. The report's `monotone_constraint` says where the requirement
bound, so a reader can see where the fit departed from the source.

**Refusals are values.** Every refusal names the rule and the value it measured,
so a caller can print it as the one sentence a menu or a CLI needs, and a test
can match on the variant.

### Example

```rust
use sfmtool_core::camera::refit_intrinsics::{refit_camera_intrinsics, RefitOptions, RefitTarget};

let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8))?;
let refit = refit_camera_intrinsics(&source, &target, &RefitOptions::default())?;
println!(
    "f {:.2}, rms {:.3} px, radial {:.3} px over θ ≤ {:.1}°",
    refit.camera.focal_lengths().0, refit.rms_px, refit.radial_rms_px, refit.theta_fit_deg,
);
for term in &refit.dropped {
    println!("{term}"); // "fx/fy aspect 0.9978 dropped (single focal)"
}
```

## Theory

### The samples

The fit samples 96 incidence angles evenly over `(0, θ_fit]` and 64 azimuths at
each, in the canonical camera frame, where the camera looks along `−Z`: the ray
at `(θ, φ)` is `(sin θ cos φ, sin θ sin φ, −cos θ)`. Each ray is projected with
the source. A source that has no pixel for a ray inside the domain is refused
(`SourceCannotProject`), since there is nothing to fit to there.

The **principal point is copied**, not fitted. Bundle adjustment never frees it,
and a fit that moved it would move every keypoint's ray for a reason the data
did not give. The image size is copied too.

### The fit's largest angle

`θ_fit` defaults to the source's trusted bound: the angle past which a
polynomial fisheye folds or its inverse blends toward the identity ray. For a
model with no trusted bound (the perspective models, the spline models, the
exact fisheye maps), the lens-only default is the incidence angle of the far
image corner under the source, capped at 180°. The reconstruction-level switch
uses the observations' extent instead; see its spec.

A caller may give a smaller `θ_fit`, never a larger one than the trusted bound
(`BeyondTrustedBound`). A fit that followed the source past its bound would copy
the fold into the new model: on the `kerry_park` rig's first lens the
polynomial's radius flattens between 95° and 102°, and a spline fitted there
flattens with it.

### A spline source refitted as a spline

A spline model has no trusted bound: its curve is defined everywhere on its
domain `[0, d_max]` by construction, and along its linear tail past it. So when
a spline camera is refitted as the same spline model with another coefficient
count (`refit_spline`), the fit samples the **whole** new domain, `θ_fit` equal
to the domain end as an incidence angle (`ThetaFitSource::SplineDomain`), and the
result is the best least-squares description of the old curve on the new
coefficient scheme, focal included. When no new domain end is given, the
source's is copied bit for bit; `SFMTOOL_PINHOLE`'s `tan θ` is not taken through
degrees and back. A new domain end is sampled the same way whether it is shorter
or longer than the old one, since the old model states a pixel at every angle
either way.

The old and new bases are not nested (a spline of `N` coefficients has `N − 1`
knot spans on the same domain), so the refit is not exact. On the `kerry_park`
first lens as an eight-coefficient `SFMTOOL_FISHEYE` over about 150°, twelve
coefficients reproduce the curve to under 0.05 px and five to under 1 px. Like
every spline fit, the refit is constrained to be monotone (below), which matters
most here: a source with a deep dip, where its slope comes close to zero, is
monotone, but a fit with more coefficients rings through the dip and can cross
below zero without the constraint.

### Spline targets: one linear solve

For `SFMTOOL_FISHEYE` the model's pixel is

```
(u, v) = (cx, cy) + f·(d + Σ cᵢ·Bᵢ(d))·û
```

with `d = θ`, `û` the unit image direction of the ray, and `Bᵢ` the fixed basis
on `[0, d_max]` (carried along its end tangent past `d_max`, exactly as the
model's linear tail is). That is linear in `x = (f, f·c₀, …, f·c_{N−1})`, so one
least-squares solve over the samples' two pixel coordinates gives the focal and
the coefficients together, with no starting point and no iteration.
`SFMTOOL_PINHOLE` is the same fit with `d = tan θ`. `EQUIDISTANT_FISHEYE` is the
fit with no coefficients (`N = 0`), which the zero-spline identity makes the
same model.

`û` and `d` come from projecting the ray through the family's base model at
focal 1 and principal point at the origin, so the sign conventions of the
optical frame are the model code's and are not restated here.

**The domain end is chosen once.** `d_max` is placed where the format spec asks:
at the far image corner, estimated as the corner's pixel radius over the
source's focal on the axis (`√(fx·fy)`), which is where every model family
agrees with its base. For the `kerry_park` fisheyes (480 × 480, f ≈ 129.6) that
is about 150°. A caller may give it instead (`spline_domain_deg`).

**The span past `θ_fit` is regularized.** Coefficients whose support lies past
the last sample have no data, and the solve would be rank-deficient. So one row
per interior coefficient penalizes the second difference `f·(c_{i−1} − 2cᵢ +
c_{i+1})`, and `δ` continues past `θ_fit` as the smoothest curve that meets the
data. The penalty's weight is small (see [Parameters](#parameters)): on a source
the target represents exactly it moves the fit by well under a thousandth of a
pixel.

### Spline targets: the monotonicity constraint

A monotone spline is the model's construction invariant: the radial map has to
be strictly increasing over its domain and along its linear tail, or the camera
has no inverse. The unconstrained least-squares solution does not always have
one. The continuation of a lens that is flattening at `θ_fit` can turn over past
it (the `kerry_park` first lens does, with `d_max` at 110°), and a fit with more
coefficients than its source rings through a deep dip in the source's slope.

So the fit is solved under the constraint that the slope stays above a floor.
The radial map's slope is linear in the unknowns,

```
r′(d) = f·(1 + Σ cᵢ·Bᵢ′(d)) = x₀ + Σ xᵢ₊₁·Bᵢ′(d),
```

and the fit requires `r′(d_k) ≥ MIN_SLOPE · f` at a grid of `d_k` over
`[0, d_max]`. `MIN_SLOPE` is relative to the focal, so each constraint is the
homogeneous linear inequality `(1 − MIN_SLOPE)·x₀ + Σ xᵢ₊₁·Bᵢ′(d_k) ≥ 0`, and
the problem stays a linear least-squares problem with linear inequality
constraints. The last grid point is `d_max`, whose slope is the end tangent the
linear tail carries, so the constraint covers the tail too.

- **The grid is the monotonicity check's own.** It has 64 points per knot span,
  at the same angles `bspline_is_monotone` samples when its sufficient test
  fails, so a fit that holds the floor at every grid point passes that check by
  construction.
- **The floor is positive.** `MIN_SLOPE = 0.05` rather than zero, so the fitted
  camera is invertible with a margin: at the floor one pixel of radius is at
  most twenty times the angle it is at the centre, which keeps the inverse's
  Newton steps bounded. It is below what a real lens reaches inside its field:
  the orthographic projection `r = f·sin θ`, the most compressive of the
  classical fisheye projections, falls to it only at 87°, and the equisolid
  projection only at 174°. It binds on a fold or a near-flat dip, not on an
  optical design.
- **An unconstrained fit is unchanged.** The unconstrained solution is computed
  first and kept, bit for bit, when it already holds the floor at every grid
  point. Only a fit that breaks it goes through the constrained solve.
- **The result is still checked.** The fitted spline is checked with
  `bspline_is_monotone` as before. With the constraint in place a failure means
  the constrained solve itself went wrong, and is refused (`NotMonotone`) with a
  sentence that says so.

The report's `monotone_constraint` gives whether the constraint bound, at how
many grid angles, and the smallest and largest of those angles as incidence
angles. Over that range the fit is not the source's curve but the closest
invertible one, and the pixel error there is part of `rms_px` and `max_px`.

On the `tk107` capture's first camera after an eight-coefficient bundle
adjustment (`SFMTOOL_FISHEYE`, f ≈ 129.53, domain 150.1°, a dip where
`1 + δ′` comes close to zero), refits to twelve and sixteen coefficients break
the floor without the constraint. With it, twelve coefficients reproduce the old curve to
rms 0.49 px and max 1.52 px over the whole domain, the floor binding at one grid
angle, 113.2°, the bottom of the dip; sixteen give rms 0.18 px and max 0.56 px,
binding at 113.3°.

### Polynomial targets: a small nonlinear fit

For the COLMAP models the pixel is not linear in the parameters, so the samples
are fitted by Levenberg–Marquardt over every parameter but the principal point.
The start copies the source's parameters by name, the focal translated between
one value and two (the mean of `fx` and `fy`, or the single focal into both).
When that start cannot project every sample, which happens when a coefficient
with the same name means something different in the two models, the distortion
starts from zero instead. The Jacobian is a central difference of the target
model's own projection.

A target that contains the source's model starts at zero error and stays there,
so `SIMPLE_RADIAL` to `RADIAL` gives the source's focal and `k1` with `k2 = 0`.

A fitted polynomial fisheye is checked the way the viewer checks one: its own
trusted bound must reach `θ_fit`, or the fit is refused (`TrustedBoundShort`).
Past 90° of distorted angle the fisheye polynomials' inverse blends toward the
identity ray, so a polynomial fitted to a near-equidistant lens out to 120° is
trusted only to about 90°.

The polynomial fit is not constrained to be monotone. Its parameters enter the
pixel nonlinearly, so the slope floor is not a linear constraint there, and the
polynomial models already state how far they can be inverted as their trusted
bound, which the check above holds the fit to.

### Perspective targets

A perspective model (`SFMTOOL_PINHOLE`, `PINHOLE`, `SIMPLE_PINHOLE`,
`SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `FULL_OPENCV`) has no pixel for a ray at 90°
or more, so it is refused when `θ_fit` reaches 90° (`PerspectivePast90`).

### What the report measures

- **`rms_px` and `max_px`**: the pixel distance between the source's and the
  fitted camera's pixel over all samples.
- **`radial_rms_px`**: at each sampled angle, the difference between the two
  cameras' radii averaged over the azimuths, then the rms over the angles. This
  is the error of the radial profile alone. A lost focal aspect shows in
  `rms_px` and not here.
- **`dropped`**: a focal aspect `fy / fx` when the source has two focals and the
  target one, and every non-zero tangential or thin-prism parameter the target
  has no parameter of the same name for. Radial terms are never listed: a
  target with a different radial parameterization represents them by fitting.
- **`extent`**: the largest incidence angle the fitted camera gives the midpoints
  of the image edges and the image corners, and the source's trusted bound and
  fold.
- **`monotone_constraint`**: for a spline target, whether the slope floor bound,
  at how many grid angles, and the range of incidence angles they cover. Always
  inactive for the other targets.

On the `kerry_park` first lens (`OPENCV_FISHEYE`, `fx` 129.718, `fy` 129.430),
an eight-coefficient `SFMTOOL_FISHEYE` fitted over its trusted bound of about
84.5° has f ≈ 129.52, radial rms ≈ 0.013 px, rms ≈ 0.13 px and max ≈ 0.29 px,
and drops the aspect 0.9978.

## Implementation notes

**The spline solve is an SVD of the design matrix**, not of the normal
equations. The normal equations square the condition number, and an exact
source (an `EQUIDISTANT_FISHEYE` fitted as `SFMTOOL_FISHEYE`) then comes back
with coefficients of 1e-11 instead of zero to rounding. The matrix is
`2·96·64 + (N − 2)` rows by `N + 1` columns, a few milliseconds to decompose.

**The penalty weight scales with the data.** Each penalty row's squared weight
is `SMOOTHING · rows / N`, so the penalty's share of the solve does not change
with the sample or coefficient count.

**The constrained solve reuses the SVD.** With the design matrix `A = U·S·Vᵀ`,
the substitution `z = S·Vᵀ·x − Uᵀ·b` turns `min ‖A·x − b‖²` subject to
`G·x ≥ 0` into the least-distance problem `min ‖z‖` subject to `E·z ≥ f`, with
`E = G·V·S⁻¹` and `f = −G·x_unconstrained`. Its dual is a non-negative least
squares problem over one variable per constraint, solved by the Lawson–Hanson
active-set algorithm (Lawson & Hanson, *Solving Least Squares Problems*,
chapter 23), and the constraints with a positive dual variable are the ones the
report counts as bound. Each row of `E` is scaled to unit length first, which
changes neither the feasible set nor the solution. The problem is at most 33
unknowns and `64·(N − 1) + 1` constraints, and it is in
[constrained_lsq.rs](../../../crates/sfmtool-core/src/camera/refit_intrinsics/constrained_lsq.rs);
no dependency of the workspace carries a quadratic-programming or NNLS routine.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `coeff_count` | `DEFAULT_COEFF_COUNT`, `8` | Spline coefficients for a spline target named without a count; `0` or `2..=MAX_COEFF_COUNT` (`32`). |
| `theta_fit_deg` | trusted bound, else far image corner | The largest incidence angle sampled. |
| `spline_domain_deg` | far image corner, `r_corner / √(fx·fy)` | Where a spline target's domain ends. |
| `THETA_SAMPLES` | `96` | Incidence angles sampled over `(0, θ_fit]`. |
| `AZIMUTHS` | `64` | Azimuths sampled at each angle. |
| `SMOOTHING` | `1e-6` | Weight of the second-difference penalty, per data row and per coefficient. |
| `MIN_SLOPE` | `0.05` | The spline fit's slope floor, `r′(d) ≥ MIN_SLOPE·f`. |
| `SLOPE_GRID_PER_SPAN` | `64` | Constraint grid points per knot span, the monotonicity check's density. |
| `LM_MAX_ITERS` | `200` | Iteration budget of the polynomial fit. |

All are constants in [refit_intrinsics.rs](../../../crates/sfmtool-core/src/camera/refit_intrinsics.rs).

## Python bindings

`CameraIntrinsics.refit(target, *, coeff_count=None, theta_fit_deg=None,
spline_domain_deg=None)` returns `(CameraIntrinsics, report)`. The report is a
dict: `model`, `theta_fit_deg`, `theta_fit_source` (`"trusted_bound"`,
`"observations"`, `"image_corner"` or `"given"`), `spline_domain_deg` (`None` for
a non-spline target), `rms_px`, `max_px`, `radial_rms_px`, `dropped` (one
sentence per term) and `extent` (`edge_deg`, `corner_deg`, `source_trusted_deg`,
`source_fold_deg`) and `monotone_constraint` (`active`, `active_angles`, and
`range_deg`, a `(from, to)` pair of incidence angles or `None`). A refusal is a
`ValueError` carrying the error's sentence.

```python
camera, report = source.refit("SFMTOOL_FISHEYE", coeff_count=8)
print(report["rms_px"], report["radial_rms_px"], report["dropped"])
```

## Testing

[refit_intrinsics/tests.rs](../../../crates/sfmtool-core/src/camera/refit_intrinsics/tests.rs):

- `EQUIDISTANT_FISHEYE` to `SFMTOOL_FISHEYE` fits to zero error with zero
  coefficients, and `EQUIDISTANT_FISHEYE` as a target is the spline fit with
  none.
- `SIMPLE_RADIAL` to `RADIAL` reproduces the copy; a COLMAP target equal to its
  source comes back unchanged.
- A synthetic `SFMTOOL_FISHEYE` fitted back on its own domain recovers its
  focal and coefficients.
- The `kerry_park` first lens: the default `θ_fit` is its trusted bound, short
  of its fold at about 101.6°; the fitted spline is monotone, its domain ends
  near 150°, and the aspect is reported as dropped.
- A polynomial fitted to a spline over 80°.
- `refit_spline`: the `kerry_park` first lens as an eight-coefficient spline
  refitted to twelve and to five coefficients over its whole domain, the domain
  end copied bit for bit and the curve reproduced within 0.05 px and 1 px; the
  same count coming back as itself; an `SFMTOOL_PINHOLE` keeping its `ρ_max`
  exactly; refusals for a source without a spline, counts of 1 and 33, and a
  domain end past 180°.
- The monotonicity constraint: a folded source fitted to a spline whose
  smallest slope is the floor; the `tk107` first camera refitted to twelve and
  sixteen coefficients, monotone with the constraint reported active; a
  nearly-flat source lifted to the floor in its flat stretch only; and an
  unconstrained refit equal bit for bit to the plain least-squares solve.
  `constrained_lsq.rs` tests the solver on hand-solved problems and an
  infeasible one.
- Refusals: a perspective target past 90°; a fit past the trusted bound; a
  polynomial trusted short of the fit; target names and coefficient counts.

`forward_fold_deg` is tested in
[report/tests.rs](../../../crates/sfmtool-core/src/camera/report/tests.rs). The
bindings are tested in
`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`.

## Non-goals

- Fitting the principal point. Nothing in the toolkit frees it, and the fit
  keeps that.
- An aspect for the spline models. They have one focal, and a lens whose `fx`
  and `fy` really differ loses the difference; the report says so.
- Choosing `d_max` from the image circle of a circular fisheye. The default is
  the far image corner, because that is the model's own reach: every pixel of
  the frame has a ray under it. On a circular fisheye it spends part of the
  knot span on black pixels past the circle, and trimming the domain to the
  circle is a choice a person makes: wherever the domain is edited, the
  outermost keypoint is shown beside it
  ([`../reconstruction/outermost-keypoint.md`](../reconstruction/outermost-keypoint.md)),
  the detected one where the images' `.sift` files can be read and otherwise
  the observed one, labelled as such, with a button that sets the domain to its
  angle. `refit_spline` makes the change on a camera already switched.

## Open questions

- **The regularization weight** past `θ_fit`, and whether a smooth continuation
  or the linear tail starting at `θ_fit` serves a later bundle adjustment better.

