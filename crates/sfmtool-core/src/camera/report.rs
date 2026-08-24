// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a camera's intrinsics *mean*: the angles its image subtends, the shape
//! of its radial projection, how far its lens displaces a pixel from where an
//! undistorted lens would have put it, and the 35 mm-equivalent focal length.
//!
//! These are the derived quantities a viewer's intrinsics panel and image
//! overlay draw (`specs/gui/gui-camera-intrinsics.md` § "`camera::report` — the
//! derived quantities"). They are pure arithmetic on a [`CameraIntrinsics`],
//! so they live here rather than in the GUI, where they can be unit-tested
//! without a window.
//!
//! # The ideal map
//!
//! "Distortion" is a displacement from somewhere, and that somewhere is the
//! **ideal map** of the model's family:
//!
//! | Family | Ideal map |
//! |---|---|
//! | Perspective | `r = f·tan θ` — a pinhole |
//! | Fisheye | `r = f·θ` — an equidistant fisheye |
//! | Equirectangular | itself; the model has no distortion parameters |
//!
//! The ideal map carries the camera's **own** `fx`, `fy`, `cx` and `cy`. That
//! is the whole point: what it measures is then pure lens distortion, with
//! neither the focal length, nor a non-square pixel aspect, nor the offset of
//! the principal point from the image centre leaking into the displacement. A
//! model whose [`CameraIntrinsics::has_distortion`] is `false` **is** its own
//! ideal map, so its residual is zero to the last bits of a pixel coordinate
//! (under `1e-12` px) rather than merely small: the difference is rounding in
//! the two paths' multiply order, not a lens.
//!
//! # Frame conventions
//!
//! Rays are in the **canonical camera frame: −Z forward, +Y up, +X right**, the
//! frame [`CameraIntrinsics::pixel_to_ray`] and
//! [`CameraIntrinsics::ray_to_pixel`] speak (see [`super::distortion`]).
//! Forward is `(0, 0, −1)`; a ray `(0, sin ε, −cos ε)` with `ε > 0` looks
//! **upward** and projects **above** the principal point, and
//! `(sin α, 0, −cos α)` with `α > 0` projects to its **right**.
//!
//! Pixel coordinates are continuous, with `(0, 0)` the top-left *corner* of the
//! image and `(0.5, 0.5)` the centre of the top-left pixel. The image centre is
//! `(w/2, h/2)`, which is generally **not** the principal point.
//!
//! # Where the numbers stop meaning anything
//!
//! Every quantity here is defined over the whole image rectangle, and for a
//! circular fisheye that rectangle's corners are outside the lens's image
//! circle — a region no calibration constrained and where a distortion
//! polynomial is free to fold. [`trustworthy_max_theta_deg`] is how a consumer
//! finds out where that starts, so a panel or a plot can bound its own domain
//! rather than presenting an extrapolation as a measurement. See that
//! function's docs for which model families are bounded and why.

use crate::camera::distortion::FISHEYE_BLEND_START_RAD;
use crate::camera::{CameraIntrinsics, CameraModel};

#[cfg(test)]
mod tests;

/// Diagonal of the 36 × 24 mm still-photography frame, in millimetres — the
/// reference "35 mm" format every equivalent focal length is quoted against.
///
/// `√(36² + 24²)`, which is the 43.267 the spec rounds it to.
const FRAME_35MM_DIAGONAL_MM: f64 = 43.266_615_305_567_87;

/// Angles, in degrees, subtended by an image's edges and corners.
///
/// Measured between rays through the image *boundary*, not through the centres
/// of the boundary pixels: the left edge is `u = 0` and the right edge is
/// `u = w`, so a pinhole with a centred principal point gives the closed form
/// `2·atan(w / 2fx)` and an equidistant fisheye with `f = w/π` gives exactly
/// 180°.
///
/// # Why the angles are swept, not subtracted
///
/// Each of the three spans is the sum of two hops — boundary to image centre,
/// image centre to opposite boundary — rather than the single angle between the
/// two boundary rays. The two agree below 180° and only there: the angle
/// between two directions saturates at 180° and folds back, so a 200° fisheye
/// measured end to end reports 160°, and the two edges of a full
/// equirectangular panorama, being *the same ray*, report 0°. Both of those are
/// wrong in the direction that matters most — the wide lenses are exactly the
/// ones a user is asking this question about.
///
/// The sum is exact whenever the three rays are coplanar, which they are for a
/// perspective or equidistant model with a centred principal point. An
/// off-centre principal point puts them on a shallow cone instead and the sum
/// runs a little over; that is a far smaller error than the fold it replaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldOfView {
    /// Mid-left to mid-right, swept through the image centre: `(0, h/2)` to
    /// `(w/2, h/2)` to `(w, h/2)`.
    pub horizontal: f64,
    /// Mid-top to mid-bottom: `(w/2, 0)` to `(w/2, h)`.
    pub vertical: f64,
    /// Corner to opposite corner: `(0, 0)` to `(w, h)`.
    pub diagonal: f64,
    /// The largest incidence angle `θ` over the four corners — the angle off
    /// the optical axis, which is what answers "is this fisheye really 180°?".
    ///
    /// Note this is a *half*-angle and so is roughly half of
    /// [`Self::diagonal`], and only roughly: the two differ whenever the
    /// principal point is off centre, since then no corner is opposite
    /// another through the optical axis.
    ///
    /// For an equirectangular panorama the four corners are the two poles, so
    /// this is 90° whatever the panorama covers — the one span for which the
    /// corners are not where the extreme angles are.
    pub max_off_axis: f64,
}

/// One sample of a model's radial map along a given azimuth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialSample {
    /// Incidence angle off the optical axis, in degrees.
    pub theta_deg: f64,
    /// `|project(ray(θ, φ)) − principal point|`, in pixels.
    pub radius_px: f64,
    /// The same distance under the family's ideal map.
    pub reference_px: f64,
}

/// The displacement of one pixel from where the ideal map would place it.
///
/// [`Self::pixel`] and [`Self::reference`] are the projection of the **same**
/// ray, so their difference is the lens's contribution and nothing else. An
/// overlay draws the arrow from [`Self::reference`] to [`Self::pixel`] — "the
/// lens moved this ray *here* from *there*", the direction a rectification
/// would undo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistortionSample {
    /// Where the model actually projects the ray.
    pub pixel: [f64; 2],
    /// Where the family's ideal map projects it.
    pub reference: [f64; 2],
    /// Incidence angle of the sampled ray, in degrees off the optical axis.
    ///
    /// The angle of `pixel_to_ray(node)` — the direction the grid node looks —
    /// so a consumer can drop the samples that fall outside
    /// [`trustworthy_max_theta_deg`] instead of reporting a fold in a
    /// polynomial as if it were a lens. Without it the field's own maximum is
    /// unfilterable: the source pixel is recoverable from the sample's index,
    /// but the angle it looks at is not recoverable from anything.
    pub theta_deg: f64,
}

/// The angles the image subtends, or `None` for a camera with no image to
/// subtend them (zero width or height).
///
/// Four corner rays and five span rays (the three spans share the image
/// centre), so it is cheap enough to compute per selection and dear enough to
/// be worth caching per camera.
pub fn field_of_view(cam: &CameraIntrinsics) -> Option<FieldOfView> {
    if cam.width == 0 || cam.height == 0 {
        return None;
    }
    let w = f64::from(cam.width);
    let h = f64::from(cam.height);
    let ray = |u: f64, v: f64| cam.pixel_to_ray(u, v);

    // Each span is swept in two hops through the image centre, which is the
    // midpoint of all three of them — see the type's docs for why the single
    // end-to-end angle will not do.
    let centre = ray(w / 2.0, h / 2.0);
    let sweep = |from: (f64, f64), to: (f64, f64)| {
        angle_between_deg(ray(from.0, from.1), centre) + angle_between_deg(centre, ray(to.0, to.1))
    };

    let max_off_axis = [[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]]
        .into_iter()
        .map(|[u, v]| angle_between_deg(ray(u, v), FORWARD))
        .fold(0.0_f64, f64::max);

    Some(FieldOfView {
        horizontal: sweep((0.0, h / 2.0), (w, h / 2.0)),
        vertical: sweep((w / 2.0, 0.0), (w / 2.0, h)),
        diagonal: sweep((0.0, 0.0), (w, h)),
        max_off_axis,
    })
}

/// The model's radial map along one azimuth: `samples` incidence angles evenly
/// spaced over `[0°, max_off_axis]`, each with the radius the model puts it at
/// and the radius the family's ideal map would.
///
/// `azimuth_deg` rotates the sampled ray about the optical axis in the
/// canonical frame — `0°` is `+X` (right), `90°` is `+Y` (up) — so a ray at
/// `(θ, φ)` is `(sin θ·cos φ, sin θ·sin φ, −cos θ)`. Azimuth matters only for a
/// model that is not radially symmetric: `fx ≠ fy`, or live tangential or
/// thin-prism terms. Sweeping it is how decentring distortion becomes visible
/// at all, since a single azimuth hides it completely.
///
/// The upper end is [`FieldOfView::max_off_axis`], the largest angle the image
/// actually contains; a plot wanting headroom beyond the data adds it to its
/// axis, not here. Angles the model cannot project — beyond a perspective
/// model's 90°, or past a fold in a distortion polynomial — are **dropped**, so
/// the result can be shorter than `samples` and can be empty. It is also empty
/// for a camera [`field_of_view`] returns `None` for.
pub fn radial_profile(
    cam: &CameraIntrinsics,
    azimuth_deg: f64,
    samples: usize,
) -> Vec<RadialSample> {
    let Some(fov) = field_of_view(cam) else {
        return Vec::new();
    };
    let (cx, cy) = cam.principal_point();
    let (sin_phi, cos_phi) = azimuth_deg.to_radians().sin_cos();

    (0..samples)
        .filter_map(|i| {
            // `samples == 1` is the optical axis alone; anything more is an
            // inclusive linspace, so the last sample is the corner angle
            // itself rather than one step short of it.
            let t = if samples > 1 {
                i as f64 / (samples - 1) as f64
            } else {
                0.0
            };
            let theta_deg = fov.max_off_axis * t;
            let (sin_theta, cos_theta) = theta_deg.to_radians().sin_cos();
            let ray = [sin_theta * cos_phi, sin_theta * sin_phi, -cos_theta];

            let (u, v) = cam.ray_to_pixel(ray)?;
            let (u_ref, v_ref) = reference_project(cam, ray)?;
            Some(RadialSample {
                theta_deg,
                radius_px: (u - cx).hypot(v - cy),
                reference_px: (u_ref - cx).hypot(v_ref - cy),
            })
        })
        .collect()
}

/// The lens's displacement field over a `cols × rows` grid of the image.
///
/// Node `(i, j)` is the centre of grid cell `(i, j)` —
/// `u = (i + ½)·w/cols`, `v = (j + ½)·h/rows` — and the samples come back
/// row-major, `j` outer. Cell centres rather than a boundary-to-boundary grid
/// so that every node is strictly inside the image, which matters for a fisheye
/// whose image rectangle has corners outside the lens's circle.
///
/// Each node is round-tripped: the pixel is unprojected to a ray, and both that
/// ray's actual projection and its ideal-map projection are recorded. A node
/// either projection cannot represent is **dropped**, so the result can be
/// shorter than `cols × rows`; it is empty for a degenerate camera or a zero
/// `cols` or `rows`.
///
/// Identically zero — every `pixel` equal to its `reference` to under
/// `1e-12` px — for a model whose [`CameraIntrinsics::has_distortion`] is
/// `false`. A caller should skip the field for those rather than draw
/// zero-length arrows.
pub fn distortion_field(cam: &CameraIntrinsics, cols: usize, rows: usize) -> Vec<DistortionSample> {
    if cam.width == 0 || cam.height == 0 || cols == 0 || rows == 0 {
        return Vec::new();
    }
    let w = f64::from(cam.width);
    let h = f64::from(cam.height);

    (0..rows)
        .flat_map(|j| (0..cols).map(move |i| (i, j)))
        .filter_map(|(i, j)| {
            let u = (i as f64 + 0.5) * w / cols as f64;
            let v = (j as f64 + 0.5) * h / rows as f64;
            let ray = cam.pixel_to_ray(u, v);
            let (pu, pv) = cam.ray_to_pixel(ray)?;
            let (ru, rv) = reference_project(cam, ray)?;
            Some(DistortionSample {
                pixel: [pu, pv],
                reference: [ru, rv],
                theta_deg: angle_between_deg(ray, FORWARD),
            })
        })
        .collect()
}

/// 35 mm-equivalent focal length in millimetres: `f_px · 43.267 / diagonal_px`.
///
/// `None` for every fisheye and equirectangular model, whose focal length is
/// pixels **per radian** rather than pixels — the number would be arithmetic on
/// two different units, and quoting a fisheye's "equivalent focal length" is
/// the single easiest way to make its intrinsics look absurd. `None` also for
/// a camera with no image diagonal to divide by.
///
/// Sensor-independent by construction: both focal length and diagonal are in
/// pixels, so the pixel pitch cancels and no sensor size has to be guessed. A
/// model with `fx ≠ fy` contributes the geometric mean `√(fx·fy)`, the
/// isotropic focal length with the same image scale (the same `det K`), since
/// the two axes are in different pixel units and the diagonal is in neither.
pub fn equiv_focal_length_35mm(cam: &CameraIntrinsics) -> Option<f64> {
    if cam.model.needs_ray_path() {
        return None;
    }
    let diagonal_px = f64::from(cam.width).hypot(f64::from(cam.height));
    if diagonal_px <= 0.0 {
        return None;
    }
    let (fx, fy) = cam.focal_lengths();
    Some((fx * fy).sqrt() * FRAME_35MM_DIAGONAL_MM / diagonal_px)
}

/// The incidence angle in degrees of the ray through pixel `(u, v)` — how far
/// off the optical axis that pixel looks.
///
/// The one place the frame convention is spelled, so an overlay's hover
/// readout, a plot's edge marker and this module's own corner angles cannot
/// disagree about which direction is forward. `(w/2, h/2)` is generally not
/// 0°: the angle is measured from the **optical axis**, which is where the
/// principal point is, not where the image centre is.
pub fn off_axis_angle_deg(cam: &CameraIntrinsics, u: f64, v: f64) -> f64 {
    angle_between_deg(cam.pixel_to_ray(u, v), FORWARD)
}

/// The largest incidence angle, in degrees, at which the model still describes
/// a lens rather than extrapolating — or `None` when it does so at every angle
/// the model can represent.
///
/// # Why this exists
///
/// [`field_of_view`], [`radial_profile`] and [`distortion_field`] are all
/// defined over the whole image rectangle, and a consumer that plots or
/// maximizes over that domain unfiltered will sooner or later report a number
/// that is about a polynomial rather than about a camera. On the `kerry_park`
/// rig's `OPENCV_FISHEYE` — a circular fisheye in a square 480 × 480 frame —
/// the image corners sit 150° off-axis, outside the lens's image circle, and
/// the `k1..k4` polynomial folds long before that: its forward map takes
/// θ = 132° to a *smaller* radius than θ = 60°. The displacement field's
/// maximum over the full rectangle is 273 px, which is a true statement about
/// two forward maps and a false one about the lens.
///
/// # Which models are bounded
///
/// This is a property of the *parameterization*, not of fisheyes:
///
/// - **Bounded** — the multi-coefficient polynomial fisheye models
///   [`CameraModel::OpenCVFisheye`], [`CameraModel::RadialFisheye`],
///   [`CameraModel::ThinPrismFisheye`] and
///   [`CameraModel::RadTanThinPrismFisheye`]. These are exactly the three call
///   sites of the wide-angle blend in [`super::distortion`]: past
///   [`FISHEYE_BLEND_START_RAD`] of distorted radius their own inverse stops
///   inverting the polynomial and slews toward the identity ray, on the stated
///   grounds that a high-order polynomial approaching its peak is not to be
///   trusted. Where the forward map peaks *before* that radius, the peak is
///   the bound instead: past a fold there is no inverse to have.
/// - **Unbounded** — [`CameraModel::SimpleRadialFisheye`], deliberately: with
///   one coefficient `θ_d = θ·(1 + k1·θ²)` there is nothing to distrust, and
///   its ray conversion is excluded from the blend for that reason. The two
///   spline models likewise, from the other direction: they hold `δ` constant
///   past their domain end so the radial map continues linearly, and they
///   enforce `1 + δ'> 0` as a construction invariant, so there is no peak to
///   approach at any angle. And the exact maps — the plain pinholes,
///   [`CameraModel::EquidistantFisheye`], [`CameraModel::Equirectangular`] —
///   along with the perspective polynomials, whose domain is already hard
///   bounded at the 90° their projective divide refuses.
///
/// A camera whose [`CameraIntrinsics::has_distortion`] is `false` is `None`
/// whatever its model: with every coefficient zero it **is** its family's
/// ideal map, the blend interpolates between two identical rays, and there is
/// no polynomial to fold.
///
/// # How the bound is found
///
/// By walking the camera's own forward map, not by reading its coefficients:
/// a coarse sweep in θ over eight azimuths brackets the first step at which
/// the distorted radius either reaches [`FISHEYE_BLEND_START_RAD`] or stops
/// increasing, and the bracket is then refined. Eight azimuths because the
/// thin-prism models are not radially symmetric and the bound is the *first*
/// azimuth to go, and the camera's own [`CameraIntrinsics::ray_to_pixel`]
/// because a second spelling of four different polynomials is a second thing
/// to keep in step. The sweep is a bracket, not a proof: a fold narrower than
/// `TRUST_SCAN_STEP_DEG` would be stepped over.
pub fn trustworthy_max_theta_deg(cam: &CameraIntrinsics) -> Option<f64> {
    // No live coefficients, no polynomial: the model is its own ideal map at
    // every angle, and that is true of a zeroed OPENCV_FISHEYE as much as of a
    // pinhole.
    if !cam.has_distortion() {
        return None;
    }
    // Exhaustive on purpose, with no `_` arm: a newly registered model has to
    // be classified here before it will build, rather than defaulting to
    // "trustworthy everywhere" because nobody thought about it.
    match &cam.model {
        CameraModel::Pinhole { .. }
        | CameraModel::SimplePinhole { .. }
        | CameraModel::Equirectangular { .. }
        | CameraModel::EquidistantFisheye { .. }
        | CameraModel::SimpleRadial { .. }
        | CameraModel::Radial { .. }
        | CameraModel::OpenCV { .. }
        | CameraModel::FullOpenCV { .. }
        | CameraModel::SfmtoolPinhole { .. }
        | CameraModel::SfmtoolFisheye { .. }
        | CameraModel::SimpleRadialFisheye { .. } => None,

        CameraModel::OpenCVFisheye { .. }
        | CameraModel::RadialFisheye { .. }
        | CameraModel::ThinPrismFisheye { .. }
        | CameraModel::RadTanThinPrismFisheye { .. } => Some(polynomial_fisheye_limit(cam)),
    }
}

/// Angular step of the sweep [`trustworthy_max_theta_deg`] brackets with.
const TRUST_SCAN_STEP_DEG: f64 = 0.5;

/// Upper end of that sweep. A fisheye rectangle's corner can exceed 180°, but
/// a polynomial that has neither folded nor reached the blend radius by then
/// is not going to be plotted usefully past it either.
const TRUST_SCAN_MAX_DEG: f64 = 180.0;

/// Azimuths the sweep samples at each θ. Enough to catch the thin-prism
/// models' azimuthal asymmetry; the bound is the smallest θ at which *any*
/// azimuth goes.
const TRUST_SCAN_AZIMUTHS: usize = 8;

/// Bisection / ternary refinement steps applied to the bracketing interval.
/// `(1/2)^60` and `(2/3)^60` of half a degree are both far below what any
/// consumer can render.
const TRUST_REFINE_STEPS: usize = 60;

/// The bound for one of the four polynomial fisheye models: the first θ at
/// which the distorted radius reaches [`FISHEYE_BLEND_START_RAD`], or the
/// radius's peak if it turns over first.
fn polynomial_fisheye_limit(cam: &CameraIntrinsics) -> f64 {
    let radius = |theta_deg: f64| max_distorted_radius(cam, theta_deg);
    let steps = (TRUST_SCAN_MAX_DEG / TRUST_SCAN_STEP_DEG) as usize;
    let mut previous = 0.0_f64;
    let mut last_good = 0.0_f64;

    for i in 1..=steps {
        let theta = i as f64 * TRUST_SCAN_STEP_DEG;
        let Some(r) = radius(theta) else {
            // The model refused the ray outright; the last angle it accepted
            // is as far as anything can be said about.
            return last_good;
        };
        if r >= FISHEYE_BLEND_START_RAD {
            // Bisect for the crossing. `radius` is increasing across the
            // bracket, so the usual sign test applies.
            let (mut lo, mut hi) = (last_good, theta);
            for _ in 0..TRUST_REFINE_STEPS {
                let mid = 0.5 * (lo + hi);
                match radius(mid) {
                    Some(r) if r < FISHEYE_BLEND_START_RAD => lo = mid,
                    _ => hi = mid,
                }
            }
            return lo;
        }
        if r <= previous {
            // The map turned over inside this step. Ternary-search the bracket
            // for the peak, which is unimodal across it by construction.
            let (mut lo, mut hi) = ((last_good - TRUST_SCAN_STEP_DEG).max(0.0), theta);
            for _ in 0..TRUST_REFINE_STEPS {
                let third = (hi - lo) / 3.0;
                let (a, b) = (lo + third, hi - third);
                if radius(a).unwrap_or(0.0) < radius(b).unwrap_or(0.0) {
                    lo = a;
                } else {
                    hi = b;
                }
            }
            return 0.5 * (lo + hi);
        }
        previous = r;
        last_good = theta;
    }
    last_good
}

/// The largest **distorted** radius, in normalized image-plane units, that the
/// model puts an incidence angle of `theta_deg` at over
/// [`TRUST_SCAN_AZIMUTHS`] azimuths.
///
/// Recovered from the model's own projection rather than from its
/// coefficients: `ray_to_pixel` writes `u = fx·x_d + cx`, so `x_d` and `y_d`
/// come back exactly by undoing that. `None` for a degenerate focal length, or
/// for an angle the model will not project at all.
fn max_distorted_radius(cam: &CameraIntrinsics, theta_deg: f64) -> Option<f64> {
    let (fx, fy) = cam.focal_lengths();
    let (cx, cy) = cam.principal_point();
    if fx == 0.0 || fy == 0.0 {
        return None;
    }
    let (sin_theta, cos_theta) = theta_deg.to_radians().sin_cos();
    let mut max = 0.0_f64;
    for i in 0..TRUST_SCAN_AZIMUTHS {
        let phi = std::f64::consts::TAU * i as f64 / TRUST_SCAN_AZIMUTHS as f64;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let ray = [sin_theta * cos_phi, sin_theta * sin_phi, -cos_theta];
        let (u, v) = cam.ray_to_pixel(ray)?;
        max = max.max(((u - cx) / fx).hypot((v - cy) / fy));
    }
    Some(max)
}

/// The optical axis in the canonical camera frame.
const FORWARD: [f64; 3] = [0.0, 0.0, -1.0];

/// Angle in degrees between two direction vectors.
///
/// `atan2(|a × b|, a · b)` rather than `acos` of a normalized dot product: it
/// needs neither input to be unit-length and stays accurate at both ends of the
/// range, where `acos` loses most of its significant digits.
fn angle_between_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let cross_len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    cross_len.atan2(dot).to_degrees()
}

/// Where the camera's family ideal map — the module docs' table — places a
/// canonical-frame ray, in pixels. `None` on the same domain the model's own
/// [`CameraIntrinsics::ray_to_pixel`] refuses: a perspective ray that is not in
/// front of the camera.
///
/// Carries the camera's own `fx`, `fy`, `cx`, `cy`, so the perspective ideal is
/// a pinhole and the fisheye ideal an equidistant fisheye with **this** camera's
/// focal lengths and principal point. The fisheye ideal keeps `fx` and `fy`
/// separate even though [`crate::camera::CameraModel::EquidistantFisheye`] has
/// only one focal length: three of the fisheye models carry two, and collapsing
/// them here would fold the pixel aspect ratio into what is meant to be a
/// measurement of the lens alone.
///
/// Written as the same arithmetic as the kernels in [`super::distortion`], so
/// a model that *is* its own ideal reproduces it to the last bits of a pixel
/// coordinate. Not bit for bit in every case: the fisheye kernels do not all
/// group the `θ · direction · f` product the same way, and no single spelling
/// can match all of them at once.
fn reference_project(cam: &CameraIntrinsics, ray: [f64; 3]) -> Option<(f64, f64)> {
    // Equirectangular is its own ideal: it has no distortion parameters, so
    // there is nothing to subtract and the model itself is the reference.
    if cam.model.is_equirectangular() {
        return cam.ray_to_pixel(ray);
    }

    let (fx, fy) = cam.focal_lengths();
    let (cx, cy) = cam.principal_point();
    // Canonical (−Z forward, +Y up) → optical frame (+Z forward, y down),
    // which is the frame the pixel scaling below is written in — `v` grows
    // downward, so an upward ray lands above `cy`.
    let [rx, ry, rz] = [ray[0], -ray[1], -ray[2]];

    let (x_d, y_d) = if cam.model.is_fisheye() {
        // Equidistant: `r = f·θ`, the distorted coordinate being θ times the
        // unit 2D direction.
        let r_xy = (rx * rx + ry * ry).sqrt();
        if r_xy < 1e-15 {
            // On the axis (or at the antipode, which the model maps to the
            // principal point too).
            return Some((cx, cy));
        }
        let theta = r_xy.atan2(rz);
        (theta * rx / r_xy, theta * ry / r_xy)
    } else {
        // Pinhole: `r = f·tan θ`, i.e. the perspective divide, undistorted.
        if rz <= 0.0 {
            return None;
        }
        (rx / rz, ry / rz)
    };

    Some((fx * x_d + cx, fy * y_d + cy))
}
