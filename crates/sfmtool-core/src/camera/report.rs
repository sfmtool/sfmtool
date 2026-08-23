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

use crate::camera::CameraIntrinsics;

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
/// Both fields are the projection of the **same** ray, so their difference is
/// the lens's contribution and nothing else. An overlay draws the arrow from
/// [`Self::reference`] to [`Self::pixel`] — "the lens moved this ray *here*
/// from *there*", the direction a rectification would undo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistortionSample {
    /// Where the model actually projects the ray.
    pub pixel: [f64; 2],
    /// Where the family's ideal map projects it.
    pub reference: [f64; 2],
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
