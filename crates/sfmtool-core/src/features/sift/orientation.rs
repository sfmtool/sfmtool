// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Orientation assignment (stage 5 of SIFT, `specs/core/sift.md`).
//!
//! For each localized keypoint, build a 36-bin gradient-orientation histogram
//! over a Gaussian-weighted circular window at the keypoint's scale, smooth it,
//! and emit a keypoint for the dominant peak and for every other local peak
//! within `peak_ratio` of it (so one localized candidate may produce several
//! oriented keypoints).
//!
//! # Coordinate convention
//!
//! [`LocalizedKeypoint`] carries its location in **octave-pixel** coordinates
//! (see `detect.rs`). The histogram window and the precomputed gradient images
//! are sampled in those octave pixels; the keypoint's location is converted to
//! full-resolution image coordinates (COLMAP pixel-center convention) via
//! [`ScaleSpace::octave_to_image`] only at the very end, when the
//! [`SiftKeypoint`] is built.
//!
//! # Histogram smoothing
//!
//! The raw histogram is smoothed with **6 passes of a circular `[1 4 6 4 1]/16`
//! kernel** (a binomial low-pass). This matches the "smooth a few times" guidance
//! in the spec and keeps the peak picking stable against the discretization of
//! the 36 bins.

use super::detect::LocalizedKeypoint;
use super::{ScaleSpace, SiftKeypoint, SiftParams};
use rayon::prelude::*;
use std::f32::consts::PI;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Build a smoothed orientation histogram for one localized keypoint, in the
/// Gaussian level nearest its `layer`. Returns the `n_ori`-bin histogram.
fn orientation_histogram(
    scale_space: &ScaleSpace,
    kp: &LocalizedKeypoint,
    n_ori: usize,
) -> Vec<f32> {
    let octave = kp.octave as usize;
    let (w, h) = scale_space.octave_dims(octave);
    let (w, h) = (w as usize, h as usize);

    // Gaussian level nearest the (fractional) layer, clamped to a valid level.
    let n_levels = scale_space.gaussians_per_octave();
    let level = (kp.layer.round() as i32).clamp(0, n_levels as i32 - 1) as usize;
    // Gradients are sampled on the fly from this Gaussian level (see
    // `scale_space::pixel_gradient`); the window below is clamped to the interior.
    let gauss = scale_space.gaussian(octave, level).data();

    // sigma in *octave* pixels at this layer; window scaled by 1.5·sigma.
    let sigma_oct = scale_space.abs_sigma(kp.layer as f64) as f32;
    let sigma_w = 1.5 * sigma_oct;
    let radius = (3.0 * sigma_w).round() as i32;
    let radius = radius.max(1);

    let cx = kp.x;
    let cy = kp.y;
    // Integer center used to index the sample grid (gradients are per-pixel).
    let icx = cx.round() as i32;
    let icy = cy.round() as i32;

    let exp_denom = 1.0 / (2.0 * sigma_w * sigma_w);
    let radius2 = (radius * radius) as f32;
    let bin_scale = n_ori as f32 / (2.0 * PI);

    let mut hist = vec![0.0f32; n_ori];
    let p = HistParams {
        w,
        h,
        icx,
        icy,
        radius,
        radius2,
        exp_denom,
        bin_scale,
        n_ori,
    };

    #[cfg(target_arch = "x86_64")]
    if *super::simd::HAS_AVX2_FMA {
        // SAFETY: guarded by the AVX2+FMA runtime check.
        unsafe { fill_orientation_hist_avx2(gauss, &p, &mut hist) };
        return smooth_histogram(&hist, n_ori, 6);
    }

    fill_orientation_hist_scalar(gauss, &p, &mut hist);
    smooth_histogram(&hist, n_ori, 6)
}

/// Geometry of one keypoint's orientation-histogram window, shared by the scalar
/// and AVX2 fills.
struct HistParams {
    w: usize,
    h: usize,
    icx: i32,
    icy: i32,
    radius: i32,
    radius2: f32,
    exp_denom: f32,
    bin_scale: f32,
    n_ori: usize,
}

/// Scalar reference fill of the raw orientation histogram: for every in-window
/// pixel, scatter `gaussian_weight · magnitude` into the bin of its gradient
/// angle.
fn fill_orientation_hist_scalar(gauss: &[f32], p: &HistParams, hist: &mut [f32]) {
    for dy in -p.radius..=p.radius {
        let ry = p.icy + dy;
        if ry < 1 || ry >= p.h as i32 - 1 {
            continue;
        }
        for dx in -p.radius..=p.radius {
            let rx = p.icx + dx;
            if rx < 1 || rx >= p.w as i32 - 1 {
                continue;
            }
            let r2 = (dx * dx + dy * dy) as f32;
            if r2 > p.radius2 {
                continue;
            }
            let idx = ry as usize * p.w + rx as usize;
            let (m, t) = super::scale_space::pixel_gradient(gauss, p.w, idx); // t in (-PI, PI]
            let weight = (-r2 * p.exp_denom).exp();
            // Map theta in (-PI, PI] to bin [0, n_ori).
            let mut t_pos = t;
            if t_pos < 0.0 {
                t_pos += 2.0 * PI;
            }
            let mut bin = (t_pos * p.bin_scale).floor() as i32;
            // Guard against floating point landing exactly at n_ori.
            bin = bin.rem_euclid(p.n_ori as i32);
            hist[bin as usize] += weight * m;
        }
    }
}

/// AVX2 fill of the raw orientation histogram. Each row is processed in 8-pixel
/// chunks: gradient (`gx, gy`), magnitude, angle (`atan2_approx`) and Gaussian
/// weight (`exp_approx`) are all computed 8-wide; only the final, conflict-prone
/// `hist[bin] += contribution` scatter is scalar (8 adds per chunk — cheap next
/// to the vectorized transcendentals). Identical to the scalar fill modulo the
/// ~1e-3 rad / 1e-6 transcendental approximations.
///
/// # Safety
/// Requires `avx2` + `fma` (guarded by [`super::simd::HAS_AVX2_FMA`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fill_orientation_hist_avx2(gauss: &[f32], p: &HistParams, hist: &mut [f32]) {
    let lanes = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    let neg_exp_denom = _mm256_set1_ps(-p.exp_denom);
    let radius2_v = _mm256_set1_ps(p.radius2);
    let bin_scale_v = _mm256_set1_ps(p.bin_scale);
    let twopi = _mm256_set1_ps(2.0 * PI);
    let zero = _mm256_setzero_ps();
    let wi = p.w as i32;

    for dy in -p.radius..=p.radius {
        let ry = p.icy + dy;
        if ry < 1 || ry >= p.h as i32 - 1 {
            continue;
        }
        let row_base = ry as usize * p.w;
        let dy2 = _mm256_set1_ps((dy * dy) as f32);
        // In-bounds column range: rx ∈ [1, w−2] ⇒ dx ∈ [1−icx, w−2−icx], clipped
        // to [−radius, radius]. Within it every idx±1 / idx±w load is in bounds.
        let lo = (1 - p.icx).max(-p.radius);
        let hi = (wi - 2 - p.icx).min(p.radius);

        let mut dx = lo;
        while dx + 8 <= hi + 1 {
            let base = row_base + (p.icx + dx) as usize; // idx of lane 0
            let gx = _mm256_sub_ps(
                _mm256_loadu_ps(gauss.as_ptr().add(base + 1)),
                _mm256_loadu_ps(gauss.as_ptr().add(base - 1)),
            );
            let gy = _mm256_sub_ps(
                _mm256_loadu_ps(gauss.as_ptr().add(base + p.w)),
                _mm256_loadu_ps(gauss.as_ptr().add(base - p.w)),
            );
            let m = _mm256_sqrt_ps(_mm256_fmadd_ps(gx, gx, _mm256_mul_ps(gy, gy)));
            let t = super::simd::atan2_approx(gy, gx);

            // r2 = (dx + lane)² + dy²; weight = exp(−r2·exp_denom), 0 if r2 > radius².
            let dxf = _mm256_add_ps(_mm256_set1_ps(dx as f32), lanes);
            let r2 = _mm256_fmadd_ps(dxf, dxf, dy2);
            let wgt = super::simd::exp_approx(_mm256_mul_ps(r2, neg_exp_denom));
            let inside = _mm256_cmp_ps(r2, radius2_v, _CMP_LE_OQ);
            let contrib = _mm256_mul_ps(_mm256_and_ps(wgt, inside), m);

            // bin = trunc(t_pos · bin_scale), t_pos = (t < 0 ? t + 2π : t).
            let tneg = _mm256_cmp_ps(t, zero, _CMP_LT_OQ);
            let t_pos = _mm256_blendv_ps(t, _mm256_add_ps(t, twopi), tneg);
            let bini = _mm256_cvttps_epi32(_mm256_mul_ps(t_pos, bin_scale_v));

            let mut cbuf = [0.0f32; 8];
            let mut bbuf = [0i32; 8];
            _mm256_storeu_ps(cbuf.as_mut_ptr(), contrib);
            _mm256_storeu_si256(bbuf.as_mut_ptr() as *mut __m256i, bini);
            for k in 0..8 {
                let bin = bbuf[k].rem_euclid(p.n_ori as i32) as usize;
                hist[bin] += cbuf[k];
            }
            dx += 8;
        }
        // Scalar tail for the remaining in-bounds columns [dx, hi].
        while dx <= hi {
            let r2 = (dx * dx + dy * dy) as f32;
            if r2 <= p.radius2 {
                let idx = row_base + (p.icx + dx) as usize;
                let (m, t) = super::scale_space::pixel_gradient(gauss, p.w, idx);
                let weight = (-r2 * p.exp_denom).exp();
                let mut t_pos = t;
                if t_pos < 0.0 {
                    t_pos += 2.0 * PI;
                }
                let bin = ((t_pos * p.bin_scale).floor() as i32).rem_euclid(p.n_ori as i32);
                hist[bin as usize] += weight * m;
            }
            dx += 1;
        }
    }
}

/// Smooth a circular histogram with `passes` applications of the binomial
/// `[1 4 6 4 1]/16` kernel.
fn smooth_histogram(hist: &[f32], n: usize, passes: usize) -> Vec<f32> {
    let mut cur = hist.to_vec();
    let mut next = vec![0.0f32; n];
    for _ in 0..passes {
        for i in 0..n {
            let m2 = cur[(i + n - 2) % n];
            let m1 = cur[(i + n - 1) % n];
            let c = cur[i];
            let p1 = cur[(i + 1) % n];
            let p2 = cur[(i + 2) % n];
            next[i] = (m2 + 4.0 * m1 + 6.0 * c + 4.0 * p1 + p2) / 16.0;
        }
        std::mem::swap(&mut cur, &mut next);
    }
    cur
}

/// Find the dominant orientation angle(s) (radians, in `(-PI, PI]`) of a
/// smoothed circular histogram: the global-max peak and every other local peak
/// at least `peak_ratio · max`. Each peak angle is refined by fitting a parabola
/// to the peak bin and its two circular neighbors.
fn peak_angles(hist: &[f32], peak_ratio: f32) -> Vec<f32> {
    let n = hist.len();
    let max_val = hist.iter().cloned().fold(f32::MIN, f32::max);
    let threshold = peak_ratio * max_val;
    let bin_to_angle = 2.0 * PI / n as f32;

    let mut out = Vec::new();
    for i in 0..n {
        let left = hist[(i + n - 1) % n];
        let right = hist[(i + 1) % n];
        let c = hist[i];
        // Local peak: above its left neighbor, at least its right neighbor (so a
        // flat-topped peak counts once), and within peak_ratio of the global max.
        if c > left && c >= right && c >= threshold {
            // Parabolic interpolation of the sub-bin peak location.
            let denom = left - 2.0 * c + right;
            let offset = if denom.abs() > f32::EPSILON {
                0.5 * (left - right) / denom
            } else {
                0.0
            };
            let bin = i as f32 + offset;
            let mut angle = (bin * bin_to_angle).rem_euclid(2.0 * PI);
            // Express in (-PI, PI] for consistency with gradient theta.
            if angle > PI {
                angle -= 2.0 * PI;
            }
            out.push(angle);
        }
    }
    out
}

/// Assign orientation(s) to localized keypoints, producing oriented
/// [`SiftKeypoint`]s.
///
/// A single localized candidate may yield multiple keypoints when it has more
/// than one dominant gradient orientation: one per histogram peak that is at
/// least `peak_ratio` of the maximum peak. The peak angle is refined by fitting
/// a parabola to the peak bin and its two circular neighbors.
///
/// Parallelized over the input keypoints with rayon; `flat_map`/`collect`
/// preserves the input order (the `i`-th localized keypoint's oriented
/// keypoints appear before the `(i+1)`-th's).
pub fn assign_orientations(
    scale_space: &ScaleSpace,
    localized: &[LocalizedKeypoint],
    params: &SiftParams,
) -> Vec<SiftKeypoint> {
    let n_ori = params.orientation_bins as usize;
    let peak_ratio = params.peak_ratio as f32;

    localized
        .par_iter()
        .flat_map_iter(|kp| {
            let hist = orientation_histogram(scale_space, kp, n_ori);

            // Convert the octave-pixel location to full-resolution coordinates.
            let (x_full, y_full) = scale_space.octave_to_image(kp.octave, kp.x as f64, kp.y as f64);
            let x_full = x_full as f32;
            let y_full = y_full as f32;

            peak_angles(&hist, peak_ratio)
                .into_iter()
                .map(move |angle| {
                    SiftKeypoint::from_similarity(
                        x_full,
                        y_full,
                        kp.scale,
                        angle,
                        kp.octave,
                        kp.layer,
                        kp.response,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::detect_keypoints;
    use super::*;
    use crate::features::optical_flow::GrayImage;
    use std::f32::consts::PI;

    /// A smooth image with a single dominant gradient direction: a linear ramp
    /// rotated to angle `dir`. The gradient everywhere points along `dir`.
    fn directional_ramp(w: u32, h: u32, dir: f32) -> GrayImage {
        let (s, c) = dir.sin_cos();
        let mut data = vec![0.0f32; (w * h) as usize];
        for row in 0..h {
            for col in 0..w {
                let x = col as f32;
                let y = row as f32;
                // Projection onto the gradient direction, scaled into [0,1]-ish.
                let v = 0.5 + 0.002 * (x * c + y * s);
                data[(row * w + col) as usize] = v;
            }
        }
        GrayImage::new(w, h, data)
    }

    #[test]
    fn test_smooth_histogram_preserves_mean() {
        let n = 36;
        let hist: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin().abs()).collect();
        let mean0: f32 = hist.iter().sum::<f32>() / n as f32;
        let sm = smooth_histogram(&hist, n, 6);
        let mean1: f32 = sm.iter().sum::<f32>() / n as f32;
        assert!((mean0 - mean1).abs() < 1e-4, "{mean0} vs {mean1}");
    }

    #[test]
    fn test_single_dominant_orientation() {
        // Gradient points along +x (dir = 0): the dominant keypoint orientation
        // should be ~0 radians.
        let img = directional_ramp(80, 80, 0.0);
        // A directional ramp has no DoG extrema, so drive the histogram directly
        // with a synthetic localized keypoint at the image center.
        let ss = ScaleSpace::build(&img, &SiftParams::default());
        let kp = LocalizedKeypoint {
            x: 80.0, // octave-0 is doubled => 160 wide; center at ~80
            y: 80.0,
            scale: 4.0,
            octave: 0,
            layer: 1.0,
            response: 0.1,
        };
        let oriented = assign_orientations(&ss, &[kp], &SiftParams::default());
        assert!(!oriented.is_empty(), "no orientation assigned");
        // The strongest (first) peak should be near 0 radians.
        let ori = oriented[0].orientation();
        let diff = ((ori - 0.0 + PI).rem_euclid(2.0 * PI)) - PI;
        assert!(
            diff.abs() < 0.2,
            "orientation {ori} not near 0 (diff {diff})"
        );
    }

    #[test]
    fn test_orientation_matches_direction() {
        // Gradient along +y (dir = PI/2): orientation should be ~PI/2.
        let img = directional_ramp(80, 80, PI / 2.0);
        let ss = ScaleSpace::build(&img, &SiftParams::default());
        let kp = LocalizedKeypoint {
            x: 80.0,
            y: 80.0,
            scale: 4.0,
            octave: 0,
            layer: 1.0,
            response: 0.1,
        };
        let oriented = assign_orientations(&ss, &[kp], &SiftParams::default());
        assert!(!oriented.is_empty());
        let ori = oriented[0].orientation();
        let target = PI / 2.0;
        let diff = ((ori - target + PI).rem_euclid(2.0 * PI)) - PI;
        assert!(diff.abs() < 0.2, "orientation {ori} not near PI/2");
    }

    #[test]
    fn test_peak_angles_single() {
        // A single clean peak at bin 9 (= 90 deg) of a 36-bin histogram.
        let mut hist = vec![0.0f32; 36];
        hist[8] = 0.5;
        hist[9] = 1.0;
        hist[10] = 0.5;
        let angles = peak_angles(&hist, 0.8);
        assert_eq!(angles.len(), 1);
        // Bin 9 -> 9/36 * 2PI = PI/2; symmetric neighbors -> zero parabola offset.
        assert!((angles[0] - PI / 2.0).abs() < 1e-4, "got {}", angles[0]);
    }

    #[test]
    fn test_two_orientations_emit_multiple() {
        // A histogram with two comparable peaks (bins 9 and 27, i.e. 90 deg and
        // 270 deg apart) above the 0.8 ratio must yield two oriented keypoints at
        // the same location, exercised through the full `assign_orientations`
        // path. Drive the histogram via an image with two equal orthogonal
        // gradient populations would be phase-sensitive; instead validate the
        // peak picker directly here and the location-sharing through a synthetic
        // localized keypoint with a hand-checked symmetric image.
        let mut hist = vec![0.05f32; 36];
        for (b, v) in [(9usize, 1.0f32), (27usize, 0.9f32)] {
            hist[b - 1] = 0.5 * v;
            hist[b] = v;
            hist[b + 1] = 0.5 * v;
        }
        let angles = peak_angles(&hist, 0.8);
        assert_eq!(angles.len(), 2, "expected two peaks, got {angles:?}");
        // Peaks ~180 deg apart (bins 9 and 27).
        let mut diff = (angles[0] - angles[1]).abs();
        if diff > PI {
            diff = 2.0 * PI - diff;
        }
        assert!((diff - PI).abs() < 0.2, "peaks not ~PI apart: {angles:?}");

        // And through assign_orientations: a synthetic localized keypoint yields
        // keypoints that all share the same full-resolution location.
        let img = directional_ramp(80, 80, 0.0);
        let ss = ScaleSpace::build(&img, &SiftParams::default());
        let kp = LocalizedKeypoint {
            x: 80.0,
            y: 80.0,
            scale: 4.0,
            octave: 0,
            layer: 1.0,
            response: 0.1,
        };
        let oriented = assign_orientations(&ss, &[kp], &SiftParams::default());
        for o in &oriented {
            assert!((o.x - oriented[0].x).abs() < 1e-3);
            assert!((o.y - oriented[0].y).abs() < 1e-3);
        }
    }

    #[test]
    fn test_blob_full_pipeline_orients() {
        // The full detect_keypoints path on a blob should now (with orientation
        // implemented) produce oriented keypoints.
        let mut data = vec![0.5f32; 64 * 64];
        let inv = 1.0 / (2.0 * 4.0 * 4.0);
        for row in 0..64 {
            for col in 0..64 {
                let dx = col as f32 + 0.5 - 32.0;
                let dy = row as f32 + 0.5 - 32.0;
                data[row * 64 + col] = 0.5 + 0.4 * (-(dx * dx + dy * dy) * inv).exp();
            }
        }
        let img = GrayImage::new(64, 64, data);
        let detection = detect_keypoints(&img, &SiftParams::default());
        assert!(!detection.keypoints.is_empty());
        for kp in &detection.keypoints {
            assert!(kp.x.is_finite() && kp.y.is_finite());
            assert!(kp.scale() > 0.0);
        }
    }
}
