// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Scoring kernels for subpixel keypoint refinement.
//!
//! [`znorm_core`] (z-normalize), [`ecc_score`] (the ECC criterion),
//! [`view_jacobian`] (the analytic ECC Gauss-Newton normal equations), and
//! [`solve_2x2`] (the damped normal-equation solve).

use crate::patch::normal_refine::{weighted_moments_pub, Support, FLAT_NORM_SQ_EPS};

use crate::patch::keypoint_subpixel::prof;

/// z-normalize a raw core (`raw[channel * n + k]`, all channels) over the windowed
/// support into `out` (`out[channel * n + k]`), folding `√w` in so a plain dot
/// realizes the windowed inner product. A channel flat in this core (windowed
/// norm² below [`FLAT_NORM_SQ_EPS`]) is written as zeros. Mirrors
/// `keypoint_localize::znorm_core` / `normal_refine::znormalize_into`.
pub(in crate::patch::keypoint_subpixel) fn znorm_core(
    raw: &[f32],
    support: &Support,
    channels: usize,
    out: &mut [f32],
) {
    prof::ZNORM.time(|| {
        let n = support.pixels.len();
        for c in 0..channels {
            let col = &raw[c * n..][..n];
            let (s1, s2) = weighted_moments_pub(col, &support.weights);
            let mean = (s1 / support.total_weight) as f32;
            let norm_sq = s2 - s1 * (mean as f64);
            let dst = &mut out[c * n..][..n];
            if norm_sq < FLAT_NORM_SQ_EPS {
                dst.fill(0.0);
            } else {
                let inv = (1.0 / norm_sq.sqrt()) as f32;
                for (d, (&x, &sw)) in dst.iter_mut().zip(col.iter().zip(&support.sqrt_weights)) {
                    *d = sw * (x - mean) * inv;
                }
            }
        }
    })
}

/// Channel-averaged windowed ZNCC of a z-normalized core against the unit-norm
/// consensus template (both `[c * n + k]`): the ECC score `S(δ)`.
pub(in crate::patch::keypoint_subpixel) fn ecc_score(
    znorm: &[f32],
    tmpl: &[f32],
    channels: usize,
    n: usize,
) -> f64 {
    prof::ECC.time(|| {
        let mut s = 0.0;
        for c in 0..channels {
            let a = &znorm[c * n..][..n];
            let b = &tmpl[c * n..][..n];
            s += a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x as f64) * (y as f64))
                .sum::<f64>();
        }
        s / channels as f64
    })
}

/// The analytic ECC Gauss–Newton normal equations at the current offset. Given
/// the raw core `g` at `δ` and the **pre-composed** raw image Jacobian
/// `Jg = (Jg_u, Jg_v) = ∇_src I · J` (one render of the value+gradient sampler
/// composed per-pixel with the warp Jacobian — see [`render_core_with_jg`](super::render::render_core_with_jg)),
/// this composes the z-normalization derivative
/// `∂ẑ_c[k]/∂δ = (∂a/∂δ)/N − a·(a·∂a/∂δ)/N³` (with `a = √w(g − μ)`, `N = ‖a‖`)
/// and accumulates `H = Σ(∂ẑ)(∂ẑ)ᵀ` and `b = Σ(∂ẑ)·T`. Returns `(H, b)` as
/// `([Hxx, Hxy, Hyy], [bx, by])`, or `None` if every channel is flat (no
/// texture to localize on — the aperture/low-texture case the guard keeps the
/// seed for).
#[allow(clippy::too_many_arguments)]
pub(in crate::patch::keypoint_subpixel) fn view_jacobian(
    g: &[f32],
    jg_u: &[f32],
    jg_v: &[f32],
    tmpl: &[f32],
    support: &Support,
    channels: usize,
) -> Option<([f64; 3], [f64; 2])> {
    prof::JACOBIAN.time(|| view_jacobian_impl(g, jg_u, jg_v, tmpl, support, channels))
}

/// Untimed body of [`view_jacobian`] (split so the phase timer stays a single
/// wrap).
#[allow(clippy::too_many_arguments)]
fn view_jacobian_impl(
    g: &[f32],
    jg_u: &[f32],
    jg_v: &[f32],
    tmpl: &[f32],
    support: &Support,
    channels: usize,
) -> Option<([f64; 3], [f64; 2])> {
    let n = support.pixels.len();
    let mut hxx = 0.0;
    let mut hxy = 0.0;
    let mut hyy = 0.0;
    let mut bx = 0.0;
    let mut by = 0.0;
    let mut any_textured = false;

    // Per-pixel ∂ẑ/∂δ, reused per channel.
    let mut dzu = vec![0.0f64; n];
    let mut dzv = vec![0.0f64; n];
    for c in 0..channels {
        let gc = &g[c * n..][..n];
        let (s1, s2) = weighted_moments_pub(gc, &support.weights);
        let mean = s1 / support.total_weight;
        let norm_sq = s2 - s1 * mean;
        if norm_sq < FLAT_NORM_SQ_EPS {
            continue; // flat channel: zeros into ẑ, no gradient contribution
        }
        any_textured = true;
        let nrm = norm_sq.sqrt();
        let inv_n = 1.0 / nrm;
        let inv_n3 = inv_n / norm_sq;

        // a = √w (g − μ); raw image Jacobian Jg = ∂g/∂δ supplied analytically.
        // ∂a/∂δ = √w (Jg − μ'), where μ' = Σ_k w_k·Jg_k / W (∂(weighted mean)/∂δ).
        let jgu_c = &jg_u[c * n..][..n];
        let jgv_c = &jg_v[c * n..][..n];

        // ∂(weighted mean)/∂δ (the centering's mean term).
        let mut mu_du = 0.0;
        let mut mu_dv = 0.0;
        for k in 0..n {
            let w = support.weights[k];
            mu_du += w * jgu_c[k] as f64;
            mu_dv += w * jgv_c[k] as f64;
        }
        mu_du /= support.total_weight;
        mu_dv /= support.total_weight;

        // a·(∂a/∂δ) for the norm-derivative term (Σ_k a_k · ∂a_k/∂δ).
        let mut a_dau = 0.0;
        let mut a_dav = 0.0;
        for k in 0..n {
            let sw = support.sqrt_weights[k] as f64;
            let a = sw * (gc[k] as f64 - mean);
            let dau = sw * (jgu_c[k] as f64 - mu_du);
            let dav = sw * (jgv_c[k] as f64 - mu_dv);
            a_dau += a * dau;
            a_dav += a * dav;
        }

        // ∂ẑ/∂δ per pixel, then accumulate H and b against the template.
        let tc = &tmpl[c * n..][..n];
        for k in 0..n {
            let sw = support.sqrt_weights[k] as f64;
            let a = sw * (gc[k] as f64 - mean);
            let dau = sw * (jgu_c[k] as f64 - mu_du);
            let dav = sw * (jgv_c[k] as f64 - mu_dv);
            dzu[k] = dau * inv_n - a * a_dau * inv_n3;
            dzv[k] = dav * inv_n - a * a_dav * inv_n3;
        }
        for k in 0..n {
            let zu = dzu[k];
            let zv = dzv[k];
            hxx += zu * zu;
            hxy += zu * zv;
            hyy += zv * zv;
            let t = tc[k] as f64;
            bx += zu * t;
            by += zv * t;
        }
    }
    if !any_textured {
        return None;
    }
    Some(([hxx, hxy, hyy], [bx, by]))
}

/// Solve the 2×2 SPD system `H δ = b` (`H = [[Hxx, Hxy], [Hxy, Hyy]]`), with a
/// small Levenberg damping for conditioning. Returns `None` when the (damped)
/// system is near-singular — the aperture problem / low-texture case, where the
/// guard keeps the seed.
pub(in crate::patch::keypoint_subpixel) fn solve_2x2(h: [f64; 3], b: [f64; 2]) -> Option<[f64; 2]> {
    let [hxx, hxy, hyy] = h;
    // Levenberg damping relative to the trace keeps a degenerate (rank-1) Hessian
    // from producing a huge step along the unconstrained direction.
    let lambda = 1e-3 * (hxx + hyy).max(1e-12);
    let a = hxx + lambda;
    let d = hyy + lambda;
    let det = a * d - hxy * hxy;
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (d * b[0] - hxy * b[1]) * inv_det,
        (a * b[1] - hxy * b[0]) * inv_det,
    ])
}
