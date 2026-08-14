// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Warp-consistency residuals: a reconstruction-free per-member misfit
//! signal from a joint weak-perspective factorization of all cluster warps.
//!
//! See `specs/core/cluster-warp-consistency.md`. Every image is modeled as a
//! scaled-orthographic camera `M_k` (2×3) and every refined cluster as a
//! planar patch with tangent frame `T_c` (3×2), parameterized so the
//! reference member's warp is the identity; each stored member warp must
//! then factor as `J_ck = M_k · T_c` (the affine correspondence is the
//! Jacobian of the local image-to-image map). Per cluster this
//! decomposition is inherently ambiguous — each view adds exactly as many
//! camera unknowns as its warp adds measurements — but the cameras are
//! shared across every cluster in an image, so the joint bilinear system is
//! massively over-determined and is solved here by deterministic
//! alternating least squares. The reported per-member relative residual
//! `‖M_k·T_c − J_ck‖_F / ‖J_ck‖_F` flags members whose warp cannot be
//! reconciled with any common plane under the globally consistent cameras —
//! a contamination signal, stored (not gated) so consumers pick their own
//! threshold.
//!
//! Only the fit residual is needed, so no metric upgrade is performed (the
//! residual is invariant to the factorization's global `GL(3)` gauge).

use ndarray::ArrayView3;
use rayon::prelude::*;

use crate::geometry::numeric::splitmix64;

use super::params::MemberStatus;
use super::{inv2, mul2, REFERENCE_UNREFINABLE};

/// A member's warp 2×2 must clear this determinant floor to enter the fit.
const MIN_ABS_DET: f64 = 1e-6;

/// Tikhonov ridge on the 3×3 normal matrices (both ALS half-steps).
const RIDGE: f64 = 1e-9;

/// ALS sweep cap and the RMS-change early-stop threshold (checked every 10
/// sweeps; real data hits its noise floor and stops long before the cap).
const MAX_SWEEPS: usize = 500;
const RMS_STOP: f64 = 1e-9;

/// Independent deterministic ALS restarts; the lowest-RMS solution wins
/// (bilinear factorization with structured missing data has local minima).
const RESTARTS: usize = INIT_NOISE.len();

/// Camera-init perturbation amplitudes, one per restart: the first stays
/// near the orthographic identity (right basin for gently-varying camera
/// graphs), later restarts start from effectively random cameras so at
/// least one lands in the global basin when the true cameras are far from
/// identity. Also sets [`RESTARTS`].
const INIT_NOISE: [f64; 4] = [0.15, 0.5, 1.0, 2.0];

type Mat2x3 = [[f64; 3]; 2];
type Mat3x2 = [[f64; 2]; 3];
type Mat2 = [[f64; 2]; 2];

/// Uniform in [-1, 1) from SplitMix64 (53 mantissa bits -> [0, 2), shifted).
fn noise(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Inverse of a symmetric-plus-ridge 3×3 (callers add [`RIDGE`] to the
/// diagonal first). Returns `None` for a numerically singular matrix.
fn inv3(a: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    if det.abs() < 1e-30 || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            c00 * inv_det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det,
        ],
        [
            c01 * inv_det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det,
        ],
        [
            c02 * inv_det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det,
        ],
    ])
}

/// One fit member: its flat index into the output array, image, and warp.
struct FitMember {
    member_index: u32,
    image: u32,
    j: Mat2,
}

/// Solve one cluster's tangent frame from fixed cameras:
/// `T = (Σ MᵀM + ridge·I)⁻¹ (Σ MᵀJ)`.
fn solve_tangent(cameras: &[Mat2x3], members: &[FitMember]) -> Mat3x2 {
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [[0.0f64; 2]; 3];
    for fm in members {
        let m = &cameras[fm.image as usize];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += m[0][r] * m[0][c] + m[1][r] * m[1][c];
            }
            b[r][0] += m[0][r] * fm.j[0][0] + m[1][r] * fm.j[1][0];
            b[r][1] += m[0][r] * fm.j[0][1] + m[1][r] * fm.j[1][1];
        }
    }
    for (r, row) in a.iter_mut().enumerate() {
        row[r] += RIDGE;
    }
    let Some(ai) = inv3(&a) else {
        return [[0.0; 2]; 3];
    };
    let mut t = [[0.0f64; 2]; 3];
    for r in 0..3 {
        for c in 0..2 {
            t[r][c] = ai[r][0] * b[0][c] + ai[r][1] * b[1][c] + ai[r][2] * b[2][c];
        }
    }
    t
}

/// `M_k · T_c` (2×2 prediction).
fn predict(m: &Mat2x3, t: &Mat3x2) -> Mat2 {
    let mut p = [[0.0f64; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            p[r][c] = m[r][0] * t[0][c] + m[r][1] * t[1][c] + m[r][2] * t[2][c];
        }
    }
    p
}

fn frob_sq(m: &Mat2) -> f64 {
    m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[1][0] * m[1][0] + m[1][1] * m[1][1]
}

fn frob(m: &Mat2) -> f64 {
    frob_sq(m).sqrt()
}

fn sub(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [a[0][0] - b[0][0], a[0][1] - b[0][1]],
        [a[1][0] - b[1][0], a[1][1] - b[1][1]],
    ]
}

/// Compute the per-member warp-consistency residuals for a refined cluster
/// set (see the module docs). `member_affines` / `member_status` /
/// `reference_members` are the [`refine_cluster_patches`] outputs
/// (member-parallel; `(M, 2, 3)` — only the leading 2×2 blocks enter the fit;
/// the last column, the member's absolute refined keypoint position, is never
/// read). The stored block is the member's ABSOLUTE affine shape
/// `S = W·S_ref`, so each cluster's reference row is inverted once to recover
/// the reference-relative warps `W = S·S_ref⁻¹` the factorization is
/// parameterized on; a cluster whose `S_ref` is singular is skipped whole.
/// Members that participate
/// in the fit — the reference (`J = I`) plus every kept member with a
/// non-degenerate warp, in clusters with at least 2 such members — get a
/// residual; everything else is NaN. Deterministic: fixed seed, fixed
/// iteration order, parallelism only across independent solves.
///
/// [`refine_cluster_patches`]: super::refine_cluster_patches
pub fn warp_consistency_residuals(
    cluster_starts: &[u32],
    member_images: &[u32],
    member_status: &[MemberStatus],
    reference_members: &[u32],
    member_affines: ArrayView3<'_, f64>,
    n_images: usize,
) -> Vec<f32> {
    let m_total = member_status.len();
    let mut residuals = vec![f32::NAN; m_total];
    if n_images == 0 {
        return residuals;
    }

    // Gather fit members, grouped per cluster (CSR over `fit`).
    let mut fit: Vec<FitMember> = Vec::new();
    let mut row_starts: Vec<u32> = vec![0];
    for (c, &ref_k) in reference_members.iter().enumerate() {
        if ref_k == REFERENCE_UNREFINABLE {
            continue;
        }
        // The fit is parameterized on the REFERENCE-RELATIVE warp `W` (the
        // reference's own is the identity by construction), while the stored
        // block is the absolute shape `S = W·S_ref`. Recover `W = S·S_ref⁻¹`
        // through this cluster's reference row; a singular `S_ref` (rejected
        // by the writer, but this entry point takes raw arrays) leaves the
        // whole cluster out of the fit.
        let rk = ref_k as usize;
        let s_ref: Mat2 = [
            [member_affines[[rk, 0, 0]], member_affines[[rk, 0, 1]]],
            [member_affines[[rk, 1, 0]], member_affines[[rk, 1, 1]]],
        ];
        let s_ref_det = s_ref[0][0] * s_ref[1][1] - s_ref[0][1] * s_ref[1][0];
        if s_ref_det.abs() < MIN_ABS_DET || !s_ref_det.is_finite() {
            continue;
        }
        let s_ref_inv = inv2(&s_ref);
        let begin = fit.len();
        for k in cluster_starts[c] as usize..cluster_starts[c + 1] as usize {
            let j: Mat2 = if k as u32 == ref_k {
                [[1.0, 0.0], [0.0, 1.0]]
            } else if member_status[k] == MemberStatus::Kept {
                let s: Mat2 = [
                    [member_affines[[k, 0, 0]], member_affines[[k, 0, 1]]],
                    [member_affines[[k, 1, 0]], member_affines[[k, 1, 1]]],
                ];
                mul2(&s, &s_ref_inv)
            } else {
                continue;
            };
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det.abs() < MIN_ABS_DET || !det.is_finite() {
                continue;
            }
            fit.push(FitMember {
                member_index: k as u32,
                image: member_images[k],
                j,
            });
        }
        if fit.len() - begin >= 2 {
            row_starts.push(fit.len() as u32);
        } else {
            fit.truncate(begin);
        }
    }
    let n_rows = row_starts.len() - 1;
    if n_rows == 0 {
        return residuals;
    }

    // Per-image membership index lists (fixed order -> deterministic sums).
    let mut by_image: Vec<Vec<u32>> = vec![Vec::new(); n_images];
    for (i, fm) in fit.iter().enumerate() {
        by_image[fm.image as usize].push(i as u32);
    }

    // Precompute each fit member's row (for the camera half-step).
    let mut fit_row = vec![0u32; fit.len()];
    for (row, w) in row_starts.windows(2).enumerate() {
        fit_row[w[0] as usize..w[1] as usize].fill(row as u32);
    }

    let row_members = |row: usize| &fit[row_starts[row] as usize..row_starts[row + 1] as usize];

    // Deterministic RMS (sequential sum) so the best-restart selection and
    // early stop are schedule-independent.
    let rms = |cameras: &[Mat2x3], tangents: &[Mat3x2]| -> f64 {
        let mut sse = 0.0;
        for (row, t) in tangents.iter().enumerate() {
            for fm in row_members(row) {
                let p = predict(&cameras[fm.image as usize], t);
                sse += frob_sq(&sub(&p, &fm.j));
            }
        }
        (sse / (4.0 * fit.len() as f64)).sqrt()
    };

    let run_als = |seed: u64| -> (f64, Vec<Mat2x3>, Vec<Mat3x2>) {
        // Deterministically-perturbed orthographic init, per restart.
        let mut cameras: Vec<Mat2x3> = (0..n_images)
            .map(|k| {
                let mut s = 0x5f37_59df_u64
                    ^ seed.wrapping_mul(0xd129_9f7d)
                    ^ (k as u64).wrapping_mul(0x9e37_79b9);
                let mut m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
                let amplitude = INIT_NOISE[seed as usize % INIT_NOISE.len()];
                for row in m.iter_mut() {
                    for v in row.iter_mut() {
                        *v += amplitude * noise(&mut s);
                    }
                }
                m
            })
            .collect();
        let mut tangents: Vec<Mat3x2> = (0..n_rows)
            .into_par_iter()
            .map(|row| solve_tangent(&cameras, row_members(row)))
            .collect();

        let mut last_rms = f64::INFINITY;
        for sweep in 0..MAX_SWEEPS {
            // Camera half-step: M = (Σ J·Tᵀ)(Σ T·Tᵀ + ridge·I)⁻¹ per image.
            cameras = (0..n_images)
                .into_par_iter()
                .map(|img| {
                    let members = &by_image[img];
                    if members.is_empty() {
                        return cameras[img];
                    }
                    let mut a = [[0.0f64; 3]; 3];
                    let mut b = [[0.0f64; 3]; 2];
                    for &fi in members {
                        let fm = &fit[fi as usize];
                        let t = &tangents[fit_row[fi as usize] as usize];
                        for r in 0..3 {
                            for c in 0..3 {
                                a[r][c] += t[r][0] * t[c][0] + t[r][1] * t[c][1];
                            }
                        }
                        for c in 0..3 {
                            b[0][c] += fm.j[0][0] * t[c][0] + fm.j[0][1] * t[c][1];
                            b[1][c] += fm.j[1][0] * t[c][0] + fm.j[1][1] * t[c][1];
                        }
                    }
                    for (r, row) in a.iter_mut().enumerate() {
                        row[r] += RIDGE;
                    }
                    let Some(ai) = inv3(&a) else {
                        return cameras[img];
                    };
                    let mut m = [[0.0f64; 3]; 2];
                    for r in 0..2 {
                        for c in 0..3 {
                            m[r][c] = b[r][0] * ai[0][c] + b[r][1] * ai[1][c] + b[r][2] * ai[2][c];
                        }
                    }
                    m
                })
                .collect();
            // Tangent half-step.
            tangents = (0..n_rows)
                .into_par_iter()
                .map(|row| solve_tangent(&cameras, row_members(row)))
                .collect();

            if sweep % 10 == 9 {
                let cur = rms(&cameras, &tangents);
                if (last_rms - cur).abs() < RMS_STOP {
                    break;
                }
                last_rms = cur;
            }
        }
        (rms(&cameras, &tangents), cameras, tangents)
    };

    // Best of RESTARTS independent runs (ties keep the lowest seed).
    let (mut best_rms, mut cameras, mut tangents) = run_als(0);
    for seed in 1..RESTARTS as u64 {
        let (r, c, t) = run_als(seed);
        if r < best_rms {
            best_rms = r;
            cameras = c;
            tangents = t;
        }
    }

    // Per-member relative residual.
    let cameras_ref: &[Mat2x3] = &cameras;
    let scattered: Vec<(u32, f32)> = (0..n_rows)
        .into_par_iter()
        .flat_map_iter(|row| {
            let t = tangents[row];
            row_members(row)
                .iter()
                .map(move |fm| {
                    let p = predict(&cameras_ref[fm.image as usize], &t);
                    let rel = frob(&sub(&p, &fm.j)) / frob(&fm.j).max(1e-9);
                    (fm.member_index, rel as f32)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    for (k, r) in scattered {
        residuals[k as usize] = r;
    }
    residuals
}

#[cfg(test)]
mod tests;
