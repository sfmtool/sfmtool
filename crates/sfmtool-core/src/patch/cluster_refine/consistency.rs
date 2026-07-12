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

use super::params::MemberStatus;
use super::REFERENCE_UNREFINABLE;

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

/// SplitMix64 — the deterministic init-noise generator (no rand dependency,
/// identical across platforms).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

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
/// (member-parallel; `(M, 2, 3)` — only the leading 2×2 warp blocks enter
/// the fit; the last column, the member's absolute refined keypoint
/// position, is never read). Members that participate
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
        let begin = fit.len();
        for k in cluster_starts[c] as usize..cluster_starts[c + 1] as usize {
            let j: Mat2 = if k as u32 == ref_k {
                [[1.0, 0.0], [0.0, 1.0]]
            } else if member_status[k] == MemberStatus::Kept {
                [
                    [member_affines[[k, 0, 0]], member_affines[[k, 0, 1]]],
                    [member_affines[[k, 1, 0]], member_affines[[k, 1, 1]]],
                ]
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
mod tests {
    use super::super::params::MemberStatus;
    use super::*;
    use ndarray::Array3;

    /// `(cluster_starts, member_images, member_status, reference_members,
    /// member_affines)` — the [`warp_consistency_residuals`] inputs.
    type SyntheticScene = (Vec<u32>, Vec<u32>, Vec<MemberStatus>, Vec<u32>, Array3<f64>);

    /// Build a synthetic scene: `n_images` scaled-orthographic cameras and
    /// `n_clusters` planar frames, emitted as clusters whose reference warp
    /// is re-gauged to the identity (exactly the stored representation).
    fn synthetic(n_images: usize, n_clusters: usize, members_per_cluster: usize) -> SyntheticScene {
        let mut state = 12345u64;
        let mut rnd = move || {
            let mut s = state;
            let v = noise(&mut s);
            state = s;
            v
        };
        // Cameras: random small rotations around identity, orthographic.
        let cams: Vec<Mat2x3> = (0..n_images)
            .map(|_| {
                let (a, b, c) = (0.4 * rnd(), 0.4 * rnd(), 0.4 * rnd());
                let (ca, sa) = (a.cos(), a.sin());
                let (cb, sb) = (b.cos(), b.sin());
                let (cc, sc) = (c.cos(), c.sin());
                // Rz(c)·Ry(b)·Rx(a), top two rows.
                [
                    [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
                    [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
                ]
            })
            .collect();
        let m = n_clusters * members_per_cluster;
        let mut cluster_starts = vec![0u32];
        let mut member_images = Vec::with_capacity(m);
        let mut status = Vec::with_capacity(m);
        let mut refs = Vec::with_capacity(n_clusters);
        let mut affines = Array3::<f64>::zeros((m, 2, 3));
        for c in 0..n_clusters {
            // Random tangent frame with a definite out-of-plane component.
            let t: Mat3x2 = [
                [1.0 + 0.2 * rnd(), 0.2 * rnd()],
                [0.2 * rnd(), 1.0 + 0.2 * rnd()],
                [0.6 * rnd(), 0.6 * rnd()],
            ];
            let base = c * members_per_cluster;
            let ref_img = c % n_images;
            let jr = predict(&cams[ref_img], &t);
            // Re-gauge so the reference sees the identity.
            let det = jr[0][0] * jr[1][1] - jr[0][1] * jr[1][0];
            let jr_inv = [
                [jr[1][1] / det, -jr[0][1] / det],
                [-jr[1][0] / det, jr[0][0] / det],
            ];
            for i in 0..members_per_cluster {
                let k = base + i;
                // Distinct pseudo-random member images per cluster (dense
                // cross-view coupling, the shape of real cluster graphs;
                // n_images - 1 must be coprime with the stride multiplier).
                let img = if i == 0 {
                    ref_img
                } else {
                    (ref_img + 1 + (c * 7 + i * 5) % (n_images - 1)) % n_images
                };
                member_images.push(img as u32);
                if i == 0 {
                    status.push(MemberStatus::Reference);
                    refs.push(k as u32);
                    affines[[k, 0, 0]] = 1.0;
                    affines[[k, 1, 1]] = 1.0;
                } else {
                    status.push(MemberStatus::Kept);
                    let jm = predict(&cams[img], &t);
                    for r in 0..2 {
                        for cc2 in 0..2 {
                            affines[[k, r, cc2]] =
                                jm[r][0] * jr_inv[0][cc2] + jm[r][1] * jr_inv[1][cc2];
                        }
                    }
                }
            }
            cluster_starts.push((base + members_per_cluster) as u32);
        }
        (cluster_starts, member_images, status, refs, affines)
    }

    #[test]
    fn oracle_cameras_fit_exactly() {
        // With the TRUE cameras, per-cluster tangent solves must reproduce
        // every warp exactly (validates generation + solve_tangent).
        let n_images = 12;
        let mut state = 12345u64;
        let mut rnd = move || {
            let mut s = state;
            let v = noise(&mut s);
            state = s;
            v
        };
        let cams: Vec<Mat2x3> = (0..n_images)
            .map(|_| {
                let (a, b, c) = (0.4 * rnd(), 0.4 * rnd(), 0.4 * rnd());
                let (ca, sa) = (a.cos(), a.sin());
                let (cb, sb) = (b.cos(), b.sin());
                let (cc, sc) = (c.cos(), c.sin());
                [
                    [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
                    [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
                ]
            })
            .collect();
        // One cluster: ref image 0, members 1..3.
        let t: Mat3x2 = [[1.1, 0.1], [-0.05, 0.95], [0.4, -0.3]];
        let jr = predict(&cams[0], &t);
        let det = jr[0][0] * jr[1][1] - jr[0][1] * jr[1][0];
        let jr_inv = [
            [jr[1][1] / det, -jr[0][1] / det],
            [-jr[1][0] / det, jr[0][0] / det],
        ];
        let mut members = vec![FitMember {
            member_index: 0,
            image: 0,
            j: [[1.0, 0.0], [0.0, 1.0]],
        }];
        for i in 1..4u32 {
            let jm = predict(&cams[i as usize], &t);
            let mut j = [[0.0; 2]; 2];
            for r in 0..2 {
                for c in 0..2 {
                    j[r][c] = jm[r][0] * jr_inv[0][c] + jm[r][1] * jr_inv[1][c];
                }
            }
            members.push(FitMember {
                member_index: i,
                image: i,
                j,
            });
        }
        let tsol = solve_tangent(&cams, &members);
        let mut worst = 0.0f64;
        for fm in &members {
            let p = predict(&cams[fm.image as usize], &tsol);
            let d = [
                [p[0][0] - fm.j[0][0], p[0][1] - fm.j[0][1]],
                [p[1][0] - fm.j[1][0], p[1][1] - fm.j[1][1]],
            ];
            worst = worst.max(frob(&d));
        }
        // The ridge (1e-9) biases the exact solve by a comparable amount.
        assert!(worst < 1e-6, "oracle-camera misfit {worst} should be ~0");
    }

    #[test]
    fn consistent_synthetic_scene_has_near_zero_residuals() {
        let (starts, images, status, refs, affines) = synthetic(12, 400, 4);
        let res = warp_consistency_residuals(&starts, &images, &status, &refs, affines.view(), 12);
        let finite: Vec<f32> = res.iter().copied().filter(|v| v.is_finite()).collect();
        assert_eq!(finite.len(), 400 * 4);
        let median = {
            let mut s = finite.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        assert!(median < 1e-3, "median residual {median} should be ~0");
    }

    #[test]
    fn contaminated_member_scores_highest() {
        let (starts, images, status, refs, mut affines) = synthetic(12, 400, 4);
        // Corrupt one kept member's warp (wrong-match simulation).
        let bad = 4 * 7 + 2;
        affines[[bad, 0, 0]] = -0.3;
        affines[[bad, 0, 1]] = 1.1;
        affines[[bad, 1, 0]] = 0.9;
        affines[[bad, 1, 1]] = 0.4;
        let res = warp_consistency_residuals(&starts, &images, &status, &refs, affines.view(), 12);
        assert!(
            res[bad] > 0.2,
            "contaminated member residual {} should be large",
            res[bad]
        );
        // Members of untouched clusters stay near zero.
        let clean_max = res
            .iter()
            .enumerate()
            .filter(|(k, v)| *k / 4 != bad / 4 && v.is_finite())
            .map(|(_, v)| *v)
            .fold(0.0f32, f32::max);
        assert!(
            clean_max < 0.05,
            "clean clusters should stay consistent (max {clean_max})"
        );
    }

    #[test]
    fn non_participants_are_nan_and_runs_are_deterministic() {
        let (starts, images, mut status, refs, affines) = synthetic(12, 50, 4);
        // Demote one member: it must come back NaN.
        status[4 * 3 + 1] = MemberStatus::RejectedLowZncc;
        // Demote ALL of cluster 5's non-reference members: with fewer than
        // 2 fitted members the whole cluster leaves the fit, so even its
        // reference is NaN.
        for i in 1..4 {
            status[4 * 5 + i] = MemberStatus::RejectedLowZncc;
        }
        let run =
            || warp_consistency_residuals(&starts, &images, &status, &refs, affines.view(), 12);
        let a = run();
        let b = run();
        assert!(a[4 * 3 + 1].is_nan());
        for i in 0..4 {
            assert!(a[4 * 5 + i].is_nan(), "member {i} of the dropped cluster");
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b), "runs must be bit-identical");
    }
}
