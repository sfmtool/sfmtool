// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Staged bundle adjustment for images sharing one camera model.
//!
//! Jointly refines world-to-camera poses, world points, and optionally the
//! shared focal length by minimizing soft-L1 pixel reprojection error over a
//! trim schedule with inter-round retriangulation — the multi-view
//! generalization of [`crate::geometry::pose_refine`], and the native
//! replacement for the cluster-bootstrap experiments' scipy BA
//! (`specs/core/bundle-adjustment.md`).
//!
//! Canonical camera frame throughout (the camera looks along `−Z`; a point in
//! front has `z < 0`). Each Levenberg–Marquardt step is taken over a local
//! `SO(3) × ℝ³` perturbation per image, `ℝ³` per point, and an optional focal
//! scalar, with analytic Jacobians; points are eliminated by a Schur
//! complement and the dense reduced camera system is solved by LU.

use nalgebra::{DMatrix, DVector, Matrix3, Point3, SMatrix, UnitQuaternion, Vector2, Vector3};

use crate::camera::{CameraModel, PixelJacobian};
use crate::reconstruction::triangulation::triangulate_batch;
use crate::CameraIntrinsics;

/// A point behind the camera / outside the model domain contributes this pixel
/// residual per component — large enough to be trimmed, finite so the robust
/// cost stays well-posed (matches `reprojection_residuals` / `pose_refine`).
const INVALID_RESIDUAL: f64 = 1e6;

/// One round of the trim schedule.
#[derive(Clone, Copy, Debug)]
pub struct BaSchedule {
    /// Pre-round trim threshold on the reprojection residual norm, px.
    pub trim_px: f64,
    /// Soft-L1 scale for the round's solve, px.
    pub loss_scale: f64,
}

/// The default staged schedule (gross-outlier trim → tighten → final).
pub const DEFAULT_SCHEDULE: [BaSchedule; 3] = [
    BaSchedule {
        trim_px: 50.0,
        loss_scale: 5.0,
    },
    BaSchedule {
        trim_px: 12.0,
        loss_scale: 2.0,
    },
    BaSchedule {
        trim_px: 4.0,
        loss_scale: 1.0,
    },
];

/// Default widening multiplier on a stage's `loss_scale` for protected
/// observations (see [`bundle_adjust`]'s `protected`).
pub const DEFAULT_PROTECTED_LOSS_SCALE: f64 = 3.0;

/// Result of [`bundle_adjust`]. Poses and points are refined in place; this
/// carries what has no in-place home.
#[derive(Clone, Debug)]
pub struct BundleAdjustment {
    /// The shared focal length after the solve (the input focal unless
    /// `opt_f`).
    pub focal: f64,
    /// Unweighted reprojection residual norm of every supplied observation at
    /// the final state; `+∞` where the point is non-finite, behind the
    /// camera, or outside the model domain. All-`∞` signals the degenerate
    /// exit (fewer than `min_obs` observations survived a trim).
    pub residual_norms: Vec<f64>,
}

/// Soft-L1 robust cost of a squared-residual-over-scale² argument:
/// `ρ(z) = 2·(√(1 + z) − 1)`, applied per residual COMPONENT (matching
/// scipy's element-wise `loss="soft_l1"` that this kernel replaces).
#[inline]
fn rho(z: f64) -> f64 {
    2.0 * ((1.0 + z).sqrt() - 1.0)
}

/// Second-order (Triggs-style) robust scaling of one residual component,
/// exactly scipy's `scale_for_robust_loss_function`: with `z = (r/s)²`,
/// scale the Jacobian row by `√(ρ' + 2·ρ''·z)` and the residual by
/// `ρ'/√(ρ' + 2·ρ''·z)`. For soft-L1 the curvature term collapses to
/// `ρ' + 2ρ''z = (1 + z)^(−3/2)`, so the row scale is `(1 + z)^(−¾)` and the
/// residual scale `(1 + z)^(+¼)`; the resulting `Jᵀr` equals the true robust
/// gradient `ρ'·Jᵀr` while `JᵀJ` carries the corrected curvature.
#[inline]
fn robust_scales(z: f64) -> (f64, f64) {
    let js = (1.0 + z).powf(-0.75);
    let rs = (1.0 + z).powf(0.25);
    (js, rs)
}

/// The shared camera at focal `f` (identity for every model but
/// SIMPLE_PINHOLE — `opt_f` is gated on that model, so no other ever sees a
/// moved focal).
fn cam_at(cam: &CameraIntrinsics, f: f64) -> CameraIntrinsics {
    let mut out = cam.clone();
    if let CameraModel::SimplePinhole { focal_length, .. } = &mut out.model {
        *focal_length = f;
    }
    out
}

/// Per-observation residual norm and canonical in-front depth (`−z_cam`) at
/// the given state. Invalid observations (non-finite point, behind camera,
/// outside domain) report `INVALID_RESIDUAL` and their (possibly negative)
/// depth.
fn residual_norms_depths(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
) -> (Vec<f64>, Vec<f64>) {
    let n_obs = obs_img.len();
    let mut norms = vec![INVALID_RESIDUAL; n_obs];
    let mut depths = vec![f64::NEG_INFINITY; n_obs];
    for k in 0..n_obs {
        let p = points[obs_pt[k] as usize];
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }
        let i = obs_img[k] as usize;
        let c = quats[i] * Vector3::new(p[0], p[1], p[2]) + trans[i];
        depths[k] = -c.z;
        if let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) {
            norms[k] = (u - uv[k][0]).hypot(v - uv[k][1]);
        }
    }
    (norms, depths)
}

/// Rebuild every point from all supplied observations at the current poses
/// (ray-midpoint batch triangulation). Tracks with fewer than two
/// observations — and points with none — become `NaN`; callers refill from
/// their full observation set (the bootstrap's post-BA refill rule).
fn retriangulate(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
) {
    let n_obs = obs_img.len();
    let mut order: Vec<u32> = (0..n_obs as u32).collect();
    order.sort_unstable_by_key(|&k| obs_pt[k as usize]);

    let mut dirs = Vec::with_capacity(n_obs);
    let mut centers = Vec::with_capacity(n_obs);
    let mut offsets = Vec::new();
    let mut track_pt = Vec::new();
    let mut prev: Option<u32> = None;
    for &k in &order {
        let k = k as usize;
        let p = obs_pt[k];
        if prev != Some(p) {
            offsets.push(dirs.len());
            track_pt.push(p as usize);
            prev = Some(p);
        }
        let i = obs_img[k] as usize;
        let r_inv = quats[i].inverse();
        let d = cam.pixel_to_ray(uv[k][0], uv[k][1]);
        dirs.push(r_inv * Vector3::new(d[0], d[1], d[2]));
        centers.push(Point3::from(-(r_inv * trans[i])));
    }
    offsets.push(dirs.len());

    for p in points.iter_mut() {
        *p = [f64::NAN; 3];
    }
    let tris = triangulate_batch(&dirs, &centers, &offsets);
    for (t, tri) in tris.iter().enumerate() {
        if offsets[t + 1] - offsets[t] >= 2 {
            points[track_pt[t]] = [tri.point.x, tri.point.y, tri.point.z];
        }
    }
}

/// Projected pixel and the 2×3 projection Jacobian `∂(u, v)/∂p_cam` at a
/// camera-frame point. Analytic for the perspective family; a central
/// difference of `ray_to_pixel` for fisheye / equirectangular models, which
/// have no analytic Jacobian yet (same fallback as `pose_refine`). `None`
/// when the point is outside the model domain.
fn project_with_jac(
    cam: &CameraIntrinsics,
    p_cam: Vector3<f64>,
    analytic: bool,
) -> Option<PixelJacobian> {
    if analytic {
        return cam.ray_to_pixel_with_jacobian([p_cam.x, p_cam.y, p_cam.z]);
    }
    let uv = cam.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z])?;
    let h = 1e-6;
    let mut j = [[0.0f64; 3]; 2];
    for c in 0..3 {
        let mut pp = p_cam;
        let mut pm = p_cam;
        pp[c] += h;
        pm[c] -= h;
        let (up, vp) = cam.ray_to_pixel([pp.x, pp.y, pp.z])?;
        let (um, vm) = cam.ray_to_pixel([pm.x, pm.y, pm.z])?;
        j[0][c] = (up - um) / (2.0 * h);
        j[1][c] = (vp - vm) / (2.0 * h);
    }
    Some((uv, j))
}

/// Linearization of one observation: weighted residual and the weighted
/// camera-side (`[f | δθ | δt]`, 2×7) and point-side (2×3) Jacobian blocks.
struct ObsBlocks {
    /// Compact image index.
    ci: usize,
    /// Compact point index.
    cp: usize,
    res: Vector2<f64>,
    cam_j: SMatrix<f64, 2, 7>,
    pt_j: SMatrix<f64, 2, 3>,
}

/// Robust cost over the kept observations at a candidate state. `s2s` is the
/// per-kept-observation squared loss scale (uniform except where a protected
/// observation widens it).
#[allow(clippy::too_many_arguments)]
fn robust_cost(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[Vector3<f64>],
    uv: &[[f64; 2]],
    kept: &[usize],
    obs_ci: &[usize],
    obs_cp: &[usize],
    s2s: &[f64],
) -> f64 {
    kept.iter()
        .enumerate()
        .map(|(kk, &k)| {
            let s2 = s2s[kk];
            let p = points[obs_cp[kk]];
            // A non-finite point (possible only for protected observations,
            // which the trim never excludes) is penalized like an
            // out-of-domain projection.
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                return s2 * rho(INVALID_RESIDUAL * INVALID_RESIDUAL / s2);
            }
            let c = quats[obs_ci[kk]] * p + trans[obs_ci[kk]];
            match cam.ray_to_pixel([c.x, c.y, c.z]) {
                Some((u, v)) => {
                    let dx = u - uv[k][0];
                    let dy = v - uv[k][1];
                    s2 * (rho(dx * dx / s2) + rho(dy * dy / s2))
                }
                None => s2 * rho(INVALID_RESIDUAL * INVALID_RESIDUAL / s2),
            }
        })
        .sum()
}

/// One robust sparse LM solve over the kept observations (compact-indexed).
/// Updates `quats` / `trans` / `points` in place and returns the focal.
#[allow(clippy::too_many_arguments)]
fn solve_lm(
    cam0: &CameraIntrinsics,
    f0: f64,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    kept: &[usize],
    opt_f: bool,
    loss_scale: f64,
    max_iters: usize,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
) -> f64 {
    // Compact the images and points the kept observations touch.
    let mut img_ids: Vec<usize> = kept.iter().map(|&k| obs_img[k] as usize).collect();
    img_ids.sort_unstable();
    img_ids.dedup();
    let mut pt_ids: Vec<usize> = kept.iter().map(|&k| obs_pt[k] as usize).collect();
    pt_ids.sort_unstable();
    pt_ids.dedup();
    let n_im = img_ids.len();
    let n_pt = pt_ids.len();
    let ci_of: std::collections::HashMap<usize, usize> =
        img_ids.iter().enumerate().map(|(c, &i)| (i, c)).collect();
    let cp_of: std::collections::HashMap<usize, usize> =
        pt_ids.iter().enumerate().map(|(c, &p)| (p, c)).collect();
    let obs_ci: Vec<usize> = kept
        .iter()
        .map(|&k| ci_of[&(obs_img[k] as usize)])
        .collect();
    let obs_cp: Vec<usize> = kept.iter().map(|&k| cp_of[&(obs_pt[k] as usize)]).collect();

    // Per-point observation lists (compact indices into `kept`).
    let mut pt_obs: Vec<Vec<usize>> = vec![Vec::new(); n_pt];
    for (kk, &cp) in obs_cp.iter().enumerate() {
        pt_obs[cp].push(kk);
    }

    // Working state (compact copies).
    let mut f = f0;
    let mut q: Vec<UnitQuaternion<f64>> = img_ids.iter().map(|&i| quats[i]).collect();
    let mut t: Vec<Vector3<f64>> = img_ids.iter().map(|&i| trans[i]).collect();
    let mut x: Vec<Vector3<f64>> = pt_ids
        .iter()
        .map(|&p| Vector3::new(points[p][0], points[p][1], points[p][2]))
        .collect();

    // Reduced camera system: [f | 6 per image], the focal slot always present
    // (pinned when !opt_f) to keep the indexing uniform.
    let d = 1 + 6 * n_im;
    // Per-kept-observation squared loss scale: the round's scale everywhere,
    // widened by `protected_loss_scale` for protected observations.
    let s2 = loss_scale * loss_scale;
    let s2s: Vec<f64> = kept
        .iter()
        .map(|&k| match protected {
            Some(m) if m[k] => {
                let s = loss_scale * protected_loss_scale;
                s * s
            }
            _ => s2,
        })
        .collect();
    let mut lambda = 1e-3;
    let mut tiny_steps = 0usize;
    let mut cam = cam_at(cam0, f);
    let mut prev_cost = robust_cost(&cam, &q, &t, &x, uv, kept, &obs_ci, &obs_cp, &s2s);

    let analytic = cam.model.supports_pixel_jacobian();
    for _ in 0..max_iters {
        // ── Linearize at the current state ───────────────────────────────
        let (cx, cy) = cam.principal_point();
        let blocks: Vec<ObsBlocks> = kept
            .iter()
            .enumerate()
            .map(|(kk, &k)| {
                let ci = obs_ci[kk];
                let cp = obs_cp[kk];
                let s2 = s2s[kk];
                let rot_pt = q[ci] * x[cp];
                let p_cam = rot_pt + t[ci];
                let mut res = Vector2::new(INVALID_RESIDUAL, 0.0);
                let mut cam_j = SMatrix::<f64, 2, 7>::zeros();
                let mut pt_j = SMatrix::<f64, 2, 3>::zeros();
                // A non-finite point (protected observations only — the trim
                // never excludes them) keeps the penalized residual and zero
                // Jacobian rows: penalized, never steering.
                let proj = if x[cp].x.is_finite() && x[cp].y.is_finite() && x[cp].z.is_finite() {
                    project_with_jac(&cam, p_cam, analytic)
                } else {
                    None
                };
                if let Some(((u, v), jp)) = proj {
                    res = Vector2::new(u - uv[k][0], v - uv[k][1]);
                    let jp = SMatrix::<f64, 2, 3>::from_rows(&[
                        SMatrix::<f64, 1, 3>::from_row_slice(&jp[0]),
                        SMatrix::<f64, 1, 3>::from_row_slice(&jp[1]),
                    ]);
                    if opt_f {
                        cam_j[(0, 0)] = (u - cx) / f;
                        cam_j[(1, 0)] = (v - cy) / f;
                    }
                    // Rotation block: ∂p_cam/∂δθ = −[R·X]ₓ.
                    let nskew = Matrix3::new(
                        0.0, rot_pt.z, -rot_pt.y, //
                        -rot_pt.z, 0.0, rot_pt.x, //
                        rot_pt.y, -rot_pt.x, 0.0,
                    );
                    cam_j.fixed_view_mut::<2, 3>(0, 1).copy_from(&(jp * nskew));
                    // Translation block: identity.
                    cam_j.fixed_view_mut::<2, 3>(0, 4).copy_from(&jp);
                    // Point block: ∂p_cam/∂X = R.
                    let r_mat: Matrix3<f64> = q[ci].to_rotation_matrix().into_inner();
                    pt_j.copy_from(&(jp * r_mat));
                }
                for row in 0..2 {
                    let z = res[row] * res[row] / s2;
                    let (js, rs) = robust_scales(z);
                    res[row] *= rs;
                    for col in 0..7 {
                        cam_j[(row, col)] *= js;
                    }
                    for col in 0..3 {
                        pt_j[(row, col)] *= js;
                    }
                }
                ObsBlocks {
                    ci,
                    cp,
                    res,
                    cam_j,
                    pt_j,
                }
            })
            .collect();

        // ── Accumulate the normal-equation blocks ────────────────────────
        let mut h_cc = DMatrix::<f64>::zeros(d, d);
        let mut g_c = DVector::<f64>::zeros(d);
        let mut v_pp: Vec<Matrix3<f64>> = vec![Matrix3::zeros(); n_pt];
        let mut g_p: Vec<Vector3<f64>> = vec![Vector3::zeros(); n_pt];
        let mut w_cp: Vec<SMatrix<f64, 7, 3>> = Vec::with_capacity(blocks.len());
        for b in &blocks {
            let idx = [
                0,
                1 + 6 * b.ci,
                2 + 6 * b.ci,
                3 + 6 * b.ci,
                4 + 6 * b.ci,
                5 + 6 * b.ci,
                6 + 6 * b.ci,
            ];
            let h_local = b.cam_j.transpose() * b.cam_j;
            let g_local = b.cam_j.transpose() * b.res;
            for (a, &ia) in idx.iter().enumerate() {
                g_c[ia] += g_local[a];
                for (c, &ic) in idx.iter().enumerate() {
                    h_cc[(ia, ic)] += h_local[(a, c)];
                }
            }
            v_pp[b.cp] += b.pt_j.transpose() * b.pt_j;
            g_p[b.cp] += b.pt_j.transpose() * b.res;
            w_cp.push(b.cam_j.transpose() * b.pt_j);
        }

        // ── Damping ladder: re-damp and re-solve from this linearization ──
        let mut improved = false;
        for _ in 0..12 {
            let mut s = h_cc.clone();
            for dd in 0..d {
                s[(dd, dd)] += lambda * h_cc[(dd, dd)].max(1e-12);
            }
            let mut g_red = g_c.clone();
            // Schur-eliminate the points.
            let mut v_inv: Vec<Matrix3<f64>> = Vec::with_capacity(n_pt);
            let mut singular = false;
            for v in &v_pp {
                let mut vd = *v;
                for dd in 0..3 {
                    vd[(dd, dd)] += lambda * v[(dd, dd)].max(1e-12);
                }
                match vd.try_inverse() {
                    Some(inv) => v_inv.push(inv),
                    None => {
                        singular = true;
                        break;
                    }
                }
            }
            if singular {
                lambda *= 4.0;
                continue;
            }
            for (p, obs) in pt_obs.iter().enumerate() {
                let y = v_inv[p] * g_p[p];
                for &a in obs {
                    let wa = &w_cp[a];
                    let ba = blocks[a].ci;
                    let ia = [
                        0,
                        1 + 6 * ba,
                        2 + 6 * ba,
                        3 + 6 * ba,
                        4 + 6 * ba,
                        5 + 6 * ba,
                        6 + 6 * ba,
                    ];
                    let contrib = wa * y;
                    for (r, &ir) in ia.iter().enumerate() {
                        g_red[ir] -= contrib[r];
                    }
                    for &b in obs {
                        let m = wa * v_inv[p] * w_cp[b].transpose();
                        let bb = blocks[b].ci;
                        let ib = [
                            0,
                            1 + 6 * bb,
                            2 + 6 * bb,
                            3 + 6 * bb,
                            4 + 6 * bb,
                            5 + 6 * bb,
                            6 + 6 * bb,
                        ];
                        for (r, &ir) in ia.iter().enumerate() {
                            for (c, &ic) in ib.iter().enumerate() {
                                s[(ir, ic)] -= m[(r, c)];
                            }
                        }
                    }
                }
            }
            if !opt_f {
                // Pin the focal slot.
                for dd in 0..d {
                    s[(0, dd)] = 0.0;
                    s[(dd, 0)] = 0.0;
                }
                s[(0, 0)] = 1.0;
                g_red[0] = 0.0;
            }

            let Some(delta) = s.lu().solve(&(-g_red)) else {
                lambda *= 4.0;
                continue;
            };

            // Candidate state.
            let f_cand = if opt_f { f + delta[0] } else { f };
            if opt_f && !(f_cand.is_finite() && f_cand > 1e-6) {
                lambda *= 4.0;
                continue;
            }
            let mut q_cand = q.clone();
            let mut t_cand = t.clone();
            for c in 0..n_im {
                let o = 1 + 6 * c;
                let dtheta = Vector3::new(delta[o], delta[o + 1], delta[o + 2]);
                q_cand[c] = UnitQuaternion::from_scaled_axis(dtheta) * q[c];
                t_cand[c] = t[c] + Vector3::new(delta[o + 3], delta[o + 4], delta[o + 5]);
            }
            let mut x_cand = x.clone();
            for (p, obs) in pt_obs.iter().enumerate() {
                // δp = −V⁻¹(g_p + Wᵀ δc), the Wᵀδc gathered over the point's
                // observations' camera blocks.
                let mut wt_dc = Vector3::zeros();
                for &a in obs {
                    let ba = blocks[a].ci;
                    let mut dc = SMatrix::<f64, 7, 1>::zeros();
                    dc[0] = delta[0];
                    for r in 0..6 {
                        dc[1 + r] = delta[1 + 6 * ba + r];
                    }
                    wt_dc += w_cp[a].transpose() * dc;
                }
                x_cand[p] = x[p] - v_inv[p] * (g_p[p] + wt_dc);
            }

            let cam_cand = cam_at(cam0, f_cand);
            let new_cost = robust_cost(
                &cam_cand, &q_cand, &t_cand, &x_cand, uv, kept, &obs_ci, &obs_cp, &s2s,
            );
            if new_cost < prev_cost {
                let rel = (prev_cost - new_cost) / prev_cost.max(1e-300);
                f = f_cand;
                q = q_cand;
                t = t_cand;
                x = x_cand;
                cam = cam_cand;
                prev_cost = new_cost;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                // Converged only after tiny improvements twice in a row: a
                // single small step is how a traverse of a nearly-flat
                // valley STARTS (the focal release walks −20% through one),
                // so one is not proof of convergence.
                if rel < 1e-8 {
                    tiny_steps += 1;
                    if tiny_steps >= 2 {
                        lambda = f64::INFINITY;
                    }
                } else {
                    tiny_steps = 0;
                }
                break;
            }
            lambda *= 4.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved || lambda.is_infinite() {
            break;
        }
    }

    // Scatter the compact state back.
    for (c, &i) in img_ids.iter().enumerate() {
        quats[i] = q[c];
        trans[i] = t[c];
    }
    for (c, &p) in pt_ids.iter().enumerate() {
        points[p] = [x[c].x, x[c].y, x[c].z];
    }
    f
}

/// Staged bundle adjustment over images sharing one camera model.
///
/// Per schedule round: retriangulate every point from all supplied
/// observations at the current poses (rounds after the first), trim to
/// observations under `trim_px` with in-front depth and a finite point whose
/// track keeps at least `min_track` survivors, then run one robust sparse LM
/// solve at the round's `loss_scale`. Poses and points are refined in place;
/// the returned [`BundleAdjustment`] carries the focal and the per-observation
/// residual norms at the final state (`+∞` where invalid — and everywhere,
/// with the state passed through, when fewer than `min_obs` observations
/// survive a trim).
///
/// `point_at_infinity` optionally marks per-point directions: a marked row of
/// `points` is a world-frame direction (normalized on input and output) whose
/// observations depend on rotation and camera model only — see "Points at
/// infinity" in `specs/core/bundle-adjustment.md`. An absent or all-`false`
/// mask takes the finite-only code path untouched.
///
/// `protected` optionally marks per-observation protection (parallel to the
/// observation arrays): a protected observation is never removed by the
/// inter-round trim gates — it stays in the solve set every round regardless
/// of its residual and always counts toward `min_track` survival — and passes
/// through the robust loss at the wider scale
/// `protected_loss_scale · loss_scale` (bounded pull, never trimmed nor
/// dominating). See "Protected observations" in
/// `specs/core/bundle-adjustment.md`. An absent or all-`false` mask
/// reproduces the unprotected behavior bit for bit.
///
/// `opt_f` releases the shared focal (SIMPLE_PINHOLE only — the binding
/// rejects other models loudly; the core silently degrades them to a
/// fixed-focal solve).
#[allow(clippy::too_many_arguments)]
pub fn bundle_adjust(
    cam: &CameraIntrinsics,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    point_at_infinity: Option<&[bool]>,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
) -> BundleAdjustment {
    if let Some(mask) = protected {
        assert_eq!(
            mask.len(),
            obs_img.len(),
            "protected and observation length mismatch"
        );
    }
    // An all-`false` protection mask is exactly no mask.
    let protected = protected.filter(|m| m.iter().any(|&b| b));
    if let Some(mask) = point_at_infinity {
        assert_eq!(
            mask.len(),
            points.len(),
            "point_at_infinity and points length mismatch"
        );
        if mask.iter().any(|&b| b) {
            return bundle_adjust_mixed(
                cam,
                quats,
                trans,
                points,
                uv,
                obs_img,
                obs_pt,
                mask,
                protected,
                protected_loss_scale,
                opt_f,
                schedule,
                max_iters,
                min_track,
                min_obs,
            );
        }
    }
    bundle_adjust_finite(
        cam,
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
        protected,
        protected_loss_scale,
        opt_f,
        schedule,
        max_iters,
        min_track,
        min_obs,
    )
}

/// The finite-only staged loop — the original kernel, untouched by the
/// points-at-infinity extension (the dispatcher branches here whenever no
/// direction is marked, preserving it bit for bit).
#[allow(clippy::too_many_arguments)]
fn bundle_adjust_finite(
    cam: &CameraIntrinsics,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
) -> BundleAdjustment {
    let n_obs = obs_img.len();
    assert_eq!(obs_pt.len(), n_obs, "obs_img and obs_pt length mismatch");
    assert_eq!(uv.len(), n_obs, "uv and obs_img length mismatch");
    let is_prot = |k: usize| protected.is_some_and(|m| m[k]);

    // Focal release is SIMPLE_PINHOLE-only: `cam_at` applies `f` to no other
    // model, so an opt_f linearization elsewhere would model a focal DOF the
    // cost does not have (garbage `δf` perturbing every coupled step). The
    // binding rejects this loudly; at the core boundary it degrades to a
    // fixed-focal solve.
    let opt_f = opt_f && matches!(cam.model, CameraModel::SimplePinhole { .. });

    let mut f = cam.focal_lengths().0;

    for (rnd, stage) in schedule.iter().enumerate() {
        let cam_now = cam_at(cam, f);
        if rnd > 0 {
            retriangulate(&cam_now, quats, trans, points, uv, obs_img, obs_pt);
        }
        let (norms, depths) =
            residual_norms_depths(&cam_now, quats, trans, points, uv, obs_img, obs_pt);
        // Protected observations bypass the trim gates entirely: they stay in
        // the solve set every round regardless of residual or depth.
        let mut keep: Vec<bool> = (0..n_obs)
            .map(|k| is_prot(k) || (norms[k] < stage.trim_px && depths[k] > 1e-3 * f))
            .collect();
        // Track survival: drop observations of points with < min_track kept
        // (protected observations count as survivors and are never dropped).
        let mut surv = vec![0usize; points.len()];
        for k in 0..n_obs {
            if keep[k] {
                surv[obs_pt[k] as usize] += 1;
            }
        }
        for k in 0..n_obs {
            keep[k] = keep[k] && (is_prot(k) || surv[obs_pt[k] as usize] >= min_track);
        }
        let kept: Vec<usize> = (0..n_obs).filter(|&k| keep[k]).collect();
        if kept.len() < min_obs {
            // Degenerate (e.g. a wildly wrong focal): state passes through.
            return BundleAdjustment {
                focal: f,
                residual_norms: vec![f64::INFINITY; n_obs],
            };
        }
        f = solve_lm(
            cam,
            f,
            quats,
            trans,
            points,
            uv,
            obs_img,
            obs_pt,
            &kept,
            opt_f,
            stage.loss_scale,
            max_iters,
            protected,
            protected_loss_scale,
        );
    }

    let cam_final = cam_at(cam, f);
    let (norms, _depths) =
        residual_norms_depths(&cam_final, quats, trans, points, uv, obs_img, obs_pt);
    let residual_norms = norms
        .iter()
        .map(|&r| {
            if r >= INVALID_RESIDUAL {
                f64::INFINITY
            } else {
                r
            }
        })
        .collect();
    BundleAdjustment {
        focal: f,
        residual_norms,
    }
}

// ── Points at infinity ──────────────────────────────────────────────────────
//
// The mixed path below handles per-point direction masks
// (`specs/core/bundle-adjustment.md`, "Points at infinity"). A marked row of
// `points` is a world-frame direction `d` projecting as
// `uv_pred = ray_to_pixel(R·d)` — no translation dependence — parameterized
// by a 2-DOF tangent-plane perturbation `d ← normalize(d + B(d)·δ)`. It is
// only entered when at least one direction is marked; the finite-only kernel
// above stays bit-for-bit intact otherwise.

/// Normalize a direction row. Zero-norm and non-finite rows come back `NaN`
/// (a `NaN` direction behaves like a `NaN` finite point: invalid until
/// re-estimated).
fn normalized_dir(p: [f64; 3]) -> [f64; 3] {
    let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        [p[0] / n, p[1] / n, p[2] / n]
    } else {
        [f64::NAN; 3]
    }
}

/// An orthonormal basis `B(d) = [b1 | b2]` of the tangent plane `d⊥` of a
/// unit direction (rebuilt at each linearization).
fn tangent_basis(d: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let anchor = if d.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let b1 = d.cross(&anchor).normalize();
    let b2 = d.cross(&b1);
    (b1, b2)
}

/// Mixed-path residual norms and in-front measures. Finite observations
/// report the canonical depth `−z_cam` (checked against the `1e-3·f` floor by
/// the caller); direction observations report `−(R·d)_z` (cheirality: any
/// positive value is in front). Invalid observations report
/// `INVALID_RESIDUAL` and a non-positive in-front measure.
#[allow(clippy::too_many_arguments)]
fn residual_norms_depths_mixed(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[[f64; 3]],
    is_dir: &[bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
) -> (Vec<f64>, Vec<f64>) {
    let n_obs = obs_img.len();
    let mut norms = vec![INVALID_RESIDUAL; n_obs];
    let mut depths = vec![f64::NEG_INFINITY; n_obs];
    for k in 0..n_obs {
        let pi = obs_pt[k] as usize;
        let p = points[pi];
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }
        let i = obs_img[k] as usize;
        let rot = quats[i] * Vector3::new(p[0], p[1], p[2]);
        let c = if is_dir[pi] { rot } else { rot + trans[i] };
        depths[k] = -c.z;
        if let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) {
            norms[k] = (u - uv[k][0]).hypot(v - uv[k][1]);
        }
    }
    (norms, depths)
}

/// Mixed-path re-estimation (rounds after the first): finite points rebuild
/// by ray-midpoint batch triangulation exactly as [`retriangulate`];
/// direction points re-estimate in closed form as the normalized mean of
/// their observations' back-rotated rays `R_iᵀ · pixel_to_ray(uv)`. Tracks
/// with fewer than two observations — and points with none — become `NaN`.
#[allow(clippy::too_many_arguments)]
fn reestimate_points(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &[bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
) {
    let n_obs = obs_img.len();
    let mut order: Vec<u32> = (0..n_obs as u32).collect();
    order.sort_unstable_by_key(|&k| obs_pt[k as usize]);

    // Direction means accumulate directly; finite tracks feed the batch
    // triangulation exactly as in the finite path.
    let mut dir_sum: Vec<(usize, Vector3<f64>, usize)> = Vec::new();
    let mut dirs = Vec::new();
    let mut centers = Vec::new();
    let mut offsets = Vec::new();
    let mut track_pt = Vec::new();
    let mut prev: Option<u32> = None;
    for &k in &order {
        let k = k as usize;
        let p = obs_pt[k];
        let pu = p as usize;
        if prev != Some(p) {
            if is_dir[pu] {
                dir_sum.push((pu, Vector3::zeros(), 0));
            } else {
                offsets.push(dirs.len());
                track_pt.push(pu);
            }
            prev = Some(p);
        }
        let i = obs_img[k] as usize;
        let r_inv = quats[i].inverse();
        let d = cam.pixel_to_ray(uv[k][0], uv[k][1]);
        let world_ray = r_inv * Vector3::new(d[0], d[1], d[2]);
        if is_dir[pu] {
            let last = dir_sum.last_mut().unwrap();
            last.1 += world_ray;
            last.2 += 1;
        } else {
            dirs.push(world_ray);
            centers.push(Point3::from(-(r_inv * trans[i])));
        }
    }
    offsets.push(dirs.len());

    for p in points.iter_mut() {
        *p = [f64::NAN; 3];
    }
    let tris = triangulate_batch(&dirs, &centers, &offsets);
    for (t, tri) in tris.iter().enumerate() {
        if offsets[t + 1] - offsets[t] >= 2 {
            points[track_pt[t]] = [tri.point.x, tri.point.y, tri.point.z];
        }
    }
    for &(p, sum, count) in &dir_sum {
        if count >= 2 {
            let mean = sum / count as f64;
            points[p] = normalized_dir([mean.x, mean.y, mean.z]);
        }
    }
}

/// Mixed-path robust cost over the kept observations at a candidate state
/// (`cp_dir` flags direction points by compact index; `s2s` is the
/// per-kept-observation squared loss scale as in [`robust_cost`]).
#[allow(clippy::too_many_arguments)]
fn robust_cost_mixed(
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &[Vector3<f64>],
    cp_dir: &[bool],
    uv: &[[f64; 2]],
    kept: &[usize],
    obs_ci: &[usize],
    obs_cp: &[usize],
    s2s: &[f64],
) -> f64 {
    kept.iter()
        .enumerate()
        .map(|(kk, &k)| {
            let s2 = s2s[kk];
            let p = points[obs_cp[kk]];
            // A non-finite point (possible only for protected observations,
            // which the trim never excludes) is penalized like an
            // out-of-domain projection.
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                return s2 * rho(INVALID_RESIDUAL * INVALID_RESIDUAL / s2);
            }
            let rot = quats[obs_ci[kk]] * p;
            let c = if cp_dir[obs_cp[kk]] {
                rot
            } else {
                rot + trans[obs_ci[kk]]
            };
            match cam.ray_to_pixel([c.x, c.y, c.z]) {
                Some((u, v)) => {
                    let dx = u - uv[k][0];
                    let dy = v - uv[k][1];
                    s2 * (rho(dx * dx / s2) + rho(dy * dy / s2))
                }
                None => s2 * rho(INVALID_RESIDUAL * INVALID_RESIDUAL / s2),
            }
        })
        .sum()
}

/// One robust sparse LM solve over the kept observations with mixed finite
/// and direction points. Direction points use 2-DOF tangent-plane parameters
/// stored in the first two slots of the uniform 3-wide point block (the third
/// slot carries exact zeros and is pinned at the Schur inversion); images
/// whose kept observations are all directions have their translation slots
/// pinned in the reduced system (frozen for the round).
#[allow(clippy::too_many_arguments)]
fn solve_lm_mixed(
    cam0: &CameraIntrinsics,
    f0: f64,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    is_dir: &[bool],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    kept: &[usize],
    opt_f: bool,
    loss_scale: f64,
    max_iters: usize,
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
) -> f64 {
    // Compact the images and points the kept observations touch.
    let mut img_ids: Vec<usize> = kept.iter().map(|&k| obs_img[k] as usize).collect();
    img_ids.sort_unstable();
    img_ids.dedup();
    let mut pt_ids: Vec<usize> = kept.iter().map(|&k| obs_pt[k] as usize).collect();
    pt_ids.sort_unstable();
    pt_ids.dedup();
    let n_im = img_ids.len();
    let n_pt = pt_ids.len();
    let ci_of: std::collections::HashMap<usize, usize> =
        img_ids.iter().enumerate().map(|(c, &i)| (i, c)).collect();
    let cp_of: std::collections::HashMap<usize, usize> =
        pt_ids.iter().enumerate().map(|(c, &p)| (p, c)).collect();
    let obs_ci: Vec<usize> = kept
        .iter()
        .map(|&k| ci_of[&(obs_img[k] as usize)])
        .collect();
    let obs_cp: Vec<usize> = kept.iter().map(|&k| cp_of[&(obs_pt[k] as usize)]).collect();
    let cp_dir: Vec<bool> = pt_ids.iter().map(|&p| is_dir[p]).collect();

    // Translation observability: an image whose kept observations are all
    // directions gets its translation pinned for this round (a direction
    // observation's translation Jacobian is identically zero, so the block
    // would otherwise be pure zero curvature in the reduced system).
    let mut img_has_finite = vec![false; n_im];
    for (kk, &ci) in obs_ci.iter().enumerate() {
        if !cp_dir[obs_cp[kk]] {
            img_has_finite[ci] = true;
        }
    }
    let any_frozen = img_has_finite.iter().any(|&h| !h);

    // Per-point observation lists (compact indices into `kept`).
    let mut pt_obs: Vec<Vec<usize>> = vec![Vec::new(); n_pt];
    for (kk, &cp) in obs_cp.iter().enumerate() {
        pt_obs[cp].push(kk);
    }

    // Working state (compact copies). Direction rows arrive unit-normalized
    // (input normalization / re-estimation) and every accepted step
    // re-normalizes them.
    let mut f = f0;
    let mut q: Vec<UnitQuaternion<f64>> = img_ids.iter().map(|&i| quats[i]).collect();
    let mut t: Vec<Vector3<f64>> = img_ids.iter().map(|&i| trans[i]).collect();
    let mut x: Vec<Vector3<f64>> = pt_ids
        .iter()
        .map(|&p| Vector3::new(points[p][0], points[p][1], points[p][2]))
        .collect();

    // Reduced camera system: [f | 6 per image], the focal slot always present
    // (pinned when !opt_f) to keep the indexing uniform.
    let d = 1 + 6 * n_im;
    // Per-kept-observation squared loss scale: the round's scale everywhere,
    // widened by `protected_loss_scale` for protected observations.
    let s2 = loss_scale * loss_scale;
    let s2s: Vec<f64> = kept
        .iter()
        .map(|&k| match protected {
            Some(m) if m[k] => {
                let s = loss_scale * protected_loss_scale;
                s * s
            }
            _ => s2,
        })
        .collect();
    let mut lambda = 1e-3;
    let mut tiny_steps = 0usize;
    let mut cam = cam_at(cam0, f);
    let mut prev_cost =
        robust_cost_mixed(&cam, &q, &t, &x, &cp_dir, uv, kept, &obs_ci, &obs_cp, &s2s);

    let analytic = cam.model.supports_pixel_jacobian();
    for _ in 0..max_iters {
        // ── Linearize at the current state ───────────────────────────────
        // Tangent bases B(d) = [b1 | b2] for the direction points, rebuilt at
        // each linearization.
        let bases: Vec<(Vector3<f64>, Vector3<f64>)> = x
            .iter()
            .zip(&cp_dir)
            .map(|(xd, &dir)| {
                if dir {
                    tangent_basis(xd)
                } else {
                    (Vector3::zeros(), Vector3::zeros())
                }
            })
            .collect();
        let (cx, cy) = cam.principal_point();
        let blocks: Vec<ObsBlocks> = kept
            .iter()
            .enumerate()
            .map(|(kk, &k)| {
                let ci = obs_ci[kk];
                let cp = obs_cp[kk];
                let s2 = s2s[kk];
                let dir = cp_dir[cp];
                let rot_pt = q[ci] * x[cp];
                let p_cam = if dir { rot_pt } else { rot_pt + t[ci] };
                let mut res = Vector2::new(INVALID_RESIDUAL, 0.0);
                let mut cam_j = SMatrix::<f64, 2, 7>::zeros();
                let mut pt_j = SMatrix::<f64, 2, 3>::zeros();
                // A non-finite point (protected observations only — the trim
                // never excludes them) keeps the penalized residual and zero
                // Jacobian rows: penalized, never steering.
                let proj = if x[cp].x.is_finite() && x[cp].y.is_finite() && x[cp].z.is_finite() {
                    project_with_jac(&cam, p_cam, analytic)
                } else {
                    None
                };
                if let Some(((u, v), jp)) = proj {
                    res = Vector2::new(u - uv[k][0], v - uv[k][1]);
                    let jp = SMatrix::<f64, 2, 3>::from_rows(&[
                        SMatrix::<f64, 1, 3>::from_row_slice(&jp[0]),
                        SMatrix::<f64, 1, 3>::from_row_slice(&jp[1]),
                    ]);
                    if opt_f {
                        cam_j[(0, 0)] = (u - cx) / f;
                        cam_j[(1, 0)] = (v - cy) / f;
                    }
                    // Rotation block: ∂p_cam/∂δθ = −[R·X]ₓ (finite) or
                    // −[R·d]ₓ (direction) — same composition either way.
                    let nskew = Matrix3::new(
                        0.0, rot_pt.z, -rot_pt.y, //
                        -rot_pt.z, 0.0, rot_pt.x, //
                        rot_pt.y, -rot_pt.x, 0.0,
                    );
                    cam_j.fixed_view_mut::<2, 3>(0, 1).copy_from(&(jp * nskew));
                    let r_mat: Matrix3<f64> = q[ci].to_rotation_matrix().into_inner();
                    if dir {
                        // Translation block: zero (a direction observes no
                        // translation). Point block: 2-DOF tangent-plane
                        // parameters, ∂p_cam/∂δ = R·B(d) (columns b1, b2;
                        // the third slot stays exactly zero).
                        let (b1, b2) = bases[cp];
                        let col0 = jp * (r_mat * b1);
                        let col1 = jp * (r_mat * b2);
                        pt_j.set_column(0, &col0);
                        pt_j.set_column(1, &col1);
                    } else {
                        // Translation block: identity.
                        cam_j.fixed_view_mut::<2, 3>(0, 4).copy_from(&jp);
                        // Point block: ∂p_cam/∂X = R.
                        pt_j.copy_from(&(jp * r_mat));
                    }
                }
                for row in 0..2 {
                    let z = res[row] * res[row] / s2;
                    let (js, rs) = robust_scales(z);
                    res[row] *= rs;
                    for col in 0..7 {
                        cam_j[(row, col)] *= js;
                    }
                    for col in 0..3 {
                        pt_j[(row, col)] *= js;
                    }
                }
                ObsBlocks {
                    ci,
                    cp,
                    res,
                    cam_j,
                    pt_j,
                }
            })
            .collect();

        // ── Accumulate the normal-equation blocks ────────────────────────
        let mut h_cc = DMatrix::<f64>::zeros(d, d);
        let mut g_c = DVector::<f64>::zeros(d);
        let mut v_pp: Vec<Matrix3<f64>> = vec![Matrix3::zeros(); n_pt];
        let mut g_p: Vec<Vector3<f64>> = vec![Vector3::zeros(); n_pt];
        let mut w_cp: Vec<SMatrix<f64, 7, 3>> = Vec::with_capacity(blocks.len());
        for b in &blocks {
            let idx = [
                0,
                1 + 6 * b.ci,
                2 + 6 * b.ci,
                3 + 6 * b.ci,
                4 + 6 * b.ci,
                5 + 6 * b.ci,
                6 + 6 * b.ci,
            ];
            let h_local = b.cam_j.transpose() * b.cam_j;
            let g_local = b.cam_j.transpose() * b.res;
            for (a, &ia) in idx.iter().enumerate() {
                g_c[ia] += g_local[a];
                for (c, &ic) in idx.iter().enumerate() {
                    h_cc[(ia, ic)] += h_local[(a, c)];
                }
            }
            v_pp[b.cp] += b.pt_j.transpose() * b.pt_j;
            g_p[b.cp] += b.pt_j.transpose() * b.res;
            w_cp.push(b.cam_j.transpose() * b.pt_j);
        }

        // ── Damping ladder: re-damp and re-solve from this linearization ──
        let mut improved = false;
        for _ in 0..12 {
            let mut s = h_cc.clone();
            for dd in 0..d {
                s[(dd, dd)] += lambda * h_cc[(dd, dd)].max(1e-12);
            }
            let mut g_red = g_c.clone();
            // Schur-eliminate the points. A direction's block is 2×2 in the
            // first two slots (its third row/column carry exact zeros); the
            // third diagonal is pinned to 1 so the uniform 3×3 inversion
            // stays regular while contributing an exactly-zero update.
            let mut v_inv: Vec<Matrix3<f64>> = Vec::with_capacity(n_pt);
            let mut singular = false;
            for (p, v) in v_pp.iter().enumerate() {
                let mut vd = *v;
                for dd in 0..3 {
                    vd[(dd, dd)] += lambda * v[(dd, dd)].max(1e-12);
                }
                if cp_dir[p] {
                    vd[(2, 2)] = 1.0;
                }
                match vd.try_inverse() {
                    Some(inv) => v_inv.push(inv),
                    None => {
                        singular = true;
                        break;
                    }
                }
            }
            if singular {
                lambda *= 4.0;
                continue;
            }
            for (p, obs) in pt_obs.iter().enumerate() {
                let y = v_inv[p] * g_p[p];
                for &a in obs {
                    let wa = &w_cp[a];
                    let ba = blocks[a].ci;
                    let ia = [
                        0,
                        1 + 6 * ba,
                        2 + 6 * ba,
                        3 + 6 * ba,
                        4 + 6 * ba,
                        5 + 6 * ba,
                        6 + 6 * ba,
                    ];
                    let contrib = wa * y;
                    for (r, &ir) in ia.iter().enumerate() {
                        g_red[ir] -= contrib[r];
                    }
                    for &b in obs {
                        let m = wa * v_inv[p] * w_cp[b].transpose();
                        let bb = blocks[b].ci;
                        let ib = [
                            0,
                            1 + 6 * bb,
                            2 + 6 * bb,
                            3 + 6 * bb,
                            4 + 6 * bb,
                            5 + 6 * bb,
                            6 + 6 * bb,
                        ];
                        for (r, &ir) in ia.iter().enumerate() {
                            for (c, &ic) in ib.iter().enumerate() {
                                s[(ir, ic)] -= m[(r, c)];
                            }
                        }
                    }
                }
            }
            if !opt_f {
                // Pin the focal slot.
                for dd in 0..d {
                    s[(0, dd)] = 0.0;
                    s[(dd, 0)] = 0.0;
                }
                s[(0, 0)] = 1.0;
                g_red[0] = 0.0;
            }
            if any_frozen {
                // Pin the translation slots of all-direction images (frozen
                // for the round; their rotations still update).
                for (c, &has_finite) in img_has_finite.iter().enumerate() {
                    if has_finite {
                        continue;
                    }
                    for r in 0..3 {
                        let slot = 4 + 6 * c + r;
                        for dd in 0..d {
                            s[(slot, dd)] = 0.0;
                            s[(dd, slot)] = 0.0;
                        }
                        s[(slot, slot)] = 1.0;
                        g_red[slot] = 0.0;
                    }
                }
            }

            let Some(delta) = s.lu().solve(&(-g_red)) else {
                lambda *= 4.0;
                continue;
            };

            // Candidate state.
            let f_cand = if opt_f { f + delta[0] } else { f };
            if opt_f && !(f_cand.is_finite() && f_cand > 1e-6) {
                lambda *= 4.0;
                continue;
            }
            let mut q_cand = q.clone();
            let mut t_cand = t.clone();
            for c in 0..n_im {
                let o = 1 + 6 * c;
                let dtheta = Vector3::new(delta[o], delta[o + 1], delta[o + 2]);
                q_cand[c] = UnitQuaternion::from_scaled_axis(dtheta) * q[c];
                if img_has_finite[c] {
                    t_cand[c] = t[c] + Vector3::new(delta[o + 3], delta[o + 4], delta[o + 5]);
                }
                // Frozen images keep their translation untouched (a `+ 0.0`
                // would still flip the sign of a `−0.0` component).
            }
            let mut x_cand = x.clone();
            for (p, obs) in pt_obs.iter().enumerate() {
                // δp = −V⁻¹(g_p + Wᵀ δc), the Wᵀδc gathered over the point's
                // observations' camera blocks.
                let mut wt_dc = Vector3::zeros();
                for &a in obs {
                    let ba = blocks[a].ci;
                    let mut dc = SMatrix::<f64, 7, 1>::zeros();
                    dc[0] = delta[0];
                    for r in 0..6 {
                        dc[1 + r] = delta[1 + 6 * ba + r];
                    }
                    wt_dc += w_cp[a].transpose() * dc;
                }
                let dxp = v_inv[p] * (g_p[p] + wt_dc);
                if cp_dir[p] {
                    // d ← normalize(d + B(d)·δ), the 2-DOF tangent update.
                    let (b1, b2) = bases[p];
                    x_cand[p] = (x[p] - (b1 * dxp[0] + b2 * dxp[1])).normalize();
                } else {
                    x_cand[p] = x[p] - dxp;
                }
            }

            let cam_cand = cam_at(cam0, f_cand);
            let new_cost = robust_cost_mixed(
                &cam_cand, &q_cand, &t_cand, &x_cand, &cp_dir, uv, kept, &obs_ci, &obs_cp, &s2s,
            );
            if new_cost < prev_cost {
                let rel = (prev_cost - new_cost) / prev_cost.max(1e-300);
                f = f_cand;
                q = q_cand;
                t = t_cand;
                x = x_cand;
                cam = cam_cand;
                prev_cost = new_cost;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                // Converged only after tiny improvements twice in a row (see
                // the finite path).
                if rel < 1e-8 {
                    tiny_steps += 1;
                    if tiny_steps >= 2 {
                        lambda = f64::INFINITY;
                    }
                } else {
                    tiny_steps = 0;
                }
                break;
            }
            lambda *= 4.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved || lambda.is_infinite() {
            break;
        }
    }

    // Scatter the compact state back.
    for (c, &i) in img_ids.iter().enumerate() {
        quats[i] = q[c];
        trans[i] = t[c];
    }
    for (c, &p) in pt_ids.iter().enumerate() {
        points[p] = [x[c].x, x[c].y, x[c].z];
    }
    f
}

/// The mixed staged loop (at least one direction marked): the finite loop's
/// semantics with direction-aware residuals, trims, re-estimation, and the
/// finite-survivors-only `min_obs` floor.
#[allow(clippy::too_many_arguments)]
fn bundle_adjust_mixed(
    cam: &CameraIntrinsics,
    quats: &mut [UnitQuaternion<f64>],
    trans: &mut [Vector3<f64>],
    points: &mut [[f64; 3]],
    uv: &[[f64; 2]],
    obs_img: &[u32],
    obs_pt: &[u32],
    is_dir: &[bool],
    protected: Option<&[bool]>,
    protected_loss_scale: f64,
    opt_f: bool,
    schedule: &[BaSchedule],
    max_iters: usize,
    min_track: usize,
    min_obs: usize,
) -> BundleAdjustment {
    let n_obs = obs_img.len();
    assert_eq!(obs_pt.len(), n_obs, "obs_img and obs_pt length mismatch");
    assert_eq!(uv.len(), n_obs, "uv and obs_img length mismatch");
    let is_prot = |k: usize| protected.is_some_and(|m| m[k]);

    // Direction rows are world-frame directions: normalized on input (and
    // kept normalized throughout, so they return normalized too).
    for (p, row) in points.iter_mut().enumerate() {
        if is_dir[p] {
            *row = normalized_dir(*row);
        }
    }

    let opt_f = opt_f && matches!(cam.model, CameraModel::SimplePinhole { .. });

    let mut f = cam.focal_lengths().0;

    for (rnd, stage) in schedule.iter().enumerate() {
        let cam_now = cam_at(cam, f);
        if rnd > 0 {
            reestimate_points(&cam_now, quats, trans, points, is_dir, uv, obs_img, obs_pt);
        }
        let (norms, depths) = residual_norms_depths_mixed(
            &cam_now, quats, trans, points, is_dir, uv, obs_img, obs_pt,
        );
        // In-front: canonical depth over the 1e-3·f floor for finite
        // observations; cheirality (R·d)_z < 0 for directions. Protected
        // observations bypass the trim gates entirely.
        let mut keep: Vec<bool> = (0..n_obs)
            .map(|k| {
                let floor = if is_dir[obs_pt[k] as usize] {
                    0.0
                } else {
                    1e-3 * f
                };
                is_prot(k) || (norms[k] < stage.trim_px && depths[k] > floor)
            })
            .collect();
        // Track survival: drop observations of points with < min_track kept
        // (protected observations count as survivors and are never dropped).
        let mut surv = vec![0usize; points.len()];
        for k in 0..n_obs {
            if keep[k] {
                surv[obs_pt[k] as usize] += 1;
            }
        }
        for k in 0..n_obs {
            keep[k] = keep[k] && (is_prot(k) || surv[obs_pt[k] as usize] >= min_track);
        }
        let kept: Vec<usize> = (0..n_obs).filter(|&k| keep[k]).collect();
        // The degenerate floor counts finite-point survivors only: direction
        // observations constrain no structure depth, so they cannot vouch
        // for a usable finite solve.
        let kept_finite = kept
            .iter()
            .filter(|&&k| !is_dir[obs_pt[k] as usize])
            .count();
        if kept_finite < min_obs {
            // Degenerate (e.g. a wildly wrong focal): state passes through.
            return BundleAdjustment {
                focal: f,
                residual_norms: vec![f64::INFINITY; n_obs],
            };
        }
        f = solve_lm_mixed(
            cam,
            f,
            quats,
            trans,
            points,
            is_dir,
            uv,
            obs_img,
            obs_pt,
            &kept,
            opt_f,
            stage.loss_scale,
            max_iters,
            protected,
            protected_loss_scale,
        );
    }

    let cam_final = cam_at(cam, f);
    let (norms, _depths) = residual_norms_depths_mixed(
        &cam_final, quats, trans, points, is_dir, uv, obs_img, obs_pt,
    );
    let residual_norms = norms
        .iter()
        .map(|&r| {
            if r >= INVALID_RESIDUAL {
                f64::INFINITY
            } else {
                r
            }
        })
        .collect();
    BundleAdjustment {
        focal: f,
        residual_norms,
    }
}

#[cfg(test)]
mod tests;
