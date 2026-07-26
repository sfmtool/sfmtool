// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Group consistency: is the census's cross-group disagreement *coherent*?
//!
//! The census score says how much cross-group evidence a candidate leaves
//! unsatisfied. This companion asks whether that disagreement is explainable by
//! group-level pose error. Jointly over all viewpoint groups it estimates the
//! per-group 7-dof similarity (rotation, translation, log scale) that best
//! satisfies the eligible bridges, with the largest group holding the identity
//! so the gauge is fixed, and reports how much of the disagreement the
//! corrections explain.
//!
//! A correction `(Q, t, s)` acts on its group's content as the world similarity
//! `W(x) = s·Q·x + t`; equivalently, on that group's cameras,
//!
//! ```text
//! R' = R·Qᵀ        C' = s·Q·C + t
//! ```
//!
//! which leaves the group's own projections untouched (its structure moves with
//! it) and changes only where its rays meet the other groups'. Bridges are
//! therefore **re-triangulated** at the corrected poses and re-scored; there is
//! no fixed structure to hold on to.
//!
//! The estimate is a Levenberg–Marquardt descent on a soft-L1 cost over the
//! bridges' per-observation pixel residuals — 7 × (n_groups − 1) parameters, so
//! the dense normal equations are trivial and a central-difference Jacobian of
//! the (smooth) triangulate-and-project chain is affordable. Robustness matters
//! more than the last digit of convergence here: the fit population is the
//! eligible bridges, which by construction include whatever false matches
//! survived the census's eligibility screen, and the answer being extracted is
//! coherent-or-not rather than a pose to ship.
//!
//! Deterministic: corrections start at the identity, the fit subsample is a
//! fixed stride, and every reduction is a fixed function of the inputs.

use nalgebra::{DMatrix, DVector, Point3, UnitQuaternion, Vector3};

use super::{median_in_place, GroupConsistency, GroupCorrection, INVALID_RESIDUAL_PX};
use crate::reconstruction::triangulation::triangulate_batch;
use crate::CameraIntrinsics;

/// Largest bridge population the robust fit runs on. Beyond it the fit set is
/// strided down (the *scoring* set stays complete) — the corrections are a
/// 7-dof-per-group quantity, so a few hundred bridges already over-determine
/// them and the Jacobian cost is linear in the fit set.
const MAX_FIT_BRIDGES: usize = 1200;

/// Soft-L1 transition scale (px) of the robust cost. Residuals well under it
/// are quadratic, residuals well over it linear — set above the `sat_px` bar so
/// a satisfied bridge is inside the quadratic regime and a false match cannot
/// drag the solve.
const ROBUST_SCALE_PX: f64 = 3.0;

/// Levenberg–Marquardt iteration cap.
const MAX_ITER: usize = 30;

/// Damping retries per iteration before giving up on a descent direction.
const MAX_DAMPING_STEPS: usize = 12;

/// Central-difference step for the numeric Jacobian. The parameters are
/// radians, scene-relative translation, and log scale — all `O(1)` — so one
/// step serves every block.
const JACOBIAN_STEP: f64 = 1e-6;

/// Everything the solve needs from the census pass. Observation arrays are the
/// census's posed-observation arrays (`obs_*`, `dirs` in world frame at the
/// candidate), segmented CSR-style by `seg_offsets`.
pub(super) struct GroupConsistencyInput<'a> {
    /// Shared camera model of the candidate.
    pub camera: &'a CameraIntrinsics,
    /// World-to-camera rotation per image (unposed entries unused).
    pub quats: &'a [UnitQuaternion<f64>],
    /// Group id per image, `-1` for an unposed image.
    pub group_of: &'a [i32],
    /// Number of viewpoint groups.
    pub n_groups: usize,
    /// Camera centers of the posed images (for the scene scale).
    pub posed_centers: &'a [Point3<f64>],
    /// Unit world-frame ray per posed observation at the candidate.
    pub dirs: &'a [Vector3<f64>],
    /// Camera center per posed observation.
    pub obs_center: &'a [Point3<f64>],
    /// Image id per posed observation.
    pub obs_image: &'a [u32],
    /// Observed pixels per posed observation, flattened `(u, v)`.
    pub obs_uv: &'a [f64],
    /// CSR segment offsets into the posed-observation arrays.
    pub seg_offsets: &'a [usize],
    /// Segments to solve and score on: the eligible, measurable bridges.
    pub eval_segs: &'a [usize],
    /// Parallel to `eval_segs`: does the bridge clear the parallax floor?
    pub eval_hi_parallax: &'a [bool],
    /// Satisfied / unsatisfied bar on a bridge's median residual (px).
    pub sat_px: f64,
}

/// One evaluation of the corrected placement: per-observation pixel residuals
/// (`None` outside the camera model's domain) and per-segment median residual
/// norms.
struct Evaluation {
    res: Vec<Option<[f64; 2]>>,
    med: Vec<f64>,
}

/// The packed problem: which observations are evaluated, which of them are
/// fitted, and which group owns which parameter block.
struct Solver<'a> {
    input: &'a GroupConsistencyInput<'a>,
    /// Parameter block per group id; `None` for the gauge group.
    block_of_group: Vec<Option<usize>>,
    /// Packed observation order — indices into the census's posed-observation
    /// arrays, grouped by evaluated segment.
    obs: Vec<usize>,
    /// CSR offsets into `obs`, one run per evaluated segment.
    offsets: Vec<usize>,
    /// Parallel to `obs`: does the observation enter the robust fit?
    in_fit: Vec<bool>,
    /// World units per unit of translation parameter.
    scene: f64,
    /// `7 × (n_groups − 1)`.
    n_params: usize,
}

impl Solver<'_> {
    /// Corrected placement at `p`: re-triangulate every evaluated segment from
    /// its corrected rays and centers, then reproject.
    fn evaluate(&self, p: &[f64]) -> Evaluation {
        let n_blocks = self.n_params / 7;
        let mut rot = Vec::with_capacity(n_blocks);
        let mut scale = Vec::with_capacity(n_blocks);
        let mut shift = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let o = b * 7;
            rot.push(UnitQuaternion::from_scaled_axis(Vector3::new(
                p[o],
                p[o + 1],
                p[o + 2],
            )));
            shift.push(Vector3::new(p[o + 3], p[o + 4], p[o + 5]) * self.scene);
            scale.push(p[o + 6].exp());
        }

        let n = self.obs.len();
        let mut dirs = Vec::with_capacity(n);
        let mut centers = Vec::with_capacity(n);
        let mut rots = Vec::with_capacity(n);
        for &o in &self.obs {
            let img = self.input.obs_image[o] as usize;
            let g = self.input.group_of[img];
            let block = if g >= 0 {
                self.block_of_group[g as usize]
            } else {
                None
            };
            match block {
                Some(b) => {
                    dirs.push(rot[b] * self.input.dirs[o]);
                    centers.push(Point3::from(
                        scale[b] * (rot[b] * self.input.obs_center[o].coords) + shift[b],
                    ));
                    rots.push(self.input.quats[img] * rot[b].inverse());
                }
                None => {
                    dirs.push(self.input.dirs[o]);
                    centers.push(self.input.obs_center[o]);
                    rots.push(self.input.quats[img]);
                }
            }
        }

        let tris = triangulate_batch(&dirs, &centers, &self.offsets);
        let n_seg = self.offsets.len() - 1;
        let mut res = vec![None; n];
        let mut med = vec![INVALID_RESIDUAL_PX; n_seg];
        let mut buf: Vec<f64> = Vec::new();
        for (s, m) in med.iter_mut().enumerate() {
            let (lo, hi) = (self.offsets[s], self.offsets[s + 1]);
            let x = tris[s].point;
            let finite = x.x.is_finite() && x.y.is_finite() && x.z.is_finite();
            buf.clear();
            for k in lo..hi {
                // `R·X + t` with `t = −R·C`, the same form the census's
                // per-observation residuals use, so the identity correction
                // reproduces them exactly.
                let r = if finite {
                    let rp = rots[k];
                    let t = -(rp * centers[k].coords);
                    let xc = rp * x.coords + t;
                    self.input
                        .camera
                        .ray_to_pixel([xc.x, xc.y, xc.z])
                        .map(|(u, v)| {
                            let o = self.obs[k] * 2;
                            [u - self.input.obs_uv[o], v - self.input.obs_uv[o + 1]]
                        })
                } else {
                    None
                };
                buf.push(match r {
                    Some(v) => (v[0] * v[0] + v[1] * v[1]).sqrt(),
                    None => INVALID_RESIDUAL_PX,
                });
                res[k] = r;
            }
            *m = median_in_place(&mut buf);
        }
        Evaluation { res, med }
    }

    /// Soft-L1 cost over the fit observations: `Σ 2·f²·(√(1 + ‖r‖²/f²) − 1)`.
    /// An observation outside the model's domain contributes the cost of
    /// [`INVALID_RESIDUAL_PX`], which makes a step that pushes structure behind
    /// a camera unacceptable rather than merely expensive.
    fn cost(&self, ev: &Evaluation) -> f64 {
        let fs2 = ROBUST_SCALE_PX * ROBUST_SCALE_PX;
        let mut total = 0.0;
        for (k, r) in ev.res.iter().enumerate() {
            if !self.in_fit[k] {
                continue;
            }
            let n2 = match r {
                Some(r) => r[0] * r[0] + r[1] * r[1],
                None => INVALID_RESIDUAL_PX * INVALID_RESIDUAL_PX,
            };
            total += 2.0 * fs2 * ((1.0 + n2 / fs2).sqrt() - 1.0);
        }
        total
    }

    /// Levenberg–Marquardt from the identity correction. The soft-L1 loss
    /// enters as the Gauss–Newton reweighting `w = 1/√(1 + ‖r‖²/f²)` of each
    /// observation's normal-equation contribution; observations outside the
    /// model's domain get a zero Jacobian row, so they penalize the cost
    /// without steering the step.
    fn solve(&self) -> Vec<f64> {
        let np = self.n_params;
        let mut p = vec![0.0f64; np];
        let fit_obs: Vec<usize> = (0..self.obs.len()).filter(|&k| self.in_fit[k]).collect();
        if np == 0 || fit_obs.is_empty() {
            return p;
        }
        let fs2 = ROBUST_SCALE_PX * ROBUST_SCALE_PX;
        let mut ev = self.evaluate(&p);
        let mut prev = self.cost(&ev);
        let mut lambda = 1e-3f64;

        for _ in 0..MAX_ITER {
            // Central-difference Jacobian of the triangulate-and-project chain,
            // rows in `fit_obs` order, two rows (du, dv) per observation.
            let m = fit_obs.len();
            let mut jac = vec![0.0f64; m * 2 * np];
            for j in 0..np {
                let mut pp = p.clone();
                pp[j] += JACOBIAN_STEP;
                let mut pm = p.clone();
                pm[j] -= JACOBIAN_STEP;
                let ep = self.evaluate(&pp);
                let em = self.evaluate(&pm);
                for (i, &k) in fit_obs.iter().enumerate() {
                    // A perturbation that leaves the model's domain has no
                    // usable derivative; the row stays zero.
                    let (Some(a), Some(b)) = (ep.res[k], em.res[k]) else {
                        continue;
                    };
                    if ev.res[k].is_none() {
                        continue;
                    }
                    jac[(i * 2) * np + j] = (a[0] - b[0]) / (2.0 * JACOBIAN_STEP);
                    jac[(i * 2 + 1) * np + j] = (a[1] - b[1]) / (2.0 * JACOBIAN_STEP);
                }
            }

            let mut a = DMatrix::<f64>::zeros(np, np);
            let mut g = DVector::<f64>::zeros(np);
            for (i, &k) in fit_obs.iter().enumerate() {
                let Some(r) = ev.res[k] else { continue };
                let w = 1.0 / (1.0 + (r[0] * r[0] + r[1] * r[1]) / fs2).sqrt();
                for c in 0..2 {
                    let row = &jac[(i * 2 + c) * np..(i * 2 + c + 1) * np];
                    for j in 0..np {
                        let wj = w * row[j];
                        g[j] += wj * r[c];
                        for l in j..np {
                            a[(j, l)] += wj * row[l];
                        }
                    }
                }
            }
            for j in 0..np {
                for l in (j + 1)..np {
                    a[(l, j)] = a[(j, l)];
                }
            }

            let mut improved = false;
            for _ in 0..MAX_DAMPING_STEPS {
                let mut damped = a.clone();
                for d in 0..np {
                    let diag = damped[(d, d)];
                    damped[(d, d)] = diag + lambda * diag.max(1e-12);
                }
                let Some(delta) = damped.lu().solve(&(-&g)) else {
                    lambda *= 4.0;
                    continue;
                };
                if !delta.iter().all(|v| v.is_finite()) {
                    lambda *= 4.0;
                    continue;
                }
                let cand: Vec<f64> = (0..np).map(|j| p[j] + delta[j]).collect();
                let cand_ev = self.evaluate(&cand);
                let cand_cost = self.cost(&cand_ev);
                if cand_cost < prev {
                    p = cand;
                    ev = cand_ev;
                    prev = cand_cost;
                    lambda = (lambda * 0.5).max(1e-12);
                    improved = true;
                    break;
                }
                lambda *= 4.0;
                if lambda > 1e12 {
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        p
    }
}

/// Component-wise median center, then the median distance to it — the scene
/// radius the translation parameters are measured in, so all seven parameters
/// of a block are `O(1)` and one damping scale serves them. Falls back to 1
/// when the cameras are coincident or non-finite.
fn scene_scale(centers: &[Point3<f64>]) -> f64 {
    if centers.is_empty() {
        return 1.0;
    }
    let mut axis = vec![0.0f64; centers.len()];
    let mut mid = Vector3::zeros();
    for c in 0..3 {
        for (i, center) in centers.iter().enumerate() {
            axis[i] = center.coords[c];
        }
        mid[c] = median_in_place(&mut axis);
    }
    let mut d: Vec<f64> = centers.iter().map(|c| (c.coords - mid).norm()).collect();
    let s = median_in_place(&mut d);
    if s.is_finite() && s > 0.0 {
        s
    } else {
        1.0
    }
}

/// Estimate the joint per-group corrections and report how much of the
/// candidate's cross-group disagreement they explain.
///
/// `None` when the operation has nothing to say: fewer than two viewpoint
/// groups (no group structure), or no eligible, measurable bridge to solve on
/// (no cross-group evidence). A solve that cannot descend — singular normal
/// equations, no admissible step — returns the identity corrections it started
/// from, which score `explained_pct == 0` and `net_after == net_before`.
pub(super) fn group_consistency(input: &GroupConsistencyInput<'_>) -> Option<GroupConsistency> {
    if input.n_groups < 2 || input.eval_segs.is_empty() {
        return None;
    }

    // Gauge: the largest group by posed image count, ties to the lowest id.
    let mut sizes = vec![0usize; input.n_groups];
    for &g in input.group_of {
        if g >= 0 && (g as usize) < input.n_groups {
            sizes[g as usize] += 1;
        }
    }
    let mut gauge = 0usize;
    for g in 1..input.n_groups {
        if sizes[g] > sizes[gauge] {
            gauge = g;
        }
    }
    let mut block_of_group = vec![None; input.n_groups];
    let mut n_moving = 0usize;
    for (g, block) in block_of_group.iter_mut().enumerate() {
        if g != gauge {
            *block = Some(n_moving);
            n_moving += 1;
        }
    }

    // Pack the evaluated observations; every `stride`-th bridge also fits.
    let n_eval = input.eval_segs.len();
    let stride = if n_eval > MAX_FIT_BRIDGES {
        n_eval / MAX_FIT_BRIDGES + 1
    } else {
        1
    };
    let mut obs = Vec::new();
    let mut offsets = vec![0usize];
    let mut in_fit = Vec::new();
    for (e, &s) in input.eval_segs.iter().enumerate() {
        let fits = e % stride == 0;
        for k in input.seg_offsets[s]..input.seg_offsets[s + 1] {
            obs.push(k);
            in_fit.push(fits);
        }
        offsets.push(obs.len());
    }

    let solver = Solver {
        input,
        block_of_group,
        obs,
        offsets,
        in_fit,
        scene: scene_scale(input.posed_centers),
        n_params: 7 * (input.n_groups - 1),
    };

    let before = solver.evaluate(&vec![0.0f64; solver.n_params]);
    let p = solver.solve();
    let after = solver.evaluate(&p);

    let sat_px = input.sat_px;
    let net_before = before.med.iter().filter(|&&m| m < sat_px).count();
    let net_after = after.med.iter().filter(|&&m| m < sat_px).count();
    let mut n_unsat = 0usize;
    let mut n_fixed = 0usize;
    for e in 0..n_eval {
        if input.eval_hi_parallax[e] && before.med[e] >= sat_px {
            n_unsat += 1;
            if after.med[e] < sat_px {
                n_fixed += 1;
            }
        }
    }
    let explained_pct = if n_unsat > 0 {
        100.0 * n_fixed as f64 / n_unsat as f64
    } else {
        0.0
    };

    let mut corrections = Vec::with_capacity(input.n_groups);
    for g in 0..input.n_groups {
        let (rotation_wxyz, translation, log_scale) = match solver.block_of_group[g] {
            None => ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
            Some(b) => {
                let o = b * 7;
                let q = UnitQuaternion::from_scaled_axis(Vector3::new(p[o], p[o + 1], p[o + 2]));
                let qi = q.into_inner();
                let t = Vector3::new(p[o + 3], p[o + 4], p[o + 5]) * solver.scene;
                ([qi.w, qi.i, qi.j, qi.k], [t.x, t.y, t.z], p[o + 6])
            }
        };
        corrections.push(GroupCorrection {
            group: g as u32,
            rotation_wxyz,
            translation,
            log_scale,
        });
    }

    Some(GroupConsistency {
        corrections,
        explained_pct,
        net_before,
        net_after,
    })
}
