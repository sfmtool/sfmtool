// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How [`grown_plane_normal`](super::grown_plane_normal) combines planes: the
//! information-weighted mean, the precision it carries, the agreement
//! distance, and the robust combination that leaves out the estimates that
//! disagree and the views that no longer match at the combined plane.
//!
//! Every estimate is a tilt `t` of the starting patch frame with its information
//! `I` about the tilt, read at that tilt. The combined plane is
//! `t = (Σ I)⁻¹ Σ I·t`, with covariance `2τ·(Σ I)⁻¹`: the estimate a sum of
//! independent quadratic score losses `½·(t − tₑ)ᵀ Iₑ (t − tₑ)` is maximised
//! by.

use nalgebra::{DMatrix, Matrix2, Vector2};

use super::{GrownPlaneAttempt, GrownPlaneRefusal, Sequence};
use crate::progress::Progress;
use crate::progress_note;

/// The share of the information's scale added along every direction when the
/// mean is solved, pulling a direction no estimate informs to the plain mean
/// of the estimates' tilts rather than leaving it undetermined.
const MEAN_REGULARISATION: f64 = 1e-6;

/// One view's own plane, grown against the reference.
#[derive(Clone)]
pub(super) struct Estimate {
    /// Its index among the candidates.
    pub(super) index: usize,
    /// Its own plane and shift.
    pub(super) tilt: [f64; 2],
    pub(super) shift: [f64; 2],
    /// Its information about the tilt with the shift eliminated.
    pub(super) information: Matrix2<f64>,
    /// The normal matrix over `[a, b, sx, sy]` that information came from,
    /// which moves its shift to another plane; `None` where nothing could be
    /// read.
    pub(super) normal: Option<DMatrix<f64>>,
    /// The growth it came from.
    pub(super) attempt: GrownPlaneAttempt,
}

/// One pair of views' own plane, matched against each other.
pub(super) struct PairEstimate {
    /// The two views, as indices into the estimates.
    pub(super) members: [usize; 2],
    pub(super) tilt: [f64; 2],
    pub(super) information: Matrix2<f64>,
    /// Refused before it was combined: under the floor.
    pub(super) refused: Option<GrownPlaneRefusal>,
}

/// The verdict on one estimate of a combination.
#[derive(Clone, Copy)]
pub(super) struct Verdict {
    pub(super) refusal: Option<GrownPlaneRefusal>,
    /// Its distance from the combined plane, or from the mean that left it
    /// out.
    pub(super) distance: Option<f64>,
    /// A view's score at the combined plane, at its shift moved there.
    pub(super) held: Option<f64>,
}

/// What a combination ended at.
pub(super) struct Combined {
    /// `None` when no view was kept.
    pub(super) tilt: Option<[f64; 2]>,
    /// The kept estimates' information summed.
    pub(super) information: Matrix2<f64>,
    /// The same with each kept pair's information halved.
    pub(super) discounted: Matrix2<f64>,
    /// One per view estimate.
    pub(super) views: Vec<Verdict>,
    /// One per pair estimate.
    pub(super) pairs: Vec<Verdict>,
}

impl Combined {
    /// The combined plane's weakest `σ`, degrees; `NaN` with no plane.
    pub(super) fn sigma_weak_deg(&self, tau: f64) -> f64 {
        match self.tilt {
            Some(_) => sigmas_deg(&self.information, tau)[0],
            None => f64::NAN,
        }
    }
}

/// The information-weighted mean of `estimates`, each a tilt and its
/// information: `(Σ I + ε)⁻¹ (Σ I·t + ε·t̄)`, with `t̄` the plain mean of the
/// tilts and `ε` [`MEAN_REGULARISATION`] of the summed information's largest
/// eigenvalue (or 1 where it is zero), so that the mean of one estimate is
/// that estimate and a direction no estimate informs takes the plain mean.
/// `None` with no estimate.
pub(super) fn weighted_mean(estimates: &[([f64; 2], Matrix2<f64>)]) -> Option<[f64; 2]> {
    if estimates.is_empty() {
        return None;
    }
    let mut sum = Matrix2::zeros();
    let mut rhs = Vector2::zeros();
    let mut plain = Vector2::zeros();
    for (t, info) in estimates {
        let t = Vector2::new(t[0], t[1]);
        sum += info;
        rhs += info * t;
        plain += t;
    }
    plain /= estimates.len() as f64;
    let largest = sum.symmetric_eigen().eigenvalues.max();
    let eps = MEAN_REGULARISATION * if largest > 0.0 { largest } else { 1.0 };
    let a = sum + Matrix2::identity() * eps;
    let t = a.try_inverse()? * (rhs + plain * eps);
    Some([t[0], t[1]])
}

/// `σ` along the weakest and strongest directions of `information`, degrees:
/// `√(2τ/λ)` for each eigenvalue `λ`, floored at `1e-12`, a small tilt
/// `(a, b)` being that many radians.
pub(super) fn sigmas_deg(information: &Matrix2<f64>, tau: f64) -> [f64; 2] {
    let eigen = information.symmetric_eigen().eigenvalues;
    let (lo, hi) = if eigen[0] <= eigen[1] {
        (eigen[0], eigen[1])
    } else {
        (eigen[1], eigen[0])
    };
    let sigma = |lambda: f64| (2.0 * tau / lambda.max(1e-12)).sqrt().to_degrees();
    [sigma(lo), sigma(hi)]
}

/// The covariance of the tilt that `information` gives: `2τ·(I + 10⁻⁶)⁻¹`,
/// regularised so a direction with no information has a large variance
/// rather than none.
pub(super) fn covariance(information: &Matrix2<f64>, tau: f64) -> Matrix2<f64> {
    let reg = information + Matrix2::identity() * 1e-6;
    reg.try_inverse()
        .unwrap_or_else(|| Matrix2::identity() * 1e6)
        * (2.0 * tau)
}

/// The Mahalanobis distance between the tilt `a` of information `ia` and the
/// tilt `b` of information `ib`, the two covariances added.
pub(super) fn distance(
    a: [f64; 2],
    ia: &Matrix2<f64>,
    b: [f64; 2],
    ib: &Matrix2<f64>,
    tau: f64,
) -> f64 {
    let d = Vector2::new(a[0] - b[0], a[1] - b[1]);
    let c = covariance(ia, tau) + covariance(ib, tau);
    let d2 = c
        .try_inverse()
        .map_or(f64::INFINITY, |inv| (d.transpose() * inv * d)[(0, 0)]);
    d2.max(0.0).sqrt()
}

impl Sequence<'_, '_, '_> {
    /// Combine `views` and `pairs` into one plane, opening a `mean` phase.
    ///
    /// Every view estimate and every pair not refused already is in the
    /// first mean. While some kept estimate lies further than
    /// [`max_agreement_distance`](super::GrownPlaneNormalOptions::max_agreement_distance)
    /// from the mean of the kept ones, the furthest is left out (`TooFar`),
    /// with any pair holding a view left out (`ViewLeftOut`), and the mean is
    /// read again. Then each kept view is scored at the mean, at its shift
    /// moved there by its own normal matrix, one render each; a view under
    /// [`min_held_zncc`](super::GrownPlaneNormalOptions::min_held_zncc) is
    /// left out (`HeldUnderFloor`) and the whole is repeated, until no view
    /// is left out. Leaving out one at a time, the furthest first, makes the
    /// result independent of the order the estimates are given in and keeps
    /// at least one estimate, which is at no distance from its own mean.
    pub(super) fn combine(
        &self,
        views: &[Estimate],
        pairs: &[PairEstimate],
        progress: &Progress<'_>,
    ) -> Combined {
        let options = self.options;
        let tau = options.zncc_tolerance;
        let mut phase = progress.detail_phase("mean");
        let renders_before = phase.is_recording().then(|| self.renders_done());
        let mut v_verdict: Vec<Verdict> = views
            .iter()
            .map(|_| Verdict {
                refusal: None,
                distance: None,
                held: None,
            })
            .collect();
        let mut p_verdict: Vec<Verdict> = pairs
            .iter()
            .map(|p| Verdict {
                refusal: p.refused,
                distance: None,
                held: None,
            })
            .collect();
        let mean_of = |v: &[Verdict], p: &[Verdict]| {
            let kept: Vec<([f64; 2], Matrix2<f64>)> = views
                .iter()
                .zip(v)
                .filter(|(_, v)| v.refusal.is_none())
                .map(|(e, _)| (e.tilt, e.information))
                .chain(
                    pairs
                        .iter()
                        .zip(p)
                        .filter(|(_, v)| v.refusal.is_none())
                        .map(|(e, _)| (e.tilt, e.information)),
                )
                .collect();
            let information = kept.iter().map(|(_, i)| i).sum::<Matrix2<f64>>();
            (weighted_mean(&kept), information)
        };
        let leave_out_pairs_of = |view: usize, p: &mut [Verdict]| {
            for (pair, verdict) in pairs.iter().zip(p.iter_mut()) {
                if verdict.refusal.is_none() && pair.members.contains(&view) {
                    verdict.refusal = Some(GrownPlaneRefusal::ViewLeftOut);
                }
            }
        };
        let mut rounds = 0;
        let mut held_out = 0;
        loop {
            // The furthest estimate out, one at a time.
            while let (Some(mean), information) = mean_of(&v_verdict, &p_verdict) {
                let mut worst: Option<(bool, usize, f64)> = None;
                let mut consider = |is_view: bool, k: usize, t: [f64; 2], i: &Matrix2<f64>| {
                    let d = distance(t, i, mean, &information, tau);
                    let d = if d.is_nan() { f64::INFINITY } else { d };
                    if worst.is_none_or(|(_, _, w)| d > w) {
                        worst = Some((is_view, k, d));
                    }
                };
                for (k, e) in views.iter().enumerate() {
                    if v_verdict[k].refusal.is_none() {
                        consider(true, k, e.tilt, &e.information);
                    }
                }
                for (k, e) in pairs.iter().enumerate() {
                    if p_verdict[k].refusal.is_none() {
                        consider(false, k, e.tilt, &e.information);
                    }
                }
                rounds += 1;
                match worst {
                    Some((is_view, k, d)) if d > options.max_agreement_distance => {
                        let verdict = if is_view {
                            &mut v_verdict[k]
                        } else {
                            &mut p_verdict[k]
                        };
                        verdict.refusal = Some(GrownPlaneRefusal::TooFar);
                        verdict.distance = Some(d);
                        if is_view {
                            leave_out_pairs_of(k, &mut p_verdict);
                        }
                    }
                    _ => break,
                }
            }
            // Every kept view still matches at the mean.
            let (Some(mean), _) = mean_of(&v_verdict, &p_verdict) else {
                break;
            };
            let mut dropped = false;
            for (k, e) in views.iter().enumerate() {
                if v_verdict[k].refusal.is_some() {
                    continue;
                }
                let shift = self.moved_shift(e, mean);
                let z = self.score_at(e.index, mean, shift);
                v_verdict[k].held = Some(z);
                if z.is_nan() || z < options.min_held_zncc {
                    v_verdict[k].refusal = Some(GrownPlaneRefusal::HeldUnderFloor);
                    leave_out_pairs_of(k, &mut p_verdict);
                    dropped = true;
                    held_out += 1;
                }
            }
            if !dropped {
                break;
            }
        }
        let (tilt, information) = mean_of(&v_verdict, &p_verdict);
        let mut discounted = Matrix2::zeros();
        if let Some(mean) = tilt {
            for (k, e) in views.iter().enumerate() {
                let verdict = &mut v_verdict[k];
                if verdict.refusal.is_none() {
                    discounted += e.information;
                }
                if verdict.refusal != Some(GrownPlaneRefusal::TooFar) {
                    verdict.distance =
                        Some(distance(e.tilt, &e.information, mean, &information, tau));
                }
            }
            for (k, e) in pairs.iter().enumerate() {
                let verdict = &mut p_verdict[k];
                if verdict.refusal.is_none() {
                    discounted += e.information * 0.5;
                }
                if verdict.refusal.is_none()
                    || verdict.refusal == Some(GrownPlaneRefusal::ViewLeftOut)
                {
                    verdict.distance =
                        Some(distance(e.tilt, &e.information, mean, &information, tau));
                }
            }
        }
        let kept_views = v_verdict.iter().filter(|v| v.refusal.is_none()).count();
        let kept_pairs = p_verdict.iter().filter(|v| v.refusal.is_none()).count();
        progress_note!(
            phase,
            "{kept_views} of {} views and {kept_pairs} of {} pairs kept, {held_out} held under the floor, {rounds} means: \u{3c3} {:.2}\u{b0} weakest, {} renders",
            views.len(),
            pairs.len(),
            if tilt.is_some() {
                sigmas_deg(&information, tau)[0]
            } else {
                f64::NAN
            },
            self.renders_done() - renders_before.unwrap_or(0)
        );
        Combined {
            tilt,
            information,
            discounted,
            views: v_verdict,
            pairs: p_verdict,
        }
    }
}
