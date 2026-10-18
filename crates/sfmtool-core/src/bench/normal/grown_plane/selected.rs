// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The selection, the pairs and the shift-only fits of
//! [`grown_plane_normal`](super::grown_plane_normal).
//!
//! The selection predicts each view's information about the tilt from its
//! geometry and matches views greedily until the combined plane is precise
//! enough. Every view matched grows from the starting plane, from its
//! keypoint-implied shift first and from the stage-1 search where that growth
//! is under the floor, and counts when a growth passes the floor on its own.
//! The robust combination (`combine`) then keeps the planes that agree. A
//! view the combination leaves out whose kept growth had no search is grown
//! once more from the search, as the growth with no search is followed by
//! the search wherever it is refused, and the planes are combined again.
//!
//! The pairs are the widest among the kept views, each view in one pair at
//! most; each is fitted from the combined plane with the two views' shifts
//! held, and its plane joins the combination. When the step turns the patch,
//! every view not picked has its shift alone fitted at the final plane.

use nalgebra::Matrix2;

use super::combine::{sigmas_deg, Combined, Estimate, PairEstimate};
use super::solve::{maximise, tilt_information, PairFit};
use super::{
    same_shift, start_word, verdict_words, Fate, GrownPlaneAttempt, GrownPlanePair, GrownPlanePick,
    GrownPlaneRefusal, GrownPlaneStart, GrownPlaneTry, GrownPlaneTryOutcome, NormalError, Outcome,
    Sequence,
};
use crate::progress::Progress;
use crate::progress_note;

/// A view grown against the reference: its estimate when a growth passed the
/// floor, else its last growth.
type Matched = Result<Estimate, GrownPlaneAttempt>;

impl Sequence<'_, '_, '_> {
    /// The selection, the combination of the selected views' planes, the
    /// retry from the search of a view it left out, and the pairs. Every view
    /// not picked is left [`Fate::NotTried`], for [`Self::fit_shift_only`].
    pub(super) fn run(&self, progress: &Progress<'_>) -> Result<Outcome, NormalError> {
        let n = self.candidates.len();
        let mut trace = Vec::new();
        let mut picks = Vec::new();
        // Each candidate's estimate, or the growth that left it under the
        // floor; `None` for a view not matched.
        let mut matched: Vec<Option<Matched>> = (0..n).map(|_| None).collect();
        self.select(progress, &mut matched, &mut trace, &mut picks)?;

        let mut estimates: Vec<Estimate> = matched
            .iter()
            .filter_map(|m| m.as_ref().and_then(|r| r.as_ref().ok()).cloned())
            .collect();
        let mut combined = self.combine(&estimates, &[], progress);

        // A view left out whose growth had no search is grown from the
        // search, once.
        let mut retried = false;
        for (e, verdict) in estimates.iter_mut().zip(&combined.views) {
            if verdict.refusal.is_none() || e.attempt.start != GrownPlaneStart::NoSearch {
                continue;
            }
            if let Some(Ok(found)) = self.match_view(e.index, false, progress, &mut trace)? {
                *e = found;
                retried = true;
            }
        }
        if retried {
            combined = self.combine(&estimates, &[], progress);
        }

        let mut pairs = Vec::new();
        if let Some(tilt) = combined.tilt {
            let pair_estimates;
            (pair_estimates, pairs) = self.fit_pairs(&estimates, &combined, tilt, progress)?;
            if !pair_estimates.is_empty() {
                combined = self.combine(&estimates, &pair_estimates, progress);
            }
        }
        for (pair, verdict) in pairs.iter_mut().zip(&combined.pairs) {
            pair.refusal = verdict.refusal;
            pair.distance = verdict.distance;
        }

        let mut fates: Vec<Fate> = (0..n).map(|_| Fate::NotTried).collect();
        for (i, m) in matched.iter().enumerate() {
            if let Some(Err(attempt)) = m {
                fates[i] = Fate::LeftOut {
                    reason: attempt.refusal.unwrap_or(GrownPlaneRefusal::UnderFloor),
                    distance: None,
                    held: None,
                };
            }
        }
        for (e, verdict) in estimates.iter().zip(&combined.views) {
            fates[e.index] = match verdict.refusal {
                None => Fate::Joined {
                    estimate: Box::new(e.clone()),
                    distance: verdict.distance,
                    held: verdict.held,
                    final_shift: None,
                },
                Some(reason) => Fate::LeftOut {
                    reason,
                    distance: verdict.distance,
                    held: verdict.held,
                },
            };
        }
        Ok(Outcome {
            tilt: combined.tilt,
            information: combined.information,
            discounted: combined.discounted,
            fates,
            trace,
            picks,
            pairs,
        })
    }

    /// Fit the shift alone of every view not picked, each left
    /// [`Fate::NotTried`] by [`Self::run`], at the final plane `tilt`.
    pub(super) fn fit_shift_only(
        &self,
        tilt: [f64; 2],
        fates: &mut [Fate],
        progress: &Progress<'_>,
    ) -> Result<(), NormalError> {
        for (i, fate) in fates.iter_mut().enumerate() {
            if matches!(fate, Fate::NotTried) {
                progress.check_cancel()?;
                *fate = self.shift_only(i, tilt, progress);
            }
        }
        Ok(())
    }

    /// Grow candidate `i` from the starting plane against the reference
    /// alone, in a `grown plane view` phase, and record the try: from its
    /// keypoint-implied shift first where `unsearched`, then from the search
    /// where that is under the floor. Its estimate when a growth passes the
    /// floor, else its last growth; `None` where no growth was made, which
    /// only a search landing on the implied shift with `unsearched` off
    /// leaves.
    fn match_view(
        &self,
        i: usize,
        unsearched: bool,
        progress: &Progress<'_>,
        trace: &mut Vec<GrownPlaneTry>,
    ) -> Result<Option<Matched>, NormalError> {
        progress.check_cancel()?;
        let candidate = &self.candidates[i];
        let mut phase = progress.phase("grown plane view");
        let judged = self.growths(i, unsearched, &phase)?;
        let taken = judged
            .iter()
            .find(|j| j.grown.attempt.refusal.is_none())
            .map(|j| j.estimate.clone());
        let outcome = if taken.is_some() {
            GrownPlaneTryOutcome::Matched
        } else {
            GrownPlaneTryOutcome::LeftOut
        };
        progress_note!(
            phase,
            "image {}: {}",
            candidate.image_index,
            if taken.is_some() {
                "matched"
            } else {
                "left out"
            }
        );
        let last = judged.last().map(|j| j.grown.attempt.clone());
        trace.push(GrownPlaneTry {
            observation: candidate.row,
            image: candidate.image_index,
            angle_deg: candidate.angle_deg,
            attempts: judged.into_iter().map(|j| j.grown.attempt).collect(),
            outcome,
            combined_normal: None,
            combined_sigma_deg: f64::NAN,
        });
        Ok(match (taken, last) {
            (Some(e), _) => Some(Ok(e)),
            (None, Some(a)) => Some(Err(a)),
            (None, None) => None,
        })
    }

    /// The widest pairs among the views `combined` kept, each view in one
    /// pair at most, up to `max_pairs`, each fitted from the combined plane
    /// `tilt` in a `pair fit` phase under one `pairs` phase.
    ///
    /// The pairs are taken greedily: every pair of kept views in order of
    /// the angle between the two views' directions from the patch centre,
    /// widest first, ties by the two views' order among the candidates; a
    /// pair is taken when neither view is in a pair taken already.
    fn fit_pairs(
        &self,
        estimates: &[Estimate],
        combined: &Combined,
        tilt: [f64; 2],
        progress: &Progress<'_>,
    ) -> Result<(Vec<PairEstimate>, Vec<GrownPlanePair>), NormalError> {
        let options = self.options;
        let mut phase = progress.detail_phase("pairs");
        let kept: Vec<usize> = (0..estimates.len())
            .filter(|&k| combined.views[k].refusal.is_none())
            .collect();
        let angle = |a: usize, b: usize| {
            let (ca, cb) = (
                &self.candidates[estimates[a].index],
                &self.candidates[estimates[b].index],
            );
            match (ca.direction, cb.direction) {
                (Some(x), Some(y)) => x.dot(&y).clamp(-1.0, 1.0).acos().to_degrees(),
                _ => 0.0,
            }
        };
        let mut all: Vec<(f64, usize, usize)> = Vec::new();
        for (x, &a) in kept.iter().enumerate() {
            for &b in &kept[x + 1..] {
                all.push((angle(a, b), a, b));
            }
        }
        all.sort_by(|p, q| {
            q.0.total_cmp(&p.0)
                .then(estimates[p.1].index.cmp(&estimates[q.1].index))
                .then(estimates[p.2].index.cmp(&estimates[q.2].index))
        });
        let mut used = vec![false; estimates.len()];
        let mut chosen = Vec::new();
        for (deg, a, b) in all {
            if chosen.len() >= options.max_pairs {
                break;
            }
            if used[a] || used[b] {
                continue;
            }
            used[a] = true;
            used[b] = true;
            chosen.push((deg, a, b));
        }
        let mut pair_estimates = Vec::new();
        let mut pairs = Vec::new();
        for (deg, a, b) in chosen {
            progress.check_cancel()?;
            let mut fit_phase = phase.detail_phase("pair fit");
            let (ea, eb) = (&estimates[a], &estimates[b]);
            let renders_before =
                self.renders[ea.index].renders() + self.renders[eb.index].renders();
            let fit = PairFit {
                first: &self.renders[ea.index],
                second: &self.renders[eb.index],
                live: self.live,
                shifts: [self.moved_shift(ea, tilt), self.moved_shift(eb, tilt)],
                min_coverage: options.min_coverage,
            };
            let (p, zncc, taken) = maximise(&fit, &[tilt[0], tilt[1], 0.0, 0.0], &self.iteration);
            let own = [p[0], p[1]];
            let information = tilt_information(&fit, &p, &self.iteration);
            let sigma = sigmas_deg(&information, options.zncc_tolerance)[0];
            let refused = (zncc.is_nan() || zncc < options.min_held_zncc)
                .then_some(GrownPlaneRefusal::UnderFloor);
            let (ca, cb) = (&self.candidates[ea.index], &self.candidates[eb.index]);
            progress_note!(
                fit_phase,
                "images {} and {}, {deg:.1}\u{b0} apart: ZNCC {zncc:.3}, \u{3c3} {sigma:.2}\u{b0}, {}, {taken} iterations, {} renders",
                ca.image_index,
                cb.image_index,
                verdict_words(refused),
                self.renders[ea.index].renders() + self.renders[eb.index].renders() - renders_before
            );
            drop(fit_phase);
            pair_estimates.push(PairEstimate {
                members: [a, b],
                tilt: own,
                information,
                refused,
            });
            pairs.push(GrownPlanePair {
                observations: [ca.row, cb.row],
                images: [ca.image_index, cb.image_index],
                angle_deg: deg,
                normal: self.grid.plane_normal(own).into(),
                relative_shift: [p[2], p[3]],
                zncc,
                sigma_deg: sigma,
                distance: None,
                refusal: refused,
            });
        }
        progress_note!(phase, "{} pairs", pairs.len());
        Ok((pair_estimates, pairs))
    }

    /// The selection: pick views one at a time by their predicted
    /// information and match each against the
    /// reference, until the combined plane's weakest `σ` reaches
    /// `target_sigma_deg` or `max_selected_views` views have passed the
    /// floor. A pick under the floor is replaced by the next pick.
    ///
    /// A view's predicted information is `P = g · (α·d dᵀ + β·e eᵀ)`, with
    /// `d` the unit direction of its offset from the reference in the
    /// starting frame, `e` the direction across it, and the geometric gain
    /// `g = (s / ρ)² · sin²θ`: `θ` its angle from the reference, `s` its
    /// photograph px per world unit at the anchor capped at the working
    /// grid's samples per world unit `ρ` (the grid samples no finer). `α` and
    /// `β` are read from the views matched so far, each the median over them
    /// of their measured information along `d` or across it divided by `g`;
    /// a model with `β` zero is the rank-one prediction along the offset. The
    /// first pick, and every pick while no view has been matched, is the
    /// widest view left; every later one maximises the smallest eigenvalue of
    /// the measured information of the kept views plus the view's
    /// prediction, ties by the larger eigenvalue, then by the wider view and
    /// then by the earlier one.
    fn select(
        &self,
        progress: &Progress<'_>,
        matched: &mut [Option<Matched>],
        trace: &mut Vec<GrownPlaneTry>,
        picks: &mut Vec<GrownPlanePick>,
    ) -> Result<(), NormalError> {
        let options = self.options;
        let tau = options.zncc_tolerance;
        let n = self.candidates.len();
        let density = self.grid.density();
        // Each view's geometric gain `g = (s/ρ)²·sin²θ` and offset direction.
        let gain: Vec<f64> = self
            .candidates
            .iter()
            .map(|c| {
                let s = if c.scale_px.is_finite() {
                    c.scale_px.min(density) / density
                } else {
                    0.0
                };
                let sin = if c.angle_deg.is_finite() {
                    c.angle_deg.to_radians().sin()
                } else {
                    0.0
                };
                s * s * sin * sin
            })
            .collect();
        let along_of = |i: usize| {
            let d = self.candidates[i].offset_direction;
            nalgebra::Vector2::new(d[0], d[1])
        };
        let across_of = |i: usize| {
            let d = along_of(i);
            nalgebra::Vector2::new(-d[1], d[0])
        };
        let predicted = |i: usize, along: f64, across: f64| {
            let (d, e) = (along_of(i), across_of(i));
            if d.norm_squared() < 0.5 {
                return Matrix2::identity() * (gain[i] * 0.5 * (along + across));
            }
            (d * d.transpose() * along + e * e.transpose() * across) * gain[i]
        };
        let mut estimates: Vec<Estimate> = Vec::new();
        let mut kept_information = Matrix2::zeros();
        let mut passed = 0;
        let mut tried = vec![false; n];
        while passed < options.max_selected_views {
            progress.check_cancel()?;
            // The model's two coefficients, from the views matched so far.
            let (mut along, mut across): (Vec<f64>, Vec<f64>) = estimates
                .iter()
                .filter(|e| gain[e.index] > 0.0)
                .map(|e| {
                    let (d, x) = (along_of(e.index), across_of(e.index));
                    let g = gain[e.index];
                    (
                        (d.transpose() * e.information * d)[(0, 0)] / g,
                        (x.transpose() * e.information * x)[(0, 0)] / g,
                    )
                })
                .unzip();
            let model = (!along.is_empty()).then(|| {
                (
                    crate::numeric::median_in_place(&mut along).max(0.0),
                    crate::numeric::median_in_place(&mut across).max(0.0),
                )
            });
            let wider = |a: usize, b: usize| {
                self.candidates[a]
                    .angle_deg
                    .total_cmp(&self.candidates[b].angle_deg)
            };
            let weakest_with = |i: usize, (along, across): (f64, f64)| {
                let e = (kept_information + predicted(i, along, across))
                    .symmetric_eigen()
                    .eigenvalues;
                (e.min(), e.max())
            };
            let pick = match model {
                None => (0..n)
                    .filter(|&i| !tried[i])
                    .max_by(|&a, &b| wider(a, b).then(b.cmp(&a))),
                Some(m) => (0..n)
                    .filter(|&i| !tried[i])
                    .map(|i| {
                        let (lo, hi) = weakest_with(i, m);
                        (i, lo, hi)
                    })
                    .max_by(|a, b| {
                        a.1.total_cmp(&b.1)
                            .then(a.2.total_cmp(&b.2))
                            .then(wider(a.0, b.0))
                            .then(b.0.cmp(&a.0))
                    })
                    .map(|(i, _, _)| i),
            };
            let Some(i) = pick else {
                break;
            };
            tried[i] = true;
            let candidate = &self.candidates[i];
            let predicted_weak = model.map_or(f64::NAN, |m| weakest_with(i, m).0);
            let mut phase = progress.detail_phase("selection");
            progress_note!(
                phase,
                "image {}, {:.1}\u{b0} from the reference, {:.2} px per unit: predicted weakest information {predicted_weak:.3}",
                candidate.image_index,
                candidate.angle_deg,
                candidate.scale_px
            );
            drop(phase);
            let Some(result) = self.match_view(i, true, progress, trace)? else {
                continue;
            };
            let ok = result.is_ok();
            let mut sigma_after = f64::NAN;
            if let Ok(e) = &result {
                estimates.push(e.clone());
                passed += 1;
                let combined = self.combine(&estimates, &[], progress);
                kept_information = estimates
                    .iter()
                    .zip(&combined.views)
                    .filter(|(_, v)| v.refusal.is_none())
                    .map(|(e, _)| e.information)
                    .sum();
                sigma_after = combined.sigma_weak_deg(tau);
                if let (Some(t), Some(last)) = (combined.tilt, trace.last_mut()) {
                    last.combined_normal = Some(self.grid.plane_normal(t).into());
                    last.combined_sigma_deg = sigma_after;
                }
            }
            matched[i] = Some(result);
            picks.push(GrownPlanePick {
                observation: candidate.row,
                image: candidate.image_index,
                angle_deg: candidate.angle_deg,
                offset_direction: candidate.offset_direction,
                scale_px: candidate.scale_px,
                predicted_weak,
                matched: ok,
                sigma_after_deg: sigma_after,
            });
            if sigma_after <= options.target_sigma_deg {
                break;
            }
        }
        Ok(())
    }

    /// Candidate `i`'s shift alone at the final plane `tilt`, in a
    /// `shift-only fit` phase: from its keypoint-implied shift, the first
    /// window's ZNCC read and the whole grid fitted with the tilt held; where
    /// that is under the floor, or the view has no keypoint, from the stage-1
    /// search at `tilt` the same way, unless it lands on the implied shift.
    fn shift_only(&self, i: usize, tilt: [f64; 2], progress: &Progress<'_>) -> Fate {
        let options = self.options;
        let candidate = &self.candidates[i];
        let mut phase = progress.detail_phase("shift-only fit");
        let renders_before = self.renders[i].renders();
        let rows = self.grid.window(self.radii[0]);
        let fit = self.grid_fit(i, Some(tilt));
        let mut best: Option<(GrownPlaneStart, [f64; 2], f64, f64)> = None;
        if let Some(shift) = candidate.kp_shift {
            let z1 = self.spot_zncc(i, tilt, &rows, shift);
            let (q, z, _) = maximise(&fit, &shift, &self.iteration);
            best = Some((GrownPlaneStart::NoSearch, [q[0], q[1]], z, z1));
        }
        let under = best.is_none_or(|(_, _, z, _)| z.is_nan() || z < options.min_held_zncc);
        if under {
            let (z1, shift, start) = self.search_first_window(i, tilt, &rows, &phase);
            if !candidate.kp_shift.is_some_and(|s| same_shift(s, shift)) {
                let (q, z, _) = maximise(&fit, &shift, &self.iteration);
                if best.is_none_or(|(_, _, b, _)| b.is_nan() || z > b) {
                    best = Some((start, [q[0], q[1]], z, z1));
                }
            }
        }
        let (start, shift, grid_zncc, z1) = best.expect("a keypoint growth or a search");
        progress_note!(
            phase,
            "image {}, {}: first window ZNCC {z1:.3}, grid ZNCC {grid_zncc:.3}, {}, {} renders",
            candidate.image_index,
            start_word(start),
            if grid_zncc >= options.min_held_zncc {
                "written"
            } else {
                "under the floor"
            },
            self.renders[i].renders() - renders_before
        );
        Fate::ShiftOnly {
            start,
            shift,
            grid_zncc,
        }
    }
}
