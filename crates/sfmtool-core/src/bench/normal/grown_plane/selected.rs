// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The selection, the pairs and the shift-only fits of
//! [`grown_plane_normal`](super::grown_plane_normal).
//!
//! The selection matches views near to far by their angle from the
//! reference, in rungs, until the combined plane is precise enough, picking
//! within a rung by each view's information about the tilt predicted from its
//! geometry. The first rung grows from the starting plane, and every later
//! pick from the combined plane of the views kept so far, from its
//! keypoint-implied shift; a pick of the second rung refused there is grown
//! again from the starting plane, at the same shift, and a pick of a later
//! rung is not. Every growth starts from the keypoint-implied shift, and none
//! searches for another. A pick counts when a growth passes the floor on its
//! own. The robust combination (`combine`) then keeps the planes that agree.
//!
//! The pairs are the widest among the kept views, each view in one pair at
//! most; each is fitted from the combined plane with the two views' shifts
//! held, and its plane joins the combination. When the step turns the patch,
//! every view not picked has its shift alone fitted at the final plane, over
//! a mid-size window and then for two iterations over the whole grid.

use nalgebra::{Matrix2, Vector2};

use super::combine::{sigmas_deg, Combined, Estimate, PairEstimate};
use super::solve::{maximise, tilt_information, Iteration, PairFit, Reading, ViewFit};
use super::{
    verdict_words, Fate, GrownPlaneAttempt, GrownPlanePair, GrownPlanePick, GrownPlaneRefusal,
    GrownPlaneTiltStart, GrownPlaneTry, GrownPlaneTryOutcome, NormalError, Outcome, Sequence,
    SHIFT_ONLY_GRID_ITERATIONS, SHIFT_ONLY_WINDOW,
};
use crate::progress::Progress;
use crate::progress_note;

/// The rungs of the selection, counted from the first at 0, whose picks
/// refused from the combined plane are grown again from the starting plane:
/// the first, grown from the starting plane anyway, and the second. A wrong
/// first pick makes a wrong combined plane, which refuses the second rung's
/// picks; grown from the starting plane, they can pass and pull the plane
/// back. Most picks refused from the combined plane are on later rungs, where
/// the second growth costs more than it finds.
const STARTING_PLANE_RETRY_RUNGS: u32 = 2;

/// A view grown against the reference: its estimate when a growth passed the
/// floor, else its last growth.
type Matched = Result<Estimate, GrownPlaneAttempt>;

impl Sequence<'_, '_, '_> {
    /// The selection, the combination of the selected views' planes and the
    /// pairs. Every view not picked is left [`Fate::NotTried`], for
    /// [`Self::fit_shift_only`].
    pub(super) fn run(&self, progress: &Progress<'_>) -> Result<Outcome, NormalError> {
        let n = self.candidates.len();
        let mut trace = Vec::new();
        let mut picks = Vec::new();
        // Each candidate's estimate, or the growth that left it under the
        // floor; `None` for a view not matched.
        let mut matched: Vec<Option<Matched>> = (0..n).map(|_| None).collect();
        self.select(progress, &mut matched, &mut trace, &mut picks)?;

        let estimates: Vec<Estimate> = matched
            .iter()
            .filter_map(|m| m.as_ref().and_then(|r| r.as_ref().ok()).cloned())
            .collect();
        let mut combined = self.combine(&estimates, &[], progress);

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

    /// Grow candidate `i` against the reference alone, in a `grown plane
    /// view` phase, and record the try. Where `combined` gives a plane, the
    /// view is grown from it first. Where `combined` is `None`, or that
    /// growth is refused and `retry` is set, it is grown from the starting
    /// plane. Each growth starts from the view's keypoint-implied shift. Its
    /// estimate when a growth passes the floor, else its last growth.
    fn match_view(
        &self,
        i: usize,
        combined: Option<[f64; 2]>,
        retry: bool,
        progress: &Progress<'_>,
        trace: &mut Vec<GrownPlaneTry>,
    ) -> Result<Matched, NormalError> {
        progress.check_cancel()?;
        let candidate = &self.candidates[i];
        let mut phase = progress.phase("grown plane view");
        let starting = (GrownPlaneTiltStart::Starting, [0.0; 2]);
        let first = match combined {
            Some(tilt) => (GrownPlaneTiltStart::Combined, tilt),
            None => starting,
        };
        let mut judged = vec![self.growth(i, first, &phase)?];
        if combined.is_some() && retry && judged[0].grown.attempt.refusal.is_some() {
            judged.push(self.growth(i, starting, &phase)?);
        }
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
        let result = taken.ok_or_else(|| judged[judged.len() - 1].grown.attempt.clone());
        trace.push(GrownPlaneTry {
            observation: candidate.row,
            image: candidate.image_index,
            angle_deg: candidate.angle_deg,
            attempts: judged.into_iter().map(|j| j.grown.attempt).collect(),
            outcome,
            combined_normal: None,
            combined_sigma_deg: f64::NAN,
        });
        Ok(result)
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
            let information = tilt_information(&fit, &p);
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

    /// The selection: views picked one at a time near to far by angle from
    /// the reference, in rungs, and matched against the reference until the
    /// combined plane's weakest `σ` reaches `target_sigma_deg` or
    /// `max_selected_views` views have passed the floor, or no candidate is
    /// left.
    ///
    /// The first rung, grown from the starting plane, takes the view nearest
    /// `first_rung_deg` within `first_rung_min_deg ..= first_rung_max_deg`,
    /// nearest first, ties by the earlier view; while none in that range has
    /// passed the floor and none is left there, the view nearest
    /// `first_rung_deg` among all those left, so the widest view under the
    /// range where nothing is wider. Each later rung aims at `rung_ratio`
    /// times the widest angle the combination keeps, and takes the views
    /// within a factor `√rung_ratio` of that aim; a rung with no view left in
    /// its range aims `rung_ratio` times further out, and one with an aim of
    /// zero, where every view kept is 0° from the reference, aims at the
    /// widest candidate. The rung whose range
    /// reaches the widest candidate is the last: it takes every view from
    /// the smaller of its aim and the widest candidate's angle, over
    /// `√rung_ratio`, out, and the selection stays on it. Once no view is
    /// left there, one more rung takes every view left, narrower ones among
    /// them.
    ///
    /// Within a later rung the pick is the view whose predicted information
    /// most raises the weakest information measured so far, which spreads
    /// the picks' offset directions, ties by the larger eigenvalue, then by
    /// the view nearer the aim and then by the earlier one. A view's
    /// predicted information is `P = g · (α·d dᵀ + β·e eᵀ)`, with `d` the
    /// unit direction of its offset from the reference in the starting patch
    /// frame, `e` the direction across it, and the geometric gain
    /// `g = (s / ρ)² · sin²θ`: `θ` its angle from the reference, `s` its
    /// photograph px per world unit at the anchor capped at the working
    /// grid's samples per world unit `ρ` (the grid samples no finer). `α` and
    /// `β` are read from the views matched so far, each the median over them
    /// of their measured information along `d` or across it divided by `g`;
    /// a model with `β` zero is the rank-one prediction along the offset.
    ///
    /// Every pick of a later rung is grown from the combined plane of the
    /// views kept so far, from its keypoint-implied shift, and, on the second
    /// rung, where that is refused, from the starting plane as the first rung
    /// grows it
    /// ([`STARTING_PLANE_RETRY_RUNGS`]). A rung stays until one of
    /// its views is kept, so a refused pick is followed by another view in
    /// the same range. A candidate whose angle is not finite is never
    /// picked.
    fn select(
        &self,
        progress: &Progress<'_>,
        matched: &mut [Option<Matched>],
        trace: &mut Vec<GrownPlaneTry>,
        picks: &mut Vec<GrownPlanePick>,
    ) -> Result<(), NormalError> {
        let mut state = Selection {
            prediction: Prediction::new(self),
            estimates: Vec::new(),
            kept_information: Matrix2::zeros(),
            kept_tilt: None,
            widest_kept: None,
            passed: 0,
            tried: vec![false; self.candidates.len()],
        };
        let state = &mut state;
        let options = self.options;
        let n = self.candidates.len();
        let angle = |i: usize| self.candidates[i].angle_deg;
        let widest_candidate = (0..n)
            .map(angle)
            .filter(|a| a.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let root = options.rung_ratio.sqrt();
        let mut rung: u32 = 0;
        let mut aim = options.first_rung_deg;
        // The last rung's lower bound, once the ladder has reached it.
        let mut last: Option<f64> = None;
        // Whether the ladder is past its last rung, on the views left under it.
        let mut rest = false;
        while state.passed < options.max_selected_views {
            progress.check_cancel()?;
            let left = |i: &usize| !state.tried[*i] && angle(*i).is_finite();
            let nearer = |a: &usize, b: &usize| {
                (angle(*a) - aim)
                    .abs()
                    .total_cmp(&(angle(*b) - aim).abs())
                    .then(a.cmp(b))
            };
            let first_rung = state.widest_kept.is_none();
            let (pick, predicted_weak) = if first_rung {
                let range = options.first_rung_min_deg..=options.first_rung_max_deg;
                let pick = (0..n)
                    .filter(left)
                    .filter(|&i| range.contains(&angle(i)))
                    .min_by(nearer)
                    .or_else(|| (0..n).filter(left).min_by(nearer));
                (pick, f64::NAN)
            } else {
                let model = state.prediction.model(&state.estimates);
                let pick = loop {
                    let (lo, hi) = match last {
                        Some(lo) => (lo, f64::INFINITY),
                        None if aim * root >= widest_candidate => {
                            let lo = aim.min(widest_candidate) / root;
                            last = Some(lo);
                            (lo, f64::INFINITY)
                        }
                        None => (aim / root, aim * root),
                    };
                    let pick = state.best_predicted_near(
                        self,
                        model,
                        aim,
                        (0..n)
                            .filter(left)
                            .filter(|&i| (lo..=hi).contains(&angle(i))),
                    );
                    if pick.is_some() {
                        break pick;
                    }
                    match last {
                        // Past the last rung, every view left, by predicted
                        // information.
                        Some(_) if !rest => {
                            last = Some(f64::NEG_INFINITY);
                            rest = true;
                            rung += 1;
                        }
                        Some(_) => break None,
                        // An aim of zero, where every kept view is 0°
                        // from the reference, never widens; it goes
                        // to the last rung.
                        None if aim <= 0.0 => {
                            aim = widest_candidate;
                            rung += 1;
                        }
                        None => {
                            aim *= options.rung_ratio;
                            rung += 1;
                        }
                    }
                };
                let weak = match (pick, model) {
                    (Some(i), Some(m)) => state.weakest_with(i, m).0,
                    _ => f64::NAN,
                };
                (pick, weak)
            };
            let Some(i) = pick else {
                break;
            };
            let widest_before = state.widest_kept;
            if self.take(
                state,
                i,
                rung,
                predicted_weak,
                progress,
                matched,
                trace,
                picks,
            )? {
                break;
            }
            // The rung is passed once the widest kept view widens; the next
            // aims `rung_ratio` times beyond it.
            if let Some(widest) = state.widest_kept {
                if widest_before.is_none_or(|w| widest > w) && last.is_none() {
                    aim = options.rung_ratio * widest;
                    rung += 1;
                }
            }
        }
        Ok(())
    }

    /// Match pick `i`, of rung `rung`, against the reference, in a
    /// `selection` phase, and record it: from the combined plane of the views
    /// kept so far first where a view has been kept, from the starting plane
    /// otherwise or, on a rung under [`STARTING_PLANE_RETRY_RUNGS`], where
    /// that is refused. A pick that passes the floor is
    /// combined with the picks before it. Whether the combined plane's
    /// weakest `σ` has reached `target_sigma_deg`.
    #[allow(clippy::too_many_arguments)]
    fn take(
        &self,
        state: &mut Selection,
        i: usize,
        rung: u32,
        predicted_weak: f64,
        progress: &Progress<'_>,
        matched: &mut [Option<Matched>],
        trace: &mut Vec<GrownPlaneTry>,
        picks: &mut Vec<GrownPlanePick>,
    ) -> Result<bool, NormalError> {
        let options = self.options;
        state.tried[i] = true;
        let candidate = &self.candidates[i];
        let mut phase = progress.detail_phase("selection");
        progress_note!(
            phase,
            "image {}, {:.1}\u{b0} from the reference, {:.2} px per unit, rung {rung}: predicted weakest information {predicted_weak:.3}",
            candidate.image_index,
            candidate.angle_deg,
            candidate.scale_px
        );
        drop(phase);
        let retry = rung < STARTING_PLANE_RETRY_RUNGS;
        let result = self.match_view(i, state.kept_tilt, retry, progress, trace)?;
        let ok = result.is_ok();
        let mut sigma_after = f64::NAN;
        if let Ok(e) = &result {
            state.estimates.push(e.clone());
            state.passed += 1;
            let combined = self.combine(&state.estimates, &[], progress);
            let kept: Vec<&Estimate> = state
                .estimates
                .iter()
                .zip(&combined.views)
                .filter(|(_, v)| v.refusal.is_none())
                .map(|(e, _)| e)
                .collect();
            state.kept_information = kept.iter().map(|e| e.information).sum();
            state.widest_kept = kept
                .iter()
                .map(|e| self.candidates[e.index].angle_deg)
                .filter(|a| a.is_finite())
                .reduce(f64::max);
            state.kept_tilt = combined.tilt;
            sigma_after = combined.sigma_weak_deg(options.zncc_tolerance);
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
            rung,
            sigma_after_deg: sigma_after,
        });
        Ok(sigma_after <= options.target_sigma_deg)
    }

    /// Fit candidate `i`'s shift alone at the plane `tilt` from `shift`: over
    /// the window of [`SHIFT_ONLY_WINDOW`] first, then for
    /// [`SHIFT_ONLY_GRID_ITERATIONS`] iterations over the whole grid on the
    /// robust cell score. The shift reached and its whole-grid score.
    fn fit_shift(&self, i: usize, tilt: [f64; 2], shift: [f64; 2]) -> ([f64; 2], f64) {
        let half = self.radii[SHIFT_ONLY_WINDOW.min(self.radii.len() - 1)];
        let window = ViewFit {
            render: &self.renders[i],
            reference: self.reference,
            live: self.live,
            reading: Reading::Window(self.grid.window(half)),
            held: Some(tilt),
            min_coverage: self.options.min_coverage,
        };
        let (q, _, _) = maximise(&window, &shift, &self.iteration);
        let brief = Iteration {
            max_iterations: SHIFT_ONLY_GRID_ITERATIONS,
            ..self.iteration
        };
        let (q, z, _) = maximise(&self.grid_fit(i, Some(tilt)), &q, &brief);
        ([q[0], q[1]], z)
    }

    /// Candidate `i`'s shift alone at the final plane `tilt`, in a
    /// `shift-only fit` phase: from its keypoint-implied shift, the first
    /// window's ZNCC read and the shift fitted with the tilt held
    /// ([`Self::fit_shift`]). A fit under the floor is kept as it is, and its
    /// keypoint is not written.
    fn shift_only(&self, i: usize, tilt: [f64; 2], progress: &Progress<'_>) -> Fate {
        let options = self.options;
        let candidate = &self.candidates[i];
        let mut phase = progress.detail_phase("shift-only fit");
        let renders_before = self.renders[i].renders();
        let rows = self.grid.window(self.radii[0]);
        let z1 = self.spot_zncc(i, tilt, &rows, candidate.kp_shift);
        let (shift, grid_zncc) = self.fit_shift(i, tilt, candidate.kp_shift);
        progress_note!(
            phase,
            "image {}: first window ZNCC {z1:.3}, grid ZNCC {grid_zncc:.3}, {}, {} renders",
            candidate.image_index,
            if grid_zncc >= options.min_held_zncc {
                "written"
            } else {
                "under the floor"
            },
            self.renders[i].renders() - renders_before
        );
        Fate::ShiftOnly { shift, grid_zncc }
    }
}

/// What the selection has measured so far.
struct Selection {
    prediction: Prediction,
    /// The estimates of every pick that passed the floor, in the order
    /// picked.
    estimates: Vec<Estimate>,
    /// The information of those the combination keeps, summed.
    kept_information: Matrix2<f64>,
    /// Their combined plane; `None` before any pick passed.
    kept_tilt: Option<[f64; 2]>,
    /// The widest angle from the reference among them, degrees.
    widest_kept: Option<f64>,
    /// How many picks passed the floor.
    passed: usize,
    /// Whether each candidate was picked.
    tried: Vec<bool>,
}

impl Selection {
    /// The smallest and largest eigenvalue of the kept information with
    /// candidate `i`'s prediction under the model `m` added.
    fn weakest_with(&self, i: usize, m: (f64, f64)) -> (f64, f64) {
        let e = (self.kept_information + self.prediction.predicted(i, m))
            .symmetric_eigen()
            .eigenvalues;
        (e.min(), e.max())
    }

    /// Among `candidates`, the one whose prediction under `m` most raises
    /// the weakest kept information, ties by the larger eigenvalue, then by
    /// the view nearer `aim` degrees and then by the earlier one; with no
    /// model, the view nearest `aim`, ties by the earlier one.
    fn best_predicted_near(
        &self,
        sequence: &Sequence<'_, '_, '_>,
        m: Option<(f64, f64)>,
        aim: f64,
        candidates: impl Iterator<Item = usize>,
    ) -> Option<usize> {
        let off = |i: usize| (sequence.candidates[i].angle_deg - aim).abs();
        candidates
            .map(|i| {
                let (lo, hi) = m.map_or((0.0, 0.0), |m| self.weakest_with(i, m));
                (i, lo, hi)
            })
            .max_by(|a, b| {
                a.1.total_cmp(&b.1)
                    .then(a.2.total_cmp(&b.2))
                    .then(off(b.0).total_cmp(&off(a.0)))
                    .then(b.0.cmp(&a.0))
            })
            .map(|(i, _, _)| i)
    }
}

/// Each candidate's information about the tilt, predicted from its
/// geometry: `g · (α·d dᵀ + β·e eᵀ)` (see [`Sequence::select`]).
struct Prediction {
    /// Each candidate's geometric gain `g = (s/ρ)²·sin²θ`.
    gain: Vec<f64>,
    /// Each candidate's offset direction `d`; zero where it has none.
    along: Vec<Vector2<f64>>,
}

impl Prediction {
    fn new(sequence: &Sequence<'_, '_, '_>) -> Self {
        let density = sequence.grid.density();
        let gain = sequence
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
        let along = sequence
            .candidates
            .iter()
            .map(|c| Vector2::new(c.offset_direction[0], c.offset_direction[1]))
            .collect();
        Self { gain, along }
    }

    fn across(&self, i: usize) -> Vector2<f64> {
        let d = self.along[i];
        Vector2::new(-d[1], d[0])
    }

    /// Candidate `i`'s predicted information under the model's two
    /// coefficients `(α, β)`.
    fn predicted(&self, i: usize, (along, across): (f64, f64)) -> Matrix2<f64> {
        let (d, e) = (self.along[i], self.across(i));
        if d.norm_squared() < 0.5 {
            return Matrix2::identity() * (self.gain[i] * 0.5 * (along + across));
        }
        (d * d.transpose() * along + e * e.transpose() * across) * self.gain[i]
    }

    /// The model's two coefficients `(α, β)` from `estimates`, each the
    /// median of their measured information along or across their offset
    /// over their gain; `None` with no estimate of positive gain.
    fn model(&self, estimates: &[Estimate]) -> Option<(f64, f64)> {
        let (mut along, mut across): (Vec<f64>, Vec<f64>) = estimates
            .iter()
            .filter(|e| self.gain[e.index] > 0.0)
            .map(|e| {
                let (d, x) = (self.along[e.index], self.across(e.index));
                let g = self.gain[e.index];
                (
                    (d.transpose() * e.information * d)[(0, 0)] / g,
                    (x.transpose() * e.information * x)[(0, 0)] / g,
                )
            })
            .unzip();
        (!along.is_empty()).then(|| {
            (
                crate::numeric::median_in_place(&mut along).max(0.0),
                crate::numeric::median_in_place(&mut across).max(0.0),
            )
        })
    }
}
