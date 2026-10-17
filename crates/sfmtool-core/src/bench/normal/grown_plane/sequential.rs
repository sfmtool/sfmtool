// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`SequentialMean`](super::GrownPlaneStrategy::SequentialMean) and
//! [`SequentialPairs`](super::GrownPlaneStrategy::SequentialPairs): the views
//! in order of angle from the reference, each grown from the plane the views
//! before it share and joining by its agreement with it, the shared plane
//! after each join the information-weighted mean of the joined views' own
//! planes; and, for the second, the widest pairs of joined views matched
//! against each other at the end.

use std::collections::{BTreeSet, VecDeque};

use nalgebra::Matrix2;

use super::combine::{distance, sigmas_deg, weighted_mean, Estimate};
use super::solve::maximise;
use super::{
    verdict_words, Fate, Grown, GrownPlaneRefusal, GrownPlaneTry, GrownPlaneTryOutcome, Judged,
    NormalError, Outcome, Sequence,
};
use crate::progress::Progress;
use crate::progress_note;

/// A view in the shared plane, with its growth's agreement when it joined.
struct Member {
    estimate: Estimate,
    distance: Option<f64>,
    held: Option<f64>,
}

impl Sequence<'_, '_, '_> {
    /// Take the views in order until every one has joined or been left out.
    pub(super) fn run_sequential(&self, progress: &Progress<'_>) -> Result<Outcome, NormalError> {
        let tau = self.options.zncc_tolerance;
        let mut queue: VecDeque<usize> = (0..self.candidates.len()).collect();
        let mut deferred = BTreeSet::new();
        let mut tilt = [0.0; 2];
        let mut members: Vec<Member> = Vec::new();
        let mut left_out: Vec<(usize, super::GrownPlaneAttempt)> = Vec::new();
        let mut trace = Vec::new();
        while let Some(i) = queue.pop_front() {
            progress.check_cancel()?;
            let candidate = &self.candidates[i];
            let mut phase = progress.phase("grown plane view");
            progress_note!(phase, "image {}", candidate.image_index);

            // From the shared plane, then from the starting plane; from each,
            // at the keypoint-implied shift first and by the search after.
            let starts: &[bool] = if members.is_empty() {
                &[true]
            } else {
                &[false, true]
            };
            let shared = members
                .iter()
                .map(|m| m.estimate.information)
                .sum::<Matrix2<f64>>();
            let judge = |grown: Grown, growth: &Progress<'_>| {
                self.judge(i, grown, tilt, &shared, members.is_empty(), growth)
            };
            let mut attempts: Vec<Judged> = Vec::new();
            for &from_start in starts {
                let tilt0 = if from_start { [0.0; 2] } else { tilt };
                let plane = if from_start { "starting" } else { "shared" };
                let judged = self.growths(i, tilt0, from_start, plane, true, &phase, &judge)?;
                let taken = judged.iter().any(|j| j.grown.attempt.refusal.is_none());
                attempts.extend(judged);
                if taken {
                    break;
                }
            }
            let joining = attempts
                .iter()
                .position(|j| j.grown.attempt.refusal.is_none());
            let outcome = match joining {
                Some(k) => {
                    let j = &attempts[k];
                    members.push(Member {
                        estimate: j.estimate.clone(),
                        distance: j.grown.attempt.distance,
                        held: j.grown.attempt.held_zncc,
                    });
                    let mut mean = phase.detail_phase("mean");
                    let planes: Vec<_> = members
                        .iter()
                        .map(|m| (m.estimate.tilt, m.estimate.information))
                        .collect();
                    tilt = weighted_mean(&planes).expect("one member at least");
                    let information = planes.iter().map(|(_, i)| i).sum::<Matrix2<f64>>();
                    progress_note!(
                        mean,
                        "{} views: \u{3c3} {:.2}\u{b0} weakest",
                        members.len(),
                        sigmas_deg(&information, tau)[0]
                    );
                    drop(mean);
                    GrownPlaneTryOutcome::Joined
                }
                None if deferred.insert(i) => {
                    queue.push_back(i);
                    GrownPlaneTryOutcome::Deferred
                }
                None => {
                    let last = attempts.last().expect("one growth at least");
                    left_out.push((i, last.grown.attempt.clone()));
                    GrownPlaneTryOutcome::LeftOut
                }
            };
            progress_note!(
                phase,
                "image {}: {}",
                candidate.image_index,
                match outcome {
                    GrownPlaneTryOutcome::Joined => "joined",
                    GrownPlaneTryOutcome::Deferred => "deferred",
                    _ => "left out",
                }
            );
            let shared = members
                .iter()
                .map(|m| m.estimate.information)
                .sum::<Matrix2<f64>>();
            trace.push(GrownPlaneTry {
                observation: candidate.row,
                image: candidate.image_index,
                angle_deg: candidate.angle_deg,
                attempts: attempts.into_iter().map(|j| j.grown.attempt).collect(),
                outcome,
                shared_normal: (!members.is_empty()).then(|| self.grid.plane_normal(tilt).into()),
                shared_sigma_deg: if members.is_empty() {
                    f64::NAN
                } else {
                    sigmas_deg(&shared, tau)[0]
                },
            });
        }
        let information = members
            .iter()
            .map(|m| m.estimate.information)
            .sum::<Matrix2<f64>>();
        let mut fates: Vec<Fate> = (0..self.candidates.len()).map(|_| Fate::NotTried).collect();
        for (i, attempt) in left_out {
            fates[i] = Fate::LeftOut {
                reason: attempt.refusal.unwrap_or(GrownPlaneRefusal::UnderFloor),
                distance: attempt.distance,
                held: attempt.held_zncc,
            };
        }
        let joined = !members.is_empty();
        if self.options.strategy.pairs() && joined {
            return self.pair_members(members, fates, trace, progress);
        }
        for m in members {
            let index = m.estimate.index;
            fates[index] = Fate::Joined {
                distance: m.distance,
                held: m.held,
                estimate: Box::new(m.estimate),
                final_shift: None,
                final_zncc: None,
            };
        }
        Ok(Outcome {
            tilt: joined.then_some(tilt),
            information,
            discounted: information,
            fates,
            trace,
            picks: Vec::new(),
            pairs: Vec::new(),
        })
    }

    /// [`SequentialPairs`](super::GrownPlaneStrategy::SequentialPairs)'s end:
    /// the joined views' planes combined as the independent strategies combine
    /// them, the widest pairs among the views kept fitted from that plane, and
    /// everything combined again. Each joined view's distance and held score become the
    /// combination's.
    fn pair_members(
        &self,
        members: Vec<Member>,
        mut fates: Vec<Fate>,
        trace: Vec<GrownPlaneTry>,
        progress: &Progress<'_>,
    ) -> Result<Outcome, NormalError> {
        let estimates: Vec<Estimate> = members.into_iter().map(|m| m.estimate).collect();
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
        for (e, verdict) in estimates.iter().zip(&combined.views) {
            fates[e.index] = match verdict.refusal {
                None => Fate::Joined {
                    estimate: Box::new(e.clone()),
                    distance: verdict.distance,
                    held: verdict.held,
                    final_shift: None,
                    final_zncc: None,
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
            picks: Vec::new(),
            pairs,
        })
    }

    /// Judge a growth of candidate `i` against the shared plane at `tilt` of
    /// information `shared`, or, for the `first` view, against the floor
    /// alone. Its information is read at its own plane either way.
    fn judge(
        &self,
        i: usize,
        mut grown: Grown,
        tilt: [f64; 2],
        shared: &Matrix2<f64>,
        first: bool,
        progress: &Progress<'_>,
    ) -> Judged {
        let options = self.options;
        let mut phase = progress.detail_phase("judge");
        let renders_before = self.renders[i].renders();
        let estimate = self.estimate_of(i, &grown);
        let sigma = sigmas_deg(&estimate.information, options.zncc_tolerance)[0];
        let attempt = &mut grown.attempt;
        attempt.sigma_deg = sigma;
        if first {
            attempt.refusal = (attempt.grid_zncc < options.min_held_zncc
                || attempt.grid_zncc.is_nan())
            .then_some(GrownPlaneRefusal::UnderFloor);
            progress_note!(
                phase,
                "first view, grid ZNCC {:.3}, \u{3c3} {sigma:.2}\u{b0}: {}, {} renders",
                attempt.grid_zncc,
                verdict_words(attempt.refusal),
                self.renders[i].renders() - renders_before
            );
        } else {
            let d = distance(
                grown.tilt,
                &estimate.information,
                tilt,
                shared,
                options.zncc_tolerance,
            );
            let (_, held, _) = maximise(
                &self.grid_fit(i, Some(tilt)),
                &attempt.shift,
                &self.iteration,
            );
            attempt.refusal = if d.is_nan() || d > options.max_agreement_distance {
                Some(GrownPlaneRefusal::TooFar)
            } else if held.is_nan() || held < options.min_held_zncc {
                Some(GrownPlaneRefusal::HeldUnderFloor)
            } else {
                None
            };
            attempt.distance = Some(d);
            attempt.held_zncc = Some(held);
            progress_note!(
                phase,
                "distance {d:.2}, held ZNCC {held:.3}, \u{3c3} {sigma:.2}\u{b0}, {}, {} renders",
                verdict_words(attempt.refusal),
                self.renders[i].renders() - renders_before
            );
        }
        drop(phase);
        let estimate = Estimate {
            attempt: grown.attempt.clone(),
            ..estimate
        };
        Judged { grown, estimate }
    }
}
