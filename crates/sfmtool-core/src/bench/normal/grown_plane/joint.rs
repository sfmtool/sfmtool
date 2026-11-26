// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The joint growth of [`grown_plane_normal`](super::grown_plane_normal)'s
//! three-way ladder: a pick and its partner from the same rung grown together
//! against the reference, one plane shared by the two views and one image
//! shift for each.
//!
//! It runs the stages a single view's growth runs (`Sequence::grow`): the
//! first window read at each view's keypoint-implied shift, the growing
//! windows with the early stop, and the whole-grid rounds, each stage on the
//! grid laid at the plane the growth has reached with the turned support.
//! Each stage fits the tilt and both shifts on the sum of the two views'
//! scores against the reference ([`JointFit`]); the views are never compared
//! with each other. Every score it reports, and the floor it is judged by, is
//! the mean of the two views' scores. Its information about the tilt is the
//! Schur complement of the joint normal matrix with both shifts eliminated,
//! which is the two views' information at the shared plane summed.

use std::rc::Rc;

use nalgebra::{Matrix2, Vector3};

use super::combine::{sigmas_deg_at, Estimate};
use super::render::ViewRender;
use super::solve::{
    carried_information, maximise, schur_tilt, score_and_normal_matrix, JointFit, Reading,
};
use super::{
    reached, tilt_start_word, verdict_words, GrownPlaneAttempt, GrownPlaneGridRound,
    GrownPlaneRefusal, GrownPlaneTiltStart, NormalError, Sequence, EARLY_REFUSAL_FROM_STAGE,
    EARLY_REFUSAL_MARGIN,
};
use crate::progress::{Phase, Progress};
use crate::progress_note;

/// One joint growth of a pick and its partner, judged.
pub(super) struct JointGrown {
    /// Its plane, as a tilt of the starting frame.
    pub(super) tilt: [f64; 2],
    /// The growth as reported; `refusal` is `StoppedEarly` or `UnderFloor`
    /// for a growth whose mean score is under the floor, `None` otherwise.
    pub(super) attempt: GrownPlaneAttempt,
    /// The estimate it gives, the pick's index and the partner's, with the
    /// joint normal matrix; information zero for a growth that stopped early.
    pub(super) estimate: Estimate,
}

impl<'a> Sequence<'_, 'a> {
    /// Grow candidates `i` and `j` jointly against the reference from the
    /// plane `from`, each from its keypoint-implied shift, in a `growth`
    /// phase on `phase`, and judge the growth on the mean of the two views'
    /// whole-grid scores at the plane it reached.
    pub(super) fn joint_growth(
        &self,
        (i, j): (usize, usize),
        from: (GrownPlaneTiltStart, [f64; 2]),
        phase: &Progress<'_>,
    ) -> Result<JointGrown, NormalError> {
        let options = self.options;
        let turned = self.turned();
        let members = [i, j];
        let images = [&self.candidates[i].image, &self.candidates[j].image];
        let renders_of = || images.iter().map(|v| v.renders()).sum::<u64>();
        let mut growth = phase.detail_phase("growth");
        let templates_before = self.reference.renders();
        let renders_before = renders_of();
        let (mut laid, t0) = self.laid_for(from.1);
        let shifts = [self.candidates[i].kp_shift, self.candidates[j].kp_shift];
        let mut p = vec![
            t0[0],
            t0[1],
            shifts[0][0],
            shifts[0][1],
            shifts[1][0],
            shifts[1][1],
        ];
        let mut normal = if turned {
            laid.grid.frame.normal
        } else {
            laid.grid.plane_normal(t0)
        };
        let rows = laid.grid.window(self.radii[0]);
        let z1 = members
            .iter()
            .zip(&shifts)
            .map(|(&k, &s)| self.spot_zncc(k, &laid, t0, &rows, s))
            .sum::<f64>()
            / 2.0;
        let mut stage_zncc = vec![z1];
        let mut stage_turn_deg = Vec::new();
        let window_iteration = self.capped_iteration(options.stage_turn_cap_deg);
        let shifts_of = |p: &[f64]| ([p[2], p[3]], [p[4], p[5]]);
        let mut stopped = false;
        for &half in &self.radii[1..] {
            growth.check_cancel()?;
            let mut stage = growth.detail_phase("window stage");
            let before = renders_of();
            if turned {
                laid = self.lay(normal);
                (p[0], p[1]) = (0.0, 0.0);
            }
            let renders = [
                ViewRender::new(&laid.grid, images[0]),
                ViewRender::new(&laid.grid, images[1]),
            ];
            let fit = JointFit {
                views: renders.iter().collect(),
                reference: &laid.template,
                live: &laid.live,
                reading: Reading::Window(laid.grid.window(half)),
                min_coverage: options.min_coverage,
            };
            let (q, s, taken) = maximise(&fit, &p, &window_iteration);
            let (next, turn) = reached(&laid.grid, normal, [p[0], p[1]], [q[0], q[1]]);
            (normal, p) = (next, q);
            let s = s / 2.0;
            stage_zncc.push(s);
            stage_turn_deg.push(turn);
            progress_note!(
                stage,
                "half-width {half}: mean ZNCC {s:.3}, turned {turn:.2}\u{b0}, {taken} iterations, {} renders",
                renders_of() - before
            );
            let early_floor = options.min_held_zncc - EARLY_REFUSAL_MARGIN;
            if stage_zncc.len() > EARLY_REFUSAL_FROM_STAGE && (s.is_nan() || s < early_floor) {
                stopped = true;
                break;
            }
        }
        let tilt_of = |normal: Vector3<f64>, p: &[f64]| {
            if turned {
                self.start_tilt(normal)
            } else {
                [p[0], p[1]]
            }
        };
        if stopped {
            let tilt = tilt_of(normal, &p);
            let (s0, s1) = shifts_of(&p);
            let attempt = GrownPlaneAttempt {
                tilt_start: from.0,
                stage_zncc,
                shift: s0,
                normal: normal.into(),
                grid_zncc: f64::NAN,
                sigma_deg: f64::NAN,
                refusal: Some(GrownPlaneRefusal::StoppedEarly),
                stage_turn_deg,
                grid_rounds: Vec::new(),
                partner_shift: Some(s1),
                view_zncc: None,
            };
            let mut judge = growth.detail_phase("judge");
            progress_note!(judge, "{}", verdict_words(attempt.refusal));
            drop(judge);
            note_growth(
                &mut growth,
                &attempt,
                renders_of() - renders_before,
                self.reference.renders() - templates_before,
            );
            let estimate = Estimate {
                index: i,
                tilt,
                shift: s0,
                information: Matrix2::zeros(),
                normal: None,
                frame: None,
                partner: Some((j, s1)),
            };
            return Ok(JointGrown {
                tilt,
                attempt,
                estimate,
            });
        }
        growth.check_cancel()?;
        let mut rounds = self.rounds();
        let mut grid_rounds = Vec::new();
        let mut last_score;
        loop {
            let mut stage = growth.detail_phase("whole-grid stage");
            let before = renders_of();
            if turned {
                laid = self.lay(normal);
                (p[0], p[1]) = (0.0, 0.0);
            }
            let cap_deg = rounds.cap_deg();
            let renders = [
                ViewRender::new(&laid.grid, images[0]),
                ViewRender::new(&laid.grid, images[1]),
            ];
            let fit = JointFit {
                views: renders.iter().collect(),
                reference: &laid.template,
                live: &laid.live,
                reading: Reading::Grid,
                min_coverage: options.min_coverage,
            };
            let (q, s, taken) = maximise(&fit, &p, &self.capped_iteration(cap_deg));
            let (next, turn) = reached(&laid.grid, normal, [p[0], p[1]], [q[0], q[1]]);
            let more = rounds.after(turn, next - normal);
            (normal, p) = (next, q);
            last_score = s / 2.0;
            grid_rounds.push(GrownPlaneGridRound {
                cap_deg,
                turn_deg: turn,
                zncc: last_score,
            });
            progress_note!(
                stage,
                "round {}, cap {cap_deg:.2}\u{b0}: turned {turn:.2}\u{b0}, mean robust ZNCC {last_score:.3}, {taken} iterations, {} renders",
                grid_rounds.len(),
                renders_of() - before
            );
            drop(stage);
            if !more {
                break;
            }
            growth.check_cancel()?;
        }
        stage_zncc.push(last_score);

        // Judged on the grid laid at its plane, at no tilt, with the turned
        // support; on the starting grid at its tilt without it.
        let mut judge = growth.detail_phase("judge");
        let before = renders_of();
        let (laid, own) = if turned {
            (self.lay(normal), [0.0; 2])
        } else {
            (Rc::clone(&laid), [p[0], p[1]])
        };
        let renders = [
            ViewRender::new(&laid.grid, images[0]),
            ViewRender::new(&laid.grid, images[1]),
        ];
        let fit = JointFit {
            views: renders.iter().collect(),
            reference: &laid.template,
            live: &laid.live,
            reading: Reading::Grid,
            min_coverage: options.min_coverage,
        };
        let mut q = p.clone();
        q[0] = own[0];
        q[1] = own[1];
        let (score_sum, matrix) = score_and_normal_matrix(&fit, &q);
        let local = matrix.as_ref().map_or_else(Matrix2::zeros, schur_tilt);
        let (information, frame) = if turned {
            (
                carried_information(&local, &laid.grid.start_chart_jacobian()),
                Some(laid.grid.frame),
            )
        } else {
            (local, None)
        };
        let score = score_sum / 2.0;
        let tilt = tilt_of(normal, &p);
        let sigma = sigmas_deg_at(
            &information,
            &self.start.grid.start,
            tilt,
            options.zncc_tolerance,
        )[0];
        let refusal = (score.is_nan() || score < options.min_held_zncc)
            .then_some(GrownPlaneRefusal::UnderFloor);
        progress_note!(
            judge,
            "mean grid ZNCC {score:.3}, \u{3c3} {sigma:.2}\u{b0}: {}, {} renders",
            verdict_words(refusal),
            renders_of() - before
        );
        drop(judge);
        let (s0, s1) = shifts_of(&p);
        let attempt = GrownPlaneAttempt {
            tilt_start: from.0,
            stage_zncc,
            shift: s0,
            normal: normal.into(),
            grid_zncc: score,
            sigma_deg: sigma,
            refusal,
            stage_turn_deg,
            grid_rounds,
            partner_shift: Some(s1),
            view_zncc: None,
        };
        note_growth(
            &mut growth,
            &attempt,
            renders_of() - renders_before,
            self.reference.renders() - templates_before,
        );
        let estimate = Estimate {
            index: i,
            tilt,
            shift: s0,
            information,
            normal: matrix,
            frame,
            partner: Some((j, s1)),
        };
        Ok(JointGrown {
            tilt,
            attempt,
            estimate,
        })
    }
}

/// The `growth` phase's note for a joint growth.
fn note_growth(growth: &mut Phase<'_>, attempt: &GrownPlaneAttempt, renders: u64, templates: u64) {
    progress_note!(
        *growth,
        "joint, {}: first window mean ZNCC {:.3}, grid mean ZNCC {:.3}, {}, {renders} renders, {templates} template renders",
        tilt_start_word(attempt.tilt_start),
        attempt.stage_zncc[0],
        attempt.grid_zncc,
        verdict_words(attempt.refusal)
    );
}
