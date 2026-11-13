// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk along one sequence: predict the next frame from the recent rate,
//! or measure the next frame first when there is none, measure the step,
//! accept it or re-aim, aiming no further than twice the longest span
//! measured, keep every step beyond the next frame at or above the
//! match floor, and swap a frame blurred against its neighbours for a sharper
//! neighbour; with the local density term, measure each accepted step's
//! essential split and set the next step's target from the last three.
//! See `specs/core/analysis/pick-images.md` § "The walk", § "The local
//! density term" and § "Run-ahead".

use std::collections::VecDeque;

use super::capture::{FrameId, PickImagesError, PickImagesOptions};
use super::density::{self, DensityTerm};
use super::essential_split::EssentialSplitOptions;
use super::frame_pair::{FrameFeatures, FramePairStats};
use super::frames::Frames;
use super::pilot::PilotVote;
use super::report::{StepDensity, StepKind, Swap, WalkStep};
use crate::camera::CameraIntrinsics;
use crate::numeric::median;
use crate::progress::Progress;

/// How many times a step is re-aimed at the band before the walk settles
/// below it.
const MAX_REAIMS: usize = 4;

/// How many accepted steps' rates predict the next step.
const RECENT_RATES: usize = 3;

/// How many times the longest span a step has measured, of the recent steps
/// and of its own candidates that held below the band, its aim may lie from
/// the last chosen frame. The band's ends are about 2 apart, `1.3 / 0.7`, so
/// a candidate at twice a span measured below the band reads close to the
/// band rather than far past it, where frames that share only the distant
/// scene can read in the band.
const REACH_FACTOR: usize = 2;

/// An accepted step that kept the match floor: its rate and its span in
/// positions.
#[derive(Clone, Copy)]
struct Recent {
    rate: f64,
    span: usize,
}

/// How many positions on either side of a candidate lie the frames it is
/// compared with for blur against its neighbours.
const NEIGHBOUR_WINDOW: usize = 2;

/// What the walk of one sequence chose.
pub(super) struct Walked {
    pub(super) chosen: Vec<usize>,
    pub(super) steps: Vec<WalkStep>,
    pub(super) swaps: Vec<Swap>,
    pub(super) density_term: DensityTerm,
    pub(super) pilot_vote: Option<PilotVote>,
}

/// The bounds one step works within.
#[derive(Clone, Copy)]
struct StepBounds {
    from: usize,
    low: f64,
    high: f64,
    target: f64,
    /// The farthest frame the step, or a swap, may reach: the sequence's
    /// last frame, or the frame before the nearest candidate measured short
    /// of the floor.
    limit: usize,
}

struct Walker<'w, 'a> {
    frames: &'w mut Frames<'a>,
    options: &'w PickImagesOptions,
    progress: &'w Progress<'w>,
    sequence: usize,
    last: usize,
    /// Whether this is the pilot's walk, which keeps the matched positions
    /// of every pair it matches for the walk after it.
    pilot: bool,
    /// The last chosen frame and its features, read when a pair from it is
    /// first matched.
    current: (usize, Option<FrameFeatures>),
    /// The candidates of the step matched from the last chosen frame, with
    /// their features, so that the one accepted keeps them.
    candidates: Vec<(usize, FrameFeatures)>,
    /// The rates and spans of the last accepted steps that kept the match
    /// floor; empty at the start and after a step to the next frame that did
    /// not.
    recent: VecDeque<Recent>,
    /// The local density term's split options, `None` with the term off.
    split: Option<EssentialSplitOptions>,
    /// The measures of the last accepted steps, at most
    /// [`density::WINDOW`].
    measures: VecDeque<f64>,
    /// The next step's target, and the gain and median measure it came
    /// from.
    target: f64,
    gain: f64,
    median_measure: f64,
    walked: Walked,
}

/// Walk one sequence of `len` frames from its first frame to its last, with
/// the local density term measuring with `camera` when it is given, or, with
/// `stop_at`, until it has chosen that many frames.
///
/// With the term on, the matched positions of the pairs the walk may accept
/// are kept in `frames`, so each accepted step's split is measured from the
/// walk's own match of it. A walk with `stop_at`, the pilot's, keeps those of
/// every pair it matches, for the walk after it, which takes its steps again
/// from the pairs' statistics; any other walk drops them all when it ends.
pub(super) fn walk_sequence(
    frames: &mut Frames,
    sequence: usize,
    len: usize,
    camera: Option<&CameraIntrinsics>,
    options: &PickImagesOptions,
    progress: &Progress,
    stop_at: Option<usize>,
) -> Result<Walked, PickImagesError> {
    let first = FrameId { sequence, frame: 0 };
    frames.ensure(&[first], progress)?;
    let (width, height) = frames.image_size(first);
    let split = density::split_options(camera, width, height, options.max_density_gain);
    let density_term = match &split {
        Ok(split) => {
            let (fx, fy) = split.camera.focal_lengths();
            DensityTerm::On {
                focal_px: 0.5 * (fx + fy),
            }
        }
        Err(off) => DensityTerm::Off(off.clone()),
    };
    let walked = Walked {
        chosen: vec![0],
        steps: Vec::new(),
        swaps: Vec::new(),
        density_term,
        pilot_vote: None,
    };
    if len == 1 {
        return Ok(walked);
    }
    let pilot = stop_at.is_some();
    frames.keep_positions(pilot || split.is_ok());
    let mut walker = Walker {
        frames,
        options,
        progress,
        sequence,
        last: len - 1,
        pilot,
        current: (0, None),
        candidates: Vec::new(),
        recent: VecDeque::new(),
        split: split.ok(),
        measures: VecDeque::new(),
        target: options.target_step_percent,
        gain: 1.0,
        median_measure: 1.0,
        walked,
    };
    let stop_at = stop_at.unwrap_or(usize::MAX);
    while walker.current.0 < walker.last && walker.walked.chosen.len() < stop_at {
        walker.step()?;
    }
    if !pilot {
        walker.frames.keep_positions(false);
    }
    Ok(walker.walked)
}

impl Walker<'_, '_> {
    fn id(&self, frame: usize) -> FrameId {
        FrameId {
            sequence: self.sequence,
            frame,
        }
    }

    /// Read the frames of `positions` not yet read, in one wait.
    fn read(&mut self, positions: &[usize]) -> Result<(), PickImagesError> {
        let ids: Vec<FrameId> = positions.iter().map(|&p| self.id(p)).collect();
        self.frames.ensure(&ids, self.progress)
    }

    /// Whether the step from `f` to `n`, measured as `stats`, holds: the step
    /// to the next frame always does, and one beyond it when it keeps at
    /// least `min_step_matches` matches, the match floor. A solve follows a
    /// capture through the tracks consecutive chosen frames share, which the
    /// mutual matches count, so no step passes over frames to keep fewer.
    fn holds(&self, f: usize, n: usize, stats: &FramePairStats) -> bool {
        n == f + 1 || stats.matches >= self.options.min_step_matches
    }

    /// Read `frame` and match it with the last chosen frame.
    fn measure(&mut self, frame: usize) -> Result<FramePairStats, PickImagesError> {
        self.read(&[frame])?;
        let (a, b) = (self.id(self.current.0), self.id(frame));
        if let Some(stats) = self.frames.pair(a, b) {
            return Ok(stats.clone());
        }
        let current = match self.current.1.take() {
            Some(current) => current,
            None => self.frames.features(a)?,
        };
        let features = self.frames.features(b)?;
        let stats = self
            .frames
            .match_pair(a, &current, b, &features, self.progress)?;
        self.current.1 = Some(current);
        self.candidates.push((frame, features));
        Ok(stats)
    }

    /// Ask the source to prefetch `frames` after `f` not yet read, which the
    /// walk predicts it will read next. Nothing is read; the walk decides
    /// only from frames it waits for.
    fn predict(&mut self, f: usize, frames: &[usize]) {
        let ids: Vec<FrameId> = frames
            .iter()
            .filter(|&&n| n > f && n <= self.last)
            .map(|&n| self.id(n))
            .filter(|&id| !self.frames.is_read(id))
            .collect();
        if !ids.is_empty() {
            self.frames.prefetch(&ids);
        }
    }

    /// The recent steps once a step from `f` to `c`, measured as `stats`,
    /// is accepted: a step that keeps the floor adds its rate and span, and a
    /// step to the next frame that does not clears them, so the next step
    /// measures its next frame first.
    fn recent_after(&self, f: usize, c: usize, stats: &FramePairStats) -> VecDeque<Recent> {
        let mut recent = self.recent.clone();
        if stats.matches >= self.options.min_step_matches {
            recent.push_back(Recent {
                rate: stats.step_percent / (c - f) as f64,
                span: c - f,
            });
            if recent.len() > RECENT_RATES {
                recent.pop_front();
            }
        } else {
            recent.clear();
        }
        recent
    }

    /// The farthest frame a step from `f` may aim at, its reach:
    /// [`REACH_FACTOR`] times the longest span of the recent steps `recent`
    /// and of `measured`, the step's own farthest candidate that held below
    /// the band, from `f`, and at most the last frame.
    fn reach(&self, f: usize, recent: &VecDeque<Recent>, measured: Option<usize>) -> usize {
        let span = recent
            .iter()
            .map(|r| r.span)
            .chain(measured.map(|x| x - f))
            .max()
            .unwrap_or(0)
            .max(1);
        (f + REACH_FACTOR * span).min(self.last)
    }

    /// The first candidate of a step from `f` with the recent steps `recent`:
    /// `target / rate` frames on, at the median of their rates, within their
    /// reach, or the next frame when there are none. Past the last frame when
    /// `f` is the last.
    fn first_candidate(&self, f: usize, recent: &VecDeque<Recent>) -> usize {
        if recent.is_empty() || f >= self.last {
            return f + 1;
        }
        let rates: Vec<f64> = recent.iter().map(|r| r.rate).collect();
        Self::aim(
            f,
            self.target / median(&rates),
            f + 1,
            self.reach(f, recent, None),
        )
    }

    fn bounds(&self) -> StepBounds {
        let target = self.target;
        StepBounds {
            from: self.current.0,
            low: target * (1.0 - self.options.step_band),
            high: target * (1.0 + self.options.step_band),
            target,
            limit: self.last,
        }
    }

    /// `from + distance`, rounded, kept within `[low, high]`.
    fn aim(from: usize, distance: f64, low: usize, high: usize) -> usize {
        let span = (high - from) as f64;
        let d = if distance.is_finite() {
            distance.round().clamp(1.0, span.max(1.0))
        } else {
            span
        };
        (from + d as usize).clamp(low, high)
    }

    /// Take one step from the last chosen frame.
    fn step(&mut self) -> Result<(), PickImagesError> {
        let b = self.bounds();
        let f = b.from;
        // With no recent step, the step to the next frame is measured first
        // and the walk aims from its rate, which is the step's prediction.
        let mut predicting = self.recent.is_empty();
        let mut c = self.first_candidate(f, &self.recent);
        // The largest candidate below the band that held, the smallest above
        // it, and the smallest short of the floor.
        let mut below: Option<(usize, FramePairStats)> = None;
        let mut above: Option<usize> = None;
        let mut short: Option<usize> = None;
        let mut reaims = 0;
        loop {
            let stats = self.measure(c)?;
            let m = stats.step_percent;
            if c == f + 1 && stats.matches < self.options.min_step_matches {
                // Matches fall with the distance, so every step beyond the
                // next frame would keep fewer than the floor too.
                return self.settle(
                    Self::within_floor(b, Some(c + 1)),
                    c,
                    stats,
                    StepKind::AtFloor,
                );
            }
            // A candidate short of the floor is too far, whatever its step.
            let short_of_floor = !self.holds(f, c, &stats);
            if short_of_floor {
                short = Some(short.map_or(c, |s| s.min(c)));
            } else if m > b.high {
                above = Some(above.map_or(c, |a| a.min(c)));
                if c == f + 1 {
                    return self.settle(
                        Self::within_floor(b, short),
                        c,
                        stats,
                        StepKind::AboveBand,
                    );
                }
            } else if m >= b.low {
                return self.settle(Self::within_floor(b, short), c, stats, StepKind::InBand);
            } else {
                if c == b.limit {
                    return self.settle(Self::within_floor(b, short), c, stats, StepKind::AtLimit);
                }
                if below.as_ref().is_none_or(|(x, _)| c > *x) {
                    below = Some((c, stats.clone()));
                }
            }
            let low = below.as_ref().map_or(f + 1, |(x, _)| x + 1);
            let high = [above, short]
                .into_iter()
                .flatten()
                .min()
                .map_or(b.limit, |a| a - 1);
            if low > high {
                // Two candidates one frame apart lie on either side of the
                // band, where the motion per frame is wider than the band, or
                // on either side of the floor.
                return match below {
                    Some((x, s)) => {
                        let kind = if short == Some(x + 1) {
                            StepKind::AtFloor
                        } else {
                            StepKind::Bracketed
                        };
                        self.settle(Self::within_floor(b, short), x, s, kind)
                    }
                    None => self.settle_next(b, short),
                };
            }
            if short_of_floor {
                // The farthest frame that keeps the floor lies in
                // `[low, high]`: try its middle, rounded up. The range
                // shrinks with every candidate, so these tries need no
                // limit of their own.
                c = low + (high - low).div_ceil(2);
                // Its two possible successors, one of which the search tries
                // next unless it ends at `c`.
                let middle = |a: usize, z: usize| (a <= z).then(|| a + (z - a).div_ceil(2));
                let next: Vec<usize> = [middle(low, c - 1), middle(c + 1, high)]
                    .into_iter()
                    .flatten()
                    .collect();
                self.predict(f, &next);
                continue;
            }
            // The aim stays within the step's reach. An aim the reach stops
            // is a probe outward, which doubles the reach when it measures
            // below the band, and not a re-aim; nor is the step's prediction
            // from the next frame's rate.
            let distance = (c - f) as f64 * b.target / m;
            let reach = self.reach(f, &self.recent, below.as_ref().map(|(x, _)| *x));
            let probe = Self::aim(f, distance, low, high) > reach;
            if !std::mem::take(&mut predicting) && !probe {
                if reaims == MAX_REAIMS {
                    return match below {
                        Some((x, s)) => {
                            self.settle(Self::within_floor(b, short), x, s, StepKind::BelowBand)
                        }
                        None => self.settle_next(b, short),
                    };
                }
                reaims += 1;
            }
            c = Self::aim(f, distance, low, high.min(reach));
        }
    }

    /// `b` with its limit before `short`, the nearest candidate measured short
    /// of the floor, so that a swap does not look at or past it.
    fn within_floor(b: StepBounds, short: Option<usize>) -> StepBounds {
        StepBounds {
            limit: short.map_or(b.limit, |s| (s - 1).min(b.limit)),
            ..b
        }
    }

    /// Settle on the next frame, when no candidate below the band held.
    fn settle_next(&mut self, b: StepBounds, short: Option<usize>) -> Result<(), PickImagesError> {
        let c = b.from + 1;
        let stats = self.measure(c)?;
        let m = stats.step_percent;
        let kind = if stats.matches < self.options.min_step_matches {
            StepKind::AtFloor
        } else if m > b.high {
            StepKind::AboveBand
        } else if m >= b.low {
            StepKind::InBand
        } else {
            StepKind::BelowBand
        };
        self.settle(Self::within_floor(b, short), c, stats, kind)
    }

    /// Accept `c`, which `step` reached with `stats`, or a sharper neighbour
    /// in its place when it is blurred against its neighbours.
    fn settle(
        &mut self,
        b: StepBounds,
        c: usize,
        stats: FramePairStats,
        kind: StepKind,
    ) -> Result<(), PickImagesError> {
        // The next step's first candidate, should `c` be accepted, is
        // extracted while the walk reads and judges `c`'s neighbours. With no
        // recent rates left, the walk reads the frames after `c` one after
        // another for as long as they keep fewer matches than the floor, so
        // as many of them as the source works on at once are predicted.
        let recent = self.recent_after(b.from, c, &stats);
        let next = self.first_candidate(c, &recent);
        let ahead = if recent.is_empty() {
            self.frames.prefetch_window().max(1)
        } else {
            1
        };
        let mut frames: Vec<usize> = (next..next + ahead).collect();
        if ahead == 1 {
            // The frames the judgement of blur reads when it is accepted.
            frames.extend(
                (1..=NEIGHBOUR_WINDOW)
                    .flat_map(|k| [next.checked_sub(k), Some(next + k)].into_iter().flatten()),
            );
        }
        self.predict(b.from, &frames);
        match self.sharper_neighbour(b, c, &stats)? {
            Some((n, s)) => {
                self.walked.swaps.push(Swap {
                    candidate: c,
                    chosen: n,
                });
                self.accept(n, s, kind)
            }
            None => self.accept(c, stats, kind),
        }
    }

    /// The neighbour to accept in place of `c`, which `step` reached with
    /// `stats`, when `c` is blurred against its neighbours; `None` when it is
    /// not, or when no neighbour qualifies.
    ///
    /// Every frame within [`NEIGHBOUR_WINDOW`] positions of `c` is read, in
    /// one wait, and `c` is blurred against a neighbour when, on some sensor,
    /// its sharpness is below `neighbour_blur_ratio` times the neighbour's;
    /// so `c` is blurred against its neighbours when it is against the
    /// sharpest of them on some sensor. A neighbour `c` is blurred against
    /// qualifies when it is after the last chosen frame and within the step's
    /// limit, is not itself blurred against `c` on any sensor, and its step
    /// holds and is no longer than the band and no shorter than the band or
    /// `c`'s own step. They are tried sharpest first: by the least, over the
    /// sensors, of the neighbour's sharpness over `c`'s, then nearest first,
    /// then earliest.
    fn sharper_neighbour(
        &mut self,
        b: StepBounds,
        c: usize,
        stats: &FramePairStats,
    ) -> Result<Option<(usize, FramePairStats)>, PickImagesError> {
        let window: Vec<usize> = (c.saturating_sub(NEIGHBOUR_WINDOW)
            ..=(c + NEIGHBOUR_WINDOW).min(self.last))
            .filter(|&n| n != c)
            .collect();
        self.read(&window)?;
        let ratio = self.options.neighbour_blur_ratio;
        let sharpness = |walker: &Self, n: usize| -> Vec<f64> {
            walker
                .frames
                .readings(walker.id(n))
                .iter()
                .map(|r| f64::from(r.sharpness))
                .collect()
        };
        // Whether `a` is blurred against `b` on some sensor.
        let blurred_against = |a: &[f64], b: &[f64]| a.iter().zip(b).any(|(&x, &y)| x < ratio * y);
        let own = sharpness(self, c);
        let f = b.from;
        let mut blurred = false;
        let mut sharper: Vec<(usize, f64)> = Vec::new();
        for &n in &window {
            let theirs = sharpness(self, n);
            if !blurred_against(&own, &theirs) {
                continue;
            }
            blurred = true;
            if n <= f || n > b.limit || blurred_against(&theirs, &own) {
                continue;
            }
            // The least gain over the sensors. A sensor at 0 in `c` gains
            // without bound where the neighbour's is above 0, and nothing
            // where it is 0 too.
            let gain = own
                .iter()
                .zip(&theirs)
                .map(|(&a, &t)| {
                    if a > 0.0 {
                        t / a
                    } else if t > 0.0 {
                        f64::INFINITY
                    } else {
                        1.0
                    }
                })
                .fold(f64::INFINITY, f64::min);
            sharper.push((n, gain));
        }
        if !blurred {
            return Ok(None);
        }
        sharper.sort_by(|x, y| {
            y.1.total_cmp(&x.1)
                .then(x.0.abs_diff(c).cmp(&y.0.abs_diff(c)))
                .then(x.0.cmp(&y.0))
        });
        let floor = stats.step_percent.min(b.low);
        for (n, _) in sharper {
            let s = self.measure(n)?;
            if self.holds(f, n, &s) && s.step_percent >= floor && s.step_percent <= b.high {
                return Ok(Some((n, s)));
            }
        }
        Ok(None)
    }

    /// Make `c`, reached with `stats`, the last chosen frame. A step that
    /// keeps the floor adds its rate to the recent rates; a step to the next
    /// frame that does not clears them, so the next step measures its next
    /// frame first. With the local density term, the step's essential split
    /// is measured and the next step's target set from the last measures.
    fn accept(
        &mut self,
        c: usize,
        stats: FramePairStats,
        kind: StepKind,
    ) -> Result<(), PickImagesError> {
        let f = self.current.0;
        let rate = stats.step_percent / (c - f) as f64;
        self.recent = self.recent_after(f, c, &stats);
        let features = self
            .candidates
            .drain(..)
            .find_map(|(n, features)| (n == c).then_some(features));
        let density = self.density_after(c)?;
        self.walked.steps.push(WalkStep {
            from: f,
            to: c,
            stats,
            rate,
            kind,
            density,
        });
        self.walked.chosen.push(c);
        self.current = (c, features);
        if !self.pilot {
            self.frames.drop_positions_before(self.id(c));
        }
        Ok(())
    }

    /// The local density term's reading of the step from the last chosen
    /// frame to `c`, with the target it was walked with. With the term on,
    /// the step's split is measured from the pair's matched positions and the
    /// next step's target set from the measures of the last
    /// [`density::WINDOW`] steps.
    fn density_after(&mut self, c: usize) -> Result<StepDensity, PickImagesError> {
        let walked_with = StepDensity {
            target_percent: self.target,
            gain: self.gain,
            median_measure: self.median_measure,
            essential_split: None,
            measure: 1.0,
        };
        let Some(split) = &self.split else {
            return Ok(walked_with);
        };
        let essential_split =
            self.frames
                .measure_split(self.id(self.current.0), self.id(c), split, self.progress)?;
        let measure = density::step_measure(&essential_split);
        self.measures.push_back(measure);
        if self.measures.len() > density::WINDOW {
            self.measures.pop_front();
        }
        let measures: Vec<f64> = self.measures.iter().copied().collect();
        let (median_measure, gain) = density::gain(&measures, self.options.max_density_gain);
        self.median_measure = median_measure;
        self.gain = gain;
        self.target = self.options.target_step_percent / gain;
        Ok(StepDensity {
            essential_split: Some(essential_split),
            measure,
            ..walked_with
        })
    }
}
