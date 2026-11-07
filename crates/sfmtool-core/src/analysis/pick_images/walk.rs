// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk along one sequence: predict the next frame from the recent rate,
//! or measure the next frame first when there is none, measure the step,
//! accept it or re-aim, keep every step beyond the next frame at or above the
//! match floor, and swap a frame blurred against its neighbours for a sharper
//! neighbour. See `specs/core/analysis/pick-images.md` § "The walk" and
//! § "Run-ahead".

use std::collections::VecDeque;

use super::capture::{FrameId, PickImagesError, PickImagesOptions};
use super::frame_pair::{FrameFeatures, FramePairStats};
use super::frames::Frames;
use super::report::{StepKind, Swap, WalkStep};
use crate::numeric::median;
use crate::progress::Progress;

/// How many times a step is re-aimed at the band before the walk settles
/// below it.
const MAX_REAIMS: usize = 4;

/// How many accepted steps' rates predict the next step.
const RECENT_RATES: usize = 3;

/// How many positions on either side of a candidate lie the frames it is
/// compared with for blur against its neighbours.
const NEIGHBOUR_WINDOW: usize = 2;

/// What the walk of one sequence chose.
pub(super) struct Walked {
    pub(super) chosen: Vec<usize>,
    pub(super) steps: Vec<WalkStep>,
    pub(super) swaps: Vec<Swap>,
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
    /// The last chosen frame and its features.
    current: (usize, FrameFeatures),
    /// The last candidate measured and its features, reused when it is
    /// accepted.
    candidate: Option<(usize, FrameFeatures)>,
    /// The rates of the last accepted steps that kept the match floor; empty
    /// at the start and after a step to the next frame that did not.
    rates: VecDeque<f64>,
    walked: Walked,
}

/// Walk one sequence of `len` frames from its first frame to its last.
pub(super) fn walk_sequence(
    frames: &mut Frames,
    sequence: usize,
    len: usize,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<Walked, PickImagesError> {
    let first = FrameId { sequence, frame: 0 };
    frames.ensure(&[first], progress)?;
    let walked = Walked {
        chosen: vec![0],
        steps: Vec::new(),
        swaps: Vec::new(),
    };
    if len == 1 {
        return Ok(walked);
    }
    let features = frames.features(first)?;
    let mut walker = Walker {
        frames,
        options,
        progress,
        sequence,
        last: len - 1,
        current: (0, features),
        candidate: None,
        rates: VecDeque::new(),
        walked,
    };
    while walker.current.0 < walker.last {
        walker.step()?;
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
        let features = self.frames.features(b)?;
        let stats = self
            .frames
            .match_pair(a, &self.current.1, b, &features, self.progress)?;
        self.candidate = Some((frame, features));
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

    /// The recent rates once a step from `f` to `c`, measured as `stats`,
    /// is accepted: a step that keeps the floor adds its rate, and a step to
    /// the next frame that does not clears them, so the next step measures
    /// its next frame first.
    fn rates_after(&self, f: usize, c: usize, stats: &FramePairStats) -> VecDeque<f64> {
        let mut rates = self.rates.clone();
        if stats.matches >= self.options.min_step_matches {
            rates.push_back(stats.step_percent / (c - f) as f64);
            if rates.len() > RECENT_RATES {
                rates.pop_front();
            }
        } else {
            rates.clear();
        }
        rates
    }

    /// The first candidate of a step from `f` with the recent rates `rates`:
    /// `target / rate` frames on, at the median of the rates, or the next
    /// frame when there are none. Past the last frame when `f` is the last.
    fn first_candidate(&self, f: usize, rates: &VecDeque<f64>) -> usize {
        if rates.is_empty() || f >= self.last {
            return f + 1;
        }
        let rates: Vec<f64> = rates.iter().copied().collect();
        let target = self.options.target_step_percent;
        Self::aim(f, target / median(&rates), f + 1, self.last)
    }

    fn bounds(&self) -> StepBounds {
        let target = self.options.target_step_percent;
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
        // With no recent rate, the step to the next frame is measured first
        // and the walk re-aims from its rate.
        let from_next = self.rates.is_empty();
        let mut c = self.first_candidate(f, &self.rates);
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
            // Aiming from the next frame's rate is the step's prediction,
            // not a re-aim.
            if reaims == MAX_REAIMS + usize::from(from_next) {
                return match below {
                    Some((x, s)) => {
                        self.settle(Self::within_floor(b, short), x, s, StepKind::BelowBand)
                    }
                    None => self.settle_next(b, short),
                };
            }
            reaims += 1;
            c = Self::aim(f, (c - f) as f64 * b.target / m, low, high);
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
        let rates = self.rates_after(b.from, c, &stats);
        let next = self.first_candidate(c, &rates);
        let ahead = if rates.is_empty() {
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
    /// frame first.
    fn accept(
        &mut self,
        c: usize,
        stats: FramePairStats,
        kind: StepKind,
    ) -> Result<(), PickImagesError> {
        let f = self.current.0;
        let rate = stats.step_percent / (c - f) as f64;
        self.rates = self.rates_after(f, c, &stats);
        self.walked.steps.push(WalkStep {
            from: f,
            to: c,
            stats,
            rate,
            kind,
        });
        self.walked.chosen.push(c);
        let features = match self.candidate.take() {
            Some((n, features)) if n == c => features,
            _ => self.frames.features(self.id(c))?,
        };
        self.current = (c, features);
        Ok(())
    }
}
