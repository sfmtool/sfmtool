// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk along one sequence: predict the next frame from the recent rate,
//! measure the step to it, accept it, re-aim, or swap a blurred frame for a
//! sharp neighbour; halve the gap where a step fails, and start a new segment
//! after a break. See `specs/core/analysis/pick-images.md` § "The walk".

use std::collections::{BTreeSet, VecDeque};

use super::baseline::{matches_well, SequenceBaseline};
use super::capture::{FrameId, PickImagesError, PickImagesOptions};
use super::frame_pair::{FrameFeatures, FramePairStats};
use super::frames::Frames;
use super::report::{
    Break, Gap, KeptBlurred, Segment, Stage, StepKind, Swap, WalkReport, WalkStep, WeakPoint,
};
use crate::numeric::median;
use crate::progress::Progress;

/// How many times a step is re-aimed at the band before it fails.
const MAX_REAIMS: usize = 4;

/// How many accepted steps' rates predict the next step.
const RECENT_RATES: usize = 3;

/// A step with fewer than this many times `min_matches` matches is a weak
/// point.
const WEAK_MATCH_FACTOR: usize = 2;

/// A step whose F-inlier share is within this of `min_inlier_share` is a weak
/// point.
const WEAK_SHARE_MARGIN: f64 = 0.1;

/// A segment of fewer chosen frames than this is a weak point.
const MIN_SEGMENT_FRAMES: usize = 3;

/// The bounds one step works within.
#[derive(Clone, Copy)]
struct StepBounds {
    from: usize,
    /// The band, halved with the target in a flagged region.
    low: f64,
    high: f64,
    target: f64,
    /// The farthest frame the step may reach: the sequence's last frame, or
    /// the first frame of the next flagged region.
    limit: usize,
}

/// What one step ended in.
enum Outcome {
    Accepted,
    Break,
}

/// A frame chosen with the step that reached it.
struct Choice {
    frame: usize,
    stats: FramePairStats,
    kind: StepKind,
    kept_blurred: Option<(usize, usize)>,
}

struct Walker<'w, 'a> {
    frames: &'w mut Frames<'a>,
    baseline: &'w SequenceBaseline,
    options: &'w PickImagesOptions,
    progress: &'w Progress<'w>,
    sequence: usize,
    last: usize,
    /// The last chosen frame and its features.
    current: (usize, FrameFeatures),
    /// The last candidate measured and its features, reused when it is
    /// accepted.
    candidate: Option<(usize, FrameFeatures)>,
    rates: VecDeque<f64>,
    used: BTreeSet<usize>,
    report: WalkReport,
    /// Whether the last step pushed was a gap fill in the current segment.
    in_gap: bool,
}

/// Walk one sequence that passed the verdict.
pub(super) fn walk_sequence(
    frames: &mut Frames,
    baseline: &SequenceBaseline,
    sequence: usize,
    len: usize,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<WalkReport, PickImagesError> {
    let mut report = WalkReport {
        chosen: Vec::new(),
        segments: Vec::new(),
        steps: Vec::new(),
        swaps: Vec::new(),
        gaps: Vec::new(),
        kept_blurred: Vec::new(),
        breaks: Vec::new(),
        weak_points: Vec::new(),
        frames_used: 0,
        frames_from_baseline: 0,
    };
    let mut used = BTreeSet::new();
    let last = len - 1;
    let Some((start, features)) =
        find_start(frames, sequence, 0, last, &mut used, options, progress)?
    else {
        finish(&mut report, &used, frames, sequence, options);
        return Ok(report);
    };
    report.segments.push(Segment { start: 0, len: 1 });
    report.chosen.push(start);
    let mut walker = Walker {
        frames: &mut *frames,
        baseline,
        options,
        progress,
        sequence,
        last,
        current: (start, features),
        candidate: None,
        rates: VecDeque::new(),
        used,
        report,
        in_gap: false,
    };
    while walker.current.0 < walker.last {
        match walker.step()? {
            Outcome::Accepted => {}
            Outcome::Break => {
                let after = walker.current.0;
                let resumed = find_start(
                    walker.frames,
                    sequence,
                    after + 1,
                    last,
                    &mut walker.used,
                    options,
                    progress,
                )?;
                walker.report.breaks.push(Break {
                    after,
                    resumes_at: resumed.as_ref().map(|(g, _)| *g),
                });
                let Some((g, features)) = resumed else {
                    break;
                };
                walker.report.segments.push(Segment {
                    start: walker.report.chosen.len(),
                    len: 1,
                });
                walker.report.chosen.push(g);
                walker.current = (g, features);
                walker.candidate = None;
                walker.rates.clear();
                walker.in_gap = false;
            }
        }
    }
    let Walker {
        mut report, used, ..
    } = walker;
    finish(&mut report, &used, frames, sequence, options);
    Ok(report)
}

/// The first frame from `from` on that matches its next frame well, with its
/// features, reading the frames one pair at a time.
fn find_start(
    frames: &mut Frames,
    sequence: usize,
    from: usize,
    last: usize,
    used: &mut BTreeSet<usize>,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<Option<(usize, FrameFeatures)>, PickImagesError> {
    let id = |frame| FrameId { sequence, frame };
    let mut held: Option<(usize, FrameFeatures)> = None;
    for g in from..last {
        frames.ensure(&[id(g), id(g + 1)], Stage::Walk, progress)?;
        used.extend([g, g + 1]);
        let fg = match held.take() {
            Some((h, features)) if h == g => features,
            _ => frames.features(id(g))?,
        };
        let next = frames.features(id(g + 1))?;
        let stats = frames.match_pair(id(g), &fg, id(g + 1), &next, Stage::Walk, progress)?;
        if matches_well(&stats, options) {
            return Ok(Some((g, fg)));
        }
        held = Some((g + 1, next));
    }
    Ok(None)
}

/// Fill in what the walk's report derives at the end: the weak points and
/// the counts of frames used.
fn finish(
    report: &mut WalkReport,
    used: &BTreeSet<usize>,
    frames: &Frames,
    sequence: usize,
    options: &PickImagesOptions,
) {
    let min_matches = options.pair.min_matches;
    for (i, step) in report.steps.iter().enumerate() {
        let few_matches = step.stats.matches < WEAK_MATCH_FACTOR * min_matches;
        let low_inlier_share = step
            .stats
            .f_inlier_share
            .is_some_and(|s| s < options.min_inlier_share + WEAK_SHARE_MARGIN);
        if few_matches || low_inlier_share {
            report.weak_points.push(WeakPoint::Step {
                step: i,
                few_matches,
                low_inlier_share,
            });
        }
    }
    for (i, segment) in report.segments.iter().enumerate() {
        if segment.len < MIN_SEGMENT_FRAMES {
            report
                .weak_points
                .push(WeakPoint::ShortSegment { segment: i });
        }
    }
    report.frames_used = used.len();
    report.frames_from_baseline = frames
        .read_frames()
        .iter()
        .filter(|f| {
            f.id.sequence == sequence && f.stage != Stage::Walk && used.contains(&f.id.frame)
        })
        .count();
}

impl Walker<'_, '_> {
    fn id(&self, frame: usize) -> FrameId {
        FrameId {
            sequence: self.sequence,
            frame,
        }
    }

    /// Read the frames of `positions` not yet read, in one call.
    fn read(&mut self, positions: &[usize]) -> Result<(), PickImagesError> {
        let ids: Vec<FrameId> = positions.iter().map(|&p| self.id(p)).collect();
        self.frames.ensure(&ids, Stage::Walk, self.progress)?;
        self.used.extend(positions.iter().copied());
        Ok(())
    }

    /// Whether a frame read is blurred, and its sharpness over the median.
    fn blur(&self, frame: usize) -> (bool, f64) {
        self.baseline.blur(
            self.frames.readings(self.id(frame)),
            self.options.blur_ratio,
        )
    }

    fn blurred(&self, frame: usize) -> bool {
        self.blur(frame).0
    }

    /// Read `frame` and match it with the last chosen frame.
    fn measure(&mut self, frame: usize) -> Result<FramePairStats, PickImagesError> {
        self.read(&[frame])?;
        let (a, b) = (self.id(self.current.0), self.id(frame));
        if let Some(stats) = self.frames.pair(a, b) {
            return Ok(stats.clone());
        }
        let features = self.frames.features(b)?;
        let stats =
            self.frames
                .match_pair(a, &self.current.1, b, &features, Stage::Walk, self.progress)?;
        self.candidate = Some((frame, features));
        Ok(stats)
    }

    /// The rate the next step is predicted from: the median of the last
    /// three accepted steps' rates, or the baseline's near the current frame
    /// at the start of a segment.
    fn predicted_rate(&self) -> f64 {
        if self.rates.is_empty() {
            self.baseline.rate_near(self.current.0)
        } else {
            let rates: Vec<f64> = self.rates.iter().copied().collect();
            median(&rates)
        }
    }

    fn bounds(&self) -> StepBounds {
        let from = self.current.0;
        let flagged = self.baseline.is_flagged(from);
        let target = self.options.target_step_percent * if flagged { 0.5 } else { 1.0 };
        let limit = if flagged {
            self.last
        } else {
            self.baseline
                .next_flagged(from)
                .map_or(self.last, |f| f.min(self.last))
        };
        StepBounds {
            from,
            low: target * (1.0 - self.options.step_band),
            high: target * (1.0 + self.options.step_band),
            target,
            limit,
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
    fn step(&mut self) -> Result<Outcome, PickImagesError> {
        let b = self.bounds();
        let f = b.from;
        let mut c = Self::aim(f, b.target / self.predicted_rate(), f + 1, b.limit);
        // The largest candidate below the band that matched well, and the
        // smallest above it.
        let mut below: Option<(usize, FramePairStats)> = None;
        let mut above: Option<usize> = None;
        let mut reaims = 0;
        loop {
            let stats = self.measure(c)?;
            let m = stats.step_percent;
            let well = matches_well(&stats, self.options);
            if m > b.high {
                above = Some(above.map_or(c, |a| a.min(c)));
                if c == f + 1 {
                    return if well {
                        self.settle(b, c, stats, StepKind::AboveBand)
                    } else {
                        self.fill_gap(b, c, below)
                    };
                }
            } else if m >= b.low {
                return if well {
                    self.settle(b, c, stats, StepKind::InBand)
                } else {
                    self.fill_gap(b, c, below)
                };
            } else {
                // Below the band, or no matches at all.
                if !well {
                    return self.fill_gap(b, c, below);
                }
                if c == b.limit {
                    return self.settle(b, c, stats, StepKind::AtLimit);
                }
                if below.as_ref().is_none_or(|(x, _)| c > *x) {
                    below = Some((c, stats.clone()));
                }
            }
            let low = below.as_ref().map_or(f + 1, |(x, _)| x + 1);
            let high = above.map_or(b.limit, |a| a - 1);
            if low > high {
                // Two candidates one frame apart lie on either side of the
                // band: the motion per frame is wider than the band.
                return match below {
                    Some((x, s)) => self.settle(b, x, s, StepKind::Bracketed),
                    None => self.fill_gap(b, c, None),
                };
            }
            if reaims == MAX_REAIMS {
                return self.fill_gap(b, c, below);
            }
            reaims += 1;
            c = Self::aim(f, (c - f) as f64 * b.target / m, low, high);
        }
    }

    /// Accept `c`, which `step` reached with `stats`, or a sharp frame in its
    /// place when it is blurred.
    fn settle(
        &mut self,
        b: StepBounds,
        c: usize,
        stats: FramePairStats,
        kind: StepKind,
    ) -> Result<Outcome, PickImagesError> {
        if !self.blurred(c) {
            return self.accept(Choice {
                frame: c,
                stats,
                kind,
                kept_blurred: None,
            });
        }
        let f = b.from;
        let rate = stats.step_percent / (c - f) as f64;
        let reach = |step: f64| f as f64 + step / rate;
        // The frames whose step, at the candidate's rate, is in the band, and
        // the candidate's own neighbours when it is not.
        let first = (reach(b.low).ceil() as usize).clamp(f + 1, c);
        let last = (reach(b.high).floor() as usize).clamp(c, b.limit);
        let floor = stats.step_percent.min(b.low);

        // A sharp neighbour, nearest first.
        for k in 1..=(c - first).max(last - c) {
            let near: Vec<usize> = [c.checked_sub(k), c.checked_add(k)]
                .into_iter()
                .flatten()
                .filter(|&n| first <= n && n <= last)
                .collect();
            self.read(&near)?;
            for n in near {
                if self.blurred(n) {
                    continue;
                }
                let s = self.measure(n)?;
                if matches_well(&s, self.options)
                    && s.step_percent >= floor
                    && s.step_percent <= b.high
                {
                    self.report.swaps.push(Swap {
                        candidate: c,
                        chosen: n,
                    });
                    return self.accept(Choice {
                        frame: n,
                        stats: s,
                        kind,
                        kept_blurred: None,
                    });
                }
            }
        }

        // A blurred run: every frame up to the band's upper edge, the
        // farthest sharp one that matches well, or else the sharpest.
        let run: Vec<usize> = (f + 1..=last).collect();
        self.read(&run)?;
        let fits = |s: &FramePairStats, n: usize, walker: &Self| {
            matches_well(s, walker.options) && (s.step_percent <= b.high || n == f + 1)
        };
        for &n in run.iter().rev() {
            if self.blurred(n) {
                continue;
            }
            let s = self.measure(n)?;
            if fits(&s, n, self) {
                return self.accept(Choice {
                    frame: n,
                    stats: s,
                    kind: StepKind::BlurredRun,
                    kept_blurred: None,
                });
            }
        }
        let mut by_sharpness: Vec<(usize, f64)> =
            run.iter().map(|&n| (n, self.blur(n).1)).collect();
        by_sharpness.sort_by(|x, y| y.1.total_cmp(&x.1).then(y.0.cmp(&x.0)));
        for (n, _) in by_sharpness {
            let s = self.measure(n)?;
            if fits(&s, n, self) {
                return self.accept(Choice {
                    frame: n,
                    stats: s,
                    kind: StepKind::BlurredRun,
                    kept_blurred: Some((f + 1, last)),
                });
            }
        }
        // Not reached: `c` itself matched well within the band's upper edge
        // or is the next frame.
        self.fill_gap(b, c, None)
    }

    /// The step to `failed` did not match well or could not be brought into
    /// the band: accept the largest candidate below the band that matched
    /// well, or else halve the gap until a frame matches well, down to the
    /// next frame. A break when even that fails.
    fn fill_gap(
        &mut self,
        b: StepBounds,
        failed: usize,
        below: Option<(usize, FramePairStats)>,
    ) -> Result<Outcome, PickImagesError> {
        let f = b.from;
        if let Some((x, stats)) = below {
            return self.accept_gap(f, x, failed.max(x + 1), stats);
        }
        let mut failed = failed;
        loop {
            let h = f + (failed - f) / 2;
            if h == f {
                return Ok(Outcome::Break);
            }
            let stats = self.measure(h)?;
            if matches_well(&stats, self.options) {
                return self.accept_gap(f, h, failed, stats);
            }
            failed = h;
        }
    }

    /// Accept `h` as a gap fill, or a sharp neighbour of it before `bound`
    /// when it is blurred, or `h` kept though blurred.
    fn accept_gap(
        &mut self,
        f: usize,
        h: usize,
        bound: usize,
        stats: FramePairStats,
    ) -> Result<Outcome, PickImagesError> {
        if self.blurred(h) {
            let near: Vec<usize> = [h - 1, h + 1]
                .into_iter()
                .filter(|&n| n > f && n < bound && n <= self.last)
                .collect();
            self.read(&near)?;
            for &n in &near {
                if self.blurred(n) {
                    continue;
                }
                let s = self.measure(n)?;
                if matches_well(&s, self.options) {
                    self.report.swaps.push(Swap {
                        candidate: h,
                        chosen: n,
                    });
                    return self.accept(Choice {
                        frame: n,
                        stats: s,
                        kind: StepKind::GapFill,
                        kept_blurred: None,
                    });
                }
            }
            let searched = near.iter().copied().chain([h]);
            let first = searched.clone().min().unwrap_or(h);
            let last = searched.max().unwrap_or(h);
            return self.accept(Choice {
                frame: h,
                stats,
                kind: StepKind::GapFill,
                kept_blurred: Some((first, last)),
            });
        }
        self.accept(Choice {
            frame: h,
            stats,
            kind: StepKind::GapFill,
            kept_blurred: None,
        })
    }

    /// Make `choice` the last chosen frame.
    fn accept(&mut self, choice: Choice) -> Result<Outcome, PickImagesError> {
        let f = self.current.0;
        let c = choice.frame;
        let rate = choice.stats.step_percent / (c - f) as f64;
        let flagged = self.baseline.is_flagged(f);
        if let Some((first, last)) = choice.kept_blurred {
            self.report.kept_blurred.push(KeptBlurred {
                frame: c,
                searched_first: first,
                searched_last: last,
                sharpness_ratio: self.blur(c).1,
            });
        }
        let step = self.report.steps.len();
        if choice.kind == StepKind::GapFill {
            match self.report.gaps.last_mut() {
                Some(gap) if self.in_gap => gap.end_step = step + 1,
                _ => self.report.gaps.push(Gap {
                    first_step: step,
                    end_step: step + 1,
                }),
            }
            self.in_gap = true;
        } else {
            self.in_gap = false;
        }
        self.report.steps.push(WalkStep {
            from: f,
            to: c,
            stats: choice.stats,
            rate,
            kind: choice.kind,
            flagged,
        });
        self.rates.push_back(rate);
        if self.rates.len() > RECENT_RATES {
            self.rates.pop_front();
        }
        self.report.chosen.push(c);
        if let Some(segment) = self.report.segments.last_mut() {
            segment.len += 1;
        }
        let features = match self.candidate.take() {
            Some((n, features)) if n == c => features,
            _ => self.frames.features(self.id(c))?,
        };
        self.current = (c, features);
        Ok(Outcome::Accepted)
    }
}
