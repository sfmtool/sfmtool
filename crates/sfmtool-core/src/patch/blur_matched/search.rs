// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The width of a blur, found by measuring the blurred tile: blur, read the
//! self-similarity ellipse again with the reading the target was read with,
//! and correct the width until the two lengths agree.

use super::blur::{blur_tile, BlurScratch};
use super::tiles::TilePlanes;
use super::{
    estimated_blur_sigma, length_along, BlurCovariance, BlurDirection, BLUR_MATCH_MAX_PROBES,
    BLUR_MATCH_TOLERANCE, MAX_BLUR_SIGMA,
};

/// Buffers [`match_blur`] reuses between calls.
#[derive(Debug, Default, Clone)]
pub struct MatchScratch {
    blur: BlurScratch,
}

/// What [`match_blur`] settled on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchedBlur {
    /// The blur the tile in `out` was blurred by.
    pub cov: BlurCovariance,
    /// How many blurred tiles had their ellipse read.
    pub probes: u32,
    /// The largest relative difference, over the directions, between a
    /// blurred tile's length and the length aimed for, `|l − t| / t`; `0`
    /// along a direction already blurred by [`MAX_BLUR_SIGMA`] and still short
    /// of it. Within [`BLUR_MATCH_TOLERANCE`], it is that of `cov` itself.
    /// Where the probes ran out, it is the smallest read, and `cov` is either
    /// that blur or a step between two widths read, which is not read again.
    /// `NaN` where the last blurred tile could not be read.
    pub error: f64,
}

/// The search along one direction: `σ²` and the length² it gave, as a bracket.
#[derive(Debug, Clone, Copy)]
struct Search {
    /// `σ²` to try next.
    next: f64,
    /// The widest `σ²` known to fall short, with its length², starting at no
    /// blur.
    short: (f64, f64),
    /// The one before it, for extrapolating while nothing has overshot.
    before: Option<(f64, f64)>,
    /// The narrowest `σ²` known to overshoot, with its length².
    over: Option<(f64, f64)>,
    /// Whether the last reading was within the tolerance along this
    /// direction, so `next` is the width read.
    settled: bool,
}

impl Search {
    fn new(d: &BlurDirection) -> Self {
        let s = estimated_blur_sigma(d.blurrier, d.sharper);
        Self {
            next: s * s,
            short: (0.0, d.sharper * d.sharper),
            before: None,
            over: None,
            settled: false,
        }
    }

    /// Take in that `σ²` gave `length`, aiming for `target`, and choose the
    /// next `σ²`: a secant step on length² against `σ²` through the bracket's
    /// ends, or through the last two short points while nothing has overshot.
    fn update(&mut self, sigma2: f64, length: f64, target: f64) {
        let point = (sigma2, length * length);
        if length < target {
            self.before = Some(self.short);
            self.short = point;
        } else {
            self.over = Some(point);
        }
        let (a, b) = match (self.over, self.before) {
            (Some(over), _) => (self.short, over),
            (None, Some(before)) => (before, self.short),
            (None, None) => (self.short, self.short),
        };
        let slope = if b.0 > a.0 {
            (b.1 - a.1) / (b.0 - a.0)
        } else {
            0.0
        };
        let max2 = MAX_BLUR_SIGMA * MAX_BLUR_SIGMA;
        self.next = if slope > 1e-6 {
            a.0 + (target * target - a.1) / slope
        } else if let Some(over) = self.over {
            0.5 * (self.short.0 + over.0)
        } else {
            // No growth seen yet: try twice as wide.
            (2.0 * self.short.0).max(1.0 / 16.0)
        };
        if let Some(over) = self.over {
            // Stay strictly inside the bracket.
            let w = over.0 - self.short.0;
            self.next = self.next.clamp(self.short.0 + 0.05 * w, over.0 - 0.05 * w);
        }
        self.next = self.next.clamp(0.0, max2);
    }
}

/// Blur `tile` along each of `dirs` until, read by `read`, its self-similarity ellipse is as long
/// along each direction as [`BlurDirection::blurrier`], and leave the blurred
/// tile in `out`.
///
/// `read` must read the ellipse matrix the way the lengths in `dirs` were
/// read: the same reading of the same render, so that the blur is set by the
/// measurement it is compared with and any bias the reading has cancels.
///
/// The first blur tried is [`estimated_blur_sigma`] along each direction. The
/// blurred tile's ellipse is read, and where a length along a direction is
/// more than [`BLUR_MATCH_TOLERANCE`] off, that direction's `σ²` is corrected
/// by a secant step of length² against `σ²`, which grows about linearly, kept
/// inside the bracket of widths known to fall short and to overshoot. At most
/// [`BLUR_MATCH_MAX_PROBES`] blurred tiles are read. Where the last is still
/// off, the tile is blurred once more by the step past it, unread, if along
/// every direction still off that step lies between a width that fell short
/// and one that overshot; otherwise by the blur read closest to its targets.
/// A direction blurred by [`MAX_BLUR_SIGMA`] and still short is left there.
/// Where `read` gives no ellipse, the search stops at the blur it just tried.
///
/// # Panics
///
/// Panics if `tile`'s planes or data flags do not cover it, or `dirs` holds
/// more than two directions.
pub fn match_blur(
    tile: &TilePlanes,
    dirs: &[BlurDirection],
    mut read: impl FnMut(&[f32]) -> Option<[[f64; 2]; 2]>,
    out: &mut Vec<f32>,
    scratch: &mut MatchScratch,
) -> MatchedBlur {
    assert!(dirs.len() <= 2, "match_blur: at most two directions");
    let mut searches = [Search::new(&BlurDirection::default()); 2];
    for (s, d) in searches.iter_mut().zip(dirs) {
        *s = Search::new(d);
    }
    let searches = &mut searches[..dirs.len()];
    let (values, channels, side, data) = (&tile.values, tile.channels, tile.side, &tile.data);
    out.resize(values.len(), 0.0);
    let max2 = MAX_BLUR_SIGMA * MAX_BLUR_SIGMA;
    let mut probes = 0;
    // The blur read closest to its targets, and how far off it was.
    let mut best = (f64::INFINITY, BlurCovariance::ZERO);
    loop {
        let cov = searches
            .iter()
            .zip(dirs)
            .fold(BlurCovariance::ZERO, |cov, (s, d)| {
                cov.add(BlurCovariance::along(d.u, s.next.sqrt()))
            });
        if probes == BLUR_MATCH_MAX_PROBES {
            // Out of probes. The step past the last reading is taken, unread,
            // where it lies inside a bracket along every direction still off,
            // so between two widths read; otherwise the closest blur read.
            let bracketed = searches.iter().all(|s| s.settled || s.over.is_some());
            let cov = if bracketed { cov } else { best.1 };
            blur_tile(values, channels, side, data, cov, out, &mut scratch.blur);
            return MatchedBlur {
                cov,
                probes,
                error: best.0,
            };
        }
        blur_tile(values, channels, side, data, cov, out, &mut scratch.blur);
        probes += 1;
        let Some(e) = read(out) else {
            return MatchedBlur {
                cov,
                probes,
                error: f64::NAN,
            };
        };
        // Per direction: how far off the length is, `0` where the blur is
        // already the widest and still short.
        let mut off = [0.0f64; 2];
        let mut lengths = [0.0f64; 2];
        for (i, (s, d)) in searches.iter().zip(dirs).enumerate() {
            let l = length_along(&e, d.u);
            lengths[i] = l;
            let at_max = l < d.blurrier && s.next >= max2 * (1.0 - 1e-9);
            off[i] = if at_max {
                0.0
            } else {
                (l - d.blurrier).abs() / d.blurrier
            };
        }
        let error = off.iter().copied().fold(0.0, f64::max);
        if error <= BLUR_MATCH_TOLERANCE {
            return MatchedBlur { cov, probes, error };
        }
        if error < best.0 {
            best = (error, cov);
        }
        for (i, (s, d)) in searches.iter_mut().zip(dirs).enumerate() {
            s.settled = off[i] <= BLUR_MATCH_TOLERANCE;
            if !s.settled {
                let tried = s.next;
                s.update(tried, lengths[i], d.blurrier);
            }
        }
    }
}
