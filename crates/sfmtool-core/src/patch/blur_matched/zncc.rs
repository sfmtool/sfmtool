// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The ZNCC of two tiles over the samples that carry data in both, weighted by
//! a window: the reading a blur-matched score is taken with, on the tiles as
//! rendered or after one of them is blurred.

use super::tiles::TilePlanes;

/// The fewest samples with data in both tiles, and weight in the window, that
/// [`windowed_zncc`] reads a ZNCC over: the floor member coherence's common
/// support has (`MIN_MASK_PIXELS`).
pub const MIN_WINDOWED_SAMPLES: usize = 8;

/// Weighted sums of one channel of two tiles.
#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    w: f64,
    a: f64,
    b: f64,
    aa: f64,
    bb: f64,
    ab: f64,
}

impl Moments {
    #[inline(always)]
    fn add(&mut self, w: f64, x: f64, y: f64) {
        self.w += w;
        self.a += w * x;
        self.b += w * y;
        self.aa += w * x * x;
        self.bb += w * y * y;
        self.ab += w * x * y;
    }

    /// The ZNCC, or `None` where either side is flat.
    fn zncc(&self) -> Option<f64> {
        if self.w <= 0.0 {
            return None;
        }
        let va = self.aa - self.a * self.a / self.w;
        let vb = self.bb - self.b * self.b / self.w;
        if va <= 1e-6 || vb <= 1e-6 {
            return None;
        }
        Some((self.ab - self.a * self.b / self.w) / (va * vb).sqrt())
    }
}

/// The ZNCC of `a` and `b` over the samples that carry data in both and have
/// weight in `window` (one weight per sample, row-major, as
/// `window_weights` of the tile's side gives), per colour channel, the
/// channels' ZNCCs averaged.
///
/// A channel flat in either tile over those samples is left out of the
/// average. `NaN` with fewer than [`MIN_WINDOWED_SAMPLES`] such samples or no
/// channel left.
///
/// ```
/// use sfmtool_core::patch::blur_matched::{windowed_zncc, TilePlanes};
///
/// let side = 8;
/// let values: Vec<f32> = (0..side * side).map(|k| ((k * 37) % 11) as f32).collect();
/// let tile = TilePlanes { values, data: vec![true; side * side], side, channels: 1 };
/// let window = vec![1.0; side * side];
/// assert!((windowed_zncc(&tile, &tile, &window) - 1.0).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the tiles differ in side or `window` does not cover the tile.
pub fn windowed_zncc(a: &TilePlanes, b: &TilePlanes, window: &[f64]) -> f64 {
    assert_eq!(a.side, b.side, "the tiles differ in side");
    let n = a.side * a.side;
    assert_eq!(window.len(), n, "the window does not cover the tile");
    let channels = a.channels.min(b.channels).min(3);
    let mut moments = [Moments::default(); 3];
    let mut samples = 0usize;
    for (k, &w) in window.iter().enumerate() {
        if w <= 0.0 || !(a.data[k] && b.data[k]) {
            continue;
        }
        samples += 1;
        for (c, m) in moments.iter_mut().enumerate().take(channels) {
            m.add(
                w,
                f64::from(a.values[c * n + k]),
                f64::from(b.values[c * n + k]),
            );
        }
    }
    if samples < MIN_WINDOWED_SAMPLES {
        return f64::NAN;
    }
    let (total, used) = moments[..channels]
        .iter()
        .filter_map(Moments::zncc)
        .fold((0.0, 0usize), |(t, u), z| (t + z, u + 1));
    if used == 0 {
        f64::NAN
    } else {
        total / used as f64
    }
}
