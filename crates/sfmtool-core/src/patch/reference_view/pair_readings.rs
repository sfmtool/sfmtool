// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pairwise readings of a track's views' tiles, plain and blur-matched:
//! each pair correlated over the samples that carry data in both, over the
//! whole tile with a window and over each cell of the ZNCC grid. A
//! blur-matched pair's sharper tile is blurred first, by the pairing rule of
//! [`pair_sharpness`](crate::patch::pair_sharpness) and the blur of
//! [`blur_matched`](crate::patch::blur_matched).

use super::agreement::finite_middle;
use super::REFERENCE_MIN_CELL_SAMPLES;
use crate::patch::blur_matched::{
    assess_blur, blur_to_length_into, read_tile_ellipse, BlurScratch, TilePlanes,
    GROWTH_PROBE_SIGMAS,
};
use crate::patch::normal_refine::{grid_bounds, window_weights, PatchWindow};
use crate::patch::pair_sharpness::{PairMatching, TrackBlurs};
use crate::progress::Progress;
use crate::progress_note;

/// The fewest samples with data in both tiles, and inside the window, that a
/// whole-tile reading of [`pair_zncc_readings`] is taken over: the floor
/// member coherence's common support has (`MIN_MASK_PIXELS`).
pub const MIN_WINDOWED_SAMPLES: usize = 8;

/// Two tiles' ZNCC over the whole tile and over each cell of the ZNCC grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairReadings {
    /// The windowed ZNCC over the samples with data in both tiles and weight in
    /// the window, per colour channel, the channels averaged; `NaN` under
    /// [`MIN_WINDOWED_SAMPLES`] samples or with no channel textured in both.
    pub whole: f64,
    /// Per cell, `grid[row][col]` from the top-left cell: the ZNCC with every
    /// sample weighted equally, read as
    /// [`pair_zncc_grid`](super::pair_zncc_grid) reads
    /// it.
    pub grid: [[f64; 3]; 3],
}

/// Sums of one channel of two tiles over a set of samples, with weights.
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

/// The channel average of the channels' ZNCCs, `NaN` with fewer than `min`
/// samples or no channel textured in both tiles.
fn channel_mean(moments: &[Moments], samples: usize, min: usize) -> f64 {
    if samples < min {
        return f64::NAN;
    }
    let (total, used) = moments
        .iter()
        .filter_map(Moments::zncc)
        .fold((0.0, 0usize), |(t, u), z| (t + z, u + 1));
    if used == 0 {
        f64::NAN
    } else {
        total / used as f64
    }
}

/// [`PairReadings`] of two tiles as they are, `window` one weight per sample
/// (`window_weights` of the tile's side).
///
/// One walk over the samples with data in both tiles gathers the whole
/// tile's windowed sums and each cell's plain ones, so a cell reads exactly
/// what [`pair_zncc_grid`](super::pair_zncc_grid) reads
/// on the same samples.
///
/// # Panics
///
/// Panics if the tiles differ in side or `window` does not cover the tile.
pub fn pair_zncc_readings(a: &TilePlanes, b: &TilePlanes, window: &[f64]) -> PairReadings {
    assert_eq!(a.side, b.side, "the tiles differ in side");
    let side = a.side;
    let n = side * side;
    assert_eq!(window.len(), n, "the window does not cover the tile");
    let channels = a.channels.min(b.channels).min(3);
    let bounds = grid_bounds(side as u32);
    let cell_of = |i: usize| (1..3).filter(|&c| i >= bounds[c]).count();
    let mut whole = [Moments::default(); 3];
    let mut whole_samples = 0usize;
    let mut cells = [[Moments::default(); 3]; 9];
    let mut cell_samples = [0usize; 9];
    for row in 0..side {
        let ci = cell_of(row);
        for col in 0..side {
            let k = row * side + col;
            if !(a.data[k] && b.data[k]) {
                continue;
            }
            let cell = ci * 3 + cell_of(col);
            cell_samples[cell] += 1;
            let w = window[k];
            if w > 0.0 {
                whole_samples += 1;
            }
            for c in 0..channels {
                let x = f64::from(a.values[c * n + k]);
                let y = f64::from(b.values[c * n + k]);
                cells[cell][c].add(1.0, x, y);
                if w > 0.0 {
                    whole[c].add(w, x, y);
                }
            }
        }
    }
    let mut grid = [[f64::NAN; 3]; 3];
    for (cell, m) in cells.iter().enumerate() {
        grid[cell / 3][cell % 3] = channel_mean(
            &m[..channels],
            cell_samples[cell],
            REFERENCE_MIN_CELL_SAMPLES,
        );
    }
    PairReadings {
        whole: channel_mean(&whole[..channels], whole_samples, MIN_WINDOWED_SAMPLES),
        grid,
    }
}

/// The pairwise readings of a track's views' tiles, each pair blur-matched.
#[derive(Debug, Clone, PartialEq)]
pub struct BlurMatchedPairs {
    /// The view count `k`.
    pub k: usize,
    /// Row-major `k×k` [`PairReadings::whole`]; the diagonal is `1`.
    pub whole: Vec<f64>,
    /// Row-major `k×k` [`PairReadings::grid`]; the diagonal is `NaN`
    /// throughout.
    pub grid: Vec<[[f64; 3]; 3]>,
    /// Row-major `k×k`: whether a tile of the pair was blurred.
    pub blurred: Vec<bool>,
    /// Row-major `k×k`: the width, in grid px, by which the row's tile was
    /// blurred against the column's; `0` where it was not blurred.
    pub sigma: Vec<f64>,
    /// How many pairs were read.
    pub pairs: usize,
    /// How many of them were blurred.
    pub pairs_blurred: usize,
    /// How many blurred tiles had their ellipse read: one per probe for each
    /// view some pair blurs, to assess its blur ([`TrackBlurs`]).
    pub ellipse_reads: usize,
}

impl BlurMatchedPairs {
    /// The middle of view `v`'s whole-tile readings with the other views, their
    /// median by `finite_middle`, `NaN` where it has none.
    pub fn row_middle(&self, v: usize) -> f64 {
        let others: Vec<f64> = (0..self.k)
            .filter(|&w| w != v)
            .map(|w| self.whole[v * self.k + w])
            .collect();
        finite_middle(&others)
    }
}

/// Read every pair of `tiles` with [`pair_zncc_readings`] after blur-matching
/// it under `matching`, `ellipses[v]` view `v`'s self-similarity ellipse
/// matrix in grid px² (`None` where it has none, which leaves every pair it
/// is in plain).
///
/// Each view some pair blurs is assessed once ([`assess_blur`], through
/// [`TrackBlurs`]), its blurred probes read with [`read_tile_ellipse`]. A pair
/// is blurred where [`TrackBlurs::pair`] names a sharper view: that view's
/// tile is blurred to the target ([`blur_to_length_into`]) and the other tile
/// is read as it is. Where the view's assessment cannot be read, or does not
/// grow, the pair is read plain.
///
/// The ellipses must be the whole-tile reading over the samples with data
/// ([`read_tile_ellipse`]), the reading the probes are read with, so the
/// lengths compared are of one reading. Each view's assessment depends on its
/// own tile alone, so a pair reads the same whatever other pairs are read
/// with it.
///
/// The blur work is timed in the `blur-matched pairs` detail phase of
/// `progress`, with a note of how many pairs were blurred and how many blurred
/// tiles were read.
///
/// # Panics
///
/// Panics if `ellipses` is not parallel to `tiles`, or the tiles differ in
/// side.
pub fn blur_matched_pairs(
    tiles: &[&TilePlanes],
    ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching,
    window: PatchWindow,
    progress: &Progress<'_>,
) -> BlurMatchedPairs {
    let k = tiles.len();
    assert_eq!(ellipses.len(), k, "ellipses must be parallel to tiles");
    let mut phase = progress.detail_phase("blur-matched pairs");
    let mut whole = vec![f64::NAN; k * k];
    let mut grid = vec![[[f64::NAN; 3]; 3]; k * k];
    let mut blurred = vec![false; k * k];
    let mut widths = vec![0.0; k * k];
    for v in 0..k {
        whole[v * k + v] = 1.0;
    }
    let side = tiles.first().map_or(0, |t| t.side);
    let weights = window_weights(window, side as u32);
    let mut scratch = BlurScratch::default();
    let blurs = TrackBlurs::assess(ellipses, matching, |v, e| {
        let t = tiles[v];
        let read = |values: &[f32]| read_tile_ellipse(values, t.channels, t.side, &t.data);
        assess_blur(t, e, read, &mut scratch)
    });
    // The blurred tile, reused pair after pair.
    let mut sharp_blurred = TilePlanes::default();
    let (mut pairs, mut pairs_blurred) = (0usize, 0usize);
    for a in 0..k {
        for b in (a + 1)..k {
            pairs += 1;
            let blur = blurs.pair(a, b).and_then(|p| {
                let sigma = blur_to_length_into(
                    tiles[p.view],
                    p.assessment,
                    p.target,
                    &mut sharp_blurred,
                    &mut scratch,
                )?;
                (sigma > 0.0).then_some((p.view, sigma))
            });
            let readings = match blur {
                None => pair_zncc_readings(tiles[a], tiles[b], &weights),
                Some((sharp, sigma)) => {
                    let other = if sharp == a { b } else { a };
                    pairs_blurred += 1;
                    blurred[a * k + b] = true;
                    blurred[b * k + a] = true;
                    widths[sharp * k + other] = sigma;
                    // Read in the pair's own order, so a pair reads the same
                    // whichever of its tiles is blurred.
                    if sharp == a {
                        pair_zncc_readings(&sharp_blurred, tiles[other], &weights)
                    } else {
                        pair_zncc_readings(tiles[other], &sharp_blurred, &weights)
                    }
                }
            };
            whole[a * k + b] = readings.whole;
            whole[b * k + a] = readings.whole;
            grid[a * k + b] = readings.grid;
            grid[b * k + a] = readings.grid;
        }
    }
    let reads = blurs.assessed() * GROWTH_PROBE_SIGMAS.len();
    progress_note!(
        phase,
        "{pairs_blurred} of {pairs} pairs blurred, {reads} blurred tiles read"
    );
    BlurMatchedPairs {
        k,
        whole,
        grid,
        blurred,
        sigma: widths,
        pairs,
        pairs_blurred,
        ellipse_reads: reads,
    }
}

#[cfg(test)]
mod tests;
