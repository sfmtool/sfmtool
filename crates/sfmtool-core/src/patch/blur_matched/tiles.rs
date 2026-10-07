// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur-matched ZNCC between whole tiles: each pair correlated over the
//! samples that carry data in both, over the whole tile with a window and over
//! each cell of the ZNCC grid, after the blur [`pair_blur`] chooses, and the
//! same over every pair of a track's views.

use super::blur::{blur_tile, BlurScratch};
use super::{
    isotropic_blur_sigma, ladder_level, pair_blur, BlurCovariance, PairBlur, PairMatching,
    LADDER_SIGMAS,
};
use crate::patch::normal_refine::{grid_bounds, window_weights, PatchWindow};
use crate::patch::reference_view::REFERENCE_MIN_CELL_SAMPLES;
use crate::progress::Progress;
use crate::progress_note;

/// The fewest samples with data in both tiles, and inside the window, that a
/// whole-tile reading of [`pair_zncc_readings`] is taken over: the floor
/// member coherence's common support has (`MIN_MASK_PIXELS`).
pub const MIN_WINDOWED_SAMPLES: usize = 8;

/// One view's tile as planar colour channels in `f32`, with the samples that
/// carry data.
#[derive(Debug, Clone, PartialEq)]
pub struct TilePlanes {
    /// `channels` planes of `side × side`, row-major.
    pub values: Vec<f32>,
    /// Per sample, row-major: whether it carries data.
    pub data: Vec<bool>,
    /// The tile's side, `R`.
    pub side: usize,
    /// The colour channels, 1 to 3.
    pub channels: usize,
}

impl TilePlanes {
    /// The colour planes of an interleaved `side × side × stride` `u8` tile,
    /// its first `min(stride, 3)` channels, with `data` flagging the samples
    /// that carry data.
    ///
    /// # Panics
    ///
    /// Panics if `samples` or `data` do not cover the tile, or `stride` is 0.
    pub fn from_interleaved(samples: &[u8], side: usize, stride: usize, data: &[bool]) -> Self {
        assert!(stride > 0, "TilePlanes: no channels");
        assert_eq!(
            samples.len(),
            side * side * stride,
            "TilePlanes: samples do not cover the tile"
        );
        assert_eq!(
            data.len(),
            side * side,
            "TilePlanes: one data flag per sample"
        );
        let channels = stride.min(3);
        let n = side * side;
        let mut values = vec![0.0f32; channels * n];
        for (k, pixel) in samples.chunks_exact(stride).enumerate() {
            for c in 0..channels {
                values[c * n + k] = f32::from(pixel[c]);
            }
        }
        Self {
            values,
            data: data.to_vec(),
            side,
            channels,
        }
    }

    /// The same tile blurred by `cov` ([`blur_tile`]).
    pub fn blurred(&self, cov: BlurCovariance, scratch: &mut BlurScratch) -> Self {
        let mut values = vec![0.0f32; self.values.len()];
        blur_tile(
            &self.values,
            self.channels,
            self.side,
            &self.data,
            cov,
            &mut values,
            scratch,
        );
        Self {
            values,
            data: self.data.clone(),
            side: self.side,
            channels: self.channels,
        }
    }
}

/// Two tiles' ZNCC over the whole tile and over each cell of the ZNCC grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairReadings {
    /// The windowed ZNCC over the samples with data in both tiles and weight in
    /// the window, per colour channel, the channels averaged; `NaN` under
    /// [`MIN_WINDOWED_SAMPLES`] samples or with no channel textured in both.
    pub whole: f64,
    /// Per cell, `grid[row][col]` from the top-left cell: the ZNCC with every
    /// sample weighted equally, read as
    /// [`pair_zncc_grid`](crate::patch::reference_view::pair_zncc_grid) reads
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
/// what [`pair_zncc_grid`](crate::patch::reference_view::pair_zncc_grid) reads
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

// The shape of the blur a pair is matched with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlurMatchKernel {
    /// [`pair_blur`]: each tile blurred along the directions in which its
    /// ellipse is the shorter, by the anisotropic mapping. Every pair blurs
    /// its own tiles.
    #[default]
    Anisotropic,
    /// The sharper tile, by semi-major axis, blurred isotropically by the
    /// isotropic mapping, the width snapped to the nearest of
    /// [`LADDER_SIGMAS`], so each view is blurred at most once per level
    /// however many pairs it is in.
    IsotropicLadder,
}

impl BlurMatchKernel {
    /// The name the bindings spell it with: `"anisotropic"` or
    /// `"isotropic_ladder"`.
    pub fn name(self) -> &'static str {
        match self {
            BlurMatchKernel::Anisotropic => "anisotropic",
            BlurMatchKernel::IsotropicLadder => "isotropic_ladder",
        }
    }

    /// Parse [`Self::name`].
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "anisotropic" => Some(BlurMatchKernel::Anisotropic),
            "isotropic_ladder" => Some(BlurMatchKernel::IsotropicLadder),
            _ => None,
        }
    }
}

/// The pairwise readings of a track's views' tiles, each pair blur-matched.
#[derive(Debug, Clone, PartialEq)]
pub struct BlurMatchedPairs {
    /// The view count `k`.
    pub k: usize,
    /// Row-major `k×k` [`PairReadings::whole`]; the diagonal is `1`, and a
    /// pair not read (outside `rows`) is `NaN`.
    pub whole: Vec<f64>,
    /// Row-major `k×k` [`PairReadings::grid`]; the diagonal and a pair not read
    /// are `NaN` throughout.
    pub grid: Vec<[[f64; 3]; 3]>,
    /// Row-major `k×k`: whether either tile of the pair was blurred.
    pub blurred: Vec<bool>,
    /// How many pairs were read.
    pub pairs: usize,
    /// How many of them were blurred.
    pub pairs_blurred: usize,
}

impl BlurMatchedPairs {
    /// The middle of view `v`'s whole-tile readings with the other views, their
    /// median by `finite_middle`, `NaN` where it has none.
    pub fn row_middle(&self, v: usize) -> f64 {
        let others: Vec<f64> = (0..self.k)
            .filter(|&w| w != v)
            .map(|w| self.whole[v * self.k + w])
            .collect();
        crate::patch::reference_view::finite_middle(&others)
    }
}

/// The semi-major axis of the ellipse `E`: the square root of its larger
/// eigenvalue.
fn semi_major(e: &[[f64; 2]; 2]) -> f64 {
    let (a, b, d) = (e[0][0], 0.5 * (e[0][1] + e[1][0]), e[1][1]);
    (0.5 * (a + d + ((a - d) * (a - d) + 4.0 * b * b).sqrt()))
        .max(0.0)
        .sqrt()
}

/// Read every pair of `tiles` (or every pair with a view in `rows`, when
/// given) with [`pair_zncc_readings`] after blur-matching it under `matching`
/// with `kernel`, `ellipses[v]` view `v`'s self-similarity ellipse matrix in
/// grid px² (`None` where it has none, which leaves every pair it is in plain).
///
/// The blur work is timed in the `blur-matched pairs` detail phase of
/// `progress`, with a note of how many pairs were blurred.
///
/// # Panics
///
/// Panics if `ellipses` or `rows` is not parallel to `tiles`, or the tiles
/// differ in side.
pub fn blur_matched_pairs(
    tiles: &[&TilePlanes],
    ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching,
    kernel: BlurMatchKernel,
    window: PatchWindow,
    rows: Option<&[bool]>,
    progress: &Progress<'_>,
) -> BlurMatchedPairs {
    let k = tiles.len();
    assert_eq!(ellipses.len(), k, "ellipses must be parallel to tiles");
    if let Some(r) = rows {
        assert_eq!(r.len(), k, "rows must be parallel to tiles");
    }
    let mut phase = progress.detail_phase("blur-matched pairs");
    let mut whole = vec![f64::NAN; k * k];
    let mut grid = vec![[[f64::NAN; 3]; 3]; k * k];
    let mut blurred = vec![false; k * k];
    for v in 0..k {
        whole[v * k + v] = 1.0;
    }
    let side = tiles.first().map_or(0, |t| t.side);
    let weights = window_weights(window, side as u32);
    let mut scratch = BlurScratch::default();
    // The isotropic ladder's levels, rendered once per view and level.
    let mut ladder: Vec<[Option<TilePlanes>; 6]> = match kernel {
        BlurMatchKernel::IsotropicLadder => (0..k).map(|_| Default::default()).collect(),
        BlurMatchKernel::Anisotropic => Vec::new(),
    };
    let (mut pairs, mut pairs_blurred) = (0usize, 0usize);
    for a in 0..k {
        for b in (a + 1)..k {
            if rows.is_some_and(|r| !(r[a] || r[b])) {
                continue;
            }
            pairs += 1;
            let readings = match (ellipses[a], ellipses[b], matching.min_ratio()) {
                (Some(ea), Some(eb), Some(ratio)) => match kernel {
                    BlurMatchKernel::Anisotropic => {
                        let blur = pair_blur(&ea, &eb, ratio);
                        if blur.is_none() {
                            pair_zncc_readings(tiles[a], tiles[b], &weights)
                        } else {
                            pairs_blurred += 1;
                            blurred[a * k + b] = true;
                            blurred[b * k + a] = true;
                            read_blurred(tiles[a], tiles[b], blur, &weights, &mut scratch)
                        }
                    }
                    BlurMatchKernel::IsotropicLadder => {
                        let (ma, mb) = (semi_major(&ea), semi_major(&eb));
                        let (sharp, other, longer, shorter) = if ma <= mb {
                            (a, b, mb, ma)
                        } else {
                            (b, a, ma, mb)
                        };
                        let level = (longer >= ratio * shorter.max(super::BLUR_MAP_MIN_LENGTH))
                            .then(|| ladder_level(isotropic_blur_sigma(longer, shorter)))
                            .flatten();
                        match level {
                            None => pair_zncc_readings(tiles[a], tiles[b], &weights),
                            Some(level) => {
                                pairs_blurred += 1;
                                blurred[a * k + b] = true;
                                blurred[b * k + a] = true;
                                let slot = &mut ladder[sharp][level];
                                if slot.is_none() {
                                    *slot = Some(tiles[sharp].blurred(
                                        BlurCovariance::isotropic(LADDER_SIGMAS[level]),
                                        &mut scratch,
                                    ));
                                }
                                let level_tile = slot.as_ref().expect("filled just above");
                                pair_zncc_readings(level_tile, tiles[other], &weights)
                            }
                        }
                    }
                },
                _ => pair_zncc_readings(tiles[a], tiles[b], &weights),
            };
            whole[a * k + b] = readings.whole;
            whole[b * k + a] = readings.whole;
            grid[a * k + b] = readings.grid;
            grid[b * k + a] = readings.grid;
        }
    }
    progress_note!(phase, "{pairs_blurred} of {pairs} pairs blurred");
    BlurMatchedPairs {
        k,
        whole,
        grid,
        blurred,
        pairs,
        pairs_blurred,
    }
}

fn read_blurred(
    a: &TilePlanes,
    b: &TilePlanes,
    blur: PairBlur,
    weights: &[f64],
    scratch: &mut BlurScratch,
) -> PairReadings {
    let a_blurred;
    let b_blurred;
    let a_ref = if blur.a.is_zero() {
        a
    } else {
        a_blurred = a.blurred(blur.a, scratch);
        &a_blurred
    };
    let b_ref = if blur.b.is_zero() {
        b
    } else {
        b_blurred = b.blurred(blur.b, scratch);
        &b_blurred
    };
    pair_zncc_readings(a_ref, b_ref, weights)
}
