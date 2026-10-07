// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur-matched ZNCC between whole tiles: each pair correlated over the
//! samples that carry data in both, over the whole tile with a window and over
//! each cell of the ZNCC grid, after the sharper tile is blurred by the width
//! its view's growth gives ([`ViewGrowths`]), and the same over every pair of
//! a track's views.

use super::blur::{blur_tile, BlurScratch};
use super::growth::ViewGrowths;
use super::{pair_blur_for, PairMatching};
use crate::patch::normal_refine::{grid_bounds, window_weights, PatchWindow};
use crate::patch::reference_view::REFERENCE_MIN_CELL_SAMPLES;
use crate::patch::self_similarity::{zncc_self_similarity_radius, PatchTile, SelfSimilarityParams};
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
    /// with `data` flagging the samples that carry data. One or two channels
    /// are grey, and grey and alpha; three or four are RGB, and RGB and alpha.
    /// Alpha is not a colour plane: the caller reads it into `data` where it
    /// says which samples carry data.
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
        let channels = if stride <= 2 { 1 } else { 3 };
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

    /// The same tile blurred isotropically by `sigma` grid px ([`blur_tile`]).
    pub fn blurred(&self, sigma: f64, scratch: &mut BlurScratch) -> Self {
        let mut out = Self::empty();
        self.blur_into(sigma, &mut out, scratch);
        out
    }

    /// A tile of no samples, to blur into.
    fn empty() -> Self {
        Self {
            values: Vec::new(),
            data: Vec::new(),
            side: 0,
            channels: 0,
        }
    }

    /// [`Self::blurred`] into `out`, reusing its buffers.
    fn blur_into(&self, sigma: f64, out: &mut Self, scratch: &mut BlurScratch) {
        out.values.resize(self.values.len(), 0.0);
        out.data.clone_from(&self.data);
        out.side = self.side;
        out.channels = self.channels;
        blur_tile(
            &self.values,
            self.channels,
            self.side,
            &self.data,
            sigma,
            &mut out.values,
            scratch,
        );
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

/// The whole-tile self-similarity ellipse matrix of the planar tile `values`
/// (`channels` colour planes of `side × side`), read the overlap way over the
/// samples `data` marks, with the default parameters; `None` where the
/// reading has no finite axes. This is the reading [`blur_matched_pairs`]
/// reads its blurred tiles with, and the one the bench and the bindings read
/// each view's tile with, so the lengths it compares are of one reading.
pub fn read_tile_ellipse(
    values: &[f32],
    channels: usize,
    side: usize,
    data: &[bool],
) -> Option<[[f64; 2]; 2]> {
    let reading = zncc_self_similarity_radius(
        &PatchTile {
            values,
            channels: channels.min(3),
            width: side,
            height: side,
        },
        Some(data),
        [0, 0, side, side],
        &SelfSimilarityParams::default(),
    );
    let e = reading.ellipse;
    e.axes.iter().all(|a| a.is_finite()).then_some(e.matrix)
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
    /// view blurred in any pair, to read its growth ([`ViewGrowths`]).
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
        crate::patch::reference_view::finite_middle(&others)
    }
}

/// Read every pair of `tiles` (or every pair with a view in `rows`, when
/// given) with [`pair_zncc_readings`] after blur-matching it under
/// `matching`, `ellipses[v]` view `v`'s self-similarity ellipse matrix in
/// grid px² (`None` where it has none, which leaves every pair it is in
/// plain).
///
/// A pair is blurred where [`pair_blur_for`] finds a
/// sharper tile: that tile is blurred isotropically by the width at which its
/// view's growth brings its semi-major axis to the target
/// ([`BlurGrowth::sigma_for`](super::BlurGrowth::sigma_for)), and the other
/// tile is read as it is. Where the view's growth cannot be read, or does
/// not grow, the pair is read plain.
///
/// The ellipses must be the whole-tile reading over the samples with data
/// ([`read_tile_ellipse`]): each view's growth is read that way on its tile
/// blurred by each probe ([`ViewGrowths`]), and the width is read off those
/// readings and the view's own. A view's growth is read the first time a
/// pair blurs it, so a pair reads the same whatever other pairs are read with
/// it.
///
/// The blur work is timed in the `blur-matched pairs` detail phase of
/// `progress`, with a note of how many pairs were blurred and how many blurred
/// tiles were read.
///
/// # Panics
///
/// Panics if `ellipses` or `rows` is not parallel to `tiles`, or the tiles
/// differ in side.
pub fn blur_matched_pairs(
    tiles: &[&TilePlanes],
    ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching,
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
    let mut widths = vec![0.0; k * k];
    for v in 0..k {
        whole[v * k + v] = 1.0;
    }
    let side = tiles.first().map_or(0, |t| t.side);
    let weights = window_weights(window, side as u32);
    let mut growths = ViewGrowths::new(k);
    let mut blur_scratch = BlurScratch::default();
    // The blurred tile, reused pair after pair.
    let mut sharp_blurred = TilePlanes::empty();
    let (mut pairs, mut pairs_blurred) = (0usize, 0usize);
    for a in 0..k {
        for b in (a + 1)..k {
            if rows.is_some_and(|r| !(r[a] || r[b])) {
                continue;
            }
            pairs += 1;
            let blur = match (ellipses[a], ellipses[b]) {
                (Some(ea), Some(eb)) => pair_blur_for(&ea, &eb, matching).and_then(|pb| {
                    let (sharp, other) = if pb.sharper == 0 { (a, b) } else { (b, a) };
                    let e = if pb.sharper == 0 { ea } else { eb };
                    let t = tiles[sharp];
                    let growth = growths.get(sharp, t, &e, |values| {
                        read_tile_ellipse(values, t.channels, t.side, &t.data)
                    })?;
                    let sigma = growth.sigma_for(pb.target)?;
                    (sigma > 0.0).then_some((sharp, other, sigma))
                }),
                _ => None,
            };
            let readings = match blur {
                None => pair_zncc_readings(tiles[a], tiles[b], &weights),
                Some((sharp, other, sigma)) => {
                    pairs_blurred += 1;
                    blurred[a * k + b] = true;
                    blurred[b * k + a] = true;
                    widths[sharp * k + other] = sigma;
                    tiles[sharp].blur_into(sigma, &mut sharp_blurred, &mut blur_scratch);
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
    let reads = growths.reads();
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
