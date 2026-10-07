// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur-matched ZNCC between whole tiles: each pair correlated over the
//! samples that carry data in both, over the whole tile with a window and over
//! each cell of the ZNCC grid, after the blur [`match_blur`] finds, and the
//! same over every pair of a track's views.

use super::blur::{blur_tile, BlurScratch};
use super::search::{match_blur, MatchScratch};
use super::{
    pair_directions, BlurCovariance, PairMatching, TileDirections, BLUR_MATCH_TOLERANCE,
    LADDER_SIGMAS, MAX_MATCHED_LENGTH, MIN_SHARPER_LENGTH,
};
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

    /// The same tile blurred by `cov` ([`blur_tile`]).
    pub fn blurred(&self, cov: BlurCovariance, scratch: &mut BlurScratch) -> Self {
        let mut out = Self::empty();
        self.blur_into(cov, &mut out, scratch);
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
    fn blur_into(&self, cov: BlurCovariance, out: &mut Self, scratch: &mut BlurScratch) {
        out.values.resize(self.values.len(), 0.0);
        out.data.clone_from(&self.data);
        out.side = self.side;
        out.channels = self.channels;
        blur_tile(
            &self.values,
            self.channels,
            self.side,
            &self.data,
            cov,
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

/// The shape of the blur a pair is matched with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlurMatchKernel {
    /// Each tile blurred along the directions in which its ellipse is the
    /// shorter ([`pair_directions`]), the widths found by measuring the
    /// blurred tile ([`match_blur`]). Every pair blurs its own tiles.
    #[default]
    Anisotropic,
    /// The sharper tile, by semi-major axis, blurred isotropically by one of
    /// [`LADDER_SIGMAS`]: the level whose blurred tile's semi-major axis comes
    /// closest to the other tile's. Each view is blurred, and its ellipse read,
    /// at most once per level however many pairs it is in.
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
    /// How many blurred tiles had their ellipse read, over every pair: the
    /// anisotropic kernel's probes ([`match_blur`]) or the ladder's levels.
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

/// The semi-major axis of the ellipse `E`: the square root of its larger
/// eigenvalue.
fn semi_major(e: &[[f64; 2]; 2]) -> f64 {
    let (a, b, d) = (e[0][0], 0.5 * (e[0][1] + e[1][0]), e[1][1]);
    (0.5 * (a + d + ((a - d) * (a - d) + 4.0 * b * b).sqrt()))
        .max(0.0)
        .sqrt()
}

/// One view's ladder: each level's blurred tile and the semi-major axis of its
/// ellipse, filled as pairs ask for them.
type Ladder = [Option<(TilePlanes, f64)>; LADDER_SIGMAS.len()];

/// The ladder level of `tile` whose semi-major axis comes closest to `target`
/// in the logarithm, `None` where the tile as it is comes closer, filling the
/// levels from the narrowest until one reaches `target` within
/// [`BLUR_MATCH_TOLERANCE`]. `unblurred` is the tile's own semi-major axis.
fn ladder_pick(
    tile: &TilePlanes,
    unblurred: f64,
    target: f64,
    ladder: &mut Ladder,
    scratch: &mut BlurScratch,
    reads: &mut usize,
) -> Option<usize> {
    let mut pick: Option<usize> = None;
    let mut off = (unblurred.max(MIN_SHARPER_LENGTH) / target).ln().abs();
    for (level, slot) in ladder.iter_mut().enumerate() {
        let (_, length) = slot.get_or_insert_with(|| {
            let blurred = tile.blurred(BlurCovariance::isotropic(LADDER_SIGMAS[level]), scratch);
            *reads += 1;
            let length = read_tile_ellipse(
                &blurred.values,
                blurred.channels,
                blurred.side,
                &blurred.data,
            )
            .map_or(f64::NAN, |e| semi_major(&e));
            (blurred, length)
        });
        if !length.is_finite() {
            break;
        }
        let o = (length.max(MIN_SHARPER_LENGTH) / target).ln().abs();
        if o < off {
            off = o;
            pick = Some(level);
        }
        if *length >= target * (1.0 - BLUR_MATCH_TOLERANCE) {
            break;
        }
    }
    pick
}

/// Read every pair of `tiles` (or every pair with a view in `rows`, when
/// given) with [`pair_zncc_readings`] after blur-matching it under `matching`
/// with `kernel`, `ellipses[v]` view `v`'s self-similarity ellipse matrix in
/// grid px² (`None` where it has none, which leaves every pair it is in plain).
///
/// The ellipses must be the whole-tile reading over the samples with data
/// ([`read_tile_ellipse`]): the blurred tiles are read that way, and the blur
/// is set by comparing the two.
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
    let mut scratch = MatchScratch::default();
    let mut blur_scratch = BlurScratch::default();
    // The anisotropic kernel's two blurred tiles, reused pair after pair.
    let mut blurred_pair = [TilePlanes::empty(), TilePlanes::empty()];
    // The isotropic ladder's levels, rendered and read once per view and level.
    let mut ladder: Vec<Ladder> = match kernel {
        BlurMatchKernel::IsotropicLadder => (0..k).map(|_| Default::default()).collect(),
        BlurMatchKernel::Anisotropic => Vec::new(),
    };
    let (mut pairs, mut pairs_blurred, mut reads) = (0usize, 0usize, 0usize);
    for a in 0..k {
        for b in (a + 1)..k {
            if rows.is_some_and(|r| !(r[a] || r[b])) {
                continue;
            }
            pairs += 1;
            let readings = match (ellipses[a], ellipses[b], matching.min_ratio()) {
                (Some(ea), Some(eb), Some(ratio)) => match kernel {
                    BlurMatchKernel::Anisotropic => {
                        let dirs = pair_directions(&ea, &eb, ratio);
                        if dirs.is_none() {
                            pair_zncc_readings(tiles[a], tiles[b], &weights)
                        } else {
                            pairs_blurred += 1;
                            blurred[a * k + b] = true;
                            blurred[b * k + a] = true;
                            read_matched(
                                [tiles[a], tiles[b]],
                                [dirs.a, dirs.b],
                                &weights,
                                &mut blurred_pair,
                                &mut scratch,
                                &mut reads,
                            )
                        }
                    }
                    BlurMatchKernel::IsotropicLadder => {
                        let (ma, mb) = (semi_major(&ea), semi_major(&eb));
                        let (sharp, other, longer, shorter) = if ma <= mb {
                            (a, b, mb, ma)
                        } else {
                            (b, a, ma, mb)
                        };
                        let longer = longer.min(MAX_MATCHED_LENGTH);
                        let level = (longer > shorter
                            && longer >= ratio * shorter.max(MIN_SHARPER_LENGTH))
                        .then(|| {
                            ladder_pick(
                                tiles[sharp],
                                shorter,
                                longer,
                                &mut ladder[sharp],
                                &mut blur_scratch,
                                &mut reads,
                            )
                        })
                        .flatten();
                        match level {
                            None => pair_zncc_readings(tiles[a], tiles[b], &weights),
                            Some(level) => {
                                pairs_blurred += 1;
                                blurred[a * k + b] = true;
                                blurred[b * k + a] = true;
                                let (level_tile, _) =
                                    ladder[sharp][level].as_ref().expect("filled by the pick");
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
    progress_note!(
        phase,
        "{pairs_blurred} of {pairs} pairs blurred, {reads} blurred tiles read"
    );
    BlurMatchedPairs {
        k,
        whole,
        grid,
        blurred,
        pairs,
        pairs_blurred,
        ellipse_reads: reads,
    }
}

/// The readings of the pair `tiles` after each is blurred along its
/// directions `dirs` to the other's sharpness ([`match_blur`]), a tile with
/// none read as it is, the blurred tiles written into `out`, the ellipse
/// readings counted into `reads`.
fn read_matched(
    tiles: [&TilePlanes; 2],
    dirs: [TileDirections; 2],
    weights: &[f64],
    out: &mut [TilePlanes; 2],
    scratch: &mut MatchScratch,
    reads: &mut usize,
) -> PairReadings {
    for ((tile, d), out) in tiles.iter().zip(&dirs).zip(out.iter_mut()) {
        if d.is_empty() {
            continue;
        }
        out.data.clone_from(&tile.data);
        out.side = tile.side;
        out.channels = tile.channels;
        let matched = match_blur(
            tile,
            d.as_slice(),
            |values| read_tile_ellipse(values, tile.channels, tile.side, &tile.data),
            &mut out.values,
            scratch,
        );
        *reads += matched.probes as usize;
    }
    let a = if dirs[0].is_empty() {
        tiles[0]
    } else {
        &out[0]
    };
    let b = if dirs[1].is_empty() {
        tiles[1]
    } else {
        &out[1]
    };
    pair_zncc_readings(a, b, weights)
}
