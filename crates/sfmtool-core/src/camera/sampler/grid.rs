// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The samplers of affine maps over a patch grid: the view selection's affine
//! fast path ([`sample_support_affine`], [`sample_support_affine_aniso`]) and
//! the cluster refinement's grids ([`sample_grid`]), each with the rule for
//! the region it reads.
//!
//! Each reads its level through a [`LevelTexels`], so the same arithmetic runs
//! on a whole level, a copied region and photograph tiles. A map is
//! `x = a[0]·col + a[1]·row + a[2]`, `y = a[3]·col + a[4]·row + a[5]` from
//! grid indices to continuous level pixels.

use crate::camera::remap::{aniso_sample, prof, AnisoTally, LevelTexels, PyramidTexels};
use crate::camera::warp_map::svd_2x2;

use super::{LevelRegions, MAX_ANISOTROPY};

/// What a cluster grid sampler does with a sample whose taps leave the level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GridEdge {
    /// Clamp to the nearest valid pixel (border replicate), and refuse the
    /// whole grid on a non-finite coordinate.
    Clamp,
    /// Write `NaN` in every channel of that sample, non-finite coordinates
    /// included.
    Missing,
    /// Refuse the whole grid when any sample's taps leave the level or its
    /// coordinate is not finite.
    InFrame,
}

/// The grid points a cluster grid sampler reads.
#[derive(Clone, Copy, Debug)]
pub enum GridPoints<'a> {
    /// The `size` x `size` window of grid indices from `first` on both axes.
    Window { first: i64, size: usize },
    /// The grid pixels `pixels` (row-major indices into a grid of
    /// `resolution` columns).
    Pixels {
        pixels: &'a [usize],
        resolution: usize,
    },
}

impl GridPoints<'_> {
    /// The number of points.
    pub fn len(&self) -> usize {
        match *self {
            Self::Window { size, .. } => size * size,
            Self::Pixels { pixels, .. } => pixels.len(),
        }
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Point `k`'s grid column and row.
    #[inline]
    fn at(&self, k: usize) -> (f64, f64) {
        match *self {
            Self::Window { first, size } => (
                (first + (k % size) as i64) as f64,
                (first + (k / size) as i64) as f64,
            ),
            Self::Pixels { pixels, resolution } => {
                let p = pixels[k];
                ((p % resolution) as f64, (p / resolution) as f64)
            }
        }
    }
}

/// One grid point's bilinear cell: its top-left tap and fractions, or no
/// sample.
#[derive(Clone, Copy)]
pub(crate) enum GridCell {
    Cell { ix: i64, iy: i64, fx: f32, fy: f32 },
    Missing,
}

/// The cells a cluster grid sampler reads for `points` through the level map
/// `a` on a level of `size` pixels, by `edge`'s rule; `None` where the grid
/// is refused: a level too small to bilinear-sample, or a non-finite
/// coordinate under [`GridEdge::Clamp`], or any sample out of frame under
/// [`GridEdge::InFrame`].
pub(crate) fn grid_cells(
    a: &[f64; 6],
    points: GridPoints<'_>,
    edge: GridEdge,
    size: (u32, u32),
) -> Option<Vec<GridCell>> {
    let (w_img, h_img) = (i64::from(size.0), i64::from(size.1));
    if w_img < 2 || h_img < 2 {
        return None;
    }
    let (max_gx, max_gy) = ((w_img - 1) as f64, (h_img - 1) as f64);
    let mut cells = Vec::with_capacity(points.len());
    for k in 0..points.len() {
        let (col, row) = points.at(k);
        let x = a[0] * col + a[1] * row + a[2];
        let y = a[3] * col + a[4] * row + a[5];
        // `bilinear_cell`'s pixel-center convention.
        let gx = x - 0.5;
        let gy = y - 0.5;
        let cell = match edge {
            GridEdge::Clamp => {
                if !gx.is_finite() || !gy.is_finite() {
                    return None;
                }
                // Nearest-valid-pixel clamp (border replicate): the coordinate
                // clamps to the outermost pixel center and the tap base to the
                // last valid 2x2 cell, so fx/fy saturate and the blend reads
                // the edge pixel.
                let gx = gx.clamp(0.0, max_gx);
                let gy = gy.clamp(0.0, max_gy);
                let ix = (gx.floor() as i64).min(w_img - 2);
                let iy = (gy.floor() as i64).min(h_img - 2);
                GridCell::Cell {
                    ix,
                    iy,
                    fx: (gx - ix as f64) as f32,
                    fy: (gy - iy as f64) as f32,
                }
            }
            GridEdge::Missing | GridEdge::InFrame => {
                let in_frame = gx.is_finite()
                    && gy.is_finite()
                    && gx >= 0.0
                    && gy >= 0.0
                    && (gx.floor() as i64) + 1 < w_img
                    && (gy.floor() as i64) + 1 < h_img;
                if !in_frame {
                    if edge == GridEdge::InFrame {
                        return None;
                    }
                    cells.push(GridCell::Missing);
                    continue;
                }
                let (x0, y0) = (gx.floor(), gy.floor());
                GridCell::Cell {
                    ix: x0 as i64,
                    iy: y0 as i64,
                    fx: (gx - x0) as f32,
                    fy: (gy - y0) as f32,
                }
            }
        };
        cells.push(cell);
    }
    Some(cells)
}

/// The regions the cells read at `level`.
pub(crate) fn grid_regions(regions: &mut LevelRegions, level: usize, cells: &[GridCell]) {
    for cell in cells {
        if let GridCell::Cell { ix, iy, .. } = *cell {
            // In-frame cells, whose taps are inside the level.
            regions.add_cell(level, ix as i32, iy as i32);
        }
    }
}

/// The interleaved `n × C` `f32` samples of `level` at the cells: each
/// cell's bilinear blend, `NaN` in every channel of a missing one.
pub(crate) fn sample_grid<T: LevelTexels>(level: &T, cells: &[GridCell]) -> Vec<f32> {
    let ch = level.channels();
    let mut out = vec![0f32; cells.len() * ch];
    for (px, cell) in cells.iter().enumerate() {
        let GridCell::Cell { ix, iy, fx, fy } = *cell else {
            out[px * ch..][..ch].fill(f32::NAN);
            continue;
        };
        level.with_cell(ix as i32, iy as i32, |data, idx| {
            for c in 0..ch {
                let v00 = data[idx[0] + c] as f32;
                let v10 = data[idx[1] + c] as f32;
                let v01 = data[idx[2] + c] as f32;
                let v11 = data[idx[3] + c] as f32;
                out[px * ch + c] = (1.0 - fx) * (1.0 - fy) * v00
                    + fx * (1.0 - fy) * v10
                    + (1.0 - fx) * fy * v01
                    + fx * fy * v11;
            }
        });
    }
    out
}

/// The cell of the affine fast path's bilinear sample of grid pixel `p`: the
/// map's position minus 0.5 in `f64`, cast to `f32`, floored.
#[inline]
fn affine_cell(a: &[f64; 6], p: usize, resolution: usize) -> (f32, f32, f32, f32) {
    let col = (p % resolution) as f64;
    let row = (p / resolution) as f64;
    // Same pixel-center convention as `bilinear_cell` (x − 0.5).
    let gx = (a[0] * col + a[1] * row + a[2] - 0.5) as f32;
    let gy = (a[3] * col + a[4] * row + a[5] - 0.5) as f32;
    let x0 = gx.floor();
    let y0 = gy.floor();
    (x0, y0, gx - x0, gy - y0)
}

/// Sample the reference-support pixels of `level` through the affine map
/// into the planar `out` (`[channel · n + support_index]`, `f32`), with
/// exactly [`remap_bilinear`](crate::camera::remap::remap_bilinear)'s
/// per-sample value convention (same `x − 0.5` pixel-center geometry, same
/// bilinear blend grouping, same `u8` rounding), so where the affine position
/// matches the exact warp the sampled value is bit-identical.
///
/// `level` is whichever pyramid level the view selection's `affine_core_map`
/// selected, and `a` is already expressed in that level's pixels. Its border
/// gate keeps every sample at least 1 px inside the frame, so no tap is
/// clamped.
pub(crate) fn sample_support_affine<T: LevelTexels>(
    level: &T,
    a: &[f64; 6],
    pixels: &[usize],
    resolution: usize,
    out: &mut [f32],
) {
    let n = pixels.len();
    let ch = level.channels();
    for (k, &p) in pixels.iter().enumerate() {
        let (x0, y0, fx, fy) = affine_cell(a, p, resolution);
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;
        level.with_cell(x0 as i32, y0 as i32, |data, idx| {
            for c in 0..ch {
                let val = w00 * data[idx[0] + c] as f32
                    + w10 * data[idx[1] + c] as f32
                    + w01 * data[idx[2] + c] as f32
                    + w11 * data[idx[3] + c] as f32;
                // remap_bilinear's u8 rounding, kept for value parity.
                out[c * n + k] = (val + 0.5).clamp(0.0, 255.0).floor();
            }
        });
    }
    // Keep the shared resample counters honest (one call, n samples, one
    // bilinear tap per sample per channel — same accounting as
    // `remap_bilinear`, which the exact path routes through).
    prof::add(&prof::CALLS, 1);
    prof::add(&prof::PX_TOTAL, n as u64);
    prof::add(&prof::PX_SAMPLED, n as u64);
    prof::add(&prof::TAPS, (n * ch) as u64);
}

/// The SVD of an affine map's constant Jacobian, as
/// [`WarpMap::compute_svd`](crate::camera::WarpMap::compute_svd) computes
/// it per pixel.
fn affine_svd(a: &[f64; 6]) -> (f32, f32, f32, f32) {
    svd_2x2(a[0] as f32, a[1] as f32, a[3] as f32, a[4] as f32)
}

/// The level-0 position of grid pixel `p`, rounded to `f32` as a warp map
/// stores it.
#[inline]
fn affine_position(a: &[f64; 6], p: usize, resolution: usize) -> (f32, f32) {
    let col = (p % resolution) as f64;
    let row = (p / resolution) as f64;
    (
        (a[0] * col + a[1] * row + a[2]) as f32,
        (a[3] * col + a[4] * row + a[5]) as f32,
    )
}

/// Sample the reference-support pixels of `pyramid` through the affine map
/// with the anisotropic sampler, into the planar `out` (`[channel · n +
/// support_index]`, `f32`).
///
/// `a` is in level-0 px. Its Jacobian `[[a0, a1], [a3, a4]]` is the same at
/// every pixel, so one SVD gives the footprint of every sample, and each
/// support pixel takes [`aniso_sample`] at its affine position, rounded to
/// `f32` as a warp map stores it: the per-pixel path's sample for that
/// position and SVD.
pub(crate) fn sample_support_affine_aniso<P: PyramidTexels>(
    pyramid: &P,
    a: &[f64; 6],
    pixels: &[usize],
    resolution: usize,
    out: &mut [f32],
) {
    let n = pixels.len();
    let ch = pyramid.channels();
    let svd = affine_svd(a);
    let mut tally = AnisoTally::default();
    let mut sample = vec![0u8; ch];
    for (k, &p) in pixels.iter().enumerate() {
        let (x, y) = affine_position(a, p, resolution);
        aniso_sample(pyramid, x, y, svd, MAX_ANISOTROPY, &mut sample, &mut tally);
        for (c, &v) in sample.iter().enumerate() {
            out[c * n + k] = v as f32;
        }
    }
    // The shared resample counters, as `remap_aniso_with_pyramid` keeps them.
    prof::add(&prof::CALLS, 1);
    prof::add(&prof::PX_TOTAL, n as u64);
    prof::add(&prof::PX_SAMPLED, n as u64);
    prof::add(&prof::TAPS, tally.taps);
    prof::add(&prof::ANISO_FAST, tally.fast);
    prof::add(&prof::ANISO_MULTI, tally.multi);
    prof::add(&prof::ANISO_SUM_N, tally.sum_n);
}

/// The regions [`sample_support_affine`] at `level`, or under `aniso`
/// [`sample_support_affine_aniso`], reads.
pub(crate) fn affine_regions(
    regions: &mut LevelRegions,
    a: &[f64; 6],
    level: usize,
    aniso: bool,
    pixels: &[usize],
    resolution: usize,
) {
    if aniso {
        let svd = affine_svd(a);
        for &p in pixels {
            let (x, y) = affine_position(a, p, resolution);
            regions.add_aniso(x, y, svd);
        }
    } else {
        for &p in pixels {
            let (x0, y0, _, _) = affine_cell(a, p, resolution);
            regions.add_cell(level, x0 as i32, y0 as i32);
        }
    }
}
