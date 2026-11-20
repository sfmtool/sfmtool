// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The rectangle of each pyramid level a read takes its taps from:
//! [`required_regions`] for a render through a warp map, and
//! [`LevelRegions::for_samples`] for a read at given positions.
//!
//! Each rule walks the same positions and levels as the sampler it is for,
//! with the same [`bilinear_cell`], and records the cell's taps clamped to
//! the level, so a region pyramid of these rectangles gives a render the
//! whole pyramid's values (`specs/core/camera/view-region-pyramids.md`).

use crate::camera::mipmap::PyramidLayout;
use crate::camera::remap::{aniso_footprint, aniso_tap, bilinear_cell, mip_level_for_sigma};
use crate::camera::warp_map::{parallelize_rows, REMAP_PAR_MIN_PIXELS};
use crate::camera::WarpMap;
use rayon::prelude::*;

use super::{Sampler, MAX_ANISOTROPY};

/// How far a held region extends past the region a render requires, as a
/// share of the required region's width (left and right) and height (above
/// and below), each rounded up to whole pixels, before clipping to the level.
///
/// It changes how much a render reserves from its kernel call's region pool,
/// and so which renders take the tile path, never a value.
pub const REGION_REUSE_MARGIN: f64 = 0.25;

/// A rectangle of one level's pixels: columns `x0..x1` and rows `y0..y1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl LevelRect {
    /// The number of columns.
    pub fn width(&self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// The number of rows.
    pub fn height(&self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }

    /// Whether `other` lies inside this rectangle.
    pub fn contains(&self, other: &LevelRect) -> bool {
        self.x0 <= other.x0 && self.y0 <= other.y0 && other.x1 <= self.x1 && other.y1 <= self.y1
    }

    /// The smallest rectangle holding both.
    fn union(self, other: LevelRect) -> LevelRect {
        LevelRect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

/// The rectangle of each level of a pyramid that a read takes its taps from;
/// `None` for a level it does not read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelRegions {
    regions: Vec<Option<LevelRect>>,
    /// Each level's size, which the taps are clamped to.
    sizes: Vec<(u32, u32)>,
}

impl LevelRegions {
    /// No region of any level of `layout`.
    pub fn new(layout: &PyramidLayout) -> Self {
        Self {
            regions: vec![None; layout.levels()],
            sizes: (0..layout.levels()).map(|l| layout.level_size(l)).collect(),
        }
    }

    /// The region of level `level`; `None` where the read takes nothing from
    /// it, or past the last level.
    pub fn region(&self, level: usize) -> Option<LevelRect> {
        self.regions.get(level).copied().flatten()
    }

    /// The number of levels.
    pub fn levels(&self) -> usize {
        self.regions.len()
    }

    /// Whether the read takes no pixel.
    pub fn is_empty(&self) -> bool {
        self.regions.iter().all(Option::is_none)
    }

    /// The bytes the regions hold at `channels` bytes a pixel.
    pub fn byte_len(&self, channels: u32) -> u64 {
        self.regions
            .iter()
            .flatten()
            .map(|r| u64::from(r.width()) * u64::from(r.height()) * u64::from(channels))
            .sum()
    }

    /// The regions of a read at `positions` of level `level` (level pixels,
    /// pixel centres at half-integers): for each position the bounding box
    /// of its bilinear taps `floor(x − 0.5)` and `floor(x − 0.5) + 1` on each
    /// axis, widened by `border` pixels on every side (0 for a bilinear read,
    /// 1 for a read of the 4 x 4 pixels about it), each address clamped to
    /// the level. No region for a level `layout` does not have.
    pub fn for_samples(
        layout: &PyramidLayout,
        level: usize,
        positions: &[[f32; 2]],
        border: u32,
    ) -> Self {
        let mut regions = Self::new(layout);
        if level < regions.levels() {
            let border = border.min(i32::MAX as u32) as i32;
            for &[x, y] in positions {
                let (x0, y0, _, _) = bilinear_cell(x, y);
                regions.add_span(
                    level,
                    [x0.saturating_sub(border), x0.saturating_add(1 + border)],
                    [y0.saturating_sub(border), y0.saturating_add(1 + border)],
                );
            }
        }
        regions
    }

    /// The regions grown by `margin`: a `w` x `h` region by `ceil(margin · w)`
    /// pixels on the left and on the right and `ceil(margin · h)` pixels
    /// above and below, clipped to its level.
    pub fn grown(&self, margin: f64) -> Self {
        let mut grown = self.clone();
        for (region, &(w, h)) in grown.regions.iter_mut().zip(&self.sizes) {
            if let Some(r) = region {
                let gx = (margin * f64::from(r.width())).ceil() as u32;
                let gy = (margin * f64::from(r.height())).ceil() as u32;
                *r = LevelRect {
                    x0: r.x0.saturating_sub(gx),
                    y0: r.y0.saturating_sub(gy),
                    x1: r.x1.saturating_add(gx).min(w),
                    y1: r.y1.saturating_add(gy).min(h),
                };
            }
        }
        grown
    }

    /// The regions of `self` and `other` together: each level's bounding box
    /// of both.
    fn union(mut self, other: Self) -> Self {
        for (slot, theirs) in self.regions.iter_mut().zip(other.regions) {
            *slot = match (*slot, theirs) {
                (Some(a), Some(b)) => Some(a.union(b)),
                (a, b) => a.or(b),
            };
        }
        self
    }

    /// Replace the region of `level`, for the tests that shrink one.
    #[cfg(test)]
    pub(crate) fn set_region(&mut self, level: usize, rect: LevelRect) {
        self.regions[level] = Some(rect);
    }

    /// Add the taps of the bilinear cell whose top-left tap is `(x0, y0)` at
    /// `level`, each clamped to the level as the samplers clamp it.
    #[inline]
    pub(crate) fn add_cell(&mut self, level: usize, x0: i32, y0: i32) {
        self.add_span(
            level,
            [x0, x0.saturating_add(1)],
            [y0, y0.saturating_add(1)],
        );
    }

    /// Add the pixels from `xs[0]` to `xs[1]` and `ys[0]` to `ys[1]`
    /// inclusive at `level`, each end clamped to the level.
    fn add_span(&mut self, level: usize, xs: [i32; 2], ys: [i32; 2]) {
        let (w, h) = self.sizes[level];
        if w == 0 || h == 0 {
            return;
        }
        let cx = |v: i32| v.clamp(0, w as i32 - 1) as u32;
        let cy = |v: i32| v.clamp(0, h as i32 - 1) as u32;
        let rect = LevelRect {
            x0: cx(xs[0]),
            y0: cy(ys[0]),
            x1: cx(xs[1]) + 1,
            y1: cy(ys[1]) + 1,
        };
        let slot = &mut self.regions[level];
        *slot = Some(match *slot {
            Some(held) => held.union(rect),
            None => rect,
        });
    }

    /// Add the taps of a bilinear sample at `(x, y)` of `level`.
    #[inline]
    pub(crate) fn add_sample(&mut self, level: usize, x: f32, y: f32) {
        let (x0, y0, _, _) = bilinear_cell(x, y);
        self.add_cell(level, x0, y0);
    }

    /// Add the taps of the anisotropic sample at the level-0 position
    /// `(sx, sy)` with the SVD `svd`, as `aniso_sample` and its gradient form
    /// take them.
    pub(crate) fn add_aniso(&mut self, sx: f32, sy: f32, svd: (f32, f32, f32, f32)) {
        let (sigma_major, sigma_minor, _, _) = svd;
        if sigma_major <= 1.0 {
            self.add_sample(0, sx, sy);
            return;
        }
        let fp = aniso_footprint(sigma_major, sigma_minor, self.levels(), MAX_ANISOTROPY);
        let scale_lo = (1u32 << fp.level_lo) as f32;
        let scale_hi = (1u32 << fp.level_hi) as f32;
        for i in 0..fp.n {
            let (x, y) = aniso_tap(sx, sy, svd, i, fp.n);
            self.add_sample(fp.level_lo, x / scale_lo, y / scale_lo);
            if fp.frac > 0.0 {
                self.add_sample(fp.level_hi, x / scale_hi, y / scale_hi);
            }
        }
    }
}

/// The region of each level a render of `map` with `sampler` reads from a
/// photograph of `layout`, for its value or its value and gradients: the
/// bounding box of every tap the render takes there, clamped to the level.
///
/// It walks the map once, by the render's own rules: `Bilinear` takes level 0
/// at each valid texel's `(sx, sy)`, `BilinearMip` the level
/// `mip_level_for_sigma(σ_major)` at `(sx, sy) / 2^l`, and `Anisotropic` the
/// samples of `aniso_footprint` along the major axis at its one or two
/// levels. Computes the map's SVD first where the sampler needs it, as
/// [`render_tile`](super::render_tile) does.
pub fn required_regions(
    map: &mut WarpMap,
    sampler: Sampler,
    layout: &PyramidLayout,
) -> LevelRegions {
    if sampler.needs_svd() && !map.has_svd() {
        map.compute_svd();
    }
    let empty = LevelRegions::new(layout);
    if empty.levels() == 0 {
        return empty;
    }
    let num_levels = empty.levels();
    let map = &*map;
    let add_row = |regions: &mut LevelRegions, row: u32| {
        for col in 0..map.width() {
            let (sx, sy) = map.get(col, row);
            if sx.is_nan() || sy.is_nan() {
                continue;
            }
            match sampler {
                Sampler::Bilinear => regions.add_sample(0, sx, sy),
                Sampler::BilinearMip => {
                    let (sigma_major, _, _, _) = map.get_svd(col, row);
                    let level = mip_level_for_sigma(sigma_major, num_levels);
                    let scale = (1u32 << level) as f32;
                    regions.add_sample(level, sx / scale, sy / scale);
                }
                Sampler::Anisotropic => regions.add_aniso(sx, sy, map.get_svd(col, row)),
            }
        }
    };
    // The regions are bounding boxes, so taking the rows in any grouping and
    // order gives the same rectangles; the rows of a large map are walked on
    // the rayon pool, as its render's are.
    let pixels = map.width() as usize * map.height() as usize;
    if parallelize_rows(pixels, REMAP_PAR_MIN_PIXELS) {
        (0..map.height())
            .into_par_iter()
            .fold(
                || empty.clone(),
                |mut regions, row| {
                    add_row(&mut regions, row);
                    regions
                },
            )
            .reduce(|| empty.clone(), LevelRegions::union)
    } else {
        let mut regions = empty;
        for row in 0..map.height() {
            add_row(&mut regions, row);
        }
        regions
    }
}

#[cfg(test)]
mod tests;
