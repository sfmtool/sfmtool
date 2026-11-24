// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where the samplers' taps come from.
//!
//! Every sampler of [`super`] finds its taps with [`bilinear_cell`] at
//! full-level coordinates and reads them through a [`LevelTexels`]: a level
//! held in memory, whole or as a rectangle of it ([`HeldLevel`]), or the
//! photograph tiles of the tile-addressed path
//! (`camera::sampler::source`). Both clamp each tap to the full level and
//! read the same pixel, so a sampler gives the same bits through either.

use crate::camera::image::{ImageU8, ImageU8Pyramid};

/// The cell a bilinear sample at `(x, y)` (level pixels, pixel centres at
/// half-integers) reads: the top-left tap `(floor(x − 0.5), floor(y − 0.5))`,
/// saturated to `i32`, and the in-cell fractions. The single source of the
/// `x − 0.5` half-pixel convention for every sampler and region rule.
#[inline]
pub(crate) fn bilinear_cell(x: f32, y: f32) -> (i32, i32, f32, f32) {
    let gx = x - 0.5;
    let gy = y - 0.5;
    let x0 = gx.floor() as i32;
    let y0 = gy.floor() as i32;
    (x0, y0, gx - x0 as f32, gy - y0 as f32)
}

/// One level of a pyramid as the samplers read it.
pub(crate) trait LevelTexels {
    /// The channels of each pixel.
    fn channels(&self) -> usize;

    /// Calls `f` with the pixel bytes and the base indices of the taps
    /// `(x0, y0)`, `(x0 + 1, y0)`, `(x0, y0 + 1)` and `(x0 + 1, y0 + 1)`, in
    /// full-level pixels, each clamped to the level as the whole level
    /// clamps it.
    fn with_cell<R>(&self, x0: i32, y0: i32, f: impl FnOnce(&[u8], [usize; 4]) -> R) -> R;

    /// Calls `f` with the channels of the pixel at `(x, y)` in full-level
    /// pixels, clamped to the level.
    fn with_pixel<R>(&self, x: i32, y: i32, f: impl FnOnce(&[u8]) -> R) -> R;
}

/// A pyramid as the samplers read it.
pub(crate) trait PyramidTexels {
    /// One level of it.
    type Level<'a>: LevelTexels
    where
        Self: 'a;

    /// Level `level`.
    fn texels(&self, level: usize) -> Self::Level<'_>;

    /// The number of levels.
    fn num_levels(&self) -> usize;

    /// The channels of each pixel.
    fn channels(&self) -> usize;
}

/// A level held in memory: the whole level, or a rectangle of it whose
/// top-left pixel is `origin` in the full level.
///
/// A tap is clamped to the held rectangle after the origin is subtracted.
/// Where the rectangle is clipped to the full level and holds every tap the
/// whole level's clamp gives, that is the same pixel the whole level reads
/// (`specs/core/camera/view-region-pyramids.md`); the subtraction is in
/// integers, after `floor`, so the fractions are the whole level's.
#[derive(Clone, Copy)]
pub(crate) struct HeldLevel<'a> {
    image: &'a ImageU8,
    origin: [i32; 2],
}

impl<'a> HeldLevel<'a> {
    /// A whole level, or an image read as one.
    #[inline]
    pub(crate) fn whole(image: &'a ImageU8) -> Self {
        Self {
            image,
            origin: [0, 0],
        }
    }

    /// A rectangle of a level whose top-left pixel is `origin` in the full
    /// level.
    #[inline]
    pub(crate) fn at(image: &'a ImageU8, origin: [i32; 2]) -> Self {
        Self { image, origin }
    }

    /// Level `level` of `pyramid`, at its origin.
    #[inline]
    pub(crate) fn of(pyramid: &'a ImageU8Pyramid, level: usize) -> Self {
        let [ox, oy] = pyramid.level_origin(level);
        Self {
            image: pyramid.level(level),
            // A level's side is under 2^16, so its origin fits.
            origin: [ox as i32, oy as i32],
        }
    }

    /// The held pixels.
    #[inline]
    pub(crate) fn image(&self) -> &'a ImageU8 {
        self.image
    }

    /// The held rectangle's top-left pixel in the full level.
    #[inline]
    pub(crate) fn origin(&self) -> [i32; 2] {
        self.origin
    }

    /// The base indices of the four taps of the cell at `(x0, y0)`.
    #[inline]
    pub(crate) fn corners(&self, x0: i32, y0: i32) -> [usize; 4] {
        let img = self.image;
        let w = img.width() as i32;
        let h = img.height() as i32;
        let c = img.channels() as usize;
        let x0 = x0.saturating_sub(self.origin[0]);
        let y0 = y0.saturating_sub(self.origin[1]);
        let x1 = x0.saturating_add(1);
        let y1 = y0.saturating_add(1);
        let cx0 = x0.clamp(0, w - 1) as usize;
        let cx1 = x1.clamp(0, w - 1) as usize;
        let cy0 = y0.clamp(0, h - 1) as usize;
        let cy1 = y1.clamp(0, h - 1) as usize;
        let stride = img.width() as usize * c;
        [
            cy0 * stride + cx0 * c,
            cy0 * stride + cx1 * c,
            cy1 * stride + cx0 * c,
            cy1 * stride + cx1 * c,
        ]
    }
}

impl LevelTexels for HeldLevel<'_> {
    #[inline]
    fn channels(&self) -> usize {
        self.image.channels() as usize
    }

    #[inline]
    fn with_cell<R>(&self, x0: i32, y0: i32, f: impl FnOnce(&[u8], [usize; 4]) -> R) -> R {
        f(self.image.data(), self.corners(x0, y0))
    }

    #[inline]
    fn with_pixel<R>(&self, x: i32, y: i32, f: impl FnOnce(&[u8]) -> R) -> R {
        let img = self.image;
        let c = img.channels() as usize;
        let x = x
            .saturating_sub(self.origin[0])
            .clamp(0, img.width() as i32 - 1) as usize;
        let y = y
            .saturating_sub(self.origin[1])
            .clamp(0, img.height() as i32 - 1) as usize;
        let at = (y * img.width() as usize + x) * c;
        f(&img.data()[at..at + c])
    }
}

impl PyramidTexels for ImageU8Pyramid {
    type Level<'a> = HeldLevel<'a>;

    #[inline]
    fn texels(&self, level: usize) -> HeldLevel<'_> {
        HeldLevel::of(self, level)
    }

    #[inline]
    fn num_levels(&self) -> usize {
        ImageU8Pyramid::num_levels(self)
    }

    #[inline]
    fn channels(&self) -> usize {
        self.level(0).channels() as usize
    }
}
