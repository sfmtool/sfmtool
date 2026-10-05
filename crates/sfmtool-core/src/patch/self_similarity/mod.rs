// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The ZNCC self-similarity radius: how far a patch can slide over itself, by
//! whole pixels, and still look as much like itself as a true match between
//! two views would.
//!
//! See `specs/core/patch/zncc-self-similarity-radius.md`. A reading is of a
//! bitmap as it is, with no pixels from outside it: for a template rectangle
//! inside a tile, [`zncc_self_similarity_radius`] computes the
//! channel-averaged ZNCC of the template against the window of the same size
//! moved by every shift `d` of the square `|dx|, |dy| ≤ r`, over only the
//! samples inside the tile, and carrying data, on both sides (the overlap
//! reading); counts a shift as indistinguishable when
//! `1 − z(d) ≤ ε + mean_c (n / s_c)²`; and summarises the region of those
//! shifts, the surface taken as bilinear between them, by the ellipse with the
//! same second moments per unit area about the true position, whose semi-major
//! axis, capped at `r`, is the radius.
//! [`zncc_self_similarity_parts`] reads a whole `R×R` bitmap, its
//! middle square and the nine cells of the ZNCC grid's split, sharing the
//! cross sums between them.
//!
//! [`SelfSimilarityEllipse::mapped`] takes the ellipse into source-image px
//! through [`crate::camera::warp_map::patch_grid_jacobian`], and
//! [`SelfSimilarityEllipse::on_patch`] along the patch's own axes in the
//! scene's world-space unit; [`SelfSimilarityEllipseUnits::read`] does both.
//!
//! Where every sample carries data (the dense route), each template is centred
//! by its own mean per channel before the `f32` cross-sum kernel runs, so the
//! products stay on the scale of the template's own spread, and the moments
//! come from per-channel summed-area tables in `f64`. Where some sample carries
//! no data (the masked route), every sum is taken in `f64` sample by sample.
//! The combine, the tolerance test and the ellipse run in `f64` on both routes.

mod ellipse;
mod kernels;
mod overlap;

#[cfg(test)]
mod tests;

use crate::patch::normal_refine::{grid_bounds, middle_span, FLAT_NORM_SQ_EPS};

pub use ellipse::{PatchEllipse, SelfSimilarityEllipse, SelfSimilarityEllipseUnits};
pub use overlap::{zncc_self_similarity_parts, zncc_self_similarity_radius};

/// The template spread, in grey levels, under which a channel carries no
/// texture: it is left out of the channel average, and a template with no
/// channel at or above it scores `max_radius`.
pub const FLAT_FLOOR: f64 = 0.5;

/// How far a patch can slide over itself and stay indistinguishable from its
/// true position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfSimilarityParams {
    /// How far the shifts searched reach along each axis, in tile pixels,
    /// and the largest radius read; a radius of `max_radius` reads "this far
    /// or further".
    pub max_radius: u32,
    /// ε: the ZNCC deficit two views of the same surface show from warp, blur
    /// and lighting, as a fraction.
    pub relative_tolerance: f64,
    /// n: the noise between two views, in grey levels.
    pub noise: f64,
}

impl Default for SelfSimilarityParams {
    fn default() -> Self {
        Self {
            max_radius: 3,
            relative_tolerance: 0.05,
            noise: 2.0,
        }
    }
}

/// One template's reading.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfSimilarity {
    /// The ZNCC self-similarity radius, `0 ..= max_radius`, in tile pixels:
    /// the semi-major axis of [`Self::ellipse`], capped at `max_radius` as
    /// the ellipse's axes are, so `max_radius` reads "this far or further"; it
    /// is also the reading of a template with no textured channel.
    /// [`Self::radius_is_at_least`] says whether the true length may be
    /// larger. `NaN` for a template with no data.
    pub radius: f64,
    /// The ellipse with the same second moments per unit area about `d = 0` as the
    /// region of shifts where the surface, bilinear between the whole-pixel
    /// shifts, is at or above `1 − tolerance`, in the grid frame (`x`
    /// column-right, `y` row-down).
    pub ellipse: SelfSimilarityEllipse,
    /// The tolerance τ the template was judged by; `f64::INFINITY` for a
    /// template with no textured channel, where every shift counts.
    pub tolerance: f64,
    /// The channel-averaged ZNCC `z(d)` for every shift of the `(2r + 1)²`
    /// square, row-major from `(dx, dy) = (−r, −r)`, `z(0, 0) = 1`: the
    /// surface the ellipse is read from. Every value is `NaN` for a template
    /// with no textured channel.
    pub surface: Vec<f64>,
}

impl SelfSimilarity {
    /// Whether the true radius may be larger than [`Self::radius`]: the
    /// ellipse's major axis is a lower bound (the region at the level runs off
    /// the square of shifts, a gap with no reading beside it could hide more
    /// of it, or it reached `max_radius`).
    pub fn radius_is_at_least(&self) -> bool {
        self.ellipse.axes_is_at_least[0]
    }
}

/// A bitmap whole, its middle square and the nine cells of the ZNCC grid's
/// split, each read as its own template.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfSimilarityParts {
    /// The whole `R×R` bitmap.
    pub whole: SelfSimilarity,
    /// The middle square, the bitmap's rows and columns `R/4 .. R − R/4`.
    pub middle: SelfSimilarity,
    /// Each cell of the bitmap's split at `R/3` and `R − R/3`, `grid[row][col]`
    /// from the top-left cell.
    pub grid: [[SelfSimilarity; 3]; 3],
}

/// A tile, planar: channel `c`'s row-major `width × height` plane at
/// `values[c * width * height ..]`. Up to three colour channels.
#[derive(Debug, Clone, Copy)]
pub struct PatchTile<'a> {
    pub values: &'a [f32],
    pub channels: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a> PatchTile<'a> {
    /// The planar colour channels of an interleaved `height × width × C`
    /// patch, the layout a rendered bitmap arrives in, dropping a fourth
    /// channel (alpha): returns the planes and the colour channel count.
    ///
    /// # Panics
    ///
    /// Panics if `patch.len() != width * height * channels` or `channels` is 0.
    pub fn planes_from_interleaved(
        patch: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> (Vec<f32>, usize) {
        assert!(channels > 0, "planes_from_interleaved: no channels");
        assert_eq!(
            patch.len(),
            width * height * channels,
            "planes_from_interleaved: a {width}×{height}×{channels} patch has {} values, not {}",
            width * height * channels,
            patch.len()
        );
        let colour = channels.min(3);
        let n = width * height;
        let mut planes = vec![0.0f32; colour * n];
        for (p, pixel) in patch.chunks_exact(channels).enumerate() {
            for c in 0..colour {
                planes[c * n + p] = pixel[c];
            }
        }
        (planes, colour)
    }

    /// Which samples of an interleaved `height × width × C` patch carry data,
    /// for the overlap reading: with a fourth channel (alpha), the samples
    /// whose alpha is above 0; without one, every sample, returned as `None`.
    ///
    /// A fused consensus bitmap writes alpha 0 where no view covered the
    /// sample, where only one view did, and where the views disagree so much
    /// that the confidence rounds to 0; its colour there is zero or one view's
    /// reading, so none of those samples is treated as data.
    ///
    /// # Panics
    ///
    /// Panics if `patch.len() != width * height * channels` or `channels` is 0.
    pub fn data_from_interleaved(
        patch: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Option<Vec<bool>> {
        assert!(channels > 0, "data_from_interleaved: no channels");
        assert_eq!(
            patch.len(),
            width * height * channels,
            "data_from_interleaved: a {width}×{height}×{channels} patch has {} values, not {}",
            width * height * channels,
            patch.len()
        );
        (channels >= 4).then(|| {
            patch
                .chunks_exact(channels)
                .map(|pixel| pixel[3] > 0.0)
                .collect()
        })
    }

    /// Check the tile's own shape: one to three channels and a value for
    /// every pixel of every channel.
    fn validate(&self) {
        assert!(
            (1..=3).contains(&self.channels),
            "PatchTile: {} channels, expected 1 to 3 colour channels",
            self.channels
        );
        assert_eq!(
            self.values.len(),
            self.channels * self.width * self.height,
            "PatchTile: a {}×{}×{} tile has {} values, not {}",
            self.width,
            self.height,
            self.channels,
            self.channels * self.width * self.height,
            self.values.len()
        );
    }
}

/// Which cross-sum kernel the dense route runs: the dispatching one, or the
/// scalar reference (for the equivalence test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Dispatch,
    #[cfg_attr(not(test), allow(dead_code))]
    Scalar,
}

/// The reading of a filled surface judged by `tolerance`: the ellipse of the
/// region at or above `1 − tolerance`, read over the whole square of shifts,
/// and the radius, its semi-major axis.
fn read_surface(surface: Vec<f64>, r: usize, tolerance: f64) -> SelfSimilarity {
    let ellipse = ellipse::fit_ellipse(&surface, r, 1.0 - tolerance);
    SelfSimilarity {
        radius: ellipse.axes[0],
        ellipse,
        tolerance,
        surface,
    }
}

/// The reading of a template with no sample carrying data: `NaN` throughout.
fn no_reading(r: usize) -> SelfSimilarity {
    let side = 2 * r + 1;
    SelfSimilarity {
        radius: f64::NAN,
        ellipse: SelfSimilarityEllipse::no_data(),
        tolerance: f64::NAN,
        surface: vec![f64::NAN; side * side],
    }
}

/// The reading of a template with no textured channel: every shift counts, so
/// the radius is `r`, a lower bound, the ellipse a circle of radius `r`, and
/// the surface `NaN`.
fn flat_reading(r: usize) -> SelfSimilarity {
    let side = 2 * r + 1;
    SelfSimilarity {
        radius: r as f64,
        ellipse: SelfSimilarityEllipse::flat(r),
        tolerance: f64::INFINITY,
        surface: vec![f64::NAN; side * side],
    }
}
