// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur matching of one tile: how sharp the tile is, how its sharpness
//! changes under round blur, and the blur that brings it to a given
//! sharpness.
//!
//! `specs/core/patch/blur-matched-zncc.md` is the design. A tile's sharpness
//! is read from its ZNCC self-similarity ellipse
//! ([`SelfSimilarityEllipse::matrix`](crate::patch::self_similarity::SelfSimilarityEllipse::matrix)),
//! short along the directions in which it holds fine detail. A tile's
//! **blur assessment** ([`BlurAssessment`], [`assess_blur`]) keeps the
//! ellipse's semi-axes and those of the tile blurred isotropically by each of
//! [`GROWTH_PROBE_SIGMAS`], each blurred tile read with the reading the tile's
//! own ellipse came from. From it, the width of the round blur that brings
//! the tile's semi-major axis to a length is read without blurring the tile
//! again ([`BlurAssessment::sigma_to_reach`]), and [`blur_to_length`] blurs
//! the tile by that width.
//!
//! Which tile of a pair is blurred, and to what length, is the caller's
//! choice: member coherence's rule is in
//! [`pair_sharpness`](crate::patch::pair_sharpness), and the scores of a
//! point's observations against its stored bitmap, which blur only the bitmap,
//! in [`stored_bitmap`](crate::patch::stored_bitmap). Both read the pair with
//! [`windowed_zncc`].
//!
//! The blur ([`blur_tile`]) is an isotropic Gaussian applied as two 1-D
//! passes, one down the columns and one along the rows, by normalized
//! convolution over the samples that carry data, so a sample without data
//! neither contributes nor receives a value.
//!
//! Blur-matched ZNCC is for scores and judging. Blurring a template before
//! aligning a view to it lowers the curvature of the correlation peak and
//! places the peak no closer to the truth, so alignment runs against the
//! unblurred tiles.

mod assess;
mod blur;
mod tiles;
mod zncc;

#[cfg(test)]
pub(crate) mod test_tiles;
#[cfg(test)]
mod tests;

pub use assess::{assess_blur, blur_to_length, blur_to_length_into, BlurAssessment};
#[cfg(test)]
pub(crate) use blur::blur_tile_direct;
pub use blur::{blur_tile, BlurScratch};
pub use tiles::{read_tile_ellipse, TilePlanes};
pub use zncc::{windowed_zncc, MIN_WINDOWED_SAMPLES};

/// The widths, in grid px, of the isotropic blurs a tile is blurred by, once
/// each, to read how its self-similarity ellipse grows ([`assess_blur`]).
/// Most pairs need a blur of 0.3 to 1 grid px; the narrower probe reads the
/// growth there, the wider one how it bends further out, where a sharp tile's
/// axis starts to lengthen faster.
pub const GROWTH_PROBE_SIGMAS: [f64; 2] = [0.4, 1.0];

/// The widest blur, in grid px. The ellipse's own axes are capped at the
/// self-similarity reading's `max_radius` (3 grid px by default).
pub const MAX_BLUR_SIGMA: f64 = 3.0;

/// The semi-axes `[major, minor]` of the ellipse `E`, in grid px: the square
/// roots of its eigenvalues. Both are `NaN` where an entry of `E` is not
/// finite.
pub fn semi_axes(e: &[[f64; 2]; 2]) -> [f64; 2] {
    if !e.iter().flatten().all(|v| v.is_finite()) {
        return [f64::NAN; 2];
    }
    let (a, b, d) = (e[0][0], 0.5 * (e[0][1] + e[1][0]), e[1][1]);
    let mean = 0.5 * (a + d);
    let half = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    [(mean + half).max(0.0).sqrt(), (mean - half).max(0.0).sqrt()]
}
