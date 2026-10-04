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
//! `1 − z(d) ≤ ε + mean_c (n / s_c)²`; and reports how far from the centre the
//! surface crosses that level at its furthest (interpolated linearly along the
//! grid edges between neighbouring shifts, and capped at `r`) and the
//! direction the indistinguishable shifts line up in.
//! [`zncc_self_similarity_parts`] reads a whole `R×R` bitmap, its
//! middle square and the nine cells of the ZNCC grid's split, sharing the
//! cross sums between them.
//!
//! [`SelfSimilarity::contour`] gives the points the radius is read from, and
//! [`SelfSimilarityReach::read`] measures their reach: along each grid axis,
//! in source-image px through
//! [`crate::camera::warp_map::patch_grid_jacobian`], and along the patch's
//! own axes in the scene's world-space unit.
//!
//! Where every sample carries data (the dense route), each template is centred
//! by its own mean per channel before the `f32` cross-sum kernel runs, so the
//! products stay on the scale of the template's own spread, and the moments
//! come from per-channel summed-area tables in `f64`. Where some sample carries
//! no data (the masked route), every sum is taken in `f64` sample by sample.
//! The combine, the tolerance test and the slide run in `f64` on both routes.

mod contour;
mod kernels;
mod overlap;

#[cfg(test)]
mod tests;

use crate::patch::normal_refine::{grid_bounds, middle_span, FLAT_NORM_SQ_EPS};

pub use contour::{
    BoundedLength, ContourPoint, PatchAxisReach, SelfSimilarityContour, SelfSimilarityReach,
};
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
    /// How far from the centre the ZNCC surface crosses the level `1 −
    /// tolerance`, `0 ..= max_radius`, in tile pixels: over the grid edges of
    /// the square of shifts between neighbouring shifts where one is at or
    /// above the level and the other below, the furthest point where the ZNCC
    /// interpolated linearly along the edge equals the level, capped at
    /// `max_radius`. A shift at or above the level on the square's border
    /// counts its own distance, at least `max_radius`, since the crossing past
    /// it was not searched. `max_radius` reads "this far or further"; it is
    /// also the reading of a template with no textured channel.
    pub radius: f64,
    /// The direction of the indistinguishable shifts, in the grid frame (`x`
    /// column-right, `y` row-down), scaled by how strongly they line up:
    /// the unit eigenvector of their second moment's larger eigenvalue `μ₁`
    /// times `1 − μ₂/μ₁`. `[0, 0]` when there are none, and for a template
    /// with no textured channel. Its sign means nothing.
    pub slide: [f64; 2],
    /// The tolerance τ the template was judged by; `f64::INFINITY` for a
    /// template with no textured channel, where every shift counts.
    pub tolerance: f64,
    /// The channel-averaged ZNCC `z(d)` for every shift of the `(2r + 1)²`
    /// square, row-major from `(dx, dy) = (−r, −r)`, `z(0, 0) = 1`: the
    /// surface the radius and the slide are read from. Every value is `NaN`
    /// for a template with no textured channel.
    pub surface: Vec<f64>,
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

/// The reading of a filled surface judged by `tolerance`: the radius, where
/// the surface crosses `1 − tolerance` furthest from the centre, capped at `r`,
/// and the slide of the indistinguishable shifts. Both read the whole square of
/// shifts. The centre does not count as a shift.
fn read_surface(surface: Vec<f64>, r: usize, tolerance: f64) -> SelfSimilarity {
    let side = 2 * r + 1;
    let ri = r as i64;
    let level = 1.0 - tolerance;
    let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if dx == 0 && dy == 0 {
                continue;
            }
            // A `NaN` reading compares false and is not indistinguishable.
            if surface[((dy + ri) as usize) * side + (dx + ri) as usize] >= level {
                let (fx, fy) = (dx as f64, dy as f64);
                sxx += fx * fx;
                sxy += fx * fy;
                syy += fy * fy;
            }
        }
    }
    SelfSimilarity {
        radius: crossing_radius(&surface, r, level),
        slide: slide_of(sxx, sxy, syy),
        tolerance,
        surface,
    }
}

/// How far from the centre the surface crosses `level`, capped at `r`.
///
/// Over every grid edge of the `(2r + 1)²` square between two neighbouring
/// shifts where one is at or above the level and the other below it, the point
/// where the ZNCC, interpolated linearly along the edge, equals the level; the
/// largest distance of those points from the centre. A shift at or above the
/// level whose neighbour was not read, because the neighbour lies past the
/// square's border or its reading is not finite, has its crossing in that
/// direction somewhere past itself, so its own distance counts as a lower
/// bound. Every shift on the border is at least `r` from the centre, so a
/// region at the level that reaches the border reads `r`. Tracked as a squared
/// length, with one square root at the end. The centre is always at or above
/// the level, so a patch that locks reads the fraction of a pixel its peak
/// takes to fall through it.
fn crossing_radius(surface: &[f64], r: usize, level: f64) -> f64 {
    radius_of_points(&crossing_points(surface, r, level), r)
}

/// The largest distance of `points` from the centre, capped at `r`: the
/// radius [`crossing_radius`] reads. Tracked as a squared length, with one
/// square root at the end.
fn radius_of_points(points: &[ContourPoint], r: usize) -> f64 {
    let mut furthest = 0.0f64;
    for point in points {
        let [px, py] = point.offset;
        furthest = furthest.max(px * px + py * py);
    }
    furthest.sqrt().min(r as f64)
}

/// The points of the contour where `surface` falls through `level`, the
/// points [`crossing_radius`] reads its radius from: over every grid edge of
/// the `(2r + 1)²` square between a shift at or above the level and a
/// neighbour below it, the point where the ZNCC, interpolated linearly along
/// the edge, equals the level; and for a shift at or above the level whose
/// neighbour was not read (past the square's border, or not finite), the
/// shift itself, open towards that neighbour. Each shift contributes its edges
/// in the order `+x`, `−x`, `+y`, `−y`, and the shifts are visited row-major
/// from `(−r, −r)`.
fn crossing_points(surface: &[f64], r: usize, level: f64) -> Vec<ContourPoint> {
    let side = 2 * r + 1;
    let ri = r as i64;
    let at = |dx: i64, dy: i64| -> Option<f64> {
        if dx.abs() > ri || dy.abs() > ri {
            return None;
        }
        let z = surface[((dy + ri) as usize) * side + (dx + ri) as usize];
        z.is_finite().then_some(z)
    };
    let mut points = Vec::new();
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let Some(z) = at(dx, dy) else { continue };
            if z < level {
                continue;
            }
            for (ex, ey) in [(1i8, 0i8), (-1, 0), (0, 1), (0, -1)] {
                let point = match at(dx + i64::from(ex), dy + i64::from(ey)) {
                    Some(zn) if zn >= level => continue,
                    Some(zn) => {
                        let t = (z - level) / (z - zn);
                        ContourPoint {
                            offset: [dx as f64 + t * f64::from(ex), dy as f64 + t * f64::from(ey)],
                            open_towards: None,
                        }
                    }
                    // Not read: the shift itself is as far as is known.
                    None => ContourPoint {
                        offset: [dx as f64, dy as f64],
                        open_towards: Some([ex, ey]),
                    },
                };
                points.push(point);
            }
        }
    }
    points
}

/// The slide of a set of shifts from the sums of their second moments: the
/// unit eigenvector of the larger eigenvalue `μ₁` scaled by `1 − μ₂/μ₁`, or
/// `[0, 0]` for an empty set. The common `1/|A|` factor cancels, so the raw
/// sums serve.
fn slide_of(sxx: f64, sxy: f64, syy: f64) -> [f64; 2] {
    let half_trace = 0.5 * (sxx + syy);
    let spread = (0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy).sqrt();
    let (mu1, mu2) = (half_trace + spread, half_trace - spread);
    if mu1 <= 0.0 {
        return [0.0; 2];
    }
    let strength = (1.0 - mu2.max(0.0) / mu1).clamp(0.0, 1.0);
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    [strength * theta.cos(), strength * theta.sin()]
}
