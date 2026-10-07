// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One tile's blur assessment, and the blur that brings the tile's semi-major
//! axis to a length.

use super::blur::{blur_tile, BlurScratch};
use super::tiles::TilePlanes;
use super::{semi_axes, GROWTH_PROBE_SIGMAS, MAX_BLUR_SIGMA};

/// How sharp one tile is, and how that changes under round blur: the
/// semi-axes of its self-similarity ellipse as read, and those of the tile
/// blurred isotropically by each of [`GROWTH_PROBE_SIGMAS`], all read the same
/// way ([`assess_blur`]).
///
/// It depends on the tile alone, so it is read once per tile, whatever the
/// tile is later compared with. The square of the semi-major axis against
/// `σ²` is taken to be piecewise linear through the readings, from no blur to
/// the widest probe, and to go on past the widest probe along its last piece;
/// the width that brings the axis to a length is read back off that line
/// ([`Self::sigma_to_reach`]) without blurring the tile again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurAssessment {
    /// The semi-axes `[major, minor]` of the tile as read, in grid px.
    pub semi_axes: [f64; 2],
    /// The semi-axes `[major, minor]` of the tile blurred by each of
    /// [`GROWTH_PROBE_SIGMAS`], in that order, in grid px.
    pub growth: [[f64; 2]; GROWTH_PROBE_SIGMAS.len()],
}

impl BlurAssessment {
    /// The width of the round blur that brings the semi-major axis to
    /// `length`, in grid px, capped at [`MAX_BLUR_SIGMA`]; `0` where the
    /// unblurred axis is already that long, `None` where the readings do not
    /// grow or are not finite.
    ///
    /// The line goes on past the widest probe along its last piece, and a
    /// reading shorter than the one before it is passed over.
    ///
    /// ```
    /// use sfmtool_core::patch::blur_matched::BlurAssessment;
    ///
    /// // A semi-major axis whose square is 0.25 + σ², read at σ = 0.4 and 1.
    /// let a = BlurAssessment {
    ///     semi_axes: [0.5, 0.4],
    ///     growth: [[0.41f64.sqrt(), 0.5], [1.25f64.sqrt(), 1.2]],
    /// };
    /// let sigma = a.sigma_to_reach(1.0).unwrap();
    /// assert!((sigma - 0.75f64.sqrt()).abs() < 1e-12);
    /// assert_eq!(a.sigma_to_reach(0.5), Some(0.0));
    /// ```
    pub fn sigma_to_reach(&self, length: f64) -> Option<f64> {
        let mut l2 = [self.semi_axes[0] * self.semi_axes[0]; GROWTH_PROBE_SIGMAS.len() + 1];
        for (l, axes) in l2[1..].iter_mut().zip(&self.growth) {
            *l = axes[0] * axes[0];
        }
        let t2 = length * length;
        if !(t2.is_finite() && l2.iter().all(|v| v.is_finite())) {
            return None;
        }
        let mut prev = (0.0, l2[0]);
        if t2 <= prev.1 {
            return Some(0.0);
        }
        let last = GROWTH_PROBE_SIGMAS.len() - 1;
        for (i, &w) in GROWTH_PROBE_SIGMAS.iter().enumerate() {
            let cur = (w * w, l2[i + 1]);
            let slope = (cur.1 - prev.1) / (cur.0 - prev.0);
            if slope > 0.0 && (t2 <= cur.1 || i == last) {
                let s2 = prev.0 + (t2 - prev.1) / slope;
                return Some(s2.max(0.0).sqrt().min(MAX_BLUR_SIGMA));
            }
            if cur.1 >= prev.1 {
                prev = cur;
            }
        }
        None
    }
}

/// The blur assessment of `tile`, whose self-similarity ellipse matrix is
/// `ellipse` (in grid px²): the tile blurred by each of
/// [`GROWTH_PROBE_SIGMAS`] in turn, each blurred tile's ellipse read by
/// `read`, the semi-axes of every reading kept beside those of `ellipse`.
/// `None` where `read` gives no ellipse for any probe.
///
/// `read` takes a tile's colour planes, laid out as `tile.values`, and must
/// read them the way `ellipse` was read, the same reading of the same render,
/// so that the growth is the reading's own and the lengths later compared
/// are of one reading. [`read_tile_ellipse`](super::read_tile_ellipse) is the
/// whole-tile reading over the samples with data.
///
/// ```
/// use sfmtool_core::patch::blur_matched::{
///     assess_blur, read_tile_ellipse, BlurScratch, TilePlanes,
/// };
///
/// fn assess(tile: &TilePlanes, scratch: &mut BlurScratch) {
///     let read = |values: &[f32]| read_tile_ellipse(values, tile.channels, tile.side, &tile.data);
///     let Some(ellipse) = read(&tile.values) else { return };
///     if let Some(a) = assess_blur(tile, &ellipse, read, scratch) {
///         println!("semi-axes {:?}, after the probes {:?}", a.semi_axes, a.growth);
///     }
/// }
/// ```
///
/// # Panics
///
/// Panics if `tile`'s planes or data flags do not cover it.
pub fn assess_blur(
    tile: &TilePlanes,
    ellipse: &[[f64; 2]; 2],
    mut read: impl FnMut(&[f32]) -> Option<[[f64; 2]; 2]>,
    scratch: &mut BlurScratch,
) -> Option<BlurAssessment> {
    let mut probe = std::mem::take(&mut scratch.probe);
    probe.resize(tile.values.len(), 0.0);
    let mut growth = [[f64::NAN; 2]; GROWTH_PROBE_SIGMAS.len()];
    let mut read_all = true;
    for (axes, &sigma) in growth.iter_mut().zip(&GROWTH_PROBE_SIGMAS) {
        blur_tile(
            &tile.values,
            tile.channels,
            tile.side,
            &tile.data,
            sigma,
            &mut probe,
            scratch,
        );
        match read(&probe) {
            Some(e) => *axes = semi_axes(&e),
            None => {
                read_all = false;
                break;
            }
        }
    }
    scratch.probe = probe;
    read_all.then(|| BlurAssessment {
        semi_axes: semi_axes(ellipse),
        growth,
    })
}

/// `tile` blurred isotropically until its semi-major axis reaches `length`,
/// by the width `assessment` (the tile's own, [`assess_blur`]) gives
/// ([`BlurAssessment::sigma_to_reach`]), and that width in grid px. A tile
/// whose semi-major axis is already that long is returned unblurred, with a
/// width of `0`. `None` where the assessment gives no width.
pub fn blur_to_length(
    tile: &TilePlanes,
    assessment: &BlurAssessment,
    length: f64,
    scratch: &mut BlurScratch,
) -> Option<(TilePlanes, f64)> {
    let mut out = TilePlanes::default();
    let sigma = blur_to_length_into(tile, assessment, length, &mut out, scratch)?;
    Some((out, sigma))
}

/// [`blur_to_length`] into `out`, reusing its buffers, returning the width.
/// `out` is left as it was where the assessment gives no width.
pub fn blur_to_length_into(
    tile: &TilePlanes,
    assessment: &BlurAssessment,
    length: f64,
    out: &mut TilePlanes,
    scratch: &mut BlurScratch,
) -> Option<f64> {
    let sigma = assessment.sigma_to_reach(length)?;
    tile.blur_into(sigma, out, scratch);
    Some(sigma)
}
