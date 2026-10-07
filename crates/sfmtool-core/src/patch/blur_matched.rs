// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur-matched ZNCC: the ZNCC of two views' tiles after the sharper one is
//! blurred, by a round Gaussian, until it is no sharper than the other.
//!
//! `specs/core/patch/blur-matched-zncc.md` is the design. Plain ZNCC between a
//! sharp and a blurry tile of the same surface is low because the detail the
//! sharp tile carries has nothing in the blurry one to match. Each tile's
//! sharpness is read from its ZNCC self-similarity ellipse
//! ([`SelfSimilarityEllipse::matrix`](crate::patch::self_similarity::SelfSimilarityEllipse::matrix)).
//! A tile is the sharper of a pair when its semi-major axis is shorter than
//! the other tile's semi-minor axis: it is sharper along every direction.
//! That tile is blurred isotropically until its semi-major axis reaches the
//! other's semi-minor axis ([`pair_blur`]), so the blur takes no direction
//! past what the blurrier tile shows along its sharpest one. Where neither
//! tile is the sharper in that sense, as where one tile is blurry along one
//! direction only or where the two hold a texture with a grain at different
//! angles, the pair is correlated plain. Where the semi-major axis is short
//! of the target by less than a ratio, the pair is correlated plain as well
//! ([`PairMatching`]).
//!
//! The width of the blur comes from how the tile's own semi-major axis grows
//! when the tile is blurred ([`BlurGrowth`]). Each view's tile is blurred by
//! each of [`GROWTH_PROBE_SIGMAS`], and each blurred tile's ellipse read with
//! the reading the view's own ellipse came from ([`read_growth`]); every pair
//! the view is blurred in then reads the width off those readings
//! ([`BlurGrowth::sigma_for`]) without reading again.
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

mod blur;
mod growth;
mod tiles;

#[cfg(test)]
mod tests;

#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use blur::blur_tile_direct;
pub use blur::{blur_tile, BlurScratch};
pub use growth::{read_growth, ViewGrowths};
pub use tiles::{
    blur_matched_pairs, pair_zncc_readings, read_tile_ellipse, BlurMatchedPairs, PairReadings,
    TilePlanes, MIN_WINDOWED_SAMPLES,
};

/// The widths, in grid px, of the isotropic blurs each view's tile is
/// blurred by, once each, to read how its semi-major axis grows
/// ([`read_growth`]). Most pairs need a blur of 0.3 to 1 grid px; the
/// narrower probe reads the growth there, the wider one how it bends further
/// out, where a sharp tile's axis starts to lengthen faster.
pub const GROWTH_PROBE_SIGMAS: [f64; 2] = [0.4, 1.0];

/// The shortest semi-major axis, in grid px, the skip ratio reads: a shorter
/// one is read as this one.
pub const MIN_SHARPER_LENGTH: f64 = 0.05;

/// The widest blur, in grid px. The ellipse's own axes are capped at the
/// self-similarity reading's `max_radius` (3 grid px by default).
pub const MAX_BLUR_SIGMA: f64 = 3.0;

/// The longest semi-major axis, in grid px, blur matching aims for: a
/// blurrier tile's semi-minor axis past it is read as this one. Past about
/// 2 grid px a tile holds little detail to tell it from another surface's,
/// and blurring members further towards smooth tiles of other points lowered
/// how well wrong views are told apart.
pub const MAX_MATCHED_LENGTH: f64 = 2.0;

/// The fraction by which the target must exceed the sharper tile's
/// semi-major axis for the pair to be blurred at all, whatever the ratio
/// asked for: a smaller difference is within the spread of the width the
/// growth gives.
pub const MATCHED_LENGTH_TOLERANCE: f64 = 0.05;

/// The default ratio of [`PairMatching::BlurMatchedAboveRatio`]: the target
/// must exceed the sharper tile's semi-major axis by more than this factor
/// for the pair to be blurred.
pub const DEFAULT_MIN_ELLIPSE_RATIO: f64 = 1.25;

/// How a pair of views' tiles is correlated: as they are, or blur-matched.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PairMatching {
    /// The tiles as they are.
    #[default]
    Plain,
    /// The sharper tile blurred wherever [`pair_blur`] finds one, its
    /// semi-major axis short of the target by more than
    /// [`MATCHED_LENGTH_TOLERANCE`].
    BlurMatched,
    /// The sharper tile blurred where its semi-major axis is short of the
    /// target by more than the given factor (`> 1`); a pair that differs by
    /// less is correlated plain, which costs nothing beyond the plain ZNCC.
    BlurMatchedAboveRatio(f64),
}

impl PairMatching {
    /// The factor the target must exceed the sharper semi-major axis by for
    /// the pair to be blurred, or `None` for [`PairMatching::Plain`]. `1` for
    /// [`PairMatching::BlurMatched`]; [`pair_blur`] blurs no pair whose
    /// lengths are within [`MATCHED_LENGTH_TOLERANCE`] of each other,
    /// whatever the factor.
    pub fn min_ratio(self) -> Option<f64> {
        match self {
            PairMatching::Plain => None,
            PairMatching::BlurMatched => Some(1.0),
            PairMatching::BlurMatchedAboveRatio(r) => Some(r.max(1.0)),
        }
    }

    /// Whether this blur-matches at all.
    pub fn is_blur_matched(self) -> bool {
        self.min_ratio().is_some()
    }

    /// The name the wire and the bindings spell it with:
    /// `"plain"`, `"blur_matched"` or `"blur_matched_above_ratio"`.
    pub fn name(self) -> &'static str {
        match self {
            PairMatching::Plain => "plain",
            PairMatching::BlurMatched => "blur_matched",
            PairMatching::BlurMatchedAboveRatio(_) => "blur_matched_above_ratio",
        }
    }

    /// Parse [`Self::name`], with `ratio` the factor of
    /// [`PairMatching::BlurMatchedAboveRatio`]; `None` for an unknown name.
    pub fn from_name(name: &str, ratio: f64) -> Option<Self> {
        match name {
            "plain" => Some(PairMatching::Plain),
            "blur_matched" => Some(PairMatching::BlurMatched),
            "blur_matched_above_ratio" => Some(PairMatching::BlurMatchedAboveRatio(ratio)),
            _ => None,
        }
    }
}

/// How one view's self-similarity semi-major axis grows when its tile is
/// blurred: the tile's semi-major axis and those of the tile blurred
/// isotropically by each of [`GROWTH_PROBE_SIGMAS`], all read the same way
/// ([`read_growth`]).
///
/// The square of the semi-major axis against `σ²` is taken to be piecewise
/// linear through the readings, from no blur to the widest probe, and to go
/// on past the widest probe along its last piece. The width that brings the
/// axis to a target is read back off that line ([`BlurGrowth::sigma_for`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurGrowth {
    /// The unblurred tile's semi-major axis, then the probes' in the order of
    /// [`GROWTH_PROBE_SIGMAS`], in grid px.
    pub semi_major: [f64; GROWTH_PROBE_SIGMAS.len() + 1],
}

impl BlurGrowth {
    /// The width that brings the semi-major axis to `target`, capped at
    /// [`MAX_BLUR_SIGMA`]; `0` where the unblurred axis is already there,
    /// `None` where the readings do not grow or are not finite.
    ///
    /// The line goes on past the widest probe along its last piece, and a
    /// reading shorter than the one before it is passed over.
    pub fn sigma_for(&self, target: f64) -> Option<f64> {
        let l2 = self.semi_major.map(|l| l * l);
        let t2 = target * target;
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

/// The semi-axes `[major, minor]` of the ellipse `E`, in grid px: the square
/// roots of its eigenvalues.
pub fn semi_axes(e: &[[f64; 2]; 2]) -> [f64; 2] {
    let (a, b, d) = (e[0][0], 0.5 * (e[0][1] + e[1][0]), e[1][1]);
    let mean = 0.5 * (a + d);
    let half = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    [(mean + half).max(0.0).sqrt(), (mean - half).max(0.0).sqrt()]
}

/// Which tile of a pair is to be blurred, and to what semi-major axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairBlur {
    /// The tile to blur: `0` for the first of the pair, `1` for the second.
    pub sharper: usize,
    /// Its semi-major axis, in grid px.
    pub major: f64,
    /// The semi-major axis the blur aims for: the other tile's semi-minor
    /// axis, at most [`MAX_MATCHED_LENGTH`].
    pub target: f64,
}

/// Which tile of a pair is to be blurred, from their self-similarity
/// ellipses `a` and `b` (each `E`,
/// [`SelfSimilarityEllipse::matrix`](crate::patch::self_similarity::SelfSimilarityEllipse::matrix),
/// in grid px²); `None` where the pair is correlated plain.
///
/// A tile is blurred when its semi-major axis is shorter than the other
/// tile's semi-minor axis, the target, by more than `min_ratio` (and by more
/// than [`MATCHED_LENGTH_TOLERANCE`]). The target is at most
/// [`MAX_MATCHED_LENGTH`]. At most one tile of a pair can be shorter in that
/// way. A pair with an ellipse that cannot be read is not blurred.
///
/// ```
/// use sfmtool_core::patch::blur_matched::pair_blur;
///
/// // `a` is round and 0.5 long; `b` is 1 long along x and 1.5 along y.
/// let a = [[0.25, 0.0], [0.0, 0.25]];
/// let b = [[1.0, 0.0], [0.0, 2.25]];
/// let blur = pair_blur(&a, &b, 1.0).unwrap();
/// assert_eq!(blur.sharper, 0);
/// assert!((blur.major - 0.5).abs() < 1e-12 && (blur.target - 1.0).abs() < 1e-12);
/// // `c` is as long as `a` along x: it is not blurrier along every direction.
/// let c = [[0.25, 0.0], [0.0, 2.25]];
/// assert!(pair_blur(&a, &c, 1.0).is_none());
/// ```
pub fn pair_blur(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], min_ratio: f64) -> Option<PairBlur> {
    if !(readable(a) && readable(b)) {
        return None;
    }
    let ([ma, na], [mb, nb]) = (semi_axes(a), semi_axes(b));
    let (sharper, major, minor) = if ma < nb {
        (0, ma, nb)
    } else if mb < na {
        (1, mb, na)
    } else {
        return None;
    };
    let target = minor.min(MAX_MATCHED_LENGTH);
    let ratio = min_ratio.max(1.0 + MATCHED_LENGTH_TOLERANCE);
    (target > major && target >= ratio * major.max(MIN_SHARPER_LENGTH)).then_some(PairBlur {
        sharper,
        major,
        target,
    })
}

/// [`pair_blur`] under `matching`: `None` for [`PairMatching::Plain`].
pub fn pair_blur_for(
    a: &[[f64; 2]; 2],
    b: &[[f64; 2]; 2],
    matching: PairMatching,
) -> Option<PairBlur> {
    pair_blur(a, b, matching.min_ratio()?)
}

/// Whether an ellipse matrix can be read: every entry finite.
fn readable(e: &[[f64; 2]; 2]) -> bool {
    e.iter().flatten().all(|v| v.is_finite())
}
