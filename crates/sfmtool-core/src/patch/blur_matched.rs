// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur-matched ZNCC: the ZNCC of two views' tiles after the sharper one is
//! blurred, direction by direction, to the other's sharpness.
//!
//! `specs/core/patch/blur-matched-zncc.md` is the design. Plain ZNCC between a
//! sharp and a blurry tile of the same surface is low because the detail the
//! sharp tile carries has nothing in the blurry one to match. Each tile's
//! sharpness is read from its ZNCC self-similarity ellipse
//! ([`SelfSimilarityEllipse::matrix`](crate::patch::self_similarity::SelfSimilarityEllipse::matrix)).
//! Along each eigenvector `u` of the difference of the two ellipses, the tile
//! whose ellipse is shorter along `u` is to be blurred along `u` until its
//! ellipse is as long there as the other's ([`pair_directions`]). Either tile,
//! or both, may be blurred, each only along the directions in which it is the
//! sharper one. Where the two lengths along a direction differ by less than a
//! ratio, that direction is left alone, and a pair left alone along both is
//! correlated plain ([`PairMatching`]).
//!
//! The width of each blur is found by measurement ([`match_blur`]): the tile is
//! blurred, its ellipse read again with the same reading, and the width
//! corrected by a secant step on `σ²` until the blurred tile's length along
//! each direction is within [`BLUR_MATCH_TOLERANCE`] of the blurrier tile's,
//! at most [`BLUR_MATCH_MAX_PROBES`] times. The first width tried comes from
//! [`estimated_blur_sigma`], a difference of squares calibrated on real tiles.
//!
//! The blur ([`blur_tile`]) is an anisotropic Gaussian applied as two 1-D
//! passes, one along a grid axis and one along a slanted line, by normalized
//! convolution over the samples that carry data, so a sample without data
//! neither contributes nor receives a value.
//!
//! Blur-matched ZNCC is for scores and judging. Blurring a template before
//! aligning a view to it lowers the curvature of the correlation peak and
//! places the peak no closer to the truth, so alignment runs against the
//! unblurred tiles.

mod blur;
mod search;
mod tiles;

#[cfg(test)]
mod tests;

#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use blur::blur_tile_direct;
pub use blur::{blur_tile, BlurScratch};
pub use search::{match_blur, MatchScratch, MatchedBlur};
pub use tiles::{
    blur_matched_pairs, pair_zncc_readings, read_tile_ellipse, BlurMatchKernel, BlurMatchedPairs,
    PairReadings, TilePlanes, MIN_WINDOWED_SAMPLES,
};

/// `k` of [`estimated_blur_sigma`]'s growth `k · s^p`: how fast the square of
/// a tile's ellipse length along a direction grows with the square of a blur
/// along it, `d(l²)/d(σ²)`, for a sharper length `s` of 1 grid px.
pub const BLUR_GROWTH_SCALE: f64 = 1.243;

/// `p` of [`estimated_blur_sigma`]'s growth `k · s^p`: the power of the
/// sharper length.
pub const BLUR_GROWTH_POWER: f64 = 1.247;

/// The shortest sharper length, in grid px, the skip ratio and
/// [`estimated_blur_sigma`] read: shorter lengths are read as this one.
pub const MIN_SHARPER_LENGTH: f64 = 0.05;

/// The widest blur, in grid px, along a direction. The ellipse's own axes are
/// capped at the self-similarity reading's `max_radius` (3 grid px by
/// default), and a blur of 3 takes the median sharp tile to about 2.5.
pub const MAX_BLUR_SIGMA: f64 = 3.0;

/// The longest ellipse length, in grid px, blur matching aims for: a blurrier
/// tile's length past it is read as this one. Past about 2 grid px a tile
/// holds little detail to tell it from another surface's, and matching the
/// full length blurred members towards smooth tiles of other points enough to
/// lower how well wrong views are told apart.
pub const MAX_MATCHED_LENGTH: f64 = 2.0;

/// How close, as a fraction of the length aimed for, the blurred tile's
/// ellipse length along each direction must come for [`match_blur`] to stop.
pub const BLUR_MATCH_TOLERANCE: f64 = 0.05;

/// The most blurs [`match_blur`] tries, and reads the ellipse of, for one tile
/// of a pair.
pub const BLUR_MATCH_MAX_PROBES: u32 = 2;

/// The blur widths of the isotropic ladder ([`BlurMatchKernel::IsotropicLadder`]),
/// in grid px: `0.25 · √2ⁿ` for `n = 0 .. 7`.
pub const LADDER_SIGMAS: [f64; 8] = [
    0.25,
    0.25 * std::f64::consts::SQRT_2,
    0.5,
    0.5 * std::f64::consts::SQRT_2,
    1.0,
    std::f64::consts::SQRT_2,
    2.0,
    2.0 * std::f64::consts::SQRT_2,
];

/// The default ratio of [`PairMatching::BlurMatchedAboveRatio`]: the two
/// ellipses' lengths along a direction must differ by more than this factor
/// for that direction to be blurred.
pub const DEFAULT_MIN_ELLIPSE_RATIO: f64 = 1.25;

/// How a pair of views' tiles is correlated: as they are, or blur-matched.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PairMatching {
    /// The tiles as they are.
    #[default]
    Plain,
    /// The sharper tile blurred along every direction in which the two
    /// ellipses differ at all.
    BlurMatched,
    /// The sharper tile blurred along the directions in which the two
    /// ellipses' lengths differ by more than the given factor (`> 1`); a pair
    /// whose ellipses differ by less along both directions is correlated
    /// plain, which costs nothing beyond the plain ZNCC.
    BlurMatchedAboveRatio(f64),
}

impl PairMatching {
    /// The factor two lengths must differ by for a direction to be blurred, or
    /// `None` for [`PairMatching::Plain`]. `1` for
    /// [`PairMatching::BlurMatched`].
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

/// The covariance of a 2-D Gaussian blur in grid px², in the grid's frame
/// (`x` column-right, `y` row-down), the frame the self-similarity ellipse is
/// read in.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BlurCovariance {
    /// `Σ_xx`.
    pub xx: f64,
    /// `Σ_xy`.
    pub xy: f64,
    /// `Σ_yy`.
    pub yy: f64,
}

impl BlurCovariance {
    /// No blur.
    pub const ZERO: Self = Self {
        xx: 0.0,
        xy: 0.0,
        yy: 0.0,
    };

    /// An isotropic blur of width `sigma`.
    pub fn isotropic(sigma: f64) -> Self {
        Self {
            xx: sigma * sigma,
            xy: 0.0,
            yy: sigma * sigma,
        }
    }

    /// A 1-D blur of width `sigma` along the unit direction `u`.
    pub fn along(u: [f64; 2], sigma: f64) -> Self {
        let s2 = sigma * sigma;
        Self {
            xx: s2 * u[0] * u[0],
            xy: s2 * u[0] * u[1],
            yy: s2 * u[1] * u[1],
        }
    }

    /// Whether the blur moves nothing: its trace is under `1e-6` px².
    pub fn is_zero(&self) -> bool {
        self.xx + self.yy < 1e-6
    }

    /// The variance of the blur along the unit direction `u`, `uᵀ Σ u`.
    pub fn variance_along(&self, u: [f64; 2]) -> f64 {
        u[0] * u[0] * self.xx + 2.0 * u[0] * u[1] * self.xy + u[1] * u[1] * self.yy
    }

    fn add(self, other: Self) -> Self {
        Self {
            xx: self.xx + other.xx,
            xy: self.xy + other.xy,
            yy: self.yy + other.yy,
        }
    }
}

/// The width of the 1-D Gaussian, in grid px, expected to blur a tile whose
/// self-similarity ellipse is `sharper` long along a direction until it is
/// `blurrier` long there: the first width [`match_blur`] tries.
///
/// The square of the length grows about linearly with the square of the
/// blur, at a rate that rises with the sharper length:
///
/// `σ² = (blurrier² − s²) / (k · s^p)`, `s = max(sharper, `[`MIN_SHARPER_LENGTH`]`)`,
///
/// with `k` [`BLUR_GROWTH_SCALE`] and `p` [`BLUR_GROWTH_POWER`], capped at
/// [`MAX_BLUR_SIGMA`]. `0` where `blurrier ≤ s` or either length is not
/// finite. The constants were fitted on real tiles of ten datasets; a single
/// tile's growth is spread about them by a factor of about 2 either way, which
/// is why the width is then measured rather than taken from here.
///
/// ```
/// use sfmtool_core::patch::blur_matched::estimated_blur_sigma;
///
/// assert_eq!(estimated_blur_sigma(0.5, 0.5), 0.0);
/// // Half as long again along a direction: a blur of about 0.77 grid px.
/// let s = estimated_blur_sigma(0.75, 0.5);
/// assert!(s > 0.75 && s < 0.8, "{s}");
/// ```
pub fn estimated_blur_sigma(blurrier: f64, sharper: f64) -> f64 {
    if !(blurrier.is_finite() && sharper.is_finite()) {
        return 0.0;
    }
    let s = sharper.max(MIN_SHARPER_LENGTH);
    // Two lengths equal but for rounding are equal.
    if blurrier <= s * (1.0 + 1e-9) {
        return 0.0;
    }
    let growth = BLUR_GROWTH_SCALE * s.powf(BLUR_GROWTH_POWER);
    ((blurrier * blurrier - s * s) / growth)
        .sqrt()
        .min(MAX_BLUR_SIGMA)
}

/// The half-width of the ellipse `E` along the unit direction `u`:
/// `sqrt(uᵀ E u)`.
pub fn length_along(e: &[[f64; 2]; 2], u: [f64; 2]) -> f64 {
    (u[0] * u[0] * e[0][0] + 2.0 * u[0] * u[1] * e[0][1] + u[1] * u[1] * e[1][1])
        .max(0.0)
        .sqrt()
}

/// Whether an ellipse matrix can be read: every entry finite.
fn readable(e: &[[f64; 2]; 2]) -> bool {
    e.iter().flatten().all(|v| v.is_finite())
}

/// One direction along which one tile of a pair is to be blurred.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BlurDirection {
    /// The unit direction, in grid px (`x` column-right, `y` row-down).
    pub u: [f64; 2],
    /// The length along `u` of the tile to be blurred, the sharper one.
    pub sharper: f64,
    /// The length the blur aims for: the other tile's length along `u`, at
    /// most [`MAX_MATCHED_LENGTH`].
    pub blurrier: f64,
}

/// The directions, at most two, along which one tile of a pair is to be
/// blurred.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TileDirections {
    dirs: [BlurDirection; 2],
    len: usize,
}

impl TileDirections {
    /// The directions.
    pub fn as_slice(&self) -> &[BlurDirection] {
        &self.dirs[..self.len]
    }

    /// Whether the tile is not to be blurred.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The blur [`estimated_blur_sigma`] gives along each direction, added up.
    pub fn estimated_blur(&self) -> BlurCovariance {
        self.as_slice().iter().fold(BlurCovariance::ZERO, |cov, d| {
            cov.add(BlurCovariance::along(
                d.u,
                estimated_blur_sigma(d.blurrier, d.sharper),
            ))
        })
    }

    fn push(&mut self, d: BlurDirection) {
        self.dirs[self.len] = d;
        self.len += 1;
    }
}

/// The directions along which each tile of a pair is to be blurred.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PairDirections {
    /// The first tile's.
    pub a: TileDirections,
    /// The second tile's.
    pub b: TileDirections,
}

impl PairDirections {
    /// Whether neither tile is to be blurred: the pair is correlated plain.
    pub fn is_none(&self) -> bool {
        self.a.is_empty() && self.b.is_empty()
    }
}

/// Which tile of a pair is to be blurred along which direction, from their
/// self-similarity ellipses `a` and `b` (each `E`,
/// [`SelfSimilarityEllipse::matrix`](crate::patch::self_similarity::SelfSimilarityEllipse::matrix),
/// in grid px²), leaving a direction alone where the two lengths along it
/// differ by less than `min_ratio`.
///
/// Along each eigenvector `u` of `E_b − E_a`, the tile whose ellipse is
/// shorter along `u` is to be blurred along `u` until it is as long there as
/// the other, or [`MAX_MATCHED_LENGTH`] long where the other is longer. A pair
/// with an ellipse that cannot be read is not blurred.
///
/// ```
/// use sfmtool_core::patch::blur_matched::pair_directions;
///
/// // `a` is sharp along x and as blurry as `b` along y; `b` is round.
/// let a = [[0.25, 0.0], [0.0, 1.0]];
/// let b = [[1.0, 0.0], [0.0, 1.0]];
/// let dirs = pair_directions(&a, &b, 1.0);
/// assert!(dirs.b.is_empty());
/// let [d] = dirs.a.as_slice() else { panic!() };
/// assert!(d.u[0].abs() > 0.999 && (d.sharper - 0.5).abs() < 1e-9 && (d.blurrier - 1.0).abs() < 1e-9);
/// ```
pub fn pair_directions(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], min_ratio: f64) -> PairDirections {
    let mut out = PairDirections::default();
    if !(readable(a) && readable(b)) {
        return out;
    }
    let d = [
        b[0][0] - a[0][0],
        0.5 * (b[0][1] + b[1][0] - a[0][1] - a[1][0]),
        b[1][1] - a[1][1],
    ];
    // The eigenvectors of a symmetric 2×2.
    let theta = 0.5 * (2.0 * d[1]).atan2(d[0] - d[2]);
    let (s, c) = theta.sin_cos();
    for u in [[c, s], [-s, c]] {
        let la = length_along(a, u);
        let lb = length_along(b, u);
        let (longer, shorter) = if la > lb { (la, lb) } else { (lb, la) };
        let longer = longer.min(MAX_MATCHED_LENGTH);
        if longer <= shorter || longer < min_ratio * shorter.max(MIN_SHARPER_LENGTH) {
            continue;
        }
        if estimated_blur_sigma(longer, shorter) <= 0.0 {
            continue;
        }
        let dir = BlurDirection {
            u,
            sharper: shorter,
            blurrier: longer,
        };
        if lb > la {
            out.a.push(dir);
        } else {
            out.b.push(dir);
        }
    }
    out
}

/// [`pair_directions`] under `matching`: none for [`PairMatching::Plain`].
pub fn pair_directions_for(
    a: &[[f64; 2]; 2],
    b: &[[f64; 2]; 2],
    matching: PairMatching,
) -> PairDirections {
    match matching.min_ratio() {
        Some(ratio) => pair_directions(a, b, ratio),
        None => PairDirections::default(),
    }
}
