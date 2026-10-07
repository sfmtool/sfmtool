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
//! whose ellipse is shorter along `u` is blurred along `u` by a 1-D Gaussian
//! whose width the fitted mapping [`blur_sigma`] reads off the two lengths
//! ([`pair_blur`]). Either tile, or both, may be blurred, each only along the
//! directions in which it is the sharper one. Where the two lengths along a
//! direction differ by less than a ratio, that direction is left alone, and a
//! pair left alone along both is correlated plain ([`PairMatching`]).
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
mod tiles;

#[cfg(test)]
mod tests;

#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use blur::blur_tile_direct;
pub use blur::{blur_tile, BlurScratch};
pub use tiles::{
    blur_matched_pairs, pair_zncc_readings, BlurMatchKernel, BlurMatchedPairs, PairReadings,
    TilePlanes, MIN_WINDOWED_SAMPLES,
};

/// `k` of the fitted mapping [`blur_sigma`]: `σ = k · s^q · (r² − 1)^p`.
pub const BLUR_MAP_SCALE: f64 = 0.9975;

/// `p` of the fitted mapping [`blur_sigma`], the power of `r² − 1`.
pub const BLUR_MAP_RATIO_POWER: f64 = 0.1744;

/// `q` of the fitted mapping [`blur_sigma`], the power of the sharper length.
pub const BLUR_MAP_LENGTH_POWER: f64 = 0.2613;

/// The shortest sharper length, in grid px, [`blur_sigma`] reads: shorter
/// lengths are read as this one. It is the shortest length among the samples
/// the mapping was fitted on.
pub const BLUR_MAP_MIN_LENGTH: f64 = 0.05;

/// The widest blur [`blur_sigma`] returns, in grid px. The ellipse's own
/// axes are capped at the self-similarity reading's `max_radius` (3 grid px by
/// default), and over that range the mapping stays under about 1.9, so the cap
/// only guards against a reading from a wider search.
pub const MAX_BLUR_SIGMA: f64 = 3.0;

/// `k`, `p` and `q` of the isotropic mapping [`isotropic_blur_sigma`], fitted
/// as [`blur_sigma`] was but on the semi-major axes and isotropic blurs.
pub const ISOTROPIC_BLUR_MAP: [f64; 3] = [0.8361, 0.3635, 0.5588];

/// The blur widths of the isotropic ladder ([`BlurMatchKernel::IsotropicLadder`]),
/// in grid px: `0.5 · √2ⁿ` for `n = 0 .. 5`.
pub const LADDER_SIGMAS: [f64; 6] = [
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

    fn add(self, other: Self) -> Self {
        Self {
            xx: self.xx + other.xx,
            xy: self.xy + other.xy,
            yy: self.yy + other.yy,
        }
    }
}

/// The blur each tile of a pair gets before the two are correlated.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PairBlur {
    /// The first tile's blur.
    pub a: BlurCovariance,
    /// The second tile's blur.
    pub b: BlurCovariance,
}

impl PairBlur {
    /// Whether neither tile is blurred: the pair is correlated plain.
    pub fn is_none(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }
}

/// The width of the 1-D Gaussian, in grid px, that blurs a tile whose
/// self-similarity ellipse is `sharper` long along a direction until it is
/// about `blurrier` long there: the fitted mapping
///
/// `σ = k · s^q · (r² − 1)^p`, `r = blurrier / s`, `s = max(sharper, `[`BLUR_MAP_MIN_LENGTH`]`)`,
///
/// with `k` [`BLUR_MAP_SCALE`], `p` [`BLUR_MAP_RATIO_POWER`] and `q`
/// [`BLUR_MAP_LENGTH_POWER`], capped at [`MAX_BLUR_SIGMA`]. `0` where
/// `blurrier ≤ s` or either length is not finite.
///
/// The mapping was fitted, not derived: tiles of the ground truths were blurred
/// by known anisotropic Gaussians and their ellipses read again, and the model
/// fitted in logs over the samples whose length along the blur changed
/// measurably. The response is far from the `σ² = blurrier² − sharper²` a
/// Gaussian model of the ellipse predicts: a blur under about half a grid px
/// does not register in the reading at all, so any measurable change in length
/// maps to a blur of at least about 0.5 grid px.
///
/// ```
/// use sfmtool_core::patch::blur_matched::blur_sigma;
///
/// assert_eq!(blur_sigma(0.5, 0.5), 0.0);
/// // Twice as long along a direction: a blur of about one grid px.
/// let s = blur_sigma(1.0, 0.5);
/// assert!(s > 0.95 && s < 1.05, "{s}");
/// ```
pub fn blur_sigma(blurrier: f64, sharper: f64) -> f64 {
    let [k, p, q] = [BLUR_MAP_SCALE, BLUR_MAP_RATIO_POWER, BLUR_MAP_LENGTH_POWER];
    fitted_sigma(blurrier, sharper, k, p, q)
}

/// [`blur_sigma`] with the isotropic mapping [`ISOTROPIC_BLUR_MAP`], read on
/// the two ellipses' semi-major axes.
pub fn isotropic_blur_sigma(blurrier: f64, sharper: f64) -> f64 {
    let [k, p, q] = ISOTROPIC_BLUR_MAP;
    fitted_sigma(blurrier, sharper, k, p, q)
}

fn fitted_sigma(blurrier: f64, sharper: f64, k: f64, p: f64, q: f64) -> f64 {
    if !(blurrier.is_finite() && sharper.is_finite()) {
        return 0.0;
    }
    let s = sharper.max(BLUR_MAP_MIN_LENGTH);
    let r = blurrier / s;
    // Two lengths equal but for rounding are equal.
    if r <= 1.0 + 1e-9 {
        return 0.0;
    }
    (k * s.powf(q) * (r * r - 1.0).powf(p)).min(MAX_BLUR_SIGMA)
}

/// The half-width of the ellipse `E` along the unit direction `u`:
/// `sqrt(uᵀ E u)`.
fn length_along(e: &[[f64; 2]; 2], u: [f64; 2]) -> f64 {
    (u[0] * u[0] * e[0][0] + 2.0 * u[0] * u[1] * e[0][1] + u[1] * u[1] * e[1][1])
        .max(0.0)
        .sqrt()
}

/// Whether an ellipse matrix can be read: every entry finite.
fn readable(e: &[[f64; 2]; 2]) -> bool {
    e.iter().flatten().all(|v| v.is_finite())
}

/// The blur each of two tiles gets, from their self-similarity ellipses `a`
/// and `b` (each `E`, [`SelfSimilarityEllipse::matrix`](crate::patch::self_similarity::SelfSimilarityEllipse::matrix),
/// in grid px²), blurring a direction only where the two lengths along it
/// differ by more than `min_ratio`.
///
/// Along each eigenvector `u` of `E_b − E_a`, the tile whose ellipse is
/// shorter along `u` is blurred along `u` by [`blur_sigma`] of the two
/// lengths. A pair with an ellipse that cannot be read is not blurred.
///
/// ```
/// use sfmtool_core::patch::blur_matched::pair_blur;
///
/// // `a` is sharp along x and as blurry as `b` along y; `b` is round.
/// let a = [[0.25, 0.0], [0.0, 1.0]];
/// let b = [[1.0, 0.0], [0.0, 1.0]];
/// let blur = pair_blur(&a, &b, 1.0);
/// assert!(blur.a.xx > 0.3 && blur.a.yy.abs() < 1e-9 && blur.b.xx == 0.0);
/// ```
pub fn pair_blur(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], min_ratio: f64) -> PairBlur {
    if !(readable(a) && readable(b)) {
        return PairBlur::default();
    }
    let d = [
        b[0][0] - a[0][0],
        0.5 * (b[0][1] + b[1][0] - a[0][1] - a[1][0]),
        b[1][1] - a[1][1],
    ];
    // The eigenvectors of a symmetric 2×2.
    let theta = 0.5 * (2.0 * d[1]).atan2(d[0] - d[2]);
    let (s, c) = theta.sin_cos();
    let mut blur = PairBlur::default();
    for u in [[c, s], [-s, c]] {
        let la = length_along(a, u);
        let lb = length_along(b, u);
        let (longer, shorter) = if la > lb { (la, lb) } else { (lb, la) };
        if longer <= shorter || longer < min_ratio * shorter.max(BLUR_MAP_MIN_LENGTH) {
            continue;
        }
        let sigma = blur_sigma(longer, shorter);
        if sigma <= 0.0 {
            continue;
        }
        let along = BlurCovariance::along(u, sigma);
        if lb > la {
            blur.a = blur.a.add(along);
        } else {
            blur.b = blur.b.add(along);
        }
    }
    blur
}

/// [`pair_blur`] under `matching`: no blur for [`PairMatching::Plain`].
pub fn pair_blur_for(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], matching: PairMatching) -> PairBlur {
    match matching.min_ratio() {
        Some(ratio) => pair_blur(a, b, ratio),
        None => PairBlur::default(),
    }
}

/// The ladder level, an index into [`LADDER_SIGMAS`], nearest to the isotropic
/// blur `sigma` in the logarithm, or `None` for a blur under 0.35 grid px,
/// which the ladder does not apply.
pub fn ladder_level(sigma: f64) -> Option<usize> {
    // A NaN width is no blur either.
    if sigma.is_nan() || sigma < 0.35 {
        return None;
    }
    let n = (2.0 * (sigma / LADDER_SIGMAS[0]).log2()).round();
    Some((n.max(0.0) as usize).min(LADDER_SIGMAS.len() - 1))
}
