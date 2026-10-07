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
//! The width of each blur comes from how the tile's own ellipse grows when
//! the tile is blurred ([`BlurGrowth`]). Each view's tile is blurred
//! isotropically by each of [`GROWTH_PROBE_SIGMAS`], and each blurred tile's
//! ellipse read with the reading the view's own ellipse came from
//! ([`read_growth`]); every pair the view is blurred in then reads the width
//! off those readings ([`BlurDirection::sigma`]) without reading again. Where
//! a view's growth cannot be read, a rate fitted on real tiles stands in
//! ([`estimated_blur_sigma`]).
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
mod growth;
mod tiles;

#[cfg(test)]
mod tests;

#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use blur::blur_tile_direct;
pub use blur::{blur_tile, BlurScratch};
pub use growth::{read_growth, ViewGrowths};
pub use tiles::{
    blur_matched_pairs, pair_zncc_readings, read_tile_ellipse, BlurMatchKernel, BlurMatchedPairs,
    PairReadings, TilePlanes, MIN_WINDOWED_SAMPLES,
};

/// The widths, in grid px, of the isotropic blurs each view's tile is
/// blurred by, once each, to read how its ellipse grows ([`read_growth`]).
/// Most pairs need a blur of 0.3 to 1 grid px; the narrower probe reads the
/// growth there, the wider one how it bends further out.
pub const GROWTH_PROBE_SIGMAS: [f64; 2] = [0.4, 1.0];

/// `k` of [`estimated_blur_sigma`]'s rate `k · s^p`: how fast the square of
/// a tile's ellipse length along a direction grows with the square of a blur
/// along it, `d(l²)/d(σ²)`, for a sharper length `s` of 1 grid px, fitted on
/// real tiles.
pub const BLUR_GROWTH_SCALE: f64 = 1.243;

/// `p` of [`estimated_blur_sigma`]'s rate `k · s^p`: the power of the
/// sharper length.
pub const BLUR_GROWTH_POWER: f64 = 1.247;

/// The shortest sharper length, in grid px, the skip ratio and the width
/// read: shorter lengths are read as this one.
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

/// The fraction by which two lengths along a direction must differ for the
/// direction to be blurred at all, whatever the ratio asked for: a smaller
/// difference is within the spread of the width the growth gives.
pub const MATCHED_LENGTH_TOLERANCE: f64 = 0.05;

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
    /// ellipses' lengths differ by more than [`MATCHED_LENGTH_TOLERANCE`].
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
    /// [`PairMatching::BlurMatched`]; [`pair_directions`] blurs no direction
    /// whose lengths are within [`MATCHED_LENGTH_TOLERANCE`] of each other,
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

/// How one view's self-similarity ellipse grows when its tile is blurred:
/// the tile's ellipse and the ellipses of the tile blurred isotropically by
/// each of [`GROWTH_PROBE_SIGMAS`], all read the same way ([`read_growth`]).
///
/// Along a unit direction `u`, the square of the length `uᵀ E u` against `σ²`
/// is taken to be piecewise linear through the readings, from no blur to the
/// widest probe, and to go on past the widest probe along its last piece. The
/// width that brings the length to a target is read back off that line
/// ([`BlurGrowth::sigma_along`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurGrowth {
    /// The unblurred tile's ellipse, then the probes' in the order of
    /// [`GROWTH_PROBE_SIGMAS`].
    pub ellipses: [[[f64; 2]; 2]; GROWTH_PROBE_SIGMAS.len() + 1],
}

impl BlurGrowth {
    /// The growth from a tile's ellipse `unblurred` and the ellipses `probed`
    /// of the same tile blurred by each of [`GROWTH_PROBE_SIGMAS`].
    pub fn from_ellipses(
        unblurred: &[[f64; 2]; 2],
        probed: &[[[f64; 2]; 2]; GROWTH_PROBE_SIGMAS.len()],
    ) -> Self {
        let mut ellipses = [*unblurred; GROWTH_PROBE_SIGMAS.len() + 1];
        ellipses[1..].copy_from_slice(probed);
        Self { ellipses }
    }

    /// The width that brings the square of the length along the unit
    /// direction `u` to `target²`, capped at [`MAX_BLUR_SIGMA`]; `0` where the
    /// unblurred length is already there, `None` where the readings do not
    /// grow along `u`.
    pub fn sigma_along(&self, u: [f64; 2], target: f64) -> Option<f64> {
        let l2 = self.ellipses.map(|e| length_along(&e, u).powi(2));
        sigma_on_curve(&l2, target)
    }

    /// [`Self::sigma_along`] for the semi-major axis, which the isotropic
    /// ladder reads.
    pub fn sigma_semi_major(&self, target: f64) -> Option<f64> {
        let l2 = self.ellipses.map(|e| semi_major(&e).powi(2));
        sigma_on_curve(&l2, target)
    }

    /// The semi-major axis the tile is expected to have once blurred
    /// isotropically by `sigma`, read off the same line.
    pub fn semi_major_after(&self, sigma: f64) -> f64 {
        let l2 = self.ellipses.map(|e| semi_major(&e).powi(2));
        let s2 = sigma * sigma;
        let mut prev = (0.0, l2[0]);
        for (i, &w) in GROWTH_PROBE_SIGMAS.iter().enumerate() {
            let cur = (w * w, l2[i + 1]);
            if s2 <= cur.0 || i + 1 == GROWTH_PROBE_SIGMAS.len() {
                let slope = (cur.1 - prev.1) / (cur.0 - prev.0);
                return (prev.1 + slope.max(0.0) * (s2 - prev.0)).max(0.0).sqrt();
            }
            prev = cur;
        }
        l2[0].max(0.0).sqrt()
    }
}

/// The `σ` at which the line through `(0, l2[0])` and `(w², l2[i + 1])` for
/// each probe width `w` reaches `target²`, the line going on past the widest
/// probe along its last piece; a reading shorter than the one before it is
/// passed over. `None` where no piece grows.
fn sigma_on_curve(l2: &[f64; GROWTH_PROBE_SIGMAS.len() + 1], target: f64) -> Option<f64> {
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

/// The semi-major axis of the ellipse `E`: the square root of its larger
/// eigenvalue.
pub(crate) fn semi_major(e: &[[f64; 2]; 2]) -> f64 {
    let (a, b, d) = (e[0][0], 0.5 * (e[0][1] + e[1][0]), e[1][1]);
    (0.5 * (a + d + ((a - d) * (a - d) + 4.0 * b * b).sqrt()))
        .max(0.0)
        .sqrt()
}

/// The rate `k · s^p` fitted on real tiles, at the sharper length `sharper`.
pub(crate) fn fitted_rate(sharper: f64) -> f64 {
    BLUR_GROWTH_SCALE * sharper.max(MIN_SHARPER_LENGTH).powf(BLUR_GROWTH_POWER)
}

/// The width of the 1-D Gaussian, in grid px, expected to blur a tile whose
/// self-similarity ellipse is `sharper` long along a direction until it is
/// `blurrier` long there, from the rate fitted on real tiles: the width used
/// where a view's own growth could not be read, and the widest a width read
/// past the widest probe may be (`BlurDirection::sigma`).
///
/// The square of the length grows about linearly with the square of the
/// blur, at a rate that rises with the sharper length:
///
/// `σ² = (blurrier² − s²) / (k · s^p)`, `s = max(sharper, `[`MIN_SHARPER_LENGTH`]`)`,
///
/// with `k` [`BLUR_GROWTH_SCALE`] and `p` [`BLUR_GROWTH_POWER`], capped at
/// [`MAX_BLUR_SIGMA`]. `0` where `blurrier ≤ s` or either length is not
/// finite.
///
/// The rate rises with the sharper length, because a longer ellipse comes
/// from a fainter or smoother texture. The constants were fitted on real
/// tiles of ten datasets; a single tile's rate is spread about the fit, its
/// quartiles 1.8 times below and 1.5 times above it, which is why each view's
/// own growth is read instead where it can be.
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
    ((blurrier * blurrier - s * s) / fitted_rate(sharper))
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

impl BlurDirection {
    /// The width of the blur along `u`: the width at which `growth`'s
    /// readings reach the blurrier length ([`BlurGrowth::sigma_along`]), or
    /// [`estimated_blur_sigma`] where there is no growth or it does not grow
    /// along `u`.
    ///
    /// A width past the widest probe is extrapolated, and a tile whose
    /// ellipse grows faster the wider the blur would be blurred too far by
    /// it, so there the width is at most [`estimated_blur_sigma`]'s (and no
    /// less than the widest probe).
    pub fn sigma(&self, growth: Option<&BlurGrowth>) -> f64 {
        let fitted = estimated_blur_sigma(self.blurrier, self.sharper);
        let widest = GROWTH_PROBE_SIGMAS[GROWTH_PROBE_SIGMAS.len() - 1];
        match growth.and_then(|g| g.sigma_along(self.u, self.blurrier)) {
            Some(sigma) if sigma > widest => sigma.min(fitted.max(widest)),
            Some(sigma) => sigma,
            None => fitted,
        }
    }
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

    /// The blur along every direction, each by [`BlurDirection::sigma`] at the
    /// tile's `growth`, added up.
    pub fn blur(&self, growth: Option<&BlurGrowth>) -> BlurCovariance {
        self.as_slice().iter().fold(BlurCovariance::ZERO, |cov, d| {
            cov.add(BlurCovariance::along(d.u, d.sigma(growth)))
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
/// differ by less than `min_ratio`, or by no more than
/// [`MATCHED_LENGTH_TOLERANCE`].
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
    let ratio = min_ratio.max(1.0 + MATCHED_LENGTH_TOLERANCE);
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
        if longer <= shorter || longer < ratio * shorter.max(MIN_SHARPER_LENGTH) {
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
