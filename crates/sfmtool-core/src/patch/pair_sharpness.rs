// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which tile of a pair is the sharper, and to what length it is blurred
//! before the pair is correlated: the pairing rule the blur-matched readings
//! of the reference-view rule and of member coherence share.
//!
//! `specs/core/patch/blur-matched-zncc.md` § "Which tile is blurred, and to
//! what length" is the design. A tile is the sharper of a pair when its
//! self-similarity semi-major axis is shorter than the other tile's
//! semi-minor axis: it is sharper along every direction. That tile is blurred
//! isotropically until its semi-major axis reaches the other's semi-minor
//! axis, at most [`MAX_MATCHED_LENGTH`] ([`pair_blur`]), so the blur takes no
//! direction past what the blurrier tile shows along its sharpest one. Where
//! neither tile is the sharper in that sense, as where one tile is blurry
//! along one direction only or where the two hold a texture with a grain at
//! different angles, the pair is correlated plain. Where the target is less
//! than a ratio times the semi-major axis, the pair is correlated plain as
//! well ([`PairMatching`]).
//!
//! The blur itself is done to one tile, by
//! [`blur_matched`](crate::patch::blur_matched): each view whose tile some
//! pair blurs is assessed once ([`TrackBlurs`]), and every pair it is the
//! sharper of blurs it by the width its assessment gives.

use crate::patch::blur_matched::{semi_axes, BlurAssessment};

/// The shortest semi-major axis, in grid px, the skip ratio reads: a shorter
/// one is read as this one.
pub const MIN_SHARPER_LENGTH: f64 = 0.05;

/// The longest semi-major axis, in grid px, blur matching aims for: a
/// blurrier tile's semi-minor axis past it is read as this one. Past about
/// 2 grid px a tile holds little detail to tell it from another surface's,
/// and blurring members further towards smooth tiles of other points lowered
/// how well wrong views are told apart.
pub const MAX_MATCHED_LENGTH: f64 = 2.0;

/// The fraction by which the target must at least exceed the sharper tile's
/// semi-major axis for the pair to be blurred at all, whatever the ratio
/// asked for: a smaller difference is within the spread of the width the
/// blur assessment gives.
pub const MATCHED_LENGTH_TOLERANCE: f64 = 0.05;

/// The default ratio of [`PairMatching::BlurMatchedAboveRatio`]: the target
/// must be at least this factor times the sharper tile's semi-major axis for
/// the pair to be blurred.
pub const DEFAULT_MIN_ELLIPSE_RATIO: f64 = 1.25;

/// How a pair of views' tiles is correlated: as they are, or blur-matched.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PairMatching {
    /// The tiles as they are.
    #[default]
    Plain,
    /// The sharper tile blurred wherever [`pair_blur`] finds one, the target
    /// at least `1 +` [`MATCHED_LENGTH_TOLERANCE`] times its semi-major axis.
    BlurMatched,
    /// The sharper tile blurred where the target is at least the given factor
    /// (`> 1`) times its semi-major axis; a pair that differs by less is
    /// correlated plain, which costs nothing beyond the plain ZNCC.
    BlurMatchedAboveRatio(f64),
}

impl PairMatching {
    /// The least ratio of the target to the sharper semi-major axis at which
    /// the pair is blurred, or `None` for [`PairMatching::Plain`]. `1` for
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
/// A tile is blurred when the other tile's semi-minor axis, the target, is at
/// least `min_ratio` times its semi-major axis (and at least `1 +`
/// [`MATCHED_LENGTH_TOLERANCE`] times it). The target is at most
/// [`MAX_MATCHED_LENGTH`]. At most one tile of a pair can be shorter in that
/// way. A pair with an ellipse that cannot be read is not blurred.
///
/// ```
/// use sfmtool_core::patch::pair_sharpness::pair_blur;
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

/// Whether an ellipse matrix can be read: every entry finite.
fn readable(e: &[[f64; 2]; 2]) -> bool {
    e.iter().flatten().all(|v| v.is_finite())
}

/// The view of a pair to blur, its blur assessment, and the semi-major axis
/// to blur it to ([`TrackBlurs::pair`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairTarget<'a> {
    /// The view to blur, an index into the track's views.
    pub view: usize,
    /// Its blur assessment.
    pub assessment: &'a BlurAssessment,
    /// The semi-major axis to blur it to ([`PairBlur::target`]).
    pub target: f64,
}

/// A track's blur matching: which view of each pair is blurred, and to what
/// length ([`pair_blur`]), with the blur assessment of every view some pair
/// blurs, each read once.
#[derive(Debug, Clone)]
pub struct TrackBlurs {
    ellipses: Vec<Option<[[f64; 2]; 2]>>,
    min_ratio: Option<f64>,
    assessments: Vec<Option<BlurAssessment>>,
    assessed: usize,
}

impl TrackBlurs {
    /// The pairing of the views whose self-similarity ellipse matrices are
    /// `ellipses` (`None` where a view has none, which leaves every pair it
    /// is in plain), under `matching`. `assess(v, e)` is called once, in view
    /// order, for each view `v` that [`pair_blur`] names the sharper of some
    /// pair, `e` its ellipse, and gives the view's blur assessment
    /// ([`assess_blur`](crate::patch::blur_matched::assess_blur)) or `None`,
    /// which leaves the view's pairs plain.
    ///
    /// Which views are assessed depends on the ellipses alone, and each
    /// assessment on its own view alone, so a pair is blurred the same
    /// whatever other views are in the track and whatever order they come
    /// in.
    pub fn assess(
        ellipses: &[Option<[[f64; 2]; 2]>],
        matching: PairMatching,
        mut assess: impl FnMut(usize, &[[f64; 2]; 2]) -> Option<BlurAssessment>,
    ) -> Self {
        let k = ellipses.len();
        let min_ratio = matching.min_ratio();
        let mut wanted = vec![false; k];
        if let Some(ratio) = min_ratio {
            for a in 0..k {
                for b in (a + 1)..k {
                    if let (Some(ea), Some(eb)) = (&ellipses[a], &ellipses[b]) {
                        if let Some(pb) = pair_blur(ea, eb, ratio) {
                            wanted[if pb.sharper == 0 { a } else { b }] = true;
                        }
                    }
                }
            }
        }
        let mut assessed = 0;
        let assessments = (0..k)
            .map(|v| {
                let e = ellipses[v].as_ref().filter(|_| wanted[v])?;
                assessed += 1;
                assess(v, e)
            })
            .collect();
        Self {
            ellipses: ellipses.to_vec(),
            min_ratio,
            assessments,
            assessed,
        }
    }

    /// The view of the pair `a`, `b` to blur, and to what length; `None`
    /// where the pair is correlated plain: under [`PairMatching::Plain`],
    /// where [`pair_blur`] finds no sharper tile, or where the sharper view
    /// has no assessment.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` is not a view of the track.
    pub fn pair(&self, a: usize, b: usize) -> Option<PairTarget<'_>> {
        let pb = pair_blur(
            self.ellipses[a].as_ref()?,
            self.ellipses[b].as_ref()?,
            self.min_ratio?,
        )?;
        let view = if pb.sharper == 0 { a } else { b };
        Some(PairTarget {
            view,
            assessment: self.assessments[view].as_ref()?,
            target: pb.target,
        })
    }

    /// View `v`'s blur assessment, `None` where it was not assessed or its
    /// assessment could not be read.
    pub fn assessment(&self, v: usize) -> Option<&BlurAssessment> {
        self.assessments[v].as_ref()
    }

    /// How many views were assessed.
    pub fn assessed(&self) -> usize {
        self.assessed
    }
}

#[cfg(test)]
mod tests;
