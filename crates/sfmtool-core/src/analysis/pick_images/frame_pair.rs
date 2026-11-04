// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One frame's features with its forest, and the statistics of a pair of
//! frames: the step, the fundamental-matrix inlier share, the share of those
//! inliers that fit one homography, and that homography's distance from a
//! pure rotation. See `specs/core/analysis/frame-pair-matching.md`.
//!
//! Precision: positions stay `f32`, as a `.sift` file stores them, and
//! descriptors `u8`. They are widened to `f64` once, per match, because the
//! two RANSAC kernels and [`crate::numeric::median`] take `f64`; widening an
//! `f32` is exact, and the per-match arithmetic is a few thousand values per
//! pair. The forest's squared descriptor distances are `f32`, and exact: the
//! largest, `128 · 255²`, is below `2²⁴`.

use nalgebra::Matrix3;
use sfmtool_sift_format::{SiftFeatures, SiftMetadata, DESCRIPTOR_DIM};

use crate::features::kdforest::{KdForestParams, KdForestU8};
use crate::geometry::epipolar_estimation::{estimate_fundamental, FundamentalOptions};
use crate::geometry::homography_estimation::{estimate_homography, HomographyOptions};
use crate::numeric::median;
use crate::progress::Progress;

#[cfg(test)]
mod tests;

/// The long side, in pixels, that the inlier bounds of [`FramePairOptions`]
/// are stated against.
const REFERENCE_LONG_SIDE_PX: f64 = 960.0;

/// The focal lengths [`FramePairStats::rotation_error`] tries, as multiples of
/// the image's long side: from the first to the second, inclusive.
const ROTATION_FOCAL_RANGE: (f64, f64) = (0.3, 3.0);

/// How many focal lengths [`FramePairStats::rotation_error`] tries, spaced
/// evenly in `log f`.
const ROTATION_FOCAL_STEPS: usize = 80;

/// The fewest correspondences a fundamental matrix is fitted from (the
/// seven-point solver's sample), and so the least `min_matches` may be.
const FUNDAMENTAL_SAMPLE: usize = 7;

/// One frame's features, with the forest over its descriptors built once, so
/// that a frame matched with several others is indexed once.
pub struct FrameFeatures {
    positions_xy: Vec<[f32; 2]>,
    forest: KdForestU8,
    width: u32,
    height: u32,
}

impl std::fmt::Debug for FrameFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameFeatures")
            .field("len", &self.positions_xy.len())
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl FrameFeatures {
    /// One frame's features: `positions_xy` in pixels, `descriptors`
    /// `positions_xy.len() × 128` bytes, row-major, as a `.sift` file holds
    /// them, and the image's size. Builds the forest over the descriptors
    /// with `forest`.
    ///
    /// Takes no `Progress`: one frame's forest is milliseconds of work (about
    /// 3 ms for 8192 features and 50 ms for 100,000 on 32 cores), so a caller
    /// walking many frames reports and checks for cancellation between them.
    ///
    /// # Errors
    ///
    /// [`FramePairError::Options`] for a forest parameter of 0,
    /// [`FramePairError::FeatureArrays`] when `descriptors` is not
    /// `128 × positions_xy.len()` bytes, a position is not finite or there are
    /// more features than a forest indexes (`u32::MAX`), and
    /// [`FramePairError::ImageSize`] for a width or height of 0.
    pub fn new(
        positions_xy: Vec<[f32; 2]>,
        descriptors: Vec<u8>,
        width: u32,
        height: u32,
        forest: KdForestParams,
    ) -> Result<Self, FramePairError> {
        for (name, value) in [
            ("forest.num_trees", forest.num_trees),
            ("forest.split_dim_candidates", forest.split_dim_candidates),
            ("forest.leaf_size", forest.leaf_size),
        ] {
            if value == 0 {
                return Err(FramePairError::Options {
                    name,
                    reason: "must be positive".to_string(),
                });
            }
        }
        if width == 0 || height == 0 {
            return Err(FramePairError::ImageSize { width, height });
        }
        let n = positions_xy.len();
        if n > u32::MAX as usize {
            return Err(FramePairError::FeatureArrays {
                reason: format!("{n} features are more than a forest indexes ({})", u32::MAX),
            });
        }
        if n.checked_mul(DESCRIPTOR_DIM) != Some(descriptors.len()) {
            return Err(FramePairError::FeatureArrays {
                reason: format!(
                    "descriptors has {} bytes but {n} features of {DESCRIPTOR_DIM} bytes \
                     each need {}",
                    descriptors.len(),
                    n.saturating_mul(DESCRIPTOR_DIM)
                ),
            });
        }
        if let Some(i) = positions_xy
            .iter()
            .position(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err(FramePairError::FeatureArrays {
                reason: format!(
                    "position {i} is not finite: ({}, {})",
                    positions_xy[i][0], positions_xy[i][1]
                ),
            });
        }
        let forest = KdForestU8::build(&descriptors, n, DESCRIPTOR_DIM, forest, &Progress::none())
            .expect("a build behind Progress::none is never cancelled");
        Ok(Self {
            positions_xy,
            forest,
            width,
            height,
        })
    }

    /// The features of a `.sift` file read by `read_sift_features`, with the
    /// image size from its `read_sift_metadata`.
    ///
    /// # Errors
    ///
    /// As [`FrameFeatures::new`], and [`FramePairError::FeatureArrays`] when
    /// `features` holds a different number of features than `metadata`
    /// records, which means the two were read from different files.
    pub fn from_sift(
        features: SiftFeatures,
        metadata: &SiftMetadata,
        forest: KdForestParams,
    ) -> Result<Self, FramePairError> {
        let n = features.positions_xy.len();
        if n != metadata.feature_count as usize {
            return Err(FramePairError::FeatureArrays {
                reason: format!(
                    "the features hold {n} features but the metadata records {}",
                    metadata.feature_count
                ),
            });
        }
        Self::new(
            features.positions_xy,
            features.descriptors,
            metadata.image_width,
            metadata.image_height,
            forest,
        )
    }

    /// The number of features.
    pub fn len(&self) -> usize {
        self.positions_xy.len()
    }

    /// Whether the frame has no features.
    pub fn is_empty(&self) -> bool {
        self.positions_xy.is_empty()
    }

    /// The image's width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The image's height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// How [`match_frame_pair`] matches and what it fits.
#[derive(Clone, Debug, PartialEq)]
pub struct FramePairOptions {
    /// Lowe's ratio on descriptor distances: a feature's nearest neighbour is
    /// kept when it is closer than `ratio` times the second nearest. In
    /// `(0, 1]`.
    pub ratio: f32,
    /// The fewest mutual matches the shares are computed from; fewer gives
    /// `None` shares. At least 7, the sample a fundamental matrix is fitted
    /// from.
    pub min_matches: usize,
    /// The per-query budget of distance computations, or `None` for the
    /// queried forest's own `max_leaf_checks`.
    pub max_leaf_checks: Option<usize>,
    /// The fundamental matrix's inlier bound on the Sampson distance, in
    /// pixels per 960 pixels of the image's long side.
    pub fundamental_error_per_960px: f64,
    /// The homography's inlier bound on the symmetric transfer error, in
    /// pixels per 960 pixels of the image's long side.
    pub homography_error_per_960px: f64,
    /// The seed of both RANSAC samplers, so a pair's statistics repeat.
    pub seed: u64,
}

impl Default for FramePairOptions {
    fn default() -> Self {
        Self {
            ratio: 0.8,
            min_matches: 30,
            max_leaf_checks: None,
            fundamental_error_per_960px: 1.0,
            homography_error_per_960px: 2.0,
            seed: 0,
        }
    }
}

impl FramePairOptions {
    fn validate(&self) -> Result<(), FramePairError> {
        if !(self.ratio > 0.0 && self.ratio <= 1.0) {
            return Err(FramePairError::Options {
                name: "ratio",
                reason: format!("must be in (0, 1], got {}", self.ratio),
            });
        }
        if self.min_matches < FUNDAMENTAL_SAMPLE {
            return Err(FramePairError::Options {
                name: "min_matches",
                reason: format!(
                    "must be at least {FUNDAMENTAL_SAMPLE}, the matches a fundamental matrix \
                     is fitted from, got {}",
                    self.min_matches
                ),
            });
        }
        for (name, value) in [
            (
                "fundamental_error_per_960px",
                self.fundamental_error_per_960px,
            ),
            (
                "homography_error_per_960px",
                self.homography_error_per_960px,
            ),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(FramePairError::Options {
                    name,
                    reason: format!("must be positive and finite, got {value}"),
                });
            }
        }
        Ok(())
    }
}

/// Why [`FrameFeatures::new`] or [`match_frame_pair`] refused its input.
#[derive(Clone, Debug, PartialEq)]
pub enum FramePairError {
    /// An option out of range: a ratio outside `(0, 1]`, a `min_matches`
    /// below 7, an error bound that is not positive and finite, or a forest
    /// parameter of 0, which the forest's build asserts against. `name` is
    /// the field.
    Options { name: &'static str, reason: String },
    /// The descriptors are not `128 × positions_xy.len()` bytes, a position is
    /// not finite, there are more features than a forest indexes, or a
    /// `.sift` file's features and metadata disagree on the count.
    FeatureArrays { reason: String },
    /// A width or height of 0.
    ImageSize { width: u32, height: u32 },
    /// The two frames' images differ in size, so one step has no single long
    /// side to be measured against.
    SizeMismatch { a: (u32, u32), b: (u32, u32) },
}

impl std::fmt::Display for FramePairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Options { name, reason } => write!(f, "{name} {reason}"),
            Self::FeatureArrays { reason } => write!(f, "{reason}"),
            Self::ImageSize { width, height } => write!(
                f,
                "the image size {width} x {height} has a side of 0 pixels"
            ),
            Self::SizeMismatch { a, b } => write!(
                f,
                "the two frames differ in size, {} x {} and {} x {}, so a step between them \
                 has no single long side to be measured against",
                a.0, a.1, b.0, b.1
            ),
        }
    }
}

impl std::error::Error for FramePairError {}

/// One pair's statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct FramePairStats {
    /// The number of mutual matches.
    pub matches: usize,
    /// The median length of the mutual matches' displacements, in percent of
    /// the image's long side. NaN when there are no matches.
    pub step_percent: f64,
    /// The share of the matches that are inliers of one fundamental matrix,
    /// 0 when no fundamental matrix fits them. `None` under `min_matches`.
    pub f_inlier_share: Option<f64>,
    /// The share of the fundamental matrix's inliers that also fit one
    /// homography, 0 when no homography fits them. `None` under
    /// `min_matches` matches or `min_matches` inliers.
    pub h_over_f: Option<f64>,
    /// How far that homography is from a pure rotation: the smallest
    /// `‖M Mᵀ − I‖` (Frobenius) over the tried focal lengths, with
    /// `M = K⁻¹ H K` scaled to determinant 1. `None` where `h_over_f` is
    /// `None`, where no homography fits, or where no tried focal length gives
    /// a finite `M` with a non-zero determinant.
    pub rotation_error: Option<f64>,
}

/// Match two frames and measure the motion between them.
///
/// The matches are the mutual best matches of two `k = 2` queries, each
/// frame's descriptors against the other's forest, that pass the ratio test
/// both ways. The fundamental matrix is fitted to all of them, and the
/// homography to its inliers, by LO-RANSAC with `options.seed`; a homography
/// fitted to all of them at the fundamental matrix's bound stands in for the
/// fundamental matrix when it has more inliers, since matches that fit one
/// homography fit a whole family of fundamental matrices, which the
/// seven-point solver cannot single out.
///
/// # Errors
///
/// [`FramePairError::Options`] for an option out of range and
/// [`FramePairError::SizeMismatch`] when the frames' images differ in size.
pub fn match_frame_pair(
    a: &FrameFeatures,
    b: &FrameFeatures,
    options: &FramePairOptions,
) -> Result<FramePairStats, FramePairError> {
    options.validate()?;
    if (a.width, a.height) != (b.width, b.height) {
        return Err(FramePairError::SizeMismatch {
            a: (a.width, a.height),
            b: (b.width, b.height),
        });
    }
    let pairs = mutual_matches(a, b, options);
    let long_side = f64::from(a.width.max(a.height));

    let x1: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(i, _)| widen(a.positions_xy[i]))
        .collect();
    let x2: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(_, j)| widen(b.positions_xy[j]))
        .collect();
    let lengths: Vec<f64> = x1
        .iter()
        .zip(&x2)
        .map(|(p, q)| (q[0] - p[0]).hypot(q[1] - p[1]))
        .collect();
    let step_percent = 100.0 * median(&lengths) / long_side;

    let mut stats = FramePairStats {
        matches: pairs.len(),
        step_percent,
        f_inlier_share: None,
        h_over_f: None,
        rotation_error: None,
    };
    if pairs.len() < options.min_matches {
        return Ok(stats);
    }

    let scale = long_side / REFERENCE_LONG_SIDE_PX;
    let fundamental_error_px = options.fundamental_error_per_960px * scale;
    let f_inliers = fundamental_consensus(&x1, &x2, fundamental_error_px, options.seed);
    let (y1, y2): (Vec<[f64; 2]>, Vec<[f64; 2]>) = f_inliers
        .iter()
        .zip(x1.iter().zip(&x2))
        .filter(|(&inlier, _)| inlier)
        .map(|(_, (&p, &q))| (p, q))
        .unzip();
    stats.f_inlier_share = Some(y1.len() as f64 / pairs.len() as f64);
    if y1.len() < options.min_matches {
        return Ok(stats);
    }

    let homography = estimate_homography(
        &y1,
        &y2,
        &HomographyOptions {
            max_error_px: options.homography_error_per_960px * scale,
            seed: options.seed,
            ..HomographyOptions::default()
        },
    );
    let Some(homography) = homography else {
        stats.h_over_f = Some(0.0);
        return Ok(stats);
    };
    let h_inliers = homography.inliers.iter().filter(|&&inlier| inlier).count();
    stats.h_over_f = Some(h_inliers as f64 / y1.len() as f64);
    stats.rotation_error = rotation_error(&homography.h_matrix, a.width, a.height);
    Ok(stats)
}

/// The inliers of the fundamental matrix with the largest consensus.
///
/// Matches that all fit one homography fit every fundamental matrix
/// `[e']ₓ H`, and the seven-point solver finds none of them: its sample has
/// fewer than seven independent constraints. A homography fitted at the same
/// bound is therefore a fundamental matrix's consensus too, and the larger of
/// the two inlier sets is the one returned, the fundamental matrix's on a
/// tie. All `false` when neither fits.
fn fundamental_consensus(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    max_error_px: f64,
    seed: u64,
) -> Vec<bool> {
    let fundamental = estimate_fundamental(
        x1,
        x2,
        &FundamentalOptions {
            max_error_px,
            // A minimal sample fits its own seven matches exactly, so with
            // this floor the share reports the best consensus however small.
            min_inliers: FUNDAMENTAL_SAMPLE,
            seed,
            ..FundamentalOptions::default()
        },
    )
    .map(|estimate| estimate.inliers);
    let homography = estimate_homography(
        x1,
        x2,
        &HomographyOptions {
            max_error_px,
            seed,
            ..HomographyOptions::default()
        },
    )
    .map(|estimate| estimate.inliers);
    let count = |mask: &Option<Vec<bool>>| {
        mask.as_ref()
            .map_or(0, |mask| mask.iter().filter(|&&inlier| inlier).count())
    };
    let best = if count(&homography) > count(&fundamental) {
        homography
    } else {
        fundamental
    };
    best.unwrap_or_else(|| vec![false; x1.len()])
}

fn widen(p: [f32; 2]) -> [f64; 2] {
    [f64::from(p[0]), f64::from(p[1])]
}

/// The mutual best matches `(index in a, index in b)` that pass the ratio
/// test both ways, in the order of `a`'s features.
fn mutual_matches(
    a: &FrameFeatures,
    b: &FrameFeatures,
    options: &FramePairOptions,
) -> Vec<(usize, usize)> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let a_to_b = best_passing(a, b, options);
    let b_to_a = best_passing(b, a, options);
    a_to_b
        .iter()
        .enumerate()
        .filter_map(|(i, &j)| {
            let j = j?;
            (b_to_a[j] == Some(i)).then_some((i, j))
        })
        .collect()
}

/// For each feature of `from`, its nearest neighbour in `to`'s forest when
/// that neighbour passes the ratio test, comparing squared distances against
/// `ratio²`. A feature whose query found only one neighbour has nothing to be
/// compared with and keeps it.
fn best_passing(
    from: &FrameFeatures,
    to: &FrameFeatures,
    options: &FramePairOptions,
) -> Vec<Option<usize>> {
    let checks = options
        .max_leaf_checks
        .unwrap_or(to.forest.params().max_leaf_checks);
    let (indices, distances_sq) =
        to.forest
            .search_batch_with_distances(from.forest.points(), from.len(), 2, checks, None);
    let ratio_sq = options.ratio * options.ratio;
    indices
        .as_chunks::<2>()
        .0
        .iter()
        .zip(distances_sq.as_chunks::<2>().0)
        .map(|(index, d)| {
            let passes = index[0] != u32::MAX && d[0] < ratio_sq * d[1];
            passes.then_some(index[0] as usize)
        })
        .collect()
}

/// The smallest `‖M Mᵀ − I‖` over the tried focal lengths, with `K` the
/// camera matrix of focal `f` and the principal point at the image centre,
/// and `M = K⁻¹ H K` scaled to determinant 1.
fn rotation_error(h: &Matrix3<f64>, width: u32, height: u32) -> Option<f64> {
    let long_side = f64::from(width.max(height));
    let (cx, cy) = (f64::from(width) / 2.0, f64::from(height) / 2.0);
    let (low, high) = ROTATION_FOCAL_RANGE;
    let log_step = (high / low).ln() / (ROTATION_FOCAL_STEPS - 1) as f64;
    let mut best = f64::INFINITY;
    for k in 0..ROTATION_FOCAL_STEPS {
        let f = long_side * low * (k as f64 * log_step).exp();
        let k_mat = Matrix3::new(f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0);
        let k_inv = Matrix3::new(1.0 / f, 0.0, -cx / f, 0.0, 1.0 / f, -cy / f, 0.0, 0.0, 1.0);
        let m = k_inv * h * k_mat;
        let det = m.determinant();
        if !det.is_finite() || det == 0.0 {
            continue;
        }
        let m = m / det.cbrt();
        let error = (m * m.transpose() - Matrix3::identity()).norm();
        if error < best {
            best = error;
        }
    }
    best.is_finite().then_some(best)
}
