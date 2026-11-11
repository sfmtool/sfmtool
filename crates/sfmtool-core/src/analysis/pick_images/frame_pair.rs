// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One frame's features with its forest, and the statistics of a pair of
//! frames: the number of mutual matches, the step and, given a camera, the
//! essential split. See
//! `specs/core/analysis/frame-pair-matching.md`.
//!
//! Precision: positions stay `f32`, as a `.sift` file stores them, and
//! descriptors `u8`. A match's displacement is computed in `f64`, from the
//! two positions widened, because [`crate::numeric::median`] takes `f64`;
//! widening an `f32` is exact, and the per-match arithmetic is a few thousand
//! values per pair. The forest's squared descriptor distances are `f32`, and
//! exact: the largest, `128 · 255²`, is below `2²⁴`.

use sfmtool_sift_format::{SiftFeatures, SiftMetadata, DESCRIPTOR_DIM};

use super::essential_split::{
    measure_essential_split, EssentialSplitOptions, EssentialSplitOutcome,
};
use crate::features::kdforest::{KdForestParams, KdForestU8};
use crate::numeric::median;
use crate::progress::Progress;

#[cfg(test)]
mod tests;

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
        validate_forest(&forest)?;
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

/// How [`match_frame_pair`] matches.
#[derive(Clone, Debug, PartialEq)]
pub struct FramePairOptions {
    /// Lowe's ratio on descriptor distances: a feature's nearest neighbour is
    /// kept when it is closer than `ratio` times the second nearest. In
    /// `(0, 1]`.
    pub ratio: f32,
    /// The per-query budget of distance computations, or `None` for the
    /// queried forest's own `max_leaf_checks`.
    pub max_leaf_checks: Option<usize>,
    /// With a camera, also measure each pair's essential split
    /// ([`measure_essential_split`]); `None` measures nothing more.
    pub essential_split: Option<EssentialSplitOptions>,
}

impl Default for FramePairOptions {
    fn default() -> Self {
        Self {
            ratio: 0.8,
            max_leaf_checks: None,
            essential_split: None,
        }
    }
}

/// The forest parameters `KdForestU8::build` asserts against: a tree count,
/// split candidate count or leaf size of 0. [`FrameFeatures::new`] checks them
/// before it builds, and a caller that builds many frames can check them once
/// first.
pub(super) fn validate_forest(forest: &KdForestParams) -> Result<(), FramePairError> {
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
    Ok(())
}

impl FramePairOptions {
    /// The checks [`match_frame_pair`] makes on the options, for a caller that
    /// checks them once before matching many pairs.
    pub(super) fn validate(&self) -> Result<(), FramePairError> {
        if !(self.ratio > 0.0 && self.ratio <= 1.0) {
            return Err(FramePairError::Options {
                name: "ratio",
                reason: format!("must be in (0, 1], got {}", self.ratio),
            });
        }
        if let Some(split) = &self.essential_split {
            split.validate()?;
        }
        Ok(())
    }
}

/// Why [`FrameFeatures::new`] or [`match_frame_pair`] refused its input.
#[derive(Clone, Debug, PartialEq)]
pub enum FramePairError {
    /// An option out of range: a ratio outside `(0, 1]`, a forest parameter
    /// of 0, which the forest's build asserts against, or an essential split
    /// option out of range, its camera among them. `name` is the field.
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
    /// The essential split's camera is not of the frames' image size, so its
    /// pixels are not the frames' pixels.
    CameraSize {
        camera: (u32, u32),
        frames: (u32, u32),
    },
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
            Self::CameraSize { camera, frames } => write!(
                f,
                "the essential split's camera is {} x {} but the frames are {} x {}",
                camera.0, camera.1, frames.0, frames.1
            ),
        }
    }
}

impl std::error::Error for FramePairError {}

/// One pair's statistics.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FramePairStats {
    /// The number of mutual matches.
    pub matches: usize,
    /// The median length of the mutual matches' displacements, in percent of
    /// the image's long side. NaN when there are no matches.
    pub step_percent: f64,
    /// The essential split of the matches, or why none was measured; `None`
    /// when the options gave no camera, so none was attempted.
    pub essential_split: Option<EssentialSplitOutcome>,
}

/// Match two frames and measure the motion between them.
///
/// The matches are the mutual best matches of two `k = 2` queries, each
/// frame's descriptors against the other's forest, that pass the ratio test
/// both ways. The step is the median length of their displacements. With
/// [`FramePairOptions::essential_split`] set, the matches' essential split is
/// measured too ([`measure_essential_split`]).
///
/// # Errors
///
/// [`FramePairError::Options`] for an option out of range,
/// [`FramePairError::SizeMismatch`] when the frames' images differ in size and
/// [`FramePairError::CameraSize`] when the essential split's camera is of
/// another size than the frames.
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
    if let Some(split) = &options.essential_split {
        let camera = (split.camera.width, split.camera.height);
        if camera != (a.width, a.height) {
            return Err(FramePairError::CameraSize {
                camera,
                frames: (a.width, a.height),
            });
        }
    }
    let pairs = mutual_matches(a, b, options);
    let long_side = f64::from(a.width.max(a.height));
    let lengths: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| {
            let (p, q) = (widen(a.positions_xy[i]), widen(b.positions_xy[j]));
            (q[0] - p[0]).hypot(q[1] - p[1])
        })
        .collect();
    let essential_split = match &options.essential_split {
        None => None,
        Some(split) => {
            let (points_a, points_b): (Vec<[f64; 2]>, Vec<[f64; 2]>) = pairs
                .iter()
                .map(|&(i, j)| (widen(a.positions_xy[i]), widen(b.positions_xy[j])))
                .unzip();
            Some(measure_essential_split(&points_a, &points_b, split)?)
        }
    };
    Ok(FramePairStats {
        matches: pairs.len(),
        step_percent: 100.0 * median(&lengths) / long_side,
        essential_split,
    })
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
