// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Feature matching algorithms.
//!
//! This module provides descriptor distance computation, best-match
//! search, and sort-and-sweep matching for SIFT features.

pub mod descriptor;
mod gather;
pub mod geometric_filter;
pub mod polar;
pub mod sweep;
mod window;

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

pub use descriptor::{
    descriptor_distance_l2, descriptor_distance_l2_squared, find_best_match,
    find_best_match_contiguous,
};
pub use geometric_filter::{
    two_stage_geometric_filter, CameraParams, GeometricFilterConfig, StereoPairGeometry,
};
pub use polar::{polar_mutual_best_match, polar_mutual_best_match_geometric};
pub use sweep::{
    match_one_way_sweep, match_one_way_sweep_geometric, mutual_best_match_sweep,
    mutual_best_match_sweep_geometric,
};

/// The mutual matches of one image pair, as [`match_image_pair`] returns
/// them: `(index1, index2, distance)` in original feature indices, sorted by
/// `index1`.
pub type PairMatches = Vec<(usize, usize, f64)>;

/// Why [`match_image_pair`] or [`match_image_pairs_batch`] refused its input.
///
/// Both functions check every length and index they read before matching, so
/// a caller's mistake is this error rather than a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchImagePairError {
    /// A per-feature array does not hold `rows × width` elements. `array` is
    /// the parameter name, and `image` the index into the batch's per-image
    /// lists (`None` from [`match_image_pair`], whose names say the image).
    FeatureArrayLength {
        array: &'static str,
        image: Option<usize>,
        rows: usize,
        width: usize,
        len: usize,
    },
    /// [`match_image_pairs_batch`]: a per-image or per-camera list's length
    /// differs from that of `reference`, the list it must match.
    ListLength {
        list: &'static str,
        len: usize,
        reference: &'static str,
        expected: usize,
    },
    /// [`match_image_pairs_batch`]: a pair names an image past the end of
    /// the per-image lists.
    ImageIndexOutOfRange {
        pair: (usize, usize),
        num_images: usize,
    },
    /// [`match_image_pairs_batch`]: an image's camera index is past the end
    /// of the per-camera lists.
    CameraIndexOutOfRange {
        image: usize,
        camera: usize,
        num_cameras: usize,
    },
}

impl std::fmt::Display for MatchImagePairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FeatureArrayLength {
                array,
                image,
                rows,
                width,
                len,
            } => {
                write!(f, "{array}")?;
                if let Some(image) = image {
                    write!(f, "[{image}]")?;
                }
                write!(
                    f,
                    " has {len} elements but {rows} features of {width} elements each need {}",
                    rows.saturating_mul(*width)
                )
            }
            Self::ListLength {
                list,
                len,
                reference,
                expected,
            } => write!(f, "{list} has {len} entries but {reference} has {expected}"),
            Self::ImageIndexOutOfRange { pair, num_images } => write!(
                f,
                "pair ({}, {}) has an image index out of range for {num_images} images",
                pair.0, pair.1
            ),
            Self::CameraIndexOutOfRange {
                image,
                camera,
                num_cameras,
            } => write!(
                f,
                "image {image} has camera index {camera}, out of range for {num_cameras} cameras"
            ),
        }
    }
}

impl std::error::Error for MatchImagePairError {}

/// `Err` unless `array` holds `rows × width` elements.
fn check_feature_array<T>(
    name: &'static str,
    image: Option<usize>,
    array: &[T],
    rows: usize,
    width: usize,
) -> Result<(), MatchImagePairError> {
    if rows.checked_mul(width) == Some(array.len()) {
        Ok(())
    } else {
        Err(MatchImagePairError::FeatureArrayLength {
            array: name,
            image,
            rows,
            width,
            len: array.len(),
        })
    }
}

/// Match features between an image pair given camera poses and SIFT features.
///
/// Internally:
/// 1. Computes fundamental matrix and epipole
/// 2. Checks rectification safety (epipole outside image)
/// 3. If safe: computes stereo rectification, rectifies points, runs sweep matching
/// 4. If unsafe: runs polar sweep matching
/// 5. Optionally applies geometric filtering
///
/// Returns `(index1, index2, distance)` for every mutual match, in the
/// caller's original feature indices, sorted by `index1`.
///
/// # Errors
///
/// [`MatchImagePairError::FeatureArrayLength`], before anything is matched,
/// when `positions1` does not hold `2 × n1` values, `descriptors1` does not
/// hold `desc_len × n1` bytes, or `affines1`, when `Some`, does not hold
/// `4 × n1` values; and the same for image 2 with `n2`.
///
/// **Convention.** This pipeline is pixel-space `K`-matrix algebra
/// throughout (F, rectification homographies, epipolar filters), so the
/// poses must be in the COLMAP/OpenCV **optical** camera convention
/// (+Z forward, y down). A caller holding canonical `.sfmr` poses must
/// pre-multiply each pose by `S = diag(1, −1, −1)` (`r → S·r`, `t → S·t`;
/// see [`crate::geometry::convention`]).
#[allow(clippy::too_many_arguments)]
pub fn match_image_pair(
    // Camera intrinsics
    k1: &Matrix3<f64>,
    k2: &Matrix3<f64>,
    // Camera poses (cam_from_world)
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
    // Image dimensions (for rectification safety check)
    width1: u32,
    height1: u32,
    width2: u32,
    height2: u32,
    // Features (flat row-major arrays)
    positions1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    positions2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    // Matching parameters
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    rectification_margin: u32,
    // Optional geometric filtering
    affines1: Option<&[f64]>,
    affines2: Option<&[f64]>,
    geometric_config: Option<&GeometricFilterConfig>,
) -> Result<PairMatches, MatchImagePairError> {
    check_feature_array("positions1", None, positions1, n1, 2)?;
    check_feature_array("descriptors1", None, descriptors1, n1, desc_len)?;
    check_feature_array("positions2", None, positions2, n2, 2)?;
    check_feature_array("descriptors2", None, descriptors2, n2, desc_len)?;
    if let Some(a) = affines1 {
        check_feature_array("affines1", None, a, n1, 4)?;
    }
    if let Some(a) = affines2 {
        check_feature_array("affines2", None, a, n2, 4)?;
    }
    Ok(match_checked_image_pair(
        k1,
        k2,
        r1,
        t1,
        r2,
        t2,
        width1,
        height1,
        width2,
        height2,
        positions1,
        descriptors1,
        n1,
        positions2,
        descriptors2,
        n2,
        desc_len,
        window_size,
        threshold,
        rectification_margin,
        affines1,
        affines2,
        geometric_config,
    ))
}

/// The body of [`match_image_pair`], for feature arrays whose lengths have
/// been checked against `n1`, `n2` and `desc_len`.
#[allow(clippy::too_many_arguments)]
fn match_checked_image_pair(
    k1: &Matrix3<f64>,
    k2: &Matrix3<f64>,
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
    width1: u32,
    height1: u32,
    width2: u32,
    height2: u32,
    positions1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    positions2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    rectification_margin: u32,
    affines1: Option<&[f64]>,
    affines2: Option<&[f64]>,
    geometric_config: Option<&GeometricFilterConfig>,
) -> Vec<(usize, usize, f64)> {
    use crate::camera::{epipolar, rectification};

    // A singular intrinsic matrix (e.g. a degenerate camera with zero focal
    // length) has no fundamental matrix; such a pair cannot be matched.
    let Some(f) = epipolar::compute_fundamental_matrix(k1, r1, t1, k2, r2, t2) else {
        return Vec::new();
    };
    // Check both epipoles: right epipole (in image 2) and left epipole (in image 1)
    let is_safe =
        rectification::check_rectification_safe_from_f(&f, width2, height2, rectification_margin)
            && rectification::check_rectification_safe_from_f(
                &f.transpose(),
                width1,
                height1,
                rectification_margin,
            );

    let use_geometric = affines1.is_some() && affines2.is_some() && geometric_config.is_some();

    if !is_safe {
        // Polar path
        let f_data = f.as_slice();
        let f_arr: [f64; 9] = [
            f_data[0], f_data[3], f_data[6], // row 0 (nalgebra is column-major)
            f_data[1], f_data[4], f_data[7], // row 1
            f_data[2], f_data[5], f_data[8], // row 2
        ];
        let min_radius = 10.0;

        let polar_result = if use_geometric {
            let geom = StereoPairGeometry::new(k1, k2, r1, r2, t1, t2);
            polar_mutual_best_match_geometric(
                positions1,
                descriptors1,
                n1,
                positions2,
                descriptors2,
                n2,
                affines1.unwrap(),
                affines2.unwrap(),
                desc_len,
                &f_arr,
                window_size,
                threshold,
                min_radius,
                &geom,
                geometric_config.unwrap(),
            )
        } else {
            polar_mutual_best_match(
                positions1,
                descriptors1,
                n1,
                positions2,
                descriptors2,
                n2,
                desc_len,
                &f_arr,
                window_size,
                threshold,
                min_radius,
            )
        };

        // If polar returns Some, use it; if None (epipoles at infinity), fall through to rectified path
        if let Some(matches) = polar_result {
            return matches;
        }
    }

    // Rectified sweep path
    let r_rel = r2 * r1.transpose();
    let t_rel = t2 - r_rel * t1;
    let rect = rectification::compute_stereo_rectification(k1, k2, &r_rel, &t_rel, width1, height1);

    let k1_inv = k1
        .try_inverse()
        .expect("Intrinsic matrix K1 must be invertible");
    let k2_inv = k2
        .try_inverse()
        .expect("Intrinsic matrix K2 must be invertible");

    let rectified1 =
        rectification::rectify_points(positions1, n1, &k1_inv, &rect.r1_rect, &rect.p1);
    let rectified2 =
        rectification::rectify_points(positions2, n2, &k2_inv, &rect.r2_rect, &rect.p2);

    if use_geometric {
        // The sweep sorts by the rectified positions, but `geom` holds the
        // original cameras, so the filter reads the raw positions (and the
        // affine shapes, which are also in raw pixels).
        let geom = StereoPairGeometry::new(k1, k2, r1, r2, t1, t2);
        sweep::mutual_best_match_sweep_inner(
            &rectified1,
            descriptors1,
            n1,
            &rectified2,
            descriptors2,
            n2,
            desc_len,
            window_size,
            threshold,
            Some(sweep::SweepFilterInputs {
                positions1,
                positions2,
                affines1: affines1.unwrap(),
                affines2: affines2.unwrap(),
                geom: &geom,
                config: geometric_config.unwrap(),
            }),
        )
    } else {
        mutual_best_match_sweep(
            &rectified1,
            descriptors1,
            n1,
            &rectified2,
            descriptors2,
            n2,
            desc_len,
            window_size,
            threshold,
        )
    }
}

/// Match features across multiple image pairs in parallel using Rayon.
///
/// Each pair is processed independently using `match_image_pair`.
///
/// # Parameters
///
/// * `pairs` - List of (image_i_index, image_j_index) pairs to match
/// * `intrinsics` - Intrinsic matrix for each camera (indexed by camera_index)
/// * `rotations` - Per-image rotation matrix (cam_from_world)
/// * `translations` - Per-image translation vector (cam_from_world)
/// * `camera_indices` - Maps image index to camera index in `intrinsics`
/// * `positions` - Per-image feature positions (flat N×2 arrays)
/// * `descriptors` - Per-image feature descriptors (flat N×128 arrays)
/// * `num_features` - Number of features per image
/// * `widths` - Per-camera image width in pixels (indexed by camera_index)
/// * `heights` - Per-camera image height in pixels (indexed by camera_index)
/// * All other params same as match_image_pair
///
/// # Returns
///
/// Vec of match results, one per pair, in the same order as `pairs`.
///
/// # Errors
///
/// Every list and index is checked before any pair is matched:
/// [`MatchImagePairError::ListLength`] when a per-image list (`rotations`,
/// `translations`, `positions`, `descriptors`, `num_features`, and `affines`
/// when `Some`) is not as long as `camera_indices`, or a per-camera list
/// (`widths`, `heights`) is not as long as `intrinsics`;
/// [`MatchImagePairError::CameraIndexOutOfRange`] when an entry of
/// `camera_indices` is not below `intrinsics.len()`;
/// [`MatchImagePairError::ImageIndexOutOfRange`] when a pair names an image
/// not below `camera_indices.len()`; and
/// [`MatchImagePairError::FeatureArrayLength`] when an image's positions,
/// descriptors or affine shapes do not hold `num_features[i]` rows, checked
/// for every image whether or not a pair names it.
#[allow(clippy::too_many_arguments)]
pub fn match_image_pairs_batch(
    pairs: &[(usize, usize)],
    intrinsics: &[Matrix3<f64>],
    rotations: &[Matrix3<f64>],
    translations: &[Vector3<f64>],
    camera_indices: &[usize],
    positions: &[&[f64]],
    descriptors: &[&[u8]],
    num_features: &[usize],
    widths: &[u32],
    heights: &[u32],
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    rectification_margin: u32,
    affines: Option<&[&[f64]]>,
    geometric_config: Option<&GeometricFilterConfig>,
) -> Result<Vec<PairMatches>, MatchImagePairError> {
    check_batch_inputs(
        pairs,
        intrinsics,
        rotations,
        translations,
        camera_indices,
        positions,
        descriptors,
        num_features,
        widths,
        heights,
        desc_len,
        affines,
    )?;
    Ok(pairs
        .par_iter()
        .map(|&(i, j)| {
            let cam_idx_i = camera_indices[i];
            let cam_idx_j = camera_indices[j];
            let k1 = &intrinsics[cam_idx_i];
            let k2 = &intrinsics[cam_idx_j];
            let r1 = &rotations[i];
            let r2 = &rotations[j];
            let t1 = &translations[i];
            let t2 = &translations[j];

            let aff1 = affines.map(|a| a[i]);
            let aff2 = affines.map(|a| a[j]);

            match_checked_image_pair(
                k1,
                k2,
                r1,
                t1,
                r2,
                t2,
                widths[cam_idx_i],
                heights[cam_idx_i],
                widths[cam_idx_j],
                heights[cam_idx_j],
                positions[i],
                descriptors[i],
                num_features[i],
                positions[j],
                descriptors[j],
                num_features[j],
                desc_len,
                window_size,
                threshold,
                rectification_margin,
                aff1,
                aff2,
                geometric_config,
            )
        })
        .collect())
}

/// Checks every length and index [`match_image_pairs_batch`] reads, in the
/// order its `# Errors` section lists them.
#[allow(clippy::too_many_arguments)]
fn check_batch_inputs(
    pairs: &[(usize, usize)],
    intrinsics: &[Matrix3<f64>],
    rotations: &[Matrix3<f64>],
    translations: &[Vector3<f64>],
    camera_indices: &[usize],
    positions: &[&[f64]],
    descriptors: &[&[u8]],
    num_features: &[usize],
    widths: &[u32],
    heights: &[u32],
    desc_len: usize,
    affines: Option<&[&[f64]]>,
) -> Result<(), MatchImagePairError> {
    let num_images = camera_indices.len();
    let num_cameras = intrinsics.len();
    let mut lists = vec![
        ("rotations", rotations.len(), "camera_indices", num_images),
        (
            "translations",
            translations.len(),
            "camera_indices",
            num_images,
        ),
        ("positions", positions.len(), "camera_indices", num_images),
        (
            "descriptors",
            descriptors.len(),
            "camera_indices",
            num_images,
        ),
        (
            "num_features",
            num_features.len(),
            "camera_indices",
            num_images,
        ),
        ("widths", widths.len(), "intrinsics", num_cameras),
        ("heights", heights.len(), "intrinsics", num_cameras),
    ];
    if let Some(a) = affines {
        lists.push(("affines", a.len(), "camera_indices", num_images));
    }
    for (list, len, reference, expected) in lists {
        if len != expected {
            return Err(MatchImagePairError::ListLength {
                list,
                len,
                reference,
                expected,
            });
        }
    }
    if let Some((image, &camera)) = camera_indices
        .iter()
        .enumerate()
        .find(|&(_, &c)| c >= num_cameras)
    {
        return Err(MatchImagePairError::CameraIndexOutOfRange {
            image,
            camera,
            num_cameras,
        });
    }
    if let Some(&pair) = pairs
        .iter()
        .find(|&&(i, j)| i >= num_images || j >= num_images)
    {
        return Err(MatchImagePairError::ImageIndexOutOfRange { pair, num_images });
    }
    for (image, &n) in num_features.iter().enumerate() {
        check_feature_array("positions", Some(image), positions[image], n, 2)?;
        check_feature_array("descriptors", Some(image), descriptors[image], n, desc_len)?;
        if let Some(a) = affines {
            check_feature_array("affines", Some(image), a[image], n, 4)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
