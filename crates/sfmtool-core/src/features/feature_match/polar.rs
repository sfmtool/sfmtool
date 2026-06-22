// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Polar sweep matching for in-frame epipole cases.
//!
//! When standard stereo rectification fails because the epipole is inside
//! the image (forward/backward camera motion), this module transforms
//! features to polar coordinates centered at the epipole and performs
//! sort-and-sweep matching in angular space.

use std::collections::HashMap;
use std::f64::consts::PI;

use nalgebra::Matrix3;

use crate::camera::epipolar;

use super::descriptor::find_best_match_contiguous;
use super::geometric_filter::{
    two_stage_geometric_filter, GeometricFilterConfig, StereoPairGeometry,
};

/// Compute both epipoles from a flat row-major 3x3 fundamental matrix.
///
/// Thin wrapper around [`epipolar::compute_epipole_pair`] that accepts
/// the flat `&[f64; 9]` layout used by this module's public API.
fn compute_epipole_pair(f_matrix: &[f64; 9]) -> Option<([f64; 2], [f64; 2])> {
    let f = Matrix3::from_row_slice(f_matrix);
    epipolar::compute_epipole_pair(&f)
}

/// Transform 2D feature positions to polar coordinates centered at an epipole.
///
/// Each point is converted to `(θ, r)` where `θ = atan2(dy, dx)` and
/// `r = sqrt(dx² + dy²)` relative to the epipole. Points closer than
/// `min_radius` to the epipole are excluded (they are too close for
/// reliable angular sorting).
///
/// # Parameters
///
/// * `points` – Feature positions, flat row-major Nx2.
/// * `n` – Number of points.
/// * `epipole` – The `[x, y]` center for the polar transform.
/// * `min_radius` – Minimum distance from the epipole; closer points
///   are filtered out.
///
/// # Returns
///
/// `(theta, radius, valid_indices)` — parallel vectors for the points
/// that passed the radius filter, plus their original indices into the
/// input array.
pub fn cartesian_to_polar(
    points: &[f64],
    n: usize,
    epipole: [f64; 2],
    min_radius: f64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut theta = Vec::new();
    let mut radius = Vec::new();
    let mut valid_indices = Vec::new();

    let min_radius_sq = min_radius * min_radius;

    for i in 0..n {
        let dx = points[i * 2] - epipole[0];
        let dy = points[i * 2 + 1] - epipole[1];
        let r_sq = dx * dx + dy * dy;
        if r_sq >= min_radius_sq {
            theta.push(dy.atan2(dx));
            radius.push(r_sq.sqrt());
            valid_indices.push(i);
        }
    }

    (theta, radius, valid_indices)
}

/// Compute the angle offset between polar coordinate systems using the fundamental matrix.
///
/// A ray from epipole e1 at angle θ passes through the point
/// `p1 = [e1x + r·cosθ, e1y + r·sinθ, 1]`. The epipolar line in image 2 is
/// `l2 = F · p1 = F·e1_h + r·F·[cosθ, sinθ, 0]^T`. Since e1 is the null
/// space of F (`F·e1_h = 0`), this simplifies to `l2 = r · (f0·cosθ + f1·sinθ)`
/// where f0 and f1 are the first two columns of F. The factor r cancels when
/// computing the line direction, so the mapped angle depends only on θ, not on
/// the distance from the epipole.
///
/// The corresponding ray direction in image 2 is perpendicular to l2 = [a, b, c],
/// giving `θ2 = atan2(a, -b)`. In general the offset `θ2 - θ1` varies with θ,
/// so we sample 36 angles uniformly and return the median offset as a robust
/// central estimate.
pub fn compute_angle_offset(f_matrix: &[f64; 9], _e1: [f64; 2], _e2: [f64; 2]) -> f64 {
    let f = Matrix3::from_row_slice(f_matrix);
    let f0 = f.column(0);
    let f1 = f.column(1);
    let n_samples = 36;

    let mut diffs = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let theta1 = -PI + (2.0 * PI * i as f64) / n_samples as f64;

        // l2 = f0·cosθ + f1·sinθ  (the r factor cancels in direction computation)
        let cos_t = theta1.cos();
        let sin_t = theta1.sin();
        let a = f0[0] * cos_t + f1[0] * sin_t;
        let b = f0[1] * cos_t + f1[1] * sin_t;

        // Direction perpendicular to epipolar line [a, b, _] is [-b, a]
        let mut theta2 = a.atan2(-b);

        // Resolve 180° ambiguity
        let diff = theta2 - theta1;
        let wrapped = diff.sin().atan2(diff.cos());
        if wrapped.abs() > PI / 2.0 {
            theta2 = (-a).atan2(b);
        }

        let final_diff = theta2 - theta1;
        diffs.push(final_diff.sin().atan2(final_diff.cos()));
    }

    // Return median using O(n) selection instead of O(n log n) sort
    let mid = diffs.len() / 2;
    diffs.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    diffs[mid]
}

/// Extend sorted angular arrays so that sliding-window searches near the ±π
/// boundary see neighbors that wrap around the circle.
///
/// The last `window_size` elements (nearest +π) are copied before the start
/// with θ−2π, and the first `window_size` elements (nearest −π) are copied
/// after the end with θ+2π. This guarantees the sliding window always has
/// enough candidates even at the wraparound boundary.
///
/// # Parameters
///
/// * `sorted_theta` – Angles in (−π, π], sorted in ascending order.
/// * `sorted_descs` – Flat descriptor array laid out as
///   `[desc₀ … descₙ₋₁]`, each descriptor `desc_len` bytes.
/// * `desc_len` – Number of bytes per descriptor.
/// * `window_size` – Number of nearest angular neighbors the caller's
///   sliding window will examine on each side. This many elements are
///   duplicated at each boundary.
///
/// # Returns
///
/// `(extended_theta, extended_descs, original_len, n_prepended)` where
/// `n_prepended` is the count of ghost copies inserted before the original
/// data (needed to recover original indices from extended indices).
fn extend_for_wraparound(
    sorted_theta: &[f64],
    sorted_descs: &[u8],
    desc_len: usize,
    window_size: usize,
) -> (Vec<f64>, Vec<u8>, usize, usize) {
    let n = sorted_theta.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0, 0);
    }

    // Use angular threshold to determine which features to copy,
    // matching the Python implementation's approach.
    let angular_threshold = (PI / 4.0).min(window_size as f64 / n.max(1) as f64 * 2.0 * PI);

    // Count features near +π (to prepend with θ-2π)
    let n_prepended = sorted_theta
        .iter()
        .rev()
        .take_while(|&&t| t > (PI - angular_threshold))
        .count();

    // Count features near -π (to append with θ+2π)
    let n_appended = sorted_theta
        .iter()
        .take_while(|&&t| t < (-PI + angular_threshold))
        .count();

    let total = n_prepended + n + n_appended;
    let mut ext_theta = Vec::with_capacity(total);
    let mut ext_descs = Vec::with_capacity(total * desc_len);

    // Prepend: features near +π shifted by -2π
    for i in (n - n_prepended)..n {
        ext_theta.push(sorted_theta[i] - 2.0 * PI);
        ext_descs.extend_from_slice(&sorted_descs[i * desc_len..(i + 1) * desc_len]);
    }

    // Original
    ext_theta.extend_from_slice(sorted_theta);
    ext_descs.extend_from_slice(sorted_descs);

    // Append: features near -π shifted by +2π
    for i in 0..n_appended {
        ext_theta.push(sorted_theta[i] + 2.0 * PI);
        ext_descs.extend_from_slice(&sorted_descs[i * desc_len..(i + 1) * desc_len]);
    }

    (ext_theta, ext_descs, n, n_prepended)
}

/// One-way polar-sweep nearest-neighbor match.
///
/// Walks through features in image 1 (sorted by polar angle around the
/// epipole) and, for each one, finds the best descriptor match among the
/// `window_size` angularly-closest features in image 2. A sliding window
/// over the (wraparound-extended) image-2 array keeps this linear in the
/// number of features rather than quadratic.
///
/// # Parameters
///
/// * `sorted_theta1` – Polar angles for image 1, sorted ascending.
/// * `sorted_descs1` – Flat descriptors for image 1, each `desc_len` bytes,
///   in the same sorted order as `sorted_theta1`.
/// * `sorted_theta2` – Polar angles for image 2, sorted ascending.
/// * `sorted_descs2` – Flat descriptors for image 2, each `desc_len` bytes,
///   in the same sorted order as `sorted_theta2`.
/// * `desc_len` – Number of bytes per descriptor.
/// * `window_size` – Number of angular neighbors to consider from image 2
///   for each feature in image 1.
/// * `threshold` – Optional L2 distance ceiling. When `Some(t)`, a match
///   is rejected if its descriptor distance exceeds `t`.
///
/// # Returns
///
/// A map from sorted index in image 1 to `(original_index_in_image_2, distance)`.
#[allow(clippy::too_many_arguments)]
fn polar_match_one_way(
    sorted_theta1: &[f64],
    sorted_descs1: &[u8],
    sorted_theta2: &[f64],
    sorted_descs2: &[u8],
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
) -> HashMap<usize, (usize, f64)> {
    let mut matches = HashMap::new();
    let n2 = sorted_theta2.len();

    if n2 == 0 {
        return matches;
    }

    let (ext_theta2, ext_descs2, orig_len2, n_prepended) =
        extend_for_wraparound(sorted_theta2, sorted_descs2, desc_len, window_size);

    let num_extended = ext_theta2.len();
    let mut win_start: usize = 0;

    for idx1 in 0..sorted_theta1.len() {
        let query_theta = sorted_theta1[idx1];

        // Slide window
        while win_start + window_size < num_extended {
            let diff_next = ext_theta2[win_start + window_size] - query_theta;
            let diff_start = ext_theta2[win_start] - query_theta;
            if diff_next.abs() < diff_start.abs() {
                win_start += 1;
            } else {
                break;
            }
        }

        let win_end = (win_start + window_size).min(num_extended);
        if win_end <= win_start {
            continue;
        }

        let query_desc = &sorted_descs1[idx1 * desc_len..(idx1 + 1) * desc_len];
        let window_descs = &ext_descs2[win_start * desc_len..win_end * desc_len];

        if let Some((rel_idx, dist)) =
            find_best_match_contiguous(query_desc, window_descs, desc_len, threshold)
        {
            let ext_idx = win_start + rel_idx;
            let orig_idx =
                ((ext_idx as isize - n_prepended as isize).rem_euclid(orig_len2 as isize)) as usize;
            matches.insert(idx1, (orig_idx, dist));
        }
    }

    matches
}

/// Full bidirectional polar sweep matching with mutual consistency check.
///
/// This is the polar-coordinate analog of [`super::sweep::mutual_best_match_sweep`].
/// Use it when standard Y-sweep rectification fails because the epipole lies
/// inside the image (e.g. forward/backward camera motion).
///
/// The algorithm:
/// 1. Computes both epipoles from the fundamental matrix.
/// 2. Transforms features to polar coordinates centered at each epipole.
/// 3. Aligns the two angular systems using an F-derived angle offset.
/// 4. Sorts by angle and runs forward + backward one-way polar sweeps.
/// 5. Keeps only mutual best matches and maps back to original indices.
///
/// Returns `None` if either epipole is at infinity (fall back to Y-sweep).
///
/// # Parameters
///
/// * `positions1` – Feature positions in image 1, flat row-major Nx2.
/// * `descriptors1` – Descriptors for image 1, flat row-major Nx(desc_len).
/// * `n1` – Number of features in image 1.
/// * `positions2` – Feature positions in image 2, flat row-major Mx2.
/// * `descriptors2` – Descriptors for image 2, flat row-major Mx(desc_len).
/// * `n2` – Number of features in image 2.
/// * `desc_len` – Number of bytes per descriptor (typically 128).
/// * `f_matrix` – 3×3 fundamental matrix, row-major flat.
/// * `window_size` – Number of angular neighbors to consider in each
///   direction.
/// * `threshold` – Optional L2 distance ceiling.
/// * `min_radius` – Minimum distance from the epipole; features closer
///   are excluded from matching.
///
/// # Returns
///
/// `Some(Vec<(orig_idx1, orig_idx2, distance)>)` for mutual matches,
/// or `None` if either epipole is at infinity.
#[allow(clippy::too_many_arguments)]
pub fn polar_mutual_best_match(
    positions1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    positions2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    desc_len: usize,
    f_matrix: &[f64; 9],
    window_size: usize,
    threshold: Option<f64>,
    min_radius: f64,
) -> Option<Vec<(usize, usize, f64)>> {
    if n1 == 0 || n2 == 0 {
        return Some(Vec::new());
    }

    // Compute epipoles
    let (e1, e2) = compute_epipole_pair(f_matrix)?;

    // Transform to polar
    let (theta1, _radius1, valid1) = cartesian_to_polar(positions1, n1, e1, min_radius);
    let (theta2, _radius2, valid2) = cartesian_to_polar(positions2, n2, e2, min_radius);

    if theta1.is_empty() || theta2.is_empty() {
        return Some(Vec::new());
    }

    // Compute angle offset
    let offset = compute_angle_offset(f_matrix, e1, e2);

    // Align theta2
    let theta2_aligned: Vec<f64> = theta2
        .iter()
        .map(|&t| {
            let adj = t - offset;
            adj.sin().atan2(adj.cos())
        })
        .collect();

    // Sort by theta (with radius as tiebreaker for determinism)
    let sort_idx1 = {
        let mut idx: Vec<usize> = (0..theta1.len()).collect();
        idx.sort_by(|&a, &b| {
            theta1[a]
                .partial_cmp(&theta1[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    _radius1[a]
                        .partial_cmp(&_radius1[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        idx
    };

    let sort_idx2 = {
        let mut idx: Vec<usize> = (0..theta2_aligned.len()).collect();
        idx.sort_by(|&a, &b| {
            theta2_aligned[a]
                .partial_cmp(&theta2_aligned[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    _radius2[a]
                        .partial_cmp(&_radius2[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        idx
    };

    // Gather sorted arrays
    let sorted_theta1: Vec<f64> = sort_idx1.iter().map(|&i| theta1[i]).collect();
    let sorted_theta2: Vec<f64> = sort_idx2.iter().map(|&i| theta2_aligned[i]).collect();

    // Gather descriptors using valid_indices mapping
    let sorted_descs1 = gather_descs(descriptors1, &valid1, &sort_idx1, desc_len);
    let sorted_descs2 = gather_descs(descriptors2, &valid2, &sort_idx2, desc_len);

    // Forward matching
    let forward = polar_match_one_way(
        &sorted_theta1,
        &sorted_descs1,
        &sorted_theta2,
        &sorted_descs2,
        desc_len,
        window_size,
        threshold,
    );

    // Backward matching
    let backward = polar_match_one_way(
        &sorted_theta2,
        &sorted_descs2,
        &sorted_theta1,
        &sorted_descs1,
        desc_len,
        window_size,
        threshold,
    );

    // Mutual consistency + map to original indices
    let mut mutual = Vec::new();
    for (&s_idx1, &(s_idx2, dist)) in &forward {
        if let Some(&(back_idx1, _)) = backward.get(&s_idx2) {
            if back_idx1 == s_idx1 {
                let orig_idx1 = valid1[sort_idx1[s_idx1]];
                let orig_idx2 = valid2[sort_idx2[s_idx2]];
                mutual.push((orig_idx1, orig_idx2, dist));
            }
        }
    }

    Some(mutual)
}

/// Gather descriptors in sorted angular order.
///
/// Composes two levels of indirection: `sort_idx` gives the angular
/// ordering, `valid_indices` maps those back to original feature indices
/// (some features may have been filtered by `min_radius`). The result
/// is a flat contiguous descriptor array ready for the sweep matcher.
fn gather_descs(
    descs: &[u8],
    valid_indices: &[usize],
    sort_idx: &[usize],
    desc_len: usize,
) -> Vec<u8> {
    let mut result = Vec::with_capacity(sort_idx.len() * desc_len);
    for &si in sort_idx {
        let orig = valid_indices[si];
        let start = orig * desc_len;
        result.extend_from_slice(&descs[start..start + desc_len]);
    }
    result
}

/// Gather positions (Nx2) in sorted angular order, composing valid_indices and sort_idx.
fn gather_positions(positions: &[f64], valid_indices: &[usize], sort_idx: &[usize]) -> Vec<f64> {
    let mut result = Vec::with_capacity(sort_idx.len() * 2);
    for &si in sort_idx {
        let orig = valid_indices[si];
        let start = orig * 2;
        result.extend_from_slice(&positions[start..start + 2]);
    }
    result
}

/// Gather affines (Nx4) in sorted angular order, composing valid_indices and sort_idx.
fn gather_affines(affines: &[f64], valid_indices: &[usize], sort_idx: &[usize]) -> Vec<f64> {
    let mut result = Vec::with_capacity(sort_idx.len() * 4);
    for &si in sort_idx {
        let orig = valid_indices[si];
        let start = orig * 4;
        result.extend_from_slice(&affines[start..start + 4]);
    }
    result
}

/// Extend sorted angular arrays for wraparound, including positions and affines.
///
/// Same angular-threshold approach as [`extend_for_wraparound`], but also
/// extends parallel position (Nx2 flat) and affine (Nx4 flat) arrays.
///
/// # Returns
///
/// `(ext_theta, ext_descs, ext_positions, ext_affines, original_len, n_prepended)`
fn extend_for_wraparound_geometric(
    sorted_theta: &[f64],
    sorted_descs: &[u8],
    sorted_positions: &[f64],
    sorted_affines: &[f64],
    desc_len: usize,
    window_size: usize,
) -> (Vec<f64>, Vec<u8>, Vec<f64>, Vec<f64>, usize, usize) {
    let n = sorted_theta.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0, 0);
    }

    let angular_threshold = (PI / 4.0).min(window_size as f64 / n.max(1) as f64 * 2.0 * PI);

    // Count features near +π (to prepend with θ-2π)
    let n_prepended = sorted_theta
        .iter()
        .rev()
        .take_while(|&&t| t > (PI - angular_threshold))
        .count();

    // Count features near -π (to append with θ+2π)
    let n_appended = sorted_theta
        .iter()
        .take_while(|&&t| t < (-PI + angular_threshold))
        .count();

    let total = n_prepended + n + n_appended;
    let mut ext_theta = Vec::with_capacity(total);
    let mut ext_descs = Vec::with_capacity(total * desc_len);
    let mut ext_positions = Vec::with_capacity(total * 2);
    let mut ext_affines = Vec::with_capacity(total * 4);

    // Prepend: features near +π shifted by -2π
    for i in (n - n_prepended)..n {
        ext_theta.push(sorted_theta[i] - 2.0 * PI);
        ext_descs.extend_from_slice(&sorted_descs[i * desc_len..(i + 1) * desc_len]);
        ext_positions.extend_from_slice(&sorted_positions[i * 2..(i + 1) * 2]);
        ext_affines.extend_from_slice(&sorted_affines[i * 4..(i + 1) * 4]);
    }

    // Original
    ext_theta.extend_from_slice(sorted_theta);
    ext_descs.extend_from_slice(sorted_descs);
    ext_positions.extend_from_slice(sorted_positions);
    ext_affines.extend_from_slice(sorted_affines);

    // Append: features near -π shifted by +2π
    for i in 0..n_appended {
        ext_theta.push(sorted_theta[i] + 2.0 * PI);
        ext_descs.extend_from_slice(&sorted_descs[i * desc_len..(i + 1) * desc_len]);
        ext_positions.extend_from_slice(&sorted_positions[i * 2..(i + 1) * 2]);
        ext_affines.extend_from_slice(&sorted_affines[i * 4..(i + 1) * 4]);
    }

    (
        ext_theta,
        ext_descs,
        ext_positions,
        ext_affines,
        n,
        n_prepended,
    )
}

/// One-way polar-sweep nearest-neighbor match with geometric filtering.
///
/// Like [`polar_match_one_way`] but applies a two-stage geometric filter
/// (orientation + size consistency) to the sliding window *before* descriptor
/// comparison. Only candidates that pass the geometric check are considered
/// for the best descriptor match.
#[allow(clippy::too_many_arguments)]
fn polar_match_one_way_geometric(
    sorted_theta1: &[f64],
    sorted_descs1: &[u8],
    sorted_positions1: &[f64],
    sorted_affines1: &[f64],
    sorted_theta2: &[f64],
    sorted_descs2: &[u8],
    sorted_positions2: &[f64],
    sorted_affines2: &[f64],
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    geom: &StereoPairGeometry,
    config: &GeometricFilterConfig,
) -> HashMap<usize, (usize, f64)> {
    let mut matches = HashMap::new();
    let n2 = sorted_theta2.len();

    if n2 == 0 {
        return matches;
    }

    let (ext_theta2, ext_descs2, ext_positions2, ext_affines2, orig_len2, n_prepended) =
        extend_for_wraparound_geometric(
            sorted_theta2,
            sorted_descs2,
            sorted_positions2,
            sorted_affines2,
            desc_len,
            window_size,
        );

    let num_extended = ext_theta2.len();
    let mut win_start: usize = 0;
    let mut filtered_descs_buf: Vec<u8> = Vec::new();

    for idx1 in 0..sorted_theta1.len() {
        let query_theta = sorted_theta1[idx1];

        // Slide window
        while win_start + window_size < num_extended {
            let diff_next = ext_theta2[win_start + window_size] - query_theta;
            let diff_start = ext_theta2[win_start] - query_theta;
            if diff_next.abs() < diff_start.abs() {
                win_start += 1;
            } else {
                break;
            }
        }

        let win_end = (win_start + window_size).min(num_extended);
        if win_end <= win_start {
            continue;
        }

        let window_len = win_end - win_start;

        // Extract query point and affine
        let x1 = [sorted_positions1[idx1 * 2], sorted_positions1[idx1 * 2 + 1]];
        let affine1 = [
            sorted_affines1[idx1 * 4],
            sorted_affines1[idx1 * 4 + 1],
            sorted_affines1[idx1 * 4 + 2],
            sorted_affines1[idx1 * 4 + 3],
        ];

        // Window positions and affines from extended arrays
        let window_positions = &ext_positions2[win_start * 2..win_end * 2];
        let window_affines = &ext_affines2[win_start * 4..win_end * 4];

        // Apply geometric filter
        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            window_positions,
            window_affines,
            window_len,
            geom,
            config,
        );

        // Gather filtered descriptors into contiguous buffer
        filtered_descs_buf.clear();
        let mut valid_indices: Vec<usize> = Vec::new();
        for (i, &passes) in mask.iter().enumerate() {
            if passes {
                valid_indices.push(i);
                let src_idx = win_start + i;
                let src_start = src_idx * desc_len;
                filtered_descs_buf.extend_from_slice(&ext_descs2[src_start..src_start + desc_len]);
            }
        }

        if valid_indices.is_empty() {
            continue;
        }

        let query_desc = &sorted_descs1[idx1 * desc_len..(idx1 + 1) * desc_len];

        if let Some((rel_idx, dist)) =
            find_best_match_contiguous(query_desc, &filtered_descs_buf, desc_len, threshold)
        {
            // Map back: rel_idx -> valid_indices[rel_idx] -> window index -> ext index -> original
            let ext_idx = win_start + valid_indices[rel_idx];
            let orig_idx =
                ((ext_idx as isize - n_prepended as isize).rem_euclid(orig_len2 as isize)) as usize;
            matches.insert(idx1, (orig_idx, dist));
        }
    }

    matches
}

/// Full bidirectional polar sweep matching with geometric filtering and mutual consistency.
///
/// Like [`polar_mutual_best_match`] but applies two-stage geometric filtering
/// (orientation + size consistency) before descriptor comparison in both directions.
///
/// # Parameters
///
/// * `positions1` – Feature positions in image 1, flat row-major Nx2.
/// * `descriptors1` – Descriptors for image 1, flat row-major Nx(desc_len).
/// * `n1` – Number of features in image 1.
/// * `positions2` – Feature positions in image 2, flat row-major Mx2.
/// * `descriptors2` – Descriptors for image 2, flat row-major Mx(desc_len).
/// * `n2` – Number of features in image 2.
/// * `affines1` – Affine shapes for image 1, flat row-major Nx4.
/// * `affines2` – Affine shapes for image 2, flat row-major Mx4.
/// * `desc_len` – Number of bytes per descriptor (typically 128).
/// * `f_matrix` – 3×3 fundamental matrix, row-major flat.
/// * `window_size` – Number of angular neighbors to consider in each direction.
/// * `threshold` – Optional L2 distance ceiling.
/// * `min_radius` – Minimum distance from the epipole; features closer are excluded.
/// * `geom` – Precomputed camera geometry.
/// * `config` – Geometric filter configuration.
///
/// # Returns
///
/// `Some(Vec<(orig_idx1, orig_idx2, distance)>)` for mutual matches,
/// or `None` if either epipole is at infinity.
#[allow(clippy::too_many_arguments)]
pub fn polar_mutual_best_match_geometric(
    positions1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    positions2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    affines1: &[f64],
    affines2: &[f64],
    desc_len: usize,
    f_matrix: &[f64; 9],
    window_size: usize,
    threshold: Option<f64>,
    min_radius: f64,
    geom: &StereoPairGeometry,
    config: &GeometricFilterConfig,
) -> Option<Vec<(usize, usize, f64)>> {
    if n1 == 0 || n2 == 0 {
        return Some(Vec::new());
    }

    // Compute epipoles
    let (e1, e2) = compute_epipole_pair(f_matrix)?;

    // Transform to polar
    let (theta1, _radius1, valid1) = cartesian_to_polar(positions1, n1, e1, min_radius);
    let (theta2, _radius2, valid2) = cartesian_to_polar(positions2, n2, e2, min_radius);

    if theta1.is_empty() || theta2.is_empty() {
        return Some(Vec::new());
    }

    // Compute angle offset
    let offset = compute_angle_offset(f_matrix, e1, e2);

    // Align theta2
    let theta2_aligned: Vec<f64> = theta2
        .iter()
        .map(|&t| {
            let adj = t - offset;
            adj.sin().atan2(adj.cos())
        })
        .collect();

    // Sort by theta (with radius as tiebreaker for determinism)
    let sort_idx1 = {
        let mut idx: Vec<usize> = (0..theta1.len()).collect();
        idx.sort_by(|&a, &b| {
            theta1[a]
                .partial_cmp(&theta1[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    _radius1[a]
                        .partial_cmp(&_radius1[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        idx
    };

    let sort_idx2 = {
        let mut idx: Vec<usize> = (0..theta2_aligned.len()).collect();
        idx.sort_by(|&a, &b| {
            theta2_aligned[a]
                .partial_cmp(&theta2_aligned[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    _radius2[a]
                        .partial_cmp(&_radius2[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        idx
    };

    // Gather sorted arrays
    let sorted_theta1: Vec<f64> = sort_idx1.iter().map(|&i| theta1[i]).collect();
    let sorted_theta2: Vec<f64> = sort_idx2.iter().map(|&i| theta2_aligned[i]).collect();

    let sorted_descs1 = gather_descs(descriptors1, &valid1, &sort_idx1, desc_len);
    let sorted_descs2 = gather_descs(descriptors2, &valid2, &sort_idx2, desc_len);

    let sorted_positions1 = gather_positions(positions1, &valid1, &sort_idx1);
    let sorted_positions2 = gather_positions(positions2, &valid2, &sort_idx2);

    let sorted_affines1 = gather_affines(affines1, &valid1, &sort_idx1);
    let sorted_affines2 = gather_affines(affines2, &valid2, &sort_idx2);

    // Forward matching
    let forward = polar_match_one_way_geometric(
        &sorted_theta1,
        &sorted_descs1,
        &sorted_positions1,
        &sorted_affines1,
        &sorted_theta2,
        &sorted_descs2,
        &sorted_positions2,
        &sorted_affines2,
        desc_len,
        window_size,
        threshold,
        geom,
        config,
    );

    // Backward matching (swap camera geometry)
    let geom_swapped = geom.swapped();
    let backward = polar_match_one_way_geometric(
        &sorted_theta2,
        &sorted_descs2,
        &sorted_positions2,
        &sorted_affines2,
        &sorted_theta1,
        &sorted_descs1,
        &sorted_positions1,
        &sorted_affines1,
        desc_len,
        window_size,
        threshold,
        &geom_swapped,
        config,
    );

    // Mutual consistency + map to original indices
    let mut mutual = Vec::new();
    for (&s_idx1, &(s_idx2, dist)) in &forward {
        if let Some(&(back_idx1, _)) = backward.get(&s_idx2) {
            if back_idx1 == s_idx1 {
                let orig_idx1 = valid1[sort_idx1[s_idx1]];
                let orig_idx2 = valid2[sort_idx2[s_idx2]];
                mutual.push((orig_idx1, orig_idx2, dist));
            }
        }
    }

    Some(mutual)
}

#[cfg(test)]
mod tests;
