// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Polar sweep matching for in-frame epipole cases.
//!
//! When standard stereo rectification fails because the epipole is inside
//! the image (forward/backward camera motion), this module transforms
//! features to polar coordinates centered at the epipole and performs
//! sort-and-sweep matching in angular space.
//!
//! ## Plain and geometric matching share one path
//!
//! The two public entry points — [`polar_mutual_best_match`] and
//! [`polar_mutual_best_match_geometric`] — run the *same* sweep. The geometric
//! one additionally carries per-feature positions and affine shapes through the
//! angular sort and the wraparound extension, and narrows each sliding window
//! to the candidates passing the two-stage geometric filter before any
//! descriptor is compared. That is the whole difference, and it is carried by
//! one `Option` parameter rather than by a parallel family of `_geometric`
//! functions.

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

    // Upper median via O(n) selection rather than a sort. `total_cmp` is a real
    // total order, so the selection is defined even if a degenerate `F` puts a
    // NaN in the samples; `partial_cmp().unwrap_or(Equal)` is not, and left the
    // result unspecified in exactly that case.
    let mid = diffs.len() / 2;
    diffs.select_nth_unstable_by(mid, f64::total_cmp);
    diffs[mid]
}

/// How many entries at each end of an angle-sorted array get ghost copies, so a
/// sliding window near the ±π seam still sees its neighbours across it.
///
/// Built once per candidate array by [`Self::plan`], then used to extend every
/// parallel array the matcher carries (angles, descriptors, and — on the
/// geometric path — positions and affines) with one consistent layout. It also
/// owns the inverse map, [`Self::to_original`], which is the only place the
/// extended-to-original index arithmetic is written.
struct Wraparound {
    /// Length of the original, un-extended array.
    n: usize,
    /// Ghost copies inserted before the start, taken from the +π end.
    n_prepended: usize,
    /// Ghost copies appended after the end, taken from the −π end.
    n_appended: usize,
}

impl Wraparound {
    /// Decide the copy counts for an angle-sorted array.
    ///
    /// Features within `angular_threshold` of either end of (−π, π] are the
    /// ones a window at the opposite end needs to see. The threshold matches
    /// the Python implementation's approach: a quarter turn, or the angular
    /// span the window covers on average, whichever is smaller.
    fn plan(sorted_theta: &[f64], window_size: usize) -> Self {
        let n = sorted_theta.len();
        if n == 0 {
            return Self {
                n: 0,
                n_prepended: 0,
                n_appended: 0,
            };
        }

        let angular_threshold = (PI / 4.0).min(window_size as f64 / n.max(1) as f64 * 2.0 * PI);

        Self {
            n,
            // Near +π, to be prepended with θ−2π.
            n_prepended: sorted_theta
                .iter()
                .rev()
                .take_while(|&&t| t > (PI - angular_threshold))
                .count(),
            // Near −π, to be appended with θ+2π.
            n_appended: sorted_theta
                .iter()
                .take_while(|&&t| t < (-PI + angular_threshold))
                .count(),
        }
    }

    /// Length of every extended array this plan produces.
    fn extended_len(&self) -> usize {
        self.n_prepended + self.n + self.n_appended
    }

    /// Extend the angles, shifting each ghost copy onto the continuation of the
    /// real range so the window's arithmetic stays monotonic across the seam.
    ///
    /// `sorted_theta` must be the array the plan was measured from. A plan is a
    /// separate value from the arrays it describes, so nothing in the type
    /// system ties the two together; the assertion stands in for that.
    fn extend_theta(&self, sorted_theta: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            sorted_theta.len(),
            self.n,
            "plan was measured from an array of a different length"
        );

        let mut out = Vec::with_capacity(self.extended_len());
        out.extend(
            sorted_theta[self.n - self.n_prepended..self.n]
                .iter()
                .map(|t| t - 2.0 * PI),
        );
        out.extend_from_slice(&sorted_theta[..self.n]);
        out.extend(sorted_theta[..self.n_appended].iter().map(|t| t + 2.0 * PI));
        out
    }

    /// Extend a payload array holding `stride` elements per feature, in the
    /// same layout [`Self::extend_theta`] produces.
    ///
    /// Every array extended through one plan lands in that same layout, which
    /// is what lets a window index descriptors, positions and affines
    /// interchangeably. `rows` must hold exactly one row per planned feature.
    fn extend_rows<T: Copy>(&self, rows: &[T], stride: usize) -> Vec<T> {
        debug_assert_eq!(
            rows.len(),
            self.n * stride,
            "payload length does not match the planned feature count"
        );

        let mut out = Vec::with_capacity(self.extended_len() * stride);
        out.extend_from_slice(&rows[(self.n - self.n_prepended) * stride..self.n * stride]);
        out.extend_from_slice(&rows[..self.n * stride]);
        out.extend_from_slice(&rows[..self.n_appended * stride]);
        out
    }

    /// Map an index in an extended array back to the original array, folding
    /// the ghost copies at either end onto the entries they duplicate.
    ///
    /// Panics if the plan is empty; callers return early on an empty candidate
    /// side before any index reaches here.
    fn to_original(&self, ext_idx: usize) -> usize {
        ((ext_idx as isize - self.n_prepended as isize).rem_euclid(self.n as isize)) as usize
    }
}

/// Gather a payload array of `stride` elements per feature into angular order.
///
/// Composes two levels of indirection: `sort_idx` gives the angular ordering,
/// `valid_indices` maps those back to original feature indices (some features
/// may have been filtered by `min_radius`). The result is flat and contiguous,
/// ready for the sweep matcher.
fn gather_rows<T: Copy>(
    rows: &[T],
    stride: usize,
    valid_indices: &[usize],
    sort_idx: &[usize],
) -> Vec<T> {
    let mut out = Vec::with_capacity(sort_idx.len() * stride);
    for &si in sort_idx {
        let start = valid_indices[si] * stride;
        out.extend_from_slice(&rows[start..start + stride]);
    }
    out
}

/// The extra per-feature data the geometric path needs, already in angular
/// order: query-side positions and affines, candidate-side positions and
/// affines, and the filter to apply.
struct GeometricInputs<'a> {
    positions1: &'a [f64],
    affines1: &'a [f64],
    positions2: &'a [f64],
    affines2: &'a [f64],
    geom: &'a StereoPairGeometry,
    config: &'a GeometricFilterConfig,
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
/// * `geometric` – When `Some`, each window is narrowed to the candidates
///   passing the two-stage orientation/size filter before any descriptor is
///   compared. When `None`, the whole window is compared.
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
    geometric: Option<&GeometricInputs<'_>>,
) -> HashMap<usize, (usize, f64)> {
    let mut matches = HashMap::new();

    if sorted_theta2.is_empty() {
        return matches;
    }

    let plan = Wraparound::plan(sorted_theta2, window_size);
    let ext_theta2 = plan.extend_theta(sorted_theta2);
    let ext_descs2 = plan.extend_rows(sorted_descs2, desc_len);
    let ext_geometric2 = geometric.map(|g| {
        (
            plan.extend_rows(g.positions2, 2),
            plan.extend_rows(g.affines2, 4),
        )
    });

    let num_extended = ext_theta2.len();
    let mut win_start: usize = 0;

    // Reused across iterations on the geometric path: the descriptors that
    // survived the filter, and their offsets within the current window.
    let mut passing_descs: Vec<u8> = Vec::new();
    let mut passing_offsets: Vec<usize> = Vec::new();

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

        // The candidate set is either the whole window or the part of it the
        // geometric filter admits. `window_offsets` is `Some` exactly when the
        // set was narrowed, and maps a candidate back to its offset within the
        // window; deciding it here, in the arm that knows, keeps it from
        // drifting out of step with the descriptors it indexes.
        let (candidate_descs, window_offsets): (&[u8], Option<&[usize]>) =
            match (geometric, &ext_geometric2) {
                (Some(g), Some((ext_positions2, ext_affines2))) => {
                    let x1 = [g.positions1[idx1 * 2], g.positions1[idx1 * 2 + 1]];
                    let affine1 = [
                        g.affines1[idx1 * 4],
                        g.affines1[idx1 * 4 + 1],
                        g.affines1[idx1 * 4 + 2],
                        g.affines1[idx1 * 4 + 3],
                    ];

                    let mask = two_stage_geometric_filter(
                        x1,
                        &affine1,
                        &ext_positions2[win_start * 2..win_end * 2],
                        &ext_affines2[win_start * 4..win_end * 4],
                        win_end - win_start,
                        g.geom,
                        g.config,
                    );

                    passing_descs.clear();
                    passing_offsets.clear();
                    for (offset, &passes) in mask.iter().enumerate() {
                        if passes {
                            passing_offsets.push(offset);
                            let start = (win_start + offset) * desc_len;
                            passing_descs.extend_from_slice(&ext_descs2[start..start + desc_len]);
                        }
                    }

                    if passing_offsets.is_empty() {
                        continue;
                    }
                    (&passing_descs, Some(passing_offsets.as_slice()))
                }
                (None, None) => (&ext_descs2[win_start * desc_len..win_end * desc_len], None),
                // `ext_geometric2` is built by mapping over `geometric`, so the
                // two are `Some` together or not at all.
                _ => unreachable!("geometric inputs and their extensions disagree"),
            };

        if let Some((rel_idx, dist)) =
            find_best_match_contiguous(query_desc, candidate_descs, desc_len, threshold)
        {
            // `rel_idx` indexes the candidate set, which is the window itself
            // unless the filter narrowed it.
            let offset = match window_offsets {
                Some(offsets) => offsets[rel_idx],
                None => rel_idx,
            };
            matches.insert(idx1, (plan.to_original(win_start + offset), dist));
        }
    }

    matches
}

/// The geometric filter's inputs, gathered into angular order once and held for
/// both sweep directions.
struct SortedGeometry<'a> {
    positions1: Vec<f64>,
    positions2: Vec<f64>,
    affines1: Vec<f64>,
    affines2: Vec<f64>,
    geom: &'a StereoPairGeometry,
    /// The backward sweep reverses the roles of query and target, so it needs
    /// the geometry swapped to match.
    geom_swapped: StereoPairGeometry,
    config: &'a GeometricFilterConfig,
}

impl SortedGeometry<'_> {
    fn forward(&self) -> GeometricInputs<'_> {
        GeometricInputs {
            positions1: &self.positions1,
            affines1: &self.affines1,
            positions2: &self.positions2,
            affines2: &self.affines2,
            geom: self.geom,
            config: self.config,
        }
    }

    fn backward(&self) -> GeometricInputs<'_> {
        GeometricInputs {
            positions1: &self.positions2,
            affines1: &self.affines2,
            positions2: &self.positions1,
            affines2: &self.affines1,
            geom: &self.geom_swapped,
            config: self.config,
        }
    }
}

/// Sort feature indices by polar angle, breaking ties on radius for determinism.
fn angular_order(theta: &[f64], radius: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..theta.len()).collect();
    idx.sort_by(|&a, &b| {
        theta[a]
            .partial_cmp(&theta[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                radius[a]
                    .partial_cmp(&radius[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    idx
}

/// The shared body of both public entry points.
///
/// `geometric` carries the only difference between them: `Some((affines1,
/// affines2, geom, config))` runs the geometric filter over each window,
/// `None` compares the whole window.
#[allow(clippy::too_many_arguments)]
fn polar_mutual_best_match_inner(
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
    geometric: Option<(&[f64], &[f64], &StereoPairGeometry, &GeometricFilterConfig)>,
) -> Option<Vec<(usize, usize, f64)>> {
    if n1 == 0 || n2 == 0 {
        return Some(Vec::new());
    }

    // Compute epipoles
    let (e1, e2) = compute_epipole_pair(f_matrix)?;

    // Transform to polar
    let (theta1, radius1, valid1) = cartesian_to_polar(positions1, n1, e1, min_radius);
    let (theta2, radius2, valid2) = cartesian_to_polar(positions2, n2, e2, min_radius);

    if theta1.is_empty() || theta2.is_empty() {
        return Some(Vec::new());
    }

    // Align theta2 onto image 1's angular frame
    let offset = compute_angle_offset(f_matrix, e1, e2);
    let theta2_aligned: Vec<f64> = theta2
        .iter()
        .map(|&t| {
            let adj = t - offset;
            adj.sin().atan2(adj.cos())
        })
        .collect();

    let sort_idx1 = angular_order(&theta1, &radius1);
    let sort_idx2 = angular_order(&theta2_aligned, &radius2);

    // Gather sorted arrays
    let sorted_theta1: Vec<f64> = sort_idx1.iter().map(|&i| theta1[i]).collect();
    let sorted_theta2: Vec<f64> = sort_idx2.iter().map(|&i| theta2_aligned[i]).collect();

    let sorted_descs1 = gather_rows(descriptors1, desc_len, &valid1, &sort_idx1);
    let sorted_descs2 = gather_rows(descriptors2, desc_len, &valid2, &sort_idx2);

    let sorted_geometry = geometric.map(|(affines1, affines2, geom, config)| SortedGeometry {
        positions1: gather_rows(positions1, 2, &valid1, &sort_idx1),
        positions2: gather_rows(positions2, 2, &valid2, &sort_idx2),
        affines1: gather_rows(affines1, 4, &valid1, &sort_idx1),
        affines2: gather_rows(affines2, 4, &valid2, &sort_idx2),
        geom,
        geom_swapped: geom.swapped(),
        config,
    });

    let forward = polar_match_one_way(
        &sorted_theta1,
        &sorted_descs1,
        &sorted_theta2,
        &sorted_descs2,
        desc_len,
        window_size,
        threshold,
        sorted_geometry
            .as_ref()
            .map(SortedGeometry::forward)
            .as_ref(),
    );

    let backward = polar_match_one_way(
        &sorted_theta2,
        &sorted_descs2,
        &sorted_theta1,
        &sorted_descs1,
        desc_len,
        window_size,
        threshold,
        sorted_geometry
            .as_ref()
            .map(SortedGeometry::backward)
            .as_ref(),
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
    polar_mutual_best_match_inner(
        positions1,
        descriptors1,
        n1,
        positions2,
        descriptors2,
        n2,
        desc_len,
        f_matrix,
        window_size,
        threshold,
        min_radius,
        None,
    )
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
    polar_mutual_best_match_inner(
        positions1,
        descriptors1,
        n1,
        positions2,
        descriptors2,
        n2,
        desc_len,
        f_matrix,
        window_size,
        threshold,
        min_radius,
        Some((affines1, affines2, geom, config)),
    )
}

#[cfg(test)]
mod tests;
