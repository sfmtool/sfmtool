// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Alternating-least-squares affine factorization (Tomasi–Kanade with missing
//! data and residual trimming) plus its metric upgrade.
//!
//! Given 2D observations of clusters across a small image group — most
//! (cluster, image) combinations unobserved, some observations junk — jointly
//! estimate an affine camera per image (`u ≈ M_i·X + t_i`), a 3D point per
//! cluster in a shared affine frame, and a per-observation keep mask. The
//! metric upgrade then finds the 3×3 gauge that makes the cameras
//! rotation-times-scale, returning both reflection hypotheses.
//!
//! Everything is deterministic: fixed round count, exact linear
//! least-squares sub-solves, no randomness, no iteration-order dependence in
//! the results. The trimming quantile uses linear interpolation between
//! order statistics (numpy's default `quantile` method) — this is
//! contractual; consumer parity depends on it.
//!
//! See `specs/core/affine-factorization.md` for the design.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Dense-init size bound on `num_images × num_clusters`. The initialization
/// builds a dense 2N×C `f64` matrix (16·N·C bytes): 64 MB at this bound,
/// mirroring the dense-covisibility budget. The intended inputs are small
/// image groups against a few thousand clusters — orders of magnitude below
/// this. [`factorize_affine`] errors with [`FactorizationError::TooLarge`]
/// above it.
pub const MAX_DENSE_ENTRIES: usize = 4_194_304;

/// Tuning for [`factorize_affine`].
#[derive(Clone, Debug)]
pub struct AffineFactorizationParams {
    /// Fixed number of alternation rounds (default 25). Rounds are numbered
    /// from 0; trimming runs from round `rounds / 2` (integer division)
    /// onward.
    pub rounds: usize,
    /// Per-trim fraction (default 0.05): each trimming round keeps the
    /// observations whose residual norm is strictly below the
    /// `(1 - trim_fraction)` quantile of the currently-kept norms.
    pub trim_fraction: f64,
}

impl Default for AffineFactorizationParams {
    fn default() -> Self {
        Self {
            rounds: 25,
            trim_fraction: 0.05,
        }
    }
}

/// Errors from [`factorize_affine`] input validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactorizationError {
    /// The observation arrays have different lengths.
    NotParallel {
        clusters: usize,
        images: usize,
        xy: usize,
    },
    /// An observation's cluster index is out of range.
    ClusterIndexOutOfRange { index: u32, num_clusters: usize },
    /// An observation's image index is out of range.
    ImageIndexOutOfRange { index: u32, num_images: usize },
    /// `num_images × num_clusters` exceeds [`MAX_DENSE_ENTRIES`].
    TooLarge {
        num_images: usize,
        num_clusters: usize,
    },
}

impl std::fmt::Display for FactorizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotParallel {
                clusters,
                images,
                xy,
            } => write!(
                f,
                "obs_clusters ({clusters}), obs_images ({images}), and obs_xy ({xy}) must be \
                 parallel arrays"
            ),
            Self::ClusterIndexOutOfRange {
                index,
                num_clusters,
            } => write!(
                f,
                "observation cluster index {index} is out of range for {num_clusters} clusters"
            ),
            Self::ImageIndexOutOfRange { index, num_images } => write!(
                f,
                "observation image index {index} is out of range for {num_images} images"
            ),
            Self::TooLarge {
                num_images,
                num_clusters,
            } => write!(
                f,
                "num_images × num_clusters ({num_images} × {num_clusters}) exceeds the dense \
                 factorization bound ({MAX_DENSE_ENTRIES}); the mean-filled 2N×C init matrix \
                 would need {} MB — intended inputs are small image groups against a few \
                 thousand clusters",
                num_images.saturating_mul(*num_clusters).saturating_mul(16) / (1024 * 1024),
            ),
        }
    }
}

impl std::error::Error for FactorizationError {}

/// Result of [`factorize_affine`]: per-image affine cameras, per-cluster 3D
/// points in a shared affine frame (defined up to an invertible 3×3 gauge),
/// and the per-observation residuals and keep mask from the final round.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineFactorization {
    /// `M_i`, per image (2×3 rows).
    pub cameras: Vec<[[f64; 3]; 2]>,
    /// `t_i`, per image.
    pub translations: Vec<[f64; 2]>,
    /// `X_c`, per cluster (affine frame).
    pub points: Vec<[f64; 3]>,
    /// `u − (M_i·X_c + t_i)` per observation, final round.
    pub residuals: Vec<[f64; 2]>,
    /// Per-observation keep mask after the final round.
    pub keep: Vec<bool>,
    /// Per image: has ≥ 4 kept observations after the final round.
    pub used_images: Vec<bool>,
}

/// One reflection hypothesis of the metric upgrade: the gauge `A` and the
/// per-image rotation/scale decomposition of `M_i·A` (identity rotation and
/// zero scale where the image is unused).
#[derive(Debug, Clone, PartialEq)]
pub struct MetricHypothesis {
    /// The gauge `A`.
    pub gauge: [[f64; 3]; 3],
    /// Per-image rotations; identity where unused.
    pub rotations: Vec<[[f64; 3]; 3]>,
    /// Per-image scales; 0 where unused.
    pub scales: Vec<f64>,
}

/// The `(1 - trim_fraction)`-style quantile with linear interpolation
/// between order statistics, bit-matching numpy's default `quantile` method
/// (including its `t >= 0.5` lerp branch). `sorted` must be ascending and
/// non-empty; `q` in `[0, 1]`.
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let h = q * (n - 1) as f64;
    let lo = h.floor() as usize;
    let t = h - lo as f64;
    if lo + 1 >= n {
        return sorted[n - 1];
    }
    let (a, b) = (sorted[lo], sorted[lo + 1]);
    // numpy's _lerp: the t >= 0.5 branch computes from `b` for accuracy.
    if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// Exact linear least-squares solve of `a·x = b` (any shape, multiple RHS)
/// via SVD — deterministic, minimum-norm, matching `numpy.linalg.lstsq`.
fn lstsq(a: DMatrix<f64>, b: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let svd = a.svd(true, true);
    svd.solve(b, 0.0).ok()
}

/// Alternating-least-squares affine factorization with trimming; see the
/// module docs and `specs/core/affine-factorization.md` § Algorithm.
///
/// Inputs are parallel per-observation arrays (no ordering required):
/// cluster index, image index, and centered 2D position. `params.rounds`
/// alternation rounds run over a mean-filled dense SVD initialization;
/// trimming starts at round `rounds / 2` (rounds numbered from 0).
pub fn factorize_affine(
    obs_clusters: &[u32],
    obs_images: &[u32],
    obs_xy: &[[f64; 2]],
    num_images: usize,
    num_clusters: usize,
    params: &AffineFactorizationParams,
) -> Result<AffineFactorization, FactorizationError> {
    let k = obs_clusters.len();
    if obs_images.len() != k || obs_xy.len() != k {
        return Err(FactorizationError::NotParallel {
            clusters: k,
            images: obs_images.len(),
            xy: obs_xy.len(),
        });
    }
    if let Some(&bad) = obs_clusters.iter().find(|&&c| c as usize >= num_clusters) {
        return Err(FactorizationError::ClusterIndexOutOfRange {
            index: bad,
            num_clusters,
        });
    }
    if let Some(&bad) = obs_images.iter().find(|&&i| i as usize >= num_images) {
        return Err(FactorizationError::ImageIndexOutOfRange {
            index: bad,
            num_images,
        });
    }
    if num_images
        .checked_mul(num_clusters)
        .is_none_or(|prod| prod > MAX_DENSE_ENTRIES)
    {
        return Err(FactorizationError::TooLarge {
            num_images,
            num_clusters,
        });
    }

    // ── Initialization: mean-filled dense 2N×C + top-3 right singular
    //    vectors as X ──────────────────────────────────────────────────────
    let rows = 2 * num_images;
    let mut w = DMatrix::from_element(rows, num_clusters, f64::NAN);
    for o in 0..k {
        let r = 2 * obs_images[o] as usize;
        let c = obs_clusters[o] as usize;
        w[(r, c)] = obs_xy[o][0];
        w[(r + 1, c)] = obs_xy[o][1];
    }
    // Center each row on the mean of its observed entries (a fully-missing
    // row centers on 0); missing entries become 0 — equivalent to filling
    // them with the row mean and then subtracting it.
    for r in 0..rows {
        let (mut sum, mut cnt) = (0.0, 0usize);
        for c in 0..num_clusters {
            let v = w[(r, c)];
            if !v.is_nan() {
                sum += v;
                cnt += 1;
            }
        }
        let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
        for c in 0..num_clusters {
            let v = w[(r, c)];
            w[(r, c)] = if v.is_nan() { 0.0 } else { v - mean };
        }
    }
    let svd = w.svd(false, true);
    let v_t = svd.v_t.expect("v_t requested");
    // Defensive descending order (nalgebra's `svd` sorts, but the top-3
    // choice is contractual, so don't rely on it).
    let mut order: Vec<usize> = (0..svd.singular_values.len()).collect();
    order.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
    let mut points = vec![[0.0f64; 3]; num_clusters];
    for (axis, &src) in order.iter().take(3).enumerate() {
        for c in 0..num_clusters {
            points[c][axis] = v_t[(src, c)];
        }
    }

    let mut cameras = vec![[[0.0f64; 3]; 2]; num_images];
    let mut translations = vec![[0.0f64; 2]; num_images];
    // With zero initial cameras, the pre-round residual is the observation
    // itself; this is only visible when `rounds == 0`.
    let mut residuals: Vec<[f64; 2]> = obs_xy.to_vec();
    let mut keep = vec![true; k];

    // Per-image / per-cluster observation index lists (built once).
    let mut obs_by_image: Vec<Vec<usize>> = vec![Vec::new(); num_images];
    let mut obs_by_cluster: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
    for o in 0..k {
        obs_by_image[obs_images[o] as usize].push(o);
        obs_by_cluster[obs_clusters[o] as usize].push(o);
    }

    for round in 0..params.rounds {
        // a. Camera sweep: per image, fit (M_i | t_i) over kept observations;
        //    fewer than 4 kept keeps the previous values.
        for i in 0..num_images {
            let kept: Vec<usize> = obs_by_image[i]
                .iter()
                .copied()
                .filter(|&o| keep[o])
                .collect();
            if kept.len() < 4 {
                continue;
            }
            let mut a = DMatrix::zeros(kept.len(), 4);
            let mut b = DMatrix::zeros(kept.len(), 2);
            for (r, &o) in kept.iter().enumerate() {
                let x = points[obs_clusters[o] as usize];
                a[(r, 0)] = x[0];
                a[(r, 1)] = x[1];
                a[(r, 2)] = x[2];
                a[(r, 3)] = 1.0;
                b[(r, 0)] = obs_xy[o][0];
                b[(r, 1)] = obs_xy[o][1];
            }
            if let Some(p) = lstsq(a, &b) {
                for row in 0..2 {
                    for col in 0..3 {
                        cameras[i][row][col] = p[(col, row)];
                    }
                    translations[i][row] = p[(3, row)];
                }
            }
        }

        // b. Point sweep: per cluster, fit X_c over kept observations; fewer
        //    than 2 kept keeps the previous values.
        for c in 0..num_clusters {
            let kept: Vec<usize> = obs_by_cluster[c]
                .iter()
                .copied()
                .filter(|&o| keep[o])
                .collect();
            if kept.len() < 2 {
                continue;
            }
            let mut a = DMatrix::zeros(2 * kept.len(), 3);
            let mut b = DMatrix::zeros(2 * kept.len(), 1);
            for (r, &o) in kept.iter().enumerate() {
                let i = obs_images[o] as usize;
                for row in 0..2 {
                    for col in 0..3 {
                        a[(2 * r + row, col)] = cameras[i][row][col];
                    }
                    b[(2 * r + row, 0)] = obs_xy[o][row] - translations[i][row];
                }
            }
            if let Some(x) = lstsq(a, &b) {
                points[c] = [x[(0, 0)], x[(1, 0)], x[(2, 0)]];
            }
        }

        // c. Residuals for every observation.
        for o in 0..k {
            let i = obs_images[o] as usize;
            let x = points[obs_clusters[o] as usize];
            for row in 0..2 {
                let pred = cameras[i][row][0] * x[0]
                    + cameras[i][row][1] * x[1]
                    + cameras[i][row][2] * x[2]
                    + translations[i][row];
                residuals[o][row] = obs_xy[o][row] - pred;
            }
        }

        // d. Trimming, from round `rounds / 2` onward: the kept set becomes
        //    the observations whose residual norm is strictly below the
        //    (1 - trim_fraction) quantile of the currently-kept norms.
        if round >= params.rounds / 2 {
            let mut kept_norms: Vec<f64> = (0..k)
                .filter(|&o| keep[o])
                .map(|o| residuals[o][0].hypot(residuals[o][1]))
                .collect();
            if !kept_norms.is_empty() {
                kept_norms.sort_by(f64::total_cmp);
                let thr = quantile_linear(&kept_norms, 1.0 - params.trim_fraction);
                for (o, kept) in keep.iter_mut().enumerate() {
                    *kept = residuals[o][0].hypot(residuals[o][1]) < thr;
                }
            }
        }
    }

    let used_images: Vec<bool> = obs_by_image
        .iter()
        .map(|obs| obs.iter().filter(|&&o| keep[o]).count() >= 4)
        .collect();

    Ok(AffineFactorization {
        cameras,
        translations,
        points,
        residuals,
        keep,
        used_images,
    })
}

/// Coefficients of `aᵀ·Q·b` as a linear form in the 6 unknowns of the
/// symmetric `Q`, ordered `(q11, q12, q13, q22, q23, q33)`.
fn qform_coeffs(a: &[f64; 3], b: &[f64; 3]) -> [f64; 6] {
    [
        a[0] * b[0],
        a[0] * b[1] + a[1] * b[0],
        a[0] * b[2] + a[2] * b[0],
        a[1] * b[1],
        a[1] * b[2] + a[2] * b[1],
        a[2] * b[2],
    ]
}

/// SVD orthonormalization of a 3×3 matrix with determinant corrected to +1.
fn nearest_rotation(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let u = svd.u.expect("u requested");
    let v_t = svd.v_t.expect("v_t requested");
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        // Flip the least-significant singular direction.
        let mut u = u;
        for row in 0..3 {
            u[(row, 2)] = -u[(row, 2)];
        }
        r = u * v_t;
    }
    r
}

/// Metric upgrade of an affine factorization: solve for the symmetric
/// `Q = A·Aᵀ` that makes each used camera's rows equal-norm and orthogonal,
/// then decompose `M_i·A` into rotation × scale. Returns both reflection
/// hypotheses (`A` and `A·diag(1, 1, −1)`), or `None` when no image is used
/// or the constraint system is degenerate. See
/// `specs/core/affine-factorization.md` § Metric upgrade.
pub fn metric_upgrade(factorization: &AffineFactorization) -> Option<[MetricHypothesis; 2]> {
    let num_images = factorization.cameras.len();
    let used: Vec<usize> = (0..num_images)
        .filter(|&i| factorization.used_images[i])
        .collect();
    if used.is_empty() {
        return None;
    }

    // ── Linear LSQ on the 6 unknowns of symmetric Q ───────────────────────
    let mut a = DMatrix::zeros(2 * used.len() + 1, 6);
    let mut b = DVector::zeros(2 * used.len() + 1);
    let mut norm_row = [0.0f64; 6];
    for (r, &i) in used.iter().enumerate() {
        let m1 = &factorization.cameras[i][0];
        let m2 = &factorization.cameras[i][1];
        let c11 = qform_coeffs(m1, m1);
        let c22 = qform_coeffs(m2, m2);
        let c12 = qform_coeffs(m1, m2);
        for col in 0..6 {
            a[(2 * r, col)] = c11[col] - c22[col]; // equal-norm rows
            a[(2 * r + 1, col)] = c12[col]; // orthogonal rows
            norm_row[col] += (c11[col] + c22[col]) / used.len() as f64;
        }
    }
    let last = 2 * used.len();
    for col in 0..6 {
        a[(last, col)] = norm_row[col];
    }
    b[last] = 2.0; // mean of m1ᵀQm1 + m2ᵀQm2 set to 2

    let svd = a.svd(true, true);
    let s_max = svd.singular_values.iter().cloned().fold(0.0f64, f64::max);
    let s_min = svd
        .singular_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    if svd.singular_values.len() < 6 || s_min.is_nan() || s_min <= 1e-10 * s_max {
        return None; // degenerate constraint system
    }
    let q = svd.solve(&b, 0.0).ok()?;
    let q_mat = Matrix3::new(q[0], q[1], q[2], q[1], q[3], q[4], q[2], q[4], q[5]);

    // ── A = V·√Λ with eigenvalues clamped to a 1e-8·λ_max floor ──────────
    let eig = q_mat.symmetric_eigen();
    let mut order = [0usize, 1, 2];
    order.sort_by(|&x, &y| eig.eigenvalues[y].total_cmp(&eig.eigenvalues[x]));
    let lambda_max = eig.eigenvalues[order[0]];
    if lambda_max.is_nan() || lambda_max <= 0.0 {
        return None;
    }
    let floor = 1e-8 * lambda_max;
    let mut gauge = Matrix3::zeros();
    for (col, &src) in order.iter().enumerate() {
        let lambda = eig.eigenvalues[src].max(floor);
        let s = lambda.sqrt();
        for row in 0..3 {
            gauge[(row, col)] = eig.eigenvectors[(row, src)] * s;
        }
    }

    // ── Both reflection hypotheses ────────────────────────────────────────
    let hypothesis = |reflect: bool| -> MetricHypothesis {
        let mut a = gauge;
        if reflect {
            for row in 0..3 {
                a[(row, 2)] = -a[(row, 2)];
            }
        }
        let mut rotations = vec![[[0.0f64; 3]; 3]; num_images];
        for r in rotations.iter_mut() {
            *r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }
        let mut scales = vec![0.0f64; num_images];
        for &i in &used {
            let cam = &factorization.cameras[i];
            let m1 = a.transpose() * Vector3::new(cam[0][0], cam[0][1], cam[0][2]);
            let m2 = a.transpose() * Vector3::new(cam[1][0], cam[1][1], cam[1][2]);
            let s = (m1.norm() + m2.norm()) / 2.0;
            scales[i] = s;
            let cross = m1.cross(&m2);
            if s.is_nan() || s <= 0.0 || cross.norm() == 0.0 {
                continue; // degenerate camera: identity rotation
            }
            let r3 = cross / cross.norm();
            let stack =
                Matrix3::from_rows(&[(m1 / s).transpose(), (m2 / s).transpose(), r3.transpose()]);
            let r = nearest_rotation(&stack);
            for row in 0..3 {
                for col in 0..3 {
                    rotations[i][row][col] = r[(row, col)];
                }
            }
        }
        let mut gauge_out = [[0.0f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                gauge_out[row][col] = a[(row, col)];
            }
        }
        MetricHypothesis {
            gauge: gauge_out,
            rotations,
            scales,
        }
    };

    Some([hypothesis(false), hypothesis(true)])
}

#[cfg(test)]
mod tests;
