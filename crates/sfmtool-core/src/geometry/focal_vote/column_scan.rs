// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera-model columns of the structure-free focal vote
//! (see `specs/core/geometry/focal-vote.md`, "Camera-Model Columns").
//!
//! A **column** is a camera-model hypothesis supplying an invertible pixel→ray
//! map parameterized by its own focal ([`CameraModel`]). Both estimator
//! families of [`super::focal_vote`] generalize over the camera model through
//! that map, and each family becomes a **cell** that scans candidate focals for
//! self-consistency:
//!
//! - **Epipolar cell** — per candidate focal, map the pair's correspondences to
//!   unit rays and robustly estimate the ray-space epipolar matrix; the cost is
//!   the essentialness residual `(σ₁ − σ₂)/(σ₁ + σ₂)` of the consensus refit.
//!   The two correspondence directions are scanned separately with **one-sided**
//!   residuals (image-2 rays against `E·x₁`, image-1 rays against `Eᵀ·x₂`), and
//!   their minima must agree within the direction band. A symmetric residual
//!   would make that certificate vacuous: the epipolar matrix of the swapped
//!   correspondences is exactly the transpose, with identical singular values.
//! - **Rotation cell** — per candidate focal, fit a rotation of rays directly
//!   (robust orthogonal fit); the cost is the fit's trimmed RMS angular
//!   residual. The inlier support is frozen **once per pair** — both maps shrink
//!   every ray angle as `1/f`, so a per-candidate support would let a bad focal
//!   buy a low cost by keeping fewer points and the scan would pin at the top of
//!   the grid instead of showing an interior minimum.
//!
//! Estimation and residuals live on the ray **sphere** throughout: an
//! equidistant field of view can exceed 180°, and rays with `θ ≥ 90°` have no
//! planar projection at all, yet are exactly the model-informative ones. The
//! RANSAC consensus bound is an angle whose value derives per candidate focal
//! from a pixel tolerance through the map's local scale `dr/dθ`; a fixed
//! angular threshold does not transfer across lenses and resolutions.
//!
//! Everything here is seeded and deterministic: the minimal-sample index sets
//! are drawn once per candidate pair from the input seed and reused at every
//! candidate focal and in every column, so the cost curves carry no RANSAC
//! jitter and the columns are directly comparable.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use rayon::prelude::*;

use crate::geometry::numeric::splitmix64;

// ── Scan configuration (see the spec's Camera-Model Columns section) ─────────

/// Scan band, as a multiple of `max(width, height)`. The low end reaches the
/// focals a beyond-180° field of view implies (`f = r_edge / θ_edge`), well
/// below the pinhole plausibility band's `0.2` floor; both columns scan the
/// same band so their certificates are comparable.
const SCAN_BAND_LO: f64 = 0.075;
/// Upper end of the scan band, as a multiple of `max(width, height)`.
const SCAN_BAND_HI: f64 = 3.0;
/// Log-spaced candidate focals per scan.
const SCAN_GRID_N: usize = 64;
/// Minimal samples per cell, drawn once per pair and reused at every candidate.
const SCAN_SAMPLES: usize = 128;
/// Correspondences per pair fed to the scans (deterministically subsampled).
const SCAN_MAX_CORR: usize = 600;
/// Keypoint localization tolerance the angular consensus bound derives from.
const SCAN_TOL_PX: f64 = 3.0;
/// Minimal-sample size of the epipolar cell (linear 8-point constraint).
const EPI_SAMPLE: usize = 8;
/// Minimal-sample size of the rotation cell (three rays fix a rotation).
const ROT_SAMPLE: usize = 3;
/// Local-optimization rounds on the consensus set.
const LO_ROUNDS: usize = 3;
/// Inliers an epipolar consensus needs.
const EPI_MIN_INLIERS: usize = 8;
/// Far-field correspondences a rotation support set needs.
const ROT_MIN_SUPPORT: usize = 20;
/// Quantile of the frozen support kept by the rotation fit's trimming.
const ROT_TRIM_Q: f64 = 0.90;
/// Trimming rounds inside the frozen support.
const ROT_TRIM_ROUNDS: usize = 2;
/// Every fourth grid point takes part in the support-freezing pass.
const ROT_SUPPORT_STRIDE: usize = 4;
/// Grid points a curve needs before its minimum is meaningful.
const MIN_CURVE_POINTS: usize = 5;

/// Credible half-field-of-view window, degrees: `f = r_edge / θ_edge` turns it
/// into a focal window at the pair's own edge radius. Band containment is a
/// model-evidence covariate, not a gate.
const FOV_HALF_LO_DEG: f64 = 50.0;
/// Upper end of the credible half-field-of-view window, degrees.
const FOV_HALF_HI_DEG: f64 = 110.0;

/// Essentialness-residual validity floor of the epipolar cell — a pair whose
/// best residual stays above it has no essential explanation at any focal.
const ESSENTIALNESS_FLOOR: f64 = 0.03;
/// Trimmed-RMS angular validity floor of the rotation cell, radians.
const ROTATION_FIT_FLOOR_RAD: f64 = 0.02;
/// Shape gate: `2·cost(f*)` must not exceed the median cost over the grid, i.e.
/// `cost(f*) / median ≤ 0.5` — a flat scan has no focal opinion.
const SHAPE_GATE: f64 = 0.5;
/// Maximum `|ln(f_side2 / f_side1)|` for a pair's two one-sided scans to count
/// as two measurements of the same focal (the pinhole cell's band, unchanged).
const DIRECTION_BAND: f64 = 0.05;
/// Radial-coverage floor, as a fraction of the half-diagonal at the vote's
/// inliers' radial p90. Below it the pinhole and equidistant maps agree to
/// first order and the vote cannot discriminate the columns, so it is excluded
/// from the model verdict (it still enters its column's focal pool).
const COVERAGE_FLOOR: f64 = 0.50;
/// Rotation-domination gate: absolute floor on the rotation consensus, the
/// analog of the pinhole family's `n_H ≥ 16`.
const ROTATION_DOMINATION_MIN: usize = 16;
/// Rotation-domination gate: fraction of the best essential consensus the
/// rotation consensus must reach, the analog of `n_H ≥ 0.8 · n_F`.
const ROTATION_DOMINATION_FRAC: f64 = 0.8;
/// Essential consensus a pair needs before its rotation/essential ratio counts
/// toward the column's parallax-poverty diagnostic.
const RATIO_MIN_ESSENTIAL: usize = 16;

// ── Columns ──────────────────────────────────────────────────────────────────

/// A camera-model hypothesis: an invertible pixel→unit-ray map parameterized by
/// its own focal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CameraModel {
    /// `ray ∝ ((x − cx)/f, (y − cy)/f, 1)`.
    Pinhole,
    /// `θ = r/f`, `ray = (sin θ cos φ, sin θ sin φ, cos θ)`.
    EquidistantFisheye,
}

impl CameraModel {
    /// Stable string name for the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            CameraModel::Pinhole => "Pinhole",
            CameraModel::EquidistantFisheye => "EquidistantFisheye",
        }
    }

    /// Parse the binding's column name (case-insensitive, `_`/`-` insensitive).
    pub fn from_str_name(s: &str) -> Option<Self> {
        let norm: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        match norm.as_str() {
            "pinhole" => Some(CameraModel::Pinhole),
            "equidistant" | "equidistantfisheye" | "fisheye" => {
                Some(CameraModel::EquidistantFisheye)
            }
            _ => None,
        }
    }

    /// Unit ray for a principal-point-centred pixel at candidate focal `f`.
    ///
    /// `r` is the pixel's radial distance `hypot(uv)`. It does not depend on the
    /// candidate focal, so the scans precompute it once per pair
    /// ([`ScanCandidate::rad1`]) instead of rebuilding it at each of the grid's
    /// candidate focals.
    fn ray(self, uv: [f64; 2], r: f64, f: f64) -> Vector3<f64> {
        match self {
            CameraModel::Pinhole => Vector3::new(uv[0] / f, uv[1] / f, 1.0).normalize(),
            CameraModel::EquidistantFisheye => {
                let th = r / f;
                let s = if r > 1e-12 { th.sin() / r } else { 0.0 };
                Vector3::new(uv[0] * s, uv[1] * s, th.cos())
            }
        }
    }

    /// Local `dr/dθ` of the map at radius `r` — the pixels-per-radian that turns
    /// a keypoint localization tolerance into an angular consensus bound. Both
    /// maps are radially symmetric, so the radius is all a pixel contributes.
    fn scale(self, r: f64, f: f64) -> f64 {
        match self {
            CameraModel::Pinhole => f * (1.0 + (r / f) * (r / f)),
            CameraModel::EquidistantFisheye => f,
        }
    }

    /// Whether a candidate focal keeps the map injective over radii up to
    /// `r_hi`: the equidistant map folds once `θ = r/f` passes `π`; the pinhole
    /// map images every pixel at every positive focal.
    fn admits(self, f: f64, r_hi: f64) -> bool {
        match self {
            CameraModel::Pinhole => f > 0.0,
            CameraModel::EquidistantFisheye => f > 0.0 && r_hi / f < std::f64::consts::PI,
        }
    }
}

/// Which cell of a column produced a scan vote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanCell {
    /// Ray-space epipolar essentialness scan (parallax-rich pairs).
    Epipolar,
    /// Robust ray-space rotation fit over a frozen support (far-field pairs).
    Rotation,
}

// ── Scan output ──────────────────────────────────────────────────────────────

/// One candidate pair's scan through one cell of one column.
#[derive(Clone, Copy, Debug)]
pub struct ScanVote {
    /// Which cell produced this scan.
    pub cell: ScanCell,
    /// First image of the candidate pair.
    pub image_a: u32,
    /// Second image of the candidate pair.
    pub image_b: u32,
    /// The scan's minimizing focal in pixels (parabolically refined).
    pub focal_px: f64,
    /// The gated cost at the minimum: the essentialness residual for the
    /// epipolar cell, the trimmed RMS angular residual (radians) for the
    /// rotation cell.
    pub cost: f64,
    /// `cost(f*) / median cost over the grid` — the shape metric.
    pub sharpness: f64,
    /// `|ln(f_side2 / f_side1)|` of the two one-sided scans (epipolar cell
    /// only; `None` for the rotation cell or when a side produced no minimum).
    /// Recorded whether or not the pair was gated, so a caller can reconstruct
    /// exactly which votes the rotation-domination gate removed.
    pub dir_disagreement: Option<f64>,
    /// Epipolar cell only: the pair's best ray-rotation consensus is at least
    /// `max(16, 0.8 × best essential consensus)`, so a rotation explains it and
    /// it casts no epipolar vote — the analog of the pinhole family's
    /// homography domination.
    pub rotation_dominated: bool,
    /// Epipolar cell only: best rotation consensus over the best essential
    /// consensus, the analog of `n_H / n_F`. `None` for the rotation cell and
    /// for pairs whose essential consensus is below `16`.
    pub rotation_ratio: Option<f64>,
    /// Radial p90 of the vote's inliers, as a fraction of the half-diagonal.
    pub coverage_p90: f64,
    /// Inliers supporting the vote.
    pub n_inliers: usize,
    /// Whether the unconstrained minimum falls inside the credible half-FOV
    /// window at the pair's own edge radius (a model-evidence covariate).
    pub in_fov_band: bool,
    /// Whether the minimum sits at an end of the scan band instead of being
    /// interior — a scan that pins at the grid edge has not located a focal and
    /// is never certified.
    pub at_grid_edge: bool,
    /// Rotation cell only: where the RAW angular cost curve bottoms out, as
    /// against the pixel-scaled curve the vote reads. The two agree to a
    /// fraction of a percent when the support is frozen; with a per-candidate
    /// support the angular minimum slides to the top of the grid, which is what
    /// makes freezing load-bearing. `None` for the epipolar cell, whose
    /// essentialness cost is already dimensionless.
    pub angular_focal_px: Option<f64>,
    /// Whether the vote's own geometry certifies it (floor, shape, edge and —
    /// for the epipolar cell — direction agreement).
    pub certified: bool,
    /// Whether a certified vote also clears the radial-coverage floor and so
    /// counts toward the model verdict.
    pub model_informative: bool,
}

/// The scans of one column over one capture's candidate pairs.
#[derive(Clone, Debug)]
pub struct ColumnScan {
    /// The column's camera model.
    pub model: CameraModel,
    /// Epipolar-cell scans, one per epipolar candidate pair that produced a
    /// curve.
    pub epipolar: Vec<ScanVote>,
    /// Rotation-cell scans, one per rotation candidate pair that produced a
    /// curve.
    pub rotation: Vec<ScanVote>,
}

impl ColumnScan {
    /// Epipolar candidate pairs gated out as rotation-dominated — the analog of
    /// the pinhole family's `n_h_dominated`.
    pub fn n_rotation_dominated(&self) -> usize {
        self.epipolar
            .iter()
            .filter(|v| v.rotation_dominated)
            .count()
    }

    /// Median rotation/essential consensus ratio over the epipolar candidate
    /// pairs with at least `16` essential inliers — this column's counterpart
    /// of `parallax_poverty`. High poverty means most correspondences are
    /// explained by a rotation alone, the regime where the column's pool should
    /// be rotation-dominated. `0` with no qualifying pair.
    pub fn parallax_poverty(&self) -> f64 {
        let ratios: Vec<f64> = self
            .epipolar
            .iter()
            .filter_map(|v| v.rotation_ratio)
            .collect();
        if ratios.is_empty() {
            0.0
        } else {
            quantile(&ratios, 0.5)
        }
    }

    /// Certified votes over both cells.
    pub fn n_certified(&self) -> usize {
        self.epipolar.iter().filter(|v| v.certified).count()
            + self.rotation.iter().filter(|v| v.certified).count()
    }

    /// Certified **and** model-informative votes over both cells — the mass the
    /// model verdict compares.
    pub fn n_informative(&self) -> usize {
        self.epipolar.iter().filter(|v| v.model_informative).count()
            + self.rotation.iter().filter(|v| v.model_informative).count()
    }

    /// Certified focals of one cell, in scan order.
    pub fn certified_focals(&self, cell: ScanCell) -> Vec<f64> {
        let src = match cell {
            ScanCell::Epipolar => &self.epipolar,
            ScanCell::Rotation => &self.rotation,
        };
        src.iter()
            .filter(|v| v.certified)
            .map(|v| v.focal_px)
            .collect()
    }
}

/// One candidate pair handed to the scans: the two images and their
/// principal-point-centred correspondences (already capped and deterministic).
#[derive(Clone, Debug)]
pub struct ScanCandidate {
    /// First image of the pair.
    pub image_a: u32,
    /// Second image of the pair.
    pub image_b: u32,
    /// Centred positions in image `a`.
    pub uv1: Vec<[f64; 2]>,
    /// Centred positions in image `b`.
    pub uv2: Vec<[f64; 2]>,
    /// Radial distance of each `uv1` position from the principal point.
    ///
    /// Both pixel→ray maps are radially symmetric and the radius is not a
    /// function of the candidate focal, so it is computed once here and read at
    /// every grid point, in both cells and in both columns, rather than
    /// rebuilt from the pixels 64 times per scan.
    pub rad1: Vec<f64>,
    /// Radial distance of each `uv2` position from the principal point.
    pub rad2: Vec<f64>,
    /// Per-pair sampler seed (derived from the kernel seed and the pair's
    /// position in the candidate list).
    pub seed: u64,
}

impl ScanCandidate {
    /// Build a candidate from correspondences already centred on the principal
    /// point, filling in the focal-independent radii.
    pub fn from_centred(
        image_a: u32,
        image_b: u32,
        uv1: Vec<[f64; 2]>,
        uv2: Vec<[f64; 2]>,
        seed: u64,
    ) -> Self {
        let rad = |uv: &[[f64; 2]]| -> Vec<f64> { uv.iter().map(|p| p[0].hypot(p[1])).collect() };
        Self {
            image_a,
            image_b,
            rad1: rad(&uv1),
            rad2: rad(&uv2),
            uv1,
            uv2,
            seed,
        }
    }

    /// Build a candidate from a pair's full correspondence lists, centring on
    /// the principal point and capping the population at `SCAN_MAX_CORR` by a
    /// seeded selection that preserves input order.
    pub fn new(
        image_a: u32,
        image_b: u32,
        x1: &[[f64; 2]],
        x2: &[[f64; 2]],
        pp: [f64; 2],
        seed: u64,
    ) -> Self {
        let n = x1.len().min(x2.len());
        let keep: Vec<usize> = if n <= SCAN_MAX_CORR {
            (0..n).collect()
        } else {
            // Partial Fisher-Yates over the index list, then sorted back into
            // input order so the population is a deterministic subsample.
            let mut idx: Vec<usize> = (0..n).collect();
            let mut state = seed;
            for i in 0..SCAN_MAX_CORR {
                let j = i + (splitmix64(&mut state) % (n - i) as u64) as usize;
                idx.swap(i, j);
            }
            let mut k = idx[..SCAN_MAX_CORR].to_vec();
            k.sort_unstable();
            k
        };
        Self::from_centred(
            image_a,
            image_b,
            keep.iter()
                .map(|&i| [x1[i][0] - pp[0], x1[i][1] - pp[1]])
                .collect(),
            keep.iter()
                .map(|&i| [x2[i][0] - pp[0], x2[i][1] - pp[1]])
                .collect(),
            seed,
        )
    }
}

// ── Small numeric helpers ────────────────────────────────────────────────────

/// Linear-interpolated quantile of an already-sorted, non-empty slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let t = p * (sorted.len() - 1) as f64;
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (t - lo as f64)
}

/// Linear-interpolated quantile of an unsorted, non-empty slice.
fn quantile(vals: &[f64], p: f64) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(f64::total_cmp);
    quantile_sorted(&v, p)
}

/// Smallest eigenvector of `AᵀA` over the selected rows, reshaped row-major
/// into a `3×3` matrix. `None` for a design that carries no constraint.
pub(crate) fn null_from_rows(
    rows: &[SVector<f64, 9>],
    idx: impl Iterator<Item = usize>,
) -> Option<Matrix3<f64>> {
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for i in idx {
        let r = &rows[i];
        ata += r * r.transpose();
    }
    if ata.iter().cloned().fold(0.0, f64::max) <= 0.0 {
        return None;
    }
    let eig = ata.symmetric_eigen();
    let mut best = 0usize;
    for j in 1..9 {
        if eig.eigenvalues[j] < eig.eigenvalues[best] {
            best = j;
        }
    }
    let v = eig.eigenvectors.column(best).into_owned();
    let m = Matrix3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    m.iter().all(|x| x.is_finite()).then_some(m)
}

/// Rotation taking `r1` onto `r2` in the least-squares sense (orthogonal
/// Procrustes with a reflection guard): `R r1ᵢ ≈ r2ᵢ`.
pub(crate) fn kabsch(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    idx: &[usize],
) -> Option<Matrix3<f64>> {
    let mut m = Matrix3::zeros();
    for &i in idx {
        m += r2[i] * r1[i].transpose();
    }
    let svd = m.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let d = (u * v_t).determinant().signum();
    let rot = u * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d) * v_t;
    rot.iter().all(|x| x.is_finite()).then_some(rot)
}

/// Descending singular values of a `3×3` matrix.
pub(crate) fn singular_values_desc(m: &Matrix3<f64>) -> [f64; 3] {
    let sv = m.svd(false, false).singular_values;
    let mut s = [sv[0], sv[1], sv[2]];
    s.sort_by(|a, b| b.total_cmp(a));
    s
}

// ── Cost curves ──────────────────────────────────────────────────────────────

/// Location and shape of a cost curve's minimum over the grid.
#[derive(Clone, Copy, Debug)]
struct CurveMinimum {
    /// Grid index of the minimum.
    k: usize,
    /// Parabolically refined focal in pixels.
    focal_px: f64,
    /// Cost at the minimum.
    cost_min: f64,
    /// `cost_min / median cost over the valid grid points`.
    sharpness: f64,
    /// Whether the minimum sits at an end of the valid range.
    edge: bool,
}

/// Minimum of `costs` over the grid, with parabolic refinement of the winning
/// bracket in `log f`. `None` with fewer than [`MIN_CURVE_POINTS`] finite costs.
fn curve_minimum(costs: &[f64], grid: &[f64]) -> Option<CurveMinimum> {
    let idx: Vec<usize> = (0..costs.len()).filter(|&k| costs[k].is_finite()).collect();
    if idx.len() < MIN_CURVE_POINTS {
        return None;
    }
    let mut k = idx[0];
    for &j in &idx {
        if costs[j] < costs[k] {
            k = j;
        }
    }
    let vals: Vec<f64> = idx.iter().map(|&j| costs[j]).collect();
    let med = quantile(&vals, 0.5);
    let cost_min = costs[k];
    let (first, last) = (idx[0], idx[idx.len() - 1]);
    let mut focal_px = grid[k];
    if first < k && k < last && costs[k - 1].is_finite() && costs[k + 1].is_finite() {
        let (y0, y1, y2) = (costs[k - 1], costs[k], costs[k + 1]);
        let den = y0 - 2.0 * y1 + y2;
        if den > 0.0 {
            let step = 0.5 * (y0 - y2) / den;
            if step.abs() <= 1.0 {
                focal_px = (grid[k].ln() + step * (grid[k + 1].ln() - grid[k].ln())).exp();
            }
        }
    }
    Some(CurveMinimum {
        k,
        focal_px,
        cost_min,
        sharpness: if med > 0.0 { cost_min / med } else { 1.0 },
        edge: k == first || k == last,
    })
}

// ── Epipolar cell ────────────────────────────────────────────────────────────

/// One robust ray-space epipolar fit at one candidate focal.
struct EpipolarFit {
    /// Essentialness residual `(σ₁ − σ₂)/(σ₁ + σ₂)` of the consensus refit.
    cost: f64,
    /// Consensus mask over the pair's correspondences.
    inliers: Vec<bool>,
}

/// Angular epipolar residual of every correspondence against `e`, one-sided.
///
/// `side_two` measures the angle between each image-2 ray and the epipolar
/// plane `E·x₁`; otherwise the image-1 ray is measured against `Eᵀ·x₂`. The two
/// are genuinely different measurements — a symmetric residual would score the
/// swapped correspondences identically, because the epipolar matrix of the swap
/// is exactly the transpose.
pub(crate) fn epipolar_residuals(
    e: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    side_two: bool,
    out: &mut [f64],
) {
    let et = e.transpose();
    for i in 0..r1.len() {
        let (n, other) = if side_two {
            (e * r1[i], r2[i])
        } else {
            (et * r2[i], r1[i])
        };
        let nn = n.norm().max(1e-15);
        out[i] = (n.dot(&other).abs() / nn).min(1.0);
    }
}

/// Design rows of the linear epipolar constraint `x₂ᵀ E x₁ = 0`, one row per
/// correspondence, `E` flattened row-major.
pub(crate) fn epipolar_rows(r1: &[Vector3<f64>], r2: &[Vector3<f64>]) -> Vec<SVector<f64, 9>> {
    (0..r1.len().min(r2.len()))
        .map(|i| {
            let (a, b) = (&r2[i], &r1[i]);
            SVector::<f64, 9>::from_column_slice(&[
                a[0] * b[0],
                a[0] * b[1],
                a[0] * b[2],
                a[1] * b[0],
                a[1] * b[1],
                a[1] * b[2],
                a[2] * b[0],
                a[2] * b[1],
                a[2] * b[2],
            ])
        })
        .collect()
}

/// Robust ray-space epipolar matrix at one candidate focal: score the frozen
/// minimal samples by the one-sided angular residual, then locally optimize on
/// the consensus set. `sin_tol[i]` is the per-point consensus bound.
fn fit_epipolar(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    sin_tol: &[f64],
    samples: &[usize],
    side_two: bool,
) -> Option<EpipolarFit> {
    let n = r1.len();
    let rows = epipolar_rows(r1, r2);

    let mut resid = vec![0.0f64; n];
    let mut best_count = 0usize;
    let mut best_e: Option<Matrix3<f64>> = None;
    let mut best_mask = vec![false; n];
    for s in samples.chunks_exact(EPI_SAMPLE) {
        let Some(e) = null_from_rows(&rows, s.iter().copied()) else {
            continue;
        };
        epipolar_residuals(&e, r1, r2, side_two, &mut resid);
        let count = (0..n).filter(|&i| resid[i] < sin_tol[i]).count();
        if count > best_count {
            best_count = count;
            best_e = Some(e);
            for i in 0..n {
                best_mask[i] = resid[i] < sin_tol[i];
            }
        }
    }
    if best_count < EPI_MIN_INLIERS {
        return None;
    }
    let mut e = best_e?;
    let mut inliers = best_mask;
    for _ in 0..LO_ROUNDS {
        let refit = null_from_rows(&rows, (0..n).filter(|&i| inliers[i]))?;
        e = refit;
        epipolar_residuals(&e, r1, r2, side_two, &mut resid);
        let new: Vec<bool> = (0..n).map(|i| resid[i] < sin_tol[i]).collect();
        if new.iter().filter(|&&b| b).count() < EPI_MIN_INLIERS {
            break;
        }
        let done = new == inliers;
        inliers = new;
        if done {
            break;
        }
    }
    if inliers.iter().filter(|&&b| b).count() < EPI_MIN_INLIERS {
        return None;
    }
    let s = singular_values_desc(&e);
    let denom = s[0] + s[1];
    if denom <= 0.0 || !denom.is_finite() {
        return None;
    }
    Some(EpipolarFit {
        cost: (s[0] - s[1]) / denom,
        inliers,
    })
}

/// One direction of the epipolar cell over the whole grid.
struct EpipolarScan {
    minimum: CurveMinimum,
    inliers: Vec<bool>,
    /// Largest essential consensus any candidate focal mustered — the
    /// denominator of the rotation-domination gate, the analog of `n_F`.
    best_consensus: usize,
}

fn scan_epipolar(
    model: CameraModel,
    cand: &ScanCandidate,
    grid: &[f64],
    r_hi: f64,
    samples: &[usize],
    side_two: bool,
) -> Option<EpipolarScan> {
    let n = cand.uv1.len();
    let rad_side = if side_two { &cand.rad2 } else { &cand.rad1 };
    let mut costs = vec![f64::INFINITY; grid.len()];
    let mut masks: Vec<Option<Vec<bool>>> = vec![None; grid.len()];
    let mut r1 = vec![Vector3::zeros(); n];
    let mut r2 = vec![Vector3::zeros(); n];
    let mut sin_tol = vec![0.0f64; n];
    let mut best_consensus = 0usize;
    for (k, &f) in grid.iter().enumerate() {
        if !model.admits(f, r_hi) {
            costs[k] = f64::NAN;
            continue;
        }
        for i in 0..n {
            r1[i] = model.ray(cand.uv1[i], cand.rad1[i], f);
            r2[i] = model.ray(cand.uv2[i], cand.rad2[i], f);
            sin_tol[i] = (SCAN_TOL_PX / model.scale(rad_side[i], f)).min(1.0).sin();
        }
        match fit_epipolar(&r1, &r2, &sin_tol, samples, side_two) {
            Some(fit) => {
                costs[k] = fit.cost;
                best_consensus = best_consensus.max(fit.inliers.iter().filter(|&&b| b).count());
                masks[k] = Some(fit.inliers);
            }
            None => costs[k] = f64::NAN,
        }
    }
    let minimum = curve_minimum(&costs, grid)?;
    let inliers = masks[minimum.k].clone()?;
    Some(EpipolarScan {
        minimum,
        inliers,
        best_consensus,
    })
}

// ── Rotation cell ────────────────────────────────────────────────────────────

/// Angle between each rotated ray and its measured partner.
pub(crate) fn rotation_residuals(
    rot: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    idx: &[usize],
    out: &mut [f64],
) {
    for (o, &i) in out.iter_mut().zip(idx.iter()) {
        *o = (rot * r1[i]).dot(&r2[i]).clamp(-1.0, 1.0).acos();
    }
}

/// Largest rotation consensus at one candidate focal, or `None` when it stays
/// under `min_support`.
fn rotation_support_at(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    tol: &[f64],
    samples: &[usize],
    min_support: usize,
) -> Option<Vec<bool>> {
    let n = r1.len();
    let all: Vec<usize> = (0..n).collect();
    let mut resid = vec![0.0f64; n];
    let mut best_count = 0usize;
    let mut best: Option<Vec<bool>> = None;
    for s in samples.chunks_exact(ROT_SAMPLE) {
        let Some(rot) = kabsch(r1, r2, s) else {
            continue;
        };
        rotation_residuals(&rot, r1, r2, &all, &mut resid);
        let count = (0..n).filter(|&i| resid[i] < tol[i]).count();
        if count > best_count {
            best_count = count;
            best = Some((0..n).map(|i| resid[i] < tol[i]).collect());
        }
    }
    let mut inl = best?;
    if best_count < min_support {
        return None;
    }
    for _ in 0..LO_ROUNDS {
        let idx: Vec<usize> = (0..n).filter(|&i| inl[i]).collect();
        let rot = kabsch(r1, r2, &idx)?;
        rotation_residuals(&rot, r1, r2, &all, &mut resid);
        let new: Vec<bool> = (0..n).map(|i| resid[i] < tol[i]).collect();
        if new.iter().filter(|&&b| b).count() < min_support {
            return None;
        }
        let done = new == inl;
        inl = new;
        if done {
            break;
        }
    }
    Some(inl)
}

/// The largest rotation consensus the pair musters at any candidate focal, with
/// the support that achieved it.
///
/// This one pass serves two purposes: it freezes the rotation cell's support
/// (so its costs are comparable across candidate focals) and it supplies the
/// epipolar cell's rotation-domination gate with the count. A rotation is the
/// far-field model of the pair, so the size of its consensus is exactly what
/// "does a rotation already explain this pair?" means.
///
/// A coarse sub-grid locates the winning bracket and the grid points inside it
/// are then swept at full resolution. The refinement is load-bearing for the
/// gate rather than for the freeze: a parallax-free pair's rotation consensus
/// peaks sharply at its own focal, and the sub-grid's stride can straddle that
/// peak by more than a tenth — on the pure-rotation fixture the coarse pass
/// alone reports 222 of 300 correspondences where the refined pass reports 300,
/// which is the difference between clearing the `0.8` gate and missing it. The
/// essential consensus it is compared against carries no such bias, being read
/// off the full grid.
fn freeze_rotation_support(
    model: CameraModel,
    cand: &ScanCandidate,
    grid: &[f64],
    r_hi: f64,
    samples: &[usize],
    min_support: usize,
) -> Option<(usize, Vec<usize>)> {
    let n = cand.uv1.len();
    let mut r1 = vec![Vector3::zeros(); n];
    let mut r2 = vec![Vector3::zeros(); n];
    let mut tol = vec![0.0f64; n];
    let valid: Vec<usize> = (0..grid.len())
        .filter(|&k| model.admits(grid[k], r_hi))
        .collect();
    let consensus_at = |k: usize,
                        r1: &mut [Vector3<f64>],
                        r2: &mut [Vector3<f64>],
                        tol: &mut [f64]|
     -> Option<Vec<usize>> {
        let f = grid[k];
        for i in 0..n {
            r1[i] = model.ray(cand.uv1[i], cand.rad1[i], f);
            r2[i] = model.ray(cand.uv2[i], cand.rad2[i], f);
            tol[i] = SCAN_TOL_PX / model.scale(cand.rad2[i], f);
        }
        let mask = rotation_support_at(r1, r2, tol, samples, min_support)?;
        Some((0..n).filter(|&i| mask[i]).collect())
    };

    let mut best: Option<(usize, usize, Vec<usize>)> = None; // (position, count, support)
    for (pos, &k) in valid.iter().enumerate().step_by(ROT_SUPPORT_STRIDE) {
        if let Some(idx) = consensus_at(k, &mut r1, &mut r2, &mut tol) {
            if best.as_ref().is_none_or(|(_, b, _)| idx.len() > *b) {
                best = Some((pos, idx.len(), idx));
            }
        }
    }
    let (pos, _, _) = best.as_ref()?;
    let lo = pos.saturating_sub(ROT_SUPPORT_STRIDE - 1);
    let hi = (pos + ROT_SUPPORT_STRIDE).min(valid.len());
    // Sweep the winning bracket at full grid resolution, skipping the points
    // the coarse pass already visited.
    for (p, &k) in valid.iter().enumerate().take(hi).skip(lo) {
        if p % ROT_SUPPORT_STRIDE == 0 {
            continue;
        }
        if let Some(idx) = consensus_at(k, &mut r1, &mut r2, &mut tol) {
            if best.as_ref().is_none_or(|(_, b, _)| idx.len() > *b) {
                best = Some((p, idx.len(), idx));
            }
        }
    }
    best.map(|(_, count, idx)| (count, idx))
}

/// The rotation fit's trimmed residual at one candidate focal, in both units.
struct RotationFit {
    /// Trimmed RMS residual carried through the map's local `dr/dθ` (pixels) —
    /// this is what locates the minimum, because it removes the `1/f` drift
    /// both maps share.
    cost_px: f64,
    /// Trimmed RMS angular residual (radians) — this is what the floor gates,
    /// because it transfers across capture resolutions.
    cost_rad: f64,
}

/// Fit a rotation on the FROZEN support and report its trimmed residual.
fn fit_rotation(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    px_scale: &[f64],
    support: &[usize],
) -> Option<RotationFit> {
    let n_keep = ROT_MIN_SUPPORT.max((ROT_TRIM_Q * support.len() as f64).round() as usize);
    let n_keep = n_keep.min(support.len());
    let mut keep: Vec<usize> = support.to_vec();
    let mut rot = kabsch(r1, r2, &keep)?;
    let mut resid = vec![0.0f64; support.len()];
    for _ in 0..ROT_TRIM_ROUNDS {
        rotation_residuals(&rot, r1, r2, support, &mut resid);
        let mut order: Vec<usize> = (0..support.len()).collect();
        order.sort_by(|&a, &b| resid[a].total_cmp(&resid[b]));
        keep = order[..n_keep].iter().map(|&j| support[j]).collect();
        rot = kabsch(r1, r2, &keep)?;
    }
    rotation_residuals(&rot, r1, r2, support, &mut resid);
    let mut order: Vec<usize> = (0..support.len()).collect();
    order.sort_by(|&a, &b| resid[a].total_cmp(&resid[b]));
    let mut sq_px = 0.0;
    let mut sq_rad = 0.0;
    for &j in &order[..n_keep] {
        let e = resid[j];
        sq_rad += e * e;
        let s = e * px_scale[support[j]];
        sq_px += s * s;
    }
    let m = n_keep as f64;
    Some(RotationFit {
        cost_px: (sq_px / m).sqrt(),
        cost_rad: (sq_rad / m).sqrt(),
    })
}

/// The rotation cell over the whole grid, with a support frozen once per pair.
struct RotationScan {
    /// Minimum of the pixel-scaled curve — what the vote reports.
    minimum: CurveMinimum,
    /// The angular cost at that minimum — what the floor gates.
    cost_rad_at_min: f64,
    /// Minimum of the raw angular curve. Diagnostic, and the curve the frozen
    /// support is what makes interior at all: with a per-candidate support a
    /// bad focal buys a low angular cost by keeping fewer points and this
    /// minimum slides to the top of the grid.
    rad_minimum: Option<CurveMinimum>,
    /// The frozen support.
    support: Vec<usize>,
}

fn scan_rotation(
    model: CameraModel,
    cand: &ScanCandidate,
    grid: &[f64],
    r_hi: f64,
    samples: &[usize],
    freeze_support: bool,
) -> Option<RotationScan> {
    let n = cand.uv1.len();
    let valid: Vec<usize> = (0..grid.len())
        .filter(|&k| model.admits(grid[k], r_hi))
        .collect();
    if valid.len() < MIN_CURVE_POINTS {
        return None;
    }
    let mut r1 = vec![Vector3::zeros(); n];
    let mut r2 = vec![Vector3::zeros(); n];
    let mut tol = vec![0.0f64; n];
    let fill = |f: f64, r1: &mut [Vector3<f64>], r2: &mut [Vector3<f64>], tol: &mut [f64]| {
        for i in 0..n {
            r1[i] = model.ray(cand.uv1[i], cand.rad1[i], f);
            r2[i] = model.ray(cand.uv2[i], cand.rad2[i], f);
            tol[i] = SCAN_TOL_PX / model.scale(cand.rad2[i], f);
        }
    };

    // One pass over a coarse sub-grid fixes the pair's far-field support: the
    // largest rotation consensus any candidate focal can muster. Freezing it is
    // what makes the costs comparable across the grid.
    let (_, frozen) = freeze_rotation_support(model, cand, grid, r_hi, samples, ROT_MIN_SUPPORT)?;

    let mut costs = vec![f64::NAN; grid.len()];
    let mut costs_rad = vec![f64::NAN; grid.len()];
    let mut px_scale = vec![0.0f64; n];
    for &k in &valid {
        let f = grid[k];
        fill(f, &mut r1, &mut r2, &mut tol);
        for (s, r) in px_scale.iter_mut().zip(cand.rad2.iter()) {
            *s = model.scale(*r, f);
        }
        // The unfrozen variant re-derives the support at every candidate: a bad
        // focal then buys a low cost by keeping fewer points, and the scan pins
        // at the top of the grid instead of showing an interior minimum.
        let sup = if freeze_support {
            frozen.clone()
        } else {
            match rotation_support_at(&r1, &r2, &tol, samples, ROT_MIN_SUPPORT) {
                Some(mask) => (0..n).filter(|&i| mask[i]).collect(),
                None => continue,
            }
        };
        if sup.len() < ROT_MIN_SUPPORT {
            continue;
        }
        if let Some(fit) = fit_rotation(&r1, &r2, &px_scale, &sup) {
            costs[k] = fit.cost_px;
            costs_rad[k] = fit.cost_rad;
        }
    }
    let minimum = curve_minimum(&costs, grid)?;
    Some(RotationScan {
        cost_rad_at_min: costs_rad[minimum.k],
        rad_minimum: curve_minimum(&costs_rad, grid),
        minimum,
        support: frozen,
    })
}

// ── Per-pair drivers ─────────────────────────────────────────────────────────

/// Log-spaced candidate focals of the shared scan band.
fn scan_grid(max_wh: f64) -> Vec<f64> {
    let (l0, l1) = ((SCAN_BAND_LO * max_wh).ln(), (SCAN_BAND_HI * max_wh).ln());
    (0..SCAN_GRID_N)
        .map(|k| (l0 + (l1 - l0) * k as f64 / (SCAN_GRID_N - 1) as f64).exp())
        .collect()
}

/// Draw `count` minimal samples of `k` distinct indices in `0..n`.
pub(crate) fn draw_samples(state: &mut u64, n: usize, k: usize, count: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(count * k);
    for _ in 0..count {
        let start = out.len();
        while out.len() - start < k {
            let cand = (splitmix64(state) % n as u64) as usize;
            if !out[start..].contains(&cand) {
                out.push(cand);
            }
        }
    }
    out
}

/// Radial p90 of a mask's members over both images, as a fraction of the
/// half-diagonal.
fn coverage_p90(cand: &ScanCandidate, members: impl Iterator<Item = usize>, half_diag: f64) -> f64 {
    let mut radii: Vec<f64> = Vec::new();
    for i in members {
        radii.push(cand.rad1[i]);
        radii.push(cand.rad2[i]);
    }
    if radii.is_empty() || half_diag <= 0.0 {
        return 0.0;
    }
    quantile(&radii, 0.90) / half_diag
}

/// p99 of the pair's correspondence radii over both images — the pair's own
/// edge radius, which sets both the map-injectivity check and the credible
/// half-FOV window.
fn pair_edge_radius(cand: &ScanCandidate) -> f64 {
    let mut radii: Vec<f64> = Vec::with_capacity(2 * cand.uv1.len());
    for i in 0..cand.uv1.len() {
        radii.push(cand.rad1[i]);
        radii.push(cand.rad2[i]);
    }
    if radii.is_empty() {
        return 0.0;
    }
    quantile(&radii, 0.99)
}

/// Whether a focal sits in the credible half-FOV window at `r_hi`.
fn in_fov_band(f: f64, r_hi: f64) -> bool {
    let lo = r_hi / FOV_HALF_HI_DEG.to_radians();
    let hi = r_hi / FOV_HALF_LO_DEG.to_radians();
    (lo..=hi).contains(&f)
}

/// Run both cells of one column over a capture's candidate pairs.
///
/// `epipolar` and `rotation` are the same candidate pairs the pinhole kernel's
/// two families select, so every column's certificates come from the same
/// machinery and their masses are comparable.
pub fn scan_column(
    model: CameraModel,
    epipolar: &[ScanCandidate],
    rotation: &[ScanCandidate],
    max_wh: f64,
    half_diag: f64,
) -> ColumnScan {
    scan_column_inner(model, epipolar, rotation, max_wh, half_diag, true)
}

/// [`scan_column`] with the rotation cell's support freezing switchable — the
/// unfrozen variant exists so tests can pin what freezing buys.
pub(crate) fn scan_column_inner(
    model: CameraModel,
    epipolar: &[ScanCandidate],
    rotation: &[ScanCandidate],
    max_wh: f64,
    half_diag: f64,
    freeze_support: bool,
) -> ColumnScan {
    let grid = scan_grid(max_wh);
    let epi: Vec<ScanVote> = epipolar
        .par_iter()
        .filter_map(|c| epipolar_vote(model, c, &grid, half_diag))
        .collect();
    let rot: Vec<ScanVote> = rotation
        .par_iter()
        .filter_map(|c| rotation_vote(model, c, &grid, half_diag, freeze_support))
        .collect();
    ColumnScan {
        model,
        epipolar: epi,
        rotation: rot,
    }
}

/// One pair's epipolar-cell vote: two one-sided scans, certified by the floor,
/// the shape, an interior minimum, their direction agreement — and by the pair
/// not already being explained by a rotation.
fn epipolar_vote(
    model: CameraModel,
    cand: &ScanCandidate,
    grid: &[f64],
    half_diag: f64,
) -> Option<ScanVote> {
    let n = cand.uv1.len();
    if n < EPI_SAMPLE * 2 {
        return None;
    }
    let r_hi = pair_edge_radius(cand);
    let mut state = cand.seed;
    let samples = draw_samples(&mut state, n, EPI_SAMPLE, SCAN_SAMPLES);
    let two = scan_epipolar(model, cand, grid, r_hi, &samples, true)?;

    // Rotation domination — the analog of the pinhole family's homography
    // domination, and the same statement: fit the rotation family's model and
    // abstain when it explains the pair. A near-zero baseline makes
    // `E = [t]×R` degenerate and its essentialness minima broad, exactly as the
    // pinhole `F` collapses toward `H`. The rotation consensus comes from the
    // same coarse sub-grid pass the rotation cell uses to freeze its support.
    let rot_samples = draw_samples(&mut state, n, ROT_SAMPLE, SCAN_SAMPLES);
    let n_rotation = freeze_rotation_support(
        model,
        cand,
        grid,
        r_hi,
        &rot_samples,
        ROTATION_DOMINATION_MIN,
    )
    .map_or(0, |(count, _)| count);
    let n_essential = two.best_consensus;
    let rotation_ratio =
        (n_essential >= RATIO_MIN_ESSENTIAL).then(|| n_rotation as f64 / n_essential as f64);
    let rotation_dominated = n_rotation as f64
        >= (ROTATION_DOMINATION_MIN as f64).max(ROTATION_DOMINATION_FRAC * n_essential as f64);

    // Both directions are always scanned, gated or not: `dir_disagreement` is
    // then a complete covariate, and a caller can reconstruct exactly which
    // votes the domination gate removed.
    let one = scan_epipolar(model, cand, grid, r_hi, &samples, false);
    let dir_disagreement = one
        .as_ref()
        .map(|s| (two.minimum.focal_px.ln() - s.minimum.focal_px.ln()).abs());
    // The two cameras share the focal, so the two one-sided scans are two
    // measurements of the same quantity: the vote is their geometric mean.
    let focal_px = match &one {
        Some(s) => (two.minimum.focal_px * s.minimum.focal_px).sqrt(),
        None => two.minimum.focal_px,
    };
    let coverage = coverage_p90(cand, (0..n).filter(|&i| two.inliers[i]), half_diag);
    let certified = !rotation_dominated
        && !two.minimum.edge
        && two.minimum.cost_min <= ESSENTIALNESS_FLOOR
        && two.minimum.sharpness <= SHAPE_GATE
        && dir_disagreement.is_some_and(|d| d <= DIRECTION_BAND);
    Some(ScanVote {
        cell: ScanCell::Epipolar,
        image_a: cand.image_a,
        image_b: cand.image_b,
        focal_px,
        cost: two.minimum.cost_min,
        sharpness: two.minimum.sharpness,
        dir_disagreement,
        rotation_dominated,
        rotation_ratio,
        coverage_p90: coverage,
        n_inliers: two.inliers.iter().filter(|&&b| b).count(),
        in_fov_band: in_fov_band(grid[two.minimum.k], r_hi),
        at_grid_edge: two.minimum.edge,
        angular_focal_px: None,
        certified,
        model_informative: certified && coverage >= COVERAGE_FLOOR,
    })
}

/// One pair's rotation-cell vote over its frozen support.
fn rotation_vote(
    model: CameraModel,
    cand: &ScanCandidate,
    grid: &[f64],
    half_diag: f64,
    freeze_support: bool,
) -> Option<ScanVote> {
    let n = cand.uv1.len();
    if n < ROT_MIN_SUPPORT {
        return None;
    }
    let r_hi = pair_edge_radius(cand);
    let mut state = cand.seed;
    let samples = draw_samples(&mut state, n, ROT_SAMPLE, SCAN_SAMPLES);
    let scan = scan_rotation(model, cand, grid, r_hi, &samples, freeze_support)?;
    let coverage = coverage_p90(cand, scan.support.iter().copied(), half_diag);
    let certified = !scan.minimum.edge
        && scan.cost_rad_at_min.is_finite()
        && scan.cost_rad_at_min <= ROTATION_FIT_FLOOR_RAD
        && scan.minimum.sharpness <= SHAPE_GATE;
    Some(ScanVote {
        cell: ScanCell::Rotation,
        image_a: cand.image_a,
        image_b: cand.image_b,
        focal_px: scan.minimum.focal_px,
        cost: scan.cost_rad_at_min,
        sharpness: scan.minimum.sharpness,
        dir_disagreement: None,
        rotation_dominated: false,
        rotation_ratio: None,
        coverage_p90: coverage,
        n_inliers: scan.support.len(),
        in_fov_band: in_fov_band(grid[scan.minimum.k], r_hi),
        at_grid_edge: scan.minimum.edge,
        angular_focal_px: scan.rad_minimum.map(|m| m.focal_px),
        certified,
        model_informative: certified && coverage >= COVERAGE_FLOOR,
    })
}

#[cfg(test)]
mod tests;
