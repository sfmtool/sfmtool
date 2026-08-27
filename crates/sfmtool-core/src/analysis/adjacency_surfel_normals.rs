// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Adjacency surfel normals: a robust plane fit through each point, over the
//! directions of its image-space neighbours.
//!
//! The neighbours come from the
//! [observation adjacency graph](super::observation_adjacency), so "next to" is
//! decided on the imaged surface rather than in the point cloud, and a
//! neighbour that belongs to a different surface can still turn up in the set.
//! The plane is anchored at the point itself (surfel semantics) and fitted on
//! the *unit* directions to the neighbours, so each neighbour contributes its
//! angular deviation once rather than in proportion to its distance: distant
//! neighbours supply angular leverage without dominating, and a neighbour's own
//! position noise enters only through angle. A Tukey-redescending IRLS loop
//! then discards the ones that do not lie on the point's surface.
//!
//! Alongside each normal the kernel reports how well-determined it is —
//! effective support, angular coverage, in-plane anisotropy — and a boolean
//! verdict from thresholds on those. Callers route on the verdict (keep the
//! normal, go acquire more neighbours, or fall back and mark the point
//! low-confidence); the kernel never substitutes a fallback normal itself, so
//! an unfittable point comes back `NaN` rather than as something that looks
//! estimated.
//!
//! See `specs/core/analysis/adjacency-surfel-normals.md` for the design.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;

use crate::numeric::median_in_place;

/// A displacement this short carries no direction worth fitting.
const EPS_DISPLACEMENT: f64 = 1e-12;
/// Denominator floor for the weight ratios and eigenvalue ratios below; every
/// one of those quantities is non-negative, so the floor only ever guards a
/// division by an underflowed sum.
const EPS_DENOM: f64 = 1e-30;
/// A point whose redescended weights sum to no more than this has nothing left
/// to fit and stops iterating.
const EPS_STALL: f64 = 1e-12;
/// A row counts toward the sector coverage when its weight is at least this
/// fraction of the point's largest weight.
const LIVE_WEIGHT_FRACTION: f64 = 0.25;
/// Median-absolute-deviation to standard-deviation conversion for a Gaussian.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Tuning for [`estimate_adjacency_surfel_normals`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdjacencySurfelParams {
    /// Number of IRLS passes before the final solve.
    pub irls_iters: u32,
    /// Tukey biweight tuning constant.
    pub tukey_c: f64,
    /// Floor on the robust scale, as an angle: below this the residuals are
    /// treated as noise, not as structure.
    pub sigma_floor_deg: f64,
    /// Number of equal tangent sectors the angular coverage is measured in.
    pub n_sectors: u32,
    /// `determined` floor on the effective neighbour count.
    pub det_n_eff: f64,
    /// `determined` floor on the number of occupied sectors.
    pub det_sectors: u32,
    /// `determined` floor on the in-plane anisotropy.
    pub det_aniso: f64,
}

impl Default for AdjacencySurfelParams {
    /// The spec's defaults.
    fn default() -> Self {
        Self {
            irls_iters: 3,
            tukey_c: 4.685,
            sigma_floor_deg: 2.0,
            n_sectors: 8,
            det_n_eff: 4.0,
            det_sectors: 3,
            det_aniso: 0.10,
        }
    }
}

/// Caller-synthesized neighbour positions, in CSR over the cloud.
///
/// These are neighbours the adjacency graph does not have — typically helper
/// patches congealed on purpose for points whose graph neighbourhood is
/// under-determined. They enter the fit exactly like graph neighbours.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtraNeighbours {
    /// CSR row boundaries; length `n_points + 1`, or empty for "no extras".
    pub offsets: Vec<u32>,
    /// Extra neighbour positions, in the same frame as `positions`.
    pub positions: Vec<[f64; 3]>,
}

impl ExtraNeighbours {
    /// No extra neighbours anywhere.
    pub fn none() -> Self {
        Self::default()
    }

    /// The extra positions supplied for point `p`; empty when there are none.
    fn row(&self, p: usize) -> &[[f64; 3]] {
        if self.offsets.len() < p + 2 {
            return &[];
        }
        let (lo, hi) = (self.offsets[p] as usize, self.offsets[p + 1] as usize);
        &self.positions[lo..hi]
    }
}

/// Per-point normals and the diagnostics that say how well determined they are.
///
/// Every field is dense over the cloud. Points that were not selected, and
/// selected points with fewer than two usable neighbours, are `NaN` throughout
/// (and `false` in [`Self::determined`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdjacencySurfelNormals {
    /// Unit normals, sign-aligned to the caller's `view_dirs`.
    pub normals: Vec<[f64; 3]>,
    /// `(Σw)² / Σw²` — the neighbour count left after redescending.
    pub n_eff: Vec<f64>,
    /// `λ_mid / λ_max` of the final scatter: how two-dimensional the in-plane
    /// spread is. A rank-1 line of neighbours cannot pin a plane.
    pub anisotropy: Vec<f64>,
    /// Tangent sectors occupied by live neighbours.
    pub sectors: Vec<f64>,
    /// The robust scale as an angle, `asin(min(σ, 1))` in degrees.
    pub sigma_deg: Vec<f64>,
    /// The weighted RMS residual as an angle, in degrees.
    pub resid_deg: Vec<f64>,
    /// Neighbour rows that survived the displacement-length test.
    pub n_support: Vec<f64>,
    /// The determinacy verdict: support, coverage and anisotropy all pass.
    pub determined: Vec<bool>,
}

impl AdjacencySurfelNormals {
    /// All-`NaN` output over `n_points` points.
    fn empty(n_points: usize) -> Self {
        Self {
            normals: vec![[f64::NAN; 3]; n_points],
            n_eff: vec![f64::NAN; n_points],
            anisotropy: vec![f64::NAN; n_points],
            sectors: vec![f64::NAN; n_points],
            sigma_deg: vec![f64::NAN; n_points],
            resid_deg: vec![f64::NAN; n_points],
            n_support: vec![f64::NAN; n_points],
            determined: vec![false; n_points],
        }
    }
}

/// One point's fit, before it is scattered into the dense output.
struct PointFit {
    normal: Vector3<f64>,
    n_eff: f64,
    anisotropy: f64,
    sectors: f64,
    sigma_deg: f64,
    resid_deg: f64,
    n_support: f64,
    determined: bool,
}

/// Estimate a surfel normal at every selected point.
///
/// # Arguments
/// * `positions` — per point.
/// * `offsets`, `neighbours` — the adjacency CSR; `offsets` has `n_points + 1`
///   entries.
/// * `view_dirs` — per point, the reference direction the normal's sign and the
///   sector basis are taken from (typically the mean unit direction toward the
///   observing cameras).
/// * `selected` — which points to fit.
/// * `extras` — synthesized neighbour positions, in CSR; may be empty. Extras
///   for unselected points are ignored.
/// * `params` — the fit and the determinacy thresholds.
///
/// The per-point work is independent and runs in parallel, but the fit itself
/// involves no randomness and a fixed pass count, so the result is a function of
/// the inputs alone and never of the thread count.
///
/// # Panics
/// If the per-point slices disagree on length, if `offsets` is not
/// `n_points + 1` long, or if a neighbour index is out of range.
pub fn estimate_adjacency_surfel_normals(
    positions: &[[f64; 3]],
    offsets: &[u32],
    neighbours: &[u32],
    view_dirs: &[[f64; 3]],
    selected: &[bool],
    extras: &ExtraNeighbours,
    params: &AdjacencySurfelParams,
) -> AdjacencySurfelNormals {
    let n_points = positions.len();
    assert_eq!(
        view_dirs.len(),
        n_points,
        "view_dirs must have one entry per point"
    );
    assert_eq!(
        selected.len(),
        n_points,
        "selected must have one entry per point"
    );
    if n_points == 0 {
        return AdjacencySurfelNormals::empty(0);
    }
    assert_eq!(
        offsets.len(),
        n_points + 1,
        "offsets must have n_points + 1 entries"
    );
    assert!(
        neighbours.iter().all(|&q| (q as usize) < n_points),
        "neighbour index out of range"
    );
    assert!(
        extras.offsets.is_empty() || extras.offsets.len() == n_points + 1,
        "extras.offsets must be empty or have n_points + 1 entries"
    );

    let mut out = AdjacencySurfelNormals::empty(n_points);
    let ids: Vec<usize> = (0..n_points).filter(|&p| selected[p]).collect();
    if ids.is_empty() {
        return out;
    }

    let fits: Vec<Option<PointFit>> = ids
        .par_iter()
        .map(|&p| fit_point(p, positions, offsets, neighbours, view_dirs, extras, params))
        .collect();

    for (&p, fit) in ids.iter().zip(fits) {
        let Some(fit) = fit else { continue };
        out.normals[p] = [fit.normal[0], fit.normal[1], fit.normal[2]];
        out.n_eff[p] = fit.n_eff;
        out.anisotropy[p] = fit.anisotropy;
        out.sectors[p] = fit.sectors;
        out.sigma_deg[p] = fit.sigma_deg;
        out.resid_deg[p] = fit.resid_deg;
        out.n_support[p] = fit.n_support;
        out.determined[p] = fit.determined;
    }
    out
}

/// Fit one point. `None` when fewer than two neighbour directions survive, which
/// is the kernel's way of saying "no normal", not "a normal of zero quality".
fn fit_point(
    p: usize,
    positions: &[[f64; 3]],
    offsets: &[u32],
    neighbours: &[u32],
    view_dirs: &[[f64; 3]],
    extras: &ExtraNeighbours,
    params: &AdjacencySurfelParams,
) -> Option<PointFit> {
    let anchor = Vector3::from(positions[p]);

    // Unit directions to the neighbours. The residual, the scatter and the
    // sector angle are all scale-free, so the length is dropped here and the
    // fit never sees it. A row that contributes no direction is dropped before
    // anything counts it.
    let row = offsets[p] as usize..offsets[p + 1] as usize;
    let dirs: Vec<Vector3<f64>> = neighbours[row]
        .iter()
        .map(|&q| Vector3::from(positions[q as usize]))
        .chain(extras.row(p).iter().map(|&e| Vector3::from(e)))
        .filter_map(|q| {
            let d = q - anchor;
            let len = d.norm();
            if len > EPS_DISPLACEMENT {
                Some(d / len)
            } else {
                None
            }
        })
        .collect();

    let n_support = dirs.len();
    if n_support < 2 {
        return None;
    }

    // ── IRLS ──────────────────────────────────────────────────────────────
    let sigma_floor = params.sigma_floor_deg.to_radians().sin();
    let mut weights = vec![1.0f64; n_support];
    let mut residuals = vec![0.0f64; n_support];
    let mut scratch = vec![0.0f64; n_support];
    let mut redescended = vec![0.0f64; n_support];
    let mut sigma = f64::NAN;

    for _ in 0..params.irls_iters {
        let normal = smallest_eigenvector(&weighted_scatter(&dirs, &weights));
        fill_residuals(&dirs, &normal, &mut residuals);

        // Robust scale, floored: a median at or below the floor means the
        // residuals are noise, and letting σ chase them would redescend the
        // whole neighbourhood away.
        scratch.copy_from_slice(&residuals);
        let mut scale = MAD_TO_SIGMA * median_in_place(&mut scratch);
        if !scale.is_finite() {
            scale = sigma_floor;
        }
        let scale = scale.max(sigma_floor);

        let mut sum = 0.0;
        for (rob, &r) in redescended.iter_mut().zip(residuals.iter()) {
            let u = r / (params.tukey_c * scale);
            *rob = if u < 1.0 {
                let t = 1.0 - u * u;
                t * t
            } else {
                0.0
            };
            sum += *rob;
        }

        // The point is still active here, so this is the σ it keeps if the next
        // test stalls it.
        sigma = scale;

        // Stall exit: weights that all redescend to zero would leave an
        // all-zero scatter matrix, whose eigenvectors are an arbitrary axis
        // frame. Keep the last usable weights instead and stop.
        if sum <= EPS_STALL {
            break;
        }
        weights.copy_from_slice(&redescended);
    }

    // ── Final solve and diagnostics ───────────────────────────────────────
    let scatter = weighted_scatter(&dirs, &weights);
    let eig = SymmetricEigen::new(scatter);
    // Explicit ordering: nalgebra makes no promise about the eigenvalue order.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let mut normal = eig.eigenvectors.column(order[0]).into_owned();
    let lam_mid = eig.eigenvalues[order[1]];
    let lam_max = eig.eigenvalues[order[2]];

    let view = Vector3::from(view_dirs[p]);
    if normal.dot(&view) < 0.0 {
        normal = -normal;
    }

    fill_residuals(&dirs, &normal, &mut residuals);
    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let sum_wr2: f64 = weights
        .iter()
        .zip(residuals.iter())
        .map(|(w, r)| w * r * r)
        .sum();

    let n_eff = sum_w * sum_w / sum_w2.max(EPS_DENOM);
    let anisotropy = if lam_max > EPS_DENOM {
        lam_mid / lam_max.max(EPS_DENOM)
    } else {
        0.0
    };
    let rms = (sum_wr2.max(0.0) / sum_w.max(EPS_DENOM)).sqrt();
    let sectors = occupied_sectors(&dirs, &weights, &view, params.n_sectors) as f64;

    let n_support = n_support as f64;
    let determined = n_eff >= params.det_n_eff
        && sectors >= params.det_sectors as f64
        && anisotropy >= params.det_aniso;

    Some(PointFit {
        normal,
        n_eff,
        anisotropy,
        sectors,
        sigma_deg: asin_deg(sigma),
        resid_deg: asin_deg(rms),
        n_support,
        determined,
    })
}

/// `Σ w_q d̂_q d̂_qᵀ` — the weighted scatter of the unit directions.
fn weighted_scatter(dirs: &[Vector3<f64>], weights: &[f64]) -> Matrix3<f64> {
    let mut m = Matrix3::zeros();
    for (d, &w) in dirs.iter().zip(weights) {
        m += (w * d) * d.transpose();
    }
    m
}

/// The eigenvector of the smallest eigenvalue, chosen by explicit comparison.
fn smallest_eigenvector(scatter: &Matrix3<f64>) -> Vector3<f64> {
    let eig = SymmetricEigen::new(*scatter);
    let mut best = 0;
    for i in 1..3 {
        if eig.eigenvalues[i].total_cmp(&eig.eigenvalues[best]).is_lt() {
            best = i;
        }
    }
    eig.eigenvectors.column(best).into_owned()
}

/// `r_q = |d̂_q · n|` — the sine of the angle by which `q` sits off the plane.
fn fill_residuals(dirs: &[Vector3<f64>], normal: &Vector3<f64>, out: &mut [f64]) {
    for (r, d) in out.iter_mut().zip(dirs) {
        *r = d.dot(normal).abs();
    }
}

/// An orthonormal basis of the plane orthogonal to `view`.
///
/// Normal-free by construction — a pure function of the point's viewing
/// direction — so the coverage measure it supports cannot be circular with the
/// normal it qualifies.
fn tangent_basis(view: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let mut v = *view;
    if v.norm() < 1e-9 {
        v = Vector3::z();
    }
    v /= v.norm().max(1e-12);
    let seed = if v.z.abs() > 0.95 {
        Vector3::x()
    } else {
        Vector3::z()
    };
    let mut e1 = seed - seed.dot(&v) * v;
    e1 /= e1.norm().max(1e-12);
    let e2 = v.cross(&e1);
    (e1, e2)
}

/// How many of the `n_sectors` tangent sectors hold a live neighbour — one whose
/// weight is within a factor of four of the point's largest.
fn occupied_sectors(
    dirs: &[Vector3<f64>],
    weights: &[f64],
    view: &Vector3<f64>,
    n_sectors: u32,
) -> u32 {
    if n_sectors == 0 {
        return 0;
    }
    let (e1, e2) = tangent_basis(view);
    let max_w = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let live_floor = LIVE_WEIGHT_FRACTION * max_w.max(EPS_DENOM);

    let mut occupied = vec![false; n_sectors as usize];
    for (d, &w) in dirs.iter().zip(weights) {
        if w < live_floor {
            continue;
        }
        let angle = d.dot(&e2).atan2(d.dot(&e1));
        let raw = (angle / std::f64::consts::TAU + 1.0) * n_sectors as f64;
        let bin = (raw as i64).rem_euclid(n_sectors as i64) as usize;
        occupied[bin] = true;
    }
    occupied.iter().filter(|&&o| o).count() as u32
}

/// A sine reported as the angle it stands for, in degrees; `NaN` stays `NaN`.
fn asin_deg(sine: f64) -> f64 {
    if sine.is_nan() {
        return f64::NAN;
    }
    sine.min(1.0).asin().to_degrees()
}

#[cfg(test)]
mod tests;
