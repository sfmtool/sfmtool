// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Structure-free focal-length estimation by pairwise voting ([`focal_vote`]).
//!
//! Image pairs drawn from cluster-track observations each cast one focal vote
//! through whichever of two estimators their geometry can observe — the
//! Bougnoux focal of a robustly estimated fundamental matrix for pairs carrying
//! parallax, rotation self-calibration (`H = K R K⁻¹`) for pairs dominated by a
//! parallax-free homography — and the consensus focal is the median of the
//! pooled votes from both families.
//!
//! Every focal median in this module is taken in **log space**, so an
//! even-length median is the geometric mean of the two central votes; the
//! agreement bands and the reported spreads are log-focal quantities to match.
//!
//! Both families generalize over the camera model through the pixel→ray map, so
//! the caller may ask for more than one **column** (camera-model hypothesis):
//! see [`column_scan`]. The default is pinhole-only, and then no scan runs at
//! all.
//!
//! The pair-table pass is deterministic and the RANSAC estimators derive their
//! sampling from the input seed, so identical inputs and seed reproduce
//! identical output.
//!
//! See `specs/core/geometry/focal-vote.md` for the design.

use std::collections::{HashMap, HashSet};

use nalgebra::Matrix3;
use rayon::prelude::*;

use crate::geometry::epipolar_estimation::{
    estimate_fundamental, focal_from_fundamental, FundamentalOptions,
};
use crate::geometry::homography_estimation::{estimate_homography, HomographyOptions};
use crate::numeric::median;

pub mod column_scan;

pub use column_scan::{CameraModel, ColumnScan, ScanCell, ScanVote};

/// Which family contributed the majority of the pooled votes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteFamily {
    /// Bougnoux focal of a fundamental matrix (parallax-rich pairs).
    Epipolar,
    /// Rotation self-calibration of a conjugate homography (far-field pairs).
    Rotation,
}

impl VoteFamily {
    /// Stable string name for the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            VoteFamily::Epipolar => "Epipolar",
            VoteFamily::Rotation => "Rotation",
        }
    }
}

/// One in-band directional Bougnoux focal with the candidate-pair covariates
/// that produced it (diagnostic detail, independent of what pools; two entries
/// per pair when both F and Fᵀ yield an in-band focal).
#[derive(Clone, Copy, Debug)]
pub struct EpipolarVote {
    /// First image of the candidate pair.
    pub image_a: u32,
    /// Second image of the candidate pair.
    pub image_b: u32,
    /// Shared-cluster covisibility count of the pair.
    pub shared_clusters: f64,
    /// Mean feature displacement of the pair in pixels.
    pub mean_disp_px: f64,
    /// Fundamental-matrix RANSAC inlier count.
    pub n_f_inliers: usize,
    /// Homography RANSAC inlier count on the same correspondences.
    pub n_h_inliers: usize,
    /// `false` for the vote from F, `true` for the vote from Fᵀ.
    pub transposed: bool,
    /// The directional Bougnoux focal in pixels.
    pub focal_px: f64,
}

/// One accepted rotation self-calibration vote with its pair covariates
/// (diagnostic detail; one entry per unordered image pair).
#[derive(Clone, Copy, Debug)]
pub struct RotationVote {
    /// The sampled image.
    pub image: u32,
    /// Its widest-displacement partner.
    pub partner: u32,
    /// Mean feature displacement of the pair in pixels.
    pub mean_disp_px: f64,
    /// Homography RANSAC inlier count.
    pub n_inliers: usize,
    /// The self-calibration focal vote in pixels.
    pub focal_px: f64,
}

/// Result of [`focal_vote`].
#[derive(Clone, Debug)]
pub struct FocalVoteResult {
    /// Consensus focal in pixels — the log-space median of the pooled votes, or
    /// of the majority family's votes when the two families' medians disagree
    /// beyond the family-disagreement band. `None` with fewer than 2 pooled
    /// votes.
    pub focal_px: Option<f64>,
    /// Majority contributor to the pool (ties go to `Rotation`), `None` when
    /// there is no consensus. Under the family-disagreement rule `focal_px` is
    /// this family's median; otherwise diagnostic.
    pub family: Option<VoteFamily>,
    /// Log-space median of the epipolar pair votes (diagnostic).
    pub epipolar_focal_px: Option<f64>,
    /// Log-space median of the rotation votes (diagnostic).
    pub rotation_focal_px: Option<f64>,
    /// Epipolar pair votes entering the pool (one per direction-consistent
    /// pair).
    pub n_epipolar: usize,
    /// Rotation (self-calibration) votes entering the pool (one per unordered
    /// image pair).
    pub n_rotation: usize,
    /// Total pooled votes (`n_epipolar + n_rotation`).
    pub n_pool: usize,
    /// Interquartile range, in log-focal space, of the votes behind `focal_px`
    /// — the whole pool, or the majority family's votes under the
    /// family-disagreement rule. `0` without a consensus.
    pub pool_spread: f64,
    /// `|ln(epipolar_focal_px / rotation_focal_px)|`, the gap between the two
    /// families' medians; `None` unless both families voted (available even
    /// when the pool is too small for a consensus).
    pub family_disagreement: Option<f64>,
    /// Median H/F inlier ratio over the epipolar candidate pairs.
    pub parallax_poverty: f64,
    /// Interquartile range of the epipolar pair votes in log-focal space —
    /// junk vote populations scatter where genuine consensus is tight.
    pub epipolar_spread: f64,
    /// Interquartile range of the rotation votes in log-focal space.
    pub rotation_spread: f64,
    /// Every in-band directional Bougnoux focal with its pair covariates
    /// (both directions; diagnostic detail, not the pooled population).
    pub epipolar_votes: Vec<EpipolarVote>,
    /// Every accepted rotation vote with its pair covariates (one per unordered
    /// image pair).
    pub rotation_votes: Vec<RotationVote>,
    /// Epipolar candidate pairs skipped as homography-dominated (pair count).
    pub n_h_dominated: usize,
    /// Epipolar candidate pairs whose estimator produced no usable F (pair
    /// count).
    pub n_estimator_failed: usize,
    /// Directional Bougnoux focals rejected by the plausibility band
    /// (direction count).
    pub n_band_rejected: usize,
    /// Directions whose Bougnoux extraction produced no value at all
    /// (direction count).
    pub n_degenerate: usize,
    /// Epipolar candidate pairs whose two directional focals disagree (or
    /// where only one direction is in-band), and so cast no vote (pair count).
    pub n_inconsistent_pairs: usize,
    /// The model verdict: the requested column with the greater certified mass
    /// of model-informative votes (ties go to `Pinhole`). With a single
    /// requested column there is nothing to arbitrate and the verdict is that
    /// column; with several it is `None` when no column has any
    /// model-informative vote, and the reported focal then falls back to the
    /// pinhole column (or, absent it, the first requested column).
    pub camera_model: Option<CameraModel>,
    /// Per-column diagnostics, mirroring the per-family ones, in canonical
    /// order (`Pinhole`, then `EquidistantFisheye`). The winning column's
    /// consensus is what the top-level fields report; the losing column's
    /// median, counts and spread survive here.
    pub columns: Vec<ColumnDiagnostics>,
}

/// One camera-model column's own consensus and certificate counts.
///
/// The focal fields mirror the top-level per-family ones over that column's
/// votes alone: for the pinhole column they are the closed-form Bougnoux and
/// `K⁻¹HK` votes (identical to the top-level fields when pinhole wins), for the
/// equidistant column the two cells' scan votes. The certificate counts come
/// from the self-consistency scans in **both** columns, because certified
/// masses are only comparable when they come from the same machinery.
#[derive(Clone, Debug)]
pub struct ColumnDiagnostics {
    /// The column's camera model.
    pub model: CameraModel,
    /// The column's own consensus focal (its two families pooled through the
    /// same rule as the top-level consensus).
    pub focal_px: Option<f64>,
    /// Majority contributor to this column's pool.
    pub family: Option<VoteFamily>,
    /// Log-space median of this column's epipolar votes.
    pub epipolar_focal_px: Option<f64>,
    /// Log-space median of this column's rotation votes.
    pub rotation_focal_px: Option<f64>,
    /// Epipolar votes in this column's pool.
    pub n_epipolar: usize,
    /// Rotation votes in this column's pool.
    pub n_rotation: usize,
    /// `n_epipolar + n_rotation`.
    pub n_pool: usize,
    /// Log-focal interquartile range of the votes behind `focal_px`.
    pub pool_spread: f64,
    /// Log-focal gap between this column's two family medians.
    pub family_disagreement: Option<f64>,
    /// Log-focal interquartile range of this column's epipolar votes.
    pub epipolar_spread: f64,
    /// Log-focal interquartile range of this column's rotation votes.
    pub rotation_spread: f64,
    /// Median rotation/essential consensus ratio over this column's epipolar
    /// candidate pairs with at least 16 essential inliers — the scan-side
    /// counterpart of the top-level `parallax_poverty`.
    pub parallax_poverty: f64,
    /// Epipolar candidate pairs this column gated out as rotation-dominated —
    /// the analog of the top-level `n_h_dominated`.
    pub n_rotation_dominated: usize,
    /// Candidate pairs whose epipolar-cell scan produced a curve.
    pub n_scanned_epipolar: usize,
    /// Candidate pairs whose rotation-cell scan produced a curve.
    pub n_scanned_rotation: usize,
    /// Epipolar-cell scans certified by their own geometry.
    pub n_certified_epipolar: usize,
    /// Rotation-cell scans certified by their own geometry.
    pub n_certified_rotation: usize,
    /// Certified epipolar-cell scans that also clear the radial-coverage floor.
    pub n_informative_epipolar: usize,
    /// Certified rotation-cell scans that also clear the radial-coverage floor.
    pub n_informative_rotation: usize,
    /// Certified scans over both cells.
    pub n_certified: usize,
    /// Model-informative certified scans over both cells — the mass the model
    /// verdict compares.
    pub n_informative: usize,
    /// Every scan of this column, epipolar cell first (diagnostic detail).
    pub scan_votes: Vec<ScanVote>,
}

// ── Vote thresholds (see the spec) ───────────────────────────────────────────

const MIN_SHARED_STRICT: usize = 30;
const MIN_SHARED_RELAXED: usize = 16;
const MIN_QUALIFYING_PAIRS: usize = 6;
const EPIPOLAR_MIN_DISP_FRAC: f64 = 0.02;
const MAX_PAIRS_PER_IMAGE: u32 = 2;
const MAX_EPIPOLAR_PAIRS: usize = 18;
const RATIO_MIN_F_INLIERS: usize = 16;

const ROTATION_MAX_IMAGES: usize = 60;
const ROTATION_MIN_SHARED: usize = 25;
const ROTATION_MIN_DISP_FRAC: f64 = 0.08;
const ROTATION_MIN_INLIERS: usize = 12;

const ORTHO_GRID_N: usize = 48;
const ORTHO_GRID_LO: f64 = 0.3;
const ORTHO_GRID_HI: f64 = 4.0;
const ORTHO_COST_FLOOR: f64 = 0.15;

const FOCAL_BAND_LO: f64 = 0.2;
const FOCAL_BAND_HI: f64 = 4.0;

/// Maximum `|ln(f_F / f_Fᵀ)|` for a pair's two directional Bougnoux focals to
/// count as two measurements of the same shared focal.
const DIRECTION_AGREEMENT_BAND: f64 = 0.05;

/// Log-focal gap between the two families' medians beyond which the pool is
/// bimodal: blending it would report a focal no pair voted for, so the
/// consensus falls back to the majority family's median.
const FAMILY_DISAGREEMENT_BAND: f64 = 0.25;

/// Pooled votes needed for a consensus.
const MIN_POOL: usize = 2;

/// Tuning for [`focal_vote_with_options`].
#[derive(Clone, Debug)]
pub struct FocalVoteOptions {
    /// SplitMix64 seed for the RANSAC estimators and the column scans.
    pub seed: u64,
    /// Wide-baseline gate for epipolar candidate pairs, as a fraction of the
    /// image diagonal their mean feature displacement must reach. Too low
    /// admits near-static pairs whose ill-conditioned fundamental matrices vote
    /// junk focals into the pool.
    pub epipolar_min_disp_frac: f64,
    /// Camera-model columns to evaluate. The default is pinhole-only, which
    /// runs no scan at all and reproduces the closed-form kernel exactly;
    /// asking for more than one column adds the self-consistency scans that
    /// arbitrate between them. Duplicates are ignored and an empty list means
    /// pinhole-only.
    pub columns: Vec<CameraModel>,
}

impl Default for FocalVoteOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            epipolar_min_disp_frac: EPIPOLAR_MIN_DISP_FRAC,
            columns: vec![CameraModel::Pinhole],
        }
    }
}

/// The requested columns in canonical order, deduplicated; an empty request
/// means pinhole-only.
fn canonical_columns(requested: &[CameraModel]) -> Vec<CameraModel> {
    let mut out: Vec<CameraModel> = [CameraModel::Pinhole, CameraModel::EquidistantFisheye]
        .into_iter()
        .filter(|m| requested.contains(m))
        .collect();
    if out.is_empty() {
        out.push(CameraModel::Pinhole);
    }
    out
}

/// The two families' consensus over one column's votes: the pooled log-space
/// median, or the majority family's median when the two medians are further
/// apart than the family-disagreement band.
struct FamilyConsensus {
    focal_px: Option<f64>,
    family: Option<VoteFamily>,
    pool_spread: f64,
    epipolar_focal_px: Option<f64>,
    rotation_focal_px: Option<f64>,
    epipolar_spread: f64,
    rotation_spread: f64,
    family_disagreement: Option<f64>,
}

fn family_consensus(bou: &[f64], rot: &[f64]) -> FamilyConsensus {
    let epipolar_focal_px = log_median(bou);
    let rotation_focal_px = log_median(rot);
    let n_epipolar = bou.len();
    let n_rotation = rot.len();
    let epipolar_spread = log_iqr(bou);
    let rotation_spread = log_iqr(rot);
    let pool: Vec<f64> = bou.iter().chain(rot.iter()).copied().collect();
    // Computable whenever both families voted, consensus or not.
    let family_disagreement = match (epipolar_focal_px, rotation_focal_px) {
        (Some(e), Some(r)) => Some((e.ln() - r.ln()).abs()),
        _ => None,
    };
    let (focal_px, family, pool_spread) = if n_epipolar + n_rotation >= MIN_POOL {
        let fam = if n_epipolar > n_rotation {
            VoteFamily::Epipolar
        } else {
            VoteFamily::Rotation
        };
        let bimodal = family_disagreement.is_some_and(|d| d > FAMILY_DISAGREEMENT_BAND);
        let backing: &[f64] = if bimodal {
            // Majority family only — `family` reports it, so `focal_px` is
            // exactly that family's median.
            match fam {
                VoteFamily::Epipolar => bou,
                VoteFamily::Rotation => rot,
            }
        } else {
            &pool
        };
        (log_median(backing), Some(fam), log_iqr(backing))
    } else {
        (None, None, 0.0)
    };
    FamilyConsensus {
        focal_px,
        family,
        pool_spread,
        epipolar_focal_px,
        rotation_focal_px,
        epipolar_spread,
        rotation_spread,
        family_disagreement,
    }
}

/// Interquartile range in log space (linear-interpolated quartiles), `0`
/// for fewer than 2 votes.
fn log_iqr(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let mut v: Vec<f64> = vals.iter().map(|x| x.ln()).collect();
    v.sort_by(f64::total_cmp);
    let q = |p: f64| -> f64 {
        let t = p * (v.len() - 1) as f64;
        let lo = t.floor() as usize;
        let hi = t.ceil() as usize;
        v[lo] + (v[hi] - v[lo]) * (t - lo as f64)
    };
    q(0.75) - q(0.25)
}

/// Median of a focal population in log space: odd length takes the middle
/// vote, even length their geometric mean. Every focal median in this kernel
/// is taken this way, consistent with the log-space agreement bands and
/// spreads. `None` for an empty population.
fn log_median(vals: &[f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    let mut v: Vec<f64> = vals.iter().map(|x| x.ln()).collect();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let l = if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    };
    Some(l.exp())
}

/// Per-image-pair accumulator over the exhaustive per-cluster pass: how many
/// clusters the pair shares, and the sum of their feature displacements.
#[derive(Clone, Copy, Default)]
struct PairAccum {
    count: f64,
    disp_sum: f64,
}

impl PairAccum {
    fn mean_disp(&self) -> f64 {
        if self.count > 0.0 {
            self.disp_sum / self.count
        } else {
            0.0
        }
    }
}

/// Per-image observation list (cluster run, pixel position), sorted by run.
type ImageClusters = Vec<Vec<(u32, [f64; 2])>>;

/// Full-correspondence merge-join of two images over their shared cluster runs.
/// Returns `(positions in image `a`, positions in image `b`)`.
fn pair_correspondences(
    image_clusters: &ImageClusters,
    a: usize,
    b: usize,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let (la, lb) = (&image_clusters[a], &image_clusters[b]);
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < la.len() && j < lb.len() {
        match la[i].0.cmp(&lb[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                x1.push(la[i].1);
                x2.push(lb[j].1);
                i += 1;
                j += 1;
            }
        }
    }
    (x1, x2)
}

/// Orthogonality residual `‖G/(tr G/3) − I‖_F` with `G = M Mᵀ`,
/// `M = K⁻¹ H K`, `K = diag(f, f, 1)`. `+∞` for a degenerate `G`.
/// Shared with the far-field rotation initialization
/// (`crate::geometry::rotation_init`), which gates its edges on the same
/// residual.
pub(crate) fn ortho_cost(h: &Matrix3<f64>, f: f64) -> f64 {
    let kinv = Matrix3::new(1.0 / f, 0.0, 0.0, 0.0, 1.0 / f, 0.0, 0.0, 0.0, 1.0);
    let k = Matrix3::new(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0);
    let m = kinv * h * k;
    let g = m * m.transpose();
    let tr = g.trace() / 3.0;
    if !tr.is_finite() || tr.abs() < 1e-300 {
        return f64::INFINITY;
    }
    (g / tr - Matrix3::identity()).norm()
}

/// Focal from a homography's conjugate-rotation orthogonality scan, or `None`
/// when the residual floor (finite-plane homography) or flatness (roll-only /
/// too-small rotation) rejects it. `max_wh = max(width, height)`.
fn rotation_self_calib_focal(h: &Matrix3<f64>, max_wh: f64) -> Option<f64> {
    let l0 = ORTHO_GRID_LO.log10();
    let l1 = ORTHO_GRID_HI.log10();
    let mut fs = [0.0f64; ORTHO_GRID_N];
    let mut costs = [0.0f64; ORTHO_GRID_N];
    for k in 0..ORTHO_GRID_N {
        let e = l0 + (l1 - l0) * (k as f64) / ((ORTHO_GRID_N - 1) as f64);
        let f = max_wh * 10f64.powf(e);
        fs[k] = f;
        costs[k] = ortho_cost(h, f);
    }
    let mut kmin = 0usize;
    for k in 1..ORTHO_GRID_N {
        if costs[k] < costs[kmin] {
            kmin = k;
        }
    }
    // `costs` is the fixed-width scan grid, so it is never empty and the median
    // is always a real number.
    let med = median(&costs);
    // Residual floor validates the H as a conjugate rotation; the flatness test
    // (min far below the median) validates observability.
    if costs[kmin] > ORTHO_COST_FLOOR || costs[kmin] * 2.0 > med {
        return None;
    }
    if kmin > 0 && kmin < ORTHO_GRID_N - 1 {
        // Parabolic refinement in log f over the bracketing grid points.
        let la = fs[kmin - 1].ln();
        let lb = fs[kmin].ln();
        let (ca, cb, cc) = (costs[kmin - 1], costs[kmin], costs[kmin + 1]);
        let mut denom = ca - 2.0 * cb + cc;
        if denom == 0.0 {
            denom = 1e-12;
        }
        let lf = lb + 0.5 * (ca - cc) / denom * (lb - la);
        Some(lf.exp())
    } else {
        Some(fs[kmin])
    }
}

/// Estimate a shared focal length from cluster-track observations without any
/// reconstruction. See `specs/core/geometry/focal-vote.md`.
///
/// `cluster_indexes` must be nondecreasing (each distinct cluster is a
/// contiguous run); `image_indexes` and `positions_xy` are the image id and
/// full-pixel keypoint position per observation. The principal point is the
/// image centre `(width/2, height/2)`.
pub fn focal_vote(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    width: u32,
    height: u32,
    seed: u64,
) -> FocalVoteResult {
    focal_vote_with_min_disp(
        cluster_indexes,
        image_indexes,
        positions_xy,
        width,
        height,
        seed,
        EPIPOLAR_MIN_DISP_FRAC,
    )
}

/// `focal_vote` with an explicit epipolar displacement floor (fraction of the
/// image diagonal a candidate pair's mean feature displacement must reach).
/// The floor is the wide-baseline gate: too low admits near-static pairs whose
/// ill-conditioned fundamental matrices vote junk focals into the pool.
pub fn focal_vote_with_min_disp(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    width: u32,
    height: u32,
    seed: u64,
    epipolar_min_disp_frac: f64,
) -> FocalVoteResult {
    focal_vote_with_options(
        cluster_indexes,
        image_indexes,
        positions_xy,
        width,
        height,
        &FocalVoteOptions {
            seed,
            epipolar_min_disp_frac,
            ..Default::default()
        },
    )
}

/// `focal_vote` over an explicit camera-model column set (see
/// `specs/core/geometry/focal-vote.md`, "Camera-Model Columns").
///
/// With the default pinhole-only column set this is exactly the closed-form
/// kernel; adding the equidistant-fisheye column runs the two self-consistency
/// scans in **both** columns (so their certified masses are comparable), picks
/// the model by certified model-informative mass, and reports the winning
/// column's consensus.
pub fn focal_vote_with_options(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    width: u32,
    height: u32,
    options: &FocalVoteOptions,
) -> FocalVoteResult {
    let seed = options.seed;
    let epipolar_min_disp_frac = options.epipolar_min_disp_frac;
    let columns = canonical_columns(&options.columns);
    let empty = FocalVoteResult {
        focal_px: None,
        family: None,
        epipolar_focal_px: None,
        rotation_focal_px: None,
        n_epipolar: 0,
        n_rotation: 0,
        n_pool: 0,
        pool_spread: 0.0,
        family_disagreement: None,
        parallax_poverty: 0.0,
        epipolar_spread: 0.0,
        rotation_spread: 0.0,
        epipolar_votes: Vec::new(),
        rotation_votes: Vec::new(),
        n_h_dominated: 0,
        n_estimator_failed: 0,
        n_band_rejected: 0,
        n_degenerate: 0,
        n_inconsistent_pairs: 0,
        camera_model: None,
        columns: Vec::new(),
    };
    let n_obs = cluster_indexes.len();
    if n_obs == 0 || image_indexes.len() != n_obs || positions_xy.len() != n_obs {
        return empty;
    }

    let n_img = match image_indexes.iter().max() {
        Some(&m) => m as usize + 1,
        None => return empty,
    };
    let pp = [width as f64 / 2.0, height as f64 / 2.0];
    let max_wh = width.max(height) as f64;
    let diag = (width as f64).hypot(height as f64);

    // ── Pair tables: one pass over cluster runs ──────────────────────────────
    // Each cluster's covisible member pairs contribute to their image pair's
    // shared-cluster count and mean feature displacement. The same pass builds,
    // per image, the (run, position) list used for the full-correspondence
    // merge-join. Counts are the true shared-cluster covisibility; a sampled
    // single pair per cluster undercounts too far to reach the 25/30 cluster
    // thresholds on parallax-poor captures — see the spec's "Pair tables".
    let mut image_clusters: ImageClusters = vec![Vec::new(); n_img];
    let mut pair_accum: HashMap<(u32, u32), PairAccum> = HashMap::new();

    let mut run_start = 0usize;
    let mut run_idx: u32 = 0;
    while run_start < n_obs {
        let cid = cluster_indexes[run_start];
        let mut run_end = run_start + 1;
        while run_end < n_obs && cluster_indexes[run_end] == cid {
            run_end += 1;
        }

        // Per-image dedupe (last observation wins, mirroring the reference's
        // (cluster, image) row map) for the correspondence lists.
        let mut last_seen: HashMap<u32, [f64; 2]> = HashMap::new();
        for r in run_start..run_end {
            last_seen.insert(image_indexes[r], positions_xy[r]);
        }
        let mut members: Vec<(u32, [f64; 2])> = last_seen.into_iter().collect();
        members.sort_by_key(|m| m.0);
        for &(img, pos) in &members {
            image_clusters[img as usize].push((run_idx, pos));
        }

        // Every covisible member pair (a < b) of this cluster.
        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                let (ia, pa) = members[a];
                let (ib, pb) = members[b];
                let d = (pa[0] - pb[0]).hypot(pa[1] - pb[1]);
                let e = pair_accum.entry((ia, ib)).or_default();
                e.count += 1.0;
                e.disp_sum += d;
            }
        }

        run_start = run_end;
        run_idx += 1;
    }

    // ── Epipolar votes ───────────────────────────────────────────────────────
    // Candidate pairs: shared-cluster count >= min_shared (30, relaxing to 16
    // when fewer than 6 qualify) and mean displacement >= 0.02·diagonal; admit
    // at most 2 pairs per image, up to 18.
    let qualifying = |min_shared: usize| -> Vec<(f64, u32, u32)> {
        let mut cands: Vec<(f64, u32, u32)> = pair_accum
            .iter()
            .filter(|(_, acc)| {
                acc.count as usize >= min_shared && acc.mean_disp() >= epipolar_min_disp_frac * diag
            })
            .map(|(&(a, b), acc)| (acc.count, a, b))
            .collect();
        // Deterministic: shared count descending, then pair index ascending.
        cands.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        cands
    };
    let mut cands = qualifying(MIN_SHARED_STRICT);
    if cands.len() < MIN_QUALIFYING_PAIRS {
        cands = qualifying(MIN_SHARED_RELAXED);
    }
    let mut used: HashMap<u32, u32> = HashMap::new();
    let mut epipolar_pairs: Vec<(u32, u32)> = Vec::new();
    for (_c, a, b) in cands {
        if *used.get(&a).unwrap_or(&0) >= MAX_PAIRS_PER_IMAGE
            || *used.get(&b).unwrap_or(&0) >= MAX_PAIRS_PER_IMAGE
        {
            continue;
        }
        *used.entry(a).or_insert(0) += 1;
        *used.entry(b).or_insert(0) += 1;
        epipolar_pairs.push((a, b));
        if epipolar_pairs.len() >= MAX_EPIPOLAR_PAIRS {
            break;
        }
    }

    let f_opts = FundamentalOptions {
        max_error_px: 3.0,
        seed,
        ..Default::default()
    };
    let h_opts = HomographyOptions {
        max_error_px: 3.0,
        seed,
        min_inliers: 4,
        ..Default::default()
    };

    // Each candidate pair's estimators depend on nothing but that pair's own
    // correspondences and the shared options — both `estimate_fundamental` and
    // `estimate_homography` seed their sampler from `options.seed` alone, so no
    // sampler state crosses pairs — and the pairs therefore run in parallel.
    // The outcomes come back in pair order and are folded sequentially, so
    // every counter, every pooled vote and every diagnostic entry lands exactly
    // where the serial pass put it.
    let outcomes: Vec<EpipolarPairOutcome> = epipolar_pairs
        .par_iter()
        .map(|&(a, b)| {
            epipolar_pair_outcome(
                &image_clusters,
                &pair_accum,
                a,
                b,
                pp,
                max_wh,
                &f_opts,
                &h_opts,
            )
        })
        .collect();

    // `bou` holds the pooled epipolar votes — one geometric-mean vote per
    // direction-consistent pair; `bou_detail` holds every in-band directional
    // focal (the diagnostic layer, independent of what pools).
    let mut bou: Vec<f64> = Vec::new();
    let mut bou_detail: Vec<EpipolarVote> = Vec::new();
    let mut ratios: Vec<f64> = Vec::new();
    let mut n_h_dominated = 0usize;
    let mut n_estimator_failed = 0usize;
    let mut n_band_rejected = 0usize;
    let mut n_degenerate = 0usize;
    let mut n_inconsistent_pairs = 0usize;
    for outcome in outcomes {
        if let Some(r) = outcome.ratio {
            ratios.push(r);
        }
        n_estimator_failed += usize::from(outcome.estimator_failed);
        n_h_dominated += usize::from(outcome.h_dominated);
        n_degenerate += outcome.n_degenerate;
        n_band_rejected += outcome.n_band_rejected;
        n_inconsistent_pairs += usize::from(outcome.inconsistent);
        bou_detail.extend(outcome.detail);
        if let Some(v) = outcome.vote {
            bou.push(v);
        }
    }

    // ── Rotation votes ───────────────────────────────────────────────────────
    // For a sample of images spaced to visit at most 60, the partner with the
    // largest mean displacement among pairs sharing >= 25 clusters, when that
    // displacement is >= 0.08·diagonal. Each unordered pair votes at most once:
    // two images that are each other's widest partner are reached twice, and
    // the inverse homography over the same correspondences is the same
    // measurement, not a second one — so the later occurrence is skipped.
    let step = (n_img / ROTATION_MAX_IMAGES).max(1);
    let rot_h_opts = HomographyOptions {
        max_error_px: 3.0,
        seed,
        min_inliers: ROTATION_MIN_INLIERS,
        ..Default::default()
    };
    // The images the scan visits, and each one's widest qualifying partner. The
    // partner search reduces over the pair table under a total order (mean
    // displacement descending, partner index ascending), so its answer does not
    // depend on the table's iteration order and the images may be searched in
    // parallel.
    let scanned: Vec<usize> = (0..n_img).step_by(step).collect();
    let widest: Vec<Option<(f64, u32)>> = scanned
        .par_iter()
        .map(|&i| widest_partner(&pair_accum, i))
        .collect();

    // Every visited image whose partner clears the displacement floor fits its
    // homography and self-calibrates, in parallel and independently — each fit
    // seeds its sampler from the shared options alone. A pair reached twice
    // (two images that are each other's widest partner) is therefore computed
    // twice and the second result discarded below, where the serial pass would
    // have skipped the work: the surviving values are the same either way.
    let candidates: Vec<Option<(f64, RotationVote)>> = scanned
        .par_iter()
        .zip(widest.par_iter())
        .map(|(&i, best)| {
            let &(dmean, j) = best.as_ref()?;
            if dmean < ROTATION_MIN_DISP_FRAC * diag {
                return None;
            }
            let (x1, x2) = pair_correspondences(&image_clusters, i, j as usize);
            // Centre on the principal point: H = K R K⁻¹ has K at the origin.
            let x1c: Vec<[f64; 2]> = x1.iter().map(|p| [p[0] - pp[0], p[1] - pp[1]]).collect();
            let x2c: Vec<[f64; 2]> = x2.iter().map(|p| [p[0] - pp[0], p[1] - pp[1]]).collect();
            let hest = estimate_homography(&x1c, &x2c, &rot_h_opts)?;
            let fv = rotation_self_calib_focal(&hest.h_matrix, max_wh)?;
            if fv <= FOCAL_BAND_LO * max_wh || fv >= FOCAL_BAND_HI * max_wh {
                return None;
            }
            Some((
                fv,
                RotationVote {
                    image: i as u32,
                    partner: j,
                    mean_disp_px: dmean,
                    n_inliers: hest.inliers.iter().filter(|&&k| k).count(),
                    focal_px: fv,
                },
            ))
        })
        .collect();

    let mut rot: Vec<f64> = Vec::new();
    let mut rot_detail: Vec<RotationVote> = Vec::new();
    let mut voted_pairs: HashSet<(u32, u32)> = HashSet::new();
    // The pairs the scan REACHES, deduplicated — the rotation cell's candidate
    // list for the column scans, independent of whether the closed-form
    // self-calibration accepted them.
    let mut rotation_pairs: Vec<(u32, u32)> = Vec::new();
    let mut rotation_seen: HashSet<(u32, u32)> = HashSet::new();
    for ((&i, best), candidate) in scanned.iter().zip(&widest).zip(candidates) {
        let Some(&(dmean, j)) = best.as_ref() else {
            continue;
        };
        if dmean < ROTATION_MIN_DISP_FRAC * diag {
            continue;
        }
        let key = ((i as u32).min(j), (i as u32).max(j));
        if rotation_seen.insert(key) {
            rotation_pairs.push(key);
        }
        if voted_pairs.contains(&key) {
            continue;
        }
        if let Some((fv, vote)) = candidate {
            rot.push(fv);
            voted_pairs.insert(key);
            rot_detail.push(vote);
        }
    }

    // ── Consensus ────────────────────────────────────────────────────────────
    // Per-pair gating already certified every surviving vote, so both families
    // pool into a single population and the consensus is its log-space median —
    // unless the two families' medians disagree so far that the pool is bimodal
    // and a blend would report a focal no pair voted for.
    // `parallax_poverty` medians H/F inlier ratios, not focals, so it stays a
    // linear median.
    // No certified pairs means no poverty evidence, which reads as zero rather
    // than as the empty median's NaN.
    let poverty = if ratios.is_empty() {
        0.0
    } else {
        median(&ratios)
    };
    let pinhole = family_consensus(&bou, &rot);

    // ── Camera-model columns ─────────────────────────────────────────────────
    // Model precedes motion family: every requested column runs the same two
    // self-consistency scans over the same candidate pairs (so the certified
    // masses are commensurable), the column with the greater model-informative
    // mass wins, and the winner's own two-family consensus is what the top
    // level reports. Pinhole-only — the default — skips all of this and keeps
    // the closed forms, bit for bit.
    let (camera_model, column_diags) = if columns == [CameraModel::Pinhole] {
        (Some(CameraModel::Pinhole), Vec::new())
    } else {
        let epi_cands = scan_candidates(&image_clusters, &epipolar_pairs, pp, seed, 0);
        let rot_cands = scan_candidates(&image_clusters, &rotation_pairs, pp, seed, 1);
        let half_diag = 0.5 * diag;
        let diags: Vec<ColumnDiagnostics> = columns
            .iter()
            .map(|&model| {
                let scan =
                    column_scan::scan_column(model, &epi_cands, &rot_cands, max_wh, half_diag);
                let (col_bou, col_rot) = match model {
                    // The pinhole column keeps its closed-form focal answer;
                    // its scans only supply arbitration certificates.
                    CameraModel::Pinhole => (bou.clone(), rot.clone()),
                    CameraModel::EquidistantFisheye => (
                        scan.certified_focals(ScanCell::Epipolar),
                        scan.certified_focals(ScanCell::Rotation),
                    ),
                };
                column_diagnostics(model, &scan, &col_bou, &col_rot)
            })
            .collect();
        (model_verdict(&diags), diags)
    };

    // Column focals are not blended — a pinhole focal and an equidistant focal
    // parameterize different maps — so the top level simply reports the winning
    // column's consensus.
    let winner = camera_model
        .filter(|_| !column_diags.is_empty())
        .or_else(|| {
            column_diags
                .iter()
                .map(|c| c.model)
                .find(|&m| m == CameraModel::Pinhole)
                .or_else(|| column_diags.first().map(|c| c.model))
        });
    let consensus = match winner.and_then(|m| column_diags.iter().find(|c| c.model == m)) {
        Some(c) if c.model != CameraModel::Pinhole => FamilyConsensus {
            focal_px: c.focal_px,
            family: c.family,
            pool_spread: c.pool_spread,
            epipolar_focal_px: c.epipolar_focal_px,
            rotation_focal_px: c.rotation_focal_px,
            epipolar_spread: c.epipolar_spread,
            rotation_spread: c.rotation_spread,
            family_disagreement: c.family_disagreement,
        },
        _ => pinhole,
    };
    let (n_epipolar, n_rotation) = match winner {
        Some(CameraModel::EquidistantFisheye) => {
            let c = column_diags
                .iter()
                .find(|c| c.model == CameraModel::EquidistantFisheye);
            c.map_or((0, 0), |c| (c.n_epipolar, c.n_rotation))
        }
        _ => (bou.len(), rot.len()),
    };

    FocalVoteResult {
        focal_px: consensus.focal_px,
        family: consensus.family,
        epipolar_focal_px: consensus.epipolar_focal_px,
        rotation_focal_px: consensus.rotation_focal_px,
        n_epipolar,
        n_rotation,
        n_pool: n_epipolar + n_rotation,
        pool_spread: consensus.pool_spread,
        family_disagreement: consensus.family_disagreement,
        parallax_poverty: poverty,
        epipolar_spread: consensus.epipolar_spread,
        rotation_spread: consensus.rotation_spread,
        epipolar_votes: bou_detail,
        rotation_votes: rot_detail,
        n_h_dominated,
        n_estimator_failed,
        n_band_rejected,
        n_degenerate,
        n_inconsistent_pairs,
        camera_model,
        columns: column_diags,
    }
}

/// Image `i`'s widest-displacement partner among the pairs sharing at least
/// [`ROTATION_MIN_SHARED`] clusters, with that mean displacement.
///
/// The reduction is a maximum under a total order — mean displacement
/// descending, partner index ascending — so it is independent of the pair
/// table's iteration order.
fn widest_partner(pair_accum: &HashMap<(u32, u32), PairAccum>, i: usize) -> Option<(f64, u32)> {
    let mut best: Option<(f64, u32)> = None;
    for (&(a, b), acc) in pair_accum {
        let partner = if a as usize == i {
            b
        } else if b as usize == i {
            a
        } else {
            continue;
        };
        if (acc.count as usize) < ROTATION_MIN_SHARED {
            continue;
        }
        let dmean = acc.mean_disp();
        let better = match best {
            None => true,
            Some((bd, bj)) => dmean > bd || (dmean == bd && partner < bj),
        };
        if better {
            best = Some((dmean, partner));
        }
    }
    best
}

/// One epipolar candidate pair's contribution to the closed-form pass.
///
/// The pair loop runs in parallel, so a pair reports what it found instead of
/// mutating the accumulators directly; the driver folds these back in pair
/// order, which is what keeps the parallel pass bit-identical to a serial one.
struct EpipolarPairOutcome {
    /// H/F inlier ratio, present only when the pair has enough F inliers to
    /// count toward `parallax_poverty`.
    ratio: Option<f64>,
    /// Too few correspondences, or no usable fundamental matrix.
    estimator_failed: bool,
    /// Homography-dominated, so the pair casts no epipolar vote.
    h_dominated: bool,
    /// Directions whose Bougnoux extraction produced no value.
    n_degenerate: usize,
    /// Directional focals outside the plausibility band.
    n_band_rejected: usize,
    /// In-band directional focals, in direction order (`F` then `Fᵀ`).
    detail: Vec<EpipolarVote>,
    /// The pair's single pooled vote, the geometric mean of two agreeing
    /// directional focals.
    vote: Option<f64>,
    /// The pair's two directions disagree, or only one was usable.
    inconsistent: bool,
}

impl EpipolarPairOutcome {
    /// A pair that produced nothing but the given failure flag.
    fn failed() -> Self {
        Self {
            ratio: None,
            estimator_failed: true,
            h_dominated: false,
            n_degenerate: 0,
            n_band_rejected: 0,
            detail: Vec::new(),
            vote: None,
            inconsistent: false,
        }
    }
}

/// The closed-form epipolar cell over one candidate pair: a robust fundamental
/// matrix, the homography-domination gate on the same correspondences, and the
/// two directional Bougnoux focals the pair's single vote is the geometric mean
/// of.
#[allow(clippy::too_many_arguments)]
fn epipolar_pair_outcome(
    image_clusters: &ImageClusters,
    pair_accum: &HashMap<(u32, u32), PairAccum>,
    a: u32,
    b: u32,
    pp: [f64; 2],
    max_wh: f64,
    f_opts: &FundamentalOptions,
    h_opts: &HomographyOptions,
) -> EpipolarPairOutcome {
    let (x1, x2) = pair_correspondences(image_clusters, a as usize, b as usize);
    if x1.len() < 8 {
        return EpipolarPairOutcome::failed();
    }
    let Some(fest) = estimate_fundamental(&x1, &x2, f_opts) else {
        return EpipolarPairOutcome::failed();
    };
    let n_f = fest.inliers.iter().filter(|&&b| b).count();
    let n_h = estimate_homography(&x1, &x2, h_opts)
        .map(|h| h.inliers.iter().filter(|&&b| b).count())
        .unwrap_or(0);
    let ratio = (n_f >= RATIO_MIN_F_INLIERS).then(|| n_h as f64 / n_f as f64);
    // Homography-dominated: F is collapsing toward H, no epipolar vote.
    if (n_h as f64) >= 16.0_f64.max(0.8 * n_f as f64) {
        return EpipolarPairOutcome {
            ratio,
            estimator_failed: false,
            h_dominated: true,
            n_degenerate: 0,
            n_band_rejected: 0,
            detail: Vec::new(),
            vote: None,
            inconsistent: false,
        };
    }
    let acc = pair_accum[&(a, b)];
    let mut n_degenerate = 0usize;
    let mut n_band_rejected = 0usize;
    let mut detail: Vec<EpipolarVote> = Vec::with_capacity(2);
    let mut in_band: Vec<f64> = Vec::with_capacity(2);
    for (transposed, f_dir) in [(false, fest.f_matrix), (true, fest.f_matrix.transpose())] {
        // A direction whose Bougnoux extraction is degenerate yields no value at
        // all — separate from an out-of-band value.
        let Some(v) = focal_from_fundamental(&f_dir, pp, pp) else {
            n_degenerate += 1;
            continue;
        };
        if v > FOCAL_BAND_LO * max_wh && v < FOCAL_BAND_HI * max_wh {
            in_band.push(v);
            detail.push(EpipolarVote {
                image_a: a,
                image_b: b,
                shared_clusters: acc.count,
                mean_disp_px: acc.mean_disp(),
                n_f_inliers: n_f,
                n_h_inliers: n_h,
                transposed,
                focal_px: v,
            });
        } else {
            n_band_rejected += 1;
        }
    }
    // The two cameras share the focal, so the two directional focals are two
    // measurements of the same quantity. Agreement certifies the pair; it then
    // casts ONE vote, their geometric mean. A pair that disagrees (or has only
    // one in-band direction) carries no consistent focal.
    let (vote, inconsistent) = match in_band.as_slice() {
        [f0, f1] if (f0.ln() - f1.ln()).abs() <= DIRECTION_AGREEMENT_BAND => {
            (Some((f0 * f1).sqrt()), false)
        }
        [_, _] | [_] => (None, true),
        _ => (None, false),
    };
    EpipolarPairOutcome {
        ratio,
        estimator_failed: false,
        h_dominated: false,
        n_degenerate,
        n_band_rejected,
        detail,
        vote,
        inconsistent,
    }
}

/// Build the scan candidates of one cell: each pair's full correspondence
/// merge-join, centred on the principal point and capped. `cell_tag` separates
/// the two cells' sampler streams so a pair reached by both draws different
/// minimal samples in each.
fn scan_candidates(
    image_clusters: &ImageClusters,
    pairs: &[(u32, u32)],
    pp: [f64; 2],
    seed: u64,
    cell_tag: u64,
) -> Vec<column_scan::ScanCandidate> {
    pairs
        .iter()
        .enumerate()
        .map(|(k, &(a, b))| {
            let (x1, x2) = pair_correspondences(image_clusters, a as usize, b as usize);
            // Deterministic per-pair stream: position in the candidate list and
            // cell tag, mixed with the kernel seed.
            let pair_seed = seed
                ^ (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ cell_tag.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            column_scan::ScanCandidate::new(a, b, &x1, &x2, pp, pair_seed)
        })
        .collect()
}

/// Roll one column's scans and its own family votes into the diagnostics block.
fn column_diagnostics(
    model: CameraModel,
    scan: &ColumnScan,
    bou: &[f64],
    rot: &[f64],
) -> ColumnDiagnostics {
    let c = family_consensus(bou, rot);
    let count = |v: &[ScanVote], f: fn(&ScanVote) -> bool| v.iter().filter(|s| f(s)).count();
    let mut scan_votes = scan.epipolar.clone();
    scan_votes.extend(scan.rotation.iter().copied());
    ColumnDiagnostics {
        model,
        focal_px: c.focal_px,
        family: c.family,
        epipolar_focal_px: c.epipolar_focal_px,
        rotation_focal_px: c.rotation_focal_px,
        n_epipolar: bou.len(),
        n_rotation: rot.len(),
        n_pool: bou.len() + rot.len(),
        pool_spread: c.pool_spread,
        family_disagreement: c.family_disagreement,
        epipolar_spread: c.epipolar_spread,
        rotation_spread: c.rotation_spread,
        parallax_poverty: scan.parallax_poverty(),
        n_rotation_dominated: scan.n_rotation_dominated(),
        n_scanned_epipolar: scan.epipolar.len(),
        n_scanned_rotation: scan.rotation.len(),
        n_certified_epipolar: count(&scan.epipolar, |s| s.certified),
        n_certified_rotation: count(&scan.rotation, |s| s.certified),
        n_informative_epipolar: count(&scan.epipolar, |s| s.model_informative),
        n_informative_rotation: count(&scan.rotation, |s| s.model_informative),
        n_certified: scan.n_certified(),
        n_informative: scan.n_informative(),
        scan_votes,
    }
}

/// The model verdict: the column with the greater certified model-informative
/// mass. A single column has nothing to arbitrate and wins by construction;
/// with several, ties go to the earlier column in canonical order (pinhole),
/// and a capture where no column has any model-informative vote gets no
/// verdict at all.
fn model_verdict(diags: &[ColumnDiagnostics]) -> Option<CameraModel> {
    let best = diags.iter().max_by_key(|c| c.n_informative)?;
    if diags.len() > 1 && best.n_informative == 0 {
        return None;
    }
    // `max_by_key` returns the LAST maximum; ties must go to the first.
    diags
        .iter()
        .find(|c| c.n_informative == best.n_informative)
        .map(|c| c.model)
}

// The synthetic captures here are the crate's fixtures for the vote, and
// `geometry::estimate_intrinsics` reads its verdicts off the same ones rather
// than growing a second copy of them.
#[cfg(test)]
pub(crate) mod tests;
