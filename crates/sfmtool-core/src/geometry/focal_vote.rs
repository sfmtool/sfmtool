// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Structure-free focal-length estimation by pairwise voting
//! ([`focal_vote`]). See `specs/core/focal-vote.md`.
//!
//! Image pairs drawn from cluster-track observations each cast one focal vote
//! through whichever of two estimators their geometry can observe, and the
//! consensus focal is the median of the winning family:
//!
//! - **Epipolar** — pairs with parallax vote the Bougnoux focal of a robustly
//!   estimated fundamental matrix.
//! - **Rotation** — pairs dominated by a parallax-free homography vote by
//!   rotation self-calibration: `H = K R K⁻¹`, so the focal is the `f` that
//!   makes `K⁻¹ H K` orthogonal.
//!
//! Each estimator is degenerate exactly where the other is informative; a
//! per-pair split plus a capture-level poverty arbitration keeps each on its
//! own ground. Because no structure is estimated, the vote cannot be biased by
//! the depth/focal (bas-relief) compensation that afflicts structure-based
//! focal estimation.
//!
//! The pair-table pass is deterministic and the RANSAC estimators derive their
//! sampling from the input seed, so identical inputs and seed reproduce
//! identical output.

use std::collections::HashMap;

use nalgebra::Matrix3;

use crate::geometry::epipolar_estimation::{
    estimate_fundamental, focal_from_fundamental, FundamentalOptions,
};
use crate::geometry::homography_estimation::{estimate_homography, HomographyOptions};

/// Which family produced the consensus focal.
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

/// Result of [`focal_vote`].
#[derive(Clone, Debug)]
pub struct FocalVoteResult {
    /// Consensus focal in pixels, `None` when neither family reaches quorum.
    pub focal_px: Option<f64>,
    /// Which family produced `focal_px`, `None` when there is no consensus.
    pub family: Option<VoteFamily>,
    /// Median of the epipolar votes (diagnostic).
    pub epipolar_focal_px: Option<f64>,
    /// Median of the rotation votes (diagnostic).
    pub rotation_focal_px: Option<f64>,
    /// Number of epipolar (Bougnoux) votes.
    pub n_epipolar: usize,
    /// Number of rotation (self-calibration) votes.
    pub n_rotation: usize,
    /// Median H/F inlier ratio over the epipolar candidate pairs.
    pub parallax_poverty: f64,
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

const POVERTY_THRESHOLD: f64 = 0.55;
const ROT_QUORUM_HIGH: usize = 5;
const EPIPOLAR_QUORUM: usize = 8;
const ROT_QUORUM_LOW: usize = 6;

/// numpy-style median (even length averages the two central elements).
fn median(vals: &[f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    let mut v = vals.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

/// Per-image-pair accumulator from the sampled pass: how many clusters sampled
/// this pair, and the sum of their feature displacements.
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
fn ortho_cost(h: &Matrix3<f64>, f: f64) -> f64 {
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
    let med = median(&costs)?;
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
/// reconstruction. See `specs/core/focal-vote.md`.
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
    let empty = FocalVoteResult {
        focal_px: None,
        family: None,
        epipolar_focal_px: None,
        rotation_focal_px: None,
        n_epipolar: 0,
        n_rotation: 0,
        parallax_poverty: 0.0,
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
    // merge-join. Counts are the true shared-cluster covisibility (the sampled
    // single-pair estimate of the spec undercounts too far to reach the 25/30
    // thresholds on parallax-poor captures — see the spec's deviation note).
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
                acc.count as usize >= min_shared && acc.mean_disp() >= EPIPOLAR_MIN_DISP_FRAC * diag
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

    let mut bou: Vec<f64> = Vec::new();
    let mut ratios: Vec<f64> = Vec::new();
    for (a, b) in epipolar_pairs {
        let (x1, x2) = pair_correspondences(&image_clusters, a as usize, b as usize);
        if x1.len() < 8 {
            continue;
        }
        let Some(fest) = estimate_fundamental(&x1, &x2, &f_opts) else {
            continue;
        };
        let n_f = fest.inliers.iter().filter(|&&b| b).count();
        let n_h = estimate_homography(&x1, &x2, &h_opts)
            .map(|h| h.inliers.iter().filter(|&&b| b).count())
            .unwrap_or(0);
        if n_f >= RATIO_MIN_F_INLIERS {
            ratios.push(n_h as f64 / n_f as f64);
        }
        // Homography-dominated: F is collapsing toward H, no epipolar vote.
        if (n_h as f64) >= 16.0_f64.max(0.8 * n_f as f64) {
            continue;
        }
        for f_dir in [fest.f_matrix, fest.f_matrix.transpose()] {
            if let Some(v) = focal_from_fundamental(&f_dir, pp, pp) {
                if v > FOCAL_BAND_LO * max_wh && v < FOCAL_BAND_HI * max_wh {
                    bou.push(v);
                }
            }
        }
    }

    // ── Rotation votes ───────────────────────────────────────────────────────
    // For a sample of images spaced to visit at most 60, the partner with the
    // largest mean displacement among pairs sharing >= 25 clusters, when that
    // displacement is >= 0.08·diagonal.
    let step = (n_img / ROTATION_MAX_IMAGES).max(1);
    let rot_h_opts = HomographyOptions {
        max_error_px: 3.0,
        seed,
        min_inliers: ROTATION_MIN_INLIERS,
        ..Default::default()
    };
    let mut rot: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i < n_img {
        let mut best: Option<(f64, u32)> = None;
        for (&(a, b), acc) in &pair_accum {
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
        if let Some((dmean, j)) = best {
            if dmean >= ROTATION_MIN_DISP_FRAC * diag {
                let (x1, x2) = pair_correspondences(&image_clusters, i, j as usize);
                // Centre on the principal point: H = K R K⁻¹ has K at the origin.
                let x1c: Vec<[f64; 2]> = x1.iter().map(|p| [p[0] - pp[0], p[1] - pp[1]]).collect();
                let x2c: Vec<[f64; 2]> = x2.iter().map(|p| [p[0] - pp[0], p[1] - pp[1]]).collect();
                if let Some(hest) = estimate_homography(&x1c, &x2c, &rot_h_opts) {
                    if let Some(fv) = rotation_self_calib_focal(&hest.h_matrix, max_wh) {
                        if fv > FOCAL_BAND_LO * max_wh && fv < FOCAL_BAND_HI * max_wh {
                            rot.push(fv);
                        }
                    }
                }
            }
        }
        i += step;
    }

    // ── Arbitration ──────────────────────────────────────────────────────────
    let poverty = median(&ratios).unwrap_or(0.0);
    let epipolar_focal_px = median(&bou);
    let rotation_focal_px = median(&rot);
    let n_epipolar = bou.len();
    let n_rotation = rot.len();

    let (focal_px, family) = if n_rotation >= ROT_QUORUM_HIGH && poverty >= POVERTY_THRESHOLD {
        (rotation_focal_px, Some(VoteFamily::Rotation))
    } else if n_epipolar >= EPIPOLAR_QUORUM {
        (epipolar_focal_px, Some(VoteFamily::Epipolar))
    } else if n_rotation >= ROT_QUORUM_LOW {
        (rotation_focal_px, Some(VoteFamily::Rotation))
    } else {
        (None, None)
    };
    // A chosen family with no median (empty vote list) collapses to no consensus.
    let (focal_px, family) = match focal_px {
        Some(f) => (Some(f), family),
        None => (None, None),
    };

    FocalVoteResult {
        focal_px,
        family,
        epipolar_focal_px,
        rotation_focal_px,
        n_epipolar,
        n_rotation,
        parallax_poverty: poverty,
    }
}

#[cfg(test)]
mod tests;
