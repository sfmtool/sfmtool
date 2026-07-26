// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cluster match census: does a candidate solve agree with the raw
//! correspondence evidence it did *not* consume?
//!
//! A reconstruction can be internally consistent and wrong — a viewpoint group glued
//! at the wrong relative pose, or poses and structure bent to absorb a wrong
//! focal. Neither shows up in the solve's own reprojection error, because the
//! solve chose the tracks it is measured on. What does discriminate is the raw
//! cluster set the solve never used: eligible, high-parallax clusters whose
//! members span distinct viewpoint groups constrain those groups' relative
//! placement, and a misplaced (or focal-bent) solve necessarily leaves a
//! fraction of them unsatisfiable.
//!
//! [`cluster_census`] measures that fraction:
//!
//! 1. Partition the posed images into **viewpoint groups** by greedy-modularity
//!    (CNM) communities of the *raw* cluster-covisibility graph
//!    ([`ClusterCovisibility`]) — never the solve's own track graph, which is
//!    glued across a bad seam by construction.
//! 2. Triangulate **every** raw cluster at the candidate poses
//!    ([`triangulate_batch`]) and take each cluster's *median* reprojection
//!    residual and its triangulation parallax.
//! 3. Accept a cluster as a genuine correspondence when its matching-time
//!    warp-consistency residual is within the **P95 of the residuals of the
//!    clusters the candidate satisfies** — a data-derived bar, tight on a clean
//!    capture and loose on a noisy one.
//! 4. Census each group pair over its eligible, high-parallax bridges: the
//!    Wilson lower bound of the unsatisfied fraction. The score is the maximum
//!    over pairs, so a fine partition cannot dilute one bad seam with the
//!    satisfied bridges of good seams.
//!
//! Fewer than two groups means the capture has no group structure to census: the
//! result is *unverifiable* (`n_groups < 2`, score 0), which callers must read
//! as "no evidence", not "clean".
//!
//! Deterministic: the community merge order, the tie-breaking rule, and every
//! reduction below are fixed functions of the input arrays.
//!
//! Phase 1 (this module) implements the score, the groups, the per-pair stats,
//! and `sat_pct`. The group-consistency companion ([`GroupConsistency`]) is
//! phase 2; [`CensusReport::group_consistency`] is always `None` today.
//!
//! See `specs/core/cluster-census.md` for the design.

use std::collections::BTreeMap;

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};

use crate::features::cluster_match::covisibility::{
    ClusterCovisibility, CovisibilityError, MAX_DENSE_IMAGES,
};
use crate::geometry::reprojection::reprojection_residuals;
use crate::reconstruction::triangulation::triangulate_batch;
use crate::CameraIntrinsics;

/// Residual magnitude assigned to an observation whose point is behind the
/// camera or outside the model's valid domain. Large but finite, so the
/// per-cluster median stays a number and the cluster reads as unsatisfied.
const INVALID_RESIDUAL_PX: f64 = 1e6;

/// Tuning for [`cluster_census`]; [`Default`] is the spec's parameter table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CensusParams {
    /// Satisfied / unsatisfied bar on a cluster's median reprojection residual
    /// (px). The pipeline's shared inlier threshold.
    pub sat_px: f64,
    /// Parallax floor (degrees) for a bridge to carry constraint. Low-parallax
    /// bridges tolerate large placement and focal error, so they contribute
    /// count without evidence.
    pub hi_parallax_deg: f64,
    /// Percentile of the satisfied clusters' warp-consistency residuals that
    /// becomes the eligibility threshold.
    pub warp_percentile: f64,
    /// Wilson bound z (1.96 = standard 95 %).
    pub wilson_z: f64,
}

impl Default for CensusParams {
    fn default() -> Self {
        Self {
            sat_px: 2.0,
            hi_parallax_deg: 5.0,
            warp_percentile: 95.0,
            wilson_z: 1.96,
        }
    }
}

/// One group pair's census.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairStats {
    /// Lower group id of the pair.
    pub group_a: u32,
    /// Higher group id of the pair.
    pub group_b: u32,
    /// Eligible, high-parallax bridges of this pair (the denominator).
    pub n_eligible_hi: u32,
    /// How many of those the candidate cannot satisfy (the numerator).
    pub n_unsatisfied_hi: u32,
    /// Wilson lower bound of `n_unsatisfied_hi / n_eligible_hi`.
    pub wilson_lb: f64,
}

/// One group's pose correction from the phase-2 group-consistency solve:
/// the 7-dof similarity that, applied to this group, best satisfies the
/// eligible bridges (the largest group fixes the gauge with an identity
/// correction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupCorrection {
    /// The corrected group.
    pub group: u32,
    /// Rotation of the similarity (WXYZ).
    pub rotation_wxyz: [f64; 4],
    /// Translation of the similarity, in world units.
    pub translation: [f64; 3],
    /// Log scale of the similarity.
    pub log_scale: f64,
}

/// Phase-2 group consistency: the joint per-group corrections that best
/// explain the cross-group disagreement, and how much of it they explain.
/// Not yet computed — [`CensusReport::group_consistency`] is always `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupConsistency {
    /// Per-group corrections; the gauge group carries the identity.
    pub corrections: Vec<GroupCorrection>,
    /// Percent of the previously-unsatisfied high-parallax bridges
    /// satisfied at the corrected placements — the coherence of the
    /// disagreement.
    pub explained_pct: f64,
    /// Satisfied bridges before the corrections.
    pub net_before: usize,
    /// Satisfied bridges after them. `net_after > net_before` is a
    /// necessary condition for the corrections to be a genuine
    /// explanation.
    pub net_after: usize,
}

/// Result of [`cluster_census`].
#[derive(Clone, Debug)]
pub struct CensusReport {
    /// Maximum per-pair Wilson lower bound — the census score. Vacuous (0)
    /// when `n_groups < 2`.
    pub score: f64,
    /// Number of viewpoint groups. Below 2 the capture is *unverifiable*.
    pub n_groups: usize,
    /// Group id per input image, `-1` for an unposed image.
    pub group_of: Vec<i32>,
    /// Per-pair census, ascending by `(group_a, group_b)`. Every group pair
    /// joined by at least one bridge cluster appears, including pairs with no
    /// eligible high-parallax evidence (`n_eligible_hi == 0`, `wilson_lb == 0`).
    pub pairs: Vec<PairStats>,
    /// Percent of all eligible, measurable clusters the candidate satisfies. A
    /// globally-deformed solve degrades this without producing a large
    /// pairwise census, so gating callers should test both.
    pub sat_pct: f64,
    /// Phase 2; always `None` today.
    pub group_consistency: Option<GroupConsistency>,
}

/// Input-validation failures of [`cluster_census`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CensusError {
    /// Two arrays that must be parallel are not.
    NotParallel {
        /// Name of the offending array.
        name: &'static str,
        /// Its length.
        len: usize,
        /// The length it had to match.
        expected: usize,
    },
    /// `cluster_indexes` is not nondecreasing (clusters must be contiguous
    /// runs, the `.matches` clusters-backbone layout).
    ClusterIndexesNotSorted {
        /// Position of the first descending step.
        at: usize,
    },
    /// An observation names a cluster with no `cluster_warp_consistency` entry.
    ClusterIndexOutOfRange {
        /// The offending cluster id.
        index: u32,
        /// Number of clusters (`cluster_warp_consistency.len()`).
        num_clusters: usize,
    },
    /// The covisibility graph could not be built.
    Covisibility(CovisibilityError),
}

impl std::fmt::Display for CensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotParallel {
                name,
                len,
                expected,
            } => write!(f, "{name} has length {len}, expected {expected}"),
            Self::ClusterIndexesNotSorted { at } => write!(
                f,
                "cluster_indexes must be nondecreasing; it decreases at index {at}"
            ),
            Self::ClusterIndexOutOfRange {
                index,
                num_clusters,
            } => write!(
                f,
                "cluster index {index} is out of range for {num_clusters} clusters"
            ),
            Self::Covisibility(e) => write!(f, "covisibility construction failed: {e}"),
        }
    }
}

impl std::error::Error for CensusError {}

impl From<CovisibilityError> for CensusError {
    fn from(e: CovisibilityError) -> Self {
        Self::Covisibility(e)
    }
}

/// Wilson lower confidence bound of a binomial proportion `k / n`. Zero for an
/// empty denominator; shrinks small denominators toward zero.
pub fn wilson_lower_bound(k: u32, n: u32, z: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n = f64::from(n);
    let p = f64::from(k) / n;
    let d = 1.0 + z * z / n;
    let c = p + z * z / (2.0 * n);
    let r = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
    ((c - r) / d).max(0.0)
}

/// Linearly-interpolated percentile of `values` (NumPy's default `linear`
/// method, including its `t >= 0.5` reformulation, so the threshold matches
/// the reference prototype bit for bit). `values` is sorted in place;
/// `percentile` is in `[0, 100]`. Empty input gives `+∞`.
fn percentile_linear(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let pos = (n - 1) as f64 * (percentile / 100.0);
    let lo = pos.floor();
    let lo_i = (lo as usize).min(n - 1);
    let hi_i = (lo_i + 1).min(n - 1);
    let t = pos - lo;
    let (a, b) = (values[lo_i], values[hi_i]);
    if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// Greedy-modularity (CNM) communities of a dense symmetric weight matrix,
/// returning `(label per node, group count)`.
///
/// Communities start as singletons and merge, one pair per step, by the
/// largest modularity gain `ΔQ = 2·(w_ab/2m − k_a·k_b/(2m)²)`; the reported
/// partition is the best-`Q` one over the whole merge path (first partition to
/// attain the maximum wins, comparison strictly greater). Group ids are the
/// positions of the communities in the live community list at that point.
///
/// **Tie-breaking (determinism contract).** Pairs are scanned in ascending
/// `(a, b)` order with `a < b` over the live community list and the *last*
/// maximal gain wins — equivalently, the merge maximizes the tuple
/// `(ΔQ, a, b)` lexicographically. This reproduces the reference prototype's
/// `max()` over `(gain, a, b)` tuples exactly; with integer edge weights the
/// gains are computed from exact integer sums, so the whole merge path is bit
/// reproducible.
///
/// Bookkeeping is incremental: the inter-community weight matrix and degree
/// sums are updated in place on each merge (`O(m)` for the merge, `O(m²)` for
/// the pair scan), so `Q` is re-formed from exact community sums rather than
/// re-summed over every node pair.
///
/// Degenerate inputs (no edges, or two or fewer nodes) collapse to one group.
fn modularity_groups(w: &[f64], n: usize) -> (Vec<usize>, usize) {
    if n == 0 {
        return (Vec::new(), 0);
    }
    let mut k = vec![0.0f64; n];
    for (i, ki) in k.iter_mut().enumerate() {
        *ki = w[i * n..(i + 1) * n].iter().sum();
    }
    let two_m: f64 = k.iter().sum();
    if two_m == 0.0 || n <= 2 {
        return (vec![0; n], 1);
    }

    // Live community state: membership lists, the inter-community weight
    // matrix `e` (raw sums of `w`, exact for integer counts), and the
    // per-community degree sums `kc`.
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut e: Vec<Vec<f64>> = (0..n).map(|i| w[i * n..(i + 1) * n].to_vec()).collect();
    let mut kc = k;

    let modularity = |e: &[Vec<f64>], kc: &[f64]| -> f64 {
        (0..kc.len())
            .map(|c| e[c][c] / two_m - kc[c] * kc[c] / (two_m * two_m))
            .sum()
    };

    let mut q = modularity(&e, &kc);
    let mut best_q = q;
    let mut best = members.clone();

    while members.len() > 1 {
        let m = members.len();
        // Scan in ascending (a, b); `>=` keeps the last maximal gain, which is
        // the lexicographically largest (a, b) — see the doc comment.
        let mut best_gain = f64::NEG_INFINITY;
        let mut merge = (0usize, 1usize);
        for a in 0..m {
            for b in (a + 1)..m {
                let gain = 2.0 * (e[a][b] / two_m - kc[a] * kc[b] / (two_m * two_m));
                if gain >= best_gain {
                    best_gain = gain;
                    merge = (a, b);
                }
            }
        }
        let (a, b) = merge;

        // Absorb b into a: the merged community's internal weight picks up
        // both cross terms, and every other community's edge to a gains its
        // edge to b.
        let e_aa = e[a][a] + e[b][b] + 2.0 * e[a][b];
        let row_b = e[b].clone();
        for (x, v_b) in row_b.iter().enumerate().take(m) {
            if x != a && x != b {
                let v = e[a][x] + v_b;
                e[a][x] = v;
                e[x][a] = v;
            }
        }
        e[a][a] = e_aa;
        kc[a] += kc[b];
        let absorbed = members.remove(b);
        members[a].extend(absorbed);
        e.remove(b);
        for row in e.iter_mut() {
            row.remove(b);
        }
        kc.remove(b);

        q = modularity(&e, &kc);
        if q > best_q {
            best_q = q;
            best = members.clone();
        }
    }

    let mut labels = vec![0usize; n];
    for (g, community) in best.iter().enumerate() {
        for &i in community {
            labels[i] = g;
        }
    }
    (labels, best.len())
}

/// Median of a scratch buffer, matching the prototype's two-index form
/// (average of the middle pair for an even count).
fn median_in_place(buf: &mut [f64]) -> f64 {
    if buf.is_empty() {
        return f64::NAN;
    }
    buf.sort_by(f64::total_cmp);
    let n = buf.len();
    0.5 * (buf[(n - 1) / 2] + buf[n / 2])
}

/// Length check helper.
fn require_len(name: &'static str, len: usize, expected: usize) -> Result<(), CensusError> {
    if len == expected {
        Ok(())
    } else {
        Err(CensusError::NotParallel {
            name,
            len,
            expected,
        })
    }
}

/// Score a candidate solve against the raw correspondence evidence it did not
/// consume. See the module docs and `specs/core/cluster-census.md`.
///
/// Observations are the raw `.matches` clusters backbone flattened CSR-style:
/// `cluster_indexes` must be nondecreasing (each cluster a contiguous run),
/// with `image_indexes` and `positions_xy` (full-pixel) parallel to it.
/// `cluster_warp_consistency` is one entry per cluster — the worst finite
/// member residual of the matching-time consistency fit, lower is better,
/// non-finite for a cluster that never entered the fit.
///
/// The candidate is `quaternions_wxyz` / `translations` (canonical
/// world-to-camera, the camera looks along −Z) for the images named by
/// `posed_indexes`, all sharing `camera`. The candidate's own tracks, points,
/// and residuals are deliberately not inputs.
///
/// Deterministic: identical inputs give identical output.
#[allow(clippy::too_many_arguments)]
pub fn cluster_census(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    cluster_warp_consistency: &[f64],
    camera: &CameraIntrinsics,
    quaternions_wxyz: &[[f64; 4]],
    translations: &[[f64; 3]],
    posed_indexes: &[u32],
    params: &CensusParams,
) -> Result<CensusReport, CensusError> {
    let n_obs = cluster_indexes.len();
    require_len("image_indexes", image_indexes.len(), n_obs)?;
    require_len("positions_xy", positions_xy.len(), n_obs)?;
    let n_posed = posed_indexes.len();
    require_len("quaternions_wxyz", quaternions_wxyz.len(), n_posed)?;
    require_len("translations", translations.len(), n_posed)?;
    if let Some(at) = cluster_indexes.windows(2).position(|w| w[1] < w[0]) {
        return Err(CensusError::ClusterIndexesNotSorted { at: at + 1 });
    }
    let n_cl = cluster_warp_consistency.len();
    if let Some(&index) = cluster_indexes.iter().find(|&&c| c as usize >= n_cl) {
        return Err(CensusError::ClusterIndexOutOfRange {
            index,
            num_clusters: n_cl,
        });
    }
    let n_img = image_indexes
        .iter()
        .chain(posed_indexes)
        .map(|&i| i as usize + 1)
        .max()
        .unwrap_or(0);
    // The same bound `from_clusters` enforces, hoisted above the dense
    // per-image allocations so a stray huge index fails in O(1) instead of
    // sizing gigabytes first.
    if n_img > MAX_DENSE_IMAGES {
        return Err(CovisibilityError::TooManyImages { num_images: n_img }.into());
    }

    // ── Candidate poses ──────────────────────────────────────────────────
    let mut posed = vec![false; n_img];
    let mut quats = vec![UnitQuaternion::<f64>::identity(); n_img];
    let mut centers = vec![Point3::<f64>::origin(); n_img];
    let mut quats_flat = vec![0.0f64; n_img * 4];
    let mut trans_flat = vec![0.0f64; n_img * 3];
    for (s, &i) in posed_indexes.iter().enumerate() {
        let i = i as usize;
        let [qw, qx, qy, qz] = quaternions_wxyz[s];
        let q = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
        let t = Vector3::new(translations[s][0], translations[s][1], translations[s][2]);
        quats[i] = q;
        // Camera center C = −Rᵀ·t.
        centers[i] = Point3::from(-(q.inverse() * t));
        posed[i] = true;
        let qn = q.into_inner();
        quats_flat[i * 4] = qn.w;
        quats_flat[i * 4 + 1] = qn.i;
        quats_flat[i * 4 + 2] = qn.j;
        quats_flat[i * 4 + 3] = qn.k;
        trans_flat[i * 3] = t.x;
        trans_flat[i * 3 + 1] = t.y;
        trans_flat[i * 3 + 2] = t.z;
    }

    // ── 1. Viewpoint groups: CNM communities of raw cluster covisibility ──
    let mut cluster_starts = vec![0u32; n_cl + 1];
    for &c in cluster_indexes {
        cluster_starts[c as usize + 1] += 1;
    }
    for c in 0..n_cl {
        cluster_starts[c + 1] += cluster_starts[c];
    }
    let accepted: Vec<bool> = image_indexes.iter().map(|&i| posed[i as usize]).collect();
    let covis =
        ClusterCovisibility::from_clusters(&cluster_starts, image_indexes, Some(&accepted), n_img)?;

    let posed_images: Vec<u32> = (0..n_img).filter(|&i| posed[i]).map(|i| i as u32).collect();
    let np = posed_images.len();
    let mut w = vec![0.0f64; np * np];
    for (a, &ia) in posed_images.iter().enumerate() {
        for (b, &ib) in posed_images.iter().enumerate() {
            w[a * np + b] = f64::from(covis.count(ia, ib));
        }
    }
    let (labels, n_groups) = modularity_groups(&w, np);
    let mut group_of = vec![-1i32; n_img];
    for (a, &i) in posed_images.iter().enumerate() {
        group_of[i as usize] = labels[a] as i32;
    }

    // ── 2. Cluster placement at the candidate ────────────────────────────
    // Posed observations only, grouped into contiguous per-cluster segments
    // (the input's cluster order is preserved by the filter).
    let mut seg_cluster: Vec<u32> = Vec::new();
    let mut seg_offsets: Vec<usize> = Vec::new();
    let mut obs_image: Vec<u32> = Vec::new();
    let mut obs_uv: Vec<f64> = Vec::new();
    let mut dirs: Vec<Vector3<f64>> = Vec::new();
    let mut obs_center: Vec<Point3<f64>> = Vec::new();
    for k in 0..n_obs {
        let img = image_indexes[k] as usize;
        if !posed[img] {
            continue;
        }
        let c = cluster_indexes[k];
        if seg_cluster.last() != Some(&c) {
            seg_cluster.push(c);
            seg_offsets.push(obs_image.len());
        }
        let uv = positions_xy[k];
        let d = camera.pixel_to_ray(uv[0], uv[1]);
        dirs.push(quats[img].inverse() * Vector3::new(d[0], d[1], d[2]));
        obs_center.push(centers[img]);
        obs_image.push(img as u32);
        obs_uv.push(uv[0]);
        obs_uv.push(uv[1]);
    }
    seg_offsets.push(obs_image.len());
    let n_seg = seg_cluster.len();

    let tris = triangulate_batch(&dirs, &obs_center, &seg_offsets);

    let mut points_flat = vec![0.0f64; n_seg * 3];
    for (s, tri) in tris.iter().enumerate() {
        points_flat[s * 3] = tri.point.x;
        points_flat[s * 3 + 1] = tri.point.y;
        points_flat[s * 3 + 2] = tri.point.z;
    }
    let mut obs_point = vec![0u32; obs_image.len()];
    for s in 0..n_seg {
        for slot in obs_point[seg_offsets[s]..seg_offsets[s + 1]].iter_mut() {
            *slot = s as u32;
        }
    }
    let residuals = reprojection_residuals(
        camera,
        &quats_flat,
        &trans_flat,
        &points_flat,
        &obs_uv,
        &obs_image,
        &obs_point,
        INVALID_RESIDUAL_PX,
    );
    let residual_norms: Vec<f64> = residuals
        .chunks_exact(2)
        .map(|r| (r[0] * r[0] + r[1] * r[1]).sqrt())
        .collect();

    // Per-cluster median residual, measurability, spanned groups, parallax.
    let mut med = vec![f64::NAN; n_seg];
    let mut measurable = vec![false; n_seg];
    let mut groups_of_seg: Vec<Vec<u32>> = Vec::with_capacity(n_seg);
    let mut parallax = vec![f64::NAN; n_seg];
    let mut buf: Vec<f64> = Vec::new();
    let mut units: Vec<Vector3<f64>> = Vec::new();
    for s in 0..n_seg {
        let (lo, hi) = (seg_offsets[s], seg_offsets[s + 1]);
        buf.clear();
        buf.extend_from_slice(&residual_norms[lo..hi]);
        med[s] = median_in_place(&mut buf);
        let point = tris[s].point;
        let finite_point = point.x.is_finite() && point.y.is_finite() && point.z.is_finite();
        measurable[s] = hi - lo >= 2 && finite_point && med[s].is_finite();

        let mut gs: Vec<u32> = obs_image[lo..hi]
            .iter()
            .map(|&i| group_of[i as usize] as u32)
            .collect();
        gs.sort_unstable();
        gs.dedup();
        let is_bridge = measurable[s] && gs.len() >= 2;
        groups_of_seg.push(gs);
        if !is_bridge {
            continue;
        }
        // Parallax: the widest angle between the observation rays to the
        // triangulated point.
        units.clear();
        for center in &obs_center[lo..hi] {
            let v = point - center;
            let norm = v.norm();
            if norm > 0.0 {
                units.push(v / norm);
            }
        }
        if units.len() < 2 {
            continue;
        }
        let mut min_cos = f64::INFINITY;
        for i in 0..units.len() {
            for j in (i + 1)..units.len() {
                min_cos = min_cos.min(units[i].dot(&units[j]));
            }
        }
        parallax[s] = min_cos.clamp(-1.0, 1.0).acos().to_degrees();
    }

    // ── 3. Evidence eligibility (data-derived) ───────────────
    let warp: Vec<f64> = seg_cluster
        .iter()
        .map(|&c| cluster_warp_consistency[c as usize])
        .collect();
    let mut satisfied_warp: Vec<f64> = (0..n_seg)
        .filter(|&s| measurable[s] && med[s] < params.sat_px && warp[s].is_finite())
        .map(|s| warp[s])
        .collect();
    let q_eligible = percentile_linear(&mut satisfied_warp, params.warp_percentile);
    let eligible: Vec<bool> = warp
        .iter()
        .map(|&q| q.is_finite() && q <= q_eligible)
        .collect();

    // ── 4. Per-pair census ───────────────────────────────────────────────
    let mut pair_counts: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
    for s in 0..n_seg {
        let gs = &groups_of_seg[s];
        if !measurable[s] || gs.len() < 2 {
            continue;
        }
        let counts_here =
            eligible[s] && parallax[s].is_finite() && parallax[s] >= params.hi_parallax_deg;
        let unsatisfied = counts_here && med[s] >= params.sat_px;
        for a in 0..gs.len() {
            for b in (a + 1)..gs.len() {
                let entry = pair_counts.entry((gs[a], gs[b])).or_insert((0, 0));
                if counts_here {
                    entry.0 += 1;
                    if unsatisfied {
                        entry.1 += 1;
                    }
                }
            }
        }
    }
    let mut score = 0.0f64;
    let pairs: Vec<PairStats> = pair_counts
        .into_iter()
        .map(|((group_a, group_b), (n_eligible_hi, n_unsatisfied_hi))| {
            let wilson_lb = wilson_lower_bound(n_unsatisfied_hi, n_eligible_hi, params.wilson_z);
            score = score.max(wilson_lb);
            PairStats {
                group_a,
                group_b,
                n_eligible_hi,
                n_unsatisfied_hi,
                wilson_lb,
            }
        })
        .collect();

    // ── 5. Companion: global satisfaction ────────────────────────────────
    let mut n_eval = 0usize;
    let mut n_sat = 0usize;
    for s in 0..n_seg {
        if measurable[s] && eligible[s] {
            n_eval += 1;
            if med[s] < params.sat_px {
                n_sat += 1;
            }
        }
    }
    let sat_pct = if n_eval > 0 {
        100.0 * n_sat as f64 / n_eval as f64
    } else {
        0.0
    };

    Ok(CensusReport {
        score,
        n_groups,
        group_of,
        pairs,
        sat_pct,
        group_consistency: None,
    })
}

#[cfg(test)]
mod tests;
