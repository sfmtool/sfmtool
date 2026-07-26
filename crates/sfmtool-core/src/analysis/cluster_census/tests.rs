// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::CameraModel;
use nalgebra::{Matrix3, Rotation3};

const W: u32 = 800;
const H: u32 = 800;
const F0: f64 = 700.0;

/// Deterministic LCG so fixtures need no `rand` and are bitwise-stable.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn test_cam() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: F0,
            principal_point_x: W as f64 / 2.0,
            principal_point_y: H as f64 / 2.0,
        },
        width: W,
        height: H,
    }
}

/// World-to-camera rotation of a camera at `c` looking at `target`
/// (canonical frame: the camera looks along −Z, +Y up).
fn look_at(c: Vector3<f64>, target: Vector3<f64>) -> UnitQuaternion<f64> {
    let z_cam = -(target - c).normalize();
    let x_cam = Vector3::y().cross(&z_cam).normalize();
    let y_cam = z_cam.cross(&x_cam);
    let r = Matrix3::from_rows(&[x_cam.transpose(), y_cam.transpose(), z_cam.transpose()]);
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r))
}

/// A synthetic two-group capture: `n_a` cameras on one group, `n_b` on a second
/// group well away from it, both looking at a cloud around the origin.
///
/// Clusters come in two flavours: per-group clusters seen by every camera of one
/// group (they build the intra-group covisibility the community detection needs)
/// and bridge clusters seen by three cameras of each group (the cross-group
/// evidence the census scores).
struct TwoGroupScene {
    cluster: Vec<u32>,
    image: Vec<u32>,
    pos: Vec<[f64; 2]>,
    warp: Vec<f64>,
    quats: Vec<UnitQuaternion<f64>>,
    centers: Vec<Vector3<f64>>,
    /// Cluster ids of the bridge clusters, in order.
    bridge_ids: Vec<u32>,
    n_a: usize,
}

impl TwoGroupScene {
    fn n_img(&self) -> usize {
        self.quats.len()
    }

    /// Candidate pose arrays for every camera, with the whole of group B shifted
    /// `misplace` world units along +Y. The move is rigid, so group B still
    /// explains its own clusters perfectly and only the bridges disagree — and
    /// it is vertical, which both groups see *laterally*, so the disagreement
    /// cannot be absorbed into either group's depth direction the way an
    /// in-plane move can.
    fn candidate(&self, misplace: f64) -> (Vec<[f64; 4]>, Vec<[f64; 3]>, Vec<u32>) {
        let shift = Vector3::new(0.0, misplace, 0.0);
        let mut quats = Vec::new();
        let mut trans = Vec::new();
        for i in 0..self.n_img() {
            let q = self.quats[i];
            let c = if i < self.n_a {
                self.centers[i]
            } else {
                self.centers[i] + shift
            };
            let qi = q.into_inner();
            quats.push([qi.w, qi.i, qi.j, qi.k]);
            let t = -(q * c);
            trans.push([t.x, t.y, t.z]);
        }
        (quats, trans, (0..self.n_img() as u32).collect())
    }

    fn census(&self, misplace: f64) -> CensusReport {
        self.census_full(misplace, &self.warp, &CensusParams::default())
    }

    fn census_with(&self, misplace: f64, warp: &[f64]) -> CensusReport {
        self.census_full(misplace, warp, &CensusParams::default())
    }

    fn census_params(&self, misplace: f64, params: &CensusParams) -> CensusReport {
        self.census_full(misplace, &self.warp, params)
    }

    fn census_full(&self, misplace: f64, warp: &[f64], params: &CensusParams) -> CensusReport {
        let (quats, trans, posed) = self.candidate(misplace);
        cluster_census(
            &self.cluster,
            &self.image,
            &self.pos,
            warp,
            &test_cam(),
            &quats,
            &trans,
            &posed,
            params,
        )
        .expect("census inputs are well formed")
    }

    /// Census of a candidate with the whole of group B misplaced by the world
    /// similarity `x ↦ s·M·x + t` (cameras `C' = s·M·C + t`, `R' = R·Mᵀ`).
    /// The correction that undoes it is the inverse similarity: `Q = Mᵀ`,
    /// `log_scale = −ln s`, `t' = −(1/s)·Mᵀ·t`.
    fn census_similarity(
        &self,
        m: UnitQuaternion<f64>,
        s: f64,
        t: Vector3<f64>,
        params: &CensusParams,
    ) -> CensusReport {
        let mut quats = Vec::new();
        let mut trans = Vec::new();
        for i in 0..self.n_img() {
            let (q, c) = if i < self.n_a {
                (self.quats[i], self.centers[i])
            } else {
                (self.quats[i] * m.inverse(), s * (m * self.centers[i]) + t)
            };
            let qi = q.into_inner();
            quats.push([qi.w, qi.i, qi.j, qi.k]);
            let tt = -(q * c);
            trans.push([tt.x, tt.y, tt.z]);
        }
        let posed: Vec<u32> = (0..self.n_img() as u32).collect();
        cluster_census(
            &self.cluster,
            &self.image,
            &self.pos,
            &self.warp,
            &test_cam(),
            &quats,
            &trans,
            &posed,
            params,
        )
        .expect("census inputs are well formed")
    }
}

/// Append one cluster observed by `members`, projecting a fresh random world
/// point. Returns the cluster id, or `None` when any member cannot see the
/// point (nothing is appended in that case).
#[allow(clippy::too_many_arguments)]
fn push_cluster(
    members: &[usize],
    quats: &[UnitQuaternion<f64>],
    centers: &[Vector3<f64>],
    rng: &mut Lcg,
    cluster: &mut Vec<u32>,
    image: &mut Vec<u32>,
    pos: &mut Vec<[f64; 2]>,
    warp: &mut Vec<f64>,
) -> Option<u32> {
    let cam = test_cam();
    let x = Vector3::new(
        rng.uniform(-2.5, 2.5),
        rng.uniform(-2.0, 2.0),
        rng.uniform(-2.5, 2.5),
    );
    let mut rows = Vec::new();
    for &i in members {
        let xc = quats[i] * (x - centers[i]);
        match cam.ray_to_pixel([xc.x, xc.y, xc.z]) {
            Some((u, v)) if u >= 0.0 && v >= 0.0 && u < W as f64 && v < H as f64 => {
                rows.push((i as u32, [u, v]));
            }
            _ => return None,
        }
    }
    let id = cluster.last().map_or(0, |&c| c + 1);
    for (i, uv) in rows {
        cluster.push(id);
        image.push(i);
        pos.push(uv);
    }
    warp.push(rng.uniform(0.05, 0.5));
    Some(id)
}

/// Append one cluster observed by `members`, each member projecting a *different*
/// random world point: a false match — one cluster id over several unrelated
/// physical points, jointly unsatisfiable by any placement of the cameras.
#[allow(clippy::too_many_arguments)]
fn push_incoherent_cluster(
    members: &[usize],
    quats: &[UnitQuaternion<f64>],
    centers: &[Vector3<f64>],
    rng: &mut Lcg,
    cluster: &mut Vec<u32>,
    image: &mut Vec<u32>,
    pos: &mut Vec<[f64; 2]>,
    warp: &mut Vec<f64>,
) -> Option<u32> {
    let cam = test_cam();
    let mut rows = Vec::new();
    for &i in members {
        let x = Vector3::new(
            rng.uniform(-2.5, 2.5),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.5, 2.5),
        );
        let xc = quats[i] * (x - centers[i]);
        match cam.ray_to_pixel([xc.x, xc.y, xc.z]) {
            Some((u, v)) if u >= 0.0 && v >= 0.0 && u < W as f64 && v < H as f64 => {
                rows.push((i as u32, [u, v]));
            }
            _ => return None,
        }
    }
    let id = cluster.last().map_or(0, |&c| c + 1);
    for (i, uv) in rows {
        cluster.push(id);
        image.push(i);
        pos.push(uv);
    }
    warp.push(rng.uniform(0.05, 0.5));
    Some(id)
}

/// `n_a` + `n_b` cameras on two groups, `n_intra` clusters per group, `n_bridge`
/// clusters spanning both. Every cluster gets a low (genuine)
/// warp-consistency residual; tests that need phantoms override the array.
fn two_group_scene(n_a: usize, n_b: usize, n_intra: usize, n_bridge: usize) -> TwoGroupScene {
    two_group_scene_inner(n_a, n_b, n_intra, n_bridge, true)
}

/// Same shape as [`two_group_scene`], but the bridges are false matches: each
/// member sees a different physical point, while the warp-consistency residuals
/// stay in the genuine range so the eligibility screen admits them. No rigid
/// correction can satisfy them.
fn junk_bridge_scene(n_a: usize, n_b: usize, n_intra: usize, n_bridge: usize) -> TwoGroupScene {
    two_group_scene_inner(n_a, n_b, n_intra, n_bridge, false)
}

fn two_group_scene_inner(
    n_a: usize,
    n_b: usize,
    n_intra: usize,
    n_bridge: usize,
    coherent_bridges: bool,
) -> TwoGroupScene {
    let mut rng = Lcg(0x5eed_1234);
    let r_orbit = 10.0;

    let mut quats = Vec::new();
    let mut centers = Vec::new();
    // Group A spans azimuth −25°..25°, group B 65°..115° — far enough apart that
    // the bridge parallax is tens of degrees.
    for (n, lo, hi) in [(n_a, -25.0f64, 25.0f64), (n_b, 65.0f64, 115.0f64)] {
        for i in 0..n {
            let frac = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.5
            };
            let az = (lo + (hi - lo) * frac).to_radians();
            let c = Vector3::new(
                r_orbit * az.sin(),
                rng.uniform(-0.3, 0.3),
                r_orbit * az.cos(),
            );
            quats.push(look_at(c, Vector3::zeros()));
            centers.push(c);
        }
    }
    let n_img = n_a + n_b;

    let mut cluster = Vec::new();
    let mut image = Vec::new();
    let mut pos = Vec::new();
    let mut warp = Vec::new();
    let mut bridge_ids = Vec::new();

    let arc_a: Vec<usize> = (0..n_a).collect();
    let arc_b: Vec<usize> = (n_a..n_img).collect();
    for group in [&arc_a, &arc_b] {
        if group.is_empty() {
            continue;
        }
        let mut made = 0;
        let mut tries = 0;
        while made < n_intra {
            tries += 1;
            assert!(tries < 100 * n_intra + 100, "scene generation stalled");
            if push_cluster(
                group,
                &quats,
                &centers,
                &mut rng,
                &mut cluster,
                &mut image,
                &mut pos,
                &mut warp,
            )
            .is_some()
            {
                made += 1;
            }
        }
    }

    // Bridge clusters: three cameras of each group, rotated through the groups so
    // the cross-group covisibility spreads evenly.
    let mut made = 0;
    let mut turn = 0usize;
    let mut tries = 0;
    while made < n_bridge {
        tries += 1;
        assert!(tries < 100 * n_bridge + 100, "scene generation stalled");
        let members: Vec<usize> = (0..3)
            .map(|k| arc_a[(turn + k) % n_a])
            .chain((0..3).map(|k| arc_b[(turn + k) % n_b]))
            .collect();
        let push = if coherent_bridges {
            push_cluster
        } else {
            push_incoherent_cluster
        };
        if let Some(id) = push(
            &members,
            &quats,
            &centers,
            &mut rng,
            &mut cluster,
            &mut image,
            &mut pos,
            &mut warp,
        ) {
            bridge_ids.push(id);
            made += 1;
            turn += 1;
        }
    }

    TwoGroupScene {
        cluster,
        image,
        pos,
        warp,
        quats,
        centers,
        bridge_ids,
        n_a,
    }
}

// ── Wilson bound ─────────────────────────────────────────────────────────

#[test]
fn wilson_bound_shrinks_small_denominators() {
    assert_eq!(wilson_lower_bound(0, 0, 1.96), 0.0);
    // The same observed fraction, three sample sizes: the bound climbs toward
    // the point estimate as the evidence accumulates.
    let small = wilson_lower_bound(4, 4, 1.96);
    let mid = wilson_lower_bound(10, 10, 1.96);
    let large = wilson_lower_bound(40, 40, 1.96);
    assert!((small - 0.510_1).abs() < 1e-3, "small = {small}");
    assert!(small < mid && mid < large, "{small} {mid} {large}");
    assert!((large - 0.912_4).abs() < 1e-3, "large = {large}");
    // Three of four unsatisfied is suspicion, not certainty.
    let three_of_four = wilson_lower_bound(3, 4, 1.96);
    assert!(
        (three_of_four - 0.300_7).abs() < 1e-3,
        "three_of_four = {three_of_four}"
    );
    // A zero numerator floors at zero, never negative.
    assert_eq!(wilson_lower_bound(0, 25, 1.96), 0.0);
}

// ── Percentile ───────────────────────────────────────────────────────────

#[test]
fn percentile_matches_numpy_linear() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0];
    assert!((percentile_linear(&mut v, 50.0) - 2.5).abs() < 1e-12);
    let mut v = vec![1.0, 2.0, 3.0, 4.0];
    assert!((percentile_linear(&mut v, 95.0) - 3.85).abs() < 1e-12);
    let mut v = vec![7.0];
    assert!((percentile_linear(&mut v, 95.0) - 7.0).abs() < 1e-12);
    let mut v: Vec<f64> = Vec::new();
    assert!(percentile_linear(&mut v, 95.0).is_infinite());
}

// ── Grouping ─────────────────────────────────────────────────────────────

#[test]
fn modularity_splits_two_blocks_and_keeps_one_clique_whole() {
    // Two 3-cliques joined by a single weak edge.
    let n = 6;
    let mut w = vec![0.0; n * n];
    for (i, j) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)] {
        w[i * n + j] = 100.0;
        w[j * n + i] = 100.0;
    }
    w[2 * n + 3] = 1.0;
    w[3 * n + 2] = 1.0;
    let (labels, n_groups) = modularity_groups(&w, n);
    assert_eq!(n_groups, 2);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);

    // One uniform clique has no split worth making.
    let mut w = vec![10.0; n * n];
    for i in 0..n {
        w[i * n + i] = 0.0;
    }
    assert_eq!(modularity_groups(&w, n).1, 1);

    // Degenerate inputs collapse to one group.
    assert_eq!(modularity_groups(&[0.0; 4], 2).1, 1);
    assert_eq!(modularity_groups(&[0.0; 25], 5).1, 1);
    assert_eq!(modularity_groups(&[], 0), (Vec::new(), 0));
}

#[test]
fn modularity_merge_path_is_pinned_and_reproducible() {
    // A 4-node path 0—1—2—3 with equal weights. The first merge is a tie
    // between (0, 1) and (2, 3); the documented rule takes the last maximal
    // pair, and the best-Q partition on the path is the two end pairs.
    let n = 4;
    let mut w = vec![0.0; n * n];
    for (i, j) in [(0, 1), (1, 2), (2, 3)] {
        w[i * n + j] = 5.0;
        w[j * n + i] = 5.0;
    }
    let (labels, n_groups) = modularity_groups(&w, n);
    assert_eq!(n_groups, 2);
    assert_eq!(labels, vec![0, 0, 1, 1]);
    assert_eq!(modularity_groups(&w, n), (labels, n_groups));
}

// ── Census on synthetic captures ─────────────────────────────────────────

#[test]
fn two_clean_arcs_score_zero() {
    let scene = two_group_scene(6, 6, 150, 60);
    let report = scene.census(0.0);
    assert_eq!(report.n_groups, 2, "group_of = {:?}", report.group_of);
    // The two groups land in different communities.
    let a = report.group_of[0];
    let b = report.group_of[scene.n_a];
    assert_ne!(a, b);
    for i in 0..scene.n_a {
        assert_eq!(report.group_of[i], a);
    }
    for i in scene.n_a..scene.n_img() {
        assert_eq!(report.group_of[i], b);
    }
    // One group pair, plenty of eligible high-parallax evidence, all satisfied.
    assert_eq!(report.pairs.len(), 1);
    assert!(
        report.pairs[0].n_eligible_hi > 40,
        "pairs = {:?}",
        report.pairs
    );
    assert_eq!(report.pairs[0].n_unsatisfied_hi, 0);
    // A zero numerator over a large denominator can leave float dust in the
    // Wilson bound, not an exact zero.
    assert!(report.score < 1e-12, "score = {}", report.score);
    assert!(report.sat_pct > 99.0, "sat_pct = {}", report.sat_pct);
    assert!(report.group_consistency.is_none());
}

#[test]
fn a_misplaced_arc_scores_high() {
    let scene = two_group_scene(6, 6, 150, 60);
    let report = scene.census(0.5);
    assert_eq!(report.n_groups, 2);
    assert_eq!(report.pairs.len(), 1);
    let pair = report.pairs[0];
    assert_eq!(
        pair.n_unsatisfied_hi, pair.n_eligible_hi,
        "every bridge should fail: {pair:?}"
    );
    assert!(report.score > 0.9, "score = {}", report.score);
    // The failure is local to the seam: each group still explains its own
    // clusters, so global satisfaction stays high.
    assert!(report.sat_pct > 80.0, "sat_pct = {}", report.sat_pct);
}

#[test]
fn score_falls_as_the_candidate_approaches_the_truth() {
    let scene = two_group_scene(6, 6, 150, 60);
    let far = scene.census(0.5).score;
    let near = scene.census(0.06).score;
    let exact = scene.census(0.0).score;
    assert!(far >= near, "{far} !>= {near}");
    assert!(near >= exact, "{near} !>= {exact}");
    assert!(exact < 1e-12, "exact = {exact}");
}

#[test]
fn fewer_than_two_groups_is_unverifiable() {
    // A single group: one community, so there is nothing to census.
    let scene = two_group_scene(6, 0, 150, 0);
    let report = scene.census(0.0);
    assert_eq!(report.n_groups, 1);
    assert!(report.pairs.is_empty());
    assert_eq!(report.score, 0.0);
    // sat_pct is still meaningful — it needs no grouping.
    assert!(report.sat_pct > 99.0, "sat_pct = {}", report.sat_pct);
}

#[test]
fn no_posed_images_reports_no_groups() {
    let scene = two_group_scene(4, 4, 20, 5);
    let report = cluster_census(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &scene.warp,
        &test_cam(),
        &[],
        &[],
        &[],
        &CensusParams::default(),
    )
    .unwrap();
    assert_eq!(report.n_groups, 0);
    assert_eq!(report.score, 0.0);
    assert!(report.group_of.iter().all(|&g| g == -1));
    assert_eq!(report.sat_pct, 0.0);
}

#[test]
fn poor_warp_consistency_bridges_are_not_evidence() {
    let scene = two_group_scene(6, 6, 150, 60);
    // The seam is genuinely misplaced, but this time the bridges carry a
    // warp-consistency residual far outside the satisfied clusters' P95 — they
    // are phantom correspondences, not evidence, and must be dropped.
    let mut warp = scene.warp.clone();
    for &id in &scene.bridge_ids {
        warp[id as usize] = 8.0;
    }
    let flagged = scene.census(0.5);
    let suppressed = scene.census_with(0.5, &warp);
    assert!(flagged.score > 0.9, "flagged = {}", flagged.score);
    assert_eq!(suppressed.pairs.len(), 1);
    assert_eq!(suppressed.pairs[0].n_eligible_hi, 0);
    assert_eq!(suppressed.score, 0.0);
}

#[test]
fn a_thin_seam_is_shrunk_toward_zero() {
    // Same failure, same unsatisfied fraction (all of them) — but four bridges
    // of evidence instead of sixty.
    let thin = two_group_scene(6, 6, 150, 4).census(0.5);
    let thick = two_group_scene(6, 6, 150, 60).census(0.5);
    assert_eq!(thin.pairs[0].n_unsatisfied_hi, thin.pairs[0].n_eligible_hi);
    assert_eq!(
        thick.pairs[0].n_unsatisfied_hi,
        thick.pairs[0].n_eligible_hi
    );
    assert!(
        thin.score < thick.score,
        "{} !< {}",
        thin.score,
        thick.score
    );
    assert!(thin.score < 0.7, "thin = {}", thin.score);
    assert!(thick.score > 0.9, "thick = {}", thick.score);
}

/// Three camera bands around the same cloud. Cross-group evidence comes in two
/// flavours: pairwise bridges (three cameras of each of two groups) and
/// tri-spanning bridges (two cameras of every group), which fan into all three
/// pair entries. The candidate shifts the third band.
struct ThreeGroupScene {
    cluster: Vec<u32>,
    image: Vec<u32>,
    pos: Vec<[f64; 2]>,
    warp: Vec<f64>,
    quats: Vec<UnitQuaternion<f64>>,
    centers: Vec<Vector3<f64>>,
    n_per: usize,
}

impl ThreeGroupScene {
    fn census(&self, shift_c: f64) -> CensusReport {
        self.census_params(shift_c, &CensusParams::default())
    }

    fn census_params(&self, shift_c: f64, params: &CensusParams) -> CensusReport {
        let n_per = self.n_per;
        let shift = Vector3::new(0.0, shift_c, 0.0);
        let mut cq = Vec::new();
        let mut ct = Vec::new();
        for i in 0..3 * n_per {
            let q = self.quats[i];
            let c = if i >= 2 * n_per {
                self.centers[i] + shift
            } else {
                self.centers[i]
            };
            let qi = q.into_inner();
            cq.push([qi.w, qi.i, qi.j, qi.k]);
            let t = -(q * c);
            ct.push([t.x, t.y, t.z]);
        }
        let posed: Vec<u32> = (0..(3 * n_per) as u32).collect();
        cluster_census(
            &self.cluster,
            &self.image,
            &self.pos,
            &self.warp,
            &test_cam(),
            &cq,
            &ct,
            &posed,
            params,
        )
        .unwrap()
    }
}

fn three_group_scene() -> ThreeGroupScene {
    let mut rng = Lcg(0x5eed_3333);
    let r_orbit = 10.0;
    let n_per = 6usize;
    let mut quats = Vec::new();
    let mut centers = Vec::new();
    for (lo, hi) in [(-25.0f64, 25.0), (65.0, 115.0), (155.0, 205.0)] {
        for i in 0..n_per {
            let az = (lo + (hi - lo) * i as f64 / (n_per - 1) as f64).to_radians();
            let c = Vector3::new(
                r_orbit * az.sin(),
                rng.uniform(-0.3, 0.3),
                r_orbit * az.cos(),
            );
            quats.push(look_at(c, Vector3::zeros()));
            centers.push(c);
        }
    }
    let mut cluster = Vec::new();
    let mut image = Vec::new();
    let mut pos = Vec::new();
    let mut warp = Vec::new();
    let groups: Vec<Vec<usize>> = (0..3)
        .map(|g| (g * n_per..(g + 1) * n_per).collect())
        .collect();
    let mut fill = |members_of: &dyn Fn(usize) -> Vec<usize>,
                    n: usize,
                    cluster: &mut Vec<u32>,
                    image: &mut Vec<u32>,
                    pos: &mut Vec<[f64; 2]>,
                    warp: &mut Vec<f64>| {
        let mut made = 0;
        let mut tries = 0;
        while made < n {
            tries += 1;
            assert!(tries < 100 * n + 100, "scene generation stalled");
            if push_cluster(
                &members_of(made),
                &quats,
                &centers,
                &mut rng,
                cluster,
                image,
                pos,
                warp,
            )
            .is_some()
            {
                made += 1;
            }
        }
    };
    for g in &groups {
        let g = g.clone();
        fill(
            &move |_| g.clone(),
            150,
            &mut cluster,
            &mut image,
            &mut pos,
            &mut warp,
        );
    }
    for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
        let (ga, gb) = (groups[a].clone(), groups[b].clone());
        fill(
            &move |turn| {
                (0..3)
                    .map(|k| ga[(turn + k) % n_per])
                    .chain((0..3).map(|k| gb[(turn + k) % n_per]))
                    .collect()
            },
            60,
            &mut cluster,
            &mut image,
            &mut pos,
            &mut warp,
        );
    }
    let tri = groups.clone();
    fill(
        &move |turn| {
            tri.iter()
                .flat_map(|g| (0..2).map(move |k| g[(turn + k) % n_per]))
                .collect()
        },
        20,
        &mut cluster,
        &mut image,
        &mut pos,
        &mut warp,
    );

    ThreeGroupScene {
        cluster,
        image,
        pos,
        warp,
        quats,
        centers,
        n_per,
    }
}

#[test]
fn three_groups_fan_bridges_into_pairs_and_score_is_the_max() {
    let scene = three_group_scene();
    let n_per = scene.n_per;
    let candidate = |shift_c: f64| scene.census(shift_c);

    // At the truth every pair entry exists, in ascending (group_a, group_b)
    // order, with more eligible evidence than the 60 pairwise bridges alone
    // could supply — the tri-spanning bridges fanned into every pair.
    let clean = candidate(0.0);
    assert_eq!(clean.n_groups, 3, "group_of = {:?}", clean.group_of);
    assert_eq!(clean.pairs.len(), 3);
    for w in clean.pairs.windows(2) {
        assert!(
            (w[0].group_a, w[0].group_b) < (w[1].group_a, w[1].group_b),
            "pairs = {:?}",
            clean.pairs
        );
    }
    for p in &clean.pairs {
        assert!(p.group_a < p.group_b, "pairs = {:?}", clean.pairs);
        assert!(p.n_eligible_hi > 65, "pairs = {:?}", clean.pairs);
        assert_eq!(p.n_unsatisfied_hi, 0, "pairs = {:?}", clean.pairs);
    }
    assert!(clean.score < 1e-12, "score = {}", clean.score);

    // Shift the third band: both of its seams fail while the untouched pair
    // stays comparatively clean. The score is the max over pairs — a pooled
    // fraction would dilute the two failing seams with the clean pair's
    // evidence.
    let shifted = candidate(0.5);
    assert_eq!(shifted.n_groups, 3);
    assert_eq!(shifted.pairs.len(), 3);
    let gc = shifted.group_of[2 * n_per];
    let mut max_lb = 0.0f64;
    for p in &shifted.pairs {
        max_lb = max_lb.max(p.wilson_lb);
        if p.group_a as i32 == gc || p.group_b as i32 == gc {
            assert!(p.wilson_lb > 0.9, "seam pair = {p:?}");
        } else {
            assert!(p.wilson_lb < 0.5, "clean pair = {p:?}");
        }
    }
    assert_eq!(shifted.score, max_lb);
}

#[test]
fn census_is_deterministic() {
    let scene = two_group_scene(6, 6, 150, 60);
    let a = scene.census(0.5);
    let b = scene.census(0.5);
    assert_eq!(a.score, b.score);
    assert_eq!(a.n_groups, b.n_groups);
    assert_eq!(a.group_of, b.group_of);
    assert_eq!(a.pairs, b.pairs);
    assert_eq!(a.sat_pct, b.sat_pct);
}

// ── Group consistency (the opt-in companion) ─────────────────────────────

fn with_consistency() -> CensusParams {
    CensusParams {
        compute_group_consistency: true,
        ..CensusParams::default()
    }
}

/// Rotation angle (degrees) of a WXYZ quaternion.
fn angle_deg(q: [f64; 4]) -> f64 {
    2.0 * q[0].abs().min(1.0).acos().to_degrees()
}

fn correction_of(gc: &GroupConsistency, group: u32) -> GroupCorrection {
    *gc.corrections
        .iter()
        .find(|c| c.group == group)
        .expect("every group carries a correction")
}

#[test]
fn the_companion_is_opt_in_and_leaves_the_census_untouched() {
    let scene = two_group_scene(8, 6, 150, 60);
    let off = scene.census(0.5);
    let on = scene.census_params(0.5, &with_consistency());
    assert!(off.group_consistency.is_none());
    assert!(on.group_consistency.is_some());
    // Every phase-1 field is bit-identical; only the companion appears.
    assert_eq!(off.score, on.score);
    assert_eq!(off.n_groups, on.n_groups);
    assert_eq!(off.group_of, on.group_of);
    assert_eq!(off.pairs, on.pairs);
    assert_eq!(off.sat_pct, on.sat_pct);
}

#[test]
fn a_rigidly_misplaced_group_is_coherent() {
    // Group B is larger-than-life wrong but *rigidly* so: the correction that
    // undoes the +0.5 shift of its cameras is exactly t = (0, −0.5, 0), and the
    // solve must recover it from the bridges alone. Group A is the larger group,
    // so it holds the gauge and group B carries the correction.
    let scene = two_group_scene(8, 6, 150, 60);
    let report = scene.census_params(0.5, &with_consistency());
    assert_eq!(report.n_groups, 2);
    let gc = report.group_consistency.expect("two groups with bridges");
    assert_eq!(gc.corrections.len(), 2);

    let gauge = report.group_of[0] as u32;
    let moved = report.group_of[scene.n_a] as u32;
    assert_ne!(gauge, moved);
    let identity = correction_of(&gc, gauge);
    assert_eq!(identity.rotation_wxyz, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(identity.translation, [0.0, 0.0, 0.0]);
    assert_eq!(identity.log_scale, 0.0);

    let fix = correction_of(&gc, moved);
    assert!(angle_deg(fix.rotation_wxyz) < 1.0, "fix = {fix:?}");
    assert!(fix.log_scale.abs() < 0.02, "fix = {fix:?}");
    for (got, want) in fix.translation.iter().zip([0.0, -0.5, 0.0]) {
        assert!((got - want).abs() < 0.05, "fix = {fix:?}");
    }

    // The disagreement is coherent, and the repair is a net gain rather than a
    // trade of one seam for another.
    assert!(gc.explained_pct > 90.0, "explained = {}", gc.explained_pct);
    assert!(
        gc.n_unsatisfied_before > 40,
        "n_unsat = {}",
        gc.n_unsatisfied_before
    );
    assert_eq!(
        gc.explained_pct,
        100.0 * gc.n_explained as f64 / gc.n_unsatisfied_before as f64
    );
    assert!(
        gc.net_after > gc.net_before,
        "net {} -> {}",
        gc.net_before,
        gc.net_after
    );
}

#[test]
fn a_rotationally_misplaced_group_recovers_the_inverse_rotation() {
    // Group B rotated +4° about world Y (M = rot_y(4°), s = 1, t = 0). The
    // correction that undoes it is Q = Mᵀ exactly — a transposed-Q convention
    // passes every pure-translation fixture and fails only here.
    let scene = two_group_scene(8, 6, 150, 60);
    let m = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 4.0f64.to_radians());
    let report = scene.census_similarity(m, 1.0, Vector3::zeros(), &with_consistency());
    assert_eq!(report.n_groups, 2);
    let gc = report.group_consistency.expect("two groups with bridges");
    let moved = report.group_of[scene.n_a] as u32;
    let fix = correction_of(&gc, moved);
    // Q ≈ Mᵀ ⇔ Q·M ≈ I. The transposed reading leaves an 8° residual.
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        fix.rotation_wxyz[0],
        fix.rotation_wxyz[1],
        fix.rotation_wxyz[2],
        fix.rotation_wxyz[3],
    ));
    let residual = (q * m).angle().to_degrees();
    assert!(residual < 0.5, "residual = {residual}° fix = {fix:?}");
    assert!(fix.log_scale.abs() < 0.01, "fix = {fix:?}");
    assert!(gc.explained_pct > 90.0, "explained = {}", gc.explained_pct);
    assert!(gc.net_after > gc.net_before);
}

#[test]
fn a_scale_misplaced_group_recovers_the_inverse_log_scale() {
    // Group B scaled ×1.06 about the world origin: the correction is the
    // inverse similarity with log_scale = −ln 1.06 — a negated log-scale
    // convention fails only here.
    let scene = two_group_scene(8, 6, 150, 60);
    let report = scene.census_similarity(
        UnitQuaternion::identity(),
        1.06,
        Vector3::zeros(),
        &with_consistency(),
    );
    assert_eq!(report.n_groups, 2);
    let gc = report.group_consistency.expect("two groups with bridges");
    let moved = report.group_of[scene.n_a] as u32;
    let fix = correction_of(&gc, moved);
    assert!(
        (fix.log_scale + 1.06f64.ln()).abs() < 0.01,
        "log_scale = {} want {}",
        fix.log_scale,
        -1.06f64.ln()
    );
    assert!(angle_deg(fix.rotation_wxyz) < 0.5, "fix = {fix:?}");
    assert!(gc.explained_pct > 90.0, "explained = {}", gc.explained_pct);
    assert!(gc.net_after > gc.net_before);
}

#[test]
fn junk_bridges_that_slipped_the_screen_are_incoherent() {
    // The candidate is at the truth; what it cannot satisfy is a population of
    // false matches whose warp consistency looks genuine. No similarity of
    // either group can place unrelated physical points on top of each other, so
    // the disagreement is incoherent even though the census flags it.
    let scene = junk_bridge_scene(8, 6, 150, 60);
    let report = scene.census_params(0.0, &with_consistency());
    assert_eq!(report.n_groups, 2);
    assert!(report.score > 0.9, "score = {}", report.score);
    let gc = report.group_consistency.expect("two groups with bridges");
    assert!(gc.explained_pct < 10.0, "explained = {}", gc.explained_pct);
}

#[test]
fn a_truthful_candidate_needs_no_correction() {
    let scene = two_group_scene(8, 6, 150, 60);
    let report = scene.census_params(0.0, &with_consistency());
    let gc = report.group_consistency.expect("two groups with bridges");
    for c in &gc.corrections {
        assert!(angle_deg(c.rotation_wxyz) < 0.05, "correction = {c:?}");
        assert!(c.log_scale.abs() < 1e-3, "correction = {c:?}");
        let shift =
            (c.translation[0].powi(2) + c.translation[1].powi(2) + c.translation[2].powi(2)).sqrt();
        assert!(shift < 0.02, "correction = {c:?}");
    }
    // Nothing was unsatisfied, so there is nothing to explain, and the solve
    // cannot have broken what was already satisfied.
    assert_eq!(gc.explained_pct, 0.0);
    assert_eq!(gc.n_unsatisfied_before, 0);
    assert_eq!(gc.n_explained, 0);
    assert_eq!(gc.net_after, gc.net_before);
}

#[test]
fn the_joint_solve_separates_one_band_of_three() {
    // Three groups, one shifted: the joint solve must place the shifted band
    // relative to the other two, whichever of them holds the gauge. Comparing
    // corrections *between* groups makes the assertion gauge-independent.
    let scene = three_group_scene();
    let report = scene.census_params(0.5, &with_consistency());
    assert_eq!(report.n_groups, 3);
    let gc = report.group_consistency.expect("three groups with bridges");
    assert_eq!(gc.corrections.len(), 3);

    let moved = report.group_of[2 * scene.n_per] as u32;
    let fix = correction_of(&gc, moved);
    for g in 0..3u32 {
        if g == moved {
            continue;
        }
        let still = correction_of(&gc, g);
        assert!(angle_deg(still.rotation_wxyz) < 1.0, "still = {still:?}");
        // The shifted band has to fall 0.5 in +Y relative to every band that
        // stayed put; the two that stayed must agree with each other.
        let dy = fix.translation[1] - still.translation[1];
        assert!((dy + 0.5).abs() < 0.06, "dy = {dy}");
        assert!(
            (fix.translation[0] - still.translation[0]).abs() < 0.06,
            "fix = {fix:?} still = {still:?}"
        );
        assert!(
            (fix.translation[2] - still.translation[2]).abs() < 0.06,
            "fix = {fix:?} still = {still:?}"
        );
    }
    assert!(gc.explained_pct > 80.0, "explained = {}", gc.explained_pct);
    assert!(
        gc.net_after > gc.net_before,
        "net {} -> {}",
        gc.net_before,
        gc.net_after
    );
}

#[test]
fn group_consistency_is_deterministic() {
    let scene = two_group_scene(8, 6, 150, 60);
    let a = scene.census_params(0.5, &with_consistency());
    let b = scene.census_params(0.5, &with_consistency());
    assert_eq!(a.group_consistency, b.group_consistency);
}

#[test]
fn group_consistency_declines_without_group_structure_or_evidence() {
    // One group: nothing to be consistent about.
    let single = two_group_scene(6, 0, 150, 0).census_params(0.0, &with_consistency());
    assert_eq!(single.n_groups, 1);
    assert!(single.group_consistency.is_none());

    // Two groups, but every bridge falls outside the eligibility screen: no
    // evidence to solve on.
    let scene = two_group_scene(8, 6, 150, 60);
    let mut warp = scene.warp.clone();
    for &id in &scene.bridge_ids {
        warp[id as usize] = 8.0;
    }
    let starved = scene.census_full(0.5, &warp, &with_consistency());
    assert_eq!(starved.n_groups, 2);
    assert!(starved.group_consistency.is_none());
}

/// `n_groups` camera bands 90° apart in azimuth, `n_per` cameras each,
/// `n_intra` clusters per band, and `n_bridge` bridge clusters over drawn band
/// pairs (three cameras from each side). Every warp-consistency residual is the
/// same low value, so the eligibility screen admits every bridge and the
/// eligible-bridge population is exactly `n_bridge` — the knob the § 6 stride
/// and timing tests scale.
struct BandScene {
    cluster: Vec<u32>,
    image: Vec<u32>,
    pos: Vec<[f64; 2]>,
    warp: Vec<f64>,
    quats: Vec<UnitQuaternion<f64>>,
    centers: Vec<Vector3<f64>>,
    n_per: usize,
    n_groups: usize,
}

impl BandScene {
    /// Census of a candidate with the last band shifted `misplace` world units
    /// along +Y.
    fn census(&self, misplace: f64, params: &CensusParams) -> CensusReport {
        let n_img = self.n_per * self.n_groups;
        let shift = Vector3::new(0.0, misplace, 0.0);
        let mut cq = Vec::new();
        let mut ct = Vec::new();
        for i in 0..n_img {
            let q = self.quats[i];
            let c = if i >= n_img - self.n_per {
                self.centers[i] + shift
            } else {
                self.centers[i]
            };
            let qi = q.into_inner();
            cq.push([qi.w, qi.i, qi.j, qi.k]);
            let t = -(q * c);
            ct.push([t.x, t.y, t.z]);
        }
        let posed: Vec<u32> = (0..n_img as u32).collect();
        cluster_census(
            &self.cluster,
            &self.image,
            &self.pos,
            &self.warp,
            &test_cam(),
            &cq,
            &ct,
            &posed,
            params,
        )
        .unwrap()
    }
}

fn band_scene(n_groups: usize, n_per: usize, n_intra: usize, n_bridge: usize) -> BandScene {
    let mut rng = Lcg(0x5eed_7777);
    let r_orbit = 10.0;
    let mut quats = Vec::new();
    let mut centers = Vec::new();
    for g in 0..n_groups {
        let mid = 90.0 * g as f64;
        for i in 0..n_per {
            let az = (mid - 25.0 + 50.0 * i as f64 / (n_per - 1) as f64).to_radians();
            let c = Vector3::new(
                r_orbit * az.sin(),
                rng.uniform(-0.3, 0.3),
                r_orbit * az.cos(),
            );
            quats.push(look_at(c, Vector3::zeros()));
            centers.push(c);
        }
    }
    let groups: Vec<Vec<usize>> = (0..n_groups)
        .map(|g| (g * n_per..(g + 1) * n_per).collect())
        .collect();
    let pairs: Vec<(usize, usize)> = (0..n_groups)
        .flat_map(|a| ((a + 1)..n_groups).map(move |b| (a, b)))
        .collect();

    let mut cluster = Vec::new();
    let mut image = Vec::new();
    let mut pos = Vec::new();
    let mut warp = Vec::new();
    let push = |members: &[usize],
                rng: &mut Lcg,
                cluster: &mut Vec<u32>,
                image: &mut Vec<u32>,
                pos: &mut Vec<[f64; 2]>,
                warp: &mut Vec<f64>| {
        push_cluster(members, &quats, &centers, rng, cluster, image, pos, warp).is_some()
    };

    for g in &groups {
        let mut made = 0;
        while made < n_intra {
            if push(g, &mut rng, &mut cluster, &mut image, &mut pos, &mut warp) {
                made += 1;
            }
        }
    }
    let mut made = 0;
    let mut turn = 0usize;
    while made < n_bridge {
        // Draw the band pair rather than cycling it: a fixed cycle aliases
        // against the fit stride and can starve a band of fit evidence.
        let (a, b) = pairs[(rng.next_f64() * pairs.len() as f64) as usize % pairs.len()];
        let members: Vec<usize> = (0..3)
            .map(|k| groups[a][(turn + k) % n_per])
            .chain((0..3).map(|k| groups[b][(turn + k) % n_per]))
            .collect();
        if push(
            &members,
            &mut rng,
            &mut cluster,
            &mut image,
            &mut pos,
            &mut warp,
        ) {
            made += 1;
            turn += 1;
        }
    }

    let warp = vec![0.1; warp.len()];
    BandScene {
        cluster,
        image,
        pos,
        warp,
        quats,
        centers,
        n_per,
        n_groups,
    }
}

#[test]
fn the_fit_stride_caps_the_fit_set_without_capping_the_scoring_set() {
    use group_consistency::{fit_stride, MAX_FIT_BRIDGES};

    // Up to the cap every bridge fits; past it the stride is the smallest step
    // that brings ⌈n / stride⌉ back under the cap.
    for n in [0usize, 1, 7, MAX_FIT_BRIDGES] {
        assert_eq!(fit_stride(n), 1, "n = {n}");
    }
    assert_eq!(fit_stride(MAX_FIT_BRIDGES + 1), 2);
    assert_eq!(fit_stride(2 * MAX_FIT_BRIDGES), 3);
    assert_eq!(fit_stride(10 * MAX_FIT_BRIDGES), 11);
    for n in [1201usize, 2399, 2400, 2401, 12_000, 1_000_000] {
        assert!(
            n.div_ceil(fit_stride(n)) <= MAX_FIT_BRIDGES,
            "n = {n} leaves {} fit bridges",
            n.div_ceil(fit_stride(n))
        );
    }

    // End to end past the cap: the band scene's uniform warp-consistency admits
    // every bridge, so the eligible population is exactly the 1250 generated
    // ones and the fit set strides down to 625. The correction still comes out
    // of the subsample, and the *scoring* still runs on the whole population —
    // which is what a denominator above the cap shows.
    let scene = band_scene(2, 6, 1250, 1250);
    let report = scene.census(0.3, &with_consistency());
    assert_eq!(report.n_groups, 2);
    let gc = report.group_consistency.expect("two groups with bridges");
    assert!(
        gc.n_unsatisfied_before > MAX_FIT_BRIDGES,
        "n_unsat = {} — the scoring set must exceed the fit cap for this to \
         exercise the stride",
        gc.n_unsatisfied_before
    );
    // The shifted band has to fall 0.3 in +Y relative to the one that stayed,
    // whichever of them holds the gauge.
    let moved = correction_of(&gc, report.group_of[scene.n_per] as u32);
    let still = correction_of(&gc, report.group_of[0] as u32);
    let dy = moved.translation[1] - still.translation[1];
    assert!((dy + 0.3).abs() < 0.05, "dy = {dy}");
    assert!(gc.explained_pct > 90.0, "explained = {}", gc.explained_pct);
}

/// Wall-clock cost of the § 6 companion, isolated as the difference between a
/// census with it and one without, over bridge populations and group counts.
/// Ignored — it is a measurement, not an assertion. Run with
/// `cargo test --release -p sfmtool-core -- --ignored --nocapture timing_probe`.
#[test]
#[ignore]
fn timing_probe_group_consistency() {
    for n_groups in [2usize, 4] {
        for n_bridge in [2400usize, 4800] {
            let scene = band_scene(n_groups, 6, n_bridge, n_bridge);
            let t0 = std::time::Instant::now();
            let off = scene.census(0.3, &CensusParams::default());
            let census_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = std::time::Instant::now();
            let on = scene.census(0.3, &with_consistency());
            let total_ms = t1.elapsed().as_secs_f64() * 1e3;
            let gc = on.group_consistency.expect("bridges in every band pair");
            assert_eq!(off.n_groups, n_groups);
            println!(
                "groups={n_groups} bridges={n_bridge} census={census_ms:.1}ms \
                 companion={:.1}ms explained={:.1}% unsat_before={}",
                total_ms - census_ms,
                gc.explained_pct,
                gc.n_unsatisfied_before,
            );
        }
    }
}

// ── Input validation ─────────────────────────────────────────────────────

#[test]
fn input_validation_rejects_malformed_arrays() {
    let cam = test_cam();
    let params = CensusParams::default();
    let q = vec![[1.0, 0.0, 0.0, 0.0]];
    let t = vec![[0.0, 0.0, 0.0]];

    let err = cluster_census(
        &[0, 0],
        &[0],
        &[[0.0, 0.0], [1.0, 1.0]],
        &[0.1],
        &cam,
        &q,
        &t,
        &[0],
        &params,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        CensusError::NotParallel {
            name: "image_indexes",
            ..
        }
    ));

    let err = cluster_census(
        &[1, 0],
        &[0, 0],
        &[[0.0, 0.0], [1.0, 1.0]],
        &[0.1, 0.1],
        &cam,
        &q,
        &t,
        &[0],
        &params,
    )
    .unwrap_err();
    assert_eq!(err, CensusError::ClusterIndexesNotSorted { at: 1 });

    let err = cluster_census(
        &[0, 5],
        &[0, 0],
        &[[0.0, 0.0], [1.0, 1.0]],
        &[0.1, 0.1],
        &cam,
        &q,
        &t,
        &[0],
        &params,
    )
    .unwrap_err();
    assert_eq!(
        err,
        CensusError::ClusterIndexOutOfRange {
            index: 5,
            num_clusters: 2
        }
    );

    let err = cluster_census(
        &[0, 0],
        &[0, 0],
        &[[0.0, 0.0], [1.0, 1.0]],
        &[0.1],
        &cam,
        &q,
        &[],
        &[0],
        &params,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        CensusError::NotParallel {
            name: "translations",
            ..
        }
    ));
}
