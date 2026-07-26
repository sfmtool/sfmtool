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
        self.census_with(misplace, &self.warp)
    }

    fn census_with(&self, misplace: f64, warp: &[f64]) -> CensusReport {
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
            &CensusParams::default(),
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

/// `n_a` + `n_b` cameras on two groups, `n_intra` clusters per group, `n_bridge`
/// clusters spanning both. Every cluster gets a low (genuine)
/// warp-consistency residual; tests that need phantoms override the array.
fn two_group_scene(n_a: usize, n_b: usize, n_intra: usize, n_bridge: usize) -> TwoGroupScene {
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
        if let Some(id) = push_cluster(
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

#[test]
fn three_groups_fan_bridges_into_pairs_and_score_is_the_max() {
    // Three camera bands around the same cloud. Cross-group evidence comes in
    // two flavours: pairwise bridges (three cameras of each of two groups) and
    // tri-spanning bridges (two cameras of every group) — the latter must fan
    // into all three pair entries.
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

    let candidate = |shift_c: f64| {
        let shift = Vector3::new(0.0, shift_c, 0.0);
        let mut cq = Vec::new();
        let mut ct = Vec::new();
        for i in 0..3 * n_per {
            let q = quats[i];
            let c = if i >= 2 * n_per {
                centers[i] + shift
            } else {
                centers[i]
            };
            let qi = q.into_inner();
            cq.push([qi.w, qi.i, qi.j, qi.k]);
            let t = -(q * c);
            ct.push([t.x, t.y, t.z]);
        }
        let posed: Vec<u32> = (0..(3 * n_per) as u32).collect();
        cluster_census(
            &cluster,
            &image,
            &pos,
            &warp,
            &test_cam(),
            &cq,
            &ct,
            &posed,
            &CensusParams::default(),
        )
        .unwrap()
    };

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
