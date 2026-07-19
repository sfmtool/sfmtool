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
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn test_cam(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: f,
            principal_point_x: W as f64 / 2.0,
            principal_point_y: H as f64 / 2.0,
        },
        width: W,
        height: H,
    }
}

/// Ground-truth orbit capture: cameras on a circle of radius 10 looking at
/// the origin, world points on a jittered cylinder of radius ~4. A point at
/// cylinder angle `phi` is visible only from cameras within the front-facing
/// cone (`cos(theta - phi) > vis_cos`), so covisibility is local — a long
/// orbit closes a loop rather than being globally rigid from the start.
struct Scene {
    cluster: Vec<u32>,
    image: Vec<u32>,
    pos: Vec<[f64; 2]>,
    quats: Vec<UnitQuaternion<f64>>,
    centers: Vec<Vector3<f64>>,
    /// World point per cluster id.
    world: Vec<Vector3<f64>>,
}

impl Scene {
    fn seed(&self, n: usize) -> (Vec<[f64; 4]>, Vec<[f64; 3]>, Vec<u32>) {
        let q = (0..n)
            .map(|i| {
                let q = self.quats[i].into_inner();
                [q.w, q.i, q.j, q.k]
            })
            .collect();
        let t = (0..n)
            .map(|i| {
                let t = -(self.quats[i] * self.centers[i]);
                [t.x, t.y, t.z]
            })
            .collect();
        (q, t, (0..n as u32).collect())
    }
}

fn orbit_scene(n_img: usize, n_pts: usize, noise: f64, vis_cos: f64, rng: &mut Lcg) -> Scene {
    use std::f64::consts::TAU;
    let cam = test_cam(F0);
    let r_orbit = 10.0;

    let mut quats = Vec::with_capacity(n_img);
    let mut centers = Vec::with_capacity(n_img);
    let mut thetas = Vec::with_capacity(n_img);
    for i in 0..n_img {
        let th = TAU * i as f64 / n_img as f64;
        let c = Vector3::new(
            r_orbit * th.sin(),
            rng.uniform(-0.3, 0.3),
            r_orbit * th.cos(),
        );
        let target = Vector3::new(
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2),
        );
        let z_cam = -(target - c).normalize();
        let x_cam = Vector3::y().cross(&z_cam).normalize();
        let y_cam = z_cam.cross(&x_cam);
        let r = Matrix3::from_rows(&[x_cam.transpose(), y_cam.transpose(), z_cam.transpose()]);
        quats.push(UnitQuaternion::from_rotation_matrix(
            &Rotation3::from_matrix_unchecked(r),
        ));
        centers.push(c);
        thetas.push(th);
    }

    let mut cluster = Vec::new();
    let mut image = Vec::new();
    let mut pos = Vec::new();
    let mut world = Vec::new();
    let mut cid = 0u32;
    for _ in 0..n_pts {
        let phi = rng.uniform(0.0, TAU);
        let r_cyl = 4.0 + rng.uniform(-1.0, 1.0);
        let x = Vector3::new(r_cyl * phi.sin(), rng.uniform(-3.0, 3.0), r_cyl * phi.cos());
        let mut members: Vec<(u32, [f64; 2])> = Vec::new();
        for i in 0..n_img {
            if (thetas[i] - phi).cos() <= vis_cos {
                continue;
            }
            let pc = quats[i] * (x - centers[i]);
            if pc.z >= -1e-6 {
                continue;
            }
            let Some((u, v)) = cam.ray_to_pixel([pc.x, pc.y, pc.z]) else {
                continue;
            };
            if !(0.0..W as f64).contains(&u) || !(0.0..H as f64).contains(&v) {
                continue;
            }
            members.push((i as u32, [u, v]));
        }
        if members.len() < 2 {
            continue;
        }
        for (i, p) in members {
            cluster.push(cid);
            image.push(i);
            pos.push([p[0] + noise * rng.gaussian(), p[1] + noise * rng.gaussian()]);
        }
        world.push(x);
        cid += 1;
    }
    Scene {
        cluster,
        image,
        pos,
        quats,
        centers,
        world,
    }
}

/// Replace a fraction of one image's observation positions with junk
/// (uniform in-frame pixels), returning how many rows stayed good.
fn corrupt_image(scene: &mut Scene, img: u32, junk_fraction: f64, rng: &mut Lcg) -> (usize, usize) {
    let mut n_junk = 0;
    let mut n_good = 0;
    for k in 0..scene.image.len() {
        if scene.image[k] != img {
            continue;
        }
        if rng.next_f64() < junk_fraction {
            scene.pos[k] = [rng.uniform(0.0, W as f64), rng.uniform(0.0, H as f64)];
            n_junk += 1;
        } else {
            n_good += 1;
        }
    }
    (n_good, n_junk)
}

// ── Alignment metrics ────────────────────────────────────────────────────────

fn polar_rotation(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let p = svd.u.unwrap() * svd.v_t.unwrap();
    if p.determinant() < 0.0 {
        -p
    } else {
        p
    }
}

fn rotation_angle(r: &Matrix3<f64>) -> f64 {
    (((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)).acos()
}

/// Gauge-align estimated rotations to ground truth and return per-image
/// angular errors in degrees.
fn aligned_rotation_errors(gt: &[Matrix3<f64>], est: &[Matrix3<f64>]) -> Vec<f64> {
    let mut sum = Matrix3::<f64>::zeros();
    for (g, e) in gt.iter().zip(est) {
        sum += g.transpose() * e;
    }
    let gauge = polar_rotation(&sum);
    gt.iter()
        .zip(est)
        .map(|(g, e)| rotation_angle(&(e * (g * gauge).transpose())).to_degrees())
        .collect()
}

/// Similarity fit of `xs` onto `ys`; per-point residual norms and the spread
/// of `ys`.
fn similarity_residuals(xs: &[Vector3<f64>], ys: &[Vector3<f64>]) -> (Vec<f64>, f64) {
    let n = xs.len() as f64;
    let cx = xs.iter().sum::<Vector3<f64>>() / n;
    let cy = ys.iter().sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::<f64>::zeros();
    let mut var_x = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        cov += (y - cy) * (x - cx).transpose();
        var_x += (x - cx).norm_squared();
    }
    let r = polar_rotation(&cov);
    let s = (r.transpose() * cov).trace() / var_x.max(1e-300);
    let res = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| ((y - cy) - s * (r * (x - cx))).norm())
        .collect();
    let spread = (ys.iter().map(|y| (y - cy).norm_squared()).sum::<f64>() / n).sqrt();
    (res, spread)
}

/// Posed-camera center and rotation errors of a growth result against the
/// scene ground truth: `(max center residual / spread, max rotation error
/// deg)`.
fn camera_errors(scene: &Scene, out: &ReconstructionGrowth) -> (f64, f64) {
    let mut est_c = Vec::new();
    let mut gt_c = Vec::new();
    let mut est_r = Vec::new();
    let mut gt_r = Vec::new();
    for i in 0..out.posed.len() {
        if !out.posed[i] {
            continue;
        }
        let q = out.quaternions_wxyz[i];
        let rq = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
        let t = Vector3::new(
            out.translations[i][0],
            out.translations[i][1],
            out.translations[i][2],
        );
        est_c.push(-(rq.inverse() * t));
        gt_c.push(scene.centers[i]);
        est_r.push(rq.to_rotation_matrix().into_inner());
        gt_r.push(scene.quats[i].to_rotation_matrix().into_inner());
    }
    let (res, spread) = similarity_residuals(&est_c, &gt_c);
    let max_c = res.iter().cloned().fold(0.0, f64::max) / spread;
    let rot_errs = aligned_rotation_errors(&gt_r, &est_r);
    let max_r = rot_errs.iter().cloned().fold(0.0, f64::max);
    (max_c, max_r)
}

fn grow(
    scene: &Scene,
    cam: &CameraIntrinsics,
    n_seed: usize,
    opts: &GrowOptions,
) -> ReconstructionGrowth {
    let (q, t, idx) = scene.seed(n_seed);
    grow_reconstruction(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        cam,
        &q,
        &t,
        &idx,
        opts,
    )
}

fn assert_growth_bits_eq(a: &ReconstructionGrowth, b: &ReconstructionGrowth) {
    let bits3 = |v: &[[f64; 3]]| -> Vec<[u64; 3]> {
        v.iter()
            .map(|r| [r[0].to_bits(), r[1].to_bits(), r[2].to_bits()])
            .collect()
    };
    assert_eq!(a.posed, b.posed);
    for (qa, qb) in a.quaternions_wxyz.iter().zip(&b.quaternions_wxyz) {
        assert_eq!(
            qa.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            qb.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    assert_eq!(bits3(&a.translations), bits3(&b.translations));
    assert_eq!(bits3(&a.points), bits3(&b.points));
    assert_eq!(a.focal.to_bits(), b.focal.to_bits());
    assert_eq!(
        a.residual_norms
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        b.residual_norms
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
}

// ── Growth on a synthetic orbit ──────────────────────────────────────────────

#[test]
fn orbit_grows_to_full_registration() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(16, 240, 0.2, 0.4, &mut rng);
    let out = grow(&scene, &test_cam(F0), 3, &GrowOptions::default());

    assert!(out.posed.iter().all(|&p| p), "posed {:?}", out.posed);
    let (max_c, max_r) = camera_errors(&scene, &out);
    assert!(max_c < 0.02, "center error {max_c} of spread");
    assert!(max_r < 0.5, "rotation error {max_r} deg");
    assert!(
        (out.focal - F0).abs() / F0 < 0.01,
        "released focal {} vs truth {F0}",
        out.focal
    );

    // Residuals: most observations reproject tightly at the final state.
    let finite: Vec<f64> = out
        .residual_norms
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    assert!(
        finite.len() * 10 >= out.residual_norms.len() * 9,
        "finite residuals {}/{}",
        finite.len(),
        out.residual_norms.len()
    );
    let med = median(&finite);
    assert!(med < 1.0, "median residual {med} px");

    // Structure: triangulated points match the ground-truth cloud under a
    // similarity fit.
    let mut est_p = Vec::new();
    let mut gt_p = Vec::new();
    for (c, p) in out.points.iter().enumerate() {
        if p[0].is_finite() {
            est_p.push(Vector3::new(p[0], p[1], p[2]));
            gt_p.push(scene.world[c]);
        }
    }
    assert!(est_p.len() * 10 >= scene.world.len() * 9);
    let (res, spread) = similarity_residuals(&est_p, &gt_p);
    assert!(median(&res) < 0.02 * spread);
}

#[test]
fn focal_release_pulls_toward_truth() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(16, 240, 0.2, 0.4, &mut rng);
    // Growth runs at a 5%-off focal; the finishing release recovers it.
    let f_off = 1.05 * F0;
    let out = grow(&scene, &test_cam(f_off), 3, &GrowOptions::default());
    assert!(out.posed.iter().all(|&p| p));
    assert!(
        (out.focal - F0).abs() < (f_off - F0).abs(),
        "release did not improve: {} vs input {f_off}",
        out.focal
    );
    assert!(
        (out.focal - F0).abs() / F0 < 0.02,
        "released focal {} vs truth {F0}",
        out.focal
    );
}

#[test]
fn determinism_same_inputs() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(14, 200, 0.2, 0.4, &mut rng);
    let opts = GrowOptions {
        seed: 42,
        ..Default::default()
    };
    let a = grow(&scene, &test_cam(F0), 3, &opts);
    let b = grow(&scene, &test_cam(F0), 3, &opts);
    assert_growth_bits_eq(&a, &b);
}

// ── Bounded adjustments ──────────────────────────────────────────────────────

#[test]
fn ba_window_at_or_above_posed_reproduces_unbounded() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(14, 200, 0.2, 0.4, &mut rng);
    let unbounded = grow(&scene, &test_cam(F0), 3, &GrowOptions::default());
    let windowed = grow(
        &scene,
        &test_cam(F0),
        3,
        &GrowOptions {
            ba_window: 14,
            ..Default::default()
        },
    );
    assert_growth_bits_eq(&unbounded, &windowed);
}

#[test]
fn bounded_window_still_registers_full_orbit() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(16, 240, 0.2, 0.4, &mut rng);
    let out = grow(
        &scene,
        &test_cam(F0),
        3,
        &GrowOptions {
            ba_window: 5,
            ..Default::default()
        },
    );
    assert!(out.posed.iter().all(|&p| p), "posed {:?}", out.posed);
    let (max_c, max_r) = camera_errors(&scene, &out);
    assert!(max_c < 0.05, "center error {max_c} of spread");
    assert!(max_r < 2.0, "rotation error {max_r} deg");
}

#[test]
fn anchor_every_beats_frontier_only_on_long_loop() {
    let mut rng = Lcg(19);
    let scene = orbit_scene(30, 420, 0.4, 0.35, &mut rng);
    let frontier = grow(
        &scene,
        &test_cam(F0),
        3,
        &GrowOptions {
            ba_window: 5,
            ..Default::default()
        },
    );
    let anchored = grow(
        &scene,
        &test_cam(F0),
        3,
        &GrowOptions {
            ba_window: 5,
            anchor_every: 3,
            ..Default::default()
        },
    );
    assert!(frontier.posed.iter().all(|&p| p));
    assert!(anchored.posed.iter().all(|&p| p));
    let (front_c, _) = camera_errors(&scene, &frontier);
    let (anch_c, _) = camera_errors(&scene, &anchored);
    assert!(
        anch_c < front_c,
        "anchored max center error {anch_c} should beat frontier-only {front_c}"
    );
}

#[test]
fn ba_cluster_cap_still_registers() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(14, 200, 0.2, 0.4, &mut rng);
    let n_cl = scene.world.len();
    let out = grow(
        &scene,
        &test_cam(F0),
        3,
        &GrowOptions {
            ba_cluster_cap: n_cl / 2,
            ..Default::default()
        },
    );
    assert!(out.posed.iter().all(|&p| p));
    let (max_c, max_r) = camera_errors(&scene, &out);
    assert!(max_c < 0.05, "center error {max_c} of spread");
    assert!(max_r < 2.0, "rotation error {max_r} deg");
}

// ── Gates: deferral, verified force-accept, restore-on-reject ────────────────

#[test]
fn junk_dominated_image_is_deferred_then_force_accepted() {
    let mut rng = Lcg(7);
    let mut scene = orbit_scene(14, 320, 0.2, 0.4, &mut rng);
    // The victim sits opposite the seed (0..3) on the orbit, so by the time
    // growth reaches it the acceptance gate is armed with a clean median.
    let victim = 7u32;
    let (n_good, n_junk) = corrupt_image(&mut scene, victim, 0.8, &mut rng);
    assert!(
        n_good >= 15,
        "fixture needs a P3P-findable consensus: {n_good}"
    );
    assert!(n_junk > 3 * n_good, "fixture must be junk-dominated");

    let out = grow(&scene, &test_cam(F0), 3, &GrowOptions::default());
    assert!(
        out.posed[victim as usize],
        "junk-dominated image should force-accept through its consensus"
    );
    // Its verified pose is accurate despite the junk observations.
    let q = out.quaternions_wxyz[victim as usize];
    let rq = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
    let t = out.translations[victim as usize];
    let est_center = -(rq.inverse() * Vector3::new(t[0], t[1], t[2]));
    // Compare in the reconstruction gauge via all posed cameras.
    let (max_c, max_r) = camera_errors(&scene, &out);
    assert!(max_c < 0.05, "center error {max_c} of spread");
    assert!(max_r < 2.0, "rotation error {max_r} deg");
    assert!(est_center.iter().all(|v| v.is_finite()));
}

#[test]
fn all_junk_image_is_force_rejected() {
    let mut rng = Lcg(7);
    let mut scene = orbit_scene(14, 240, 0.2, 0.4, &mut rng);
    // Opposite the seed, so the gate is armed before growth reaches it (a
    // seed-adjacent junk image would be the FIRST resection, which is ungated
    // by design — nothing has been accepted to gate against yet).
    let victim = 7u32;
    corrupt_image(&mut scene, victim, 1.0, &mut rng);

    let out = grow(&scene, &test_cam(F0), 3, &GrowOptions::default());
    assert!(
        !out.posed[victim as usize],
        "an all-junk image must not survive force-accept verification"
    );
    assert_eq!(out.posed.iter().filter(|&&p| p).count(), 13);
    // Its observations are invalid in the output.
    for (k, &i) in scene.image.iter().enumerate() {
        if i == victim {
            assert!(out.residual_norms[k].is_infinite());
        }
    }
    // The rejected force-accept restored the prior state: the remaining
    // cameras are as accurate as a clean growth.
    let (max_c, max_r) = camera_errors(&scene, &out);
    assert!(max_c < 0.02, "center error {max_c} of spread");
    assert!(max_r < 0.5, "rotation error {max_r} deg");
}

// ── Degenerate inputs ────────────────────────────────────────────────────────

#[test]
fn no_seed_poses_returns_input_state() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(10, 120, 0.2, 0.4, &mut rng);
    let out = grow_reconstruction(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &[],
        &[],
        &[],
        &GrowOptions::default(),
    );
    assert!(out.posed.iter().all(|&p| !p));
    assert!(out.points.iter().all(|p| p[0].is_nan()));
    assert!(out.residual_norms.iter().all(|r| r.is_infinite()));
    assert_eq!(out.focal, F0);
    for q in &out.quaternions_wxyz {
        assert_eq!(q, &[1.0, 0.0, 0.0, 0.0]);
    }
}

#[test]
fn all_images_below_min_obs_returns_input_state() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(10, 120, 0.2, 0.4, &mut rng);
    let (q, t, idx) = scene.seed(3);
    let out = grow_reconstruction(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &q,
        &t,
        &idx,
        &GrowOptions {
            min_obs: 100_000,
            ..Default::default()
        },
    );
    // Seed poses pass through (quaternions modulo the input unit
    // re-normalization); nothing else registers.
    assert_eq!(out.posed.iter().filter(|&&p| p).count(), 3);
    for (k, &i) in idx.iter().enumerate() {
        for (a, b) in out.quaternions_wxyz[i as usize].iter().zip(&q[k]) {
            assert!((a - b).abs() < 1e-14);
        }
        assert_eq!(out.translations[i as usize], t[k]);
    }
    assert_eq!(out.focal, F0);
    // The seed still triangulates its covisible clusters.
    assert!(out.points.iter().any(|p| p[0].is_finite()));
}

#[test]
fn no_triangulable_clusters_returns_input_state() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(10, 120, 0.2, 0.4, &mut rng);
    // A single seed pose triangulates nothing, so no image ever has a valid
    // 2D-3D candidate set.
    let (q, t, idx) = scene.seed(1);
    let out = grow_reconstruction(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &q,
        &t,
        &idx,
        &GrowOptions::default(),
    );
    assert_eq!(out.posed.iter().filter(|&&p| p).count(), 1);
    for (a, b) in out.quaternions_wxyz[0].iter().zip(&q[0]) {
        assert!((a - b).abs() < 1e-14);
    }
    assert_eq!(out.translations[0], t[0]);
    assert!(out.points.iter().all(|p| p[0].is_nan()));
    assert_eq!(out.focal, F0);
}

#[test]
fn empty_inputs_return_empty_growth() {
    let out = grow_reconstruction(
        &[],
        &[],
        &[],
        &test_cam(F0),
        &[],
        &[],
        &[],
        &GrowOptions::default(),
    );
    assert!(out.posed.is_empty());
    assert!(out.points.is_empty());
    assert!(out.residual_norms.is_empty());
    assert_eq!(out.focal, F0);
}

// ── resect_images_batch ──────────────────────────────────────────────────────

/// Ground-truth structure and seed poses for the batch registration tests.
type BatchFixture = (Vec<[f64; 3]>, Vec<[f64; 4]>, Vec<[f64; 3]>, Vec<u32>);

fn batch_fixture(scene: &Scene) -> BatchFixture {
    let points: Vec<[f64; 3]> = scene.world.iter().map(|w| [w.x, w.y, w.z]).collect();
    let (q, t, idx) = scene.seed(3);
    (points, q, t, idx)
}

#[test]
fn batch_resection_registers_all_images_against_structure() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(14, 240, 0.2, 0.4, &mut rng);
    let (points, q, t, idx) = batch_fixture(&scene);
    let image_list: Vec<u32> = (0..14).collect();
    let out = resect_images_batch(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &points,
        &image_list,
        &q,
        &t,
        &idx,
        &ResectOptions::default(),
    );
    assert!(
        out.accepted.iter().all(|&a| a),
        "accepted {:?}",
        out.accepted
    );
    // The structure is ground truth, so poses match without a gauge fit.
    for (k, &i) in image_list.iter().enumerate() {
        let qk = out.quaternions_wxyz[k];
        let rq = UnitQuaternion::from_quaternion(Quaternion::new(qk[0], qk[1], qk[2], qk[3]));
        let tk = Vector3::new(
            out.translations[k][0],
            out.translations[k][1],
            out.translations[k][2],
        );
        let center = -(rq.inverse() * tk);
        assert!(
            (center - scene.centers[i as usize]).norm() < 0.05,
            "image {i} center off by {}",
            (center - scene.centers[i as usize]).norm()
        );
        let rel = rq.angle_to(&scene.quats[i as usize]).to_degrees();
        assert!(rel < 0.3, "image {i} rotation off by {rel} deg");
        assert!(out.inlier_fractions[k] > 0.9);
    }
}

#[test]
fn batch_resection_is_deterministic_and_matches_single_calls() {
    let mut rng = Lcg(7);
    let scene = orbit_scene(14, 240, 0.2, 0.4, &mut rng);
    let (points, q, t, idx) = batch_fixture(&scene);
    let image_list: Vec<u32> = (0..14).collect();
    let opts = ResectOptions {
        seed: 5,
        ..Default::default()
    };
    let run = |list: &[u32]| {
        resect_images_batch(
            &scene.cluster,
            &scene.image,
            &scene.pos,
            &test_cam(F0),
            &points,
            list,
            &q,
            &t,
            &idx,
            &opts,
        )
    };
    let a = run(&image_list);
    let b = run(&image_list);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for k in 0..image_list.len() {
        assert_eq!(bits(&a.quaternions_wxyz[k]), bits(&b.quaternions_wxyz[k]));
        assert_eq!(bits(&a.translations[k]), bits(&b.translations[k]));
    }
    assert_eq!(bits(&a.inlier_fractions), bits(&b.inlier_fractions));
    assert_eq!(a.accepted, b.accepted);

    // Each image's RANSAC is a pure function of (seed, image index), so a
    // one-image call matches its batch row bit for bit.
    for (k, &i) in image_list.iter().enumerate() {
        let single = run(&[i]);
        assert_eq!(
            bits(&single.quaternions_wxyz[0]),
            bits(&a.quaternions_wxyz[k])
        );
        assert_eq!(bits(&single.translations[0]), bits(&a.translations[k]));
        assert_eq!(
            single.inlier_fractions[0].to_bits(),
            a.inlier_fractions[k].to_bits()
        );
        assert_eq!(single.accepted[0], a.accepted[k]);
    }
}

#[test]
fn batch_resection_gates_junk_and_min_obs() {
    let mut rng = Lcg(7);
    let mut scene = orbit_scene(14, 240, 0.2, 0.4, &mut rng);
    corrupt_image(&mut scene, 13, 1.0, &mut rng);
    let (points, q, t, idx) = batch_fixture(&scene);
    let out = resect_images_batch(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &points,
        &[12, 13],
        &q,
        &t,
        &idx,
        &ResectOptions::default(),
    );
    assert!(out.accepted[0], "clean image should register");
    assert!(!out.accepted[1], "all-junk image must not clear the gate");

    // min_obs above any candidate count skips every image.
    let skipped = resect_images_batch(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &points,
        &[12],
        &q,
        &t,
        &idx,
        &ResectOptions {
            min_obs: 100_000,
            ..Default::default()
        },
    );
    assert!(!skipped.accepted[0]);
    assert_eq!(skipped.quaternions_wxyz[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(skipped.inlier_fractions[0], 0.0);

    // All-NaN structure: nothing to resect against, empty growth.
    let nan_points = vec![[f64::NAN; 3]; points.len()];
    let none = resect_images_batch(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &nan_points,
        &[12],
        &q,
        &t,
        &idx,
        &ResectOptions::default(),
    );
    assert!(!none.accepted[0]);
}
