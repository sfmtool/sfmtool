// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::CameraModel;
use nalgebra::Rotation3;

const W: u32 = 800;
const H: u32 = 800;
const F0: f64 = 700.0;

/// Deterministic LCG so fixtures need no `rand` and are bitwise-stable
/// (mirrors `reconstruction_growth::tests`).
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

/// Ground-truth capture in the `reconstruction_growth::tests` orbit style,
/// but with *stations*: `per_station` cameras share each orbit position
/// (identical centre, small yaw/pitch offsets). Same-station pairs are pure
/// rotations — genuine near-duplicate viewpoints where the conjugate
/// homography holds exactly — while cross-station pairs carry real parallax,
/// so the displacement neighborhood's `nearest` ranking has actual structure
/// to find. World points sit on a jittered cylinder with visibility limited
/// to front-facing stations.
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
    fn n_img(&self) -> usize {
        self.quats.len()
    }

    /// World-to-camera pose arrays for every camera.
    fn pose_arrays(&self) -> (Vec<[f64; 4]>, Vec<[f64; 3]>, Vec<u32>) {
        let q = self
            .quats
            .iter()
            .map(|r| {
                let q = r.into_inner();
                [q.w, q.i, q.j, q.k]
            })
            .collect();
        let t = self
            .quats
            .iter()
            .zip(&self.centers)
            .map(|(r, c)| {
                let t = -(r * c);
                [t.x, t.y, t.z]
            })
            .collect();
        (q, t, (0..self.n_img() as u32).collect())
    }

    fn points(&self) -> Vec<[f64; 3]> {
        self.world.iter().map(|w| [w.x, w.y, w.z]).collect()
    }

    /// The displacement neighborhood over this scene's cluster tracks.
    fn neighborhood(&self) -> DisplacementNeighborhood {
        let n_cl = self.world.len();
        let mut starts = vec![0u32; n_cl + 1];
        for &c in &self.cluster {
            starts[c as usize + 1] += 1;
        }
        for c in 0..n_cl {
            starts[c + 1] += starts[c];
        }
        DisplacementNeighborhood::from_clusters(&starts, &self.image, None, self.n_img(), &self.pos)
            .unwrap()
    }
}

fn station_scene(
    n_stations: usize,
    per_station: usize,
    n_pts: usize,
    noise: f64,
    vis_cos: f64,
    rng: &mut Lcg,
) -> Scene {
    use std::f64::consts::TAU;
    let cam = test_cam(F0);
    let r_orbit = 10.0;
    let n_img = n_stations * per_station;

    let mut quats = Vec::with_capacity(n_img);
    let mut centers = Vec::with_capacity(n_img);
    let mut station_theta = Vec::with_capacity(n_stations);
    for s in 0..n_stations {
        let th = TAU * s as f64 / n_stations as f64;
        station_theta.push(th);
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
        let r_base = Matrix3::from_rows(&[x_cam.transpose(), y_cam.transpose(), z_cam.transpose()]);
        let q_base =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_base));
        for m in 0..per_station {
            // Small camera-frame yaw spread plus pitch jitter; the centre is
            // shared, so same-station pairs are pure rotations.
            let yaw = (m as f64 - (per_station as f64 - 1.0) / 2.0) * 2.0f64.to_radians();
            let pitch = rng.uniform(-1.0, 1.0).to_radians();
            let delta = UnitQuaternion::from_scaled_axis(Vector3::new(pitch, yaw, 0.0));
            quats.push(delta * q_base);
            centers.push(c);
        }
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
            let s = i / per_station;
            if (station_theta[s] - phi).cos() <= vis_cos {
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

/// The standard fixture: 4 stations x 5 cameras, so every camera's four
/// nearest low-displacement partners are its same-station (pure-rotation)
/// mates.
fn standard_scene(seed: u64) -> Scene {
    station_scene(4, 5, 500, 0.2, 0.4, &mut Lcg(seed))
}

/// Corrupt one camera's pose: rotate by `angle_deg` about a camera-frame
/// axis and shift the centre.
fn corrupt_pose(scene: &mut Scene, img: usize, angle_deg: f64) {
    let delta = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, angle_deg.to_radians(), 0.0));
    scene.quats[img] = delta * scene.quats[img];
    scene.centers[img] += Vector3::new(0.5, 0.3, -0.4);
}

/// Replace a fraction of one image's observation positions with junk
/// (uniform in-frame pixels).
fn corrupt_observations(scene: &mut Scene, img: u32, junk_fraction: f64, rng: &mut Lcg) {
    for k in 0..scene.image.len() {
        if scene.image[k] == img && rng.next_f64() < junk_fraction {
            scene.pos[k] = [rng.uniform(0.0, W as f64), rng.uniform(0.0, H as f64)];
        }
    }
}

fn verify(scene: &Scene, options: &VerifyOptions) -> PoseVerification {
    let (q, t, idx) = scene.pose_arrays();
    verify_poses(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        &scene.points(),
        &q,
        &t,
        &idx,
        &scene.neighborhood(),
        options,
    )
}

fn repair(scene: &Scene, points: &[[f64; 3]], options: &RepairOptions) -> PoseRepair {
    let (q, t, idx) = scene.pose_arrays();
    repair_poses(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        &test_cam(F0),
        points,
        &q,
        &t,
        &idx,
        &scene.neighborhood(),
        options,
    )
}

// ── Screens ──────────────────────────────────────────────────────────────────

#[test]
fn clean_scene_yields_no_flags() {
    let scene = standard_scene(7);
    let out = verify(&scene, &VerifyOptions::default());
    assert!(!out.flagged.iter().any(|&f| f), "flags {:?}", out.flagged);
    assert!(!out.resect_flags.iter().any(|&f| f));
    assert!(!out.rotation_flags.iter().any(|&f| f));
    // Every camera has four same-station neighbours, so screen B measures
    // everywhere, and the pure-rotation homographies agree with the poses.
    for (i, &s) in out.rotation_scores_deg.iter().enumerate() {
        assert!(s.is_finite(), "camera {i} score not measured");
        assert!(s < 1.0, "camera {i} clean score {s} deg");
    }
    for &f in &out.resect_inlier_fractions {
        assert!(f > 0.9, "clean self-resection inliers {f}");
    }
}

#[test]
fn wrong_pose_flagged_by_rotation_screen() {
    let mut scene = standard_scene(7);
    let victim = 12usize;
    corrupt_pose(&mut scene, victim, 8.0);
    let out = verify(&scene, &VerifyOptions::default());

    assert!(out.rotation_flags[victim], "victim not flagged by screen B");
    assert!(out.flagged[victim]);
    assert!(
        out.rotation_scores_deg[victim] > 3.0,
        "victim score {}",
        out.rotation_scores_deg[victim]
    );
    // Screen A re-derives a pose from the victim's healthy observations, so
    // it does not implicate the stored pose — that is screen B's job.
    assert!(!out.resect_flags[victim]);
    // The per-image median keeps the victim's neighbours clean: each sees
    // exactly one discrepant pair (with the victim) among four.
    for i in 0..scene.n_img() {
        if i != victim {
            assert!(!out.flagged[i], "camera {i} falsely flagged");
        }
    }
}

#[test]
fn wrong_pose_with_junk_support_flagged_by_both_screens() {
    let mut scene = standard_scene(7);
    let victim = 7usize;
    corrupt_pose(&mut scene, victim, 8.0);
    // 80% junk: below the resection accept gate, while the clean 20% of the
    // pair correspondences still validates the neighbour homographies.
    corrupt_observations(&mut scene, victim as u32, 0.8, &mut Lcg(99));
    let out = verify(&scene, &VerifyOptions::default());

    assert!(out.resect_flags[victim], "screen A missed the junk support");
    assert!(out.rotation_flags[victim], "screen B missed the wrong pose");
    assert!(out.flagged[victim]);
    for i in 0..scene.n_img() {
        if i != victim {
            assert!(!out.flagged[i], "camera {i} falsely flagged");
        }
    }
}

#[test]
fn high_parallax_pair_alone_never_flags_rotation_screen() {
    // Two wide-baseline cameras (a quarter orbit apart, one per station):
    // translation-rich correspondences are exactly where the conjugate-
    // homography model breaks, so screen B must abstain rather than measure.
    let mut rng = Lcg(11);
    let mut scene = station_scene(2, 1, 300, 0.2, -0.5, &mut rng);
    let nb = scene.neighborhood();
    let (shared, _) = nb.pair(0, 1).expect("pair must be covisible");
    assert!(shared >= 50, "fixture needs a realized pair, got {shared}");

    let clean = verify(&scene, &VerifyOptions::default());
    assert!(!clean.rotation_flags.iter().any(|&f| f));
    assert!(clean.rotation_scores_deg.iter().all(|s| s.is_nan()));

    // Even a genuinely wrong pose stays unflagged by screen B here — a
    // single translation-rich pair is never evidence.
    corrupt_pose(&mut scene, 1, 10.0);
    let bad = verify(&scene, &VerifyOptions::default());
    assert!(!bad.rotation_flags.iter().any(|&f| f));
}

// ── Repair ───────────────────────────────────────────────────────────────────

#[test]
fn repair_restores_implanted_wrong_pose() {
    let mut scene = standard_scene(7);
    let victim = 12usize;
    let true_quat = scene.quats[victim];
    let true_center = scene.centers[victim];
    corrupt_pose(&mut scene, victim, 8.0);
    let (q_in, t_in, _) = scene.pose_arrays();

    let out = repair(&scene, &scene.points(), &RepairOptions::default());
    assert!(out.verification.flagged[victim]);
    assert!(out.repaired[victim], "repair not accepted");
    assert!(
        out.inlier_after[victim] > 0.9,
        "after {}",
        out.inlier_after[victim]
    );
    assert!(
        out.inlier_before[victim] < 0.1,
        "before {}",
        out.inlier_before[victim]
    );

    // The repaired pose is restored to within tight bounds of truth.
    let q = out.quaternions_wxyz[victim];
    let rq = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
    let t = Vector3::new(
        out.translations[victim][0],
        out.translations[victim][1],
        out.translations[victim][2],
    );
    let center = -(rq.inverse() * t);
    assert!(
        rq.angle_to(&true_quat).to_degrees() < 0.3,
        "rotation off by {} deg",
        rq.angle_to(&true_quat).to_degrees()
    );
    assert!(
        (center - true_center).norm() < 0.05,
        "centre off by {}",
        (center - true_center).norm()
    );

    // Every other camera passes through bit for bit.
    for i in 0..scene.n_img() {
        if i == victim {
            continue;
        }
        assert_eq!(out.quaternions_wxyz[i], q_in[i], "camera {i} rotation");
        assert_eq!(out.translations[i], t_in[i], "camera {i} translation");
        assert!(!out.repaired[i]);
        assert!(out.inlier_before[i].is_nan() && out.inlier_after[i].is_nan());
    }
}

#[test]
fn repair_rejected_when_cluster_points_corrupted() {
    // A leaner scene than the screen tests: each of the flagged cameras'
    // resections exhausts its RANSAC budget against the junk structure, so
    // observation count is what debug-build time scales with here.
    let scene = station_scene(3, 4, 250, 0.2, 0.4, &mut Lcg(7));
    // Corrupt the world points of every cluster station 0 observes: its
    // cameras are flagged (screen A cannot re-derive a pose from junk
    // structure), but pose-only repair cannot fix broken structure either —
    // every repair must be rejected and the state left untouched.
    let mut points = scene.points();
    let mut junk = Lcg(13);
    let mut broken = vec![false; points.len()];
    for (k, &i) in scene.image.iter().enumerate() {
        if (i as usize) < 4 {
            broken[scene.cluster[k] as usize] = true;
        }
    }
    for (c, p) in points.iter_mut().enumerate() {
        if broken[c] {
            *p = [
                junk.uniform(-20.0, 20.0),
                junk.uniform(-20.0, 20.0),
                junk.uniform(-20.0, 20.0),
            ];
        }
    }
    let (q_in, t_in, _) = scene.pose_arrays();

    let out = repair(&scene, &points, &RepairOptions::default());
    for i in 0..4 {
        assert!(out.verification.resect_flags[i], "camera {i} not flagged");
        assert!(!out.repaired[i], "camera {i} repair must be rejected");
        assert!(
            out.inlier_after[i].is_nan() || out.inlier_after[i] < 0.10,
            "camera {i} junk repair reached {}",
            out.inlier_after[i]
        );
    }
    assert!(!out.repaired.iter().any(|&r| r));
    // Rejected repairs leave every pose untouched, bit for bit.
    for i in 0..scene.n_img() {
        assert_eq!(out.quaternions_wxyz[i], q_in[i], "camera {i} rotation");
        assert_eq!(out.translations[i], t_in[i], "camera {i} translation");
    }
    // The flags stand.
    assert!(out.verification.flagged[..4].iter().all(|&f| f));
}

// ── Determinism ──────────────────────────────────────────────────────────────

fn assert_verification_bits_eq(a: &PoseVerification, b: &PoseVerification) {
    assert_eq!(a.resect_flags, b.resect_flags);
    assert_eq!(a.rotation_flags, b.rotation_flags);
    assert_eq!(a.flagged, b.flagged);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&a.resect_inlier_fractions),
        bits(&b.resect_inlier_fractions)
    );
    assert_eq!(bits(&a.rotation_scores_deg), bits(&b.rotation_scores_deg));
}

#[test]
fn determinism_same_inputs() {
    // A leaner fixture than the screen tests (the junk camera's RANSAC
    // paths dominate debug-build time), still exercising both screens, an
    // accepted repair, and the junk-support path.
    let mut scene = station_scene(3, 5, 320, 0.2, 0.4, &mut Lcg(7));
    corrupt_pose(&mut scene, 7, 8.0);
    corrupt_observations(&mut scene, 7, 0.7, &mut Lcg(99));
    corrupt_pose(&mut scene, 12, 6.0);
    let options = RepairOptions {
        verify: VerifyOptions {
            seed: 42,
            ..Default::default()
        },
        ..Default::default()
    };

    let a = repair(&scene, &scene.points(), &options);
    let b = repair(&scene, &scene.points(), &options);
    assert_verification_bits_eq(&a.verification, &b.verification);
    assert_eq!(a.repaired, b.repaired);
    let bits3 = |v: &[[f64; 3]]| -> Vec<[u64; 3]> {
        v.iter()
            .map(|r| [r[0].to_bits(), r[1].to_bits(), r[2].to_bits()])
            .collect()
    };
    let bits4 = |v: &[[f64; 4]]| -> Vec<[u64; 4]> {
        v.iter()
            .map(|r| {
                [
                    r[0].to_bits(),
                    r[1].to_bits(),
                    r[2].to_bits(),
                    r[3].to_bits(),
                ]
            })
            .collect()
    };
    assert_eq!(bits4(&a.quaternions_wxyz), bits4(&b.quaternions_wxyz));
    assert_eq!(bits3(&a.translations), bits3(&b.translations));
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.inlier_before), bits(&b.inlier_before));
    assert_eq!(bits(&a.inlier_after), bits(&b.inlier_after));
}
