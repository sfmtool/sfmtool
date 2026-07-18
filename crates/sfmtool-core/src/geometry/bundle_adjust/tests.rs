// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::CameraModel;

fn simple_pinhole(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: f,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

/// Deterministic pseudo-random in [-1, 1] from an index (no rand dependency).
fn jitter(i: usize, salt: u64) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z ^= z >> 27;
    ((z % 20001) as f64 / 10000.0) - 1.0
}

/// A synthetic multi-view scene: ground-truth poses (cameras on an arc
/// looking at the origin), world points, and observations of every point in
/// every camera that sees it.
struct Scene {
    cam: CameraIntrinsics,
    quats: Vec<UnitQuaternion<f64>>,
    trans: Vec<Vector3<f64>>,
    points: Vec<[f64; 3]>,
    uv: Vec<[f64; 2]>,
    obs_img: Vec<u32>,
    obs_pt: Vec<u32>,
}

fn make_scene(n_img: usize, n_pt: usize) -> Scene {
    make_scene_cam(simple_pinhole(500.0), n_img, n_pt)
}

fn make_scene_cam(cam: CameraIntrinsics, n_img: usize, n_pt: usize) -> Scene {
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        // Cameras on a shallow arc at radius 8, looking at the origin.
        let ang = 0.15 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(8.0 * ang.sin(), 0.5 * jitter(i, 11), 8.0 * ang.cos());
        // Canonical look-at: the camera looks along −Z, so its local +Z axis
        // points AWAY from the origin (along `center`).
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
    }
    let mut points = Vec::new();
    for p in 0..n_pt {
        points.push([2.0 * jitter(p, 1), 2.0 * jitter(p, 2), 1.5 * jitter(p, 3)]);
    }
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    for (p, x) in points.iter().enumerate() {
        for i in 0..n_img {
            let c = quats[i] * Vector3::new(x[0], x[1], x[2]) + trans[i];
            if c.z >= -0.5 {
                continue;
            }
            let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            if !(0.0..cam.width as f64).contains(&u) || !(0.0..cam.height as f64).contains(&v) {
                continue;
            }
            uv.push([u, v]);
            obs_img.push(i as u32);
            obs_pt.push(p as u32);
        }
    }
    assert!(
        uv.len() >= n_img * n_pt / 2,
        "degenerate synthetic scene: only {} observations",
        uv.len()
    );
    Scene {
        cam,
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
    }
}

fn run(s: &mut Scene, opt_f: bool, schedule: &[BaSchedule]) -> BundleAdjustment {
    bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        opt_f,
        schedule,
        60,
        2,
        12,
    )
}

#[test]
fn perfect_data_stays_put() {
    let mut s = make_scene(6, 60);
    let q0 = s.quats.clone();
    let t0 = s.trans.clone();
    let out = run(&mut s, false, &DEFAULT_SCHEDULE);
    for k in 0..s.quats.len() {
        assert!(s.quats[k].angle_to(&q0[k]) < 1e-6, "camera {k} rotated");
        assert!((s.trans[k] - t0[k]).norm() < 1e-5, "camera {k} moved");
    }
    let max_res = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_res < 1e-5, "max residual {max_res}");
    assert_eq!(out.focal, 500.0);
}

#[test]
fn recovers_from_perturbed_state() {
    let mut s = make_scene(6, 60);
    let q_true = s.quats.clone();
    let t_true = s.trans.clone();
    // Perturb every pose and point (first camera held to pin the gauge —
    // with a shared camera the similarity gauge is otherwise free and the
    // absolute pose comparison below would need an alignment step).
    for i in 1..s.quats.len() {
        let d = Vector3::new(
            0.03 * jitter(i, 21),
            0.03 * jitter(i, 22),
            0.03 * jitter(i, 23),
        );
        s.quats[i] = UnitQuaternion::from_scaled_axis(d) * s.quats[i];
        s.trans[i] += Vector3::new(
            0.05 * jitter(i, 24),
            0.05 * jitter(i, 25),
            0.05 * jitter(i, 26),
        );
    }
    for (p, x) in s.points.iter_mut().enumerate() {
        for (c, xc) in x.iter_mut().enumerate() {
            *xc += 0.05 * jitter(p, 30 + c as u64);
        }
    }
    let out = run(&mut s, false, &DEFAULT_SCHEDULE);
    let med = {
        let mut r: Vec<f64> = out.residual_norms.clone();
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r[r.len() / 2]
    };
    assert!(med < 0.05, "median residual {med} px");
    // Gauge-pinned by camera 0, the other cameras should land near truth.
    for i in 0..s.quats.len() {
        let ang = s.quats[i].angle_to(&q_true[i]);
        assert!(ang < 5e-3, "camera {i} rotation err {ang} rad");
        let terr = (s.trans[i] - t_true[i]).norm();
        assert!(terr < 5e-2, "camera {i} translation err {terr}");
    }
}

#[test]
fn recovers_focal_started_20_percent_off() {
    let mut s = make_scene(8, 80);
    // Observations were generated at f = 500; hand the solver f = 600.
    s.cam = simple_pinhole(600.0);
    let out = run(&mut s, true, &DEFAULT_SCHEDULE);
    assert!(
        (out.focal - 500.0).abs() < 5.0,
        "focal {} (want ~500)",
        out.focal
    );
}

#[test]
fn junk_observations_do_not_pull_the_solution() {
    let mut s = make_scene(6, 60);
    let q_true = s.quats.clone();
    // Corrupt every 10th POINT's whole track with large per-observation
    // offsets — the bootstrap's real contamination is junk clusters (wrong
    // matches for every member), which the trim + min-track machinery drops
    // track-wise. Per-member corruption inside otherwise-good tracks is NOT
    // handled by design: the inter-round retriangulation rebuilds each point
    // from ALL its observations (the script's re-admission semantics), so a
    // junk member drags its own track until the trim excludes the track.
    let junk_track = |p: u32| p.is_multiple_of(10);
    for k in 0..s.uv.len() {
        if junk_track(s.obs_pt[k]) {
            s.uv[k][0] += 80.0 + 40.0 * jitter(k, 41);
            s.uv[k][1] -= 70.0 + 30.0 * jitter(k, 43);
        }
    }
    let out = run(&mut s, false, &DEFAULT_SCHEDULE);
    for (i, (q, qt)) in s.quats.iter().zip(&q_true).enumerate() {
        let ang = q.angle_to(qt);
        assert!(ang < 5e-3, "camera {i} rotation err {ang} rad");
    }
    // The clean tracks end sub-pixel; the junk tracks stay outliers.
    let mut clean: Vec<f64> = (0..s.uv.len())
        .filter(|&k| !junk_track(s.obs_pt[k]))
        .map(|k| out.residual_norms[k])
        .collect();
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        clean[clean.len() / 2] < 0.1,
        "clean median {}",
        clean[clean.len() / 2]
    );
    let junk: Vec<usize> = (0..s.uv.len())
        .filter(|&k| junk_track(s.obs_pt[k]))
        .collect();
    let junk_big = junk
        .iter()
        .filter(|&&k| out.residual_norms[k] > 10.0)
        .count();
    assert!(
        junk_big * 10 >= 8 * junk.len(),
        "only {junk_big} of {} junk obs stayed large",
        junk.len()
    );
}

#[test]
fn degenerate_exit_passes_state_through() {
    let mut s = make_scene(3, 5);
    let q0 = s.quats.clone();
    // A trim threshold no observation can pass (norm < 0 is impossible).
    let schedule = [BaSchedule {
        trim_px: 0.0,
        loss_scale: 1.0,
    }];
    let out = run_with_schedule(&mut s, &schedule);
    assert!(out.residual_norms.iter().all(|r| r.is_infinite()));
    for (q, q_orig) in s.quats.iter().zip(&q0) {
        assert!(q.angle_to(q_orig) < 1e-12, "state must pass through");
    }
}

fn run_with_schedule(s: &mut Scene, schedule: &[BaSchedule]) -> BundleAdjustment {
    run(s, false, schedule)
}

#[test]
fn min_track_drops_starved_points() {
    let mut s = make_scene(4, 30);
    // Perturb every point so the solve visibly moves the survivors, then
    // push one point's observations (except one) far off so trimming leaves
    // a single survivor — the whole track must leave the solve, and with a
    // single-round schedule (no retriangulation to overwrite it) the
    // starved point must come back bit-identical while clean points move.
    for (p, x) in s.points.iter_mut().enumerate() {
        for (c, xc) in x.iter_mut().enumerate() {
            *xc += 0.03 * jitter(p, 60 + c as u64);
        }
    }
    let victim = s.obs_pt[0] as usize;
    let victim_before = s.points[victim];
    let mut first = true;
    for k in 0..s.uv.len() {
        if s.obs_pt[k] as usize == victim {
            if first {
                first = false;
                continue;
            }
            s.uv[k][0] += 500.0;
        }
    }
    let schedule = [BaSchedule {
        trim_px: 25.0,
        loss_scale: 1.0,
    }];
    let out = run_with_schedule(&mut s, &schedule);
    assert_eq!(
        s.points[victim], victim_before,
        "starved track must be dropped from the solve (point untouched)"
    );
    let moved = (0..s.points.len())
        .filter(|&p| p != victim && s.points[p] != [0.0; 3])
        .filter(|&p| {
            let d: f64 = (0..3)
                .map(|c| (s.points[p][c] - victim_before[c]).abs())
                .sum();
            d > 0.0 // touched points differ from the victim; just count them
        })
        .count();
    assert!(moved > 0);
    // The corrupted rows end as outliers; the survivor row fits.
    let bad = (0..s.uv.len())
        .filter(|&k| s.obs_pt[k] as usize == victim && out.residual_norms[k] < 25.0)
        .count();
    assert!(bad <= 1, "corrupted track kept {bad} obs under the trim");
}

#[test]
fn fisheye_solve_via_numeric_jacobian() {
    // Non-perspective models have no analytic pixel Jacobian; the solve must
    // fall back to the central difference (a zero-Jacobian regression left
    // the LM unable to move anything while retriangulation still ran).
    let cam = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 200.0,
            focal_length_y: 200.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.01,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 480,
        height: 480,
    };
    let mut s = make_scene_cam(cam, 5, 40);
    let t_true = s.trans.clone();
    for i in 0..s.trans.len() {
        s.trans[i] += Vector3::new(
            0.04 * jitter(i, 71),
            0.04 * jitter(i, 72),
            0.04 * jitter(i, 73),
        );
    }
    // Single round: no retriangulation, so any improvement is the LM's.
    let schedule = [BaSchedule {
        trim_px: 50.0,
        loss_scale: 1.0,
    }];
    let out = run_with_schedule(&mut s, &schedule);
    let mut r: Vec<f64> = out.residual_norms.clone();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(r[r.len() / 2] < 0.05, "median residual {}", r[r.len() / 2]);
    let moved = (0..s.trans.len())
        .filter(|&i| (s.trans[i] - t_true[i]).norm() < 0.02)
        .count();
    assert!(
        moved >= s.trans.len() - 1,
        "only {moved} cameras recovered toward truth"
    );
}

#[test]
fn retriangulation_readmits_nan_points() {
    let mut s = make_scene(5, 40);
    // Wipe half the points to NaN; two rounds should re-create them from
    // the (posed) observations and leave their residuals small.
    for p in (0..s.points.len()).step_by(2) {
        s.points[p] = [f64::NAN; 3];
    }
    let schedule = [
        BaSchedule {
            trim_px: 50.0,
            loss_scale: 2.0,
        },
        BaSchedule {
            trim_px: 4.0,
            loss_scale: 1.0,
        },
    ];
    let out = run_with_schedule(&mut s, &schedule);
    for p in (0..s.points.len()).step_by(2) {
        assert!(s.points[p][0].is_finite(), "point {p} not re-admitted");
    }
    let max_res = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_res < 0.1, "max residual {max_res} after re-admission");
}

#[test]
fn untouched_images_pass_through() {
    let mut s = make_scene(6, 60);
    // Add an extra image and point never referenced by an observation.
    let spare_q = UnitQuaternion::from_scaled_axis(Vector3::new(0.7, -0.3, 0.2));
    let spare_t = Vector3::new(1.0, 2.0, 3.0);
    s.quats.push(spare_q);
    s.trans.push(spare_t);
    let n_pt_before = s.points.len();
    s.points.push([7.0, 8.0, 9.0]);
    // Single-round schedule: no retriangulation, so untouched points also
    // pass through (a retriangulation round would reset them to NaN).
    let schedule = [BaSchedule {
        trim_px: 4.0,
        loss_scale: 1.0,
    }];
    run_with_schedule(&mut s, &schedule);
    let last = s.quats.len() - 1;
    assert!(s.quats[last].angle_to(&spare_q) < 1e-15);
    assert!((s.trans[last] - spare_t).norm() < 1e-15);
    assert_eq!(s.points[n_pt_before], [7.0, 8.0, 9.0]);
}
