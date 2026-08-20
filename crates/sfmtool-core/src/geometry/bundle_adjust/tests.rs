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
#[derive(Clone)]
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
        None,
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        opt_f,
        false,
        false,
        schedule,
        60,
        2,
        12,
    )
}

/// [`run`] with a `point_at_infinity` mask and an explicit `min_obs`.
fn run_masked(
    s: &mut Scene,
    mask: &[bool],
    opt_f: bool,
    schedule: &[BaSchedule],
    min_obs: usize,
) -> BundleAdjustment {
    bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        Some(mask),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        opt_f,
        false,
        false,
        schedule,
        60,
        2,
        min_obs,
    )
}

/// [`run`] with a `protected` observation mask (default widening).
fn run_protected(
    s: &mut Scene,
    protected: &[bool],
    opt_f: bool,
    schedule: &[BaSchedule],
) -> BundleAdjustment {
    bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        None,
        Some(protected),
        DEFAULT_PROTECTED_LOSS_SCALE,
        opt_f,
        false,
        false,
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

// ── Points at infinity ──────────────────────────────────────────────────────

/// Append far-field direction tracks (world-frame unit directions, mostly
/// along world −Z like the scenes' viewing directions), observed by exact
/// projection `ray_to_pixel(R_i · d)` — plus deterministic pixel noise of
/// amplitude `noise` — in every camera where they land in-image and in
/// front. Returns the direction rows' point indices.
fn add_direction_tracks(s: &mut Scene, n_dir: usize, salt: u64, noise: f64) -> Vec<usize> {
    let mut ids = Vec::new();
    for j in 0..n_dir {
        let d = Vector3::new(0.35 * jitter(j, salt), 0.26 * jitter(j, salt + 1), -1.0).normalize();
        let p = s.points.len();
        s.points.push([d.x, d.y, d.z]);
        let mut n_obs = 0;
        for i in 0..s.quats.len() {
            let c = s.quats[i] * d;
            if c.z >= 0.0 {
                continue;
            }
            let Some((u, v)) = s.cam.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            if !(0.0..s.cam.width as f64).contains(&u) || !(0.0..s.cam.height as f64).contains(&v) {
                continue;
            }
            let k = s.uv.len();
            s.uv.push([
                u + noise * jitter(k, salt + 2),
                v + noise * jitter(k, salt + 3),
            ]);
            s.obs_img.push(i as u32);
            s.obs_pt.push(p as u32);
            n_obs += 1;
        }
        assert!(n_obs >= 2, "direction {j} observed only {n_obs} times");
        ids.push(p);
    }
    ids
}

fn dir_mask(s: &Scene, ids: &[usize]) -> Vec<bool> {
    let mut m = vec![false; s.points.len()];
    for &p in ids {
        m[p] = true;
    }
    m
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let av = Vector3::new(a[0], a[1], a[2]).normalize();
    let bv = Vector3::new(b[0], b[1], b[2]).normalize();
    av.dot(&bv).clamp(-1.0, 1.0).acos()
}

/// A low-parallax, near-planar scene with observation noise: cameras on a
/// shallow look-at arc (so the tight central cloud stays near the principal
/// point in every view) over a thin cloud whose depth relief is ~1% of the
/// viewing distance. The finite focal signal — quadratic in the small image
/// footprint — drowns in the pixel noise, so a released focal converges
/// wherever rotation bends and translation compensations carry it; the
/// full-FOV direction tracks keep a signal well above the noise.
fn make_lowpar_scene(n_img: usize, n_pt: usize, noise: f64) -> Scene {
    let cam = simple_pinhole(500.0);
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        let ang = 0.04 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(8.0 * ang.sin(), 0.1 * jitter(i, 11), 8.0 * ang.cos());
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
    }
    let mut points = Vec::new();
    for p in 0..n_pt {
        points.push([
            1.2 * jitter(p, 121),
            0.9 * jitter(p, 122),
            0.06 * jitter(p, 123),
        ]);
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
            let k = uv.len();
            uv.push([u + noise * jitter(k, 501), v + noise * jitter(k, 502)]);
            obs_img.push(i as u32);
            obs_pt.push(p as u32);
        }
    }
    assert!(
        uv.len() >= n_img * n_pt / 2,
        "degenerate low-parallax scene: only {} observations",
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

/// Guards the entry point's `None` → all-`false` normalization: an absent
/// mask must build a mask the staged loop treats identically to one the
/// caller supplied.
#[test]
fn all_false_and_absent_masks_match_bit_for_bit() {
    let build = || {
        let mut s = make_scene(6, 60);
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
        s
    };
    let mut absent = build();
    let out_absent = run(&mut absent, false, &DEFAULT_SCHEDULE);
    let mut masked = build();
    let mask = vec![false; masked.points.len()];
    let out_masked = run_masked(&mut masked, &mask, false, &DEFAULT_SCHEDULE, 12);
    assert_eq!(out_absent.focal.to_bits(), out_masked.focal.to_bits());
    for (a, b) in absent.quats.iter().zip(&masked.quats) {
        for c in 0..4 {
            assert_eq!(a.coords[c].to_bits(), b.coords[c].to_bits());
        }
    }
    for (a, b) in absent.trans.iter().zip(&masked.trans) {
        for c in 0..3 {
            assert_eq!(a[c].to_bits(), b[c].to_bits());
        }
    }
    for (a, b) in absent.points.iter().zip(&masked.points) {
        for c in 0..3 {
            assert_eq!(a[c].to_bits(), b[c].to_bits());
        }
    }
    for (a, b) in out_absent
        .residual_norms
        .iter()
        .zip(&out_masked.residual_norms)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn unobserved_direction_row_does_not_perturb_finite_results() {
    // The direction machinery must not leak into a finite solve. Appending a
    // marked direction row that no observation references exercises every
    // direction branch's guard — `normalized_dir` on input, the `cp_dir`
    // lookups, the tangent bases, the frozen-translation scan, the
    // finite-survivors `min_obs` count — while changing nothing the solve is
    // allowed to see. Every finite result must come back bit-identical.
    //
    // Since the finite-only kernel was folded into the staged loop this is
    // the load-bearing guard on that reduction: a direction branch that
    // stopped checking its flag would show up here.
    let mut plain = make_perturbed_scene();
    let out_plain = run(&mut plain, false, &DEFAULT_SCHEDULE);

    let mut extra = make_perturbed_scene();
    extra.points.push([0.0, 0.0, -1.0]);
    let mut mask = vec![false; extra.points.len()];
    *mask.last_mut().unwrap() = true;
    let out_extra = run_masked(&mut extra, &mask, false, &DEFAULT_SCHEDULE, 12);

    // Compare only the rows the plain scene has; the appended direction is
    // unobserved, so re-estimation legitimately leaves it NaN.
    assert_eq!(out_plain.focal.to_bits(), out_extra.focal.to_bits());
    for (a, b) in plain.quats.iter().zip(&extra.quats) {
        for c in 0..4 {
            assert_eq!(a.coords[c].to_bits(), b.coords[c].to_bits());
        }
    }
    for (a, b) in plain.trans.iter().zip(&extra.trans) {
        for c in 0..3 {
            assert_eq!(a[c].to_bits(), b[c].to_bits());
        }
    }
    for (p, (a, b)) in plain.points.iter().zip(&extra.points).enumerate() {
        for c in 0..3 {
            assert_eq!(a[c].to_bits(), b[c].to_bits(), "point {p} component {c}");
        }
    }
    for (a, b) in out_plain
        .residual_norms
        .iter()
        .zip(&out_extra.residual_norms)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn direction_observations_fixpoint() {
    let mut s = make_scene(6, 40);
    let ids = add_direction_tracks(&mut s, 12, 300, 0.0);
    let mask = dir_mask(&s, &ids);
    let d_true: Vec<[f64; 3]> = ids.iter().map(|&p| s.points[p]).collect();
    let q0 = s.quats.clone();
    let t0 = s.trans.clone();
    let out = run_masked(&mut s, &mask, false, &DEFAULT_SCHEDULE, 12);
    let max_res = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_res < 1e-5, "max residual {max_res}");
    for k in 0..s.quats.len() {
        assert!(s.quats[k].angle_to(&q0[k]) < 1e-6, "camera {k} rotated");
        assert!((s.trans[k] - t0[k]).norm() < 1e-5, "camera {k} moved");
    }
    for (j, &p) in ids.iter().enumerate() {
        let d = s.points[p];
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "direction {j} not unit: {norm}");
        let ang = angle_between(d, d_true[j]);
        assert!(ang < 1e-6, "direction {j} moved {ang} rad");
    }
}

#[test]
fn perturbed_rotations_recover_against_directions() {
    // Directions-only scene: every image observes directions exclusively, so
    // every translation is frozen and the `min_obs` floor (finite survivors
    // only) must be lifted for the solve to run at all.
    let mut s = make_scene(6, 4);
    s.points.clear();
    s.uv.clear();
    s.obs_img.clear();
    s.obs_pt.clear();
    let ids = add_direction_tracks(&mut s, 25, 400, 0.0);
    let mask = dir_mask(&s, &ids);
    let q_true = s.quats.clone();
    let t0 = s.trans.clone();
    for i in 0..s.quats.len() {
        let d = Vector3::new(
            0.015 * jitter(i, 421),
            0.015 * jitter(i, 422),
            0.015 * jitter(i, 423),
        );
        s.quats[i] = UnitQuaternion::from_scaled_axis(d) * s.quats[i];
    }
    let schedule = [BaSchedule {
        trim_px: 50.0,
        loss_scale: 1.0,
    }];
    let out = run_masked(&mut s, &mask, false, &schedule, 0);
    // Translations are frozen: bit-identical pass-through.
    for (i, (t, t_orig)) in s.trans.iter().zip(&t0).enumerate() {
        for c in 0..3 {
            assert_eq!(
                t[c].to_bits(),
                t_orig[c].to_bits(),
                "image {i} translation moved"
            );
        }
    }
    // Sub-pixel reprojection against the direction set.
    let max_res = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_res < 0.01, "max residual {max_res}");
    // Rotations recover ground truth up to the free global rotation gauge:
    // compare the gauge-invariant relative rotations R_i · R_0⁻¹.
    for i in 1..s.quats.len() {
        let rel_est = s.quats[i] * s.quats[0].inverse();
        let rel_true = q_true[i] * q_true[0].inverse();
        let ang = rel_est.angle_to(&rel_true);
        assert!(ang < 1e-4, "camera {i} relative rotation err {ang} rad");
    }
}

#[test]
fn all_direction_image_translation_frozen_rotation_refines() {
    let mut s = make_scene(6, 40);
    // An extra image on the arc that observes only directions.
    let extra_center = Vector3::new(8.0 * 0.5f64.sin(), 0.2, 8.0 * 0.5f64.cos());
    let extra_r = UnitQuaternion::face_towards(&extra_center, &Vector3::y()).inverse();
    s.quats.push(extra_r);
    s.trans.push(-(extra_r * extra_center));
    let extra = s.quats.len() - 1;
    let ids = add_direction_tracks(&mut s, 15, 500, 0.0);
    assert!(
        s.obs_img.iter().any(|&i| i as usize == extra),
        "extra image observes no direction"
    );
    let mask = dir_mask(&s, &ids);
    let q_true = s.quats.clone();
    // Perturb the extra image's pose; the rotation must refine back while the
    // translation (unconstrained by directions) passes through bit-identical.
    s.quats[extra] =
        UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.008, 0.006)) * s.quats[extra];
    s.trans[extra] += Vector3::new(0.3, -0.2, 0.1);
    let t_frozen = s.trans[extra];
    let init_ang =
        (s.quats[extra] * s.quats[0].inverse()).angle_to(&(q_true[extra] * q_true[0].inverse()));
    // Single round: no re-estimation, so the directions stay anchored by the
    // six true-pose images while the extra image's rotation is pulled back.
    let schedule = [BaSchedule {
        trim_px: 50.0,
        loss_scale: 1.0,
    }];
    let out = run_masked(&mut s, &mask, false, &schedule, 12);
    for c in 0..3 {
        assert_eq!(
            s.trans[extra][c].to_bits(),
            t_frozen[c].to_bits(),
            "frozen translation moved"
        );
    }
    // The solve's global rotation gauge is free (a world rotation about the
    // origin moves every quaternion while leaving all translations and
    // residuals unchanged), so compare the gauge-invariant relative rotation
    // against image 0.
    let ang =
        (s.quats[extra] * s.quats[0].inverse()).angle_to(&(q_true[extra] * q_true[0].inverse()));
    assert!(
        ang < 1e-6 && ang < init_ang,
        "extra relative rotation err {ang} rad (started {init_ang})"
    );
    // Its direction observations end fit.
    for k in 0..s.uv.len() {
        if s.obs_img[k] as usize == extra {
            assert!(out.residual_norms[k] < 0.01, "extra obs {k} residual");
        }
    }
}

#[test]
fn nan_direction_reborn_as_mean_back_rotated_ray() {
    let mut s = make_scene(5, 40);
    let ids = add_direction_tracks(&mut s, 8, 600, 0.0);
    let victim = ids[3];
    let d_true = s.points[victim];
    let n_victim_obs = s.obs_pt.iter().filter(|&&p| p as usize == victim).count();
    assert!(n_victim_obs >= 2);
    s.points[victim] = [f64::NAN; 3];
    let mask = dir_mask(&s, &ids);
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
    let out = run_masked(&mut s, &mask, false, &schedule, 12);
    let d = s.points[victim];
    assert!(d[0].is_finite(), "direction not re-admitted");
    let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-9,
        "reborn direction not unit: {norm}"
    );
    // Reborn as the normalized mean of the back-rotated observation rays at
    // the (fixpoint) rotations — which is the true direction here.
    let mut sum = Vector3::zeros();
    for k in 0..s.uv.len() {
        if s.obs_pt[k] as usize != victim {
            continue;
        }
        let i = s.obs_img[k] as usize;
        let ray = s.cam.pixel_to_ray(s.uv[k][0], s.uv[k][1]);
        sum += s.quats[i].inverse() * Vector3::new(ray[0], ray[1], ray[2]);
    }
    let expected = (sum / n_victim_obs as f64).normalize();
    assert!(
        angle_between(d, [expected.x, expected.y, expected.z]) < 1e-6,
        "reborn direction differs from the mean back-rotated ray"
    );
    assert!(
        angle_between(d, d_true) < 1e-6,
        "reborn direction off truth"
    );
    // Its observations participate thereafter: they end with small residuals.
    for k in 0..s.uv.len() {
        if s.obs_pt[k] as usize == victim {
            assert!(out.residual_norms[k] < 0.1, "victim obs {k} residual");
        }
    }
}

#[test]
fn directions_lock_rotations_for_focal_release() {
    // The load-bearing coupling test: on a noisy low-parallax near-planar
    // scene a wrong focal converges wrong under opt_f (rotation bends and
    // translation compensations absorb it within the noise), and far-field
    // direction tracks — which pin rotations without touching the
    // translation side — recover it.
    let schedule = [BaSchedule {
        trim_px: 300.0,
        loss_scale: 2.0,
    }];
    // Precondition: finite-only opt_f from f = 650 lands far from 500.
    let mut plain = make_lowpar_scene(6, 80, 0.3);
    plain.cam = simple_pinhole(650.0);
    let out_plain = bundle_adjust(
        &plain.cam,
        &mut plain.quats,
        &mut plain.trans,
        &mut plain.points,
        &plain.uv,
        &plain.obs_img,
        &plain.obs_pt,
        None,
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        true,
        false,
        false,
        &schedule,
        150,
        2,
        12,
    );
    assert!(
        (out_plain.focal - 500.0).abs() > 25.0,
        "precondition failed: finite-only opt_f recovered f = {} (true 500)",
        out_plain.focal
    );
    // With far-field direction tracks (same pixel noise) the same solve
    // recovers the focal to within the noise floor.
    let mut s = make_lowpar_scene(6, 80, 0.3);
    let ids = add_direction_tracks(&mut s, 20, 700, 0.3);
    let mask = dir_mask(&s, &ids);
    s.cam = simple_pinhole(650.0);
    let out = bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        Some(&mask),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        true,
        false,
        false,
        &schedule,
        150,
        2,
        12,
    );
    assert!(
        (out.focal - 500.0).abs() < 5.0,
        "focal {} with directions (want ~500; finite-only gave {})",
        out.focal,
        out_plain.focal
    );
}

// ── Protected observations ──────────────────────────────────────────────────

/// The perturbed scene of `recovers_from_perturbed_state`, rebuilt
/// deterministically (for bit-for-bit comparisons across runs).
fn make_perturbed_scene() -> Scene {
    let mut s = make_scene(6, 60);
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
    s
}

fn assert_bitwise_equal(a: &Scene, oa: &BundleAdjustment, b: &Scene, ob: &BundleAdjustment) {
    assert_eq!(oa.focal.to_bits(), ob.focal.to_bits());
    for (qa, qb) in a.quats.iter().zip(&b.quats) {
        for c in 0..4 {
            assert_eq!(qa.coords[c].to_bits(), qb.coords[c].to_bits());
        }
    }
    for (ta, tb) in a.trans.iter().zip(&b.trans) {
        for c in 0..3 {
            assert_eq!(ta[c].to_bits(), tb[c].to_bits());
        }
    }
    for (pa, pb) in a.points.iter().zip(&b.points) {
        for c in 0..3 {
            assert_eq!(pa[c].to_bits(), pb[c].to_bits());
        }
    }
    for (ra, rb) in oa.residual_norms.iter().zip(&ob.residual_norms) {
        assert_eq!(ra.to_bits(), rb.to_bits());
    }
}

#[test]
fn protected_absent_and_all_false_match_bit_for_bit() {
    let mut absent = make_perturbed_scene();
    let out_absent = run(&mut absent, false, &DEFAULT_SCHEDULE);
    let mut masked = make_perturbed_scene();
    let prot = vec![false; masked.uv.len()];
    let out_masked = run_protected(&mut masked, &prot, false, &DEFAULT_SCHEDULE);
    assert_bitwise_equal(&absent, &out_absent, &masked, &out_masked);
}

#[test]
fn protected_all_false_with_infinity_mask_matches_bit_for_bit() {
    // The mixed (points-at-infinity) path with an all-false protection mask
    // must reproduce the protection-free mixed path bit for bit.
    let build = || {
        let mut s = make_perturbed_scene();
        let ids = add_direction_tracks(&mut s, 10, 900, 0.0);
        let mask = dir_mask(&s, &ids);
        (s, mask)
    };
    let (mut plain, mask_a) = build();
    let out_plain = run_masked(&mut plain, &mask_a, false, &DEFAULT_SCHEDULE, 12);
    let (mut prot, mask_b) = build();
    let out_prot = bundle_adjust(
        &prot.cam,
        &mut prot.quats,
        &mut prot.trans,
        &mut prot.points,
        &prot.uv,
        &prot.obs_img,
        &prot.obs_pt,
        Some(&mask_b),
        Some(&vec![false; prot.uv.len()]),
        DEFAULT_PROTECTED_LOSS_SCALE,
        false,
        false,
        false,
        &DEFAULT_SCHEDULE,
        60,
        2,
        12,
    );
    assert_bitwise_equal(&plain, &out_plain, &prot, &out_prot);
}

#[test]
fn protected_observations_survive_trim_unprotected_are_dropped() {
    // Two tracks corrupted identically with large mutually inconsistent
    // offsets (zero-mean: a consistent component would simply be absorbed by
    // the track's free point); the protected one stays in the solve set (its
    // point moves under the LM), the unprotected one is fully trimmed (its
    // point passes through bit-identical under a single-round schedule).
    let corrupt = corrupt_track_incoherent;
    let mut s = make_scene(6, 60);
    let victim_a = s.obs_pt[0] as usize;
    let victim_b = s
        .obs_pt
        .iter()
        .map(|&p| p as usize)
        .find(|&p| p != victim_a)
        .unwrap();
    corrupt(&mut s, victim_a);
    corrupt(&mut s, victim_b);
    let a_before = s.points[victim_a];
    let b_before = s.points[victim_b];
    let prot: Vec<bool> = s.obs_pt.iter().map(|&p| p as usize == victim_a).collect();
    let schedule = [BaSchedule {
        trim_px: 25.0,
        loss_scale: 1.0,
    }];
    let out = run_protected(&mut s, &prot, false, &schedule);
    assert_eq!(
        s.points[victim_b], b_before,
        "unprotected corrupted track must be trimmed (point untouched)"
    );
    assert_ne!(
        s.points[victim_a], a_before,
        "protected corrupted track must stay in the solve (point moves)"
    );
    // Bounded influence, asserted in residual space (parameters can wander
    // the scene's soft modes — the free similarity gauge and the arc's
    // bas-relief-like bend — at near-zero cost, so they are not the
    // measure): the junk stays far from fitted (pulled toward, never
    // dominated) and the clean majority still fits to ~a pixel. The pull is
    // real — soft-L1 saturates the clean data's resistance too, so a bend
    // that trades a few clean pixels for hundreds of junk pixels is by
    // design — but a dominating junk track would drive its own residuals
    // toward zero and the clean median far past a pixel.
    let mut clean: Vec<f64> = (0..s.uv.len())
        .filter(|&k| {
            let p = s.obs_pt[k] as usize;
            p != victim_a && p != victim_b
        })
        .map(|k| out.residual_norms[k])
        .collect();
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let clean_med = clean[clean.len() / 2];
    assert!(clean_med < 1.0, "clean median residual {clean_med}");
    let junk_max = (0..s.uv.len())
        .filter(|&k| s.obs_pt[k] as usize == victim_a)
        .map(|k| out.residual_norms[k])
        .fold(0.0f64, f64::max);
    assert!(
        junk_max > 100.0,
        "protected junk got fitted (max residual {junk_max}) — it must not dominate"
    );
}

/// Corrupt every observation of one track with large, mutually inconsistent,
/// zero-mean pixel offsets (±120–200 px, no common component the track's
/// free point could absorb).
fn corrupt_track_incoherent(s: &mut Scene, victim: usize) {
    for k in 0..s.uv.len() {
        if s.obs_pt[k] as usize == victim {
            let sx = if jitter(k, 85) > 0.0 { 1.0 } else { -1.0 };
            let sy = if jitter(k, 86) > 0.0 { 1.0 } else { -1.0 };
            s.uv[k][0] += sx * (160.0 + 40.0 * jitter(k, 81));
            s.uv[k][1] += sy * (160.0 + 40.0 * jitter(k, 83));
        }
    }
}

#[test]
fn protected_survives_every_round_of_a_multi_round_schedule() {
    // Under the full default schedule (with retriangulation) an unprotected
    // inconsistently-corrupted track is trimmed every round, so protecting
    // it is the only difference between the two runs: if the protected
    // observations were ever dropped, the runs would be bit-identical.
    let corrupt = corrupt_track_incoherent;
    let build = || {
        let mut s = make_perturbed_scene();
        let victim = s.obs_pt[0] as usize;
        corrupt(&mut s, victim);
        (s, victim)
    };
    let (mut plain, victim) = build();
    let out_plain = run(&mut plain, false, &DEFAULT_SCHEDULE);
    let (mut prot_scene, _) = build();
    let prot: Vec<bool> = prot_scene
        .obs_pt
        .iter()
        .map(|&p| p as usize == victim)
        .collect();
    let out_prot = run_protected(&mut prot_scene, &prot, false, &DEFAULT_SCHEDULE);
    // The corrupted observations are outliers in the unprotected run …
    let victim_obs: Vec<usize> = (0..plain.uv.len())
        .filter(|&k| plain.obs_pt[k] as usize == victim)
        .collect();
    for &k in &victim_obs {
        assert!(
            out_plain.residual_norms[k] > 12.0,
            "obs {k} unexpectedly consistent ({}) — corruption too tame",
            out_plain.residual_norms[k]
        );
    }
    // … so protection is the only difference; the runs must diverge.
    let diverged = plain
        .points
        .iter()
        .zip(&prot_scene.points)
        .any(|(a, b)| a != b)
        || plain
            .quats
            .iter()
            .zip(&prot_scene.quats)
            .any(|(a, b)| a.coords != b.coords);
    assert!(
        diverged,
        "protected observations left no trace on the solve (dropped somewhere?)"
    );
    // And the protected point ends pulled toward its (inconsistent)
    // observations: strictly smaller residuals than the unprotected run's.
    let sum =
        |out: &BundleAdjustment| -> f64 { victim_obs.iter().map(|&k| out.residual_norms[k]).sum() };
    assert!(
        sum(&out_prot) < sum(&out_plain),
        "protected track not pulled ({} vs {})",
        sum(&out_prot),
        sum(&out_plain)
    );
}

#[test]
fn protected_counts_toward_min_track_survival() {
    // A track trimmed down to one clean survivor leaves the solve entirely
    // (min_track = 2) — unless its corrupted observations are protected, in
    // which case they count as survivors and the track stays in.
    let build = || {
        let mut s = make_scene(4, 30);
        for (p, x) in s.points.iter_mut().enumerate() {
            for (c, xc) in x.iter_mut().enumerate() {
                *xc += 0.03 * jitter(p, 60 + c as u64);
            }
        }
        let victim = s.obs_pt[0] as usize;
        let mut first = true;
        for k in 0..s.uv.len() {
            if s.obs_pt[k] as usize == victim {
                if first {
                    first = false;
                    continue;
                }
                s.uv[k][0] += 500.0 + 100.0 * jitter(k, 87);
            }
        }
        (s, victim)
    };
    let schedule = [BaSchedule {
        trim_px: 25.0,
        loss_scale: 1.0,
    }];
    // Unprotected: the whole track leaves the solve, point bit-identical.
    let (mut plain, victim) = build();
    let before = plain.points[victim];
    run(&mut plain, false, &schedule);
    assert_eq!(plain.points[victim], before, "starved track not dropped");
    // Protected corrupted observations count toward min_track: the track
    // stays in the solve and its point moves.
    let (mut prot_scene, _) = build();
    let mut first = true;
    let prot: Vec<bool> = (0..prot_scene.uv.len())
        .map(|k| {
            if prot_scene.obs_pt[k] as usize == victim {
                if first {
                    first = false;
                    return false; // the clean survivor needs no protection
                }
                return true;
            }
            false
        })
        .collect();
    run_protected(&mut prot_scene, &prot, false, &schedule);
    assert_ne!(
        prot_scene.points[victim], before,
        "protected survivors must keep the track in the solve"
    );
}

#[test]
fn protected_direction_observation_composes_with_infinity_mask() {
    // A protected corrupted direction observation stays in the solve
    // (composability smoke: both masks apply, no special casing).
    let build = || {
        let mut s = make_scene(6, 40);
        let ids = add_direction_tracks(&mut s, 8, 950, 0.0);
        let mask = dir_mask(&s, &ids);
        // Corrupt one direction observation far past every trim gate.
        let victim = ids[2];
        let k = (0..s.uv.len())
            .find(|&k| s.obs_pt[k] as usize == victim)
            .unwrap();
        s.uv[k][0] += 120.0;
        s.uv[k][1] -= 90.0;
        (s, mask, victim, k)
    };
    let schedule = [BaSchedule {
        trim_px: 25.0,
        loss_scale: 1.0,
    }];
    // Unprotected: the corrupted observation is trimmed; the direction stays
    // at the truth its clean observations pin.
    let (mut plain, mask, victim, _k) = build();
    let d_true = plain.points[victim];
    run_masked(&mut plain, &mask, false, &schedule, 12);
    assert!(
        angle_between(plain.points[victim], d_true) < 1e-9,
        "unprotected corrupted direction obs must be trimmed"
    );
    // Protected: it stays in the solve and pulls the direction off truth.
    let (mut s, mask, victim, k) = build();
    let mut prot = vec![false; s.uv.len()];
    prot[k] = true;
    let out = bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        Some(&mask),
        Some(&prot),
        DEFAULT_PROTECTED_LOSS_SCALE,
        false,
        false,
        false,
        &schedule,
        60,
        2,
        12,
    );
    assert!(
        angle_between(s.points[victim], d_true) > 1e-6,
        "protected corrupted direction obs left no trace"
    );
    let norm = {
        let d = s.points[victim];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    assert!((norm - 1.0).abs() < 1e-9, "direction not unit: {norm}");
    assert!(out.residual_norms[k].is_finite());
}

/// A drifted-gauge scene: two internally rigid camera fragments (halves of
/// the standard arc) whose local tracks live entirely inside one half, tied
/// together only by a small minority (~2% of observations) of long-range
/// tracks observed by every camera. The second half — cameras and its local
/// points together — is drifted by a similarity (rotation + scale), which
/// leaves every local observation exactly consistent (a similarity applied
/// to a whole fragment is pure gauge for its own tracks) while the
/// long-range observations in the drifted half carry large corrective
/// residuals toward the true relative gauge. This is the drifted-merge
/// regime the protection exists for: the local majority cannot constrain
/// the fragments' relative similarity at all, and the only corrective
/// signal is the long-range minority whose residuals look like outliers.
struct DriftScene {
    s: Scene,
    true_centers: Vec<Vector3<f64>>,
    long_obs: Vec<usize>,
}

fn make_drifted_scene(theta: f64, sigma: f64) -> DriftScene {
    let n_img = 8usize;
    let n_local = 250usize;
    // Three non-collinear shared points: the minimum that makes the
    // fragments' relative similarity rigid (two leave a residual 1-DOF
    // family that fits every observation without recovering the gauge).
    let n_long = 3usize;
    let cam = simple_pinhole(500.0);
    // Ground-truth poses: the standard shallow arc; halves 0..4 and 4..8.
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    let mut true_centers = Vec::new();
    for i in 0..n_img {
        let ang = 0.15 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(8.0 * ang.sin(), 0.5 * jitter(i, 11), 8.0 * ang.cos());
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
        true_centers.push(center);
    }
    // Local windows of 3 consecutive cameras, never crossing the halves:
    // (0,1,2), (1,2,3) in the first half; (4,5,6), (5,6,7) in the second.
    let windows = [[0usize, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]];
    let mut points = Vec::new();
    for p in 0..n_local {
        points.push([2.0 * jitter(p, 1), 2.0 * jitter(p, 2), 1.5 * jitter(p, 3)]);
    }
    // The long-range points are placed deliberately: well off the drift's
    // rotation axis (Y) so the rotation moves them tangentially — strong
    // pixel signal in every camera (a point near the axis, or one displaced
    // along the viewing ray, drifts almost invisibly).
    points.push([2.2, 0.3, 1.2]);
    points.push([-1.8, -0.4, 1.4]);
    points.push([0.6, 1.4, -1.6]);
    assert_eq!(points.len(), n_local + n_long);
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    let mut long_obs = Vec::new();
    for (p, x) in points.iter().enumerate() {
        let cams: Vec<usize> = if p < n_local {
            windows[p % windows.len()].to_vec()
        } else {
            (0..n_img).collect()
        };
        for i in cams {
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
            if p >= n_local {
                long_obs.push(uv.len());
            }
            uv.push([u, v]);
            obs_img.push(i as u32);
            obs_pt.push(p as u32);
        }
    }
    let n_obs = uv.len();
    assert!(
        long_obs.len() * 100 >= n_obs && long_obs.len() * 25 <= n_obs,
        "long-range fraction off: {} of {n_obs}",
        long_obs.len()
    );
    // Drift the second half — cameras and its local points — by one
    // similarity: X' = s·Rd·X + b, R' = R·Rdᵀ, t' = s·t − R'·b (the
    // camera-frame point scales, so every intra-half reprojection is
    // untouched). The vertical offset `b` guarantees a strong pixel residual
    // in every camera (rotation about Y and scale about the origin can both
    // move a particular point nearly along some camera's viewing ray).
    // Long-range points stay at truth: their first-half observations remain
    // consistent, their second-half observations carry the corrective
    // residual.
    let rd = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), theta);
    let s_drift = 1.0 + sigma;
    let b = Vector3::new(0.0, 1.5, 0.0);
    for i in n_img / 2..n_img {
        let r_new = quats[i] * rd.inverse();
        trans[i] = s_drift * trans[i] - r_new * b;
        quats[i] = r_new;
    }
    for (p, x) in points.iter_mut().enumerate() {
        let in_second_half = p < n_local && windows[p % windows.len()][0] >= n_img / 2;
        if in_second_half {
            let w = s_drift * (rd * Vector3::new(x[0], x[1], x[2])) + b;
            *x = [w.x, w.y, w.z];
        }
    }
    DriftScene {
        s: Scene {
            cam,
            quats,
            trans,
            points,
            uv,
            obs_img,
            obs_pt,
        },
        true_centers,
        long_obs,
    }
}

/// RMS distance between the estimated camera centers and truth after a
/// least-squares similarity alignment (the gauge is free; only the residual
/// deformation counts).
fn aligned_center_rms(s: &Scene, true_centers: &[Vector3<f64>]) -> f64 {
    use crate::analysis::alignment::{estimate_alignment, AlignmentParams};
    let est: Vec<f64> = s
        .quats
        .iter()
        .zip(&s.trans)
        .flat_map(|(q, t)| {
            let c = -(q.inverse() * t);
            [c.x, c.y, c.z]
        })
        .collect();
    let tgt: Vec<f64> = true_centers.iter().flat_map(|c| [c.x, c.y, c.z]).collect();
    let n = true_centers.len();
    let xf = estimate_alignment(&est, &tgt, n, AlignmentParams::default()).unwrap();
    let r = xf.rotation.to_rotation_matrix();
    let mut sum = 0.0;
    for (i, c_true) in true_centers.iter().enumerate() {
        let src = Vector3::new(est[3 * i], est[3 * i + 1], est[3 * i + 2]);
        let mapped = xf.scale * (r * src) + xf.translation;
        sum += (mapped - c_true).norm_squared();
    }
    (sum / n as f64).sqrt()
}

#[test]
fn protected_long_range_observations_correct_a_drifted_gauge() {
    // The load-bearing test: a reconstruction drifted into a wrong
    // rotation/scale gauge whose local observations are self-consistent.
    // The ~2% minority of long-range observations carries the corrective
    // signal — but their large residuals are exactly what the trim gates
    // classify as outliers, so the unprotected BA silently removes the
    // correction and re-converges inside the drift. Protecting them keeps
    // the pull (at bounded influence) and moves the solution measurably
    // toward the true gauge.
    let run_ba = |s: &mut Scene, prot: Option<&[bool]>| -> BundleAdjustment {
        bundle_adjust(
            &s.cam,
            &mut s.quats,
            &mut s.trans,
            &mut s.points,
            &s.uv,
            &s.obs_img,
            &s.obs_pt,
            None,
            prot,
            DEFAULT_PROTECTED_LOSS_SCALE,
            false,
            false,
            false,
            &DEFAULT_SCHEDULE,
            150,
            2,
            12,
        )
    };
    let drift = || make_drifted_scene(0.8, 0.5);

    // Precondition: the drift is real, and every drifted-half long-range
    // observation starts beyond the first trim gate (50 px) — the corrective
    // signal is exactly what the unprotected trim removes.
    let init = drift();
    let rms_init = aligned_center_rms(&init.s, &init.true_centers);
    assert!(rms_init > 0.2, "drift too tame: initial rms {rms_init}");
    let (init_norms, _) = residual_norms_depths(
        &init.s.cam,
        &init.s.quats,
        &init.s.trans,
        &init.s.points,
        &vec![false; init.s.points.len()],
        &init.s.uv,
        &init.s.obs_img,
        &init.s.obs_pt,
    );
    for &k in &init.long_obs {
        if init.s.obs_img[k] as usize >= 4 {
            assert!(
                init_norms[k] > DEFAULT_SCHEDULE[0].trim_px,
                "drifted long-range obs {k} starts at {} px — under the first trim gate",
                init_norms[k]
            );
        }
    }

    // Unprotected: the BA trims the long-range observations and re-converges
    // inside the drift gauge.
    let mut plain = drift();
    let out_plain = run_ba(&mut plain.s, None);
    let rms_plain = aligned_center_rms(&plain.s, &plain.true_centers);
    let plain_long_trimmed = plain
        .long_obs
        .iter()
        .filter(|&&k| out_plain.residual_norms[k] > DEFAULT_SCHEDULE[2].trim_px)
        .count();
    assert!(
        plain_long_trimmed * 3 > plain.long_obs.len(),
        "long-range obs unexpectedly consistent without protection \
         ({plain_long_trimmed} of {} beyond the final trim)",
        plain.long_obs.len()
    );
    assert!(
        rms_plain > 0.5 * rms_init,
        "unprotected BA left the drift gauge on its own (rms {rms_plain} vs init {rms_init})"
    );

    // Protected: the same solve moves measurably toward the true gauge.
    let mut prot_scene = drift();
    let prot: Vec<bool> = {
        let mut m = vec![false; prot_scene.s.uv.len()];
        for &k in &prot_scene.long_obs {
            m[k] = true;
        }
        m
    };
    let out_prot = run_ba(&mut prot_scene.s, Some(&prot));
    let rms_prot = aligned_center_rms(&prot_scene.s, &prot_scene.true_centers);
    // Margin, not bitwise: the protected run must land an order of magnitude
    // closer to the true gauge (measured: ~1e-9 vs an unprotected fixpoint
    // at the full drift, ~0.87).
    assert!(
        rms_prot < 0.1 * rms_plain && rms_prot < 0.05,
        "protection did not correct the gauge: rms {rms_prot} (unprotected {rms_plain}, \
         init {rms_init})"
    );
    // The retained long-range observations end far better fit than the
    // trimmed ones did.
    let mean_long = |out: &BundleAdjustment, obs: &[usize]| -> f64 {
        obs.iter().map(|&k| out.residual_norms[k]).sum::<f64>() / obs.len() as f64
    };
    let long_prot = mean_long(&out_prot, &prot_scene.long_obs);
    let long_plain = mean_long(&out_plain, &plain.long_obs);
    assert!(
        long_prot < 0.5 * long_plain,
        "protected long-range residuals not reduced ({long_prot} vs {long_plain})"
    );
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

// ── Model-genericity: fixed equidistant-fisheye intrinsics past 90° ────────
//
// Phase 1 of the fisheye-seed campaign (`scripts/notes-fisheye-seed.md`) runs
// this kernel with FIXED `SimpleRadialFisheye { k1 = 0 }` intrinsics. Two
// distinct pinhole assumptions could bite there, and the tests below pin
// both:
//
//   * the projection/Jacobian path — analytic for this model (it shares the
//     equidistant closed form), with `RadialFisheye` standing in below for
//     the central-difference fallback the rest of the family still takes;
//   * the inter-round in-front gate. Until the model-aware measure landed it
//     compared the canonical depth `−z_cam` against the `1e-3·f` floor, which
//     DISCARDS every observation at `θ ≥ 90°` — the whole periphery of a
//     >180° capture, i.e. exactly the part that carries the model
//     information.

/// The Phase-1 fisheye-seed camera: equidistant `θ = r/f`, `k1 = 0`.
fn equidistant_seed(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
        },
        width: 480,
        height: 480,
    }
}

/// A wide-FOV multi-view scene: cameras on an arc around a point cloud that
/// SURROUNDS them, so a large share of every image's observations sit past
/// 90° off-axis. Returns the scene and the number of `z_cam > 0`
/// observations.
fn make_fisheye_scene(n_img: usize, n_pt: usize) -> (Scene, usize) {
    make_fisheye_scene_for(equidistant_seed(130.0), n_img, n_pt)
}

/// The same `θ = r/f` map as a `RadialFisheye` with both coefficients zero:
/// identical projections, but `supports_pixel_jacobian()` is false, so the
/// kernel central-differences `ray_to_pixel` instead of linearizing
/// analytically. This is the fallback path the multi-coefficient fisheye
/// family takes, carried by a camera whose exact answer is known.
fn equidistant_legacy(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::RadialFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
        },
        width: 480,
        height: 480,
    }
}

/// The same map as a native `EquidistantFisheye`: identical projections, and
/// like `SimpleRadialFisheye` it carries an analytic pixel Jacobian.
fn equidistant_native(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
    }
}

/// [`make_fisheye_scene`] under an arbitrary equidistant camera.
fn make_fisheye_scene_for(cam: CameraIntrinsics, n_img: usize, n_pt: usize) -> (Scene, usize) {
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        let ang = 0.25 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(1.2 * ang.sin(), 0.3 * jitter(i, 11), 1.2 * ang.cos());
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
    }
    // Points on a shell of radius ~6 about the origin: the rig sits inside
    // it, so each camera images a hemisphere-plus of them.
    let mut points = Vec::new();
    for p in 0..n_pt {
        let theta = std::f64::consts::PI * (0.15 + 0.7 * (p as f64) / (n_pt as f64 - 1.0));
        let phi = 2.399_963 * p as f64;
        let rad = 6.0 + 1.5 * jitter(p, 3);
        points.push([
            rad * theta.sin() * phi.cos(),
            rad * theta.cos(),
            rad * theta.sin() * phi.sin(),
        ]);
    }
    // Observations first, then keep only the points at least two images see
    // (a one-view track is re-estimated to NaN by the staged loop, which has
    // nothing to do with the camera model).
    let mut per_pt: Vec<Vec<(u32, [f64; 2], bool)>> = vec![Vec::new(); points.len()];
    for (p, x) in points.iter().enumerate() {
        for i in 0..n_img {
            let c = quats[i] * Vector3::new(x[0], x[1], x[2]) + trans[i];
            let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            // Keep the image circle out to θ = 105° (a 210° lens).
            if (u - 240.0).hypot(v - 240.0) > 130.0 * 105.0_f64.to_radians() {
                continue;
            }
            per_pt[p].push((i as u32, [u, v], c.z > 0.0));
        }
    }
    let mut kept_points = Vec::new();
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    let mut n_behind = 0usize;
    for (p, obs) in per_pt.iter().enumerate() {
        if obs.len() < 2 {
            continue;
        }
        let cp = kept_points.len() as u32;
        kept_points.push(points[p]);
        for &(i, px, behind) in obs {
            n_behind += behind as usize;
            uv.push(px);
            obs_img.push(i);
            obs_pt.push(cp);
        }
    }
    (
        Scene {
            cam,
            quats,
            trans,
            points: kept_points,
            uv,
            obs_img,
            obs_pt,
        },
        n_behind,
    )
}

#[test]
fn fixed_fisheye_intrinsics_keep_observations_past_ninety_degrees() {
    let (s, n_behind) = make_fisheye_scene(6, 90);
    assert!(
        n_behind >= 50,
        "scene is not wide enough: {n_behind}/{} past 90°",
        s.uv.len()
    );
    // The gate the staged loop applies, evaluated at the ground-truth state.
    let (norms, depths) = residual_norms_depths(
        &s.cam,
        &s.quats,
        &s.trans,
        &s.points,
        &vec![false; s.points.len()],
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
    );
    let f = s.cam.focal_lengths().0;
    let kept = (0..s.uv.len())
        .filter(|&k| norms[k] < 50.0 && depths[k] > 1e-3 * f)
        .count();
    assert_eq!(
        kept,
        s.uv.len(),
        "the in-front gate dropped {} of {} exact fisheye observations",
        s.uv.len() - kept,
        s.uv.len()
    );
    // And the counterfactual that motivates the model-aware measure: the
    // perspective family's `−z_cam` would have thrown the whole θ ≥ 90° band
    // away at the very first trim.
    let mut dropped_by_z = 0usize;
    for k in 0..s.uv.len() {
        let x = s.points[s.obs_pt[k] as usize];
        let i = s.obs_img[k] as usize;
        let c = s.quats[i] * Vector3::new(x[0], x[1], x[2]) + s.trans[i];
        if -c.z <= 1e-3 * f {
            dropped_by_z += 1;
        }
    }
    assert!(
        dropped_by_z >= n_behind,
        "the `−z_cam` measure must drop at least the backward-of-image-plane \
         band ({dropped_by_z} < {n_behind})"
    );
}

#[test]
fn fixed_fisheye_intrinsics_converge_from_a_perturbed_state() {
    let (mut s, _) = make_fisheye_scene(6, 90);
    let truth_pts = s.points.clone();
    let truth_q = s.quats.clone();
    for (i, q) in s.quats.iter_mut().enumerate() {
        *q = UnitQuaternion::from_scaled_axis(Vector3::new(
            0.01 * jitter(i, 21),
            0.01 * jitter(i, 22),
            0.01 * jitter(i, 23),
        )) * *q;
    }
    for (p, x) in s.points.iter_mut().enumerate() {
        for (c, v) in x.iter_mut().enumerate() {
            *v += 0.05 * jitter(p * 3 + c, 31);
        }
    }
    let out = run(&mut s, false, &DEFAULT_SCHEDULE);
    assert_eq!(out.focal, 130.0, "fixed intrinsics must not move");
    let finite = out.residual_norms.iter().filter(|r| r.is_finite()).count();
    assert_eq!(
        finite,
        out.residual_norms.len(),
        "some observations ended outside the model domain"
    );
    let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(worst < 0.5, "worst reprojection {worst} px after BA");
    // Structure and rotations came back to the planted values (the gauge is
    // fixed by the unmoved translations of the perturbation-free init).
    for (p, x) in s.points.iter().enumerate() {
        let d = (0..3)
            .map(|c| (x[c] - truth_pts[p][c]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(d < 0.05, "point {p} off by {d}");
    }
    for (i, q) in s.quats.iter().enumerate() {
        assert!(q.angle_to(&truth_q[i]) < 5e-3, "image {i} rotation");
    }
}

// ── Analytic vs central-difference Jacobian on the same fisheye scene ──────
//
// `EquidistantFisheye` and `RadialFisheye { k1 = k2 = 0 }` parameterize the
// same `θ = r/f` map, so the two arms differ ONLY in how each Gauss–Newton
// step is linearized: the first from the closed-form pixel Jacobian, the
// second from a central difference of `ray_to_pixel` (four extra projections
// per observation per linearization).

/// The two arms must converge to the same reconstruction from the same
/// perturbed start.
///
/// Poses agree to `1e-9` rad / `1e-9` in translation and points to `1e-8`
/// world units — three or more orders inside the `5e-3` rad / `0.05` accuracy
/// each arm is separately asserted to reach against the planted truth, and
/// far below the scene's own sub-pixel residual floor. The residual is a
/// difference of linearizations, not of models: the projections agree to
/// `1e-12` px before either solve starts, asserted below.
#[test]
fn fisheye_analytic_and_central_difference_bundles_agree() {
    let (mut legacy, n_behind) = make_fisheye_scene_for(equidistant_legacy(130.0), 6, 90);
    let (mut native, _) = make_fisheye_scene_for(equidistant_native(130.0), 6, 90);
    assert!(n_behind >= 50, "scene is not wide enough: {n_behind}");
    // The arms take different paths through `project_with_jac`.
    assert!(!legacy.cam.model.supports_pixel_jacobian());
    assert!(native.cam.model.supports_pixel_jacobian());
    // …but describe the same camera: identical observations to start from.
    assert_eq!(legacy.uv.len(), native.uv.len());
    for (a, b) in legacy.uv.iter().zip(native.uv.iter()) {
        assert!(
            (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12,
            "the two representations disagree on a projection: {a:?} vs {b:?}"
        );
    }

    let perturb = |s: &mut Scene| {
        for (i, q) in s.quats.iter_mut().enumerate() {
            *q = UnitQuaternion::from_scaled_axis(Vector3::new(
                0.01 * jitter(i, 21),
                0.01 * jitter(i, 22),
                0.01 * jitter(i, 23),
            )) * *q;
        }
        for (p, x) in s.points.iter_mut().enumerate() {
            for (c, v) in x.iter_mut().enumerate() {
                *v += 0.05 * jitter(p * 3 + c, 31);
            }
        }
    };
    perturb(&mut legacy);
    perturb(&mut native);

    let out_l = run(&mut legacy, false, &DEFAULT_SCHEDULE);
    let out_n = run(&mut native, false, &DEFAULT_SCHEDULE);

    assert_eq!(out_l.focal, 130.0);
    assert_eq!(out_n.focal, 130.0);
    for (i, (a, b)) in legacy.quats.iter().zip(native.quats.iter()).enumerate() {
        assert!(
            a.angle_to(b) < 1e-9,
            "image {i} rotation split {}",
            a.angle_to(b)
        );
        let dt = (legacy.trans[i] - native.trans[i]).norm();
        assert!(dt < 1e-9, "image {i} translation split {dt}");
    }
    for (p, (a, b)) in legacy.points.iter().zip(native.points.iter()).enumerate() {
        let d = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
        assert!(d < 1e-8, "point {p} split {d}");
    }
    let worst = out_n.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        worst < 0.5,
        "worst reprojection {worst} px under the analytic path"
    );
}

// ── Focal release under the equidistant model ─────────────────────────────
//
// `opt_f` admits exactly the two single-focal, distortion-free models (the
// gate in `bundle_adjust_staged`). The tests below pin the three things that claim
// rests on: the analytic focal column is exact for `EquidistantFisheye`
// INCLUDING past θ = 90°, a released focal actually recovers a planted one
// there, and every other model still holds its focal fixed rather than taking
// a half-modeled step.

/// `∂(u, v)/∂f` as this kernel computes it, against a central difference of
/// the projection, over the whole equidistant field including the periphery.
///
/// The column is `(u − cx)/f` because `f` multiplies a distorted coordinate
/// that never reads it (`x_d = θ·û`, `θ = atan2(ρ, rz)`). That claim is what
/// is measured here, at the same rays a >180° capture actually produces.
#[test]
fn equidistant_focal_column_matches_a_central_difference() {
    let f0 = 130.0;
    let cam = equidistant_native(f0);
    let (cx, cy) = cam.principal_point();
    let h = 1e-4 * f0;
    let cam_p = equidistant_native(f0 + h);
    let cam_m = equidistant_native(f0 - h);
    let mut worst: f64 = 0.0;
    let mut n_past_90 = 0usize;
    // θ from near-axis out to 170°, five azimuths, three ray scales (the
    // projection is scale-invariant, the derivative must be too).
    for ti in 0..18 {
        let theta = (5.0 + 10.0 * ti as f64).to_radians();
        for ai in 0..5 {
            let az = 2.0 * std::f64::consts::PI * ai as f64 / 5.0;
            for &scale in &[0.35_f64, 1.0, 7.5] {
                // Optical-frame direction at (θ, az) → canonical via
                // S = diag(1, −1, −1).
                let opt = Vector3::new(theta.sin() * az.cos(), theta.sin() * az.sin(), theta.cos());
                let r = [scale * opt.x, -scale * opt.y, -scale * opt.z];
                if theta > std::f64::consts::FRAC_PI_2 {
                    n_past_90 += 1;
                }
                let (u, v) = cam.ray_to_pixel(r).expect("equidistant is total");
                let (up, vp) = cam_p.ray_to_pixel(r).unwrap();
                let (um, vm) = cam_m.ray_to_pixel(r).unwrap();
                let (fd_u, fd_v) = ((up - um) / (2.0 * h), (vp - vm) / (2.0 * h));
                // The kernel's column, verbatim.
                let (col_u, col_v) = ((u - cx) / f0, (v - cy) / f0);
                // Relative to the column's own magnitude (it grows with θ);
                // the absolute floor keeps the near-axis rays meaningful.
                let denom = col_u.abs().max(col_v.abs()).max(1e-3);
                worst = worst
                    .max((col_u - fd_u).abs() / denom)
                    .max((col_v - fd_v).abs() / denom);
            }
        }
    }
    assert!(n_past_90 >= 100, "not enough periphery: {n_past_90}");
    // A central difference of an exactly-linear function is exact to rounding;
    // the bar is the difference quotient's own cancellation noise, not a
    // truncation allowance.
    assert!(
        worst < 1e-9,
        "analytic focal column vs central difference: worst relative error {worst}"
    );
}

/// Free-focal BA under `EquidistantFisheye` recovers a planted focal from a
/// start several percent off, on a scene whose periphery is past 90°.
#[test]
fn equidistant_opt_f_recovers_a_perturbed_focal() {
    let f_true = 130.0;
    let (mut s, n_behind) = make_fisheye_scene_for(equidistant_native(f_true), 8, 140);
    assert!(
        n_behind >= 50,
        "scene is not wide enough: {n_behind}/{} past 90°",
        s.uv.len()
    );
    // The observations were generated at 130; hand the solver 118.3 (−9%).
    let f_start = 118.3;
    s.cam = equidistant_native(f_start);
    // The focal trades against the scene scale, so the solve needs the
    // structure to move with it: perturb depths, not just the focal.
    for x in s.points.iter_mut() {
        for v in x.iter_mut() {
            *v *= f_start / f_true;
        }
    }
    for t in s.trans.iter_mut() {
        *t *= f_start / f_true;
    }
    let out = run(&mut s, true, &DEFAULT_SCHEDULE);
    let err = (out.focal - f_true).abs() / f_true;
    assert!(
        err < 0.01,
        "released focal {} from a {:.1}% start (want {f_true})",
        out.focal,
        100.0 * (f_start / f_true - 1.0)
    );
    // …and it converged, not just drifted: the fit is sub-pixel.
    let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        worst < 0.5,
        "worst reprojection {worst} px after the release"
    );
}

/// The gauge tests, re-derived against `EquidistantFisheye`: `opt_f = false`
/// holds the focal EXACTLY, and the release is what moves it.
#[test]
fn equidistant_fixed_focal_holds_exactly() {
    let f0 = 130.0;
    let (mut fixed, _) = make_fisheye_scene_for(equidistant_native(f0), 6, 90);
    let (mut freed, _) = make_fisheye_scene_for(equidistant_native(f0), 6, 90);
    // Same wrong start in both arms.
    let off = 0.94 * f0;
    for s in [&mut fixed, &mut freed] {
        s.cam = equidistant_native(off);
        for x in s.points.iter_mut() {
            for v in x.iter_mut() {
                *v *= off / f0;
            }
        }
        for t in s.trans.iter_mut() {
            *t *= off / f0;
        }
    }
    let out_fixed = run(&mut fixed, false, &DEFAULT_SCHEDULE);
    assert_eq!(
        out_fixed.focal.to_bits(),
        off.to_bits(),
        "a fixed-focal equidistant solve moved the focal to {}",
        out_fixed.focal
    );
    let out_freed = run(&mut freed, true, &DEFAULT_SCHEDULE);
    assert!(
        (out_freed.focal - f0).abs() < (off - f0).abs() / 2.0,
        "the release did not close on the planted focal: {} (start {off}, true {f0})",
        out_freed.focal
    );
}

/// The multi-coefficient fisheye family is NOT released: its `∂u/∂f` is not
/// `(u − cx)/f` (the coefficients act on a `θ` recovered from `r/f`), so the
/// core degrades `opt_f` to a fixed-focal solve rather than stepping a
/// half-modeled focal. The release admits the two one-coefficient models
/// only.
#[test]
fn polynomial_fisheye_still_holds_its_focal_under_opt_f() {
    let f0 = 130.0;
    let (mut s, _) = make_fisheye_scene_for(equidistant_legacy(f0), 6, 90);
    let off = 0.94 * f0;
    s.cam = equidistant_legacy(off);
    for x in s.points.iter_mut() {
        for v in x.iter_mut() {
            *v *= off / f0;
        }
    }
    for t in s.trans.iter_mut() {
        *t *= off / f0;
    }
    let out = run(&mut s, true, &DEFAULT_SCHEDULE);
    assert_eq!(
        out.focal.to_bits(),
        off.to_bits(),
        "RadialFisheye took a focal step under opt_f: {}",
        out.focal
    );
}

// ── The curvature rung: `opt_k1` under SIMPLE_RADIAL_FISHEYE ───────────────
//
// A real fisheye's `r(θ)` is not exactly `f·θ`. Held to the equidistant map,
// the adjustment buys the leftover θ³ curvature back with GEOMETRY — a finite
// dome that pulls sky and horizon off infinity. Releasing the one radial
// coefficient gives that residual field a parameter to land on. These tests
// pin the rung's claims: the `∂/∂k1` column is exact, a planted curvature
// comes back (focal fixed and co-released), a truly equidistant scene stays
// at `k1 = 0`, direction rows carry the column too, and every other model
// degrades instead of taking a half-modeled step.

/// A `SIMPLE_RADIAL_FISHEYE` on the same 480² frame as the equidistant
/// scenes above.
fn radial_fisheye(f: f64, k1: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: k1,
        },
        width: 480,
        height: 480,
    }
}

/// [`run`] with both shared-camera releases.
fn run_k1(s: &mut Scene, opt_f: bool, opt_k1: bool, schedule: &[BaSchedule]) -> BundleAdjustment {
    bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        None,
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        opt_f,
        opt_k1,
        false,
        schedule,
        60,
        2,
        12,
    )
}

/// [`run_k1`] with a `point_at_infinity` mask.
fn run_k1_masked(
    s: &mut Scene,
    mask: &[bool],
    opt_f: bool,
    opt_k1: bool,
    schedule: &[BaSchedule],
) -> BundleAdjustment {
    bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        Some(mask),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        opt_f,
        opt_k1,
        false,
        schedule,
        60,
        2,
        12,
    )
}

/// `∂(u, v)/∂k1` as the kernel computes it, against a central difference of
/// the projection in `k1`, over the whole field including past 90°.
///
/// The column is `f·θ³·û` because `k1` enters as `θ_d = θ·(1 + k1·θ²)` with
/// `θ` read off the ray, never off a pixel radius — so, like the focal
/// column, it is the derivative of an exactly-linear dependence and a central
/// difference must reproduce it to rounding.
#[test]
fn k1_column_matches_a_central_difference() {
    let f = 130.0;
    let h = 1e-3;
    let mut worst: f64 = 0.0;
    let mut n_past_90 = 0usize;
    for &k1_0 in &[0.0f64, 0.03, -0.02] {
        let cam_p = radial_fisheye(f, k1_0 + h);
        let cam_m = radial_fisheye(f, k1_0 - h);
        for ti in 0..18 {
            let theta = (5.0 + 10.0 * ti as f64).to_radians();
            for ai in 0..5 {
                let az = 2.0 * std::f64::consts::PI * ai as f64 / 5.0;
                for &scale in &[0.35_f64, 1.0, 7.5] {
                    // Optical-frame direction at (θ, az) → canonical via
                    // S = diag(1, −1, −1).
                    let opt =
                        Vector3::new(theta.sin() * az.cos(), theta.sin() * az.sin(), theta.cos());
                    let r = [scale * opt.x, -scale * opt.y, -scale * opt.z];
                    if theta > std::f64::consts::FRAC_PI_2 {
                        n_past_90 += 1;
                    }
                    let (up, vp) = cam_p.ray_to_pixel(r).unwrap();
                    let (um, vm) = cam_m.ray_to_pixel(r).unwrap();
                    let (fd_u, fd_v) = ((up - um) / (2.0 * h), (vp - vm) / (2.0 * h));
                    // The kernel's column, verbatim.
                    let (col_u, col_v) = k1_column(f, Vector3::new(r[0], r[1], r[2]));
                    let denom = col_u.abs().max(col_v.abs()).max(1e-3);
                    worst = worst
                        .max((col_u - fd_u).abs() / denom)
                        .max((col_v - fd_v).abs() / denom);
                }
            }
        }
    }
    assert!(n_past_90 >= 100, "not enough periphery: {n_past_90}");
    assert!(
        worst < 1e-9,
        "analytic k1 column vs central difference: worst relative error {worst}"
    );
}

/// The column is scale-invariant in the ray (the projection is), and exactly
/// zero on the optical axis where `θ³·û → 0`.
#[test]
fn k1_column_is_scale_invariant_and_vanishes_on_axis() {
    let f = 130.0;
    let p = Vector3::new(0.4, -0.3, -0.6);
    let (a_u, a_v) = k1_column(f, p);
    for s in [0.2f64, 3.0, 100.0] {
        let (b_u, b_v) = k1_column(f, p * s);
        assert!((a_u - b_u).abs() < 1e-12 && (a_v - b_v).abs() < 1e-12);
    }
    assert_eq!(k1_column(f, Vector3::new(0.0, 0.0, -2.0)), (0.0, 0.0));
}

/// A scene shot through a curved lens, handed to the solver as an equidistant
/// one (`k1 = 0`): the release recovers the planted curvature and the fit
/// comes back sub-pixel.
#[test]
fn opt_k1_recovers_a_planted_curvature() {
    let f = 130.0;
    for &k1_true in &[0.02f64, -0.02] {
        let (mut s, n_behind) = make_fisheye_scene_for(radial_fisheye(f, k1_true), 8, 140);
        assert!(n_behind >= 50, "scene is not wide enough: {n_behind}");
        // Hand the solver the same focal but no curvature at all.
        s.cam = radial_fisheye(f, 0.0);
        let out = run_k1(&mut s, false, true, &DEFAULT_SCHEDULE);
        assert_eq!(out.focal.to_bits(), f.to_bits(), "the focal was not fixed");
        let err = (out.k1 - k1_true).abs() / k1_true.abs();
        assert!(
            err < 0.1,
            "released k1 {} from a 0.0 start (want {k1_true})",
            out.k1
        );
        let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
        assert!(worst < 0.5, "worst reprojection {worst} px after the rung");
    }
}

/// The staged release the callers actually run: focal and curvature together,
/// from a focal several percent off and no curvature.
#[test]
fn opt_f_and_opt_k1_recover_together() {
    let f_true = 130.0;
    let k1_true = 0.025;
    let (mut s, _) = make_fisheye_scene_for(radial_fisheye(f_true, k1_true), 8, 140);
    let f_start = 124.0;
    s.cam = radial_fisheye(f_start, 0.0);
    // The focal trades against the scene scale: move the structure with it.
    for x in s.points.iter_mut() {
        for v in x.iter_mut() {
            *v *= f_start / f_true;
        }
    }
    for t in s.trans.iter_mut() {
        *t *= f_start / f_true;
    }
    let out = run_k1(&mut s, true, true, &DEFAULT_SCHEDULE);
    let f_err = (out.focal - f_true).abs() / f_true;
    let k_err = (out.k1 - k1_true).abs() / k1_true;
    assert!(
        f_err < 0.01 && k_err < 0.15,
        "co-released (f, k1) = ({}, {}) from ({f_start}, 0.0), want ({f_true}, {k1_true})",
        out.focal,
        out.k1
    );
    let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(worst < 0.5, "worst reprojection {worst} px");
}

/// The fixed point that matters for the promotion
/// EQUIDISTANT_FISHEYE → SIMPLE_RADIAL_FISHEYE(k1 = 0): on a scene that
/// really is equidistant, releasing `k1` leaves it at zero and leaves the
/// geometry where it was.
#[test]
fn opt_k1_holds_at_zero_on_an_equidistant_scene() {
    let f = 130.0;
    let (mut released, _) = make_fisheye_scene_for(radial_fisheye(f, 0.0), 8, 140);
    let mut fixed = released.clone();
    // A perturbed start, so both arms have real work to do.
    let perturb = |s: &mut Scene| {
        for (i, q) in s.quats.iter_mut().enumerate() {
            *q = UnitQuaternion::from_scaled_axis(Vector3::new(
                0.01 * jitter(i, 21),
                0.01 * jitter(i, 22),
                0.01 * jitter(i, 23),
            )) * *q;
        }
        for (p, x) in s.points.iter_mut().enumerate() {
            for (c, v) in x.iter_mut().enumerate() {
                *v += 0.05 * jitter(p * 3 + c, 31);
            }
        }
    };
    perturb(&mut released);
    perturb(&mut fixed);
    let out_r = run_k1(&mut released, false, true, &DEFAULT_SCHEDULE);
    let out_f = run_k1(&mut fixed, false, false, &DEFAULT_SCHEDULE);
    assert_eq!(out_f.k1.to_bits(), 0.0f64.to_bits(), "unreleased k1 moved");
    assert!(
        out_r.k1.abs() < 1e-4,
        "the released k1 walked off zero on an equidistant scene: {}",
        out_r.k1
    );
    // …and the reconstruction is the same one, not a curvature-for-geometry
    // trade that happens to end near zero.
    for (i, (a, b)) in released.quats.iter().zip(fixed.quats.iter()).enumerate() {
        assert!(a.angle_to(b) < 1e-4, "image {i} rotation split");
    }
    for (p, (a, b)) in released.points.iter().zip(fixed.points.iter()).enumerate() {
        let d = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
        assert!(d < 1e-3, "point {p} split {d}");
    }
}

/// Every other model degrades: `opt_k1` is the radial model's rung alone, and
/// the core never takes a half-modeled step (the binding rejects these loudly
/// before they get here).
#[test]
fn opt_k1_is_gated_on_the_radial_fisheye_model() {
    let f = 130.0;
    // EQUIDISTANT_FISHEYE has no k1 at all: the release is dropped, and the
    // solve is bit for bit the one without it.
    let (mut released, _) = make_fisheye_scene_for(equidistant_native(f), 6, 90);
    let mut plain = released.clone();
    let out_r = run_k1(&mut released, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_k1(&mut plain, true, false, &DEFAULT_SCHEDULE);
    assert_eq!(out_r.k1.to_bits(), 0.0f64.to_bits());
    assert_bitwise_equal(&released, &out_r, &plain, &out_p);
    // The multi-coefficient family holds BOTH parameters: its k1 is not
    // reachable through this column, and its focal is not `(u − cx)/f`.
    let (mut s, _) = make_fisheye_scene_for(equidistant_legacy(f), 6, 90);
    let off = 0.94 * f;
    s.cam = equidistant_legacy(off);
    let out = run_k1(&mut s, true, true, &DEFAULT_SCHEDULE);
    assert_eq!(out.focal.to_bits(), off.to_bits());
    assert_eq!(out.k1.to_bits(), 0.0f64.to_bits());
    // A pinhole scene the same way (the rung is a fisheye one).
    let mut ph = make_scene(6, 60);
    let out_ph = run_k1(&mut ph, true, true, &DEFAULT_SCHEDULE);
    assert_eq!(out_ph.k1.to_bits(), 0.0f64.to_bits());
}

/// The step guard: `θ_d = θ·(1 + k1·θ²)` must stay strictly increasing over
/// the field the observations occupy, or the projection folds — two incidence
/// angles onto one pixel radius, and `pixel_to_ray` picking the wrong branch.
#[test]
fn k1_step_guard_rejects_a_folded_map() {
    let f = 130.0;
    // A 480² frame at f = 130 images out to r ≈ 240 px ⇔ θ ≈ 1.85 rad.
    let field_r = 240.0;
    // Positive curvature never folds, at any magnitude.
    assert!(k1_step_admissible(f, 0.0, field_r));
    assert!(k1_step_admissible(f, 5.0, field_r));
    // A fold beyond θ = π is beyond every physical ray.
    assert!(k1_step_admissible(f, -0.03, field_r));
    // A fold inside the imaged field is not admissible…
    assert!(!k1_step_admissible(f, -0.2, field_r));
    // …but the very same k1 is, for a camera whose field stops short of it
    // (the guard reads the data, not a constant).
    assert!(k1_step_admissible(f, -0.2, 90.0));
    // Non-finite steps never pass.
    assert!(!k1_step_admissible(f, f64::NAN, field_r));
    assert!(!k1_step_admissible(f, f64::NEG_INFINITY, field_r));
    // And the boundary is where the derivation says: the peak imaged radius
    // is (2/3)·f·θ_fold.
    let k1 = -0.1f64;
    let theta_fold = 1.0 / (-3.0 * k1).sqrt();
    let peak = (2.0 / 3.0) * f * theta_fold;
    assert!(k1_step_admissible(f, k1, peak * 0.999));
    assert!(!k1_step_admissible(f, k1, peak * 1.001));
}

/// End to end, a released solve never lands on a folded map.
#[test]
fn released_k1_stays_admissible() {
    let f = 130.0;
    let (mut s, _) = make_fisheye_scene_for(radial_fisheye(f, -0.02), 8, 140);
    s.cam = radial_fisheye(f, 0.0);
    let out = run_k1(&mut s, true, true, &DEFAULT_SCHEDULE);
    let (cx, cy) = s.cam.principal_point();
    let field_r =
        s.uv.iter()
            .map(|p| (p[0] - cx).hypot(p[1] - cy))
            .fold(0.0f64, f64::max);
    assert!(
        k1_step_admissible(out.focal, out.k1, field_r),
        "the solve returned a folded map: k1 = {} at f = {}",
        out.k1,
        out.focal
    );
}

/// Direction rows carry the rung. A point at infinity projects through the
/// very same map, so `∂/∂k1` applies to it unchanged — and where the finite
/// cloud sits near the axis (no θ³ signal), the far field is the only thing
/// that can recover the curvature.
#[test]
fn directions_participate_in_the_curvature_rung() {
    let f = 130.0;
    let k1_true = 0.03;
    let cam_true = radial_fisheye(f, k1_true);

    // A near-axis finite cloud in front of an arc of cameras: enough to
    // satisfy the finite-survivor floor, far too little curvature signal to
    // fit k1 from.
    let n_img = 8;
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        let ang = 0.12 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(ang.sin(), 0.2 * jitter(i, 11), ang.cos() + 6.0);
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
    }
    let mut points: Vec<[f64; 3]> = Vec::new();
    for p in 0..40 {
        points.push([0.6 * jitter(p, 5), 0.6 * jitter(p, 6), 0.4 * jitter(p, 7)]);
    }
    let n_finite = points.len();
    // Far-field directions spread over the whole 210° field, out to θ = 105°.
    let mut dir_ids = Vec::new();
    for j in 0..60 {
        let theta = (25.0 + 80.0 * (j as f64) / 59.0f64).to_radians();
        let phi = 2.399_963 * j as f64;
        let d = Vector3::new(
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            -theta.cos(),
        );
        dir_ids.push(points.len());
        points.push([d.x, d.y, d.z]);
    }
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    for (p, x) in points.iter().enumerate() {
        let is_dir = p >= n_finite;
        for i in 0..n_img {
            let xv = Vector3::new(x[0], x[1], x[2]);
            let c = if is_dir {
                quats[i] * xv
            } else {
                quats[i] * xv + trans[i]
            };
            let Some((u, v)) = cam_true.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            if (u - 240.0).hypot(v - 240.0) > f * 105.0_f64.to_radians() {
                continue;
            }
            uv.push([u, v]);
            obs_img.push(i as u32);
            obs_pt.push(p as u32);
        }
    }
    let scene = Scene {
        cam: radial_fisheye(f, 0.0),
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
    };
    let mask = dir_mask(&scene, &dir_ids);

    // With the directions in the solve, the rung recovers the curvature.
    let mut with_dirs = scene.clone();
    let out = run_k1_masked(&mut with_dirs, &mask, false, true, &DEFAULT_SCHEDULE);
    let err = (out.k1 - k1_true).abs() / k1_true;
    assert!(
        err < 0.15,
        "k1 {} from the direction rows (want {k1_true})",
        out.k1
    );

    // The control: drop every direction observation and the near-axis finite
    // cloud alone cannot see the curvature.
    let mut finite_only = scene.clone();
    let keep: Vec<usize> = (0..finite_only.obs_pt.len())
        .filter(|&k| (finite_only.obs_pt[k] as usize) < n_finite)
        .collect();
    finite_only.uv = keep.iter().map(|&k| finite_only.uv[k]).collect();
    finite_only.obs_img = keep.iter().map(|&k| finite_only.obs_img[k]).collect();
    finite_only.obs_pt = keep.iter().map(|&k| finite_only.obs_pt[k]).collect();
    let out_finite = run_k1(&mut finite_only, false, true, &DEFAULT_SCHEDULE);
    let err_finite = (out_finite.k1 - k1_true).abs() / k1_true;
    assert!(
        err_finite > 3.0 * err,
        "the near-axis control recovered k1 too well ({} vs {}) - the test no \
         longer isolates the direction rows",
        out_finite.k1,
        out.k1
    );
}

// ── The spline rung: opt_bspline under SFMTOOL_FISHEYE ──────────────────────

use crate::camera::distortion::bspline::delta as bspline_delta;

/// Spline-domain end used by the spline-rung scenes: slightly beyond the
/// 105° image circle, so the outermost observations sit inside the basis.
const THETA_MAX: f64 = 2.0;

/// A flattening 8-coefficient spline (the Phase A fixture): monotone on
/// `[0, THETA_MAX]`, ~9 px of rim correction at `f = 130`.
const PLANTED_BSPLINE: [f64; 8] = [-0.001, -0.004, -0.01, -0.02, -0.03, -0.05, -0.07, -0.09];

fn sfmtool_fisheye(f: f64, bspline: &[f64]) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: THETA_MAX,
            bspline: bspline.to_vec(),
        },
        width: 480,
        height: 480,
    }
}

/// [`run`] with the focal and spline releases.
fn run_bspline(
    s: &mut Scene,
    opt_f: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],
) -> BundleAdjustment {
    bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        None,
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        opt_f,
        false,
        opt_bspline,
        schedule,
        60,
        2,
        12,
    )
}

/// [`run_bspline`] with a `point_at_infinity` mask.
fn run_bspline_masked(
    s: &mut Scene,
    mask: &[bool],
    opt_f: bool,
    opt_bspline: bool,
    schedule: &[BaSchedule],
) -> BundleAdjustment {
    bundle_adjust(
        &s.cam,
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        Some(mask),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        opt_f,
        false,
        opt_bspline,
        schedule,
        60,
        2,
        12,
    )
}

/// Largest composite-map discrepancy `f·|δ_a(θ) − δ_b(θ)|` in pixels over
/// `[lo, hi]` — the "compare the map, not the coefficients" metric.
fn worst_map_err_px(f: f64, a: &[f64], b: &[f64], lo: f64, hi: f64) -> f64 {
    (0..=200)
        .map(|s| {
            let theta = lo + (hi - lo) * s as f64 / 200.0;
            f * (bspline_delta(a, THETA_MAX, theta) - bspline_delta(b, THETA_MAX, theta)).abs()
        })
        .fold(0.0f64, f64::max)
}

/// `∂(u, v)/∂cᵢ` as the kernel computes it, against a central difference of
/// the projection in each coefficient, over the whole field including past
/// 90°.
///
/// The columns are `f·Bᵢ(θ)·û` because every coefficient enters as
/// `θ_d = θ + Σ cᵢ·Bᵢ(θ)` with `θ` read off the ray, never off a pixel
/// radius — the projection is exactly LINEAR in each coefficient, so a
/// central difference must reproduce the column to rounding (the same bar as
/// the k1 column's).
#[test]
fn bspline_columns_match_a_central_difference() {
    let f = 130.0;
    let h = 1e-3;
    let n = PLANTED_BSPLINE.len();
    let mut worst: f64 = 0.0;
    let mut n_past_90 = 0usize;
    for ti in 0..17 {
        let theta = (5.0 + 10.0 * ti as f64).to_radians();
        for ai in 0..5 {
            let az = 2.0 * std::f64::consts::PI * ai as f64 / 5.0;
            for &scale in &[0.35_f64, 1.0, 7.5] {
                // Optical-frame direction at (θ, az) → canonical via
                // S = diag(1, −1, −1).
                let opt = Vector3::new(theta.sin() * az.cos(), theta.sin() * az.sin(), theta.cos());
                let r = [scale * opt.x, -scale * opt.y, -scale * opt.z];
                if theta > std::f64::consts::FRAC_PI_2 {
                    n_past_90 += 1;
                }
                // The kernel's columns, assembled per coefficient.
                let (first, cols) = bspline_columns(
                    f,
                    n,
                    THETA_MAX,
                    SplineRadial::IncidenceAngle,
                    Vector3::new(r[0], r[1], r[2]),
                );
                let mut full = vec![[0.0f64; 2]; n];
                for (j, col) in cols.iter().enumerate() {
                    let fi = first + j;
                    if fi >= 2 {
                        full[fi - 2] = *col;
                    }
                }
                // Relative to the sample's largest column: an out-of-support
                // coefficient's column is exactly zero, and dividing the
                // central difference's pixel-rounding noise by a per-column
                // magnitude would measure the noise floor, not the column.
                let denom = full
                    .iter()
                    .flatten()
                    .fold(0.0f64, |m, c| m.max(c.abs()))
                    .max(1e-3);
                for (i, col) in full.iter().enumerate() {
                    let mut cp = PLANTED_BSPLINE;
                    cp[i] += h;
                    let mut cm = PLANTED_BSPLINE;
                    cm[i] -= h;
                    let (up, vp) = sfmtool_fisheye(f, &cp).ray_to_pixel(r).unwrap();
                    let (um, vm) = sfmtool_fisheye(f, &cm).ray_to_pixel(r).unwrap();
                    let (fd_u, fd_v) = ((up - um) / (2.0 * h), (vp - vm) / (2.0 * h));
                    worst = worst
                        .max((col[0] - fd_u).abs() / denom)
                        .max((col[1] - fd_v).abs() / denom);
                }
            }
        }
    }
    assert!(n_past_90 >= 100, "not enough periphery: {n_past_90}");
    assert!(
        worst < 1e-9,
        "analytic spline columns vs central difference: worst relative error {worst}"
    );
}

/// A scene shot through a flattening lens, handed to the solver with a zero
/// spline: the release recovers the planted coefficients — coefficient-wise
/// and, what actually matters, composite-map-wise — and the fit comes back
/// sub-pixel.
#[test]
fn opt_bspline_recovers_a_planted_spline() {
    let f = 130.0;
    let (mut s, n_behind) = make_fisheye_scene_for(sfmtool_fisheye(f, &PLANTED_BSPLINE), 8, 140);
    assert!(n_behind >= 50, "scene is not wide enough: {n_behind}");
    s.cam = sfmtool_fisheye(f, &[0.0; 8]);
    let out = run_bspline(&mut s, false, true, &DEFAULT_SCHEDULE);
    assert_eq!(out.focal.to_bits(), f.to_bits(), "the focal was not fixed");
    let map_err = worst_map_err_px(f, &out.bspline, &PLANTED_BSPLINE, 0.05, 1.85);
    assert!(
        map_err < 0.3,
        "recovered composite map off by {map_err} px (spline {:?})",
        out.bspline
    );
    for (i, (c, t)) in out.bspline.iter().zip(&PLANTED_BSPLINE).enumerate() {
        assert!(
            (c - t).abs() < 0.01,
            "coefficient {i}: {c} (want {t}; full spline {:?})",
            out.bspline
        );
    }
    let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(worst < 0.5, "worst reprojection {worst} px after the rung");
}

/// The staged release the callers actually run: focal and spline together,
/// from a focal several percent off and no spline.
#[test]
fn opt_f_and_opt_bspline_recover_together() {
    let f_true = 130.0;
    let (mut s, _) = make_fisheye_scene_for(sfmtool_fisheye(f_true, &PLANTED_BSPLINE), 8, 140);
    let f_start = 124.0;
    s.cam = sfmtool_fisheye(f_start, &[0.0; 8]);
    // The focal trades against the scene scale: move the structure with it.
    for x in s.points.iter_mut() {
        for v in x.iter_mut() {
            *v *= f_start / f_true;
        }
    }
    for t in s.trans.iter_mut() {
        *t *= f_start / f_true;
    }
    let out = run_bspline(&mut s, true, true, &DEFAULT_SCHEDULE);
    let f_err = (out.focal - f_true).abs() / f_true;
    let map_err = worst_map_err_px(f_true, &out.bspline, &PLANTED_BSPLINE, 0.05, 1.85);
    assert!(
        f_err < 0.01 && map_err < 0.5,
        "co-released f = {} (want {f_true}), composite map off by {map_err} px ({:?})",
        out.focal,
        out.bspline
    );
    let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(worst < 0.5, "worst reprojection {worst} px");
}

/// The fixed point that matters for the promotion
/// EQUIDISTANT_FISHEYE → SFMTOOL_FISHEYE(zero spline): on a scene that
/// really is equidistant, releasing the spline leaves it at zero and leaves
/// the geometry where it was.
#[test]
fn opt_bspline_holds_at_zero_on_an_equidistant_scene() {
    let f = 130.0;
    let (mut released, _) = make_fisheye_scene_for(sfmtool_fisheye(f, &[0.0; 8]), 8, 140);
    let mut fixed = released.clone();
    // A perturbed start, so both arms have real work to do.
    let perturb = |s: &mut Scene| {
        for (i, q) in s.quats.iter_mut().enumerate() {
            *q = UnitQuaternion::from_scaled_axis(Vector3::new(
                0.01 * jitter(i, 21),
                0.01 * jitter(i, 22),
                0.01 * jitter(i, 23),
            )) * *q;
        }
        for (p, x) in s.points.iter_mut().enumerate() {
            for (c, v) in x.iter_mut().enumerate() {
                *v += 0.05 * jitter(p * 3 + c, 31);
            }
        }
    };
    perturb(&mut released);
    perturb(&mut fixed);
    let out_r = run_bspline(&mut released, false, true, &DEFAULT_SCHEDULE);
    let out_f = run_bspline(&mut fixed, false, false, &DEFAULT_SCHEDULE);
    for (i, c) in out_f.bspline.iter().enumerate() {
        assert_eq!(c.to_bits(), 0.0f64.to_bits(), "unreleased c{i} moved");
    }
    let held = worst_map_err_px(f, &out_r.bspline, &[0.0; 8], 0.05, 1.85);
    assert!(
        held < 0.1,
        "the released spline walked off zero on an equidistant scene by \
         {held} px ({:?})",
        out_r.bspline
    );
    // …and the reconstruction is the same one, not a spline-for-geometry
    // trade that happens to end near zero.
    for (i, (a, b)) in released.quats.iter().zip(fixed.quats.iter()).enumerate() {
        assert!(a.angle_to(b) < 1e-4, "image {i} rotation split");
    }
    for (p, (a, b)) in released.points.iter().zip(fixed.points.iter()).enumerate() {
        let d = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
        assert!(d < 1e-3, "point {p} split {d}");
    }
}

/// Every other model degrades: `opt_bspline` is the spline model's rung
/// alone, and the core never takes a half-modeled step (the binding rejects
/// these loudly before they get here) — nor does it release a spline too
/// short to define the spline.
#[test]
fn opt_bspline_is_gated_on_the_sfmtool_fisheye_model() {
    let f = 130.0;
    // EQUIDISTANT_FISHEYE has no spline at all: the release is dropped, and
    // the solve is bit for bit the one without it.
    let (mut released, _) = make_fisheye_scene_for(equidistant_native(f), 6, 90);
    let mut plain = released.clone();
    let out_r = run_bspline(&mut released, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_bspline(&mut plain, true, false, &DEFAULT_SCHEDULE);
    assert!(out_r.bspline.is_empty());
    assert_bitwise_equal(&released, &out_r, &plain, &out_p);
    // SIMPLE_RADIAL_FISHEYE carries a k1, not a spline: same degrade.
    let (mut srf_r, _) = make_fisheye_scene_for(radial_fisheye(f, 0.02), 6, 90);
    let mut srf_p = srf_r.clone();
    let out_r = run_bspline(&mut srf_r, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_bspline(&mut srf_p, true, false, &DEFAULT_SCHEDULE);
    assert!(out_r.bspline.is_empty());
    assert_bitwise_equal(&srf_r, &out_r, &srf_p, &out_p);
    // The converse: SFMTOOL_FISHEYE carries no k1, so `opt_k1` degrades on
    // it the same way (the two releases are naturally exclusive).
    let (mut k1_r, _) = make_fisheye_scene_for(sfmtool_fisheye(f, &PLANTED_BSPLINE), 6, 90);
    let mut k1_p = k1_r.clone();
    let out_r = run_k1(&mut k1_r, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_k1(&mut k1_p, true, false, &DEFAULT_SCHEDULE);
    assert_eq!(out_r.k1.to_bits(), 0.0f64.to_bits());
    assert_bitwise_equal(&k1_r, &out_r, &k1_p, &out_p);
    // A coefficient vector too short to define the spline (the identity map)
    // carries nothing to release: same silent degrade, and the input comes back.
    let (mut empty_r, _) = make_fisheye_scene_for(sfmtool_fisheye(f, &[]), 6, 90);
    let mut empty_p = empty_r.clone();
    let out_r = run_bspline(&mut empty_r, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_bspline(&mut empty_p, true, false, &DEFAULT_SCHEDULE);
    assert!(out_r.bspline.is_empty());
    assert_bitwise_equal(&empty_r, &out_r, &empty_p, &out_p);
}

/// The step guard: `θ_d = θ + δ(θ)` must stay strictly increasing over the
/// spline's whole domain, or the projection folds — two incidence angles
/// onto one pixel radius, and `pixel_to_ray`'s Newton solve losing its
/// bracket.
#[test]
fn bspline_step_guard_rejects_a_folded_spline() {
    // The identity and the (gently flattening) planted spline pass.
    assert!(bspline_step_admissible(&[0.0; 8], THETA_MAX));
    assert!(bspline_step_admissible(&PLANTED_BSPLINE, THETA_MAX));
    // A steep drop mid-domain folds the map: with interior knot spans of
    // 2/7, a −0.4 step between adjacent coefficients puts δ' ≈ −1.4 < −1.
    let folded = [0.0, 0.0, 0.0, -0.4, -0.8, -0.8, -0.8, -0.8];
    assert!(!bspline_step_admissible(&folded, THETA_MAX));
    // A fold in the outermost span is rejected too: the guard covers the
    // whole domain, not just where this round's data happens to sit, because
    // the accepted spline is persisted into the camera and its Newton
    // inverse needs the global bracket.
    let rim_folded = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5];
    assert!(!bspline_step_admissible(&rim_folded, THETA_MAX));
    // Non-finite steps never pass.
    let mut bad = PLANTED_BSPLINE;
    bad[3] = f64::NAN;
    assert!(!bspline_step_admissible(&bad, THETA_MAX));
    bad[3] = f64::NEG_INFINITY;
    assert!(!bspline_step_admissible(&bad, THETA_MAX));
}

/// End to end, a released solve never lands on a folded spline.
#[test]
fn released_bspline_stays_admissible() {
    let f = 130.0;
    let (mut s, _) = make_fisheye_scene_for(sfmtool_fisheye(f, &PLANTED_BSPLINE), 8, 140);
    s.cam = sfmtool_fisheye(f, &[0.0; 8]);
    let out = run_bspline(&mut s, true, true, &DEFAULT_SCHEDULE);
    assert!(
        bspline_is_monotone(&out.bspline, THETA_MAX, THETA_MAX),
        "the solve returned a folded spline: {:?} at f = {}",
        out.bspline,
        out.focal
    );
}

/// A coefficient whose basis span no surviving observation reaches has
/// exactly-zero curvature: its shared slot is pinned per linearization, the
/// reduced system stays regular, and the coefficient comes back bit for bit
/// at its input value.
#[test]
fn unsupported_bspline_slots_hold_their_input_exactly() {
    let f = 130.0;
    // Sentinels in the outermost two coefficients, whose basis support
    // starts at θ = 10/7 ≈ 1.43 and 12/7 ≈ 1.71: invisible below θ ≈ 1.43.
    let sentinels = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.017, -0.008];
    // An equidistant capture imaged only out to θ = 1 rad (≈ 57°): well
    // inside the sentinels' support, three knot spans short of the rim.
    let (full, _) = make_fisheye_scene_for(sfmtool_fisheye(f, &[0.0; 8]), 8, 140);
    let mut s = Scene {
        cam: sfmtool_fisheye(f, &sentinels),
        quats: full.quats.clone(),
        trans: full.trans.clone(),
        points: full.points.clone(),
        uv: Vec::new(),
        obs_img: Vec::new(),
        obs_pt: Vec::new(),
    };
    for k in 0..full.uv.len() {
        let r = (full.uv[k][0] - 240.0).hypot(full.uv[k][1] - 240.0);
        if r <= f * 1.0 {
            s.uv.push(full.uv[k]);
            s.obs_img.push(full.obs_img[k]);
            s.obs_pt.push(full.obs_pt[k]);
        }
    }
    assert!(s.uv.len() >= 100, "narrow scene too small: {}", s.uv.len());
    let out = run_bspline(&mut s, false, true, &DEFAULT_SCHEDULE);
    // The solve is not degenerate…
    let finite = out.residual_norms.iter().filter(|r| r.is_finite()).count();
    assert!(
        finite >= 100,
        "solve degenerated: {finite} finite residuals"
    );
    // …the unsupported outer coefficients held their inputs exactly…
    assert_eq!(out.bspline[6].to_bits(), sentinels[6].to_bits());
    assert_eq!(out.bspline[7].to_bits(), sentinels[7].to_bits());
    // …and the supported inner ones stayed near the (true) zero.
    let inner_err = worst_map_err_px(
        f,
        &out.bspline[..6]
            .iter()
            .chain(&[0.0, 0.0])
            .copied()
            .collect::<Vec<_>>(),
        &[0.0; 8],
        0.05,
        1.0,
    );
    assert!(
        inner_err < 0.2,
        "supported coefficients drifted by {inner_err} px: {:?}",
        out.bspline
    );
}

/// Direction rows carry the rung. A point at infinity projects through the
/// very same map, so `∂/∂cᵢ` applies to it unchanged — and where the finite
/// cloud sits near the axis (no basis signal), the far field is the only
/// thing that can recover the spline.
#[test]
fn directions_participate_in_the_bspline_rung() {
    let f = 130.0;
    let cam_true = sfmtool_fisheye(f, &PLANTED_BSPLINE);

    // A near-axis finite cloud in front of an arc of cameras: enough to
    // satisfy the finite-survivor floor, far too little periphery to fit the
    // spline from.
    let n_img = 8;
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        let ang = 0.12 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(ang.sin(), 0.2 * jitter(i, 11), ang.cos() + 6.0);
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
    }
    let mut points: Vec<[f64; 3]> = Vec::new();
    for p in 0..40 {
        points.push([0.6 * jitter(p, 5), 0.6 * jitter(p, 6), 0.4 * jitter(p, 7)]);
    }
    let n_finite = points.len();
    // Far-field directions spread over the whole 210° field, out to θ = 105°.
    let mut dir_ids = Vec::new();
    for j in 0..60 {
        let theta = (25.0 + 80.0 * (j as f64) / 59.0f64).to_radians();
        let phi = 2.399_963 * j as f64;
        let d = Vector3::new(
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            -theta.cos(),
        );
        dir_ids.push(points.len());
        points.push([d.x, d.y, d.z]);
    }
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    for (p, x) in points.iter().enumerate() {
        let is_dir = p >= n_finite;
        for i in 0..n_img {
            let xv = Vector3::new(x[0], x[1], x[2]);
            let c = if is_dir {
                quats[i] * xv
            } else {
                quats[i] * xv + trans[i]
            };
            let Some((u, v)) = cam_true.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            if (u - 240.0).hypot(v - 240.0) > f * 105.0_f64.to_radians() {
                continue;
            }
            uv.push([u, v]);
            obs_img.push(i as u32);
            obs_pt.push(p as u32);
        }
    }
    let scene = Scene {
        cam: sfmtool_fisheye(f, &[0.0; 8]),
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
    };
    let mask = dir_mask(&scene, &dir_ids);

    // With the directions in the solve, the rung recovers the composite map.
    let mut with_dirs = scene.clone();
    let out = run_bspline_masked(&mut with_dirs, &mask, false, true, &DEFAULT_SCHEDULE);
    let err = worst_map_err_px(f, &out.bspline, &PLANTED_BSPLINE, 0.5, 1.7);
    assert!(
        err < 1.0,
        "composite map off by {err} px from the direction rows ({:?})",
        out.bspline
    );

    // The control: drop every direction observation and the near-axis finite
    // cloud alone cannot see the spline.
    let mut finite_only = scene.clone();
    let keep: Vec<usize> = (0..finite_only.obs_pt.len())
        .filter(|&k| (finite_only.obs_pt[k] as usize) < n_finite)
        .collect();
    finite_only.uv = keep.iter().map(|&k| finite_only.uv[k]).collect();
    finite_only.obs_img = keep.iter().map(|&k| finite_only.obs_img[k]).collect();
    finite_only.obs_pt = keep.iter().map(|&k| finite_only.obs_pt[k]).collect();
    let out_finite = run_bspline(&mut finite_only, false, true, &DEFAULT_SCHEDULE);
    let err_finite = worst_map_err_px(f, &out_finite.bspline, &PLANTED_BSPLINE, 0.5, 1.7);
    assert!(
        err_finite > 3.0 * err,
        "the near-axis control recovered the spline too well ({err_finite} \
         vs {err} px) - the test no longer isolates the direction rows"
    );
}

// ── The spline rung on SFMTOOL_PINHOLE ──────────────────────────────────────

/// Spline-domain end used by the pinhole spline-rung scenes: just beyond the
/// image circle the scene generator keeps, so the outermost observations sit
/// inside the basis.
const RHO_MAX: f64 = 0.85;

/// Largest normalized image-plane radius the pinhole scenes admit — a
/// circular acceptance, so every azimuth reaches the same `ρ` and no
/// coefficient's support is corner-only.
const RHO_LIMIT: f64 = 0.82;

/// A gently expanding 6-coefficient spline: monotone on `[0, RHO_MAX]`, ~12 px
/// of rim correction at `f = 250`.
const PLANTED_PINHOLE_BSPLINE: [f64; 6] = [0.002, 0.007, 0.014, 0.024, 0.036, 0.05];

fn sfmtool_pinhole(f: f64, bspline: &[f64]) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_rho_max: RHO_MAX,
            bspline: bspline.to_vec(),
        },
        width: 480,
        height: 480,
    }
}

fn simple_pinhole_at(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
    }
}

/// A perspective scene wide enough that every spline coefficient has
/// observation support: cameras on a shallow arc inside a shell of points,
/// keeping observations inside the circular field `ρ ≤ RHO_LIMIT`. Returns the
/// scene and the number of observations past `ρ = 0.4`, where the spline is
/// the only thing that can explain the residual.
fn make_pinhole_scene_for(cam: CameraIntrinsics, n_img: usize, n_pt: usize) -> (Scene, usize) {
    let f = cam.focal_lengths().0;
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        let ang = 0.25 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(1.2 * ang.sin(), 0.3 * jitter(i, 11), 1.2 * ang.cos());
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
    }
    let mut points = Vec::new();
    for p in 0..n_pt {
        let theta = std::f64::consts::PI * (0.15 + 0.7 * (p as f64) / (n_pt as f64 - 1.0));
        let phi = 2.399_963 * p as f64;
        let rad = 6.0 + 1.5 * jitter(p, 3);
        points.push([
            rad * theta.sin() * phi.cos(),
            rad * theta.cos(),
            rad * theta.sin() * phi.sin(),
        ]);
    }
    let mut per_pt: Vec<Vec<(u32, [f64; 2], bool)>> = vec![Vec::new(); points.len()];
    for (p, x) in points.iter().enumerate() {
        for i in 0..n_img {
            let c = quats[i] * Vector3::new(x[0], x[1], x[2]) + trans[i];
            let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            let r = (u - 240.0).hypot(v - 240.0);
            if r > f * RHO_LIMIT {
                continue;
            }
            per_pt[p].push((i as u32, [u, v], r > f * 0.4));
        }
    }
    let mut kept_points = Vec::new();
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    let mut n_wide = 0usize;
    for (p, obs) in per_pt.iter().enumerate() {
        if obs.len() < 2 {
            continue;
        }
        let cp = kept_points.len() as u32;
        kept_points.push(points[p]);
        for &(i, px, wide) in obs {
            n_wide += wide as usize;
            uv.push(px);
            obs_img.push(i);
            obs_pt.push(cp);
        }
    }
    (
        Scene {
            cam,
            quats,
            trans,
            points: kept_points,
            uv,
            obs_img,
            obs_pt,
        },
        n_wide,
    )
}

/// The composite-map metric on the pinhole's radial coordinate: the largest
/// `f·|δ_a(ρ) − δ_b(ρ)|` in pixels over `[lo, hi]`.
fn pinhole_map_err_px(f: f64, a: &[f64], b: &[f64], lo: f64, hi: f64) -> f64 {
    (0..=200)
        .map(|s| {
            let rho = lo + (hi - lo) * s as f64 / 200.0;
            f * (bspline_delta(a, RHO_MAX, rho) - bspline_delta(b, RHO_MAX, rho)).abs()
        })
        .fold(0.0f64, f64::max)
}

/// `∂(u, v)/∂cᵢ` for the pinhole model as the kernel computes it, against a
/// central difference of the projection in each coefficient.
///
/// The columns are `f·Bᵢ(ρ)·û` because every coefficient enters as
/// `ρ_d = ρ + Σ cᵢ·Bᵢ(ρ)` with `ρ = ρ_xy/rz` read off the ray, never off a
/// pixel radius — the projection is exactly LINEAR in each coefficient, so a
/// central difference must reproduce the column to rounding.
#[test]
fn pinhole_bspline_columns_match_a_central_difference() {
    let f = 250.0;
    let h = 1e-3;
    let n = PLANTED_PINHOLE_BSPLINE.len();
    let mut worst: f64 = 0.0;
    let mut n_past_seam = 0usize;
    for ti in 0..17 {
        let rho = 0.02 + 0.07 * ti as f64; // out to ρ = 1.14, past ρ_max
        for ai in 0..5 {
            let az = 2.0 * std::f64::consts::PI * ai as f64 / 5.0;
            for &scale in &[0.35_f64, 1.0, 7.5] {
                // Optical-frame direction at (ρ, az) → canonical via
                // S = diag(1, −1, −1).
                let theta = rho.atan();
                let opt = Vector3::new(theta.sin() * az.cos(), theta.sin() * az.sin(), theta.cos());
                let r = [scale * opt.x, -scale * opt.y, -scale * opt.z];
                if rho > RHO_MAX {
                    n_past_seam += 1;
                }
                let (first, cols) = bspline_columns(
                    f,
                    n,
                    RHO_MAX,
                    SplineRadial::ImagePlaneRadius,
                    Vector3::new(r[0], r[1], r[2]),
                );
                let mut full = vec![[0.0f64; 2]; n];
                for (j, col) in cols.iter().enumerate() {
                    let fi = first + j;
                    if fi >= 2 {
                        full[fi - 2] = *col;
                    }
                }
                let denom = full
                    .iter()
                    .flatten()
                    .fold(0.0f64, |m, c| m.max(c.abs()))
                    .max(1e-3);
                for (i, col) in full.iter().enumerate() {
                    let mut cp = PLANTED_PINHOLE_BSPLINE;
                    cp[i] += h;
                    let mut cm = PLANTED_PINHOLE_BSPLINE;
                    cm[i] -= h;
                    let (up, vp) = sfmtool_pinhole(f, &cp).ray_to_pixel(r).unwrap();
                    let (um, vm) = sfmtool_pinhole(f, &cm).ray_to_pixel(r).unwrap();
                    let (fd_u, fd_v) = ((up - um) / (2.0 * h), (vp - vm) / (2.0 * h));
                    worst = worst
                        .max((col[0] - fd_u).abs() / denom)
                        .max((col[1] - fd_v).abs() / denom);
                }
            }
        }
    }
    assert!(n_past_seam >= 45, "held-constant tail thin: {n_past_seam}");
    assert!(
        worst < 1e-9,
        "analytic spline columns vs central difference: worst relative error {worst}"
    );
}

/// A scene shot through an expanding lens, handed to the solver with a zero
/// spline: the release recovers the planted coefficients — composite-map-wise
/// above all — and the fit comes back sub-pixel.
#[test]
fn opt_bspline_recovers_a_planted_pinhole_spline() {
    let f = 250.0;
    let (mut s, n_wide) =
        make_pinhole_scene_for(sfmtool_pinhole(f, &PLANTED_PINHOLE_BSPLINE), 10, 500);
    assert!(n_wide >= 100, "scene is not wide enough: {n_wide}");
    s.cam = sfmtool_pinhole(f, &[0.0; 6]);
    let out = run_bspline(&mut s, false, true, &DEFAULT_SCHEDULE);
    assert_eq!(out.focal.to_bits(), f.to_bits(), "the focal was not fixed");
    let map_err = pinhole_map_err_px(f, &out.bspline, &PLANTED_PINHOLE_BSPLINE, 0.05, 0.8);
    assert!(
        map_err < 0.3,
        "recovered composite map off by {map_err} px (spline {:?})",
        out.bspline
    );
    for (i, (c, t)) in out.bspline.iter().zip(&PLANTED_PINHOLE_BSPLINE).enumerate() {
        assert!(
            (c - t).abs() < 0.01,
            "coefficient {i}: {c} (want {t}; full spline {:?})",
            out.bspline
        );
    }
    let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(worst < 0.5, "worst reprojection {worst} px after the rung");
}

/// The staged release the callers actually run: focal and spline together,
/// from a focal several percent off and no spline.
#[test]
fn opt_f_and_opt_bspline_recover_together_on_a_pinhole() {
    let f_true = 250.0;
    let (mut s, _) =
        make_pinhole_scene_for(sfmtool_pinhole(f_true, &PLANTED_PINHOLE_BSPLINE), 10, 500);
    let f_start = 238.0;
    s.cam = sfmtool_pinhole(f_start, &[0.0; 6]);
    // The focal trades against the scene scale: move the structure with it.
    for x in s.points.iter_mut() {
        for v in x.iter_mut() {
            *v *= f_start / f_true;
        }
    }
    for t in s.trans.iter_mut() {
        *t *= f_start / f_true;
    }
    let out = run_bspline(&mut s, true, true, &DEFAULT_SCHEDULE);
    let f_err = (out.focal - f_true).abs() / f_true;
    let map_err = pinhole_map_err_px(f_true, &out.bspline, &PLANTED_PINHOLE_BSPLINE, 0.05, 0.8);
    assert!(
        f_err < 0.01 && map_err < 0.5,
        "co-released f = {} (want {f_true}), composite map off by {map_err} px ({:?})",
        out.focal,
        out.bspline
    );
    let worst = out.residual_norms.iter().cloned().fold(0.0f64, f64::max);
    assert!(worst < 0.5, "worst reprojection {worst} px");
}

/// The fixed point that matters for the promotion
/// SIMPLE_PINHOLE → SFMTOOL_PINHOLE(zero spline): on a scene that really is a
/// pinhole, releasing the spline leaves it at zero and leaves the geometry
/// where it was.
#[test]
fn opt_bspline_holds_at_zero_on_a_pinhole_scene() {
    let f = 250.0;
    let (mut released, _) = make_pinhole_scene_for(sfmtool_pinhole(f, &[0.0; 6]), 10, 500);
    let mut fixed = released.clone();
    // A perturbed start, so both arms have real work to do.
    let perturb = |s: &mut Scene| {
        for (i, q) in s.quats.iter_mut().enumerate() {
            *q = UnitQuaternion::from_scaled_axis(Vector3::new(
                0.01 * jitter(i, 21),
                0.01 * jitter(i, 22),
                0.01 * jitter(i, 23),
            )) * *q;
        }
        for (p, x) in s.points.iter_mut().enumerate() {
            for (c, v) in x.iter_mut().enumerate() {
                *v += 0.05 * jitter(p * 3 + c, 31);
            }
        }
    };
    perturb(&mut released);
    perturb(&mut fixed);
    let out_r = run_bspline(&mut released, false, true, &DEFAULT_SCHEDULE);
    let out_f = run_bspline(&mut fixed, false, false, &DEFAULT_SCHEDULE);
    for (i, c) in out_f.bspline.iter().enumerate() {
        assert_eq!(c.to_bits(), 0.0f64.to_bits(), "unreleased c{i} moved");
    }
    let held = pinhole_map_err_px(f, &out_r.bspline, &[0.0; 6], 0.05, 0.8);
    assert!(
        held < 0.1,
        "the released spline walked off zero on a pinhole scene by {held} px ({:?})",
        out_r.bspline
    );
    // …and the reconstruction is the same one, not a spline-for-geometry
    // trade that happens to end near zero.
    for (i, (a, b)) in released.quats.iter().zip(fixed.quats.iter()).enumerate() {
        assert!(a.angle_to(b) < 1e-4, "image {i} rotation split");
    }
    for (p, (a, b)) in released.points.iter().zip(fixed.points.iter()).enumerate() {
        let d = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
        assert!(d < 1e-3, "point {p} split {d}");
    }
}

/// The focal release admits the model: its dimensionless spline rides on the
/// ray's own `ρ`, so `∂(u, v)/∂f = (u − cx)/f` stays exact and a wrong focal
/// is recovered with the spline fixed.
#[test]
fn opt_f_is_admitted_on_the_spline_pinhole() {
    let f_true = 250.0;
    let (mut s, _) =
        make_pinhole_scene_for(sfmtool_pinhole(f_true, &PLANTED_PINHOLE_BSPLINE), 10, 500);
    let f_start = 232.0;
    s.cam = sfmtool_pinhole(f_start, &PLANTED_PINHOLE_BSPLINE);
    for x in s.points.iter_mut() {
        for v in x.iter_mut() {
            *v *= f_start / f_true;
        }
    }
    for t in s.trans.iter_mut() {
        *t *= f_start / f_true;
    }
    let out = run_bspline(&mut s, true, false, &DEFAULT_SCHEDULE);
    let err = (out.focal - f_true).abs() / f_true;
    assert!(err < 0.005, "released focal {} (want {f_true})", out.focal);
    // The spline came back untouched: `opt_bspline` was not requested.
    for (i, (c, t)) in out.bspline.iter().zip(&PLANTED_PINHOLE_BSPLINE).enumerate() {
        assert_eq!(c.to_bits(), t.to_bits(), "unreleased c{i} moved");
    }
}

/// The gates on the pinhole model: `opt_k1` has nothing to move on it, a
/// spline too short to define carries nothing to release, and SIMPLE_PINHOLE
/// has no spline at all. All three degrade to the solve without the rung, bit
/// for bit.
#[test]
fn the_rungs_are_gated_on_the_spline_pinhole_too() {
    let f = 250.0;
    let (mut k1_r, _) =
        make_pinhole_scene_for(sfmtool_pinhole(f, &PLANTED_PINHOLE_BSPLINE), 6, 250);
    let mut k1_p = k1_r.clone();
    let out_r = run_k1(&mut k1_r, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_k1(&mut k1_p, true, false, &DEFAULT_SCHEDULE);
    assert_eq!(out_r.k1.to_bits(), 0.0f64.to_bits());
    assert_bitwise_equal(&k1_r, &out_r, &k1_p, &out_p);

    let (mut empty_r, _) = make_pinhole_scene_for(sfmtool_pinhole(f, &[]), 6, 250);
    let mut empty_p = empty_r.clone();
    let out_r = run_bspline(&mut empty_r, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_bspline(&mut empty_p, true, false, &DEFAULT_SCHEDULE);
    assert!(out_r.bspline.is_empty());
    assert_bitwise_equal(&empty_r, &out_r, &empty_p, &out_p);

    let (mut plain_r, _) = make_pinhole_scene_for(simple_pinhole_at(f), 6, 250);
    let mut plain_p = plain_r.clone();
    let out_r = run_bspline(&mut plain_r, true, true, &DEFAULT_SCHEDULE);
    let out_p = run_bspline(&mut plain_p, true, false, &DEFAULT_SCHEDULE);
    assert!(out_r.bspline.is_empty());
    assert_bitwise_equal(&plain_r, &out_r, &plain_p, &out_p);
}

/// The step guard on the pinhole's domain: `ρ_d = ρ + δ(ρ)` must stay strictly
/// increasing over `[0, ρ_max]`, and a released solve never lands on a spline
/// that is not.
#[test]
fn pinhole_bspline_step_guard_rejects_a_folded_spline() {
    assert!(bspline_step_admissible(&[0.0; 6], RHO_MAX));
    assert!(bspline_step_admissible(&PLANTED_PINHOLE_BSPLINE, RHO_MAX));
    // With interior knot spans of 0.17, a −0.2 step between adjacent
    // coefficients puts δ' well below −1.
    let folded = [0.0, 0.0, -0.2, -0.4, -0.6, -0.8];
    assert!(!bspline_step_admissible(&folded, RHO_MAX));
    // A fold in the outermost span is rejected too.
    let rim_folded = [0.0, 0.0, 0.0, 0.0, 0.0, -0.3];
    assert!(!bspline_step_admissible(&rim_folded, RHO_MAX));
    let mut bad = PLANTED_PINHOLE_BSPLINE;
    bad[3] = f64::NAN;
    assert!(!bspline_step_admissible(&bad, RHO_MAX));

    let f = 250.0;
    let (mut s, _) = make_pinhole_scene_for(sfmtool_pinhole(f, &PLANTED_PINHOLE_BSPLINE), 10, 500);
    s.cam = sfmtool_pinhole(f, &[0.0; 6]);
    let out = run_bspline(&mut s, true, true, &DEFAULT_SCHEDULE);
    assert!(
        bspline_is_monotone(&out.bspline, RHO_MAX, RHO_MAX),
        "the solve returned a folded spline: {:?} at f = {}",
        out.bspline,
        out.focal
    );
}

/// The pinhole's half of the unsupported-slot contract: a coefficient whose
/// basis span no surviving observation reaches has exactly-zero curvature, its
/// shared slot is pinned per linearization, and the coefficient comes back bit
/// for bit at its input value.
#[test]
fn unsupported_pinhole_bspline_slots_hold_their_input_exactly() {
    let f = 250.0;
    // Sentinels in the outermost two coefficients. With 6 coefficients on
    // `[0, 0.85]` the interior knots sit every 0.17, and those two slots'
    // basis support starts at ρ = 0.51 and ρ = 0.68: invisible below 0.51.
    let sentinels = [0.0, 0.0, 0.0, 0.0, 0.017, -0.008];
    // A perspective capture cropped to ρ ≤ 0.45, a knot span short of the
    // nearer sentinel's support.
    const FIELD: f64 = 0.45;
    let (full, _) = make_pinhole_scene_for(sfmtool_pinhole(f, &[0.0; 6]), 10, 500);
    let mut s = Scene {
        cam: sfmtool_pinhole(f, &sentinels),
        quats: full.quats.clone(),
        trans: full.trans.clone(),
        points: full.points.clone(),
        uv: Vec::new(),
        obs_img: Vec::new(),
        obs_pt: Vec::new(),
    };
    for k in 0..full.uv.len() {
        let r = (full.uv[k][0] - 240.0).hypot(full.uv[k][1] - 240.0);
        if r <= f * FIELD {
            s.uv.push(full.uv[k]);
            s.obs_img.push(full.obs_img[k]);
            s.obs_pt.push(full.obs_pt[k]);
        }
    }
    assert!(s.uv.len() >= 100, "narrow scene too small: {}", s.uv.len());
    let out = run_bspline(&mut s, false, true, &DEFAULT_SCHEDULE);
    // The solve is not degenerate…
    let finite = out.residual_norms.iter().filter(|r| r.is_finite()).count();
    assert!(
        finite >= 100,
        "solve degenerated: {finite} finite residuals"
    );
    // …the unsupported outer coefficients held their inputs exactly…
    assert_eq!(out.bspline[4].to_bits(), sentinels[4].to_bits());
    assert_eq!(out.bspline[5].to_bits(), sentinels[5].to_bits());
    // …and the supported inner ones stayed near the (true) zero.
    let inner_err = pinhole_map_err_px(
        f,
        &out.bspline[..4]
            .iter()
            .chain(&[0.0, 0.0])
            .copied()
            .collect::<Vec<_>>(),
        &[0.0; 6],
        0.02,
        FIELD,
    );
    assert!(
        inner_err < 0.2,
        "supported coefficients drifted by {inner_err} px: {:?}",
        out.bspline
    );
}

/// Direction rows carry the pinhole's spline rung. A point at infinity
/// projects through the very same map, so `∂/∂cᵢ` applies to it unchanged, and
/// where the finite cloud sits near the axis (no basis signal) the far field
/// is the only thing that can recover the spline.
#[test]
fn directions_participate_in_the_pinhole_bspline_rung() {
    let f = 250.0;
    let cam_true = sfmtool_pinhole(f, &PLANTED_PINHOLE_BSPLINE);

    // A near-axis finite cloud in front of an arc of cameras: enough to
    // satisfy the finite-survivor floor, far too little field to fit the
    // spline from.
    let n_img = 8;
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        let ang = 0.12 * (i as f64 - (n_img as f64 - 1.0) / 2.0);
        let center = Vector3::new(ang.sin(), 0.2 * jitter(i, 11), ang.cos() + 6.0);
        let r = UnitQuaternion::face_towards(&center, &Vector3::y()).inverse();
        quats.push(r);
        trans.push(-(r * center));
    }
    let mut points: Vec<[f64; 3]> = Vec::new();
    for p in 0..40 {
        points.push([0.4 * jitter(p, 5), 0.4 * jitter(p, 6), 0.4 * jitter(p, 7)]);
    }
    let n_finite = points.len();
    // Far-field directions spread over the field the spline actually shapes,
    // ρ = 0.15 out to the RHO_LIMIT rim.
    let mut dir_ids = Vec::new();
    for j in 0..60 {
        let rho = 0.15 + (RHO_LIMIT - 0.15) * (j as f64) / 59.0;
        let theta = rho.atan();
        let phi = 2.399_963 * j as f64;
        let d = Vector3::new(
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            -theta.cos(),
        );
        dir_ids.push(points.len());
        points.push([d.x, d.y, d.z]);
    }
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    for (p, x) in points.iter().enumerate() {
        let is_dir = p >= n_finite;
        for i in 0..n_img {
            let xv = Vector3::new(x[0], x[1], x[2]);
            let c = if is_dir {
                quats[i] * xv
            } else {
                quats[i] * xv + trans[i]
            };
            let Some((u, v)) = cam_true.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            if (u - 240.0).hypot(v - 240.0) > f * RHO_LIMIT {
                continue;
            }
            uv.push([u, v]);
            obs_img.push(i as u32);
            obs_pt.push(p as u32);
        }
    }
    let scene = Scene {
        cam: sfmtool_pinhole(f, &[0.0; 6]),
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
    };
    let mask = dir_mask(&scene, &dir_ids);

    // With the directions in the solve, the rung recovers the composite map.
    let mut with_dirs = scene.clone();
    let out = run_bspline_masked(&mut with_dirs, &mask, false, true, &DEFAULT_SCHEDULE);
    let err = pinhole_map_err_px(f, &out.bspline, &PLANTED_PINHOLE_BSPLINE, 0.15, RHO_LIMIT);
    eprintln!("[pinhole-dir-rung] direction rows recovered to {err:.3e} px");
    assert!(
        err < 1.0,
        "composite map off by {err} px from the direction rows ({:?})",
        out.bspline
    );

    // The control: drop every direction observation and the near-axis finite
    // cloud alone cannot see the spline.
    let mut finite_only = scene.clone();
    let keep: Vec<usize> = (0..finite_only.obs_pt.len())
        .filter(|&k| (finite_only.obs_pt[k] as usize) < n_finite)
        .collect();
    finite_only.uv = keep.iter().map(|&k| finite_only.uv[k]).collect();
    finite_only.obs_img = keep.iter().map(|&k| finite_only.obs_img[k]).collect();
    finite_only.obs_pt = keep.iter().map(|&k| finite_only.obs_pt[k]).collect();
    let out_finite = run_bspline(&mut finite_only, false, true, &DEFAULT_SCHEDULE);
    let err_finite = pinhole_map_err_px(
        f,
        &out_finite.bspline,
        &PLANTED_PINHOLE_BSPLINE,
        0.15,
        RHO_LIMIT,
    );
    eprintln!("[pinhole-dir-rung] the finite-only control landed at {err_finite:.3e} px");
    assert!(
        err_finite > 3.0 * err,
        "the near-axis control recovered the spline too well ({err_finite} \
         vs {err} px) - the test no longer isolates the direction rows"
    );
}
