// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::{CameraIntrinsics, CameraModel};

/// A pinhole camera wide enough for the synthetic geometry below.
fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

/// Identity world-to-camera rotation, WXYZ.
const IDENTITY_Q: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

/// Cameras on the x axis looking along -Z: quaternions and the translations
/// `t = -R c` that put each centre where it belongs.
fn views(centres: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>) {
    let mut q = Vec::new();
    let mut t = Vec::new();
    for c in centres {
        q.extend_from_slice(&IDENTITY_Q);
        t.extend_from_slice(&[-c[0], -c[1], -c[2]]);
    }
    (q, t)
}

/// The pixel a world point lands at in a camera at `centre`.
fn project(cam: &CameraIntrinsics, centre: [f64; 3], p: [f64; 3]) -> [f64; 2] {
    let ray = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
    let (u, v) = cam.ray_to_pixel(ray).expect("point projects");
    [u, v]
}

/// One track observed in every listed camera.
fn one_track(
    cam: &CameraIntrinsics,
    centres: &[[f64; 3]],
    world: [f64; 3],
) -> (Vec<f64>, Vec<u32>, Vec<u32>) {
    let mut uv = Vec::new();
    let mut img = Vec::new();
    let mut pt = Vec::new();
    for (i, c) in centres.iter().enumerate() {
        uv.extend_from_slice(&project(cam, *c, world));
        img.push(i as u32);
        pt.push(0u32);
    }
    (uv, img, pt)
}

fn obs<'a>(
    uv: &'a [f64],
    img: &'a [u32],
    pt: &'a [u32],
    q: &'a [f64],
    t: &'a [f64],
    n_tracks: usize,
) -> ObservationSet<'a> {
    ObservationSet {
        uv,
        obs_image: img,
        obs_point: pt,
        quats_wxyz: q,
        translations: t,
        n_tracks,
    }
}

const PAIR: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
const WORLD: [f64; 3] = [0.2, -0.1, -5.0];

// ── Rules off is the solve ────────────────────────────────────────────────

#[test]
fn every_rule_off_is_the_batch_triangulation_solve() {
    // Rays straight into both forms: the estimate has to be the same bits as
    // the solve's own point.
    let dirs: [f64; 9] = [
        0.0, 0.0, -1.0, //
        0.2, 0.0, -1.0, //
        -0.3, 0.1, -1.0,
    ];
    let mut unit_dirs = Vec::new();
    for c in dirs.chunks_exact(3) {
        let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        unit_dirs.extend_from_slice(&[c[0] / n, c[1] / n, c[2] / n]);
    }
    let centres = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
    let offsets = [0usize, 3];
    let want = triangulate_batch(
        &unit_dirs
            .chunks_exact(3)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect::<Vec<_>>(),
        &centres
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect::<Vec<_>>(),
        &offsets,
    );
    let got = estimate_points_from_rays(
        RaySet {
            dirs: &unit_dirs,
            centres: &centres,
            offsets: &offsets,
        },
        None,
        PointRules::default(),
    );
    assert_eq!(got.verdicts, vec![PointVerdict::Finite]);
    assert_eq!(got.xyzw[0][0], want[0].point.x);
    assert_eq!(got.xyzw[0][1], want[0].point.y);
    assert_eq!(got.xyzw[0][2], want[0].point.z);
    assert_eq!(got.xyzw[0][3], 1.0);
    assert_eq!(got.in_front, vec![want[0].in_front_of_all_cameras]);
    assert_eq!(got.census.triangulation_angle_median_deg, None);
}

#[test]
fn the_two_forms_agree_on_the_same_geometry() {
    let cam = camera();
    let (q, t) = views(&PAIR);
    let (uv, img, pt) = one_track(&cam, &PAIR, WORLD);
    let from_obs = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 1),
        None,
        PointRules::default(),
    );
    // The same rays, built the way the observation form builds them.
    let mut dirs = Vec::new();
    let mut centres = Vec::new();
    for (i, c) in PAIR.iter().enumerate() {
        let d = cam.pixel_to_ray(uv[2 * i], uv[2 * i + 1]);
        dirs.extend_from_slice(&d);
        centres.extend_from_slice(c);
    }
    let from_rays = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 2],
        },
        None,
        PointRules::default(),
    );
    assert_eq!(from_obs.xyzw, from_rays.xyzw);
    assert_eq!(from_obs.verdicts, from_rays.verdicts);
}

// ── The rules ─────────────────────────────────────────────────────────────

#[test]
fn a_marked_track_is_never_solved_even_when_its_rays_cross() {
    let cam = camera();
    let (q, t) = views(&PAIR);
    let (uv, img, pt) = one_track(&cam, &PAIR, WORLD);
    let out = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 1),
        Some(&[true]),
        PointRules {
            floor_rad: Some(1e-4),
            cheirality: true,
            ..Default::default()
        },
    );
    assert_eq!(out.verdicts, vec![PointVerdict::Marked]);
    assert_eq!(out.xyzw[0][3], 0.0);
    let n = (out.xyzw[0][0].powi(2) + out.xyzw[0][1].powi(2) + out.xyzw[0][2].powi(2)).sqrt();
    assert!((n - 1.0).abs() < 1e-12);
    assert_eq!(out.census.marked, 1);
    // Marks off, the same track is solved.
    let solved = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 1),
        None,
        PointRules {
            floor_rad: Some(1e-4),
            cheirality: true,
            ..Default::default()
        },
    );
    assert_eq!(solved.verdicts, vec![PointVerdict::Finite]);
}

#[test]
fn a_pair_exactly_at_the_floor_is_not_thin() {
    // Two rays a known angle apart, and a floor set to exactly that angle.
    let theta = 0.2_f64;
    let a = Vector3::new(0.0, 0.0, -1.0);
    let b = Vector3::new(theta.sin(), 0.0, -theta.cos());
    let dirs = [a.x, a.y, a.z, b.x, b.y, b.z];
    let centres = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let at = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 2],
        },
        None,
        PointRules {
            floor_rad: Some(a.dot(&b).clamp(-1.0, 1.0).acos()),
            ..Default::default()
        },
    );
    assert_eq!(
        at.verdicts,
        vec![PointVerdict::Finite],
        "exactly at the floor"
    );
    let inside = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 2],
        },
        None,
        PointRules {
            floor_rad: Some(theta * 1.0001),
            ..Default::default()
        },
    );
    assert_eq!(inside.verdicts, vec![PointVerdict::Thin]);
}

#[test]
fn the_floor_reads_the_same_whether_a_track_has_two_rays_or_twenty() {
    let theta = 0.2_f64;
    let a = Vector3::new(0.0, 0.0, -1.0);
    let b = Vector3::new(theta.sin(), 0.0, -theta.cos());
    let floor = Some(theta * 1.0001);
    let mut wide_dirs = Vec::new();
    let mut wide_centres = Vec::new();
    // Twenty rays spread between the same two extremes.
    for k in 0..20 {
        let f = k as f64 / 19.0;
        let ang = theta * f;
        wide_dirs.extend_from_slice(&[ang.sin(), 0.0, -ang.cos()]);
        wide_centres.extend_from_slice(&[f, 0.0, 0.0]);
    }
    let many = estimate_points_from_rays(
        RaySet {
            dirs: &wide_dirs,
            centres: &wide_centres,
            offsets: &[0, 20],
        },
        None,
        PointRules {
            floor_rad: floor,
            ..Default::default()
        },
    );
    let two = estimate_points_from_rays(
        RaySet {
            dirs: &[a.x, a.y, a.z, b.x, b.y, b.z],
            centres: &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            offsets: &[0, 2],
        },
        None,
        PointRules {
            floor_rad: floor,
            ..Default::default()
        },
    );
    assert_eq!(many.verdicts, two.verdicts);
    assert_eq!(many.verdicts, vec![PointVerdict::Thin]);
}

#[test]
fn a_track_that_is_both_thin_and_behind_reads_thin() {
    // Two nearly parallel rays that meet behind the cameras.
    let dirs = [0.0, 0.0, -1.0, 1e-4, 0.0, -1.0];
    let centres = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let out = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 2],
        },
        None,
        PointRules {
            floor_rad: Some(0.01),
            cheirality: true,
            ..Default::default()
        },
    );
    assert_eq!(out.verdicts, vec![PointVerdict::Thin]);
    // With the floor off the same track reads behind.
    let no_floor = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 2],
        },
        None,
        PointRules {
            cheirality: true,
            ..Default::default()
        },
    );
    assert_eq!(no_floor.verdicts, vec![PointVerdict::Behind]);
}

#[test]
fn cheirality_off_keeps_the_point_and_reports_the_flag() {
    let dirs = [0.0, 0.0, -1.0, 1e-2, 0.0, -1.0];
    let centres = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let out = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 2],
        },
        None,
        PointRules::default(),
    );
    assert_eq!(out.verdicts, vec![PointVerdict::Finite]);
    assert_eq!(out.in_front, vec![false]);
    assert_eq!(out.xyzw[0][3], 1.0);
}

#[test]
fn a_track_under_the_bar_with_a_camera_behind_it_reads_behind() {
    let dirs = [0.0, 0.0, -1.0, 1e-2, 0.0, -1.0];
    let centres = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    // Cheirality runs before the bar, so the bar never sees this track.
    let out = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 2],
        },
        None,
        PointRules {
            cheirality: true,
            bar_px: Some(1e-9),
            ..Default::default()
        },
    );
    assert_eq!(out.verdicts, vec![PointVerdict::Behind]);
    assert_eq!(out.census.over_bar, 0);
}

#[test]
fn the_bar_reads_the_median_of_the_finite_residuals() {
    let cam = camera();
    let triple = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
    let (q, t) = views(&triple);
    let (mut uv, img, pt) = one_track(&cam, &triple, WORLD);
    for k in 0..3 {
        uv[2 * k] += if k == 1 { -20.0 } else { 20.0 };
    }
    let loose = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 1),
        None,
        PointRules {
            bar_px: Some(1e6),
            ..Default::default()
        },
    );
    assert_eq!(loose.verdicts, vec![PointVerdict::Finite]);
    let tight = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 1),
        None,
        PointRules {
            bar_px: Some(0.5),
            ..Default::default()
        },
    );
    assert_eq!(tight.verdicts, vec![PointVerdict::OverBar]);
    assert_eq!(tight.xyzw[0][3], 0.0);
}

#[test]
fn a_track_that_projects_nowhere_is_over_the_bar() {
    // Both rays point away from the camera, so the solved point sits behind
    // every one of them and nothing projects.
    let cam = camera();
    let (q, t) = views(&PAIR);
    let uv = [320.0, 240.0, 320.0, 240.0];
    let out = estimate_points_from_observations(
        &cam,
        obs(&uv, &[0, 1], &[0, 0], &q, &t, 1),
        None,
        PointRules {
            bar_px: Some(1e6),
            ..Default::default()
        },
    );
    // Parallel rays: the minimum-norm point is at the origin, which reprojects
    // nowhere in a camera sitting on it.
    assert_eq!(out.verdicts, vec![PointVerdict::OverBar]);
}

// ── Few ───────────────────────────────────────────────────────────────────

#[test]
fn one_usable_ray_is_a_bearing_or_absent() {
    let dirs = [0.0, 0.0, -1.0];
    let centres = [0.0, 0.0, 0.0];
    let bearing = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 1],
        },
        None,
        PointRules {
            few: FewObservations::Bearing,
            ..Default::default()
        },
    );
    assert_eq!(bearing.verdicts, vec![PointVerdict::Few]);
    assert_eq!(bearing.xyzw, vec![[0.0, 0.0, -1.0, 0.0]]);
    let absent = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 1],
        },
        None,
        PointRules::default(),
    );
    assert_eq!(absent.verdicts, vec![PointVerdict::Few]);
    assert!(absent.xyzw[0].iter().all(|v| v.is_nan()));
}

#[test]
fn no_usable_ray_falls_back_to_the_forward_direction() {
    let out = estimate_points_from_rays(
        RaySet {
            dirs: &[],
            centres: &[],
            offsets: &[0, 0],
        },
        None,
        PointRules {
            few: FewObservations::Bearing,
            ..Default::default()
        },
    );
    assert_eq!(out.verdicts, vec![PointVerdict::Few]);
    assert_eq!(
        out.xyzw,
        vec![[
            FALLBACK_DIRECTION[0],
            FALLBACK_DIRECTION[1],
            FALLBACK_DIRECTION[2],
            0.0
        ]]
    );
}

#[test]
fn few_is_read_before_marks() {
    // A marked track with one ray is absent under `few = absent`, which is what
    // an adjustment's own re-estimation does with it.
    let out = estimate_points_from_rays(
        RaySet {
            dirs: &[0.0, 0.0, -1.0],
            centres: &[0.0, 0.0, 0.0],
            offsets: &[0, 1],
        },
        Some(&[true]),
        PointRules::default(),
    );
    assert_eq!(out.verdicts, vec![PointVerdict::Few]);
    assert!(out.xyzw[0].iter().all(|v| v.is_nan()));
    assert_eq!(out.census.marked, 0);
}

#[test]
fn a_track_no_observation_names_is_a_few_track() {
    let cam = camera();
    let (q, t) = views(&PAIR);
    let (uv, img, _pt) = one_track(&cam, &PAIR, WORLD);
    let out = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &[1, 1], &q, &t, 2),
        None,
        PointRules {
            few: FewObservations::Bearing,
            ..Default::default()
        },
    );
    assert_eq!(out.verdicts[0], PointVerdict::Few);
    assert_eq!(out.verdicts[1], PointVerdict::Finite);
    assert_eq!(out.census.seen, 2);
    assert_eq!(out.census.few, 1);
}

// ── Determinism and shape ─────────────────────────────────────────────────

#[test]
fn the_output_is_in_the_input_order_and_repeats_itself() {
    let cam = camera();
    let quad = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let (q, t) = views(&quad);
    let worlds = [[0.2, -0.1, -5.0], [-0.4, 0.3, -8.0], [0.05, 0.05, -3.0]];
    let mut uv = Vec::new();
    let mut img = Vec::new();
    let mut pt = Vec::new();
    // Interleaved, so the grouping has to unpick the order.
    for (i, c) in quad.iter().enumerate() {
        for (k, w) in worlds.iter().enumerate() {
            uv.extend_from_slice(&project(&cam, *c, *w));
            img.push(i as u32);
            pt.push(k as u32);
        }
    }
    let rules = PointRules {
        floor_rad: Some(1e-4),
        cheirality: true,
        prune_behind: false,
        bar_px: Some(1.0),
        few: FewObservations::Bearing,
    };
    let a = estimate_points_from_observations(&cam, obs(&uv, &img, &pt, &q, &t, 3), None, rules);
    assert_eq!(a.census.finite, 3);
    for (k, w) in worlds.iter().enumerate() {
        for (c, want) in w.iter().enumerate() {
            assert!((a.xyzw[k][c] - want).abs() < 1e-9);
        }
    }
    assert!(a.census.triangulation_angle_median_deg.unwrap() > 0.0);
    let b = estimate_points_from_observations(&cam, obs(&uv, &img, &pt, &q, &t, 3), None, rules);
    assert_eq!(a, b);
}

#[test]
fn an_empty_observation_set_leaves_every_track_absent() {
    let cam = camera();
    let (q, t) = views(&PAIR);
    let out = estimate_points_from_observations(
        &cam,
        obs(&[], &[], &[], &q, &t, 2),
        None,
        PointRules::default(),
    );
    assert_eq!(out.verdicts, vec![PointVerdict::Few, PointVerdict::Few]);
    assert!(out.xyzw.iter().all(|p| p.iter().all(|v| v.is_nan())));
    assert_eq!(out.census.seen, 2);
    assert_eq!(out.census.few, 2);
}

#[test]
fn verdict_codes_are_the_wire_contract() {
    assert_eq!(PointVerdict::Finite.code(), 0);
    assert_eq!(PointVerdict::Marked.code(), 1);
    assert_eq!(PointVerdict::Thin.code(), 2);
    assert_eq!(PointVerdict::Behind.code(), 3);
    assert_eq!(PointVerdict::OverBar.code(), 4);
    assert_eq!(PointVerdict::Few.code(), 5);
    assert_eq!(PointVerdict::FinitePruned.code(), 6);
}

// ── The cheirality prune, read per observation ────────────────────────────

/// A unit ray from `centre` through `p`. `away` reverses it, so `p` sits behind
/// the camera while still lying on the ray's own line: the midpoint solve is
/// unmoved and only the depth's sign changes.
fn ray_through(centre: [f64; 3], p: [f64; 3], away: bool) -> [f64; 3] {
    let d = Vector3::new(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]).normalize();
    let s = if away { -1.0 } else { 1.0 };
    [s * d.x, s * d.y, s * d.z]
}

/// One track's rays and centres, flattened, from `(centre, away)` views that all
/// look along `p`.
fn lines_at(p: [f64; 3], views: &[([f64; 3], bool)]) -> (Vec<f64>, Vec<f64>) {
    let mut dirs = Vec::new();
    let mut centres = Vec::new();
    for (c, away) in views {
        dirs.extend_from_slice(&ray_through(*c, p, *away));
        centres.extend_from_slice(c);
    }
    (dirs, centres)
}

/// The rules the prune is read under: cheirality on, the prune on top of it.
fn prune_rules() -> PointRules {
    PointRules {
        cheirality: true,
        prune_behind: true,
        ..Default::default()
    }
}

/// Four views that see the point and one the point sits behind.
const FIVE: [([f64; 3], bool); 5] = [
    ([0.0, 0.0, 0.0], false),
    ([1.0, 0.0, 0.0], false),
    ([-1.0, 0.0, 0.0], false),
    ([0.0, 1.0, 0.0], false),
    ([0.0, 0.0, -10.0], true),
];

#[test]
fn a_minority_behind_is_dropped_and_the_track_is_rescued() {
    let p = [0.0, 0.0, -5.0];
    let (dirs, centres) = lines_at(p, &FIVE);
    let out = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 5],
        },
        None,
        prune_rules(),
    );
    assert_eq!(out.verdicts, vec![PointVerdict::FinitePruned]);
    assert_eq!(out.xyzw[0][3], 1.0);
    for (c, want) in p.iter().enumerate() {
        assert!((out.xyzw[0][c] - want).abs() < 1e-9, "{:?}", out.xyzw[0]);
    }
    assert_eq!(out.pruned, vec![false, false, false, false, true]);
    assert_eq!(out.in_front, vec![true]);
    assert_eq!(out.census.finite_pruned, 1);
    assert_eq!(out.census.pruned_obs, 1);
    assert_eq!(out.census.finite, 0);
    assert_eq!(out.census.behind, 0);
    // Every bucket still adds up to the tracks seen.
    let c = out.census;
    assert_eq!(
        c.seen,
        c.finite + c.finite_pruned + c.marked + c.thin + c.behind + c.over_bar + c.few
    );
}

#[test]
fn the_prune_off_leaves_the_same_track_a_bearing() {
    let p = [0.0, 0.0, -5.0];
    let (dirs, centres) = lines_at(p, &FIVE);
    let rays = RaySet {
        dirs: &dirs,
        centres: &centres,
        offsets: &[0, 5],
    };
    let off = estimate_points_from_rays(
        rays,
        None,
        PointRules {
            cheirality: true,
            ..Default::default()
        },
    );
    assert_eq!(off.verdicts, vec![PointVerdict::Behind]);
    assert_eq!(off.xyzw[0][3], 0.0);
    assert_eq!(off.pruned, vec![false; 5]);
    assert_eq!(off.census.behind, 1);
    assert_eq!(off.census.pruned_obs, 0);
    // The prune with cheirality off is inert: the rule it reads never fires.
    let no_rule = estimate_points_from_rays(
        rays,
        None,
        PointRules {
            prune_behind: true,
            ..Default::default()
        },
    );
    let plain = estimate_points_from_rays(rays, None, PointRules::default());
    assert_eq!(no_rule, plain);
    assert_eq!(no_rule.verdicts, vec![PointVerdict::Finite]);
}

#[test]
fn a_majority_behind_is_a_bearing_as_it_always_was() {
    let p = [0.0, 0.0, -5.0];
    let views = [
        ([0.0, 0.0, 0.0], false),
        ([1.0, 0.0, 0.0], false),
        ([0.0, 0.0, -10.0], true),
        ([1.0, 0.0, -10.0], true),
        ([-1.0, 0.0, -10.0], true),
    ];
    let (dirs, centres) = lines_at(p, &views);
    let rays = RaySet {
        dirs: &dirs,
        centres: &centres,
        offsets: &[0, 5],
    };
    let on = estimate_points_from_rays(rays, None, prune_rules());
    let off = estimate_points_from_rays(
        rays,
        None,
        PointRules {
            cheirality: true,
            ..Default::default()
        },
    );
    assert_eq!(on.verdicts, vec![PointVerdict::Behind]);
    assert_eq!(on.xyzw, off.xyzw);
    assert_eq!(on.pruned, vec![false; 5]);
    assert_eq!(on.census.pruned_obs, 0);
}

#[test]
fn half_the_track_behind_is_not_a_minority() {
    // The tie: with half the observations behind the point, the track's own
    // observations name no majority to solve on, so nothing is dropped.
    let p = [0.0, 0.0, -5.0];
    let views = [
        ([0.0, 0.0, 0.0], false),
        ([1.0, 0.0, 0.0], false),
        ([0.0, 0.0, -10.0], true),
        ([1.0, 0.0, -10.0], true),
    ];
    let (dirs, centres) = lines_at(p, &views);
    let on = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 4],
        },
        None,
        prune_rules(),
    );
    assert_eq!(on.verdicts, vec![PointVerdict::Behind]);
    assert_eq!(on.pruned, vec![false; 4]);
    // A track of two can never show a minority, so the rescue never leaves a
    // single survivor behind.
    let (d2, c2) = lines_at(p, &[views[0], views[2]]);
    let pair = estimate_points_from_rays(
        RaySet {
            dirs: &d2,
            centres: &c2,
            offsets: &[0, 2],
        },
        None,
        prune_rules(),
    );
    assert_eq!(pair.verdicts, vec![PointVerdict::Behind]);
    assert_eq!(pair.pruned, vec![false; 2]);
}

#[test]
fn a_rescue_re_reads_the_floor_over_the_survivors() {
    // Two nearly parallel rays plus one the point sits behind. The full track
    // clears the floor only because the odd ray widens it; the survivors do
    // not, so the rescue is refused and the track is the bearing it was.
    let p = [0.0, 0.0, -5.0];
    let views = [
        ([0.0, 0.0, 0.0], false),
        ([0.001, 0.0, 0.0], false),
        ([0.0, -10.0, -5.0], true),
    ];
    let (dirs, centres) = lines_at(p, &views);
    let rays = RaySet {
        dirs: &dirs,
        centres: &centres,
        offsets: &[0, 3],
    };
    let floored = estimate_points_from_rays(
        rays,
        None,
        PointRules {
            floor_rad: Some(0.01),
            ..prune_rules()
        },
    );
    assert_eq!(floored.verdicts, vec![PointVerdict::Behind]);
    assert_eq!(floored.pruned, vec![false; 3]);
    // With the floor off the same track is rescued: the floor is the only rule
    // that refused it.
    let open = estimate_points_from_rays(rays, None, prune_rules());
    assert_eq!(open.verdicts, vec![PointVerdict::FinitePruned]);
    assert_eq!(open.pruned, vec![false, false, true]);
}

#[test]
fn a_rescue_re_reads_the_bar_over_the_survivors() {
    // Four views of a point plus a fifth camera the point sits behind. The four
    // pixels are pushed off their projections, so the reduced solve reprojects
    // badly and the bar decides whether the rescue stands.
    let cam = camera();
    let centres = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -10.0],
    ];
    let (q, t) = views(&centres);
    let mut uv = Vec::new();
    for (k, c) in centres[..4].iter().enumerate() {
        let p = project(&cam, *c, WORLD);
        uv.extend_from_slice(&[p[0] + if k == 1 { -20.0 } else { 20.0 }, p[1]]);
    }
    // The fifth camera sees nothing of the point: its ray is the optical axis
    // and the point is behind it.
    uv.extend_from_slice(&[320.0, 240.0]);
    let img: Vec<u32> = (0..5).collect();
    let pt = [0u32; 5];
    let loose = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 1),
        None,
        PointRules {
            bar_px: Some(1e6),
            ..prune_rules()
        },
    );
    assert_eq!(loose.verdicts, vec![PointVerdict::FinitePruned]);
    assert_eq!(loose.pruned, vec![false, false, false, false, true]);
    let tight = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 1),
        None,
        PointRules {
            bar_px: Some(0.5),
            ..prune_rules()
        },
    );
    assert_eq!(tight.verdicts, vec![PointVerdict::Behind]);
    assert_eq!(tight.pruned, vec![false; 5]);
    assert_eq!(tight.census.over_bar, 0, "the bar decided a behind track");
}

#[test]
fn the_prune_mask_indexes_the_caller_s_own_observations() {
    // Two tracks with their observations interleaved: the mask names the rows
    // the caller listed, not a position within a track.
    let cam = camera();
    let centres = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -10.0],
    ];
    let (q, t) = views(&centres);
    let worlds = [[0.2, -0.1, -5.0], [-0.4, 0.3, -8.0]];
    let mut uv = Vec::new();
    let mut img = Vec::new();
    let mut pt = Vec::new();
    for (i, c) in centres.iter().enumerate() {
        for (k, w) in worlds.iter().enumerate() {
            if i == 3 {
                uv.extend_from_slice(&[320.0, 240.0]);
            } else {
                uv.extend_from_slice(&project(&cam, *c, *w));
            }
            img.push(i as u32);
            pt.push(k as u32);
        }
    }
    let out = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 2),
        None,
        prune_rules(),
    );
    assert_eq!(
        out.verdicts,
        vec![PointVerdict::FinitePruned, PointVerdict::FinitePruned]
    );
    // Rows 6 and 7 are the fourth camera's two observations.
    assert_eq!(
        out.pruned,
        vec![false, false, false, false, false, false, true, true]
    );
    assert_eq!(out.census.pruned_obs, 2);
    assert_eq!(out.census.finite_pruned, 2);
    for (k, w) in worlds.iter().enumerate() {
        for (c, want) in w.iter().enumerate() {
            assert!((out.xyzw[k][c] - want).abs() < 1e-9);
        }
    }
    let again = estimate_points_from_observations(
        &cam,
        obs(&uv, &img, &pt, &q, &t, 2),
        None,
        prune_rules(),
    );
    assert_eq!(out, again);
}

#[test]
fn the_rescued_angle_is_the_survivors_own_widest_pair() {
    // The dropped ray is the widest pair of the full track, so an angle read
    // over that track would report one the estimate never rested on.
    let p = [0.0, 0.0, -5.0];
    let views = [
        ([0.0, 0.0, 0.0], false),
        ([1.0, 0.0, 0.0], false),
        ([0.0, -10.0, -5.0], true),
    ];
    let (dirs, centres) = lines_at(p, &views);
    let out = estimate_points_from_rays(
        RaySet {
            dirs: &dirs,
            centres: &centres,
            offsets: &[0, 3],
        },
        None,
        PointRules {
            floor_rad: Some(1e-4),
            ..prune_rules()
        },
    );
    assert_eq!(out.verdicts, vec![PointVerdict::FinitePruned]);
    // The two survivors are 5 units deep and 1 apart.
    let want = (5.0_f64 / 26.0_f64.sqrt()).acos().to_degrees();
    let got = out.census.triangulation_angle_median_deg.unwrap();
    assert!((got - want).abs() < 1e-9, "{got} vs {want}");
}
