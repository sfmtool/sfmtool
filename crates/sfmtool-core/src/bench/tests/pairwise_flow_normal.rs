// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pairwise-flow normal, decided against synthetic surfaces seen by
//! pinholes that look at the point from around it: a textured plane from a
//! start 40 degrees off, a flat plane, texture whose contrast falls, a crease
//! through the point, a view too oblique to the first round's plane, two
//! views, an answer far from the mean viewing direction, a view four times as
//! far as the others, and the refusals and cancellation.

use std::sync::atomic::{AtomicBool, Ordering};

use super::scene::{fixture_of, Scene, WORLD};
use super::*;
use crate::bench::{
    pairwise_flow_normal, PairwiseFlowCrop, PairwiseFlowNormalOptions, PairwiseFlowOutcome,
    PairwiseFlowRounds,
};
use crate::camera::CameraModel;

/// The photographs' side, pixels.
const SIDE: u32 = 160;
/// The cameras' focal length, pixels.
const FOCAL: f64 = 200.0;
/// How far each camera stands from the point.
const DISTANCE: f64 = 4.0;
/// The patch half-extent: 12.5 px in a view facing the plane from
/// [`DISTANCE`].
const HALF: f64 = 0.25;

fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: f64::from(SIDE) / 2.0,
            principal_point_y: f64::from(SIDE) / 2.0,
        },
        width: SIDE,
        height: SIDE,
        rolling_shutter: None,
    }
}

/// Two unit vectors perpendicular to the unit `n` and to each other.
fn basis(n: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let hint = if n.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let a = (hint - n * n.dot(&hint)).normalize();
    (a, n.cross(&a))
}

/// `n` turned by `deg` degrees toward the azimuth `azimuth_deg` about it.
fn tilted(n: Vector3<f64>, deg: f64, azimuth_deg: f64) -> Vector3<f64> {
    let (a, b) = basis(n);
    let (t, p) = (deg.to_radians(), azimuth_deg.to_radians());
    (n * t.cos() + (a * p.cos() + b * p.sin()) * t.sin()).normalize()
}

fn degrees(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    a.normalize()
        .cross(&b.normalize())
        .norm()
        .atan2(a.normalize().dot(&b.normalize()))
        .to_degrees()
}

/// Camera centres `distance` from the point, `from` degrees off `axis`
/// toward each of `azimuths`.
fn centres(axis: Vector3<f64>, from: f64, azimuths: &[f64], distance: f64) -> Vec<[f64; 3]> {
    azimuths
        .iter()
        .map(|&p| {
            let c = WORLD.coords + tilted(axis, from, p) * distance;
            [c.x, c.y, c.z]
        })
        .collect()
}

/// A smooth texture of `contrast` grey levels standard deviation: 24
/// sinusoids with wavelengths of 3 to 12 px of a view facing it from
/// [`DISTANCE`], times `scale`.
fn texture(a: f64, b: f64, contrast: f64, scale: f64) -> f64 {
    const WAVES: usize = 24;
    let pixel = DISTANCE / FOCAL * scale;
    let amplitude = contrast * (2.0 / WAVES as f64).sqrt();
    (0..WAVES)
        .map(|k| {
            let fraction = k as f64 / (WAVES - 1) as f64;
            let wavelength = pixel * (3.0 + 9.0 * fraction);
            // Directions and phases from the golden angle, so no two waves
            // run alike.
            let angle = k as f64 * 2.399_963;
            let phase = k as f64 * 1.7;
            let f = std::f64::consts::TAU / wavelength;
            amplitude * (f * (a * angle.cos() + b * angle.sin()) + phase).sin()
        })
        .sum()
}

/// White noise of `sd` grey levels, the same for the same ray.
fn grain(c: [f64; 3], d: [f64; 3], sd: f64) -> f64 {
    let mut h = 0xC0FF_EE00_u64;
    for v in c.iter().chain(&d) {
        h = (h ^ v.to_bits()).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 31;
    }
    let u = (h >> 11) as f64 / (1u64 << 53) as f64;
    // A uniform draw of the same variance.
    sd * 3f64.sqrt() * (2.0 * u - 1.0)
}

/// The distance along the ray from `c` along unit `d` to the plane through
/// `WORLD` with unit normal `n`, where it meets it in front.
fn meet(c: Vector3<f64>, d: Vector3<f64>, n: Vector3<f64>) -> Option<f64> {
    let along = n.dot(&d);
    let t = n.dot(&(WORLD.coords - c)) / along;
    (along.abs() > 1e-12 && t > 0.0).then_some(t)
}

/// A textured plane through `WORLD` with unit normal `n`.
fn plane(n: Vector3<f64>, contrast: f64, noise: f64) -> impl Fn([f64; 3], [f64; 3]) -> f64 {
    let (e1, e2) = basis(n);
    move |c, d| {
        let (cv, dv) = (Vector3::from(c), Vector3::from(d));
        let value = match meet(cv, dv, n) {
            Some(t) => {
                let p = cv + dv * t - WORLD.coords;
                128.0 + texture(p.dot(&e1), p.dot(&e2), contrast, 1.0)
            }
            None => 128.0,
        };
        value + grain(c, d, noise)
    }
}

/// Every image of `scene` observing one point at `WORLD`, on a patch facing
/// `facing`.
fn edited_of(scene: &Scene, facing: Vector3<f64>) -> EditedReconstruction {
    let (_, up) = basis(facing);
    let patch = OrientedPatch::from_center_normal(WORLD, facing, up, [HALF, HALF]);
    let observing: Vec<u32> = (0..scene.len() as u32).collect();
    EditedReconstruction::new(Arc::new(fixture_of(scene, WORLD, 1.0, &observing, &patch)))
}

/// The fixture's track with its normal set to `normal`.
fn track_facing(edited: &EditedReconstruction, normal: Vector3<f64>) -> EditableTrack {
    let (bench, label) = bench_with_point(edited, 0);
    let track = track_of(&bench, &label);
    let (tilted, _) = tilt_patch(&track, edited, normal).expect("a finite track");
    assert!(degrees(patch_frame_of(&tilted).normal(), normal) < 1e-6);
    tilted
}

fn run(
    scene: &Scene,
    edited: &EditedReconstruction,
    track: &EditableTrack,
    options: &PairwiseFlowNormalOptions,
) -> (EditableTrack, NormalReport) {
    pairwise_flow_normal(track, edited, &scene.views(), options, &Progress::none())
        .expect("the step runs on a finite track")
}

fn rounds_of(report: &NormalReport) -> &PairwiseFlowRounds {
    match &report.estimate {
        NormalEstimate::PairwiseFlow(rounds) => rounds,
        other => panic!("a pairwise flow estimate, not {other:?}"),
    }
}

/// The plane's true normal, and four cameras 25° off it, 90° apart in
/// azimuth: pairs 35° and 50° apart.
fn truth() -> Vector3<f64> {
    Vector3::new(0.2, -0.1, -1.0).normalize()
}

fn four_views() -> Vec<[f64; 3]> {
    centres(truth(), 25.0, &[0.0, 90.0, 180.0, 270.0], DISTANCE)
}

fn describe(rounds: &PairwiseFlowRounds) -> String {
    let mut s = format!("{rounds}\n");
    for (k, round) in rounds.rounds.iter().enumerate() {
        s += &format!(
            "round {k}: plane {:?} outcome {:?} radius {:?} from mean {:?}\n",
            round.plane_normal, round.outcome, round.chosen_radius, round.from_mean_view_deg
        );
        for r in &round.radii {
            s += &format!(
                "  r {} fitted {} determined {} combined {} chi2 {:.2} sigma {:.2}\n",
                r.radius,
                r.pairs_fitted,
                r.pairs_determined,
                r.pairs_combined,
                r.mean_chi2,
                r.sigma_deg
            );
        }
    }
    s
}

/// The plane of `round` against `truth`, degrees.
fn plane_off(round: &crate::bench::PairwiseFlowRound, truth: Vector3<f64>) -> f64 {
    degrees(Vector3::from(round.plane_normal), truth)
}

/// The largest σ among the pairs combined at the last round's chosen radius.
fn largest_combined_sigma(rounds: &PairwiseFlowRounds) -> f64 {
    let last = rounds.last().expect("a round ran");
    let k = last
        .radii
        .iter()
        .position(|r| Some(r.radius) == last.chosen_radius)
        .expect("a chosen radius");
    last.pairs
        .iter()
        .filter_map(|p| p.fits[k].as_ref())
        .filter(|f| f.combined)
        .map(|f| f.sigma_deg)
        .fold(0.0, f64::max)
}

/// Four views of a textured plane, given a normal 40 degrees off in two
/// directions: each round renders on a plane closer to the truth than the
/// last, and the patch is turned to within a degree of it (0.01° here),
/// within 3 times the largest σ among the pairs combined at the chosen
/// radius.
#[test]
fn a_textured_plane_given_a_normal_40_degrees_off_is_turned_onto_it() {
    let scene = Scene::looking_at(&four_views(), WORLD, camera(), plane(truth(), 30.0, 2.0));
    let edited = edited_of(&scene, truth());
    for azimuth in [0.0, 135.0] {
        let track = track_facing(&edited, tilted(truth(), 40.0, azimuth));
        let (turned, report) = run(
            &scene,
            &edited,
            &track,
            &PairwiseFlowNormalOptions::default(),
        );
        let rounds = rounds_of(&report);
        let detail = describe(rounds);
        assert_eq!(rounds.outcome, PairwiseFlowOutcome::Turned, "{detail}");
        assert_eq!(rounds.rounds.len(), 3, "{detail}");
        let offs: Vec<f64> = rounds
            .rounds
            .iter()
            .map(|r| plane_off(r, truth()))
            .collect();
        assert!((offs[0] - 40.0).abs() < 1e-6, "{offs:?}");
        assert!(offs.windows(2).all(|w| w[1] < w[0]), "{offs:?}");
        let after = degrees(patch_frame_of(&turned).normal(), truth());
        assert!(after < 1.0, "{after}° off\n{detail}");
        assert!(
            after < 3.0 * largest_combined_sigma(rounds),
            "{after}°\n{detail}"
        );
        // The patch was turned to the step's answer, the full turn.
        let answer = Vector3::from(rounds.normal.expect("turned"));
        assert!(degrees(patch_frame_of(&turned).normal(), answer) < 1e-6);
        assert!(
            (rounds.turn_deg.unwrap() - degrees(tilted(truth(), 40.0, azimuth), answer)).abs()
                < 1e-9
        );
        // Every view was rendered in every round, and every pair fitted.
        for round in &rounds.rounds {
            assert_eq!(round.rendered(), 4, "{detail}");
            assert_eq!(round.pairs.len(), 6, "{detail}");
        }
        assert!(
            report.to_string().contains("6 of 6 pairs agree at"),
            "{report}"
        );
        // The centre does not move, and the track is read back.
        assert!((patch_frame_of(&turned).center - WORLD).norm() < 1e-9);
        assert!(turned.track().expect("track stage").bitmap.is_some());
    }
}

/// A plane of constant grey with noise of 2 grey levels: no pair is
/// determined at any radius, the outcome is undetermined and the track keeps
/// the normal it was given. With two views, one pair, it is undetermined
/// too, not a single pair, after one round.
#[test]
fn a_flat_plane_is_undetermined_and_keeps_its_normal() {
    let start = tilted(truth(), 15.0, 60.0);
    for views in [four_views(), four_views()[..2].to_vec()] {
        let scene = Scene::looking_at(&views, WORLD, camera(), plane(truth(), 0.0, 2.0));
        let edited = edited_of(&scene, truth());
        let track = track_facing(&edited, start);
        let (turned, report) = run(
            &scene,
            &edited,
            &track,
            &PairwiseFlowNormalOptions::default(),
        );
        let rounds = rounds_of(&report);
        let detail = describe(rounds);
        assert_eq!(
            rounds.outcome,
            PairwiseFlowOutcome::Undetermined,
            "{detail}"
        );
        for round in &rounds.rounds {
            assert_eq!(round.outcome, PairwiseFlowOutcome::Undetermined, "{detail}");
            assert!(
                round.radii.iter().all(|r| r.pairs_determined == 0),
                "{detail}"
            );
            assert!(round.rendered() == views.len(), "{detail}");
        }
        assert_eq!(rounds.normal, None);
        assert!(degrees(patch_frame_of(&turned).normal(), start) < 1e-9);
        assert!(report
            .to_string()
            .contains("normal left unchanged: undetermined"));
        if views.len() == 2 {
            assert_eq!(rounds.rounds.len(), 1, "{detail}");
        }
    }
}

/// Texture whose contrast falls from 12 to 0.5 grey levels under noise of
/// 2: σ grows as the contrast falls, the outcome passes from turned to
/// undetermined, it is never unstable, and where it is turned the normal is
/// within 3 times the largest σ among the pairs combined at the chosen
/// radius.
#[test]
fn weak_texture_is_turned_within_its_sigma_or_undetermined_and_never_unstable() {
    let start = tilted(truth(), 20.0, 200.0);
    let mut seen = Vec::new();
    let mut sigma = Vec::new();
    for contrast in [12.0, 4.0, 2.0, 1.0, 0.5] {
        let scene = Scene::looking_at(
            &four_views(),
            WORLD,
            camera(),
            plane(truth(), contrast, 2.0),
        );
        let edited = edited_of(&scene, truth());
        let track = track_facing(&edited, start);
        let (turned, report) = run(
            &scene,
            &edited,
            &track,
            &PairwiseFlowNormalOptions::default(),
        );
        let rounds = rounds_of(&report);
        let detail = describe(rounds);
        assert_ne!(rounds.outcome, PairwiseFlowOutcome::Unstable, "{detail}");
        for round in &rounds.rounds {
            assert_ne!(round.outcome, PairwiseFlowOutcome::Unstable, "{detail}");
        }
        let after = degrees(patch_frame_of(&turned).normal(), truth());
        if rounds.outcome == PairwiseFlowOutcome::Turned {
            let bound = 3.0 * largest_combined_sigma(rounds);
            assert!(
                after < bound,
                "contrast {contrast}: {after}° against {bound}°\n{detail}"
            );
        }
        // The median σ of the pairs' fits at 1 half-extent in the first round,
        // which every contrast renders on the same plane.
        let first = &rounds.rounds[0];
        let k = first.radii.iter().position(|r| r.radius == 1.0).unwrap();
        let sigmas: Vec<f64> = first
            .pairs
            .iter()
            .filter_map(|p| p.fits[k].as_ref())
            .map(|f| f.sigma_deg)
            .collect();
        sigma.push(crate::numeric::median(&sigmas));
        seen.push(rounds.outcome);
    }
    assert_eq!(seen[0], PairwiseFlowOutcome::Turned, "{seen:?}");
    assert_eq!(
        seen[seen.len() - 1],
        PairwiseFlowOutcome::Undetermined,
        "{seen:?}"
    );
    // While turned, σ grows as the contrast falls, and every undetermined
    // contrast's σ is larger than every turned one's.
    let turned: Vec<f64> = (0..seen.len())
        .filter(|&k| seen[k] == PairwiseFlowOutcome::Turned)
        .map(|k| sigma[k])
        .collect();
    assert!(turned.windows(2).all(|w| w[1] > w[0]), "{sigma:?} {seen:?}");
    let least_undetermined = (0..seen.len())
        .filter(|&k| seen[k] == PairwiseFlowOutcome::Undetermined)
        .map(|k| sigma[k])
        .fold(f64::INFINITY, f64::min);
    assert!(
        turned.iter().all(|&s| s < least_undetermined),
        "{sigma:?} {seen:?}"
    );
}

/// The faces of a ridge through the point: two textured planes meeting at
/// 90 degrees along a line through it, each 45 degrees off `bisector`.
fn ridge(bisector: Vector3<f64>, contrast: f64) -> impl Fn([f64; 3], [f64; 3]) -> f64 {
    let (a, _) = basis(bisector);
    let faces = [tilted(bisector, 45.0, 0.0), tilted(bisector, 45.0, 180.0)];
    let bases = faces.map(basis);
    move |c, d| {
        let (cv, dv) = (Vector3::from(c), Vector3::from(d));
        let hit = (0..2)
            .filter_map(|k| {
                let t = meet(cv, dv, faces[k])?;
                let p = cv + dv * t - WORLD.coords;
                let side = p.dot(&a);
                let on = if k == 0 { side >= 0.0 } else { side <= 0.0 };
                on.then_some((t, k, p))
            })
            .min_by(|x, y| x.0.total_cmp(&y.0));
        let value = match hit {
            Some((_, k, p)) => {
                let (e1, e2) = bases[k];
                128.0 + texture(p.dot(&e1), p.dot(&e2), contrast, 1.0)
            }
            None => 128.0,
        };
        value + grain(c, d, 2.0)
    }
}

/// A ridge through the point, five cameras in front of its edge spread to
/// both sides: the pairs agree on small discs and disagree as the disc
/// grows, so the patch is turned near the bisector of the faces at a small
/// radius, and the mean χ² rises steeply beyond it.
#[test]
fn a_crease_through_the_point_is_turned_near_its_bisector_at_a_small_radius() {
    let bisector = truth();
    let views = centres(bisector, 25.0, &[0.0, 72.0, 144.0, 216.0, 288.0], DISTANCE);
    let scene = Scene::looking_at(&views, WORLD, camera(), ridge(bisector, 30.0));
    let edited = edited_of(&scene, bisector);
    let track = track_facing(&edited, tilted(bisector, 20.0, 90.0));
    let (turned, report) = run(
        &scene,
        &edited,
        &track,
        &PairwiseFlowNormalOptions::default(),
    );
    let rounds = rounds_of(&report);
    let detail = describe(rounds);
    assert_eq!(rounds.outcome, PairwiseFlowOutcome::Turned, "{detail}");
    let chosen = rounds.chosen_radius.expect("turned");
    assert!(chosen <= 0.75, "{detail}");
    // 1.6° here, against the faces' 45°.
    let after = degrees(patch_frame_of(&turned).normal(), bisector);
    assert!(after < 3.0, "{after}° off the bisector\n{detail}");
    // Past the chosen radius every radius with two determined pairs
    // disagrees, and the χ² at the largest of them is far above the bar.
    let last = rounds.last().unwrap();
    let beyond: Vec<f64> = last
        .radii
        .iter()
        .filter(|r| r.radius > chosen && r.pairs_determined >= 2)
        .map(|r| r.mean_chi2)
        .collect();
    assert!(beyond.len() >= 2, "{detail}");
    assert!(beyond.iter().all(|&c| c > 2.0), "{detail}");
    assert!(beyond[beyond.len() - 1] > 10.0, "{detail}");
}

/// Three views, given a normal from which one is more than
/// `max_view_angle_deg`, lowered to 60 degrees, away: the first round
/// renders two crops, one pair, so it has no answer and passes on its
/// fallback; from that plane all three views are in, so the step runs every
/// round, is not a single pair, and turns the patch onto the plane.
#[test]
fn a_view_left_out_as_too_oblique_is_not_a_single_pair() {
    let views = vec![
        centres(truth(), 25.0, &[0.0], DISTANCE)[0],
        centres(truth(), 25.0, &[120.0], DISTANCE)[0],
        centres(truth(), 35.0, &[180.0], DISTANCE)[0],
    ];
    let scene = Scene::looking_at(&views, WORLD, camera(), plane(truth(), 30.0, 2.0));
    let edited = edited_of(&scene, truth());
    let track = track_facing(&edited, tilted(truth(), 35.0, 0.0));
    let options = PairwiseFlowNormalOptions {
        max_view_angle_deg: 60.0,
        ..Default::default()
    };
    let (turned, report) = run(&scene, &edited, &track, &options);
    let rounds = rounds_of(&report);
    let detail = describe(rounds);
    let first = &rounds.rounds[0];
    assert!(
        matches!(first.crops[2], PairwiseFlowCrop::TooOblique { angle_deg, .. } if angle_deg > 60.0),
        "{detail}"
    );
    assert_eq!(first.rendered(), 2, "{detail}");
    assert_eq!(first.pairs.len(), 1, "{detail}");
    assert_eq!(first.pairs[0].observations, [0, 1], "{detail}");
    assert_eq!(first.outcome, PairwiseFlowOutcome::Undetermined, "{detail}");
    assert!(first.next_plane.is_some(), "{detail}");
    assert_eq!(rounds.rounds.len(), 3, "{detail}");
    assert_eq!(rounds.rounds[1].rendered(), 3, "{detail}");
    assert_eq!(rounds.outcome, PairwiseFlowOutcome::Turned, "{detail}");
    assert!(
        degrees(patch_frame_of(&turned).normal(), truth()) < 1.0,
        "{detail}"
    );
}

/// Two views of a textured plane: one round, rendered on the given normal,
/// read as a single pair and turned.
#[test]
fn one_pair_runs_one_round() {
    let views = centres(truth(), 25.0, &[0.0, 180.0], DISTANCE);
    let scene = Scene::looking_at(&views, WORLD, camera(), plane(truth(), 30.0, 2.0));
    let edited = edited_of(&scene, truth());
    let start = tilted(truth(), 20.0, 45.0);
    let track = track_facing(&edited, start);
    let (turned, report) = run(
        &scene,
        &edited,
        &track,
        &PairwiseFlowNormalOptions::default(),
    );
    let rounds = rounds_of(&report);
    let detail = describe(rounds);
    assert_eq!(rounds.rounds.len(), 1, "{detail}");
    assert!(plane_off(&rounds.rounds[0], start) < 1e-9, "{detail}");
    assert_eq!(rounds.outcome, PairwiseFlowOutcome::SinglePair, "{detail}");
    assert_eq!(rounds.chosen_radius, Some(1.0));
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(
        after < 3.0 * rounds.sigma_deg.unwrap(),
        "{after}°\n{detail}"
    );
    assert!(report.to_string().contains("single pair"), "{report}");
}

/// Four views all 30 to 50 degrees to one side of a textured plane, the bar
/// on the mean viewing direction lowered to 20 degrees: the plane, about 40
/// degrees from it, is no answer, so the step stops after one round as
/// unstable and the track keeps the normal it was given.
#[test]
fn an_answer_too_far_from_the_mean_viewing_direction_is_unstable() {
    let side = tilted(truth(), 40.0, 0.0);
    let views = centres(side, 10.0, &[0.0, 90.0, 180.0, 270.0], DISTANCE);
    let scene = Scene::looking_at(&views, WORLD, camera(), plane(truth(), 30.0, 2.0));
    let edited = edited_of(&scene, truth());
    let start = side;
    let track = track_facing(&edited, start);
    let options = PairwiseFlowNormalOptions {
        max_normal_from_mean_view_deg: 20.0,
        ..Default::default()
    };
    let (turned, report) = run(&scene, &edited, &track, &options);
    let rounds = rounds_of(&report);
    let detail = describe(rounds);
    assert_eq!(rounds.outcome, PairwiseFlowOutcome::Unstable, "{detail}");
    assert_eq!(rounds.rounds.len(), 1, "{detail}");
    let first = &rounds.rounds[0];
    assert!(first.from_mean_view_deg.unwrap() > 20.0, "{detail}");
    assert_eq!(first.next_plane, None);
    assert!(degrees(patch_frame_of(&turned).normal(), start) < 1e-9);
    assert!(
        report
            .to_string()
            .contains("from the mean viewing direction"),
        "{report}"
    );
    // At the default bar the same scene is turned onto the plane.
    let (turned, report) = run(
        &scene,
        &edited,
        &track,
        &PairwiseFlowNormalOptions::default(),
    );
    assert_eq!(
        rounds_of(&report).outcome,
        PairwiseFlowOutcome::Turned,
        "{report}"
    );
    assert!(degrees(patch_frame_of(&turned).normal(), truth()) < 1.0);
}

/// A textured plane seen by three cameras at the usual distance and one four
/// times as far: the far crop's pixel spans about 4 grid samples, so each
/// pair with it is read at level 2 of the crops' pyramids, on a square a
/// quarter as wide, and still fits; the step turns the patch onto the plane.
#[test]
fn a_view_four_times_as_far_is_read_at_level_two() {
    let mut views = centres(truth(), 25.0, &[0.0, 120.0, 240.0], DISTANCE);
    views.push(centres(truth(), 20.0, &[60.0], 4.0 * DISTANCE)[0]);
    let n = truth();
    let (e1, e2) = basis(n);
    let coarse = move |c: [f64; 3], d: [f64; 3]| {
        let (cv, dv) = (Vector3::from(c), Vector3::from(d));
        let value = match meet(cv, dv, n) {
            Some(t) => {
                let p = cv + dv * t - WORLD.coords;
                128.0 + texture(p.dot(&e1), p.dot(&e2), 30.0, 4.0)
            }
            None => 128.0,
        };
        value + grain(c, d, 2.0)
    };
    let scene = Scene::looking_at(&views, WORLD, camera(), coarse);
    let edited = edited_of(&scene, truth());
    let track = track_facing(&edited, tilted(truth(), 15.0, 300.0));
    let (turned, report) = run(
        &scene,
        &edited,
        &track,
        &PairwiseFlowNormalOptions::default(),
    );
    let rounds = rounds_of(&report);
    let detail = describe(rounds);
    for round in &rounds.rounds {
        let PairwiseFlowCrop::Rendered {
            samples_per_pixel, ..
        } = round.crops[3]
        else {
            panic!("the far view is rendered\n{detail}");
        };
        assert!(
            (3.5..4.5).contains(&samples_per_pixel),
            "{samples_per_pixel}"
        );
        for pair in &round.pairs {
            let fits = pair.fits.iter().flatten().count();
            if pair.observations[1] == 3 {
                // The square read is a quarter of the crop's side.
                assert_eq!(pair.level, 2, "{pair:?}");
                assert_eq!(pair.half_width, round.crop_half_width / 4, "{pair:?}");
                assert!(fits >= 3, "{pair:?}");
            } else {
                assert_eq!(pair.level, 0, "{pair:?}");
                assert_eq!(pair.half_width, round.crop_half_width, "{pair:?}");
            }
        }
    }
    // The far view's pairs are determined and combined at the chosen radius.
    let last = rounds.last().unwrap();
    let k = last
        .radii
        .iter()
        .position(|r| Some(r.radius) == rounds.chosen_radius)
        .expect("turned");
    let far_combined = last
        .pairs
        .iter()
        .filter(|p| p.observations[1] == 3)
        .filter(|p| p.fits[k].as_ref().is_some_and(|f| f.combined))
        .count();
    assert!(far_combined >= 1, "{detail}");
    assert_eq!(rounds.outcome, PairwiseFlowOutcome::Turned, "{detail}");
    // 0.03° here.
    let after = degrees(patch_frame_of(&turned).normal(), truth());
    assert!(after < 1.0, "{after}°\n{detail}");
}

/// A track at infinity is refused as every normal step refuses it; a setting
/// the check refuses is named; a set cancel flag stops the step between
/// pairs.
#[test]
fn the_step_refuses_a_bearing_and_bad_options_and_stops_when_cancelled() {
    let scene = Scene::looking_at(&four_views(), WORLD, camera(), plane(truth(), 30.0, 2.0));
    let edited = edited_of(&scene, truth());
    let track = track_facing(&edited, truth());
    let options = PairwiseFlowNormalOptions::default();
    let bearing = as_bearing(&track, &edited);
    assert_eq!(
        pairwise_flow_normal(
            &bearing,
            &edited,
            &scene.views(),
            &options,
            &Progress::none()
        )
        .err(),
        Some(NormalError::AtInfinity)
    );
    let bad = PairwiseFlowNormalOptions {
        rounds: 0,
        ..Default::default()
    };
    assert_eq!(
        pairwise_flow_normal(&track, &edited, &scene.views(), &bad, &Progress::none()).err(),
        Some(NormalError::BadPairwiseFlowOption("rounds"))
    );

    // The flag is set when the first pair is counted done, so the step is
    // stopped between pairs, not before it starts.
    let cancel = AtomicBool::new(false);
    let pairs_done = std::sync::atomic::AtomicUsize::new(0);
    let sink = |event: crate::progress::Event<'_>| {
        if let crate::progress::Event::Count { unit: "pair", .. } = event {
            pairs_done.fetch_add(1, Ordering::SeqCst);
            cancel.store(true, Ordering::SeqCst);
        }
    };
    let progress = Progress::to(&sink).cancelled_by(&cancel);
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let result = one_thread
        .install(|| pairwise_flow_normal(&track, &edited, &scene.views(), &options, &progress));
    assert_eq!(result.err(), Some(NormalError::Cancelled));
    assert_eq!(pairs_done.load(Ordering::SeqCst), 1);
}
