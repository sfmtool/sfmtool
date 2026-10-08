// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two normal steps, decided against the textured plane: a patch turned
//! off the plane is turned back toward it.

use super::scene::{fixture_of, Scene, WORLD};
use super::*;
use crate::bench::normal::MAX_PIECES;

/// A half-length that projects to about 16 px in the scene's views, so the
/// pieces a finite difference cuts it into still hold texture to localize.
const HALF: f64 = 0.4;

/// A version holding one point on the plane, seen by all three images, on a
/// patch facing the cameras.
fn three_view_edited(scene: &Scene) -> EditedReconstruction {
    let patch = OrientedPatch::from_center_normal(
        WORLD,
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [HALF, HALF],
    );
    EditedReconstruction::new(Arc::new(fixture_of(scene, WORLD, 1.0, &[0, 1, 2], &patch)))
}

/// The plane's outward normal, toward the cameras.
fn truth() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -1.0)
}

fn degrees(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    a.normalize()
        .dot(&b.normalize())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// The fixture's track, tilted `by` degrees off the plane about the `x` axis.
fn tilted_track(edited: &EditedReconstruction, by: f64) -> EditableTrack {
    let (bench, label) = bench_with_point(edited, 0);
    let track = track_of(&bench, &label);
    let r = by.to_radians();
    let aim = Vector3::new(0.0, r.sin(), -r.cos());
    let (tilted, report) = tilt_patch(&track, edited, aim).expect("a finite track");
    assert!(report.changed && report.stopped.is_none());
    tilted
}

#[test]
fn fit_normal_turns_a_tilted_patch_back_toward_the_plane() {
    let scene = Scene::new();
    let edited = three_view_edited(&scene);
    let track = tilted_track(&edited, 20.0);
    let before = degrees(placement_of(&track).normal(), truth());

    let (turned, report) = fit_normal(
        &track,
        &edited,
        &scene.views(),
        &FitNormalOptions::default(),
        &Progress::none(),
    )
    .expect("three views score a normal");
    let after = degrees(placement_of(&turned).normal(), truth());
    assert!(after < before / 2.0, "{before:.1} -> {after:.1}: {report}");
    let NormalEstimate::Photometric {
        before: z0,
        after: z1,
        views,
    } = report.estimate
    else {
        panic!("a photometric estimate");
    };
    assert!(z1 >= z0, "{z0} -> {z1}");
    assert_eq!(views, 3);
    // The centre does not move, and the result is read back, its bitmap
    // rendered, and every row scored against that bitmap.
    assert!((placement_of(&turned).center - WORLD).norm() < 1e-9);
    assert!(turned.track().expect("track stage").bitmap.is_some());
    assert_eq!(report.evaluate.measured, 3);
    assert_rows_scored_against_the_bitmap(&turned);
}

/// Every row that has a pixel carries a score against the track's bitmap,
/// and the row the bitmap is the render of scores 1.
fn assert_rows_scored_against_the_bitmap(track: &EditableTrack) {
    let reference = track.track().expect("track stage").reference;
    for (i, o) in track.observations.iter().enumerate() {
        let m = o.track.as_ref().expect("a reading");
        let z = m
            .bitmap_zncc
            .unwrap_or_else(|| panic!("row {i} is not scored"));
        if Some(i) == reference {
            assert_eq!(z, 1.0);
        }
    }
}

#[test]
fn finite_difference_normal_turns_a_tilted_patch_back_toward_the_plane() {
    let scene = Scene::new();
    let edited = three_view_edited(&scene);
    let track = tilted_track(&edited, 20.0);
    let before = degrees(placement_of(&track).normal(), truth());

    for (pieces, overlap) in [(2, 0.0), (3, 0.5)] {
        let options = FiniteDifferenceOptions {
            pieces,
            overlap,
            ..Default::default()
        };
        let (turned, report) =
            finite_difference_normal(&track, &edited, &scene.views(), &options, &Progress::none())
                .expect("the pieces fit");
        let after = degrees(placement_of(&turned).normal(), truth());
        assert!(
            after < before / 2.0,
            "{pieces} pieces, {overlap}: {before:.1} -> {after:.1}: {report}"
        );
        assert_eq!(
            report.estimate,
            NormalEstimate::FiniteDifference {
                pieces,
                u: pieces,
                v: pieces,
                both_axes: true,
            }
        );
        assert!((placement_of(&turned).center - WORLD).norm() < 1e-9);
        assert_rows_scored_against_the_bitmap(&turned);
    }
}

#[test]
fn a_grid_of_pieces_turns_a_tilted_patch_back_toward_the_plane() {
    let scene = Scene::new();
    let edited = three_view_edited(&scene);
    let track = tilted_track(&edited, 20.0);
    let before = degrees(placement_of(&track).normal(), truth());

    for (pieces, overlap) in [(2, 0.0), (3, 0.5)] {
        let options = FiniteDifferenceOptions {
            layout: PieceLayout::Grid,
            pieces,
            overlap,
            ..Default::default()
        };
        let (turned, report) =
            finite_difference_normal(&track, &edited, &scene.views(), &options, &Progress::none())
                .expect("the pieces fit");
        let after = degrees(placement_of(&turned).normal(), truth());
        assert!(
            after < before / 2.0,
            "{pieces}x{pieces}, {overlap}: {before:.1} -> {after:.1}: {report}"
        );
        let NormalEstimate::GridPlane {
            pieces: cut,
            fitted,
            both_axes,
            off_plane,
        } = report.estimate
        else {
            panic!("a grid estimate: {report}");
        };
        assert_eq!((cut, fitted, both_axes), (pieces, pieces * pieces, true));
        // The scene is a plane, so the centres lie close to one.
        assert!(off_plane < 0.1, "{off_plane}");
        assert!(report
            .to_string()
            .starts_with(&format!("{pieces}x{pieces} pieces")));
        assert!((placement_of(&turned).center - WORLD).norm() < 1e-9);
    }
}

#[test]
fn the_normal_steps_refuse_what_the_track_alone_rules_out() {
    let scene = Scene::new();
    let edited = three_view_edited(&scene);
    let (bench, label) = bench_with_point(&edited, 0);
    let track = track_of(&bench, &label);
    assert_eq!(normal_preconditions(&track), Ok(()));

    let mut one_in = track.clone();
    for observation in &mut one_in.observations[1..] {
        observation.verdict = Verdict::Out;
    }
    assert_eq!(
        normal_preconditions(&one_in),
        Err(NormalError::TooFewObservations(1))
    );

    let cluster = EditableTrack::empty_cluster();
    assert_eq!(
        normal_preconditions(&cluster),
        Err(NormalError::ClusterStage)
    );

    for (pieces, overlap, refusal) in [
        (1, 0.0, NormalError::BadPieces(1)),
        (MAX_PIECES + 1, 0.0, NormalError::BadPieces(MAX_PIECES + 1)),
        (2, 0.95, NormalError::BadOverlap(0.95)),
    ] {
        let options = FiniteDifferenceOptions {
            pieces,
            overlap,
            ..Default::default()
        };
        assert_eq!(
            finite_difference_normal(&track, &edited, &scene.views(), &options, &Progress::none())
                .err(),
            Some(refusal)
        );
    }
}
