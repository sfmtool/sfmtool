// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::geometry::focal_vote::tests::{
    baseline_cameras, emit_fisheye_parallax_pair, fisheye_scene, two_subcapture_scene, Lcg, Obs,
    F_TRUE, H, W,
};
use crate::geometry::focal_vote::ScanCell;

/// Run the estimator over a synthetic capture, both columns, seed 0.
fn estimate(obs: &Obs, options: &IntrinsicsOptions) -> IntrinsicsEstimate {
    estimate_intrinsics(&obs.cluster, &obs.image, &obs.pos, W, H, options)
}

/// A fisheye capture with parallax but NO far-field rotation pairs: every
/// candidate is a baseline pair, so the equidistant column's rotation cell has
/// nothing to certify and the verdict's rotation mass is zero -- structurally
/// the shape a false fisheye verdict has.
fn parallax_only_fisheye_scene(seed: u64) -> Obs {
    let mut rng = Lcg(seed);
    let n = 8;
    let cams = baseline_cameras(n, 1.0, &mut rng);
    let mut obs = Obs::default();
    for i in 0..n - 1 {
        emit_fisheye_parallax_pair(&mut obs, &cams, i, i + 1, 60, &mut rng);
    }
    for i in 0..n - 2 {
        emit_fisheye_parallax_pair(&mut obs, &cams, i, i + 2, 60, &mut rng);
    }
    obs
}

#[test]
fn fisheye_verdict_with_rotation_mass_is_confirmed() {
    let est = estimate(&fisheye_scene(2718), &IntrinsicsOptions::default());
    assert_eq!(est.camera_model, Some(CameraModel::EquidistantFisheye));
    let fish = column(&est.vote, CameraModel::EquidistantFisheye).expect("equidistant column");
    assert!(
        fish.n_certified_rotation >= 1,
        "a real fisheye carries certified rotation mass, got {}",
        fish.n_certified_rotation
    );
    assert_eq!(est.confirmed, Some(true));
}

#[test]
fn fisheye_verdict_without_rotation_mass_is_unconfirmed() {
    let est = estimate(
        &parallax_only_fisheye_scene(31337),
        &IntrinsicsOptions::default(),
    );
    assert_eq!(
        est.camera_model,
        Some(CameraModel::EquidistantFisheye),
        "{:?}",
        est.vote.columns
    );
    let fish = column(&est.vote, CameraModel::EquidistantFisheye).expect("equidistant column");
    // A capture with no far-field pair certifies no rotation cell at all, and a
    // wrong ray map cannot fake one, so the verdict rests on nothing but the
    // arbitration.
    assert_eq!(fish.n_certified_rotation, 0);
    assert_eq!(est.confirmed, Some(false));
}

#[test]
fn min_rotation_mass_floor_is_respected() {
    let obs = fisheye_scene(2718);
    let mass = {
        let est = estimate(&obs, &IntrinsicsOptions::default());
        column(&est.vote, CameraModel::EquidistantFisheye)
            .expect("equidistant column")
            .n_certified_rotation
    };
    assert!(mass >= 1);

    // Exactly at the floor still confirms; one above it does not.
    let at_floor = estimate(
        &obs,
        &IntrinsicsOptions {
            min_rotation_mass: mass,
            ..Default::default()
        },
    );
    assert_eq!(at_floor.confirmed, Some(true));
    let above_floor = estimate(
        &obs,
        &IntrinsicsOptions {
            min_rotation_mass: mass + 1,
            ..Default::default()
        },
    );
    assert_eq!(above_floor.confirmed, Some(false));
    // The verdict itself is untouched by the floor -- confirmation is a
    // separate question from which column won.
    assert_eq!(above_floor.camera_model, at_floor.camera_model);
    assert_eq!(
        above_floor.focal_px.map(f64::to_bits),
        at_floor.focal_px.map(f64::to_bits)
    );
}

#[test]
fn pinhole_verdict_has_no_confirmation_question() {
    let est = estimate(
        &two_subcapture_scene(6, 8, F_TRUE, F_TRUE, 99),
        &IntrinsicsOptions::default(),
    );
    assert_eq!(est.camera_model, Some(CameraModel::Pinhole));
    assert_eq!(est.confirmed, None);
    // The equidistant column ran and even certified votes; that is simply not
    // the question a pinhole verdict raises.
    assert!(column(&est.vote, CameraModel::EquidistantFisheye).is_some());
}

#[test]
fn single_column_run_arbitrates_nothing() {
    let obs = fisheye_scene(2718);
    let est = estimate(
        &obs,
        &IntrinsicsOptions {
            vote: FocalVoteOptions {
                columns: vec![CameraModel::EquidistantFisheye],
                ..Default::default()
            },
            ..Default::default()
        },
    );
    // One column is the verdict by construction, so there is no arbitration to
    // corroborate even when the rotation mass is there.
    assert_eq!(est.camera_model, Some(CameraModel::EquidistantFisheye));
    assert_eq!(est.confirmed, None);
}

#[test]
fn verdict_votes_belong_to_the_winning_column() {
    let est = estimate(&fisheye_scene(2718), &IntrinsicsOptions::default());
    assert_eq!(est.camera_model, Some(CameraModel::EquidistantFisheye));
    let fish = column(&est.vote, CameraModel::EquidistantFisheye).expect("equidistant column");

    // Exactly the equidistant column's certified scans, in its stored order.
    let expected: Vec<(ScanCell, u32, u32, u64)> = fish
        .scan_votes
        .iter()
        .filter(|v| v.certified)
        .map(|v| (v.cell, v.image_a, v.image_b, v.focal_px.to_bits()))
        .collect();
    let got: Vec<(ScanCell, u32, u32, u64)> = est
        .verdict_votes
        .iter()
        .map(|v| (v.cell, v.image_a, v.image_b, v.focal_px.to_bits()))
        .collect();
    assert_eq!(got, expected);
    assert!(est.verdict_votes.iter().all(|v| v.certified));

    // Both cells of the winning column are represented...
    assert!(est
        .verdict_votes
        .iter()
        .any(|v| v.cell == ScanCell::Epipolar));
    assert!(est
        .verdict_votes
        .iter()
        .any(|v| v.cell == ScanCell::Rotation));

    // ...and none of them is the pinhole column's, which is the mistake this
    // field exists to end: the result's flat vote lists always describe the
    // pinhole closed-form kernel, whichever column won.
    let pinhole = column(&est.vote, CameraModel::Pinhole).expect("pinhole column");
    let pinhole_certified: Vec<u64> = pinhole
        .scan_votes
        .iter()
        .filter(|v| v.certified)
        .map(|v| v.focal_px.to_bits())
        .collect();
    assert!(
        !pinhole_certified.is_empty(),
        "the control must be non-empty"
    );
    for v in &est.verdict_votes {
        assert!(!pinhole_certified.contains(&v.focal_px.to_bits()));
    }
    // Nor is it the closed-form detail layer, whose focals are the pinhole
    // column's Bougnoux and self-calibration answers -- and which the result
    // carries at the top level whichever column won.
    let flat: Vec<u64> = est
        .vote
        .epipolar_votes
        .iter()
        .map(|v| v.focal_px.to_bits())
        .chain(est.vote.rotation_votes.iter().map(|v| v.focal_px.to_bits()))
        .collect();
    for v in &est.verdict_votes {
        assert!(!flat.contains(&v.focal_px.to_bits()));
    }
}

#[test]
fn the_vote_comes_back_untouched() {
    let obs = fisheye_scene(2718);
    let est = estimate(&obs, &IntrinsicsOptions::default());
    let raw = crate::geometry::focal_vote::focal_vote_with_options(
        &obs.cluster,
        &obs.image,
        &obs.pos,
        W,
        H,
        &IntrinsicsOptions::default().vote,
    );
    assert_eq!(est.vote.camera_model, raw.camera_model);
    assert_eq!(
        est.vote.focal_px.map(f64::to_bits),
        raw.focal_px.map(f64::to_bits)
    );
    assert_eq!(
        est.focal_px.map(f64::to_bits),
        raw.focal_px.map(f64::to_bits)
    );
    assert_eq!(est.vote.columns.len(), raw.columns.len());
    assert_eq!(est.vote.n_pool, raw.n_pool);
}

#[test]
fn empty_input_has_no_verdict_and_no_question() {
    let est = estimate_intrinsics(&[], &[], &[], W, H, &IntrinsicsOptions::default());
    assert_eq!(est.camera_model, None);
    assert_eq!(est.confirmed, None);
    assert_eq!(est.focal_px, None);
    assert!(est.verdict_votes.is_empty());
}
