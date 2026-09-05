// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::geometry::focal_vote::tests::{
    baseline_cameras, emit_fisheye_parallax_pair, fisheye_scene, two_subcapture_scene, Lcg, Obs,
    F_TRUE, H, W,
};
use crate::geometry::focal_vote::{ScanCell, VoteFamily};

/// Run the estimator over a synthetic capture, both columns, seed 0.
fn estimate(obs: &Obs, options: &IntrinsicsOptions) -> IntrinsicsEstimate {
    estimate_intrinsics(&obs.starts, &obs.image, &obs.pos, W, H, options)
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
        &obs.starts,
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
    let est = estimate_intrinsics(&[0], &[], &[], W, H, &IntrinsicsOptions::default());
    assert_eq!(est.camera_model, None);
    assert_eq!(est.confirmed, None);
    assert_eq!(est.focal_px, None);
    assert!(est.verdict_votes.is_empty());
    // A fixed column set asks its question outright; nothing was escalated to.
    assert_eq!(est.escalation, None);
    assert!(est.screening_vote.is_none());
}

// ── The weak-vote escalation ─────────────────────────────────────────────────
//
// Each arm of the disjunction is pinned against a hand-built vote result, so a
// test says which single quantity it is moving. The scene-level tests below
// then check that `Auto` acts on the reasons: it re-runs when one fires, leaves
// the pinhole vote alone when none does, and its escalated answer is the
// two-column answer.

/// A strong pinhole vote: a consensus, one family, a wide pool. Every
/// escalation test starts here and moves exactly one field.
fn strong_vote() -> FocalVoteResult {
    FocalVoteResult {
        focal_px: Some(800.0),
        family: Some(VoteFamily::Epipolar),
        epipolar_focal_px: Some(800.0),
        rotation_focal_px: None,
        n_epipolar: 12,
        n_rotation: 0,
        n_pool: 12,
        pool_spread: 0.01,
        family_disagreement: None,
        parallax_poverty: 0.1,
        epipolar_spread: 0.01,
        rotation_spread: 0.0,
        epipolar_votes: Vec::new(),
        rotation_votes: Vec::new(),
        n_h_dominated: 0,
        n_estimator_failed: 0,
        n_band_rejected: 0,
        n_degenerate: 0,
        n_inconsistent_pairs: 0,
        camera_model: Some(CameraModel::Pinhole),
        columns: Vec::new(),
    }
}

#[test]
fn a_strong_vote_escalates_for_no_reason() {
    assert!(escalation_reasons(&strong_vote(), W, H).is_empty());
}

#[test]
fn no_consensus_escalates() {
    let vote = FocalVoteResult {
        focal_px: None,
        ..strong_vote()
    };
    assert_eq!(
        escalation_reasons(&vote, W, H),
        vec![EscalationReason::NoConsensus]
    );
}

#[test]
fn a_railed_rotation_consensus_escalates() {
    // The grid's bottom rung is `ORTHO_GRID_LO * max(w, h)`, and the cut is one
    // multiplicative step above it: on it and just under the step escalates,
    // just over does not.
    let max_wh = f64::from(W.max(H));
    let step = (ORTHO_GRID_HI / ORTHO_GRID_LO).powf(1.0 / (ORTHO_GRID_N - 1) as f64);
    let railed = |f: f64| {
        let vote = FocalVoteResult {
            family: Some(VoteFamily::Rotation),
            rotation_focal_px: Some(f),
            ..strong_vote()
        };
        escalation_reasons(&vote, W, H)
    };
    assert_eq!(
        railed(ORTHO_GRID_LO * max_wh),
        vec![EscalationReason::RotationRailed]
    );
    assert_eq!(
        railed(step * ORTHO_GRID_LO * max_wh),
        vec![EscalationReason::RotationRailed],
        "the cut is inclusive at one grid step above the floor"
    );
    assert!(railed(1.001 * step * ORTHO_GRID_LO * max_wh).is_empty());

    // The same focal from the epipolar family is not railing: the grid is the
    // rotation self-calibration's, so only its answer can sit on the floor.
    let epipolar = FocalVoteResult {
        family: Some(VoteFamily::Epipolar),
        rotation_focal_px: Some(ORTHO_GRID_LO * max_wh),
        ..strong_vote()
    };
    assert!(escalation_reasons(&epipolar, W, H).is_empty());
}

#[test]
fn family_disagreement_escalates() {
    let over = FocalVoteResult {
        family_disagreement: Some(FAMILY_DISAGREEMENT_BAND + 1e-9),
        ..strong_vote()
    };
    assert_eq!(
        escalation_reasons(&over, W, H),
        vec![EscalationReason::FamilyDisagreement]
    );
    // The band itself is not disagreement -- the vote's own pooling rule reads
    // it the same way.
    let at_band = FocalVoteResult {
        family_disagreement: Some(FAMILY_DISAGREEMENT_BAND),
        ..strong_vote()
    };
    assert!(escalation_reasons(&at_band, W, H).is_empty());
}

#[test]
fn a_thin_pool_escalates() {
    let thin = FocalVoteResult {
        n_epipolar: THIN_POOL,
        n_pool: THIN_POOL,
        ..strong_vote()
    };
    assert_eq!(
        escalation_reasons(&thin, W, H),
        vec![EscalationReason::ThinPool]
    );
    let one_more = FocalVoteResult {
        n_epipolar: THIN_POOL + 1,
        n_pool: THIN_POOL + 1,
        ..strong_vote()
    };
    assert!(escalation_reasons(&one_more, W, H).is_empty());
}

#[test]
fn every_reason_that_fires_is_reported_in_check_order() {
    let max_wh = f64::from(W.max(H));
    let all = FocalVoteResult {
        focal_px: None,
        family: Some(VoteFamily::Rotation),
        rotation_focal_px: Some(ORTHO_GRID_LO * max_wh),
        family_disagreement: Some(0.9),
        n_epipolar: 1,
        n_rotation: 1,
        n_pool: 2,
        ..strong_vote()
    };
    assert_eq!(
        escalation_reasons(&all, W, H),
        vec![
            EscalationReason::NoConsensus,
            EscalationReason::RotationRailed,
            EscalationReason::FamilyDisagreement,
            EscalationReason::ThinPool,
        ]
    );
}

#[test]
fn reason_names_are_stable() {
    assert_eq!(EscalationReason::NoConsensus.as_str(), "no_consensus");
    assert_eq!(EscalationReason::RotationRailed.as_str(), "rotation_railed");
    assert_eq!(
        EscalationReason::FamilyDisagreement.as_str(),
        "family_disagreement"
    );
    assert_eq!(EscalationReason::ThinPool.as_str(), "thin_pool");
}

/// [`ColumnPolicy::Auto`], everything else default.
fn auto() -> IntrinsicsOptions {
    IntrinsicsOptions {
        columns: ColumnPolicy::Auto,
        ..Default::default()
    }
}

#[test]
fn auto_escalates_a_fisheye_capture_to_the_two_column_answer() {
    let obs = fisheye_scene(2718);
    let est = estimate(&obs, &auto());
    let reasons = est.escalation.clone().expect("Auto records its decision");
    assert!(
        !reasons.is_empty(),
        "a fisheye capture's pinhole vote is weak: {:?}",
        est.screening_vote.as_ref().map(|v| v.n_pool)
    );

    // The escalated answer IS the both-columns answer on the same inputs.
    let direct = estimate(&obs, &IntrinsicsOptions::default());
    assert_eq!(est.camera_model, direct.camera_model);
    assert_eq!(est.confirmed, direct.confirmed);
    assert_eq!(
        est.focal_px.map(f64::to_bits),
        direct.focal_px.map(f64::to_bits)
    );
    assert_eq!(est.vote.columns.len(), direct.vote.columns.len());
    assert_eq!(est.verdict_votes.len(), direct.verdict_votes.len());

    // ...and the weak pinhole vote that triggered it is kept, because the
    // escalated result's top-level fields are the fisheye column's.
    let screening = est.screening_vote.expect("the screening vote is kept");
    assert!(screening.columns.is_empty());
    assert_eq!(screening.camera_model, Some(CameraModel::Pinhole));
    assert_eq!(escalation_reasons(&screening, W, H), reasons);
}

#[test]
fn auto_leaves_a_strong_pinhole_vote_alone() {
    let obs = two_subcapture_scene(6, 8, F_TRUE, F_TRUE, 99);
    let est = estimate(&obs, &auto());
    assert_eq!(
        est.escalation.as_deref(),
        Some(&[][..]),
        "a strong pinhole vote has nothing for the columns to overturn"
    );
    // No second run: no columns, a Pinhole verdict by construction, and no
    // confirmation question.
    assert!(est.screening_vote.is_none());
    assert!(est.vote.columns.is_empty());
    assert_eq!(est.camera_model, Some(CameraModel::Pinhole));
    assert_eq!(est.confirmed, None);
    assert!(est.verdict_votes.is_empty());

    // It is exactly the pinhole-only vote, which is what makes skipping the
    // scans free rather than a different answer.
    let pinhole = crate::geometry::focal_vote::focal_vote_with_options(
        &obs.starts,
        &obs.image,
        &obs.pos,
        W,
        H,
        &FocalVoteOptions::default(),
    );
    assert_eq!(
        est.focal_px.map(f64::to_bits),
        pinhole.focal_px.map(f64::to_bits)
    );
    assert_eq!(est.vote.n_pool, pinhole.n_pool);
}

#[test]
fn a_fixed_column_set_never_escalates() {
    let est = estimate(&fisheye_scene(2718), &IntrinsicsOptions::default());
    assert_eq!(est.escalation, None);
    assert!(est.screening_vote.is_none());
}

// ── The camera the verdict implies ───────────────────────────────────────────

/// A camera's model name, focal (both axes agree for the two single-focal
/// models this field builds) and principal point, as raw bits where it matters.
fn read(cam: &CameraIntrinsics) -> (&'static str, u64, (f64, f64), u32, u32) {
    let (fx, fy) = cam.focal_lengths();
    assert_eq!(fx.to_bits(), fy.to_bits(), "single-focal model");
    (
        cam.model_name(),
        fx.to_bits(),
        cam.principal_point(),
        cam.width,
        cam.height,
    )
}

#[test]
fn a_confirmed_equidistant_verdict_gives_an_equidistant_camera() {
    let est = estimate(&fisheye_scene(2718), &IntrinsicsOptions::default());
    assert_eq!(est.confirmed, Some(true));
    let cam = est
        .camera
        .as_ref()
        .expect("a confirmed verdict has a camera");
    assert_eq!(
        read(cam),
        (
            "EQUIDISTANT_FISHEYE",
            est.focal_px.expect("consensus").to_bits(),
            (f64::from(W) / 2.0, f64::from(H) / 2.0),
            W,
            H,
        )
    );
}

#[test]
fn a_pinhole_verdict_gives_a_pinhole_camera_at_the_vote_focal() {
    let est = estimate(
        &two_subcapture_scene(6, 8, F_TRUE, F_TRUE, 99),
        &IntrinsicsOptions::default(),
    );
    assert_eq!(est.camera_model, Some(CameraModel::Pinhole));
    // The pinhole column won, so the top-level focal IS the pinhole answer.
    let cam = est.camera.as_ref().expect("a pinhole verdict has a camera");
    assert_eq!(
        read(cam),
        (
            "SIMPLE_PINHOLE",
            est.focal_px.expect("consensus").to_bits(),
            (f64::from(W) / 2.0, f64::from(H) / 2.0),
            W,
            H,
        )
    );
}

#[test]
fn an_unconfirmed_equidistant_verdict_falls_back_to_the_pinhole_column() {
    // Fixed columns, so nothing was screened: the pinhole answer survives only
    // in the column the fisheye verdict beat.
    let est = estimate(
        &parallax_only_fisheye_scene(31337),
        &IntrinsicsOptions::default(),
    );
    assert_eq!(est.camera_model, Some(CameraModel::EquidistantFisheye));
    assert_eq!(est.confirmed, Some(false));
    assert!(est.screening_vote.is_none());

    let pinhole = column(&est.vote, CameraModel::Pinhole)
        .expect("pinhole column")
        .focal_px
        .expect("pinhole consensus");
    let cam = est.camera.as_ref().expect("a pinhole camera");
    assert_eq!(cam.model_name(), "SIMPLE_PINHOLE");
    assert_eq!(cam.focal_lengths().0.to_bits(), pinhole.to_bits());
    // ...and NOT the equidistant consensus the top-level focal reports, which
    // parameterizes a different map.
    assert_ne!(
        cam.focal_lengths().0.to_bits(),
        est.focal_px.expect("consensus").to_bits()
    );
}

#[test]
fn an_escalated_run_reads_the_pinhole_focal_off_the_screening_vote() {
    // This capture escalates on a thin pool rather than on no consensus, so the
    // screening vote it kept HAS a pinhole focal -- and its unconfirmed fisheye
    // verdict means that focal is the one the camera takes, not the top-level
    // equidistant consensus.
    let est = estimate(&parallax_only_fisheye_scene(31337), &auto());
    assert_eq!(est.camera_model, Some(CameraModel::EquidistantFisheye));
    assert_eq!(est.confirmed, Some(false));
    let screening = est.screening_vote.as_ref().expect("an escalated run");
    let cam = est.camera.as_ref().expect("a pinhole camera");
    assert_eq!(cam.model_name(), "SIMPLE_PINHOLE");
    assert_eq!(
        cam.focal_lengths().0.to_bits(),
        screening.focal_px.expect("pinhole consensus").to_bits()
    );
    assert_ne!(
        cam.focal_lengths().0.to_bits(),
        est.focal_px.expect("consensus").to_bits()
    );
}

#[test]
fn an_unescalated_auto_run_takes_the_pinhole_vote_it_ran() {
    // No second run, so `vote` IS the pinhole-only vote and its top-level focal
    // is the pinhole answer.
    let est = estimate(&two_subcapture_scene(6, 8, F_TRUE, F_TRUE, 99), &auto());
    assert_eq!(est.escalation.as_deref(), Some(&[][..]));
    assert!(est.screening_vote.is_none());
    let cam = est.camera.as_ref().expect("a pinhole camera");
    assert_eq!(cam.model_name(), "SIMPLE_PINHOLE");
    assert_eq!(
        cam.focal_lengths().0.to_bits(),
        est.vote.focal_px.expect("consensus").to_bits()
    );
}

#[test]
fn no_consensus_leaves_no_camera() {
    let est = estimate_intrinsics(&[0], &[], &[], W, H, &IntrinsicsOptions::default());
    assert_eq!(est.focal_px, None);
    assert!(est.camera.is_none());
}

#[test]
fn the_pinhole_camera_is_the_focal_a_caller_would_read_by_hand() {
    // The precedence, stated as the reading every `Auto` caller used to do:
    // the screening vote's focal when there is one, the vote's own otherwise --
    // whichever arm the capture takes, and whether or not it has a camera.
    // A rotation-mass floor nothing clears keeps every verdict on the pinhole
    // rule, so all three captures are read the same way.
    for obs in [
        fisheye_scene(2718),
        parallax_only_fisheye_scene(31337),
        two_subcapture_scene(6, 8, F_TRUE, F_TRUE, 99),
    ] {
        let est = estimate(
            &obs,
            &IntrinsicsOptions {
                min_rotation_mass: usize::MAX,
                ..auto()
            },
        );
        assert_ne!(est.confirmed, Some(true));
        let by_hand = est.screening_vote.as_ref().unwrap_or(&est.vote).focal_px;
        assert_eq!(
            est.camera
                .as_ref()
                .map(|c| (c.model_name(), c.focal_lengths().0.to_bits())),
            by_hand.map(|f| ("SIMPLE_PINHOLE", f.to_bits()))
        );
    }
}

// ── The from-matches entry ───────────────────────────────────────────────────

/// A parsed `.matches` value carrying a synthetic capture as its cluster
/// backbone, at the given per-image dimensions.
///
/// The member arrays ARE the capture's observation arrays -- the file's layout,
/// the vote's layout and the vote's element type are all the same -- so the
/// from-matches entry and the array entry see identical values with nothing
/// converted between them.
fn matches_of(obs: &Obs, dims: &[(u32, u32)]) -> MatchesData {
    let n_members = obs.image.len();
    let mut positions = ndarray::Array2::<f32>::zeros((n_members, 2));
    for (row, p) in obs.pos.iter().enumerate() {
        positions[[row, 0]] = p[0];
        positions[[row, 1]] = p[1];
    }
    let mut image_dims = ndarray::Array2::<u32>::zeros((dims.len(), 2));
    for (row, &(w, h)) in dims.iter().enumerate() {
        image_dims[[row, 0]] = w;
        image_dims[[row, 1]] = h;
    }
    MatchesData {
        metadata: matches_format::MatchesMetadata {
            version: matches_format::MATCHES_FORMAT_VERSION,
            matching_method: "test".into(),
            matching_tool: "test".into(),
            matching_tool_version: "0".into(),
            matching_options: std::collections::BTreeMap::new(),
            workspace: matches_format::WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: matches_format::WorkspaceContents {
                    feature_tool: "none".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::json!({}),
                    feature_prefix_dir: String::new(),
                },
            },
            timestamp: String::new(),
            image_count: dims.len() as u32,
            image_pair_count: None,
            match_count: None,
            cluster_count: Some(obs.starts.len() as u32 - 1),
            cluster_member_count: Some(n_members as u32),
            has_two_view_geometries: false,
            has_clusters: true,
            has_cluster_patches: false,
        },
        content_hash: matches_format::MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: None,
            clusters_xxh128: None,
            cluster_patches_xxh128: None,
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names: (0..dims.len()).map(|i| format!("images/{i}.jpg")).collect(),
        feature_tool_hashes: vec![[0u8; 16]; dims.len()],
        sift_content_hashes: vec![[0u8; 16]; dims.len()],
        feature_counts: ndarray::Array1::zeros(dims.len()),
        image_dims: Some(image_dims),
        image_pairs: None,
        clusters: Some(matches_format::ClustersData {
            cluster_starts: ndarray::Array1::from(obs.starts.clone()),
            member_images: ndarray::Array1::from(obs.image.clone()),
            member_features: ndarray::Array1::from_iter(0..n_members as u32),
            member_positions: Some(positions),
            member_affine_shapes: None,
            matcher_options: serde_json::json!({}),
        }),
        cluster_patches: None,
        two_view_geometries: None,
    }
}

#[test]
fn the_from_matches_entry_is_the_array_entry() {
    let obs = fisheye_scene(2718);
    let n_img = obs.image.iter().max().expect("members") + 1;
    let dims = vec![(W, H); n_img as usize];
    let options = IntrinsicsOptions::default();

    let from_file =
        estimate_intrinsics_from_matches(&matches_of(&obs, &dims), &options).expect("readable");
    let from_arrays = estimate_intrinsics(&obs.starts, &obs.image, &obs.pos, W, H, &options);

    assert_eq!(from_file.camera_model, from_arrays.camera_model);
    assert_eq!(from_file.confirmed, from_arrays.confirmed);
    assert_eq!(
        from_file.focal_px.map(f64::to_bits),
        from_arrays.focal_px.map(f64::to_bits)
    );
    assert_eq!(from_file.vote.n_pool, from_arrays.vote.n_pool);
    assert_eq!(
        from_file.verdict_votes.len(),
        from_arrays.verdict_votes.len()
    );
    for (a, b) in from_file
        .verdict_votes
        .iter()
        .zip(&from_arrays.verdict_votes)
    {
        assert_eq!(a.focal_px.to_bits(), b.focal_px.to_bits());
    }
}

#[test]
fn mixed_image_dimensions_are_refused() {
    // The vote places the principal point at the centre of ONE camera, so a
    // file whose images are not one resolution is not one estimate -- named
    // here rather than silently answered from the first image's dimensions.
    let obs = fisheye_scene(2718);
    let n_img = (obs.image.iter().max().expect("members") + 1) as usize;
    let mut dims = vec![(W, H); n_img];
    dims[2] = (W / 2, H);
    let err = estimate_intrinsics_from_matches(&matches_of(&obs, &dims), &Default::default())
        .expect_err("mixed dimensions");
    assert_eq!(
        err,
        MatchesInputError::MixedDimensions {
            expected: (W, H),
            found: (W / 2, H),
            image: "images/2.jpg".to_string(),
        }
    );
    assert!(err.to_string().contains("images/2.jpg"), "{err}");
}

#[test]
fn a_file_without_clusters_is_refused() {
    let obs = fisheye_scene(2718);
    let n_img = (obs.image.iter().max().expect("members") + 1) as usize;
    let mut matches = matches_of(&obs, &vec![(W, H); n_img]);
    matches.clusters = None;
    assert_eq!(
        estimate_intrinsics_from_matches(&matches, &Default::default()).expect_err("no clusters"),
        MatchesInputError::NoClusters
    );

    // ...as is a cluster backbone with no member positions, and a file that
    // never recorded its image dimensions.
    let mut matches = matches_of(&obs, &vec![(W, H); n_img]);
    matches
        .clusters
        .as_mut()
        .expect("clusters")
        .member_positions = None;
    assert_eq!(
        estimate_intrinsics_from_matches(&matches, &Default::default()).expect_err("no positions"),
        MatchesInputError::NoMemberPositions
    );

    let mut matches = matches_of(&obs, &vec![(W, H); n_img]);
    matches.image_dims = None;
    assert_eq!(
        estimate_intrinsics_from_matches(&matches, &Default::default()).expect_err("no dims"),
        MatchesInputError::NoImageDimensions
    );
}
