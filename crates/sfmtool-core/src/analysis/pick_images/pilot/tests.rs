// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn a_vote_is_used_only_with_a_consensus_whose_spread_is_at_most_the_bound() {
    assert_eq!(judge(None, 0.0), PilotOutcome::NoConsensus);
    assert_eq!(judge(Some(800.0), 0.0), PilotOutcome::Used);
    assert_eq!(judge(Some(800.0), MAX_POOL_SPREAD), PilotOutcome::Used);
    assert_eq!(
        judge(Some(800.0), MAX_POOL_SPREAD + 1e-9),
        PilotOutcome::WideSpread {
            max_pool_spread: MAX_POOL_SPREAD
        }
    );
}

#[test]
fn the_verdict_camera_is_a_pinhole_or_an_equidistant_fisheye_at_the_centre() {
    let pinhole = verdict_camera(Some(CameraModel::Pinhole), 700.0, 640, 480);
    assert_eq!(
        pinhole.model,
        LensModel::SimplePinhole {
            focal_length: 700.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        }
    );
    assert_eq!((pinhole.width, pinhole.height), (640, 480));
    // No verdict, with a focal, is the pinhole column's focal.
    assert_eq!(verdict_camera(None, 700.0, 640, 480), pinhole);
    let fisheye = verdict_camera(Some(CameraModel::EquidistantFisheye), 300.0, 480, 640);
    assert_eq!(
        fisheye.model,
        LensModel::EquidistantFisheye {
            focal_length: 300.0,
            principal_point_x: 240.0,
            principal_point_y: 320.0,
        }
    );
}

#[test]
fn the_outcome_serializes_with_its_bound() {
    let json = |outcome| serde_json::to_value(outcome).expect("serializes");
    assert_eq!(
        json(PilotOutcome::Used),
        serde_json::json!({"status": "used"})
    );
    assert_eq!(
        json(PilotOutcome::NoConsensus),
        serde_json::json!({"status": "no_consensus"})
    );
    assert_eq!(
        json(PilotOutcome::TooFewFrames { min_frames: 3 }),
        serde_json::json!({"status": "too_few_frames", "min_frames": 3})
    );
    assert_eq!(
        json(PilotOutcome::WideSpread {
            max_pool_spread: 0.35
        }),
        serde_json::json!({"status": "wide_spread", "max_pool_spread": 0.35})
    );
}
