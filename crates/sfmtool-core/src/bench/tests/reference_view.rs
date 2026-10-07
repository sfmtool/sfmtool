// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reference-view readings an evaluation writes, on a real track: the
//! seoul_bull ground truth and its photographs, which are checked in.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::bench::{create_track, evaluate, Bench, CreateTrackOptions, EvaluateOptions, Verdict};
use crate::camera::PhotographCache;
use crate::geometry::RigidTransform;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::{
    ReferenceFallback, REFERENCE_MAX_VIEWING_ANGLE_DEG, REFERENCE_MIN_COVERAGE,
};
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;
use crate::reconstruction::SfmrReconstruction;

/// The point of the ground truth the test reads: the first with at least
/// eight observations.
const MIN_TRACK: usize = 8;

#[test]
fn an_evaluation_picks_a_whole_view_facing_the_patch_on_a_ground_truth_track() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/images/seoul_bull_sculpture");
    let recon = SfmrReconstruction::load(
        &dir.join("seoul_bull_sculpture_ground_truth.sfmr"),
        &Progress::none(),
    )
    .expect("the ground truth loads");
    let offsets = &recon.point_set.observation_offsets;
    let point = (0..recon.point_count())
        .find(|&p| offsets[p + 1] - offsets[p] >= MIN_TRACK)
        .expect("the ground truth has a long track") as u32;

    let images = &recon.image_table.images;
    let paths: Vec<PathBuf> = images.iter().map(|i| dir.join(&i.name)).collect();
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let (pyramids, _) = PhotographCache::new(0, 8)
        .get_many(&refs, &Progress::none())
        .expect("nothing cancels the decode");
    let pyramids: Vec<_> = pyramids
        .into_iter()
        .map(|p| p.expect("every photograph is checked in"))
        .collect();
    let poses: Vec<RigidTransform> = images
        .iter()
        .map(|image| {
            let q = image.quaternion_wxyz;
            let t = image.translation_xyz;
            RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
        })
        .collect();
    let views: Vec<ProjectedImage<'_>> = images
        .iter()
        .zip(&poses)
        .zip(&pyramids)
        .map(|((image, cam_from_world), pyramid)| ProjectedImage {
            camera: &recon.image_table.cameras[image.camera_index as usize],
            cam_from_world,
            pyramid,
        })
        .collect();

    let edited = EditedReconstruction::new(Arc::new(recon.clone()));
    let (bench, report) = create_track(
        &Bench::new(),
        &edited,
        point,
        &CreateTrackOptions::default(),
    )
    .expect("the point is live");
    let track = bench.track(&report.label).expect("just put on");
    let (read, _) = evaluate(
        track,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .expect("the track reads");

    let mut picked = Vec::new();
    for (i, observation) in read.observations.iter().enumerate() {
        let m = observation.track.as_ref().expect("every row is read");
        assert_eq!(observation.verdict, Verdict::In);
        // Every row carries the per-view readings and a standing.
        assert!(m.viewing_angle_deg.is_some(), "row {i}");
        assert!(m.coverage.is_some(), "row {i}");
        assert!(m.clipped_share.is_some(), "row {i}");
        assert!(m.pair_zncc_grid.is_some(), "row {i}");
        let standing = m.reference_view.expect("an in row has a standing");
        assert_eq!(standing.fallback, ReferenceFallback::None);
        if standing.is_reference() {
            picked.push(i);
        }
    }
    assert_eq!(picked.len(), 1, "one reference view");
    let reference = read.observations[picked[0]].track.as_ref().unwrap();
    assert!(reference.coverage.unwrap() >= REFERENCE_MIN_COVERAGE);
    assert!(reference.viewing_angle_deg.unwrap() <= REFERENCE_MAX_VIEWING_ANGLE_DEG);
    assert!(reference.pair_zncc.is_some());
    // The reference is the sharpest of the views that passed every other test.
    let radius = |i: usize| {
        read.observations[i]
            .track
            .as_ref()
            .and_then(|m| m.zncc_self_similarity_radius)
            .unwrap()
    };
    for (i, observation) in read.observations.iter().enumerate() {
        let standing = observation.track.as_ref().unwrap().reference_view.unwrap();
        if standing.rejected_by == Some(crate::patch::reference_view::ReferenceTest::Sharpness) {
            assert!(radius(i) >= radius(picked[0]), "row {i}");
        }
    }
}
