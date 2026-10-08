// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cancellation of the patch batches that render views: each polls its
//! `Progress` before every patch, and a cancelled batch returns `Cancelled`
//! rather than the results of the patches it finished.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nalgebra::{Point3, Vector3};

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::{localize_patch_cloud_keypoints, KeypointLocalizeParams};
use crate::patch::keypoint_subpixel::{refine_patch_cloud_keypoints, KeypointSubpixelParams};
use crate::patch::member_coherence::{
    validate_patch_cloud_member_coherence, MemberCoherenceParams,
};
use crate::patch::normal_refine::{refine_patch_cloud_normals, NormalRefineParams, ProjectedImage};
use crate::patch::stored_bitmap::render_patch_cloud_bitmaps;
use crate::patch::view_selection::{select_patch_cloud_views, ViewSelectParams};
use crate::patch::PatchCloud;
use crate::progress::{Event, Progress};
use crate::SfmrReconstruction;

const R: u32 = 16;

fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 300.0,
            focal_length_y: 300.0,
            principal_point_x: 160.0,
            principal_point_y: 120.0,
        },
        width: 320,
        height: 240,
    }
}

/// A textured photograph and its pyramid. What it shows does not matter here:
/// only whether a batch starts on a patch.
fn pyramid() -> ImageU8Pyramid {
    let (w, h) = (320u32, 240u32);
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (x * 7 + y * 13) ^ (x * y);
            data.extend_from_slice(&[(v % 251) as u8, (v % 241) as u8, ((x + y) % 256) as u8]);
        }
    }
    ImageU8Pyramid::build(&ImageU8::new(w, h, 3, data), 5)
}

/// `n` small squares at depth 4 facing three cameras along `x`.
fn cloud(n: usize) -> PatchCloud {
    let patches = (0..n)
        .map(|i| {
            let x = -0.4 + 0.8 * i as f64 / n as f64;
            OrientedPatch::from_center_normal(
                Point3::new(x, 0.0, -4.0),
                Vector3::z(),
                Vector3::y(),
                [0.1, 0.1],
            )
        })
        .collect();
    PatchCloud {
        patches,
        point_indexes: (0..n as u32).collect(),
    }
}

/// What a batch reported: how many `Count` events, and how many render detail
/// phases it entered.
#[derive(Default)]
struct Reported {
    counts: AtomicUsize,
    renders: AtomicUsize,
}

impl Reported {
    fn sink(&self) -> impl Fn(Event<'_>) + Sync + '_ {
        move |event: Event<'_>| match event {
            Event::Count { .. } => {
                self.counts.fetch_add(1, Ordering::Relaxed);
            }
            Event::Enter { phase, .. } if phase.starts_with("render ") => {
                self.renders.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

/// A batch handed a `Progress` cancelled before it starts returns `Cancelled`,
/// finishes no patch, reports no count and renders nothing, and normal
/// refinement leaves the cloud as it was.
#[test]
fn a_batch_cancelled_before_it_starts_does_no_work() {
    let cams = [camera(), camera(), camera()];
    let poses: Vec<RigidTransform> = [-0.2, 0.0, 0.2]
        .iter()
        .map(|&x| RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-x, 0.0, 0.0]))
        .collect();
    let pyr = pyramid();
    let views: Vec<ProjectedImage<'_>> = cams
        .iter()
        .zip(&poses)
        .map(|(camera, cam_from_world)| ProjectedImage {
            camera,
            cam_from_world,
            pyramid: &pyr,
        })
        .collect();
    let cloud = cloud(40);
    let view_sets: Vec<Vec<u32>> = vec![vec![0, 1, 2]; cloud.len()];
    let cancel = AtomicBool::new(true);

    let check = |name: &str, run: &dyn Fn(&Progress<'_>, &AtomicUsize) -> bool| {
        let reported = Reported::default();
        let sink = reported.sink();
        let progress = Progress::to(&sink).detailed(true).cancelled_by(&cancel);
        let done = AtomicUsize::new(0);
        assert!(run(&progress, &done), "{name} did not return Cancelled");
        assert_eq!(done.load(Ordering::Relaxed), 0, "{name} finished a patch");
        assert_eq!(reported.counts.load(Ordering::Relaxed), 0, "{name} counted");
        assert_eq!(
            reported.renders.load(Ordering::Relaxed),
            0,
            "{name} rendered"
        );
    };

    check("normal refinement", &|progress, done| {
        let mut refined = cloud.clone();
        let params = NormalRefineParams::default();
        let out = refine_patch_cloud_normals(
            &mut refined,
            &views,
            &view_sets,
            R,
            &params,
            None,
            Some(done),
            progress,
        );
        assert_eq!(
            format!("{:?}", refined.patches),
            format!("{:?}", cloud.patches),
            "the cloud changed"
        );
        out.is_err()
    });
    check("view selection", &|progress, done| {
        let params = ViewSelectParams {
            resolution: R,
            ..ViewSelectParams::default()
        };
        select_patch_cloud_views(
            &cloud,
            &views,
            &view_sets,
            None,
            &params,
            Some(done),
            progress,
        )
        .is_err()
    });
    check("the localizer", &|progress, done| {
        let params = KeypointLocalizeParams {
            resolution: R,
            ..KeypointLocalizeParams::default()
        };
        localize_patch_cloud_keypoints(
            &cloud,
            &views,
            &view_sets,
            None,
            None,
            &params,
            Some(done),
            progress,
        )
        .is_err()
    });
    check("the sub-pixel refiner", &|progress, _| {
        let params = KeypointSubpixelParams {
            resolution: R,
            ..KeypointSubpixelParams::default()
        };
        refine_patch_cloud_keypoints(&cloud, &views, &view_sets, None, &params, progress).is_err()
    });
    check("member coherence", &|progress, done| {
        let params = MemberCoherenceParams {
            resolution: R,
            ..MemberCoherenceParams::default()
        };
        validate_patch_cloud_member_coherence(
            &cloud,
            &views,
            &view_sets,
            None,
            &params,
            Some(done),
            progress,
        )
        .is_err()
    });

    // The stored bitmaps are read from a reconstruction's tracks and keypoints.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr");
    let recon = SfmrReconstruction::load(&path, &Progress::none()).expect("the ground truth loads");
    let stored = PatchCloud::from_stored_frames(&recon).expect("the ground truth has frames");
    let none: Vec<Option<ProjectedImage<'_>>> = vec![None; recon.image_count()];
    check("the stored bitmaps", &|progress, done| {
        render_patch_cloud_bitmaps(
            &stored,
            &recon,
            &none,
            &KeypointSubpixelParams::default(),
            Some(done),
            progress,
        )
        .is_err()
    });
}

/// Normal refinement cancelled part of the way through returns `Cancelled`,
/// not the patches it refined, and leaves the cloud as it was: the flag is set
/// by the first `patches` count, which the batch sends after its first patch.
/// The cloud is many times larger than the patches the worker threads can
/// have started by then.
#[test]
fn normal_refinement_cancelled_part_way_leaves_the_cloud() {
    let cams = [camera(), camera(), camera()];
    let poses: Vec<RigidTransform> = [-0.2, 0.0, 0.2]
        .iter()
        .map(|&x| RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-x, 0.0, 0.0]))
        .collect();
    let pyr = pyramid();
    let views: Vec<ProjectedImage<'_>> = cams
        .iter()
        .zip(&poses)
        .map(|(camera, cam_from_world)| ProjectedImage {
            camera,
            cam_from_world,
            pyramid: &pyr,
        })
        .collect();
    let original = cloud(2000);
    let view_sets: Vec<Vec<u32>> = vec![vec![0, 1, 2]; original.len()];
    let cancel = AtomicBool::new(false);
    let sink = |event: Event<'_>| {
        if let Event::Count { .. } = event {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let progress = Progress::to(&sink).cancelled_by(&cancel);
    let done = AtomicUsize::new(0);
    let mut refined = original.clone();
    let out = refine_patch_cloud_normals(
        &mut refined,
        &views,
        &view_sets,
        R,
        &NormalRefineParams::default(),
        None,
        Some(&done),
        &progress,
    );
    assert!(out.is_err(), "a cancelled batch returned results");
    let finished = done.load(Ordering::Relaxed);
    assert!(
        (1..original.len()).contains(&finished),
        "the batch should stop part of the way through, finished {finished}"
    );
    assert_eq!(
        format!("{:?}", refined.patches),
        format!("{:?}", original.patches),
        "the cloud changed"
    );
}
