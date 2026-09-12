// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Adding an observation from a pixel: the refusals, and that the fit lands the
//! keypoint on the truth and the re-triangulation moves the point toward it.
//!
//! The scene is the one the localization kernel's own tests use -- pinhole
//! cameras looking down world `+z` at a textured plane -- wrapped in an
//! `embedded_patches` reconstruction whose stored keypoints are the exact
//! projections, so the truth for the added observation is known to the pixel.

use std::sync::Arc;

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::{Event, Progress};
use crate::reconstruction::data::{
    ObservationSource, Point3D, SfmrImage, SfmrReconstruction, TrackObservation,
};

use super::*;

pub(crate) const IMG_W: u32 = 128;
pub(crate) const IMG_H: u32 = 128;
const FOCAL: f64 = 160.0;
const PLANE_Z: f64 = 4.0;
const HALF_EXTENT: f64 = 0.12;
/// The camera centres, in world space. The first two observe the point; the
/// third is the one an observation is added in.
const CENTERS: [[f64; 3]; 3] = [[-0.5, -0.3, 0.0], [0.45, 0.25, 0.0], [0.1, -0.55, 0.0]];

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

fn texture(x: f64, y: f64) -> f64 {
    127.5 + 55.0 * (x * 17.0).sin() + 45.0 * (y * 23.0).cos() + 25.0 * ((x + y) * 31.0).sin()
}

/// What a pinhole at `center` looking down world `+z` sees of the textured
/// plane `z = PLANE_Z`.
fn render_plane_view(center: [f64; 3]) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            let lambda = PLANE_Z - center[2];
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            data.push(texture(x, y).clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8::new(IMG_W, IMG_H, 1, data)
}

/// The pose of a camera at `center`: a half-turn about `x`, which puts the
/// canonical `-z` camera axis along world `+z`.
fn pose(center: [f64; 3]) -> RigidTransform {
    RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-center[0], center[1], center[2]])
}

/// The decoded views, held so the borrows in [`views`] have something to point
/// at.
pub(crate) struct Scene {
    cameras: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyramids: Vec<ImageU8Pyramid>,
}

impl Scene {
    pub(crate) fn new() -> Self {
        Self {
            cameras: CENTERS.iter().map(|_| pinhole()).collect(),
            poses: CENTERS.iter().map(|&c| pose(c)).collect(),
            pyramids: CENTERS
                .iter()
                .map(|&c| ImageU8Pyramid::build(&render_plane_view(c), 4))
                .collect(),
        }
    }

    pub(crate) fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.cameras
            .iter()
            .zip(&self.poses)
            .zip(&self.pyramids)
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect()
    }

    /// Where `world` lands in image `i`, in source-image px.
    pub(crate) fn project(&self, i: usize, world: Point3<f64>) -> [f64; 2] {
        let cam = self.poses[i].transform_point(&world);
        let (u, v) = self.cameras[i]
            .ray_to_pixel([cam.x, cam.y, cam.z])
            .expect("the point is in front of every camera of this scene");
        [u, v]
    }
}

/// The patch the point carries: on the plane, facing the cameras.
fn plane_patch(center: Point3<f64>) -> OrientedPatch {
    OrientedPatch::from_center_normal(
        center,
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [HALF_EXTENT, HALF_EXTENT],
    )
}

/// An `embedded_patches` reconstruction over [`Scene`] holding one point on the
/// plane, observed by images 0 and 1 at their exact projections. Image 2 does
/// not observe it, and is where an observation is added.
pub(crate) fn fixture(scene: &Scene, world: Point3<f64>) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    let n = CENTERS.len();

    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..n)
        .map(|i| SfmrImage {
            name: format!("image_{i}.jpg"),
            camera_index: 0,
            quaternion_wxyz: UnitQuaternion::from_quaternion(Quaternion::new(0.0, 1.0, 0.0, 0.0)),
            translation_xyz: Vector3::new(-CENTERS[i][0], CENTERS[i][1], CENTERS[i][2]),
        })
        .collect();
    recon.image_table.thumbnails_y_x_rgb = Arc::new(Array4::zeros((
        n,
        sfmr_format::THUMBNAIL_SIZE,
        sfmr_format::THUMBNAIL_SIZE,
        3,
    )));
    recon.image_table.depth_statistics.images.truncate(n);
    recon.image_table.depth_histogram_counts.truncate(n);

    let patch = plane_patch(world);
    let set = &mut recon.point_set;
    set.points = vec![Point3D {
        position: world,
        w: 1.0,
        color: [120, 130, 140],
        error: 0.5,
        normal: Vector3::new(0.0, 0.0, -1.0),
    }];
    set.tracks = vec![
        TrackObservation {
            image_index: 0,
            point_index: 0,
        },
        TrackObservation {
            image_index: 1,
            point_index: 0,
        },
    ];
    set.observation_counts = vec![2];
    let mut keypoints = Array2::<f32>::zeros((2, 2));
    for i in 0..2 {
        let p = scene.project(i, world);
        keypoints[[i, 0]] = p[0] as f32;
        keypoints[[i, 1]] = p[1] as f32;
    }
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; n],
    };
    let halfvec = |axis: Vector3<f64>, half: f64| {
        let v = axis * half;
        [v.x as f32, v.y as f32, v.z as f32]
    };
    let u = halfvec(patch.u_axis, patch.half_extent[0]);
    let v = halfvec(patch.v_axis, patch.half_extent[1]);
    set.patch_u_halfvec_xyz = Some(Array2::from_shape_vec((1, 3), u.to_vec()).unwrap());
    set.patch_v_halfvec_xyz = Some(Array2::from_shape_vec((1, 3), v.to_vec()).unwrap());
    recon.metadata.feature_source = sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// [`fixture`] carrying the optional per-observation and per-point columns a
/// created point has to fill in: an `(P, r, r, 4)` bitmap column, an
/// observation confidence and a normal confidence.
pub(crate) fn fixture_with_columns(
    scene: &Scene,
    world: Point3<f64>,
    r: usize,
) -> SfmrReconstruction {
    let mut recon = fixture(scene, world);
    let set = &mut recon.point_set;
    set.patch_bitmaps_y_x_rgba = Some(Arc::new(Array4::zeros((set.points.len(), r, r, 4))));
    set.observation_confidence = Some(vec![200; set.tracks.len()]);
    set.normal_confidence = Some(vec![180; set.points.len()]);
    recon.rebuild_derived_fields();
    recon
}

/// The fixture wrapped as a version with no edits.
pub(crate) fn edited(scene: &Scene, world: Point3<f64>) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(fixture(scene, world)))
}

pub(crate) const WORLD: Point3<f64> = Point3::new(0.0, 0.0, PLANE_Z);

// ── The refusals ─────────────────────────────────────────────────────

#[test]
fn a_sift_files_reconstruction_is_refused() {
    // `demo` is a `sift_files` value: its observations are feature indexes, and
    // a clicked pixel is not one.
    let scene = Scene::new();
    let base = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(4)));
    let err = add_observation(
        &base,
        0,
        2,
        [10.0, 10.0],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect_err("a sift_files base has no room for a featureless observation");
    assert_eq!(err, AddObservationError::NotEmbeddedPatches);
}

#[test]
fn an_image_already_in_the_track_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let err = add_observation(
        &value,
        0,
        1,
        [64.0, 64.0],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect_err("image 1 already observes the point");
    assert_eq!(err, AddObservationError::ImageAlreadyInTrack(1));
}

#[test]
fn an_image_past_the_table_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let err = add_observation(
        &value,
        0,
        9,
        [64.0, 64.0],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect_err("there is no image 9");
    assert_eq!(
        err,
        AddObservationError::ImageOutOfRange {
            image: 9,
            image_count: 3
        }
    );
}

#[test]
fn a_point_that_is_not_live_is_refused() {
    let scene = Scene::new();
    let mut value = edited(&scene, WORLD);
    value.delete_point(0).expect("a live point");
    let err = add_observation(
        &value,
        0,
        2,
        [64.0, 64.0],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect_err("point 0 was deleted");
    assert_eq!(err, AddObservationError::NoSuchPoint(0));
}

#[test]
fn a_pixel_outside_the_image_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    for pixel in [[-1.0, 10.0], [10.0, -1.0], [IMG_W as f32, 10.0]] {
        let err = add_observation(
            &value,
            0,
            2,
            pixel,
            &scene.views(),
            &AddObservationOptions::default(),
            &Progress::none(),
        )
        .expect_err("the pixel is off the sensor");
        assert!(matches!(err, AddObservationError::PixelOutsideImage { .. }));
    }
}

#[test]
fn too_few_views_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let views = scene.views();
    let err = add_observation(
        &value,
        0,
        2,
        [64.0, 64.0],
        &views[..2],
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect_err("the fit has no pixels for image 2");
    assert_eq!(
        err,
        AddObservationError::ViewsMissing {
            got: 2,
            expected: 3
        }
    );
}

#[test]
fn an_unreachable_acceptance_bar_refuses_rather_than_adding() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let truth = scene.project(2, WORLD);
    let options = AddObservationOptions {
        min_zncc: 1.5,
        ..AddObservationOptions::default()
    };
    let err = add_observation(
        &value,
        0,
        2,
        [truth[0] as f32, truth[1] as f32],
        &scene.views(),
        &options,
        &Progress::none(),
    )
    .expect_err("no ZNCC clears a bar above one");
    assert!(matches!(
        err,
        AddObservationError::BelowAcceptanceBar { bar, .. } if bar == 1.5
    ));
    assert_eq!(value.point_count(), 1, "a refusal changed the value");
}

// ── The edit itself ──────────────────────────────────────────────────

#[test]
fn the_fit_lands_the_added_observation_on_the_truth() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let truth = scene.project(2, WORLD);
    // Click two pixels off the truth: the fit has to walk there.
    let clicked = [(truth[0] + 2.0) as f32, (truth[1] - 1.5) as f32];

    let (next, report) = add_observation(
        &value,
        0,
        2,
        clicked,
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect("the patch registers in image 2");

    assert_eq!(report.image, 2);
    assert_eq!(report.replaced, 0);
    assert_eq!(report.observation_count, 3);
    assert!(
        report.zncc > 0.9,
        "the same surface should register almost perfectly, got {}",
        report.zncc
    );
    let clicked_err = (clicked[0] as f64 - truth[0]).hypot(clicked[1] as f64 - truth[1]);
    let err = (report.keypoint[0] as f64 - truth[0]).hypot(report.keypoint[1] as f64 - truth[1]);
    assert!(
        err < 1.0 && err < clicked_err,
        "the fit put the keypoint {err:.3} px from the truth, from a click {clicked_err:.3} px off"
    );
    assert!(report.shift_px > 1.0, "the fit did not move off the click");

    // The track gained the observation, in image order.
    let view = next.point(report.point).expect("the modified point");
    let images: Vec<u32> = view.observations().iter().map(|o| o.image_index).collect();
    assert_eq!(images, vec![0, 1, 2]);
}

#[test]
fn the_retriangulation_moves_the_point_toward_the_truth() {
    let scene = Scene::new();
    // A point displaced along the plane normal: the two stored observations are
    // still the truth's projections, so the third view's ray is what pulls the
    // solve back onto the plane.
    let displaced = Point3::new(WORLD.x, WORLD.y, WORLD.z + 0.08);
    let value = edited(&scene, displaced);
    // The stored keypoints in the fixture are the projections of the *stored*
    // position, so re-fit the fixture at the truth and copy them in.
    let value = {
        let truth_recon = fixture(&scene, WORLD);
        let mut moved = (*value.base).clone();
        moved.point_set.observations = truth_recon.point_set.observations.clone();
        moved.point_set.points[0].position = displaced;
        moved.rebuild_derived_fields();
        EditedReconstruction::new(Arc::new(moved))
    };

    let before = (value.point(0).expect("live").point().position - WORLD).norm();
    let truth = scene.project(2, WORLD);
    let (next, report) = add_observation(
        &value,
        0,
        2,
        [truth[0] as f32, truth[1] as f32],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect("the patch registers in image 2");

    let after = (next.point(report.point).expect("live").point().position - WORLD).norm();
    assert!(
        after < before,
        "the re-triangulation moved the point from {before:.4} to {after:.4} off the truth"
    );
    assert!(report.condition_number.is_finite());
    assert!(report.position_shift > 0.0);
}

#[test]
fn the_edit_leaves_the_base_and_the_input_value_alone() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let truth = scene.project(2, WORLD);
    let (next, _) = add_observation(
        &value,
        0,
        2,
        [truth[0] as f32, truth[1] as f32],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect("the patch registers in image 2");

    assert!(
        Arc::ptr_eq(&value.base, &next.base),
        "the edit produced a new base"
    );
    assert!(
        value.added.points.is_empty(),
        "the input value gained a point"
    );
    assert!(
        value.deleted_points.is_empty(),
        "the input value lost a point"
    );
    assert_eq!(value.point(0).expect("live").observations().len(), 2);
}

#[test]
fn the_point_keeps_its_place_and_its_columns_through_materialisation() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let truth = scene.project(2, WORLD);
    let (next, report) = add_observation(
        &value,
        0,
        2,
        [truth[0] as f32, truth[1] as f32],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect("the patch registers in image 2");

    // Everything the edit was not asked to change is unchanged.
    let before = value.point(0).expect("live");
    let after = next.point(report.point).expect("live");
    assert_eq!(before.point().color, after.point().color);
    assert_eq!(before.point().normal, after.point().normal);
    assert_eq!(before.patch_u_halfvec(), after.patch_u_halfvec());
    assert_eq!(before.patch_v_halfvec(), after.patch_v_halfvec());

    // The modification lands back at the base index it replaced, and a scan of
    // the materialised value agrees with the overlay read.
    let (plain, map) = next.materialize();
    assert_eq!(map.forward(report.point), Some(0));
    assert_eq!(plain.point_count(), 1);
    assert_eq!(plain.observations_for_point(0).len(), 3);
    let keypoints = plain.keypoints_xy().expect("embedded patches");
    assert_eq!(
        [keypoints[[2, 0]], keypoints[[2, 1]]],
        report.keypoint,
        "the added observation is the third row of the track"
    );
    assert_eq!(
        plain.point_set.points[0].position,
        after.point().position,
        "the materialised point is the edited one"
    );
}

// ── What the edit says it is doing ───────────────────────────────────

/// One phase a run closed: its name, its depth, and what it said it did.
type ClosedPhase = (&'static str, u8, Option<String>);

/// The `Leave` of every phase a run closed, in order.
fn closed_phases(events: &std::sync::Mutex<Vec<ClosedPhase>>) -> Vec<ClosedPhase> {
    events.lock().unwrap().clone()
}

/// Everything the edit writes into the point it replaced, as bit patterns: the
/// position, the patch frame and the track's keypoints. Bits rather than a
/// tolerance, because the question is whether reporting moved anything at all.
fn edit_bits(value: &EditedReconstruction, point: u32) -> Vec<u64> {
    let view = value.point(point).expect("the edited point is live");
    let mut bits = Vec::new();
    bits.extend(view.point().position.coords.iter().map(|c| c.to_bits()));
    bits.push(view.point().w.to_bits());
    bits.extend(view.point().normal.iter().map(|c| u64::from(c.to_bits())));
    for (k, observation) in view.observations().iter().enumerate() {
        bits.push(u64::from(observation.image_index));
        if let Some(keypoint) = view.keypoint_xy(k) {
            bits.extend(keypoint.iter().map(|c| u64::from(c.to_bits())));
        }
    }
    for halfvec in [view.patch_u_halfvec(), view.patch_v_halfvec()]
        .into_iter()
        .flatten()
    {
        bits.extend(halfvec.iter().map(|c| u64::from(c.to_bits())));
    }
    bits
}

/// The same for the report.
fn report_bits(report: &AddObservationReport) -> Vec<u64> {
    let mut bits = vec![
        u64::from(report.point),
        u64::from(report.replaced),
        u64::from(report.image),
        report.shift_px.to_bits(),
        report.zncc.to_bits(),
        report.observation_count as u64,
        report.position_shift.to_bits(),
        u64::from(report.from_infinity),
        report.condition_number.to_bits(),
    ];
    bits.extend(report.clicked_pixel.iter().map(|c| u64::from(c.to_bits())));
    bits.extend(report.keypoint.iter().map(|c| u64::from(c.to_bits())));
    bits
}

/// The two kernel calls are the two stages, and each says how many views it
/// ran over.
#[test]
fn a_recorded_run_names_the_localize_and_the_refine() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let truth = scene.project(2, WORLD);

    let events = std::sync::Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Leave {
            phase, depth, note, ..
        } = event
        {
            events
                .lock()
                .unwrap()
                .push((phase, depth, note.map(str::to_string)));
        }
    };
    add_observation(
        &value,
        0,
        2,
        [truth[0] as f32, truth[1] as f32],
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::to(&sink),
    )
    .expect("the patch registers in image 2");

    let phases = closed_phases(&events);
    assert_eq!(
        phases
            .iter()
            .map(|(name, depth, _)| (*name, *depth))
            .collect::<Vec<_>>(),
        [("localize", 0), ("refine", 0)],
        "{phases:?}"
    );
    // Three views: the two the track already held, seeded at their stored
    // keypoints, and the one the observation is being added in.
    for (name, _, note) in &phases {
        assert_eq!(
            note.as_deref(),
            Some("3 views"),
            "{name} did not say how many views it ran over"
        );
    }
}

/// Reporting is a side channel, so a recorded run has to place the observation
/// exactly where a silent one does.
#[test]
fn reporting_moves_not_one_bit_of_the_added_observation() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let truth = scene.project(2, WORLD);
    let clicked = [(truth[0] + 1.5) as f32, (truth[1] - 1.0) as f32];

    let (quiet, quiet_report) = add_observation(
        &value,
        0,
        2,
        clicked,
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect("the patch registers in image 2");

    let events = std::sync::Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Leave {
            phase, depth, note, ..
        } = event
        {
            events
                .lock()
                .unwrap()
                .push((phase, depth, note.map(str::to_string)));
        }
    };
    // Detail on as well, so every phase this call can open is open while the
    // arithmetic runs.
    let (loud, loud_report) = add_observation(
        &value,
        0,
        2,
        clicked,
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::to(&sink).detailed(true),
    )
    .expect("the patch registers in image 2");

    assert!(
        !closed_phases(&events).is_empty(),
        "the recorded run reported nothing, so it proves nothing"
    );
    assert_eq!(
        report_bits(&quiet_report),
        report_bits(&loud_report),
        "the two runs disagree about the report"
    );
    assert_eq!(
        edit_bits(&quiet, quiet_report.point),
        edit_bits(&loud, loud_report.point),
        "the two runs disagree about the value"
    );
}
