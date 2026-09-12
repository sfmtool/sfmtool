// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Creating a point at infinity from a pixel: the refusals, the bearing the
//! click names, the radius the frame is built from, and the second observation
//! that turns the bearing into a place.
//!
//! The scene is the one [`crate::reconstruction::add_observation`]'s tests
//! build -- pinhole cameras looking down world `+z` at a textured plane -- so a
//! click at a known projection has a known ray behind it.

use std::sync::Arc;

use nalgebra::Point3;

use crate::progress::Progress;
use crate::reconstruction::add_observation::tests::{
    edited, fixture, fixture_with_columns, Scene, IMG_W, WORLD,
};
use crate::reconstruction::add_observation::{
    add_observation, AddObservationError, AddObservationOptions, AddObservationReport,
};
use crate::reconstruction::data::SfmrReconstruction;
use crate::reconstruction::RowMap;

use super::*;

/// The patch radius most of these tests ask for, in image pixels.
const RADIUS: f32 = 12.0;

/// The focal length of the fixture's pinhole, which is what a radius in pixels
/// turns into an angle through.
const FOCAL: f64 = 160.0;

/// The value with every optional column present, so a created record has to
/// fill each of them in.
fn edited_with_columns(scene: &Scene, world: Point3<f64>, r: usize) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(fixture_with_columns(scene, world, r)))
}

// ── The refusals ─────────────────────────────────────────────────────

#[test]
fn a_sift_files_reconstruction_is_refused() {
    let scene = Scene::new();
    let base = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(4)));
    let err = create_point(
        &base,
        0,
        [10.0, 10.0],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect_err("a sift_files base has no room for a featureless observation");
    assert_eq!(err, CreatePointError::NotEmbeddedPatches);
}

#[test]
fn an_image_past_the_table_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let err = create_point(
        &value,
        9,
        [64.0, 64.0],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect_err("there is no image 9");
    assert_eq!(
        err,
        CreatePointError::ImageOutOfRange {
            image: 9,
            image_count: 3
        }
    );
}

#[test]
fn a_pixel_outside_the_image_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    for pixel in [[-1.0, 10.0], [10.0, -1.0], [IMG_W as f32, 10.0]] {
        let err = create_point(
            &value,
            2,
            pixel,
            RADIUS,
            &scene.views(),
            &CreatePointOptions::default(),
        )
        .expect_err("the pixel is off the sensor");
        assert!(matches!(err, CreatePointError::PixelOutsideImage { .. }));
    }
}

#[test]
fn a_radius_that_is_not_a_size_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    for radius in [0.0, -3.0, f32::NAN] {
        let err = create_point(
            &value,
            2,
            [64.0, 64.0],
            radius,
            &scene.views(),
            &CreatePointOptions::default(),
        )
        .expect_err("a patch has a positive size or it has none");
        assert!(matches!(err, CreatePointError::BadRadius(_)));
    }
}

#[test]
fn too_few_views_is_refused() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let views = scene.views();
    let err = create_point(
        &value,
        2,
        [64.0, 64.0],
        RADIUS,
        &views[..2],
        &CreatePointOptions::default(),
    )
    .expect_err("the call has no pixels for image 2");
    assert_eq!(
        err,
        CreatePointError::ViewsMissing {
            got: 2,
            expected: 3
        }
    );
}

// ── The point the click creates ──────────────────────────────────────

#[test]
fn the_direction_is_the_clicked_pixels_own_ray() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    // Click where the fixture's point projects in image 2: the ray behind that
    // pixel is the one from that camera centre to the known world point.
    let at = scene.project(2, WORLD);
    let (next, report) = create_point(
        &value,
        2,
        [at[0] as f32, at[1] as f32],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect("a pixel on the sensor has a ray");

    let view = next.point(report.point).expect("just created");
    assert_eq!(view.point().w, 0.0, "one sighting fixes no distance");
    let stored = view.point().position.coords;
    assert!((stored.norm() - 1.0).abs() < 1e-9, "the bearing is a unit");

    let centre = scene.views()[2].cam_from_world.inverse_translation_origin();
    let truth = (WORLD - centre).normalize();
    let angle = stored.dot(&truth).clamp(-1.0, 1.0).acos();
    assert!(
        angle < 1e-3,
        "the stored bearing is {angle} rad off the known ray"
    );

    // One observation, in the image that was clicked, at the clicked pixel.
    let track = view.observations();
    assert_eq!(track.len(), 1);
    assert_eq!(track[0].image_index, 2);
    assert_eq!(view.keypoint_xy(0), Some([at[0] as f32, at[1] as f32]));
}

#[test]
fn the_radius_sets_the_frames_angular_size() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    // At the principal point a pinhole's `radius_px` subtends exactly
    // `atan(radius / f)`, so the half-vector length is `radius / f`.
    for radius in [6.0_f32, 12.0, 24.0] {
        let (next, report) = create_point(
            &value,
            2,
            [64.0, 64.0],
            radius,
            &scene.views(),
            &CreatePointOptions::default(),
        )
        .expect("a pixel on the sensor has a ray");
        let expected = radius as f64 / FOCAL;
        assert!(
            (report.half_extent - expected).abs() < 1e-6,
            "{radius} px gave a half-extent of {} rather than {expected}",
            report.half_extent
        );
        let view = next.point(report.point).expect("just created");
        let u = view.patch_u_halfvec().expect("the frame column");
        let length = (u[0] as f64).hypot(u[1] as f64).hypot(u[2] as f64);
        assert!(
            (length - expected).abs() < 1e-5,
            "the stored half-vector is {length} long rather than {expected}"
        );
        // The frame is square and tangent to the bearing.
        let v = view.patch_v_halfvec().expect("the frame column");
        let v_len = (v[0] as f64).hypot(v[1] as f64).hypot(v[2] as f64);
        assert!((v_len - length).abs() < 1e-6, "the frame is not square");
        let d = view.point().position.coords;
        let u = nalgebra::Vector3::new(u[0] as f64, u[1] as f64, u[2] as f64);
        assert!(u.dot(&d).abs() < 1e-9, "the frame is not tangent to `d`");
    }
}

#[test]
fn the_optional_columns_are_filled_the_way_the_schema_needs() {
    let scene = Scene::new();
    let r = 9;
    let value = edited_with_columns(&scene, WORLD, r);
    let (next, report) = create_point(
        &value,
        2,
        [64.0, 64.0],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect("a pixel on the sensor has a ray");

    let view = next.point(report.point).expect("just created");
    assert_eq!(view.observation_confidence(), Some(&[u8::MAX][..]));
    assert_eq!(view.normal_confidence(), Some(0));
    assert_eq!(
        view.point().normal,
        nalgebra::Vector3::zeros(),
        "a point at infinity carries no normal"
    );
    let bitmap = view.patch_bitmap().expect("the bitmap column");
    assert_eq!(bitmap.shape(), &[r, r, 4]);
    assert!(
        (0..r).all(|y| (0..r).all(|x| bitmap[[y, x, 3]] == u8::MAX)),
        "the tile is opaque"
    );
    assert!(
        (0..r).any(|y| (0..r).any(|x| bitmap[[y, x, 0]] != 0)),
        "the tile sampled nothing"
    );
    // The colour is read out of the image at the pixel, so it is the texture's
    // grey rather than a default.
    assert!(report.color[0] > 0);
    assert_eq!(report.color[0], report.color[1]);
}

#[test]
fn the_edit_leaves_the_base_and_the_input_value_alone() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let (next, _) = create_point(
        &value,
        2,
        [64.0, 64.0],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect("a pixel on the sensor has a ray");
    assert!(
        Arc::ptr_eq(&value.base, &next.base),
        "the edit produced a new base"
    );
    assert!(value.added.points.is_empty());
    assert_eq!(value.point_count(), 1);
    assert_eq!(next.point_count(), 2);
}

#[test]
fn the_created_point_materialises_as_a_row_the_scan_calls_created() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let (next, report) = create_point(
        &value,
        2,
        [64.0, 64.0],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect("a pixel on the sensor has a ray");

    let (plain, map) = next.materialize();
    assert_eq!(plain.point_count(), 2, "the addition is appended");
    let row = map.forward(report.point).expect("the created point");
    assert_eq!(plain.point_set.points[row as usize].w, 0.0);

    let scan = RowMap::by_scan(&value.base, &plain, None).expect("the scan reads both values");
    assert_eq!(
        scan.inverse(row),
        None,
        "a created row has no index in the base"
    );
    assert_eq!(scan.forward(0), Some(0), "the base's point is carried over");
}

#[test]
fn the_point_edit_hash_names_the_content_and_nothing_else() {
    let scene = Scene::new();
    let value = edited(&scene, WORLD);
    let create = |pixel: [f32; 2]| {
        let (next, report) = create_point(
            &value,
            2,
            pixel,
            RADIUS,
            &scene.views(),
            &CreatePointOptions::default(),
        )
        .expect("a pixel on the sensor has a ray");
        let record = next.point(report.point).expect("just created").to_record();
        value
            .point_edit_hash(std::slice::from_ref(&record))
            .expect("a hashable record")
    };
    assert_eq!(
        create([64.0, 64.0]),
        create([64.0, 64.0]),
        "the same click twice is the same content"
    );
    assert_ne!(
        create([64.0, 64.0]),
        create([70.0, 60.0]),
        "a different pixel is different content"
    );
}

// ── The second observation, which makes it finite ────────────────────

/// How far off the truth the second sighting is clicked, in each axis, so the
/// fit has somewhere to walk from.
const CLICK_OFFSET: [f64; 2] = [2.0, -1.5];

/// Create a point on the known world point's bearing in image 0, then add the
/// sighting of it in image 1, clicked a couple of pixels off the truth.
fn create_then_observe(scene: &Scene) -> (EditedReconstruction, AddObservationReport, f64) {
    let value = edited_with_columns(scene, WORLD, 11);
    let at0 = scene.project(0, WORLD);
    let (created, report) = create_point(
        &value,
        0,
        [at0[0] as f32, at0[1] as f32],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect("a pixel on the sensor has a ray");

    let at1 = scene.project(1, WORLD);
    let clicked = [
        (at1[0] + CLICK_OFFSET[0]) as f32,
        (at1[1] + CLICK_OFFSET[1]) as f32,
    ];
    let (next, add) = add_observation(
        &created,
        report.point,
        1,
        clicked,
        &scene.views(),
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect("the second sighting triangulates the bearing");
    (next, add, report.half_extent)
}

#[test]
fn a_second_observation_triangulates_the_bearing_into_a_place() {
    let scene = Scene::new();
    let (next, add, _) = create_then_observe(&scene);
    assert!(add.from_infinity, "the point was a bearing before this");
    assert_eq!(add.observation_count, 2);

    let view = next.point(add.point).expect("the promoted point");
    assert_eq!(view.point().w, 1.0, "two bearings fix a place");
    let off = (view.point().position - WORLD).norm();
    assert!(
        off < 0.05,
        "the triangulation landed {off:.4} from the truth"
    );

    // The fit ran: the provisional triangulation put a finite patch in front of
    // both views, so the localizer had a projection to anchor on and reported a
    // real score, and the sub-pixel stage walked the click onto the surface.
    let truth = scene.project(1, WORLD);
    let clicked_err = CLICK_OFFSET[0].hypot(CLICK_OFFSET[1]);
    let err = (add.keypoint[0] as f64 - truth[0]).hypot(add.keypoint[1] as f64 - truth[1]);
    assert!(
        add.zncc.is_finite() && add.zncc > 0.9,
        "the same surface should register almost perfectly, got {}",
        add.zncc
    );
    assert!(
        add.shift_px > 1.0,
        "the fit did not move off the click: {}",
        add.shift_px
    );
    // Within a pixel and a half rather than a fraction of one: a created point's
    // frame is fronto-parallel to the camera it was created in, and this plane
    // is not, so the registration is limited by that orientation error rather
    // than by the fit. What is asserted is that the fit moved the sighting
    // toward the truth, which is what the two-pass path exists for.
    assert!(
        err < 1.5 && err < clicked_err,
        "the fit put the keypoint {err:.3} px from the truth, from a click {clicked_err:.3} px off"
    );
}

#[test]
fn the_promoted_frame_is_resized_at_the_triangulated_depth() {
    let scene = Scene::new();
    let (next, add, half_extent) = create_then_observe(&scene);
    let view = next.point(add.point).expect("the promoted point");

    // The angular half-extent times the placement distance -- the distance from
    // the camera-cloud centroid, which is what the infinity-to-finite
    // conversion scales by.
    let views = scene.views();
    let mut centroid = nalgebra::Vector3::zeros();
    for v in &views {
        centroid += v.cam_from_world.inverse_translation_origin().coords;
    }
    centroid /= views.len() as f64;
    let distance = (view.point().position.coords - centroid).norm();
    let expected = half_extent * distance;

    let u = view.patch_u_halfvec().expect("the frame column");
    let length = (u[0] as f64).hypot(u[1] as f64).hypot(u[2] as f64);
    assert!(
        (length - expected).abs() < 1e-3 * expected.max(1.0),
        "the promoted half-vector is {length} long rather than {expected}"
    );
    assert!(
        view.point().normal.norm() > 0.5,
        "a finite patch states a normal"
    );
    // The bitmap the fit was accepted against is the one the point still holds.
    assert!(view.patch_bitmap().is_some());
}

#[test]
fn a_created_point_keeps_its_bearing_when_the_base_carries_no_bitmaps() {
    // The bitmap column is optional, and a base without it gets a record
    // without it rather than a refusal.
    let scene = Scene::new();
    let value = EditedReconstruction::new(Arc::new(fixture(&scene, WORLD)));
    let (next, report) = create_point(
        &value,
        2,
        [64.0, 64.0],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect("a pixel on the sensor has a ray");
    let view = next.point(report.point).expect("just created");
    assert!(view.patch_bitmap().is_none());
    assert!(view.patch_u_halfvec().is_some());
}

#[test]
fn a_second_sighting_along_the_same_bearing_is_refused() {
    // Clicking in the second image at the pixel the bearing itself projects to
    // states the same direction twice: the two rays are parallel, and there is
    // no depth in them to solve for.
    let scene = Scene::new();
    let value = edited_with_columns(&scene, WORLD, 11);
    let at0 = scene.project(0, WORLD);
    let (created, report) = create_point(
        &value,
        0,
        [at0[0] as f32, at0[1] as f32],
        RADIUS,
        &scene.views(),
        &CreatePointOptions::default(),
    )
    .expect("a pixel on the sensor has a ray");

    let d = created
        .point(report.point)
        .expect("just created")
        .point()
        .position
        .coords;
    let views = scene.views();
    let cam = views[1].cam_from_world.to_rotation_matrix() * d;
    let (u, v) = views[1]
        .camera
        .ray_to_pixel([cam.x, cam.y, cam.z])
        .expect("the bearing points into image 1");
    let err = add_observation(
        &created,
        report.point,
        1,
        [u as f32, v as f32],
        &views,
        &AddObservationOptions::default(),
        &Progress::none(),
    )
    .expect_err("parallel rays have no intersection");
    assert_eq!(err, AddObservationError::Triangulation);
}
