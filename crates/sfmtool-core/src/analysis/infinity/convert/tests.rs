// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::*;
use crate::analysis::reprojection_noise::tests::observe;
use crate::reconstruction::data::PointConstraintColumns;
use sfmtool_sfmr_format::{POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD};

#[test]
fn camera_extents_is_bbox_diagonal() {
    let centers = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(3.0, 4.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    ];
    assert!((camera_extents(&centers) - 5.0).abs() < 1e-9);
    assert_eq!(camera_extents(&[]), 0.0);
}

/// The per-axis noise every observed demo below is generated with.
const NOISE_PX: f64 = 0.5;

/// `SfmrReconstruction::demo(n)` with its points as stored observed: inline
/// keypoints at their projections plus Gaussian noise of [`NOISE_PX`].
fn observed_demo(n: usize, seed: u64) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(n);
    observe(&mut recon, [NOISE_PX, NOISE_PX], seed);
    recon
}

fn centers_of(recon: &SfmrReconstruction) -> Vec<Point3<f64>> {
    recon
        .image_table
        .images
        .iter()
        .map(|im| im.camera_center())
        .collect()
}

/// A point `dist` from the midpoint of the two cameras observing demo point
/// `p`, in the direction from that midpoint towards the world origin, which
/// both cameras look at: in view of both at any distance.
fn far_point(recon: &SfmrReconstruction, p: usize, dist: f64) -> Point3<f64> {
    let centers = centers_of(recon);
    let obs = recon.observations_for_point(p);
    let mid = (centers[obs[0].image_index as usize].coords
        + centers[obs[1].image_index as usize].coords)
        / 2.0;
    Point3::from(mid + dist * (-mid).normalize())
}

/// Store point `p` as a point at infinity along `dir`.
fn store_as_bearing(recon: &mut SfmrReconstruction, p: usize, dir: Vector3<f64>) {
    let point = &mut recon.point_set.points[p];
    point.position = Point3::from(dir.normalize());
    point.w = 0.0;
    point.normal = Vector3::zeros();
    recon.point_set.infinity_point_count = count_points_at_infinity(&recon.point_set.points);
}

#[test]
fn classify_leaves_well_conditioned_points_finite() {
    // demo() points sit ~1 unit from cameras ~5 units away: wide parallax.
    let recon = observed_demo(100, 1);
    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    let n_inf = classified
        .point_set
        .points
        .iter()
        .filter(|p| p.w == 0.0)
        .count();
    assert_eq!(n_inf, 0, "well-conditioned demo points must stay finite");
    assert_eq!(classified.point_set.infinity_point_count, 0);
    assert_eq!(summary.kept, 100);
    assert_eq!(summary.demoted + summary.promoted + summary.unscored, 0);
    // σ is the measured noise level, about the 0.5 px the keypoints carry.
    let sigma = summary.sigma_px.unwrap();
    assert!((sigma - NOISE_PX).abs() < 0.1, "sigma {sigma}");
    let noise = summary.noise.as_ref().unwrap();
    assert_eq!(noise.sigma_px, Some(sigma));
    assert_eq!(noise.observation_count, 200);
}

#[test]
fn classify_demotes_a_far_point_to_its_scored_bearing() {
    // Point 0 observed as if a million units away: its two rays are parallel
    // to far under the noise, so the verdict is a bearing.
    let mut recon = SfmrReconstruction::demo(50);
    let far = far_point(&recon, 0, 1.0e6);
    recon.point_set.points[0].position = far;
    observe(&mut recon, [NOISE_PX, NOISE_PX], 2);

    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    let pt = &classified.point_set.points[0];
    assert!(pt.is_at_infinity());
    assert_eq!(classified.point_set.infinity_point_count, 1);
    assert_eq!(classified.metadata.infinity_point_count, 1);
    assert_eq!(summary.demoted, 1);
    assert_eq!(summary.kept, 49);

    // The stored direction is the closed-form bearing of the score, a unit
    // vector, close to the true direction.
    let scores = recon
        .point_or_bearing_scores(Some(&[0]), summary.sigma_px, None)
        .unwrap();
    let bearing = scores.scores[0].unwrap().bearing;
    assert!((pt.position.coords - bearing.normalize()).norm() < 1e-12);
    assert!((pt.position.coords.norm() - 1.0).abs() < 1e-12);
    let truth = (far.coords - camera_cloud_centroid(&centers_of(&recon)).coords).normalize();
    assert!(pt.position.coords.angle(&truth) < 1e-3);
    // The normal is zeroed, and the error is the bearing's own residual,
    // which is of the order of the noise.
    assert_eq!(pt.normal, Vector3::zeros());
    assert!(
        pt.error > 0.0 && pt.error < 4.0 * NOISE_PX as f32,
        "{}",
        pt.error
    );
}

#[test]
fn classify_sigma_override_decides() {
    // Point 0 at 300 units: about 12 px of parallax between its two views,
    // finite at the measured 0.5 px but not at a stated 50 px.
    let mut recon = SfmrReconstruction::demo(50);
    recon.point_set.points[0].position = far_point(&recon, 0, 300.0);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 3);

    let (measured, summary) = recon.classify_points_at_infinity(None).unwrap();
    assert!(!measured.point_set.points[0].is_at_infinity());
    assert_eq!(summary.demoted, 0);

    let (stated, summary) = recon.classify_points_at_infinity(Some(50.0)).unwrap();
    assert!(stated.point_set.points[0].is_at_infinity());
    assert_eq!(summary.sigma_px, Some(50.0));
    assert!(summary.noise.is_none());
    // A larger σ can only add bearing verdicts.
    assert!(summary.demoted >= 1);
}

#[test]
fn classify_rejects_an_invalid_sigma() {
    let recon = observed_demo(10, 4);
    for s in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            recon.classify_points_at_infinity(Some(s)),
            Err(PointOrBearingError::InvalidNoiseLevel(_))
        ));
    }
}

#[test]
fn classify_without_its_pixels_is_an_error() {
    // demo() has no inline keypoints and no .sift files behind it.
    let recon = SfmrReconstruction::demo(10);
    assert!(matches!(
        recon.classify_points_at_infinity(None),
        Err(PointOrBearingError::Reconstruction(_))
    ));
}

#[test]
fn classify_promotes_a_bearing_whose_rays_ask_for_a_depth() {
    // Point 0 is stored as a bearing, but its keypoints are those of the near
    // point it is: the verdict is finite, and the fit places it there.
    let mut recon = observed_demo(50, 5);
    let truth = recon.point_set.points[0].position;
    let origin = camera_cloud_centroid(&centers_of(&recon));
    store_as_bearing(&mut recon, 0, truth - origin);

    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    assert_eq!(summary.promoted, 1);
    assert_eq!(summary.kept, 49);
    let pt = &classified.point_set.points[0];
    assert_eq!(pt.w, 1.0);
    assert_eq!(classified.point_set.infinity_point_count, 0);
    assert!(
        (pt.position - truth).norm() < 0.05,
        "placed at {:?}, truth {truth:?}",
        pt.position
    );
    assert_eq!(pt.normal, Vector3::zeros());
    // The error is recomputed at the placed point: of the order of the noise,
    // where the bearing's residuals were the full parallax.
    assert!(pt.error < 4.0 * NOISE_PX as f32, "{}", pt.error);
}

#[test]
fn classify_without_finite_points_changes_nothing() {
    // Every point at infinity: there is no finite point to measure the noise
    // from, so nothing is decided.
    let mut recon = observed_demo(20, 6);
    let origin = camera_cloud_centroid(&centers_of(&recon));
    for p in 0..20 {
        let d = recon.point_set.points[p].position - origin;
        store_as_bearing(&mut recon, p, d);
    }
    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    assert_eq!(summary.sigma_px, None);
    assert_eq!(summary.noise.as_ref().unwrap().observation_count, 0);
    assert_eq!(
        summary.promoted + summary.demoted + summary.kept + summary.unscored,
        0
    );
    assert!(classified
        .point_set
        .points
        .iter()
        .all(|p| p.is_at_infinity()));
    assert_eq!(classified.point_set.infinity_point_count, 20);

    // A stated noise level decides them, and since their keypoints are those
    // of near points, promotes them. The minimum depth falls back to the
    // camera extents for want of a finite point.
    let (promoted, summary) = recon.classify_points_at_infinity(Some(NOISE_PX)).unwrap();
    assert_eq!(summary.promoted, 20);
    assert_eq!(promoted.point_set.infinity_point_count, 0);
}

#[test]
fn classify_does_not_demote_a_bearing_behind_a_camera() {
    // Point 0 between two cameras that face each other: its rays are opposite,
    // so a bearing along either fits them with no cost and the verdict is a
    // bearing, but the bearing is behind one of the two cameras. The point is
    // left as the solve produced it.
    let mut recon = SfmrReconstruction::demo(20);
    // Point 0 is seen by images 0 and 1; move the second sighting to image 4,
    // opposite image 0 across the ring.
    recon.point_set.tracks[1].image_index = 4;
    recon.rebuild_derived_fields();
    recon.point_set.points[0].position = Point3::new(0.0, 0.0, 1.5);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 7);

    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    assert_eq!(summary.bearing_behind_camera, 1);
    assert_eq!(summary.demoted, 0);
    assert_eq!(
        classified.point_set.points[0].position,
        recon.point_set.points[0].position
    );
    assert!(!classified.point_set.points[0].is_at_infinity());
}

#[test]
fn classify_does_not_promote_without_a_usable_point() {
    // Image 1 sits one unit behind image 0 and looks the same way, and point 0
    // is a thousandth of a unit in front of image 0, off its axis: the rays
    // cross at a wide angle, so the verdict is finite, but the only point
    // that explains them is on top of image 0's centre, far under the minimum
    // depth. The point stays a bearing.
    let mut recon = SfmrReconstruction::demo(50);
    let (rotation, c0) = {
        let image = &recon.image_table.images[0];
        (image.quaternion_wxyz, image.camera_center())
    };
    // The camera looks down its −z axis.
    let forward = rotation.inverse() * -Vector3::z();
    let right = rotation.inverse() * Vector3::x();
    let c1 = c0 - forward;
    recon.image_table.images[1].quaternion_wxyz = rotation;
    recon.image_table.images[1].translation_xyz = -(rotation * c1.coords);
    let near = c0 + 1.0e-3 * (forward + 0.5 * right);
    recon.point_set.points[0].position = near;
    observe(&mut recon, [NOISE_PX, NOISE_PX], 8);
    let origin = camera_cloud_centroid(&centers_of(&recon));
    store_as_bearing(&mut recon, 0, near - origin);

    let scores = recon
        .point_or_bearing_scores(Some(&[0]), None, None)
        .unwrap();
    assert!(is_finite(
        &scores.scores[0].unwrap(),
        DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD
    ));

    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    assert_eq!(summary.no_usable_point, 1);
    assert_eq!(summary.promoted, 0);
    assert!(classified.point_set.points[0].is_at_infinity());
    assert_eq!(
        classified.point_set.points[0].position,
        recon.point_set.points[0].position
    );
}

#[test]
fn classify_leaves_a_point_neither_a_point_nor_a_bearing_describes() {
    // Point 0 is stored finite a thousandth of a unit in front of image 0,
    // off its axis, and image 1 sits one unit further along image 0's axis,
    // looking back at it. The two rays cross at the point at a wide angle, so
    // the verdict is finite, but the point is on top of image 0's centre,
    // under the minimum depth, and the fit from it finds nothing else. The
    // rays point almost opposite ways, so the bearing is behind one of the
    // cameras. Neither representation describes both sightings: the point is
    // left as stored and counted.
    let mut recon = SfmrReconstruction::demo(1);
    let (rotation, c0) = {
        let image = &recon.image_table.images[0];
        (image.quaternion_wxyz, image.camera_center())
    };
    // The camera looks down its −z axis; turning it half a turn about its own
    // y axis makes it look back.
    let forward = rotation.inverse() * -Vector3::z();
    let right = rotation.inverse() * Vector3::x();
    let c1 = c0 + forward;
    let back = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI)
        * rotation;
    recon.image_table.images[1].quaternion_wxyz = back;
    recon.image_table.images[1].translation_xyz = -(back * c1.coords);
    let near = c0 + 1.0e-3 * (forward + 0.5 * right);
    recon.point_set.points[0].position = near;
    observe(&mut recon, [NOISE_PX, NOISE_PX], 18);
    let before = recon.point_set.points[0].clone();

    let scores = recon
        .point_or_bearing_scores(Some(&[0]), Some(NOISE_PX), None)
        .unwrap();
    let score = scores.scores[0].unwrap();
    assert!(is_finite(&score, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD));
    assert!(!score.bearing_in_front_of_all_cameras);

    let (classified, summary) = recon.classify_points_at_infinity(Some(NOISE_PX)).unwrap();
    assert_eq!(summary.left_unusable, 1);
    assert_eq!(summary.refitted + summary.demoted + summary.kept, 0);
    let after = &classified.point_set.points[0];
    assert_eq!(after.position, before.position);
    assert_eq!(after.w, before.w);
    assert_eq!(after.error, before.error);
}

#[test]
fn classify_releases_the_constraints_of_the_points_it_changes() {
    // Point 0 is demoted, point 1 promoted, and point 2 kept; each is held.
    let mut recon = SfmrReconstruction::demo(50);
    recon.point_set.points[0].position = far_point(&recon, 0, 1.0e6);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 9);
    let origin = camera_cloud_centroid(&centers_of(&recon));
    let d1 = recon.point_set.points[1].position - origin;
    store_as_bearing(&mut recon, 1, d1);
    let mut constraints = PointConstraintColumns::all_free(50);
    for p in 0..3 {
        constraints.point_constraints[p] = POINT_CONSTRAINT_HELD;
    }
    recon.point_set.point_constraints = Some(constraints);

    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    assert_eq!((summary.demoted, summary.promoted), (1, 1));
    let released = classified.point_set.point_constraints.as_ref().unwrap();
    assert_eq!(released.point_constraints[0], POINT_CONSTRAINT_FREE);
    assert_eq!(released.point_constraints[1], POINT_CONSTRAINT_FREE);
    assert_eq!(released.point_constraints[2], POINT_CONSTRAINT_HELD);
}

#[test]
fn classify_zeroes_the_normal_confidence_of_the_points_it_changes() {
    let mut recon = SfmrReconstruction::demo(50);
    recon.point_set.points[0].position = far_point(&recon, 0, 1.0e6);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 10);
    recon.point_set.normal_confidence = Some(vec![255; 50]);

    let (classified, _) = recon.classify_points_at_infinity(None).unwrap();
    let confidence = classified.point_set.normal_confidence.as_ref().unwrap();
    assert_eq!(confidence[0], 0);
    assert!(confidence[1..].iter().all(|&c| c == 255));
}

#[test]
fn materialize_makes_every_point_finite() {
    let mut recon = SfmrReconstruction::demo(50);
    for pt in &mut recon.point_set.points {
        pt.position = Point3::from(pt.position.coords.normalize());
        pt.w = 0.0;
        pt.normal = Vector3::zeros();
    }
    recon.rebuild_derived_fields();
    recon.metadata.infinity_point_count = 50;
    let materialised = recon.materialize_points_at_infinity();
    assert!(materialised.point_set.points.iter().all(|p| p.w == 1.0));
    assert_eq!(materialised.point_set.infinity_point_count, 0);
    assert_eq!(materialised.metadata.infinity_point_count, 0);
}

#[test]
fn materialize_places_point_along_its_direction() {
    let mut recon = SfmrReconstruction::demo(1);
    let dir = Vector3::new(0.0, 0.0, 1.0);
    recon.point_set.points[0].position = Point3::from(dir);
    recon.point_set.points[0].w = 0.0;

    let materialised = recon.materialize_points_at_infinity();
    let pt = &materialised.point_set.points[0];
    assert_eq!(pt.w, 1.0);
    // The point lies on the ray origin + t·d, so (pt - origin) ∥ d.
    let origin = camera_cloud_centroid(&centers_of(&recon)).coords;
    let offset = pt.position.coords - origin;
    let perp = offset - offset.dot(&dir) * dir;
    assert!(perp.norm() < 1e-6, "materialised point must lie along d");
    assert!(offset.dot(&dir) > 0.0, "placed in the +d half-line");
}

#[test]
fn materialize_renormalises_non_unit_directions() {
    // A w = 0 point whose stored direction drifted off the unit sphere
    // must materialise to the same finite point as the unit direction:
    // the placement geometry depends only on the normalised direction.
    let mut unit = SfmrReconstruction::demo(1);
    unit.point_set.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 1.0));
    unit.point_set.points[0].w = 0.0;

    let mut scaled = SfmrReconstruction::demo(1);
    scaled.point_set.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 7.5));
    scaled.point_set.points[0].w = 0.0;

    let from_unit = unit.materialize_points_at_infinity();
    let from_scaled = scaled.materialize_points_at_infinity();
    let delta = (from_unit.point_set.points[0].position.coords
        - from_scaled.point_set.points[0].position.coords)
        .norm();
    assert!(
        delta < 1e-9,
        "non-unit direction must materialise identically"
    );
}

#[test]
fn materialize_leaves_zero_direction_point_untouched() {
    // A malformed w = 0 point with a zero-length direction cannot be
    // placed — it is left at w = 0 rather than producing a bogus finite
    // coordinate.
    let mut recon = SfmrReconstruction::demo(1);
    recon.point_set.points[0].position = Point3::from(Vector3::zeros());
    recon.point_set.points[0].w = 0.0;

    let materialised = recon.materialize_points_at_infinity();
    assert!(materialised.point_set.points[0].is_at_infinity());
}

#[test]
fn materialize_is_inverse_free_but_round_trips_classification() {
    // A genuine infinity point, materialised then reclassified, returns to
    // infinity: its keypoints still show no parallax.
    let mut recon = SfmrReconstruction::demo(50);
    let far = far_point(&recon, 0, 1.0e6);
    let origin = camera_cloud_centroid(&centers_of(&recon));
    store_as_bearing(&mut recon, 0, far - origin);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 11);

    let materialised = recon.materialize_points_at_infinity();
    assert!(!materialised.point_set.points[0].is_at_infinity());
    assert_eq!(materialised.point_set.infinity_point_count, 0);

    let (reclassified, _) = materialised.classify_points_at_infinity(None).unwrap();
    assert!(reclassified.point_set.points[0].is_at_infinity());
    assert_eq!(reclassified.point_set.infinity_point_count, 1);
}

/// Attach a patch frame (and a bitmap) to point 0 of a demo reconstruction.
fn with_patch(recon: &mut SfmrReconstruction, u0: Vector3<f64>, v0: Vector3<f64>) {
    let n = recon.point_set.points.len();
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    for k in 0..3 {
        u[[0, k]] = u0[k] as f32;
        v[[0, k]] = v0[k] as f32;
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    let mut b = Array4::<u8>::zeros((n, 2, 2, 4));
    b.index_axis_mut(Axis(0), 0).fill(200);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(b));
}

fn patch_row(a: &Option<Array2<f32>>) -> Vector3<f64> {
    let a = a.as_ref().unwrap();
    Vector3::new(
        f64::from(a[[0, 0]]),
        f64::from(a[[0, 1]]),
        f64::from(a[[0, 2]]),
    )
}

/// A patch for point 0 facing the camera-cloud centroid, with half-extents
/// `half_u` and `half_v`: `u × v` along the direction from the point to the
/// centroid, the convention a demotion produces.
fn facing_patch(
    recon: &SfmrReconstruction,
    half_u: f64,
    half_v: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let origin = camera_cloud_centroid(&centers_of(recon));
    let d = (recon.point_set.points[0].position - origin).normalize();
    let a = d.cross(&Vector3::z()).normalize();
    let b = d.cross(&a).normalize();
    let (u, v) = if a.cross(&b).dot(&d) < 0.0 {
        (a, b)
    } else {
        (b, a)
    };
    (u * half_u, v * half_v)
}

#[test]
fn classify_rescales_patch_frame_to_angular_extents() {
    // Demoting a far finite point to infinity turns its anchor into a unit
    // direction; the world-unit patch half-vectors must become angular
    // extents (divided by the demotion-time distance from the camera-cloud
    // centroid, preserving apparent size), tangent to the direction sphere,
    // with u x v along -d — the format's infinity-patch convention.
    let mut recon = SfmrReconstruction::demo(50);
    recon.point_set.points[0].position = far_point(&recon, 0, 10_000.0);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 12);
    with_patch(
        &mut recon,
        Vector3::new(50.0, 0.0, 0.0),
        Vector3::new(0.0, 25.0, 0.0),
    );
    let origin = camera_cloud_centroid(&centers_of(&recon));
    let dist = (recon.point_set.points[0].position.coords - origin.coords).norm();

    let (classified, _) = recon.classify_points_at_infinity(None).unwrap();
    assert!(classified.point_set.points[0].is_at_infinity());
    let d = classified.point_set.points[0].position.coords.normalize();
    let u = patch_row(&classified.point_set.patch_u_halfvec_xyz);
    let v = patch_row(&classified.point_set.patch_v_halfvec_xyz);
    // Tangent to the direction sphere.
    assert!(u.dot(&d).abs() < 1e-6 * u.norm());
    assert!(v.dot(&d).abs() < 1e-6 * v.norm());
    // Right-handed with the outward normal along -d.
    assert!(u.cross(&v).dot(&d) < 0.0);
    // The projection onto the tangent plane keeps each vector's in-plane
    // part, scaled by 1 / dist (the swap may exchange the axes).
    let in_plane = |w: Vector3<f64>| (w - w.dot(&d) * d).norm() / dist;
    let mut expected = [
        in_plane(Vector3::new(50.0, 0.0, 0.0)),
        in_plane(Vector3::new(0.0, 25.0, 0.0)),
    ];
    expected.sort_by(f64::total_cmp);
    let mut sizes = [u.norm(), v.norm()];
    sizes.sort_by(f64::total_cmp);
    for k in 0..2 {
        assert!((sizes[k] - expected[k]).abs() / expected[k] < 0.01);
    }
    // The bitmap is untouched by a demotion that keeps the patch.
    assert_eq!(
        classified
            .point_set
            .patch_bitmaps_y_x_rgba
            .as_ref()
            .unwrap()[[0, 0, 0, 0]],
        200
    );
}

#[test]
fn classify_leaves_finite_point_patches_untouched() {
    let mut recon = observed_demo(50, 13);
    with_patch(
        &mut recon,
        Vector3::new(5.0, 0.0, 0.0),
        Vector3::new(0.0, 5.0, 0.0),
    );
    let (classified, _) = recon.classify_points_at_infinity(None).unwrap();
    assert!(!classified.point_set.points[0].is_at_infinity());
    assert_eq!(
        classified.point_set.patch_u_halfvec_xyz.as_ref().unwrap()[[0, 0]],
        5.0
    );
}

#[test]
fn materialize_rescales_patch_frame_to_world_extents() {
    // The inverse boundary crossing: angular extents on the direction sphere
    // become world-unit extents at the placement depth.
    let mut recon = SfmrReconstruction::demo(1);
    recon.point_set.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 1.0));
    recon.point_set.points[0].w = 0.0;
    recon.point_set.points[0].normal = Vector3::zeros();
    with_patch(
        &mut recon,
        Vector3::new(0.01, 0.0, 0.0),
        Vector3::new(0.0, 0.005, 0.0),
    );

    let materialised = recon.materialize_points_at_infinity();
    let pt = &materialised.point_set.points[0];
    assert_eq!(pt.w, 1.0);
    let origin = camera_cloud_centroid(&centers_of(&recon));
    let t = (pt.position.coords - origin.coords).norm();
    let u = materialised.point_set.patch_u_halfvec_xyz.as_ref().unwrap();
    assert!((f64::from(u[[0, 0]]) - 0.01 * t).abs() / (0.01 * t) < 1e-6);
}

#[test]
fn classify_then_materialize_preserves_apparent_patch_size() {
    // Round trip: world extent / distance is invariant across the demotion
    // and the re-materialisation (the depths differ, the ratio must not).
    let mut recon = SfmrReconstruction::demo(50);
    recon.point_set.points[0].position = far_point(&recon, 0, 10_000.0);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 14);
    let (u0, v0) = facing_patch(&recon, 50.0, 25.0);
    with_patch(&mut recon, u0, v0);
    let origin = camera_cloud_centroid(&centers_of(&recon));
    let dist0 = (recon.point_set.points[0].position.coords - origin.coords).norm();
    let apparent0 = 50.0 / dist0;

    let (demoted, _) = recon.classify_points_at_infinity(None).unwrap();
    assert!(demoted.point_set.points[0].is_at_infinity());
    let round = demoted.materialize_points_at_infinity();
    assert!(!round.point_set.points[0].is_at_infinity());
    let dist1 = (round.point_set.points[0].position.coords - origin.coords).norm();
    let ua = patch_row(&round.point_set.patch_u_halfvec_xyz);
    let va = patch_row(&round.point_set.patch_v_halfvec_xyz);
    let apparent1 = ua.norm().max(va.norm()) / dist1;
    assert!(
        (apparent0 - apparent1).abs() / apparent0 < 0.01,
        "apparent size must survive the round trip: {apparent0} vs {apparent1}"
    );
}

#[test]
fn demotion_then_promotion_restores_the_patch_frame() {
    // Point 0, 50 units out, has about 76 px of parallax. A noise level far
    // above that demotes it, and the measured one promotes it again at about
    // the same place, so its patch comes back to the world extents it had.
    // The patch faces the camera-cloud centroid, a few degrees from the
    // bearing at that distance, so the demotion's projection onto the
    // direction sphere turns it a little and keeps its extents.
    let mut recon = SfmrReconstruction::demo(50);
    recon.point_set.points[0].position = far_point(&recon, 0, 50.0);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 15);
    let (u0, v0) = facing_patch(&recon, 2.0, 1.0);
    with_patch(&mut recon, u0, v0);
    let original = recon.point_set.points[0].position;
    let dist = (original - camera_cloud_centroid(&centers_of(&recon))).norm();

    let (demoted, summary) = recon.classify_points_at_infinity(Some(1.0e4)).unwrap();
    assert!(summary.demoted >= 1);
    assert!(demoted.point_set.points[0].is_at_infinity());
    // World extents became angular ones at the distance from the centroid.
    let ud = patch_row(&demoted.point_set.patch_u_halfvec_xyz);
    assert!((ud.norm() - 2.0 / dist).abs() < 0.01 * 2.0 / dist);

    let (promoted, summary) = demoted.classify_points_at_infinity(Some(NOISE_PX)).unwrap();
    assert!(summary.promoted >= 1);
    let pt = &promoted.point_set.points[0];
    assert_eq!(pt.w, 1.0);
    assert!(
        (pt.position - original).norm() < 0.02 * 50.0,
        "placed at {:?}, was {original:?}",
        pt.position
    );
    let u = patch_row(&promoted.point_set.patch_u_halfvec_xyz);
    let v = patch_row(&promoted.point_set.patch_v_halfvec_xyz);
    // The extents come back; the frame turns by the small angle between the
    // centroid's direction and the bearing it was projected to face.
    for (w, w0) in [(u, u0), (v, v0)] {
        assert!(
            (w.norm() - w0.norm()).abs() < 0.01 * w0.norm(),
            "{w:?} vs {w0:?}"
        );
        assert!(w.angle(&w0) < 0.1, "{w:?} vs {w0:?}");
    }
}

#[test]
fn classify_demotes_a_point_seen_from_one_centre() {
    // Point 0's two cameras collapse onto one centre after its keypoints were
    // taken, and the point is stored on that centre, as a solver that
    // collapsed a run of frames leaves it. The rays leave that centre in
    // directions 45 degrees apart, so no bearing fits them, and the two
    // centres differ only by round-off. They are one centre to the test,
    // which gives no depth score and a bearing verdict, and the point is
    // demoted. (Without that, the verdict would rest on the round-off, and a
    // finite one would still end in demotion: the stored position is on a
    // camera and the point fit finds nothing else.)
    let mut recon = SfmrReconstruction::demo(50);
    observe(&mut recon, [NOISE_PX, NOISE_PX], 16);
    with_patch(
        &mut recon,
        Vector3::new(0.1, 0.0, 0.0),
        Vector3::new(0.0, 0.1, 0.0),
    );
    let c0 = recon.image_table.images[0].camera_center();
    let r1 = recon.image_table.images[1].quaternion_wxyz;
    recon.image_table.images[1].translation_xyz = -(r1 * c0.coords);
    recon.point_set.points[0].position = c0;
    let c1 = recon.image_table.images[1].camera_center();
    assert!(
        (c1 - c0).norm() < 1e-12,
        "the centres coincide to round-off"
    );

    let scores = recon
        .point_or_bearing_scores(Some(&[0]), Some(NOISE_PX), None)
        .unwrap();
    let score = scores.scores[0].unwrap();
    assert!(score.bearing_cost > DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD);
    assert_eq!(score.depth_score, 0.0);
    assert_eq!(score.midpoint_bound, 0.0);

    let (classified, summary) = recon.classify_points_at_infinity(Some(NOISE_PX)).unwrap();
    let pt = &classified.point_set.points[0];
    assert!(pt.is_at_infinity());
    assert!(summary.demoted >= 1);
    assert!((pt.position.coords.norm() - 1.0).abs() < 1e-12);
    // The patch was solved at no depth, so there is no extent to convert.
    assert_eq!(
        patch_row(&classified.point_set.patch_u_halfvec_xyz),
        Vector3::zeros()
    );
}

#[test]
fn classify_refits_a_finite_point_stored_behind_its_cameras() {
    // Point 0's keypoints are those of the near point it is, but it is stored
    // reflected through its cameras, behind both. The verdict is finite, the
    // stored position is not one to keep, and the point fit from it finds the
    // point in front.
    let mut recon = observed_demo(50, 17);
    let truth = recon.point_set.points[0].position;
    let centers = centers_of(&recon);
    let obs = recon.observations_for_point(0);
    let mid = (centers[obs[0].image_index as usize].coords
        + centers[obs[1].image_index as usize].coords)
        / 2.0;
    recon.point_set.points[0].position = Point3::from(2.0 * mid - truth.coords);

    let (classified, summary) = recon.classify_points_at_infinity(None).unwrap();
    assert_eq!(summary.refitted, 1);
    assert_eq!(summary.kept, 49);
    let pt = &classified.point_set.points[0];
    assert_eq!(pt.w, 1.0);
    assert!((pt.position - truth).norm() < 0.05, "{:?}", pt.position);
}
