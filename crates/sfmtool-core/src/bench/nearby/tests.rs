// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The nearby-tracks building blocks, decided against the bench's synthetic
//! capture: pinhole cameras looking down world `+z` at a textured plane, put
//! near, far, or so far that it reads as infinity.

use nalgebra::{Point3, Vector3};

use crate::bench::tests::scene::{edited, Scene};
use crate::bench::track_at_pixel::ViewCamera;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;

use super::far_field::{average_linkage, best3, prominence};
use super::grey::blurred_grey;
use super::*;

/// Four cameras a metre or so apart, all looking down world `+z`.
const CENTERS: [[f64; 3]; 4] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 0.0],
    [0.3, 0.4, 0.0],
];

/// The pixel the tests query in image 0: the middle of the photograph.
const PIXEL: [f64; 2] = [64.0, 64.0];

fn sweep(scene: &Scene, depth: f64, options: &FarFieldOptions) -> FarFieldSweep {
    let edited = edited(scene, Point3::new(0.0, 0.0, depth));
    let views = scene.views();
    let grey = GreyImages::new(views.len());
    far_field_sweep(&edited, &views, &grey, 0, PIXEL, options, &Progress::none())
        .expect("the query is valid")
}

/// The pixels per unit of inverse distance of the image that moves the query
/// pixel most: for cameras that share a rotation, the focal length times the
/// largest baseline across the view axis.
fn widest_rate() -> f64 {
    let c0 = CENTERS[0];
    CENTERS[1..]
        .iter()
        .map(|c| 160.0 * (c[0] - c0[0]).hypot(c[1] - c0[1]))
        .fold(0.0, f64::max)
}

// ---- The grey images ---------------------------------------------------------

#[test]
fn the_blur_keeps_a_constant_image_constant_and_its_mass() {
    let image = ImageU8::new(12, 9, 3, vec![77; 12 * 9 * 3]);
    let grey = blurred_grey(&image);
    assert!(grey.data().iter().all(|&v| (v - 77.0).abs() < 1e-3));

    // A single bright pixel spreads to its neighbours and keeps its mass away
    // from the edges.
    let mut data = vec![0u8; 21 * 21];
    data[10 * 21 + 10] = 255;
    let grey = blurred_grey(&ImageU8::new(21, 21, 1, data));
    let total: f32 = grey.data().iter().sum();
    assert!((total - 255.0).abs() < 1e-2, "{total}");
    let peak = grey.data()[10 * 21 + 10];
    // The centre tap of the one-pixel Gaussian, squared: 0.39894348^2.
    assert!(
        (peak - 255.0 * 0.398_943_5 * 0.398_943_5).abs() < 1e-3,
        "{peak}"
    );
}

#[test]
fn the_grey_conversion_is_opencvs() {
    // cv2.cvtColor of (200, 100, 50) with COLOR_RGB2GRAY is 124.
    let grey = blurred_grey(&ImageU8::new(1, 1, 3, vec![200, 100, 50]));
    assert_eq!(grey.data()[0], 124.0);
}

#[test]
fn a_sample_needs_all_four_pixels() {
    let data: Vec<f32> = (0..20).map(|v| v as f32).collect();
    let image = GreyImage::new(5, 4, data);
    // At a pixel centre the sample is that pixel.
    assert_eq!(sample_grey(&image, 2.5, 1.5), Some(7.0));
    // Halfway between two pixels, their mean.
    assert_eq!(sample_grey(&image, 3.0, 1.5), Some(7.5));
    // The last pixel centre in a row reads a pixel past the edge.
    assert_eq!(sample_grey(&image, 4.5, 1.5), None);
    assert_eq!(sample_grey(&image, 0.49, 1.5), None);
    assert!(sample_grey(&image, 0.5, 0.5).is_some());
}

// ---- Projecting a direction --------------------------------------------------

#[test]
fn a_direction_projects_where_the_scene_puts_it() {
    let scene = Scene::from_centers(&CENTERS, 4.0);
    let views = scene.views();
    let d = Vector3::new(0.05, -0.08, 1.0).normalize();
    for (i, view) in views.iter().enumerate() {
        let got = ViewCamera::new(view)
            .project_direction(&d)
            .expect("in front of every camera");
        let want = scene.project_homogeneous(i, Point3::from(d), 0.0);
        assert!((got[0] - want[0]).abs() < 1e-9 && (got[1] - want[1]).abs() < 1e-9);
    }
    // Behind a perspective camera, a direction has no pixel.
    let camera = ViewCamera::new(&views[0]);
    assert!(camera.project_direction(&-d).is_none());
}

// ---- Reading a patch along a ray ---------------------------------------------

#[test]
fn the_patch_reads_best_at_the_planes_distance() {
    let depth = 4.0;
    let scene = Scene::from_centers(&CENTERS, depth);
    let views = scene.views();
    let grey = GreyImages::new(views.len());
    let camera = ViewCamera::new(&views[0]);
    let ray = camera.ray(PIXEL).normalize();
    // The plane z = depth, along the pixel's ray.
    let t = (depth - camera.center.z) / ray.z;
    let patch = RayPatch {
        image: 0,
        pixel: PIXEL,
        radius_px: 8.0,
    };
    let distances = [t * 0.8, t, t * 1.25, f64::INFINITY];
    let read = read_patch_along_ray(&views, &grey, &patch, &distances, &[1, 2, 3], true)
        .expect("the patch is textured and inside the photograph");
    let point = camera.center + ray * t;
    for v in 0..3 {
        assert!(read.whole[[1, v]] > 0.99, "{}", read.whole[[1, v]]);
        assert!(read.middle[[1, v]] > 0.98, "{}", read.middle[[1, v]]);
        for k in [0, 2, 3] {
            assert!(read.whole[[k, v]] < read.whole[[1, v]]);
        }
        let want = scene.project(v + 1, Point3::from(point));
        assert!((read.centres[[1, v, 0]] - want[0]).abs() < 1e-6);
        assert!((read.centres[[1, v, 1]] - want[1]).abs() < 1e-6);
    }
    assert!(read.middle_std > 8.0);
    let samples = read.samples.expect("asked for");
    assert_eq!(samples.template.len(), PATCH_GRID * PATCH_GRID);
    assert_eq!(samples.middle.iter().filter(|&&m| m).count(), 25);
    assert!(samples.values.iter().all(|v| v.is_finite()));
}

#[test]
fn a_patch_off_the_photograph_or_flat_is_not_read() {
    let scene = Scene::from_centers(&CENTERS, 4.0);
    let views = scene.views();
    let grey = GreyImages::new(views.len());
    let off = RayPatch {
        image: 0,
        pixel: [3.0, 64.0],
        radius_px: 8.0,
    };
    assert!(read_patch_along_ray(&views, &grey, &off, &[f64::INFINITY], &[1], false).is_none());

    let camera = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 160.0,
            focal_length_y: 160.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
        },
        width: 128,
        height: 128,
    };
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let pyramid = ImageU8Pyramid::build(&ImageU8::new(128, 128, 1, vec![90; 128 * 128]), 2);
    let flat = vec![
        ProjectedImage {
            camera: &camera,
            cam_from_world: &pose,
            pyramid: &pyramid,
        };
        2
    ];
    let patch = RayPatch {
        image: 0,
        pixel: PIXEL,
        radius_px: 8.0,
    };
    let grey = GreyImages::new(2);
    assert!(read_patch_along_ray(&flat, &grey, &patch, &[f64::INFINITY], &[1], false).is_none());
}

// ---- The far-field sweep -----------------------------------------------------

#[test]
fn a_plane_at_infinity_reads_at_disparity_zero() {
    let scene = Scene::from_centers(&CENTERS, 1e7);
    let found = sweep(&scene, 1e7, &FarFieldOptions::default());
    assert!(found.dropped.is_empty());
    let [reading] = found.readings.as_slice() else {
        panic!("one reading, got {:?}", found.readings);
    };
    assert_eq!(reading.disparity, 0.0);
    assert!(reading.at_infinity);
    assert_eq!(reading.depth, f64::INFINITY);
    assert_eq!(reading.refit, Some(Refit::Agrees));
    assert_eq!(reading.views.len(), CENTERS.len());
    assert_eq!(reading.views[0], (0, PIXEL));
    let range = reading.range.expect("a sweep reading has a range");
    let rate = reading.metrics.widest_px;
    assert!((rate - widest_rate()).abs() < 1e-3 * rate, "{rate}");
    assert!((range[0] - rate / 0.5).abs() < 1e-9 * rate);
    assert_eq!(range[1], f64::INFINITY);
    assert!(reading.metrics.whole > 0.99);
    assert_eq!(reading.metrics.peak_rank, 1);
    assert!(!reading.metrics.middle_flat);
    let grouping = reading.grouping.as_ref().expect("the refit ran");
    assert_eq!(grouping.groups, vec![vec![0, 1, 2, 3]]);
    assert_eq!(grouping.left_out, 0);
}

#[test]
fn a_plane_in_the_far_field_reads_at_its_disparity() {
    let rate = widest_rate();
    let depth = rate / 6.0;
    let scene = Scene::from_centers(&CENTERS, depth);
    let found = sweep(&scene, depth, &FarFieldOptions::default());
    let best = found
        .readings
        .first()
        .expect("the plane's distance is a peak");
    assert_eq!(best.disparity, 6.0);
    assert!(!best.at_infinity);
    assert!((best.depth - depth).abs() < 0.02 * depth, "{}", best.depth);
    let [near, far] = best.range.expect("a sweep reading has a range");
    assert!(near < depth && depth < far, "{near} {depth} {far}");
    assert!(best.metrics.whole > 0.99);
    assert!(best.max_ray_angle_deg > 0.0);
    assert_eq!(best.refit, Some(Refit::Agrees));
    // Nothing reads the plane at infinity.
    assert!(found.readings.iter().all(|r| !r.at_infinity));
}

#[test]
fn a_near_plane_is_not_read_as_far() {
    // At 4 units the widest image moves the pixel by about 44 px, beyond the
    // sweep's 16.
    let scene = Scene::from_centers(&CENTERS, 4.0);
    let found = sweep(&scene, 4.0, &FarFieldOptions::default());
    assert!(
        found
            .readings
            .iter()
            .all(|r| !r.at_infinity && r.disparity > 0.0),
        "{:?}",
        found.readings
    );
}

#[test]
fn without_the_refit_the_readings_carry_no_grouping() {
    let scene = Scene::from_centers(&CENTERS, 1e7);
    let options = FarFieldOptions {
        refit: false,
        ..FarFieldOptions::default()
    };
    let found = sweep(&scene, 1e7, &options);
    let [reading] = found.readings.as_slice() else {
        panic!("one reading");
    };
    assert!(reading.grouping.is_none() && reading.refit.is_none());
    assert_eq!(reading.views.len(), CENTERS.len());
}

#[test]
fn the_sweep_refuses_a_query_that_names_no_place() {
    let scene = Scene::from_centers(&CENTERS, 4.0);
    let edited = edited(&scene, Point3::new(0.0, 0.0, 4.0));
    let views = scene.views();
    let grey = GreyImages::new(views.len());
    let run = |image, pixel, grey: &GreyImages| {
        far_field_sweep(
            &edited,
            &views,
            grey,
            image,
            pixel,
            &FarFieldOptions::default(),
            &Progress::none(),
        )
    };
    assert!(matches!(
        run(9, PIXEL, &grey),
        Err(FarFieldError::NoSuchImage { .. })
    ));
    assert!(matches!(
        run(0, [-1.0, 3.0], &grey),
        Err(FarFieldError::PixelOffImage { .. })
    ));
    assert!(matches!(
        run(0, PIXEL, &GreyImages::new(2)),
        Err(FarFieldError::InputMismatch { input: "grey", .. })
    ));
}

// ---- The arithmetic ----------------------------------------------------------

#[test]
fn best3_is_the_mean_of_the_three_highest_read_values() {
    assert!((best3([0.5, -1.0, 0.9, 0.7, 0.8].into_iter()) - 0.8).abs() < 1e-12);
    assert_eq!(best3([0.4, -1.0].into_iter()), 0.4);
    assert!(best3([-1.0, -1.0].into_iter()).is_nan());
}

#[test]
fn prominence_is_measured_to_the_nearest_higher_peak() {
    let key = [0.9, 0.5, 0.7, 0.6, 0.95, 0.2];
    // The 0.7 peak: the lowest toward 0.9 is 0.5, toward 0.95 is 0.6; the
    // higher of the two is the base.
    assert!((prominence(&key, 2) - 0.1).abs() < 1e-12);
    // The highest stands above the lowest of all.
    assert!((prominence(&key, 4) - 0.75).abs() < 1e-12);
    // A first peak with nothing higher on its side is measured from the other.
    assert!((prominence(&key, 0) - 0.4).abs() < 1e-12);
    // An unread value counts as the lowest.
    let key = [0.8, f64::NEG_INFINITY, 0.9];
    assert_eq!(prominence(&key, 0), f64::INFINITY);
}

#[test]
fn average_linkage_splits_two_groups() {
    // Images 0, 2 and 4 agree with each other, 1 and 3 with each other, and
    // the two sets do not.
    let n = 5;
    let mut similar = vec![0.2; n * n];
    for a in 0..n {
        for b in 0..n {
            if a == b || a % 2 == b % 2 {
                similar[a * n + b] = if a == b { 1.0 } else { 0.95 };
            }
        }
    }
    let mut groups = average_linkage(&similar, n, 0.9);
    groups.iter_mut().for_each(|g| g.sort_unstable());
    assert_eq!(groups, vec![vec![0, 2, 4], vec![1, 3]]);
    // At a low enough cut they merge.
    assert_eq!(average_linkage(&similar, n, 0.1).len(), 1);
    // A flat similarity (NaN) merges nothing.
    assert_eq!(average_linkage(&[f64::NAN; 4], 2, 0.9).len(), 2);
}
