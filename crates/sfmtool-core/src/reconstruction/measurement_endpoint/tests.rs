// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Resolving measurement endpoints on small synthetic reconstructions.
//!
//! The fixture is two cameras on a short arc at radius 8 around the origin
//! and a third posed image that observes nothing, with points near the origin
//! whose square patch frames face the cameras (`u = x`, `v = y`, `n = z`).
//! Every keypoint is the exact projection of its point unless a test moves it.
//! The scale tests (`measurement_scale/tests.rs`) build their scenes with it too.

use std::sync::Arc;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::{array, Array2};

use crate::camera::sampler::{Sampler, SamplerChoice};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::{RigidTransform, RotQuaternion, Se3Transform};
use crate::patch::normal_refine::PatchWindow;
use crate::reconstruction::data::{
    observation_reading_options, ObservationReading, ObservationReadings, ObservationSource,
    Point3D, SfmrImage, SfmrReconstruction, TrackObservation,
};
use crate::reconstruction::EditedReconstruction;
use crate::{DistanceMeasurementsData, LENGTH_UNIT_M};

use super::*;

const IMG_W: u32 = 640;
const IMG_H: u32 = 480;
const IMAGES: usize = 3;
/// The patch resolution the fixture's readings are taken at.
const R: u32 = 24;

/// One point of a fixture: its homogeneous position, its two half-vectors,
/// and the images that observe it.
pub(crate) struct Pt {
    pos: [f64; 3],
    w: f64,
    u: [f64; 3],
    v: [f64; 3],
    images: Vec<usize>,
}

/// A finite point at `(x, y, z)` with a square patch of half-size `h` facing
/// `+z`, observed by images 0 and 1.
pub(crate) fn pt(x: f64, y: f64, z: f64, h: f64) -> Pt {
    Pt {
        pos: [x, y, z],
        w: 1.0,
        u: [h, 0.0, 0.0],
        v: [0.0, h, 0.0],
        images: vec![0, 1],
    }
}

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
        rolling_shutter: None,
    }
}

/// Where the homogeneous point `(xyz, w)` lands in image `i`.
fn project(recon: &SfmrReconstruction, i: usize, xyz: Vector3<f64>, w: f64) -> [f64; 2] {
    let image = &recon.image_set.images[i];
    let local = image.quaternion_wxyz * xyz + image.translation_xyz * w;
    let (x, y) = recon.image_set.cameras[0]
        .ray_to_pixel([local.x, local.y, local.z])
        .expect("the fixture's points are in front of every camera");
    [x, y]
}

/// The fixture over `points`, with no readings.
pub(crate) fn scene(points: &[Pt]) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_set.cameras = vec![pinhole()];
    recon.image_set.images = (0..IMAGES)
        .map(|i| {
            // Images 0 and 2 share a pose; image 1 is 0.2 rad round the arc.
            let angle = if i == 1 { 0.1 } else { -0.1 };
            let centre = Vector3::new(8.0 * f64::sin(angle), 0.0, 8.0 * f64::cos(angle));
            let rotation = UnitQuaternion::face_towards(&centre, &Vector3::y()).inverse();
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: 0,
                quaternion_wxyz: rotation,
                translation_xyz: -(rotation * centre),
                intrinsic_adjustment: Default::default(),
                readings: None,
            }
        })
        .collect();
    let stats = recon.image_set.depth_statistics.images[0].clone();
    recon.image_set.depth_statistics.images = vec![stats; IMAGES];
    recon.image_set.depth_histogram_counts =
        vec![recon.image_set.depth_histogram_counts[0].clone(); IMAGES];

    recon.point_set.points = points
        .iter()
        .map(|p| Point3D {
            position: Point3::from(p.pos),
            w: p.w,
            color: [10, 20, 30],
            rms_reprojection_error: 0.25,
            normal: Vector3::new(0.0, 0.0, 1.0),
        })
        .collect();
    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f64; 2]> = Vec::new();
    for (p, spec) in points.iter().enumerate() {
        for &i in &spec.images {
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push(project(&recon, i, Vector3::from(spec.pos), spec.w));
        }
        counts.push(spec.images.len() as u32);
    }
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, xy) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = xy[0] as f32;
        keypoints_xy[[row, 1]] = xy[1] as f32;
    }
    let mut u = Array2::<f32>::zeros((points.len(), 3));
    let mut v = Array2::<f32>::zeros((points.len(), 3));
    for (p, spec) in points.iter().enumerate() {
        for c in 0..3 {
            u[[p, c]] = spec.u[c] as f32;
            v[[p, c]] = spec.v[c] as f32;
        }
    }
    let set = &mut recon.point_set;
    set.tracks = tracks;
    set.observation_counts = counts;
    set.keypoints_xy = Some(keypoints_xy);
    set.observations = ObservationSource::EmbeddedPatches {
        image_file_hashes: vec![[0u8; 16]; IMAGES],
    };
    set.patch_u_halfvec_xyz = Some(u);
    set.patch_v_halfvec_xyz = Some(v);
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// A stored reading with a round self-similarity ellipse of `radius` grid px
/// and a blur-matched score of `zncc` against the bitmap.
fn reading(radius: f32, zncc: f32) -> ObservationReading {
    ObservationReading {
        ellipse_axes: [radius, radius],
        ellipse_axes_is_at_least: [false, false],
        ellipse_major_angle: f32::NAN,
        cos_view_angle: 1.0,
        tilt_angle: f32::NAN,
        zoom: [1.0, 1.0],
        plain_bitmap_zncc: zncc,
        blur_matched_bitmap_zncc: zncc,
    }
}

/// `recon` with `rows` as its observation readings, at resolution [`R`].
fn with_readings(
    mut recon: SfmrReconstruction,
    rows: Vec<ObservationReading>,
) -> SfmrReconstruction {
    assert_eq!(rows.len(), recon.point_set.tracks.len());
    recon.point_set.observation_readings = Some(ObservationReadings {
        rows,
        options: observation_reading_options(
            SamplerChoice::Fixed(Sampler::Bilinear),
            R as usize,
            PatchWindow::Uniform,
        ),
    });
    recon
}

fn edited(recon: SfmrReconstruction) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(recon))
}

fn resolve(
    value: &EditedReconstruction,
    image: u32,
    pixel: [f64; 2],
) -> Result<EndpointResolution, EndpointRefusal> {
    resolve_measurement_endpoint(value, image, pixel, &MeasurementEndpointOptions::default())
}

/// The pixel of image `i` that shows texel `(s, t)` of point `p`'s square
/// there: the patch slid onto the observation's keypoint.
pub(crate) fn pixel_of_texel(
    recon: &SfmrReconstruction,
    p: u32,
    i: usize,
    s: f64,
    t: f64,
) -> [f64; 2] {
    let edited = edited(recon.clone());
    let view = edited.point(p).unwrap();
    let k = view
        .observations()
        .iter()
        .position(|o| o.image_index as usize == i)
        .expect("the point is observed in the image");
    let xy = view.keypoint_xy(k).unwrap();
    let image = &recon.image_set.images[i];
    let pose = RigidTransform::new(
        RotQuaternion::from_nalgebra(image.quaternion_wxyz),
        image.translation_xyz,
    );
    let patch = view.patch_frame().unwrap();
    let square = patch.seen_at_keypoint(
        &recon.image_set.cameras[0],
        &pose,
        Some([f64::from(xy[0]), f64::from(xy[1])]),
    );
    let (xyz, w) = square.corner_homogeneous(s, t);
    project(recon, i, xyz, w)
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} against {b}");
}

fn ranked(c: &EndpointCandidate) -> EndpointCost {
    match c.standing {
        CandidateStanding::Ranked(cost) => cost,
        other => panic!("point {} is not ranked: {other:?}", c.point),
    }
}

// ─── one square ───────────────────────────────────────────────────────────

/// A pixel in exactly one square resolves to it as `Alone`, and reads no
/// stored reading: stripping the readings leaves the resolution unchanged.
#[test]
fn a_pixel_in_one_square_resolves_alone_without_readings() {
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2), pt(1.5, 0.0, 0.0, 0.2)]);
    let rows = vec![reading(0.6, 0.9); recon.point_set.tracks.len()];
    let read = with_readings(recon.clone(), rows);
    let pixel = pixel_of_texel(&read, 0, 0, 0.4, -0.3);

    let with = resolve(&edited(read), 0, pixel).expect("one square holds the pixel");
    let without = resolve(&edited(recon), 0, pixel).expect("one square holds the pixel");
    assert_eq!(with.candidates.len(), 1);
    assert_eq!(with.chosen().point, 0);
    assert_eq!(with.chosen().standing, CandidateStanding::Alone);
    assert_eq!(with, without);
}

/// An observation's own keypoint is texel `(0, 0)` of its point's patch, and
/// a pixel off it is the texel its render shows there. The position is that
/// texel on the point's own patch, not on the slid square.
#[test]
fn the_pixel_maps_to_the_texel_its_render_shows() {
    let mut recon = scene(&[pt(0.1, -0.2, 0.0, 0.25)]);
    // Move image 0's keypoint 3 px right and 2 px down of the projection, so
    // the square slides off the point.
    if let Some(keypoints_xy) = recon.point_set.keypoints_xy.as_mut() {
        keypoints_xy[[0, 0]] += 3.0;
        keypoints_xy[[0, 1]] += 2.0;
    }
    let keypoint = {
        let view = edited(recon.clone());
        let xy = view.point(0).unwrap().keypoint_xy(0).unwrap();
        [f64::from(xy[0]), f64::from(xy[1])]
    };
    let value = edited(recon.clone());

    let at_keypoint = resolve(&value, 0, keypoint).unwrap();
    let [s, t] = at_keypoint.chosen().patch_coords;
    assert_close(s, 0.0, 1e-6, "s at the keypoint");
    assert_close(t, 0.0, 1e-6, "t at the keypoint");

    let pixel = pixel_of_texel(&recon, 0, 0, 0.5, -0.7);
    let off = resolve(&value, 0, pixel).unwrap();
    let [s, t] = off.chosen().patch_coords;
    assert_close(s, 0.5, 1e-6, "s off the keypoint");
    assert_close(t, -0.7, 1e-6, "t off the keypoint");
    let own = value
        .point(0)
        .unwrap()
        .patch_frame()
        .unwrap()
        .to_world(0.5, -0.7);
    let position = off.position().finite().unwrap();
    for c in 0..3 {
        assert_close(
            position[c],
            own[c],
            1e-9,
            "position on the point's own patch",
        );
    }
    // The slid square lies off the patch, so the same texel on it, where the
    // pixel's ray meets it, is another position.
    let image = &recon.image_set.images[0];
    let pose = RigidTransform::new(
        RotQuaternion::from_nalgebra(image.quaternion_wxyz),
        image.translation_xyz,
    );
    let slid = value
        .point(0)
        .unwrap()
        .patch_frame()
        .unwrap()
        .seen_at_keypoint(&recon.image_set.cameras[0], &pose, Some(keypoint))
        .to_world(0.5, -0.7);
    assert!((Vector3::from(position) - slid.coords).norm() > 0.01);
    assert!(off.chosen().depth > 7.0 && off.chosen().depth < 9.0);
}

// ─── several squares ──────────────────────────────────────────────────────

/// Two equal patches overlap; the pixel goes to the one it is nearer the
/// centre of.
#[test]
fn equal_patches_go_to_the_one_nearer_its_centre() {
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2), pt(0.25, 0.0, 0.0, 0.2)]);
    let value = edited(recon.clone());
    // Texel (0.85, 0) of point 0 is 0.17 from its centre and 0.08 from point
    // 1's.
    let pixel = pixel_of_texel(&recon, 0, 0, 0.85, 0.0);
    let near_one = resolve(&value, 0, pixel).unwrap();
    assert_eq!(near_one.candidates.len(), 2);
    assert_eq!(near_one.chosen().point, 1);
    let costs: Vec<_> = near_one.candidates.iter().map(ranked).collect();
    assert!(costs[1].offset < costs[0].offset);

    // And the other way round, nearer point 0's centre.
    let pixel = pixel_of_texel(&recon, 0, 0, 0.45, 0.0);
    assert_eq!(resolve(&value, 0, pixel).unwrap().chosen().point, 0);
}

/// A small patch holding the pixel at its centre beats a larger one, which
/// wins once the pixel is near the small one's edge.
#[test]
fn a_small_patch_wins_at_its_centre_and_loses_at_its_edge() {
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.1), pt(0.1, 0.0, 0.0, 0.15)]);
    let value = edited(recon.clone());

    let centre = pixel_of_texel(&recon, 0, 0, 0.0, 0.0);
    let at_centre = resolve(&value, 0, centre).unwrap();
    assert_eq!(at_centre.candidates.len(), 2);
    assert_eq!(at_centre.chosen().point, 0);
    let small = ranked(&at_centre.candidates[0]);
    let large = ranked(&at_centre.candidates[1]);
    assert!(small.localization < large.localization);

    let edge = pixel_of_texel(&recon, 0, 0, 0.95, 0.0);
    let at_edge = resolve(&value, 0, edge).unwrap();
    assert_eq!(at_edge.candidates.len(), 2);
    assert_eq!(at_edge.chosen().point, 1);
}

/// Squares of stacked surfaces: the far one is behind the front layer even
/// where its cost is the lower, and wins on cost once the layer reaches it.
#[test]
fn a_far_square_is_behind_the_front_layer_whatever_its_cost() {
    // Point 0 is 2 units nearer the cameras than point 1, on its line of
    // sight from image 0, and blurry; point 1 is sharp.
    let near = pt(0.0, 0.0, 2.0, 0.2);
    let recon0 = scene(&[near, pt(0.0, 0.0, 0.0, 0.2)]);
    let near_pixel = project(&recon0, 0, Vector3::new(0.0, 0.0, 2.0), 1.0);
    let far = {
        // The far point where image 0's ray through the near one meets z = 0.
        let dir = recon0.image_set.world_ray(0, near_pixel).unwrap();
        let o = recon0.image_set.images[0].camera_center();
        let lambda = -o.z / dir.z;
        o + dir * lambda
    };
    let recon = scene(&[pt(0.0, 0.0, 2.0, 0.2), pt(far.x, far.y, 0.0, 0.2)]);
    let rows = vec![
        reading(2.0, 0.9),
        reading(2.0, 0.9),
        reading(0.3, 0.9),
        reading(0.3, 0.9),
    ];
    let recon = with_readings(recon, rows);
    let value = edited(recon);

    let res = resolve(&value, 0, near_pixel).unwrap();
    assert_eq!(res.candidates.len(), 2);
    assert_eq!(res.chosen().point, 0);
    assert_eq!(
        res.candidates[1].standing,
        CandidateStanding::Excluded(EndpointExclusion::BehindFrontLayer)
    );
    assert!(res.candidates[1].depth > res.candidates[0].depth + 1.5);

    // A layer deep enough to hold both ranks them, and the far one's cost is
    // the lower.
    let deep = MeasurementEndpointOptions {
        layer_depth_factor: 100.0,
        ..Default::default()
    };
    let both = resolve_measurement_endpoint(&value, 0, near_pixel, &deep).unwrap();
    assert_eq!(both.chosen().point, 1);
    assert!(ranked(&both.candidates[1]).cost < ranked(&both.candidates[0]).cost);
}

/// A patch whose point has no observation in the image has no square there.
#[test]
fn a_patch_unobserved_in_the_image_is_no_candidate() {
    let mut lone = pt(0.0, 0.0, 0.0, 0.2);
    lone.images = vec![1];
    let recon = scene(&[lone, pt(1.5, 0.0, 0.0, 0.2)]);
    let pixel = project(&recon, 0, Vector3::zeros(), 1.0);
    assert_eq!(
        resolve(&edited(recon), 0, pixel),
        Err(EndpointRefusal::NoCoveringSquare { image: 0, pixel })
    );
    // Observed in the image, the same patch holds the pixel.
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2), pt(1.5, 0.0, 0.0, 0.2)]);
    assert_eq!(resolve(&edited(recon), 0, pixel).unwrap().chosen().point, 0);
}

/// The resolution reads the edited value: with the chosen point deleted, the
/// pixel goes to the other square that holds it.
#[test]
fn a_deleted_point_is_no_candidate() {
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2), pt(0.25, 0.0, 0.0, 0.2)]);
    let pixel = pixel_of_texel(&recon, 0, 0, 0.85, 0.0);
    let mut value = edited(recon);
    assert_eq!(resolve(&value, 0, pixel).unwrap().chosen().point, 1);
    value.delete_point(1).unwrap();
    let res = resolve(&value, 0, pixel).unwrap();
    assert_eq!(res.chosen().point, 0);
    assert_eq!(res.chosen().standing, CandidateStanding::Alone);
}

// ─── refusals ─────────────────────────────────────────────────────────────

#[test]
fn each_refusal_names_its_remedy() {
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2)]);
    let centre = project(&recon, 0, Vector3::zeros(), 1.0);
    let value = edited(recon.clone());

    let e = resolve(&value, 3, centre).unwrap_err();
    assert_eq!(
        e,
        EndpointRefusal::NoSuchImage {
            image: 3,
            image_count: 3
        }
    );
    assert!(e.to_string().contains("place the endpoint in one of them"));

    let mut unposed = recon.clone();
    unposed.image_set.images[0].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    let e = resolve(&edited(unposed), 0, centre).unwrap_err();
    assert_eq!(e, EndpointRefusal::NoPose { image: 0 });
    assert!(e.to_string().contains("resect the image"));

    for pixel in [[-1.0, 10.0], [10.0, 480.0], [f64::NAN, 10.0]] {
        let e = resolve(&value, 0, pixel).unwrap_err();
        assert!(
            matches!(e, EndpointRefusal::PixelOutsideImage { image: 0, .. }),
            "{pixel:?}: {e:?}"
        );
        assert!(e
            .to_string()
            .contains("place the endpoint inside the image"));
    }

    let mut frameless = recon.clone();
    frameless.point_set.patch_u_halfvec_xyz = None;
    frameless.point_set.patch_v_halfvec_xyz = None;
    let e = resolve(&edited(frameless), 0, centre).unwrap_err();
    assert_eq!(e, EndpointRefusal::NoPatchFrames);
    assert!(e.to_string().contains("sfm xform --to-embedded-patches"));

    let e = resolve(&value, 2, centre).unwrap_err();
    assert_eq!(e, EndpointRefusal::NoObservations { image: 2 });
    assert!(e.to_string().contains("Create Track Here"));

    let corner = [5.0, 5.0];
    let e = resolve(&value, 0, corner).unwrap_err();
    assert_eq!(
        e,
        EndpointRefusal::NoCoveringSquare {
            image: 0,
            pixel: corner
        }
    );
    assert!(e.to_string().contains("Create Track Here"));

    // A patch facing away from the cameras (`u = y`, `v = x`, `n = −z`).
    let mut behind = pt(0.0, 0.0, 0.0, 0.2);
    std::mem::swap(&mut behind.u, &mut behind.v);
    let e = resolve(&edited(scene(&[behind])), 0, centre).unwrap_err();
    match &e {
        EndpointRefusal::AllExcluded { candidates } => {
            assert_eq!(candidates.len(), 1);
            assert_eq!(
                candidates[0].standing,
                CandidateStanding::Excluded(EndpointExclusion::BackFacing)
            );
        }
        other => panic!("not AllExcluded: {other:?}"),
    }
    assert!(e.to_string().contains("seen from behind"));
}

// ─── points at infinity ───────────────────────────────────────────────────

/// A pixel in the square of a point at infinity resolves to a direction, and
/// a measurement with such an endpoint has no scene distance; one between two
/// finite endpoints has the distance between their positions.
#[test]
fn an_endpoint_at_infinity_has_a_direction_and_no_distance() {
    // A direction 0.1 rad off image 0's line of sight to the origin, with a
    // patch on the direction sphere 0.02 rad across.
    let d = Vector3::new(0.0, 0.0, -1.0);
    let sky = Pt {
        pos: [d.x, d.y, d.z],
        w: 0.0,
        u: [0.02, 0.0, 0.0],
        v: [0.0, -0.02, 0.0],
        images: vec![0, 1],
    };
    let mut recon = scene(&[sky, pt(1.5, 0.0, 0.0, 0.2), pt(-1.5, 0.0, 0.0, 0.2)]);
    let sky_pixel = pixel_of_texel(&recon, 0, 0, 0.3, 0.2);
    let a = pixel_of_texel(&recon, 1, 0, 0.0, 0.0);
    let b = pixel_of_texel(&recon, 2, 1, 0.0, 0.0);
    let value = edited(recon.clone());
    let res = resolve(&value, 0, sky_pixel).unwrap();
    assert_eq!(res.chosen().point, 0);
    assert_eq!(res.chosen().depth, f64::INFINITY);
    let [s, t] = res.chosen().patch_coords;
    assert_close(s, 0.3, 1e-6, "s on the sky");
    assert_close(t, 0.2, 1e-6, "t on the sky");
    let EndpointPosition::AtInfinity { direction } = res.position() else {
        panic!("not at infinity: {:?}", res.position());
    };
    let expected = (d + Vector3::new(0.3 * 0.02, 0.2 * -0.02, 0.0)).normalize();
    for c in 0..3 {
        assert_close(direction[c], expected[c], 1e-6, "direction");
    }

    recon.image_set.measurements.distance = Some(DistanceMeasurementsData {
        distances: array![1.0_f32, 1.0_f32],
        units: array![LENGTH_UNIT_M, LENGTH_UNIT_M],
        image_indexes: array![[0_u32, 0_u32], [0_u32, 1_u32]],
        endpoints_xy: array![
            [
                [sky_pixel[0] as f32, sky_pixel[1] as f32],
                [a[0] as f32, a[1] as f32]
            ],
            [[a[0] as f32, a[1] as f32], [b[0] as f32, b[1] as f32]]
        ],
        labels: vec!["to the sky".into(), "across".into()],
    });
    let resolved =
        resolve_distance_measurements(&edited(recon), &MeasurementEndpointOptions::default());
    assert_eq!(resolved.len(), 2);
    assert!(resolved[0].endpoints[0].is_ok() && resolved[0].endpoints[1].is_ok());
    assert_eq!(resolved[0].scene_distance, None);
    // The two finite endpoints are 3 apart, to within the f32 pixels.
    assert_close(resolved[1].scene_distance.unwrap(), 3.0, 1e-3, "distance");
}

// ─── a similarity transform ───────────────────────────────────────────────

/// A similarity transform keeps every texel, moves every position with it,
/// and scales the distance and every cost by its scale.
#[test]
fn a_similarity_transform_keeps_texels_and_scales_lengths() {
    let recon = scene(&[
        pt(0.0, 0.0, 0.0, 0.2),
        pt(0.25, 0.0, 0.0, 0.2),
        pt(-1.5, 0.3, 0.0, 0.2),
    ]);
    let rows: Vec<_> = (0..recon.point_set.tracks.len())
        .map(|j| reading(0.4 + 0.1 * j as f32, 0.7 + 0.03 * j as f32))
        .collect();
    let mut recon = with_readings(recon, rows);
    let p = pixel_of_texel(&recon, 0, 0, 0.85, 0.1);
    let q = pixel_of_texel(&recon, 2, 1, -0.4, 0.3);
    recon.image_set.measurements.distance = Some(DistanceMeasurementsData {
        distances: array![1.0_f32],
        units: array![LENGTH_UNIT_M],
        image_indexes: array![[0_u32, 1_u32]],
        endpoints_xy: array![[[p[0] as f32, p[1] as f32], [q[0] as f32, q[1] as f32]]],
        labels: vec![String::new()],
    });
    let scale = 2.5;
    let xform = Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 0.5).normalize(), 0.7).unwrap(),
        Vector3::new(3.0, -1.0, 4.0),
        scale,
    );
    let moved = recon.apply_se3_transform(&xform);
    let opts = MeasurementEndpointOptions::default();
    let before = resolve_distance_measurements(&edited(recon), &opts);
    let after = resolve_distance_measurements(&edited(moved), &opts);

    assert_close(
        after[0].scene_distance.unwrap(),
        scale * before[0].scene_distance.unwrap(),
        1e-6,
        "distance",
    );
    let first = before[0].endpoints[0].as_ref().unwrap();
    assert_eq!(first.candidates.len(), 2, "the first endpoint is ranked");
    for k in 0..2 {
        let b = before[0].endpoints[k].as_ref().unwrap();
        let a = after[0].endpoints[k].as_ref().unwrap();
        assert_eq!(a.chosen, b.chosen);
        assert_eq!(a.candidates.len(), b.candidates.len());
        for (ca, cb) in a.candidates.iter().zip(&b.candidates) {
            assert_eq!(ca.point, cb.point);
            for c in 0..2 {
                assert_close(ca.patch_coords[c], cb.patch_coords[c], 1e-5, "texel");
            }
            let expected = xform.apply_to_point(&Point3::from(cb.position.finite().unwrap()));
            let got = ca.position.finite().unwrap();
            for c in 0..3 {
                assert_close(got[c], expected[c], 1e-5, "position");
            }
            if let (CandidateStanding::Ranked(x), CandidateStanding::Ranked(y)) =
                (ca.standing, cb.standing)
            {
                assert_close(x.cost, scale * y.cost, 1e-6 * y.cost.abs(), "cost");
            }
        }
    }
}

// ─── the views of an endpoint ─────────────────────────────────────────────

fn views(
    value: &EditedReconstruction,
    image: u32,
    pixel: [f64; 2],
) -> Result<Vec<EndpointView>, EndpointRefusal> {
    endpoint_views(value, image, pixel, &MeasurementEndpointOptions::default())
}

/// The offers of an endpoint on a known texel land at the pixels where the
/// other observations' squares hold that texel, the square slid onto each
/// keypoint rather than the patch itself; the resolver at an offered pixel
/// finds the same point and the same texel, so a switch keeps both.
#[test]
fn views_land_where_the_other_squares_hold_the_texel() {
    let mut p = pt(0.1, -0.2, 0.0, 0.25);
    p.images = vec![0, 1, 2];
    let mut recon = scene(&[p]);
    // Move image 1's keypoint (observation 1) off the projection, so its
    // square slides off the point.
    let keypoints_xy = recon
        .point_set
        .keypoints_xy
        .as_mut()
        .expect("the scene carries keypoints");
    keypoints_xy[[1, 0]] += 3.0;
    keypoints_xy[[1, 1]] -= 2.0;
    let recon = with_readings(recon, vec![reading(0.6, 0.9); 3]);
    let value = edited(recon.clone());
    let (s, t) = (0.4, -0.3);
    let pixel = pixel_of_texel(&recon, 0, 0, s, t);

    let texel = resolve(&value, 0, pixel).unwrap().chosen().patch_coords;
    let offered = views(&value, 0, pixel).unwrap();
    let images: Vec<u32> = offered.iter().map(|v| v.image).collect();
    assert_eq!(images, [1, 2]);
    for view in &offered {
        assert_eq!(view.point, 0);
        assert_eq!(view.observation, view.image as usize);
        assert_eq!(view.patch_coords, texel);
        let expected = pixel_of_texel(&recon, 0, view.image as usize, s, t);
        assert_close(view.pixel[0], expected[0], 1e-6, "x");
        assert_close(view.pixel[1], expected[1], 1e-6, "y");
        assert_eq!(view.resolves_to, Some(0));
        assert!(!view.resolves_elsewhere());
        // 0.6 grid px of a 24 px grid across the patch's 0.5 width.
        assert_close(view.localization.unwrap(), 0.6 * 0.5 / 24.0, 1e-6, "l");
        let angle = view.viewing_angle_deg.unwrap();
        assert!(angle > 0.0 && angle < 15.0, "{angle}");

        let moved = resolve(&value, view.image, view.pixel).unwrap();
        assert_eq!(moved.chosen().point, 0);
        let [ms, mt] = moved.chosen().patch_coords;
        assert_close(ms, s, 1e-6, "s after the switch");
        assert_close(mt, t, 1e-6, "t after the switch");
    }
    // The slid square puts the texel elsewhere than the patch itself would.
    let unslid = {
        let patch = value.point(0).unwrap().patch_frame().unwrap();
        let (xyz, w) = patch.corner_homogeneous(s, t);
        project(&recon, 1, xyz, w)
    };
    assert!((offered[0].pixel[0] - unslid[0]).hypot(offered[0].pixel[1] - unslid[1]) > 1.0);
}

/// A view whose texel projects outside its image, and one whose square faces
/// away from its camera, are dropped; the others are offered.
#[test]
fn a_view_off_the_image_or_seen_from_behind_is_dropped() {
    // Far to the side, inside image 0 and outside image 1.
    let mut side = pt(5.0, 0.0, 0.0, 0.2);
    side.images = vec![0, 1, 2];
    let recon = scene(&[side]);
    let at = pixel_of_texel(&recon, 0, 0, 0.0, 0.0);
    let inside = |x: f64| (0.0..f64::from(IMG_W)).contains(&x);
    assert!(inside(at[0]), "the point is inside image 0: {at:?}");
    let off = pixel_of_texel(&recon, 0, 1, 0.0, 0.0);
    assert!(!inside(off[0]), "the point is outside image 1: {off:?}");
    let images: Vec<u32> = views(&edited(recon), 0, at)
        .unwrap()
        .iter()
        .map(|v| v.image)
        .collect();
    assert_eq!(images, [2], "the view in image 1 is off the image");

    // A patch nearly edge on, its normal `(-1, 0, 0.05)`: image 0 and image 2
    // (the same pose) see its front, image 1 its back.
    let h = 0.2;
    let u = Vector3::new(0.05, 0.0, 1.0).normalize() * h;
    let tilted = Pt {
        pos: [0.0, 0.0, 0.0],
        w: 1.0,
        u: [u.x, u.y, u.z],
        v: [0.0, h, 0.0],
        images: vec![0, 1, 2],
    };
    let recon = scene(&[tilted]);
    let at = pixel_of_texel(&recon, 0, 0, 0.0, 0.0);
    let value = edited(recon);
    assert!(resolve(&value, 0, at).is_ok());
    let images: Vec<u32> = views(&value, 0, at)
        .unwrap()
        .iter()
        .map(|v| v.image)
        .collect();
    assert_eq!(images, [2], "the view in image 1 sees the square's back");
}

/// `texel_pixel` gives the pixel where an observation's square, slid onto its
/// keypoint, holds the texel, and the resolver there finds the same point at
/// the same texel.
#[test]
fn texel_pixel_lands_where_the_square_holds_the_texel() {
    let mut p = pt(0.1, -0.2, 0.0, 0.25);
    p.images = vec![0, 1, 2];
    let mut recon = scene(&[p]);
    // Image 1's keypoint off the projection, so its square slides off the
    // point and the unslid patch would put the texel elsewhere.
    let keypoints_xy = recon.point_set.keypoints_xy.as_mut().unwrap();
    keypoints_xy[[1, 0]] += 3.0;
    keypoints_xy[[1, 1]] -= 2.0;
    let value = edited(recon.clone());
    let (s, t) = (0.4, -0.3);
    for k in 0..3 {
        let pixel = texel_pixel(&value, 0, k, [s, t]).expect("the texel is in view");
        let expected = pixel_of_texel(&recon, 0, k, s, t);
        assert_close(pixel[0], expected[0], 1e-9, "x");
        assert_close(pixel[1], expected[1], 1e-9, "y");
        let back = resolve(&value, k as u32, pixel).unwrap();
        assert_eq!(back.chosen().point, 0);
        assert_eq!(back.chosen().observation, k);
        let [bs, bt] = back.chosen().patch_coords;
        assert_close(bs, s, 1e-6, "s");
        assert_close(bt, t, 1e-6, "t");
    }
    // The centre texel is the keypoint itself.
    let centre = texel_pixel(&value, 0, 1, [0.0, 0.0]).unwrap();
    let kp = recon.point_set.keypoints_xy.as_ref().unwrap();
    assert_close(centre[0], f64::from(kp[[1, 0]]), 1e-3, "keypoint x");
    assert_close(centre[1], f64::from(kp[[1, 1]]), 1e-3, "keypoint y");
}

/// `texel_pixel` answers `None` for a texel outside the square or not a
/// number, a point or observation that does not exist, a texel whose place is
/// off the image, and a square seen from behind.
#[test]
fn texel_pixel_refuses_what_no_square_shows() {
    let mut side = pt(5.0, 0.0, 0.0, 0.2);
    side.images = vec![0, 1];
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2), side]);
    let value = edited(recon);
    assert!(texel_pixel(&value, 0, 0, [0.5, -0.5]).is_some());
    for texel in [
        [1.01, 0.0],
        [0.0, -1.5],
        [f64::NAN, 0.0],
        [0.0, f64::INFINITY],
    ] {
        assert_eq!(texel_pixel(&value, 0, 0, texel), None, "{texel:?}");
    }
    assert_eq!(
        texel_pixel(&value, 0, 2, [0.0, 0.0]),
        None,
        "no observation 2"
    );
    assert_eq!(texel_pixel(&value, 9, 0, [0.0, 0.0]), None, "no point 9");
    // Point 1 is inside image 0 and outside image 1.
    assert!(texel_pixel(&value, 1, 0, [0.0, 0.0]).is_some());
    assert_eq!(texel_pixel(&value, 1, 1, [0.0, 0.0]), None, "off image 1");

    // A patch nearly edge on, its normal `(-1, 0, 0.05)`: image 0 sees its
    // front, image 1 its back.
    let h = 0.2;
    let u = Vector3::new(0.05, 0.0, 1.0).normalize() * h;
    let tilted = Pt {
        pos: [0.0, 0.0, 0.0],
        w: 1.0,
        u: [u.x, u.y, u.z],
        v: [0.0, h, 0.0],
        images: vec![0, 1],
    };
    let value = edited(scene(&[tilted]));
    assert!(texel_pixel(&value, 0, 0, [0.0, 0.0]).is_some());
    assert_eq!(
        texel_pixel(&value, 0, 1, [0.0, 0.0]),
        None,
        "seen from behind"
    );
}

/// Where another square holds an offered pixel and wins the ranking there,
/// the view says the endpoint would resolve to that other point.
#[test]
fn a_view_another_square_wins_resolves_elsewhere() {
    let mut a = pt(0.0, 0.0, 0.0, 0.2);
    a.images = vec![0, 1, 2];
    // A smaller patch observed only in image 1, centred where A's texel
    // (0.5, 0) lies.
    let mut b = pt(0.1, 0.0, 0.0, 0.1);
    b.images = vec![1];
    let recon = scene(&[a, b]);
    let value = edited(recon.clone());
    let at = pixel_of_texel(&recon, 0, 0, 0.5, 0.0);
    assert_eq!(resolve(&value, 0, at).unwrap().chosen().point, 0);

    let offered = views(&value, 0, at).unwrap();
    assert_eq!(offered.len(), 2);
    let in_1 = offered.iter().find(|v| v.image == 1).unwrap();
    assert_eq!(in_1.point, 0);
    assert_eq!(in_1.resolves_to, Some(1));
    assert!(in_1.resolves_elsewhere());
    let in_2 = offered.iter().find(|v| v.image == 2).unwrap();
    assert_eq!(in_2.resolves_to, Some(0));
    assert!(!in_2.resolves_elsewhere());
}

/// Every candidate is offered, chosen or not: the chosen point's views come
/// first, each at its own candidate's texel.
#[test]
fn the_chosen_points_views_come_first() {
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2), pt(0.25, 0.0, 0.0, 0.2)]);
    let value = edited(recon.clone());
    let pixel = pixel_of_texel(&recon, 0, 0, 0.85, 0.0);
    let res = resolve(&value, 0, pixel).unwrap();
    assert_eq!(res.chosen().point, 1);

    let offered = views(&value, 0, pixel).unwrap();
    let points: Vec<u32> = offered.iter().map(|v| v.point).collect();
    assert_eq!(points, [1, 0]);
    for view in &offered {
        let own = res
            .candidates
            .iter()
            .find(|c| c.point == view.point)
            .unwrap();
        assert_eq!(view.patch_coords, own.patch_coords);
        assert_eq!(view.image, 1);
    }
}

/// The views share the resolver's refusals, sentence and all.
#[test]
fn the_views_refuse_as_the_resolver_does() {
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2)]);
    let value = edited(recon.clone());
    let centre = project(&recon, 0, Vector3::zeros(), 1.0);
    for (image, pixel) in [(0, [5.0, 5.0]), (2, centre), (3, centre), (0, [-1.0, 3.0])] {
        let refused = views(&value, image, pixel).unwrap_err();
        assert_eq!(Err(refused), resolve(&value, image, pixel));
    }
    let mut behind = pt(0.0, 0.0, 0.0, 0.2);
    std::mem::swap(&mut behind.u, &mut behind.v);
    let value = edited(scene(&[behind]));
    assert!(matches!(
        views(&value, 0, centre),
        Err(EndpointRefusal::AllExcluded { .. })
    ));
}

/// The views of an endpoint on a point at infinity, seen in three images:
/// each view is the pixel where that observation's square holds the texel's
/// direction, the resolver there finds the same point and texel at infinity,
/// and `ℓ` reads in degrees.
#[test]
fn the_views_of_a_point_at_infinity_carry_its_direction() {
    let d = Vector3::new(0.0, 0.0, -1.0);
    let sky = Pt {
        pos: [d.x, d.y, d.z],
        w: 0.0,
        u: [0.02, 0.0, 0.0],
        v: [0.0, -0.02, 0.0],
        images: vec![0, 1, 2],
    };
    let recon = with_readings(scene(&[sky]), vec![reading(0.6, 0.9); 3]);
    let value = edited(recon.clone());
    let at = pixel_of_texel(&recon, 0, 0, 0.3, 0.2);
    let texel = resolve(&value, 0, at).unwrap().chosen().patch_coords;

    let offered = views(&value, 0, at).unwrap();
    let images: Vec<u32> = offered.iter().map(|v| v.image).collect();
    assert_eq!(images, [1, 2]);
    // 0.6 grid px of a 24 px grid across 0.04 of the direction sphere.
    let expected_l = (0.6 * 0.04 / 24.0_f64).to_degrees();
    for view in &offered {
        assert_eq!(view.patch_coords, texel);
        let expected = pixel_of_texel(&recon, 0, view.image as usize, 0.3, 0.2);
        assert_close(view.pixel[0], expected[0], 1e-6, "x");
        assert_close(view.pixel[1], expected[1], 1e-6, "y");
        assert_eq!(view.resolves_to, Some(0));
        let l = view.localization.unwrap();
        assert!(
            (l - expected_l).abs() < 0.01 * expected_l,
            "{l} against {expected_l}"
        );

        let moved = resolve(&value, view.image, view.pixel).unwrap();
        assert_eq!(moved.chosen().point, 0);
        assert_eq!(moved.chosen().depth, f64::INFINITY);
        assert!(matches!(
            moved.position(),
            EndpointPosition::AtInfinity { .. }
        ));
        let [s, t] = moved.chosen().patch_coords;
        assert_close(s, 0.3, 1e-6, "s after the switch");
        assert_close(t, 0.2, 1e-6, "t after the switch");
    }
}

/// The views resolve through the observations gathered for every image at
/// once, and that reading is the resolver's own: over a grid of pixels of
/// each image, a resolution against an image's entry of the gathered lists
/// is the resolver's answer exactly, and each view's `resolves_to` is the
/// resolver's pick at its pixel.
#[test]
fn the_gathered_observations_resolve_as_the_resolver_does() {
    let mut far = pt(0.05, 0.02, -0.3, 0.25);
    far.images = vec![0, 1, 2];
    let mut lone = pt(-1.5, 0.3, 0.0, 0.2);
    lone.images = vec![1];
    let recon = scene(&[pt(0.0, 0.0, 0.0, 0.2), pt(0.25, 0.0, 0.0, 0.2), far, lone]);
    let rows: Vec<_> = (0..recon.point_set.tracks.len())
        .map(|j| reading(0.4 + 0.1 * j as f32, 0.7 + 0.03 * j as f32))
        .collect();
    let recon = with_readings(recon, rows);
    let value = edited(recon.clone());
    let opts = MeasurementEndpointOptions::default();
    let by_image = observations_by_image(&value, None);
    let mut resolved = 0;
    for image in 0..IMAGES as u32 {
        for gx in 0..60 {
            for gy in 0..60 {
                let pixel = [200.5 + f64::from(gx) * 4.0, 120.25 + f64::from(gy) * 4.0];
                let one = resolve_measurement_endpoint(&value, image, pixel, &opts);
                let all = resolve_among(&value, image, pixel, &opts, &by_image[image as usize]);
                assert_eq!(one, all, "image {image} pixel {pixel:?}");
                resolved += usize::from(one.is_ok());
            }
        }
    }
    assert!(resolved > 200, "the grid hit {resolved} squares");

    let at = pixel_of_texel(&recon, 0, 0, 0.85, 0.0);
    let offered = views(&value, 0, at).unwrap();
    assert!(offered.len() >= 3, "{offered:?}");
    for view in &offered {
        let pick = resolve_measurement_endpoint(&value, view.image, view.pixel, &opts)
            .ok()
            .map(|r| r.chosen().point);
        assert_eq!(view.resolves_to, pick);
    }
}
