// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`add_image_to_tracks`] on a synthetic capture: pinhole cameras
//! looking at a textured plane, points on the plane observed at their exact
//! projections, and one image that observes none of them.

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use ndarray::Array2;

use super::*;
use crate::camera::remap::ImageU8;
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::reconstruction::data::{Point3D, SfmrImage};

const IMG: u32 = 128;
const FOCAL: f64 = 160.0;
const PLANE_Z: f64 = 4.0;
const HALF_EXTENT: f64 = 0.12;
/// Camera centres. Images 0-3 observe the points; image 4 is the one added.
const CENTERS: [[f64; 3]; 5] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 0.0],
    [-0.35, 0.4, 0.0],
    [0.05, 0.1, 0.0],
];
const TARGET: usize = 4;

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: f64::from(IMG) / 2.0,
            principal_point_y: f64::from(IMG) / 2.0,
        },
        width: IMG,
        height: IMG,
    }
}

fn texture(x: f64, y: f64) -> f64 {
    127.5 + 55.0 * (x * 17.0).sin() + 45.0 * (y * 23.0).cos() + 25.0 * ((x + y) * 31.0).sin()
}

/// A texture unrelated to [`texture`], for a photograph of something else.
fn other_texture(x: f64, y: f64) -> f64 {
    // Blocks of hashed grey: uncorrelated with anything smooth.
    let (i, j) = ((x * 60.0).floor(), (y * 60.0).floor());
    let h = ((i * 12.9898 + j * 78.233).sin() * 43758.5453)
        .fract()
        .abs();
    40.0 + 175.0 * h
}

/// A camera at `center` looking down world `+z` (a half-turn about `x`).
fn down_z(center: [f64; 3]) -> (UnitQuaternion<f64>, Vector3<f64>) {
    (
        UnitQuaternion::from_quaternion(Quaternion::new(0.0, 1.0, 0.0, 0.0)),
        Vector3::new(-center[0], center[1], center[2]),
    )
}

/// What `camera` at `(q, t)` sees of the plane `z = PLANE_Z` painted with `tex`.
fn render(
    camera: &CameraIntrinsics,
    q: &UnitQuaternion<f64>,
    t: &Vector3<f64>,
    tex: fn(f64, f64) -> f64,
) -> ImageU8 {
    let pose = RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z]);
    let c = pose.inverse_translation_origin();
    let rinv = pose.rotation.inverse();
    let mut data = Vec::with_capacity((IMG * IMG) as usize);
    for row in 0..IMG {
        for col in 0..IMG {
            let ray = camera.pixel_to_ray(f64::from(col) + 0.5, f64::from(row) + 0.5);
            let d = rinv.rotate_vector(&Vector3::new(ray[0], ray[1], ray[2]));
            let lambda = (PLANE_Z - c.z) / d.z;
            let x = c.x + lambda * d.x;
            let y = c.y + lambda * d.y;
            data.push(tex(x, y).clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8::new(IMG, IMG, 1, data)
}

struct Capture {
    recon: SfmrReconstruction,
    pyramids: Vec<ImageU8Pyramid>,
}

impl Capture {
    fn pyramids(&self) -> Vec<Option<&ImageU8Pyramid>> {
        self.pyramids.iter().map(Some).collect()
    }

    /// Where `world` lands in image `i`.
    fn project(&self, i: usize, world: Point3<f64>) -> [f64; 2] {
        let im = &self.recon.image_table.images[i];
        let q = im.quaternion_wxyz;
        let t = im.translation_xyz;
        let pose = RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z]);
        let cam = pose.transform_point(&world);
        let (u, v) = pinhole().ray_to_pixel([cam.x, cam.y, cam.z]).unwrap();
        [u, v]
    }

    /// Replace image `i`'s pose and photograph.
    fn repose(
        &mut self,
        i: usize,
        q: UnitQuaternion<f64>,
        t: Vector3<f64>,
        tex: fn(f64, f64) -> f64,
    ) {
        self.recon.image_table.images[i].quaternion_wxyz = q;
        self.recon.image_table.images[i].translation_xyz = t;
        self.pyramids[i] = ImageU8Pyramid::build(&render(&pinhole(), &q, &t, tex), 4);
    }
}

/// Points on the plane at `points`, each observed by the images listed beside
/// it at its exact projections, over the five cameras of [`CENTERS`].
fn capture(points: &[(Point3<f64>, &[u32])]) -> Capture {
    let mut recon = SfmrReconstruction::demo(1);
    let n = CENTERS.len();
    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = CENTERS
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let (q, t) = down_z(c);
            SfmrImage {
                name: format!("image_{i}.jpg"),
                camera_index: 0,
                quaternion_wxyz: q,
                translation_xyz: t,
            }
        })
        .collect();
    let stats_row = recon.image_table.depth_statistics.images[0].clone();
    recon
        .image_table
        .depth_statistics
        .images
        .resize(n, stats_row);
    let histogram_row = recon.image_table.depth_histogram_counts[0].clone();
    recon
        .image_table
        .depth_histogram_counts
        .resize(n, histogram_row);

    let pyramids: Vec<ImageU8Pyramid> = CENTERS
        .iter()
        .map(|&c| {
            let (q, t) = down_z(c);
            ImageU8Pyramid::build(&render(&pinhole(), &q, &t, texture), 4)
        })
        .collect();
    let mut cap = Capture { recon, pyramids };

    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut kps = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();
    for (p, (world, observing)) in points.iter().enumerate() {
        counts.push(observing.len() as u32);
        for &image in observing.iter() {
            tracks.push(TrackObservation {
                image_index: image,
                point_index: p as u32,
            });
            let xy = cap.project(image as usize, *world);
            kps.push(xy[0] as f32);
            kps.push(xy[1] as f32);
        }
        us.extend_from_slice(&[HALF_EXTENT as f32, 0.0, 0.0]);
        vs.extend_from_slice(&[0.0, HALF_EXTENT as f32, 0.0]);
    }
    let set = &mut cap.recon.point_set;
    set.points = points
        .iter()
        .map(|(world, _)| Point3D {
            position: *world,
            w: 1.0,
            color: [120, 130, 140],
            error: 0.5,
            normal: Vector3::new(0.0, 0.0, -1.0),
        })
        .collect();
    let m = tracks.len();
    set.tracks = tracks;
    set.observation_counts = counts;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::from_shape_vec((m, 2), kps).unwrap(),
        image_file_hashes: vec![[0u8; 16]; n],
    };
    set.patch_u_halfvec_xyz = Some(Array2::from_shape_vec((points.len(), 3), us).unwrap());
    set.patch_v_halfvec_xyz = Some(Array2::from_shape_vec((points.len(), 3), vs).unwrap());
    cap.recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    cap.recon.rebuild_derived_fields();
    cap
}

const FOUR: &[u32] = &[0, 1, 2, 3];

fn grid_points() -> Vec<Point3<f64>> {
    let mut out = Vec::new();
    for iy in -1..=1 {
        for ix in -1..=1 {
            out.push(Point3::new(
                0.25 * f64::from(ix),
                0.25 * f64::from(iy),
                PLANE_Z,
            ));
        }
    }
    out
}

/// The fixed-bar rule: on a noiseless capture every reference agrees with
/// every other to the fifth decimal, so a bar drawn from them sits where
/// rounding decides, and the tests of everything but the rule use a fixed one.
fn fixed() -> AddImageToTracksOptions {
    AddImageToTracksOptions {
        rule: AcceptRule::FixedZncc,
        min_zncc: 0.9,
        ..AddImageToTracksOptions::default()
    }
}

fn run(
    cap: &Capture,
    options: &AddImageToTracksOptions,
) -> (SfmrReconstruction, AddImageToTracksReport) {
    add_image_to_tracks(
        &cap.recon,
        TARGET,
        &cap.pyramids(),
        options,
        &Progress::none(),
    )
    .unwrap()
}

#[test]
fn a_removed_observation_is_found_again_where_it_was() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let cap = capture(&points);
    let (next, report) = run(&cap, &fixed());

    assert_eq!(report.candidates.len(), points.len());
    assert_eq!(
        report.accepted,
        points.len(),
        "{:?}",
        report.refusal_counts()
    );
    assert_eq!(
        report.observations_after,
        report.observations_before + points.len()
    );
    let kps = next.keypoints_xy().unwrap();
    for (p, (world, _)) in points.iter().enumerate() {
        let obs = next.observations_for_point(p);
        let images: Vec<u32> = obs.iter().map(|o| o.image_index).collect();
        assert_eq!(images, vec![0, 1, 2, 3, 4], "tracks stay in image order");
        let row = next.point_set.observation_offsets[p] + 4;
        let expected = cap.project(TARGET, *world);
        let err =
            (f64::from(kps[[row, 0]]) - expected[0]).hypot(f64::from(kps[[row, 1]]) - expected[1]);
        assert!(
            err < 0.1,
            "point {p}: keypoint {err} px from its projection"
        );
        let c = &report.candidates[p];
        assert!(c.zncc > 0.95, "point {p}: zncc {}", c.zncc);
        assert_eq!(c.references, vec![0, 1, 2, 3]);
    }
}

#[test]
fn the_existing_observations_and_points_come_back_unchanged() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let cap = capture(&points);
    let (next, _) = run(&cap, &fixed());
    let before = cap.recon.keypoints_xy().unwrap();
    let after = next.keypoints_xy().unwrap();
    for p in 0..points.len() {
        assert_eq!(next.point_set.points[p], cap.recon.point_set.points[p]);
        for k in 0..4 {
            let old = cap.recon.point_set.observation_offsets[p] + k;
            let new = next.point_set.observation_offsets[p] + k;
            assert_eq!(before.row(old), after.row(new));
        }
    }
    assert_eq!(
        next.point_set.patch_u_halfvec_xyz,
        cap.recon.point_set.patch_u_halfvec_xyz
    );
    assert_eq!(
        next.metadata.observation_count as usize,
        next.point_set.tracks.len()
    );
}

#[test]
fn a_two_observation_track_is_judged_by_the_pair_rule() {
    let cap = capture(&[(Point3::new(0.1, -0.05, PLANE_Z), &[1, 3])]);
    let (_, report) = run(
        &cap,
        &AddImageToTracksOptions {
            rule: AcceptRule::TrackBasis {
                statistic: BasisStatistic::Min,
                pair: PairRule {
                    statistic: PairStatistic::Mean,
                    factor: 0.95,
                },
            },
            ..AddImageToTracksOptions::default()
        },
    );
    let c = &report.candidates[0];
    assert_eq!(c.references.len(), 2);
    assert_eq!(c.pair_zncc.len(), 2);
    assert!(c.bar.is_finite() && c.judged.is_finite());
    assert_eq!(c.refusal, None, "judged {} against bar {}", c.judged, c.bar);
}

#[test]
fn a_photograph_of_something_else_is_refused_photometrically() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    let (q, t) = down_z(CENTERS[TARGET]);
    cap.repose(TARGET, q, t, other_texture);
    for rule in [
        AcceptRule::FixedZncc,
        AddImageToTracksOptions::default().rule,
        AcceptRule::PooledBasis {
            statistic: BasisStatistic::Min,
        },
    ] {
        let options = AddImageToTracksOptions {
            rule,
            min_zncc: 0.7,
            ..fixed()
        };
        let (next, report) = run(&cap, &options);
        assert_eq!(
            report.accepted,
            0,
            "{rule:?}: {:?}",
            report.refusal_counts()
        );
        assert_eq!(
            next.point_set.tracks.len(),
            cap.recon.point_set.tracks.len()
        );
    }
}

#[test]
fn a_point_outside_the_frame_is_not_in_frame() {
    let cap = capture(&[(Point3::new(3.0, 0.0, PLANE_Z), &[0, 1])]);
    let (_, report) = run(&cap, &fixed());
    assert_eq!(report.candidates[0].refusal, Some(Refusal::NotInFrame));
}

#[test]
fn a_camera_behind_the_plane_is_back_facing() {
    let points: Vec<(Point3<f64>, &[u32])> = vec![(Point3::new(0.0, 0.0, PLANE_Z), FOUR)];
    let mut cap = capture(&points);
    // Looking down world -z from beyond the plane: the identity rotation puts
    // the camera's forward axis along -z.
    let q = UnitQuaternion::identity();
    let t = -Vector3::new(0.05, 0.1, 8.0);
    cap.repose(TARGET, q, t, texture);
    let (_, report) = run(&cap, &fixed());
    assert_eq!(report.candidates[0].refusal, Some(Refusal::BackFacing));
    let options = AddImageToTracksOptions {
        require_facing: false,
        ..fixed()
    };
    let (_, report) = run(&cap, &options);
    assert_ne!(report.candidates[0].refusal, Some(Refusal::BackFacing));
}

#[test]
fn two_points_at_one_place_keep_one_observation() {
    let world = Point3::new(0.0, 0.0, PLANE_Z);
    let cap = capture(&[(world, FOUR), (world, FOUR)]);
    let (_, report) = run(&cap, &fixed());
    assert_eq!(report.accepted, 1);
    let shared = report
        .candidates
        .iter()
        .filter(|c| c.refusal == Some(Refusal::SharedKeypoint))
        .count();
    assert_eq!(shared, 1);
}

#[test]
fn the_confidence_column_is_extended_in_lockstep() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    cap.recon.point_set.observation_confidence = Some(vec![200; cap.recon.point_set.tracks.len()]);
    let (next, report) = run(&cap, &fixed());
    let conf = next.point_set.observation_confidence.as_ref().unwrap();
    assert_eq!(conf.len(), next.point_set.tracks.len());
    for (p, c) in report.candidates.iter().enumerate() {
        let start = next.point_set.observation_offsets[p];
        assert_eq!(&conf[start..start + 4], &[200; 4]);
        assert_eq!(conf[start + 4], confidence_byte(c.zncc));
        assert!(conf[start + 4] >= 1);
    }
}

#[test]
fn a_position_gate_refuses_what_lands_too_far() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let cap = capture(&points);
    let options = AddImageToTracksOptions {
        position_gate: PositionGate::MaxPx(-1.0),
        ..fixed()
    };
    let (_, report) = run(&cap, &options);
    assert_eq!(report.accepted, 0);
    assert!(report
        .candidates
        .iter()
        .all(|c| c.refusal == Some(Refusal::TooFar)));
}

#[test]
fn the_calls_preconditions_are_refused_by_name() {
    let cap = capture(&[(Point3::new(0.0, 0.0, PLANE_Z), FOUR)]);
    let options = fixed();
    let pyr = cap.pyramids();
    assert!(matches!(
        add_image_to_tracks(&cap.recon, 9, &pyr, &options, &Progress::none()),
        Err(AddImageToTracksError::NoSuchImage { .. })
    ));
    let mut missing = pyr.clone();
    missing[TARGET] = None;
    assert!(matches!(
        add_image_to_tracks(&cap.recon, TARGET, &missing, &options, &Progress::none()),
        Err(AddImageToTracksError::NoTargetImage(TARGET))
    ));
    let mut unposed = cap.recon.clone();
    unposed.image_table.images[TARGET].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    assert!(matches!(
        add_image_to_tracks(&unposed, TARGET, &pyr, &options, &Progress::none()),
        Err(AddImageToTracksError::Unposed(TARGET))
    ));
}

#[test]
fn a_missing_reference_image_leaves_that_reference_out() {
    let points: Vec<(Point3<f64>, &[u32])> = vec![(Point3::new(0.0, 0.0, PLANE_Z), FOUR)];
    let cap = capture(&points);
    let mut pyr = cap.pyramids();
    pyr[0] = None;
    pyr[1] = None;
    let (_, report) =
        add_image_to_tracks(&cap.recon, TARGET, &pyr, &fixed(), &Progress::none()).unwrap();
    assert_eq!(report.candidates[0].references, vec![2, 3]);
    pyr[2] = None;
    let (_, report) =
        add_image_to_tracks(&cap.recon, TARGET, &pyr, &fixed(), &Progress::none()).unwrap();
    assert_eq!(
        report.candidates[0].refusal,
        Some(Refusal::TooFewReferences)
    );
}

#[test]
fn basis_statistics_read_as_documented() {
    let v = [0.9, 0.8, 0.7, 0.95, 0.85];
    assert_eq!(BasisStatistic::Min.bar(&v), 0.7);
    assert!((BasisStatistic::FractionOfMedian { fraction: 0.5 }.bar(&v) - 0.425).abs() < 1e-12);
    let mad = BasisStatistic::MedianMinusMad { k: 1.0 }.bar(&v);
    assert!((mad - (0.85 - MAD_TO_SIGMA * 0.05)).abs() < 1e-12);
    assert!(BasisStatistic::Min.bar(&[]).is_nan());
}

#[test]
fn a_stored_bitmap_template_finds_the_same_keypoint() {
    use crate::patch::keypoint_subpixel::fuse_patch_bitmap;
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    let r = 24usize;
    let poses: Vec<RigidTransform> = cap
        .recon
        .image_table
        .images
        .iter()
        .map(|im| {
            let (q, t) = (im.quaternion_wxyz, im.translation_xyz);
            RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
        })
        .collect();
    let camera = pinhole();
    let views: Vec<ProjectedImage<'_>> = (0..CENTERS.len())
        .map(|i| ProjectedImage {
            camera: &camera,
            cam_from_world: &poses[i],
            pyramid: &cap.pyramids[i],
        })
        .collect();
    let mut column = ndarray::Array4::<u8>::zeros((points.len(), r, r, 4));
    for (p, (world, observing)) in points.iter().enumerate() {
        let mut patch = OrientedPatch::new(
            *world,
            Vector3::x(),
            Vector3::y(),
            [HALF_EXTENT, HALF_EXTENT],
        );
        patch.w = 1.0;
        let kps: Vec<[f64; 2]> = observing
            .iter()
            .map(|&i| cap.project(i as usize, *world))
            .collect();
        let params = KeypointSubpixelParams {
            resolution: r as u32,
            ..KeypointSubpixelParams::default()
        };
        let bitmap = fuse_patch_bitmap(&patch, &views, observing, &kps, &params).unwrap();
        column
            .index_axis_mut(ndarray::Axis(0), p)
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(&bitmap);
    }
    drop(views);
    cap.recon.point_set.patch_bitmaps_y_x_rgba = Some(std::sync::Arc::new(column));
    let options = AddImageToTracksOptions {
        template: TemplateSource::StoredBitmap,
        subpixel: false,
        ..fixed()
    };
    let (_, report) = run(&cap, &options);
    assert_eq!(
        report.accepted,
        points.len(),
        "{:?}",
        report.refusal_counts()
    );
    for (p, (world, _)) in points.iter().enumerate() {
        let c = &report.candidates[p];
        let expected = cap.project(TARGET, *world);
        let kp = c.keypoint.unwrap();
        let err = (kp[0] - expected[0]).hypot(kp[1] - expected[1]);
        assert!(err < 0.25, "point {p}: {err} px");
        assert!(c.zncc > 0.95, "point {p}: zncc {}", c.zncc);
    }
}

#[test]
fn pooled_or_track_accepts_what_either_bar_accepts() {
    let options = AddImageToTracksOptions::default();
    let mut c = CandidateReport::new(0);
    c.references = vec![0, 1, 2];
    c.reference_loo_zncc = vec![0.6, 0.62, 0.64];
    // Below the pooled bar, above 0.9 of its own track's median.
    c.zncc = 0.58;
    judge(&mut c, &options, Some(0.7));
    assert_eq!(c.refusal, None);
    assert!((c.bar - 0.9 * 0.62).abs() < 1e-12);
    // Below both.
    let mut c2 = c.clone();
    c2.refusal = None;
    c2.zncc = 0.5;
    c2.reference_loo_zncc = vec![0.8, 0.85, 0.9];
    judge(&mut c2, &options, Some(0.7));
    assert_eq!(c2.refusal, Some(Refusal::BelowBar));
    // Above the pooled bar, whatever the track says.
    let mut c3 = c2.clone();
    c3.refusal = None;
    c3.zncc = 0.72;
    judge(&mut c3, &options, Some(0.7));
    assert_eq!(c3.refusal, None);
    assert_eq!(c3.bar, 0.7);
}
