// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`add_image_to_tracks`] on a synthetic capture: pinhole cameras
//! looking at a textured plane, points on the plane observed at their exact
//! projections, and one image that observes none of them.

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use ndarray::Array2;

use super::*;
use crate::camera::image::ImageU8;
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
        // `u × v` is `-z`: the patch faces the cameras, as its normal says,
        // so the reference-view rule has candidates.
        vs.extend_from_slice(&[0.0, -HALF_EXTENT as f32, 0.0]);
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
/// The default options with the member self-similarity gate off. The plane's
/// texture renders smooth on the patch grid at this capture's scale, so about
/// half its cores read a radius over the default bar; these tests are about the
/// search and the rules, and
/// `the_self_similarity_gate_refuses_what_is_over_its_bar` covers the gate.
fn gate_off() -> AddImageToTracksOptions {
    let mut options = AddImageToTracksOptions::default();
    options.localize.max_member_zncc_self_similarity_radius = 0.0;
    options
}

fn fixed() -> AddImageToTracksOptions {
    AddImageToTracksOptions {
        rule: AcceptRule::FixedZncc,
        min_zncc: 0.9,
        ..gate_off()
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
fn the_default_self_similarity_gate_refuses_exactly_the_cores_over_its_bar() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let cap = capture(&points);
    let (_, open) = run(&cap, &fixed());
    let gated_options = AddImageToTracksOptions {
        rule: AcceptRule::FixedZncc,
        min_zncc: 0.9,
        ..AddImageToTracksOptions::default()
    };
    let bar = gated_options
        .localize
        .max_member_zncc_self_similarity_radius;
    assert_eq!(bar, 2.5);
    let (_, gated) = run(&cap, &gated_options);
    let mut refused = 0;
    for (o, g) in open.candidates.iter().zip(&gated.candidates) {
        assert!(o.zncc_self_similarity_radius.is_finite());
        assert_eq!(o.zncc_self_similarity_radius, g.zncc_self_similarity_radius);
        if o.zncc_self_similarity_radius > bar {
            assert_eq!(g.refusal, Some(Refusal::Unlocalizable));
            refused += 1;
        } else {
            assert_eq!(g.refusal, o.refusal);
        }
    }
    assert!(refused > 0, "the scene has cores over the bar");
    assert!(refused < points.len(), "and cores under it");
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
            ..gate_off()
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
fn a_score_is_stored_on_the_confidence_byte_scale() {
    // A measured score is never the `0` that means unmeasured; a score that
    // is not a number is unmeasured.
    assert_eq!(observation_confidence_byte(1.0), 255);
    assert_eq!(observation_confidence_byte(0.5), 128);
    assert_eq!(observation_confidence_byte(0.0), 1);
    assert_eq!(observation_confidence_byte(-0.4), 1);
    assert_eq!(observation_confidence_byte(1.7), 255);
    assert_eq!(observation_confidence_byte(f64::NAN), 0);
    assert_eq!(observation_confidence_byte(f64::INFINITY), 0);
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
        assert_eq!(conf[start + 4], observation_confidence_byte(c.zncc));
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

/// The poses of the capture's images, for building [`ProjectedImage`]s.
fn poses_of(cap: &Capture) -> Vec<RigidTransform> {
    cap.recon
        .image_table
        .images
        .iter()
        .map(|im| {
            let (q, t) = (im.quaternion_wxyz, im.translation_xyz);
            RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
        })
        .collect()
}

/// Point `world`'s patch as [`capture`] frames it.
fn patch_at(world: Point3<f64>) -> OrientedPatch {
    let mut patch = OrientedPatch::new(world, Vector3::x(), -Vector3::y(), [HALF_EXTENT; 2]);
    patch.w = 1.0;
    patch
}

/// A stored bitmap column for `points`, each the fused mean of its observing
/// views at the projections of its position moved by `shift`, on a 24 px grid.
fn fused_bitmaps(
    cap: &Capture,
    points: &[(Point3<f64>, &[u32])],
    shift: Vector3<f64>,
) -> ndarray::Array4<u8> {
    use crate::patch::keypoint_subpixel::fuse_patch_bitmap;
    let r = 24usize;
    let poses = poses_of(cap);
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
        let kps: Vec<[f64; 2]> = observing
            .iter()
            .map(|&i| cap.project(i as usize, world + shift))
            .collect();
        let params = KeypointSubpixelParams {
            resolution: r as u32,
            ..KeypointSubpixelParams::default()
        };
        let bitmap =
            fuse_patch_bitmap(&patch_at(*world), &views, observing, &kps, &params).unwrap();
        column
            .index_axis_mut(ndarray::Axis(0), p)
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(&bitmap);
    }
    column
}

#[test]
fn a_stored_bitmap_template_finds_the_same_keypoint() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    let column = fused_bitmaps(&cap, &points, Vector3::zeros());
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
        assert_eq!(c.template, Some(TemplateKind::StoredBitmap));
        let expected = cap.project(TARGET, *world);
        let kp = c.keypoint.unwrap();
        let err = (kp[0] - expected[0]).hypot(kp[1] - expected[1]);
        assert!(err < 0.25, "point {p}: {err} px");
        assert!(c.zncc > 0.95, "point {p}: zncc {}", c.zncc);
    }
}

/// The template is the stored bitmap where the value stores one: a bitmap
/// rendered a little way along the plane from the point puts the new keypoint
/// at that place, where a template rendered from the observations would put it
/// at the point's projection.
#[test]
fn the_template_is_the_stored_bitmap_where_there_is_one() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    // Four grid px along the patch's u axis.
    let shift = Vector3::new(0.04, 0.0, 0.0);
    let column = fused_bitmaps(&cap, &points, shift);
    cap.recon.point_set.patch_bitmaps_y_x_rgba = Some(std::sync::Arc::new(column));
    for (template, moved) in [
        (TemplateSource::StoredBitmap, shift),
        (TemplateSource::Rendered, Vector3::zeros()),
    ] {
        let options = AddImageToTracksOptions {
            template,
            position_gate: PositionGate::Off,
            ..fixed()
        };
        let (_, report) = run(&cap, &options);
        assert_eq!(
            report.accepted,
            points.len(),
            "{template:?}: {:?}",
            report.refusal_counts()
        );
        for (p, (world, _)) in points.iter().enumerate() {
            let c = &report.candidates[p];
            let expected_kind = match template {
                TemplateSource::StoredBitmap => TemplateKind::StoredBitmap,
                TemplateSource::Rendered => TemplateKind::ReferenceObservation,
            };
            assert_eq!(c.template, Some(expected_kind));
            let expected = cap.project(TARGET, world + moved);
            let kp = c.keypoint.unwrap();
            let err = (kp[0] - expected[0]).hypot(kp[1] - expected[1]);
            assert!(err < 0.25, "{template:?} point {p}: {err} px");
        }
    }
}

/// Without bitmaps, the template is the point's stored reference observation
/// rendered at its keypoint: moving that keypoint moves where the new view is
/// placed by as much, the reference reads `1.0`, and neither its keypoint nor
/// the point's reference index is changed by the new observation.
#[test]
fn without_bitmaps_the_stored_reference_observation_is_the_template() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    // Image 1's observation (index 1 in each track) is the reference, its
    // keypoint one source px to the right of the projection. The cameras are
    // parallel at one depth, so one px there is one px in the target.
    let n = points.len();
    cap.recon.point_set.reference_observations = Some(vec![1; n]);
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
        &mut cap.recon.point_set.observations
    else {
        unreachable!()
    };
    for p in 0..n {
        let row = cap.recon.point_set.observation_offsets[p] + 1;
        keypoints_xy[[row, 0]] += 1.0;
    }
    let before = cap.recon.keypoints_xy().unwrap().clone();
    let options = AddImageToTracksOptions {
        position_gate: PositionGate::Off,
        ..fixed()
    };
    let (next, report) = run(&cap, &options);
    assert_eq!(report.accepted, n, "{:?}", report.refusal_counts());
    let after = next.keypoints_xy().unwrap();
    for (p, (world, _)) in points.iter().enumerate() {
        let c = &report.candidates[p];
        assert_eq!(c.template, Some(TemplateKind::ReferenceObservation));
        assert_eq!(c.reference_observation, Some(1));
        assert_eq!(c.references[1], 1);
        assert_eq!(c.reference_zncc[1], 1.0);
        let proj = cap.project(TARGET, *world);
        let kp = c.keypoint.unwrap();
        let err = (kp[0] - (proj[0] + 1.0)).hypot(kp[1] - proj[1]);
        assert!(err < 0.25, "point {p}: {err} px from the moved place");
        // The reference's keypoint, and every other, is as it was.
        for k in 0..4 {
            let old = cap.recon.point_set.observation_offsets[p] + k;
            let new = next.point_set.observation_offsets[p] + k;
            assert_eq!(before.row(old), after.row(new));
        }
    }
    // Image 4 lands after image 1 in every track, so the index stands.
    assert_eq!(next.point_set.reference_observations, Some(vec![1; n]));
}

/// Without a stored reference observation (no column, or `-1`), the
/// reference-view rule picks one from the existing observations' renders at
/// their keypoints, and its render is the template.
#[test]
fn without_a_stored_reference_the_rule_picks_one() {
    use crate::patch::stored_bitmap::{render_reference, stored_view};
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    let options = fixed();
    let poses = poses_of(&cap);
    let camera = pinhole();
    let picks: Vec<Option<usize>> = {
        let views: Vec<ProjectedImage<'_>> = (0..CENTERS.len())
            .map(|i| ProjectedImage {
                camera: &camera,
                cam_from_world: &poses[i],
                pyramid: &cap.pyramids[i],
            })
            .collect();
        points
            .iter()
            .map(|(world, observing)| {
                let kps: Vec<Option<[f64; 2]>> = observing
                    .iter()
                    .map(|&i| Some(cap.project(i as usize, *world)))
                    .collect();
                let render = render_reference(
                    &patch_at(*world),
                    &views,
                    observing,
                    &kps,
                    options.localize.resolution,
                    options.localize.sampler,
                    &Progress::none(),
                );
                stored_view(&render.reading.choice)
            })
            .collect()
    };
    assert!(
        picks.iter().all(Option::is_some),
        "the rule picks a view: {picks:?}"
    );
    for column in [None, Some(vec![-1; points.len()])] {
        cap.recon.point_set.reference_observations = column.clone();
        let (next, report) = run(&cap, &options);
        assert_eq!(report.accepted, points.len());
        for (p, c) in report.candidates.iter().enumerate() {
            assert_eq!(c.template, Some(TemplateKind::ReferenceObservation));
            assert_eq!(c.reference_observation, picks[p], "point {p}");
            assert_eq!(c.reference_zncc[picks[p].unwrap()], 1.0);
        }
        // The pick is not written as the point's reference observation.
        assert_eq!(next.point_set.reference_observations, column);
    }
}

/// Where the rule picks no view it would store (here every camera sees the
/// patch from behind its frame, so no view is a candidate), the template is the
/// fused mean of the references, as the stored bitmap would be, and no
/// reference is left out of the bars.
#[test]
fn where_the_rule_picks_none_the_template_is_the_fused_mean() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    let n = points.len();
    let v = Array2::from_shape_fn(
        (n, 3),
        |(_, k)| if k == 1 { HALF_EXTENT as f32 } else { 0.0 },
    );
    cap.recon.point_set.patch_v_halfvec_xyz = Some(v);
    let (_, report) = run(&cap, &fixed());
    assert_eq!(report.accepted, n, "{:?}", report.refusal_counts());
    for (p, (world, _)) in points.iter().enumerate() {
        let c = &report.candidates[p];
        assert_eq!(c.template, Some(TemplateKind::FusedMean));
        assert_eq!(c.reference_observation, None);
        assert_eq!(c.bar_zncc().count(), 4);
        let expected = cap.project(TARGET, *world);
        let kp = c.keypoint.unwrap();
        let err = (kp[0] - expected[0]).hypot(kp[1] - expected[1]);
        assert!(err < 0.1, "point {p}: {err} px");
    }
}

/// A texture fine enough that the references' renders carry detail on the
/// patch grid, for a stored bitmap sharper than an out-of-focus view.
fn fine_texture(x: f64, y: f64) -> f64 {
    127.5 + 50.0 * (x * 90.0).sin() + 40.0 * (y * 75.0).cos() + 30.0 * ((x - y) * 60.0).sin()
}

/// [`fine_texture`] as an out-of-focus photograph shows it: every sinusoid
/// attenuated, the finest most.
fn blurred_fine_texture(x: f64, y: f64) -> f64 {
    127.5
        + 50.0 * 0.15 * (x * 90.0).sin()
        + 40.0 * 0.25 * (y * 75.0).cos()
        + 30.0 * 0.4 * ((x - y) * 60.0).sin()
}

/// The new view's score, the one the bars judge and the confidence column
/// stores, is its blur-matched score against the template, read as the bench
/// reads a row against the stored bitmap, both where the template is the
/// reference observation's render and where it is the stored bitmap. The
/// references see a fine texture and the new view an out-of-focus copy of it,
/// so the template is blurred for some candidates and their score is not the
/// plain one.
#[test]
fn the_new_views_score_is_blur_matched_against_the_template() {
    use crate::patch::reference_view::render_view_tile;
    use crate::patch::stored_bitmap::{bitmap_from_tile, bitmap_planes, BitmapScorer};

    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    let n = points.len();
    for (i, &center) in CENTERS.iter().enumerate() {
        let (q, t) = down_z(center);
        let tex = if i == TARGET {
            blurred_fine_texture
        } else {
            fine_texture
        };
        cap.repose(i, q, t, tex);
    }
    cap.recon.point_set.reference_observations = Some(vec![2; n]);
    cap.recon.point_set.observation_confidence = Some(vec![200; cap.recon.point_set.tracks.len()]);
    let column = fused_bitmaps(&cap, &points, Vector3::zeros());
    cap.recon.point_set.patch_bitmaps_y_x_rgba = Some(std::sync::Arc::new(column.clone()));
    let poses = poses_of(&cap);
    let camera = pinhole();
    let view = |i: usize| ProjectedImage {
        camera: &camera,
        cam_from_world: &poses[i],
        pyramid: &cap.pyramids[i],
    };
    for template in [TemplateSource::Rendered, TemplateSource::StoredBitmap] {
        let options = AddImageToTracksOptions {
            rule: AcceptRule::FixedZncc,
            min_zncc: -2.0,
            position_gate: PositionGate::Off,
            template,
            ..gate_off()
        };
        let (next, report) = run(&cap, &options);
        let r = options.localize.resolution as usize;
        let sampler = options.localize.sampler;
        let (mut scored, mut blurred) = (0, 0);
        let conf = next.point_set.observation_confidence.as_ref().unwrap();
        for (p, (world, _)) in points.iter().enumerate() {
            let c = &report.candidates[p];
            let Some(kp) = c.keypoint else { continue };
            if c.refusal.is_some() {
                continue;
            }
            let patch = patch_at(*world);
            let bitmap = match template {
                TemplateSource::Rendered => bitmap_from_tile(&render_view_tile(
                    &patch,
                    &view(2),
                    Some(cap.project(2, *world)),
                    r,
                    sampler,
                    &Progress::none(),
                )),
                TemplateSource::StoredBitmap => {
                    assert_eq!(c.template, Some(TemplateKind::StoredBitmap));
                    column
                        .index_axis(ndarray::Axis(0), p)
                        .as_slice()
                        .unwrap()
                        .to_vec()
                }
            };
            let planes = bitmap_planes(&bitmap, r);
            let mut scorer = BitmapScorer::new(&planes, options.localize.window);
            let tile = render_view_tile(
                &patch,
                &view(TARGET),
                Some(kp),
                r,
                sampler,
                &Progress::none(),
            );
            let score = scorer.score(&tile.planes(), None);
            assert_eq!(c.zncc, score.blur_matched_zncc, "{template:?} point {p}");
            scored += 1;
            if score.blur_sigma > 0.0 && score.blur_matched_zncc != score.plain_zncc {
                blurred += 1;
            }
            let start = next.point_set.observation_offsets[p];
            assert_eq!(
                conf[start + 4],
                observation_confidence_byte(c.zncc),
                "{template:?} point {p}"
            );
        }
        assert!(
            scored > n / 2,
            "{template:?}: {scored} of {n} candidates scored"
        );
        assert!(
            blurred > 0,
            "{template:?}: the template was blurred for no candidate"
        );
    }
}

/// A candidate refused before its new view was scored carries no reference
/// scores, so the report holds blur-matched scores only; a scored one carries
/// one per reference.
#[test]
fn a_candidate_refused_before_scoring_carries_no_reference_scores() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let cap = capture(&points);
    // The default member gate refuses about half the cores as unlocalizable.
    let options = AddImageToTracksOptions {
        rule: AcceptRule::FixedZncc,
        min_zncc: -2.0,
        ..AddImageToTracksOptions::default()
    };
    let (_, report) = run(&cap, &options);
    let mut refused = 0;
    for c in &report.candidates {
        match c.refusal {
            Some(Refusal::Unlocalizable | Refusal::NoPeak | Refusal::PeakAtEdge) => {
                refused += 1;
                assert!(c.reference_zncc.is_empty(), "{:?}", c.refusal);
            }
            None => assert_eq!(c.reference_zncc.len(), c.references.len()),
            _ => {}
        }
    }
    assert!(refused > 0, "{:?}", report.refusal_counts());
}

/// A new view whose tile could not be read against the template (a `NaN`
/// score) is refused as unscorable, not left for the rule to refuse as below
/// its floor or bar.
#[test]
fn a_nan_score_is_unscorable() {
    let mut c = CandidateReport::new(0);
    settle_score(&mut c, vec![1.0, 0.8], f64::NAN);
    assert_eq!(c.refusal, Some(Refusal::Unscorable));
    let mut c = CandidateReport::new(0);
    settle_score(&mut c, vec![1.0, 0.8], 0.2);
    assert_eq!(c.refusal, None);
    assert_eq!(c.zncc, 0.2);
    assert_eq!(c.reference_zncc, vec![1.0, 0.8]);
}

/// The bars read the existing observations' blur-matched scores against the
/// template, the reference observation's left out.
#[test]
fn the_bars_read_the_references_scores_against_the_template() {
    let points: Vec<(Point3<f64>, &[u32])> = grid_points().into_iter().map(|p| (p, FOUR)).collect();
    let mut cap = capture(&points);
    cap.recon.point_set.reference_observations = Some(vec![2; points.len()]);
    // Pooled: the bar is the smallest score of any reference but the
    // reference observation, over the candidates that reached the verdict.
    let pooled = AddImageToTracksOptions {
        rule: AcceptRule::PooledBasis {
            statistic: BasisStatistic::Min,
        },
        min_zncc: 0.0,
        ..gate_off()
    };
    let (_, report) = run(&cap, &pooled);
    let mut want = f64::INFINITY;
    for c in &report.candidates {
        assert_eq!(c.reference_observation, Some(2));
        assert_eq!(c.reference_zncc.len(), 4);
        assert_eq!(c.reference_zncc[2], 1.0);
        let scores: Vec<f64> = c.bar_zncc().collect();
        assert_eq!(scores.len(), 3);
        // The views see the same synthetic texture, so each score is 1 up to
        // the rounding of a dot product of two `f32` unit vectors, which can
        // land just above 1 (it does on the non-AVX2 render path).
        assert!(
            scores.iter().all(|&z| z <= 1.0 + 1e-6 && z > 0.5),
            "{scores:?}"
        );
        want = scores.iter().copied().fold(want, f64::min);
    }
    assert_eq!(report.pooled_bar, Some(want));
    // A track's own: the statistic over its three other references' scores.
    let track = AddImageToTracksOptions {
        rule: AcceptRule::TrackBasis {
            statistic: BasisStatistic::FractionOfMedian { fraction: 0.9 },
            pair: PairRule {
                statistic: PairStatistic::Mean,
                factor: 0.9,
            },
        },
        min_zncc: 0.0,
        ..gate_off()
    };
    let (_, report) = run(&cap, &track);
    for c in &report.candidates {
        let scores: Vec<f64> = c.bar_zncc().collect();
        let want = BasisStatistic::FractionOfMedian { fraction: 0.9 }.bar(&scores);
        assert_eq!(c.bar, want);
    }
    // And on a hand-built candidate: a reference observation's `1.0` would
    // raise the median from 0.62 to 0.63.
    let mut c = CandidateReport::new(0);
    c.references = vec![0, 1, 2, 3];
    c.reference_observation = Some(0);
    c.reference_zncc = vec![1.0, 0.6, 0.62, 0.64];
    c.zncc = 0.58;
    judge(&mut c, &track, None);
    assert!((c.bar - 0.9 * 0.62).abs() < 1e-12, "{}", c.bar);
}

#[test]
fn pooled_or_track_accepts_what_either_bar_accepts() {
    let options = AddImageToTracksOptions::default();
    let mut c = CandidateReport::new(0);
    c.references = vec![0, 1, 2];
    c.reference_zncc = vec![0.6, 0.62, 0.64];
    // Below the pooled bar, above 0.9 of its own track's median.
    c.zncc = 0.58;
    judge(&mut c, &options, Some(0.7));
    assert_eq!(c.refusal, None);
    assert!((c.bar - 0.9 * 0.62).abs() < 1e-12);
    // Below both.
    let mut c2 = c.clone();
    c2.refusal = None;
    c2.zncc = 0.5;
    c2.reference_zncc = vec![0.8, 0.85, 0.9];
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

/// An observation added before a point's reference observation moves the
/// index down one place, so it still names the same observation; one added
/// after it leaves the index alone, and the bitmap is not re-rendered.
#[test]
fn an_added_observation_moves_the_reference_index_past_it() {
    let mut recon = SfmrReconstruction::demo(4);
    let m = recon.point_set.tracks.len();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::<f32>::zeros((m, 2)),
        image_file_hashes: vec![[0u8; 16]; recon.image_count()],
    };
    // The reference column is present with the patch frame, so the points
    // carry one.
    let n = recon.point_count();
    recon.point_set.patch_u_halfvec_xyz = Some(Array2::from_elem((n, 3), 0.5));
    recon.point_set.patch_v_halfvec_xyz = Some(Array2::from_elem((n, 3), 0.25));
    // Point 2 sees images 2 and 3; point 3 sees images 3 and 4.
    recon.point_set.reference_observations = Some(vec![-1, -1, 1, 0]);
    let image_of = |r: &SfmrReconstruction, p: usize| {
        r.point_set
            .reference_observation_row(p)
            .map(|row| r.point_set.tracks[row].image_index)
    };
    // Image 0 lands before both observations of points 2 and 3.
    let out = insert_observations(&recon, 0, &[(2, [1.0, 1.0], 0.9), (3, [1.0, 1.0], 0.9)]);
    assert_eq!(
        out.point_set.reference_observations,
        Some(vec![-1, -1, 2, 1])
    );
    assert_eq!(image_of(&out, 2), Some(3));
    assert_eq!(image_of(&out, 3), Some(3));
    // Image 7 lands after them.
    let out = insert_observations(&recon, 7, &[(2, [1.0, 1.0], 0.9)]);
    assert_eq!(
        out.point_set.reference_observations,
        Some(vec![-1, -1, 1, 0])
    );
    out.validate_point_columns().unwrap();
}

#[test]
fn the_default_pooled_bar_is_the_median_minus_two_scaled_deviations() {
    // Measured on the reference render, the bar at three deviations let
    // through more bad extra observations on kerry_park than at two; see the
    // spec's "How the bars were measured on the reference render".
    match AddImageToTracksOptions::default().rule {
        AcceptRule::PooledOrTrack { pooled, .. } => {
            assert_eq!(pooled, BasisStatistic::MedianMinusMad { k: 2.0 });
        }
        other => panic!("the default rule is PooledOrTrack, got {other:?}"),
    }
}
