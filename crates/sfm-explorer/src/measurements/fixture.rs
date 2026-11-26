// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A small synthetic reconstruction with distance measurements whose readings
//! are known by construction, for the tests of the selection, the readings,
//! the Scene tree rows, Measurement View and the wire.
//!
//! Two cameras on a short arc at radius 8 around the origin and a third posed
//! image that observes nothing, as core's resolver tests build theirs, with
//! three finite points on the x axis and one point at infinity. Every point's
//! square patch faces the cameras, and every keypoint is the exact projection
//! of its point, so an endpoint placed on an observation's keypoint resolves to
//! texel `(0, 0)` of that point's patch, the point itself.
//!
//! The five measurements, in metres in the file's unit when it has one:
//!
//! | # | A | B | Stated | Reading |
//! |---|---|---|---|---|
//! | 0 | point 0 in image 0 | point 1 in image 0 | 153 cm | 1.5 m apart, the scale 1.02 |
//! | 1 | point 1 in image 0 | point 2 in image 1 | 3.06 m | 3.0 m apart, the scale 1.02 |
//! | 2 | point 2 in image 0 | a pixel no square holds | 1 m | B refused |
//! | 3 | point 0 in image 0 | the point at infinity | 1 m | B at infinity |
//! | 4 | point 0 in image 1 | point 2 in image 0 | 1.8 m | 1.5 m apart, the scale 1.2 |

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::{Array1, Array2, Array3};
use sfmtool_core::{
    CameraIntrinsics, CameraModel, DistanceMeasurementsData, ObservationSource, Point3D, SfmrImage,
    SfmrReconstruction, TrackObservation, LENGTH_UNIT_CM, LENGTH_UNIT_M,
};

const IMG_W: u32 = 640;
const IMG_H: u32 = 480;
const IMAGES: usize = 3;

/// The scale every good measurement of the fixture implies.
pub(crate) const GOOD_SCALE: f64 = 1.02;

/// The scale measurement 4, the bad one, implies.
pub(crate) const BAD_SCALE: f64 = 1.2;

/// The fixture's finite points, in scene units.
pub(crate) const POINTS: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]];

/// The index of the point at infinity.
pub(crate) const SKY: u32 = 3;

/// The pixel of image 0 no square holds.
pub(crate) const BARE_PIXEL: [f32; 2] = [5.0, 5.0];

/// One measurement of the fixture: its two endpoints, each an image and a
/// pixel, its stated distance, its unit code and its label.
type Row = ([(u32, [f32; 2]); 2], f32, u8, &'static str);

/// One point of the fixture: its homogeneous position and its patch's two
/// half-vectors.
struct Pt {
    pos: [f64; 3],
    w: f64,
    u: [f64; 3],
    v: [f64; 3],
}

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: f64::from(IMG_W) / 2.0,
            principal_point_y: f64::from(IMG_H) / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
        rolling_shutter: None,
    }
}

/// Where the homogeneous point `(xyz, w)` lands in image `i`.
fn project(recon: &SfmrReconstruction, i: usize, xyz: Vector3<f64>, w: f64) -> [f32; 2] {
    let image = &recon.image_set.images[i];
    let local = image.quaternion_wxyz * xyz + image.translation_xyz * w;
    let (x, y) = recon.image_set.cameras[0]
        .ray_to_pixel([local.x, local.y, local.z])
        .expect("the fixture's points are in front of every camera");
    [x as f32, y as f32]
}

/// The fixture, its file declaring `world_space_unit` (`None` for scene
/// units).
pub(crate) fn measured_reconstruction(world_space_unit: Option<&str>) -> SfmrReconstruction {
    let mut points: Vec<Pt> = POINTS
        .iter()
        .map(|&pos| Pt {
            pos,
            w: 1.0,
            u: [0.2, 0.0, 0.0],
            v: [0.0, 0.2, 0.0],
        })
        .collect();
    // A direction 0.1 rad off image 0's line of sight to the origin, with a
    // patch on the direction sphere 0.02 rad across, as core's tests make one.
    points.push(Pt {
        pos: [0.0, 0.0, -1.0],
        w: 0.0,
        u: [0.02, 0.0, 0.0],
        v: [0.0, -0.02, 0.0],
    });

    let mut recon = SfmrReconstruction::demo(1);
    recon.image_set.cameras = vec![pinhole()];
    recon.image_set.images = (0..IMAGES)
        .map(|i| {
            // Images 0 and 2 share a pose; image 1 is 0.2 rad round the arc.
            let angle: f64 = if i == 1 { 0.1 } else { -0.1 };
            let centre = Vector3::new(8.0 * angle.sin(), 0.0, 8.0 * angle.cos());
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
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for (p, spec) in points.iter().enumerate() {
        for i in [0, 1] {
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push(project(&recon, i, Vector3::from(spec.pos), spec.w));
        }
        counts.push(2);
    }
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, xy) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = xy[0];
        keypoints_xy[[row, 1]] = xy[1];
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
    recon.metadata.world_space_unit = world_space_unit.map(str::to_string);
    recon.rebuild_derived_fields();

    // The keypoint of point `p` in image `i`: its observation `2p + i`.
    let kp = |p: usize, i: usize| keypoints[2 * p + i];
    let rows: [Row; 5] = [
        (
            [(0, kp(0, 0)), (0, kp(1, 0))],
            (150.0 * GOOD_SCALE) as f32,
            LENGTH_UNIT_CM,
            "ear to ear",
        ),
        (
            [(0, kp(1, 0)), (1, kp(2, 1))],
            (3.0 * GOOD_SCALE) as f32,
            LENGTH_UNIT_M,
            "nose to tail",
        ),
        ([(0, kp(2, 0)), (0, BARE_PIXEL)], 1.0, LENGTH_UNIT_M, ""),
        (
            [(0, kp(0, 0)), (0, kp(3, 0))],
            1.0,
            LENGTH_UNIT_M,
            "to the sky",
        ),
        (
            [(1, kp(0, 1)), (0, kp(2, 0))],
            (1.5 * BAD_SCALE) as f32,
            LENGTH_UNIT_M,
            "paw width",
        ),
    ];
    let n = rows.len();
    let mut image_indexes = Array2::<u32>::zeros((n, 2));
    let mut endpoints_xy = Array3::<f32>::zeros((n, 2, 2));
    for (i, (ends, _, _, _)) in rows.iter().enumerate() {
        for (k, (image, xy)) in ends.iter().enumerate() {
            image_indexes[[i, k]] = *image;
            endpoints_xy[[i, k, 0]] = xy[0];
            endpoints_xy[[i, k, 1]] = xy[1];
        }
    }
    recon.image_set.measurements.distance = Some(DistanceMeasurementsData {
        distances: Array1::from_iter(rows.iter().map(|r| r.1)),
        units: Array1::from_iter(rows.iter().map(|r| r.2)),
        image_indexes,
        endpoints_xy,
        labels: rows.iter().map(|r| r.3.to_string()).collect(),
    });
    recon
}

/// `recon`, a reconstruction made by [`measured_reconstruction`], with only
/// its distance measurements at `keep`, in that order.
pub(crate) fn keeping(mut recon: SfmrReconstruction, keep: &[usize]) -> SfmrReconstruction {
    let m = recon
        .image_set
        .measurements
        .distance
        .take()
        .expect("the fixture stores distance measurements");
    let n = keep.len();
    let mut image_indexes = Array2::<u32>::zeros((n, 2));
    let mut endpoints_xy = Array3::<f32>::zeros((n, 2, 2));
    for (row, &i) in keep.iter().enumerate() {
        for k in 0..2 {
            image_indexes[[row, k]] = m.image_indexes[[i, k]];
            for c in 0..2 {
                endpoints_xy[[row, k, c]] = m.endpoints_xy[[i, k, c]];
            }
        }
    }
    recon.image_set.measurements.distance = Some(DistanceMeasurementsData {
        distances: Array1::from_iter(keep.iter().map(|&i| m.distances[i])),
        units: Array1::from_iter(keep.iter().map(|&i| m.units[i])),
        image_indexes,
        endpoints_xy,
        labels: keep.iter().map(|&i| m.labels[i].clone()).collect(),
    });
    recon
}

/// The fixture in a fresh viewer state, as its one node, with its id.
pub(crate) fn measured_state(
    world_space_unit: Option<&str>,
) -> (crate::state::AppState, crate::scene::ReconId) {
    let mut state = crate::state::AppState::new();
    let node = crate::scene::SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        measured_reconstruction(world_space_unit),
    );
    let id = state.append_node(node);
    (state, id)
}

/// The reference observation of each point of
/// [`referenced_reconstruction`], as indexes into its track: point 0's is its
/// observation in image 1, point 1 has none, and point 2's and the point at
/// infinity's are their observations in image 0.
pub(crate) const REFERENCES: [i32; 4] = [1, -1, 0, 0];

/// The fixture with a reference observation per point
/// ([`REFERENCES`]), as a file with patch frames stores them, for the tests
/// of a point's reference pixel.
pub(crate) fn referenced_reconstruction(world_space_unit: Option<&str>) -> SfmrReconstruction {
    let mut recon = measured_reconstruction(world_space_unit);
    recon.point_set.reference_observations = Some(REFERENCES.to_vec());
    recon
}

/// [`referenced_reconstruction`] in a fresh viewer state, as its one node,
/// with its id.
pub(crate) fn referenced_state(
    world_space_unit: Option<&str>,
) -> (crate::state::AppState, crate::scene::ReconId) {
    let mut state = crate::state::AppState::new();
    let node = crate::scene::SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        referenced_reconstruction(world_space_unit),
    );
    let id = state.append_node(node);
    (state, id)
}
