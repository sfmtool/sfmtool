// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! *Create Track Here* as a pair of versions: what a run that builds a track
//! leaves on the bench and in the reconstruction, what one that is refused
//! leaves (nothing), and when the gesture is greyed.
//!
//! The fixture is the capture core's track-at-pixel tests are decided against
//! (`sfmtool-core`'s `bench/tests/scene.rs`), rebuilt here because that one is
//! private to core's test build: pinhole cameras looking down world `+z` at a
//! textured plane, with a grid of points on the plane that every camera sees at
//! its exact projection. The grid's middle point is left out, and the query is
//! made where it would be, which is where core's own tests hold a point out.

use std::sync::Arc;

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};
use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::{CameraIntrinsics, CameraModel};
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::{ObservationSource, Point3D, SfmrImage, SfmrReconstruction, TrackObservation};

use super::{CreatedTrack, NOT_EMBEDDED_PATCHES, NOT_POSED};
use crate::action_log::Kind;
use crate::background::Operation;
use crate::progress::Detail;
use crate::scene::{ImageRef, ReconId, SceneNode};
use crate::state::AppState;

// ── The plane capture ───────────────────────────────────────────────────

const IMG_W: u32 = 128;
const IMG_H: u32 = 128;
const FOCAL: f64 = 160.0;
const PLANE_Z: f64 = 4.0;
const HALF_EXTENT: f64 = 0.12;
/// The camera centres, in world space: core's three.
const CENTERS: [[f64; 3]; 3] = [[-0.5, -0.3, 0.0], [0.45, 0.25, 0.0], [0.1, -0.55, 0.0]];
/// The point the query is made at: the middle of the grid, which the
/// reconstruction does not hold.
const HELD_OUT: Point3<f64> = Point3::new(0.0, 0.0, PLANE_Z);

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: f64::from(IMG_W) / 2.0,
            principal_point_y: f64::from(IMG_H) / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// The plane's texture, in world units: three sinusoids at different periods
/// and directions, so every patch of it registers in two dimensions.
fn texture(x: f64, y: f64) -> f64 {
    127.5 + 55.0 * (x * 17.0).sin() + 45.0 * (y * 23.0).cos() + 25.0 * ((x + y) * 31.0).sin()
}

/// What a pinhole at `center` looking down world `+z` sees of the plane.
fn photograph(center: [f64; 3]) -> ImageU8 {
    let (cx, cy) = (f64::from(IMG_W) / 2.0, f64::from(IMG_H) / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (f64::from(col) + 0.5 - cx) / FOCAL;
            let dy = (f64::from(row) + 0.5 - cy) / FOCAL;
            let lambda = PLANE_Z - center[2];
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            data.push(texture(x, y).clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8::new(IMG_W, IMG_H, 1, data)
}

/// The image row of a camera at `center`: a half-turn about `x`, which puts
/// the camera's `-z` axis along world `+z`.
fn image_row(i: usize) -> SfmrImage {
    let center = CENTERS[i];
    SfmrImage {
        name: format!("image_{i}.jpg"),
        camera_index: 0,
        quaternion_wxyz: UnitQuaternion::from_quaternion(Quaternion::new(0.0, 1.0, 0.0, 0.0)),
        translation_xyz: Vector3::new(-center[0], center[1], center[2]),
    }
}

/// Where `world` lands in image `i`, in source-image px.
pub(crate) fn project(i: usize, world: Point3<f64>) -> [f64; 2] {
    let row = image_row(i);
    let cam = row.quaternion_wxyz.to_rotation_matrix() * world.coords + row.translation_xyz;
    let (u, v) = pinhole()
        .ray_to_pixel([cam.x, cam.y, cam.z])
        .expect("every camera of the plane capture sees every point on it");
    [u, v]
}

/// Where the held-out point would be seen in image 0: a pixel on the textured
/// plane with neighbours all around it.
pub(crate) fn textured_pixel() -> [f64; 2] {
    project(0, HELD_OUT)
}

/// A pixel of image 0 more than every member's neighbourhood radius from every
/// point's projection, so no member finds anything to build from.
pub(crate) fn lonely_pixel() -> [f64; 2] {
    [2.0, 125.0]
}

/// The grid: five by five points on the plane 0.3 apart, less the middle one.
fn grid() -> Vec<Point3<f64>> {
    (-2..=2)
        .flat_map(|j| {
            (-2..=2).map(move |i| Point3::new(0.3 * f64::from(i), 0.3 * f64::from(j), PLANE_Z))
        })
        .filter(|p| *p != HELD_OUT)
        .collect()
}

/// An `embedded_patches` reconstruction of the plane capture: every grid
/// point observed by every image at its exact projection, standing on a patch
/// that faces the cameras.
pub(crate) fn plane_recon() -> SfmrReconstruction {
    let points = grid();
    let n = CENTERS.len();
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..n).map(image_row).collect();
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

    let set = &mut recon.point_set;
    set.points = points
        .iter()
        .map(|&position| Point3D {
            position,
            w: 1.0,
            color: [120, 130, 140],
            error: 0.5,
            normal: Vector3::new(0.0, 0.0, -1.0),
        })
        .collect();
    set.tracks = (0..points.len() as u32)
        .flat_map(|point_index| {
            (0..n as u32).map(move |image_index| TrackObservation {
                image_index,
                point_index,
            })
        })
        .collect();
    set.observation_counts = vec![n as u32; points.len()];
    let mut keypoints = Array2::<f32>::zeros((set.tracks.len(), 2));
    for (row, obs) in set.tracks.iter().enumerate() {
        let p = project(obs.image_index as usize, points[obs.point_index as usize]);
        keypoints[[row, 0]] = p[0] as f32;
        keypoints[[row, 1]] = p[1] as f32;
    }
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; n],
    };
    let mut u = Array2::<f32>::zeros((points.len(), 3));
    let mut v = Array2::<f32>::zeros((points.len(), 3));
    for (p, &center) in points.iter().enumerate() {
        let patch = OrientedPatch::from_center_normal(
            center,
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 0.0),
            [HALF_EXTENT, HALF_EXTENT],
        );
        for c in 0..3 {
            u[[p, c]] = (patch.u_axis[c] * HALF_EXTENT) as f32;
            v[[p, c]] = (patch.v_axis[c] * HALF_EXTENT) as f32;
        }
    }
    set.patch_u_halfvec_xyz = Some(u);
    set.patch_v_halfvec_xyz = Some(v);
    // A bitmap per point, as a file with patches carries, so the commit the
    // run ends in needs the track to bring one.
    set.patch_bitmaps_y_x_rgba = Some(Arc::new(Array4::zeros((points.len(), 8, 8, 4))));
    recon.metadata.feature_source = "embedded_patches".to_string();
    recon.rebuild_derived_fields();
    recon
}

/// Put the capture's photographs in the node's full-resolution cache, so the
/// run finds them without a file on disk.
pub(crate) fn cache_photographs(state: &mut AppState, id: ReconId) {
    for (i, &center) in CENTERS.iter().enumerate() {
        state.full_res_cache.insert(
            ImageRef::new(id, i),
            Some(Arc::new(ImageU8Pyramid::from_image(
                photograph(center),
                crate::state::PYRAMID_LEVELS,
            ))),
        );
    }
}

/// A state holding the plane capture as its one node, with its photographs
/// cached.
pub(crate) fn plane_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(plane_recon()));
    let id = state.selected_recon.expect("a selected reconstruction");
    cache_photographs(&mut state, id);
    (state, id)
}

fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

fn live_points(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").edited().point_count()
}

/// The Action Log's rows written since `since` entries, as `(kind, failed,
/// text)`.
fn rows_after(state: &AppState, since: usize) -> Vec<(Kind, bool, String)> {
    state
        .action_log
        .entries()
        .skip(since)
        .map(|entry| (entry.kind, entry.failed, entry.text.clone()))
        .collect()
}

// ── The run that builds a track ────────────────────────────────────────

/// A pixel on the textured plane between the points: the cascade builds a
/// track there, which lands as two versions -- the track put on the bench and
/// active, then committed as a new point -- and the item settles on the point
/// it wrote, which is the selection.
#[test]
fn a_track_built_at_a_pixel_is_put_on_the_bench_and_committed() {
    let (mut state, id) = plane_state();
    let image = ImageRef::new(id, 0);
    let points_before = live_points(&state, id);
    let versions_before = versions(&state, id);
    let rows_before = state.action_log.len();

    state
        .start_create_track_at_pixel(image, textured_pixel())
        .expect("a posed image of an embedded_patches node");
    assert_eq!(
        state.background_task().map(|task| task.operation.name),
        Some(Operation::CREATE_TRACK_AT_PIXEL.name),
        "the run is on a worker"
    );
    state.finish_background_task();

    assert_eq!(
        versions(&state, id) - versions_before,
        2,
        "the put and the commit"
    );
    assert_eq!(
        live_points(&state, id),
        points_before + 1,
        "one point created"
    );

    let Some(CreatedTrack::Committed {
        item,
        member,
        point,
    }) = state
        .last_background_task
        .as_ref()
        .and_then(|task| task.created_track.clone())
    else {
        panic!(
            "expected a committed track, got {:?}",
            state.last_background_task.as_ref().map(|t| &t.outcome)
        );
    };
    // No index files beside this node, so the clusters member has nothing to
    // read and the neighbours' transfer is what builds it.
    assert_eq!(member, "transfer");
    assert!(point.changed && point.replaced.is_none(), "{point:?}");
    // Labelled as a cluster started at the same pixel would be.
    let p = textured_pixel();
    assert_eq!(
        item,
        format!("image_0@{},{}", p[0].round() as i64, p[1].round() as i64)
    );

    // The item is on the bench, active, and seated on the point it wrote.
    let node = state.node(id).expect("loaded");
    let bench = node.history.current_bench();
    assert_eq!(crate::bench::active_track_label(bench), Some(item.as_str()));
    let track = bench.track(&item).expect("on the bench");
    assert_eq!(state.resolved_origin(node, track), Some(point.point));
    // The committed point is the selection, which Track View follows.
    assert_eq!(
        state.selected_point.map(|p| p.point),
        Some(point.point),
        "the commit selects the point it wrote"
    );

    // One bench row, then the commit's edit row, then the selection's.
    let rows = rows_after(&state, rows_before);
    let kinds: Vec<Kind> = rows.iter().map(|(kind, _, _)| *kind).collect();
    assert_eq!(
        kinds,
        [Kind::Bench, Kind::Edit, Kind::Selection],
        "{rows:?}"
    );
    assert!(rows.iter().all(|(_, failed, _)| !failed), "{rows:?}");
    assert!(
        rows[0].2.starts_with(&format!("Created {item} at ("))
            && rows[0].2.contains("with the transfer member"),
        "{}",
        rows[0].2
    );

    // One undo takes the point back and leaves the track on the bench.
    state.undo(id).expect("an undo");
    assert_eq!(live_points(&state, id), points_before);
    let bench = state.node(id).expect("loaded").history.current_bench();
    assert!(bench.track(&item).is_some(), "the track stays on the bench");
}

// ── The run that is refused ────────────────────────────────────────────

/// A pixel with no point anywhere near it: every member refuses, nothing is
/// put on the bench or committed, and one failed row says so in one sentence
/// -- the last member's stage and reason, then the missing index files --
/// with every member's refusal among its detail lines.
#[test]
fn a_refused_run_commits_nothing_and_logs_each_member() {
    let (mut state, id) = plane_state();
    let image = ImageRef::new(id, 0);
    // The premise: the pixel is further from every point's projection than any
    // member looks.
    let lonely = lonely_pixel();
    for point in grid() {
        let p = project(0, point);
        let d = ((p[0] - lonely[0]).powi(2) + (p[1] - lonely[1]).powi(2)).sqrt();
        assert!(d > 60.0, "a point projects {d} px from the lonely pixel");
    }
    let versions_before = versions(&state, id);
    let rows_before = state.action_log.len();
    let bench_before = Arc::clone(state.node(id).expect("loaded").history.current_bench());

    state
        .start_create_track_at_pixel(image, lonely)
        .expect("the run starts; the cascade is what refuses");
    state.finish_background_task();

    assert_eq!(versions(&state, id), versions_before, "nothing pushed");
    assert!(Arc::ptr_eq(
        &bench_before,
        state.node(id).expect("loaded").history.current_bench()
    ));
    let entry = state
        .action_log
        .entries()
        .skip(rows_before)
        .find(|entry| entry.failed)
        .expect("a failed row");
    assert_eq!(entry.kind, Kind::Bench);
    assert!(
        entry
            .text
            .starts_with("Cannot create a track at (2.0, 125.0) in image_0.jpg: every member refused; the last, constellation, at constellation: "),
        "{}",
        entry.text
    );
    // The index files the clusters and constellation members read were not
    // there, and the demo node has no path to build them beside.
    assert!(
        entry
            .text
            .contains("No SIFT index is open, so the constellation member had nothing to read.")
            && entry.text.contains(
                "No cluster patches file is open, so the clusters member had nothing to read."
            ),
        "{}",
        entry.text
    );
    let messages: Vec<&str> = entry
        .detail
        .iter()
        .filter_map(|row| match row {
            Detail::Message { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    for (member, stage) in [
        ("clusters", "clusters"),
        ("transfer", "neighbourhood"),
        ("sweep", "prior"),
        ("constellation", "constellation"),
    ] {
        let line = format!("{member} refused at {stage}: ");
        assert!(
            messages.iter().any(|m| m.starts_with(&line)),
            "no detail line {line:?} in {messages:?}"
        );
    }
    let Some(CreatedTrack::Refused { refusals }) = state
        .last_background_task
        .as_ref()
        .and_then(|task| task.created_track.clone())
    else {
        panic!("expected a refusal");
    };
    assert_eq!(refusals.len(), 4, "{refusals:?}");
}

// ── When the gesture is greyed ─────────────────────────────────────────

#[test]
fn create_track_here_is_greyed_where_it_cannot_run() {
    let (mut state, id) = plane_state();
    let image = ImageRef::new(id, 0);
    assert_eq!(state.create_track_here_refusal(image), None);

    // An image with no pose has no ray through the pixel.
    state.scene[0].recon_mut().image_table.images[1]
        .translation_xyz
        .x = f64::NAN;
    assert_eq!(
        state
            .create_track_here_refusal(ImageRef::new(id, 1))
            .as_deref(),
        Some(NOT_POSED)
    );

    // A pixel off the photograph is the gesture's refusal, in one row, and no
    // task.
    let rows_before = state.action_log.len();
    assert!(state
        .start_create_track_at_pixel(image, [-5.0, 10.0])
        .is_err());
    assert!(state.background_task().is_none());
    let rows = rows_after(&state, rows_before);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        rows[0].1 && rows[0].2.contains("not on the 128x128 photograph"),
        "{rows:?}"
    );

    // A busy node refuses every step on it.
    state
        .start_create_track_at_pixel(image, textured_pixel())
        .expect("starts");
    let why = state.create_track_here_refusal(image).expect("busy");
    assert!(why.contains("is busy"), "{why}");
    state.cancel_background_task();
    state.finish_background_task();

    // A node whose observations are `.sift` features cannot take the commit.
    let mut sift = AppState::new();
    sift.append_node(SceneNode::demo(SfmrReconstruction::demo(4)));
    let sift_id = sift.selected_recon.expect("selected");
    assert_eq!(
        sift.create_track_here_refusal(ImageRef::new(sift_id, 0))
            .as_deref(),
        Some(NOT_EMBEDDED_PATCHES)
    );
}
