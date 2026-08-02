// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;

// The synthetic scene of the sibling patch kernels' tests: pinhole cameras
// (looking down world +z) viewing a textured plane at z = PLANE_Z. The parent
// patch sits on that plane with its normal pointing back at the cameras; a
// candidate at an in-plane offset therefore lies on the same textured plane and
// must congeal onto it.
//
// Every camera renders the SAME texture with no per-view offset, so a correctly
// placed candidate is photometrically consistent across views and its true 3D
// position is exactly its requested centre — which is what the assertions below
// check against.

const PLANE_Z: f64 = 4.0;
const IMG_W: u32 = 320;
const IMG_H: u32 = 240;
const FOCAL: f64 = 260.0;
const HALF_EXTENT: f64 = 0.12;
const RES: u32 = 20;

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

/// Broadband, non-periodic plane texture: the spawned candidate has to find a
/// unique correlation peak away from the parent's own location.
fn texture(x: f64, y: f64) -> f64 {
    127.5
        + 46.0 * (x * 17.0).sin()
        + 38.0 * (y * 23.0).cos()
        + 22.0 * ((x + y) * 31.0).sin()
        + 14.0 * ((x - 2.0 * y) * 7.3).cos()
}

/// What a pinhole at `center` (looking down world +z) sees of the textured plane.
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

struct Scene {
    cams: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyrs: Vec<ImageU8Pyramid>,
}

impl Scene {
    fn new(centers: &[[f64; 3]]) -> Self {
        Self {
            cams: centers.iter().map(|_| pinhole()).collect(),
            poses: centers
                .iter()
                .map(|c| {
                    RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-c[0], c[1], c[2]])
                })
                .collect(),
            pyrs: centers
                .iter()
                .map(|c| ImageU8Pyramid::build(&render_plane_view(*c), 5))
                .collect(),
        }
    }

    fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.cams
            .iter()
            .zip(&self.poses)
            .zip(&self.pyrs)
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect()
    }
}

/// Four cameras with real parallax about the patch, well clear of degeneracy.
fn scene() -> Scene {
    Scene::new(&[
        [0.55, 0.10, 0.0],
        [-0.50, -0.15, 0.0],
        [0.05, 0.60, 0.0],
        [-0.20, -0.55, 0.30],
    ])
}

/// One parent patch on the plane, normal toward the cameras (-z).
fn parent_cloud() -> PatchCloud {
    PatchCloud {
        patches: vec![OrientedPatch::from_center_normal(
            Point3::new(0.0, 0.0, PLANE_Z),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 0.0),
            [HALF_EXTENT, HALF_EXTENT],
        )],
        point_indexes: vec![0],
    }
}

fn params() -> SpawnParams {
    SpawnParams {
        resolution: RES,
        ..SpawnParams::default()
    }
}

/// `n` copies of the full view set `[0, 1, 2, 3]`.
fn all_views(n: usize) -> Vec<Vec<u32>> {
    vec![vec![0, 1, 2, 3]; n]
}

/// The candidate's true world centre for offset `(du, dv)` off the parent.
fn expected_center(cloud: &PatchCloud, du: f64, dv: f64) -> Point3<f64> {
    let p = cloud.patch(0);
    p.center + p.u_axis * (du * p.half_extent[0]) + p.v_axis * (dv * p.half_extent[1])
}

fn distance(a: [f64; 3], b: Point3<f64>) -> f64 {
    ((a[0] - b.x).powi(2) + (a[1] - b.y).powi(2) + (a[2] - b.z).powi(2)).sqrt()
}

#[test]
fn candidate_on_the_textured_plane_spawns() {
    // A candidate two half-extents off the parent, inside the textured region and
    // in frame everywhere: it must localize in every view, triangulate back onto
    // the plane, and pass every gate.
    let scene = scene();
    let cloud = parent_cloud();
    let out = spawn_candidate_tracks(
        &scene.views(),
        &cloud,
        &[0],
        &[[2.0, 0.0]],
        &all_views(1),
        &params(),
    );

    assert_eq!(out.len(), 1);
    assert_eq!(
        out.status[0],
        SpawnStatus::Spawned as u8,
        "expected a spawn, got status {} (rms {})",
        out.status[0],
        out.reproj_rms_px[0]
    );
    assert_eq!(out.n_views[0], 4, "every view should survive");

    let truth = expected_center(&cloud, 2.0, 0.0);
    // The requested centre is exact arithmetic on the parent's frame.
    assert!(distance(out.requested_centers[0], truth) < 1e-12);
    // The congealed position lands on the plane, well inside a patch half-extent.
    assert!(
        distance(out.positions[0], truth) < 0.25 * HALF_EXTENT,
        "spawned at {:?}, truth {truth:?}",
        out.positions[0]
    );
    assert!((out.positions[0][2] - PLANE_Z).abs() < 0.05 * HALF_EXTENT);
    assert!(out.reproj_rms_px[0] > 0.0 && out.reproj_rms_px[0] < 0.5);
}

#[test]
fn candidate_pushed_off_every_image_is_too_few_views() {
    // A candidate hundreds of patch widths out projects outside every frame, so
    // the localizer keeps nothing.
    let scene = scene();
    let cloud = parent_cloud();
    let out = spawn_candidate_tracks(
        &scene.views(),
        &cloud,
        &[0],
        &[[400.0, 400.0]],
        &all_views(1),
        &params(),
    );

    assert_eq!(out.status[0], SpawnStatus::TooFewViews as u8);
    assert_eq!(out.n_views[0], 0);
    assert!(out.positions[0].iter().all(|c| c.is_nan()));
    assert!(out.reproj_rms_px[0].is_nan());
    // A candidate that never reached triangulation reports no observations.
    assert_eq!(out.obs_offsets, vec![0, 0]);
    // The requested centre is still reported, so the caller can see what it asked.
    assert!(
        distance(
            out.requested_centers[0],
            expected_center(&cloud, 400.0, 400.0)
        ) < 1e-9
    );
}

#[test]
fn empty_view_set_is_too_few_views() {
    let scene = scene();
    let cloud = parent_cloud();
    let out = spawn_candidate_tracks(
        &scene.views(),
        &cloud,
        &[0],
        &[[2.0, 0.0]],
        &[Vec::new()],
        &params(),
    );

    assert_eq!(out.status[0], SpawnStatus::TooFewViews as u8);
    assert_eq!(out.n_views[0], 0);
}

#[test]
fn min_views_floor_above_the_survivors_kills_the_candidate() {
    // Same candidate as the spawning test, with the floor set one above the four
    // views that survive — a clean margin from the boundary in both directions.
    let scene = scene();
    let cloud = parent_cloud();
    let strict = SpawnParams {
        min_views: 5,
        ..params()
    };
    let out = spawn_candidate_tracks(
        &scene.views(),
        &cloud,
        &[0],
        &[[2.0, 0.0]],
        &all_views(1),
        &strict,
    );

    assert_eq!(out.status[0], SpawnStatus::TooFewViews as u8);
    assert_eq!(out.n_views[0], 0, "no views are carried past the floor");
}

#[test]
fn unreachable_reprojection_gate_reports_high_reproj() {
    // Take the RMS the spawning candidate actually achieves and set the gate an
    // order of magnitude below it: same geometry, same observations, only the
    // verdict changes.
    let scene = scene();
    let views = scene.views();
    let cloud = parent_cloud();
    let baseline = spawn_candidate_tracks(
        &views,
        &cloud,
        &[0],
        &[[2.0, 0.0]],
        &all_views(1),
        &params(),
    );
    let achieved = baseline.reproj_rms_px[0];
    assert!(
        achieved > 0.0 && achieved.is_finite(),
        "need a positive baseline RMS to set an unreachable gate, got {achieved}"
    );

    let strict = SpawnParams {
        max_reproj_rms_px: achieved / 10.0,
        ..params()
    };
    let out = spawn_candidate_tracks(&views, &cloud, &[0], &[[2.0, 0.0]], &all_views(1), &strict);

    assert_eq!(out.status[0], SpawnStatus::HighReproj as u8);
    assert!(out.positions[0].iter().all(|c| c.is_nan()));
    // The RMS that failed is reported, and the observations that produced it stay.
    assert!((out.reproj_rms_px[0] - achieved).abs() < 1e-12);
    assert_eq!(out.n_views[0], 4);
    assert_eq!(out.obs_view_indexes.len(), 4);
}

#[test]
fn discrete_only_still_spawns() {
    // `subpixel_sweeps = 0` skips refinement entirely; the discrete keypoints go
    // straight to triangulation and still clear the gates.
    let scene = scene();
    let cloud = parent_cloud();
    let discrete = SpawnParams {
        subpixel_sweeps: 0,
        ..params()
    };
    let out = spawn_candidate_tracks(
        &scene.views(),
        &cloud,
        &[0],
        &[[2.0, 0.0]],
        &all_views(1),
        &discrete,
    );

    assert_eq!(out.status[0], SpawnStatus::Spawned as u8);
    assert_eq!(out.n_views[0], 4);
    assert!(distance(out.positions[0], expected_center(&cloud, 2.0, 0.0)) < 0.5 * HALF_EXTENT);
}

#[test]
fn batched_candidates_match_the_same_candidates_spawned_alone() {
    // Several candidates off one parent in a single batch must equal the same
    // candidates run one at a time — candidates are independent.
    let scene = scene();
    let views = scene.views();
    let cloud = parent_cloud();
    let offsets = [[2.0, 0.0], [-2.0, 0.5], [0.0, 2.0], [1.5, -1.5]];
    let parents = [0u32; 4];

    let batched =
        spawn_candidate_tracks(&views, &cloud, &parents, &offsets, &all_views(4), &params());

    for (i, off) in offsets.iter().enumerate() {
        let alone = spawn_candidate_tracks(
            &views,
            &cloud,
            &[0],
            std::slice::from_ref(off),
            &all_views(1),
            &params(),
        );
        assert_eq!(batched.status[i], alone.status[0], "candidate {i} status");
        assert_eq!(batched.n_views[i], alone.n_views[0], "candidate {i} views");
        for k in 0..3 {
            let (b, a) = (batched.positions[i][k], alone.positions[0][k]);
            assert!(
                (b.is_nan() && a.is_nan()) || (b - a).abs() < 1e-12,
                "candidate {i} position component {k}: {b} vs {a}"
            );
        }
    }
    // At least some of the batch actually spawned, or the comparison is vacuous.
    assert!(
        batched
            .status
            .iter()
            .filter(|&&s| s == SpawnStatus::Spawned as u8)
            .count()
            >= 2
    );
}

#[test]
fn csr_bookkeeping_is_consistent() {
    // A batch mixing survivors with a candidate that dies before triangulation:
    // the offsets must still be a valid CSR over the observation arrays, and each
    // candidate's views must come back ascending.
    let scene = scene();
    let cloud = parent_cloud();
    let offsets = [[2.0, 0.0], [400.0, 400.0], [0.0, -2.0]];
    let out = spawn_candidate_tracks(
        &scene.views(),
        &cloud,
        &[0, 0, 0],
        &offsets,
        // A deliberately unsorted view set: the CSR contract is view-index order
        // regardless of how the caller listed them.
        &vec![vec![2, 0, 3, 1]; 3],
        &params(),
    );

    assert_eq!(out.obs_offsets.len(), out.len() + 1);
    assert_eq!(out.obs_offsets[0], 0);
    assert!(out.obs_offsets.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(
        *out.obs_offsets.last().unwrap() as usize,
        out.obs_view_indexes.len()
    );
    assert_eq!(out.obs_keypoints_xy.len(), out.obs_view_indexes.len());

    for i in 0..out.len() {
        let (lo, hi) = (out.obs_offsets[i] as usize, out.obs_offsets[i + 1] as usize);
        let block = &out.obs_view_indexes[lo..hi];
        assert!(
            block.windows(2).all(|w| w[0] < w[1]),
            "candidate {i} observations are not in ascending view order: {block:?}"
        );
        if out.status[i] == SpawnStatus::TooFewViews as u8 {
            assert_eq!(lo, hi, "candidate {i} died before triangulation");
        } else {
            assert_eq!(hi - lo, out.n_views[i] as usize);
        }
    }
    assert_eq!(out.status[1], SpawnStatus::TooFewViews as u8);
}

#[test]
fn empty_batch_returns_an_empty_csr() {
    let scene = scene();
    let cloud = parent_cloud();
    let out = spawn_candidate_tracks(&scene.views(), &cloud, &[], &[], &[], &params());

    assert!(out.is_empty());
    assert_eq!(out.obs_offsets, vec![0]);
    assert!(out.obs_view_indexes.is_empty());
}
