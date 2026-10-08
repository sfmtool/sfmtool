// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use sfmtool_matches_format::{ClusterCellStatus, ClusterMemberStatus};

use super::*;
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::{RigidTransform, RotQuaternion};

const RESOLUTION: u32 = 25;
const PATCH_SIZE: f64 = 12.0;

fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 800.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

/// `cam_from_world` of a camera at `centre` looking at the origin, +Y up in
/// the image, −Z forward.
fn look_at(centre: Vector3<f64>) -> RigidTransform {
    let forward = (-centre).normalize();
    let up_world = Vector3::y();
    let right = forward.cross(&up_world).normalize();
    let up = right.cross(&forward);
    let back = -forward;
    let r = Matrix3::from_rows(&[right.transpose(), up.transpose(), back.transpose()]);
    let t = -(r * centre);
    RigidTransform::new(RotQuaternion::from_rotation_matrix(r), t)
}

/// The plane through the origin with normal `n`, seen by cameras at
/// `centres`; the first camera holds the reference member.
struct Scene {
    normal: Vector3<f64>,
    poses: Vec<RigidTransform>,
    cam: CameraIntrinsics,
}

impl Scene {
    fn new(normal: Vector3<f64>, centres: &[Vector3<f64>]) -> Scene {
        Scene {
            normal: normal.normalize(),
            poses: centres.iter().map(|&c| look_at(c)).collect(),
            cam: camera(),
        }
    }

    fn project(&self, i: usize, x: Vector3<f64>) -> Vector2<f64> {
        let p = self.poses[i].transform_point(&x.into());
        let (u, v) = self.cam.project(p.x / -p.z, p.y / -p.z);
        Vector2::new(u, v)
    }

    /// Where the ray through `pixel` of image `i` meets the plane.
    fn on_plane(&self, i: usize, pixel: Vector2<f64>) -> Vector3<f64> {
        let pose = &self.poses[i];
        let r_wc = pose.to_rotation_matrix().transpose();
        let centre = -(r_wc * pose.translation);
        let d = self.cam.pixel_to_ray(pixel.x, pixel.y);
        let d = r_wc * Vector3::new(d[0], d[1], d[2]);
        let s = -centre.dot(&self.normal) / d.dot(&self.normal);
        centre + s * d
    }

    /// Image `i`'s pixel of the plane point under reference pixel `x`.
    fn transfer(&self, i: usize, x: Vector2<f64>) -> Vector2<f64> {
        self.project(i, self.on_plane(0, x))
    }

    fn cam_from_world(&self) -> Vec<Option<RigidTransform>> {
        self.poses.iter().cloned().map(Some).collect()
    }
}

/// One cluster's member rows, the reference first.
struct Members {
    images: Vec<u32>,
    status: Vec<ClusterMemberStatus>,
    positions: Vec<[f32; 2]>,
    shapes: Vec<[[f32; 2]; 2]>,
    shifts: Vec<[[f32; 2]; 9]>,
    cells: Vec<[ClusterCellStatus; 9]>,
}

/// A cluster with the reference in image 0 and one kept member in every other
/// image, each member's shape the plane's affine at the patch centre and its
/// cell displacements the exact remainder: the homography's second-order term.
fn planar_members(scene: &Scene) -> Members {
    planar_members_with(scene, scene)
}

/// [`planar_members`] with every member's shape the affine of `shape_scene`'s
/// plane instead, a plane through the same point at another tilt: the shape a
/// wrong tilt would give. The displacements, measured from those shapes to the
/// true plane's transfer, carry the correction.
fn planar_members_with(scene: &Scene, shape_scene: &Scene) -> Members {
    let step = PATCH_SIZE / RESOLUTION as f64;
    let centres = cell_centres(RESOLUTION);
    let s_ref = Matrix2::new(4.0, 0.5, -0.3, 3.5);
    let x0 = scene.project(0, Vector3::zeros());
    let round = |m: Matrix2<f64>| {
        [
            [m[(0, 0)] as f32, m[(0, 1)] as f32],
            [m[(1, 0)] as f32, m[(1, 1)] as f32],
        ]
    };
    let unround = |s: [[f32; 2]; 2]| {
        Matrix2::new(
            f64::from(s[0][0]),
            f64::from(s[0][1]),
            f64::from(s[1][0]),
            f64::from(s[1][1]),
        )
    };
    let ref_pos = [x0.x as f32, x0.y as f32];
    let ref_shape = round(s_ref);
    let ref_pos64 = Vector2::new(f64::from(ref_pos[0]), f64::from(ref_pos[1]));
    let s_ref64 = unround(ref_shape);

    let mut m = Members {
        images: vec![0],
        status: vec![ClusterMemberStatus::Reference],
        positions: vec![ref_pos],
        shapes: vec![ref_shape],
        shifts: vec![[[f32::NAN; 2]; 9]],
        cells: vec![[ClusterCellStatus::NotAttempted; 9]],
    };
    for i in 1..scene.poses.len() {
        // Central-difference Jacobian of the reference → member transfer.
        let h = 1e-3;
        let col = |e: Vector2<f64>| {
            (shape_scene.transfer(i, ref_pos64 + h * e)
                - shape_scene.transfer(i, ref_pos64 - h * e))
                / (2.0 * h)
        };
        let jx = col(Vector2::x());
        let jy = col(Vector2::y());
        let j = Matrix2::new(jx.x, jy.x, jx.y, jy.y);
        let pos = scene.transfer(i, ref_pos64);
        let pos = [pos.x as f32, pos.y as f32];
        let shape = round(j * s_ref64);
        let pos64 = Vector2::new(f64::from(pos[0]), f64::from(pos[1]));
        let inv = (step * unround(shape)).try_inverse().unwrap();
        let shifts = std::array::from_fn(|c| {
            let u = Vector2::new(centres[c][0], centres[c][1]);
            let x_ref = ref_pos64 + step * s_ref64 * u;
            let g = inv * (scene.transfer(i, x_ref) - pos64);
            [(g.x - u.x) as f32, (g.y - u.y) as f32]
        });
        m.images.push(i as u32);
        m.status.push(ClusterMemberStatus::Kept);
        m.positions.push(pos);
        m.shapes.push(shape);
        m.shifts.push(shifts);
        m.cells.push([ClusterCellStatus::Fitted; 9]);
    }
    m
}

/// Cameras on a 10-unit ring in front of the plane.
fn ring_centres() -> Vec<Vector3<f64>> {
    vec![
        Vector3::new(0.5, 0.3, 10.0),
        Vector3::new(3.5, 0.0, 9.5),
        Vector3::new(-3.0, 1.0, 9.5),
        Vector3::new(0.5, 3.5, 9.5),
        Vector3::new(0.0, -3.0, 9.5),
    ]
}

fn run(
    scene: &Scene,
    clusters: &[&Members],
    references: &[u32],
    poses: &[Option<RigidTransform>],
    params: &CellPlaneParams,
) -> Vec<CellPlaneNormal> {
    let mut starts = vec![0u32];
    let mut images = Vec::new();
    let mut status = Vec::new();
    let mut positions = Vec::new();
    let mut shapes = Vec::new();
    let mut shifts = Vec::new();
    let mut cells = Vec::new();
    for m in clusters {
        images.extend_from_slice(&m.images);
        status.extend_from_slice(&m.status);
        positions.extend_from_slice(&m.positions);
        shapes.extend_from_slice(&m.shapes);
        shifts.extend_from_slice(&m.shifts);
        cells.extend_from_slice(&m.cells);
        starts.push(images.len() as u32);
    }
    let cameras = [scene.cam.clone()];
    let image_camera = vec![0u32; poses.len()];
    cell_plane_normals(
        &CellPlaneClusters {
            cluster_starts: &starts,
            reference_members: references,
            member_images: &images,
            member_status: &status,
            member_positions: &positions,
            member_shapes: &shapes,
            cell_shift_px: &shifts,
            cell_status: &cells,
            patch_size: PATCH_SIZE,
            resolution: RESOLUTION,
        },
        &CellPlaneCameras {
            cameras: &cameras,
            image_camera: &image_camera,
            cam_from_world: poses,
        },
        params,
    )
}

fn angle_deg(a: [f64; 3], b: Vector3<f64>) -> f64 {
    let a = Vector3::from(a);
    (a.dot(&b) / (a.norm() * b.norm()))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn tilted_normal() -> Vector3<f64> {
    Vector3::new(0.45, -0.3, 1.0).normalize()
}

#[test]
fn planar_cluster_recovers_its_normal_with_both_axes() {
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let members = planar_members(&scene);
    // The displacements carry a second-order term: without it the cells would
    // all sit on the affine placement.
    let largest = members.shifts[1..]
        .iter()
        .flat_map(|s| s.iter())
        .map(|d| d[0].hypot(d[1]))
        .fold(0.0f32, f32::max);
    assert!(largest > 0.02, "second-order term too small: {largest}");

    let out = run(
        &scene,
        &[&members],
        &[0],
        &scene.cam_from_world(),
        &CellPlaneParams::default(),
    );
    let r = &out[0];
    assert_eq!(r.determinacy, NormalDeterminacy::BothAxes);
    let err = angle_deg(r.normal, scene.normal);
    assert!(err < 0.5, "normal off by {err}°");
    assert!(r.cell_status.iter().all(|&s| s == CellPlaneStatus::InPlane));
    assert!(r.cell_rays.iter().all(|&n| n == 5));
    // Sign: toward the cameras.
    assert!(Vector3::from(r.normal).dot(&Vector3::from(r.view_dir)) > 0.0);
    assert!(r.anisotropy > 0.5);
    for p in r.cell_positions {
        assert!(Vector3::from(p).dot(&scene.normal).abs() < 1e-3);
    }
}

#[test]
fn the_displacements_correct_a_wrong_stored_shape() {
    // Every member's stored shape is the affine of a plane tilted 20° away
    // from the true one; the displacements, measured from those shapes, carry
    // the correction. Zeroing them leaves the cells where the wrong shapes put
    // them, on the wrong plane.
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let wrong = Scene::new(Vector3::new(0.0, -0.3, 1.0), &ring_centres());
    let tilt = angle_deg(wrong.normal.into(), scene.normal);
    assert!((15.0..30.0).contains(&tilt), "shape plane tilted {tilt}°");
    let mut members = planar_members_with(&scene, &wrong);
    let params = CellPlaneParams::default();
    let exact = run(&scene, &[&members], &[0], &scene.cam_from_world(), &params);
    for s in members.shifts[1..].iter_mut() {
        *s = [[0.0; 2]; 9];
    }
    let zeroed = run(&scene, &[&members], &[0], &scene.cam_from_world(), &params);
    let e_exact = angle_deg(exact[0].normal, scene.normal);
    assert!(e_exact < 0.5, "with displacements {e_exact}°");
    // Without them the cells are where the wrong shapes put them: they either
    // leave the plane fit or tilt it.
    let e_zeroed = angle_deg(zeroed[0].normal, scene.normal);
    assert!(
        zeroed[0].determinacy != NormalDeterminacy::BothAxes || e_zeroed > 2.0,
        "zeroed {:?} at {e_zeroed}°",
        zeroed[0].determinacy
    );
    // They lie on the plane the shapes came from.
    let e_wrong = angle_deg(zeroed[0].normal, wrong.normal);
    assert!(e_wrong < 0.5, "zeroed off the shapes' plane by {e_wrong}°");
}

#[test]
fn collinear_cells_give_one_axis_naming_the_line() {
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let mut members = planar_members(&scene);
    for cells in members.cells[1..].iter_mut() {
        for (j, c) in cells.iter_mut().enumerate() {
            if j / 3 != 1 {
                *c = ClusterCellStatus::RefusedZncc;
            }
        }
    }
    let out = run(
        &scene,
        &[&members],
        &[0],
        &scene.cam_from_world(),
        &CellPlaneParams::default(),
    );
    let r = &out[0];
    let NormalDeterminacy::OneAxis { free_axis } = r.determinacy else {
        panic!("expected one axis, got {:?}", r.determinacy);
    };
    let line =
        (Vector3::from(r.cell_positions[5]) - Vector3::from(r.cell_positions[3])).normalize();
    let axis = Vector3::from(free_axis);
    assert!(axis.dot(&line).abs() > 0.9999, "free axis off the row");
    // The fixed component: the normal is perpendicular to the line, and so is
    // the true normal, since the line lies in the plane.
    assert!(Vector3::from(r.normal).dot(&axis).abs() < 1e-9);
    assert!(scene.normal.dot(&axis).abs() < 1e-3);
    // About the line it stays at the prior: in the plane of the axis and the
    // viewing direction.
    let prior = Vector3::from(r.view_dir);
    assert!(Vector3::from(r.normal).dot(&axis.cross(&prior)).abs() < 1e-9);
    for j in [0, 1, 2, 6, 7, 8] {
        assert_eq!(r.cell_status[j], CellPlaneStatus::TooFewRays);
        assert_eq!(r.cell_rays[j], 1);
        assert!(r.cell_positions[j][0].is_nan());
    }
}

#[test]
fn fewer_than_three_cells_give_no_normal() {
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let mut members = planar_members(&scene);
    for cells in members.cells[1..].iter_mut() {
        for (j, c) in cells.iter_mut().enumerate() {
            if j != 0 && j != 4 {
                *c = ClusterCellStatus::RefusedCurvature;
            }
        }
    }
    let out = run(
        &scene,
        &[&members],
        &[0],
        &scene.cam_from_world(),
        &CellPlaneParams::default(),
    );
    let r = &out[0];
    assert_eq!(r.determinacy, NormalDeterminacy::None);
    assert!(r.normal.iter().all(|v| v.is_nan()));
    assert_eq!(r.cell_status[0], CellPlaneStatus::Positioned);
    assert_eq!(r.cell_status[4], CellPlaneStatus::Positioned);
    assert!(r.plane_rms.is_nan());
}

#[test]
fn a_cell_seen_along_one_ray_is_skipped() {
    let scene = Scene::new(tilted_normal(), &ring_centres());
    // Cluster 0: the reference is posed and cell 0 is fitted in no kept member,
    // so only the reference's ray reaches it.
    let mut a = planar_members(&scene);
    for cells in a.cells[1..].iter_mut() {
        cells[0] = ClusterCellStatus::RefusedBound;
    }
    // Cluster 1: the same scene, the reference's image unposed, and cell 0
    // fitted in one kept member only.
    let mut b = planar_members(&scene);
    for cells in b.cells[2..].iter_mut() {
        cells[0] = ClusterCellStatus::RefusedZncc;
    }
    let n_a = a.images.len() as u32;
    // Cluster 1's reference sits in an image of its own, left unposed.
    let unposed = scene.poses.len() as u32;
    b.images[0] = unposed;
    let mut poses = scene.cam_from_world();
    poses.push(None);
    let out = run(
        &scene,
        &[&a, &b],
        &[0, n_a],
        &poses,
        &CellPlaneParams::default(),
    );

    for r in &out {
        assert_eq!(r.cell_rays[0], 1);
        assert_eq!(r.cell_status[0], CellPlaneStatus::TooFewRays);
        assert!(r.cell_positions[0][0].is_nan());
        assert_eq!(r.determinacy, NormalDeterminacy::BothAxes);
        assert!(angle_deg(r.normal, scene.normal) < 0.5);
    }
    assert_eq!(out[0].cell_rays[1], 5);
    assert_eq!(out[1].cell_rays[1], 4);
}

#[test]
fn refused_outlier_cells_enter_only_on_request() {
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let mut members = planar_members(&scene);
    members.cells[1][2] = ClusterCellStatus::RefusedOutlier;
    let poses = scene.cam_from_world();
    let off = run(
        &scene,
        &[&members],
        &[0],
        &poses,
        &CellPlaneParams::default(),
    );
    let on = run(
        &scene,
        &[&members],
        &[0],
        &poses,
        &CellPlaneParams {
            include_refused_outlier: true,
            ..CellPlaneParams::default()
        },
    );
    assert_eq!(off[0].cell_rays[2], 4);
    assert_eq!(on[0].cell_rays[2], 5);
}

#[test]
fn a_narrow_baseline_is_not_triangulated() {
    let centres = vec![
        Vector3::new(0.5, 0.3, 10.0),
        Vector3::new(0.6, 0.3, 10.0),
        Vector3::new(0.5, 0.4, 10.0),
    ];
    let scene = Scene::new(tilted_normal(), &centres);
    let members = planar_members(&scene);
    let out = run(
        &scene,
        &[&members],
        &[0],
        &scene.cam_from_world(),
        &CellPlaneParams::default(),
    );
    assert!(out[0]
        .cell_status
        .iter()
        .all(|&s| s == CellPlaneStatus::NarrowBaseline));
    assert_eq!(out[0].determinacy, NormalDeterminacy::None);
}

#[test]
fn an_off_plane_cell_is_refused_by_the_plane_fit() {
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let mut members = planar_members(&scene);
    // Cell 8's content in every member is displaced as a point well off the
    // plane would be: shift it by a consistent parallax.
    let off = Scene::new(tilted_normal(), &ring_centres());
    let step = PATCH_SIZE / RESOLUTION as f64;
    let centres = cell_centres(RESOLUTION);
    let r0 = &members;
    let x_ref = {
        let s = r0.shapes[0];
        let p = r0.positions[0];
        let u = centres[8];
        Vector2::new(
            f64::from(p[0]) + step * (f64::from(s[0][0]) * u[0] + f64::from(s[0][1]) * u[1]),
            f64::from(p[1]) + step * (f64::from(s[1][0]) * u[0] + f64::from(s[1][1]) * u[1]),
        )
    };
    // The reference ray's point, lifted 0.1 units toward the camera.
    let x_true = off.on_plane(0, x_ref);
    let lifted = x_true + 0.1 * (Vector3::new(0.5, 0.3, 10.0) - x_true).normalize();
    let mut shifted = Vec::new();
    for k in 1..members.images.len() {
        let target = off.project(k, lifted);
        let s = members.shapes[k];
        let p = members.positions[k];
        let sm = Matrix2::new(
            f64::from(s[0][0]),
            f64::from(s[0][1]),
            f64::from(s[1][0]),
            f64::from(s[1][1]),
        );
        let g = (step * sm).try_inverse().unwrap()
            * (target - Vector2::new(f64::from(p[0]), f64::from(p[1])));
        shifted.push([(g.x - centres[8][0]) as f32, (g.y - centres[8][1]) as f32]);
    }
    for (k, d) in shifted.into_iter().enumerate() {
        members.shifts[k + 1][8] = d;
    }
    let out = run(
        &scene,
        &[&members],
        &[0],
        &scene.cam_from_world(),
        &CellPlaneParams::default(),
    );
    let r = &out[0];
    assert_eq!(r.cell_status[8], CellPlaneStatus::PlaneOutlier);
    assert_eq!(r.determinacy, NormalDeterminacy::BothAxes);
    assert!(angle_deg(r.normal, scene.normal) < 0.5);
}

#[test]
fn the_answer_does_not_depend_on_the_thread_count() {
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let clusters: Vec<Members> = (0..16).map(|_| planar_members(&scene)).collect();
    let refs: Vec<&Members> = clusters.iter().collect();
    let per = clusters[0].images.len() as u32;
    let references: Vec<u32> = (0..16).map(|c| c * per).collect();
    let poses = scene.cam_from_world();
    let go = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                run(
                    &scene,
                    &refs,
                    &references,
                    &poses,
                    &CellPlaneParams::default(),
                )
            })
    };
    assert_eq!(go(1), go(4));
}

#[test]
fn two_ray_cells_at_the_floor_count_toward_the_verdict() {
    // Cell 4 is fitted in every member; cells 0, 2 and 7 only in the member of
    // image 1, so each of them is triangulated from two rays with no residual
    // and sits at the noise floor. Their precision is far below cell 4's, but
    // the robust fit keeps them, and so they decide the verdict with it.
    let scene = Scene::new(tilted_normal(), &ring_centres());
    let mut members = planar_members(&scene);
    for (k, cells) in members.cells.iter_mut().enumerate().skip(1) {
        for (j, c) in cells.iter_mut().enumerate() {
            let two_ray = [0, 2, 7].contains(&j) && k == 1;
            if j != 4 && !two_ray {
                *c = ClusterCellStatus::RefusedZncc;
            }
        }
    }
    let out = run(
        &scene,
        &[&members],
        &[0],
        &scene.cam_from_world(),
        &CellPlaneParams::default(),
    );
    let r = &out[0];
    assert_eq!(r.cell_rays[4], 5);
    let max_w = r.cell_weight[4];
    for j in [0, 2, 7] {
        assert_eq!(r.cell_rays[j], 2);
        assert_eq!(r.cell_status[j], CellPlaneStatus::InPlane);
        assert!(r.cell_residual_px[j] < 0.05, "cell {j} off the floor");
        // Below a quarter of the largest weight: a live set chosen by the
        // precision weight would have dropped these cells.
        assert!(r.cell_weight[j] < 0.25 * max_w, "cell {j} weight too high");
    }
    assert_eq!(r.determinacy, NormalDeterminacy::BothAxes);
    let err = angle_deg(r.normal, scene.normal);
    assert!(err < 0.5, "normal off by {err}°");
}

#[test]
fn the_ray_floor_follows_the_shape_and_the_grid_step() {
    let shape = [[3.0f32, 0.0], [0.0, 4.0]];
    assert!((shape_gain(&shape) - 4.0).abs() < 1e-12);
    let rotated = [[0.0f32, -2.0], [2.0, 0.0]];
    assert!((shape_gain(&rotated) - 2.0).abs() < 1e-6);
}
