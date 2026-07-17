// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::WarpMap;
use crate::camera::{CameraIntrinsics, CameraModel};
use nalgebra::{Point3, UnitQuaternion, Vector3};

fn pinhole(f: f64, cx: f64, cy: f64, w: u32, h: u32) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: f,
            focal_length_y: f,
            principal_point_x: cx,
            principal_point_y: cy,
        },
        width: w,
        height: h,
    }
}

/// Source→grid Jacobian at the centre pixel of an `r×r` warp, from the
/// 4-neighbours of pixel `(1, 1)`: `[dx_dcol, dy_dcol, dx_drow, dy_drow]`.
fn center_jacobian(wm: &WarpMap) -> [f64; 4] {
    let (xl, yl) = wm.get(0, 1);
    let (xr, yr) = wm.get(2, 1);
    let (xt, yt) = wm.get(1, 0);
    let (xb, yb) = wm.get(1, 2);
    [
        (xr - xl) as f64, // dx_dcol
        (yr - yl) as f64, // dy_dcol
        (xb - xt) as f64, // dx_drow
        (yb - yt) as f64, // dy_drow
    ]
}

#[test]
fn from_center_normal_render_is_upright() {
    // Regression (patch orientation): a patch built via `from_center_normal`
    // with the observing camera's up as `up_hint` must render UPRIGHT, not
    // rotated 90°. Canonical camera at origin looking down −Z (identity pose);
    // a patch in front sits at −Z, facing the camera. camera up = +Y, right = +X.
    // Upright means +col → +source_x (right) and +row → +source_y (down), i.e.
    // the Jacobian is a positive diagonal with ~zero off-diagonals.
    let (f, cx, cy) = (500.0, 320.0, 240.0);
    let cam = pinhole(f, cx, cy, 640, 480);
    let pose = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let patch = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(0.0, 0.0, 1.0), // outward normal toward the camera (+Z)
        Vector3::new(0.0, 1.0, 0.0), // up_hint = camera up (+Y)
        [0.5, 0.5],
    );
    let wm = WarpMap::from_patch(&patch, &cam, &pose, 3);
    let [dx_dcol, dy_dcol, dx_drow, dy_drow] = center_jacobian(&wm);
    assert!(
        dx_dcol > 0.0 && dy_drow > 0.0 && dx_drow.abs() < 1.0 && dy_dcol.abs() < 1.0,
        "expected upright render; J = [[{dx_dcol:.2}, {dx_drow:.2}], [{dy_dcol:.2}, {dy_drow:.2}]]"
    );
}

#[test]
fn repose_preserves_upright_orientation() {
    // Reposing a patch onto a new (still camera-facing) normal must keep it
    // upright — the in-plane orientation is preserved across the normal change,
    // it does not accumulate a 90° turn per repose. Uses the public
    // `from_center_normal`/`normal` surface (repose is crate-internal): rebuild
    // the frame from the patch's own `v_axis`, exactly as `repose_patch` does.
    let (f, cx, cy) = (500.0, 320.0, 240.0);
    let cam = pinhole(f, cx, cy, 640, 480);
    let pose = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let base = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.5, 0.5],
    );
    // A slightly tilted, still camera-facing normal.
    let tilted = Vector3::new(0.1, -0.05, 1.0).normalize();
    let reposed =
        OrientedPatch::from_center_normal(base.center, tilted, base.v_axis, base.half_extent);
    let wm = WarpMap::from_patch(&reposed, &cam, &pose, 3);
    let [dx_dcol, dy_dcol, dx_drow, dy_drow] = center_jacobian(&wm);
    assert!(
        dx_dcol > 0.0 && dy_drow > 0.0 && dx_drow.abs() < dx_dcol && dy_dcol.abs() < dy_drow,
        "reposed patch must stay upright; J = [[{dx_dcol:.2}, {dx_drow:.2}], [{dy_dcol:.2}, {dy_drow:.2}]]"
    );
}

#[test]
fn to_world_and_normal() {
    let p = OrientedPatch::new(
        Point3::new(1.0, 2.0, 3.0),
        Vector3::x(),
        Vector3::y(),
        [2.0, 4.0],
    );
    assert!((p.to_world(0.0, 0.0) - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
    assert!((p.to_world(1.0, 1.0) - Point3::new(3.0, 6.0, 3.0)).norm() < 1e-12);
    // normal = u_axis × v_axis = x̂ × ŷ = ẑ (right-handed frame).
    assert!((p.normal() - Vector3::z()).norm() < 1e-12);
}

#[test]
fn from_center_normal_is_orthonormal_and_preserves_normal() {
    let normal = Vector3::new(0.3, -0.5, 0.8).normalize();
    let p = OrientedPatch::from_center_normal(Point3::origin(), normal, Vector3::y(), [1.0, 1.0]);
    assert!((p.u_axis.norm() - 1.0).abs() < 1e-9);
    assert!((p.v_axis.norm() - 1.0).abs() < 1e-9);
    assert!(p.u_axis.dot(&p.v_axis).abs() < 1e-9);
    assert!((p.normal() - normal).norm() < 1e-9);
}

#[test]
fn from_center_normal_handles_up_hint_parallel_to_normal() {
    let normal = Vector3::z();
    let p = OrientedPatch::from_center_normal(Point3::origin(), normal, Vector3::z(), [1.0, 1.0]);
    assert!(p.u_axis.dot(&p.v_axis).abs() < 1e-9);
    assert!((p.normal() - normal).norm() < 1e-9);
}

#[test]
fn is_front_facing_uses_outward_normal() {
    // Camera at origin looking down world +Z: the canonical (−Z-forward)
    // camera rotated 180° about X.
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let front = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 5.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::y(),
        [1.0, 1.0],
    );
    assert!(front.is_front_facing(&pose));
    let back = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 5.0),
        Vector3::z(),
        Vector3::y(),
        [1.0, 1.0],
    );
    assert!(!back.is_front_facing(&pose));
}

#[test]
fn from_patch_projects_fronto_parallel_plane() {
    let (f, cx, cy) = (500.0, 320.0, 240.0);
    let cam = pinhole(f, cx, cy, 640, 480);
    // Camera at origin looking down world +Z (canonical camera rotated 180°
    // about X), so the plane at +z is in front.
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let (d, h) = (4.0, 0.5);
    let patch = OrientedPatch::new(Point3::new(0.0, 0.0, d), Vector3::x(), Vector3::y(), [h, h]);
    let r = 2u32;
    let wm = WarpMap::from_patch(&patch, &cam, &pose, r);

    let step = 2.0 / r as f64;
    for col in 0..r {
        for row in 0..r {
            let s = (col as f64 + 0.5) * step - 1.0;
            let t = (row as f64 + 0.5) * step - 1.0;
            // Columns run with +u_axis (+x); rows run with −v_axis (−y), since
            // the raster reverses `v` to render un-mirrored.
            let exp_x = f * (s * h) / d + cx;
            let exp_y = f * (-t * h) / d + cy;
            let (gx, gy) = wm.get(col, row);
            assert!((gx as f64 - exp_x).abs() < 1e-3, "x {gx} vs {exp_x}");
            assert!((gy as f64 - exp_y).abs() < 1e-3, "y {gy} vs {exp_y}");
        }
    }
}

#[test]
fn from_center_normal_render_is_not_mirrored() {
    // Regression: a fronto-parallel patch built via `from_center_normal` (normal
    // toward the camera) must render un-mirrored — the source→grid map must be a
    // proper (orientation-preserving) map, i.e. its Jacobian determinant is
    // positive. The frame is right-handed (`u = v × n`, outward normal `u × v`),
    // so `WarpMap::from_patch` reverses `v` for the raster row; walking `+v`
    // instead would reflect (negative determinant) and bake mirror-imaged
    // `inspect --strips` tiles and patch bitmaps.
    let (f, cx, cy) = (500.0, 320.0, 240.0);
    let cam = pinhole(f, cx, cy, 640, 480);
    // Camera at origin looking down world +Z (canonical camera rotated 180°
    // about X).
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let patch = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 4.0),
        Vector3::new(0.0, 0.0, -1.0), // outward normal toward the camera
        Vector3::new(0.0, -1.0, 0.0), // world "up on screen"
        [0.5, 0.5],
    );
    // The outward normal is preserved regardless of the in-plane handedness.
    assert!((patch.normal() - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-9);

    let r = 3u32;
    let wm = WarpMap::from_patch(&patch, &cam, &pose, r);
    // Discrete source→grid Jacobian at the centre pixel from its 4-neighbours.
    let (xl, yl) = wm.get(0, 1);
    let (xr, yr) = wm.get(2, 1);
    let (xt, yt) = wm.get(1, 0);
    let (xb, yb) = wm.get(1, 2);
    let dx_dcol = (xr - xl) as f64;
    let dy_dcol = (yr - yl) as f64;
    let dx_drow = (xb - xt) as f64;
    let dy_drow = (yb - yt) as f64;
    let det = dx_dcol * dy_drow - dx_drow * dy_dcol;
    assert!(
        det > 0.0,
        "render must be orientation-preserving (not mirrored); det = {det}",
    );
}

#[test]
fn mean_viewing_normal_points_at_camera_cluster() {
    let center = Point3::origin();
    // Cameras symmetric about the +Z axis -> mean viewing direction is +Z.
    let cams = [
        Point3::new(1.0, 0.0, 2.0),
        Point3::new(-1.0, 0.0, 2.0),
        Point3::new(0.0, 1.0, 2.0),
        Point3::new(0.0, -1.0, 2.0),
    ];
    let n = mean_viewing_normal(&center, &cams);
    assert!((n - Vector3::z()).norm() < 1e-9);
}

#[test]
fn pca_plane_normal_of_coplanar_points() {
    // Points spread in the z = 3 plane -> normal is +/- Z.
    let pts = [
        Point3::new(0.0, 0.0, 3.0),
        Point3::new(1.0, 0.0, 3.0),
        Point3::new(0.0, 1.0, 3.0),
        Point3::new(1.0, 1.0, 3.0),
        Point3::new(-1.0, 0.5, 3.0),
    ];
    let n = pca_plane_normal(&pts);
    assert!(n.z.abs() > 1.0 - 1e-6);
    assert!(n.x.abs() < 1e-6 && n.y.abs() < 1e-6);
}

#[test]
fn reduce_across_views() {
    let base = [2.0, 5.0, 3.0, 9.0];
    assert_eq!(reduce(&mut base.to_vec(), ViewReduce::Min), 2.0);
    assert_eq!(reduce(&mut base.to_vec(), ViewReduce::Max), 9.0);
    assert!((reduce(&mut base.to_vec(), ViewReduce::Mean) - 4.75).abs() < 1e-12);
    // even count -> average of the two middle values ([2,3,5,9] -> (3+5)/2)
    assert_eq!(reduce(&mut base.to_vec(), ViewReduce::Median), 4.0);
    assert_eq!(reduce(&mut [3.0, 1.0, 2.0], ViewReduce::Median), 2.0);
}

#[test]
fn feature_size_without_sift_is_an_error() {
    // The demo reconstruction has no workspace `.sift` files, so no keypoint
    // scale is readable for any point. FeatureSize must error rather than fall
    // back to a substitute size; the other extent policies still succeed.
    let recon = SfmrReconstruction::demo(12);

    let err = PatchCloud::from_reconstruction(
        &recon,
        PatchNormal::MeanViewing,
        PatchExtent::FeatureSize {
            factor: 5.0,
            across: ViewReduce::Median,
        },
        true,
    )
    .unwrap_err();
    // Every observation fails for the unreadable-scale reason (no `.sift`); none
    // for the coincident-camera reason, since the demo points are not on top of
    // their cameras.
    let PatchCloudError::MissingFeatureScale {
        observations,
        unreadable_scale,
        coincident_with_camera,
        ..
    } = err;
    assert!(observations > 0);
    assert_eq!(unreadable_scale, observations);
    assert_eq!(coincident_with_camera, 0);

    let cloud = PatchCloud::from_reconstruction(
        &recon,
        PatchNormal::MeanViewing,
        PatchExtent::Fixed(0.1),
        true,
    )
    .expect("Fixed extent needs no sift files");
    assert_eq!(cloud.len(), 12);
}

#[test]
fn from_patch_behind_camera_is_invalid() {
    let cam = pinhole(500.0, 320.0, 240.0, 640, 480);
    // Identity pose: the canonical camera looks down −Z, so a patch at +z is
    // behind it.
    let pose = RigidTransform::identity();
    let patch = OrientedPatch::new(
        Point3::new(0.0, 0.0, 4.0),
        Vector3::x(),
        Vector3::y(),
        [0.5, 0.5],
    );
    let wm = WarpMap::from_patch(&patch, &cam, &pose, 4);
    assert!(!wm.is_valid(0, 0));
}

#[test]
fn from_patch_infinity_ignores_camera_translation() {
    // A point at infinity in the −Z direction (in front of identity-pose
    // canonical cameras); its corners are directions, so its projection is
    // translation-invariant (every viewing ray is parallel to d). Rendering it
    // must rotate the corners without translating.
    let cam = pinhole(500.0, 320.0, 240.0, 640, 480);
    let patch = OrientedPatch::from_infinity_direction(
        Point3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.02, 0.02],
    );
    assert_eq!(patch.w, 0.0);
    // Outward normal faces back toward the observers: normalize(-d).
    assert!((patch.normal() - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-9);

    let at_origin = RigidTransform::identity();
    let translated = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [12.0, -7.0, 4.0]);
    let a = WarpMap::from_patch(&patch, &cam, &at_origin, 6);
    let b = WarpMap::from_patch(&patch, &cam, &translated, 6);
    for row in 0..6 {
        for col in 0..6 {
            assert_eq!(a.is_valid(col, row), b.is_valid(col, row));
            if a.is_valid(col, row) {
                let (ax, ay) = a.get(col, row);
                let (bx, by) = b.get(col, row);
                assert!((ax - bx).abs() < 1e-3 && (ay - by).abs() < 1e-3);
            }
        }
    }
    // The patch is in view and lands near the principal point.
    assert!(a.is_valid(3, 3));
    let (px, py) = a.get(3, 3);
    assert!((px - 320.0).abs() < 5.0 && (py - 240.0).abs() < 5.0);
}

// ---------------------------------------------------------------------------
// `PatchCloud::from_tracks` (the array-fed counterpart of `from_reconstruction`)
// ---------------------------------------------------------------------------

/// The per-image `cam_from_world` quaternions / translations / focal lengths of a
/// reconstruction, in the layout `from_tracks` consumes.
fn scene_arrays(
    recon: &SfmrReconstruction,
) -> (Vec<UnitQuaternion<f64>>, Vec<Vector3<f64>>, Vec<f64>) {
    let quats = recon.images.iter().map(|im| im.quaternion_wxyz).collect();
    let trans = recon.images.iter().map(|im| im.translation_xyz).collect();
    let focals = recon
        .images
        .iter()
        .map(|im| recon.cameras[im.camera_index as usize].focal_lengths().0)
        .collect();
    (quats, trans, focals)
}

/// Assert two clouds are patch-for-patch identical (point ids + frames + weight).
fn assert_clouds_equal(a: &PatchCloud, b: &PatchCloud) {
    assert_eq!(a.len(), b.len(), "patch counts differ");
    assert_eq!(a.point_indexes, b.point_indexes, "point_indexes differ");
    for i in 0..a.len() {
        let (pa, pb) = (a.patch(i), b.patch(i));
        assert!((pa.center - pb.center).norm() < 1e-12, "center {i}");
        assert!((pa.u_axis - pb.u_axis).norm() < 1e-12, "u_axis {i}");
        assert!((pa.v_axis - pb.v_axis).norm() < 1e-12, "v_axis {i}");
        assert!(
            (pa.half_extent[0] - pb.half_extent[0]).abs() < 1e-12,
            "half_u {i}"
        );
        assert!(
            (pa.half_extent[1] - pb.half_extent[1]).abs() < 1e-12,
            "half_v {i}"
        );
        assert_eq!(pa.w, pb.w, "w {i}");
    }
}

/// Write a `.sift` file per image of the demo recon under a fresh workspace, with
/// each feature's affine column-0 norm set to a distinct `σ(image, feature)`, and
/// point the recon's workspace/prefix at it. Returns the per-observation scale
/// vector (parallel to `recon.tracks`) that `from_reconstruction` will read back,
/// ready to hand to `from_tracks`.
fn write_demo_sift(recon: &mut SfmrReconstruction, tag: &str) -> Vec<f64> {
    use ndarray::{Array2, Array3};

    // Distinct, positive per-observation scale.
    let sigma = |img: usize, feat: usize| 2.0 + 0.37 * img as f64 + 0.11 * feat as f64;

    let dir = std::env::temp_dir().join(format!("patch_from_tracks_{tag}_{}", std::process::id()));
    let features_dir = dir.join("features");
    std::fs::create_dir_all(&features_dir).unwrap();
    recon.workspace_dir = dir.clone();
    recon.metadata.workspace.contents.feature_prefix_dir = "features".into();

    for img in 0..recon.images.len() {
        let count = recon.max_track_feature_index[img] as usize + 1;
        let mut affine = Array3::<f32>::zeros((count, 2, 2));
        for f in 0..count {
            let s = sigma(img, f) as f32;
            affine[[f, 0, 0]] = s; // column-0 norm = s (a10 = 0)
            affine[[f, 1, 1]] = s;
        }
        let data = sift_format::SiftData {
            feature_tool_metadata: sift_format::FeatureToolMetadata {
                feature_tool: "test".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
            },
            metadata: sift_format::SiftMetadata {
                version: sift_format::SIFT_FORMAT_VERSION,
                image_name: recon.images[img].name.clone(),
                image_file_xxh128: "0".repeat(32),
                image_file_size: 1,
                image_width: recon.cameras[0].width,
                image_height: recon.cameras[0].height,
                feature_count: count as u32,
            },
            content_hash: sift_format::SiftContentHash::default(),
            positions_xy: Array2::<f32>::zeros((count, 2)),
            affine_shapes: affine,
            descriptors: Array2::<u8>::zeros((count, 128)),
            thumbnail_y_x_rgb: Array3::<u8>::zeros((128, 128, 3)),
        };
        sift_format::write_sift(&recon.sift_path_for_image(img), &data, 3).unwrap();
    }

    // Per-observation scale in track order, matching what `from_reconstruction`
    // resolves through feature_index → image `.sift`.
    // The scale is stored (and read back) as f32, so round-trip through f32 to
    // match exactly what `read_image_scales` recovers (`a00 as f64`).
    let feats = recon.feature_indexes().unwrap().to_vec();
    recon
        .tracks
        .iter()
        .enumerate()
        .map(|(j, o)| sigma(o.image_index as usize, feats[j] as usize) as f32 as f64)
        .collect()
}

#[test]
fn from_tracks_reproduces_from_reconstruction_feature_size() {
    // The headline equivalence: `from_tracks` fed the same scales the `.sift`
    // files carry reproduces `from_reconstruction` patch-for-patch under the
    // scale-reading FeatureSize policy.
    let mut recon = SfmrReconstruction::demo(12);
    let obs_scales = write_demo_sift(&mut recon, "eq");

    let extent = PatchExtent::FeatureSize {
        factor: 5.0,
        across: ViewReduce::Median,
    };
    let from_recon =
        PatchCloud::from_reconstruction(&recon, PatchNormal::MeanViewing, extent, false).unwrap();

    let positions: Vec<Point3<f64>> = recon.points.iter().map(|p| p.position).collect();
    let weights: Vec<f64> = recon.points.iter().map(|p| p.w).collect();
    let obs_images: Vec<u32> = recon.tracks.iter().map(|o| o.image_index).collect();
    let (quats, trans, focals) = scene_arrays(&recon);
    let from_arrays = PatchCloud::from_tracks(
        &positions,
        &weights,
        None,
        &recon.observation_offsets,
        &obs_images,
        Some(&obs_scales),
        &quats,
        &trans,
        &focals,
        PatchNormal::MeanViewing,
        extent,
        false,
    )
    .unwrap();

    assert_clouds_equal(&from_recon, &from_arrays);
    // FeatureSize actually sized the patches (non-trivial equivalence).
    assert!(from_arrays.len() == 12 && from_arrays.patch(0).half_extent[0] > 0.0);

    std::fs::remove_dir_all(&recon.workspace_dir).ok();
}

#[test]
fn from_tracks_matches_reconstruction_pixel_radius_and_stored_normal() {
    // The non-`.sift` policies match too: PixelRadius extent + Stored normals
    // (fed as an array) reproduce `from_reconstruction` on the same geometry.
    let recon = SfmrReconstruction::demo(9);
    let extent = PatchExtent::PixelRadius {
        radius_px: 4.0,
        across: ViewReduce::Min,
    };
    let from_recon =
        PatchCloud::from_reconstruction(&recon, PatchNormal::Stored, extent, false).unwrap();

    let positions: Vec<Point3<f64>> = recon.points.iter().map(|p| p.position).collect();
    let weights: Vec<f64> = recon.points.iter().map(|p| p.w).collect();
    let stored: Vec<Vector3<f64>> = recon
        .points
        .iter()
        .map(|p| Vector3::new(p.normal.x as f64, p.normal.y as f64, p.normal.z as f64))
        .collect();
    let obs_images: Vec<u32> = recon.tracks.iter().map(|o| o.image_index).collect();
    let (quats, trans, focals) = scene_arrays(&recon);
    let from_arrays = PatchCloud::from_tracks(
        &positions,
        &weights,
        Some(&stored),
        &recon.observation_offsets,
        &obs_images,
        None,
        &quats,
        &trans,
        &focals,
        PatchNormal::Stored,
        extent,
        false,
    )
    .unwrap();
    assert_clouds_equal(&from_recon, &from_arrays);
}

#[test]
fn from_tracks_nan_scale_counts_as_unreadable() {
    // Two points, each with one observation. A NaN scale entry is treated exactly
    // like an unreadable `.sift` scale, so FeatureSize errors with the same
    // MissingFeatureScale taxonomy (all observations unreadable, none coincident).
    let positions = vec![Point3::new(0.0, 0.0, 3.0), Point3::new(1.0, 0.0, 3.0)];
    let weights = vec![1.0, 1.0];
    let obs_offsets = vec![0usize, 1, 2];
    let obs_images = vec![0u32, 0];
    let obs_scales = vec![f64::NAN, f64::NAN];
    let quats = vec![UnitQuaternion::identity()];
    // Camera at origin looking down −Z; points at +z=3 are a distance 3 away.
    let trans = vec![Vector3::zeros()];
    let focals = vec![500.0];

    let err = PatchCloud::from_tracks(
        &positions,
        &weights,
        None,
        &obs_offsets,
        &obs_images,
        Some(&obs_scales),
        &quats,
        &trans,
        &focals,
        PatchNormal::MeanViewing,
        PatchExtent::FeatureSize {
            factor: 5.0,
            across: ViewReduce::Median,
        },
        false,
    )
    .unwrap_err();
    let PatchCloudError::MissingFeatureScale {
        point_index,
        observations,
        unreadable_scale,
        coincident_with_camera,
    } = err;
    assert_eq!(point_index, 0);
    assert_eq!(observations, 1);
    assert_eq!(unreadable_scale, 1);
    assert_eq!(coincident_with_camera, 0);

    // A readable (finite) scale for the same geometry sizes the patch fine.
    let ok = PatchCloud::from_tracks(
        &positions,
        &weights,
        None,
        &obs_offsets,
        &obs_images,
        Some(&[3.0, 3.0]),
        &quats,
        &trans,
        &focals,
        PatchNormal::MeanViewing,
        PatchExtent::FeatureSize {
            factor: 5.0,
            across: ViewReduce::Median,
        },
        false,
    )
    .unwrap();
    assert_eq!(ok.len(), 2);
    // half = factor * σ * d / f = 5 * 3 * 3 / 500.
    assert!((ok.patch(0).half_extent[0] - 5.0 * 3.0 * 3.0 / 500.0).abs() < 1e-9);
}

#[test]
fn from_tracks_builds_infinity_tangent_frames() {
    // A finite point and a point at infinity (w = 0). Under
    // exclude_points_at_infinity = false the infinity row gets a tangent-sphere
    // frame (w = 0, normal = normalize(-d)); the finite one stays w = 1. Fixed
    // extent needs no scales.
    let positions = vec![
        Point3::new(0.0, 0.0, -3.0), // finite, in front of the camera
        Point3::new(0.0, 0.0, -1.0), // direction (at infinity)
    ];
    let weights = vec![1.0, 0.0];
    let obs_offsets = vec![0usize, 1, 2];
    let obs_images = vec![0u32, 0];
    let quats = vec![UnitQuaternion::identity()];
    let trans = vec![Vector3::zeros()];
    let focals = vec![500.0];

    let cloud = PatchCloud::from_tracks(
        &positions,
        &weights,
        None,
        &obs_offsets,
        &obs_images,
        None,
        &quats,
        &trans,
        &focals,
        PatchNormal::MeanViewing,
        PatchExtent::Fixed(0.1),
        false,
    )
    .unwrap();
    assert_eq!(cloud.len(), 2);
    // Finite patch first (point 0), then the infinity patch (point 1).
    assert_eq!(cloud.point_indexes, vec![0, 1]);
    assert_eq!(cloud.patch(0).w, 1.0);
    assert_eq!(cloud.patch(1).w, 0.0);
    // Infinity normal faces back toward the observers: normalize(-d) with d = -Z.
    assert!((cloud.patch(1).normal() - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-9);

    // exclude_points_at_infinity = true drops the infinity row.
    let finite_only = PatchCloud::from_tracks(
        &positions,
        &weights,
        None,
        &obs_offsets,
        &obs_images,
        None,
        &quats,
        &trans,
        &focals,
        PatchNormal::MeanViewing,
        PatchExtent::Fixed(0.1),
        true,
    )
    .unwrap();
    assert_eq!(finite_only.len(), 1);
    assert_eq!(finite_only.point_indexes, vec![0]);
}
