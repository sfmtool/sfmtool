// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Camera Intrinsics panel.
//!
//! egui lays a frame out with no GPU, so the panel really runs — the tables are
//! really built, the grids really allocate, and the strings the panel decided
//! to paint come back out of the frame's galleys
//! (`test_support::painted_texts`, the `track_view/body/tests.rs` pattern).
//! The assertions target what the panel *says* rather than pixels.
//!
//! The numbers themselves are asserted where they are computed — `P` against
//! the camera's own forward projection, the transformed pose against
//! `Se3Transform::apply_to_camera_pose` — because those are the two places a
//! sign error would produce a plausible-looking matrix that is silently wrong.

use nalgebra::Vector3;
use ndarray::{Array1, Array2};
use sfmtool_core::camera::{CameraIntrinsics, CameraModel};
use sfmtool_core::{
    FramesMetadata, RigDefinition, RigFrameData, RigsMetadata, RotQuaternion, Se3Transform,
    SfmrReconstruction,
};

use super::{extrinsics::Pose, IntrinsicsDetail, PoseFrame};
use crate::scene::SceneNode;

/// Tall enough that nothing the panel draws is scrolled out of the frame:
/// `painted_texts` only sees what was laid out, and a clipped table would make
/// every "the panel says X" assertion vacuous.
const VIEWPORT: egui::Vec2 = egui::vec2(560.0, 2400.0);

// ── Fixtures ────────────────────────────────────────────────────────────

/// A node whose single camera is the demo's 1920×1080 pinhole.
fn pinhole_node() -> SceneNode {
    SceneNode::from_path(
        std::path::Path::new("/runs/demo.sfmr"),
        SfmrReconstruction::demo(32),
    )
}

/// `kerry_park`'s real intrinsics: an `OPENCV_FISHEYE` on a 480×480 frame.
fn kerry_park_camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 129.1499937015594,
            focal_length_y: 129.2573627423474,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.038113353966529886,
            radial_distortion_k2: -0.00800851799065643,
            radial_distortion_k3: 0.008329720504707577,
            radial_distortion_k4: -0.0026901578801066814,
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    }
}

/// The demo node with its camera swapped for a fisheye — the model family with
/// no `P` and a focal length in pixels per radian.
fn fisheye_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon_mut().image_set.cameras[0] = kerry_park_camera();
    node
}

/// A node with two cameras: the demo pinhole for the first four images and a
/// second, single-focal-length model for the rest.
fn two_camera_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon_mut().image_set.cameras.push(CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 344.0,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.035,
        },
        width: 270,
        height: 480,
        rolling_shutter: None,
    });
    for image in node.recon_mut().image_set.images.iter_mut().skip(4) {
        image.camera_index = 1;
    }
    node
}

/// The demo node under a similarity with a rotation, a translation and a scale
/// change, so a transform that is only partly applied cannot pass.
fn transformed_node() -> SceneNode {
    let mut node = pinhole_node();
    *node.history.transform_mut() = Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.3, -0.7, 0.6), 0.9).unwrap(),
        Vector3::new(4.0, -2.5, 1.25),
        2.0,
    );
    node
}

/// The demo node as a two-sensor rig: `left` is the reference with an identity
/// `sensor_from_rig`, `right` is offset along X. Four frames of two images.
fn rig_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon_mut().image_set.images.truncate(8);
    let images = node.recon_mut().image_set.images.len();
    node.recon_mut().image_set.rig_frame_data = Some(RigFrameData {
        rigs_metadata: RigsMetadata {
            rig_count: 1,
            sensor_count: 2,
            rigs: vec![RigDefinition {
                name: "kerry_rig".to_string(),
                sensor_count: 2,
                sensor_offset: 0,
                ref_sensor_name: "left".to_string(),
                sensor_names: vec!["left".to_string(), "right".to_string()],
            }],
        },
        sensor_camera_indexes: Array1::from_vec(vec![0u32, 0]),
        sensor_quaternions_wxyz: Array2::from_shape_vec(
            (2, 4),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        .unwrap(),
        sensor_translations_xyz: Array2::from_shape_vec(
            (2, 3),
            vec![0.0, 0.0, 0.0, 0.0307, 0.0, 0.0],
        )
        .unwrap(),
        frames_metadata: FramesMetadata {
            frame_count: (images / 2) as u32,
        },
        rig_indexes: Array1::from_vec(vec![0u32; images / 2]),
        image_sensor_indexes: Array1::from_vec((0..images as u32).map(|i| i % 2).collect()),
        image_frame_indexes: Array1::from_vec((0..images as u32).map(|i| i / 2).collect()),
    });
    node
}

/// Every string the panel painted in one frame.
fn texts(
    panel: &mut IntrinsicsDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    camera: Option<usize>,
    image: Option<usize>,
) -> Vec<String> {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        ..Default::default()
    };
    crate::test_support::painted_texts(ctx, input, |ui| {
        panel.show(ui, node, camera, image, None, None);
    })
}

/// One frame of the panel, with the strings it painted.
fn show(node: &SceneNode, camera: Option<usize>, image: Option<usize>) -> Vec<String> {
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    texts(&mut panel, &ctx, node, camera, image)
}

/// Whether any painted string contains `needle`.
fn says(texts: &[String], needle: &str) -> bool {
    texts.iter().any(|t| t.contains(needle))
}

// ── Empty state ─────────────────────────────────────────────────────────

#[test]
fn no_selection_shows_the_empty_state() {
    let painted = show(&pinhole_node(), None, None);
    assert!(says(&painted, "No camera selected"));
    assert!(says(&painted, "Select a camera under Camera Intrinsics"));
    // Nothing of the populated state leaks through.
    assert!(!says(&painted, "Parameters"));
}

#[test]
fn a_stale_camera_index_falls_back_to_the_empty_state() {
    let painted = show(&pinhole_node(), Some(99), None);
    assert!(says(&painted, "No camera selected"));
}

// ── Header and tables ───────────────────────────────────────────────────

#[test]
fn the_header_names_the_node_the_camera_and_the_model() {
    let painted = show(&pinhole_node(), Some(0), None);
    assert!(says(&painted, "demo"));
    assert!(says(&painted, "Camera #0"));
    assert!(says(&painted, "PINHOLE"));
    assert!(says(&painted, "1920×1080"));
    assert!(says(&painted, "8 images"));
    // No beta model, no marker.
    assert!(!says(&painted, "(beta)"));
}

#[test]
fn a_beta_model_marks_the_header() {
    let mut node = pinhole_node();
    node.recon_mut().image_set.cameras[0] = CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 129.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: 1.6,
            bspline: vec![0.0, 0.01, 0.02, 0.0],
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    let painted = show(&node, Some(0), None);
    assert!(says(&painted, "(beta)"));
}

/// The spline's coefficient count prints as an integer and its angular domain
/// end in degrees, as every other angle in the panel does, in the table and in
/// its copied text alike; the column keeps its width.
#[test]
fn the_spline_count_is_an_integer_and_its_domain_end_is_in_degrees() {
    let mut node = pinhole_node();
    node.recon_mut().image_set.cameras[0] = CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 129.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: 150f64.to_radians(),
            bspline: vec![0.0; 8],
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    let painted = show(&node, Some(0), None);
    let count = format!("{:>14}", 8);
    let domain = format!("{:>13}°", "150.000000");
    assert!(painted.contains(&count), "{painted:?}");
    assert!(painted.contains(&domain), "{painted:?}");
    assert!(!says(&painted, "8.000000"), "{painted:?}");
    assert!(!says(&painted, "2.617994"), "{painted:?}");
    assert_eq!(domain.chars().count(), 14);

    let text = super::header::parameters_text(&node.recon().image_set.cameras[0]);
    assert!(text
        .lines()
        .any(|l| l.ends_with(&count) && l.starts_with("bspline_coeff_count")));
    assert!(text
        .lines()
        .any(|l| l.ends_with(&domain) && l.starts_with("bspline_theta_max")));
}

#[test]
fn the_parameter_table_is_in_declaration_order() {
    // Asserted against `parameter_names()` rather than hard-coded, so a model
    // gaining a parameter cannot silently reorder the table — and so the panel
    // and `sfm inspect` cannot drift into two different orders.
    let node = fisheye_node();
    let painted = show(&node, Some(0), None);
    let declared: Vec<String> = node.recon().image_set.cameras[0]
        .model
        .parameter_names()
        .iter()
        .map(|n| n.to_string())
        .collect();
    assert!(declared.len() > 4);
    let painted_names: Vec<String> = painted
        .iter()
        .filter(|t| declared.contains(t))
        .cloned()
        .collect();
    assert_eq!(painted_names, declared);
}

#[test]
fn the_derived_table_labels_a_fisheye_focal_length_in_pixels_per_radian() {
    let painted = show(&fisheye_node(), Some(0), None);
    assert!(says(&painted, "px/rad"));
    // 35 mm equivalence is meaningless for a pixels-per-radian focal length,
    // so the row is absent rather than wrong.
    assert!(!says(&painted, "35 mm equivalent"));
}

#[test]
fn the_derived_table_labels_a_perspective_focal_length_in_pixels() {
    let painted = show(&pinhole_node(), Some(0), None);
    assert!(says(&painted, "1000.000, 1000.000 px"));
    assert!(!says(&painted, "px/rad"));
    assert!(says(&painted, "35 mm equivalent"));
}

#[test]
fn a_distortion_free_model_reports_no_distortion() {
    let painted = show(&pinhole_node(), Some(0), None);
    assert!(says(&painted, "none"));
    assert!(!says(&painted, "yes — max"));
}

#[test]
fn a_distorted_model_reports_its_largest_displacement() {
    let painted = show(&fisheye_node(), Some(0), None);
    assert!(says(&painted, "yes — max"));
}

/// The distortion row is bounded by the model's trustworthy domain rather than
/// by the image rectangle, and says which.
///
/// `kerry_park`'s fisheye is the case that forced this: over the whole
/// rectangle the maximum is 241 px, because the corners are 150° off-axis
/// where the `k1..k4` polynomial has folded. Inside the bound it is 12 px,
/// which is a lens.
#[test]
fn a_bounded_model_reports_its_displacement_inside_the_bound() {
    let camera = kerry_park_camera();
    let limit = sfmtool_core::camera::report::trustworthy_max_theta_deg(&camera).unwrap();
    let painted = show(&fisheye_node(), Some(0), None);

    assert!(says(&painted, &format!("inside {limit:.1}°")));
    // The unqualified phrasing is gone, and so is the number it used to carry.
    assert!(!says(&painted, "over the image"));
    assert!(!says(&painted, "241"));
}

/// A model that is trustworthy everywhere keeps the plain phrasing: the
/// qualifier is only earned when it actually excludes something.
#[test]
fn an_unbounded_model_reports_its_displacement_over_the_whole_image() {
    let painted = show(&two_camera_node(), Some(1), None);
    assert!(says(&painted, "yes — max"));
    assert!(says(&painted, "over the image"));
    assert!(!says(&painted, "inside "));
}

#[test]
fn the_aspect_row_is_hidden_for_a_single_focal_length_model() {
    // The demo pinhole carries `fx` and `fy` separately, so the row is there
    // even though they are equal; SIMPLE_RADIAL cannot express an aspect ratio
    // at all, so it is not.
    assert!(says(&show(&pinhole_node(), Some(0), None), "aspect fy/fx"));
    assert!(!says(
        &show(&two_camera_node(), Some(1), None),
        "aspect fy/fx"
    ));
}

#[test]
fn the_field_of_view_rows_are_the_cameras_own() {
    let painted = show(&fisheye_node(), Some(0), None);
    // kerry_park's fisheye: wide spans and a large corner angle. The exact
    // values are `camera::report`'s business; what this pins is that the panel
    // shows that camera's numbers rather than a placeholder.
    let fov = sfmtool_core::camera::report::field_of_view(&kerry_park_camera()).unwrap();
    assert!(fov.diagonal > 180.0);
    assert!(says(&painted, &format!("{:.1}°", fov.diagonal)));
    assert!(says(&painted, &format!("{:.1}°", fov.max_off_axis)));
}

// ── `K` and `P` ─────────────────────────────────────────────────────────

#[test]
fn k_carries_the_frame_note() {
    let painted = show(&pinhole_node(), Some(0), None);
    assert!(says(&painted, "P = K · S · [R|t]"));
    assert!(says(&painted, "S = diag(1, −1, −1)"));
}

#[test]
fn a_fisheye_replaces_the_p_row_with_a_statement() {
    let painted = show(&fisheye_node(), Some(0), Some(0));
    assert!(says(
        &painted,
        "Not a linear projection — this model has no 3×4 P"
    ));
    assert!(!says(&painted, "P = K · S · [ R | t ]"));
}

#[test]
fn a_perspective_model_shows_p() {
    let painted = show(&pinhole_node(), Some(0), Some(0));
    assert!(says(&painted, "P = K · S · [ R | t ]"));
    assert!(!says(&painted, "Not a linear projection"));
}

#[test]
fn p_is_k_times_s_times_rt_and_not_k_times_rt() {
    let node = pinhole_node();
    let camera = &node.recon().image_set.cameras[0];
    let pose = Pose::resolve(&node, 0, camera, PoseFrame::Stored);
    let p = pose.projection.expect("a pinhole has a P");

    // A point this image actually sees, projected two ways: through `P`, and
    // through the camera's own forward map. They have to agree — that is what
    // makes `P` worth pasting anywhere.
    let world = node.recon().point_set.points[0].position;
    let homogeneous = p * nalgebra::Vector4::new(world.x, world.y, world.z, 1.0);
    let through_p = (
        homogeneous[0] / homogeneous[2],
        homogeneous[1] / homogeneous[2],
    );
    let in_camera = pose.rotation * world.coords + pose.translation;
    let (u, v) = camera
        .ray_to_pixel([in_camera.x, in_camera.y, in_camera.z])
        .expect("the demo point is in front of the demo camera");
    assert!(
        (through_p.0 - u).abs() < 1e-9,
        "{through_p:?} vs ({u}, {v})"
    );
    assert!(
        (through_p.1 - v).abs() < 1e-9,
        "{through_p:?} vs ({u}, {v})"
    );

    // And the `S`-less spelling does not: `K · [R|t]` is the plausible-looking
    // matrix this whole row exists to avoid handing out. `S` mirrors the
    // camera's x axis, so the naive projection lands at the point's mirror
    // image about the principal point's column, exactly, for every point.
    // (A distance bound would let a point near that column slip through.)
    let mut rt = nalgebra::Matrix3x4::zeros();
    rt.fixed_view_mut::<3, 3>(0, 0).copy_from(&pose.rotation);
    rt.fixed_view_mut::<3, 1>(0, 3).copy_from(&pose.translation);
    let naive = camera.intrinsic_matrix() * rt;
    let naive = naive * nalgebra::Vector4::new(world.x, world.y, world.z, 1.0);
    let (cx, _) = camera.principal_point();
    let mirrored_u = 2.0 * cx - u;
    assert!(
        (naive[0] / naive[2] - mirrored_u).abs() < 1e-9,
        "naive u {} vs mirrored {mirrored_u}",
        naive[0] / naive[2]
    );
    assert!((naive[1] / naive[2] - v).abs() < 1e-9);
}

// ── Extrinsics ──────────────────────────────────────────────────────────

#[test]
fn the_extrinsics_block_appears_only_with_an_image_selected() {
    let node = pinhole_node();
    assert!(!says(&show(&node, Some(0), None), "Rotation R"));
    let with_image = show(&node, Some(0), Some(3));
    assert!(says(&with_image, "Rotation R"));
    assert!(says(&with_image, "Camera centre C"));
    assert!(says(&with_image, "Axes in world"));
    assert!(says(&with_image, &node.recon().image_set.images[3].name));
}

#[test]
fn an_image_taken_through_another_camera_gets_no_extrinsics_block() {
    // The selection coupling already rules this out; the panel does not lean
    // on that, because printing one lens's K beside another lens's pose is the
    // one mistake the block must not be able to make.
    let node = two_camera_node();
    assert!(!says(&show(&node, Some(0), Some(6)), "Rotation R"));
    assert!(says(&show(&node, Some(1), Some(6)), "Rotation R"));
}

#[test]
fn the_camera_centre_carries_the_world_space_unit_when_the_file_names_one() {
    let mut node = pinhole_node();
    assert!(says(&show(&node, Some(0), Some(0)), "Camera centre C"));
    node.recon_mut().metadata.world_space_unit = Some("m".to_string());
    assert!(says(&show(&node, Some(0), Some(0)), "Camera centre C (m)"));
}

// ── The node transform ──────────────────────────────────────────────────

#[test]
fn the_node_transform_toggle_appears_only_for_a_transformed_node() {
    assert!(!says(
        &show(&pinhole_node(), Some(0), Some(0)),
        "× node transform"
    ));
    assert!(says(
        &show(&transformed_node(), Some(0), Some(0)),
        "× node transform"
    ));
}

#[test]
fn the_stored_pose_is_the_default_and_the_marker_follows_the_frame() {
    let node = transformed_node();
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();

    let stored = texts(&mut panel, &ctx, &node, Some(0), Some(0));
    assert!(!says(&stored, "(transformed)"));

    panel.pose_frame = PoseFrame::NodeTransform;
    let moved = texts(&mut panel, &ctx, &node, Some(0), Some(0));
    assert!(says(&moved, "(transformed)"));
}

#[test]
fn the_transformed_pose_is_the_stored_pose_through_the_node_transform() {
    let node = transformed_node();
    let camera = &node.recon().image_set.cameras[0];
    let image = &node.recon().image_set.images[0];

    let stored = Pose::resolve(&node, 0, camera, PoseFrame::Stored);
    let moved = Pose::resolve(&node, 0, camera, PoseFrame::NodeTransform);
    assert!(!moved.transformed || stored.translation != moved.translation);

    let (expected_q, expected_t) = node.transform().apply_to_camera_pose(
        &RotQuaternion::from_nalgebra(image.quaternion_wxyz),
        &image.translation_xyz,
    );
    assert!((moved.translation - expected_t).norm() < 1e-12);
    assert!((moved.rotation - expected_q.to_rotation_matrix()).norm() < 1e-12);
    // And the camera centre follows the transform's own action on a point.
    let expected_centre = node.transform().apply_to_point(&stored.centre);
    assert!((moved.centre - expected_centre).norm() < 1e-9);
}

#[test]
fn an_untransformed_node_ignores_the_transformed_frame() {
    let node = pinhole_node();
    let camera = &node.recon().image_set.cameras[0];
    let stored = Pose::resolve(&node, 0, camera, PoseFrame::Stored);
    let asked = Pose::resolve(&node, 0, camera, PoseFrame::NodeTransform);
    assert!(!asked.transformed);
    assert_eq!(stored.translation, asked.translation);
}

// ── Rigs ────────────────────────────────────────────────────────────────

#[test]
fn the_rig_block_names_the_rig_the_sensor_and_the_frame() {
    let node = rig_node();
    // Image 3 is the second sensor of frame 1.
    let painted = show(&node, Some(0), Some(3));
    assert!(says(&painted, "kerry_rig"));
    // Spelled out rather than looked for loosely: "right" is also an axis
    // label two blocks up, and a rig row that never drew would still match it.
    assert!(says(&painted, "right  ·  index 1"));
    assert!(says(&painted, "1  ·  2 images"));
    assert!(!says(&painted, "identity (reference sensor)"));
}

#[test]
fn the_reference_sensors_pose_says_it_is_the_identity() {
    let node = rig_node();
    // Image 2 is the reference sensor of frame 1.
    let painted = show(&node, Some(0), Some(2));
    assert!(says(&painted, "left  ·  index 0 (reference sensor)"));
    assert!(says(&painted, "identity (reference sensor)"));
}

#[test]
fn a_reconstruction_with_no_rig_draws_no_rig_block() {
    assert!(!says(
        &show(&pinhole_node(), Some(0), Some(0)),
        "sensor_from_rig"
    ));
}

// ── Copy ────────────────────────────────────────────────────────────────

#[test]
fn the_copied_parameter_text_is_the_table() {
    let node = pinhole_node();
    let camera = &node.recon().image_set.cameras[0];
    let text = super::header::parameters_text(camera);
    let lines: Vec<&str> = text.lines().collect();
    let declared = camera.model.parameter_names();
    assert_eq!(lines.len(), declared.len());
    for (line, name) in lines.iter().zip(declared.iter()) {
        assert!(
            line.starts_with(name.as_ref()),
            "{line} does not start with {name}"
        );
    }
    assert!(lines[0].contains("1000.000000"));
}

#[test]
fn the_copied_parameter_json_names_the_model_and_the_size() {
    let camera = kerry_park_camera();
    let json = super::header::parameters_json(&camera);
    assert!(json.contains("\"model\": \"OPENCV_FISHEYE\""));
    assert!(json.contains("\"width\": 480"));
    // Full precision, not the table's six decimals: a pasted `k1` rounded to
    // six places is a different lens.
    assert!(json.contains("0.038113353966529886"));
}

// ── Caching ─────────────────────────────────────────────────────────────

#[test]
fn the_derived_report_is_cached_per_camera_and_dropped_with_the_node() {
    let node = two_camera_node();
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();

    texts(&mut panel, &ctx, &node, Some(0), None);
    assert_eq!(panel.derived.len(), 1);
    texts(&mut panel, &ctx, &node, Some(0), None);
    assert_eq!(panel.derived.len(), 1);
    texts(&mut panel, &ctx, &node, Some(1), None);
    assert_eq!(panel.derived.len(), 2);

    panel.forget_recon(node.id);
    assert!(panel.derived.is_empty());
}

// ── Projection plot ─────────────────────────────────────────────────────

/// A node whose camera is an `EQUIDISTANT_FISHEYE` — a fisheye with no
/// distortion parameters at all, so its residual is identically zero.
fn equidistant_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon_mut().image_set.cameras[0] = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 152.8,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    node
}

/// A node whose camera is an `EQUIRECTANGULAR` panorama, which is its own
/// reference map rather than a pinhole's or an equidistant fisheye's.
fn equirectangular_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon_mut().image_set.cameras[0] = CameraIntrinsics {
        model: CameraModel::Equirectangular {
            focal_length_x: 2048.0 / std::f64::consts::TAU,
            focal_length_y: 1024.0 / std::f64::consts::PI,
            principal_point_x: 1024.0,
            principal_point_y: 512.0,
        },
        width: 2048,
        height: 1024,
        rolling_shutter: None,
    };
    node
}

#[test]
fn the_plot_draws_both_axes_and_names_the_ideal_map_by_family() {
    let painted = show(&pinhole_node(), Some(0), None);
    assert!(says(&painted, "Projection"));
    assert!(says(&painted, "r (px)"));
    assert!(says(&painted, "Δr (px)"));
    assert!(says(&painted, "θ off-axis"));
    // A perspective model is measured against `f·tan θ`...
    assert!(says(&painted, "dashed: ideal r = f·tan θ"));

    // ...a fisheye against `f·θ`...
    let painted = show(&fisheye_node(), Some(0), None);
    assert!(says(&painted, "dashed: ideal r = f·θ"));
    assert!(!says(&painted, "f·tan θ"));

    // ...and an equirectangular panorama against itself.
    let painted = show(&equirectangular_node(), Some(0), None);
    assert!(says(&painted, "dashed: ideal (itself)"));
}

/// The rules the plot draws are the frame's own geometry, so they carry the
/// same angles the derived table above them does.
#[test]
fn the_plot_marks_the_frames_edges_and_corner() {
    let painted = show(&fisheye_node(), Some(0), None);
    let fov = sfmtool_core::camera::report::field_of_view(&kerry_park_camera()).unwrap();
    // A square frame with a centred principal point: one edge rule, not two.
    assert!(says(&painted, "edge "));
    assert!(!says(&painted, "h edge"));
    assert!(says(&painted, &format!("corner {:.1}°", fov.max_off_axis)));
}

/// The bounded case: the extrapolated region is named on the plot and
/// explained under it, and the caption says how far the frame goes past it.
#[test]
fn the_plot_marks_and_explains_the_extrapolated_region() {
    let camera = kerry_park_camera();
    let limit = sfmtool_core::camera::report::trustworthy_max_theta_deg(&camera).unwrap();
    let corner = sfmtool_core::camera::report::field_of_view(&camera)
        .unwrap()
        .max_off_axis;
    let painted = show(&fisheye_node(), Some(0), None);

    assert!(says(&painted, &format!("extrapolated past {limit:.1}°")));
    assert!(says(&painted, &format!("Shaded past {limit:.1}°")));
    assert!(says(&painted, &format!("The frame reaches {corner:.1}°")));
    // The two numbers are the point: two thirds of this frame's angular
    // extent is outside what the model was fitted to.
    assert!(corner > 1.5 * limit);
}

/// The unbounded case says nothing about extrapolation, rather than saying it
/// with a limit equal to the corner.
#[test]
fn an_unbounded_model_has_no_extrapolated_region() {
    for painted in [
        show(&pinhole_node(), Some(0), None),
        show(&two_camera_node(), Some(1), None),
    ] {
        assert!(!says(&painted, "extrapolated past"));
        assert!(!says(&painted, "Shaded past"));
    }
}

/// The band is drawn when the model treats its azimuths differently, and the
/// legend is what says so. `kerry_park` carries `fx ≠ fy`; `SIMPLE_RADIAL`
/// cannot.
#[test]
fn the_azimuth_band_is_declared_only_when_there_is_one() {
    assert!(says(
        &show(&fisheye_node(), Some(0), None),
        "band: spread over 32 azimuths"
    ));
    assert!(!says(
        &show(&two_camera_node(), Some(1), None),
        "band: spread over 32 azimuths"
    ));
}

/// A model with no distortion still gets both plots — the projection curve is
/// a fact about the camera either way — with a banner across the residual
/// saying what the model exactly *is*, not only what it is not.
#[test]
fn a_distortion_free_model_gets_the_banner_naming_its_family() {
    for (node, expected) in [
        (pinhole_node(), "exactly a pinhole"),
        (equidistant_node(), "exactly an equidistant fisheye"),
        (equirectangular_node(), "exactly its own reference map"),
    ] {
        let painted = show(&node, Some(0), None);
        assert!(
            says(&painted, expected),
            "expected the banner to say {expected:?}"
        );
        // Both plots are still there.
        assert!(says(&painted, "r (px)"));
        assert!(says(&painted, "Δr (px)"));
    }
    // And a lens that does distort gets no banner.
    assert!(!says(
        &show(&fisheye_node(), Some(0), None),
        "No distortion — this model"
    ));
}

/// A camera with no image has no angle to span, so the plot says so rather
/// than allocating an empty frame with no axis.
#[test]
fn a_camera_with_no_image_gets_a_statement_instead_of_a_plot() {
    let mut node = pinhole_node();
    node.recon_mut().image_set.cameras[0].width = 0;
    node.recon_mut().image_set.cameras[0].height = 0;
    let painted = show(&node, Some(0), None);
    assert!(says(&painted, "No projection to plot"));
    assert!(!says(&painted, "θ off-axis"));
}

// ── Switch Model… and the proposal ──────────────────────────────────────

/// The header's one edit is `Switch Model…`; there is no `Refit spline…`.
#[test]
fn the_header_offers_switch_model_and_no_refit_spline() {
    let painted = show(&fisheye_node(), Some(0), None);
    assert!(says(&painted, "Switch Model…"));
    assert!(!says(&painted, "Refit spline"));
}

/// A node whose camera 1 is a four-coefficient `SFMTOOL_PINHOLE` on a 45°
/// domain, with a flat spline, and camera 0 a `SIMPLE_PINHOLE`.
fn spline_proposal_state() -> (crate::state::AppState, crate::scene::ReconId) {
    let (mut state, id) = crate::state::edits::tests::adjustable_state();
    {
        let recon = state.scene[0].recon_mut();
        let camera = &mut recon.image_set.cameras[0];
        let (f, _) = camera.focal_lengths();
        let (cx, cy) = camera.principal_point();
        camera.model = CameraModel::SimplePinhole {
            focal_length: f,
            principal_point_x: cx,
            principal_point_y: cy,
        };
        let mut second = recon.image_set.cameras[0].clone();
        second.model = CameraModel::SfmtoolPinhole {
            focal_length: f,
            principal_point_x: cx,
            principal_point_y: cy,
            bspline_rho_max: 1.0,
            bspline: vec![0.0; 4],
        };
        recon.image_set.cameras.push(second);
        for image in recon.image_set.images.iter_mut().skip(4) {
            image.camera_index = 1;
        }
    }
    (state, id)
}

/// One frame of the panel on the state's camera `camera`, with its proposal,
/// with `events` delivered: what it painted and what it reported.
fn proposal_frame(
    panel: &mut IntrinsicsDetail,
    ctx: &egui::Context,
    state: &mut crate::state::AppState,
    camera: usize,
    events: Vec<egui::Event>,
) -> (
    Vec<crate::test_support::PaintedText>,
    Option<(crate::scene::CameraRef, super::ProposalAction)>,
) {
    proposal_frame_at(panel, ctx, state, camera, VIEWPORT.x, events)
}

/// [`proposal_frame`] with the panel `width` pixels wide.
fn proposal_frame_at(
    panel: &mut IntrinsicsDetail,
    ctx: &egui::Context,
    state: &mut crate::state::AppState,
    camera: usize,
    width: f32,
    events: Vec<egui::Event>,
) -> (
    Vec<crate::test_support::PaintedText>,
    Option<(crate::scene::CameraRef, super::ProposalAction)>,
) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width, VIEWPORT.y),
        )),
        events,
        ..Default::default()
    };
    let mut action = None;
    let node = &state.scene[0];
    let mut proposal = state.camera_model_proposal.as_mut();
    let painted = crate::test_support::painted_text_rects(ctx, input, |ui| {
        action = panel
            .show(ui, node, Some(camera), None, proposal.as_deref_mut(), None)
            .proposal;
    });
    (painted, action)
}

fn texts_of(painted: &[crate::test_support::PaintedText]) -> Vec<String> {
    painted.iter().map(|p| p.text.clone()).collect()
}

/// A spline camera's proposal carries the refit's controls: the model as the
/// refit, Coefficients and Spline domain with the camera's own beside them,
/// no Fit to θ, the fit's line, and the two models side by side.
#[test]
fn a_refit_proposal_shows_the_refit_controls_and_both_cameras() {
    let (mut state, id) = spline_proposal_state();
    state
        .open_camera_model_proposal(crate::scene::CameraRef::new(id, 1), None)
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    let (painted, _) = proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    let painted = texts_of(&painted);
    for needle in [
        "SFMTOOL_PINHOLE (refit spline)",
        "Coefficients",
        "now 4",
        "Spline domain (°)",
        "now 45.0°",
        "rms ",
        "Current",
        "Proposed",
        "Apply",
        "Cancel",
    ] {
        assert!(says(&painted, needle), "{needle} missing from {painted:?}");
    }
    assert!(!says(&painted, "Fit to θ"), "{painted:?}");
}

/// Any other camera's proposal shows Fit to θ with its default and where it
/// came from; the panel on another camera of the node shows no strip.
#[test]
fn a_change_of_model_shows_the_fit_angle_and_its_default() {
    let (mut state, id) = spline_proposal_state();
    state
        .open_camera_model_proposal(crate::scene::CameraRef::new(id, 0), None)
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    proposal_frame(&mut panel, &ctx, &mut state, 0, Vec::new());
    let (painted, _) = proposal_frame(&mut panel, &ctx, &mut state, 0, Vec::new());
    let painted = texts_of(&painted);
    assert!(says(&painted, "Fit to θ (°)"), "{painted:?}");
    assert!(says(&painted, "default "), "{painted:?}");
    assert!(!says(&painted, "(refit spline)"), "{painted:?}");

    let (painted, _) = proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    assert!(!says(&texts_of(&painted), "Proposal"));
}

/// Apply and Cancel are reported for the camera shown, and `Escape` over the
/// panel is Cancel.
#[test]
fn apply_cancel_and_escape_are_reported() {
    let (mut state, id) = spline_proposal_state();
    let camera = crate::scene::CameraRef::new(id, 1);
    state
        .open_camera_model_proposal(camera, None)
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    let (painted, _) = proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    let at = |label: &str| {
        painted
            .iter()
            .find(|p| p.text == label)
            .unwrap_or_else(|| panic!("{label} was not painted"))
            .rect
            .center()
    };
    for (label, expected) in [
        ("Apply", super::ProposalAction::Apply),
        ("Cancel", super::ProposalAction::Cancel),
    ] {
        let pos = at(label);
        let press = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        proposal_frame(
            &mut panel,
            &ctx,
            &mut state,
            1,
            vec![egui::Event::PointerMoved(pos)],
        );
        proposal_frame(&mut panel, &ctx, &mut state, 1, vec![press(true)]);
        let (_, action) = proposal_frame(&mut panel, &ctx, &mut state, 1, vec![press(false)]);
        assert_eq!(action, Some((camera, expected)), "{label}");
    }

    let escape = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let (_, action) = proposal_frame(&mut panel, &ctx, &mut state, 1, vec![escape]);
    assert_eq!(action, Some((camera, super::ProposalAction::Cancel)));
}

/// The proposed camera's derived report is cached apart from the current
/// camera's, keyed by the proposal's generation.
#[test]
fn the_proposed_report_is_cached_apart_from_the_current_one() {
    let (mut state, id) = spline_proposal_state();
    let camera = crate::scene::CameraRef::new(id, 1);
    state
        .open_camera_model_proposal(camera, None)
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    let generation = state.camera_model_proposal.as_ref().unwrap().generation;
    assert!(matches!(
        panel.proposed_derived,
        Some((c, g, _)) if c == camera && g == generation
    ));
    assert_eq!(panel.derived.len(), 1, "the current camera's report");
    panel.forget_recon(id);
    assert!(panel.proposed_derived.is_none());
}

/// The pairs of painted strings above `until` whose glyphs overlap, shrunk by
/// a pixel so strings that merely touch are not counted.
fn overlapping(painted: &[crate::test_support::PaintedText], until: f32) -> Vec<(String, String)> {
    let shown: Vec<&crate::test_support::PaintedText> = painted
        .iter()
        .filter(|p| p.rect.min.y < until && !p.text.trim().is_empty())
        .collect();
    let mut pairs = Vec::new();
    for (i, a) in shown.iter().enumerate() {
        for b in &shown[i + 1..] {
            if a.rect.shrink(1.0).intersects(b.rect.shrink(1.0)) {
                pairs.push((a.text.clone(), b.text.clone()));
            }
        }
    }
    pairs
}

/// Where the first string painted as exactly `text` starts, top to bottom.
fn top_of(painted: &[crate::test_support::PaintedText], text: &str) -> f32 {
    painted
        .iter()
        .find(|p| p.text == text)
        .unwrap_or_else(|| panic!("{text} was not painted"))
        .rect
        .min
        .y
}

/// At the default panel width the header's identity line wraps rather than
/// running under `Switch Model…` and `Copy`.
#[test]
fn the_header_does_not_overlap_its_buttons_at_a_narrow_width() {
    let mut state = crate::state::AppState::new();
    state.append_node(fisheye_node());
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    // At 328 px the fixture's line breaks before a part that follows a `·`.
    for width in [482.0, 360.0, 341.0, 328.0] {
        for _ in 0..4 {
            proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
        }
        let (painted, _) = proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
        let until = top_of(&painted, "Parameters");
        assert_eq!(overlapping(&painted, until), Vec::new(), "at {width} px");
        assert_eq!(
            past_the_edge(&painted, until, width),
            Vec::<String>::new(),
            "at {width} px"
        );
        assert!(says(&texts_of(&painted), "480×480"), "at {width} px");
        for dot in painted.iter().filter(|p| p.text == "·") {
            assert!(
                painted.iter().any(|p| p.text != "·"
                    && (p.rect.center().y - dot.rect.center().y).abs() < 2.0
                    && p.rect.max.x <= dot.rect.min.x + 1.0),
                "a line starts with a separator at {width} px"
            );
        }
        assert!(says(&texts_of(&painted), "Switch Model…"), "at {width} px");
    }
}

/// The current and proposed tables never run into each other: one above the
/// other at a narrow width, side by side at a wide one.
#[test]
fn the_compared_tables_do_not_overlap_at_any_width() {
    let (mut state, id) = spline_proposal_state();
    state
        .open_camera_model_proposal(crate::scene::CameraRef::new(id, 0), None)
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    for width in [341.0, 455.0, 482.0, 877.0, 1800.0] {
        for _ in 0..4 {
            proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
        }
        let (painted, _) = proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
        let until = top_of(&painted, "K");
        assert_eq!(overlapping(&painted, until), Vec::new(), "at {width} px");
        assert_eq!(
            past_the_edge(&painted, until, width),
            Vec::<String>::new(),
            "at {width} px"
        );
        let current = top_of(&painted, "Current");
        let proposed = top_of(&painted, "Proposed");
        if width < 600.0 {
            assert!(proposed > current, "not stacked at {width} px");
        }
        if width > 1500.0 {
            assert_eq!(proposed, current, "not side by side at {width} px");
        }
    }
}

/// `Escape` that closes the open Model dropdown closes only the dropdown; the
/// next `Escape` cancels the proposal.
#[test]
fn escape_closing_the_model_dropdown_does_not_cancel() {
    let (mut state, id) = spline_proposal_state();
    let camera = crate::scene::CameraRef::new(id, 1);
    state
        .open_camera_model_proposal(camera, None)
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    let (painted, _) = proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    let pos = painted
        .iter()
        .find(|p| p.text == "SFMTOOL_PINHOLE (refit spline)")
        .expect("the Model dropdown")
        .rect
        .center();
    let press = |pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    proposal_frame(
        &mut panel,
        &ctx,
        &mut state,
        1,
        vec![egui::Event::PointerMoved(pos)],
    );
    proposal_frame(&mut panel, &ctx, &mut state, 1, vec![press(true)]);
    proposal_frame(&mut panel, &ctx, &mut state, 1, vec![press(false)]);
    proposal_frame(&mut panel, &ctx, &mut state, 1, Vec::new());
    assert!(egui::Popup::is_any_open(&ctx), "the dropdown did not open");

    let escape = || egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let (_, action) = proposal_frame(&mut panel, &ctx, &mut state, 1, vec![escape()]);
    assert_eq!(action, None, "closing the dropdown cancelled the proposal");
    assert!(!egui::Popup::is_any_open(&ctx), "the dropdown stayed open");
    let (_, action) = proposal_frame(&mut panel, &ctx, &mut state, 1, vec![escape()]);
    assert_eq!(action, Some((camera, super::ProposalAction::Cancel)));
}

/// A proposal whose fit is refused shows the fit angle the fit used and where
/// it came from, not a made-up 180°, and only the current camera's tables.
#[test]
fn a_refused_proposal_shows_its_default_fit_angle_and_the_current_tables() {
    let (mut state, id) = spline_proposal_state();
    let inputs = crate::camera_model_proposal::ProposalInputs {
        camera_model: "SFMTOOL_PINHOLE",
        coeff_count: Some(8),
        theta_fit_deg: None,
        spline_domain_deg: Some(95.0),
    };
    state
        .open_camera_model_proposal(crate::scene::CameraRef::new(id, 0), Some(inputs))
        .expect("opens with the refusal");
    let (deg, source) = state
        .camera_model_proposal
        .as_ref()
        .and_then(|p| p.theta_fit_default)
        .expect("a default beside the refusal");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    proposal_frame(&mut panel, &ctx, &mut state, 0, Vec::new());
    let (painted, _) = proposal_frame(&mut panel, &ctx, &mut state, 0, Vec::new());
    let painted = texts_of(&painted);
    assert!(says(&painted, "Refused: "), "{painted:?}");
    assert!(
        says(
            &painted,
            &format!("default {deg:.1}° ({})", source.as_str())
        ),
        "{painted:?}"
    );
    assert!(painted.contains(&format!("{deg:.1}")), "{painted:?}");
    assert!(!painted.contains(&"180.0".to_string()), "{painted:?}");
    assert!(!says(&painted, "Current"), "{painted:?}");
    assert!(!says(&painted, "Proposed"), "{painted:?}");
    // Drawing the strip left the wire's out-of-range domain as it was.
    let proposal = state.camera_model_proposal.as_ref().expect("open");
    assert_eq!(proposal.inputs.spline_domain_deg, Some(95.0));
    assert!(proposal.fit.is_err());
}

/// The painted strings above `until` that run past the panel's right edge.
fn past_the_edge(
    painted: &[crate::test_support::PaintedText],
    until: f32,
    width: f32,
) -> Vec<String> {
    painted
        .iter()
        .filter(|p| p.rect.min.y < until && p.rect.max.x > width + 0.5)
        .map(|p| p.text.clone())
        .collect()
}

/// At the stock window's panel width the fit angle's default and its
/// `Default` button stay inside the panel, where they can be reached.
#[test]
fn the_default_button_stays_inside_a_narrow_panel() {
    let (mut state, id) = spline_proposal_state();
    let inputs = crate::camera_model_proposal::ProposalInputs {
        camera_model: "SFMTOOL_PINHOLE",
        coeff_count: Some(8),
        theta_fit_deg: Some(10.0),
        spline_domain_deg: None,
    };
    state
        .open_camera_model_proposal(crate::scene::CameraRef::new(id, 0), Some(inputs))
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    let width = 341.0;
    // A wrapped value grows its grid row on the frames after it is laid out.
    for _ in 0..4 {
        proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
    }
    let (painted, _) = proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
    let button = painted
        .iter()
        .find(|p| p.text == crate::intrinsics_detail::proposal::DEFAULT)
        .expect("the Default button");
    assert!(button.rect.max.x <= width, "{:?}", button.rect);
    let label = painted
        .iter()
        .find(|p| p.text.starts_with("default ") && p.text.ends_with("(observations)"))
        .expect("the default's label");
    assert!(label.rect.max.x <= width, "{:?}", label.rect);
    let until = top_of(&painted, "K");
    assert_eq!(past_the_edge(&painted, until, width), Vec::<String>::new());
    assert_eq!(overlapping(&painted, until), Vec::new());
}

/// An `Escape` that ends the editing of a field ends only that, though egui
/// drops the field's focus before the frame draws; the next one cancels.
#[test]
fn escape_ending_a_fields_editing_does_not_cancel() {
    let (mut state, id) = spline_proposal_state();
    let camera = crate::scene::CameraRef::new(id, 0);
    state
        .open_camera_model_proposal(camera, None)
        .expect("opens");
    let (deg, _) = state
        .camera_model_proposal
        .as_ref()
        .and_then(|p| p.theta_fit_default)
        .expect("a default");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    proposal_frame(&mut panel, &ctx, &mut state, 0, Vec::new());
    let (painted, _) = proposal_frame(&mut panel, &ctx, &mut state, 0, Vec::new());
    let pos = painted
        .iter()
        .find(|p| p.text == format!("{deg:.1}"))
        .expect("the Fit to θ field")
        .rect
        .center();
    let press = |pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    proposal_frame(
        &mut panel,
        &ctx,
        &mut state,
        0,
        vec![egui::Event::PointerMoved(pos)],
    );
    proposal_frame(&mut panel, &ctx, &mut state, 0, vec![press(true)]);
    proposal_frame(&mut panel, &ctx, &mut state, 0, vec![press(false)]);
    proposal_frame(&mut panel, &ctx, &mut state, 0, Vec::new());
    assert!(
        ctx.egui_wants_keyboard_input(),
        "the field is not being edited"
    );

    let escape = || egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let (_, action) = proposal_frame(&mut panel, &ctx, &mut state, 0, vec![escape()]);
    assert_eq!(
        action, None,
        "ending the field's editing cancelled the proposal"
    );
    assert!(
        !ctx.egui_wants_keyboard_input(),
        "the field is still being edited"
    );
    let (_, action) = proposal_frame(&mut panel, &ctx, &mut state, 0, vec![escape()]);
    assert_eq!(action, Some((camera, super::ProposalAction::Cancel)));
}

// ── The comparison plots ────────────────────────────────────────────────

/// A frame of the panel on the fisheye fixture's camera, with or without its
/// proposal, after a few settling frames, at `width`.
fn fisheye_frames(
    state: &mut crate::state::AppState,
    width: f32,
) -> Vec<crate::test_support::PaintedText> {
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    // A wrapped value grows its grid row on the frames after it is laid out.
    for _ in 0..4 {
        proposal_frame_at(&mut panel, &ctx, state, 0, width, Vec::new());
    }
    proposal_frame_at(&mut panel, &ctx, state, 0, width, Vec::new()).0
}

/// While a proposal is open the plot draws the current and the proposed
/// camera, the change plot and the rug; without one it draws the camera alone.
#[test]
fn the_plot_compares_the_two_cameras_while_a_proposal_is_open() {
    let (mut state, _) = crate::camera_model_proposal::tests::fisheye_proposal_state();
    let observations = state.scene[0].recon().point_set.tracks.len();
    let painted = texts_of(&fisheye_frames(&mut state, VIEWPORT.x));
    for needle in [
        "grey: current · blue: proposed",
        "Change (px)",
        "current extrapolated past 84.",
        "fit 84.",
        "proposed spline domain",
        "The proposed model is trusted at every angle the frame reaches.",
        &format!("Rug: {observations} observations"),
    ] {
        assert!(says(&painted, needle), "{needle} missing from {painted:?}");
    }

    let mut state = crate::state::AppState::new();
    state.append_node(crate::camera_model_proposal::tests::fisheye_node(
        "/runs/kerry.sfmr",
    ));
    let painted = texts_of(&fisheye_frames(&mut state, VIEWPORT.x));
    assert!(says(&painted, "solid: model"), "{painted:?}");
    for needle in ["Change (px)", "Rug:", "grey: current", "fit 84."] {
        assert!(!says(&painted, needle), "{needle} drawn without a proposal");
    }
}

/// The rug's caption counts the observations past the current camera's
/// trusted bound: the fixture's wide points.
#[test]
fn the_rug_counts_the_observations_past_the_trusted_bound() {
    let (mut state, _) = crate::camera_model_proposal::tests::fisheye_proposal_state();
    let limit = sfmtool_core::camera::report::trustworthy_max_theta_deg(
        &crate::camera_model_proposal::tests::kerry_camera(),
    )
    .unwrap();
    let past = state
        .camera_model_proposal
        .as_ref()
        .unwrap()
        .rug()
        .iter()
        .filter(|&&t| t > limit)
        .count();
    assert!(past > 0);
    let painted = texts_of(&fisheye_frames(&mut state, VIEWPORT.x));
    let caption = format!("; {past} past {limit:.1}°.");
    assert!(
        says(&painted, &caption),
        "{caption} missing from {painted:?}"
    );
}

/// Nothing the panel paints with a proposal open overlaps anything else or
/// runs past the panel's right edge, from the stock window's width to a wide
/// one.
#[test]
fn the_comparison_stays_inside_the_panel_at_any_width() {
    let (mut state, _) = crate::camera_model_proposal::tests::fisheye_proposal_state();
    for width in [341.0, 482.0, 1800.0] {
        let painted = fisheye_frames(&mut state, width);
        assert!(says(&texts_of(&painted), "Change (px)"), "at {width} px");
        assert_eq!(
            overlapping(&painted, f32::INFINITY),
            Vec::new(),
            "at {width} px"
        );
        assert_eq!(
            past_the_edge(&painted, f32::INFINITY, width),
            Vec::<String>::new(),
            "at {width} px"
        );
    }
}

/// Once the proposal's job has switched the camera the strip shows the
/// observation comparison, under each model and past the trusted bound, and
/// the Preview retriangulation checkbox; all of it, with the preview ticked,
/// stays inside the panel and apart from the rest from the stock window's
/// width to a wide one.
#[test]
fn the_observation_comparison_and_the_preview_box_stay_inside_the_panel() {
    let (mut state, _) = crate::camera_model_proposal::tests::fisheye_proposal_state();
    let before = texts_of(&fisheye_frames(&mut state, VIEWPORT.x));
    assert!(says(&before, super::proposal::COMPARING), "{before:?}");
    state
        .camera_model_proposal
        .as_mut()
        .unwrap()
        .preview_retriangulation = true;
    state.settle_camera_model_proposal_job();
    for width in [341.0, 482.0, 1800.0] {
        let painted = fisheye_frames(&mut state, width);
        let texts = texts_of(&painted);
        for needle in [
            super::proposal::PREVIEW_RETRIANGULATION,
            "observations ",
            "median ",
            "past 8",
        ] {
            assert!(
                says(&texts, needle),
                "{needle} missing at {width} px: {texts:?}"
            );
        }
        assert!(!says(&texts, super::proposal::COMPARING), "at {width} px");
        assert_eq!(
            overlapping(&painted, f32::INFINITY),
            Vec::new(),
            "at {width} px"
        );
        assert_eq!(
            past_the_edge(&painted, f32::INFINITY, width),
            Vec::<String>::new(),
            "at {width} px"
        );
    }
}

/// The comparison in the strip's words.
#[test]
fn the_comparison_lines_name_both_models_and_the_trusted_bound() {
    use sfmtool_core::reconstruction::switch_camera_model::{ErrorSummary, ObservationComparison};
    let summary = |median_px, p90_px, max_px| ErrorSummary {
        median_px,
        p90_px,
        max_px,
    };
    let comparison = ObservationComparison {
        observations: 4210,
        unmeasured: 0,
        max_theta_deg: 101.2,
        before: summary(0.298, 0.61, 3.1),
        after: summary(0.301, 0.62, 2.95),
        changed_over_1px: 12,
        trusted_deg: Some(84.5),
        past_trusted: 58,
        past_trusted_before: summary(12.3, 40.0, 210.0),
        past_trusted_after: summary(0.4, 1.0, f64::NAN),
    };
    assert_eq!(
        super::proposal::comparison_lines(&comparison),
        vec![
            "observations 4210: median 0.298 → 0.301 px, p90 0.610 → 0.620 px, max 3.100 → \
             2.950 px"
                .to_string(),
            "past 84.5°: 58: median 12.300 → 0.400 px, p90 40.000 → 1.000 px, max 210.000 → \
             n/a px"
                .to_string(),
        ]
    );
}

// ── Per-image statistics ────────────────────────────────────────────────

/// The seoul_bull ground truth whose `SFMTOOL_PINHOLE` camera varies all four
/// per-image parameters, read on columns, over its 17 posed images.
const SEOUL_BULL_V02: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth_v02.sfmr"
);

fn seoul_bull_v02() -> SceneNode {
    let path = std::path::Path::new(SEOUL_BULL_V02);
    let recon = SfmrReconstruction::load(path, &sfmtool_core::progress::Progress::none())
        .expect("the ground truth loads");
    let mut node = SceneNode::from_path(path, recon);
    // The file's own name is longer than the header can fit beside its
    // buttons at the narrowest width, which is the header's business.
    node.label = "seoul_bull".to_string();
    node
}

/// The same camera with its radial correction and rates dropped, so it varies
/// the focal and the principal point alone.
fn seoul_bull_focal_and_principal_point() -> SceneNode {
    use sfmtool_core::reconstruction::bundle_adjust::ImageAdjustmentParameter;
    let mut node = seoul_bull_v02();
    let dropped = sfmtool_core::reconstruction::drop_image_adjustments::drop_image_adjustments(
        node.recon(),
        0,
        &[
            ImageAdjustmentParameter::RadialBspline,
            ImageAdjustmentParameter::RollingShutter,
        ],
    )
    .expect("drops");
    *node.recon_mut() = dropped;
    node
}

/// The painted strings of one frame of the panel on camera 0 of `node`,
/// `width` pixels wide, with their rects, after the frames a wrapped row
/// needs to settle.
fn frame_at(node: &SceneNode, width: f32) -> Vec<crate::test_support::PaintedText> {
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    let input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width, VIEWPORT.y),
        )),
        ..Default::default()
    };
    for _ in 0..4 {
        crate::test_support::painted_text_rects(&ctx, input(), |ui| {
            panel.show(ui, node, Some(0), None, None, None);
        });
    }
    crate::test_support::painted_text_rects(&ctx, input(), |ui| {
        panel.show(ui, node, Some(0), None, None, None);
    })
}

/// The painted strings with each `per image` row's statistics split back
/// into their cells, its non-breaking spaces read as spaces.
fn plain(painted: &[String]) -> Vec<String> {
    painted
        .iter()
        .flat_map(|t| {
            if t.contains('\u{a0}') {
                t.split("  ").map(|c| c.replace('\u{a0}', " ")).collect()
            } else {
                vec![t.clone()]
            }
        })
        .collect()
}

/// Where in the painted strings `text` first appears exactly.
fn position_of(painted: &[String], text: &str) -> usize {
    painted
        .iter()
        .position(|t| t == text)
        .unwrap_or_else(|| panic!("{text:?} was not painted: {painted:?}"))
}

/// The cells of the `per image` row labelled `label`, read off `rows`.
fn cells_of<'a>(rows: &'a [super::parameters::StatisticsRow], label: &str) -> &'a [String] {
    &rows
        .iter()
        .find(|r| r.label == label)
        .unwrap_or_else(|| panic!("no row {label:?}"))
        .cells
}

#[test]
fn a_camera_with_no_per_image_parameters_has_no_per_image_rows() {
    for node in [pinhole_node(), fisheye_node()] {
        let painted = plain(&show(&node, Some(0), None));
        assert!(says(&painted, "Parameters"), "{painted:?}");
        assert!(!says(&painted, "per image"), "{painted:?}");
        assert!(!says(&painted, "Per image"), "{painted:?}");
    }
    // A scanlines setting alone varies nothing.
    let mut node = pinhole_node();
    node.recon_mut().image_set.per_image_adjustments =
        vec![sfmtool_core::reconstruction::PerImageAdjustments {
            rolling_shutter_scanlines: Some(sfmtool_core::camera::RollingShutterScanlines::Rows),
            ..Default::default()
        }];
    let painted = plain(&show(&node, Some(0), None));
    assert!(!says(&painted, "per image"), "{painted:?}");
}

/// The seoul_bull camera varies all four parameters: each row sits beside the
/// shared parameter it adjusts, the radial correction after the spline and
/// the rates with their scanlines at the end.
#[test]
fn every_per_image_parameter_sits_beside_what_it_adjusts() {
    let node = seoul_bull_v02();
    let painted = plain(&show(&node, Some(0), None));
    assert!(
        says(
            &painted,
            "Per image, over 17 images: focal, principal point, radial correction, \
             rolling-shutter rates."
        ),
        "{painted:?}"
    );
    assert!(!says(&painted, "Not varied"), "{painted:?}");
    assert!(!says(&painted, "Not on"), "{painted:?}");

    let at = |text: &str| position_of(&painted, text);
    let last_coefficient = node.recon().image_set.cameras[0]
        .model
        .parameter_names()
        .last()
        .unwrap()
        .to_string();
    assert!(at("focal_length") < at("  per image focal"));
    assert!(at("  per image focal") < at("principal_point_x"));
    assert!(at("principal_point_x") < at("  per image Δcx"));
    assert!(at("  per image Δcx") < at("principal_point_y"));
    assert!(at("principal_point_y") < at("  per image Δcy"));
    assert!(at("  per image Δcy") < at("bspline_rho_max"));
    assert!(at(&last_coefficient) < at("  per image e0"));
    assert!(at("  per image e0") < at("  per image e1"));
    assert!(at("  per image e1") < at("  scanlines"));
    assert!(at("  scanlines") < at("columns"));
    assert!(at("columns") < at("  per image ωx"));
    assert!(at("  per image ωx") < at("  per image ωy"));
    assert!(at("  per image ωy") < at("  per image ωz"));
    assert!(at("  per image ωz") < at("Derived"));
    assert!(says(&painted, "mrad per short side"), "{painted:?}");
    assert!(says(&painted, "of 17 past 2 sd"), "{painted:?}");
}

/// The numbers are the file's, in the panel's units, worked out here from the
/// images directly: the focal's mean is the geometric mean of `1 + s` as a
/// percentage, the principal-point offsets are in pixels of the 270 px short
/// side, the radial correction is as stored and the rates are in milliradians
/// per short side, as Image Detail's legend states them.
#[test]
fn the_statistics_are_the_images_values_in_readable_units() {
    let node = seoul_bull_v02();
    let recon = node.recon();
    let camera = &recon.image_set.cameras[0];
    let per_image = super::parameters::PerImage {
        flags: recon.image_set.camera_per_image_adjustments(0),
        statistics:
            sfmtool_core::reconstruction::image_adjustment_statistics::image_adjustment_statistics(
                &recon.image_set,
                0,
            )
            .unwrap(),
        images: &recon.image_set.images,
    };
    let rows = super::parameters::statistics_rows(camera, &per_image);
    let adjustments: Vec<_> = recon
        .image_set
        .images
        .iter()
        .map(|image| image.intrinsic_adjustment)
        .collect();
    assert_eq!(adjustments.len(), 17);
    let n = 17.0;
    let signed = |v: f64, d: usize| super::format::signed(v, d);

    let log_mean = adjustments.iter().map(|a| a.focal.ln_1p()).sum::<f64>() / n;
    let log_std = (adjustments
        .iter()
        .map(|a| (a.focal.ln_1p() - log_mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let focal_max = adjustments.iter().map(|a| a.focal).fold(f64::MIN, f64::max);
    let focal = cells_of(&rows, "  per image focal");
    assert_eq!(
        focal[0],
        format!("mean {}%", signed(100.0 * log_mean.exp_m1(), 3))
    );
    assert_eq!(focal[1], format!("sd {:.3}%", 100.0 * log_std));
    assert_eq!(focal[3], format!("max {}%", signed(100.0 * focal_max, 3)));
    assert!(focal[4].ends_with(" of 17 past 2 sd"), "{focal:?}");

    let short_side = 270.0;
    let x_min = adjustments
        .iter()
        .map(|a| a.principal_point[0])
        .fold(f64::MAX, f64::min);
    let y_max = adjustments
        .iter()
        .map(|a| a.principal_point[1])
        .fold(f64::MIN, f64::max);
    assert_eq!(
        cells_of(&rows, "  per image Δcx")[2],
        format!("min {} px", signed(short_side * x_min, 3))
    );
    assert_eq!(
        cells_of(&rows, "  per image Δcy")[3],
        format!("max {} px", signed(short_side * y_max, 3))
    );

    let e_1_mean = adjustments.iter().map(|a| a.radial_bspline[1]).sum::<f64>() / n;
    assert_eq!(
        cells_of(&rows, "  per image e1")[0],
        format!("mean {}", signed(e_1_mean, 6))
    );
    let rate_z_max = adjustments
        .iter()
        .map(|a| a.rolling_shutter[2])
        .fold(f64::MIN, f64::max);
    let rates = cells_of(&rows, "  per image ωz");
    assert_eq!(rates[3], format!("max {}", signed(1000.0 * rate_z_max, 3)));
    assert_eq!(rates[4], "mrad per short side");
    assert_eq!(cells_of(&rows, "  scanlines"), ["columns"]);
}

/// A camera that varies the focal and the principal point names what it does
/// not vary, and on a model that cannot carry a radial correction or rates,
/// that the model cannot.
#[test]
fn what_a_camera_does_not_vary_and_what_its_model_cannot_carry_are_named() {
    let painted = plain(&show(
        &seoul_bull_focal_and_principal_point(),
        Some(0),
        None,
    ));
    assert!(
        says(
            &painted,
            "Per image, over 17 images: focal, principal point. Not varied: radial correction, \
             rolling-shutter rates."
        ),
        "{painted:?}"
    );
    assert!(says(&painted, "  per image Δcy"), "{painted:?}");
    assert!(!says(&painted, "  per image e0"), "{painted:?}");
    assert!(!says(&painted, "  scanlines"), "{painted:?}");

    let mut node = fisheye_node();
    *node.recon_mut() = crate::test_support::with_adjustment(
        node.recon().clone(),
        0,
        crate::test_support::ADJUSTMENT,
    );
    let painted = plain(&show(&node, Some(0), None));
    assert!(
        says(
            &painted,
            "Per image, over 8 images: focal, principal point. Not on OPENCV_FISHEYE: radial \
             correction, rolling-shutter rates."
        ),
        "{painted:?}"
    );
    assert!(says(&painted, "  per image focal"), "{painted:?}");
    // On a two-focal model the focal's row follows the second focal length.
    let at = |text: &str| position_of(&painted, text);
    assert!(at("focal_length_y") < at("  per image focal"));
    assert!(at("  per image focal") < at("principal_point_x"));
}

/// A parameter the camera varies whose values are all zero reads as zero
/// statistics, not as a parameter it does not vary.
#[test]
fn a_varied_parameter_whose_values_are_all_zero_reads_as_zero() {
    let mut node = pinhole_node();
    *node.recon_mut() = crate::test_support::with_adjustment(
        node.recon().clone(),
        0,
        sfmtool_core::IntrinsicAdjustment::default(),
    );
    let painted = plain(&show(&node, Some(0), None));
    assert!(says(
        &painted,
        "Per image, over 8 images: focal, principal point."
    ));
    for cell in [
        "mean +0.000%",
        "sd 0.000%",
        "max +0.000 px",
        "0 of 8 past 2 sd",
    ] {
        assert!(
            painted.iter().any(|t| t == cell),
            "{cell} missing: {painted:?}"
        );
    }
}

/// The per-image rows stay inside the panel and apart from everything else
/// from the stock window's width to a wide one.
#[test]
fn the_per_image_rows_stay_inside_the_panel_at_any_width() {
    let node = seoul_bull_v02();
    for width in [341.0, 482.0, 1800.0] {
        let painted = frame_at(&node, width);
        let until = top_of(&painted, "K");
        assert!(
            says(&plain(&texts_of(&painted)), "  per image ωz"),
            "at {width} px"
        );
        assert_eq!(overlapping(&painted, until), Vec::new(), "at {width} px");
        assert_eq!(
            past_the_edge(&painted, until, width),
            Vec::<String>::new(),
            "at {width} px"
        );
    }
}

/// With a proposal open the statistics, which are the images' and not either
/// lens's, are stated once, in the current camera's table, and the two tables
/// still keep apart.
#[test]
fn a_proposal_states_the_statistics_in_the_current_table_only() {
    let mut state = crate::state::AppState::new();
    let id = state.append_node(seoul_bull_focal_and_principal_point());
    state
        .open_camera_model_proposal(crate::scene::CameraRef::new(id, 0), None)
        .expect("opens");
    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    for width in [341.0, 482.0, 1800.0] {
        for _ in 0..4 {
            proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
        }
        let (painted, _) = proposal_frame_at(&mut panel, &ctx, &mut state, 0, width, Vec::new());
        let texts = plain(&texts_of(&painted));
        assert!(says(&texts, "Proposed"), "at {width} px: {texts:?}");
        assert_eq!(
            texts.iter().filter(|t| *t == "  per image focal").count(),
            1,
            "at {width} px"
        );
        let until = top_of(&painted, "K");
        assert_eq!(overlapping(&painted, until), Vec::new(), "at {width} px");
        assert_eq!(
            past_the_edge(&painted, until, width),
            Vec::<String>::new(),
            "at {width} px"
        );
    }
}

/// Every glyph the per-image rows write is in egui's bundled fonts. Checked
/// in the proportional family, whose fonts the monospace family falls back to
/// after its own, as `scene_graph`'s and the intrinsics layer's glyph tests
/// check theirs.
#[test]
fn the_per_image_glyphs_are_in_the_bundled_fonts() {
    let ctx = egui::Context::default();
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("warm the font atlas");
    });
    let font = egui::FontId::proportional(14.0);
    for glyph in ["Δ", "ω", "\u{2212}", "\u{a0}"] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&font, glyph)),
            "{glyph:?} is not in egui's bundled fonts and would render as a box"
        );
    }
}

#[test]
fn a_signed_number_never_reads_minus_zero() {
    assert_eq!(super::format::signed(-0.0004, 3), "+0.000");
    assert_eq!(super::format::signed(-0.0006, 3), "\u{2212}0.001");
    assert_eq!(super::format::signed(0.25, 2), "+0.25");
}

// ── The per-image histograms ────────────────────────────────────────────

use super::histogram::{bin_edges, Histogram};

/// Values `0, 1, …, 8`: nine values aim for three bins, the spread of 8 over
/// three is nearest 2 on the ladder, and the bins run from the step below the
/// lower two-standard-deviation bound to the step past the upper one.
#[test]
fn the_bins_are_round_and_hold_every_value() {
    let values: Vec<(usize, f64)> = (0..9).map(|i| (i, i as f64)).collect();
    let histogram = Histogram::new(values, 4.0, [-1.16, 9.16]);
    assert_eq!(histogram.edges, [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    assert_eq!(histogram.counts(), [0, 2, 2, 2, 2, 1]);
    // A value on an edge is in the bin above it, and a bin is ordered by
    // value.
    assert_eq!(histogram.bins[3], [(4, 4.0), (5, 5.0)]);
    assert_eq!(histogram.bins[5], [(8, 8.0)]);
    // The nearest step on a log scale, not the next one up.
    assert_eq!(super::histogram::nearest_step(2.04), 2.0);
    assert_eq!(super::histogram::nearest_step(3.3), 5.0);
    assert_eq!(super::histogram::nearest_step(0.0012), 0.001);
    // A hundred values aim for ten bins, every edge a multiple of the step.
    let many: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
    let edges = bin_edges(&many, [-4.5, 4.5]);
    let step = edges[1] - edges[0];
    assert_eq!(step, 0.5, "{edges:?}");
    for edge in &edges {
        assert!(
            (edge / step - (edge / step).round()).abs() < 1e-9,
            "{edges:?}"
        );
    }
    assert!(
        edges[0] <= -4.5 && *edges.last().unwrap() > 4.5,
        "{edges:?}"
    );
}

/// Two images still bin apart, and values that are all equal, a single
/// image's among them, make one bin centred on the value.
#[test]
fn few_and_equal_values_still_bin() {
    let two = Histogram::new(vec![(0, -1.0), (1, 1.0)], 0.0, [-2.0, 2.0]);
    assert_eq!(two.edges, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    assert_eq!(two.counts(), [0, 1, 0, 1, 0]);

    let zeros = Histogram::new(vec![(0, 0.0), (1, 0.0), (2, 0.0)], 0.0, [0.0, 0.0]);
    assert_eq!(zeros.edges, [-0.5, 0.5]);
    assert_eq!(zeros.counts(), [3]);

    let one = Histogram::new(vec![(4, 0.003)], 0.003, [0.003, 0.003]);
    assert_eq!(one.counts(), [1]);
    let [lo, hi] = [one.edges[0], one.edges[1]];
    assert!(
        (lo - 0.0025).abs() < 1e-12 && (hi - 0.0035).abs() < 1e-12,
        "{one:?}"
    );

    // Values whose spread is a hundred-millionth of their size, where
    // rounding puts the edges a hair inside them, are still all binned.
    let close = Histogram::new(
        vec![(0, 0.99999999), (1, 1.00000001)],
        1.0,
        [0.99999999, 1.00000001],
    );
    assert_eq!(close.counts().iter().sum::<usize>(), 2, "{close:?}");

    // A value that is not a number is in no bin.
    let gap = Histogram::new(vec![(0, 1.0), (1, f64::NAN), (2, 2.0)], 1.5, [0.5, 2.5]);
    assert_eq!(gap.counts().iter().sum::<usize>(), 2);
    assert!(Histogram::new(Vec::new(), 0.0, [0.0, 0.0]).edges.is_empty());
}

/// An outlier far past the rest sits alone in the last bin, and the bin
/// count stays within the cap.
#[test]
fn an_outlier_sits_alone_in_its_bin() {
    let mut values: Vec<(usize, f64)> = (0..16).map(|i| (i, 0.1 * (i as f64 - 8.0))).collect();
    values.push((16, 10.0));
    let plain: Vec<f64> = values.iter().map(|(_, v)| *v).collect();
    let s =
        sfmtool_core::reconstruction::image_adjustment_statistics::ComponentStatistics::of(&plain)
            .unwrap();
    let histogram = Histogram::of_component(values, &s, 1.0, false);
    let counts = histogram.counts();
    assert_eq!(counts.iter().sum::<usize>(), 17);
    let last = counts.iter().rposition(|&c| c > 0).unwrap();
    assert_eq!(histogram.bins[last], [(16, 10.0)], "{histogram:?}");
    assert!(counts.len() <= 3 * super::histogram::MAX_BINS, "{counts:?}");
    assert!(histogram.two_sd[1] < 10.0, "the outlier is past two sd");
}

/// A click on a bin selects its first image, and a click with one of its
/// images selected selects the next, wrapping round.
#[test]
fn a_click_steps_through_a_bins_images() {
    let histogram = Histogram::new(
        vec![(7, 0.12), (3, 0.11), (9, 0.13), (1, 1.0)],
        0.34,
        [0.0, 0.8],
    );
    let k = histogram.bins.iter().position(|b| b.len() == 3).unwrap();
    assert_eq!(histogram.next_in_bin(k, None), Some(3));
    assert_eq!(histogram.next_in_bin(k, Some(3)), Some(7));
    assert_eq!(histogram.next_in_bin(k, Some(7)), Some(9));
    assert_eq!(histogram.next_in_bin(k, Some(9)), Some(3));
    assert_eq!(histogram.next_in_bin(k, Some(42)), Some(3));
}

/// The seoul_bull camera's `per image` rows, read off the panel's own
/// computation.
fn seoul_bull_rows(node: &SceneNode) -> Vec<super::parameters::StatisticsRow> {
    let recon = node.recon();
    let per_image = super::parameters::PerImage {
        flags: recon.image_set.camera_per_image_adjustments(0),
        statistics:
            sfmtool_core::reconstruction::image_adjustment_statistics::image_adjustment_statistics(
                &recon.image_set,
                0,
            )
            .unwrap(),
        images: &recon.image_set.images,
    };
    super::parameters::statistics_rows(&recon.image_set.cameras[0], &per_image)
}

/// What one frame of the panel painted, its AccessKit nodes and its
/// response.
type HistogramFrame = (
    Vec<crate::test_support::PaintedText>,
    Vec<egui::accesskit::Node>,
    super::IntrinsicsDetailResponse,
);

/// One frame of the panel on camera 0 of `node`, `width` wide, with `image`
/// selected and `events` in its input. Tooltips show at once, since a
/// headless frame has no clock to wait out their delay.
fn histogram_frame(
    panel: &mut IntrinsicsDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    image: Option<usize>,
    width: f32,
    events: Vec<egui::Event>,
) -> HistogramFrame {
    ctx.enable_accesskit();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width, VIEWPORT.y),
        )),
        events,
        ..Default::default()
    };
    let mut response = None;
    let mut output = ctx.run_ui(input, |ui| {
        response = Some(panel.show(ui, node, Some(0), image, None, None));
    });
    output.textures_delta.clear();
    let painted = crate::test_support::output_text_rects(&output);
    let nodes = output
        .platform_output
        .accesskit_update
        .expect("AccessKit is on")
        .nodes
        .into_iter()
        .map(|(_, node)| node)
        .collect();
    (painted, nodes, response.expect("the panel ran"))
}

/// The bins a frame drew, by their names, with their rectangles.
fn bins_of(nodes: &[egui::accesskit::Node]) -> Vec<(String, egui::Rect)> {
    nodes
        .iter()
        .filter(|node| {
            node.role() == egui::accesskit::Role::Button
                && node.label().is_some_and(|l| l.starts_with("per image "))
        })
        .map(|node| {
            let b = node.bounds().expect("a bin has a rectangle");
            (
                node.label().unwrap().to_string(),
                egui::Rect::from_min_max(
                    egui::pos2(b.x0 as f32, b.y0 as f32),
                    egui::pos2(b.x1 as f32, b.y1 as f32),
                ),
            )
        })
        .collect()
}

/// Every component the seoul_bull camera varies has a histogram of all 17
/// images, whose two-standard-deviation bounds leave out exactly the images
/// the row counts past two sd, the focal's mapped back from `ln(1 + s)`; and
/// the panel draws a bin for each non-empty bin of each.
#[test]
fn every_varied_component_draws_its_histogram() {
    let node = seoul_bull_v02();
    let rows = seoul_bull_rows(&node);
    let shown: Vec<&str> = rows
        .iter()
        .filter(|r| r.histogram.is_some())
        .map(|r| r.label.as_str())
        .collect();
    assert_eq!(
        shown,
        [
            "  per image focal",
            "  per image Δcx",
            "  per image Δcy",
            "  per image e0",
            "  per image e1",
            "  per image ωx",
            "  per image ωy",
            "  per image ωz",
        ]
    );
    for row in rows.iter().filter(|r| r.histogram.is_some()) {
        let histogram = row.histogram.as_ref().unwrap();
        assert_eq!(
            histogram.counts().iter().sum::<usize>(),
            17,
            "{}",
            row.label
        );
        let [lo, hi] = histogram.two_sd;
        let past = histogram
            .bins
            .iter()
            .flatten()
            .filter(|(_, v)| *v < lo || *v > hi)
            .count();
        let stated = row.cells.last().unwrap().clone();
        assert_eq!(stated, format!("{past} of 17 past 2 sd"), "{}", row.label);
    }

    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    for _ in 0..4 {
        histogram_frame(&mut panel, &ctx, &node, None, 482.0, Vec::new());
    }
    let (_, nodes, _) = histogram_frame(&mut panel, &ctx, &node, None, 482.0, Vec::new());
    let bins = bins_of(&nodes);
    for row in rows.iter().filter(|r| r.histogram.is_some()) {
        let named = format!("{} ", row.label.trim_start());
        let drawn = bins
            .iter()
            .filter(|(name, _)| name.starts_with(&named))
            .count();
        let filled = row
            .histogram
            .as_ref()
            .unwrap()
            .counts()
            .iter()
            .filter(|&&c| c > 0)
            .count();
        assert_eq!(drawn, filled, "{named}: {bins:?}");
    }
}

/// At the stock window's width the histograms go under their statistics and
/// at a wide one beside them; at every width each stays inside the panel and
/// no string is painted over a bin.
#[test]
fn the_histograms_stay_inside_the_panel_at_any_width() {
    let node = seoul_bull_v02();
    for width in [341.0, 482.0, 1800.0] {
        let mut panel = IntrinsicsDetail::new();
        let ctx = egui::Context::default();
        for _ in 0..4 {
            histogram_frame(&mut panel, &ctx, &node, None, width, Vec::new());
        }
        let (painted, nodes, _) = histogram_frame(&mut panel, &ctx, &node, None, width, Vec::new());
        let bins = bins_of(&nodes);
        assert!(!bins.is_empty(), "at {width} px");
        for (name, rect) in &bins {
            assert!(
                rect.right() <= width + 0.5,
                "{name} past the edge at {width} px"
            );
            for text in painted.iter().filter(|p| !p.text.trim().is_empty()) {
                assert!(
                    !text.rect.shrink(1.0).intersects(rect.shrink(1.0)),
                    "{:?} over {name} at {width} px",
                    text.text
                );
            }
        }
        // The focal's statistics, the first row of them painted.
        let statistics = painted
            .iter()
            .find(|p| p.text.starts_with("mean\u{a0}"))
            .expect("the focal's statistics");
        let focal: Vec<&egui::Rect> = bins
            .iter()
            .filter(|(name, _)| name.starts_with("per image focal "))
            .map(|(_, rect)| rect)
            .collect();
        assert!(!focal.is_empty(), "at {width} px");
        for rect in &focal {
            if width >= 1800.0 {
                assert!(rect.left() > statistics.rect.right(), "at {width} px");
                assert!(rect.top() < statistics.rect.bottom(), "at {width} px");
            } else {
                assert!(
                    rect.top() >= statistics.rect.bottom() - 0.5,
                    "at {width} px"
                );
            }
        }
    }
}

/// Hovering a bin lists its images with their values and names the image a
/// click selects, and raises that image as the hovered one; a click selects
/// it, and a second click with it selected selects the bin's next image.
#[test]
fn a_bins_tooltip_lists_its_images_and_a_click_selects_them() {
    let node = seoul_bull_v02();
    let rows = seoul_bull_rows(&node);
    let row = rows
        .iter()
        .find(|r| r.label == "  per image Δcy")
        .expect("the Δcy row");
    let histogram = row.histogram.as_ref().unwrap();
    // The fullest bin, which holds more than one image to step through.
    let fullest = (0..histogram.bins.len())
        .max_by_key(|&k| histogram.bins[k].len())
        .unwrap();
    let images = &histogram.bins[fullest];
    assert!(images.len() > 1, "{histogram:?}");
    let name = |i: usize| node.recon().image_set.images[i].name.clone();

    let mut panel = IntrinsicsDetail::new();
    let ctx = egui::Context::default();
    for _ in 0..4 {
        histogram_frame(&mut panel, &ctx, &node, None, 482.0, Vec::new());
    }
    let (_, nodes, _) = histogram_frame(&mut panel, &ctx, &node, None, 482.0, Vec::new());
    let bins = bins_of(&nodes);
    let count = images.len();
    let (bin_name, rect) = bins
        .iter()
        .filter(|(n, _)| n.starts_with("per image Δcy "))
        .find(|(n, _)| n.ends_with(&format!(": {count} images")))
        .cloned()
        .unwrap_or_else(|| panic!("no Δcy bin of {count} images: {bins:?}"));
    let at = rect.center();

    let mut hovered = Vec::new();
    let mut response = None;
    for _ in 0..3 {
        let (painted, _, shown) = histogram_frame(
            &mut panel,
            &ctx,
            &node,
            None,
            482.0,
            vec![egui::Event::PointerMoved(at)],
        );
        hovered = texts_of(&painted);
        response = Some(shown);
    }
    let header = hovered
        .iter()
        .find(|t| t.starts_with("Δcy from "))
        .unwrap_or_else(|| panic!("no tooltip for {bin_name}: {hovered:?}"));
    assert!(
        header.ends_with(&format!(" px: {count} of 17 images")),
        "{header}"
    );
    for &(image, value) in images {
        let line = format!("{}  {} px", name(image), super::format::signed(value, 3));
        assert!(hovered.contains(&line), "{line:?} missing: {hovered:?}");
    }
    let first = images[0].0;
    assert!(
        hovered.contains(&format!("Click to select {}.", name(first))),
        "{hovered:?}"
    );
    assert_eq!(
        response.unwrap().hovered_image,
        Some(crate::scene::ImageRef::new(node.id, first))
    );

    let mut click = |selected: Option<usize>| {
        let press = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let moved = vec![egui::Event::PointerMoved(at)];
        histogram_frame(&mut panel, &ctx, &node, selected, 482.0, moved);
        histogram_frame(&mut panel, &ctx, &node, selected, 482.0, vec![press(true)]);
        let (_, _, response) =
            histogram_frame(&mut panel, &ctx, &node, selected, 482.0, vec![press(false)]);
        response.select_image
    };
    assert_eq!(
        click(None),
        Some(crate::scene::ImageRef::new(node.id, first))
    );
    assert_eq!(
        click(Some(first)),
        Some(crate::scene::ImageRef::new(node.id, images[1].0))
    );
}
