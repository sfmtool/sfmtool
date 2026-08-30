// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Intrinsics panel.
//!
//! egui lays a frame out with no GPU, so the panel really runs — the tables are
//! really built, the grids really allocate, and the strings the panel decided
//! to paint come back out of the frame's galleys
//! (`test_support::painted_texts`, the `point_track_detail/tests.rs` pattern).
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
    }
}

/// The demo node with its camera swapped for a fisheye — the model family with
/// no `P` and a focal length in pixels per radian.
fn fisheye_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon.cameras[0] = kerry_park_camera();
    node
}

/// A node with two cameras: the demo pinhole for the first four images and a
/// second, single-focal-length model for the rest.
fn two_camera_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon.cameras.push(CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 344.0,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.035,
        },
        width: 270,
        height: 480,
    });
    for image in node.recon.images.iter_mut().skip(4) {
        image.camera_index = 1;
    }
    node
}

/// The demo node under a similarity with a rotation, a translation and a scale
/// change, so a transform that is only partly applied cannot pass.
fn transformed_node() -> SceneNode {
    let mut node = pinhole_node();
    node.transform = Se3Transform::new(
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
    node.recon.images.truncate(8);
    let images = node.recon.images.len();
    node.recon.rig_frame_data = Some(RigFrameData {
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
        panel.show(ui, node, camera, image);
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
    node.recon.cameras[0] = CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 129.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: 1.6,
            bspline: vec![0.0, 0.01, 0.02, 0.0],
        },
        width: 480,
        height: 480,
    };
    let painted = show(&node, Some(0), None);
    assert!(says(&painted, "(beta)"));
}

#[test]
fn the_parameter_table_is_in_declaration_order() {
    // Asserted against `parameter_names()` rather than hard-coded, so a model
    // gaining a parameter cannot silently reorder the table — and so the panel
    // and `sfm inspect` cannot drift into two different orders.
    let node = fisheye_node();
    let painted = show(&node, Some(0), None);
    let declared: Vec<String> = node.recon.cameras[0]
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
    let camera = &node.recon.cameras[0];
    let pose = Pose::resolve(&node, 0, camera, PoseFrame::Stored);
    let p = pose.projection.expect("a pinhole has a P");

    // A point this image actually sees, projected two ways: through `P`, and
    // through the camera's own forward map. They have to agree — that is what
    // makes `P` worth pasting anywhere.
    let world = node.recon.points[0].position;
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
    assert!(says(&with_image, &node.recon.images[3].name));
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
    node.recon.metadata.world_space_unit = Some("m".to_string());
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
    let camera = &node.recon.cameras[0];
    let image = &node.recon.images[0];

    let stored = Pose::resolve(&node, 0, camera, PoseFrame::Stored);
    let moved = Pose::resolve(&node, 0, camera, PoseFrame::NodeTransform);
    assert!(!moved.transformed || stored.translation != moved.translation);

    let (expected_q, expected_t) = node.transform.apply_to_camera_pose(
        &RotQuaternion::from_nalgebra(image.quaternion_wxyz),
        &image.translation_xyz,
    );
    assert!((moved.translation - expected_t).norm() < 1e-12);
    assert!((moved.rotation - expected_q.to_rotation_matrix()).norm() < 1e-12);
    // And the camera centre follows the transform's own action on a point.
    let expected_centre = node.transform.apply_to_point(&stored.centre);
    assert!((moved.centre - expected_centre).norm() < 1e-9);
}

#[test]
fn an_untransformed_node_ignores_the_transformed_frame() {
    let node = pinhole_node();
    let camera = &node.recon.cameras[0];
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
    let camera = &pinhole_node().recon.cameras[0];
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
    node.recon.cameras[0] = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 152.8,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
    };
    node
}

/// A node whose camera is an `EQUIRECTANGULAR` panorama, which is its own
/// reference map rather than a pinhole's or an equidistant fisheye's.
fn equirectangular_node() -> SceneNode {
    let mut node = pinhole_node();
    node.recon.cameras[0] = CameraIntrinsics {
        model: CameraModel::Equirectangular {
            focal_length_x: 2048.0 / std::f64::consts::TAU,
            focal_length_y: 1024.0 / std::f64::consts::PI,
            principal_point_x: 1024.0,
            principal_point_y: 512.0,
        },
        width: 2048,
        height: 1024,
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
    node.recon.cameras[0].width = 0;
    node.recon.cameras[0].height = 0;
    let painted = show(&node, Some(0), None);
    assert!(says(&painted, "No projection to plot"));
    assert!(!says(&painted, "θ off-axis"));
}
