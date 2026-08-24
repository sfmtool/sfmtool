// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The selected image's pose, the frame it is shown in, and the rig block.
//!
//! This is the block most likely to be pasted into somebody else's code, so it
//! is explicit about two things that are otherwise invisible:
//!
//! - **Which frame.** The stored pose is world-to-camera in the canonical
//!   `.sfmr` convention — camera looks down `−Z` with `+Y` up — and a node that
//!   has been through `Align to…` is *drawn* somewhere else entirely. The panel
//!   shows the stored pose by default and offers the other behind a toggle that
//!   only appears when the two differ.
//! - **Which projection.** `intrinsic_matrix` is `K` in the **optical** frame
//!   (`+Z` forward, `+Y` down), so the projection matrix is `P = K · S · [R|t]`
//!   with `S = diag(1, −1, −1)` — not `K · [R|t]`, which is a plausible-looking
//!   matrix that is silently wrong. And for a fisheye or equirectangular model
//!   there is no `P` at all: that row is *replaced* by the statement, never
//!   omitted and never printed anyway.

use nalgebra::{Matrix3, Matrix3x4, Point3, Vector3};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::{RotQuaternion, SfmrReconstruction};

use super::format;
use super::PoseFrame;
use crate::scene::SceneNode;

/// The optical-frame flip `S = diag(1, −1, −1)`, which takes the canonical
/// camera frame (`−Z` forward, `+Y` up) to the optical one `K` is written in.
fn optical_flip() -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
}

/// One image's pose, resolved in the frame the panel is showing it in.
pub(super) struct Pose {
    /// The image's name, as the reconstruction stores it.
    pub name: String,
    /// World-to-camera rotation `R`.
    pub rotation: Matrix3<f64>,
    /// World-to-camera translation `t`.
    pub translation: Vector3<f64>,
    /// `(w, x, y, z)` of the same rotation.
    pub quaternion_wxyz: [f64; 4],
    /// Camera centre in world coordinates, `C = −Rᵀt`.
    pub centre: Point3<f64>,
    /// `P = K · S · [R|t]`, or `None` for a model that is not a linear
    /// projection.
    pub projection: Option<Matrix3x4<f64>>,
    /// Whether this is the stored pose put through the node's transform.
    pub transformed: bool,
}

impl Pose {
    /// Resolve image `index`'s pose in `frame`.
    ///
    /// `frame` is honoured only when the node actually carries a transform;
    /// asking for the transformed pose of an untransformed node yields the
    /// stored one, unmarked, because the two are the same pose.
    pub(super) fn resolve(
        node: &SceneNode,
        index: usize,
        camera: &CameraIntrinsics,
        frame: PoseFrame,
    ) -> Self {
        let image = &node.recon.images[index];
        let transformed = frame == PoseFrame::NodeTransform && node.has_transform();
        let (quaternion, translation) = if transformed {
            node.transform.apply_to_camera_pose(
                &RotQuaternion::from_nalgebra(image.quaternion_wxyz),
                &image.translation_xyz,
            )
        } else {
            (
                RotQuaternion::from_nalgebra(image.quaternion_wxyz),
                image.translation_xyz,
            )
        };
        let rotation = quaternion.to_rotation_matrix();
        let centre = Point3::from(-(rotation.transpose() * translation));
        let projection = (!camera.model.needs_ray_path()).then(|| {
            let mut rt = Matrix3x4::zeros();
            rt.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
            rt.fixed_view_mut::<3, 1>(0, 3).copy_from(&translation);
            camera.intrinsic_matrix() * optical_flip() * rt
        });
        Self {
            name: image.name.clone(),
            rotation,
            translation,
            quaternion_wxyz: quaternion.to_wxyz_array(),
            centre,
            projection,
            transformed,
        }
    }

    /// `[R | t]` as three rows of four, the shape the panel draws and the
    /// clipboard receives.
    pub(super) fn pose_matrix(&self) -> Vec<Vec<f64>> {
        (0..3)
            .map(|r| {
                vec![
                    self.rotation[(r, 0)],
                    self.rotation[(r, 1)],
                    self.rotation[(r, 2)],
                    self.translation[r],
                ]
            })
            .collect()
    }
}

/// Draw the extrinsics section, returning whether the image name was clicked.
pub(super) fn show_extrinsics(
    ui: &mut egui::Ui,
    node: &SceneNode,
    index: usize,
    pose: &Pose,
    frame: &mut PoseFrame,
) -> bool {
    // The toggle sits above the header it qualifies, and only exists when
    // there are two answers: showing one of them silently would make the panel
    // wrong half the time for anyone who has used `Align to…`, in a way nobody
    // would notice.
    if node.has_transform() {
        ui.horizontal(|ui| {
            ui.label("Frame:");
            ui.selectable_value(frame, PoseFrame::Stored, "stored")
                .on_hover_text("The pose as the .sfmr stores it");
            ui.selectable_value(frame, PoseFrame::NodeTransform, "× node transform")
                .on_hover_text("The stored pose put through this node's alignment transform");
        });
    }

    let mut clicked = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Pose ·").strong());
        clicked = ui
            .link(egui::RichText::new(&pose.name).strong())
            .on_hover_text(&pose.name)
            .clicked();
        if pose.transformed {
            ui.label(egui::RichText::new("(transformed)").weak());
        }
    });
    ui.label(
        egui::RichText::new(
            "world-to-camera, canonical .sfmr frame: Z-up world, camera looks down −Z \
             with +Y up (specs/formats/sfmr-file-format.md)",
        )
        .weak()
        .small(),
    );
    ui.add_space(4.0);

    ui.label("Rotation R");
    let r_rows: Vec<Vec<f64>> = (0..3)
        .map(|r| (0..3).map(|c| pose.rotation[(r, c)]).collect())
        .collect();
    format::matrix_grid(ui, "intrinsics_pose_r", &r_rows);
    format::labelled_row(ui, "Translation t", format::xyz(&pose.translation));
    ui.add_space(4.0);

    ui.label("[ R | t ]");
    format::matrix_grid(ui, "intrinsics_pose_rt", &pose.pose_matrix());
    ui.add_space(4.0);

    format::labelled_row(ui, "Quaternion w x y z", pose.quaternion_wxyz);
    let centre_label = match node.recon.metadata.world_space_unit.as_deref() {
        Some(unit) => format!("Camera centre C ({unit})"),
        None => "Camera centre C".to_string(),
    };
    format::labelled_row(ui, &centre_label, format::xyz(&pose.centre.coords));

    // The rows of `R` are the camera axes in world coordinates, so `Rᵀe₀` is
    // its first row — `right = Rᵀe₀`, `up = Rᵀe₁`, and forward is `−Rᵀe₂`,
    // since the canonical frame looks down its own `−Z`.
    let axis = |c: usize, sign: f64| -> [f64; 3] {
        [
            sign * pose.rotation[(c, 0)],
            sign * pose.rotation[(c, 1)],
            sign * pose.rotation[(c, 2)],
        ]
    };
    ui.add_space(4.0);
    ui.label("Axes in world");
    format::labelled_row(ui, "right  ", axis(0, 1.0));
    format::labelled_row(ui, "up     ", axis(1, 1.0));
    format::labelled_row(ui, "forward", axis(2, -1.0));

    ui.add_space(4.0);
    match &pose.projection {
        Some(p) => {
            ui.label("P = K · S · [ R | t ]");
            let p_rows: Vec<Vec<f64>> = (0..3)
                .map(|r| (0..4).map(|c| p[(r, c)]).collect())
                .collect();
            format::matrix_grid(ui, "intrinsics_pose_p", &p_rows);
        }
        None => {
            ui.label(
                egui::RichText::new("Not a linear projection — this model has no 3×4 P").weak(),
            );
        }
    }

    show_rig_block(ui, &node.recon, index);
    clicked
}

/// The rig block: which rig, which sensor, which frame, and the sensor's
/// offset from the rig origin.
///
/// Drawn only for a reconstruction that carries `rig_frame_data` — the part of
/// "extrinsics" a rig dataset actually needs, and that nothing else in the
/// viewer surfaces.
fn show_rig_block(ui: &mut egui::Ui, recon: &SfmrReconstruction, index: usize) {
    let Some(rig) = &recon.rig_frame_data else {
        return;
    };
    let (Some(&sensor), Some(&frame)) = (
        rig.image_sensor_indexes.get(index),
        rig.image_frame_indexes.get(index),
    ) else {
        return;
    };
    let sensor = sensor as usize;
    let frame = frame as usize;

    // `image_sensor_indexes` is a *global* sensor index while `sensor_names` is
    // per rig, so the name lives at `sensor − sensor_offset` of the rig whose
    // span contains it. Found by span rather than through the frame's
    // `rig_indexes`, so a file whose two disagree still names the sensor it
    // actually has rather than indexing into the wrong rig's list.
    let definition = rig
        .rigs_metadata
        .rigs
        .iter()
        .find(|def| {
            let start = def.sensor_offset as usize;
            sensor >= start && sensor < start + def.sensor_count as usize
        })
        .or_else(|| {
            rig.rig_indexes
                .get(frame)
                .and_then(|&r| rig.rigs_metadata.rigs.get(r as usize))
        });

    let sensor_name = definition.and_then(|def| {
        def.sensor_names
            .get(sensor.saturating_sub(def.sensor_offset as usize))
            .cloned()
    });
    let is_reference = definition
        .zip(sensor_name.as_deref())
        .is_some_and(|(def, name)| def.ref_sensor_name == name);
    let frame_images = rig
        .image_frame_indexes
        .iter()
        .filter(|&&f| f as usize == frame)
        .count();
    let sensor_from_rig = (sensor < rig.sensor_quaternions_wxyz.nrows()
        && sensor < rig.sensor_translations_xyz.nrows())
    .then(|| {
        let q = rig.sensor_quaternions_wxyz.row(sensor);
        let t = rig.sensor_translations_xyz.row(sensor);
        ([q[0], q[1], q[2], q[3]], [t[0], t[1], t[2]])
    });

    ui.separator();
    // "Rig and frame" rather than "Rig", so the section heading and the row
    // that names the rig are not the same word twice running.
    ui.label(egui::RichText::new("Rig and frame").strong());
    egui::Grid::new("intrinsics_rig")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            ui.label("Rig");
            ui.label(definition.map_or_else(|| "—".to_string(), |def| def.name.clone()));
            ui.end_row();

            ui.label("Sensor");
            let name = sensor_name.clone().unwrap_or_else(|| "—".to_string());
            // The marker rides the *sensor* row rather than the rig row: it is
            // a statement about which sensor this image came from, and the rig
            // row names the rig.
            let marker = if is_reference {
                " (reference sensor)"
            } else {
                ""
            };
            ui.label(format!("{name}  ·  index {sensor}{marker}"));
            ui.end_row();

            ui.label("Frame");
            let images = if frame_images == 1 { "image" } else { "images" };
            ui.label(format!("{frame}  ·  {frame_images} {images}"));
            ui.end_row();

            ui.label("sensor_from_rig");
            show_sensor_from_rig(ui, sensor_from_rig, is_reference);
            ui.end_row();
        });
}

/// The sensor's pose in the rig, or the statement that it is the identity.
///
/// The reference sensor's `sensor_from_rig` *is* the identity by construction,
/// so it says so rather than printing four ones and three zeros. Verified
/// rather than assumed: a file that stores something else for its reference
/// sensor gets the numbers, since claiming an identity that is not there would
/// hide exactly the corruption worth seeing.
fn show_sensor_from_rig(
    ui: &mut egui::Ui,
    sensor_from_rig: Option<([f64; 4], [f64; 3])>,
    is_reference: bool,
) {
    let Some((quaternion, translation)) = sensor_from_rig else {
        ui.label("—");
        return;
    };
    let identity = (quaternion[0].abs() - 1.0).abs() < 1e-12
        && quaternion[1..].iter().all(|v| v.abs() < 1e-12)
        && translation.iter().all(|v| v.abs() < 1e-12);
    if is_reference && identity {
        ui.label("identity (reference sensor)");
        return;
    }
    ui.vertical(|ui| {
        format::labelled_row(ui, "q wxyz", quaternion);
        format::labelled_row(ui, "t xyz ", translation);
    });
}
