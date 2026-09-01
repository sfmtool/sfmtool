// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `set_view`: the tool an agent calls immediately before `screenshot`.
//!
//! Framing goes through the paths the keyboard and double-click use
//! (`ViewportCamera::zoom_to_fit` over `scene::world_points`,
//! `Viewer3D::jump_to_camera_view`), so the agent's framing is the framing a
//! human gets from the same request.
//!
//! **The animated transition is skipped throughout.** `Viewer3D` eases the
//! camera over roughly 200 ms, and an agent that sets the view and screenshots
//! straight afterward would photograph the middle of the ease. MCP view
//! commands jump, and cancel any ease already running so a change the human
//! started does not slide over the top of the one the agent asked for.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use serde_json::json;

use super::{
    announce, render, resolve_camera_image, resolve_reconstruction, JsonReply, ToolError,
    ViewCommand,
};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

/// The narrowest field of view `set_view` will accept, in degrees, matching
/// what `ViewportCamera::zoom_fov` clamps interactive zoom to.
const MIN_FOV_DEG: f64 = 5.0;

/// The widest, likewise.
const MAX_FOV_DEG: f64 = 160.0;

pub(super) fn set_view(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    view: ViewCommand,
) -> JsonReply {
    viewer.cancel_transition();
    let what = match view {
        ViewCommand::Fit {
            reconstruction_label,
        } => fit(state, viewer, reconstruction_label.as_deref())?,
        ViewCommand::LookThrough {
            reconstruction_label,
            camera_image,
        } => {
            let id = resolve_reconstruction(state, reconstruction_label.as_deref())?;
            let image = resolve_camera_image(state, id, &camera_image)?;
            let node = state.node(id).expect("just resolved");
            let name = node.recon.images[image.index()].name.clone();
            viewer.jump_to_camera_view(image, node);
            format!("looking through {name}")
        }
        ViewCommand::ExitCameraView => {
            viewer.camera_view = None;
            "left camera view".to_string()
        }
        ViewCommand::LookAt {
            position,
            target,
            up,
            fov_short_axis_deg,
        } => {
            let position = Point3::new(position[0], position[1], position[2]);
            let target = Point3::new(target[0], target[1], target[2]);
            let distance = (target - position).norm();
            if !distance.is_finite() || distance <= 0.0 {
                return Err(ToolError::new(
                    "position and target are the same point — the view has no direction.",
                ));
            }
            // `up` is the roll, and defaults to the roll the view already has.
            // Supplying a different one re-rolls the view exactly as
            // `ViewportCamera::tilt` does, which is why it is written to
            // `world_up` and not merely used to build the orientation.
            let world_up = match up {
                Some(up) => normalized(up, "up")?,
                None => viewer.camera.world_up,
            };
            let forward = (target - position).normalize();
            if forward.cross(&world_up).norm() < 1e-9 {
                return Err(ToolError::new(
                    "up is parallel to the view direction — the roll is undefined.",
                ));
            }
            // Leaving camera view: this is a free camera placement, and the
            // background image belongs to a viewpoint that has just been left.
            viewer.camera_view = None;
            viewer.camera.world_up = world_up;
            viewer.camera.camera.position = position;
            viewer.camera.camera.target_distance = distance;
            viewer.camera.set_orientation_from_forward(forward);
            set_fov(viewer, fov_short_axis_deg)?;
            "camera placed".to_string()
        }
        ViewCommand::Exact {
            position,
            orientation_wxyz,
            target_distance,
            world_up,
            fov_short_axis_deg,
        } => {
            let orientation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                orientation_wxyz[0],
                orientation_wxyz[1],
                orientation_wxyz[2],
                orientation_wxyz[3],
            ));
            if !orientation_wxyz.iter().all(|c| c.is_finite())
                || nalgebra::Vector4::from(orientation_wxyz).norm() < 1e-9
            {
                return Err(ToolError::new(
                    "orientation_wxyz is not a rotation — expected four finite numbers that are \
                     not all zero.",
                ));
            }
            if !target_distance.is_finite() || target_distance <= 0.0 {
                return Err(ToolError::new("target_distance must be greater than zero."));
            }
            viewer.camera_view = None;
            viewer.camera.camera.position = Point3::new(position[0], position[1], position[2]);
            viewer.camera.camera.orientation = orientation;
            viewer.camera.camera.target_distance = target_distance;
            if let Some(world_up) = world_up {
                viewer.camera.world_up = normalized(world_up, "world_up")?;
            }
            set_fov(viewer, fov_short_axis_deg)?;
            "camera restored".to_string()
        }
        ViewCommand::Fov { fov_short_axis_deg } => {
            set_fov(viewer, Some(fov_short_axis_deg))?;
            format!("field of view {fov_short_axis_deg:.1}°")
        }
    };

    announce(state, format!("view — {what}"));
    Ok(json!({ "view": render::view(state, viewer) }))
}

/// Frame everything drawn, or one named reconstruction.
///
/// Fits over `scene::world_points` — the node's points put *through its
/// transform* — so an aligned reconstruction is framed where it is drawn rather
/// than where its own coordinates say it is.
fn fit(state: &AppState, viewer: &mut Viewer3D, label: Option<&str>) -> Result<String, ToolError> {
    let aspect = viewer.panel_aspect().ok_or_else(|| {
        ToolError::new(
            "The 3D viewport has not been laid out yet — there is no aspect ratio to frame \
             against.",
        )
    })?;
    let (points, what) = match label {
        Some(label) => {
            let id = resolve_reconstruction(state, Some(label))?;
            let node = state.node(id).expect("just resolved");
            (crate::scene::world_points(node), format!("framed {label}"))
        }
        None => {
            let points: Vec<Point3<f64>> = state
                .scene
                .iter()
                .filter(|node| crate::scene::is_visible(node, state.solo))
                .flat_map(crate::scene::world_points)
                .collect();
            (points, "framed the scene".to_string())
        }
    };
    if points.is_empty() {
        return Err(ToolError::new(
            "Nothing is drawn — there are no points to frame.",
        ));
    }
    viewer.camera.zoom_to_fit(&points, aspect);
    Ok(what)
}

/// Apply a field of view, if the call carried one.
fn set_fov(viewer: &mut Viewer3D, degrees: Option<f64>) -> Result<(), ToolError> {
    let Some(degrees) = degrees else {
        return Ok(());
    };
    if !(MIN_FOV_DEG..=MAX_FOV_DEG).contains(&degrees) {
        return Err(ToolError::new(format!(
            "fov_short_axis_deg must be between {MIN_FOV_DEG} and {MAX_FOV_DEG} degrees — got \
             {degrees}."
        )));
    }
    viewer.camera.fov = degrees.to_radians();
    Ok(())
}

/// A direction argument as a unit vector, or a refusal naming the field.
fn normalized(v: [f64; 3], field: &str) -> Result<Vector3<f64>, ToolError> {
    let v = Vector3::new(v[0], v[1], v[2]);
    let norm = v.norm();
    if !norm.is_finite() || norm < 1e-9 {
        return Err(ToolError::new(format!(
            "{field} has no direction — expected a non-zero vector."
        )));
    }
    Ok(v / norm)
}
