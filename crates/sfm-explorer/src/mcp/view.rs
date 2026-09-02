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
    render, resolve_camera_image, resolve_reconstruction, JsonReply, Placement, ToolError,
    ViewCommand,
};
use crate::action_log::Kind;
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
        } => {
            let what = fit(state, viewer, reconstruction_label.as_deref())?;
            // Leaving camera view: framing is a statement about the free
            // camera, and the Z key's own fit ends its animated transition
            // with the camera view dropped. The MCP form jumps past that
            // transition, so it lands the same state directly (a fit that
            // left the render inside a camera view would frame nothing the
            // caller can see).
            viewer.camera_view = None;
            what
        }
        ViewCommand::LookThrough {
            reconstruction_label,
            camera_image,
        } => {
            let id = resolve_reconstruction(state, reconstruction_label.as_deref())?;
            let image = resolve_camera_image(state, id, &camera_image)?;
            let node = state.node(id).expect("just resolved");
            let name = node.recon.images[image.index()].name.clone();
            viewer.jump_to_camera_view(image, node);
            format!("Looking through {name}")
        }
        ViewCommand::ExitCameraView => {
            viewer.camera_view = None;
            "Left camera view".to_string()
        }
        ViewCommand::Place(placement) => place(viewer, placement)?,
        ViewCommand::Fov { fov_short_axis_deg } => {
            set_fov(viewer, Some(fov_short_axis_deg))?;
            format!("Field of view {fov_short_axis_deg:.1}°")
        }
    };

    // The one entry `set_view` writes, in the catalogue's own words: the five
    // forms all end here, and `jump_to_camera_view` records nothing of its own
    // so that a look-through is one line and not two.
    state.action_log.record(Kind::View, what);
    Ok(json!({ "view": render::view(state, viewer) }))
}

/// Place the explicit camera from the pieces one call carried, preserving
/// every piece it did not.
///
/// One path for the whole explicit family, because the look-at form, the exact
/// form and a lone `forward` differ only in where the three unknowns come
/// from. They are resolved in turn:
///
/// - the **orientation**, from `orientation_wxyz`, from `forward`, from the
///   direction `position` to `target`, or standing;
/// - the **distance**, from `target_distance`, from the separation of
///   `position` and `target`, or standing;
/// - the **anchor** the view is hung from -- `target` where the call named
///   one, else `position` where it named one, else the standing orbit target,
///   which is the same `Camera::target()` the view block reports as
///   `derived.target`. The other end of the view follows from the anchor, the
///   orientation and the distance.
///
/// So `forward` alone swings the camera around what it is looking at rather
/// than turning it in place, `target_distance` alone dollies toward a fixed
/// target, and `target` alone re-centres the view without re-aiming it.
fn place(viewer: &mut Viewer3D, placement: Placement) -> Result<String, ToolError> {
    // Both ends given: their difference is the one thing that is degenerate if
    // they coincide, so it is checked once here and then serves as both the
    // direction and the distance.
    let separation = match (placement.position, placement.target) {
        (Some(position), Some(target)) => {
            let separation = point(target) - point(position);
            let distance = separation.norm();
            if !distance.is_finite() || distance <= 0.0 {
                return Err(ToolError::new(
                    "position and target are the same point — the view has no direction.",
                ));
            }
            Some((separation / distance, distance))
        }
        _ => None,
    };

    // The roll. `up` and `world_up` are the same quantity named for the two
    // forms that carry it, and a supplied one re-rolls the view exactly as
    // `ViewportCamera::tilt` does, which is why it is written to `world_up`
    // and not merely used to build the orientation.
    let mut world_up = viewer.camera.world_up;
    if let Some(up) = placement.up {
        world_up = normalized(up, "up")?;
    } else if let Some(up) = placement.world_up {
        world_up = normalized(up, "world_up")?;
    }

    let facing = match (placement.orientation_wxyz, placement.forward, separation) {
        (Some(wxyz), _, _) => {
            if !wxyz.iter().all(|c| c.is_finite()) || nalgebra::Vector4::from(wxyz).norm() < 1e-9 {
                return Err(ToolError::new(
                    "orientation_wxyz is not a rotation — expected four finite numbers that are \
                     not all zero.",
                ));
            }
            Facing::Stated(UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                wxyz[0], wxyz[1], wxyz[2], wxyz[3],
            )))
        }
        (None, Some(forward), _) => Facing::Derived(normalized(forward, "forward")?),
        (None, None, Some((forward, _))) => Facing::Derived(forward),
        (None, None, None) => Facing::Stated(viewer.camera.camera.orientation),
    };
    if let Facing::Derived(forward) = facing {
        if forward.cross(&world_up).norm() < 1e-9 {
            return Err(ToolError::new(
                "up is parallel to the view direction — the roll is undefined.",
            ));
        }
    }

    let distance = match (placement.target_distance, separation) {
        (Some(distance), _) => {
            if !distance.is_finite() || distance <= 0.0 {
                return Err(ToolError::new("target_distance must be greater than zero."));
            }
            distance
        }
        (None, Some((_, distance))) => distance,
        (None, None) => viewer.camera.camera.target_distance,
    };

    // Read before anything moves: the anchor of a call that named neither end
    // is where the camera is looking *now*.
    let standing_target = viewer.camera.camera.target();
    // Leaving camera view: this is a free camera placement, and the background
    // image belongs to a viewpoint that has just been left.
    viewer.camera_view = None;
    viewer.camera.world_up = world_up;
    match facing {
        Facing::Stated(orientation) => viewer.camera.camera.orientation = orientation,
        // Goes through the camera's own derivation, which reads the `world_up`
        // just written, so a derived view rolls the way the mouse rolls it.
        Facing::Derived(forward) => viewer.camera.set_orientation_from_forward(forward),
    }
    viewer.camera.camera.target_distance = distance;
    viewer.camera.camera.position = match (placement.position, placement.target) {
        // A stated position is taken verbatim rather than reconstructed from
        // the anchor, so a view read out of `get_scene` comes back bit for bit.
        (Some(position), _) => point(position),
        (None, target) => {
            let anchor = target.map_or(standing_target, point);
            anchor - viewer.camera.camera.forward() * distance
        }
    };
    set_fov(viewer, placement.fov_short_axis_deg)?;

    Ok(if placement.orientation_wxyz.is_some() {
        "Camera restored".to_string()
    } else {
        "Camera placed".to_string()
    })
}

/// Which way the camera ends up facing, and how that was arrived at.
///
/// The two are not interchangeable at the moment of assignment: a stated
/// rotation is written straight to the camera, while a direction has to go
/// through `ViewportCamera::set_orientation_from_forward` so that the roll in
/// `world_up` completes it.
enum Facing {
    Stated(UnitQuaternion<f64>),
    Derived(Vector3<f64>),
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
            (crate::scene::world_points(node), format!("Framed {label}"))
        }
        None => {
            let points: Vec<Point3<f64>> = state
                .scene
                .iter()
                .filter(|node| crate::scene::is_visible(node, state.solo))
                .flat_map(crate::scene::world_points)
                .collect();
            (points, "Framed the scene".to_string())
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

/// A point argument as a point.
fn point(v: [f64; 3]) -> Point3<f64> {
    Point3::new(v[0], v[1], v[2])
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
