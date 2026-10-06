// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `set_view`: the tool an agent calls immediately before `screenshot`.
//!
//! Framing goes through the paths the keyboard and double-click use
//! (`ViewportCamera::compute_fit` over `scene::FitPoints`,
//! `Viewer3D::jump_to_camera_view`), so the agent's framing is the framing a
//! human gets from the same request.
//!
//! **A view change jumps unless the call asks to animate.** `Viewer3D` eases
//! the camera over roughly 200 ms, and an agent that sets the view and
//! screenshots straight afterward would photograph the middle of the ease, so
//! by default every form lands its end state at once. With `animate` the form
//! still lands its end state first, by the same code, and
//! `Viewer3D::ease_from` then puts the camera back and eases to it: the
//! animated and the instant call end in one place, and a person watching the
//! window sees where the view went. Either way a call cancels any ease already
//! running, so a change the human started does not slide over the top of the
//! one the agent asked for.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use serde_json::json;

use super::{
    render, resolve_camera_image, resolve_point, resolve_reconstruction, Angles, JsonReply,
    Movement, Placement, RelativeView, ToolError, ViewCommand,
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
    animate: bool,
) -> JsonReply {
    viewer.cancel_transition();
    let from = viewer.pose();
    // The run beside the text: an agent animating a path through `place`, or
    // walking the field of view, is one line, while a framing or a
    // look-through is a deliberate act that keeps its own.
    let (run, what) = match view {
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
            (None, what)
        }
        ViewCommand::LookThrough {
            reconstruction_label,
            camera_image,
        } => {
            let id = resolve_reconstruction(state, reconstruction_label.as_deref())?;
            let image = resolve_camera_image(state, id, &camera_image)?;
            let node = state.node(id).expect("just resolved");
            let name = node.recon().image_table.images[image.index()].name.clone();
            viewer.jump_to_camera_view(image, node);
            (None, format!("Looking through {name}"))
        }
        ViewCommand::ExitCameraView => {
            viewer.camera_view = None;
            (None, "Left camera view".to_string())
        }
        ViewCommand::Point(query) => (None, centre_on_point(state, viewer, &query)?),
        ViewCommand::BenchObservation {
            reconstruction_label,
            track,
            observation,
        } => {
            let id = resolve_reconstruction(state, reconstruction_label.as_deref())?;
            let (image, pixel) =
                super::bench::observation_place(state, id, track.as_deref(), observation)?;
            let node = state.node(id).expect("just resolved");
            let name = node.recon().image_table.images[image.index()].name.clone();
            viewer.jump_through_toward_feature(image, node, pixel);
            (
                None,
                format!("Looking through {name} toward bench observation {observation}"),
            )
        }
        ViewCommand::Place(placement) => (Some("camera"), place(viewer, placement)?),
        ViewCommand::Relative(relative_view) => {
            (Some("camera"), relative(state, viewer, &relative_view)?)
        }
        ViewCommand::Fov { fov_short_axis_deg } => {
            set_fov(viewer, Some(fov_short_axis_deg))?;
            (
                Some("field of view"),
                format!("Field of view {fov_short_axis_deg:.1}°"),
            )
        }
    };

    // The one entry `set_view` writes, in the catalogue's own words: every
    // form ends here, and `jump_to_camera_view` records nothing of its own
    // so that a look-through is one line and not two.
    match run {
        Some(run) => state.action_log.record_run(Kind::View, run, what),
        None => state.action_log.record(Kind::View, what),
    }
    // The reply is the view the call ends at, read before an ease puts the
    // camera back where it started.
    let reply = json!({ "view": render::view(state, viewer) });
    if animate {
        viewer.ease_from(from);
    }
    Ok(reply)
}

/// Put `query`'s point in the middle of the view, by the gesture a double-click
/// on a tracked feature in Image Detail makes: the camera turns first when the
/// point is far from the middle, then moves sideways onto it
/// (`Viewer3D::turn_and_move_target_to`). A point at infinity is turned toward
/// (`Viewer3D::turn_toward_bearing`).
///
/// The gesture's own method starts its transition and this lands it at once,
/// so the end state is computed by one piece of code for the human and the
/// agent alike.
fn centre_on_point(
    state: &AppState,
    viewer: &mut Viewer3D,
    query: &crate::goto_point::PointQuery,
) -> Result<String, ToolError> {
    let point = resolve_point(state, query)?;
    let id = state
        .node(point.recon)
        .map(|node| crate::scene::point_id(node, point.index()))
        .unwrap_or_else(|| format!("#{}", point.index()));
    let text = match crate::scene::world_point(&state.scene, point) {
        Some(crate::scene::WorldPoint::At(position)) => {
            if !viewer.turn_and_move_target_to(position, 0.0) {
                return Err(ToolError::new(format!(
                    "Point {id} cannot be brought to the middle of the view from here."
                )));
            }
            format!("Centred on point {id}")
        }
        Some(crate::scene::WorldPoint::Toward(direction)) => {
            // A bearing already in the middle starts no turn, which is the
            // answer too: it is where the call asked for it to be.
            viewer.turn_toward_bearing(direction, 0.0);
            format!("Turned toward point {id}, which is at infinity")
        }
        None => {
            return Err(ToolError::new(format!(
                "Point {id} is not in the version on screen."
            )))
        }
    };
    viewer.finish_transition();
    Ok(text)
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
    // A stated up that is not +Z asks for a rolled view, as Q and E do, so it
    // turns Maintain Z-up off rather than being turned back over the next
    // second.
    if (placement.up.is_some() || placement.world_up.is_some())
        && world_up.angle(&Vector3::z()) > 1e-9
    {
        viewer.maintain_z_up = false;
    }
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

/// The relative forms, from the view as it stands: the move, then the turn,
/// then the orbit, each from where the one before left the camera.
///
/// That order is the order an instruction is read in ("step back 2 m, then
/// turn right 35°"), and it puts each form where its axes are the ones the
/// caller saw: the move goes along the direction the camera faced when the
/// call arrived, the turn happens at the place the move reached, and the orbit
/// swings around the target the move and the turn left in front of the camera.
///
/// - **move** goes along the level axes (`ViewportCamera::move_level`):
///   forward is the view direction laid flat on the XY plane, right is level,
///   and up is +Z. A distance is in `move.unit` converted to the view's unit
///   (`render::view_world_space_unit`), or in the view's unit where the call
///   names none. It leaves camera view, as a fly key does.
/// - **turn** is the free look (`ViewportCamera::nodal_pan_by`), about
///   `world_up`, and keeps camera view, as a drag in camera view does.
/// - **orbit** is the orbit (`ViewportCamera::orbit_by`), about `world_up`
///   through the target, and leaves camera view, as an Alt-drag in camera view
///   does.
///
/// Everything that can refuse -- a unit the view cannot convert to, a field of
/// view out of range -- is checked before anything moves.
fn relative(
    state: &AppState,
    viewer: &mut Viewer3D,
    relative: &RelativeView,
) -> Result<String, ToolError> {
    let movement = match &relative.movement {
        Some(movement) => Some((movement, move_unit(state, movement)?)),
        None => None,
    };
    // Refuses before it writes, and the field of view enters none of the
    // arithmetic below, so it can go first.
    set_fov(viewer, relative.fov_short_axis_deg)?;

    let mut done: Vec<String> = Vec::new();
    if let Some((movement, (factor, unit))) = movement {
        viewer.camera_view = None;
        viewer.camera.move_level(
            movement.forward * factor,
            movement.right * factor,
            movement.up * factor,
        );
        done.push(describe_move(movement, &unit));
    }
    if let Some(turn) = &relative.turn {
        viewer
            .camera
            .nodal_pan_by(turn.yaw_deg.to_radians(), turn.pitch_deg.to_radians());
        done.push(describe_angles("turned", turn, ("left", "right")));
    }
    if let Some(orbit) = &relative.orbit {
        viewer.camera_view = None;
        viewer
            .camera
            .orbit_by(orbit.yaw_deg.to_radians(), orbit.pitch_deg.to_radians());
        done.push(describe_angles(
            "orbited",
            orbit,
            ("counter-clockwise", "clockwise"),
        ));
    }
    if let Some(fov) = relative.fov_short_axis_deg {
        done.push(format!("field of view {fov:.1}°"));
    }
    let mut text = done.join(", then ");
    if let Some(first) = text.get(..1).map(str::to_uppercase) {
        text.replace_range(..1, &first);
    }
    Ok(text)
}

/// The factor that takes `movement`'s distances to the view's unit, and the
/// unit they were given in as the Action Log names it.
///
/// Without `move.unit` the distances are already in the view's unit, known or
/// not. With one, the view must be in a physical unit too, or there is nothing
/// to convert to.
fn move_unit(state: &AppState, movement: &Movement) -> Result<(f64, String), ToolError> {
    let view_unit = render::view_world_space_unit(state);
    match (movement.unit, view_unit) {
        (None, Ok(view)) => Ok((1.0, view.to_string())),
        (None, Err(_)) => Ok((1.0, "scene units".to_string())),
        (Some(unit), Ok(view)) => {
            let metres = |name: &str| {
                sfmtool_core::world_space_unit_in_metres(name)
                    .expect("both units were checked against the table")
            };
            Ok((metres(unit) / metres(view), unit.to_string()))
        }
        (Some(unit), Err(reason)) => Err(ToolError::new(format!(
            "move.unit {unit:?} needs the view to be in a physical unit, and it is in scene \
             units: {reason}, so there is nothing to convert {unit} to. Send the distances \
             without unit, in scene units, or give the reconstruction a physical unit first \
             with sfm xform --scale-by-measurements."
        ))),
    }
}

/// "moved 2 m back and 1 m up", naming only the axes the move went along.
fn describe_move(movement: &Movement, unit: &str) -> String {
    let parts: Vec<String> = [
        (movement.forward, "forward", "back"),
        (movement.right, "right", "left"),
        (movement.up, "up", "down"),
    ]
    .into_iter()
    .filter(|(distance, _, _)| *distance != 0.0)
    .map(|(distance, positive, negative)| {
        let way = if distance > 0.0 { positive } else { negative };
        let amount = amount(distance.abs());
        let unit = match (unit, amount.as_str()) {
            ("scene units", "1") => "scene unit",
            _ => unit,
        };
        format!("{amount} {unit} {way}")
    })
    .collect();
    if parts.is_empty() {
        "moved nowhere".to_string()
    } else {
        format!("moved {}", parts.join(" and "))
    }
}

/// "turned 35° right and 10° up", naming only the angles that were turned
/// through. `yaw_words` names a positive yaw, then a negative one.
fn describe_angles(verb: &str, angles: &Angles, yaw_words: (&str, &str)) -> String {
    let parts: Vec<String> = [
        (angles.yaw_deg, yaw_words.0, yaw_words.1),
        (angles.pitch_deg, "up", "down"),
    ]
    .into_iter()
    .filter(|(degrees, _, _)| *degrees != 0.0)
    .map(|(degrees, positive, negative)| {
        let way = if degrees > 0.0 { positive } else { negative };
        format!("{}° {way}", amount(degrees.abs()))
    })
    .collect();
    if parts.is_empty() {
        format!("{verb} through no angle")
    } else {
        format!("{verb} {}", parts.join(" and "))
    }
}

/// A distance or an angle as a person would write it: up to three decimals,
/// with no trailing zeros (`2`, `0.5`, `1.234`).
fn amount(value: f64) -> String {
    let rounded = (value * 1000.0).round() / 1000.0;
    format!("{rounded}")
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
/// Fits over `scene::FitPoints` — the node's points put *through its
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
            (crate::scene::FitPoints::of(node), format!("Framed {label}"))
        }
        None => {
            let mut points = crate::scene::FitPoints::default();
            for node in state
                .scene
                .iter()
                .filter(|node| crate::scene::is_visible(node, state.solo))
            {
                points.extend(crate::scene::FitPoints::of(node));
            }
            (points, "Framed the scene".to_string())
        }
    };
    let end = viewer
        .camera
        .compute_fit(&points, aspect, viewer.maintain_z_up)
        .ok_or_else(|| ToolError::new("Nothing is drawn — there are no points to frame."))?;
    viewer.camera.apply_fit(&end);
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
