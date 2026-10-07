// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ── set_view ────────────────────────────────────────────────────────────

#[test]
fn looking_through_a_camera_image_reports_which_one() {
    let (mut state, mut viewer) = two_reconstructions();
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    );
    assert_eq!(out["view"]["looking_through"]["camera_image_index"], 3);
    assert_eq!(out["view"]["looking_through"]["name"], "images/A_003.jpg");

    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "exit_camera_view": true }),
    );
    assert_eq!(out["view"]["looking_through"], Value::Null);
}

/// `fit` is a statement about the free camera. The Z key's own fit ends its
/// animated transition with the camera view dropped; the MCP form jumps past
/// that transition, so it must land the same state directly.
#[test]
fn a_fit_leaves_camera_view() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    );
    let out = call(&mut state, &mut viewer, "set_view", json!({ "fit": null }));
    assert_eq!(out["view"]["looking_through"], Value::Null);
}

/// The camera must not still be easing when the reply comes back, or an agent
/// that screenshots next photographs the middle of the transition.
#[test]
fn a_view_change_lands_immediately() {
    let (mut state, mut viewer) = two_reconstructions();
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [2.0, -3.0, 1.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0] }),
    );
    assert_eq!(out["view"]["position"], json!([2.0, -3.0, 1.0]));
    let target = out["view"]["derived"]["target"]
        .as_array()
        .expect("a target");
    for (axis, value) in target.iter().enumerate() {
        let value = value.as_f64().expect("a number");
        assert!(value.abs() < 1e-9, "target axis {axis} was {value}");
    }
}

#[test]
fn the_view_forms_are_exclusive() {
    let map: Map<String, Value> = json!({ "fit": null, "exit_camera_view": true })
        .as_object()
        .cloned()
        .expect("an object");
    let error = tools::parse("set_view", Some(&map)).expect_err("rejected");
    assert!(error.0.contains("exclusive"), "{error}");
}

#[test]
fn a_degenerate_look_at_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0] }),
    );
    assert!(error.0.contains("no direction"), "{error}");
}

/// A placement whose field of view is out of range is refused before the
/// camera moves: a refused call leaves the view where it was, camera view
/// included.
#[test]
fn a_placement_with_an_out_of_range_fov_leaves_the_view_unchanged() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    )["view"]
        .clone();

    for fov in [1.0, 170.0] {
        let error = refused_call(
            &mut state,
            &mut viewer,
            "set_view",
            json!({
                "position": [2.0, -3.0, 1.0],
                "target": [0.0, 0.0, 0.0],
                "up": [1.0, 0.0, 1.0],
                "fov_short_axis_deg": fov,
            }),
        );
        assert!(error.0.contains("fov_short_axis_deg"), "{error}");
        let after = call(&mut state, &mut viewer, "get_scene", json!({}))["view"].clone();
        assert_eq!(
            after, before,
            "a refused placement at {fov}° moved the view"
        );
        assert!(
            viewer.maintain_z_up,
            "a refused rolled placement turned Maintain Z-up off"
        );
    }
}

// ── set_view: the explicit camera, a piece at a time ────────────────────

/// The view block, as a `(position, target, target_distance)` triple, for
/// tests that ask what a placement preserved.
#[track_caller]
fn placement_of(view: &Value) -> ([f64; 3], [f64; 3], f64) {
    let vector = |value: &Value| {
        let numbers: Vec<f64> = value
            .as_array()
            .expect("a vector")
            .iter()
            .map(|n| n.as_f64().expect("a number"))
            .collect();
        [numbers[0], numbers[1], numbers[2]]
    };
    (
        vector(&view["position"]),
        vector(&view["derived"]["target"]),
        view["target_distance"].as_f64().expect("a distance"),
    )
}

#[track_caller]
fn assert_close(actual: [f64; 3], expected: [f64; 3], what: &str) {
    for (axis, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{what}: axis {axis} was {actual} not {expected}"
        );
    }
}

/// A stated up that is not +Z is a rolled view the agent asked for, so it
/// turns Maintain Z-up off rather than being turned back; +Z leaves it on.
#[test]
fn a_rolled_up_turns_maintain_z_up_off() {
    let (mut state, mut viewer) = two_reconstructions();
    let look =
        |up: [f64; 3]| json!({ "position": [2.0, -3.0, 1.0], "target": [0.0, 0.0, 0.0], "up": up });

    call(&mut state, &mut viewer, "set_view", look([0.0, 0.0, 1.0]));
    assert!(viewer.maintain_z_up);

    call(&mut state, &mut viewer, "set_view", look([1.0, 0.0, 1.0]));
    assert!(!viewer.maintain_z_up);
}

/// A view the explicit camera can be moved a piece at a time from: three
/// distinct coordinates, a known distance, and nothing axis-aligned about it.
fn a_placed_view(state: &mut AppState, viewer: &mut Viewer3D) -> Value {
    call(
        state,
        viewer,
        "set_view",
        json!({ "position": [3.0, 0.0, 4.0], "target": [0.0, 0.0, 0.0] }),
    )["view"]
        .clone()
}

/// `target` alone re-centres the view: the camera keeps its orientation and
/// its distance, and moves to look at the point named.
#[test]
fn a_target_alone_recentres_the_view() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0] }),
    )["view"]
        .clone();

    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);
    assert_eq!(after["target_distance"], before["target_distance"]);
    let (position, target, _) = placement_of(&after);
    assert_close(target, [1.0, 2.0, 3.0], "the target it was given");
    assert_ne!(position, placement_of(&before).0);
}

/// `forward` alone is an orbit, not a turn in place: the camera swings around
/// what it is looking at, which stays where it was.
#[test]
fn a_forward_alone_orbits_rather_than_turning_in_place() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "forward": [0.0, 1.0, 0.0] }),
    )["view"]
        .clone();

    let (position, target, distance) = placement_of(&after);
    assert_close(target, placement_of(&before).1, "the standing target");
    assert_eq!(distance, placement_of(&before).2);
    // Looking along +y from `distance` back along it.
    assert_close(position, [0.0, -distance, 0.0], "swung onto the -y side");
}

/// `position` alone moves the camera and takes the target with it: the
/// orientation is not touched, so the view does not swing round to keep
/// looking at what it was.
#[test]
fn a_position_alone_keeps_the_orientation() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [10.0, 10.0, 10.0] }),
    )["view"]
        .clone();

    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);
    assert_eq!(after["position"], json!([10.0, 10.0, 10.0]));
}

/// `target_distance` alone is a dolly: the target holds still and the camera
/// moves along the view axis.
#[test]
fn a_target_distance_alone_dollies() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target_distance": 12.0 }),
    )["view"]
        .clone();

    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);
    assert_eq!(after["target_distance"], json!(12.0));
    let (position, target, _) = placement_of(&after);
    assert_close(target, placement_of(&before).1, "the standing target");
    // Twelve units back along the same view direction: 3/5, 0, 4/5.
    assert_close(position, [7.2, 0.0, 9.6], "dollied out along the view axis");
}

/// `target` with `forward` views a point from a direction, at the distance the
/// view already had.
#[test]
fn a_target_with_a_forward_keeps_the_distance() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [0.0, 0.0, 1.0], "forward": [-2.0, 0.0, 0.0] }),
    )["view"]
        .clone();

    assert_eq!(after["target_distance"], before["target_distance"]);
    let (position, target, distance) = placement_of(&after);
    assert_close(target, [0.0, 0.0, 1.0], "the target it was given");
    assert_close(position, [distance, 0.0, 1.0], "on the +x side, looking -x");
    assert_close(
        [
            after["derived"]["forward"][0].as_f64().expect("a number"),
            after["derived"]["forward"][1].as_f64().expect("a number"),
            after["derived"]["forward"][2].as_f64().expect("a number"),
        ],
        [-1.0, 0.0, 0.0],
        "the direction it was given, normalized",
    );
}

/// A partial placement is a free camera placement like the two whole ones, so
/// it leaves camera view: the background image belongs to a viewpoint the
/// camera has just moved off.
#[test]
fn a_partial_placement_leaves_camera_view() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    );
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0] }),
    );
    assert_eq!(out["view"]["looking_through"], Value::Null);
}

/// A call that says the same thing twice is refused rather than resolved in
/// some order the agent cannot see.
#[test]
fn an_overdetermined_placement_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let refusals = [
        (
            json!({ "position": [1.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0],
                    "target_distance": 4.0 }),
            "the separation of position and target is the distance",
        ),
        (
            json!({ "position": [1.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0],
                    "forward": [0.0, 0.0, -1.0] }),
            "already fixes the view direction",
        ),
        (
            json!({ "position": [1.0, 0.0, 0.0], "forward": [0.0, 0.0, -1.0],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "target_distance": 4.0 }),
            "there is no direction to derive one from",
        ),
    ];
    for (arguments, expected) in refusals {
        let error = refused_call(&mut state, &mut viewer, "set_view", arguments);
        assert!(error.0.contains(expected), "{error}");
    }
}

/// Every form refuses the arguments it does not read. An argument silently
/// ignored leaves the agent believing it asked for something it did not.
#[test]
fn a_placement_refuses_the_arguments_it_does_not_read() {
    let (mut state, mut viewer) = two_reconstructions();
    let refusals = [
        // No orientation is being derived, so there is no roll to steer.
        (
            json!({ "position": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0] }),
            "nothing to roll",
        ),
        // The roll of a derived view is `up`; `world_up` is the exact form's.
        (
            json!({ "position": [1.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0],
                    "world_up": [0.0, 0.0, 1.0] }),
            "world_up outside the exact form",
        ),
        // The exact form's roll is already in `world_up`.
        (
            json!({ "position": [1.0, 0.0, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "target_distance": 4.0, "up": [0.0, 0.0, 1.0] }),
            "carries its roll in world_up",
        ),
    ];
    for (arguments, expected) in refusals {
        let error = refused_call(&mut state, &mut viewer, "set_view", arguments);
        assert!(error.0.contains(expected), "{error}");
    }
}

/// `forward` is validated like the other directions: it must point somewhere,
/// and it must leave the roll defined.
#[test]
fn a_degenerate_forward_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "forward": [0.0, 0.0, 0.0] }),
    );
    assert!(error.0.contains("forward has no direction"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "forward": [0.0, 0.0, 1.0], "up": [0.0, 0.0, 2.0] }),
    );
    assert!(error.0.contains("roll is undefined"), "{error}");
}

// ── set_view: a point ───────────────────────────────────────────────────

/// `point` puts the point at the orbit target, as the Image Detail
/// double-click on a tracked feature does, and leaves camera view.
#[test]
fn a_point_is_brought_to_the_middle_of_the_view() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    );
    let alpha = state.scene[0].id;
    let Some(crate::scene::WorldPoint::At(position)) =
        crate::scene::world_point(&state.scene, PointRef::new(alpha, 5))
    else {
        panic!("point 5 is a finite point");
    };

    let out = call(&mut state, &mut viewer, "set_view", json!({ "point": 5 }));
    let (_, target, _) = placement_of(&out["view"]);
    for (axis, (actual, expected)) in target.iter().zip(position.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "target axis {axis} was {actual} not {expected}"
        );
    }
    assert_eq!(out["view"]["looking_through"], Value::Null);
    assert!(
        state.selected_point.is_none(),
        "the call moves no selection"
    );
}

/// `point` keeps the field of view and is one form among the others.
#[test]
fn a_point_refuses_a_field_of_view_and_a_second_form() {
    for arguments in [
        json!({ "point": 5, "fov_short_axis_deg": 40.0 }),
        json!({ "point": 5, "fit": null }),
        json!({ "point": 5, "bench_observation": { "observation": 0 } }),
    ] {
        let map = arguments.as_object().cloned().expect("an object");
        let error = tools::parse("set_view", Some(&map)).expect_err("refused");
        assert!(
            error.0.contains("fov_short_axis_deg") || error.0.contains("exclusive"),
            "{error}"
        );
    }
}

// ── set_view: animate ───────────────────────────────────────────────────

/// With `animate` the reply is the view the call ends at, while the camera is
/// back where it started, easing toward that view; landing the ease puts it
/// exactly where the reply said.
#[test]
fn an_animated_view_replies_with_where_it_ends_and_eases_there() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0], "animate": true }),
    );
    let (_, target, _) = placement_of(&out["view"]);
    assert_close(target, [1.0, 2.0, 3.0], "the reply's target");

    let (start, _, _) = placement_of(&before);
    let now = viewer.camera.camera.position;
    assert_close([now.x, now.y, now.z], start, "the camera before the ease");

    viewer.finish_transition();
    let (end, _, _) = placement_of(&out["view"]);
    let now = viewer.camera.camera.position;
    assert_close([now.x, now.y, now.z], end, "the camera after the ease");
}

/// An animated look-through enters camera view at the end of the ease, as the
/// double-click does, not at its start.
#[test]
fn an_animated_look_through_enters_camera_view_when_the_ease_lands() {
    let (mut state, mut viewer) = two_reconstructions();
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" }, "animate": true }),
    );
    assert_eq!(out["view"]["looking_through"]["camera_image_index"], 3);
    assert!(viewer.camera_view.is_none(), "camera view before the ease");
    viewer.finish_transition();
    assert_eq!(
        viewer.camera_view.as_ref().map(|view| view.image.index()),
        Some(3)
    );
}

/// Leaving camera view moves nothing, so asking to animate it is refused.
#[test]
fn animating_an_exit_from_camera_view_is_refused() {
    let map = json!({ "exit_camera_view": true, "animate": true })
        .as_object()
        .cloned()
        .expect("an object");
    let error = tools::parse("set_view", Some(&map)).expect_err("refused");
    assert!(error.0.contains("no move to animate"), "{error}");
}

// ── set_view: the relative forms ────────────────────────────────────────

/// A level view to move from: at `[0, -5, 1]` looking along +y at
/// `[0, 0, 1]`, five away, with `world_up` +Z.
fn a_level_view(state: &mut AppState, viewer: &mut Viewer3D) -> Value {
    call(
        state,
        viewer,
        "set_view",
        json!({ "position": [0.0, -5.0, 1.0], "target": [0.0, 0.0, 1.0], "up": [0.0, 0.0, 1.0] }),
    )["view"]
        .clone()
}

/// A vector field of the view block as an array.
#[track_caller]
fn vector_of(value: &Value) -> [f64; 3] {
    let numbers: Vec<f64> = value
        .as_array()
        .expect("a vector")
        .iter()
        .map(|n| n.as_f64().expect("a number"))
        .collect();
    [numbers[0], numbers[1], numbers[2]]
}

/// The text of the Action Log's last entry.
fn last_log_text(state: &AppState) -> Option<String> {
    state
        .action_log
        .entries()
        .last()
        .map(|entry| entry.text.clone())
}

/// One reconstruction, `metric`, whose file declares `unit` (or none).
fn one_reconstruction_in(unit: Option<&str>) -> (AppState, Viewer3D) {
    let mut reconstruction = recon(8, "M");
    reconstruction.metadata.world_space_unit = unit.map(str::to_string);
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/metric.sfmr"),
        reconstruction,
    ));
    let id = state.scene[0].id;
    state.select_recon(id);
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    state.window = Some(FakeWindow::default().info());
    (state, viewer)
}

/// `move` goes along the level axes: forward is the view direction laid flat,
/// so a camera looking down at its target moves over the ground rather than
/// into it, right is level, and up is +Z. The orientation and the target
/// distance stay, so the target comes along.
#[test]
fn a_move_goes_along_the_level_axes() {
    let (mut state, mut viewer) = two_reconstructions();
    // Looking down at the origin from behind and above.
    let before = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [0.0, -5.0, 2.0], "target": [0.0, 0.0, 0.0] }),
    )["view"]
        .clone();
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "move": { "forward": 1.0, "right": 2.0, "up": 0.5 } }),
    )["view"]
        .clone();

    let (position, target, distance) = placement_of(&after);
    assert_close(position, [2.0, -4.0, 2.5], "the moved camera");
    assert_close(target, [2.0, 1.0, 0.5], "the target, carried along");
    assert_eq!(distance, placement_of(&before).2);
    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);
}

/// A positive yaw turns the camera left, counter-clockwise seen from above; a
/// negative one turns it right; a positive pitch looks up. The camera does not
/// move.
#[test]
fn a_turn_turns_the_camera_in_place() {
    let (mut state, mut viewer) = two_reconstructions();
    a_level_view(&mut state, &mut viewer);
    let left = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "turn": { "yaw_deg": 90.0 } }),
    )["view"]
        .clone();
    assert_close(vector_of(&left["position"]), [0.0, -5.0, 1.0], "stayed put");
    assert_close(
        vector_of(&left["derived"]["forward"]),
        [-1.0, 0.0, 0.0],
        "turned left, to -x",
    );
    assert_close(
        vector_of(&left["derived"]["target"]),
        [-5.0, -5.0, 1.0],
        "the target swung round the camera",
    );

    a_level_view(&mut state, &mut viewer);
    let right = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "turn": { "yaw_deg": -35.0, "pitch_deg": 30.0 } }),
    )["view"]
        .clone();
    let (sin_yaw, cos_yaw) = 35f64.to_radians().sin_cos();
    let (sin_pitch, cos_pitch) = 30f64.to_radians().sin_cos();
    assert_close(
        vector_of(&right["derived"]["forward"]),
        [sin_yaw * cos_pitch, cos_yaw * cos_pitch, sin_pitch],
        "turned right and up",
    );
    assert_close(
        vector_of(&right["position"]),
        [0.0, -5.0, 1.0],
        "stayed put",
    );
}

/// An orbit swings the camera around the target, which stays: a positive yaw
/// carries the camera counter-clockwise seen from above, to its own right, and
/// a positive pitch raises it.
#[test]
fn an_orbit_swings_the_camera_around_the_target() {
    let (mut state, mut viewer) = two_reconstructions();
    a_level_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "orbit": { "yaw_deg": 90.0 } }),
    )["view"]
        .clone();
    let (position, target, distance) = placement_of(&after);
    assert_close(position, [5.0, 0.0, 1.0], "swung onto the +x side");
    assert_close(target, [0.0, 0.0, 1.0], "the target stays");
    assert!((distance - 5.0).abs() < 1e-9, "distance {distance}");

    a_level_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "orbit": { "pitch_deg": 30.0 } }),
    )["view"]
        .clone();
    let (sin, cos) = 30f64.to_radians().sin_cos();
    let (position, target, _) = placement_of(&after);
    assert_close(position, [0.0, -5.0 * cos, 1.0 + 5.0 * sin], "raised");
    assert_close(target, [0.0, 0.0, 1.0], "the target stays");
}

/// The relative forms ride together in the order move, turn, orbit, each from
/// where the one before left the camera, whatever order the call spells them
/// in: the move goes along the direction the camera faced when the call
/// arrived, and the orbit pivots on the target the turn swung round.
#[test]
fn the_relative_forms_apply_in_the_order_move_turn_orbit() {
    let (mut state, mut viewer) = two_reconstructions();
    a_level_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({
            "orbit": { "yaw_deg": 90.0 },
            "turn": { "yaw_deg": 90.0 },
            "move": { "forward": 1.0 },
            "fov_short_axis_deg": 50.0,
        }),
    )["view"]
        .clone();
    // Moved to [0, -4, 1] along +y; turned to face -x, so the target is at
    // [-5, -4, 1]; orbited a quarter turn counter-clockwise around it.
    let (position, target, _) = placement_of(&after);
    assert_close(
        position,
        [-5.0, 1.0, 1.0],
        "where the three left the camera",
    );
    assert_close(target, [-5.0, -4.0, 1.0], "the target the turn swung round");
    assert_close(
        vector_of(&after["derived"]["forward"]),
        [0.0, -1.0, 0.0],
        "facing back down -y",
    );
    let fov = after["fov_short_axis_deg"].as_f64().expect("a number");
    assert!((fov - 50.0).abs() < 1e-9, "fov {fov}");
    assert_eq!(
        last_log_text(&state).as_deref(),
        Some(
            "Moved 1 scene unit forward, then turned 90° left, then orbited 90° \
             counter-clockwise, then field of view 50.0°"
        )
    );
}

/// A turn in camera view is the free look a drag makes there: the camera
/// turns about the looked-through camera's up and camera view holds. A move
/// and an orbit leave it, as a fly key and an Alt-drag do.
#[test]
fn a_turn_keeps_camera_view_and_a_move_or_an_orbit_leaves_it() {
    let (mut state, mut viewer) = two_reconstructions();
    let look_through = json!({ "look_through": { "camera_image": "images/A_003.jpg" } });
    let through = call(&mut state, &mut viewer, "set_view", look_through.clone())["view"].clone();
    let turned = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "turn": { "yaw_deg": -20.0 } }),
    )["view"]
        .clone();
    assert_eq!(turned["looking_through"]["camera_image_index"], 3);
    assert_eq!(turned["position"], through["position"]);
    let up = nalgebra::Vector3::from(vector_of(&through["world_up"]));
    let before = nalgebra::Vector3::from(vector_of(&through["derived"]["forward"]));
    let after = nalgebra::Vector3::from(vector_of(&turned["derived"]["forward"]));
    let angle = before.angle(&after).to_degrees();
    assert!((angle - 20.0).abs() < 1e-6, "turned {angle}°");
    assert!(
        before.cross(&after).dot(&up) < 0.0,
        "a negative yaw turns right"
    );

    for (form, arguments) in [
        ("move", json!({ "move": { "up": 0.1 } })),
        ("orbit", json!({ "orbit": { "yaw_deg": 5.0 } })),
    ] {
        call(&mut state, &mut viewer, "set_view", look_through.clone());
        let out = call(&mut state, &mut viewer, "set_view", arguments);
        assert_eq!(out["view"]["looking_through"], Value::Null, "{form}");
    }
}

/// The relative forms start from the view as it stands, so none of them may
/// ride with an absolute form, and the refusal says what to do instead.
#[test]
fn a_relative_form_beside_an_absolute_one_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_level_view(&mut state, &mut viewer);
    for (arguments, named) in [
        (
            json!({ "move": { "forward": 1.0 }, "position": [1.0, 2.0, 3.0] }),
            "was given move and position at once",
        ),
        (
            json!({ "turn": { "yaw_deg": 10.0 }, "fit": null }),
            "was given turn and fit at once",
        ),
        (
            json!({ "orbit": { "yaw_deg": 10.0 }, "look_through": { "camera_image": 0 } }),
            "was given orbit and look_through at once",
        ),
    ] {
        let error = refused_call(&mut state, &mut viewer, "set_view", arguments);
        assert!(error.0.contains(named), "{error}");
        assert!(
            error
                .0
                .contains("combine only with one another, fov_short_axis_deg and animate"),
            "{error}"
        );
        assert!(error.0.contains("in a call of its own first"), "{error}");
    }
    let after = ok(&mut state, &mut viewer, Command::GetScene)["view"].clone();
    assert_eq!(after["position"], before["position"]);
}

/// A relative form with nothing in it, and a unit the format does not know,
/// are refused rather than read as no move.
#[test]
fn an_empty_relative_form_and_an_unknown_unit_are_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(&mut state, &mut viewer, "set_view", json!({ "move": {} }));
    assert_eq!(
        error.0,
        "set_view.move was given no distance — pass forward, right or up."
    );
    let error = refused_call(&mut state, &mut viewer, "set_view", json!({ "turn": {} }));
    assert_eq!(
        error.0,
        "set_view.turn was given no angle — pass yaw_deg or pitch_deg."
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "move": { "forward": 1.0, "unit": "yd" } }),
    );
    assert_eq!(
        error.0,
        "set_view.move wants unit to be one of mm, cm, m, in, ft — got \"yd\". Omit it to \
         move in the view's own unit."
    );
}

/// `get_scene` reports the unit each file declares, and the unit the view is
/// in; both are null for scene units.
#[test]
fn get_scene_reports_the_world_space_unit() {
    let (mut state, mut viewer) = one_reconstruction_in(Some("m"));
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(scene["scene"][0]["world_space_unit"], "m");
    assert_eq!(scene["view"]["world_space_unit"], "m");

    let (mut state, mut viewer) = one_reconstruction_in(None);
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(scene["scene"][0]["world_space_unit"], Value::Null);
    assert_eq!(scene["view"]["world_space_unit"], Value::Null);
}

/// Set `label`'s display transform to a bare scale.
fn rescale(state: &mut AppState, viewer: &mut Viewer3D, label: &str, scale: f64) {
    call(
        state,
        viewer,
        "set_reconstruction_transform",
        json!({
            "reconstruction_label": label,
            "transform": {
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
                "scale": scale,
            },
        }),
    );
}

/// A second reconstruction, `fine`, in `unit`, beside `metric`, with `metric`
/// selected again afterwards.
fn add_fine(state: &mut AppState, unit: Option<&str>) {
    let mut reconstruction = recon(8, "N");
    reconstruction.metadata.world_space_unit = unit.map(str::to_string);
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/fine.sfmr"),
        reconstruction,
    ));
    let metric = state.scene[0].id;
    state.select_recon(metric);
}

/// The view's unit is the selected reconstruction's: selecting another
/// reconstruction changes it, and with no selection it is scene units.
#[test]
fn the_view_takes_the_selected_reconstructions_unit() {
    let (mut state, mut viewer) = one_reconstruction_in(Some("m"));
    add_fine(&mut state, None);
    let view_unit = |state: &mut AppState, viewer: &mut Viewer3D| {
        ok(state, viewer, Command::GetScene)["view"]["world_space_unit"].clone()
    };
    assert_eq!(view_unit(&mut state, &mut viewer), "m", "metric selected");

    let fine = state.scene[1].id;
    state.select_recon(fine);
    assert_eq!(
        view_unit(&mut state, &mut viewer),
        Value::Null,
        "fine, with no unit, selected"
    );

    state.selected_recon = None;
    assert_eq!(
        view_unit(&mut state, &mut viewer),
        Value::Null,
        "nothing selected"
    );
}

/// The selected reconstruction is drawn at its display transform's scale, so
/// one length of the view is `metres(unit) / scale`: a scale that lands on
/// another unit names it (`mm` drawn at 0.001 is `m`), and one that lands
/// between the units is scene units.
#[test]
fn a_display_scale_carries_the_selections_unit_or_leaves_scene_units() {
    let (mut state, mut viewer) = one_reconstruction_in(Some("mm"));
    rescale(&mut state, &mut viewer, "metric", 0.001);
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(scene["scene"][0]["world_space_unit"], "mm");
    assert_eq!(scene["view"]["world_space_unit"], "m", "mm drawn at 0.001");

    rescale(&mut state, &mut viewer, "metric", 2.0);
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(
        scene["view"]["world_space_unit"],
        Value::Null,
        "mm drawn at 2"
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "move": { "forward": 1.0, "unit": "m" } }),
    );
    assert!(
        error.0.contains(
            "the selected reconstruction, metric, is in mm but drawn at a display scale of 2, \
             which makes one length of the view 0.0005 m"
        ),
        "{error}"
    );
}

/// A distance in another physical unit is converted to the selection's: ten
/// feet with a reconstruction in metres selected is 3.048 m, while the same
/// call with a reconstruction in feet selected moves ten of its units.
#[test]
fn a_move_in_feet_is_converted_to_the_selections_unit() {
    let (mut state, mut viewer) = one_reconstruction_in(Some("m"));
    add_fine(&mut state, Some("ft"));
    a_level_view(&mut state, &mut viewer);
    let feet = json!({ "move": { "forward": 10.0, "up": -1.0, "unit": "ft" } });
    let after = call(&mut state, &mut viewer, "set_view", feet.clone())["view"].clone();
    assert_close(
        vector_of(&after["position"]),
        [0.0, -5.0 + 3.048, 1.0 - 0.3048],
        "ten feet forward and one down, in metres",
    );
    assert_eq!(
        last_log_text(&state).as_deref(),
        Some("Moved 10 ft forward and 1 ft down")
    );

    let fine = state.scene[1].id;
    state.select_recon(fine);
    a_level_view(&mut state, &mut viewer);
    let after = call(&mut state, &mut viewer, "set_view", feet)["view"].clone();
    assert_close(
        vector_of(&after["position"]),
        [0.0, 5.0, 0.0],
        "ten feet forward and one down, in feet",
    );
}

/// A physical unit on a view in scene units has nothing to convert to, so it
/// is refused before anything moves, naming the selected reconstruction and
/// the three ways forward.
#[test]
fn a_physical_unit_on_a_view_in_scene_units_is_refused() {
    let (mut state, mut viewer) = one_reconstruction_in(None);
    let before = a_level_view(&mut state, &mut viewer);
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "move": { "forward": 2.0, "unit": "m" }, "turn": { "yaw_deg": 10.0 } }),
    );
    assert_eq!(
        error.0,
        "move.unit \"m\" needs the view to be in a physical unit, and it is in scene units: \
         the selected reconstruction, metric, declares no world_space_unit, so there is \
         nothing to convert m to. Select a reconstruction that declares a unit, send the \
         distances without unit (in scene units), or give metric a physical unit with sfm \
         xform --scale-by-measurements."
    );
    let after = ok(&mut state, &mut viewer, Command::GetScene)["view"].clone();
    assert_eq!(after["position"], before["position"]);
    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);

    // Without a unit the same distance is in scene units, and moves.
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "move": { "forward": 2.0 } }),
    )["view"]
        .clone();
    assert_close(
        vector_of(&after["position"]),
        [0.0, -3.0, 1.0],
        "two scene units forward",
    );
}

/// With nothing selected there is no unit to convert to, even where a loaded
/// reconstruction declares one, and the refusal says so.
#[test]
fn a_physical_unit_with_no_selection_is_refused() {
    let (mut state, mut viewer) = one_reconstruction_in(Some("m"));
    state.selected_recon = None;
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "move": { "forward": 2.0, "unit": "m" } }),
    );
    assert_eq!(
        error.0,
        "move.unit \"m\" needs the view to be in a physical unit, and it is in scene units: \
         no reconstruction is selected, so there is nothing to convert m to. Select a \
         reconstruction that declares a unit, send the distances without unit (in scene \
         units), or give the reconstruction a physical unit with sfm xform \
         --scale-by-measurements."
    );
}
