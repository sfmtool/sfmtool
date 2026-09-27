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
