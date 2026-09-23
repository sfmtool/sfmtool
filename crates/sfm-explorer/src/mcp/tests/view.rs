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
