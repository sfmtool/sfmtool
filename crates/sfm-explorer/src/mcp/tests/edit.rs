// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ── The editing surface ─────────────────────────────────────────────────
//
// One fixture for all of it: a node a resection and an adjustment can both run
// on, which is also an `embedded_patches` node with a path, so every family
// here has something to work with. It is the Scene Graph tests' own node, for
// the reason those tests borrow it from each other -- "a node an edit can run
// on" is one thing, and a second answer to it would be a second thing to keep
// in step.
//
// What these assert about an edit is the boundary and not the edit: the version
// it pushed, the reply's shape, the refusal's words, and the one Action Log row
// it left as the agent. What each family *does* to a reconstruction is asserted
// in `state::edits::tests`, over the same `AppState` calls these make.

/// A scene holding one editable node, `run_a`, selected, with a photograph
/// cached for its first two images so the edits that read pixels find them.
pub(super) fn editable() -> (AppState, Viewer3D) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    state.select_recon(id);
    let camera = &state.scene[0].recon().image_set.cameras[0];
    let (width, height) = (camera.width, camera.height);
    for index in 0..2 {
        let data: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 251) as u8).collect();
        state.insert_photograph(
            id,
            index,
            sfmtool_core::camera::image::ImageU8::new(width, height, 3, data),
        );
    }
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    state.window = Some(FakeWindow::default().info());
    (state, viewer)
}

/// How many versions `run_a` holds.
pub(super) fn version_count(state: &AppState) -> usize {
    state.scene[0].history.versions().len()
}

/// The Action Log's rows, oldest first, as `(actor, failed, text)`.
pub(super) fn rows(state: &AppState) -> Vec<(Actor, bool, String)> {
    state
        .action_log
        .entries()
        .map(|entry| (entry.actor, entry.failed, entry.text.clone()))
        .collect()
}

/// The failed rows of the Action Log, which is what "one refusal, one entry"
/// is counted over.
pub(super) fn failures(state: &AppState) -> Vec<(Actor, bool, String)> {
    rows(state).into_iter().filter(|row| row.1).collect()
}

/// Nudge image 1 off its stored pose, which is what gives a resection and an
/// adjustment something to pull back.
pub(super) fn perturb(state: &mut AppState, distance: f64) {
    let image = &mut state.scene[0].recon_mut().image_set.images[1];
    image.translation_xyz += nalgebra::Vector3::new(distance, -distance * 0.7, distance * 0.5);
}

/// A point edit pushes a version, and the reply says which one and what it did.
#[test]
fn delete_point_pushes_a_version_the_reply_names() {
    let (mut state, mut viewer) = editable();
    let reply = call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );

    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["reconstruction_label"], "run_a");
    let serial = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["serial"], serial);
    assert_eq!(reply["cursor"], serial);
    assert_eq!(reply["label"], "Deleted point 3 in run_a");
    assert_eq!(reply["dirty"], true);
    // The report is the sentence the edit recorded, serials and all, which is
    // where each family's own numbers are.
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Deleted point 3 in run_a"), "{report}");
    assert!(report.contains(&serial), "{report}");

    // And it is the agent's row, not the human's.
    let last = rows(&state).pop().expect("one entry per edit");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
}

/// Many points deleted in one call are one version, and the reply has
/// `delete_point`'s shape.
#[test]
fn delete_points_pushes_one_version_the_reply_names() {
    let (mut state, mut viewer) = editable();
    let reply = call(
        &mut state,
        &mut viewer,
        "delete_points",
        json!({ "reconstruction_label": "run_a", "points": [7, 3, 5, 3] }),
    );

    assert_eq!(version_count(&state), 2);
    for point in [3, 5, 7] {
        assert!(state.scene[0].is_point_deleted(point), "{point} survived");
    }
    assert_eq!(reply["reconstruction_label"], "run_a");
    let serial = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["serial"], serial);
    assert_eq!(reply["cursor"], serial);
    assert_eq!(reply["label"], "Deleted 3 points in run_a");
    assert_eq!(reply["dirty"], true);
    assert_eq!(reply["changed"], true);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Deleted 3 points in run_a"), "{report}");
    assert!(report.contains(&serial), "{report}");
    let last = rows(&state).pop().expect("one entry per edit");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
}

/// A list naming a dead point, an empty list and a list that is not one of
/// whole numbers are each refused, and none of them pushes a version.
#[test]
fn delete_points_refuses_a_list_it_cannot_delete_whole() {
    let (mut state, mut viewer) = editable();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 4 }),
    );
    for (points, expected) in [
        (json!([1, 4]), "no live point at edited index 4"),
        (json!([]), "No points to delete."),
        (json!([1, -2]), "points"),
        (json!("1, 2"), "points"),
    ] {
        let error = refused_call(
            &mut state,
            &mut viewer,
            "delete_points",
            json!({ "reconstruction_label": "run_a", "points": points }),
        );
        assert!(error.0.contains(expected), "{points}: {error}");
        assert_eq!(version_count(&state), 2, "{points}");
    }
    assert!(
        !state.scene[0].is_point_deleted(1),
        "half a list was deleted"
    );
}

/// A retriangulated point pushes a version, and the report names the verdict
/// its own observations supported rather than merely that something happened.
#[test]
fn retriangulate_point_pushes_a_version_that_names_its_verdict() {
    let (mut state, mut viewer) = editable();
    let reply = call(
        &mut state,
        &mut viewer,
        "retriangulate_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );

    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["reconstruction_label"], "run_a");
    let serial = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["serial"], serial);
    assert_eq!(reply["label"], "Retriangulated point 3 in run_a");
    let report = reply["report"].as_str().expect("a report");
    assert!(
        report.starts_with("Retriangulated point 3 in run_a"),
        "{report}"
    );
    assert!(
        report.contains("finite") || report.contains("infinity") || report.contains("left where"),
        "the report names no verdict: {report}"
    );
    // And where to find the point next: a rewritten point names its new
    // index, and one the answer left as it was says so.
    assert!(
        report.contains(", now point ")
            || report.contains(", unchanged")
            || report.contains("left where it was"),
        "the report names neither a new index nor an unchanged point: {report}"
    );
    let last = rows(&state).pop().expect("one entry per edit");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
}

/// The whole-value retriangulation goes to a worker, and the version it comes
/// back with is the node's next one.
#[test]
fn retriangulate_all_points_defers_and_comes_back_with_a_version() {
    let (mut state, mut viewer) = editable();
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("retriangulate_all_points", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the retriangulation must defer"),
    };
    assert_eq!(pending.operation_name, "Retriangulate all points");
    state.finish_background_tasks();

    let reply = match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["label"], json!("Retriangulated run_a"), "{reply}");
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("points moved"), "{report}");
}

/// A reconstruction whose images are taken through two cameras is
/// retriangulated over the wire, not refused.
#[test]
fn retriangulate_all_points_accepts_two_cameras() {
    let (mut state, mut viewer) = editable();
    with_two_cameras(&mut state);
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("retriangulate_all_points", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the retriangulation must defer"),
    };
    state.finish_background_tasks();

    match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(reply)) => {
            assert_eq!(reply["label"], json!("Retriangulated run_a"), "{reply}")
        }
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
    assert_eq!(version_count(&state), 2);
}

/// The refit of every point goes to a worker, and the version it comes back
/// with is the node's next one. Over the bench's fixture, an embedded_patches
/// node with a patch frame per point and a photograph cached for every image.
#[test]
fn refit_all_points_defers_and_comes_back_with_a_version() {
    let (mut state, _) = crate::bench::tests::state();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    state.window = Some(FakeWindow::default().info());
    let map = json!({ "reconstruction_label": "demo" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("refit_all_points", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the refit must defer"),
    };
    assert_eq!(pending.operation_name, "Refit all points");
    state.finish_background_tasks();

    let reply = match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["label"], json!("Refit demo"), "{reply}");
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("points refit"), "{report}");
}

/// A node whose points carry no patch frame is refused at once, in the greyed
/// menu entry's words, and nothing starts.
#[test]
fn refit_all_points_is_refused_without_patch_frames() {
    let (mut state, mut viewer) = editable();
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("refit_all_points", Some(&map)).expect("a well-formed call");
    match agent(&mut state, &mut viewer, command) {
        Outcome::Done(Err(e)) => {
            assert!(e.to_string().contains("no patch frame"), "{e}")
        }
        _ => panic!("a node with no patch frames must be refused"),
    }
    assert!(state.running_tasks().next().is_none());
    assert_eq!(version_count(&state), 1);
}

/// The refinement of every normal goes to a worker under the operation name
/// the Background Tasks panel shows, and the version it comes back with is the
/// node's next one, its report the Action Log's sentence. Over the bench's
/// fixture, as for the refit.
#[test]
fn refine_all_normals_defers_and_comes_back_with_a_version() {
    let (mut state, _) = crate::bench::tests::state();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    state.window = Some(FakeWindow::default().info());
    let map = json!({ "reconstruction_label": "demo" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("refine_all_normals", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the refinement must defer"),
    };
    assert_eq!(pending.operation_name, "Refine all normals");
    state.finish_background_tasks();

    let reply = match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(
        reply["label"],
        json!("Refined the normals of demo"),
        "{reply}"
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("finite points turned"), "{report}");
}

/// A node whose points carry no patch frame is refused at once, in the greyed
/// menu entry's words, and nothing starts.
#[test]
fn refine_all_normals_is_refused_without_patch_frames() {
    let (mut state, mut viewer) = editable();
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("refine_all_normals", Some(&map)).expect("a well-formed call");
    match agent(&mut state, &mut viewer, command) {
        Outcome::Done(Err(e)) => {
            assert!(e.to_string().contains("no patch frame"), "{e}")
        }
        _ => panic!("a node with no patch frames must be refused"),
    }
    assert!(state.running_tasks().next().is_none());
    assert_eq!(version_count(&state), 1);
}

/// The prune goes to a worker, and the version it comes back with is the
/// node's next one.
#[test]
fn prune_covered_observations_defers_and_comes_back_with_a_version() {
    let (mut state, mut viewer) = editable();
    // The fixture's points carry no patch frame, so the node the wire acts on
    // is the one that does.
    state.scene.clear();
    state.append_node(crate::scene_graph::tests::prunable_node("/runs/run_a.sfmr"));
    let id = state.scene[0].id;
    state.select_recon(id);

    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command =
        tools::parse("prune_covered_observations", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the prune must defer"),
    };
    assert_eq!(pending.operation_name, "Prune covered observations");
    state.finish_background_tasks();

    let reply = match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(
        reply["label"],
        json!("Pruned covered observations in run_a"),
        "{reply}"
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("observations retired"), "{report}");
}

/// The three thresholds cross the wire, and a call that names none takes the
/// operation's own defaults.
#[test]
fn prune_covered_observations_carries_its_thresholds() {
    let defaults = sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions::default();
    let bare = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    match tools::parse("prune_covered_observations", Some(&bare)).expect("a well-formed call") {
        Command::PruneCoveredObservations { options, .. } => assert_eq!(options, defaults),
        other => panic!("parsed to {other:?}"),
    }

    let named = json!({
        "reconstruction_label": "run_a",
        "footprint_fraction": 0.4545,
        "ratio": 3.0,
        "min_fine_radius_px": 0.0,
    })
    .as_object()
    .cloned()
    .expect("an object");
    match tools::parse("prune_covered_observations", Some(&named)).expect("a well-formed call") {
        Command::PruneCoveredObservations { options, .. } => {
            assert_eq!(options.footprint_fraction, 0.4545);
            assert_eq!(options.ratio, 3.0);
            assert_eq!(options.min_fine_radius_px, 0.0);
            assert_eq!(options.min_observations, defaults.min_observations);
        }
        other => panic!("parsed to {other:?}"),
    }
}

/// An edit the state refuses pushes no version, answers in the state's words,
/// and leaves exactly one failed row.
#[test]
fn a_refused_edit_pushes_no_version_and_is_logged_once() {
    let (mut state, mut viewer) = editable();
    state.action_log.clear();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 99 }),
    );
    assert!(error.0.contains("out of range"), "{error}");
    assert_eq!(version_count(&state), 1);
    assert_eq!(failures(&state).len(), 1, "{:?}", rows(&state));
}

/// A bulk edit renumbers the image set, and the reply is read off the version
/// it pushed rather than off the request.
#[test]
fn delete_camera_image_renumbers_and_reports_the_version() {
    let (mut state, mut viewer) = editable();
    let before = state.scene[0].recon().image_set.images.len();
    let reply = call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );
    assert_eq!(
        state.scene[0].recon().image_set.images.len(),
        before - 1,
        "the image set did not shrink"
    );
    assert!(
        reply["label"]
            .as_str()
            .expect("a label")
            .starts_with("Deleted image "),
        "{reply}"
    );
}

/// The resection lands as the node's next version, and its report is the
/// resection's own summary.
#[test]
fn resecting_pushes_a_version_and_reports_the_estimate() {
    let (mut state, mut viewer) = editable();
    let id = state.scene[0].id;
    crate::resect::tests::give_cluster_patches(&mut state, id);
    perturb(&mut state, 0.30);
    let reply = call(
        &mut state,
        &mut viewer,
        "resect_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );
    assert_eq!(version_count(&state), 2);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Resected "), "{report}");
    assert!(report.contains("inliers"), "{report}");
    assert!(report.contains(" clusters)"), "{report}");
}

/// Without a current cluster-patches file the state refuses in its own words,
/// which are the greyed menu entry's, and that refusal is the one row the log
/// gets.
#[test]
fn resecting_without_a_cluster_patches_file_is_refused_in_the_states_words() {
    let (mut state, mut viewer) = editable();
    let id = state.scene[0].id;
    let refusal = state
        .resect_image_refusal(crate::scene::ImageRef::new(id, 1))
        .expect("no file is open");
    state.action_log.clear();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "resect_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );
    assert!(error.0.contains(&refusal), "{error}");
    assert!(error.0.contains("Build Index Files"), "{error}");
    assert_eq!(version_count(&state), 1);

    let failed = failures(&state);
    assert_eq!(failed.len(), 1, "one refusal is one entry");
    // The state's own sentence, not the drain's `{tool} failed: …` wrapper.
    assert!(
        !failed[0].2.starts_with("resect_camera_image failed"),
        "{:?}",
        failed[0]
    );
    assert_eq!(failed[0].0, Actor::Mcp);
}

/// Start a `bundle_adjust` the way the frame does, let the operation finish,
/// and answer it the way the readback phase does.
///
/// The tool defers rather than answering inside the call, so a test that wants
/// the reply has to do what the frame does: start it, let the worker run, and
/// ask [`super::super::edit::background_reply`] for the answer.
#[track_caller]
pub(super) fn adjusted(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Value {
    let map = arguments.as_object().cloned().expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    let pending = match agent(state, viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("bundle_adjust must defer"),
    };
    state.finish_background_tasks();
    match super::super::edit::background_reply(state, &pending).expect("the operation finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
}

/// Every camera of the node's value switched to a four-coefficient
/// `SFMTOOL_PINHOLE` whose spline is flat, which projects exactly as the
/// pinhole it replaces.
fn with_flat_spline_cameras(state: &mut AppState) {
    let recon = state.scene[0].recon_mut();
    for camera in &mut recon.image_set.cameras {
        let (f, _) = camera.focal_lengths();
        let (cx, cy) = camera.principal_point();
        camera.model = sfmtool_core::CameraModel::SfmtoolPinhole {
            focal_length: f,
            principal_point_x: cx,
            principal_point_y: cy,
            bspline_rho_max: 1.0,
            bspline: vec![0.0; 4],
        };
    }
}

/// `run_a` with its camera made a single-focal `SIMPLE_PINHOLE`, which can
/// release its focal and has no distortion, and a copy of its original
/// two-focal `PINHOLE` as camera 1 taking image 2, whose focal cannot be
/// released. Two cameras through the same lens, so the observations still
/// describe the geometry.
fn with_two_cameras(state: &mut AppState) {
    let recon = state.scene[0].recon_mut();
    let original = recon.image_set.cameras[0].clone();
    let camera = &mut recon.image_set.cameras[0];
    let (f, _) = camera.focal_lengths();
    let (cx, cy) = camera.principal_point();
    camera.model = sfmtool_core::CameraModel::SimplePinhole {
        focal_length: f,
        principal_point_x: cx,
        principal_point_y: cy,
    };
    recon.image_set.cameras.push(original);
    recon.image_set.images[2].camera_index = 1;
}

/// The defaults reach every camera, and a `cameras` entry overrides them for
/// the camera it names; the label says what each camera released.
#[test]
fn bundle_adjust_releases_each_camera_as_its_entry_says() {
    let (mut state, mut viewer) = editable();
    with_two_cameras(&mut state);
    perturb(&mut state, 0.02);

    // The default focal release reaches camera 1 too, whose PINHOLE model
    // cannot take it: the core function's refusal, naming it.
    let map = json!({ "reconstruction_label": "run_a", "release_focal": true })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        _ => panic!("bundle_adjust must defer"),
    };
    state.finish_background_tasks();
    let Err(error) =
        super::super::edit::background_reply(&state, &pending).expect("the operation finished")
    else {
        panic!("expected a refusal");
    };
    assert!(error.0.contains("camera 1, a PINHOLE"), "{error}");
    assert_eq!(version_count(&state), 1);

    // Held through its entry, the solve runs and releases camera 0 alone.
    let reply = adjusted(
        &mut state,
        &mut viewer,
        json!({
            "reconstruction_label": "run_a",
            "release_focal": true,
            "cameras": [{ "camera_intrinsics_index": 1, "release_focal": false }]
        }),
    );
    let label = reply["label"].as_str().expect("a label");
    assert_eq!(
        label,
        "Bundle adjusted run_a, camera 0 focal released, camera 1 held"
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(
        report.contains(", camera 0 focal ") && !report.contains("camera 1 focal"),
        "{report}"
    );
    assert_eq!(version_count(&state), 2);
}

/// An entry naming a camera the reconstruction does not have, or one camera
/// twice, is refused before anything starts.
#[test]
fn bundle_adjust_refuses_a_camera_entry_it_cannot_apply() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "bundle_adjust",
        json!({
            "reconstruction_label": "run_a",
            "cameras": [{ "camera_intrinsics_index": 5 }]
        }),
    );
    assert!(
        error.0.contains("camera_intrinsics_index 5") && error.0.contains("1 camera"),
        "{error}"
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "bundle_adjust",
        json!({
            "reconstruction_label": "run_a",
            "cameras": [{ "camera_intrinsics_index": 0 }, { "camera_intrinsics_index": 0 }]
        }),
    );
    assert!(error.0.contains("twice"), "{error}");
    assert_eq!(version_count(&state), 1);
    assert!(state.running_tasks().next().is_none());
}

/// The `cameras` entries parse to overrides whose left-out fields stay open
/// for the defaults, and an entry with an argument it does not have is
/// refused.
#[test]
fn bundle_adjust_parses_its_camera_entries() {
    let map = json!({
        "reconstruction_label": "a",
        "release_focal": true,
        "cameras": [
            { "camera_intrinsics_index": 1, "release_distortion": true, "release_principal_point": true },
            { "camera_intrinsics_index": 0, "release_focal": false, "release_distortion": false }
        ]
    })
    .as_object()
    .cloned()
    .expect("an object");
    let Command::BundleAdjust { cameras, .. } =
        tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call")
    else {
        panic!("not a bundle_adjust");
    };
    assert_eq!(
        cameras,
        vec![
            super::super::CameraReleaseOverride {
                camera_intrinsics_index: 1,
                release_focal: None,
                release_distortion: Some(true),
                release_principal_point: Some(true),
                release_image_focal: None,
                release_image_principal_point: None,
            },
            super::super::CameraReleaseOverride {
                camera_intrinsics_index: 0,
                release_focal: Some(false),
                release_distortion: Some(false),
                release_principal_point: None,
                release_image_focal: None,
                release_image_principal_point: None,
            },
        ]
    );

    for (cameras, expected) in [
        (json!([{ "camera": 0 }]), "has no argument"),
        (
            json!([{ "release_focal": true }]),
            "camera_intrinsics_index",
        ),
        (json!([3]), "cameras"),
        (json!({ "camera_intrinsics_index": 0 }), "cameras"),
    ] {
        let map = json!({ "reconstruction_label": "a", "cameras": cameras })
            .as_object()
            .cloned()
            .expect("an object");
        let error = tools::parse("bundle_adjust", Some(&map)).expect_err("refused at the parse");
        assert!(error.0.contains(expected), "{error}");
    }
}

/// `schedule` parses to the rounds it lists, a `null` trim to the unbounded
/// one, and a schedule the core function would refuse is refused at the
/// parse, in its words.
#[test]
fn bundle_adjust_parses_its_schedule() {
    let schedule = |value: Value| {
        let map = json!({ "reconstruction_label": "a", "schedule": value })
            .as_object()
            .cloned()
            .expect("an object");
        tools::parse("bundle_adjust", Some(&map)).map(|command| match command {
            Command::BundleAdjust { schedule, .. } => schedule,
            _ => panic!("not a bundle_adjust"),
        })
    };
    let round = |trim_px: f64, loss_scale: f64| sfmtool_core::geometry::BaSchedule {
        trim_px,
        loss_scale,
    };
    assert_eq!(
        schedule(json!([[50, 5], [12.5, 2], [null, 1000]])).expect("well formed"),
        Some(vec![
            round(50.0, 5.0),
            round(12.5, 2.0),
            round(f64::INFINITY, 1000.0)
        ])
    );
    assert_eq!(schedule(Value::Null).expect("well formed"), None);

    for (value, expected) in [
        (json!([]), "the schedule has no rounds to run"),
        (
            json!([[4, 1], [0, 1]]),
            "round 2 of the schedule has trim 0",
        ),
        (
            json!([[4, -1]]),
            "round 1 of the schedule has loss scale -1",
        ),
        (json!([[4, null]]), "wants schedule to be an array of"),
        (json!([[4]]), "wants schedule to be an array of"),
        (json!([["inf", 1]]), "wants schedule to be an array of"),
        (json!("(inf, 1000)"), "wants schedule to be an array of"),
    ] {
        let error = schedule(value.clone()).expect_err("refused at the parse");
        assert!(error.0.contains(expected), "{value}: {error}");
    }
}

/// The per-image releases, each camera's override of them, and the prior's
/// widths parse into the request; a prior the wire cannot state is refused at
/// the parse.
#[test]
fn bundle_adjust_parses_its_per_image_releases_and_prior() {
    let map = json!({
        "reconstruction_label": "a",
        "release_image_principal_point": true,
        "cameras": [{ "camera_intrinsics_index": 1, "release_image_focal": true,
                      "release_image_principal_point": false }],
        "image_prior": { "floor": { "principal_point": 0.002 },
                         "fixed": { "focal": 0.01 } }
    })
    .as_object()
    .cloned()
    .expect("an object");
    let Command::BundleAdjust {
        release_image_focal,
        release_image_principal_point,
        cameras,
        image_prior,
        ..
    } = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call")
    else {
        panic!("not a bundle_adjust");
    };
    assert!(!release_image_focal && release_image_principal_point);
    assert_eq!(cameras[0].release_image_focal, Some(true));
    assert_eq!(cameras[0].release_image_principal_point, Some(false));
    let defaults = sfmtool_core::geometry::ImagePriorOptions::default();
    assert_eq!(image_prior.loose, defaults.loose);
    assert_eq!(image_prior.floor.principal_point, 0.002);
    assert_eq!(image_prior.floor.focal, defaults.floor.focal);
    assert_eq!(
        image_prior.fixed,
        Some(sfmtool_core::geometry::ImagePriorWidths {
            focal: 0.01,
            principal_point: defaults.loose.principal_point,
        })
    );

    for (prior, expected) in [
        (json!({ "wide": {} }), "has no argument"),
        (json!({ "loose": { "f": 0.1 } }), "has no argument"),
        (json!({ "loose": { "focal": 0.0 } }), "positive number"),
        (json!({ "floor": 3 }), "wants floor to be an object"),
        (json!([0.1]), "wants image_prior to be an object"),
    ] {
        let map = json!({ "reconstruction_label": "a", "image_prior": prior.clone() })
            .as_object()
            .cloned()
            .expect("an object");
        let error = tools::parse("bundle_adjust", Some(&map)).expect_err("refused at the parse");
        assert!(error.0.contains(expected), "{prior}: {error}");
    }
}

/// A per-image release on the wire releases each image's principal point, the
/// label says so, and the reply carries the spread of the camera's images and
/// the prior's widths.
#[test]
fn bundle_adjust_reports_the_per_image_spread() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    let reply = adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a", "release_image_principal_point": true }),
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(
        report.contains("principal point released per image"),
        "{report}"
    );
    assert!(
        report.contains(", per-image widths principal point "),
        "{report}"
    );
    let rows = reply["image_adjustments"]
        .as_array()
        .expect("the per-image spread");
    assert_eq!(rows.len(), 1, "{reply}");
    let row = &rows[0];
    assert_eq!(row["camera_intrinsics_index"], 0);
    assert_eq!(row["release_image_principal_point"], true);
    assert_eq!(row["release_image_focal"], false);
    assert_eq!(row["prior_derived"], true);
    assert!(row["prior"]["principal_point"].as_f64().is_some(), "{row}");
    assert_eq!(row["prior"]["focal"], Value::Null);
    assert!(row["principal_point_std_px"].as_array().is_some(), "{row}");
    assert!(row["near_prior"].as_array().is_some(), "{row}");

    // A run with nothing released per image carries no such field.
    let reply = adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(reply["image_adjustments"], Value::Null, "{reply}");
}

/// A schedule given on the wire is the one the solve runs, the entry names it,
/// and the dialog opens on the node with it.
#[test]
fn bundle_adjust_runs_the_schedule_it_is_given() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    let reply = adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a", "schedule": [[null, 1000]] }),
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains(", schedule (inf, 1000)"), "{report}");

    let id = state.scene[0].id;
    state.open_bundle_adjust(id);
    assert_eq!(
        state.bundle_adjust_prompt.schedule_box(),
        Some("(inf, 1000)")
    );
}

/// `switch_camera_model`'s arguments parse into the request the edit takes,
/// with every default left open.
#[test]
fn switch_camera_model_parses_its_request() {
    let map = json!({ "reconstruction_label": "a", "camera_intrinsics_index": 1 })
        .as_object()
        .cloned()
        .expect("an object");
    assert_eq!(
        tools::parse("switch_camera_model", Some(&map)).expect("a well-formed call"),
        Command::SwitchCameraModel {
            reconstruction_label: "a".to_string(),
            request: crate::state::edits::SwitchCameraModelRequest {
                camera: 1,
                ..Default::default()
            },
        }
    );
    let map = json!({
        "reconstruction_label": "a",
        "camera_intrinsics_index": 0,
        "camera_model": "SFMTOOL_FISHEYE",
        "coeff_count": 10,
        "spline_domain_deg": 108.0,
        "theta_fit_deg": 80.0
    })
    .as_object()
    .cloned()
    .expect("an object");
    assert_eq!(
        tools::parse("switch_camera_model", Some(&map)).expect("a well-formed call"),
        Command::SwitchCameraModel {
            reconstruction_label: "a".to_string(),
            request: crate::state::edits::SwitchCameraModelRequest {
                camera: 0,
                camera_model: Some("SFMTOOL_FISHEYE".to_string()),
                coeff_count: Some(10),
                spline_domain_deg: Some(108.0),
                theta_fit_deg: Some(80.0),
            },
        }
    );
    let map = json!({ "reconstruction_label": "a" })
        .as_object()
        .cloned()
        .expect("an object");
    let error = tools::parse("switch_camera_model", Some(&map)).expect_err("no camera");
    assert!(error.0.contains("camera_intrinsics_index"), "{error}");
}

/// With no model named, a spline camera's switch is a refit of its spline:
/// its own count and domain unless they are named, one version, and the fit's
/// numbers in the reply.
#[test]
fn switch_camera_model_refits_a_spline_with_its_own_count_and_domain_by_default() {
    let (mut state, mut viewer) = editable();
    with_flat_spline_cameras(&mut state);

    let reply = call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "coeff_count": 6
        }),
    );
    assert_eq!(
        reply["label"],
        "Refit spline of camera 0 of run_a: 4 → 6 coefficients, domain 45.0° kept"
    );
    assert_eq!(reply["changed"], true);
    let fit = &reply["fit"];
    assert_eq!(fit["theta_fit_source"], "spline_domain");
    assert_eq!(fit["camera_model_before"], "SFMTOOL_PINHOLE");
    assert_eq!(fit["camera_model_after"], "SFMTOOL_PINHOLE");
    assert!(fit["rms_px"].as_f64().is_some_and(|r| r >= 0.0), "{fit}");
    assert!(fit["monotone_constraint"]["active"].is_boolean(), "{fit}");
    let report = reply["report"].as_str().expect("a report");
    assert!(
        report.contains("fit rms ") && report.contains("median error"),
        "{report}"
    );
    let camera = &state.scene[0].recon().image_set.cameras[0];
    let (bspline, rho_max, _) = camera.model.radial_spline().expect("a spline");
    assert_eq!(bspline.len(), 6);
    assert_eq!(
        rho_max.to_bits(),
        1.0f64.to_bits(),
        "the domain end was not kept"
    );
    assert_eq!(version_count(&state), 2);

    // A named domain with the count left out keeps the count.
    let reply = call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "spline_domain_deg": 50.0
        }),
    );
    assert_eq!(
        reply["label"],
        "Refit spline of camera 0 of run_a: 6 coefficients kept, domain 45.0° → 50.0°"
    );
    assert!((reply["fit"]["spline_domain_deg"].as_f64().unwrap() - 50.0).abs() < 1e-9);
    assert_eq!(version_count(&state), 3);
}

/// A named model is a switch between models.
#[test]
fn switch_camera_model_switches_to_a_named_model() {
    let (mut state, mut viewer) = editable();
    let before = state.scene[0].recon().image_set.cameras[0]
        .model_name()
        .to_string();
    let reply = call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "sfmtool_pinhole"
        }),
    );
    assert_eq!(
        reply["label"],
        format!("Switched camera 0 of run_a from {before} to SFMTOOL_PINHOLE")
    );
    assert_eq!(
        state.scene[0].recon().image_set.cameras[0].model_name(),
        "SFMTOOL_PINHOLE"
    );
    assert_eq!(version_count(&state), 2);
}

/// A camera the image set does not have, and a refit the model cannot make, are
/// refused naming the camera, and push nothing.
#[test]
fn switch_camera_model_refuses_naming_the_camera() {
    let (mut state, mut viewer) = editable();
    with_flat_spline_cameras(&mut state);
    let error = refused_call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({ "reconstruction_label": "run_a", "camera_intrinsics_index": 5 }),
    );
    assert!(error.0.contains("camera 5 does not exist"), "{error}");
    // SFMTOOL_PINHOLE cannot end its domain at 95°.
    let error = refused_call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "spline_domain_deg": 95.0
        }),
    );
    assert!(error.0.contains("camera 0"), "{error}");
    assert!(error.0.contains("spline domain"), "{error}");
    assert_eq!(version_count(&state), 1);
}

/// The adjustment runs on the node's value and reports its residuals.
#[test]
fn bundle_adjust_pushes_a_version_and_reports_its_residuals() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    let reply = adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(version_count(&state), 2);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("median residual"), "{report}");
    assert!(!report.contains("focal released"), "{report}");
    // An operation that finished inside the window answers as it always did:
    // no handle, and nothing for a reader to discriminate on.
    assert_eq!(reply["running"], Value::Null, "{reply}");
}

/// An operation that outlives the window answers with a handle instead.
#[test]
fn a_slow_adjustment_answers_with_a_handle_naming_it() {
    let (mut state, _viewer) = editable();
    let id = state.scene[0].id;
    // A worker held open, so the operation is still running when the reply is
    // asked for; and a call that started long enough ago to be past the window,
    // which is what the frame's clock would have reached.
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |_progress| {
                let _ = held.recv();
                crate::background::Finished::Failed("nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    let task = state.running_tasks().next().expect("running");
    let pending = super::super::BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        queue: task.operation.queue,
        answer: super::super::Answer::Version(id),
        label: task.label.clone(),
        queued: std::time::Instant::now() - super::super::REPLY_DIRECTLY_WITHIN,
    };

    let reply =
        match super::super::edit::background_reply(&state, &pending).expect("past the window") {
            Ok(ToolOutput::Json(value)) => value,
            _ => panic!("a handle is JSON"),
        };
    assert_eq!(reply["running"], json!(true), "{reply}");
    assert_eq!(reply["operation"], json!("Bundle adjust"), "{reply}");
    assert_eq!(reply["queue"], json!("slow"), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");
    assert_eq!(
        reply["operation_id"],
        json!(pending.operation_id),
        "{reply}"
    );

    // Inside the window and still running, there is no answer yet: the frame
    // moves on and asks again.
    let fresh = super::super::BackgroundReply {
        queued: std::time::Instant::now(),
        ..pending
    };
    assert!(
        super::super::edit::background_reply(&state, &fresh).is_none(),
        "an operation inside the window answered early",
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();
}

/// A background edit that ran to its end and found nothing to change pushes no
/// version, and its reply says so: `changed: false`, at the version the node
/// still stands at, with the run's own sentence as the report.
#[test]
fn a_background_edit_that_changes_nothing_answers_changed_false() {
    let (mut state, _viewer) = editable();
    let id = state.scene[0].id;
    let before = state.node(id).expect("loaded").history.current_version();
    let (serial, label) = (before.serial.to_string(), before.label.clone());
    state
        .start_background_task(
            crate::background::Operation::PRUNE_COVERED_OBSERVATIONS,
            id,
            Box::new(|_progress| {
                crate::background::Finished::NoChange("Nothing to prune".to_string())
            }),
        )
        .expect("nothing else is running");
    let task = state.running_tasks().next().expect("running");
    let pending = super::super::BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        queue: task.operation.queue,
        answer: super::super::Answer::Version(id),
        label: task.label.clone(),
        queued: std::time::Instant::now(),
    };
    state.finish_background_tasks();
    assert_eq!(version_count(&state), 1);

    let reply = match super::super::edit::background_reply(&state, &pending)
        .expect("the operation has finished")
    {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("a version reply is JSON"),
    };
    assert_eq!(reply["changed"], json!(false), "{reply}");
    assert_eq!(reply["serial"], json!(serial), "{reply}");
    assert_eq!(reply["cursor"], json!(serial), "{reply}");
    assert_eq!(reply["label"], json!(label), "{reply}");
    assert_eq!(reply["report"], json!("Nothing to prune"), "{reply}");
}

/// The conversion is a background task on the wire, and `get_scene` says what
/// it changed: one call defers, the frame answers with the version it pushed,
/// and the node's `feature_source` has flipped.
#[test]
fn convert_to_embedded_patches_defers_and_flips_the_feature_source() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(dir.path());
    state.select_recon(id);
    state.window = Some(FakeWindow::default().info());
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let before = call(&mut state, &mut viewer, "get_scene", json!({}));
    let node = &before["scene"][0];
    assert_eq!(node["feature_source"], json!("sift_files"), "{before}");
    assert_eq!(node["has_patch_data"], json!(false), "{before}");

    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command =
        tools::parse("convert_to_embedded_patches", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the conversion must defer"),
    };
    assert_eq!(pending.operation_name, "Convert to embedded patches");
    state.finish_background_tasks();

    let reply = match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(
        reply["label"],
        json!("Converted run_a to embedded patches"),
        "{reply}"
    );
    let report = reply["report"].as_str().expect("a report");
    assert_eq!(
        report.split(" (").next().expect("a sentence"),
        "Converted run_a to embedded patches: 24 points framed, 8 images read",
        "{report}"
    );

    // The same operation, read back through the tool an agent polls, under the
    // same name and id the handle carried.
    let over = call(&mut state, &mut viewer, "get_background_tasks", json!({}))["slow"]
        ["last_finished"]
        .clone();
    assert_eq!(over["running"], json!(false), "{over}");
    assert_eq!(over["finished"], json!(true), "{over}");
    assert_eq!(
        over["operation"],
        json!("Convert to embedded patches"),
        "{over}"
    );
    assert_eq!(over["reconstruction_label"], json!("run_a"), "{over}");
    assert_eq!(over["operation_id"], json!(pending.operation_id), "{over}");

    let after = call(&mut state, &mut viewer, "get_scene", json!({}));
    let node = &after["scene"][0];
    assert_eq!(node["feature_source"], json!("embedded_patches"), "{after}");
    // This fixture has no photographs to render from, so the narrower field beside
    // the feature source does not move.
    assert_eq!(node["has_patch_data"], json!(false), "{after}");
}

/// A node that already carries embedded patches is refused inline, in the
/// sentence the greyed menu entry carries, and nothing is pushed.
#[test]
fn convert_to_embedded_patches_is_refused_on_an_embedded_node() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "convert_to_embedded_patches",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(error.0.contains("already an embedded_patches"), "{error}");
    assert_eq!(version_count(&state), 1);
    assert!(state.running_tasks().next().is_none());
}

/// The id in a handle goes on naming its operation after that operation has
/// finished, which is what an agent comes back with.
#[test]
fn a_handle_s_id_still_names_the_operation_once_it_has_finished() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        _ => panic!("bundle_adjust must defer"),
    };
    state.finish_background_tasks();
    assert!(state.running_tasks().next().is_none());

    let reply = match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        other => panic!("expected the version, got {}", other.is_err()),
    };
    let cursor = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["cursor"], json!(cursor), "{reply}");
    assert!(reply["report"]
        .as_str()
        .expect("a report")
        .contains("Bundle adjusted run_a"));

    // An id that names no operation this session is never answered with
    // somebody else's run: it is told the outcome is gone and where to look.
    let stranger = super::super::BackgroundReply {
        operation_id: pending.operation_id + 1,
        ..pending
    };
    match super::super::edit::background_reply(&state, &stranger).expect("an answer, not a wait") {
        Err(error) => assert!(error.0.contains("get_action_log"), "{error}"),
        Ok(_) => panic!("a stale handle was answered with another operation's version"),
    }
}

/// A deferred reply finds its own task by id when another task, from the other
/// queue, landed after it and before the reply was resolved: the case one slot
/// of "the last finished task" answered with the other task's outcome.
#[test]
fn a_deferred_reply_finds_its_task_after_another_has_landed() {
    let (mut state, _viewer) = editable();
    let first = state.scene[0].id;
    state.append_node(crate::scene::SceneNode::demo(
        sfmtool_core::SfmrReconstruction::demo(8),
    ));
    let second = state.scene[1].id;
    let slow = state
        .start_background_task(
            crate::background::Operation::BUNDLE_ADJUST,
            first,
            Box::new(|_progress| crate::background::Finished::Failed("the solve's own".into())),
        )
        .expect("the slow queue is idle");
    let task = state.running_task(slow).expect("running");
    let pending = super::super::BackgroundReply {
        operation_id: slow,
        operation_name: task.operation.name,
        queue: task.operation.queue,
        answer: super::super::Answer::Version(first),
        label: task.label.clone(),
        queued: task.started,
    };
    let fast = state
        .start_background_task(
            crate::background::Operation::BENCH_FIT,
            second,
            Box::new(|_progress| crate::background::Finished::Failed("the fit's own".into())),
        )
        .expect("the fast queue is idle");
    state.finish_background_task(slow);
    state.finish_background_task(fast);
    assert_eq!(state.last_finished_task().map(|task| task.id), Some(fast));

    match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Err(error) => assert_eq!(error.0, "the solve's own", "{error}"),
        Ok(_) => panic!("a failed solve was answered with a version"),
    }
}

/// Cancelling is refused when there is nothing to cancel, and stops the
/// operation when there is.
#[test]
fn cancel_background_tasks_stops_what_is_running_and_refuses_when_nothing_is() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({}),
    );
    assert!(error.0.contains("Nothing is running"), "{error}");

    perturb(&mut state, 0.02);
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(_)) => {}
        _ => panic!("bundle_adjust must defer"),
    }
    let task = state.running_tasks().next().expect("running").id;
    let reply = call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({}),
    );
    assert_eq!(
        reply,
        json!({
            "cancelling": [{
                "operation": "Bundle adjust",
                "reconstruction_label": "run_a",
                "operation_id": task,
            }],
            "removed": [],
            "refused": [],
        }),
        "{reply}"
    );

    state.finish_background_tasks();
    // Whether the solve reached a poll before finishing is a race, so what is
    // asserted is the flag's effect on the state machine rather than which of
    // the two rows landed: either way the operation is over and the node free.
    assert!(state.running_tasks().next().is_none());
    assert_eq!(state.busy_refusal(state.scene[0].id), None);

    // A task named after it has finished is refused by name, with the
    // sentence the panel's tooltip would carry, and so is one never started.
    let reply = call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({ "operation_ids": [task, task + 1] }),
    );
    assert_eq!(reply["cancelling"], json!([]), "{reply}");
    assert_eq!(
        reply["refused"],
        json!([
            {
                "operation": "Bundle adjust",
                "reconstruction_label": "run_a",
                "operation_id": task,
                "reason": "Bundle adjust of run_a has already finished.",
            },
            {
                "operation_id": task + 1,
                "reason": format!("No background task with operation_id {} is running.", task + 1),
            },
        ]),
        "{reply}"
    );
}

/// With a task in each queue, cancelling with no argument asks both to stop,
/// and naming one asks that one alone.
#[test]
fn cancel_background_tasks_names_one_or_stops_every_running_task() {
    let (mut state, mut viewer) = editable();
    let first = state.scene[0].id;
    state.append_node(crate::scene::SceneNode::demo(
        sfmtool_core::SfmrReconstruction::demo(8),
    ));
    let second = state.scene[1].id;
    let held = |state: &mut AppState, operation, node| {
        let (open, wait) = std::sync::mpsc::channel::<()>();
        let task = state
            .start_background_task(
                operation,
                node,
                Box::new(move |progress| {
                    let _ = wait.recv();
                    if progress.is_cancelled() {
                        crate::background::Finished::Cancelled
                    } else {
                        crate::background::Finished::Failed("not cancelled".to_string())
                    }
                }),
            )
            .expect("its queue is idle");
        (task, open)
    };
    let (slow, open_slow) = held(
        &mut state,
        crate::background::Operation::BUNDLE_ADJUST,
        first,
    );
    let (fast, open_fast) = held(&mut state, crate::background::Operation::BENCH_FIT, second);

    let reply = call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({ "operation_ids": [fast] }),
    );
    let named: Vec<&Value> = reply["cancelling"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|task| &task["operation_id"])
        .collect();
    assert_eq!(named, [&json!(fast)], "{reply}");
    open_fast.send(()).expect("the fit is waiting");
    state.finish_background_task(fast);
    assert_eq!(
        state.finished_task(fast).map(|task| task.outcome.is_err()),
        Some(true)
    );
    assert!(
        state.running_task(slow).is_some(),
        "the solve was cancelled too"
    );

    let reply = call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({}),
    );
    let named: Vec<&Value> = reply["cancelling"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|task| &task["operation_id"])
        .collect();
    assert_eq!(named, [&json!(slow)], "{reply}");
    open_slow.send(()).expect("the solve is waiting");
    state.finish_background_task(slow);
    let ended = state.finished_task(slow).expect("it finished");
    assert_eq!(
        ended.outcome,
        Err("Bundle adjust of run_a cancelled".to_string())
    );
}

// ── What an agent reads about a background operation ────────────────────
//
// The seam under these is `background::tests`, which drives a real worker over
// the real channel. What is asserted here is only what reaches the wire: one
// shape for a running operation and a finished one, a `get_scene` block that
// does not grow with the solve, and a breakdown that is the entry's.

/// Start a fake operation on `run_a` that reports through `work` and then waits,
/// so a test can read a running operation at an instant it chose rather than
/// one the scheduler did.
///
/// Returns the sender that lets the job finish. A fake rather than a real
/// adjustment because what is under test is the reply: a solve fast enough to
/// be deterministic reports nothing worth reading, and one slow enough to read
/// is a test that takes a minute.
pub(super) fn running_operation(
    state: &mut AppState,
    work: impl FnOnce(&sfmtool_core::progress::Progress<'_>) + Send + 'static,
) -> std::sync::mpsc::Sender<()> {
    let id = state.scene[0].id;
    let (open, held) = std::sync::mpsc::channel::<()>();
    let (said, heard) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |progress| {
                work(progress);
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                crate::background::Finished::Failed("the fake worker produced nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_tasks();
    open
}

/// A job that reports the shape a solve reports: a stage that closed, a stage
/// with a folded child and a count, and a stage still open when it is read.
fn reporting_job(progress: &sfmtool_core::progress::Progress<'_>) {
    drop(progress.phase("gather arrays"));
    {
        let solve = progress.phase("solve");
        for round in 1..=2u64 {
            drop(solve.phase("round"));
            solve.count(round, Some(3), "round");
        }
    }
    // Left open on purpose: the open stage is what `phase` names, and a reply
    // that only ever saw closed ones would not exercise it.
    std::mem::forget(progress.phase("damping ladder"));
}

/// The keys of a JSON object, sorted, for asserting a block's whole shape
/// rather than the fields a test happened to think of.
#[track_caller]
pub(super) fn keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .unwrap_or_else(|| panic!("expected an object, got {value}"))
        .keys()
        .cloned()
        .collect();
    keys.sort();
    keys
}

/// One call answers both "is it still going" and "what did it cost", and the
/// two answers are the same shape with `running` telling them apart.
#[test]
fn get_background_tasks_reads_one_shape_running_and_finished() {
    let (mut state, mut viewer) = editable();

    // Nothing has run: each queue is there, running nothing and having
    // finished nothing.
    let idle = call(&mut state, &mut viewer, "get_background_tasks", json!({}));
    assert_eq!(
        idle,
        json!({
            "slow": { "running": null, "waiting": [], "last_finished": null },
            "fast": { "running": null, "waiting": [], "last_finished": null },
        }),
        "{idle}"
    );

    let open = running_operation(&mut state, reporting_job);
    let operation_id = state.running_tasks().next().expect("running").id;
    let by_id = json!({ "operation_id": operation_id });
    let live = call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        by_id.clone(),
    );
    // The queue's view and the task's own are one answer.
    let queues = call(&mut state, &mut viewer, "get_background_tasks", json!({}));
    assert_eq!(
        queues["slow"]["running"]["operation_id"],
        live["operation_id"]
    );
    assert_eq!(queues["fast"]["running"], Value::Null, "{queues}");
    assert_eq!(live["running"], json!(true), "{live}");
    assert_eq!(live["queue"], json!("slow"), "{live}");
    assert_eq!(live.get("finished"), None, "{live}");
    assert_eq!(live["operation"], json!("Bundle adjust"), "{live}");
    assert_eq!(live["reconstruction_label"], json!("run_a"), "{live}");
    assert_eq!(live["operation_id"], json!(operation_id), "{live}");
    assert_eq!(live["cancellable"], json!(true), "{live}");
    assert!(
        live["elapsed_s"].as_f64().expect("a duration") >= 0.0,
        "{live}"
    );
    // The kernel's own count, unit and all, and the stage it is inside.
    assert_eq!(
        live["progress"],
        json!({ "done": 2, "total": 3, "unit": "round" }),
        "{live}"
    );
    assert_eq!(live["phase"], json!("damping ladder"), "{live}");
    // The stages it has reported so far, in the Background panel's own order
    // and spelling, which is unfolded: the two rounds are two rows, because
    // this tool answers about an operation being watched rather than one read
    // about afterwards.
    assert_eq!(
        wire_detail_lines(&json!({ "detail": live["phases"] })),
        [
            "gather arrays",
            "solve",
            "  round",
            "  round",
            "damping ladder"
        ],
        "{live}"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();

    let over = call(&mut state, &mut viewer, "get_background_tasks", by_id);
    let queues = call(&mut state, &mut viewer, "get_background_tasks", json!({}));
    assert_eq!(queues["slow"]["running"], Value::Null, "{queues}");
    assert_eq!(queues["slow"]["last_finished"], over, "{queues}");
    assert_eq!(over["running"], json!(false), "{over}");
    assert_eq!(over["finished"], json!(true), "{over}");
    // The same operation, under the same names: an agent parses one shape.
    for field in ["operation", "queue", "reconstruction_label", "operation_id"] {
        assert_eq!(over[field], live[field], "{field} moved: {over}");
    }
    assert!(
        over["elapsed_s"].as_f64().expect("a cost") >= live["elapsed_s"].as_f64().expect("so far"),
        "the finished cost is shorter than the elapsed it was read at: {over}"
    );
    // How it ended, in the Action Log's own two fields and its own words.
    assert_eq!(over["failed"], json!(true), "{over}");
    assert_eq!(
        over["text"],
        json!("the fake worker produced nothing"),
        "{over}"
    );

    // An id no task has had is refused rather than answered with another's.
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        json!({ "operation_id": operation_id + 1 }),
    );
    assert!(
        error.0.contains(&format!(
            "No background task has operation_id {}",
            operation_id + 1
        )),
        "{error}"
    );
}

/// The breakdown this tool reports is the transcript, and the Action Log entry
/// is the summary of that transcript: every run the wire reported is counted in
/// the row the entry folds it into.
///
/// The two present one operation differently on purpose, for the reason the
/// Background panel and the Action Log panel differ
/// (`specs/gui/operation-progress.md`). What they must never do is disagree
/// about what happened.
#[test]
fn the_entry_is_the_summary_of_the_breakdown_the_wire_reported() {
    let (mut state, mut viewer) = editable();
    let open = running_operation(&mut state, reporting_job);
    let since = state.action_log.revision();
    let task = state.running_tasks().next().expect("running").id;
    open.send(()).expect("the worker is waiting");
    state.finish_background_task(task);

    let reply = call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        json!({ "operation_id": task }),
    );
    let log = ok(&mut state, &mut viewer, action_log_detail(since));
    let row = row_starting(&log, "the fake worker produced nothing");
    let wire = reply["phases"].as_array().expect("an array");
    for folded in row["detail"].as_array().expect("an array") {
        if folded["kind"] != json!("phase") {
            continue;
        }
        // Absent means one, which is how this surface spells a stage that ran
        // once.
        let runs = folded["runs"].as_u64().unwrap_or(1);
        let reported = wire
            .iter()
            .filter(|run| {
                run["kind"] == folded["kind"]
                    && run["name"] == folded["name"]
                    && run["depth"] == folded["depth"]
            })
            .count();
        assert_eq!(
            reported as u64, runs,
            "the entry folded {runs} runs of {} and the wire reported {reported}",
            folded["name"],
        );
    }
    assert_the_panel_agrees(&state, &row);
}

/// A breakdown on the wire is capped at the same size an entry's is, and says
/// how many it dropped. The **last** rows rather than the first, because this
/// is a transcript and nothing has collapsed the repetition in it: the first
/// hundred and twenty-eight rows of a long solve are its first few seconds and
/// say nothing about where it has got to.
#[test]
fn a_long_breakdown_is_capped_and_says_how_many_it_dropped() {
    let (mut state, mut viewer) = editable();
    let reported = ActionLog::DETAIL_EVENTS + 40;
    let open = running_operation(&mut state, move |progress| {
        for i in 0..reported {
            progress.message(Level::Info, format_args!("event {i}"));
        }
    });

    let task = json!({ "operation_id": state.running_tasks().next().expect("running").id });
    let live = call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        task.clone(),
    );
    let rows = live["phases"].as_array().expect("an array").clone();
    assert_eq!(rows.len(), ActionLog::DETAIL_EVENTS + 1, "{live}");
    // What it left out is said first, and what it kept is the recent end.
    assert_eq!(
        rows[0]["text"],
        json!("40 earlier events dropped"),
        "{live}"
    );
    assert_eq!(rows[1]["text"], json!("event 40"), "{live}");
    assert_eq!(
        rows[ActionLog::DETAIL_EVENTS]["text"],
        json!(format!("event {}", reported - 1)),
        "{live}"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();
    let over = call(&mut state, &mut viewer, "get_background_tasks", task);
    assert_eq!(over["phases"], live["phases"], "{over}");
}

/// `get_scene` says the viewer is busy and stops there.
///
/// The most-polled tool on the surface carries a list whose entries' size does
/// not depend on the operation: no phase table, no open phase, no status line,
/// and nothing at all once the operation is over.
#[test]
fn get_scene_says_the_viewer_is_busy_without_carrying_the_solve() {
    let (mut state, mut viewer) = editable();
    let idle = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(
        idle["background_tasks"],
        json!([]),
        "{}",
        idle["background_tasks"]
    );

    let open = running_operation(&mut state, |progress| {
        reporting_job(progress);
        progress.set_status_message(format_args!("refining images/IMG_0007.jpg"));
    });
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    let tasks = scene["background_tasks"].as_array().expect("a list");
    assert_eq!(tasks.len(), 1, "{tasks:?}");
    let block = &tasks[0];
    assert_eq!(
        keys(block),
        [
            "elapsed_s",
            "fraction",
            "operation",
            "operation_id",
            "queue",
            "reconstruction_label",
            "running",
        ],
        "the get_scene entry has grown: {block}"
    );
    assert_eq!(block["queue"], json!("slow"), "{block}");
    assert_eq!(block["running"], json!(true), "{block}");
    assert_eq!(block["operation"], json!("Bundle adjust"), "{block}");
    assert_eq!(block["reconstruction_label"], json!("run_a"), "{block}");
    // The one field a caller polls for movement, and the whole of what
    // `get_background_tasks` would add is absent here.
    assert!(block["fraction"].is_number(), "{block}");

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();
    // Empty again, rather than the operation that just ended: the list is
    // read as "is the viewer busy", and an entry that outlived the operation
    // would be carried by every poll for the rest of the session.
    let after = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(after["background_tasks"], json!([]), "{after}");
}

/// A running task that has been asked to stop says `cancelling: true` in
/// `get_scene`'s entry and in `get_background_tasks`, from the request until it
/// lands, and carries no such field before it.
#[test]
fn a_task_asked_to_stop_says_cancelling_until_it_lands() {
    let (mut state, mut viewer) = editable();
    let open = running_operation(&mut state, reporting_job);
    let id = state.running_tasks().next().expect("running").id;
    let task = json!({ "operation_id": id });

    let before = call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        task.clone(),
    );
    assert!(before.get("cancelling").is_none(), "{before}");

    call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({ "operation_ids": [id] }),
    );
    let asked = call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        task.clone(),
    );
    assert_eq!(asked["cancelling"], json!(true), "{asked}");
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(
        scene["background_tasks"][0]["cancelling"],
        json!(true),
        "{scene}"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();
    let over = call(&mut state, &mut viewer, "get_background_tasks", task);
    assert!(over.get("cancelling").is_none(), "{over}");
}

/// A task that waits reaches the wire as one that has not started: its handle
/// carries `waiting: true`, with the window for a direct answer measured from
/// when it was queued; `get_background_tasks` lists it under its queue's
/// `waiting` and answers about it by id without phases; `get_scene` lists it
/// with `waiting`; and `cancel_background_tasks` with no argument removes it,
/// after which the deferred reply answers with its cancelled row.
#[test]
fn a_waiting_task_is_reported_waiting_and_removed_by_a_cancel() {
    let (mut state, mut viewer) = editable();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_b.sfmr",
    ));
    let open = running_operation(&mut state, reporting_job);
    let ahead = state.running_tasks().next().expect("running").id;
    let map = json!({ "reconstruction_label": "run_b" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("a busy queue refused the call: {e}"),
        _ => panic!("bundle_adjust must defer"),
    };
    let queued = pending.operation_id;
    assert!(state.waiting_task(queued).is_some(), "it did not wait");

    // Inside the window from when it was queued there is no answer yet; past
    // it, the handle says the task has not started.
    assert!(super::super::edit::background_reply(&state, &pending).is_none());
    let past = super::super::BackgroundReply {
        queued: std::time::Instant::now() - super::super::REPLY_DIRECTLY_WITHIN,
        ..pending
    };
    let handle = match super::super::edit::background_reply(&state, &past).expect("past the window")
    {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("a handle is JSON"),
    };
    assert_eq!(
        handle,
        json!({
            "running": true,
            "waiting": true,
            "operation": "Bundle adjust",
            "queue": "slow",
            "reconstruction_label": "run_b",
            "operation_id": queued,
        }),
        "{handle}"
    );

    let waiting = json!({
        "running": true,
        "waiting": true,
        "operation": "Bundle adjust",
        "queue": "slow",
        "reconstruction_label": "run_b",
        "operation_id": queued,
    });
    let queues = call(&mut state, &mut viewer, "get_background_tasks", json!({}));
    assert_eq!(queues["slow"]["waiting"], json!([waiting]), "{queues}");
    assert_eq!(
        queues["slow"]["running"]["operation_id"],
        json!(ahead),
        "{queues}"
    );
    assert_eq!(queues["fast"]["waiting"], json!([]), "{queues}");
    let by_id = call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        json!({ "operation_id": queued }),
    );
    assert_eq!(by_id, waiting, "{by_id}");
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    let tasks = scene["background_tasks"].as_array().expect("a list");
    assert_eq!(tasks.len(), 2, "{tasks:?}");
    assert_eq!(tasks[0]["operation_id"], json!(ahead), "{tasks:?}");
    assert_eq!(tasks[1], waiting, "{tasks:?}");

    let reply = call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({}),
    );
    assert_eq!(
        reply,
        json!({
            "cancelling": [{
                "operation": "Bundle adjust",
                "reconstruction_label": "run_a",
                "operation_id": ahead,
            }],
            "removed": [{
                "operation": "Bundle adjust",
                "reconstruction_label": "run_b",
                "operation_id": queued,
            }],
            "refused": [],
        }),
        "{reply}"
    );
    let removed = "Bundle adjust of run_b cancelled before it started";
    let over = call(
        &mut state,
        &mut viewer,
        "get_background_tasks",
        json!({ "operation_id": queued }),
    );
    assert_eq!(over["finished"], json!(true), "{over}");
    assert_eq!(over["failed"], json!(true), "{over}");
    assert_eq!(over["text"], json!(removed), "{over}");
    match super::super::edit::background_reply(&state, &past).expect("it is over") {
        Err(error) => assert_eq!(error.0, removed),
        Ok(_) => panic!("a removed task was answered with a version"),
    }

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();
    assert!(state.waiting_tasks().next().is_none());
}

/// The apply timeout names what is running, and keeps its old guesses when
/// nothing is.
#[test]
fn the_timeout_message_names_an_operation_only_while_one_is_running() {
    let idle = super::super::server::timeout_message(&[]);
    assert!(idle.contains("did not answer within 10 seconds"), "{idle}");
    assert!(idle.contains("modal dialog"), "{idle}");
    assert!(!idle.contains("get_background_tasks"), "{idle}");

    let adjusting = crate::background::Busy {
        operation: "Bundle adjust",
        label: "dino_dog_toy-embedded".to_string(),
    };
    let busy = super::super::server::timeout_message(std::slice::from_ref(&adjusting));
    assert!(busy.contains("did not answer within 10 seconds"), "{busy}");
    assert!(
        busy.contains(
            "Bundle adjust on dino_dog_toy-embedded is running in the background; \
             get_background_tasks says how far along it has got."
        ),
        "{busy}"
    );
    // The two guesses are gone: they name things that did not happen.
    assert!(!busy.contains("modal dialog"), "{busy}");
    assert!(!busy.contains("mid-drag"), "{busy}");

    // With a task in each queue, it names both.
    let both = super::super::server::timeout_message(&[
        adjusting,
        crate::background::Busy {
            operation: "Fit track",
            label: "bull".to_string(),
        },
    ]);
    assert!(
        both.contains(
            "Bundle adjust on dino_dog_toy-embedded and Fit track on bull are running in the \
             background; get_background_tasks says how far along each has got."
        ),
        "{both}"
    );
}

/// The notice the message reads is written where the task is written, so the
/// two cannot disagree about whether anything is running.
#[test]
fn the_busy_notice_tracks_the_operation() {
    let (mut state, _viewer) = editable();
    assert_eq!(crate::background::busy(&state.busy_notice), []);

    let open = running_operation(&mut state, |_| {});
    assert_eq!(
        crate::background::busy(&state.busy_notice),
        [crate::background::Busy {
            operation: "Bundle adjust",
            label: "run_a".to_string(),
        }]
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();
    assert_eq!(crate::background::busy(&state.busy_notice), []);
}

/// Undo, redo and the jump answer with the version now showing, and refuse at
/// the ends in the state's words.
#[test]
fn the_cursor_moves_answer_with_the_version_now_showing() {
    let (mut state, mut viewer) = editable();
    let first = state.scene[0].history.current_version().serial.to_string();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    let edited = state.scene[0].history.current_version().serial.to_string();

    let undone = call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(undone["cursor"], first);
    assert_eq!(undone["dirty"], false);
    // A cursor move made no version, so it reports none of its own.
    assert_eq!(undone["report"], Value::Null);

    let error = refused_call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(error.0, "Nothing to undo in run_a.");

    let redone = call(
        &mut state,
        &mut viewer,
        "redo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(redone["cursor"], edited);

    let jumped = call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": first }),
    );
    assert_eq!(jumped["cursor"], first);
    assert_eq!(jumped["label"], state.scene[0].history.versions()[0].label);
}

/// A serial the node does not hold is refused naming the read that lists them.
#[test]
fn a_serial_that_is_not_a_version_of_the_node_is_refused() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": "v99999" }),
    );
    assert!(error.0.contains("v99999"), "{error}");
    assert!(error.0.contains("get_reconstruction_history"), "{error}");
}

/// Every refusal of a cursor move, on the wire and in the log: the error is the
/// state's sentence, and the one failed row carries it under the tool's name.
///
/// The one sentence that is the wire's own is a serial the node never minted,
/// because the wire receives a string the state never sees: it is quoted as it
/// arrived, and the refusal names the read that lists the spellings it takes.
#[test]
fn the_cursor_moves_refuse_on_the_wire_in_the_states_words() {
    let (mut state, mut viewer) = editable();
    state.action_log.clear();
    let label = json!({ "reconstruction_label": "run_a" });
    let undo = refused_call(&mut state, &mut viewer, "undo", label.clone());
    assert_eq!(undo.0, "Nothing to undo in run_a.");
    let redo = refused_call(&mut state, &mut viewer, "redo", label.clone());
    assert_eq!(redo.0, "Nothing to redo in run_a.");

    for point in [3, 4] {
        call(
            &mut state,
            &mut viewer,
            "delete_point",
            json!({ "reconstruction_label": "run_a", "point": point }),
        );
    }
    let here = state.scene[0].history.current_version().serial.to_string();
    let first = state.scene[0].history.versions()[0].serial.to_string();
    let jump = |serial: &str| json!({ "reconstruction_label": "run_a", "serial": serial });
    let already = refused_call(&mut state, &mut viewer, "jump_to_version", jump(&here));
    assert_eq!(already.0, format!("{here} is already what run_a shows."));
    let unknown = refused_call(&mut state, &mut viewer, "jump_to_version", jump("v99999"));
    assert_eq!(
        unknown.0,
        "\"v99999\" is not a version of run_a; get_reconstruction_history lists its \
         versions, spelled as \"v12\"."
    );
    state.scene[0].history.versions_mut_for_test()[1].value = None;
    let released = refused_call(&mut state, &mut viewer, "jump_to_version", jump(&first));
    let released_text = format!(
        "Cannot go to {first}: its value, or one on the way to it, was released to keep run_a \
         inside the history budget."
    );
    assert_eq!(released.0, released_text);

    let failed: Vec<String> = failures(&state).into_iter().map(|row| row.2).collect();
    assert_eq!(
        failed,
        [
            "undo failed: Nothing to undo in run_a.".to_string(),
            "redo failed: Nothing to redo in run_a.".to_string(),
            format!("jump_to_version failed: {here} is already what run_a shows."),
            format!("jump_to_version failed: {}", unknown.0),
            format!("jump_to_version failed: {released_text}"),
        ]
    );
    assert!(failures(&state).iter().all(|row| row.0 == Actor::Mcp));
}

/// A node with an operation running on it refuses all three moves with the
/// sentence every other edit of it gets, and the cursor stays.
#[test]
fn the_cursor_moves_refuse_a_busy_node_on_the_wire() {
    let (mut state, mut viewer) = editable();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    let first = state.scene[0].history.versions()[0].serial.to_string();
    let open = running_operation(&mut state, |_| {});
    let busy = "run_a is busy: Bundle adjust is still running.";

    for (tool, arguments) in [
        ("undo", json!({ "reconstruction_label": "run_a" })),
        ("redo", json!({ "reconstruction_label": "run_a" })),
        (
            "jump_to_version",
            json!({ "reconstruction_label": "run_a", "serial": first }),
        ),
    ] {
        let error = refused_call(&mut state, &mut viewer, tool, arguments);
        assert_eq!(error.0, busy, "{tool}");
    }
    assert_eq!(state.scene[0].history.cursor(), 1, "a refusal moved");

    open.send(()).expect("the worker is waiting");
    state.finish_background_tasks();
}

/// `get_reconstruction_history` is the Edit History panel's reading of the same
/// list: every version in order, the cursor, the version on disk, and the
/// released rows.
#[test]
fn get_reconstruction_history_lists_the_versions_with_the_cursor_and_the_released_rows() {
    let (mut state, mut viewer) = editable();
    for point in [3, 4] {
        call(
            &mut state,
            &mut viewer,
            "delete_point",
            json!({ "reconstruction_label": "run_a", "point": point }),
        );
    }
    // The budget releases a version's value while keeping its row. Reaching the
    // real budget from a test would mean a reconstruction of gigabytes, so the
    // release itself is what is arranged -- on the oldest version, which is the
    // one the budget takes first and the one an undo does not need.
    state.scene[0].history.versions_mut_for_test()[0].value = None;

    let history = call(
        &mut state,
        &mut viewer,
        "get_reconstruction_history",
        json!({ "reconstruction_label": "run_a" }),
    );
    let versions = history["versions"].as_array().expect("a version list");
    assert_eq!(versions.len(), 3);
    assert_eq!(history["reconstruction_label"], "run_a");
    assert_eq!(history["dirty"], true);
    assert_eq!(history["can_undo"], true);
    assert_eq!(history["can_redo"], false);
    assert_eq!(history["cursor"], versions[2]["serial"]);
    assert_eq!(history["disk_serial"], versions[0]["serial"]);

    assert_eq!(versions[0]["is_on_disk"], true);
    assert_eq!(versions[2]["is_cursor"], true);
    assert_eq!(versions[0]["is_cursor"], false);
    assert_eq!(versions[0]["held"], false, "the released row says so");
    assert_eq!(versions[2]["held"], true);
    assert!(
        versions[2]["label"]
            .as_str()
            .expect("a label")
            .starts_with("Deleted point 4"),
        "{history}"
    );
    // The same instant format the Action Log's own rows carry.
    let at = versions[0]["at"].as_str().expect("a timestamp");
    assert!(at.len() >= 19 && at.as_bytes()[10] == b'T', "{at}");
}

/// A node that came from no file has no version on disk, and says so rather
/// than naming one.
#[test]
fn a_node_from_no_file_reports_no_version_on_disk() {
    let (mut state, mut viewer) = two_reconstructions();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));
    let history = call(
        &mut state,
        &mut viewer,
        "get_reconstruction_history",
        json!({ "reconstruction_label": "demo" }),
    );
    assert_eq!(history["path"], Value::Null);
    assert_eq!(history["disk_serial"], Value::Null);
    assert_eq!(history["versions"][0]["is_on_disk"], false);
}

/// A directory of this test's own under the system temp dir, emptied first so a
/// rerun does not read a previous run's file.
pub(super) fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("sfm_explorer_mcp_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// Save As writes where it is told and re-points the node; Save afterwards
/// writes over what Save As chose.
#[test]
fn save_reconstruction_writes_to_a_path_and_then_over_it() {
    let dir = temp_dir("save");
    let path = dir.join("edited.sfmr");
    // A plain demo node rather than [`editable`]'s: what a save is about is the
    // file it writes, and this one is a value the writer accepts whole.
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        &dir.join("recon.sfmr"),
        SfmrReconstruction::demo(64),
    ));
    let mut viewer = Viewer3D::new();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "recon", "point": 3 }),
    );

    let saved = call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "recon", "path": path.display().to_string() }),
    );
    assert!(path.exists(), "the file was not written");
    // Save As re-points and re-labels the node, so the reply says what to call
    // it next time.
    assert_eq!(saved["reconstruction_label"], "edited");
    assert_eq!(saved["path"], path.display().to_string());
    assert_eq!(
        saved["serial"],
        state.scene[0].history.disk_serial().to_string()
    );
    assert!(!state.is_dirty(state.scene[0].id), "a save leaves it clean");

    // And with no path it goes over the file the node now has.
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "edited", "point": 4 }),
    );
    let again = call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "edited" }),
    );
    assert_eq!(again["path"], path.display().to_string());
    assert!(!state.is_dirty(state.scene[0].id));
}

/// A node that came from no file is refused a pathless save, in the words the
/// File menu's own Save uses.
#[test]
fn saving_a_node_from_no_file_without_a_path_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));
    let error = refused_call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "demo" }),
    );
    assert!(error.0.contains("came from no file"), "{error}");
}

/// `minimal: true` writes a copy where it is told and leaves the node where it
/// was; it needs a path, and never goes over the node's own file.
#[test]
fn save_reconstruction_minimal_writes_a_copy_and_leaves_the_node() {
    let dir = temp_dir("save_minimal");
    let own = dir.join("recon.sfmr");
    let copy = dir.join("recon-minimal.sfmr");
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(&own, SfmrReconstruction::demo(64)));
    let mut viewer = Viewer3D::new();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "recon", "point": 3 }),
    );
    let points = state.scene[0].point_count();

    let reply = call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({
            "reconstruction_label": "recon",
            "path": copy.display().to_string(),
            "minimal": true,
        }),
    );
    assert_eq!(reply["minimal"], true);
    assert_eq!(reply["reconstruction_label"], "recon");
    assert_eq!(reply["path"], copy.display().to_string());
    let written = sfmtool_sfmr_format::read_sfmr(&copy).expect("a readable copy");
    assert!(written.thumbnails_y_x_rgb.is_none());
    assert!(written.metadata.workspace.absolute_path.is_empty());
    assert_eq!(written.metadata.operation, "minimal");
    assert_eq!(written.metadata.point_count as usize, points);
    let node = &state.scene[0];
    assert_eq!(node.path.as_deref(), Some(own.as_path()));
    assert!(
        state.is_dirty(node.id),
        "a minimal copy saves nothing of the node"
    );

    let error = refused_call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "recon", "minimal": true }),
    );
    assert!(error.0.contains("pass path"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `workspace_path` is recorded as it stands, wherever the file is written, and
/// it needs a path of its own: a save over the node's own file leaves the file
/// where it already is, so the path it records is already the right one.
#[test]
fn save_reconstruction_states_the_workspace_path_and_needs_a_path() {
    let dir = temp_dir("save_workspace_path");
    let own = dir.join("recon.sfmr");
    let copy = dir.join("published").join("recon.sfmr");
    std::fs::create_dir_all(copy.parent().expect("a parent")).expect("a writable temp dir");
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(&own, SfmrReconstruction::demo(32)));
    let mut viewer = Viewer3D::new();

    call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({
            "reconstruction_label": "recon",
            "path": copy.display().to_string(),
            "minimal": true,
            "workspace_path": "../shared/ws",
        }),
    );
    let written = sfmtool_sfmr_format::read_sfmr(&copy).expect("a readable copy");
    assert_eq!(written.metadata.workspace.relative_path, "../shared/ws");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "recon", "workspace_path": "." }),
    );
    assert!(error.0.contains("pass path with workspace_path"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The editing tools name their reconstruction rather than defaulting to the
/// selection, and an unknown label is refused naming what is loaded.
#[test]
fn every_editing_tool_requires_its_reconstruction_label() {
    let (mut state, mut viewer) = editable();
    for (name, arguments) in [
        ("get_reconstruction_history", json!({})),
        ("undo", json!({})),
        ("redo", json!({})),
        ("jump_to_version", json!({ "serial": "v0" })),
        ("save_reconstruction", json!({})),
        ("delete_point", json!({ "point": 3 })),
        ("delete_camera_image", json!({ "camera_image": 1 })),
        (
            "move_camera_image",
            json!({ "camera_image": 1, "world_from_camera": {
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0] } }),
        ),
        ("resect_camera_image", json!({ "camera_image": 1 })),
        ("add_camera_image_to_tracks", json!({ "camera_image": 1 })),
        ("bundle_adjust", json!({})),
        ("convert_to_embedded_patches", json!({})),
    ] {
        let error = refused_call(&mut state, &mut viewer, name, arguments);
        assert!(error.0.contains("reconstruction_label"), "{name}: {error}");
    }

    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_reconstruction_history",
        json!({ "reconstruction_label": "run_b" }),
    );
    assert!(error.0.contains("run_a"), "{error}");
}

/// A point id belonging to another reconstruction is refused rather than
/// edited: the two handles one call carries have to name the same node.
#[test]
fn a_point_from_another_reconstruction_is_refused() {
    let (mut state, mut viewer) = editable();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/other.sfmr"),
        SfmrReconstruction::demo(16),
    ));
    let other = state.scene[1].id;
    let id = crate::point_ids::mint(&state.scene[1], 2).expect("a point id");
    state.select_recon(other);

    let error = refused_call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": id }),
    );
    assert!(error.0.contains("other"), "{error}");
    assert_eq!(version_count(&state), 1);
}

/// The editing arguments are parsed by shape before anything is applied.
#[test]
fn the_editing_arguments_are_parsed_by_shape() {
    for (name, arguments, expected) in [
        (
            "jump_to_version",
            json!({ "reconstruction_label": "run_a", "serial": 4 }),
            "serial",
        ),
        (
            "move_camera_image",
            json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
            "world_from_camera",
        ),
        (
            "move_camera_image",
            json!({ "reconstruction_label": "run_a", "camera_image": 1,
                    "world_from_camera": { "quaternion_wxyz": [1.0, 0.0, 0.0],
                                           "translation": [0.0, 0.0, 0.0] } }),
            "quaternion_wxyz",
        ),
        (
            "bundle_adjust",
            json!({ "reconstruction_label": "run_a", "release_focal": "yes" }),
            "release_focal",
        ),
        (
            "save_reconstruction",
            json!({ "reconstruction_label": "run_a", "path": 7 }),
            "path",
        ),
        (
            "delete_point",
            json!({ "reconstruction_label": "run_a", "point_index": 3 }),
            "point_index",
        ),
        (
            "undo",
            json!({ "reconstruction_label": "run_a", "steps": 2 }),
            "steps",
        ),
    ] {
        let map = arguments.as_object().cloned().expect("an object");
        let error = tools::parse(name, Some(&map)).expect_err("refused at the parse");
        assert!(error.0.contains(expected), "{name}: {error}");
    }
}

/// The optional arguments have the defaults the tools advertise.
#[test]
fn the_editing_defaults_are_what_the_schemas_say() {
    let parse = |name: &str, arguments: Value| {
        let map = arguments.as_object().cloned().expect("an object");
        tools::parse(name, Some(&map)).expect("a well-formed call")
    };
    assert_eq!(
        parse(
            "resect_camera_image",
            json!({ "reconstruction_label": "a", "camera_image": "images/x.jpg" })
        ),
        Command::ResectCameraImage {
            reconstruction_label: "a".to_string(),
            camera_image: super::super::CameraImageSel::Name("images/x.jpg".to_string()),
        }
    );
    assert_eq!(
        parse(
            "add_camera_image_to_tracks",
            json!({ "reconstruction_label": "a", "camera_image": 2 })
        ),
        Command::AddCameraImageToTracks {
            reconstruction_label: "a".to_string(),
            camera_image: super::super::CameraImageSel::Index(2),
        }
    );
    // The pose arrives in a sub-object, and comes out of the parse as the two
    // arrays the command carries.
    assert_eq!(
        parse(
            "move_camera_image",
            json!({ "reconstruction_label": "a", "camera_image": 1,
                    "world_from_camera": { "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                                           "translation": [1.5, -2.0, 3.0] } })
        ),
        Command::MoveCameraImage {
            reconstruction_label: "a".to_string(),
            camera_image: super::super::CameraImageSel::Index(1),
            quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
            translation: [1.5, -2.0, 3.0],
        }
    );
    assert_eq!(
        parse(
            "bundle_adjust",
            json!({ "reconstruction_label": "a", "release_focal": true })
        ),
        Command::BundleAdjust {
            reconstruction_label: "a".to_string(),
            release_focal: true,
            release_distortion: false,
            release_principal_point: false,
            release_image_focal: false,
            release_image_principal_point: false,
            cameras: Vec::new(),
            free_points_cross: true,
            schedule: None,
            image_prior: Default::default(),
        }
    );
    assert_eq!(
        parse(
            "bundle_adjust",
            json!({
                "reconstruction_label": "a",
                "release_focal": true,
                "release_distortion": true
            })
        ),
        Command::BundleAdjust {
            reconstruction_label: "a".to_string(),
            release_focal: true,
            release_distortion: true,
            release_principal_point: false,
            release_image_focal: false,
            release_image_principal_point: false,
            cameras: Vec::new(),
            free_points_cross: true,
            schedule: None,
            image_prior: Default::default(),
        }
    );
    assert_eq!(
        parse(
            "bundle_adjust",
            json!({ "reconstruction_label": "a", "release_principal_point": true })
        ),
        Command::BundleAdjust {
            reconstruction_label: "a".to_string(),
            release_focal: false,
            release_distortion: false,
            release_principal_point: true,
            release_image_focal: false,
            release_image_principal_point: false,
            cameras: Vec::new(),
            free_points_cross: true,
            schedule: None,
            image_prior: Default::default(),
        }
    );
    assert_eq!(
        parse(
            "bundle_adjust",
            json!({ "reconstruction_label": "a", "free_points_cross": false })
        ),
        Command::BundleAdjust {
            reconstruction_label: "a".to_string(),
            release_focal: false,
            release_distortion: false,
            release_principal_point: false,
            release_image_focal: false,
            release_image_principal_point: false,
            cameras: Vec::new(),
            free_points_cross: false,
            schedule: None,
            image_prior: Default::default(),
        }
    );
    assert_eq!(
        parse(
            "save_reconstruction",
            json!({ "reconstruction_label": "a" })
        ),
        Command::SaveReconstruction {
            reconstruction_label: "a".to_string(),
            path: None,
            minimal: false,
            workspace_path: None,
        }
    );
}

// ── move_camera_image ───────────────────────────────────────────────────
//
// The pose edit the human makes with a lock and the viewport, which an agent
// makes by sending the pose: it has no hand to place a camera with, and the
// pose is the whole input.

/// A pose for `run_a`'s image `index`, turned and shifted off where it stands,
/// in the node's own frame -- which is the frame the tool takes and every read
/// on this surface reports.
fn moved_pose(state: &AppState, index: usize) -> (Vec<f64>, Vec<f64>) {
    let stored = sfmtool_core::reconstruction::move_camera::pose_of(state.scene[0].recon(), index);
    let turn = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.2_f64);
    let rotation = turn * stored.rotation.as_nalgebra();
    let quaternion = rotation.as_ref();
    (
        vec![quaternion.w, quaternion.i, quaternion.j, quaternion.k],
        vec![
            stored.translation.x + 0.3,
            stored.translation.y - 0.1,
            stored.translation.z + 0.05,
        ],
    )
}

/// One `move_camera_image` call, with the pose in the sub-object the wire
/// carries it in.
#[track_caller]
fn move_camera(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    index: usize,
    quaternion: &[f64],
    translation: &[f64],
) -> Value {
    call(
        state,
        viewer,
        "move_camera_image",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": index,
            "world_from_camera": {
                "quaternion_wxyz": quaternion,
                "translation": translation,
            },
        }),
    )
}

/// The pose edit pushes a version like every other edit here, and the reply is
/// that version rather than a second rendering of the pose: where the camera
/// now stands is what the reads answer.
#[test]
fn move_camera_image_pushes_a_version_the_reply_names() {
    let (mut state, mut viewer) = editable();
    let (quaternion, translation) = moved_pose(&state, 1);

    let reply = move_camera(&mut state, &mut viewer, 1, &quaternion, &translation);

    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["reconstruction_label"], "run_a");
    let serial = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["serial"], serial);
    assert_eq!(reply["cursor"], serial);
    assert_eq!(reply["dirty"], true);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Moved camera "), "{report}");
    assert!(report.contains(&serial), "{report}");
    assert!(
        reply["label"]
            .as_str()
            .expect("a label")
            .starts_with("Moved camera "),
        "{reply}"
    );

    // The camera stands where the call put it, in the node's own frame.
    let centre = state.scene[0].recon().image_set.images[1].camera_center();
    for (axis, expected) in translation.iter().enumerate() {
        assert!(
            (centre[axis] - expected).abs() < 1e-9,
            "centre {axis}: {centre:?}"
        );
    }

    // And it is the agent's row, under the kind every edit of a value is filed
    // under.
    let last = rows(&state).pop().expect("one entry per edit");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
}

/// An image the node does not have is refused before anything moves, and the
/// refusal is the one row it leaves.
#[test]
fn move_camera_image_refuses_an_image_that_is_not_there() {
    let (mut state, mut viewer) = editable();
    state.action_log.clear();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "move_camera_image",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 99,
            "world_from_camera": {
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
            },
        }),
    );
    assert!(error.0.contains("out of range"), "{error}");
    assert_eq!(version_count(&state), 1);
    assert_eq!(failures(&state).len(), 1, "{:?}", rows(&state));
}

/// A camera in hand on the node is ended before the wire's own pose lands on
/// it, and committed because it had been moved: two hands on one image, two
/// versions, in the order they happened.
#[test]
fn move_camera_image_commits_a_lock_held_on_the_same_node_first() {
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);

    let (quaternion, translation) = moved_pose(&state, 1);
    move_camera(&mut state, &mut viewer, 1, &quaternion, &translation);

    assert!(viewer.camera_lock.is_none(), "the lock survived an edit");
    assert_eq!(
        version_count(&state),
        3,
        "the hand's move and the wire's are two versions"
    );
}

/// A view command that leaves camera view is a step away from a camera in
/// hand, as `,` and `.` are: the lock ends first, as a commit when it has been
/// moved. A field-of-view change keeps camera view, so it keeps the lock.
#[test]
fn a_view_command_that_leaves_camera_view_commits_a_held_lock_first() {
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);

    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "fov_short_axis_deg": 40.0 }),
    );
    assert!(
        viewer.camera_lock.is_some(),
        "a field-of-view change ended the lock"
    );
    assert_eq!(version_count(&state), 1);

    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0] }),
    );
    assert!(
        viewer.camera_lock.is_none(),
        "the lock survived a placement"
    );
    assert!(
        viewer.camera_view.is_none(),
        "the placement left camera view up"
    );
    assert_eq!(version_count(&state), 2, "the hand's move was committed");
    let texts: Vec<String> = rows(&state).into_iter().map(|row| row.2).collect();
    assert!(
        texts.iter().any(|text| text.starts_with("Moved camera ")),
        "the commit recorded itself; the log holds {texts:?}"
    );
}

/// The lock ends before the form is applied, so a leaving form that then
/// refuses has still ended it -- as a commit, since the hand had moved -- while
/// the view itself stays where the hand left it.
#[test]
fn a_refused_view_command_that_leaves_camera_view_has_still_ended_the_lock() {
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);
    let held = crate::camera_lock::pending_pose(&viewer, &state.scene[0]);

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0], "fov_short_axis_deg": 170.0 }),
    );
    assert!(error.0.contains("fov_short_axis_deg"), "{error}");
    assert!(viewer.camera_lock.is_none(), "the lock survived");
    assert_eq!(version_count(&state), 2, "the hand's move was committed");
    assert_viewport_at(&state, &viewer, &held, "after the refused placement");
}

/// The rule is the wire's rather than the pose edit's: any edit landing on the
/// node ends the lock first, since an edit under one would leave the reviewer
/// holding a camera whose stored pose had moved beneath them.
#[test]
fn another_edit_on_the_node_commits_a_held_lock_too() {
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);

    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );

    assert!(viewer.camera_lock.is_none(), "the lock survived an edit");
    assert_eq!(version_count(&state), 3);
    // The commit is recorded before the edit that displaced it, as it happened.
    let texts: Vec<String> = rows(&state).into_iter().map(|row| row.2).collect();
    let moved = texts
        .iter()
        .position(|text| text.starts_with("Moved camera "))
        .expect("the commit recorded itself");
    let deleted = texts
        .iter()
        .position(|text| text.starts_with("Deleted point 3"))
        .expect("the edit recorded itself");
    assert!(moved < deleted, "{texts:?}");
}

/// Take image `index` of `run_a` in hand and move it past the dead band, which
/// is what makes an implicit end a commit rather than a silent drop.
fn hold_the_camera(state: &mut AppState, viewer: &mut Viewer3D, index: usize) {
    let image = crate::scene::ImageRef::new(state.scene[0].id, index);
    state.select_image(Some(image));
    viewer.jump_to_camera_view(image, &state.scene[0]);
    crate::camera_lock::enter(viewer, state).expect("camera view of a posed image");
    viewer.camera.nodal_pan(50.0, 0.0);
}

/// `run_a`'s stored pose for image `index`, in its own frame.
fn stored_pose(state: &AppState, index: usize) -> sfmtool_core::Se3Transform {
    sfmtool_core::reconstruction::move_camera::pose_of(state.scene[0].recon(), index)
}

/// Assert the viewport is standing at `expected`, in `run_a`'s own frame.
#[track_caller]
fn assert_viewport_at(
    state: &AppState,
    viewer: &Viewer3D,
    expected: &sfmtool_core::Se3Transform,
    what: &str,
) {
    let at = crate::camera_lock::pending_pose(viewer, &state.scene[0]);
    let degrees = at
        .rotation
        .as_nalgebra()
        .rotation_to(expected.rotation.as_nalgebra())
        .angle()
        .to_degrees();
    let distance = (at.translation - expected.translation).norm();
    assert!(
        degrees < 1e-9 && distance < 1e-9,
        "{what}: the viewport is {degrees} deg and {distance} away from the pose it should show"
    );
}

/// A cursor move can restore the pose of the very camera the viewport is
/// looking through, and camera view follows the value rather than staying where
/// a hand left it -- so all three moves re-snap it, as the Edit menu's own Undo
/// and Redo do.
#[test]
fn the_cursor_moves_resnap_a_camera_view_onto_the_version_they_land_on() {
    let (mut state, mut viewer) = editable();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 1);
    let first = state.scene[0].history.current_version().serial.to_string();
    let stored = stored_pose(&state, 1);

    // Moved on the wire, then looked through again: the viewport stands at the
    // new pose, which is what each cursor move below has to take it off.
    let (quaternion, translation) = moved_pose(&state, 1);
    move_camera(&mut state, &mut viewer, 1, &quaternion, &translation);
    let moved = stored_pose(&state, 1);
    state.select_image(Some(image));
    viewer.jump_to_camera_view(image, &state.scene[0]);
    assert_viewport_at(&state, &viewer, &moved, "the look-through");

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_viewport_at(&state, &viewer, &stored, "undo");

    call(
        &mut state,
        &mut viewer,
        "redo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_viewport_at(&state, &viewer, &moved, "redo");

    call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": first }),
    );
    assert_viewport_at(&state, &viewer, &stored, "jump_to_version");

    // A refused move landed on nothing, so it moves the viewport no more than
    // the version: a free-look offset the human is holding is theirs to keep.
    viewer.camera.nodal_pan(40.0, 10.0);
    let held = crate::camera_lock::pending_pose(&viewer, &state.scene[0]);
    refused_call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_viewport_at(&state, &viewer, &held, "a refused undo");
}

/// A drawn action carries what it cost on the wire, in milliseconds.
///
/// The agent's half of the Action Log's duration column: an agent driving the
/// viewer can read back how long its own command took to reach the screen,
/// which is the measurement that `get_action_log` is otherwise silent about.
#[test]
fn get_action_log_reports_what_a_drawn_action_cost() {
    let (mut state, mut viewer) = quiet_scene();
    state
        .action_log
        .record(Kind::File, "Opened alpha from /runs/alpha.sfmr");
    state
        .action_log
        .settle(std::time::Instant::now(), Vec::new());

    let reply = ok(&mut state, &mut viewer, action_log_read(0, &Actor::ALL));

    let entries = reply["entries"].as_array().expect("an array");
    let row = entries.last().expect("at least the open");
    let took = row["took_ms"].as_f64().expect("a drawn row carries a cost");
    assert!(
        (0.0..60_000.0).contains(&took),
        "a plausible number of milliseconds, got {took}",
    );
}

// ── The 3D viewport's tool ──────────────────────────────────────────────

/// `set_viewer_3d_display` switches the tool, and the document reads it back
/// under `tool`, as the tool strip would show it.
#[test]
fn set_viewer_3d_display_switches_the_tool() {
    use crate::viewer_3d::brush::Tool;
    let (mut state, mut viewer) = editable();
    let read = ok(&mut state, &mut viewer, Command::GetViewer3dDisplay);
    assert_eq!(read["viewer_3d_display"]["tool"], "navigate");

    let reply = call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "tool": "eraser_brush" }),
    );
    assert_eq!(reply["viewer_3d_display"]["tool"], "eraser_brush");
    assert_eq!(viewer.tool, Tool::EraserBrush);
    assert!(
        state
            .action_log
            .entries()
            .any(|e| e.text == "Eraser brush tool"),
        "{:?}",
        rows(&state)
    );

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "tool": "lasso" }),
    );
    assert!(error.0.contains("navigate, eraser_brush"), "{error}");
    assert_eq!(viewer.tool, Tool::EraserBrush);
}

/// The eraser brush is refused while a camera is in hand, whole: the call's
/// other fields are not applied either, and the lock is left as it was.
#[test]
fn set_viewer_3d_display_refuses_the_brush_while_a_camera_is_in_hand() {
    use crate::viewer_3d::brush::Tool;
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);
    let grid = state.show_grid;
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "tool": "eraser_brush", "show_grid": !grid }),
    );
    assert!(error.0.contains("camera is being moved"), "{error}");
    assert_eq!(viewer.tool, Tool::Navigate);
    assert_eq!(state.show_grid, grid, "the other field was applied");
    assert!(viewer.camera_lock.is_some(), "the lock was ended");

    // Navigate is not refused: it is what the lock leaves the viewport in.
    let reply = call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "tool": "navigate" }),
    );
    assert_eq!(reply["viewer_3d_display"]["tool"], "navigate");
}

// ── The camera model proposal ───────────────────────────────────────────

/// Every point position `run_a` holds.
fn point_positions(state: &AppState) -> Vec<[f64; 3]> {
    state.scene[0]
        .recon()
        .point_set
        .points
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect()
}

/// `switch_camera_model { proposal: true }` parses to the proposal's Apply,
/// and refuses the fit's arguments; `propose_camera_model` parses to the
/// request the panel's controls make.
#[test]
fn the_proposal_tools_parse() {
    let parse = |name: &str, arguments: Value| {
        let map = arguments.as_object().cloned().expect("an object");
        tools::parse(name, Some(&map))
    };
    assert_eq!(
        parse(
            "switch_camera_model",
            json!({ "reconstruction_label": "a", "proposal": true })
        )
        .expect("a well-formed call"),
        Command::ApplyCameraModelProposal {
            reconstruction_label: "a".to_string(),
            camera: None,
        }
    );
    let error = parse(
        "switch_camera_model",
        json!({ "reconstruction_label": "a", "proposal": true, "coeff_count": 6 }),
    )
    .expect_err("a fit argument with the proposal form");
    assert!(error.0.contains("takes no coeff_count"), "{error}");
    assert_eq!(
        parse(
            "propose_camera_model",
            json!({
                "reconstruction_label": "a",
                "camera_intrinsics_index": 1,
                "camera_model": "SFMTOOL_FISHEYE",
                "theta_fit_deg": 80.0,
                "preview_retriangulation": true
            })
        )
        .expect("a well-formed call"),
        Command::ProposeCameraModel {
            reconstruction_label: "a".to_string(),
            request: crate::state::edits::SwitchCameraModelRequest {
                camera: 1,
                camera_model: Some("SFMTOOL_FISHEYE".to_string()),
                theta_fit_deg: Some(80.0),
                ..Default::default()
            },
            preview_retriangulation: Some(true),
            animate: None,
        }
    );
    assert_eq!(
        parse(
            "cancel_camera_model_proposal",
            json!({ "reconstruction_label": "a" })
        )
        .expect("a well-formed call"),
        Command::CancelCameraModelProposal {
            reconstruction_label: "a".to_string(),
        }
    );
}

/// `propose_camera_model` pushes nothing and answers with the `proposed`
/// block; the same call again keeps the generation, a changed one advances
/// it; `get_camera_intrinsics` reports the block; and
/// `cancel_camera_model_proposal` drops it, refusing when none is open.
#[test]
fn propose_camera_model_opens_a_proposal_and_pushes_nothing() {
    let (mut state, mut viewer) = editable();
    with_flat_spline_cameras(&mut state);
    let arguments = json!({ "reconstruction_label": "run_a", "camera_intrinsics_index": 0 });
    let reply = call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        arguments.clone(),
    );
    let proposed = &reply["proposed"];
    assert_eq!(proposed["camera_model"], "SFMTOOL_PINHOLE");
    assert_eq!(proposed["refit_spline"], true);
    assert_eq!(proposed["request"]["coeff_count"], 4);
    assert_eq!(proposed["request"]["spline_domain_deg"], Value::Null);
    assert_eq!(proposed["fit"]["theta_fit_source"], "spline_domain");
    assert!(proposed["fit_report"].as_str().unwrap().starts_with("rms "));
    assert_eq!(proposed["camera"]["camera_model"], "SFMTOOL_PINHOLE");
    let generation = proposed["generation"].as_u64().expect("a generation");
    assert_eq!(version_count(&state), 1);

    let again = call(&mut state, &mut viewer, "propose_camera_model", arguments);
    assert_eq!(again["proposed"]["generation"].as_u64(), Some(generation));
    let changed = call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "coeff_count": 6
        }),
    );
    assert!(changed["proposed"]["generation"].as_u64() > Some(generation));

    let read = call(
        &mut state,
        &mut viewer,
        "get_camera_intrinsics",
        json!({ "reconstruction_label": "run_a", "camera_intrinsics_index": 0 }),
    );
    assert_eq!(read["proposed"]["request"]["coeff_count"], 6);
    assert_eq!(read["camera_model"], "SFMTOOL_PINHOLE");

    call(
        &mut state,
        &mut viewer,
        "cancel_camera_model_proposal",
        json!({ "reconstruction_label": "run_a" }),
    );
    let read = call(
        &mut state,
        &mut viewer,
        "get_camera_intrinsics",
        json!({ "reconstruction_label": "run_a", "camera_intrinsics_index": 0 }),
    );
    assert!(read.get("proposed").is_none(), "{read}");
    let error = refused_call(
        &mut state,
        &mut viewer,
        "cancel_camera_model_proposal",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        error.0.contains("No camera model proposal is open"),
        "{error}"
    );
    assert_eq!(version_count(&state), 1);
}

/// Opening a proposal selects the Image Detail's Change to proposed field,
/// which `get_image_detail_display` reports, and cancelling it restores the
/// mode before it.
#[test]
fn a_proposal_selects_the_change_field_on_the_wire() {
    let (mut state, mut viewer) = editable();
    with_flat_spline_cameras(&mut state);
    let field = |state: &mut AppState, viewer: &mut Viewer3D| {
        call(state, viewer, "get_image_detail_display", json!({}))["image_detail_display"]
            ["intrinsics"]["field"]
            .clone()
    };
    assert_eq!(field(&mut state, &mut viewer), "distortion");
    call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({ "reconstruction_label": "run_a", "camera_intrinsics_index": 0 }),
    );
    assert_eq!(field(&mut state, &mut viewer), "change_to_proposed");
    call(
        &mut state,
        &mut viewer,
        "cancel_camera_model_proposal",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(field(&mut state, &mut viewer), "distortion");
}

/// An unknown model and a count on a model without a spline open nothing; a
/// fit the core refuses opens a proposal carrying the refusal.
#[test]
fn propose_camera_model_refuses_a_target_and_carries_a_refused_fit() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "NOT_A_MODEL"
        }),
    );
    assert!(error.0.contains("camera 0"), "{error}");
    assert!(state.camera_model_proposal.is_none());

    let reply = call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "SFMTOOL_PINHOLE",
            "spline_domain_deg": 95.0
        }),
    );
    assert_eq!(reply["proposed"]["camera"], Value::Null);
    assert!(
        reply["proposed"]["refusal"]
            .as_str()
            .is_some_and(|why| why.contains("camera 0")),
        "{reply}"
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({ "reconstruction_label": "run_a", "proposal": true }),
    );
    assert!(error.0.contains("refused"), "{error}");
}

/// The proposal form applies the open proposal as the `Switch camera model`
/// background task: one version that switches the camera and moves its
/// tracks, labelled with the retriangulation; an undo over the wire puts both
/// back.
#[test]
fn switch_camera_model_with_proposal_applies_it_and_moves_the_tracks() {
    let (mut state, mut viewer) = editable();
    for point in state.scene[0].recon_mut().point_set.points.iter_mut() {
        point.position.y += 0.01;
    }
    let before = point_positions(&state);
    let model_before = state.scene[0].recon().image_set.cameras[0]
        .model_name()
        .to_string();
    call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "SFMTOOL_PINHOLE"
        }),
    );
    let map = json!({ "reconstruction_label": "run_a", "proposal": true })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("switch_camera_model", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the proposal's Apply must defer"),
    };
    assert_eq!(pending.operation_name, "Switch camera model");
    assert!(state.camera_model_proposal.is_none());
    state.finish_background_tasks();
    let reply = match super::super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(
        reply["label"],
        format!(
            "Switched camera 0 of run_a from {model_before} to SFMTOOL_PINHOLE and \
             retriangulated its tracks"
        )
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("points moved"), "{report}");
    // The same `fit` object the direct form answers with.
    let fit = &reply["fit"];
    assert_eq!(fit["camera_intrinsics_index"], 0);
    assert_eq!(fit["camera_model_before"], model_before.as_str());
    assert_eq!(fit["camera_model_after"], "SFMTOOL_PINHOLE");
    assert!(fit["rms_px"].as_f64().is_some(), "{fit}");
    assert!(fit["observations"].as_u64().is_some_and(|n| n > 0), "{fit}");
    assert!(fit["monotone_constraint"]["active"].is_boolean(), "{fit}");
    assert_eq!(version_count(&state), 2);
    assert_ne!(point_positions(&state), before, "no track moved");

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(point_positions(&state), before);
    assert_eq!(
        state.scene[0].recon().image_set.cameras[0].model_name(),
        model_before
    );
}

/// The direct form is an edit tool naming the node, so it applies an open
/// proposal first; the Apply makes the node busy and the direct switch is
/// refused with the busy sentence. With no proposal open, it moves no point.
#[test]
fn a_direct_switch_applies_an_open_proposal_first() {
    let (mut state, mut viewer) = editable();
    call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "SFMTOOL_PINHOLE"
        }),
    );
    let direct = json!({
        "reconstruction_label": "run_a",
        "camera_intrinsics_index": 0,
        "camera_model": "SFMTOOL_FISHEYE"
    });
    let error = refused_call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        direct.clone(),
    );
    assert!(
        error.0.contains("Switch camera model is still running"),
        "{error}"
    );
    assert!(state.camera_model_proposal.is_none());
    state.finish_background_tasks();
    assert_eq!(version_count(&state), 2);

    let before = point_positions(&state);
    call(&mut state, &mut viewer, "switch_camera_model", direct);
    assert_eq!(version_count(&state), 3);
    assert_eq!(
        point_positions(&state),
        before,
        "the direct switch moved a point"
    );
}

/// An undo over the wire drops an open proposal.
#[test]
fn an_undo_over_the_wire_drops_the_proposal() {
    let (mut state, mut viewer) = editable();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "SFMTOOL_PINHOLE"
        }),
    );
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(state.camera_model_proposal.is_none());
    assert_eq!(state.scene[0].history.cursor(), 0);
}

/// On a node whose observations carry no pixel the proposal form is the
/// switch alone, answered at once with the version and the `fit` object.
#[test]
fn switch_camera_model_with_proposal_on_a_sift_node_answers_with_fit() {
    let workspace = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(workspace.path());
    state.select_recon(id);
    let mut viewer = Viewer3D::new();
    call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "SFMTOOL_PINHOLE"
        }),
    );
    let reply = call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({ "reconstruction_label": "run_a", "proposal": true }),
    );
    assert_eq!(reply["changed"], true);
    let report = reply["report"].as_str().expect("a report");
    assert!(
        report.contains("its tracks were not retriangulated"),
        "{report}"
    );
    let fit = &reply["fit"];
    assert_eq!(fit["camera_intrinsics_index"], 0);
    assert_eq!(fit["camera_model_after"], "SFMTOOL_PINHOLE");
    assert!(fit["rms_px"].as_f64().is_some(), "{fit}");
    assert!(fit.get("median_error_before_px").is_some(), "{fit}");
}

/// The preview on the wire: `propose_camera_model` with
/// `preview_retriangulation` answers with a `pending` triangulation block,
/// `get_camera_intrinsics` reports it `ready` once the job lands, with the
/// counts and the observation comparison; under `animate: false` the blend
/// stays at 0 until the answer lands and then takes 1 in one frame; a call
/// that only changes the preview keeps the generation; and the proposal form
/// of `switch_camera_model` with the answer in hand replies with the version
/// at once.
#[test]
fn the_preview_is_reported_and_its_answer_applied_at_once() {
    let (mut state, mut viewer) = editable();
    for point in state.scene[0].recon_mut().point_set.points.iter_mut() {
        point.position.y += 0.01;
    }
    let arguments = json!({
        "reconstruction_label": "run_a",
        "camera_intrinsics_index": 0,
        "camera_model": "SFMTOOL_PINHOLE"
    });
    let opened = call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        arguments.clone(),
    );
    let generation = opened["proposed"]["generation"]
        .as_u64()
        .expect("a generation");
    assert_eq!(opened["proposed"]["triangulation"]["state"], "off");
    assert_eq!(opened["proposed"]["observation_comparison"], Value::Null);

    let mut ticked = arguments.clone();
    ticked["preview_retriangulation"] = json!(true);
    ticked["animate"] = json!(false);
    let reply = call(&mut state, &mut viewer, "propose_camera_model", ticked);
    let proposed = &reply["proposed"];
    assert_eq!(proposed["generation"].as_u64(), Some(generation));
    assert_eq!(proposed["preview_retriangulation"], true);
    assert_eq!(proposed["animate"], false);
    assert_eq!(proposed["triangulation"]["state"], "pending");
    assert_eq!(proposed["s"], 0.0);

    // Before the answer there is nothing to move toward.
    state.step_camera_model_preview(1.0 / 60.0);
    state.settle_camera_model_proposal_job();
    let read = |state: &mut AppState, viewer: &mut Viewer3D| {
        call(
            state,
            viewer,
            "get_camera_intrinsics",
            json!({ "reconstruction_label": "run_a", "camera_intrinsics_index": 0 }),
        )["proposed"]
            .clone()
    };
    let proposed = read(&mut state, &mut viewer);
    let triangulation = &proposed["triangulation"];
    assert_eq!(triangulation["state"], "ready", "{triangulation}");
    assert_eq!(triangulation["generation"].as_u64(), Some(generation));
    assert!(triangulation["tracks"].as_u64().is_some_and(|n| n > 0));
    assert!(triangulation["move"].as_u64().is_some_and(|n| n > 0));
    assert_eq!(triangulation["lose_position"], 0);
    assert!(triangulation["lost_points"].as_array().is_some());
    assert!(triangulation["median_shift"].as_f64().is_some());
    let comparison = &proposed["observation_comparison"];
    assert!(comparison["observations"].as_u64().is_some_and(|n| n > 0));
    assert!(comparison["before"]["median_px"].as_f64().is_some());
    assert_eq!(proposed["s"], 0.0, "the blend moved before a frame");

    // The frame the answer lands in sets the blend to 1.
    state.step_camera_model_preview(1.0 / 60.0);
    assert_eq!(read(&mut state, &mut viewer)["s"], 1.0);

    // Under `animate: false` the reply already holds the blend at its new
    // goal: 0 when the preview is cleared, 1 when it is ticked again on the
    // answer it holds.
    let mut cleared = arguments.clone();
    cleared["preview_retriangulation"] = json!(false);
    let reply = call(&mut state, &mut viewer, "propose_camera_model", cleared);
    assert_eq!(reply["proposed"]["s"], 0.0, "{reply}");
    assert_eq!(read(&mut state, &mut viewer)["s"], 0.0);
    let mut again = arguments.clone();
    again["preview_retriangulation"] = json!(true);
    let reply = call(&mut state, &mut viewer, "propose_camera_model", again);
    assert_eq!(reply["proposed"]["s"], 1.0, "{reply}");
    assert_eq!(reply["proposed"]["triangulation"]["state"], "ready");

    let applied = call(
        &mut state,
        &mut viewer,
        "switch_camera_model",
        json!({ "reconstruction_label": "run_a", "proposal": true }),
    );
    assert!(applied.get("running").is_none(), "{applied}");
    assert!(state.running_tasks().next().is_none(), "it solved again");
    assert!(
        applied["label"]
            .as_str()
            .is_some_and(|label| label.ends_with("and retriangulated its tracks")),
        "{applied}"
    );
    assert_eq!(applied["fit"]["camera_model_after"], "SFMTOOL_PINHOLE");
    assert_eq!(version_count(&state), 2);
}

/// The proposal's job runs as a quiet task, which the wire does not report:
/// `get_scene`'s `background_tasks` stays empty while only it runs,
/// `get_background_tasks` shows the fast queue running nothing, and
/// `cancel_background_tasks` with no argument finds nothing to stop and leaves
/// it running.
#[test]
fn a_running_proposal_job_is_not_reported_or_cancelled_on_the_wire() {
    let (mut state, mut viewer) = editable();
    call(
        &mut state,
        &mut viewer,
        "propose_camera_model",
        json!({
            "reconstruction_label": "run_a",
            "camera_intrinsics_index": 0,
            "camera_model": "SFMTOOL_PINHOLE",
            "preview_retriangulation": true
        }),
    );
    let (release, hold) = std::sync::mpsc::channel();
    state.proposal_jobs.hold_next = Some(hold);
    assert!(state.drive_camera_model_proposal_job());
    let task = state.proposal_jobs.queued_task().expect("queued");
    assert!(state.quiet_task_running(task));

    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    assert_eq!(scene["background_tasks"], json!([]), "{scene}");
    let tasks = call(&mut state, &mut viewer, "get_background_tasks", json!({}));
    assert_eq!(tasks["fast"]["running"], json!(null), "{tasks}");
    assert_eq!(tasks["fast"]["waiting"], json!([]), "{tasks}");
    let error = refused_call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({}),
    );
    assert!(error.0.contains("Nothing is running"), "{error}");
    assert!(
        !state.quiet_task_cancelled(task),
        "a cancel of everything someone asked for stopped the proposal's job"
    );

    release.send(()).expect("the job is held");
    state.settle_camera_model_proposal_job();
    let proposed = call(
        &mut state,
        &mut viewer,
        "get_camera_intrinsics",
        json!({ "reconstruction_label": "run_a", "camera_intrinsics_index": 0 }),
    )["proposed"]
        .clone();
    assert_eq!(proposed["triangulation"]["state"], "ready", "{proposed}");
}
