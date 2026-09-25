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
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (width, height) = (camera.width, camera.height);
    for index in 0..2 {
        let data: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 251) as u8).collect();
        state.full_res_cache.insert(
            crate::scene::ImageRef::new(id, index),
            Some(std::sync::Arc::new(
                sfmtool_core::camera::remap::ImageU8Pyramid::from_image(
                    sfmtool_core::camera::remap::ImageU8::new(width, height, 3, data),
                    crate::state::PYRAMID_LEVELS,
                ),
            )),
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
    let image = &mut state.scene[0].recon_mut().image_table.images[1];
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
    state.finish_background_task();

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
    state.finish_background_task();

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

/// A bulk edit renumbers the image table, and the reply is read off the version
/// it pushed rather than off the request.
#[test]
fn delete_camera_image_renumbers_and_reports_the_version() {
    let (mut state, mut viewer) = editable();
    let before = state.scene[0].recon().image_table.images.len();
    let reply = call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );
    assert_eq!(
        state.scene[0].recon().image_table.images.len(),
        before - 1,
        "the image table did not shrink"
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
    state.finish_background_task();
    match super::super::edit::background_reply(state, &pending).expect("the operation finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
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
    let task = state.background_task().expect("running");
    let pending = super::super::BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        answer: super::super::Answer::Version(id),
        label: task.label.clone(),
        started: std::time::Instant::now() - super::super::REPLY_DIRECTLY_WITHIN,
    };

    let reply =
        match super::super::edit::background_reply(&state, &pending).expect("past the window") {
            Ok(ToolOutput::Json(value)) => value,
            _ => panic!("a handle is JSON"),
        };
    assert_eq!(reply["running"], json!(true), "{reply}");
    assert_eq!(reply["operation"], json!("Bundle adjust"), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");
    assert_eq!(
        reply["operation_id"],
        json!(pending.operation_id),
        "{reply}"
    );

    // Inside the window and still running, there is no answer yet: the frame
    // moves on and asks again.
    let fresh = super::super::BackgroundReply {
        started: std::time::Instant::now(),
        ..pending
    };
    assert!(
        super::super::edit::background_reply(&state, &fresh).is_none(),
        "an operation inside the window answered early",
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
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
    state.finish_background_task();

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
    let over = call(&mut state, &mut viewer, "get_background_task", json!({}));
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
    // This fixture has no photographs to fuse, so the narrower field beside
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
    assert!(state.background_task().is_none());
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
    state.finish_background_task();
    assert!(state.background_task().is_none());

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

/// Cancelling is refused when there is nothing to cancel, and stops the
/// operation when there is.
#[test]
fn cancel_background_stops_what_is_running_and_refuses_when_nothing_is() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(&mut state, &mut viewer, "cancel_background_task", json!({}));
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
    let reply = call(&mut state, &mut viewer, "cancel_background_task", json!({}));
    assert_eq!(reply["cancelling"], json!("Bundle adjust"), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");

    state.finish_background_task();
    // Whether the solve reached a poll before finishing is a race, so what is
    // asserted is the flag's effect on the state machine rather than which of
    // the two rows landed: either way the operation is over and the node free.
    assert!(state.background_task().is_none());
    assert_eq!(state.busy_refusal(state.scene[0].id), None);
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
fn running_operation(
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
    state.poll_background_task();
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
fn get_background_task_reads_one_shape_running_and_finished() {
    let (mut state, mut viewer) = editable();

    // Nothing has run: both discriminators are there and both are false, so a
    // reader never has to tell a missing key from a false one.
    let idle = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(keys(&idle), ["finished", "running"], "{idle}");
    assert_eq!(idle["running"], json!(false), "{idle}");
    assert_eq!(idle["finished"], json!(false), "{idle}");

    let open = running_operation(&mut state, reporting_job);
    let operation_id = state.background_task().expect("running").id;
    let live = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(live["running"], json!(true), "{live}");
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
    state.finish_background_task();

    let over = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(over["running"], json!(false), "{over}");
    assert_eq!(over["finished"], json!(true), "{over}");
    // The same operation, under the same names: an agent parses one shape.
    for field in ["operation", "reconstruction_label", "operation_id"] {
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
    open.send(()).expect("the worker is waiting");
    state.finish_background_task();

    let reply = call(&mut state, &mut viewer, "get_background_task", json!({}));
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

    let live = call(&mut state, &mut viewer, "get_background_task", json!({}));
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
    state.finish_background_task();
    let over = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(over["phases"], live["phases"], "{over}");
}

/// `get_scene` says the viewer is busy and stops there.
///
/// The most-polled tool on the surface carries a block whose size does not
/// depend on the operation: no phase table, no open phase, no status line, and
/// nothing at all once the operation is over.
#[test]
fn get_scene_says_the_viewer_is_busy_without_carrying_the_solve() {
    let (mut state, mut viewer) = editable();
    let idle = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(
        idle["background_task"],
        Value::Null,
        "{}",
        idle["background_task"]
    );

    let open = running_operation(&mut state, |progress| {
        reporting_job(progress);
        progress.set_status_message(format_args!("refining images/IMG_0007.jpg"));
    });
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    let block = &scene["background_task"];
    assert_eq!(
        keys(block),
        [
            "elapsed_s",
            "fraction",
            "operation",
            "operation_id",
            "reconstruction_label",
            "running",
        ],
        "the get_scene block has grown: {block}"
    );
    assert_eq!(block["running"], json!(true), "{block}");
    assert_eq!(block["operation"], json!("Bundle adjust"), "{block}");
    assert_eq!(block["reconstruction_label"], json!("run_a"), "{block}");
    // The one field a caller polls for movement, and the whole of what
    // `get_background_task` would add is absent here.
    assert!(block["fraction"].is_number(), "{block}");

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
    // Null again, rather than the operation that just ended: `background` is
    // read as "may I edit", and a block that outlived the operation would be
    // carried by every poll for the rest of the session.
    let after = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(after["background_task"], Value::Null, "{after}");
}

/// The apply timeout names what is running, and keeps its old guesses when
/// nothing is.
#[test]
fn the_timeout_message_names_an_operation_only_while_one_is_running() {
    let idle = super::super::server::timeout_message(None);
    assert!(idle.contains("did not answer within 10 seconds"), "{idle}");
    assert!(idle.contains("modal dialog"), "{idle}");
    assert!(!idle.contains("get_background_task"), "{idle}");

    let busy = super::super::server::timeout_message(Some(crate::background::Busy {
        operation: "Bundle adjust",
        label: "dino_dog_toy-embedded".to_string(),
    }));
    assert!(busy.contains("did not answer within 10 seconds"), "{busy}");
    assert!(
        busy.contains("Bundle adjust is running in the background on dino_dog_toy-embedded"),
        "{busy}"
    );
    assert!(busy.contains("get_background_task"), "{busy}");
    // The two guesses are gone: they name things that did not happen.
    assert!(!busy.contains("modal dialog"), "{busy}");
    assert!(!busy.contains("mid-drag"), "{busy}");
}

/// The notice the message reads is written where the task is written, so the
/// two cannot disagree about whether anything is running.
#[test]
fn the_busy_notice_tracks_the_operation() {
    let (mut state, _viewer) = editable();
    assert_eq!(crate::background::busy(&state.busy_notice), None);

    let open = running_operation(&mut state, |_| {});
    assert_eq!(
        crate::background::busy(&state.busy_notice),
        Some(crate::background::Busy {
            operation: "Bundle adjust",
            label: "run_a".to_string(),
        })
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
    assert_eq!(crate::background::busy(&state.busy_notice), None);
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
    assert!(error.0.contains("get_history"), "{error}");
}

/// `get_history` is the Edit History panel's reading of the same list: every
/// version in order, the cursor, the version on disk, and the released rows.
#[test]
fn get_history_lists_the_versions_with_the_cursor_and_the_released_rows() {
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
        "get_history",
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
        "get_history",
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
    assert!(written.metadata.lineage.is_empty());
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
        ("get_history", json!({})),
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
        "get_history",
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
    let centre = state.scene[0].recon().image_table.images[1].camera_center();
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
