// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `find_points`: the query over the track table cache, answered in the frame
//! from a warm cache and on landing from a cold one.

use super::*;
use sfmtool_core::analysis::track_table::TrackTableColumn;

/// Parse and apply one `find_points` call as the agent.
fn find(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Outcome {
    let map = arguments.as_object().cloned().expect("an object");
    let command = tools::parse("find_points", Some(&map)).expect("a well-formed call");
    agent(state, viewer, command)
}

/// The reply of a call, whichever way it answers: in the frame, or once the
/// *Find points* it queued has landed.
#[track_caller]
fn answered(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Value {
    match find(state, viewer, arguments) {
        Outcome::Done(Ok(ToolOutput::Json(value))) => value,
        Outcome::Done(Err(e)) => panic!("expected success, got refusal: {e}"),
        Outcome::Deferred(super::super::Deferred::Background(pending)) => {
            state.finish_background_tasks();
            match super::super::edit::background_reply(state, &pending).expect("it landed") {
                Ok(ToolOutput::Json(value)) => value,
                Ok(_) => panic!("expected JSON"),
                Err(e) => panic!("expected success, got refusal: {e}"),
            }
        }
        Outcome::Done(Ok(_)) | Outcome::Deferred(_) => panic!("expected JSON"),
    }
}

/// The point indexes of a reply's rows.
fn indexes(reply: &Value) -> Vec<u64> {
    reply["points"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|p| p["index"].as_u64().expect("an index"))
        .collect()
}

#[test]
fn a_cold_cache_queues_find_points_and_answers_when_it_lands() {
    let (mut state, mut viewer) = editable();
    let live = state.scene[0].edited().live_indexes().count();
    let pending = match find(&mut state, &mut viewer, json!({})) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("refused: {e}"),
        _ => panic!("a cold cache must defer"),
    };
    assert_eq!(pending.operation_name, "Find points");
    state.finish_background_tasks();
    let reply = match super::super::edit::background_reply(&state, &pending).expect("landed") {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("expected JSON"),
    };
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");
    assert_eq!(reply["live_points"], json!(live), "{reply}");
    assert_eq!(reply["matched"], json!(live), "{reply}");
    assert_eq!(reply["offset"], json!(0), "{reply}");
    let points = reply["points"].as_array().unwrap();
    assert_eq!(points.len(), live.min(50));
    // The default values with no bound and no sort.
    assert_eq!(
        keys(&points[0]),
        ["id", "index", "rms_error_px", "track_length"]
    );
    let id = points[0]["id"].as_str().unwrap();
    assert!(id.starts_with("pt3d_") && id.ends_with("_0"), "{id}");
    assert!(points[0]["track_length"].is_u64(), "{}", points[0]);

    // The call's own read row, and the task's row when it landed, both the
    // agent's.
    let rows = rows(&state);
    let found = format!("Found {live} of {live} points in run_a");
    assert!(
        rows.iter()
            .any(|(actor, failed, text)| *actor == Actor::Mcp && !failed && *text == found),
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .any(|(_, _, text)| text.starts_with("find_points run_a 0..50")),
        "{rows:?}"
    );
    let entry = state
        .action_log
        .entries()
        .find(|e| e.text == found)
        .unwrap();
    assert_eq!(entry.kind, Kind::Query("find_points"));
}

#[test]
fn a_warm_cache_answers_in_the_frame_with_bounds_a_sort_and_a_page() {
    let (mut state, mut viewer) = editable();
    // Warm the cache for both columns.
    answered(
        &mut state,
        &mut viewer,
        json!({ "columns": ["track_length", "max_ray_angle_deg", "rms_error_px"] }),
    );
    let query = json!({
        "bounds": [{ "column": "track_length", "min": 3 }],
        "order_by": { "column": "max_ray_angle_deg", "descending": true },
        "limit": 500,
    });
    let whole = match find(&mut state, &mut viewer, query.clone()) {
        Outcome::Done(Ok(ToolOutput::Json(value))) => value,
        Outcome::Done(Err(e)) => panic!("refused: {e}"),
        _ => panic!("a warm cache answers in the frame"),
    };
    let points = whole["points"].as_array().unwrap();
    assert!(!points.is_empty());
    assert_eq!(whole["matched"], json!(points.len()), "{whole}");
    // The bound and sorted columns, then the two defaults.
    assert_eq!(
        keys(&points[0]),
        [
            "id",
            "index",
            "max_ray_angle_deg",
            "rms_error_px",
            "track_length"
        ]
    );
    assert!(points
        .iter()
        .all(|p| p["track_length"].as_u64().unwrap() >= 3));
    let angles: Vec<f64> = points
        .iter()
        .map(|p| p["max_ray_angle_deg"].as_f64().unwrap())
        .collect();
    assert!(angles.windows(2).all(|w| w[0] >= w[1]), "{angles:?}");

    // A page is a slice of the same order.
    let mut paged = query;
    paged["offset"] = json!(2);
    paged["limit"] = json!(3);
    let page = answered(&mut state, &mut viewer, paged);
    assert_eq!(page["offset"], json!(2));
    assert_eq!(indexes(&page), indexes(&whole)[2..5]);
    // Past the end is an empty page with the count still there.
    let past = answered(
        &mut state,
        &mut viewer,
        json!({ "bounds": [{ "column": "track_length", "min": 3 }], "offset": 100000 }),
    );
    assert_eq!(past["points"], json!([]));
    assert_eq!(past["matched"], whole["matched"]);
}

#[test]
fn seen_in_camera_image_keeps_the_points_that_image_observes() {
    let (mut state, mut viewer) = editable();
    let by_index = answered(
        &mut state,
        &mut viewer,
        json!({ "seen_in_camera_image": 1, "limit": 500 }),
    );
    let name = state.scene[0].recon().image_set.images[1].name.clone();
    let by_name = answered(
        &mut state,
        &mut viewer,
        json!({ "seen_in_camera_image": name, "limit": 500 }),
    );
    assert_eq!(indexes(&by_index), indexes(&by_name));
    let edited = state.scene[0].edited();
    let expected: Vec<u64> = edited
        .live_indexes()
        .filter(|&i| {
            edited
                .point(i)
                .unwrap()
                .observations()
                .iter()
                .any(|o| o.image_index == 1)
        })
        .map(u64::from)
        .collect();
    assert!(!expected.is_empty());
    assert_eq!(indexes(&by_index), expected);
}

#[test]
fn a_point_at_infinity_has_no_position_on_the_wire() {
    let (mut state, mut viewer) = editable();
    {
        let recon = state.scene[0].recon_mut();
        let p = &mut recon.point_set.points[5];
        p.position = nalgebra::Point3::from(p.position.coords.normalize());
        p.w = 0.0;
        recon.rebuild_derived_fields();
    }
    let reply = answered(
        &mut state,
        &mut viewer,
        json!({
            "bounds": [{ "column": "at_infinity", "min": 1 }],
            "columns": ["x", "y", "z", "at_infinity", "max_ray_angle_deg", "condition_number"],
        }),
    );
    assert_eq!(reply["matched"], json!(1), "{reply}");
    let point = &reply["points"][0];
    assert_eq!(point["index"], json!(5));
    assert_eq!(point["x"], Value::Null);
    assert_eq!(point["y"], Value::Null);
    assert_eq!(point["z"], Value::Null);
    assert_eq!(point["at_infinity"], json!(1));
    assert_eq!(point["max_ray_angle_deg"], json!(0.0));
    assert_eq!(point["condition_number"], Value::Null);
    // A finite point has its position.
    let finite = answered(
        &mut state,
        &mut viewer,
        json!({ "bounds": [{ "column": "at_infinity", "max": 0 }], "columns": ["x"], "limit": 1 }),
    );
    assert!(finite["points"][0]["x"].is_f64(), "{finite}");
}

#[test]
fn the_refusals_name_the_columns() {
    let (mut state, mut viewer) = editable();
    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "find_points",
        json!({ "order_by": { "column": "error" } }),
    );
    assert!(
        unknown.0.starts_with(
            "find_points does not know the column \"error\" — the columns are track_length, "
        ),
        "{unknown}"
    );
    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "find_points",
        json!({ "columns": ["patch"] }),
    );
    assert!(unknown.0.contains("\"patch\""), "{unknown}");

    let backwards = refused_call(
        &mut state,
        &mut viewer,
        "find_points",
        json!({ "bounds": [{ "column": "track_length", "min": 5, "max": 3 }] }),
    );
    assert_eq!(
        backwards.0,
        "find_points wants each bound's min to be at most its max — got min 5 and max 3 for \
         track_length."
    );
    let no_column = refused_call(
        &mut state,
        &mut viewer,
        "find_points",
        json!({ "bounds": [{ "min": 5 }] }),
    );
    assert!(no_column.0.contains("needs column"), "{no_column}");

    let images = state.scene[0].image_count();
    let out_of_range = refused_call(
        &mut state,
        &mut viewer,
        "find_points",
        json!({ "seen_in_camera_image": images }),
    );
    assert_eq!(
        out_of_range.0,
        format!("run_a has {images} camera images — index {images} is out of range.")
    );

    // A column the fixture cannot supply is named, with the ones it can.
    let absent = TrackTableColumn::ALL
        .into_iter()
        .find(|c| !c.is_available(state.scene[0].edited()))
        .expect("the fixture lacks an optional column");
    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "find_points",
        json!({ "columns": [absent.wire_name()] }),
    );
    assert!(
        refusal.0.starts_with(&format!(
            "run_a cannot supply the column {}: the columns it can supply are track_length, ",
            absent.wire_name()
        )),
        "{refusal}"
    );
    assert!(
        !refusal
            .0
            .split("can supply are")
            .nth(1)
            .unwrap()
            .contains(absent.wire_name()),
        "{refusal}"
    );
}

#[test]
fn a_waiting_find_points_is_listed_handed_back_and_cancelled() {
    let (mut state, mut viewer) = editable();
    // The fast queue held by a fake open, so *Find points* waits behind it.
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_task(
            crate::background::Operation::OPEN,
            None,
            "held".to_string(),
            Box::new(move |_| {
                let _ = held.recv();
                crate::background::Finished::Failed("held".to_string())
            }),
        )
        .expect("an idle queue");
    let pending = match find(&mut state, &mut viewer, json!({})) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        _ => panic!("a cold cache must defer"),
    };
    let id = pending.operation_id;

    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    let tasks = scene["background_tasks"].as_array().unwrap();
    assert!(
        tasks.iter().any(|t| t["operation"] == json!("Find points")
            && t["waiting"] == json!(true)
            && t["queue"] == json!("fast")
            && t["reconstruction_label"] == json!("run_a")),
        "{tasks:?}"
    );

    let past = super::super::BackgroundReply {
        queued: std::time::Instant::now() - super::super::REPLY_DIRECTLY_WITHIN,
        ..pending
    };
    let handle = match super::super::edit::background_reply(&state, &past).expect("past it") {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("a handle is JSON"),
    };
    assert_eq!(
        handle,
        json!({
            "running": true,
            "waiting": true,
            "operation": "Find points",
            "queue": "fast",
            "reconstruction_label": "run_a",
            "operation_id": id,
        })
    );

    let cancel = call(
        &mut state,
        &mut viewer,
        "cancel_background_tasks",
        json!({ "operation_ids": [id] }),
    );
    assert_eq!(cancel["removed"].as_array().unwrap().len(), 1, "{cancel}");
    let reply = super::super::edit::background_reply(&state, &past).expect("it is over");
    match reply {
        Err(error) => assert_eq!(error.0, "Find points of run_a cancelled before it started"),
        Ok(_) => panic!("a cancelled find answers with the cancellation"),
    }

    open.send(()).unwrap();
    state.finish_background_tasks();
    assert!(state.track_table(state.scene[0].id).is_none());
}

#[test]
fn a_read_locks_nothing_and_runs_beside_an_edit() {
    let (mut state, mut viewer) = editable();
    // A slow task holding the node does not hold up a read of it.
    let open = running_operation(&mut state, nothing);
    assert!(state.busy_refusal(state.scene[0].id).is_some());
    let pending = match find(&mut state, &mut viewer, json!({ "limit": 1 })) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("a busy node refused a read: {e}"),
        _ => panic!("a cold cache must defer"),
    };
    // The fast queue runs it beside the slow task, which still holds the node.
    state.finish_background_task(pending.operation_id);
    let reply = match super::super::edit::background_reply(&state, &pending).expect("landed") {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("expected JSON"),
    };
    assert_eq!(reply["points"].as_array().unwrap().len(), 1);
    assert!(state.busy_refusal(state.scene[0].id).is_some());
    open.send(()).unwrap();
    state.finish_background_tasks();
}

/// A job that reports nothing.
fn nothing(_: &sfmtool_core::progress::Progress<'_>) {}

/// Hold the fast queue with a fake open until the returned sender fires.
fn hold_fast_queue(state: &mut AppState) -> std::sync::mpsc::Sender<()> {
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_task(
            crate::background::Operation::OPEN,
            None,
            "held".to_string(),
            Box::new(move |_| {
                let _ = held.recv();
                crate::background::Finished::Failed("held".to_string())
            }),
        )
        .expect("the queue takes it");
    open
}

#[test]
fn a_run_whose_cached_columns_were_replaced_meanwhile_is_refused_not_a_panic() {
    let (mut state, mut viewer) = editable();
    let id = state.scene[0].id;
    // The cache holds track_length for the first version.
    answered(
        &mut state,
        &mut viewer,
        json!({ "columns": ["track_length"] }),
    );
    // A second version, and a run for it queued behind a held queue.
    state
        .delete_point(crate::scene::PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    let first = hold_fast_queue(&mut state);
    let other = state
        .start_find_points(id, vec![TrackTableColumn::RmsErrorPx], Default::default())
        .unwrap();
    let second = hold_fast_queue(&mut state);
    // Back on the first version, a call that needs track_length (cached) and
    // max_ray_angle_deg (not), so its run computes only the latter.
    state.undo(id).unwrap();
    let pending = match find(
        &mut state,
        &mut viewer,
        json!({ "columns": ["track_length", "max_ray_angle_deg"] }),
    ) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        _ => panic!("a partly cold cache must defer"),
    };
    // The other version's run lands while that version is at the cursor and
    // replaces the cached table.
    state.redo(id).unwrap();
    first.send(()).unwrap();
    state.finish_background_task(other);
    // Back again, and the call's run lands over a table that lacks
    // track_length.
    state.undo(id).unwrap();
    second.send(()).unwrap();
    state.finish_background_tasks();
    match super::super::edit::background_reply(&state, &pending).expect("landed") {
        Err(error) => assert!(error.0.contains("call find_points again"), "{}", error.0),
        Ok(_) => panic!("the table lacks track_length, so the call is refused"),
    }
}
