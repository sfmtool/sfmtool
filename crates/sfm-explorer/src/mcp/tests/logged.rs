// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ── Status, and the Action Log ──────────────────────────────────────────

/// Every mutating command says so in the place the viewer already reports what
/// it did, prefixed so a human can tell it from something they did themselves.
#[test]
fn a_mutating_call_announces_itself_in_the_status_line() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_solo",
        json!({ "reconstruction_label": "beta" }),
    );
    let message = state.status_message().expect("a status message");
    assert!(message.starts_with("MCP: "), "{message}");
    assert!(message.contains("beta"), "{message}");
}

/// The scene the attribution tests start from: `alpha` selected with one of its
/// images picked, so that every command below is a *change* — only a change is
/// logged — and the log emptied of everything setting that up put in it.
pub(super) fn quiet_scene() -> (AppState, Viewer3D) {
    let (mut state, viewer) = two_reconstructions();
    let alpha = state.scene[0].id;
    state.select_image(Some(crate::scene::ImageRef::new(alpha, 1)));
    state.action_log.clear();
    (state, viewer)
}

/// One entry per mutating command, attributed to the agent — and the ambient
/// actor back where it was, so the user's next click is not filed as the
/// agent's.
#[test]
fn each_mutating_command_records_one_entry_as_the_agent() {
    let commands: Vec<Command> = vec![
        Command::SelectReconstruction {
            reconstruction_label: "beta".into(),
        },
        Command::SelectCameraImage {
            reconstruction_label: Some("beta".into()),
            camera_image: super::super::CameraImageSel::Index(2),
        },
        Command::SelectCameraIntrinsics {
            reconstruction_label: Some("beta".into()),
            camera_intrinsics_index: 0,
        },
        Command::SelectPoint {
            point: crate::goto_point::PointQuery::Index(3),
        },
        Command::ClearSelection {
            scope: super::super::SelectionScope::All,
        },
        Command::SetSolo {
            reconstruction_label: Some("beta".into()),
        },
        Command::SetReconstructionDisplay {
            reconstruction_label: "beta".into(),
            change: super::super::DisplayChange {
                visible: Some(false),
                ..Default::default()
            },
        },
        Command::SetImageDetailDisplay {
            change: super::super::ImageDetailDisplayChange {
                tracked_only: Some(false),
                ..Default::default()
            },
        },
        Command::SetTimingDetail { enabled: true },
        Command::SetView {
            view: super::super::ViewCommand::Fit {
                reconstruction_label: None,
            },
        },
        Command::CloseReconstruction {
            target: super::super::CloseTarget::One("beta".into()),
        },
    ];
    for command in commands {
        let (mut state, mut viewer) = quiet_scene();
        let name = command.tool_name();
        ok(&mut state, &mut viewer, command);
        let entries: Vec<_> = state.action_log.entries().collect();
        assert_eq!(entries.len(), 1, "{name} recorded {entries:?}");
        assert_eq!(entries[0].actor, Actor::Mcp, "{name}: {entries:?}");
        assert!(!entries[0].failed, "{name}: {entries:?}");
        assert!(
            !matches!(entries[0].kind, Kind::Query(_)),
            "{name} was filed as a query: {entries:?}"
        );
        assert_eq!(
            state.action_log.actor(),
            Actor::User,
            "{name} left the actor moved"
        );
    }
}

/// A read is logged too — from the command, since it changes no state and has
/// no state method to log through — but it never reaches the status line, so an
/// agent polling `get_scene` does not read its own polling back as the viewer's
/// status.
#[test]
fn each_read_only_command_records_a_query_the_status_line_ignores() {
    let commands: Vec<Command> = vec![
        Command::GetScene,
        Command::ListCameraImages {
            reconstruction_label: None,
            offset: 0,
            limit: 5,
        },
        Command::GetCameraImage {
            reconstruction_label: None,
            camera_image: super::super::CameraImageSel::Index(0),
        },
        Command::GetCameraIntrinsics {
            reconstruction_label: None,
            camera_intrinsics_index: 0,
        },
        Command::GetPoint {
            point: crate::goto_point::PointQuery::Index(1),
        },
        Command::GetImageDetailDisplay,
        Command::GetTimingDetail,
    ];
    for command in commands {
        let (mut state, mut viewer) = quiet_scene();
        let name = command.tool_name();
        ok(&mut state, &mut viewer, command);
        let entries: Vec<_> = state.action_log.entries().collect();
        assert_eq!(entries.len(), 1, "{name} recorded {entries:?}");
        assert_eq!(entries[0].kind, Kind::Query(name), "{name}: {entries:?}");
        assert_eq!(entries[0].actor, Actor::Mcp, "{name}: {entries:?}");
        assert_eq!(
            state.status_message(),
            None,
            "{name} reached the status line"
        );
    }
}

/// A `screenshot` is logged in the frame it was *applied*, not when the pixels
/// come back, so its line sits in order with the commands around it.
#[test]
fn a_deferred_screenshot_is_logged_when_it_is_drained() {
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    let outcome = agent(&mut state, &mut viewer, screenshot(None, true, None));
    assert!(matches!(outcome, Outcome::Deferred(_)), "not deferred");
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    // The window, at the size the window snapshot reports.
    assert_eq!(
        entries[0].text, "screenshot window 1920×1080",
        "{entries:?}"
    );
    // …and `max_dimension` is reported at the size the picture comes back at.
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    agent(&mut state, &mut viewer, screenshot(None, true, Some(640)));
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "screenshot window 640×360"
    );
}

/// A screenshot is not a value an agent is scrubbing through: it is a picture
/// it took and presumably looked at, so every one taken is its own row however
/// fast they arrive.
#[test]
fn every_screenshot_taken_is_its_own_row() {
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    for _ in 0..3 {
        agent(&mut state, &mut viewer, screenshot(None, true, None));
    }
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 3, "{entries:?}");
    assert!(
        entries.iter().all(|entry| entry.run.is_none()),
        "a screenshot carries a run: {entries:?}"
    );
    // …while the read beside it polls into one row, as every other read does.
    let (mut state, mut viewer) = quiet_scene();
    for _ in 0..3 {
        ok(&mut state, &mut viewer, Command::GetScene);
    }
    assert_eq!(state.action_log.entries().count(), 1);
}

/// A refusal is one failed entry, in the words the agent was given — not two,
/// one from the method and one from the drain.
#[test]
fn a_refused_command_records_one_failed_entry_carrying_the_refusal() {
    let (mut state, mut viewer) = quiet_scene();
    let error = refused(
        &mut state,
        &mut viewer,
        Command::SelectReconstruction {
            reconstruction_label: "globl".into(),
        },
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].failed, "{entries:?}");
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(
        entries[0].text,
        format!("select_reconstruction failed: {error}")
    );
    // The status line shows it too, where before only a success reached it.
    assert_eq!(
        state.status_message(),
        Some(format!("MCP: select_reconstruction failed: {error}"))
    );
}

/// `load_file` returns its failure rather than writing it anywhere, so an
/// unreadable path is one entry and not two.
#[test]
fn open_reconstruction_on_an_unreadable_path_records_one_failed_entry() {
    let (mut state, mut viewer) = quiet_scene();
    refused(
        &mut state,
        &mut viewer,
        Command::OpenReconstruction {
            path: std::path::PathBuf::from("/runs/there-is-no-such-file.sfmr"),
        },
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].failed, "{entries:?}");
    assert!(
        entries[0]
            .text
            .starts_with("open_reconstruction failed: Failed to load "),
        "{entries:?}"
    );
}

/// An open goes to a worker like every other background operation, and the
/// answer it comes back with is the reconstruction entry it always gave.
#[test]
fn open_reconstruction_defers_and_answers_with_the_reconstruction() {
    let dir = temp_dir("open_defers");
    std::fs::write(dir.join(".sfm-workspace.json"), "{}").unwrap();
    let path = dir.join("scene.sfmr");
    SfmrReconstruction::demo(16).save(&path).unwrap();
    let (mut state, mut viewer) = quiet_scene();

    let open = |state: &mut AppState, viewer: &mut Viewer3D| {
        let map = json!({ "path": path.display().to_string() })
            .as_object()
            .cloned()
            .expect("an object");
        let command = tools::parse("open_reconstruction", Some(&map)).expect("a well-formed call");
        let pending = match agent(state, viewer, command) {
            Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
            Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
            _ => panic!("the open must defer"),
        };
        assert_eq!(pending.operation_name, "Open");
        state.finish_background_task();
        match super::super::edit::background_reply(state, &pending).expect("it finished") {
            Ok(ToolOutput::Json(value)) => value,
            Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
            Err(e) => panic!("expected success, got refusal: {e}"),
        }
    };
    let reply = open(&mut state, &mut viewer);
    assert_eq!(reply["label"], "scene", "{reply}");
    assert_eq!(reply["already_open"], false);
    // A second open of the same path is a second node, and says so.
    let reply = open(&mut state, &mut viewer);
    assert_eq!(reply["label"], "scene (2)", "{reply}");
    assert_eq!(reply["already_open"], true);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Reading the Action Log back ─────────────────────────────────────────

/// A `get_action_log` command with every field spelled out.
pub(super) fn action_log_read(since_revision: u64, actors: &[Actor]) -> Command {
    Command::GetActionLog {
        since_revision,
        limit: super::super::read::ACTION_LOG_DEFAULT_LIMIT,
        actors: actors.to_vec(),
        detail: false,
    }
}

/// Every entry text in a `get_action_log` reply, oldest first.
fn log_texts(reply: &Value) -> Vec<&str> {
    reply["entries"]
        .as_array()
        .expect("entries is an array")
        .iter()
        .map(|entry| entry["text"].as_str().expect("a text"))
        .collect()
}

/// The reply is a transcript: oldest first, each row saying when, who, what
/// kind and whether it failed, with the log's clock beside it.
#[test]
fn the_action_log_read_returns_a_transcript_with_the_clock_beside_it() {
    let (mut state, mut viewer) = quiet_scene();
    state
        .action_log
        .record(Kind::File, "Opened alpha from /runs/alpha.sfmr");
    let expected_revision = state.action_log.revision();
    let reply = ok(&mut state, &mut viewer, action_log_read(0, &Actor::ALL));

    assert_eq!(reply["revision"], json!(expected_revision));
    assert_eq!(reply["oldest_revision"], json!(expected_revision));
    assert_eq!(reply["truncated"], json!(false));
    let entries = reply["entries"].as_array().expect("an array");
    assert_eq!(entries.len(), 1, "{reply}");
    let row = &entries[0];
    assert_eq!(row["revision"], json!(expected_revision));
    assert_eq!(row["actor"], "user");
    assert_eq!(row["kind"], "file");
    assert_eq!(row["failed"], json!(false));
    assert_eq!(row["text"], "Opened alpha from /runs/alpha.sfmr");
    assert!(row["tool"].is_null(), "only a query row carries a tool");
    assert!(
        row["took_ms"].is_null(),
        "a row the viewer has not drawn a frame for does not claim a cost yet",
    );
    // RFC 3339 in the panel's zone, so the agent's time and the human's row
    // are the same time.
    let at = row["at"].as_str().expect("a timestamp");
    assert!(at.len() >= 24 && at.contains('T'), "{at}");
}

/// The read an agent makes most: what the human did, with none of the agent's
/// own rows in it.
#[test]
fn the_actors_filter_separates_the_human_from_the_agent() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record(Kind::Selection, "Selected image a");
    ok(
        &mut state,
        &mut viewer,
        Command::SelectReconstruction {
            reconstruction_label: "beta".into(),
        },
    );
    ok(&mut state, &mut viewer, Command::GetScene);

    let human = ok(&mut state, &mut viewer, action_log_read(0, &[Actor::User]));
    assert_eq!(log_texts(&human), ["Selected image a"]);

    let agent_rows = ok(&mut state, &mut viewer, action_log_read(0, &[Actor::Mcp]));
    let entries = agent_rows["entries"].as_array().expect("an array");
    assert_eq!(
        log_texts(&agent_rows),
        [
            "Selected reconstruction beta",
            "get_scene",
            // The human's read above was the agent's own call, and a read of
            // the log that the log did not record would be the one action the
            // human could not see.
            "get_action_log since 0",
        ],
        "the agent audits itself, queries included"
    );
    // A query row carries its tool beside the kind, which stays the one word.
    assert_eq!(entries[1]["kind"], "query");
    assert_eq!(entries[1]["tool"], "get_scene");
    assert!(entries[0]["tool"].is_null());

    // Omitted, the filter is every actor, which is the whole log.
    let all = call(&mut state, &mut viewer, "get_action_log", json!({}));
    assert_eq!(all["entries"].as_array().expect("an array").len(), 4);
}

/// A call that can return nothing by construction has asked no question, and a
/// misspelled actor is a typo rather than a filter.
#[test]
fn an_empty_or_unknown_actors_list_is_refused_at_the_parse() {
    let (mut state, mut viewer) = quiet_scene();
    let empty = refused_call(
        &mut state,
        &mut viewer,
        "get_action_log",
        json!({ "actors": [] }),
    );
    assert!(empty.0.contains("empty actors"), "{empty}");
    assert!(
        empty.0.contains("user") && empty.0.contains("mcp"),
        "{empty}"
    );

    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "get_action_log",
        json!({ "actors": ["human"] }),
    );
    assert!(unknown.0.contains("\"human\""), "{unknown}");
    assert!(unknown.0.contains("viewer"), "{unknown}");
}

/// Past `limit` the reply says so, and the last entry's revision is where the
/// next call picks up.
#[test]
fn the_read_truncates_at_its_limit_and_continues_from_the_last_revision() {
    let (mut state, mut viewer) = quiet_scene();
    for i in 0..5 {
        state.action_log.record(Kind::File, format!("Opened {i}"));
    }
    let first = ok(
        &mut state,
        &mut viewer,
        Command::GetActionLog {
            since_revision: 0,
            limit: 2,
            actors: Actor::ALL.to_vec(),
            detail: false,
        },
    );
    assert_eq!(log_texts(&first), ["Opened 0", "Opened 1"]);
    assert_eq!(first["truncated"], json!(true));

    let from = first["entries"][1]["revision"]
        .as_u64()
        .expect("a revision");
    let rest = ok(
        &mut state,
        &mut viewer,
        action_log_read(from, &[Actor::User]),
    );
    assert_eq!(log_texts(&rest), ["Opened 2", "Opened 3", "Opened 4"]);
    assert_eq!(rest["truncated"], json!(false));

    // A reader that is up to date is told nothing, which is the poll that
    // costs nothing. Under `["user"]`, which is what keeps the agent's own
    // polling rows out of the answer.
    let caught_up = ok(
        &mut state,
        &mut viewer,
        action_log_read(
            rest["revision"].as_u64().expect("a revision"),
            &[Actor::User],
        ),
    );
    assert_eq!(log_texts(&caught_up), Vec::<&str>::new());
}

/// A limit above the cap is capped rather than honoured: the surface is not a
/// data channel.
#[test]
fn a_limit_above_the_cap_is_capped() {
    let (mut state, mut viewer) = quiet_scene();
    for i in 0..super::super::read::ACTION_LOG_MAX_LIMIT + 5 {
        state.action_log.record(Kind::File, format!("Opened {i}"));
    }
    let reply = ok(
        &mut state,
        &mut viewer,
        Command::GetActionLog {
            since_revision: 0,
            limit: 100_000,
            actors: Actor::ALL.to_vec(),
            detail: false,
        },
    );
    assert_eq!(
        reply["entries"].as_array().expect("an array").len(),
        super::super::read::ACTION_LOG_MAX_LIMIT
    );
    assert_eq!(reply["truncated"], json!(true));
}

/// A read of the log that the log did not record would be the one action the
/// human could not see.
#[test]
fn the_action_log_read_records_itself_as_a_query() {
    let (mut state, mut viewer) = quiet_scene();
    ok(&mut state, &mut viewer, action_log_read(512, &Actor::ALL));
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Query("get_action_log"));
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(entries[0].text, "get_action_log since 512");
    // A poll is one row however often it asks, like every other read.
    ok(&mut state, &mut viewer, action_log_read(512, &Actor::ALL));
    assert_eq!(state.action_log.entries().count(), 1);
}

/// One field on `get_scene`, so an agent that already polls it knows whether
/// anything happened without a second call.
#[test]
fn get_scene_carries_the_action_log_revision() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record(Kind::File, "Opened alpha");
    let expected = state.action_log.revision();
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(scene["action_log_revision"], json!(expected));
    assert!(
        scene["status_message"].is_string(),
        "the status line is a different thing from the log and stays: {scene}"
    );
}

// ── The breakdown on the wire ───────────────────────────────────────────
//
// What these are really about is one property: the breakdown an agent reads is
// the breakdown the human is looking at. The Action Log panel is where a
// reader meets an expanded entry, and a wire that ordered its events
// differently, folded them differently or dropped what a row says would be
// telling the two of them different stories about the same operation. So the
// assertion most of these end on is [`assert_the_panel_agrees`], which rebuilds
// the panel's rows out of the JSON and compares them with the rows the panel
// would draw.

/// A `get_action_log` that asks for each row's breakdown as well.
pub(super) fn action_log_detail(since_revision: u64) -> Command {
    Command::GetActionLog {
        since_revision,
        limit: super::super::read::ACTION_LOG_DEFAULT_LIMIT,
        actors: Actor::ALL.to_vec(),
        detail: true,
    }
}

/// The reply's row whose text begins `prefix`: the operation a test drove,
/// picked out of the query rows the reads themselves leave.
#[track_caller]
pub(super) fn row_starting(reply: &Value, prefix: &str) -> Value {
    reply["entries"]
        .as_array()
        .expect("entries is an array")
        .iter()
        .find(|row| {
            row["text"]
                .as_str()
                .is_some_and(|text| text.starts_with(prefix))
        })
        .unwrap_or_else(|| panic!("no row starting {prefix:?} in {reply}"))
        .clone()
}

/// A phase row that ran once and said nothing, for the entries these build by
/// hand.
pub(super) fn phase(name: &'static str, depth: u8, ms: u64) -> Detail {
    folded(name, depth, ms, 1, None, None)
}

/// A phase row as the collector leaves one that ran more than once, with
/// whatever its runs said at either end.
fn folded(
    name: &'static str,
    depth: u8,
    ms: u64,
    runs: u32,
    note: Option<&str>,
    note_last: Option<&str>,
) -> Detail {
    Detail::Phase {
        name,
        depth,
        took: std::time::Duration::from_millis(ms),
        cpu: None,
        note: note.map(str::to_string),
        note_last: note_last.map(str::to_string),
        runs,
    }
}

/// A message row, for the entries these build by hand.
pub(super) fn message(level: Level, depth: u8, text: &str) -> Detail {
    Detail::Message {
        level,
        depth,
        text: text.to_string(),
    }
}

/// One wire row's breakdown, each event spelled the way the panel draws its
/// row: the indent, the marker, and the name with its count and its note.
///
/// Rebuilt from the JSON rather than read off the panel, which is the whole
/// point of it: everything the panel puts in a row has to be on the wire for
/// this to come out equal to what the panel drew.
pub(super) fn wire_detail_lines(row: &Value) -> Vec<String> {
    row["detail"]
        .as_array()
        .expect("the row carries a detail array")
        .iter()
        .map(|event| {
            let indent = " ".repeat(2 * event["depth"].as_u64().expect("a depth") as usize);
            match event["kind"].as_str().expect("a kind") {
                "phase" => {
                    let runs = match event["runs"].as_u64() {
                        Some(runs) => format!(" x{runs}"),
                        None => String::new(),
                    };
                    let note = match event["note"].as_str() {
                        Some(note) => format!("  {note}"),
                        None => String::new(),
                    };
                    format!(
                        "{indent}{}{runs}{note}",
                        event["name"].as_str().expect("a name")
                    )
                }
                "message" => format!(
                    "{indent}{} {}",
                    match event["level"].as_str().expect("a level") {
                        "info" => "\u{2022}",
                        "warn" => "!",
                        other => panic!("unknown message level {other:?}"),
                    },
                    event["text"].as_str().expect("a text")
                ),
                other => panic!("unknown detail kind {other:?}"),
            }
        })
        .collect()
}

/// The wire's breakdown is the panel's, row for row.
///
/// `elsewhere` is the one row of the panel's that is not an event: the wire
/// carries it as a field beside `took_ms`, so it is lifted out of the drawn
/// rows here and its presence checked on both sides.
#[track_caller]
pub(super) fn assert_the_panel_agrees(state: &AppState, row: &Value) {
    let revision = row["revision"].as_u64().expect("a revision");
    let entry = state
        .action_log
        .entries()
        .find(|entry| entry.revision == revision)
        .expect("the row is an entry the log still holds");
    let mut drawn = ActionLog::drawn_detail(entry);
    match (
        drawn.iter().position(|line| line == "elsewhere"),
        row.get("elsewhere_ms"),
    ) {
        (Some(at), Some(_)) => {
            drawn.remove(at);
        }
        (None, None) => {}
        (drawn_at, wire) => {
            panic!("the panel draws elsewhere at {drawn_at:?} and the wire carries {wire:?}: {row}")
        }
    }
    assert_eq!(wire_detail_lines(row), drawn, "{row}");
}

/// A frame's own events, as the viewer's frame collector leaves them.
fn frame_events() -> Vec<Detail> {
    let frame = Collector::new(false);
    {
        let uploads = frame.phase("uploads");
        drop(uploads.phase("points"));
    }
    drop(frame.phase("scene render"));
    frame.take()
}

/// Stamp what is waiting, so the entries have a cost and an account to close.
pub(super) fn settle(state: &mut AppState) {
    state
        .action_log
        .settle(std::time::Instant::now(), frame_events());
}

/// An `undo`, which is an operation that names its stages: the step, the
/// selection following it and the caches the version it left was holding.
fn undo_something(state: &mut AppState, viewer: &mut Viewer3D) {
    call(
        state,
        viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    call(
        state,
        viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
}

/// The breakdown is several times the size of the row it hangs off, so an
/// agent reading the log to find out what happened is not handed it.
#[test]
fn the_breakdown_is_omitted_until_it_is_asked_for() {
    let (mut state, mut viewer) = editable();
    undo_something(&mut state, &mut viewer);
    settle(&mut state);

    let plain = ok(&mut state, &mut viewer, action_log_read(0, &Actor::ALL));
    let row = row_starting(&plain, "Undo:");
    assert!(row["detail"].is_null(), "{row}");
    assert!(row["elsewhere_ms"].is_null(), "{row}");
    assert!(
        row["took_ms"].is_number(),
        "the cost is not the detail: {row}"
    );

    let asked = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&asked, "Undo:");
    assert!(row["detail"].is_array(), "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// The stages an operation named, at the depths it named them, in the order a
/// reader sees them.
#[test]
fn the_breakdown_carries_the_stages_at_their_depths() {
    let (mut state, mut viewer) = editable();
    undo_something(&mut state, &mut viewer);

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Undo:");
    let stages: Vec<(&str, u64)> = row["detail"]
        .as_array()
        .expect("a detail array")
        .iter()
        .map(|event| {
            assert_eq!(event["kind"], "phase", "{event}");
            (
                event["name"].as_str().expect("a name"),
                event["depth"].as_u64().expect("a depth"),
            )
        })
        .collect();
    assert_eq!(
        stages,
        [
            ("undo", 0),
            ("history step", 1),
            ("selection follow", 1),
            ("forget images", 1),
        ],
        "{row}",
    );
    // Every stage ran once, so none of them claims a count.
    assert!(
        row["detail"]
            .as_array()
            .expect("a detail array")
            .iter()
            .all(|event| event["runs"].is_null()),
        "{row}",
    );
    assert_the_panel_agrees(&state, &row);
}

/// `elsewhere` is what makes the breakdown add up, so it arrives with the cost
/// it closes the account of and not before.
#[test]
fn elsewhere_sits_beside_took_and_waits_for_the_frame() {
    let (mut state, mut viewer) = editable();
    undo_something(&mut state, &mut viewer);

    let before = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&before, "Undo:");
    assert!(row["took_ms"].is_null(), "{row}");
    assert!(
        row["elsewhere_ms"].is_null(),
        "an entry with no cost yet has no account to close: {row}",
    );
    assert_the_panel_agrees(&state, &row);

    settle(&mut state);
    let after = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&after, "Undo:");
    let took = row["took_ms"].as_f64().expect("a cost");
    let elsewhere = row["elsewhere_ms"].as_f64().expect("an elsewhere");
    let named: f64 = row["detail"]
        .as_array()
        .expect("a detail array")
        .iter()
        .filter(|event| event["depth"] == 0)
        .map(|event| event["ms"].as_f64().expect("a cost"))
        .sum();
    assert!(
        (took - named - elsewhere).abs() < 1e-6,
        "the breakdown does not reconcile with the headline: {row}",
    );
    assert_the_panel_agrees(&state, &row);
}

/// A row the collector folded says how many runs it is, and one that ran once
/// says nothing, and a note that has two ends reaches the agent with both,
/// because one end alone is a claim about the row only one run supports.
#[test]
fn a_folded_row_carries_its_runs_and_both_ends_of_its_note() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record_done(
        Kind::Edit,
        std::time::Instant::now(),
        "Bundle adjusted alpha",
        vec![
            phase("materialise", 0, 30),
            folded("round", 1, 50, 3, Some("trim 50 px"), Some("trim 4 px")),
            folded("linearise", 2, 12, 180, Some("reused"), None),
        ],
    );

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Bundle adjusted alpha");
    let events = row["detail"].as_array().expect("a detail array");
    assert!(events[0]["runs"].is_null(), "{row}");
    assert!(events[0]["note"].is_null(), "{row}");
    assert_eq!(events[1]["runs"], json!(3), "{row}");
    assert_eq!(events[1]["note"], "trim 50 px ... trim 4 px", "{row}");
    assert_eq!(events[2]["runs"], json!(180), "{row}");
    // Runs that came back to what the first one said leave one note, not a
    // span that is not there.
    assert_eq!(events[2]["note"], "reused", "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// What the operation said, marked the way the panel marks it and nested under
/// the stage it was said inside.
#[test]
fn a_message_carries_its_level_and_its_depth() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record_done(
        Kind::Edit,
        std::time::Instant::now(),
        "Bundle adjusted alpha",
        vec![
            phase("solve", 0, 30),
            message(Level::Info, 1, "3 rounds, trim 50/12/4 px"),
            message(Level::Warn, 1, "3 points left unsupported and were dropped"),
        ],
    );

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Bundle adjusted alpha");
    let events = row["detail"].as_array().expect("a detail array");
    assert_eq!(events[1]["kind"], "message", "{row}");
    assert_eq!(events[1]["level"], "info", "{row}");
    assert_eq!(events[1]["depth"], json!(1), "{row}");
    assert_eq!(events[1]["text"], "3 rounds, trim 50/12/4 px", "{row}");
    assert_eq!(events[2]["level"], "warn", "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// The order is the panel's, which is not the order the events were recorded
/// in: the operation's own stages, then the frame's overhead under the rule,
/// with `elsewhere` between them as a field rather than a row.
#[test]
fn the_overhead_comes_last_on_the_wire_as_it_does_in_the_panel() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record_done(
        Kind::Edit,
        std::time::Instant::now(),
        "Opened alpha",
        vec![phase("open", 0, 30), phase("read", 1, 20)],
    );
    settle(&mut state);

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Opened alpha");
    let names: Vec<&str> = row["detail"]
        .as_array()
        .expect("a detail array")
        .iter()
        .map(|event| event["name"].as_str().expect("a name"))
        .collect();
    assert_eq!(
        names,
        [
            "open",
            "read",
            ActionLog::OVERHEAD,
            "uploads",
            "points",
            "scene render",
        ],
        "{row}",
    );
    assert!(row["elsewhere_ms"].is_number(), "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// The level decides what is *recorded*, so it changes the next operation and
/// nothing already in the log.
#[test]
fn set_timing_detail_changes_what_the_next_operation_records() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a" }),
    );
    let overview = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&overview, "Bundle adjusted run_a");
    assert!(
        !wire_detail_lines(&row)
            .iter()
            .any(|line| line.trim_start().starts_with("linearise")),
        "the kernel's detailed stages were recorded with detail off: {row}",
    );
    assert_the_panel_agrees(&state, &row);

    let revision = state.action_log.revision();
    assert_eq!(
        call(
            &mut state,
            &mut viewer,
            "set_timing_detail",
            json!({ "enabled": true })
        )["timing_detail"]["enabled"],
        json!(true),
    );
    adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a" }),
    );
    settle(&mut state);

    let detailed = ok(&mut state, &mut viewer, action_log_detail(revision));
    let row = row_starting(&detailed, "Bundle adjusted run_a");
    assert!(
        wire_detail_lines(&row)
            .iter()
            .any(|line| line.trim_start().starts_with("linearise")),
        "detail was on and the kernel's stages did not reach the entry: {row}",
    );
    // The strongest thing here: whatever a real solve reported, the agent's
    // breakdown and the human's expanded row are the same rows in the same
    // order.
    assert_the_panel_agrees(&state, &row);

    // And the entry recorded before the change keeps the detail it was
    // recorded with: nothing is re-timed.
    let unchanged = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&unchanged, "Bundle adjusted run_a");
    assert!(
        !wire_detail_lines(&row)
            .iter()
            .any(|line| line.trim_start().starts_with("linearise")),
        "an entry was re-timed by the level changing under it: {row}",
    );
}

/// The pair reads back what it set, through the log the checkbox writes to, so
/// an agent and the window cannot hold two different levels.
#[test]
fn get_timing_detail_reads_back_what_set_timing_detail_wrote() {
    let (mut state, mut viewer) = quiet_scene();
    assert_eq!(
        call(&mut state, &mut viewer, "get_timing_detail", json!({}))["timing_detail"]["enabled"],
        json!(false),
        "detail is off until it is asked for",
    );

    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": true }),
    );
    assert!(state.action_log.detailed_timing(), "the level did not move");
    assert_eq!(
        call(&mut state, &mut viewer, "get_timing_detail", json!({}))["timing_detail"]["enabled"],
        json!(true),
    );

    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": false }),
    );
    assert!(!state.action_log.detailed_timing());
}

/// The same property the checkbox holds, because it is the same call: one
/// `Display` entry when the value changes, and none when it does not.
#[test]
fn set_timing_detail_records_the_change_and_nothing_else() {
    let (mut state, mut viewer) = quiet_scene();
    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": true }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Display);
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(entries[0].text, "Detailed timing on");

    // Handed the value it already has, it records nothing.
    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": true }),
    );
    assert_eq!(state.action_log.entries().count(), 1);

    // `enabled` is what the tool is for, so a call without it is a call that
    // asked for nothing.
    let error = refused_call(&mut state, &mut viewer, "set_timing_detail", json!({}));
    assert!(error.0.contains("enabled"), "{error}");
}
