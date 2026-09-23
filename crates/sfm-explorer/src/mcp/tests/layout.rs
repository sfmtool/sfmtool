// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ── Screenshots of the window and of a panel ────────────────────────────

/// Give a panel's node the body rectangle a frame's egui pass would have laid
/// out, since a headless dock has never been drawn.
pub(super) fn lay_out(state: &mut AppState, panel: Tab, rect: egui::Rect) {
    let path = state.dock.find_tab(&panel).expect("the panel is docked");
    state
        .dock
        .leaf_mut(path.node_path())
        .expect("the path names a leaf")
        .viewport = rect;
}

/// Bring `panel` to the front of its node without a log entry, so a test whose
/// subject is the *first* entry still starts from an empty log.
fn raise(state: &mut AppState, panel: Tab) {
    state.action_log.mute();
    state.show_panel(panel);
    state.action_log.unmute();
}

/// A 320 × 180 point body at (100, 40), which at the fixture's scale factor of
/// 1.5 is 480 × 270 physical pixels at (150, 60).
fn body() -> egui::Rect {
    egui::Rect::from_min_size(egui::pos2(100.0, 40.0), egui::vec2(320.0, 180.0))
}

/// No panel is the whole window, at the size the window snapshot reports, with
/// the frame description kept because the 3D view is in the picture.
#[test]
fn a_screenshot_with_no_panel_photographs_the_window() {
    let (mut state, mut viewer) = two_reconstructions();
    let (source, caption) =
        deferred_screenshot(&mut state, &mut viewer, screenshot(None, true, None));
    assert_eq!(source, super::super::ScreenshotSource::Window);
    assert!(caption.starts_with("The window, 1920×1080."), "{caption}");
    assert!(caption.contains("alpha"), "{caption}");
}

/// A panel defers with its tab — the rectangle is resolved at readback, not
/// here — and its caption and log line name the panel and its last laid-out
/// size.
#[test]
fn a_screenshot_of_a_panel_defers_with_the_tab_and_names_it() {
    let (mut state, mut viewer) = quiet_scene();
    // Behind the viewport in the stock grid, and a picture wants the tab in
    // front.
    raise(&mut state, Tab::ImageDetail);
    lay_out(&mut state, Tab::ImageDetail, body());
    let (source, caption) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::ImageDetail), true, None),
    );
    assert_eq!(
        source,
        super::super::ScreenshotSource::Panel(Tab::ImageDetail)
    );
    assert_eq!(caption, "The Image Detail panel, 480×270.");
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "screenshot image_detail 480×270"
    );
}

/// The 3D Viewer's crop keeps the frame description, because that picture is
/// of the scene.
#[test]
fn a_screenshot_of_the_viewport_keeps_the_frame_description() {
    let (mut state, mut viewer) = two_reconstructions();
    lay_out(&mut state, Tab::Viewer3D, body());
    let (source, caption) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::Viewer3D), true, None),
    );
    assert_eq!(source, super::super::ScreenshotSource::Panel(Tab::Viewer3D));
    assert!(
        caption.starts_with("The 3D Viewer panel, 480×270."),
        "{caption}"
    );
    assert!(caption.contains("points"), "{caption}");
}

/// A panel that is not drawn cannot be photographed, and the two ways that
/// happens have two different fixes.
#[test]
fn a_panel_that_is_not_drawn_is_refused_naming_show_panel() {
    let (mut state, mut viewer) = quiet_scene();
    state.hide_panel(Tab::ActionLog);
    let closed = refused(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::ActionLog), true, None),
    );
    assert!(closed.0.contains("closed"), "{closed}");
    assert!(closed.0.contains("show_panel"), "{closed}");
    assert!(closed.0.contains("action_log"), "{closed}");

    // Behind a sibling: a picture of it would be a picture of the tab in
    // front, so the refusal names that tab too.
    let behind = refused(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::IntrinsicsDetail), true, None),
    );
    assert!(behind.0.contains("Track View"), "{behind}");
    assert!(behind.0.contains("show_panel"), "{behind}");

    // An unknown name is the panel vocabulary's own refusal, listing all seven.
    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "screenshot",
        json!({ "panel_name": "viewport" }),
    );
    assert!(unknown.0.contains("viewer_3d") && unknown.0.contains("action_log"));
}

/// Track View is a panel like any other on the wire, and the two panels whose
/// place it took are unknown names, refused with the list of the ones that
/// exist rather than read as an alias.
#[test]
fn track_view_is_the_panel_name_and_the_retired_ones_are_refused() {
    let (mut state, mut viewer) = quiet_scene();
    let shown = call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "track_view" }),
    );
    assert_eq!(
        panel(&shown, "track_view")["active"],
        json!(true),
        "{shown}"
    );
    let arguments = json!({ "panel_name": "track_view" });
    let parsed = tools::parse("screenshot", arguments.as_object()).expect("a panel name");
    assert!(
        matches!(
            parsed,
            Command::Screenshot {
                panel: Some(Tab::TrackView),
                ..
            }
        ),
        "{parsed:?}"
    );
    for retired in ["point_track", "track_edit"] {
        for tool in ["show_panel", "hide_panel", "screenshot"] {
            let why = refused_call(
                &mut state,
                &mut viewer,
                tool,
                json!({ "panel_name": retired }),
            );
            assert!(why.0.contains(&format!("\"{retired}\"")), "{tool}: {why}");
            assert!(why.0.contains("track_view"), "{tool}: {why}");
        }
    }
}

/// Both checks are against the dock at *apply* time, so a `show_panel` earlier
/// in the same batch satisfies them.
#[test]
fn show_panel_then_a_screenshot_of_it_is_accepted_in_one_batch() {
    let (mut state, mut viewer) = quiet_scene();
    let outcomes = apply_as_agent(
        &mut state,
        &mut viewer,
        &mut NoWindow,
        vec![
            Command::ShowPanel {
                panel: Tab::IntrinsicsDetail,
            },
            screenshot(Some(Tab::IntrinsicsDetail), true, None),
        ],
    )
    .outcomes;
    assert!(
        matches!(outcomes[1], Outcome::Deferred(_)),
        "the raised panel was still refused"
    );
}

/// `hud: false` asks for the picture underneath what egui painted, and only the
/// 3D Viewer has one.
#[test]
fn hud_false_reads_the_render_target_and_is_refused_elsewhere() {
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    let (source, caption) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::Viewer3D), false, None),
    );
    assert_eq!(source, super::super::ScreenshotSource::ViewportRender);
    assert!(
        caption.starts_with("The 3D Viewer panel without its HUD, 1280×720."),
        "{caption}"
    );
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "screenshot viewer_3d 1280×720 without HUD"
    );

    for arguments in [
        json!({ "panel_name": "image_detail", "hud": false }),
        json!({ "hud": false }),
    ] {
        let error = refused_call(&mut state, &mut viewer, "screenshot", arguments.clone());
        assert!(
            error.0.contains("hud applies to the 3D Viewer only"),
            "{arguments}: {error}"
        );
    }

    // `hud: true` is accepted anywhere and changes nothing.
    raise(&mut state, Tab::ImageDetail);
    let (with_hud, _) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::ImageDetail), true, None),
    );
    assert_eq!(
        with_hud,
        super::super::ScreenshotSource::Panel(Tab::ImageDetail)
    );
}

/// The crop rectangle is the dock's own points scaled to pixels, clipped to the
/// frame — pure arithmetic over the dock, which is what puts it under headless
/// test.
#[test]
fn the_crop_rectangle_scales_the_docks_points_and_clips_to_the_frame() {
    let (mut state, _) = two_reconstructions();
    lay_out(&mut state, Tab::ImageDetail, body());
    assert_eq!(
        super::super::panel_crop(&state.dock, Tab::ImageDetail, 1.5, [1920, 1080]),
        Some([150, 60, 480, 270])
    );
    // A frame smaller than the layout clips rather than reading past the end.
    assert_eq!(
        super::super::panel_crop(&state.dock, Tab::ImageDetail, 1.5, [400, 200]),
        Some([150, 60, 250, 140])
    );
    // A panel the dock has never laid out has no rectangle to crop to, and one
    // that lies wholly outside the frame has none either.
    assert_eq!(
        super::super::panel_crop(&state.dock, Tab::SceneGraph, 1.5, [1920, 1080]),
        None
    );
    assert_eq!(
        super::super::panel_crop(&state.dock, Tab::ImageDetail, 1.5, [100, 100]),
        None
    );
}

// ── The layout tools ────────────────────────────────────────────────────

/// A panel by its wire name, for a reply's `panels` map.
#[track_caller]
fn panel(reply: &Value, name: &str) -> Value {
    reply["panels"][name].clone()
}

/// A fake in one of the four states, with the snapshot to match.
fn windowed(state: WindowState) -> (AppState, Viewer3D, FakeWindow) {
    let host = FakeWindow::in_state(state);
    let (mut app_state, viewer) = two_reconstructions();
    app_state.observe_window(&host);
    (app_state, viewer, host)
}

/// The reply's three views of one arrangement: the document the file holds,
/// the live window, and the panels.
#[test]
fn get_window_layout_returns_the_file_the_window_and_the_panels() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let reply = ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout);

    // `window_layout` is the file itself, not a rendering of it: read back
    // through the file's own parser it is what Save Layout… would write.
    let text = serde_json::to_string(&reply["window_layout"]).expect("a document");
    assert_eq!(
        WindowLayout::from_json(&text).expect("the reply parses as a layout file"),
        state.window_layout()
    );

    // The block beside it is the observation, with every monitor.
    assert_eq!(reply["window"]["state"], "normal");
    let names: Vec<&str> = reply["window"]["monitors"]
        .as_array()
        .expect("a monitor list")
        .iter()
        .map(|monitor| monitor["name"].as_str().expect("a name"))
        .collect();
    assert_eq!(names, ["DISPLAY1", "DISPLAY2"], "the current monitor first");

    let panels = reply["panels"].as_object().expect("a panels map");
    assert_eq!(panels.len(), Tab::ALL.len(), "all of them, always");
    for tab in Tab::ALL {
        assert_eq!(
            panel(&reply, tab.wire_name())["open"],
            json!(true),
            "{} is open in the default layout",
            tab.wire_name()
        );
    }
    // The default layout has three multi-tab nodes, so four of the nine sit
    // behind a sibling rather than in front of it.
    let active: Vec<&str> = Tab::ALL
        .iter()
        .filter(|tab| panel(&reply, tab.wire_name())["active"] == json!(true))
        .map(|tab| tab.wire_name())
        .collect();
    assert_eq!(
        active,
        [
            "scene",
            "background_task",
            "viewer_3d",
            "image_browser",
            "track_view"
        ]
    );
}

/// A maximized window's document says what it will restore to; the block says
/// what it is. The difference is the information.
#[test]
fn the_document_carries_the_normal_rectangle_and_the_block_the_current_one() {
    let (mut state, mut viewer) = two_reconstructions();
    let mut host = FakeWindow::default();
    state.observe_window(&host);
    host.set_state(WindowState::Maximized);
    host.inner_size = [3840, 2160];

    let reply = ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout);
    assert_eq!(reply["window"]["state"], "maximized");
    assert_eq!(reply["window"]["inner_size"], json!([3840, 2160]));
    assert_eq!(reply["window_layout"]["window"]["state"], "maximized");
    assert_eq!(
        reply["window_layout"]["window"]["inner_size"],
        json!([1920, 1080])
    );
}

/// The panels half of the answer is still an answer where there is no window.
#[test]
fn get_window_layout_answers_without_a_window() {
    let (mut state, mut viewer) = two_reconstructions();
    let reply = ok(&mut state, &mut viewer, Command::GetWindowLayout);
    assert_eq!(reply["window"], Value::Null);
    assert_eq!(reply["window_layout"]["window"], Value::Null);
    assert_eq!(panel(&reply, "scene")["open"], json!(true));
}

#[test]
fn hide_panel_closes_and_show_panel_takes_it_home() {
    let (mut state, mut viewer) = two_reconstructions();
    let hidden = call(
        &mut state,
        &mut viewer,
        "hide_panel",
        json!({ "panel_name": "action_log" }),
    );
    assert_eq!(
        panel(&hidden, "action_log"),
        json!({ "open": false, "active": false })
    );
    assert!(!state.is_panel_open(Tab::ActionLog));

    // Idempotent, as the method is: hiding a closed panel succeeds and
    // changes nothing. Both tools *set* rather than toggle, for the reason
    // `set_solo` does.
    let again = call(
        &mut state,
        &mut viewer,
        "hide_panel",
        json!({ "panel_name": "action_log" }),
    );
    assert_eq!(again, hidden);

    // Shown again it goes home to its default group-mate's node — behind the
    // Image Browser — and comes to the front of it.
    let shown = call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "action_log" }),
    );
    assert_eq!(
        panel(&shown, "action_log"),
        json!({ "open": true, "active": true })
    );
    assert_eq!(
        panel(&shown, "image_browser"),
        json!({ "open": true, "active": false })
    );
}

#[test]
fn show_panel_on_an_open_panel_raises_it_and_moves_nothing_else() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = ok(&mut state, &mut viewer, Command::GetWindowLayout);
    let raised = call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "track_view" }),
    );
    assert_eq!(panel(&raised, "track_view")["active"], json!(true));
    assert_eq!(panel(&raised, "image_detail")["active"], json!(false));
    for tab in Tab::ALL {
        assert_eq!(
            panel(&raised, tab.wire_name())["open"],
            panel(&before, tab.wire_name())["open"],
            "{} moved",
            tab.wire_name()
        );
    }
}

#[test]
fn an_unknown_panel_name_lists_the_seven() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "viewer3d" }),
    );
    assert!(error.0.contains("viewer3d"), "{error}");
    assert!(
        error.0.contains("viewer_3d") && error.0.contains("action_log"),
        "the refusal lists the panels: {error}"
    );
}

/// The document `get_window_layout` hands out is a document
/// `set_window_layout` takes back, tag and all — one schema, one parser, and a
/// file an agent saved is a file the viewer reads at startup.
#[test]
fn a_whole_reply_can_be_sent_back() {
    let (mut state, mut viewer) = two_reconstructions();
    // A fake that clamps nothing, so the round trip is a round trip rather
    // than a demonstration of the clamp.
    let mut host = FakeWindow {
        minimum: [1, 1],
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "hide_panel",
        json!({ "panel_name": "camera_intrinsics" }),
    );
    let saved = ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout)
        ["window_layout"]
        .clone();
    assert_eq!(saved["sfm_explorer_layout"], json!(2));

    let reset = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "layout": "default" }),
    );
    assert!(state.is_panel_open(Tab::IntrinsicsDetail));
    assert_eq!(panel(&reset, "camera_intrinsics")["open"], json!(true));

    // The whole reply, unedited, including its version tag and window section.
    let restored = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        saved.clone(),
    );
    assert_eq!(restored["window_layout"], saved);
    assert!(!state.is_panel_open(Tab::IntrinsicsDetail));
}

/// And a file's text, parsed and sent as the argument, is the same document.
#[test]
fn a_file_can_be_sent_as_the_argument() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let file = state.window_layout().to_json();
    state.hide_panel(Tab::ActionLog);
    let document: Value = serde_json::from_str(&file).expect("the file parses");
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        document,
    );
    assert!(state.is_panel_open(Tab::ActionLog));
}

/// Every form in the spec's list, and what each leaves behind.
#[test]
fn each_form_of_set_window_layout_does_what_it_says() {
    // Maximize where it is.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "maximized" } }),
    );
    assert_eq!(reply["window"]["state"], "maximized");
    assert_eq!(host.applied, ["state maximized"]);

    // Restore, and resize.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Maximized);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "normal", "inner_size": [1700, 1300] } }),
    );
    assert_eq!(reply["window"]["state"], "normal");
    assert_eq!(reply["window"]["inner_size"], json!([1700, 1300]));

    // A size against a maximized window changes what it restores to and leaves
    // it maximized — the rule that replaced `set_window`'s refusal.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Maximized);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "inner_size": [1700, 1300] } }),
    );
    assert_eq!(reply["window"]["state"], "maximized");
    assert_eq!(
        reply["window_layout"]["window"]["inner_size"],
        json!([1700, 1300]),
        "the normal rectangle changed"
    );

    // Both portions, in that order.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    state.hide_panel(Tab::ActionLog);
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "maximized" }, "layout": "default" }),
    );
    assert!(host.maximized);
    assert!(state.is_panel_open(Tab::ActionLog));
}

/// The reply is a read-back rather than an echo, so a size the platform
/// clamped comes back clamped — while the Action Log says what was asked for.
#[test]
fn the_reply_is_a_read_back_and_the_log_is_what_was_asked() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "inner_size": [640, 480] } }),
    );
    assert_eq!(reply["window"]["inner_size"], json!([1600, 1200]));
    assert_eq!(
        state
            .action_log
            .entries()
            .next_back()
            .expect("an entry")
            .text,
        "Resized window to 640×480"
    );
}

#[test]
fn a_piece_a_window_section_does_not_carry_is_preserved() {
    let mut host = FakeWindow {
        focused: false,
        ..FakeWindow::default()
    };
    let (mut state, mut viewer) = two_reconstructions();
    state.observe_window(&host);
    let before =
        ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout)["window"].clone();
    let after = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "focus": true } }),
    )["window"]
        .clone();
    assert_eq!(after["state"], before["state"]);
    assert_eq!(after["inner_size"], before["inner_size"]);
    assert_eq!(after["outer_position"], before["outer_position"]);
    assert_eq!(after["focused"], json!(true));
    assert_eq!(host.applied, ["focus"]);
}

#[test]
fn the_window_moves_between_the_four_states() {
    for from in WindowState::ALL {
        for to in WindowState::ALL {
            let (mut state, mut viewer, mut host) = windowed(from);
            let reply = call_with(
                &mut state,
                &mut viewer,
                &mut host,
                "set_window_layout",
                json!({ "window": { "state": to.wire_name() } }),
            );
            assert_eq!(
                reply["window"]["state"],
                json!(to.wire_name()),
                "{} -> {}",
                from.wire_name(),
                to.wire_name()
            );
        }
    }
}

/// `normal` means all three flags off. Restoring a minimized window can bring
/// a maximized one back, and the caller asked for normal.
#[test]
fn normal_clears_a_minimized_and_maximized_window() {
    let mut host = FakeWindow {
        minimized: true,
        maximized: true,
        ..FakeWindow::default()
    };
    let (mut state, mut viewer) = two_reconstructions();
    state.observe_window(&host);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "normal" } }),
    );
    assert_eq!(reply["window"]["state"], "normal");
    assert!(!host.minimized && !host.maximized, "{host:?}");
}

/// The window portion is applied before the panels, so a platform refusal in
/// it stops the call with the dock untouched.
#[test]
fn a_position_refusal_leaves_the_panels_alone() {
    let mut host = FakeWindow {
        position: None,
        ..FakeWindow::default()
    };
    let (mut state, mut viewer) = two_reconstructions();
    state.observe_window(&host);
    state.hide_panel(Tab::ActionLog);
    let error = refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "outer_position": [10, 20] }, "layout": "default" }),
    );
    assert!(error.0.contains("position its own window"), "{error}");
    assert!(!state.is_panel_open(Tab::ActionLog), "the panels changed");
}

#[test]
fn a_document_that_does_not_validate_is_refused_whole() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let before = state.layout();
    let error = refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({
            "window": { "state": "maximized" },
            "layout": {
                "main": {
                    "split": "left_right",
                    "fracton": 0.5,
                    "first": { "tabs": ["scene"] },
                    "second": { "tabs": ["viewer_3d"] },
                },
                "windows": [],
            }
        }),
    );
    // The layout parser's own message, path and all.
    assert_eq!(error.0, "layout.main: unknown key \"fracton\"", "{error}");
    assert_eq!(
        state.layout(),
        before,
        "a refusal leaves the dock untouched"
    );
    assert!(host.applied.is_empty(), "{:?}", host.applied);

    // …and a window key's own rule reads the same way.
    let error = refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "big" } }),
    );
    assert!(error.0.starts_with("window.state: "), "{error}");
}

#[test]
fn the_only_named_layout_is_the_default() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_window_layout",
        json!({ "layout": "tidy" }),
    );
    assert!(
        error.0.contains("only named layout") && error.0.contains("default"),
        "{error}"
    );
}

/// A call is a request, and these ask for nothing.
#[test]
fn a_call_that_asks_for_nothing_is_refused() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    for arguments in [
        json!({}),
        json!({ "window": {} }),
        json!({ "sfm_explorer_layout": 2 }),
    ] {
        let error = refused_call_with(
            &mut state,
            &mut viewer,
            &mut host,
            "set_window_layout",
            arguments.clone(),
        );
        assert!(error.0.contains("nothing to do"), "{arguments}: {error}");
    }
    assert!(host.applied.is_empty(), "{:?}", host.applied);
}

/// Where there is no window, a window portion is refused — and a panel portion
/// on its own succeeds, because the panels do not need one.
#[test]
fn a_window_portion_needs_a_window() {
    let (mut state, mut viewer) = two_reconstructions();
    state.window = None;
    let error = refused(
        &mut state,
        &mut viewer,
        Command::SetWindowLayout {
            document: json!({ "window": { "state": "maximized" } }),
        },
    );
    assert!(error.0.contains("no window"), "{error}");

    state.hide_panel(Tab::ActionLog);
    ok(
        &mut state,
        &mut viewer,
        Command::SetWindowLayout {
            document: json!({ "layout": "default" }),
        },
    );
    assert!(state.is_panel_open(Tab::ActionLog));
}

/// The three panel writes go through the same `AppState` methods the Panels
/// menu does, so the rows are the menu's rows with the agent in the actor
/// column.
#[test]
fn the_layout_writes_record_the_menus_own_entries_as_the_agent() {
    let cases: Vec<(Command, &str)> = vec![
        (
            Command::HidePanel {
                panel: Tab::ActionLog,
            },
            "Closed Action Log panel",
        ),
        (
            Command::ShowPanel {
                panel: Tab::TrackView,
            },
            "Raised Track View panel",
        ),
        (
            Command::SetWindowLayout {
                document: json!({ "layout": "default" }),
            },
            "Reset layout",
        ),
    ];
    for (command, text) in cases {
        let (mut state, mut viewer) = quiet_scene();
        ok(&mut state, &mut viewer, command);
        let entries: Vec<_> = state.action_log.entries().collect();
        assert_eq!(entries.len(), 1, "{entries:?}");
        assert_eq!(entries[0].text, text, "{entries:?}");
        assert_eq!(entries[0].kind, Kind::Layout, "{entries:?}");
        assert_eq!(entries[0].actor, Actor::Mcp, "{entries:?}");
    }

    // A panel that was closed is *opened*, not raised…
    let (mut state, mut viewer) = quiet_scene();
    state.hide_panel(Tab::ActionLog);
    state.action_log.clear();
    ok(
        &mut state,
        &mut viewer,
        Command::ShowPanel {
            panel: Tab::ActionLog,
        },
    );
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "Opened Action Log panel"
    );

    // …and a document says which tool set it, since `apply_layout` records
    // nothing itself.
    let (mut state, mut viewer) = quiet_scene();
    let document = ok(&mut state, &mut viewer, Command::GetWindowLayout)["window_layout"].clone();
    state.action_log.clear();
    ok(
        &mut state,
        &mut viewer,
        Command::SetWindowLayout { document },
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].text, "Set layout");
    assert_eq!(entries[0].kind, Kind::Layout);
}

/// One row per portion, because the two portions are two kinds — and the
/// window row is composed in the order the pieces were applied.
#[test]
fn a_call_carrying_both_portions_records_both() {
    let (mut state, mut viewer) = quiet_scene();
    let mut host = FakeWindow {
        minimum: [1, 1],
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    state.action_log.clear();
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({
            "window": {
                "state": "maximized",
                "outer_position": [120, 64],
                "inner_size": [1280, 720],
            },
            "layout": "default",
        }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 2, "{entries:?}");
    assert_eq!(
        entries[0].text,
        "Moved window to (120, 64); resized window to 1280×720; maximized window"
    );
    assert_eq!(entries[0].kind, Kind::Window);
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(entries[1].text, "Reset layout");
    assert_eq!(entries[1].kind, Kind::Layout);
}

/// A refusal is one failed row, filed under the kind of the portion the call
/// carried.
#[test]
fn a_refusal_is_filed_under_the_portion_it_carried() {
    let (mut state, mut viewer) = quiet_scene();
    let mut host = FakeWindow {
        position: None,
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    state.action_log.clear();
    refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "outer_position": [10, 20] } }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].failed, "{entries:?}");
    assert!(
        entries[0].text.starts_with("set_window_layout failed: "),
        "{entries:?}"
    );
    assert_eq!(entries[0].kind, Kind::Window);

    // The same call carrying panels is a layout refusal.
    let (mut state, mut viewer) = quiet_scene();
    state.observe_window(&host);
    state.action_log.clear();
    refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "outer_position": [10, 20] }, "layout": "default" }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Layout);
}

#[test]
fn the_layout_read_is_a_query_entry() {
    let (mut state, mut viewer) = quiet_scene();
    ok(&mut state, &mut viewer, Command::GetWindowLayout);
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Query("get_window_layout"));
    assert_eq!(entries[0].text, "get_window_layout");
    // A query never reaches the status line: an agent polling must not read
    // its own polling back as the viewer's status.
    assert_eq!(state.status_message(), None);
}

// ── The window block ────────────────────────────────────────────────────

#[test]
fn get_scene_embeds_the_window_block() {
    let (mut state, mut viewer, _) = windowed(WindowState::Normal);
    let window = ok(&mut state, &mut viewer, Command::GetScene)["window"].clone();
    assert_eq!(window["state"], "normal");
    assert_eq!(window["focused"], json!(true));
    assert_eq!(window["scale_factor"], json!(1.5));
    assert_eq!(window["inner_size"], json!([1920, 1080]));
    assert_eq!(window["outer_size"], json!([1936, 1119]));
    assert_eq!(window["outer_position"], json!([120, 64]));
    // Physical pixels throughout, with the logical size under `derived` next
    // to the scale factor it comes from.
    assert_eq!(
        window["derived"]["inner_size_logical"],
        json!([1280.0, 720.0])
    );
    assert_eq!(window["monitor"]["name"], "DISPLAY1");
    assert!(
        window["monitors"].is_null(),
        "get_scene does not list every monitor: {window}"
    );
    let fraction = window["derived"]["monitor_fraction"][0]
        .as_f64()
        .expect("a fraction");
    assert!((fraction - 1936.0 / 3840.0).abs() < 1e-9, "{fraction}");
}

/// A picture of a window the human cannot see answers nothing an agent asked
/// of a shared viewer, and whether a minimized window's swapchain still
/// presents is platform-dependent.
#[test]
fn a_screenshot_of_a_minimized_window_is_refused() {
    let (mut state, mut viewer, _) = windowed(WindowState::Minimized);
    let error = refused(&mut state, &mut viewer, screenshot(None, true, None));
    assert!(
        error.0.contains("minimized") && error.0.contains("set_window_layout"),
        "{error}"
    );

    // Not minimized, it defers as it always did.
    state.window = Some(FakeWindow::default().info());
    assert!(
        matches!(
            agent(&mut state, &mut viewer, screenshot(None, true, None)),
            Outcome::Deferred(_)
        ),
        "a screenshot must defer once there is something to photograph"
    );
}
