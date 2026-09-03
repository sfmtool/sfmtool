// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the layout document: the default grid, the schema, the
//! file, the window's placement, and the home positions a panel is re-opened
//! at.
//!
//! None of it needs a window — a `DockState<Tab>` is plain data, the window
//! goes through `test_support::FakeWindow`, and the Panels menu is drawn
//! through the `test_support` frame harness.

use super::*;

use crate::action_log::Kind;
use crate::test_support::FakeWindow;
use crate::window::MonitorInfo;

/// The default document as the viewer writes it. Kept verbatim in
/// `specs/gui/panel-layout.md` § "The window layout file", so the spec and the
/// code cannot drift.
const DEFAULT_JSON: &str = r#"{
  "sfm_explorer_layout": 2,
  "layout": {
    "main": {
      "split": "left_right",
      "fraction": 0.18,
      "first": {
        "tabs": ["scene"],
        "active": "scene"
      },
      "second": {
        "split": "top_bottom",
        "fraction": 0.8,
        "first": {
          "split": "left_right",
          "fraction": 0.67,
          "first": {
            "tabs": ["viewer_3d"],
            "active": "viewer_3d"
          },
          "second": {
            "tabs": ["image_detail", "point_track", "camera_intrinsics"],
            "active": "image_detail"
          }
        },
        "second": {
          "tabs": ["image_browser", "action_log"],
          "active": "image_browser"
        }
      }
    },
    "windows": []
  }
}
"#;

/// A state whose dock is the stock grid and whose log is empty.
fn state() -> AppState {
    AppState::new()
}

/// The panel arrangement a document carries, or a panic if it carries none.
#[track_caller]
fn arrangement(document: &WindowLayout) -> &Layout {
    match &document.layout {
        Some(LayoutSection::Layout(layout)) => layout,
        other => panic!("no arrangement: {other:?}"),
    }
}

/// A document holding one arrangement and nothing about the window.
fn just(layout: Layout) -> WindowLayout {
    WindowLayout {
        window: None,
        layout: Some(LayoutSection::Layout(layout)),
    }
}

/// Parse a document, expecting it to be accepted.
#[track_caller]
fn parsed(text: &str) -> WindowLayout {
    WindowLayout::from_json(text).unwrap_or_else(|error| panic!("refused: {error}"))
}

/// The arrangement a document's text describes.
#[track_caller]
fn parsed_layout(text: &str) -> Layout {
    arrangement(&parsed(text)).clone()
}

/// The `Layout` entries recorded so far, oldest first.
fn layout_entries(state: &AppState) -> Vec<(bool, String)> {
    state
        .action_log
        .entries()
        .filter(|entry| entry.kind == Kind::Layout)
        .map(|entry| (entry.failed, entry.text.clone()))
        .collect()
}

/// Every leaf of the main surface, in heap order.
fn main_leaves(layout: &Layout) -> Vec<(Vec<Tab>, Tab)> {
    fn walk(node: &LayoutNode, out: &mut Vec<(Vec<Tab>, Tab)>) {
        match node {
            LayoutNode::Leaf { tabs, active } => out.push((tabs.clone(), *active)),
            LayoutNode::Split { first, second, .. } => {
                walk(first, out);
                walk(second, out);
            }
        }
    }
    let mut out = Vec::new();
    if let Some(main) = &layout.main {
        walk(main, &mut out);
    }
    out
}

// ── The default ──────────────────────────────────────────────────────────

/// The Action Log opens docked beside the image strip, and *behind* it: the
/// viewer should come up on the strip, with the record one click away rather
/// than in front of what the user came to look at.
#[test]
fn the_action_log_shares_the_bottom_node_with_the_image_browser() {
    let dock = Layout::default().to_dock();
    let leaf = dock
        .main_surface()
        .iter()
        .filter_map(|node| node.get_leaf())
        .find(|leaf| leaf.tabs.contains(&Tab::ActionLog))
        .expect("no leaf holds the Action Log");
    assert_eq!(leaf.tabs, vec![Tab::ImageBrowser, Tab::ActionLog]);
    assert_eq!(
        leaf.tabs[leaf.active.0],
        Tab::ImageBrowser,
        "the bottom node does not open on the image strip"
    );
}

#[test]
fn the_default_round_trips_through_a_dock() {
    let layout = Layout::default();
    assert_eq!(Layout::from_dock(&layout.to_dock()), layout);
}

#[test]
fn the_default_round_trips_through_json() {
    let document = WindowLayout::default();
    assert_eq!(document.to_json(), DEFAULT_JSON);
    assert_eq!(WindowLayout::from_json(DEFAULT_JSON), Ok(document));
}

/// A saved window comes back the way it went out — including the position a
/// platform would not report, which is written as absent rather than invented.
#[test]
fn a_window_section_round_trips_with_and_without_a_position() {
    for outer_position in [Some([120, 64]), None] {
        let document = WindowLayout {
            window: Some(WindowChange {
                state: Some(WindowState::Maximized),
                outer_position,
                inner_size: Some([1280, 720]),
                monitor: Some(MonitorRect {
                    position: [0, 0],
                    size: [3840, 2160],
                }),
                focus: false,
            }),
            layout: Some(LayoutSection::Layout(Layout::default())),
        };
        let text = document.to_json();
        assert_eq!(WindowLayout::from_json(&text), Ok(document));
        assert_eq!(
            text.contains("outer_position"),
            outer_position.is_some(),
            "{text}"
        );
        // The window section comes between the tag and the panels, and `focus`
        // is a request rather than a placement, so it is never written.
        let window_at = text.find("\"window\"").expect("a window section");
        let layout_at = text.find("\"layout\"").expect("a layout section");
        assert!(window_at < layout_at, "{text}");
        assert!(!text.contains("focus"), "{text}");
    }
}

/// `"default"` is legal in a file — a file that says it is a reset — even
/// though the viewer never writes that form.
#[test]
fn the_named_layout_round_trips() {
    let document = WindowLayout {
        window: None,
        layout: Some(LayoutSection::Default),
    };
    assert_eq!(
        document.to_json(),
        "{\n  \"sfm_explorer_layout\": 2,\n  \"layout\": \"default\"\n}\n"
    );
    assert_eq!(WindowLayout::from_json(&document.to_json()), Ok(document));
}

/// A document with neither section describes no change. That is legal in a
/// file — it loads as a no-op — and is what `set_window_layout` refuses.
#[test]
fn a_document_with_neither_section_is_empty() {
    for text in [
        "{}",
        r#"{"sfm_explorer_layout": 2}"#,
        r#"{"window": {}}"#,
        r#"{"sfm_explorer_layout": 2, "window": null, "layout": null}"#,
    ] {
        assert!(parsed(text).is_empty(), "{text} is not empty");
    }
    assert!(!WindowLayout::default().is_empty());
}

#[test]
fn every_panel_appears_exactly_once_in_the_default() {
    let layout = Layout::default();
    let mut tabs: Vec<Tab> = main_leaves(&layout)
        .into_iter()
        .flat_map(|(tabs, _)| tabs)
        .collect();
    assert_eq!(tabs.len(), Tab::ALL.len());
    for tab in Tab::ALL {
        assert!(tabs.contains(&tab), "{tab:?} is not in the default layout");
    }
    tabs.dedup();
    assert_eq!(tabs.len(), Tab::ALL.len(), "a panel appears twice");
    assert!(layout.validate("layout").is_ok());
}

/// The menu lists the panels in the order the default layout reads in: down
/// the left column, then the middle, then the right, then the bottom strip.
#[test]
fn tab_all_is_in_the_menus_order() {
    assert_eq!(
        Tab::ALL,
        [
            Tab::SceneGraph,
            Tab::Viewer3D,
            Tab::ImageBrowser,
            Tab::ImageDetail,
            Tab::PointTrackDetail,
            Tab::IntrinsicsDetail,
            Tab::ActionLog,
        ]
    );
}

// ── Wire names ───────────────────────────────────────────────────────────

#[test]
fn wire_names_round_trip() {
    for tab in Tab::ALL {
        assert_eq!(Tab::from_wire_name(tab.wire_name()), Some(tab));
    }
}

#[test]
fn from_wire_name_is_exact() {
    for name in ["Scene", "viewer3d", "", "scene "] {
        assert_eq!(Tab::from_wire_name(name), None, "{name:?} was accepted");
    }
}

// ── Home positions ───────────────────────────────────────────────────────

/// Rule 1: an open panel is raised where it is, and nothing moves.
#[test]
fn showing_an_open_panel_only_changes_the_active_tab() {
    let mut state = state();
    let before = state.layout();
    state.show_panel(Tab::PointTrackDetail);
    let after = state.layout();
    assert_ne!(before, after, "the active tab did not move");
    assert_eq!(
        main_leaves(&after)
            .into_iter()
            .map(|(tabs, _)| tabs)
            .collect::<Vec<_>>(),
        main_leaves(&before)
            .into_iter()
            .map(|(tabs, _)| tabs)
            .collect::<Vec<_>>(),
        "a raise moved a tab"
    );
    let leaf = main_leaves(&after)
        .into_iter()
        .find(|(tabs, _)| tabs.contains(&Tab::PointTrackDetail))
        .expect("the Point Track panel is gone");
    assert_eq!(leaf.1, Tab::PointTrackDetail);
    assert_eq!(
        layout_entries(&state),
        [(false, "Raised Point Track panel".into())]
    );
}

/// Rule 2: a panel whose default group-mate is still open goes in behind it.
#[test]
fn a_panel_goes_home_to_a_group_mate() {
    for (tab, mate) in [
        (Tab::PointTrackDetail, Tab::ImageDetail),
        (Tab::IntrinsicsDetail, Tab::ImageDetail),
        (Tab::ActionLog, Tab::ImageBrowser),
        (Tab::ImageBrowser, Tab::ActionLog),
    ] {
        let mut state = state();
        state.hide_panel(tab);
        state.show_panel(tab);
        let leaf = main_leaves(&state.layout())
            .into_iter()
            .find(|(tabs, _)| tabs.contains(&tab))
            .unwrap_or_else(|| panic!("{tab:?} was not re-opened"));
        assert!(
            leaf.0.contains(&mate),
            "{tab:?} did not land beside {mate:?}: {leaf:?}"
        );
        assert_eq!(leaf.0.last(), Some(&tab), "{tab:?} did not go in behind");
        assert_eq!(leaf.1, tab, "{tab:?} is not the active tab");
    }
}

/// Rule 3: with no group-mate left, a panel takes a split of the root along
/// its home edge, at its home share. The numbers here are the ones the spec's
/// table gives.
#[test]
fn a_panel_with_no_group_mate_splits_the_root() {
    for (tab, split, fraction, first_is_new) in [
        (Tab::SceneGraph, SplitDirection::LeftRight, 0.18, true),
        (Tab::ImageBrowser, SplitDirection::TopBottom, 0.80, false),
        (Tab::ImageDetail, SplitDirection::LeftRight, 0.67, false),
    ] {
        let mut state = state();
        for mate in tab.group().iter().copied().chain([tab]) {
            state.hide_panel(mate);
        }
        state.show_panel(tab);
        let main = state.layout().main.expect("the dock is empty");
        let LayoutNode::Split {
            split: got_split,
            fraction: got_fraction,
            first,
            second,
        } = main
        else {
            panic!("{tab:?} did not split the root: {main:?}");
        };
        assert_eq!(got_split, split, "{tab:?}");
        assert!(
            (got_fraction - fraction).abs() < 1e-6,
            "{tab:?}: fraction {got_fraction} is not {fraction}"
        );
        let new = if first_is_new { first } else { second };
        assert_eq!(
            *new,
            LayoutNode::Leaf {
                tabs: vec![tab],
                active: tab
            },
            "{tab:?} is not alone in the new node"
        );
    }
}

/// The 3D Viewer has no home edge: it joins the root's first leaf.
#[test]
fn the_viewer_joins_the_first_leaf() {
    let mut state = state();
    state.hide_panel(Tab::Viewer3D);
    state.show_panel(Tab::Viewer3D);
    let leaves = main_leaves(&state.layout());
    let first = leaves.first().expect("the dock is empty");
    assert!(
        first.0.contains(&Tab::Viewer3D),
        "the viewport did not join the first leaf: {leaves:?}"
    );
}

#[test]
fn any_panel_opened_into_an_empty_dock_becomes_its_root() {
    for tab in Tab::ALL {
        let mut state = state();
        for open in Tab::ALL {
            state.hide_panel(open);
        }
        assert_eq!(state.layout().main, None);
        state.show_panel(tab);
        assert_eq!(
            state.layout().main,
            Some(LayoutNode::Leaf {
                tabs: vec![tab],
                active: tab
            }),
            "{tab:?} did not become the root"
        );
    }
}

#[test]
fn hiding_then_showing_leaves_the_other_panels_where_they_were() {
    let mut state = state();
    let before = main_leaves(&state.layout());
    state.hide_panel(Tab::IntrinsicsDetail);
    state.show_panel(Tab::IntrinsicsDetail);
    let after = main_leaves(&state.layout());
    let strip = |leaves: Vec<(Vec<Tab>, Tab)>| -> Vec<Vec<Tab>> {
        leaves
            .into_iter()
            .map(|(mut tabs, _)| {
                tabs.retain(|tab| *tab != Tab::IntrinsicsDetail);
                tabs
            })
            .collect()
    };
    assert_eq!(strip(after), strip(before));
}

#[test]
fn hiding_a_closed_panel_does_nothing_and_says_nothing() {
    let mut state = state();
    state.hide_panel(Tab::ActionLog);
    let before = state.layout();
    state.hide_panel(Tab::ActionLog);
    assert_eq!(state.layout(), before);
    assert_eq!(layout_entries(&state).len(), 1);
}

// ── Validation ───────────────────────────────────────────────────────────

/// The message a document is refused with, or a panic if it was accepted.
#[track_caller]
fn refusal(text: &str) -> String {
    match WindowLayout::from_json(text) {
        Ok(document) => panic!("accepted: {document:?}"),
        Err(error) => error.to_string(),
    }
}

/// The same, for a `layout` section written on its own — every version-1 rule,
/// now under its `layout.` prefix.
#[track_caller]
fn layout_refusal(section: &str) -> String {
    refusal(&format!(
        r#"{{"sfm_explorer_layout": 2, "layout": {section}}}"#
    ))
}

#[test]
fn a_document_that_claims_nothing_is_not_a_layout_file() {
    // A JSON file that is not a layout at all says so, rather than complaining
    // about its own perfectly good keys.
    assert_eq!(refusal(r#"{"cameras": []}"#), "Not a layout file");
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": "two"}"#),
        "Not a layout file"
    );
    assert_eq!(refusal("[]"), "the document must be a JSON object");
}

#[test]
fn only_this_version_is_read() {
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 3, "layout": null}"#),
        "Layout version 3 is newer than this viewer reads (2)"
    );
    // Version 1 was the panel-only document, a different shape rather than a
    // different key, so there is no upgrade path.
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 1, "main": null, "windows": []}"#),
        "Layout version 1 is not one this viewer reads (2)"
    );
}

#[test]
fn an_unknown_panel_name_lists_the_seven() {
    let message = layout_refusal(r#"{"main": {"tabs": ["viewer3d"], "active": "viewer3d"}}"#);
    assert_eq!(
        message,
        "layout.main: unknown panel \"viewer3d\"; the panels are scene, viewer_3d, image_browser, \
         image_detail, point_track, camera_intrinsics, action_log"
    );
}

#[test]
fn a_panel_may_appear_only_once() {
    let two_leaves = r#"{"main": {
        "split": "left_right", "fraction": 0.5,
        "first": {"tabs": ["scene"]},
        "second": {"tabs": ["scene"]}}}"#;
    assert_eq!(
        layout_refusal(two_leaves),
        "layout.main.second: panel \"scene\" appears more than once"
    );
    let main_and_window = r#"{
        "main": {"tabs": ["scene"]},
        "windows": [{"tree": {"tabs": ["scene"]}}]}"#;
    assert_eq!(
        layout_refusal(main_and_window),
        "layout.windows[0].tree: panel \"scene\" appears more than once"
    );
}

#[test]
fn a_leaf_needs_a_tab_and_an_active_that_is_one_of_them() {
    assert_eq!(
        layout_refusal(r#"{"main": {"tabs": []}}"#),
        "layout.main: a leaf must have at least one tab"
    );
    assert_eq!(
        layout_refusal(r#"{"main": {"tabs": ["scene"], "active": "camera_intrinsics"}}"#),
        "layout.main: active \"camera_intrinsics\" is not one of this leaf's tabs"
    );
}

#[test]
fn a_fraction_is_strictly_between_zero_and_one() {
    for fraction in ["0", "1", "1.5"] {
        let section = format!(
            r#"{{"main": {{
                "split": "left_right", "fraction": {fraction},
                "first": {{"tabs": ["scene"]}},
                "second": {{"tabs": ["viewer_3d"]}}}}}}"#
        );
        assert_eq!(
            layout_refusal(&section),
            format!("layout.main: fraction must be strictly between 0 and 1, not {fraction}")
        );
    }
}

#[test]
fn an_unknown_key_is_refused_wherever_it_is() {
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 2, "layout": null, "extra": 1}"#),
        "unknown key \"extra\""
    );
    assert_eq!(
        layout_refusal(r#"{"main": null, "extra": 1}"#),
        "layout: unknown key \"extra\""
    );
    assert_eq!(
        layout_refusal(r#"{"main": {"tabs": ["scene"], "fracton": 0.5}}"#),
        "layout.main: unknown key \"fracton\""
    );
    let nested = r#"{"main": {
        "split": "left_right", "fracton": 0.5,
        "first": {"tabs": ["scene"]},
        "second": {"tabs": ["viewer_3d"]}}}"#;
    assert_eq!(
        layout_refusal(nested),
        "layout.main: unknown key \"fracton\""
    );
}

#[test]
fn a_node_that_is_neither_a_leaf_nor_a_split_is_refused() {
    assert_eq!(
        layout_refusal(r#"{"main": {}}"#),
        "layout.main: a node must have either \"tabs\" (a leaf) or \"split\" (a split)"
    );
}

#[test]
fn the_only_named_layout_is_the_default() {
    assert_eq!(
        layout_refusal(r#""tidy""#),
        "layout: the only named layout is \"default\""
    );
    assert_eq!(
        layout_refusal("7"),
        "layout: must be an arrangement, null, or \"default\""
    );
}

/// Each `window` key's own rule, with the path that says which one.
#[test]
fn every_window_key_is_checked() {
    let window = |section: &str| refusal(&format!(r#"{{"window": {section}}}"#));
    assert_eq!(window("7"), "window: must be an object or null");
    assert_eq!(
        window(r#"{"stat": "normal"}"#),
        "window: unknown key \"stat\""
    );
    assert_eq!(
        window(r#"{"state": "big"}"#),
        "window.state: unknown window state \"big\"; the states are normal, maximized, minimized, \
         fullscreen"
    );
    assert_eq!(
        window(r#"{"inner_size": [1280, 0]}"#),
        "window.inner_size: must be two whole numbers greater than zero"
    );
    assert_eq!(
        window(r#"{"inner_size": [1280]}"#),
        "window.inner_size: must be two whole numbers greater than zero"
    );
    assert_eq!(
        window(r#"{"outer_position": "left"}"#),
        "window.outer_position: must be two whole numbers"
    );
    assert_eq!(
        window(r#"{"focus": false}"#),
        "window.focus: can only ask for the foreground; omit it to leave focus alone"
    );
}

/// A monitor is what a rectangle is fitted *from*, so a section carrying one
/// and no rectangle has asked for something that cannot happen.
#[test]
fn a_monitor_needs_a_rectangle_to_fit() {
    let window = |section: &str| refusal(&format!(r#"{{"window": {section}}}"#));
    assert_eq!(
        window(r#"{"state": "maximized", "monitor": {"position": [0, 0], "size": [3840, 2160]}}"#),
        "window.monitor: has nothing to fit — send outer_position or inner_size with it"
    );
    let shape = "window.monitor: must be an object with \"position\" (two whole numbers) and \
                 \"size\" (two whole numbers greater than zero)";
    assert_eq!(
        window(r#"{"inner_size": [800, 600], "monitor": {"position": [0, 0]}}"#),
        shape
    );
    assert_eq!(
        window(r#"{"inner_size": [800, 600], "monitor": {"position": [0, 0], "size": [0, 2160]}}"#),
        shape
    );
}

#[test]
fn a_refused_load_leaves_the_dock_untouched() {
    let mut state = state();
    let before = state.layout();
    let bad = Layout {
        main: Some(LayoutNode::Leaf {
            tabs: vec![Tab::SceneGraph, Tab::SceneGraph],
            active: Tab::SceneGraph,
        }),
        windows: Vec::new(),
    };
    assert!(state.apply_layout(&bad).is_err());
    assert_eq!(state.layout(), before);
}

// ── The all-closed state, and floating windows ───────────────────────────

#[test]
fn all_closed_is_valid_and_round_trips() {
    let empty = Layout {
        main: None,
        windows: Vec::new(),
    };
    assert!(empty.validate("layout").is_ok());
    let document = just(empty.clone());
    assert_eq!(
        document.to_json(),
        "{\n  \"sfm_explorer_layout\": 2,\n  \"layout\": {\n    \"main\": null,\n    \
         \"windows\": []\n  }\n}\n"
    );
    assert_eq!(parsed_layout(&document.to_json()), empty);
    assert_eq!(Layout::from_dock(&empty.to_dock()), empty);
}

#[test]
fn a_floating_window_round_trips_with_and_without_a_rect() {
    for rect in [
        None,
        Some(LayoutRect {
            x: 40.0,
            y: 60.0,
            width: 320.0,
            height: 240.0,
        }),
    ] {
        let layout = Layout {
            main: Some(LayoutNode::leaf(&[Tab::Viewer3D])),
            windows: vec![LayoutWindow {
                tree: LayoutNode::Split {
                    split: SplitDirection::TopBottom,
                    fraction: 0.5,
                    first: Box::new(LayoutNode::leaf(&[Tab::ImageDetail])),
                    second: Box::new(LayoutNode::leaf(&[Tab::ActionLog])),
                },
                rect,
            }],
        };
        assert!(layout.validate("layout").is_ok());
        assert_eq!(parsed_layout(&just(layout.clone()).to_json()), layout);
        // A rect the window has not been drawn with yet is not reported back,
        // which is the whole reason the field is optional.
        let read_back = Layout::from_dock(&layout.to_dock());
        assert_eq!(read_back.main, layout.main);
        assert_eq!(read_back.windows.len(), 1);
        assert_eq!(read_back.windows[0].tree, layout.windows[0].tree);
        assert_eq!(read_back.windows[0].rect, None);
    }
}

// ── The Action Log ───────────────────────────────────────────────────────

#[test]
fn every_layout_action_is_logged_under_its_own_kind_and_none_coalesce() {
    assert!(!Kind::Layout.coalesces());
    assert_eq!(Kind::Layout.label(), "Layout");

    let mut state = state();
    state.hide_panel(Tab::ActionLog);
    state.hide_panel(Tab::IntrinsicsDetail);
    state.show_panel(Tab::ActionLog);
    state.show_panel(Tab::SceneGraph);
    state.reset_layout();
    assert_eq!(
        layout_entries(&state),
        [
            (false, "Closed Action Log panel".to_string()),
            (false, "Closed Camera Intrinsics panel".to_string()),
            (false, "Opened Action Log panel".to_string()),
            (false, "Raised Scene panel".to_string()),
            (false, "Reset layout".to_string()),
        ]
    );
    assert_eq!(state.layout(), Layout::default());
}

#[test]
fn a_refused_layout_carries_the_reason_a_caller_can_log() {
    let error = WindowLayout::from_json("not json at all").expect_err("accepted");
    assert!(error.to_string().starts_with("not valid JSON: "), "{error}");
}

// ── The Panels menu ──────────────────────────────────────────────────────

/// Draw the menu body in a headless frame and report every string it painted.
fn menu_texts(state: &mut AppState) -> Vec<String> {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(400.0, 600.0),
        )),
        ..Default::default()
    };
    let mut host = FakeWindow::default();
    crate::test_support::painted_texts(&ctx, input, |ui| panels_menu(ui, state, &mut host))
}

#[test]
fn the_menu_lists_every_panel_and_the_layout_items() {
    let mut state = state();
    let texts = menu_texts(&mut state);
    for tab in Tab::ALL {
        assert!(
            texts.iter().any(|text| text == tab.title()),
            "{:?} is missing from the menu: {texts:?}",
            tab.title()
        );
    }
    for item in ["Reset Layout", "Save Layout...", "Load Layout..."] {
        assert!(
            texts.iter().any(|text| text == item),
            "{item} is missing from the menu: {texts:?}"
        );
    }
}

/// The tick is a live read of the dock, not a remembered flag.
#[test]
fn the_menu_ticks_what_is_open() {
    let mut state = state();
    for tab in Tab::ALL {
        assert!(state.is_panel_open(tab), "{tab:?} did not start open");
    }
    state.hide_panel(Tab::IntrinsicsDetail);
    assert!(!state.is_panel_open(Tab::IntrinsicsDetail));
    // Drawing the menu reads the state and must not change it.
    let before = state.layout();
    let _ = menu_texts(&mut state);
    assert_eq!(state.layout(), before);
}

// ── The window's placement ───────────────────────────────────────────────

/// The window portion of a document, or a panic if it carries none.
#[track_caller]
fn placement(document: &WindowLayout) -> &WindowChange {
    document.window.as_ref().expect("no window section")
}

/// `winit` reports only the *current* rectangle, so the one a maximized window
/// will come back to has to have been remembered from a normal frame.
#[test]
fn the_normal_rectangle_is_remembered_across_a_maximize() {
    let mut state = state();
    let mut host = FakeWindow::default();
    state.observe_window(&host);
    assert_eq!(
        state.window_normal_rect,
        Some(NormalRect {
            outer_position: Some([120, 64]),
            inner_size: [1920, 1080],
        })
    );

    // Maximizing moves the window to the whole monitor, and the frame observes
    // that — but the rectangle it remembers is still the normal one.
    host.set_state(WindowState::Maximized);
    host.inner_size = [3840, 2160];
    state.observe_window(&host);
    let document = state.window_layout();
    assert_eq!(placement(&document).state, Some(WindowState::Maximized));
    assert_eq!(placement(&document).outer_position, Some([120, 64]));
    assert_eq!(placement(&document).inner_size, Some([1920, 1080]));
    // …and the monitor it was measured on, so it can be fitted elsewhere.
    assert_eq!(
        placement(&document).monitor,
        Some(MonitorRect {
            position: [0, 0],
            size: [3840, 2160]
        })
    );
}

/// A headless `AppState` has no window to describe, and says nothing about one
/// rather than inventing a rectangle.
#[test]
fn a_document_from_a_windowless_state_has_no_window_section() {
    let state = state();
    let document = state.window_layout();
    assert_eq!(document.window, None);
    assert!(!document.to_json().contains("\"window\":"));
}

/// § "How a `window` section is applied", row by row: the primitive sequence,
/// and the flags and rectangle it leaves behind.
#[test]
fn a_window_section_is_applied_in_one_order() {
    // Maximize where it is: no geometry, so nothing but the state.
    let mut host = FakeWindow::default();
    host.apply(&WindowChange {
        state: Some(WindowState::Maximized),
        ..WindowChange::default()
    })
    .expect("applied");
    assert_eq!(host.applied, ["state maximized"]);
    assert!(host.maximized);

    // A size against a normal window: made normal (it already is), then sized.
    let mut host = FakeWindow::default();
    host.apply(&WindowChange {
        inner_size: Some([1700, 1300]),
        ..WindowChange::default()
    })
    .expect("applied");
    assert_eq!(host.applied, ["state normal", "size 1700x1300"]);
    assert_eq!(host.inner_size, [1700, 1300]);

    // A size against a *maximized* window sets what it will restore to and
    // leaves it maximized — the rule a saved maximized layout needs.
    let mut host = FakeWindow::in_state(WindowState::Maximized);
    host.apply(&WindowChange {
        inner_size: Some([1700, 1300]),
        ..WindowChange::default()
    })
    .expect("applied");
    assert_eq!(
        host.applied,
        ["state normal", "size 1700x1300", "state maximized"]
    );
    assert!(host.maximized, "{host:?}");
    assert_eq!(host.inner_size, [1700, 1300]);

    // The whole placement: geometry on a normal window, then the named state,
    // then focus.
    let mut host = FakeWindow::in_state(WindowState::Minimized);
    host.apply(&WindowChange {
        state: Some(WindowState::Maximized),
        outer_position: Some([120, 64]),
        inner_size: Some([1700, 1300]),
        monitor: None,
        focus: true,
    })
    .expect("applied");
    assert_eq!(
        host.applied,
        [
            "state normal",
            "position 120,64",
            "size 1700x1300",
            "state maximized",
            "focus"
        ]
    );

    // A restored size, and minimized.
    let mut host = FakeWindow::default();
    host.apply(&WindowChange {
        state: Some(WindowState::Minimized),
        inner_size: Some([1700, 1300]),
        ..WindowChange::default()
    })
    .expect("applied");
    assert_eq!(
        host.applied,
        ["state normal", "size 1700x1300", "state minimized"]
    );
    assert!(host.minimized);
    assert_eq!(host.inner_size, [1700, 1300]);
}

/// A platform that will not place a window refuses, and stops the call there
/// rather than applying the rest of it.
#[test]
fn a_platform_refusal_stops_the_call() {
    let mut host = FakeWindow {
        position: None,
        ..FakeWindow::default()
    };
    let error = host
        .apply(&WindowChange {
            outer_position: Some([10, 20]),
            inner_size: Some([1700, 1300]),
            ..WindowChange::default()
        })
        .expect_err("refused");
    assert!(error.0.contains("position its own window"), "{error}");
    assert_eq!(host.applied, ["state normal", "position 10,20"]);
    assert_eq!(host.inner_size, [1920, 1080], "the size was applied anyway");
}

/// There is nothing to apply to without a window, and the message says so.
#[test]
fn there_is_no_window_to_apply_to() {
    let error = crate::test_support::NoWindow
        .apply(&WindowChange {
            state: Some(WindowState::Maximized),
            ..WindowChange::default()
        })
        .expect_err("refused");
    assert_eq!(error.0, crate::window::NO_WINDOW);
}

// ── Fitting a rectangle to the desktop ───────────────────────────────────

fn monitor(position: [i32; 2], size: [u32; 2]) -> MonitorInfo {
    MonitorInfo {
        name: None,
        position,
        size,
        scale_factor: 1.0,
    }
}

/// A rectangle with the monitor it was measured on.
fn saved_on(position: [i32; 2], size: [u32; 2], monitor: MonitorRect) -> WindowChange {
    WindowChange {
        outer_position: Some(position),
        inner_size: Some(size),
        monitor: Some(monitor),
        ..WindowChange::default()
    }
}

const FOUR_K: MonitorRect = MonitorRect {
    position: [0, 0],
    size: [3840, 2160],
};

#[test]
fn a_rectangle_from_this_desktop_is_used_as_saved() {
    let monitors = [
        monitor([0, 0], [3840, 2160]),
        monitor([3840, 0], [1920, 1080]),
    ];
    // Straddling the two, which no monitor contains — but the desktop is the
    // one the file was written at, so the window goes back where it was.
    let change = saved_on([3600, 100], [640, 480], FOUR_K);
    let (fitted, from) = crate::window::fit_to_monitor(&change, &monitors, Some(&monitors[0]));
    assert_eq!(from, None);
    assert_eq!(fitted.outer_position, Some([3600, 100]));
    assert_eq!(fitted.inner_size, Some([640, 480]));
    assert_eq!(fitted.monitor, None, "the monitor never reaches a host");
}

#[test]
fn a_rectangle_that_still_lands_on_a_monitor_is_used_as_saved() {
    // The 4K monitor is gone, but the rectangle fits inside the one that is
    // left, which is all the fit was for.
    let monitors = [monitor([0, 0], [1920, 1080])];
    let change = saved_on([100, 100], [640, 480], FOUR_K);
    let (fitted, from) = crate::window::fit_to_monitor(&change, &monitors, Some(&monitors[0]));
    assert_eq!(from, None);
    assert_eq!(fitted.outer_position, Some([100, 100]));
    assert_eq!(fitted.inner_size, Some([640, 480]));
}

#[test]
fn a_rectangle_off_every_monitor_keeps_its_share() {
    // The left half of a 4K monitor becomes the left half of a 1080p one.
    let monitors = [monitor([0, 0], [1920, 1080])];
    let change = saved_on([0, 0], [1920, 2160], FOUR_K);
    let (fitted, from) = crate::window::fit_to_monitor(&change, &monitors, Some(&monitors[0]));
    assert_eq!(from, Some(FOUR_K));
    assert_eq!(fitted.outer_position, Some([0, 0]));
    assert_eq!(fitted.inner_size, Some([960, 1080]));

    // …and onto a monitor that is not at the origin, the share is measured
    // from that monitor's own corner.
    let monitors = [monitor([1920, 0], [1920, 1080])];
    let change = saved_on([1920, 1080], [1920, 1080], FOUR_K);
    let (fitted, from) = crate::window::fit_to_monitor(&change, &monitors, Some(&monitors[0]));
    assert_eq!(from, Some(FOUR_K));
    assert_eq!(fitted.outer_position, Some([1920 + 960, 540]));
    assert_eq!(fitted.inner_size, Some([960, 540]));
}

#[test]
fn a_size_alone_is_scaled_and_never_reaches_zero() {
    let monitors = [monitor([0, 0], [1920, 1080])];
    let change = WindowChange {
        inner_size: Some([2, 2]),
        monitor: Some(FOUR_K),
        ..WindowChange::default()
    };
    let (fitted, from) = crate::window::fit_to_monitor(&change, &monitors, Some(&monitors[0]));
    assert_eq!(from, Some(FOUR_K));
    assert_eq!(fitted.outer_position, None);
    assert_eq!(fitted.inner_size, Some([1, 1]), "a side is never zero");
}

#[test]
fn nothing_is_fitted_without_a_monitor_to_fit_from_or_onto() {
    let monitors = [monitor([0, 0], [1920, 1080])];
    // An agent that sends a bare rectangle means that rectangle.
    let bare = WindowChange {
        outer_position: Some([9000, 9000]),
        inner_size: Some([640, 480]),
        ..WindowChange::default()
    };
    let (fitted, from) = crate::window::fit_to_monitor(&bare, &monitors, Some(&monitors[0]));
    assert_eq!(from, None);
    assert_eq!(fitted.outer_position, Some([9000, 9000]));

    // And with no monitor to map onto, there is nothing to map with.
    let change = saved_on([9000, 9000], [640, 480], FOUR_K);
    let (fitted, from) = crate::window::fit_to_monitor(&change, &monitors, None);
    assert_eq!(from, None);
    assert_eq!(fitted.outer_position, Some([9000, 9000]));
}

// ── Applying a whole document ────────────────────────────────────────────

/// The window goes first, so a panel tree is laid out into the window it was
/// meant for.
#[test]
fn a_document_applies_the_window_before_the_panels() {
    let mut state = state();
    let mut host = FakeWindow::default();
    state.observe_window(&host);
    let document = WindowLayout {
        window: Some(WindowChange {
            state: Some(WindowState::Maximized),
            ..WindowChange::default()
        }),
        layout: Some(LayoutSection::Default),
    };
    state.hide_panel(Tab::ActionLog);
    state
        .apply_window_layout(&mut host, &document)
        .expect("applied");
    assert_eq!(host.applied, ["state maximized"]);
    assert_eq!(state.layout(), Layout::default());
    // The snapshot was refreshed, so a later call in the same batch sees it.
    assert_eq!(
        state.window.as_ref().map(|info| info.state),
        Some(WindowState::Maximized)
    );
}

#[test]
fn a_refused_window_leaves_the_panels_alone() {
    let mut state = state();
    let mut host = FakeWindow {
        position: None,
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    let before = state.layout();
    let error = state
        .apply_window_layout(
            &mut host,
            &WindowLayout {
                window: Some(WindowChange {
                    outer_position: Some([10, 20]),
                    ..WindowChange::default()
                }),
                layout: Some(LayoutSection::Default),
            },
        )
        .expect_err("refused");
    assert!(error.to_string().starts_with("window: "), "{error}");
    assert_eq!(state.layout(), before);

    // A *validation* refusal touches neither, because the document is
    // validated whole before any of it is applied.
    let mut host = FakeWindow::default();
    let error = state
        .apply_window_layout(
            &mut host,
            &WindowLayout {
                window: Some(WindowChange {
                    state: Some(WindowState::Maximized),
                    ..WindowChange::default()
                }),
                layout: Some(LayoutSection::Layout(Layout {
                    main: Some(LayoutNode::Leaf {
                        tabs: vec![Tab::SceneGraph, Tab::SceneGraph],
                        active: Tab::SceneGraph,
                    }),
                    windows: Vec::new(),
                })),
            },
        )
        .expect_err("refused");
    assert!(error.to_string().contains("more than once"), "{error}");
    assert!(host.applied.is_empty(), "{:?}", host.applied);
    assert_eq!(state.layout(), before);
}

/// The host is handed the *fitted* rectangle and never a monitor, and what
/// comes back is what the Action Log row is composed from.
#[test]
fn applying_a_document_fits_the_rectangle_before_the_host_sees_it() {
    let mut state = state();
    let mut host = FakeWindow {
        monitors: vec![MonitorInfo {
            name: Some("SMALL".to_string()),
            position: [0, 0],
            size: [1920, 1080],
            scale_factor: 1.0,
        }],
        minimum: [1, 1],
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    // Half the 4K monitor, which does not fit on the 1080p one that is left.
    let document = WindowLayout {
        window: Some(saved_on([0, 0], [2560, 1440], FOUR_K)),
        layout: None,
    };
    let applied = state
        .apply_window_layout(&mut host, &document)
        .expect("applied")
        .expect("a window portion");
    assert_eq!(applied.fitted, Some(FOUR_K));
    assert_eq!(applied.change.monitor, None);
    assert_eq!(applied.change.inner_size, Some([1280, 720]));
    assert_eq!(
        host.applied,
        ["state normal", "position 0,0", "size 1280x720"]
    );
    assert_eq!(
        applied.change.log_text(applied.fitted),
        "Moved window to (0, 0); resized window to 1280×720, fitted from a 3840×2160 monitor"
    );
}

/// The phrase for a call that was applied as it was sent.
#[test]
fn an_unfitted_change_reads_as_what_was_asked() {
    let change = WindowChange {
        state: Some(WindowState::Maximized),
        outer_position: Some([120, 64]),
        inner_size: Some([1280, 720]),
        monitor: None,
        focus: true,
    };
    assert_eq!(
        change.log_text(None),
        "Moved window to (120, 64); resized window to 1280×720; maximized window; focused window"
    );
}

// ── The default file ─────────────────────────────────────────────────────

#[test]
fn the_default_file_is_in_the_home_directory() {
    let Some(path) = default_layout_path() else {
        // A platform with no home directory: the dialogs open where the
        // platform opens them and no startup load is attempted.
        return;
    };
    assert!(path.ends_with(DEFAULT_LAYOUT_FILE_NAME), "{path:?}");
    #[allow(deprecated)]
    let home = std::env::home_dir().expect("a home directory, since the path resolved");
    assert!(path.starts_with(&home), "{path:?} is not under {home:?}");
}

#[test]
fn loading_a_file_records_where_it_came_from() {
    let directory =
        std::env::temp_dir().join(format!("sfm-explorer-layout-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("a scratch directory");
    let path = directory.join("good.json");
    let saved = WindowLayout {
        window: Some(WindowChange {
            state: Some(WindowState::Maximized),
            ..WindowChange::default()
        }),
        layout: Some(LayoutSection::Layout(Layout {
            main: Some(LayoutNode::leaf(&[Tab::ActionLog])),
            windows: Vec::new(),
        })),
    };
    std::fs::write(&path, saved.to_json()).expect("write");

    let mut state = state();
    let mut host = FakeWindow::default();
    state.observe_window(&host);
    state.load_layout_file(&mut host, &path);
    assert!(host.maximized);
    assert_eq!(state.layout(), *arrangement(&saved));
    assert_eq!(
        layout_entries(&state),
        [(false, format!("Loaded layout from {}", path.display()))]
    );

    // A file that does not validate is refused whole, and the entry says why —
    // which is what puts it on the viewport status line.
    let bad = directory.join("bad.json");
    std::fs::write(
        &bad,
        r#"{"sfm_explorer_layout": 2, "layout": {"main": {}}}"#,
    )
    .expect("write");
    let mut state = crate::state::AppState::new();
    state.observe_window(&host);
    let before = state.layout();
    state.load_layout_file(&mut host, &bad);
    assert_eq!(state.layout(), before);
    let entries = layout_entries(&state);
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].0, "the entry is a failure: {entries:?}");
    assert!(entries[0].1.contains("layout.main"), "{entries:?}");

    std::fs::remove_dir_all(&directory).ok();
}
