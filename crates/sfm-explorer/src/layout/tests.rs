// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the panel layout: the default grid, the schema, the
//! file, and the home positions a panel is re-opened at.
//!
//! None of it needs a window — a `DockState<Tab>` is plain data, and the Panels
//! menu is drawn through the `test_support` frame harness.

use super::*;

use crate::action_log::Kind;

/// The default layout as the viewer writes it. Kept verbatim in
/// `specs/gui/panel-layout.md` § "The layout file", so the spec and the code
/// cannot drift.
const DEFAULT_JSON: &str = r#"{
  "sfm_explorer_layout": 1,
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
          "tabs": ["image_detail", "point_track", "intrinsics"],
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
"#;

/// A state whose dock is the stock grid and whose log is empty.
fn state() -> AppState {
    AppState::new()
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
    let layout = Layout::default();
    assert_eq!(layout.to_json(), DEFAULT_JSON);
    assert_eq!(Layout::from_json(DEFAULT_JSON), Ok(layout));
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
    assert!(layout.validate().is_ok());
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
fn refusal(text: &str) -> String {
    match Layout::from_json(text) {
        Ok(layout) => panic!("accepted: {layout:?}"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn a_document_without_a_version_is_not_a_layout_file() {
    assert_eq!(refusal(r#"{"main": null}"#), "Not a layout file");
    assert_eq!(refusal("{}"), "Not a layout file");
    assert_eq!(refusal("[]"), "the document must be a JSON object");
}

#[test]
fn a_newer_version_says_so() {
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 2, "main": null}"#),
        "Layout version 2 is newer than this viewer reads (1)"
    );
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 0, "main": null}"#),
        "Layout version 0 is not one this viewer reads (1)"
    );
}

#[test]
fn an_unknown_panel_name_lists_the_seven() {
    let message = refusal(
        r#"{"sfm_explorer_layout": 1, "main": {"tabs": ["viewer3d"], "active": "viewer3d"}}"#,
    );
    assert_eq!(
        message,
        "main: unknown panel \"viewer3d\"; the panels are scene, viewer_3d, image_browser, \
         image_detail, point_track, intrinsics, action_log"
    );
}

#[test]
fn a_panel_may_appear_only_once() {
    let two_leaves = r#"{"sfm_explorer_layout": 1, "main": {
        "split": "left_right", "fraction": 0.5,
        "first": {"tabs": ["scene"]},
        "second": {"tabs": ["scene"]}}}"#;
    assert_eq!(
        refusal(two_leaves),
        "main.second: panel \"scene\" appears more than once"
    );
    let main_and_window = r#"{"sfm_explorer_layout": 1,
        "main": {"tabs": ["scene"]},
        "windows": [{"tree": {"tabs": ["scene"]}}]}"#;
    assert_eq!(
        refusal(main_and_window),
        "windows[0].tree: panel \"scene\" appears more than once"
    );
}

#[test]
fn a_leaf_needs_a_tab_and_an_active_that_is_one_of_them() {
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 1, "main": {"tabs": []}}"#),
        "main: a leaf must have at least one tab"
    );
    assert_eq!(
        refusal(
            r#"{"sfm_explorer_layout": 1, "main": {"tabs": ["scene"], "active": "intrinsics"}}"#
        ),
        "main: active \"intrinsics\" is not one of this leaf's tabs"
    );
}

#[test]
fn a_fraction_is_strictly_between_zero_and_one() {
    for fraction in ["0", "1", "1.5"] {
        let text = format!(
            r#"{{"sfm_explorer_layout": 1, "main": {{
                "split": "left_right", "fraction": {fraction},
                "first": {{"tabs": ["scene"]}},
                "second": {{"tabs": ["viewer_3d"]}}}}}}"#
        );
        assert_eq!(
            refusal(&text),
            format!("main: fraction must be strictly between 0 and 1, not {fraction}")
        );
    }
}

#[test]
fn an_unknown_key_is_refused_wherever_it_is() {
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 1, "main": null, "extra": 1}"#),
        "unknown key \"extra\""
    );
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 1, "main": {"tabs": ["scene"], "fracton": 0.5}}"#),
        "main: unknown key \"fracton\""
    );
    let nested = r#"{"sfm_explorer_layout": 1, "main": {
        "split": "left_right", "fracton": 0.5,
        "first": {"tabs": ["scene"]},
        "second": {"tabs": ["viewer_3d"]}}}"#;
    assert_eq!(refusal(nested), "main: unknown key \"fracton\"");
}

#[test]
fn a_node_that_is_neither_a_leaf_nor_a_split_is_refused() {
    assert_eq!(
        refusal(r#"{"sfm_explorer_layout": 1, "main": {}}"#),
        "main: a node must have either \"tabs\" (a leaf) or \"split\" (a split)"
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
    assert!(empty.validate().is_ok());
    assert_eq!(
        empty.to_json(),
        "{\n  \"sfm_explorer_layout\": 1,\n  \"main\": null,\n  \"windows\": []\n}\n"
    );
    assert_eq!(Layout::from_json(&empty.to_json()), Ok(empty.clone()));
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
        assert!(layout.validate().is_ok());
        assert_eq!(Layout::from_json(&layout.to_json()), Ok(layout.clone()));
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
            (false, "Closed Intrinsics panel".to_string()),
            (false, "Opened Action Log panel".to_string()),
            (false, "Raised Scene panel".to_string()),
            (false, "Reset layout".to_string()),
        ]
    );
    assert_eq!(state.layout(), Layout::default());
}

#[test]
fn a_refused_layout_carries_the_reason_a_caller_can_log() {
    let error = Layout::from_json("not json at all").expect_err("accepted");
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
    crate::test_support::painted_texts(&ctx, input, |ui| panels_menu(ui, state))
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
