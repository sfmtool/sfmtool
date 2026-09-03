// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The layout tools: where the window is, which panels are open, and how they
//! are arranged.
//!
//! All four answer with the same block, [`reply`], for the reason every write
//! on this surface answers with its resulting state: where a panel *landed* is
//! the home rule's decision (`specs/gui/panel-layout.md` § "Home positions"),
//! not the caller's, and an agent that assumed would be wrong the first time a
//! group-mate was closed.
//!
//! Nothing here parses or prints a document itself. `set_window_layout` hands
//! its whole argument to [`WindowLayout::from_value`] and `get_window_layout`
//! re-reads [`WindowLayout::to_json`], so the wire and the file share one
//! schema, one parser and one set of validation messages — an agent can save
//! what it read to the file the viewer reads at startup, and send back a file a
//! human saved.

use serde_json::{json, Value};

use super::{JsonReply, ToolError};
use crate::action_log::Kind;
use crate::dock::Tab;
use crate::layout::{Layout, LayoutNode, LayoutSection, WindowLayout};
use crate::state::AppState;
use crate::window::{MonitorInfo, WindowHost};

/// `get_window_layout`: the document, the live window, and the panels.
pub(super) fn get_window_layout(state: &mut AppState, host: &dyn WindowHost) -> JsonReply {
    // A read is also the freshest observation anyone has, so the snapshot the
    // rest of the surface answers from is brought up to date with it.
    state.observe_window(host);
    Ok(reply(state, host))
}

/// `set_window_layout`: the window portion, the panel portion, or both.
pub(super) fn set_window_layout(
    state: &mut AppState,
    host: &mut dyn WindowHost,
    value: &Value,
) -> JsonReply {
    // Parsed and validated as a whole before any of it is applied, so a
    // refusal leaves the window and the dock exactly as they were.
    let document =
        WindowLayout::from_value(value).map_err(|error| ToolError::new(error.to_string()))?;
    let applied = state
        .apply_window_layout(host, &document)
        .map_err(|error| ToolError::new(error.to_string()))?;
    // One row per portion, because the two portions are two kinds. Neither
    // `apply_window_layout` nor `apply_layout` records anything itself: their
    // callers word the entry differently — the menu says which file, this says
    // which tool.
    if let Some(applied) = applied {
        state
            .action_log
            .record(Kind::Window, applied.change.log_text(applied.fitted));
    }
    match &document.layout {
        Some(LayoutSection::Layout(_)) => state.action_log.record(Kind::Layout, "Set layout"),
        Some(LayoutSection::Default) => state.action_log.record(Kind::Layout, "Reset layout"),
        None => {}
    }
    Ok(reply(state, host))
}

/// `show_panel`: open a panel at its home position, or raise it if it is open.
pub(super) fn show_panel(state: &mut AppState, host: &dyn WindowHost, panel: Tab) -> JsonReply {
    state.show_panel(panel);
    Ok(reply(state, host))
}

/// `hide_panel`: close a panel. Idempotent, as the method is.
pub(super) fn hide_panel(state: &mut AppState, host: &dyn WindowHost, panel: Tab) -> JsonReply {
    state.hide_panel(panel);
    Ok(reply(state, host))
}

/// The block all four layout tools return.
///
/// Three views of one arrangement. **`window_layout` is the file**: the object
/// `WindowLayout::to_json` writes, parsed — an agent that saves it has a file
/// the menu loads and the viewer reads at startup. The **`window` block beside
/// it is the observation**: focus, scale factor, the *current* (not normal)
/// geometry, and every monitor, read live from the host. The two agree for a
/// normal window and differ for a maximized one, and that difference is the
/// information. **`panels` is the same arrangement indexed the other way**,
/// because "is the Action Log open" should not cost the agent a tree walk.
pub(super) fn reply(state: &AppState, host: &dyn WindowHost) -> Value {
    let monitors: Option<Vec<MonitorInfo>> = host.observe().map(|(_, monitors)| monitors);
    json!({
        "window_layout": document(&state.window_layout()),
        "window": state
            .window
            .as_ref()
            .map(|info| super::window::block(info, monitors.as_deref())),
        "panels": panels(&state.layout()),
    })
}

/// The document as the file spells it, read back through the file's own writer
/// so the two can never describe one arrangement differently.
fn document(layout: &WindowLayout) -> Value {
    serde_json::from_str(&layout.to_json()).unwrap_or(Value::Null)
}

/// One entry per panel, always all seven: whether it is docked anywhere, and
/// whether it is the front tab of its node.
///
/// A panel alone in a node is active; a closed one is not.
fn panels(layout: &Layout) -> Value {
    let mut open = Vec::new();
    let mut active = Vec::new();
    if let Some(main) = &layout.main {
        walk(main, &mut open, &mut active);
    }
    for window in &layout.windows {
        walk(&window.tree, &mut open, &mut active);
    }
    let entries: serde_json::Map<String, Value> = Tab::ALL
        .into_iter()
        .map(|tab| {
            (
                tab.wire_name().to_string(),
                json!({
                    "open": open.contains(&tab),
                    "active": active.contains(&tab),
                }),
            )
        })
        .collect();
    Value::Object(entries)
}

/// Collect every panel in a subtree, and the front tab of each of its leaves.
fn walk(node: &LayoutNode, open: &mut Vec<Tab>, active: &mut Vec<Tab>) {
    match node {
        LayoutNode::Leaf {
            tabs,
            active: front,
        } => {
            open.extend(tabs.iter().copied());
            active.push(*front);
        }
        LayoutNode::Split { first, second, .. } => {
            walk(first, open, active);
            walk(second, open, active);
        }
    }
}
