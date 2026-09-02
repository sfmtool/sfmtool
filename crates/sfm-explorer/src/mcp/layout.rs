// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The layout tools: which panels are open, and how they are arranged.
//!
//! All four answer with the same block, [`layout_reply`], for the reason every
//! write on this surface answers with its resulting state: where a panel
//! *landed* is the home rule's decision (`specs/gui/panel-layout.md`
//! § "Home positions"), not the caller's, and an agent that assumed would be
//! wrong the first time a group-mate was closed.
//!
//! Nothing here parses or prints a layout document itself. `set_layout` hands
//! the argument object to [`Layout::from_value`] and `get_layout` re-reads
//! [`Layout::to_json`], so the wire and the file share one schema, one parser
//! and one set of validation messages — an agent can save what it read to a
//! file the Panels menu loads, and send back a file a human saved.

use serde_json::{json, Value};

use super::{JsonReply, ToolError};
use crate::action_log::Kind;
use crate::dock::Tab;
use crate::layout::{Layout, LayoutNode};
use crate::state::AppState;

/// What `set_layout` was asked for: a document, or the one layout with a name.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayoutTarget {
    /// The argument object, unparsed.
    ///
    /// Carried as JSON rather than as a parsed [`Layout`] so that a document
    /// the viewer will not accept is a *domain* error — the agent gets the
    /// parser's path-carrying message with `isError`, and the human at the
    /// window gets the same words in the Action Log, exactly as a refused
    /// Panels ▸ Load Layout… gives them.
    Document(Value),
    /// The stock seven-panel grid, as Panels ▸ Reset Layout restores it.
    Default,
}

/// `get_layout`: the layout file, and the same information indexed by panel.
pub(super) fn get_layout(state: &AppState) -> JsonReply {
    Ok(layout_reply(state))
}

/// `set_layout`: replace the whole arrangement.
pub(super) fn set_layout(state: &mut AppState, target: &LayoutTarget) -> JsonReply {
    match target {
        LayoutTarget::Default => state.reset_layout(),
        LayoutTarget::Document(value) => {
            // Parsed and validated as a whole before any of it is applied, so
            // a refusal leaves the dock exactly as it was.
            let layout =
                Layout::from_value(value).map_err(|error| ToolError::new(error.to_string()))?;
            state
                .apply_layout(&layout)
                .map_err(|error| ToolError::new(error.to_string()))?;
            // `apply_layout` records nothing itself, because its two callers
            // word the entry differently: the menu says which file, and this
            // says which tool.
            state.action_log.record(Kind::Layout, "Set layout");
        }
    }
    Ok(layout_reply(state))
}

/// `show_panel`: open a panel at its home position, or raise it if it is open.
pub(super) fn show_panel(state: &mut AppState, panel: Tab) -> JsonReply {
    state.show_panel(panel);
    Ok(layout_reply(state))
}

/// `hide_panel`: close a panel. Idempotent, as the method is.
pub(super) fn hide_panel(state: &mut AppState, panel: Tab) -> JsonReply {
    state.hide_panel(panel);
    Ok(layout_reply(state))
}

/// The block all four layout tools return.
///
/// `layout` is the file itself, parsed — not a rendering of it and not a
/// subset. `panels` is the same information indexed the other way, because
/// "is the Action Log open" should not cost the agent a tree walk.
pub(super) fn layout_reply(state: &AppState) -> Value {
    let layout = state.layout();
    json!({
        "layout": document(&layout),
        "panels": panels(&layout),
    })
}

/// The layout as the file spells it, read back through the file's own writer
/// so the two can never describe one arrangement differently.
fn document(layout: &Layout) -> Value {
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
