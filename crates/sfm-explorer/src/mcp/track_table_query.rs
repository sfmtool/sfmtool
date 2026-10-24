// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `get_track_table_query` and `set_track_table_query`: the Track Table
//! panel's query as one document.
//!
//! See `specs/gui/mcp-server.md` § "`get_track_table_query` /
//! `set_track_table_query`". The panel keeps its query in its own state and
//! publishes it on `AppState` each frame
//! ([`crate::track_table_panel::PublishedQuery`]); these read that copy, and a
//! set leaves the merged query there for the panel to take on its next frame,
//! opening the panel when it is closed. The bounds and the sort are parsed by
//! the parsers `find_points` uses, so the two take one column vocabulary and
//! one bound syntax.

use serde_json::{json, Value};

use super::tools::Args;
use super::{JsonReply, ToolError};
use crate::action_log::Kind;
use crate::dock::Tab;
use crate::state::AppState;
use crate::track_table_panel::{
    panel_column_names, Filter3dChoice, PanelColumn, TrackTableQueryChange,
};

/// The write tool's name, which its refusals start with.
pub(super) const SET_TOOL: &str = "set_track_table_query";

/// `get_track_table_query`: the query and the reconstruction the panel
/// applies it to.
pub(super) fn get(state: &AppState) -> JsonReply {
    Ok(document(state))
}

/// `set_track_table_query`: merge the fields the call named into the query,
/// record what changed in the panel's words, open the panel when it is
/// closed, and answer with the whole document.
pub(super) fn set(state: &mut AppState, change: &TrackTableQueryChange) -> JsonReply {
    let rows = state
        .track_table_query
        .set(SET_TOOL, change)
        .map_err(ToolError::new)?;
    for row in rows {
        state.action_log.record(Kind::Display, row);
    }
    if !state.is_panel_open(Tab::TrackTable) {
        state.show_panel(Tab::TrackTable);
    }
    Ok(document(state))
}

/// The document both tools answer with: the query, and the label of the
/// reconstruction the panel shows, the selected one, or `null` with none.
fn document(state: &AppState) -> Value {
    let label = crate::scene::selected_node(&state.scene, state.selected_recon)
        .map(|node| node.label.clone());
    json!({
        "track_table_query": state.track_table_query.document(),
        "reconstruction_label": label,
    })
}

/// Every argument `set_track_table_query` takes, and the whole of what its
/// parse refuses: a column that does not exist, a picture column as a bound
/// or a sort key, two bounds on one column, a bound whose `min` is above its
/// `max`, an unknown `filter_3d`, a value of the wrong type and a call that
/// names nothing.
pub(super) fn parse_change(args: &Args) -> Result<TrackTableQueryChange, ToolError> {
    let bounds = match args.get("bounds") {
        None | Some(Value::Null) => None,
        Some(_) => {
            let bounds = args.column_bounds("bounds")?;
            for (i, bound) in bounds.iter().enumerate() {
                if bounds[..i].iter().any(|b| b.column == bound.column) {
                    return Err(args.error(format!(
                        "names the column {} in two bounds — the panel holds one minimum and one \
                         maximum per column.",
                        bound.column.wire_name()
                    )));
                }
            }
            Some(bounds)
        }
    };
    let order_by = match args.get("order_by") {
        None => None,
        Some(_) => Some(args.order_by("order_by")?),
    };
    let columns = match args.get("columns") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let expected = "an array of the Track Table's column names";
            let array = value
                .as_array()
                .ok_or_else(|| args.wrong_type("columns", expected, value))?;
            let columns = array
                .iter()
                .map(|element| {
                    let name = element
                        .as_str()
                        .ok_or_else(|| args.wrong_type("columns", expected, value))?;
                    PanelColumn::from_wire_name(name).ok_or_else(|| {
                        args.error(format!(
                            "does not know the column {name:?} — the Track Table's columns are \
                             {}.",
                            panel_column_names().join(", ")
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(columns)
        }
    };
    let filter_3d = match args.optional_string("filter_3d")? {
        None => None,
        Some(name) => Some(Filter3dChoice::from_wire_name(&name).ok_or_else(|| {
            args.error(format!(
                "wants filter_3d to be one of {} — got {name:?}.",
                Filter3dChoice::ALL
                    .iter()
                    .map(|c| format!("{:?}", c.wire_name()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?),
    };
    let change = TrackTableQueryChange {
        bounds,
        order_by,
        columns,
        seen_in_selected_camera_image: args.optional_bool("seen_in_selected_camera_image")?,
        filter_3d,
    };
    if change.is_empty() {
        return Err(args.error("was given nothing to change."));
    }
    Ok(change)
}
