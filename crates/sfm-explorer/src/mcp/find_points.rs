// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `find_points`: the track table query over the wire, a page of matched
//! points at a time.
//!
//! See `specs/gui/mcp-server.md` § "`find_points`". The call runs core's
//! `query_points` over the track table cache ([`crate::track_table`]). When
//! every column it names is cached for the version at the cursor it answers in
//! the frame it is drained; otherwise it queues *Find points* for the missing
//! columns and answers when that lands, or with a handle once
//! [`super::REPLY_DIRECTLY_WITHIN`] has passed, as every call that starts a
//! background task does.

use serde_json::{json, Map, Value};
use sfmtool_core::analysis::track_table::{
    query_points, ColumnBounds, SortOrder, TrackTable, TrackTableColumn, TrackTableQueryOptions,
};

use super::{edit, CameraImageSel, Outcome, ToolError, ToolOutput};
use crate::scene::{ReconId, SceneNode};
use crate::state::AppState;

/// What a `find_points` call asked for, parsed: every column already named
/// by a [`TrackTableColumn`], and the page.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FindPointsRequest {
    /// The reconstruction, or `None` for the selected one.
    pub(super) reconstruction_label: Option<String>,
    /// Every bound must pass.
    pub(super) bounds: Vec<ColumnBounds>,
    /// Keep only the points this camera image observes.
    pub(super) seen_in_camera_image: Option<CameraImageSel>,
    /// The column to sort by, and which way.
    pub(super) order_by: Option<(TrackTableColumn, SortOrder)>,
    /// The values to return per point, or `None` for the default set.
    pub(super) columns: Option<Vec<TrackTableColumn>>,
    /// The first matched point to return.
    pub(super) offset: usize,
    /// How many to return, already capped at [`super::read::MAX_LIMIT`].
    pub(super) limit: usize,
}

/// The page a deferred `find_points` answers with once *Find points* lands:
/// the node, the values to return and which slice of the matched points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FindPointsPage {
    node: ReconId,
    columns: Vec<TrackTableColumn>,
    offset: usize,
    limit: usize,
}

/// Every column's wire name, for a refusal that has to list them.
pub(super) fn column_names(columns: impl IntoIterator<Item = TrackTableColumn>) -> String {
    columns
        .into_iter()
        .map(TrackTableColumn::wire_name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The column a wire name names, or the refusal listing every column. A
/// picture column of the Track Table panel is refused with its own sentence:
/// it is a column of the panel and holds no numbers, so a person or an agent
/// sees it in the panel's screenshot rather than in a reply.
pub(super) fn parse_column(tool: &str, name: &str) -> Result<TrackTableColumn, ToolError> {
    TrackTableColumn::from_wire_name(name).ok_or_else(|| {
        let columns = column_names(TrackTableColumn::ALL);
        if crate::track_table_panel::picture_column_names().contains(&name) {
            ToolError::new(format!(
                "{tool} does not take the column {name:?}: picture columns belong to the Track \
                 Table panel only, and screenshot {{\"panel_name\": \"track_table\"}} shows them. \
                 The columns are {columns}."
            ))
        } else {
            ToolError::new(format!(
                "{tool} does not know the column {name:?} — the columns are {columns}."
            ))
        }
    })
}

/// The values a call returns: the ones it named, or by default its bound and
/// sorted columns followed by `track_length` and `rms_error_px`, each once, in
/// the order first named.
fn returned_columns(request: &FindPointsRequest) -> Vec<TrackTableColumn> {
    let named: Vec<TrackTableColumn> = match &request.columns {
        Some(columns) => columns.clone(),
        None => request
            .bounds
            .iter()
            .map(|b| b.column)
            .chain(request.order_by.map(|(c, _)| c))
            .chain([TrackTableColumn::TrackLength, TrackTableColumn::RmsErrorPx])
            .collect(),
    };
    let mut out = Vec::new();
    for column in named {
        if !out.contains(&column) {
            out.push(column);
        }
    }
    out
}

/// `find_points`: run the query over the cached track table, computing the
/// columns the cache lacks first.
pub(super) fn find_points(state: &mut AppState, request: FindPointsRequest) -> Outcome {
    match prepare(state, &request) {
        Err(error) => Outcome::Done(Err(error)),
        Ok((id, options, columns)) => {
            let needed: Vec<TrackTableColumn> = options
                .bounds
                .iter()
                .map(|b| b.column)
                .chain(options.order_by.map(|(c, _)| c))
                .chain(columns.iter().copied())
                .collect();
            let page = FindPointsPage {
                node: id,
                columns,
                offset: request.offset,
                limit: request.limit,
            };
            let missing: Vec<TrackTableColumn> = TrackTableColumn::ALL
                .into_iter()
                .filter(|c| needed.contains(c))
                .filter(|&c| !state.track_table(id).is_some_and(|table| table.has(c)))
                .collect();
            if missing.is_empty() {
                return super::done(answer_now(state, &page, &options));
            }
            match state.start_find_points(id, missing, options.clone()) {
                Err(message) => Outcome::Done(Err(ToolError::new(message))),
                Ok(task) => edit::started_or(
                    state,
                    Some(task),
                    super::Answer::FoundPoints(Box::new(page.clone())),
                    |state| answer_now(state, &page, &options),
                ),
            }
        }
    }
}

/// Resolve the call against the scene: the node, the core query and the
/// columns to return, or the refusal for a camera image out of range or a
/// column the reconstruction cannot supply.
fn prepare(
    state: &AppState,
    request: &FindPointsRequest,
) -> Result<(ReconId, TrackTableQueryOptions, Vec<TrackTableColumn>), ToolError> {
    let id = super::resolve_reconstruction(state, request.reconstruction_label.as_deref())?;
    let seen_in_image = match &request.seen_in_camera_image {
        None => None,
        Some(selector) => Some(super::resolve_camera_image(state, id, selector)?.index() as u32),
    };
    let node = state.node(id).expect("just resolved");
    let columns = returned_columns(request);
    let named = request
        .bounds
        .iter()
        .map(|b| b.column)
        .chain(request.order_by.map(|(c, _)| c))
        .chain(columns.iter().copied());
    for column in named {
        if !column.is_available(node.edited()) {
            return Err(ToolError::new(format!(
                "{} cannot supply the column {}: the columns it can supply are {}.",
                node.label,
                column.wire_name(),
                column_names(
                    TrackTableColumn::ALL
                        .into_iter()
                        .filter(|c| c.is_available(node.edited()))
                )
            )));
        }
    }
    Ok((
        id,
        TrackTableQueryOptions {
            bounds: request.bounds.clone(),
            seen_in_image,
            order_by: request.order_by,
        },
        columns,
    ))
}

/// The reply from the cache as it stands, for a call whose columns are all
/// cached for the version at the cursor.
fn answer_now(
    state: &AppState,
    page: &FindPointsPage,
    options: &TrackTableQueryOptions,
) -> super::JsonReply {
    let node = state
        .node(page.node)
        .ok_or_else(|| ToolError::new(crate::state::NOT_LOADED))?;
    let table = state
        .track_table(page.node)
        .filter(|table| holds(table, page))
        .ok_or_else(|| changed_since(node))?;
    let matched = query_points(node.edited(), table, options)
        .map_err(|error| ToolError::new(format!("find_points: {error}")))?;
    Ok(points_reply(node, table, &matched, page))
}

/// What a deferred `find_points` answers once its *Find points* has landed:
/// the page of what it found, read from the cached table it filled.
pub(super) fn found_points_reply(
    state: &AppState,
    page: &FindPointsPage,
    found: Option<&crate::track_table::FoundPoints>,
) -> Result<ToolOutput, ToolError> {
    let node = state
        .node(page.node)
        .ok_or_else(|| ToolError::new(crate::state::NOT_LOADED))?;
    let found = found.ok_or_else(|| ToolError::new("Find points landed with nothing found."))?;
    let table = state
        .track_table(page.node)
        .filter(|_| node.history.current_version().document_serial == found.document)
        .filter(|table| holds(table, page))
        .ok_or_else(|| changed_since(node))?;
    Ok(ToolOutput::Json(points_reply(
        node,
        table,
        &found.matched,
        page,
    )))
}

/// Whether `table` holds every value the page returns. A cached table can
/// lack one the call found cached when it queued *Find points*: a run for
/// another version landed while that version was at the cursor and replaced
/// the entry, and the cursor came back before this call's run landed.
fn holds(table: &TrackTable, page: &FindPointsPage) -> bool {
    page.columns.iter().all(|&column| table.has(column))
}

/// The refusal for a node whose version moved between the query and the
/// reply.
fn changed_since(node: &SceneNode) -> ToolError {
    ToolError::new(format!(
        "{} changed before its points could be read back; call find_points again.",
        node.label
    ))
}

/// The reply: the counts, and one row per matched point in the page with its
/// id, its index and the asked-for values.
fn points_reply(
    node: &SceneNode,
    table: &TrackTable,
    matched: &[u32],
    page: &FindPointsPage,
) -> Value {
    let start = page.offset.min(matched.len());
    let end = page.offset.saturating_add(page.limit).min(matched.len());
    let points: Vec<Value> = matched[start..end]
        .iter()
        .map(|&index| {
            let row = table
                .point_indexes()
                .binary_search(&index)
                .expect("a matched point is a row of the table it was matched in");
            let mut entry = Map::new();
            entry.insert(
                "id".into(),
                json!(crate::scene::point_id(node, index as usize)),
            );
            entry.insert("index".into(), json!(index));
            for &column in &page.columns {
                entry.insert(column.wire_name().into(), value(table, column, row));
            }
            Value::Object(entry)
        })
        .collect();
    json!({
        "reconstruction_label": node.label,
        "live_points": table.point_indexes().len(),
        "matched": matched.len(),
        "offset": page.offset,
        "points": points,
    })
}

/// One value as the wire carries it: `null` for no value, the string `"inf"`
/// for `+∞` (the `centroid_distance` of a point at infinity), which JSON has no
/// number for, an integer for a count or a byte, a position at `f64`, and any
/// other column through its shortest `f32` spelling.
fn value(table: &TrackTable, column: TrackTableColumn, row: usize) -> Value {
    if column.is_f64() {
        let v = table.values_f64(column).expect("a needed column is held")[row];
        return edit::finite(v);
    }
    let v = table.values(column).expect("a needed column is held")[row];
    if v.is_nan() {
        return Value::Null;
    }
    if v.is_infinite() {
        return json!(if v > 0.0 { "inf" } else { "-inf" });
    }
    match column {
        TrackTableColumn::TrackLength
        | TrackTableColumn::RepeatObservations
        | TrackTableColumn::NormalConfidence => json!(v as i64),
        _ => json!(v.to_string().parse::<f64>().unwrap_or(f64::from(v))),
    }
}
