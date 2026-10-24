// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The panel's query as `get_track_table_query` and `set_track_table_query`
//! read and write it.
//!
//! See `specs/gui/mcp-server.md` § "`get_track_table_query` /
//! `set_track_table_query`". The document names the columns by the wire names
//! `find_points` takes, and the bounds by the same `min` / `max` ends, a number
//! or `"inf"`, so a query set here and a `find_points` call with the same
//! bounds agree. A call is applied to a copy of the query and kept only when
//! the whole of it applies, and every field it changed is said in the
//! Action Log in the words of the panel's own controls.

use serde_json::{json, Map, Value};
use sfmtool_core::analysis::track_table::{ColumnBounds, SortOrder, TrackTableColumn};

use super::pictures::PictureColumn;
use super::{
    column_index, column_title, parse_bound, BoundText, PublishedQuery, TrackTableQuery,
    DEFECT_COLUMNS,
};
use crate::filter_3d::Filter3dMode;
use crate::table_sort::TableSort;

/// A column of the panel, by the wire name `columns` takes: a picture column
/// or a track table column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PanelColumn(Column);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Column {
    Picture(PictureColumn),
    Value(TrackTableColumn),
}

impl PanelColumn {
    /// The column a wire name names, or `None` for a name that is neither a
    /// picture column nor a track table column.
    pub(crate) fn from_wire_name(name: &str) -> Option<Self> {
        PictureColumn::ALL
            .into_iter()
            .find(|c| c.wire_name() == name)
            .map(Column::Picture)
            .or_else(|| TrackTableColumn::from_wire_name(name).map(Column::Value))
            .map(PanelColumn)
    }
}

/// Every column of the panel by its wire name, in the order the Columns menu
/// lists them: the picture columns, then the track table columns.
pub(crate) fn panel_column_names() -> Vec<&'static str> {
    PictureColumn::ALL
        .iter()
        .map(|c| c.wire_name())
        .chain(TrackTableColumn::ALL.iter().map(|c| c.wire_name()))
        .collect()
}

/// What *Filter the 3D view* does: the box clear, or ticked with *Dim* or
/// *Hide*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Filter3dChoice {
    Off,
    Dim,
    Hide,
}

impl Filter3dChoice {
    /// Every choice, in the order the wire lists them.
    pub(crate) const ALL: [Filter3dChoice; 3] = [
        Filter3dChoice::Off,
        Filter3dChoice::Dim,
        Filter3dChoice::Hide,
    ];

    /// The value `filter_3d` carries.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Filter3dChoice::Off => "off",
            Filter3dChoice::Dim => "dim",
            Filter3dChoice::Hide => "hide",
        }
    }

    /// The choice a `filter_3d` value names.
    pub(crate) fn from_wire_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.wire_name() == name)
    }

    /// The choice the query holds.
    fn of(query: &TrackTableQuery) -> Self {
        match (query.filter_3d, query.filter_3d_mode) {
            (false, _) => Filter3dChoice::Off,
            (true, Filter3dMode::Dim) => Filter3dChoice::Dim,
            (true, Filter3dMode::Hide) => Filter3dChoice::Hide,
        }
    }
}

/// What a `set_track_table_query` call asks for, parsed and checked against
/// the column names: each field `None` where the call left it out.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct TrackTableQueryChange {
    /// Every bound box's new text: each column named gets its `min` and its
    /// `max`, and every other column's boxes are cleared.
    pub(crate) bounds: Option<Vec<ColumnBounds>>,
    /// The new sort, `Some(None)` for point order.
    pub(crate) order_by: Option<Option<(TrackTableColumn, SortOrder)>>,
    /// The columns shown, every other one hidden.
    pub(crate) columns: Option<Vec<PanelColumn>>,
    /// The *Seen in selected image* box.
    pub(crate) seen_in_selected_camera_image: Option<bool>,
    /// *Filter the 3D view* and its choice.
    pub(crate) filter_3d: Option<Filter3dChoice>,
}

impl TrackTableQueryChange {
    /// Whether the call named no field.
    pub(crate) fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A bound's end as the box holds it: the number, or `inf` for `+∞`, which
/// the box parses back to `+∞` as it does when a person types it.
fn end_text(end: Option<f64>) -> String {
    match end {
        None => String::new(),
        Some(v) if v == f64::INFINITY => "inf".to_string(),
        Some(v) => v.to_string(),
    }
}

/// A box's end as the wire carries it: a number, or the string `"inf"` for
/// `+∞` (`"-inf"` for the `-∞` a person can type), which JSON has no number
/// for.
fn end_value(v: f64) -> Value {
    if v.is_infinite() {
        json!(if v > 0.0 { "inf" } else { "-inf" })
    } else {
        json!(v)
    }
}

impl TrackTableQuery {
    /// The query as the wire document: the bounds the boxes set, the sort,
    /// the columns the query shows, the selected-image box and the filter of
    /// the 3D view.
    ///
    /// A box whose text is not a number sets no bound and is left out, as the
    /// panel draws it in the error colour and filters nothing by it. A defect
    /// count the person has neither ticked nor unticked, which shows itself
    /// while the reconstruction on show has a defect in it, is not in
    /// `columns` until it is named.
    fn document(&self) -> Value {
        let bounds: Vec<Value> = TrackTableColumn::ALL
            .into_iter()
            .filter_map(|column| {
                let text = self.bound(column);
                let min = parse_bound(&text.min).ok().flatten();
                let max = parse_bound(&text.max).ok().flatten();
                if min.is_none() && max.is_none() {
                    return None;
                }
                let mut bound = Map::new();
                bound.insert("column".into(), json!(column.wire_name()));
                if let Some(min) = min {
                    bound.insert("min".into(), end_value(min));
                }
                if let Some(max) = max {
                    bound.insert("max".into(), end_value(max));
                }
                Some(Value::Object(bound))
            })
            .collect();
        let order_by = match self.sort {
            None => Value::Null,
            Some(sort) => json!({
                "column": sort.column.wire_name(),
                "descending": sort.descending,
            }),
        };
        json!({
            "bounds": bounds,
            "order_by": order_by,
            "columns": self.column_names(),
            "seen_in_selected_camera_image": self.seen_in_selected_image,
            "filter_3d": Filter3dChoice::of(self).wire_name(),
        })
    }

    /// The columns the query shows by their wire names, the picture columns
    /// first, in the order they are drawn.
    fn column_names(&self) -> Vec<&'static str> {
        self.pictures
            .iter()
            .map(|c| c.wire_name())
            .chain(self.shown.iter().map(|c| c.wire_name()))
            .collect()
    }

    /// The columns the query shows by their headings, for the Action Log.
    fn column_titles(&self) -> Vec<&'static str> {
        self.pictures
            .iter()
            .map(|c| c.title())
            .chain(self.shown.iter().map(|&c| column_title(c)))
            .collect()
    }

    /// Apply `change`, refusing a column that holds a bound or the sort and
    /// that `columns` leaves out. Without `columns`, a column a new bound or
    /// sort lands on is shown, so a hidden column never filters or orders the
    /// rows.
    fn apply(&mut self, tool: &str, change: &TrackTableQueryChange) -> Result<(), String> {
        if let Some(bounds) = &change.bounds {
            self.bounds = std::array::from_fn(|_| BoundText::default());
            for bound in bounds {
                *self.bound_mut(bound.column) = BoundText {
                    min: end_text(bound.min),
                    max: end_text(bound.max),
                };
            }
        }
        if let Some(order_by) = change.order_by {
            self.sort = order_by.map(|(column, order)| TableSort {
                column,
                descending: order == SortOrder::Descending,
            });
        }
        if let Some(seen) = change.seen_in_selected_camera_image {
            self.seen_in_selected_image = seen;
        }
        match change.filter_3d {
            None => {}
            Some(Filter3dChoice::Off) => self.filter_3d = false,
            Some(Filter3dChoice::Dim) => {
                self.filter_3d = true;
                self.filter_3d_mode = Filter3dMode::Dim;
            }
            Some(Filter3dChoice::Hide) => {
                self.filter_3d = true;
                self.filter_3d_mode = Filter3dMode::Hide;
            }
        }
        match &change.columns {
            Some(columns) => {
                let mut pictures = Vec::new();
                let mut shown = Vec::new();
                for column in columns {
                    match column.0 {
                        Column::Picture(c) if !pictures.contains(&c) => pictures.push(c),
                        Column::Value(c) if !shown.contains(&c) => shown.push(c),
                        _ => {}
                    }
                }
                pictures.sort();
                shown.sort_by_key(|&c| column_index(c));
                if let Some(held) = TrackTableColumn::ALL
                    .into_iter()
                    .find(|&c| self.holds(c) && !shown.contains(&c))
                {
                    return Err(format!(
                        "{tool} cannot leave the column {} out of columns: it holds a bound or \
                         the sort. List it in columns, or clear its bound and the sort, so that a \
                         hidden column never filters or orders the rows.",
                        held.wire_name()
                    ));
                }
                self.pictures = pictures;
                self.shown = shown;
                // Every column is decided, as a tick or an untick in the
                // Columns menu decides one.
                self.chosen_defects = DEFECT_COLUMNS.to_vec();
            }
            None => {
                for column in TrackTableColumn::ALL {
                    if self.holds(column) && !self.is_shown(column, &[]) {
                        self.set_shown(column, true);
                    }
                }
            }
        }
        Ok(())
    }

    /// The Action Log rows that say what moved from `before` to `self`, one
    /// per control, in the words of the panel's controls.
    fn changes_from(&self, before: &TrackTableQuery) -> Vec<String> {
        let mut rows = Vec::new();
        for column in TrackTableColumn::ALL {
            let (was, now) = (before.bound(column), self.bound(column));
            for (end, was, now) in [
                ("minimum", &was.min, &now.min),
                ("maximum", &was.max, &now.max),
            ] {
                if was != now {
                    let value = if now.is_empty() {
                        "cleared"
                    } else {
                        now.as_str()
                    };
                    rows.push(format!(
                        "Track Table: {end} {} {value}",
                        column_title(column)
                    ));
                }
            }
        }
        if before.sort != self.sort {
            rows.push(match self.sort {
                None => "Track Table: sorted by point".to_string(),
                Some(sort) => format!(
                    "Track Table: sorted by {}, {}",
                    column_title(sort.column),
                    if sort.descending {
                        "decreasing"
                    } else {
                        "increasing"
                    }
                ),
            });
        }
        if (&before.pictures, &before.shown, &before.chosen_defects)
            != (&self.pictures, &self.shown, &self.chosen_defects)
        {
            rows.push(format!(
                "Track Table: Columns {}",
                self.column_titles().join(", ")
            ));
        }
        if before.seen_in_selected_image != self.seen_in_selected_image {
            rows.push(format!(
                "Track Table: {} {}",
                super::SEEN_IN_SELECTED_IMAGE,
                if self.seen_in_selected_image {
                    "on"
                } else {
                    "off"
                }
            ));
        }
        let (was, now) = (Filter3dChoice::of(before), Filter3dChoice::of(self));
        if was != now {
            rows.push(match now {
                Filter3dChoice::Off => format!("Track Table: {} off", super::filter::FILTER_3D),
                Filter3dChoice::Dim => {
                    format!(
                        "Track Table: {} on, {}",
                        super::filter::FILTER_3D,
                        super::filter::DIM
                    )
                }
                Filter3dChoice::Hide => format!(
                    "Track Table: {} on, {}",
                    super::filter::FILTER_3D,
                    super::filter::HIDE
                ),
            });
        }
        rows
    }
}

impl PublishedQuery {
    /// The query as the wire document.
    pub(crate) fn document(&self) -> Value {
        self.query.document()
    }

    /// Merge `change` into the query and leave it for the panel to take,
    /// answering with the Action Log rows that say what changed; or refuse
    /// the whole call, changing nothing.
    pub(crate) fn set(
        &mut self,
        tool: &str,
        change: &TrackTableQueryChange,
    ) -> Result<Vec<String>, String> {
        let mut query = self.query.clone();
        query.apply(tool, change)?;
        let rows = query.changes_from(&self.query);
        self.query = query;
        self.set_over_wire = true;
        Ok(rows)
    }
}
