// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Table panel: every live point of the selected reconstruction as
//! one row, with the stored track table columns, sortable by a click on a
//! heading and filtered by a minimum and a maximum per column.
//!
//! See `specs/gui/track-table.md`. The panel reads the track table cache
//! ([`crate::track_table`]) that `find_points` reads, and its quiet table job
//! ([`job`]) fills it. What it keeps between frames is its query, which is UI
//! state of the panel and not part of any version, and the **row list**: the
//! point indexes that pass the query, in its order. A reconstruction can hold
//! millions of points, so the row list is computed again only when its inputs
//! change (the query, the document serial or the columns the cache holds),
//! never once per frame, and only the visible rows are drawn
//! (`ScrollArea::show_rows`).

pub(crate) mod job;

#[cfg(test)]
mod tests;

use sfmtool_core::analysis::track_table::{
    query_points, ColumnBounds, ColumnCost, SortOrder, TrackTable, TrackTableColumn,
    TrackTableQueryOptions,
};
use sfmtool_core::EditedReconstruction;

use crate::document::VersionSerial;
use crate::scene::{PointRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::table_sort::{paint_sort_triangle, SortRule, TableSort, SORT_TRIANGLE_HALF_WIDTH};

/// How many columns the track table has.
const COLUMN_COUNT: usize = TrackTableColumn::ALL.len();

/// The columns shown before any are hidden or shown: the point is always
/// shown, and of the stored columns the track's length, its image count, its
/// error and whether it is at infinity.
const DEFAULT_COLUMNS: [TrackTableColumn; 4] = [
    TrackTableColumn::TrackLength,
    TrackTableColumn::ImageCount,
    TrackTableColumn::RmsErrorPx,
    TrackTableColumn::AtInfinity,
];

/// The width of the Point column.
const POINT_WIDTH: f32 = 72.0;
/// The narrowest a value column is: room for its two bound boxes.
const MIN_VALUE_WIDTH: f32 = 84.0;
/// Space between a cell's text and its column's edges.
const CELL_PAD: f32 = 6.0;

impl SortRule for TrackTableColumn {
    /// The columns whose large values are the bad end: the error, the
    /// condition number, the inverse-depth z and the self-similarity radius.
    fn worst_first_descending(self) -> bool {
        matches!(
            self,
            TrackTableColumn::RmsErrorPx
                | TrackTableColumn::ConditionNumber
                | TrackTableColumn::InverseDepthZ
                | TrackTableColumn::LargestZnccSelfSimilarityRadius
        )
    }
}

/// Whether a column's bad end is its small values, so that its first click,
/// which orders it increasing, puts the worst rows first.
fn worst_first_ascending(column: TrackTableColumn) -> bool {
    matches!(
        column,
        TrackTableColumn::TrackLength
            | TrackTableColumn::ImageCount
            | TrackTableColumn::MaxRayAngleDeg
            | TrackTableColumn::LowestZncc
            | TrackTableColumn::NormalConfidence
    )
}

/// The heading a column is shown under.
fn column_title(column: TrackTableColumn) -> &'static str {
    match column {
        TrackTableColumn::TrackLength => "Track",
        TrackTableColumn::ImageCount => "Images",
        TrackTableColumn::RmsErrorPx => "Error (px)",
        TrackTableColumn::AtInfinity => "\u{221e}",
        TrackTableColumn::X => "X",
        TrackTableColumn::Y => "Y",
        TrackTableColumn::Z => "Z",
        TrackTableColumn::CentroidDistance => "Distance",
        TrackTableColumn::NormalConfidence => "Normal conf.",
        TrackTableColumn::LowestZncc => "Lowest ZNCC",
        TrackTableColumn::LargestZnccSelfSimilarityRadius => "Self-sim. radius",
        TrackTableColumn::MaxRayAngleDeg => "Ray angle (\u{b0})",
        TrackTableColumn::ConditionNumber => "Cond.",
        TrackTableColumn::InverseDepthZ => "Inv. depth z",
    }
}

/// What a column holds, for its heading's hover text and its entry in the
/// Columns menu.
fn column_description(column: TrackTableColumn) -> &'static str {
    match column {
        TrackTableColumn::TrackLength => "The number of observations in the track.",
        TrackTableColumn::ImageCount => {
            "The number of distinct images among the track's observations. It is less than \
             Track where one image holds several observations of the point."
        }
        TrackTableColumn::RmsErrorPx => {
            "The point's stored RMS reprojection error over its observations, in pixels."
        }
        TrackTableColumn::AtInfinity => {
            "\u{221e} for a point at infinity, which is a direction rather than a position. \
             The value is 1 for such a point and 0 for a finite one, so a minimum of 1 keeps \
             only the points at infinity."
        }
        TrackTableColumn::X | TrackTableColumn::Y | TrackTableColumn::Z => {
            "The point's position along this axis, in the reconstruction's unit. A point at \
             infinity has no value."
        }
        TrackTableColumn::CentroidDistance => {
            "The point's distance from the centroid of the posed cameras, in the unit of X. A \
             point at infinity has no value."
        }
        TrackTableColumn::NormalConfidence => "The stored normal confidence, 0 to 255.",
        TrackTableColumn::LowestZncc => {
            "The lowest measured ZNCC among the track's observations, 0 to 1. A track with no \
             measured observation has no value."
        }
        TrackTableColumn::LargestZnccSelfSimilarityRadius => {
            "The largest ZNCC self-similarity radius among the track's observation readings, \
             in patch px. A track with no measured reading has no value."
        }
        TrackTableColumn::MaxRayAngleDeg => {
            "The largest angle between two of the track's viewing rays, in degrees."
        }
        TrackTableColumn::ConditionNumber => "The condition number of the point's triangulation.",
        TrackTableColumn::InverseDepthZ => {
            "The inverse-depth z-score of the point's triangulation."
        }
    }
}

/// A column's place in [`TrackTableColumn::ALL`], which indexes the query's
/// bounds.
fn column_index(column: TrackTableColumn) -> usize {
    TrackTableColumn::ALL
        .iter()
        .position(|&c| c == column)
        .expect("every column is in ALL")
}

/// Whether the panel offers `column` for `edited`: a stored column the
/// reconstruction can supply. The computed columns are not shown yet.
fn offered(column: TrackTableColumn, edited: &EditedReconstruction) -> bool {
    column.cost() == ColumnCost::Stored && column.is_available(edited)
}

/// One column's two bound boxes, as typed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BoundText {
    /// The minimum box's text.
    min: String,
    /// The maximum box's text.
    max: String,
}

/// One box's text read as a bound: `Ok(None)` for an empty box, which sets no
/// bound, and `Err(())` for text that is not a number, which sets none either
/// and is drawn as a mistake.
fn parse_bound(text: &str) -> Result<Option<f64>, ()> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    match text.parse::<f64>() {
        Ok(value) if !value.is_nan() => Ok(Some(value)),
        _ => Err(()),
    }
}

impl BoundText {
    /// Whether either box holds anything, which keeps the column from being
    /// hidden.
    fn is_set(&self) -> bool {
        !self.min.trim().is_empty() || !self.max.trim().is_empty()
    }

    /// The bound the two boxes set on `column`, or `None` when neither sets
    /// one.
    fn parsed(&self, column: TrackTableColumn) -> Option<ColumnBounds> {
        let min = parse_bound(&self.min).ok().flatten();
        let max = parse_bound(&self.max).ok().flatten();
        (min.is_some() || max.is_some()).then_some(ColumnBounds { column, min, max })
    }
}

/// The panel's query: the bounds, the sort, the columns shown and whether only
/// the points the selected image sees are kept. UI state of the panel, like
/// the Image Detail display settings: it is held in no version, costs no undo
/// step and is not saved. One query serves whichever reconstruction is
/// selected.
#[derive(Debug, Clone, PartialEq)]
struct TrackTableQuery {
    /// Each column's bound boxes, in [`TrackTableColumn::ALL`] order.
    bounds: [BoundText; COLUMN_COUNT],
    /// The column the rows are ordered by and which way, or `None` for
    /// increasing point index.
    sort: Option<TableSort<TrackTableColumn>>,
    /// The value columns shown, in [`TrackTableColumn::ALL`] order. The Point
    /// column is always shown.
    shown: Vec<TrackTableColumn>,
    /// Whether only the points the selected image observes are kept.
    seen_in_selected_image: bool,
}

impl Default for TrackTableQuery {
    fn default() -> Self {
        Self {
            bounds: std::array::from_fn(|_| BoundText::default()),
            sort: None,
            shown: DEFAULT_COLUMNS.to_vec(),
            seen_in_selected_image: false,
        }
    }
}

impl TrackTableQuery {
    /// The bound boxes of `column`.
    fn bound(&self, column: TrackTableColumn) -> &BoundText {
        &self.bounds[column_index(column)]
    }

    /// The bound boxes of `column`, to edit.
    fn bound_mut(&mut self, column: TrackTableColumn) -> &mut BoundText {
        &mut self.bounds[column_index(column)]
    }

    /// Whether `column` holds a bound or the sort, which keeps it from being
    /// hidden: a hidden column never filters or orders the rows unseen.
    fn holds(&self, column: TrackTableColumn) -> bool {
        self.bound(column).is_set() || self.sort.is_some_and(|sort| sort.column == column)
    }

    /// Show or hide `column`. Hiding one that [`Self::holds`] does nothing.
    fn set_shown(&mut self, column: TrackTableColumn, shown: bool) {
        if shown {
            if !self.shown.contains(&column) {
                self.shown.push(column);
                self.shown.sort_by_key(|&c| column_index(c));
            }
        } else if !self.holds(column) {
            self.shown.retain(|&c| c != column);
        }
    }

    /// The order a click on `column`'s heading leaves, by the rule the
    /// viewer's tables share ([`crate::table_sort`]).
    fn click_heading(&mut self, column: TrackTableColumn) {
        self.sort = Some(match self.sort {
            Some(sort) => sort.clicked(column),
            None => TableSort::first_click(column),
        });
    }
}

/// What the row list was computed from. Compared once per frame, which costs
/// nothing beside running the query again over a million rows.
#[derive(Debug, Clone, PartialEq)]
struct RowInputs {
    node: ReconId,
    document: VersionSerial,
    /// The columns the cached table held, in [`TrackTableColumn::ALL`] order.
    held: Vec<TrackTableColumn>,
    bounds: Vec<ColumnBounds>,
    seen_in_image: Option<u32>,
    order_by: Option<(TrackTableColumn, SortOrder)>,
}

/// The rows that pass the query, in its order, kept between frames.
#[derive(Debug, Default)]
struct RowList {
    /// What the rows were computed from, or `None` before the first time.
    inputs: Option<RowInputs>,
    /// The matched point indexes, in order.
    rows: Vec<u32>,
    /// The query's refusal, when it refused.
    error: Option<String>,
    /// How many times the rows have been computed, for the test that holds
    /// the rule that an unchanged query does not compute them again.
    recomputed: u64,
}

impl RowList {
    /// Compute the rows again when `inputs` differ from the ones they were
    /// computed from, and answer whether the query or the node moved, which
    /// puts other rows at the top. A new version or a column landing in the
    /// cache moves neither.
    fn refresh(
        &mut self,
        edited: &EditedReconstruction,
        table: &TrackTable,
        inputs: RowInputs,
    ) -> bool {
        let moved = match &self.inputs {
            Some(old) if *old == inputs => return false,
            Some(old) => {
                old.node != inputs.node
                    || old.bounds != inputs.bounds
                    || old.seen_in_image != inputs.seen_in_image
                    || old.order_by != inputs.order_by
            }
            None => true,
        };
        let options = TrackTableQueryOptions {
            bounds: inputs.bounds.clone(),
            seen_in_image: inputs.seen_in_image,
            order_by: inputs.order_by,
        };
        match query_points(edited, table, &options) {
            Ok(rows) => {
                self.rows = rows;
                self.error = None;
            }
            Err(error) => {
                self.rows.clear();
                self.error = Some(error.to_string());
            }
        }
        self.inputs = Some(inputs);
        self.recomputed += 1;
        moved
    }
}

/// What the query cannot apply to the reconstruction on show, said in the
/// header so that nothing filters unseen and nothing is dropped unsaid.
#[derive(Debug, Default, PartialEq)]
struct Unapplied {
    /// Columns holding a bound or the sort that the reconstruction cannot
    /// supply: kept, and applied again once one that can is selected.
    unsupplied: Vec<TrackTableColumn>,
    /// Columns holding a bound or the sort that the cache does not hold yet.
    pending: Vec<TrackTableColumn>,
}

/// The Track Table panel's state: its query, the row list, and what it
/// remembers of the last frame to keep the selection and the hover in step.
#[derive(Debug, Default)]
pub(crate) struct TrackTablePanel {
    /// The query, which is panel state.
    query: TrackTableQuery,
    /// The rows that pass it, kept between frames.
    rows: RowList,
    /// The selected point as of the last frame, so that a selection made
    /// elsewhere scrolls its row into view once.
    seen_selection: Option<PointRef>,
    /// Whether the pointer was over the rows in the last frame, so that
    /// leaving them clears the hover once rather than every frame.
    had_pointer: bool,
    /// The rows the last frame drew, each with its rectangle.
    #[cfg(test)]
    drawn: Vec<(u32, egui::Rect)>,
}

/// What the panel's frame asked the state for, applied once the node it drew
/// is no longer borrowed.
#[derive(Debug, Default)]
struct Asked {
    select: Option<PointRef>,
    /// `Some` while the panel owns the hover: the row under the pointer, or
    /// `None` over no row.
    hover: Option<Option<PointRef>>,
    /// The rows drawn, each with its rectangle, for a test to click.
    #[cfg(test)]
    drawn: Vec<(u32, egui::Rect)>,
}

impl TrackTablePanel {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The matched point indexes, in order, as of the last frame drawn.
    #[cfg(test)]
    fn rows(&self) -> &[u32] {
        &self.rows.rows
    }

    /// How many times the row list has been computed.
    #[cfg(test)]
    fn recomputed(&self) -> u64 {
        self.rows.recomputed
    }

    /// The rows the last frame drew, each with its rectangle.
    #[cfg(test)]
    fn drawn(&self) -> &[(u32, egui::Rect)] {
        &self.drawn
    }

    /// The row inputs and what of the query they leave out, for the node and
    /// the table on show.
    fn row_inputs(
        &self,
        node: &SceneNode,
        table: &TrackTable,
        selected_image: Option<u32>,
    ) -> (RowInputs, Unapplied) {
        let edited = node.edited();
        let mut unapplied = Unapplied::default();
        let mut usable = |column: TrackTableColumn| {
            if !offered(column, edited) {
                unapplied.unsupplied.push(column);
                false
            } else if !table.has(column) {
                unapplied.pending.push(column);
                false
            } else {
                true
            }
        };
        let mut bounds = Vec::new();
        for column in TrackTableColumn::ALL {
            let text = self.query.bound(column);
            if !text.is_set() {
                continue;
            }
            if usable(column) {
                bounds.extend(text.parsed(column));
            }
        }
        let order_by = self.query.sort.and_then(|sort| {
            usable(sort.column).then_some((
                sort.column,
                if sort.descending {
                    SortOrder::Descending
                } else {
                    SortOrder::Ascending
                },
            ))
        });
        let inputs = RowInputs {
            node: node.id,
            document: node.history.current_version().document_serial,
            held: TrackTableColumn::ALL
                .into_iter()
                .filter(|&c| table.has(c))
                .collect(),
            bounds,
            seen_in_image: if self.query.seen_in_selected_image {
                selected_image
            } else {
                None
            },
            order_by,
        };
        (inputs, unapplied)
    }

    /// Draw the panel for the selected reconstruction.
    pub(crate) fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        let Some(id) =
            crate::scene::selected_node(&state.scene, state.selected_recon).map(|n| n.id)
        else {
            ui.centered_and_justified(|ui| {
                ui.label("No reconstruction loaded");
            });
            return;
        };
        state.want_track_table_columns(id);
        let before = self.query.clone();
        #[cfg_attr(not(test), allow(unused_mut))]
        let mut asked = self.show_node(ui, state, id);
        // The rows are computed from the query at the top of a frame, so an
        // edit this frame made shows in the next one, which is asked for here
        // rather than left to the next input event.
        if self.query != before {
            ui.ctx().request_repaint();
        }
        #[cfg(test)]
        {
            self.drawn = std::mem::take(&mut asked.drawn);
        }
        if let Some(point) = asked.select {
            self.seen_selection = Some(point);
            state.select_point(point);
        }
        if let Some(hover) = asked.hover {
            state.hovered_point = hover;
        }
    }

    /// The panel body for the node `id`, which is loaded.
    fn show_node(&mut self, ui: &mut egui::Ui, state: &AppState, id: ReconId) -> Asked {
        let mut asked = Asked::default();
        let node = state.node(id).expect("the selected node is loaded");
        let edited = node.edited();
        let selected_image = state
            .selected_image
            .and_then(|image| image.index_in(id))
            .map(|index| index as u32);
        let table = state.track_table(id);

        // The rows first, so the header counts the rows this frame draws.
        let mut reordered = false;
        let unapplied = table.map(|table| {
            let (inputs, unapplied) = self.row_inputs(node, table, selected_image);
            reordered = self.rows.refresh(edited, table, inputs);
            unapplied
        });

        self.show_header(ui, node, table, selected_image);
        if let Some(unapplied) = &unapplied {
            show_unapplied(ui, node, unapplied);
        }
        ui.separator();

        let Some(table) = table else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    egui::RichText::new(format!(
                        "Computing the track table of {}\u{2026}",
                        node.label
                    ))
                    .weak(),
                );
            });
            return asked;
        };
        if let Some(error) = &self.rows.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
            return asked;
        }

        let layout = ColumnLayout::new(ui, &self.query, edited);
        let selected = state.selected_point.filter(|p| p.recon == id);
        // A selection made elsewhere scrolls its row into view; otherwise a
        // new sort, new bounds or another reconstruction starts at the top,
        // where the rows the query puts first are.
        let scroll_to = if state.selected_point != self.seen_selection {
            self.seen_selection = state.selected_point;
            selected.and_then(|p| self.rows.rows.iter().position(|&row| row == p.point))
        } else {
            None
        }
        .or(reordered.then_some(0));
        let hovered = state.hovered_point.filter(|p| p.recon == id);

        // The headings and the bounds are above the rows' scroll area, so they
        // stay put while the rows scroll up and down. Their space is taken
        // now and they are drawn after the rows, from the left edge the rows
        // were drawn from on this frame, so a sideways scroll moves them with
        // the columns under them, as Track View's headings do.
        let heading_band = reserve_band(ui, heading_height(ui));
        let bounds_band = reserve_band(ui, bounds_height(ui));
        let mut left = heading_band.left();
        if self.rows.rows.is_empty() {
            ui.label(egui::RichText::new("No point passes the bounds.").weak());
        } else {
            let rows = RowsView {
                node,
                table,
                rows: &self.rows.rows,
                layout: &layout,
                selected,
                hovered,
            };
            let (over, rows_left) = rows.show(ui, scroll_to, &mut asked);
            left = rows_left;
            if over {
                self.had_pointer = true;
            } else if std::mem::take(&mut self.had_pointer) {
                asked.hover = Some(None);
            }
        }
        self.show_headings(ui, heading_band, left, &layout);
        self.show_bounds(ui, bounds_band, left, &layout);
        asked
    }

    /// The header line: the reconstruction, the count of matched rows against
    /// live points, the Columns menu and the selected-image box.
    fn show_header(
        &mut self,
        ui: &mut egui::Ui,
        node: &SceneNode,
        table: Option<&TrackTable>,
        selected_image: Option<u32>,
    ) {
        let edited = node.edited();
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(&node.label).strong());
            if let Some(table) = table {
                ui.label(format!(
                    "{} of {} points",
                    grouped(self.rows.rows.len()),
                    grouped(table.point_indexes().len())
                ));
            }
            let query = &mut self.query;
            egui::containers::menu::MenuButton::new("Columns")
                .config(
                    egui::containers::menu::MenuConfig::new()
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                )
                .ui(ui, |ui| {
                    for column in TrackTableColumn::ALL {
                        if !offered(column, edited) {
                            continue;
                        }
                        let mut shown = query.shown.contains(&column);
                        let locked = shown && query.holds(column);
                        let response = ui
                            .add_enabled(
                                !locked,
                                egui::Checkbox::new(&mut shown, column_title(column)),
                            )
                            .on_hover_text(column_description(column))
                            .on_disabled_hover_text(format!(
                                "{} holds a bound or the sort. Clear them before hiding it, so \
                                 that a hidden column never filters or orders the rows.",
                                column_title(column)
                            ));
                        if response.changed() {
                            query.set_shown(column, shown);
                        }
                    }
                });
            let image_name = selected_image.and_then(|index| {
                node.recon()
                    .image_set
                    .images
                    .get(index as usize)
                    .map(|image| image.name.clone())
            });
            let label = match &image_name {
                Some(name) => format!("Only points seen in {name}"),
                None => "Only points seen in the selected image".to_string(),
            };
            let mut text = egui::RichText::new(label);
            if image_name.is_none() && self.query.seen_in_selected_image {
                text = text.weak();
            }
            ui.checkbox(&mut self.query.seen_in_selected_image, text)
                .on_hover_text(match &image_name {
                    Some(_) => "Keep only the points the selected image observes.",
                    None => {
                        "Keep only the points the selected image observes. No image of this \
                         reconstruction is selected, so this filters nothing."
                    }
                });
        });
    }

    /// The heading row: a click on a value column's heading orders the rows
    /// by it, and a click on *Point* puts them back in point order.
    fn show_headings(
        &mut self,
        ui: &mut egui::Ui,
        band: egui::Rect,
        left: f32,
        layout: &ColumnLayout,
    ) {
        let ui = &mut band_ui(ui, band);
        let height = band.height();
        let rect = egui::Rect::from_min_size(
            egui::pos2(left, band.min.y),
            egui::vec2(layout.width, height),
        );
        let font = egui::TextStyle::Body.resolve(ui.style());
        for &(slot, x, width) in &layout.slots {
            let cell = egui::Rect::from_min_size(
                egui::pos2(rect.min.x + x, rect.min.y),
                egui::vec2(width, height),
            );
            let response = ui.interact(
                cell,
                ui.id().with(("track_table_heading", slot)),
                egui::Sense::click(),
            );
            let color = if response.hovered() {
                ui.visuals().strong_text_color()
            } else {
                ui.visuals().text_color()
            };
            let (title, column) = match slot {
                Slot::Point => ("Point", None),
                Slot::Value(column) => (column_title(column), Some(column)),
            };
            let word = ui.painter().text(
                egui::pos2(cell.min.x + CELL_PAD, cell.center().y),
                egui::Align2::LEFT_CENTER,
                title,
                font.clone(),
                color,
            );
            let sort = self.query.sort;
            match column {
                None => {
                    response.widget_info(|| {
                        egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Sort by point")
                    });
                    let clicked = response.clicked();
                    response.on_hover_text(match sort {
                        None => "The point's index. The rows are in increasing order by it.",
                        Some(_) => {
                            "The point's index. Click to put the rows back in increasing order \
                             by it."
                        }
                    });
                    if clicked {
                        self.query.sort = None;
                    }
                }
                Some(column) => {
                    if let Some(sort) = sort.filter(|sort| sort.column == column) {
                        paint_sort_triangle(
                            ui.painter(),
                            egui::pos2(
                                word.right() + 2.0 + SORT_TRIANGLE_HALF_WIDTH,
                                cell.center().y,
                            ),
                            sort.descending,
                            color,
                        );
                    }
                    response.widget_info(|| {
                        egui::WidgetInfo::labeled(
                            egui::WidgetType::Button,
                            true,
                            format!("Sort by {}", column.wire_name()),
                        )
                    });
                    let clicked = response.clicked();
                    response.on_hover_text(format!(
                        "{}\n\n`{}` on the wire.\n\n{}",
                        column_description(column),
                        column.wire_name(),
                        sort_hover(sort, column)
                    ));
                    if clicked {
                        self.query.click_heading(column);
                    }
                }
            }
        }
    }

    /// The bounds row under the headings: a minimum box and a maximum box per
    /// value column. An empty box sets no bound.
    fn show_bounds(
        &mut self,
        ui: &mut egui::Ui,
        band: egui::Rect,
        left: f32,
        layout: &ColumnLayout,
    ) {
        let ui = &mut band_ui(ui, band);
        let height = band.height();
        let rect = egui::Rect::from_min_size(
            egui::pos2(left, band.min.y),
            egui::vec2(layout.width, height),
        );
        for &(slot, x, width) in &layout.slots {
            let Slot::Value(column) = slot else {
                continue;
            };
            let cell = egui::Rect::from_min_size(
                egui::pos2(rect.min.x + x + 2.0, rect.min.y + 2.0),
                egui::vec2(width - 4.0, height - 4.0),
            );
            let mut cell_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(cell)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            cell_ui.set_clip_rect(ui.clip_rect());
            cell_ui.spacing_mut().item_spacing.x = 2.0;
            let box_width = (cell.width() - 2.0) / 2.0;
            let title = column_title(column);
            let bound = self.query.bound_mut(column);
            for (text, hint, end) in [
                (&mut bound.min, "min", "below"),
                (&mut bound.max, "max", "above"),
            ] {
                let bad = parse_bound(text).is_err();
                let mut edit = egui::TextEdit::singleline(text)
                    .hint_text(hint)
                    .desired_width(box_width)
                    .id_salt(("track_table_bound", column.wire_name(), hint));
                if bad {
                    edit = edit.text_color(cell_ui.visuals().error_fg_color);
                }
                let response = cell_ui.add_sized([box_width, cell.height()], edit);
                // Named for the column, so a screen reader and `get_widgets`
                // say which box this is rather than an unnamed text input.
                let value = text.clone();
                response.widget_info(|| {
                    let mut info = egui::WidgetInfo::text_edit(true, &value, &value, hint);
                    info.label = Some(format!("{hint}imum {title}"));
                    info
                });
                response.on_hover_text(if bad {
                    format!("This is not a number, so it sets no bound on {title}.")
                } else {
                    format!(
                        "The {hint}imum {title}: rows {end} it are hidden. An empty box sets no \
                         bound, and a row with no value fails any bound on its column."
                    )
                });
            }
        }
    }
}

/// Take a band of `height` across the panel, above the rows.
fn reserve_band(ui: &mut egui::Ui, height: f32) -> egui::Rect {
    let available = ui.available_rect_before_wrap();
    let rect = egui::Rect::from_min_size(available.min, egui::vec2(available.width(), height));
    ui.allocate_rect(rect, egui::Sense::hover()).rect
}

/// A child of `ui` that draws in `band` and is clipped to it, so a heading or
/// a box scrolled sideways past the panel's edge is cut off there.
fn band_ui(ui: &mut egui::Ui, band: egui::Rect) -> egui::Ui {
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(band));
    child.set_clip_rect(band.intersect(ui.clip_rect()));
    child
}

/// The heading row's height.
fn heading_height(ui: &egui::Ui) -> f32 {
    ui.text_style_height(&egui::TextStyle::Body) + 6.0
}

/// The bounds row's height: one line of boxes.
fn bounds_height(ui: &egui::Ui) -> f32 {
    ui.spacing().interact_size.y + 4.0
}

/// The last paragraph of a heading's hover text: what a click on it does with
/// the order `sort` stands at.
fn sort_hover(sort: Option<TableSort<TrackTableColumn>>, column: TrackTableColumn) -> &'static str {
    match sort.filter(|sort| sort.column == column) {
        Some(TableSort {
            descending: false, ..
        }) => "The rows are in increasing order by this column. Click to reverse it.",
        Some(_) => "The rows are in decreasing order by this column. Click to reverse it.",
        None if column.worst_first_descending() => {
            "Click to put the rows in decreasing order by this column, worst first."
        }
        None if worst_first_ascending(column) => {
            "Click to put the rows in increasing order by this column, worst first."
        }
        None => "Click to put the rows in increasing order by this column.",
    }
}

/// The header's second line, when the query names columns it cannot apply.
fn show_unapplied(ui: &mut egui::Ui, node: &SceneNode, unapplied: &Unapplied) {
    let titles = |columns: &[TrackTableColumn]| {
        columns
            .iter()
            .map(|&c| column_title(c))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !unapplied.unsupplied.is_empty() {
        ui.label(
            egui::RichText::new(format!(
                "Not filtering or sorting by {}: {} cannot supply {}. The bounds are kept for a \
                 reconstruction that can.",
                titles(&unapplied.unsupplied),
                node.label,
                if unapplied.unsupplied.len() == 1 {
                    "it"
                } else {
                    "them"
                }
            ))
            .weak(),
        );
    }
    if !unapplied.pending.is_empty() {
        ui.label(
            egui::RichText::new(format!(
                "Filtering and sorting when {} {} ready.",
                titles(&unapplied.pending),
                if unapplied.pending.len() == 1 {
                    "is"
                } else {
                    "are"
                }
            ))
            .weak(),
        );
    }
}

/// One column the table draws.
///
/// The Point column and the value columns are text. The picture columns of
/// the draft are further kinds of slot, and the row height then follows the
/// pictures rather than the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    /// The point index, the row's key.
    Point,
    /// One track table column.
    Value(TrackTableColumn),
}

/// The columns drawn, left to right, each with its left edge and its width
/// from the table's left edge.
struct ColumnLayout {
    slots: Vec<(Slot, f32, f32)>,
    width: f32,
}

impl ColumnLayout {
    /// The Point column, then each shown column `edited` can supply, each
    /// wide enough for its heading, its sort triangle and its two bound boxes.
    fn new(ui: &egui::Ui, query: &TrackTableQuery, edited: &EditedReconstruction) -> Self {
        let font = egui::TextStyle::Body.resolve(ui.style());
        let mut slots = vec![(Slot::Point, 0.0, POINT_WIDTH)];
        let mut x = POINT_WIDTH;
        for &column in &query.shown {
            if !offered(column, edited) {
                continue;
            }
            let title = ui
                .painter()
                .layout_no_wrap(
                    column_title(column).to_string(),
                    font.clone(),
                    egui::Color32::WHITE,
                )
                .size()
                .x;
            let width =
                (title + 2.0 * CELL_PAD + 4.0 * SORT_TRIANGLE_HALF_WIDTH).max(MIN_VALUE_WIDTH);
            slots.push((Slot::Value(column), x, width));
            x += width;
        }
        Self { slots, width: x }
    }
}

/// The rows, as one frame draws them.
struct RowsView<'a> {
    node: &'a SceneNode,
    table: &'a TrackTable,
    rows: &'a [u32],
    layout: &'a ColumnLayout,
    selected: Option<PointRef>,
    hovered: Option<PointRef>,
}

impl RowsView<'_> {
    /// The row height while no picture column is shown: a line of text.
    fn row_height(ui: &egui::Ui) -> f32 {
        ui.text_style_height(&egui::TextStyle::Body) + 6.0
    }

    /// Draw the visible rows, scrolling row `scroll_to` into view, and answer
    /// whether the pointer is over the rows and the left edge the rows were
    /// drawn from, which the headings follow. A click and a hover go into
    /// `asked`.
    fn show(&self, ui: &mut egui::Ui, scroll_to: Option<usize>, asked: &mut Asked) -> (bool, f32) {
        let row_height = Self::row_height(ui);
        // Rows touch: the selection and hover fills meet, and row `k` starts at
        // exactly `k` row heights, which is what the scroll below aims at.
        ui.spacing_mut().item_spacing.y = 0.0;
        // Both ways: the columns can be wider than the panel.
        let mut area = egui::ScrollArea::both()
            .id_salt("track_table_rows")
            .auto_shrink([false, false]);
        if let Some(row) = scroll_to {
            let centred = row as f32 * row_height - 0.5 * ui.available_height() + row_height;
            area = area.vertical_scroll_offset(centred.max(0.0));
        }
        let mut drawn_left = ui.available_rect_before_wrap().left();
        let output = area.show_rows(ui, row_height, self.rows.len(), |ui, range| {
            ui.set_min_width(self.layout.width);
            let top = ui.max_rect().top();
            let left = ui.max_rect().left();
            drawn_left = left;
            let mut hovered_row = None;
            for (k, position) in range.clone().enumerate() {
                let point = self.rows[position];
                let rect = egui::Rect::from_min_size(
                    egui::pos2(left, top + k as f32 * row_height),
                    egui::vec2(self.layout.width, row_height),
                );
                let response = ui.interact(
                    rect,
                    ui.id().with(("track_table_row", point)),
                    egui::Sense::click(),
                );
                let reference = PointRef {
                    recon: self.node.id,
                    point,
                };
                let selected = self.selected == Some(reference);
                self.paint_row(
                    ui,
                    rect,
                    position,
                    point,
                    selected,
                    self.hovered == Some(reference),
                );
                response.widget_info(|| {
                    egui::WidgetInfo::selected(
                        egui::WidgetType::SelectableLabel,
                        true,
                        selected,
                        format!("Point #{point}"),
                    )
                });
                #[cfg(test)]
                asked.drawn.push((point, rect));
                if response.clicked() {
                    asked.select = Some(reference);
                }
                if response.hovered() {
                    hovered_row = Some(reference);
                    // Minted for the row under the pointer only: an id walks
                    // the node's version graph.
                    response.on_hover_text_at_pointer(format!(
                        "{}. Click to select it.",
                        crate::scene::point_id(self.node, point as usize)
                    ));
                }
            }
            hovered_row
        });
        // The hover position rather than `rect_contains_pointer`, which reads
        // the last position a pointer that has left the window had; and only
        // where no popup or window lies over the rows.
        let over = ui.ctx().pointer_hover_pos().is_some_and(|at| {
            output.inner_rect.contains(at) && ui.ctx().layer_id_at(at) == Some(ui.layer_id())
        });
        if over {
            asked.hover = Some(output.inner);
        }
        (over, drawn_left)
    }

    /// Paint one row: its fill, then each cell's text.
    fn paint_row(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        position: usize,
        point: u32,
        selected: bool,
        hovered: bool,
    ) {
        let visuals = ui.visuals();
        let fill = if selected {
            Some(visuals.selection.bg_fill)
        } else if hovered {
            Some(visuals.widgets.hovered.weak_bg_fill)
        } else if position % 2 == 1 {
            Some(visuals.faint_bg_color)
        } else {
            None
        };
        if let Some(fill) = fill {
            ui.painter().rect_filled(rect, 0.0, fill);
        }
        let color = if selected {
            visuals.selection.stroke.color
        } else if hovered {
            // The hover fill is lighter than the panel, so the text over it
            // is the strong colour to keep its contrast.
            visuals.strong_text_color()
        } else {
            visuals.text_color()
        };
        let font = egui::TextStyle::Body.resolve(ui.style());
        let row = self
            .table
            .point_indexes()
            .binary_search(&point)
            .expect("a matched point is a row of the table it was matched in");
        for &(slot, x, width) in &self.layout.slots {
            let (text, align) = match slot {
                Slot::Point => (format!("#{point}"), egui::Align2::LEFT_CENTER),
                Slot::Value(column) => (
                    cell_text(self.table, column, row),
                    egui::Align2::RIGHT_CENTER,
                ),
            };
            let anchor = match slot {
                Slot::Point => egui::pos2(rect.min.x + x + CELL_PAD, rect.center().y),
                Slot::Value(_) => egui::pos2(rect.min.x + x + width - CELL_PAD, rect.center().y),
            };
            ui.painter().text(anchor, align, text, font.clone(), color);
        }
    }
}

/// One cell's text: `…` for a column the cache does not hold yet, `-` for a
/// point with no value, a whole number for a count, `∞` or nothing for
/// `at_infinity`, and three significant digits otherwise. Positions keep up to
/// three decimals however large they are.
fn cell_text(table: &TrackTable, column: TrackTableColumn, row: usize) -> String {
    if column.is_f64() {
        return match table.values_f64(column) {
            None => "\u{2026}".to_string(),
            Some(values) => position_text(values[row]),
        };
    }
    let Some(values) = table.values(column) else {
        return "\u{2026}".to_string();
    };
    let v = values[row];
    if v.is_nan() {
        return "-".to_string();
    }
    match column {
        TrackTableColumn::TrackLength
        | TrackTableColumn::ImageCount
        | TrackTableColumn::NormalConfidence => format!("{}", v as i64),
        TrackTableColumn::AtInfinity => {
            if v != 0.0 {
                "\u{221e}".to_string()
            } else {
                String::new()
            }
        }
        TrackTableColumn::LowestZncc => format!("{v:.2}"),
        _ => significant(f64::from(v), 3, 3),
    }
}

/// A position or a distance: four significant digits, with at most three
/// decimals and every whole digit kept.
fn position_text(v: f64) -> String {
    if v.is_nan() {
        return "-".to_string();
    }
    significant(v, 4, 3)
}

/// `v` to `digits` significant digits, with at most `max_decimals` decimals,
/// keeping every whole digit.
fn significant(v: f64, digits: i32, max_decimals: i32) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let magnitude = v.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).clamp(0, max_decimals) as usize;
    format!("{v:.decimals$}")
}

/// A count with its thousands grouped by commas: `9,941`.
fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (k, c) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
